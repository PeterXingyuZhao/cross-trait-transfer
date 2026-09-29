#!/usr/bin/env python3
"""Generate number-list teachers, response-only SFT, and cross-trait M10.

This is sampled sequence SFT, not KL distillation of 100 candidate scores.
Generation runs in an isolated vLLM process and keeps one teacher model resident
for the whole trait family. Students never receive the teacher system message.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Iterator

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_long_form_trait_cell import (  # noqa: E402
    CellSpec,
    adapter_has_weights,
    cell_m10_command,
    ensure_relative_directory_link,
    load_cell_spec,
    m10_slugify,
)
from scripts.train_trait_response_sft import (  # noqa: E402
    DEFAULT_TARGET_MODULES,
    build_checkpoint_steps,
    slugify,
    utc_now,
)

POLICY_VERSION = "strict_ascii_ten_number_sequence_sft_v1"
NUMBER_LIST = re.compile(r"[ \t\r\n]*[0-9]{2}(?: [0-9]{2}){9}[ \t\r\n]*")
STAGES = ("preflight", "tokenizer-check", "dataset-check", "generate", "train", "m10", "all", "generation-worker")
RUNTIME_DECODE_OVERRIDE_POLICY = "pinned_mistral_vllm_0_10_detokenize_false_v1"
RUNTIME_ROW_KEY = "generation_runtime_sha256"


def digest(value: Any) -> str:
    """Hash a canonical JSON value, without relying on dictionary ordering."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read a required JSON object."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically publish an artifact without partial JSON on interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _generation_runtime_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject ambiguous duplicate keys in frozen runtime evidence and sidecars."""
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate generation runtime evidence key: {key}")
        value[key] = item
    return value


def _runtime_decode_override(
    original: dict[str, Any], active: dict[str, Any], original_bytes: bytes,
) -> dict[str, Any]:
    """Describe only the audited token-preserving Mistral decoder transition."""
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    revision = "c170c708c41dac9275d15a8fff4eca08d52bab71"
    required = {
        "model_id": model_id, "model_revision": revision,
        "tokenizer_revision": revision, "generation_backend": "vllm",
        "vllm_version": "0.10.0",
        "generation_runtime_policy": "pinned_mistral_vllm_v1",
        "eos_token_ids": [2],
    }
    old_sampling, new_sampling = original.get("sampling"), active.get("sampling")
    valid = (
        all(digest(original.get(key)) == digest(value) for key, value in required.items())
        and isinstance(old_sampling, dict)
        and isinstance(new_sampling, dict)
    )
    if valid:
        valid = (
            old_sampling.get("detokenize", True) is True
            and new_sampling.get("detokenize") is False
            and old_sampling.get("stop") in (None, [])
            and digest(old_sampling.get("stop_token_ids")) == digest([2])
            and old_sampling.get("ignore_eos") is False
            and digest({**original, "sampling": {
                key: value for key, value in old_sampling.items() if key != "detokenize"
            }}) == digest({**active, "sampling": {
                key: value for key, value in new_sampling.items() if key != "detokenize"
            }})
        )
    if not valid:
        raise ValueError("Generation runtime changed; use a new output directory")
    return {
        "policy": RUNTIME_DECODE_OVERRIDE_POLICY,
        "reason": (
            "Disable only vLLM 0.10.0 incremental detokenization for pinned Mistral; "
            "sampling and native EOS stopping are unchanged, no stop strings are used, "
            "and complete sampled token IDs are decoded by the native tokenizer."
        ),
        "original_runtime_sha256": digest(original),
        "active_runtime_sha256": digest(active),
        "original_runtime_file_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "original_runtime": original,
        "active_runtime": active,
        "raw_row_runtime_key": RUNTIME_ROW_KEY,
        "untagged_rows_runtime_sha256": digest(original),
    }


def record_generation_runtime(output_dir: Path, evidence: dict[str, Any]) -> str:
    """Freeze runtime evidence, or record the sole audited decoder-only override.

    The caller holds the generation lock. Existing canonical evidence is never
    rewritten; a deterministic sidecar identifies the runtime of newly sampled
    rows while untagged historical rows retain the original runtime provenance.
    """
    runtime_path = output_dir / "generation_runtime.json"
    override_path = output_dir / "generation_runtime_decode_override.json"
    if runtime_path.is_symlink() or override_path.is_symlink():
        raise ValueError("Generation runtime evidence must not be a symbolic link")
    if not runtime_path.exists():
        if override_path.exists():
            raise ValueError("Orphan generation runtime decode override")
        write_json(runtime_path, evidence)
        return digest(evidence)
    original_bytes = runtime_path.read_bytes()
    original = json.loads(original_bytes, object_pairs_hook=_generation_runtime_object)
    if not isinstance(original, dict):
        raise ValueError("Expected generation runtime JSON object")
    active_hash = digest(evidence)
    if not override_path.exists() and digest(original) == active_hash:
        return active_hash
    override = _runtime_decode_override(original, evidence, original_bytes)
    if override_path.exists():
        saved_override = json.loads(
            override_path.read_bytes(), object_pairs_hook=_generation_runtime_object,
        )
        if digest(saved_override) != digest(override):
            raise ValueError("Generation runtime decode override does not match frozen evidence")
    else:
        write_json(override_path, override)
        logger.warning(
            "Applying audited Mistral decoder override {}; preserving {} and prior raw rows",
            RUNTIME_DECODE_OVERRIDE_POLICY, runtime_path,
        )
    return active_hash


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    """Reject duplicate writers instead of racing over datasets or adapters."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process holds {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def provenance(spec: CellSpec) -> dict[str, Any]:
    """Fingerprint the actual protocol, catalogs, and prompt-generator code."""
    return {
        "policy": POLICY_VERSION,
        "config": spec.payload,
        "traits": list(spec.traits),
        "questions": json.loads(
            (PROJECT_ROOT / spec.payload["questions_path"]).read_text()
        ),
        "prompt_generator_sha256": hashlib.sha256(
            (PROJECT_ROOT / "sl/datasets/nums_dataset.py").read_bytes()
        ).hexdigest(),
    }


def freeze_protocol(spec: CellSpec) -> str:
    """Refuse stale caches after changing the experimental protocol."""
    value = provenance(spec)
    fingerprint = digest(value)
    path = spec.output_dir / "number_sequence_protocol.json"
    if path.exists():
        if read_json(path).get("fingerprint") != fingerprint:
            raise ValueError(f"Protocol drift: use a new output_dir, not {path}")
    else:
        write_json(path, {"fingerprint": fingerprint, **value})
    return fingerprint


def preflight(spec: CellSpec) -> dict[str, Any]:
    """Check the fixed protocol without loading model weights or writing files."""
    required = {
        "carrier": "number_sequence_00_99",
        "answer_count": 10,
        "target_effective_batch_size": 64,
        "training_example_count": 10000,
        "n_epochs": 3,
        "lora_r": 8,
        "lora_alpha": 32,
    }
    for field, expected in required.items():
        if spec.payload.get(field) != expected:
            raise ValueError(f"{field} must be {expected!r} for this fixed comparison")
    if spec.payload.get("training_backend", "unsloth") == "transformers":
        from scripts.native_generation_loader import REPLICATION_REVISIONS

        model_id = spec.payload.get("model_id")
        if model_id not in REPLICATION_REVISIONS:
            raise ValueError("Native number SFT requires an audited replication model")
        if spec.payload.get("model_revision") != REPLICATION_REVISIONS[model_id]:
            raise ValueError("Native model revision must match the pinned loader")
        native_training_policy(spec.payload)
        if spec.payload.get("eval_torch_dtype") != "float32":
            raise ValueError("Native M10 measurement must still use FP32")
        generation_backend = spec.payload.get("generation_backend")
        if (generation_backend not in ("transformers", "vllm")
                or spec.payload.get("generation_dtype") != "bfloat16"):
            raise ValueError("Replication sampled generation requires explicit BF16")
        if generation_backend == "vllm":
            from scripts.native_number_generation import VLLM_RUNTIME_POLICIES

            if (model_id not in VLLM_RUNTIME_POLICIES
                    or spec.payload.get("generation_runtime_policy") != VLLM_RUNTIME_POLICIES[model_id]):
                raise ValueError("Pinned vLLM generation requires its explicit model policy")
        if spec.payload.get("generation_max_batch_size") not in (1, 4, 8, 16, 32, 64, 128, 256, 512):
            raise ValueError("Invalid native generation batch-search maximum")
    elif (spec.payload.get("training_backend", "unsloth") != "unsloth"
          or spec.payload.get("model_id") != "Qwen/Qwen2.5-7B-Instruct"):
        raise ValueError("The Unsloth number-SFT path is specific to Qwen2.5")
    if (spec.payload.get("training_backend", "unsloth") == "unsloth"
            and "model_revision" in spec.payload):
        from scripts.native_generation_loader import REPLICATION_REVISIONS

        if spec.payload["model_revision"] != REPLICATION_REVISIONS[spec.payload["model_id"]]:
            raise ValueError("Explicit Qwen2.5 SFT revision must match the audited pinned model")
    if tuple(spec.payload.get("target_modules", ())) != tuple(DEFAULT_TARGET_MODULES):
        raise ValueError("Use the same seven LoRA projections as the KL sweep")
    for field in (
        "max_generation_attempts",
        "generation_chunk_size",
        "max_new_tokens",
        "generation_max_seq_length",
        "training_max_seq_length",
        "eval_batch_size",
        "training_max_auto_batch_size",
        "checkpoint_count",
    ):
        if not isinstance(spec.payload.get(field), int) or spec.payload[field] <= 0:
            raise ValueError(f"{field} must be a positive integer")
    if spec.payload["max_generation_attempts"] < spec.payload["training_example_count"]:
        raise ValueError(
            "Generation attempt budget is smaller than retained-row target"
        )
    if spec.payload.get("temperature") != 1.0 or spec.payload.get("top_p") != 1.0:
        raise ValueError("The frozen sequence-SFT recipe uses temperature=top_p=1")
    if not 0 < float(spec.payload["generation_gpu_memory_utilization"]) < 1:
        raise ValueError("generation_gpu_memory_utilization must be in (0,1)")
    if spec.payload.get("reuse_generation_source_config"):
        from scripts.number_sft_dataset_reuse import source_spec_for_reuse

        source_spec_for_reuse(spec)
    storage_policy = spec.payload.get("generation_storage_policy", "atomic_raw_monolithic_v1")
    if storage_policy not in {"atomic_raw_monolithic_v1", "atomic_raw_chunks_v1"}:
        raise ValueError("Unknown generation raw-storage policy")
    logger.success(
        "Number-sequence SFT preflight: {} traits, {} retained rows/trait, "
        "10 sampled numbers/row, 3 epochs, response-only CE; no teacher system in FT",
        len(spec.traits),
        spec.payload["training_example_count"],
    )
    return {"status": "passed", "trait_count": len(spec.traits), **required}


def native_training_policy(payload: dict[str, Any]) -> dict[str, Any]:
    """Separate token-SFT compute precision from the strict evaluation policy.

    An explicit versioned policy opts into mixed precision. Old configs and
    their frozen result fingerprints keep their FP32 meaning; changing any
    field requires a new result directory through ``freeze_protocol``.
    """
    dtype = payload.get("training_dtype")
    policy = payload.get("training_precision_policy", "full_fp32_math_v1")
    if (dtype, policy) not in (
        ("float32", "full_fp32_math_v1"),
        ("bfloat16", "bf16_autocast_fp32_master_v1"),
    ):
        raise ValueError("Native training dtype requires a matching explicit precision policy")
    logits_mode = payload.get("training_sft_logits_mode", "full")
    if logits_mode not in ("full", "completion"):
        raise ValueError("Native SFT logits mode must be full or completion")
    audited_models = {
        "allenai/OLMo-2-1124-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen3-VL-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
    }
    if logits_mode == "completion" and payload.get("model_id") not in audited_models:
        raise ValueError("Completion-only LM-head optimization requires an audited native model")
    padding_mode = payload.get("training_sft_padding_attention_mode", "explicit")
    if padding_mode not in ("explicit", "causal"):
        raise ValueError("Training padding attention mode must be explicit or causal")
    if padding_mode == "causal" and payload.get("model_id") not in audited_models:
        raise ValueError("Causal-only right-padding optimization requires an audited native model")
    optimizer_mode = payload.get("training_adamw_implementation", "single_tensor")
    if optimizer_mode not in ("single_tensor", "foreach"):
        raise ValueError("Training AdamW implementation must be single_tensor or foreach")
    for key in ("training_gradient_checkpointing", "training_auto_gradient_checkpointing"):
        if key in payload and not isinstance(payload[key], bool):
            raise ValueError(f"{key} must be boolean")
    return {
        "torch_dtype": dtype,
        "sft_logits_mode": logits_mode,
        "sft_padding_attention_mode": padding_mode,
        "adamw_implementation": optimizer_mode,
        "gradient_checkpointing": payload.get("training_gradient_checkpointing", True),
        "auto_gradient_checkpointing": payload.get("training_auto_gradient_checkpointing", False),
    }


def tokenizer_check(spec: CellSpec) -> dict[str, Any]:
    """Audit pinned native role formatting and label boundaries without weights."""
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig
    from scripts.native_generation_loader import replication_revision_kwargs
    from scripts.native_number_generation import eos_ids
    from scripts.train_native_number_sft import tokenize_response_row
    from scripts.train_trait_response_sft import SftRow

    preflight(spec)
    revision = replication_revision_kwargs(spec.payload["model_id"])
    tokenizer = AutoTokenizer.from_pretrained(spec.payload["model_id"], **revision)
    config = AutoConfig.from_pretrained(spec.payload["model_id"], **revision)
    if config.model_type == "qwen3_vl" and config.text_config.num_hidden_layers != 36:
        raise ValueError("Pinned Qwen3-VL-8B architecture must have 36 language layers")
    generation = GenerationConfig.from_pretrained(spec.payload["model_id"], **revision)
    stops = eos_ids(generation, tokenizer)
    if not tokenizer.is_fast or not tokenizer.chat_template:
        raise ValueError("Native SFT label audit requires a fast chat tokenizer")
    # Cover prompt-format variants and every trait-bearing system message.
    prompts = build_prompts(spec)
    audit_prompts = list(dict.fromkeys([
        *prompts[:128], *sorted(prompts, key=len, reverse=True)[:32],
    ]))
    example = "00 01 12 23 34 45 56 67 78 99"
    supervised_counts: list[int] = []
    for prompt in audit_prompts:
        for completion in (example, f"\n{example}\n", f"  {example}  "):
            row = tokenize_response_row(
                tokenizer, SftRow(prompt=prompt, completion=completion),
                max_seq_length=spec.payload["training_max_seq_length"],
            )
            expected = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt},
                 {"role": "assistant", "content": completion}],
                tokenize=True, add_generation_prompt=False,
            )
            if row.input_ids != expected:
                raise ValueError("SFT tokenization differs from native full conversation")
            if not any(token in stops for token in row.labels if token != -100):
                raise ValueError("Assistant stop token is missing from SFT labels")
            supervised_counts.append(row.supervised_tokens)
    longest_prompt = max(audit_prompts, key=lambda value: len(tokenizer.encode(value)))
    for trait in spec.traits:
        system = spec.payload["system_prompt_template"].format(trait=trait)
        rendered = tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user", "content": longest_prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        if system not in rendered or longest_prompt not in rendered:
            raise ValueError(f"Native teacher template dropped content for {trait}")
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        if len(ids) + spec.payload["max_new_tokens"] > spec.payload["generation_max_seq_length"]:
            raise ValueError("Native teacher prompt exceeds generation context")
    result = {
        "status": "passed", "model_id": spec.payload["model_id"],
        "model_revision": getattr(config, "_commit_hash", None),
        "tokenizer_class": type(tokenizer).__name__,
        "prompt_variants_checked": len(audit_prompts),
        "supervised_token_count_range": [min(supervised_counts), max(supervised_counts)],
        "trait_system_prompts_checked": len(spec.traits), "eos_ids": stops,
    }
    logger.success("Native number tokenizer check: {}", result)
    return result


def build_prompts(spec: CellSpec) -> list[str]:
    """Use the project's original fixed-width 10-number prompt generator."""
    import numpy as np
    from sl.datasets.nums_dataset import PromptGenerator

    generator = PromptGenerator(
        rng=np.random.Generator(np.random.PCG64(spec.payload["prompt_seed"])),
        example_min_count=3,
        example_max_count=9,
        example_min_value=0,
        example_max_value=100,
        answer_count=10,
        answer_max_digits=2,
        answer_exact_count=True,
        example_digit_width=2,
        answer_digit_width=2,
        format_suffixes_space_separated_only=True,
    )
    return [
        generator.sample_query() for _ in range(spec.payload["max_generation_attempts"])
    ]


def rejection_reason(text: str, finish_reason: str | None) -> str | None:
    """Reject whole responses; never sanitize, strip, pad, or extract numbers."""
    if finish_reason != "stop":
        return "not_naturally_completed"
    if not NUMBER_LIST.fullmatch(text):
        return "not_exactly_ten_ascii_two_digit_numbers"
    return None


def validate_raw_rows(
    rows: list[dict[str, Any]],
    prompts: list[str],
    fingerprint: str,
    trait_fingerprint: str,
) -> None:
    """Validate deterministic resume ordering and recheck every cached label."""
    for index, row in enumerate(rows):
        if index >= len(prompts) or row.get("request_index") != index:
            raise ValueError("Raw request indices must be contiguous from zero")
        if (
            row.get("fingerprint") != fingerprint
            or row.get("trait_fingerprint") != trait_fingerprint
            or row.get("prompt") != prompts[index]
        ):
            raise ValueError(f"Raw generation provenance mismatch at request {index}")
        text = row.get("completion")
        if not isinstance(text, str):
            raise ValueError(f"Missing completion at request {index}")
        reason = rejection_reason(text, row.get("finish_reason"))
        if row.get("rejection_reason") != reason or row.get("eligible") != (
            reason is None
        ):
            raise ValueError(f"Invalid cached eligibility at request {index}")


def filtered_rows(rows: list[dict[str, Any]], target: int) -> list[dict[str, str]]:
    """Return unchanged sampled prompt/completion strings, never system messages."""
    return [
        {"prompt": row["prompt"], "completion": row["completion"]}
        for row in rows
        if row["eligible"]
    ][:target]


def write_raw_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Publish whole request batches, leaving the previous batch after a crash."""
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write("".join(json.dumps(row) + "\n" for row in rows))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def publish_dataset(
    root: Path,
    rows: list[dict[str, Any]],
    spec: CellSpec,
    trait: str,
    fingerprint: str,
) -> dict[str, Any]:
    """Atomically publish exact retained budget and reproducibility evidence."""
    selected = filtered_rows(rows, spec.payload["training_example_count"])
    if len(selected) != spec.payload["training_example_count"]:
        raise ValueError(
            f"{trait}: only {len(selected)} eligible rows; refusing shorter FT"
        )
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected)
    path = root / "filtered.jsonl"
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise ValueError(f"Refusing to replace a different frozen SFT dataset: {path}")
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
    manifest = {
        "status": "complete",
        "fingerprint": fingerprint,
        "trait": trait,
        "model_id": spec.payload["model_id"],
        "carrier": spec.payload["carrier"],
        "generated_count": len(rows),
        "filtered_count": len(selected),
        "eligible_count_before_cap": sum(row["eligible"] for row in rows),
        "rejection_counts": dict(
            Counter(row["rejection_reason"] for row in rows if not row["eligible"])
        ),
        "dataset_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "system_prompt_in_training": False,
        "response_sanitization": "none: invalid whole responses rejected",
        "generation_seconds": sum(
            row.get("batch_generation_seconds", 0.0) for row in rows
        ),
        "finished_at": utc_now(),
    }
    write_json(root / "generation_manifest.json", manifest)
    return manifest


def generation_progress(rows: list[dict[str, Any]], target: int, limit: int) -> dict[str, Any]:
    """Describe rejection sampling without relaxing eligibility or the FT budget."""
    accepted = sum(row["eligible"] for row in rows)
    rejections = Counter(
        row["rejection_reason"] for row in rows if not row["eligible"]
    )
    return generation_progress_from_counts(
        attempts=len(rows), accepted=accepted, target=target, limit=limit,
        rejection_counts=rejections,
    )


def generation_progress_from_counts(
    *, attempts: int, accepted: int, target: int, limit: int,
    rejection_counts: dict[str | None, int],
) -> dict[str, Any]:
    """Describe progress using resume-initialized counters, without scanning rows."""
    import math

    rate = accepted / attempts if attempts else 0.0
    return {
        "status": "complete" if accepted >= target else (
            "exhausted" if attempts >= limit else "collecting"
        ),
        "attempts": attempts,
        "attempt_limit": limit,
        "eligible_count": accepted,
        "target_retained_count": target,
        "acceptance_fraction": rate,
        "projected_total_attempts_at_observed_rate": math.ceil(target / rate) if rate else None,
        "rejection_counts": dict(rejection_counts),
    }


def load_pinned_qwen_vllm(spec: CellSpec) -> tuple[Any, Any]:
    """Pin opt-in Qwen runs without changing the established decoding settings.

    Legacy numeric configs still use the original shared loader. Explicit
    revisions must pin both weights and tokenizer, while the student loader
    additionally disables Unsloth's repository-name remapping.
    """
    from vllm import LLM

    p = spec.payload
    runtime = LLM(
        model=p["model_id"], tokenizer=p["model_id"],
        revision=p["model_revision"], tokenizer_revision=p["model_revision"],
        dtype="auto", max_model_len=p["generation_max_seq_length"],
        gpu_memory_utilization=p["generation_gpu_memory_utilization"],
        seed=p["seed"], trust_remote_code=True, enable_prefix_caching=True,
    )
    tokenizer = runtime.get_tokenizer()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return runtime, tokenizer


def generate_datasets(
    spec: CellSpec, fingerprint: str, *, selected_traits: tuple[str, ...] | None = None,
) -> None:
    """Reuse one teacher across traits, isolated from the student process."""
    traits = spec.traits if selected_traits is None else selected_traits
    if not traits or len(set(traits)) != len(traits) or any(t not in spec.traits for t in traits):
        raise ValueError("Selected traits must be a nonempty unique subset of the full catalog")
    if spec.payload.get("reuse_generation_source_config"):
        if selected_traits is not None:
            raise ValueError("Sharded generation cannot import a whole-family source dataset")
        from scripts.number_sft_dataset_reuse import import_frozen_datasets

        import_frozen_datasets(spec, fingerprint)
        return
    prompts = build_prompts(spec)
    native_generation = spec.payload.get("generation_backend") in ("transformers", "vllm")
    runtime: Any = None
    tokenizer: Any = None
    runtime_sha256: str | None = None
    target = spec.payload["training_example_count"]
    for trait in traits:
        root = spec.output_dir / "traits" / slugify(trait) / "data"
        root.mkdir(parents=True, exist_ok=True)
        raw_path = root / "raw.jsonl"
        chunk_store = None
        if spec.payload.get("generation_storage_policy") == "atomic_raw_chunks_v1":
            from scripts.number_sft_raw_storage import AtomicRawChunkStore

            chunk_store = AtomicRawChunkStore(root)
            rows = chunk_store.load_rows()
        else:
            rows = (
                [json.loads(line) for line in raw_path.read_text().splitlines()]
                if raw_path.exists() else []
            )
        system = spec.payload["system_prompt_template"].format(trait=trait)
        trait_fingerprint = digest(
            {
                "protocol": fingerprint,
                "trait": trait,
                "system_prompt": system,
            }
        )
        validate_raw_rows(rows, prompts, fingerprint, trait_fingerprint)
        kept = sum(row["eligible"] for row in rows)
        rejections = Counter(
            row["rejection_reason"] for row in rows if not row["eligible"]
        )
        if kept >= target:
            if chunk_store is not None:
                chunk_store.materialize_raw()
            publish_dataset(root, rows, spec, trait, fingerprint)
            logger.success(
                "Reusing {} complete number-sequence rows for {}", target, trait
            )
            continue
        if runtime is None:
            if native_generation:
                from scripts.native_number_generation import NativeNumberGenerator, VllmNumberGenerator

                runtime_class = (VllmNumberGenerator
                    if spec.payload["generation_backend"] == "vllm"
                    else NativeNumberGenerator)
                runtime = runtime_class(spec.payload)
                tokenizer = runtime.tokenizer
                runtime_evidence = runtime.evidence
            else:
                from scripts.generate_trait_response_sft_dataset import (
                    load_vllm_and_tokenizer,
                )

                if spec.payload.get("model_revision"):
                    runtime, tokenizer = load_pinned_qwen_vllm(spec)
                else:
                    runtime, tokenizer = load_vllm_and_tokenizer(
                        argparse.Namespace(
                            model_id=spec.payload["model_id"],
                            max_seq_length=spec.payload["generation_max_seq_length"],
                            vllm_gpu_memory_utilization=spec.payload[
                                "generation_gpu_memory_utilization"
                            ],
                            seed=spec.payload["seed"], vllm_enable_prefix_caching=True,
                        )
                    )
                import vllm

                model_config = getattr(runtime.llm_engine, "model_config", None)
                hf_config = getattr(model_config, "hf_config", None)
                runtime_evidence = {
                    "model_id": spec.payload["model_id"],
                    "model_revision": getattr(hf_config, "_commit_hash", None),
                    "tokenizer_class": type(tokenizer).__name__,
                    "chat_template_sha256": digest(tokenizer.chat_template),
                    "vllm_version": getattr(vllm, "__version__", None),
                }
                if (spec.payload.get("model_revision")
                        and runtime_evidence["model_revision"] != spec.payload["model_revision"]):
                    raise ValueError("Loaded number-generation model differs from the pinned revision")
            # Only metadata publication is serialized; disjoint trait shards
            # generate concurrently while keeping the full-family fingerprint.
            with (spec.output_dir / "native_metadata.lock").open("a") as metadata_lock:
                fcntl.flock(metadata_lock, fcntl.LOCK_EX)
                runtime_sha256 = record_generation_runtime(spec.output_dir, runtime_evidence)
        while kept < target and len(rows) < len(prompts):
            start = len(rows)
            stop = min(start + spec.payload["generation_chunk_size"], len(prompts))
            messages = [
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompts[index]},
                ]
                for index in range(start, stop)
            ]
            tokenized = [
                tokenizer.apply_chat_template(
                    value,
                    tokenize=True,
                    add_generation_prompt=True,
                )
                for value in messages
            ]
            if any(
                len(ids) + spec.payload["max_new_tokens"]
                > spec.payload["generation_max_seq_length"]
                for ids in tokenized
            ):
                raise ValueError(
                    "A teacher prompt exceeds the generation context budget"
                )
            generation_started = time.perf_counter()
            if native_generation:
                outputs = runtime.generate(tokenized, first_request_index=start)
                if spec.payload["generation_backend"] == "vllm":
                    write_json((root if selected_traits is not None else spec.output_dir) / "generation_scheduler.json", {
                        "backend": "vllm_continuous_batching",
                        "max_num_seqs": spec.payload["generation_max_batch_size"],
                        "fixed_physical_batch_size": None,
                        "enable_prefix_caching": True,
                        "note": "Scheduler capacity, not a measured optimal physical batch.",
                    })
                else:
                    write_json((root if selected_traits is not None else spec.output_dir) / "generation_batch_search.json", {
                        "batch_size": runtime.batch_size, "observations": runtime.probes,
                    })
            else:
                from vllm import SamplingParams

                outputs = runtime.generate(
                    [{"prompt_token_ids": ids} for ids in tokenized],
                    sampling_params=[
                        SamplingParams(
                            n=1,
                            temperature=spec.payload["temperature"],
                            top_p=spec.payload["top_p"],
                            top_k=-1,
                            min_p=0.0,
                            repetition_penalty=1.0,
                            max_tokens=spec.payload["max_new_tokens"],
                            seed=spec.payload["seed"] + index,
                        )
                        for index in range(start, stop)
                    ],
                    use_tqdm=False,
                )
            if len(outputs) != stop - start:
                raise ValueError("Generator returned a different number of requests")
            batch_generation_seconds = time.perf_counter() - generation_started
            batch = []
            for index, output, expected_ids in zip(
                range(start, stop), outputs, tokenized
            ):
                if (
                    list(output.prompt_token_ids) != list(expected_ids)
                    or len(output.outputs) != 1
                ):
                    raise ValueError("Generation request order or sample count mismatch")
                sample = output.outputs[0]
                text = str(sample.text)
                reason = rejection_reason(text, sample.finish_reason)
                if reason is None:
                    student_ids = tokenizer.apply_chat_template(
                        [
                            {"role": "user", "content": prompts[index]},
                            {"role": "assistant", "content": text},
                        ],
                        tokenize=True,
                        add_generation_prompt=False,
                    )
                    if len(student_ids) > spec.payload["training_max_seq_length"]:
                        raise ValueError(
                            "Accepted sequence would truncate during student SFT"
                        )
                batch.append(
                    {
                        "request_index": index,
                        "fingerprint": fingerprint,
                        "trait_fingerprint": trait_fingerprint,
                        RUNTIME_ROW_KEY: runtime_sha256,
                        "prompt": prompts[index],
                        "completion": text,
                        "finish_reason": sample.finish_reason,
                        "stop_reason": sample.stop_reason,
                        "completion_token_ids": list(sample.token_ids),
                        **({
                            "sampling_seed": output.sampling_seed,
                            "physical_batch_size": output.physical_batch_size,
                        } if native_generation else {}),
                        "eligible": reason is None,
                        "rejection_reason": reason,
                        "batch_generation_seconds": batch_generation_seconds
                        if index == start
                        else 0.0,
                    }
                )
            # Preserve legacy storage; large rejection-sampling runs opt into
            # immutable batches instead of rewriting the growing file each time.
            if chunk_store is None:
                write_raw_rows(raw_path, [*rows, *batch])
            else:
                chunk_store.append_batch(batch)
            rows.extend(batch)
            kept += sum(row["eligible"] for row in batch)
            rejections.update(
                row["rejection_reason"] for row in batch if not row["eligible"]
            )
            write_json(root / "generation_progress.json", generation_progress_from_counts(
                attempts=len(rows), accepted=kept, target=target, limit=len(prompts),
                rejection_counts=rejections,
            ))
            if native_generation and len(rows) >= 512 and kept == 0:
                raise RuntimeError(
                    f"{trait}: zero valid ten-number responses in {len(rows)} attempts; "
                    f"inspect {'raw_chunks/' if chunk_store else 'raw.jsonl'} "
                    "before spending the full generation budget"
                )
            logger.info(
                "{}: sampled {}, retained {}/{}", trait, len(rows), kept, target
            )
        if chunk_store is not None:
            chunk_store.materialize_raw()
        if kept < target:
            progress = generation_progress_from_counts(
                attempts=len(rows), accepted=kept, target=target, limit=len(prompts),
                rejection_counts=rejections,
            )
            write_json(root / "generation_progress.json", progress)
            raise ValueError(
                f"{trait}: only {kept}/{target} eligible rows after {len(rows)} attempts "
                f"({progress['acceptance_fraction']:.1%} acceptance); refusing shorter FT. "
                "See generation_progress.json and raw.jsonl. Increase the attempt budget "
                "in a new output/config; do not weaken the filter or truncate responses."
            )
        publish_dataset(root, rows, spec, trait, fingerprint)


def validate_dataset(spec: CellSpec, trait: str, fingerprint: str) -> dict[str, Any]:
    """Check contents, row count, protocol, and no-system schema before training."""
    root = spec.output_dir / "traits" / slugify(trait) / "data"
    manifest = read_json(root / "generation_manifest.json")
    content = (root / "filtered.jsonl").read_bytes()
    if (
        manifest.get("status") != "complete"
        or manifest.get("fingerprint") != fingerprint
        or manifest.get("trait") != trait
        or manifest.get("dataset_sha256") != hashlib.sha256(content).hexdigest()
    ):
        raise ValueError(f"Dataset provenance mismatch for {trait}")
    rows = [json.loads(line) for line in content.decode("utf-8").splitlines()]
    if len(rows) != spec.payload["training_example_count"]:
        raise ValueError(f"Incorrect retained row count for {trait}")
    for row in rows:
        if set(row) != {"prompt", "completion"} or not isinstance(row["prompt"], str):
            raise ValueError("FT rows must contain only prompt/completion strings")
        if rejection_reason(row["completion"], "stop") is not None:
            raise ValueError("Invalid number sequence in frozen FT dataset")
    return manifest


def training_command(spec: CellSpec, trait: str, output_dir: Path) -> list[str]:
    """Use the established response-only CE trainer with batch-size search."""
    p = spec.payload
    command = [
        sys.executable,
        str(PROJECT_ROOT / ("scripts/train_native_number_sft.py"
            if p.get("training_backend") == "transformers"
            else "scripts/train_trait_response_sft.py")),
    ]
    options = {
        "dataset-path": spec.output_dir
        / "traits"
        / slugify(trait)
        / "data/filtered.jsonl",
        "output-dir": output_dir,
        "trait": trait,
        "model-id": p["model_id"],
        "max-rows": p["training_example_count"],
        "min-rows": p["training_example_count"],
        "n-epochs": p["n_epochs"],
        "max-seq-length": p["training_max_seq_length"],
        "per-device-train-batch-size": 4,
        "gradient-accumulation-steps": 16,
        "target-effective-batch-size": p["target_effective_batch_size"],
        "max-auto-batch-size": p["training_max_auto_batch_size"],
        "batch-probe-steps": 3,
        "memory-headroom-fraction": 0.12,
        "lr": p["learning_rate"],
        "warmup-steps": p["warmup_steps"],
        "seed": p["seed"],
        "lora-r": p["lora_r"],
        "lora-alpha": p["lora_alpha"],
        "target-modules": ",".join(p["target_modules"]),
        "checkpoint-count": p["checkpoint_count"],
    }
    for key, value in options.items():
        command.extend([f"--{key}", str(value)])
    if p.get("training_backend") == "transformers":
        policy = native_training_policy(p)
        command.extend(["--torch-dtype", policy["torch_dtype"]])
        command.extend(["--sft-logits-mode", policy["sft_logits_mode"]])
        command.extend(["--sft-padding-attention-mode", policy["sft_padding_attention_mode"]])
        command.extend(["--adamw-implementation", policy["adamw_implementation"]])
        command.append("--gradient-checkpointing" if policy["gradient_checkpointing"]
                       else "--no-gradient-checkpointing")
        command.append("--auto-gradient-checkpointing" if policy["auto_gradient_checkpointing"]
                       else "--no-auto-gradient-checkpointing")
    command.extend(["--auto-batch-size", "--disable-wandb"])
    if (p.get("carrier") == "number_sequence_00_99"
            and p.get("training_backend", "unsloth") == "unsloth"
            and p.get("model_revision")):
        command.extend(["--model-revision", p["model_revision"]])
    return command


def validate_training(spec: CellSpec, trait: str, root: Path) -> dict[str, Any]:
    """Reject incomplete adapters or incorrect schedules before using M10."""
    manifest = read_json(root / "training_manifest.json")
    p = spec.payload
    expected = {
        "status": "completed",
        "model_id": p["model_id"],
        "trait": trait,
        "dataset_rows": p["training_example_count"],
        "n_epochs": p["n_epochs"],
        "lora_r": p["lora_r"],
        "lora_alpha": p["lora_alpha"],
        "effective_batch_size": p["target_effective_batch_size"],
        "max_seq_length": p["training_max_seq_length"],
        "actual_total_steps": 471,
    }
    pinned_unsloth = (
        p.get("carrier") == "number_sequence_00_99"
        and p.get("training_backend", "unsloth") == "unsloth"
        and bool(p.get("model_revision"))
    )
    if pinned_unsloth:
        expected["model_revision"] = p["model_revision"]
    if p.get("training_backend") == "transformers":
        policy = native_training_policy(p)
        expected.update(training_backend="transformers", torch_dtype=policy["torch_dtype"],
                        model_revision=p["model_revision"],
                        attention_policy=("math_sdpa_tf32_disabled"
                            if policy["torch_dtype"] == "float32"
                            else "sdpa_eligible_flash_memory_efficient_math_tf32_disabled"))
        # Legacy FP32 manifests predate these fields. Only legacy full-head,
        # fixed-checkpointing configs may reuse them.
        if (p.get("training_precision_policy") is not None
                or policy["sft_logits_mode"] != "full"
                or policy["auto_gradient_checkpointing"]
                or not policy["gradient_checkpointing"]):
            expected.update(
                master_torch_dtype="float32",
                training_precision_policy=("full_fp32_math_v1"
                    if policy["torch_dtype"] == "float32"
                    else "bf16_autocast_fp32_master_v1"),
                autocast_enabled=policy["torch_dtype"] == "bfloat16",
                trainable_parameter_dtype="float32",
                gradient_dtype="float32",
                optimizer_state_dtype="float32",
                cross_entropy_dtype="float32",
                sft_logits_mode=policy["sft_logits_mode"],
                requested_gradient_checkpointing=policy["gradient_checkpointing"],
                auto_gradient_checkpointing=policy["auto_gradient_checkpointing"],
            )
        if "training_sft_padding_attention_mode" in p:
            expected.update(
                sft_padding_attention_mode=policy["sft_padding_attention_mode"],
                right_padding_policy="collate_verified_suffix_padding",
                attention_mask_policy=("explicit_key_padding_mask"
                    if policy["sft_padding_attention_mode"] == "explicit"
                    else "native_causal_mask_without_redundant_right_padding_mask"),
            )
        if "training_adamw_implementation" in p:
            expected["adamw_implementation"] = policy["adamw_implementation"]
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"{trait} training {key} does not match {value!r}")
    steps = build_checkpoint_steps(
        total_steps=471,
        checkpoint_count=p["checkpoint_count"],
        steps_per_epoch=157,
        n_epochs=3,
    )
    if manifest.get("checkpoint_steps") != steps:
        raise ValueError(f"{trait} checkpoint schedule mismatch")
    adapters = [root / "adapters" / slugify(trait)]
    for step in steps:
        checkpoint = root / "trainer" / slugify(trait) / f"checkpoint-step-{step:06d}"
        if read_json(checkpoint / "metadata.json").get("global_step") != step:
            raise ValueError(f"{trait} incomplete checkpoint {step}")
        adapters.append(checkpoint / "adapter")
    for adapter in adapters:
        config = read_json(adapter / "adapter_config.json")
        if pinned_unsloth and (
            config.get("base_model_name_or_path") != p["model_id"]
            or config.get("revision") != p["model_revision"]
        ):
            raise ValueError(f"Saved adapter must name its pinned portable base/revision: {adapter}")
        if (
            not adapter_has_weights(adapter)
            or config.get("r") != p["lora_r"]
            or config.get("lora_alpha") != p["lora_alpha"]
        ):
            raise ValueError(f"Invalid saved adapter: {adapter}")
        modules = config.get("target_modules", [])
        if p["model_id"] == "Qwen/Qwen3-VL-8B-Instruct":
            # PEFT can shorten 252 explicit paths to seven equivalent leaves.
            # Prove their resolution against the pinned native module tree,
            # then check all saved A/B keys and shapes; do not trust strings.
            from scripts.native_lora_validation import validate_qwen3_vl_adapter

            validate_qwen3_vl_adapter(
                adapter, target_modules=p["target_modules"], rank=p["lora_r"],
                alpha=p["lora_alpha"], model_id=p["model_id"],
                revision=p["model_revision"],
            )
        elif set(modules) != set(p["target_modules"]):
            raise ValueError(f"Incorrect saved LoRA module coverage: {adapter}")
    return manifest


def completed_training(
    spec: CellSpec,
    trait: str,
    fingerprint: str,
    *,
    train: bool,
) -> Path:
    """Reuse finished traits; restart interrupted traits in fresh attempt paths.

    The shared trainer saves analysis adapters, not optimizer state. A fresh
    attempt prevents old analysis checkpoints mixing with a restarted fit.
    """
    dataset = validate_dataset(spec, trait, fingerprint)
    parent = spec.output_dir / "traits" / slugify(trait) / "training_attempts"
    expected_input = {
        "fingerprint": fingerprint,
        "dataset_sha256": dataset["dataset_sha256"],
    }
    attempts = sorted(parent.glob("attempt-*")) if parent.exists() else []
    for attempt in attempts:
        if read_json(attempt / "training_input.json") != expected_input:
            raise ValueError(f"Training input provenance drift: {attempt}")
        if (attempt / "training_manifest.json").exists():
            validate_training(spec, trait, attempt)
            logger.success("Reusing completed SFT for {}", trait)
            return attempt
    if not train:
        raise FileNotFoundError(f"No completed SFT for {trait}")
    attempt = parent / f"attempt-{len(attempts):04d}"
    write_json(attempt / "training_input.json", expected_input)
    command = training_command(spec, trait, attempt)
    logger.info("Training command: {}", shlex.join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    validate_training(spec, trait, attempt)
    return attempt


def collect_training(spec: CellSpec, fingerprint: str) -> None:
    """Create validated canonical M10 links, including multiword city slugs."""
    records = []
    for trait in spec.traits:
        root = completed_training(spec, trait, fingerprint, train=False)
        for category in ("adapters", "trainer"):
            ensure_relative_directory_link(
                root / category / slugify(trait),
                spec.output_dir
                / "consolidated_training"
                / category
                / m10_slugify(trait),
            )
        records.append({"trait": trait, "training_output_dir": str(root)})
    write_json(
        spec.output_dir / "consolidated_training/collection_manifest.json",
        {
            "status": "validated",
            "fingerprint": fingerprint,
            "records": records,
            "trait_count": len(spec.traits),
            "model_id": spec.payload["model_id"],
        },
    )


def run(spec: CellSpec, stage: str, execute: bool) -> None:
    """Run a complete family on one GPU, or inspect it without GPU work."""
    preflight(spec)
    if stage == "preflight":
        return
    if stage == "tokenizer-check":
        tokenizer_check(spec)
        return
    if stage == "dataset-check":
        from scripts.number_sft_dataset_reuse import verify_source_datasets

        _, _, datasets = verify_source_datasets(spec)
        logger.success("Verified {} complete source datasets without generation or writes", len(datasets))
        return
    if not execute:
        logger.info("Dry run: {} {} (pass --execute to run)", stage, spec.config_path)
        return
    lock_name = "generation.lock" if stage == "generation-worker" else "pipeline.lock"
    with exclusive_lock(spec.output_dir / lock_name):
        fingerprint = freeze_protocol(spec)
        started = time.perf_counter()
        if stage == "generation-worker":
            generate_datasets(spec, fingerprint)
            return
        timing = {"stage": stage, "started_at": utc_now(), "status": "running"}
        write_json(spec.output_dir / f"pipeline_timing_{stage}.json", timing)
        try:
            if stage in {"generate", "all"}:
                generation_python = (sys.executable
                    if spec.payload.get("generation_backend") == "transformers"
                    else os.environ.get("VLLM_PYTHON"))
                if not generation_python or not Path(generation_python).is_file():
                    raise ValueError(
                        "Set VLLM_PYTHON to the isolated vLLM environment Python"
                    )
                subprocess.run(
                    [
                        generation_python,
                        str(Path(__file__).resolve()),
                        "--config",
                        str(spec.config_path),
                        "--stage",
                        "generation-worker",
                        "--execute",
                    ],
                    cwd=PROJECT_ROOT,
                    check=True,
                )
            if stage in {"train", "all"}:
                for trait in spec.traits:
                    completed_training(spec, trait, fingerprint, train=True)
                collect_training(spec, fingerprint)
            if stage in {"m10", "all"}:
                collect_training(spec, fingerprint)
                command = cell_m10_command(spec)
                if spec.payload.get("eval_torch_dtype"):
                    command[command.index("--torch-dtype") + 1] = spec.payload["eval_torch_dtype"]
                logger.info("M10 command: {}", shlex.join(command))
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            timing["status"] = "completed"
        except Exception as exc:
            timing.update(status="failed", error=str(exc))
            raise
        finally:
            timing.update(
                finished_at=utc_now(), duration_seconds=time.perf_counter() - started
            )
            write_json(spec.output_dir / f"pipeline_timing_{stage}.json", timing)


def main() -> None:
    """Parse arguments without importing either GPU framework at preflight."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", choices=STAGES, default="preflight")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    run(load_cell_spec(args.config), args.stage, args.execute)


if __name__ == "__main__":
    main()
