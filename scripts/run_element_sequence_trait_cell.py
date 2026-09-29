#!/usr/bin/env python3
"""Sample native chemical-code sequences, response-only SFT, and cross-trait M10.

No constrained decoding, number-to-symbol output mapping, or distribution
distillation is performed. The fixed numeric *prompt example stream* is mapped
to the existing first-100 element codes before native autoregressive generation.
Codes such as H_ and He may each span multiple native model tokens.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import fcntl
from functools import lru_cache
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
from types import SimpleNamespace

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from scripts import build_element_symbol_sequence_dataset as elements  # noqa: E402
from scripts import run_number_sequence_trait_cell as number  # noqa: E402
from scripts.run_long_form_trait_cell import (  # noqa: E402
    CellSpec, cell_m10_command,
    ensure_relative_directory_link, load_cell_spec, m10_slugify,
)
from scripts.number_sft_raw_storage import AtomicRawChunkStore  # noqa: E402
from scripts.train_trait_response_sft import slugify, utc_now  # noqa: E402

POLICY_VERSION = "element100_native_sampled_sequence_sft_v1"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
MANIFEST = "experiment_plan/carriers/qwen25_element_symbols_100.json"
QWEN3VL_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
QWEN3VL_MANIFEST = "experiment_plan/carriers/qwen3vl_element_symbols_100.json"
STAGES = ("preflight", "tokenizer-check", "dataset-check", "pilot",
          "generate", "train", "m10", "all", "generation-worker")
digest = number.digest
read_json = number.read_json
write_json = number.write_json


@contextmanager
def metadata_lock(path: Path) -> Iterator[None]:
    """Serialize short shared metadata writes across independent trait workers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def selected_trait_names(spec: CellSpec, selected_traits: tuple[str, ...] | None) -> tuple[str, ...]:
    """Select work without changing the complete catalog or scientific identity."""
    if selected_traits is None:
        return spec.traits
    if (not selected_traits or len(set(selected_traits)) != len(selected_traits)
            or not set(selected_traits).issubset(spec.traits)):
        raise ValueError("Selected traits must be a nonempty unique subset of the full catalog")
    return tuple(trait for trait in spec.traits if trait in selected_traits)


def carrier_records(spec: CellSpec) -> tuple[dict[str, Any], ...]:
    """Require the canonical first-100 chemical codes and ordered numeric mapping."""
    value = read_json(PROJECT_ROOT / spec.payload["candidate_manifest"])
    rows = value.get("candidates", [])
    if (value.get("model_id") != spec.payload["model_id"]
            or value.get("candidate_count") != 100 or len(rows) != 100):
        raise ValueError("Element manifest must match the model and contain 100 candidates")
    for index, (row, symbol, code) in enumerate(zip(
            rows, elements.ELEMENT_SYMBOLS, elements.FIXED_WIDTH_ELEMENT_CODES)):
        expected = {"label": symbol, "text": code, "atomic_number": index + 1,
                    "source_number": f"{index:02d}"}
        if (not isinstance(row, dict) or any(row.get(key) != item for key, item in expected.items())
                or type(row.get("atomic_number")) is not int or "token_id" in row):
            raise ValueError(f"Element manifest differs from canonical mapping at {index:02d}")
    return tuple(rows)


def build_prompts(spec: CellSpec) -> list[str]:
    """Map the numeric example RNG stream, then request ten unforced element codes.

    The legacy generator's upper count bound is exclusive: examples number 3–8.
    Vocabulary order is fixed and identical across teachers/domains/neutral.
    """
    symbols = tuple(row["text"] for row in carrier_records(spec))
    return list(_mapped_prompts(spec.payload["prompt_seed"],
                                spec.payload["max_generation_attempts"], symbols))


@lru_cache(maxsize=3)
def _mapped_prompts(seed: int, count: int, tokens: tuple[str, ...]) -> tuple[str, ...]:
    """Share one deterministic prompt stream across repeated dataset validations."""
    prompts = []
    source = SimpleNamespace(payload={"prompt_seed": seed, "max_generation_attempts": count})
    for numeric in number.build_prompts(source):
        examples = re.findall(r"\b[0-9]{2}\b", numeric.split(".", 1)[0])
        if not 3 <= len(examples) <= 8:
            raise ValueError("Unexpected numeric prompt source example format")
        symbols = "\n".join(tokens[int(value)] for value in examples)
        prompts.append(
            "Use chemical element codes for atomic numbers 1-100 (H through Fm). "
            'Pad one-letter symbols with "_" (for example, H_ and C_); '
            "leave two-letter symbols unchanged (for example, He and Fe).\n\n"
            "Here is a sequence of element codes:\n"
            f"{symbols}\n\n"
            "Continue the sequence with exactly 10 new element codes. "
            "Write one code per line. Start immediately with the first code. "
            "Output only the 10 codes, with no numbering, spaces, or explanation."
        )
    return tuple(prompts)


def provenance(spec: CellSpec) -> dict[str, Any]:
    """Bind catalogs, implementation, and the entire frozen prompt protocol."""
    value = {
        "policy": POLICY_VERSION,
        "config": spec.payload,
        "traits": list(spec.traits),
        "questions": json.loads((PROJECT_ROOT / spec.payload["questions_path"]).read_text()),
        "carrier_manifest": read_json(PROJECT_ROOT / spec.payload["candidate_manifest"]),
        "source_sha256": {
            path: hashlib.sha256((PROJECT_ROOT / path).read_bytes()).hexdigest()
            for path in (
                "scripts/run_element_sequence_trait_cell.py",
                "scripts/build_element_symbol_sequence_dataset.py",
                "scripts/run_number_sequence_trait_cell.py",
                "scripts/train_trait_response_sft.py",
                "sl/datasets/nums_dataset.py",
            )
        },
        "prompt_policy": "numeric_seed42_example_stream_3_to_8_mapped_first100_element_codes_newline_v1",
        "sampling_policy": "native_full_vocabulary_autoregressive_no_constraints",
        "filter_policy": "exactly_ten_manifest_element_codes_newline_separated_native_eos",
        "sft_tokenization_policy": "native_chat_template_exact_text_response_only_canonical_retokenization",
    }
    if spec.payload.get("model_id") == QWEN3VL_MODEL_ID:
        value["native_generation_pilot_policy"] = "first64_requests_each_teacher_reused_as_raw_rows_v1"
        for path in (
            "scripts/native_number_generation.py", "scripts/native_generation_loader.py",
            "scripts/train_native_number_sft.py", "scripts/distribution_training_backend.py",
            "scripts/native_lora_validation.py",
        ):
            value["source_sha256"][path] = hashlib.sha256((PROJECT_ROOT / path).read_bytes()).hexdigest()
    return value


def freeze_protocol(spec: CellSpec) -> str:
    """Write immutable protocol evidence or reject a resumed run's drift."""
    value = provenance(spec)
    fingerprint = digest(value)
    path = spec.output_dir / "element_sequence_protocol.json"
    if path.exists():
        if read_json(path) != {"fingerprint": fingerprint, **value}:
            raise ValueError(f"Protocol drift: use a new output directory, not {path}")
    else:
        write_json(path, {"fingerprint": fingerprint, **value})
    return fingerprint


def preflight(spec: CellSpec) -> dict[str, Any]:
    """Validate all scientific invariants without model loading or file writes."""
    required = {
        "carrier": "element_sequence_100", "candidate_manifest": MANIFEST,
        "model_id": "Qwen/Qwen2.5-7B-Instruct", "model_revision": MODEL_REVISION,
        "training_backend": "unsloth", "generation_backend": "element_vllm",
        "answer_count": 10, "training_example_count": 10000,
        "target_effective_batch_size": 64, "n_epochs": 3,
        "lora_r": 8, "lora_alpha": 32, "temperature": 1.0, "top_p": 1.0,
        "generation_dtype": "bfloat16", "generation_storage_policy": "atomic_raw_chunks_v1",
        "generation_enforce_eager": True, "generation_pilot_count": 64,
        "prompt_seed": 42, "seed": 1, "learning_rate": 0.0002, "warmup_steps": 5,
        "eval_torch_dtype": "float32",
    }
    if spec.payload.get("model_id") == QWEN3VL_MODEL_ID:
        from scripts.native_generation_loader import REPLICATION_REVISIONS

        required.update(
            model_id=QWEN3VL_MODEL_ID,
            model_revision=REPLICATION_REVISIONS[QWEN3VL_MODEL_ID],
            candidate_manifest=QWEN3VL_MANIFEST,
            training_backend="transformers", generation_backend="transformers",
            training_dtype="bfloat16",
            training_precision_policy="bf16_autocast_fp32_master_v1",
        )
        del required["generation_enforce_eager"]
        number.native_training_policy(spec.payload)
        if spec.payload.get("generation_max_batch_size") not in (1, 4, 8, 16, 32, 64, 128, 256, 512):
            raise ValueError("Invalid native generation batch-search maximum")
    for key, value in required.items():
        if (spec.payload.get(key) != value
                or isinstance(value, bool) and type(spec.payload.get(key)) is not bool):
            raise ValueError(f"{key} must be {value!r}")
    if tuple(spec.payload["target_modules"]) != tuple(number.DEFAULT_TARGET_MODULES):
        raise ValueError("Use the established seven LoRA projection modules")
    for key in ("max_generation_attempts", "generation_chunk_size", "max_new_tokens",
                "generation_max_seq_length", "training_max_seq_length", "eval_batch_size",
                "training_max_auto_batch_size", "checkpoint_count", "generation_pilot_count"):
        if type(spec.payload.get(key)) is not int or spec.payload[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if spec.payload["max_generation_attempts"] < spec.payload["training_example_count"]:
        raise ValueError("Attempt budget smaller than the retained-row target")
    if not 0 < float(spec.payload["generation_gpu_memory_utilization"]) < 1:
        raise ValueError("Invalid vLLM memory fraction")
    carrier_records(spec)
    logger.success("Element token-SFT protocol validated for {} traits", len(spec.traits))
    return {"status": "passed", "trait_count": len(spec.traits), **required}


def rejection_reason(text: str, finish_reason: str | None,
                     native_eos: bool, carriers: tuple[dict[str, Any], ...]) -> str | None:
    """Reject whole outputs; never edit sampled symbol sequences or whitespace."""
    if finish_reason != "stop" or not native_eos:
        return "not_native_eos_completed"
    # Match whole codes, not individual characters; never strip/repair output.
    codes = "(?:" + "|".join(re.escape(row["text"]) for row in carriers) + ")"
    if re.fullmatch(rf"{codes}(?:\n{codes}){{9}}\n?", text) is None:
        return "not_exactly_ten_newline_separated_manifest_element_codes"
    return None


def audit_student_row(tokenizer: Any, prompt: str, completion: str,
                      max_length: int, eos_ids: list[int]) -> dict[str, Any]:
    """Prove exact content, native prefix boundary, completion labels, and no truncation."""
    messages = [{"role": "user", "content": prompt}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": completion}],
        tokenize=False, add_generation_prompt=False,
    )
    if not full.startswith(prefix + completion):
        raise ValueError("Native chat template changed the sampled response")
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    full_ids = tokenizer.encode(full, add_special_tokens=False)
    if full_ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError("Completion boundary merges prompt content; unsafe CE mask")
    content_ids = tokenizer.encode(completion, add_special_tokens=False)
    if (tokenizer.decode(content_ids, skip_special_tokens=False,
                         clean_up_tokenization_spaces=False) != completion
            or full_ids[len(prefix_ids):len(prefix_ids) + len(content_ids)] != content_ids):
        raise ValueError("Native chat response tokenization does not preserve exact content")
    if len(full_ids) > max_length:
        raise ValueError("Accepted element sequence would truncate during SFT")
    labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids):]
    if not any(value in eos_ids for value in labels if value != -100):
        raise ValueError("Assistant termination is missing from supervised response")
    return {"length": len(full_ids), "prompt_length": len(prefix_ids),
            "supervised_tokens": len(full_ids) - len(prefix_ids)}


def audit_tokenizer(spec: CellSpec, tokenizer: Any, eos_ids: list[int]) -> dict[str, Any]:
    """Audit variable-length code encodings and the entire student prompt stream.

    Every live teacher batch is additionally length-checked before generation;
    every accepted student response is checked before storage. No one-token
    assumption or native sampled-ID/canonical-ID equality is imposed.
    """
    if not tokenizer.is_fast or not tokenizer.chat_template:
        raise ValueError("Element SFT requires the native fast chat tokenizer")
    if not eos_ids or any(type(value) is not int for value in eos_ids):
        raise ValueError("Native generation EOS IDs must be explicit")
    carriers = carrier_records(spec)
    prompts = build_prompts(spec)
    if not prompts:
        raise ValueError("Element prompt stream cannot be empty")
    longest_index, longest_length = max(
        ((index, len(tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True))) for index, prompt in enumerate(prompts)),
        key=lambda pair: pair[1],
    )
    longest = prompts[longest_index]
    encodings = []
    for row in carriers:
        text = row["text"]
        content_ids = tokenizer.encode(text, add_special_tokens=False)
        if not content_ids or any(token_id in tokenizer.all_special_ids for token_id in content_ids):
            raise ValueError("Element vocabulary contains a special token")
        if tokenizer.decode(content_ids, skip_special_tokens=False,
                            clean_up_tokenization_spaces=False) != text:
            raise ValueError("Element code is not losslessly decoded")
        encodings.append({"label": row["label"], "text": text, "token_ids": content_ids})
        for system in (None, spec.payload["system_prompt_template"].format(trait=spec.traits[0])):
            messages = ([] if system is None else [{"role": "system", "content": system}])
            messages.append({"role": "user", "content": longest})
            prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            ids = tokenizer.encode(prefix + text + "\n" + text, add_special_tokens=False)
            if ids[:len(prefix_ids)] != prefix_ids:
                raise ValueError("Element continuation merges the native assistant boundary")
            if tokenizer.decode(ids[len(prefix_ids):], skip_special_tokens=False,
                                clean_up_tokenization_spaces=False) != text + "\n" + text:
                raise ValueError("Element code text changes adjacent to newline")
        for suffix in ("", "\n"):
            audit_student_row(tokenizer, longest, "\n".join([text] * 10) + suffix,
                              spec.payload["training_max_seq_length"], eos_ids)
    max_teacher = 0
    for trait in spec.traits:
        ids = tokenizer.apply_chat_template(
            [{"role": "system", "content": spec.payload["system_prompt_template"].format(trait=trait)},
             {"role": "user", "content": longest}], tokenize=True, add_generation_prompt=True,
        )
        max_teacher = max(max_teacher, len(ids))
    if max_teacher + spec.payload["max_new_tokens"] > spec.payload["generation_max_seq_length"]:
        raise ValueError("Element teacher prompt exceeds generation context")
    return {"status": "passed", "carrier_count": 100, "carrier_encodings": encodings,
            "min_code_tokens": min(len(row["token_ids"]) for row in encodings),
            "max_code_tokens": max(len(row["token_ids"]) for row in encodings),
            "audited_student_prompt_count": len(prompts),
            "longest_student_prompt_index": longest_index,
            "max_student_prefix_tokens": longest_length,
            "max_audited_teacher_tokens": max_teacher,
            "runtime_length_checks": "every_teacher_batch_and_every_accepted_student",
            "eos_ids": eos_ids, "model_revision": spec.payload["model_revision"]}


def tokenizer_check(spec: CellSpec) -> dict[str, Any]:
    """Audit pinned tokenizer and response boundaries without model weights."""
    from transformers import AutoTokenizer, GenerationConfig
    from scripts.native_number_generation import eos_ids

    preflight(spec)
    kwargs = {"revision": spec.payload["model_revision"]}
    tokenizer = AutoTokenizer.from_pretrained(spec.payload["model_id"], **kwargs)
    generation = GenerationConfig.from_pretrained(spec.payload["model_id"], **kwargs)
    result = audit_tokenizer(spec, tokenizer, eos_ids(generation, tokenizer))
    logger.success("Element tokenizer audit: {}", result)
    return result


def validate_raw_rows(rows: list[dict[str, Any]], prompts: list[str], fingerprint: str,
                      trait_fingerprint: str, carriers: tuple[dict[str, Any], ...]) -> None:
    """Recheck immutable request ordering, protocol identity, and eligibility."""
    for index, row in enumerate(rows):
        if (index >= len(prompts) or row.get("request_index") != index
                or row.get("prompt") != prompts[index]
                or row.get("fingerprint") != fingerprint
                or row.get("trait_fingerprint") != trait_fingerprint):
            raise ValueError(f"Raw generation provenance mismatch at request {index}")
        if not isinstance(row.get("completion"), str):
            raise ValueError("Raw row has no exact sampled completion")
        reason = rejection_reason(row["completion"], row.get("finish_reason"),
                                  row.get("native_eos") is True, carriers)
        if row.get("rejection_reason") != reason or row.get("eligible") != (reason is None):
            raise ValueError("Cached element eligibility is inconsistent")


def _trait_fingerprint(spec: CellSpec, trait: str, fingerprint: str) -> str:
    """Hash the rendered teacher identity, not only a display label."""
    return digest({"protocol": fingerprint, "trait": trait,
                   "system_prompt": spec.payload["system_prompt_template"].format(trait=trait)})


def validate_runtime_rows(rows: list[dict[str, Any]], evidence: dict[str, Any],
                          tokenizer: Any | None = None) -> None:
    """Recheck native stopping evidence and, when loaded, exact sampled-ID decoding."""
    stops = evidence["eos_ids"]
    for row in rows:
        ids = row.get("completion_token_ids")
        if (not isinstance(ids, list) or any(type(value) is not int for value in ids)
                or row.get("generation_runtime_sha256") != digest(evidence)):
            raise ValueError("Raw generation runtime identity or sampled token IDs mismatch")
        native = row.get("finish_reason") == "stop" and (
            bool(ids and ids[-1] in stops) or row.get("stop_reason") in stops
        )
        if row.get("native_eos") != native:
            raise ValueError("Saved native EOS flag contradicts sampled stopping evidence")
        if tokenizer is not None:
            content_ids = ids[:-1] if ids and ids[-1] in stops else ids
            if tokenizer.decode(content_ids, skip_special_tokens=False,
                                clean_up_tokenization_spaces=False) != row["completion"]:
                raise ValueError("Saved sampled token IDs do not decode to the exact response")


def validate_dataset(spec: CellSpec, trait: str, fingerprint: str) -> dict[str, Any]:
    """Verify dataset bytes and that selection is precisely the first valid samples."""
    root = spec.output_dir / "traits" / slugify(trait) / "data"
    manifest = read_json(root / "generation_manifest.json")
    content = (root / "filtered.jsonl").read_bytes()
    if (manifest.get("status") != "complete" or manifest.get("fingerprint") != fingerprint
            or manifest.get("trait") != trait
            or manifest.get("dataset_sha256") != hashlib.sha256(content).hexdigest()):
        raise ValueError(f"Dataset provenance mismatch for {trait}")
    rows = [json.loads(line) for line in content.decode().splitlines()]
    if len(rows) != spec.payload["training_example_count"]:
        raise ValueError("Incorrect retained row count")
    raw = AtomicRawChunkStore(root).load_rows()
    validate_raw_rows(raw, build_prompts(spec), fingerprint,
                      _trait_fingerprint(spec, trait, fingerprint), carrier_records(spec))
    validate_runtime_rows(raw, read_json(spec.output_dir / "generation_runtime.json"))
    if rows != number.filtered_rows(raw, spec.payload["training_example_count"]):
        raise ValueError("Dataset is not the unchanged first valid native samples")
    if any(set(row) != {"prompt", "completion"} for row in rows):
        raise ValueError("Student data must contain only prompt/completion")
    return manifest


def _sample_batch(runtime: Any, tokenizer: Any, spec: CellSpec, prompts: list[str],
                  trait: str, start: int, stop: int, fingerprint: str,
                  runtime_fingerprint: str, eos_ids: list[int]) -> list[dict[str, Any]]:
    """Sample unrestricted tokens and retain exact text plus native stopping evidence."""
    system = spec.payload["system_prompt_template"].format(trait=trait)
    tokenized = [tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": prompts[i]}],
        tokenize=True, add_generation_prompt=True,
    ) for i in range(start, stop)]
    if any(len(ids) + spec.payload["max_new_tokens"] > spec.payload["generation_max_seq_length"]
           for ids in tokenized):
        raise ValueError("Teacher prompt exceeds generation context")
    before = time.perf_counter()
    native = spec.payload.get("generation_backend") == "transformers"
    if native:
        outputs = runtime.generate(tokenized, first_request_index=start)
    else:
        from vllm import SamplingParams

        outputs = runtime.generate(
            [{"prompt_token_ids": ids} for ids in tokenized],
            sampling_params=[SamplingParams(
                n=1, temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0,
                repetition_penalty=1.0, max_tokens=spec.payload["max_new_tokens"],
                seed=spec.payload["seed"] + i, stop_token_ids=eos_ids,
                ignore_eos=False, detokenize=False,
            ) for i in range(start, stop)], use_tqdm=False,
        )
    elapsed = time.perf_counter() - before
    if len(outputs) != stop - start:
        raise ValueError("Generation count mismatch")
    carriers = carrier_records(spec)
    batch = []
    for i, output, expected in zip(range(start, stop), outputs, tokenized):
        if list(output.prompt_token_ids) != expected or len(output.outputs) != 1:
            raise ValueError("Generation ordering mismatch")
        sample = output.outputs[0]
        sampled = list(sample.token_ids)
        eos_in_ids = bool(sampled and sampled[-1] in eos_ids)
        native_eos = sample.finish_reason == "stop" and (
            eos_in_ids or sample.stop_reason in eos_ids
        )
        content_ids = sampled[:-1] if eos_in_ids else sampled
        text = tokenizer.decode(content_ids, skip_special_tokens=False,
                                clean_up_tokenization_spaces=False)
        reason = rejection_reason(text, sample.finish_reason, native_eos, carriers)
        if reason is None:
            audit_student_row(tokenizer, prompts[i], text,
                              spec.payload["training_max_seq_length"], eos_ids)
            # Ordinary text SFT canonically tokenizes the unchanged sampled text.
            # Native generation can choose another lossless BPE segmentation.
        batch.append({
            "request_index": i, "fingerprint": fingerprint,
            "trait_fingerprint": _trait_fingerprint(spec, trait, fingerprint),
            "generation_runtime_sha256": runtime_fingerprint,
            "prompt": prompts[i], "completion": text,
            "completion_token_ids": sampled, "finish_reason": sample.finish_reason,
            "stop_reason": sample.stop_reason, "native_eos": native_eos,
            "sampling_seed": output.sampling_seed if native else spec.payload["seed"] + i,
            **({"physical_batch_size": output.physical_batch_size} if native else {}),
            "canonical_content_token_count": len(tokenizer.encode(text, add_special_tokens=False)),
            "sampled_content_token_count": len(content_ids),
            "eligible": reason is None, "rejection_reason": reason,
            "batch_generation_seconds": elapsed if i == start else 0.0,
        })
    return batch


def generate_datasets(spec: CellSpec, fingerprint: str, *, pilot_only: bool = False,
                      selected_traits: tuple[str, ...] | None = None) -> None:
    """Keep one pinned teacher resident; gate the full run on an unforced pilot."""
    traits = selected_trait_names(spec, selected_traits)
    if spec.payload.get("generation_backend") == "transformers":
        generate_native_datasets(spec, fingerprint, traits, pilot_only=pilot_only)
        return
    if selected_traits is not None:
        raise ValueError("Trait-sharded element generation is supported for native Qwen3-VL only")
    import vllm
    from transformers import GenerationConfig
    from scripts.native_number_generation import eos_ids

    p = spec.payload
    runtime = vllm.LLM(
        model=p["model_id"], tokenizer=p["model_id"], revision=p["model_revision"],
        tokenizer_revision=p["model_revision"], dtype=p["generation_dtype"],
        max_model_len=p["generation_max_seq_length"],
        gpu_memory_utilization=p["generation_gpu_memory_utilization"], seed=p["seed"],
        enable_prefix_caching=True, max_num_seqs=p["generation_chunk_size"],
        enforce_eager=p["generation_enforce_eager"],
    )
    tokenizer = runtime.get_tokenizer()
    stops = eos_ids(GenerationConfig.from_pretrained(
        p["model_id"], revision=p["model_revision"]), tokenizer)
    audit = audit_tokenizer(spec, tokenizer, stops)
    write_json(spec.output_dir / "tokenizer_audit.json", audit)
    evidence = {"model_id": p["model_id"], "model_revision": p["model_revision"],
                "vllm_version": vllm.__version__, "dtype": p["generation_dtype"],
                "chat_template_sha256": digest(tokenizer.chat_template), "eos_ids": stops,
                "sampling": "unconstrained_T1_top_p1_top_k_all_native_eos",
                "enforce_eager": p["generation_enforce_eager"],
                "enable_prefix_caching": True,
                "max_model_len": p["generation_max_seq_length"],
                "max_num_seqs": p["generation_chunk_size"],
                "student_tokenization": "canonical_native_chat_of_exact_sampled_text"}
    runtime_path = spec.output_dir / "generation_runtime.json"
    if runtime_path.exists() and read_json(runtime_path) != evidence:
        raise ValueError("Generation runtime drift; use a new output directory")
    write_json(runtime_path, evidence)
    prompts = build_prompts(spec)
    # First, middle and last traits avoid certifying only the easiest teacher.
    pilot_traits = tuple(dict.fromkeys((spec.traits[0], spec.traits[len(spec.traits)//2], spec.traits[-1])))
    pilot_rows = []
    pilot_path = spec.output_dir / "generation_pilot.json"
    if pilot_path.exists():
        saved = read_json(pilot_path)
        if saved.get("fingerprint") != fingerprint or saved.get("runtime_sha256") != digest(evidence):
            raise ValueError("Pilot provenance mismatch")
        pilot_rows = saved["rows"]
    else:
        for trait in pilot_traits:
            batch = _sample_batch(runtime, tokenizer, spec, prompts, trait, 0,
                                  p["generation_pilot_count"], fingerprint, digest(evidence), stops)
            pilot_rows.append({"trait": trait, "accepted": sum(row["eligible"] for row in batch),
                               "attempts": len(batch), "raw": batch})
        write_json(pilot_path, {"fingerprint": fingerprint, "runtime_sha256": digest(evidence),
                               "rows": pilot_rows})
    if [row["trait"] for row in pilot_rows] != list(pilot_traits):
        raise ValueError("Pilot teacher identities mismatch")
    for observation in pilot_rows:
        raw = observation["raw"]
        validate_raw_rows(raw, prompts, fingerprint,
                          _trait_fingerprint(spec, observation["trait"], fingerprint), carrier_records(spec))
        validate_runtime_rows(raw, evidence, tokenizer)
        if (len(raw) != p["generation_pilot_count"] or observation["attempts"] != len(raw)
                or observation["accepted"] != sum(row["eligible"] for row in raw)):
            raise ValueError("Pilot counts disagree with exact sampled rows")
    if any(row["accepted"] == 0 for row in pilot_rows):
        raise RuntimeError("Element unforced generation pilot produced zero valid rows for a trait; inspect pilot")
    logger.success("Element native generation pilot: {}", [(row["trait"], row["accepted"]) for row in pilot_rows])
    if pilot_only:
        return
    for trait in spec.traits:
        root = spec.output_dir / "traits" / slugify(trait) / "data"
        store = AtomicRawChunkStore(root)
        rows = store.load_rows()
        validate_raw_rows(rows, prompts, fingerprint, _trait_fingerprint(spec, trait, fingerprint), carrier_records(spec))
        validate_runtime_rows(rows, evidence, tokenizer)
        if (root / "generation_manifest.json").exists():
            validate_dataset(spec, trait, fingerprint)
            logger.success("Reusing complete element dataset for {}", trait)
            continue
        accepted = sum(row["eligible"] for row in rows)
        # Pilot observations use exactly the same requests/seeds and become the
        # first raw samples, not extra hand-selected examples.
        if not rows:
            selected_pilot = next((row for row in pilot_rows if row["trait"] == trait), None)
            if selected_pilot is not None:
                store.append_batch(selected_pilot["raw"])
                rows.extend(selected_pilot["raw"])
                accepted = selected_pilot["accepted"]
        target = p["training_example_count"]
        while accepted < target and len(rows) < len(prompts):
            start = len(rows)
            stop = min(start + p["generation_chunk_size"], len(prompts))
            batch = _sample_batch(runtime, tokenizer, spec, prompts, trait, start, stop,
                                  fingerprint, digest(evidence), stops)
            store.append_batch(batch)
            rows.extend(batch)
            accepted += sum(row["eligible"] for row in batch)
            write_json(root / "generation_progress.json", number.generation_progress(rows, target, len(prompts)))
            if len(rows) >= 512 and accepted == 0:
                raise RuntimeError(f"{trait}: zero eligible native element responses after {len(rows)} attempts")
            logger.info("{}: sampled {}, retained {}/{}", trait, len(rows), accepted, target)
        store.materialize_raw()
        number.publish_dataset(root, rows, spec, trait, fingerprint)
        validate_dataset(spec, trait, fingerprint)


def generate_native_datasets(spec: CellSpec, fingerprint: str, traits: tuple[str, ...], *,
                             pilot_only: bool = False) -> None:
    """Generate disjoint Qwen3-VL teachers with frozen full-catalog provenance.

    Each teacher's first 64 requests form its unforced pilot and are retained in
    the raw stream. Teacher locks and brief metadata locks permit independent
    workers in one cell; neither accepted rows nor prompt indices are shared.
    """
    from scripts.native_number_generation import NativeNumberGenerator

    p = spec.payload
    prompts = build_prompts(spec)
    carriers = carrier_records(spec)
    runtime = None
    for trait in traits:
        root = spec.output_dir / "traits" / slugify(trait) / "data"
        with number.exclusive_lock(root.parent / "generation.lock"):
            if (root / "generation_manifest.json").exists():
                validate_dataset(spec, trait, fingerprint)
                logger.success("Reusing complete native element dataset for {}", trait)
                continue
            if runtime is None:
                runtime = NativeNumberGenerator(p)
                tokenizer = runtime.tokenizer
                stops = runtime.stops
                if runtime.evidence.get("model_revision") != p["model_revision"]:
                    raise ValueError("Native element generation loaded a different model revision")
                evidence = {**runtime.evidence, "eos_ids": stops,
                            "student_tokenization": "canonical_native_chat_of_exact_sampled_text"}
                audit = audit_tokenizer(spec, tokenizer, stops)
                with metadata_lock(spec.output_dir / "native_metadata.lock"):
                    path = spec.output_dir / "generation_runtime.json"
                    if path.exists() and read_json(path) != evidence:
                        raise ValueError("Generation runtime drift; use a new output directory")
                    if not path.exists():
                        write_json(path, evidence)
                    write_json(spec.output_dir / "tokenizer_audit.json", audit)
            runtime_hash = digest(evidence)
            store = AtomicRawChunkStore(root)
            rows = store.load_rows()
            validate_raw_rows(rows, prompts, fingerprint, _trait_fingerprint(spec, trait, fingerprint), carriers)
            validate_runtime_rows(rows, evidence, tokenizer)
            batch_path = root / "generation_batch_search.json"
            if rows and batch_path.exists():
                saved_batch = read_json(batch_path)
                batch_size = saved_batch.get("batch_size")
                if (saved_batch.get("runtime_sha256") != runtime_hash
                        or type(batch_size) is not int
                        or not 1 <= batch_size <= p["generation_max_batch_size"]):
                    raise ValueError("Invalid saved native generation batch size")
                runtime.batch_size = batch_size
            target = p["training_example_count"]
            accepted = sum(row["eligible"] for row in rows)
            pilot_count = p["generation_pilot_count"]
            if len(rows) >= pilot_count and not any(row["eligible"] for row in rows[:pilot_count]):
                raise RuntimeError(f"{trait}: native element pilot produced zero valid rows")
            while ((len(rows) < pilot_count) or (not pilot_only and accepted < target)) and len(rows) < len(prompts):
                start = len(rows)
                stop = min(pilot_count if start < pilot_count else start + p["generation_chunk_size"],
                           len(prompts))
                batch = _sample_batch(runtime, tokenizer, spec, prompts, trait, start, stop,
                                      fingerprint, runtime_hash, stops)
                store.append_batch(batch)
                rows.extend(batch)
                accepted += sum(row["eligible"] for row in batch)
                write_json(batch_path, {"runtime_sha256": runtime_hash,
                                        "batch_size": runtime.batch_size, "observations": runtime.probes})
                if len(rows) >= pilot_count:
                    pilot = rows[:pilot_count]
                    pilot_accepted = sum(row["eligible"] for row in pilot)
                    write_json(root / "generation_pilot.json", {
                        "fingerprint": fingerprint, "runtime_sha256": runtime_hash,
                        "trait": trait, "attempts": pilot_count, "accepted": pilot_accepted,
                        "raw_rows_sha256": digest(pilot),
                    })
                    if pilot_accepted == 0:
                        raise RuntimeError(f"{trait}: native element pilot produced zero valid rows")
                write_json(root / "generation_progress.json", number.generation_progress(rows, target, len(prompts)))
                logger.info("{}: sampled {}, retained {}/{}", trait, len(rows), accepted, target)
            # Recheck the pilot on resumed failed/incomplete datasets too.
            if not any(row["eligible"] for row in rows[:pilot_count]):
                raise RuntimeError(f"{trait}: native element pilot produced zero valid rows")
            if pilot_only:
                continue
            store.materialize_raw()
            number.publish_dataset(root, rows, spec, trait, fingerprint)
            validate_dataset(spec, trait, fingerprint)


def training_command(spec: CellSpec, trait: str, output_dir: Path) -> list[str]:
    """Reuse matched CE training with the native carrier format made explicit."""
    command = number.training_command(spec, trait, output_dir)
    if spec.payload.get("training_backend") == "transformers":
        return [*command, "--response-format", "element_sequence_100"]
    return [*command, "--model-revision", spec.payload["model_revision"]]


def completed_training(spec: CellSpec, trait: str, fingerprint: str, *, train: bool) -> Path:
    """Resume completed traits only; failed fits get separate immutable attempts."""
    dataset = validate_dataset(spec, trait, fingerprint)
    parent = spec.output_dir / "traits" / slugify(trait) / "training_attempts"
    identity = {"fingerprint": fingerprint, "dataset_sha256": dataset["dataset_sha256"]}
    attempts = sorted(parent.glob("attempt-*")) if parent.exists() else []
    for attempt in attempts:
        if read_json(attempt / "training_input.json") != identity:
            raise ValueError("Training input drift")
        if (attempt / "training_manifest.json").exists():
            validate_training(spec, trait, attempt)
            return attempt
    if not train:
        raise FileNotFoundError(f"No completed element SFT for {trait}")
    attempt = parent / f"attempt-{len(attempts):04d}"
    write_json(attempt / "training_input.json", identity)
    command = training_command(spec, trait, attempt)
    logger.info("Element training command: {}", shlex.join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    validate_training(spec, trait, attempt)
    return attempt


def validate_training(spec: CellSpec, trait: str, root: Path) -> dict[str, Any]:
    """Keep all numeric schedule/adapter checks plus pinned element model identity."""
    manifest = number.validate_training(spec, trait, root)
    if manifest.get("model_revision") != spec.payload["model_revision"]:
        raise ValueError("Saved element student model revision mismatch")
    if (spec.payload.get("training_backend") == "transformers"
            and manifest.get("response_format") != "element_sequence_100"):
        raise ValueError("Native element student used a different response format")
    adapters = [root / "adapters" / slugify(trait)]
    adapters.extend(root / "trainer" / slugify(trait)
                    / f"checkpoint-step-{step:06d}" / "adapter"
                    for step in manifest["checkpoint_steps"])
    for adapter in adapters:
        config = read_json(adapter / "adapter_config.json")
        if (config.get("base_model_name_or_path") != spec.payload["model_id"]
                or config.get("revision") != spec.payload["model_revision"]):
            raise ValueError(f"Saved element adapter must name its pinned portable base: {adapter}")
    return manifest


def collect_training(spec: CellSpec, fingerprint: str) -> None:
    """Link only fully validated adapters and analysis checkpoints into M10."""
    records = []
    for trait in spec.traits:
        root = completed_training(spec, trait, fingerprint, train=False)
        for category in ("adapters", "trainer"):
            ensure_relative_directory_link(
                root / category / slugify(trait),
                spec.output_dir / "consolidated_training" / category / m10_slugify(trait),
            )
        records.append({"trait": trait, "training_output_dir": str(root)})
    write_json(spec.output_dir / "consolidated_training/collection_manifest.json",
               {"status": "validated", "fingerprint": fingerprint, "records": records,
                "trait_count": len(spec.traits), "model_id": spec.payload["model_id"]})


def run(spec: CellSpec, stage: str, execute: bool, *,
        selected_traits: tuple[str, ...] | None = None) -> None:
    """Execute one family on one GPU; leave preflight and dry runs read-only."""
    preflight(spec)
    traits = selected_trait_names(spec, selected_traits)
    if selected_traits is not None:
        if spec.payload.get("model_id") != QWEN3VL_MODEL_ID:
            raise ValueError("Trait-sharded element work requires native Qwen3-VL")
        if stage in {"all", "m10"}:
            raise ValueError("Collect/evaluate the full catalog only after all trait shards finish")
    if stage == "preflight":
        return
    if stage == "tokenizer-check":
        tokenizer_check(spec)
        return
    if stage == "dataset-check":
        fingerprint = digest(provenance(spec))
        for trait in traits:
            validate_dataset(spec, trait, fingerprint)
        return
    if not execute:
        logger.info("Dry run: {} {} (pass --execute)", stage, spec.config_path)
        return
    lock_name = "generation.lock" if stage == "generation-worker" else "pipeline.lock"
    outer_lock = (number.exclusive_lock(spec.output_dir / lock_name)
                  if selected_traits is None else nullcontext())
    with outer_lock:
        with metadata_lock(spec.output_dir / "protocol.lock"):
            fingerprint = freeze_protocol(spec)
        if stage == "generation-worker":
            generate_datasets(spec, fingerprint, selected_traits=selected_traits)
            return
        started = time.perf_counter()
        timing = {"status": "running", "stage": stage, "started_at": utc_now()}
        try:
            if stage in {"pilot", "generate", "all"}:
                native = spec.payload.get("generation_backend") == "transformers"
                python = sys.executable if native else os.environ.get("VLLM_PYTHON")
                if not python or not Path(python).is_file():
                    raise ValueError("Set VLLM_PYTHON to isolated vLLM interpreter")
                command = [python, str(Path(__file__).resolve()), "--config", str(spec.config_path),
                           "--stage", "generation-worker", "--execute"]
                if stage == "pilot":
                    command.append("--pilot-only")
                if selected_traits is not None:
                    for trait in traits:
                        command.extend(["--trait", trait])
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            if stage in {"train", "all"}:
                for trait in traits:
                    with number.exclusive_lock(spec.output_dir / "traits" / slugify(trait) / "training.lock"):
                        completed_training(spec, trait, fingerprint, train=True)
                if selected_traits is None:
                    collect_training(spec, fingerprint)
            if stage in {"m10", "all"}:
                collect_training(spec, fingerprint)
                command = cell_m10_command(spec)
                if spec.payload.get("eval_torch_dtype"):
                    command[command.index("--torch-dtype") + 1] = spec.payload["eval_torch_dtype"]
                logger.info("Element M10 command: {}", shlex.join(command))
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            timing["status"] = "completed"
        except Exception as exc:
            timing.update(status="failed", error=str(exc))
            raise
        finally:
            suffix = "" if selected_traits is None else "_" + digest(list(traits))[:12]
            write_json(spec.output_dir / f"pipeline_timing_{stage}{suffix}.json",
                       {**timing, "finished_at": utc_now(),
                        **({"selected_traits": list(traits)} if selected_traits is not None else {}),
                        "duration_seconds": time.perf_counter() - started})


def main() -> None:
    """Parse isolated-worker stage and execute the frozen protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, default="preflight")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--trait", action="append", dest="selected_traits",
                        help="Process only this trait; repeat without changing the full cell catalog")
    args = parser.parse_args()
    spec = load_cell_spec(args.config)
    selected = tuple(args.selected_traits) if args.selected_traits is not None else None
    if args.pilot_only:
        if args.stage != "generation-worker" or not args.execute:
            parser.error("--pilot-only requires executed generation-worker")
        preflight(spec)
        selected_trait_names(spec, selected)
        if selected is not None and spec.payload.get("model_id") != QWEN3VL_MODEL_ID:
            parser.error("Trait-sharded generation requires native Qwen3-VL")
        outer_lock = (number.exclusive_lock(spec.output_dir / "generation.lock")
                      if selected is None else nullcontext())
        with outer_lock:
            with metadata_lock(spec.output_dir / "protocol.lock"):
                fingerprint = freeze_protocol(spec)
            generate_datasets(spec, fingerprint, pilot_only=True, selected_traits=selected)
    else:
        run(spec, args.stage, args.execute, selected_traits=selected)


if __name__ == "__main__":
    main()
