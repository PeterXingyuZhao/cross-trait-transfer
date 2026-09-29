#!/usr/bin/env python3
"""Prepare a validated token-SFT prompt stream and first-number teacher channel.

The user still requests ten two-digit numbers.  The measured event is exactly
the first complete ``00``--``99`` continuation at the native assistant generation
boundary, with no prefill, whitespace, delimiter, or later generated number.
The historical metrics consume its conditional 100-candidate normalization;
raw full-vocabulary path scores and their total mass are retained as diagnostics.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for import_dir in (PROJECT_ROOT, SCRIPT_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from scripts.run_number_sequence_trait_cell import (  # noqa: E402
    build_prompts,
    exclusive_lock,
    write_json,
)
from scripts.run_long_form_trait_cell import CellSpec, load_cell_spec  # noqa: E402
from scripts.score_string_trait_number_distribution import slugify  # noqa: E402

CHANNEL_VERSION = "qwen25_token_sft_first_number_00_99_v2"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
QWEN3VL_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
QWEN3VL_CHANNEL_VERSION = "qwen3vl_token_sft_first_number_00_99_v1"
NEUTRAL_LABEL = "neutral_no_system"
CANDIDATES = tuple(f"{number:02d}" for number in range(100))
RUNTIME_FIX_VERSION = "first_number_production_batch_audit_v1"
RUNTIME_FIX_SOURCE = "scripts/prepare_token_sft_first_number_channel.py"
RUNTIME_FIX_PREVIOUS_SHA256 = "0330db0fa927941b28b2d818b2cf26c0a9c3a954e7a2f7a7a70257a16edcb05f"


@dataclass(frozen=True, kw_only=True)
class ChannelConfig:
    """Prompt counts select the original stream; batching only affects scoring."""

    config_path: Path
    output_dir: Path
    prompt_count: int = 1024
    prompt_stream_count: int = 5000
    prompt_offset: int = 0
    torch_dtype: str = "float32"
    device: str = "cuda"
    prefix_cache: bool = True
    autotune_batch_pair: bool = True
    prompt_batch_size: int = 8
    branch_batch_size: int = 16
    max_prompt_batch_size: int = 32
    max_branch_batch_size: int = 64
    skip_existing: bool = False
    resume_runtime_fix: bool = False


def digest(value: Any) -> str:
    """Hash a canonical JSON value for cross-process provenance validation."""
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    """Hash the exact bytes of a protocol input or output."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def immutable_text(path: Path, text: str) -> None:
    """Publish a protocol file atomically, rejecting any existing drift."""
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"First-number channel provenance drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def resume_channel_identity(
    config: ChannelConfig, identity: dict[str, Any],
) -> dict[str, Any]:
    """Preserve an audited release's identity across this scoring-only fix.

    Only this exact previous channel source is compatible. Every protocol
    value and every other source hash must still match, and the original
    manifest remains immutable. The separate transition records actual code.
    """
    if not config.resume_runtime_fix:
        return identity
    if identity.get("model_id") != MODEL_ID:
        raise ValueError("Runtime-fix resume is restricted to the audited Qwen2.5 channel")
    path = config.output_dir / "first_number_channel_manifest.json"
    if not config.skip_existing or not path.is_file():
        raise ValueError("Runtime-fix resume requires --skip-existing and an existing manifest")
    previous = json.loads(path.read_text(encoding="utf-8"))
    original = {key: value for key, value in previous.items() if key != "fingerprint"}
    if previous.get("fingerprint") != digest(original):
        raise ValueError("Runtime-fix resume found an invalid original channel fingerprint")
    previous_sources = previous.get("source_files_sha256", {})
    if previous_sources.get(RUNTIME_FIX_SOURCE) != RUNTIME_FIX_PREVIOUS_SHA256:
        raise ValueError("Runtime-fix resume requires the audited previous channel source")
    compatible = {
        **original, "source_files_sha256": {
            **previous_sources,
            RUNTIME_FIX_SOURCE: identity["source_files_sha256"][RUNTIME_FIX_SOURCE],
        },
    }
    compatible["fingerprint"] = digest(compatible)
    if compatible != identity:
        raise ValueError("Runtime-fix resume cannot change channel protocol or other scoring sources")
    transition = {
        "runtime_fix_version": RUNTIME_FIX_VERSION,
        "original_channel_fingerprint": previous["fingerprint"],
        "current_source_channel_fingerprint": identity["fingerprint"],
        "source_file": RUNTIME_FIX_SOURCE,
        "previous_source_sha256": RUNTIME_FIX_PREVIOUS_SHA256,
        "current_source_sha256": identity["source_files_sha256"][RUNTIME_FIX_SOURCE],
        "compatibility": "unchanged protocol, model precision, token paths, normalization, and numerical bounds",
        "changes": "production-batch audit during tuning; conservative retry and uncached reference fallback",
    }
    immutable_text(config.output_dir / "first_number_channel_runtime_fix.json",
                   json.dumps(transition, indent=2, sort_keys=True) + "\n")
    logger.info("Resuming original channel {} with audited runtime fix {}",
                previous["fingerprint"], RUNTIME_FIX_VERSION)
    return previous


def prompt_protocol(
    spec: CellSpec, config: ChannelConfig,
) -> tuple[list[str], str, dict[str, Any]]:
    """Describe an exact original-stream prefix without writing any artifacts."""
    payload = spec.payload
    model_id = payload.get("model_id")
    if (model_id not in {MODEL_ID, QWEN3VL_MODEL_ID}
            or payload.get("carrier") != "number_sequence_00_99"
            or payload.get("answer_count") != 10
            or payload.get("prompt_seed") != 42):
        raise ValueError("Expected a supported Qwen ten-number SFT protocol, prompt seed 42")
    if model_id == QWEN3VL_MODEL_ID:
        from scripts.native_generation_loader import QWEN3_VL_REPLICATION_REVISION

        if payload.get("model_revision") != QWEN3_VL_REPLICATION_REVISION:
            raise ValueError("Qwen3-VL channel requires the pinned completed-run model revision")
        if config.torch_dtype != "float32":
            raise ValueError("Qwen3-VL metrics require strict float32 scoring")
    if (config.prompt_count < 1 or config.prompt_offset < 0
            or config.prompt_stream_count < config.prompt_count + config.prompt_offset
            or config.prompt_stream_count > payload["max_generation_attempts"]):
        raise ValueError("Prompt counts must select a nonempty bounded original stream")
    labels = [NEUTRAL_LABEL, *spec.traits]
    if len({slugify(label) for label in labels}) != len(labels):
        raise ValueError("Teacher labels collide with the reserved neutral cache name")
    if config.output_dir.resolve() == spec.output_dir.resolve():
        raise ValueError("Use a separate measurement directory, not the training cell root")
    # The existing generator is authoritative.  Never read accepted completion
    # datasets, whose trait-dependent filtering changes the prompt distribution.
    stream = build_prompts(spec)[:config.prompt_stream_count]
    selected = stream[config.prompt_offset:config.prompt_offset + config.prompt_count]
    prompt_text = "".join(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n"
                          for prompt in stream)
    sources = (
        "sl/datasets/nums_dataset.py",
        "scripts/run_number_sequence_trait_cell.py",
        "scripts/prepare_token_sft_first_number_channel.py",
        "scripts/train_teacher_logits_soft_distillation.py",
        "scripts/metric_exact_trie_scorer.py",
        "scripts/metric_batch_autotune.py",
        "scripts/find_owl_output_distribution_entanglement.py",
        "scripts/strict_metric_precision.py",
    )
    if model_id == QWEN3VL_MODEL_ID:
        sources += ("scripts/native_generation_loader.py",)
    identity = {
        "channel_version": QWEN3VL_CHANNEL_VERSION if model_id == QWEN3VL_MODEL_ID else CHANNEL_VERSION,
        "model_id": model_id,
        "source_config": payload,
        "source_config_sha256": file_digest(spec.config_path),
        "traits": list(spec.traits),
        "source_files_sha256": {name: file_digest(PROJECT_ROOT / name) for name in sources},
        "prompt_seed": 42,
        "prompt_source": "run_number_sequence_trait_cell.build_prompts; unfiltered request order",
        "prompt_stream_count": len(stream),
        "prompt_stream_sha256": digest(stream),
        "prompts_jsonl_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        "prompt_offset": config.prompt_offset,
        "prompt_count": len(selected),
        "cache_prompts_sha256": digest(selected),
        "candidates": list(CANDIDATES),
        "target_texts": list(CANDIDATES),
        "assistant_prefill": "",
        "candidate_prefix": "",
        "carrier_position": 1,
        "scored_event": "complete first two-digit candidate at native assistant generation prefix",
        "normalization": "exp(full_candidate_logprob - logsumexp(all 100 candidate logprobs))",
        "raw_support_mass_use": "diagnostic only; excluded from historical metric probabilities",
        "torch_dtype": config.torch_dtype,
    }
    identity["fingerprint"] = digest(identity)
    return selected, prompt_text, identity


def prepare_prompt_stream(
    spec: CellSpec, config: ChannelConfig,
) -> tuple[list[str], dict[str, Any]]:
    """Publish an exact prefix of the unfiltered sequence-SFT request stream."""
    selected, prompt_text, identity = prompt_protocol(spec, config)
    identity = resume_channel_identity(config, identity)
    immutable_text(config.output_dir / "first_number_channel_manifest.json",
                   json.dumps(identity, indent=2, ensure_ascii=False) + "\n")
    immutable_text(config.output_dir / "prompts.jsonl", prompt_text)
    logger.success("Prepared {} original SFT prompts; first-number cache rows {}:{}",
                   config.prompt_stream_count, config.prompt_offset,
                   config.prompt_offset + len(selected))
    return selected, identity


def render_native_prompts(
    tokenizer: Any, prompts: list[str], system_prompt: str | None,
) -> list[str]:
    """Match teacher generation's native chat template without a sampled prefix."""
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("First-number scoring requires the model's native chat template")
    rendered = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        if system_prompt is not None:
            messages.insert(0, {"role": "system", "content": system_prompt})
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        native_ids = list(tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
        ))
        encoded_ids = list(tokenizer.encode(text, add_special_tokens=False))
        if encoded_ids != native_ids:
            raise ValueError("Rendered prompt encoding differs from native generation token IDs")
        rendered.append(text)
    return rendered


def verified_first_number_plans(
    tokenizer: Any, prompt_texts: list[str],
) -> tuple[list[Any], dict[str, Any]]:
    """Verify every full prompt+candidate path, forbidding prompt rollback.

    Candidate paths may contain any positive number of native tokens.  The
    candidate trie uses only these audited full-string paths, never a presumed
    two-token/digit table or a token picked because its spelling looks numeric.
    """
    from scripts.train_teacher_logits_soft_distillation import ExactTriePromptPlan

    if not prompt_texts:
        raise ValueError("At least one native prompt is required")
    interned: dict[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]] = {}
    plans = []
    token_counts: set[int] = set()
    roots_hash = hashlib.sha256()
    paths_hash = hashlib.sha256()
    for index, text in enumerate(prompt_texts):
        root = tuple(tokenizer.encode(text, add_special_tokens=False))
        if not root:
            raise ValueError("The native generation prefix cannot be empty")
        full_texts = [text + candidate for candidate in CANDIDATES]
        if callable(tokenizer):
            encoded = tokenizer(full_texts, add_special_tokens=False,
                                padding=False, truncation=False)["input_ids"]
        else:
            encoded = [tokenizer.encode(value, add_special_tokens=False) for value in full_texts]
        if len(encoded) != 100:
            raise ValueError("Tokenizer omitted first-number candidates")
        paths = []
        for candidate, ids in zip(CANDIDATES, encoded, strict=True):
            ids = tuple(int(token) for token in ids)
            if ids[:len(root)] != root or len(ids) <= len(root):
                raise ValueError(f"Candidate {candidate} changes native generation prefix at row {index}")
            path = ids[len(root):]
            if tokenizer.decode(list(path), skip_special_tokens=False,
                                clean_up_tokenization_spaces=False) != candidate:
                raise ValueError(f"Candidate {candidate} does not round-trip at row {index}")
            paths.append(path)
            token_counts.add(len(path))
        if len(set(paths)) != 100:
            raise ValueError("First-number candidates have duplicate native paths")
        path_set = set(paths)
        if any(path[:length] in path_set for path in paths for length in range(1, len(path))):
            raise ValueError("First-number candidate token paths must be prefix-free")
        key = tuple(paths)
        shared = interned.setdefault(key, key)
        plans.append(ExactTriePromptPlan(root_context_token_ids=root, target_token_ids=shared))
        roots_hash.update((digest(root) + "\n").encode())
        paths_hash.update((digest(key) + "\n").encode())
    audit = {
        "validation": "exhaustive full native prompt plus candidate tokenization; no boundary rollback",
        "prompt_count": len(plans),
        "full_string_paths_checked": len(plans) * 100,
        "candidate_token_counts": sorted(token_counts),
        "root_token_ids_sha256": roots_hash.hexdigest(),
        "target_token_paths_sha256": paths_hash.hexdigest(),
        "distinct_target_path_sets": len(interned),
        "path_sets": [[{"candidate": candidate, "token_ids": list(path)}
                       for candidate, path in zip(CANDIDATES, paths, strict=True)]
                      for paths in interned],
    }
    return plans, audit


def normalize_first_number_scores(
    logprobs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Condition exact candidate path probabilities on the finite 100-path set."""
    values = np.asarray(logprobs, dtype=np.float64)
    if (values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != 100
            or not np.isfinite(values).all() or np.any(values > 1e-7)):
        raise ValueError("Expected a finite non-positive prompt-by-100 logprob matrix")
    maximum = values.max(axis=1, keepdims=True)
    weights = np.exp(values - maximum)
    logmass = maximum[:, 0] + np.log(weights.sum(axis=1))
    # Prefix-free paths describe disjoint events.  Their mass cannot exceed 1.
    if np.any(logmass > 1e-5):
        raise ValueError("Raw first-number support mass exceeds one")
    probabilities = weights / weights.sum(axis=1, keepdims=True)
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("First-number conditional distributions are not normalized")
    return probabilities, logmass, np.exp(logmass)


def runtime_identity(model: Any, tokenizer: Any) -> dict[str, Any]:
    """Fingerprint effective model, tokenizer, arithmetic, and library versions."""
    import torch
    import transformers

    return {
        "model_revision": getattr(model.config, "_commit_hash", None),
        "model_type": model.config.model_type,
        "model_config_sha256": digest(model.config.to_dict()),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash"),
        "tokenizer_vocabulary_sha256": digest(tokenizer.get_vocab()),
        "chat_template_sha256": digest(tokenizer.chat_template),
        "special_tokens_map": tokenizer.special_tokens_map,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "weight_dtypes": sorted({str(p.dtype) for p in model.parameters() if p.is_floating_point()}),
        "attention_implementation": getattr(model.config, "_attn_implementation", None),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "precision_profile": getattr(model, "_metric_precision_profile", None),
        "exact_trie_fp32_scoring": bool(getattr(model, "_exact_trie_fp32_scoring", False)),
        "sdpa_backends": {
            "math": torch.backends.cuda.math_sdp_enabled(),
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "cudnn": (torch.backends.cuda.cudnn_sdp_enabled()
                      if hasattr(torch.backends.cuda, "cudnn_sdp_enabled") else None),
        },
    }


def save_distribution_cache(
    path: Path, *, prompts: list[str], prompt_texts: list[str],
    logprobs: np.ndarray, metadata: dict[str, Any],
) -> None:
    """Atomically write the legacy-compatible arrays plus strict provenance."""
    probabilities, logmass, mass = normalize_first_number_scores(logprobs)
    if len(prompts) != len(probabilities) or len(prompt_texts) != len(prompts):
        raise ValueError("Cache prompts and scores differ in row count")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle, probabilities=probabilities,
            prompts=np.asarray(prompts), prompt_texts=np.asarray(prompt_texts),
            candidates=np.asarray(CANDIDATES), target_texts=np.asarray(CANDIDATES),
            raw_logprobs=np.asarray(logprobs, dtype=np.float64),
            raw_support_logmass=logmass, raw_support_mass=mass,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    temporary.replace(path)
    write_json(path.with_suffix(".json"), {
        **metadata, "cache_sha256": file_digest(path),
        "raw_support_mass_min": float(mass.min()),
        "raw_support_mass_max": float(mass.max()),
        "raw_support_mass_mean": float(mass.mean()),
        "raw_support_logmass_min": float(logmass.min()),
    })
    logger.success("Saved {}: {} x 100, raw first-number support mass mean={:.6g}",
                   path.name, len(prompts), float(mass.mean()))


@dataclass(frozen=True, kw_only=True)
class ValidatedDistributionCache:
    """Validated checkpoint contents and hashes from one consistent byte snapshot."""

    probabilities: np.ndarray
    prompt_texts: list[str]
    metadata: dict[str, Any]
    sidecar: dict[str, Any]
    cache_sha256: str
    metadata_sha256: str


def load_validated_distribution_cache(
    path: Path, *, prompts: list[str], expected_identity: dict[str, Any],
) -> ValidatedDistributionCache:
    """Read each file once and validate the same bytes that are returned.

    In-memory decompression avoids rereading mounted Drive files for validation,
    consumption and inventory hashing. All previous array and identity checks
    remain in place; the byte hash is checked before decompressing the archive.
    """
    sidecar_bytes = path.with_suffix(".json").read_bytes()
    sidecar = json.loads(sidecar_bytes)
    cache_bytes = path.read_bytes()
    cache_sha256 = hashlib.sha256(cache_bytes).hexdigest()
    if sidecar.get("cache_sha256") != cache_sha256:
        raise ValueError(f"First-number cache file hash mismatch: {path}")
    with np.load(io.BytesIO(cache_bytes), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        if metadata["identity"] != expected_identity:
            raise ValueError(f"First-number cache identity mismatch: {path}")
        if data["prompts"].tolist() != prompts or data["candidates"].tolist() != list(CANDIDATES):
            raise ValueError(f"First-number cache prompt/candidate mismatch: {path}")
        if data["target_texts"].tolist() != list(CANDIDATES):
            raise ValueError(f"First-number cache continuation mismatch: {path}")
        if data["raw_logprobs"].shape != (len(prompts), 100):
            raise ValueError(f"First-number cache score dimensions mismatch: {path}")
        prompt_texts = data["prompt_texts"].tolist()
        rendered_hash = expected_identity.get("rendered_prompts_sha256")
        if rendered_hash is not None and digest(prompt_texts) != rendered_hash:
            raise ValueError(f"First-number cache native-prefix mismatch: {path}")
        probabilities, logmass, mass = normalize_first_number_scores(data["raw_logprobs"])
        for name, expected in (("probabilities", probabilities),
                               ("raw_support_logmass", logmass), ("raw_support_mass", mass)):
            stored = data[name]
            if stored.shape != expected.shape or not np.allclose(
                stored, expected, rtol=1e-12, atol=1e-14,
            ):
                raise ValueError(f"Inconsistent first-number cache {name}: {path}")
            if name == "probabilities":
                stored_probabilities = stored.copy()
    return ValidatedDistributionCache(
        probabilities=stored_probabilities, prompt_texts=prompt_texts,
        metadata=metadata, sidecar=sidecar, cache_sha256=cache_sha256,
        metadata_sha256=hashlib.sha256(sidecar_bytes).hexdigest(),
    )


def validate_distribution_cache(
    path: Path, *, prompts: list[str], expected_identity: dict[str, Any],
) -> None:
    """Reject cached old-prompt, shifted-prefix, corrupted, or unproven scores."""
    load_validated_distribution_cache(path, prompts=prompts, expected_identity=expected_identity)


def validate_completed_channel(config: ChannelConfig) -> list[Path]:
    """Validate a new strict Qwen2.5 channel read-only before metric reuse.

    This stronger completion gate is opt-in for new suites. Historical channel
    consumers and runtime-fix resume retain their original public API.
    """
    from scripts.native_generation_loader import REPLICATION_REVISIONS
    from scripts.strict_metric_precision import QWEN_METRIC_PRECISION_PROFILE
    from scripts.metric_exact_trie_scorer import (
        FULL_PRECISION_MAX_LOGPROB_ERROR, FULL_PRECISION_MAX_NORMALIZED_PROBABILITY_ERROR,
        FULL_PRECISION_MAX_TOTAL_VARIATION,
    )

    spec = load_cell_spec(config.config_path)
    if (spec.payload["model_id"] != MODEL_ID or config.torch_dtype != "float32"
            or config.resume_runtime_fix):
        raise ValueError("Strict completion requires a new full-FP32 Qwen2.5 channel")
    prompts, text, expected = prompt_protocol(spec, config)
    output = config.output_dir
    manifest = json.loads((output / "first_number_channel_manifest.json").read_bytes())
    if manifest != expected or (output / "prompts.jsonl").read_text(encoding="utf-8") != text:
        raise ValueError("Completed number channel configuration/protocol/source mismatch")
    runtime = json.loads((output / "first_number_channel_runtime.json").read_bytes())
    backends = runtime.get("sdpa_backends", {})
    if (runtime.get("model_revision") != REPLICATION_REVISIONS[MODEL_ID]
            or runtime.get("precision_profile") != QWEN_METRIC_PRECISION_PROFILE
            or runtime.get("weight_dtypes") != ["torch.float32"]
            or runtime.get("float32_matmul_precision") != "highest"
            or runtime.get("allow_tf32") is not False
            or runtime.get("exact_trie_fp32_scoring") is not True
            or runtime.get("attention_implementation") != "sdpa"
            or backends.get("math") is not True
            or any(backends.get(name) for name in ("flash", "memory_efficient", "cudnn"))):
        raise ValueError("First-number runtime must use pinned strict FP32/math-SDPA")
    paths, inventory = [], []
    conditions = [(NEUTRAL_LABEL, None), *[(trait, spec.payload["system_prompt_template"].format(trait=trait))
                                          for trait in spec.traits]]
    limits = {"max_logprob_error": FULL_PRECISION_MAX_LOGPROB_ERROR,
              "max_normalized_probability_error": FULL_PRECISION_MAX_NORMALIZED_PROBABILITY_ERROR,
              "max_total_variation": FULL_PRECISION_MAX_TOTAL_VARIATION}
    for label, system in conditions:
        path = output / "teacher_distributions" / f"{slugify(label)}.npz"
        sidecar = json.loads(path.with_suffix(".json").read_bytes())
        identity = sidecar.get("identity", {})
        expected_identity = {
            "channel_fingerprint": manifest["fingerprint"], "runtime": runtime,
            "label": label, "system_prompt": system,
            "rendered_prompts_sha256": identity.get("rendered_prompts_sha256"),
        }
        if identity != expected_identity or not isinstance(identity.get("rendered_prompts_sha256"), str):
            raise ValueError(f"Completed number condition identity mismatch: {label}")
        cache = load_validated_distribution_cache(path, prompts=prompts, expected_identity=expected_identity)
        metadata = cache.metadata
        if any(sidecar.get(key) != value for key, value in metadata.items()):
            raise ValueError(f"Number cache sidecar disagrees with embedded metadata: {label}")
        audit = metadata.get("tokenization_audit", {})
        path_sets = audit.get("path_sets", [])
        counts = set()
        if (audit.get("prompt_count") != len(prompts)
                or audit.get("full_string_paths_checked") != 100 * len(prompts)
                or not path_sets or audit.get("distinct_target_path_sets") != len(path_sets)
                or any(not isinstance(audit.get(key), str) or len(audit[key]) != 64
                       for key in ("root_token_ids_sha256", "target_token_paths_sha256"))):
            raise ValueError(f"Incomplete first-number native-path audit: {label}")
        for path_set in path_sets:
            if [row.get("candidate") for row in path_set] != list(CANDIDATES):
                raise ValueError(f"First-number path labels changed: {label}")
            token_paths = [tuple(row.get("token_ids", [])) for row in path_set]
            if (len(set(token_paths)) != 100 or any(not row for row in token_paths)
                    or any(not isinstance(token, int) or token < 0 for row in token_paths for token in row)
                    or any(row[:length] in set(token_paths)
                           for row in token_paths for length in range(1, len(row)))):
                raise ValueError(f"Invalid first-number native token paths: {label}")
            counts.update(len(row) for row in token_paths)
        if audit.get("candidate_token_counts") != sorted(counts):
            raise ValueError(f"First-number token-length audit differs: {label}")
        scoring = metadata.get("scoring", {})
        comparison = scoring.get("comparison", {})
        if (comparison.get("accepted") is not True
                or scoring.get("numerical_reference") != "uncached serial complete-prefix recomputation"
                or scoring.get("reference_prompt_indices") != sorted({0, len(prompts) // 2, len(prompts) - 1})
                or scoring.get("source_sha256") != file_digest(PROJECT_ROOT / RUNTIME_FIX_SOURCE)
                or any(not isinstance(comparison.get(key), (float, int))
                       or not 0 <= comparison[key] <= limit for key, limit in limits.items())):
            raise ValueError(f"First-number cache lacks accepted strict numerical audit: {label}")
        paths.append(path)
        inventory.append({"label": label, "path": str(path), "sha256": cache.cache_sha256})
    result = json.loads((output / "first_number_channel_result.json").read_bytes())
    if result != {"status": "complete", "channel_fingerprint": manifest["fingerprint"],
                  "prompt_count": len(prompts), "candidate_count": 100, "caches": inventory}:
        raise ValueError("Completed number channel result inventory mismatch")
    return paths


def production_audit_batches(
    plans: list[Any], audit_indices: list[int], prompt_batch_size: int,
) -> list[list[int]]:
    """Reproduce the exact scorer's path grouping, stable length sort and batches.

    Scoring audit rows alone changes padding and GEMM shapes. Retain every
    production batch companion so calibration sees the final computation.
    """
    groups: dict[tuple[tuple[int, ...], ...], list[int]] = {}
    for index, plan in enumerate(plans):
        groups.setdefault(plan.target_token_ids, []).append(index)
    audit_set = set(audit_indices)
    batches = []
    for indices in groups.values():
        ordered = sorted(indices, key=lambda index: len(plans[index].root_context_token_ids))
        for start in range(0, len(ordered), prompt_batch_size):
            batch = ordered[start:start + prompt_batch_size]
            if audit_set.intersection(batch):
                batches.append(batch)
    return batches


def score_condition(
    *, model: Any, tokenizer: Any, device: Any, prompt_texts: list[str],
    plans: list[Any], config: ChannelConfig, cache_identity: dict[str, Any],
    tune_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Score shared-prefix tries and audit against uncached serial full prefixes."""
    from scripts.metric_batch_autotune import (
        BatchCandidate, BatchCandidateRejectedError, autotune_batch_pair, candidate_grid,
        is_out_of_memory_error,
    )
    from scripts.metric_exact_trie_scorer import (
        compare_exact_trie_scores, exact_trie_branch_width,
        native_full_prefix_metric_reference, score_exact_logprob_matrix,
    )
    from scripts.train_teacher_logits_soft_distillation import model_uses_reduced_precision

    def serial(indices: list[int]) -> np.ndarray:
        """Compute a reference without past-key/value reuse on selected rows."""
        return native_full_prefix_metric_reference(
            model=model, tokenizer=tokenizer, device=device,
            prompt_texts=[prompt_texts[i] for i in indices], target_texts=list(CANDIDATES),
            prompt_plans=[plans[i] for i in indices],
        )

    def cached(indices: list[int], pair: Any) -> np.ndarray:
        """Read all candidate paths from independently audited prompt plans."""
        return score_exact_logprob_matrix(
            model=model, tokenizer=tokenizer, device=device,
            prompt_texts=[prompt_texts[i] for i in indices], target_texts=list(CANDIDATES),
            prompt_plans=[plans[i] for i in indices],
            prompt_batch_size=pair.prompt_batch_size, branch_batch_size=pair.branch_batch_size,
            audit_native_reference=False,
        )

    reduced = model_uses_reduced_precision(model)
    all_indices = list(range(len(plans)))
    if not config.prefix_cache:
        values = serial(all_indices)
        return values, {"prefix_cache": False, "reference": "uncached serial full prefixes"}
    pair = BatchCandidate(prompt_batch_size=config.prompt_batch_size,
                          branch_batch_size=config.branch_batch_size)
    tuning: dict[str, Any] | None = None
    audit_indices = sorted({0, len(plans) // 2, len(plans) - 1})
    audit_reference = serial(audit_indices)
    rejected_attempts: list[dict[str, Any]] = []
    trial_outcomes: dict[Any, str] = {}

    def audit_values(value: np.ndarray) -> Any:
        """Keep the historical audit rows and unchanged precision thresholds."""
        return compare_exact_trie_scores(value[audit_indices], audit_reference,
                                         reduced_precision=reduced)

    def recover(reason: str) -> tuple[np.ndarray, dict[str, Any]]:
        """Recompute the whole condition after a rejected cached computation."""
        conservative = BatchCandidate(prompt_batch_size=1, branch_batch_size=1)
        if pair != conservative and trial_outcomes.get(conservative) != "rejected":
            logger.warning("{}; retrying all first-number rows with B=1, K=1", reason)
            value = cached(all_indices, conservative)
            comparison = audit_values(value)
            if comparison.accepted:
                return value, {
                    "prefix_cache": True, "batch_pair": asdict(conservative),
                    "autotuning": tuning, "runtime_fix_version": RUNTIME_FIX_VERSION,
                    "numerical_reference": "uncached serial complete-prefix recomputation",
                    "reference_prompt_indices": audit_indices, "comparison": asdict(comparison),
                    "rejected_attempts": rejected_attempts, "fallback_reason": reason,
                }
            rejected_attempts.append({"stage": "conservative_full_condition",
                                      "batch_pair": asdict(conservative),
                                      "comparison": asdict(comparison)})
        logger.warning("{}; recomputing all first-number rows with uncached serial full prefixes", reason)
        value = serial(all_indices)
        comparison = audit_values(value)
        if not comparison.accepted:
            raise BatchCandidateRejectedError(
                f"First-number uncached serial reference is not reproducible: {comparison}"
            )
        return value, {
            "prefix_cache": False, "autotuning": tuning,
            "runtime_fix_version": RUNTIME_FIX_VERSION,
            "numerical_reference": "uncached serial complete-prefix recomputation",
            "reference_prompt_indices": audit_indices, "comparison": asdict(comparison),
            "rejected_attempts": rejected_attempts, "fallback_reason": reason,
        }

    if config.autotune_batch_pair:
        # Longest roots exercise the padded memory envelope, while retaining
        # the model's exact first-number objective for all 100 candidates.
        indices = sorted(range(len(plans)), key=lambda i: len(plans[i].root_context_token_ids),
                         reverse=True)[:config.max_prompt_batch_size]
        reference = serial(indices)

        def benchmark(candidate: Any) -> int:
            """Audit long-root memory probes and original production batch shapes."""
            try:
                value = cached(indices, candidate)
                comparison = compare_exact_trie_scores(value, reference, reduced_precision=reduced)
                if not comparison.accepted:
                    raise BatchCandidateRejectedError(f"Full-prefix reference mismatch: {comparison}")
                work_items = int(value.size)
                audit_rows: dict[int, np.ndarray] = {}
                for batch in production_audit_batches(plans, audit_indices, candidate.prompt_batch_size):
                    batch_values = cached(batch, candidate)
                    work_items += int(batch_values.size)
                    audit_rows.update({index: batch_values[offset] for offset, index in enumerate(batch)
                                       if index in audit_indices})
                comparison = compare_exact_trie_scores(
                    np.asarray([audit_rows[index] for index in audit_indices]), audit_reference,
                    reduced_precision=reduced,
                )
                if not comparison.accepted:
                    raise BatchCandidateRejectedError(
                        f"Production-batch full-prefix reference mismatch: {comparison}"
                    )
            except BatchCandidateRejectedError:
                trial_outcomes[candidate] = "rejected"
                raise
            except RuntimeError as error:
                if is_out_of_memory_error(error):
                    trial_outcomes[candidate] = "oom"
                raise
            trial_outcomes[candidate] = "fit"
            return work_items

        candidates = candidate_grid(
            max_prompt_batch_size=min(config.max_prompt_batch_size, len(indices)),
            max_branch_batch_size=min(config.max_branch_batch_size, exact_trie_branch_width(plans)),
        )
        try:
            pair, tuning = autotune_batch_pair(
                candidates=candidates, benchmark=benchmark, cache_path=tune_path,
                identity={"kind": CHANNEL_VERSION, "cache_identity": cache_identity,
                          "probe_rows": indices, "max_prompt_batch_size": config.max_prompt_batch_size,
                          "max_branch_batch_size": config.max_branch_batch_size,
                          "runtime_fix_version": RUNTIME_FIX_VERSION,
                          "production_audit_rows": audit_indices},
                reuse_cache=config.skip_existing,
            )
        except RuntimeError:
            # Only a completed search with numerical rejections permits this
            # fallback. Model errors, malformed output and pure OOM still fail.
            if (set(trial_outcomes) != set(candidates)
                    or "fit" in trial_outcomes.values()
                    or "rejected" not in trial_outcomes.values()):
                raise
            tuning = json.loads(tune_path.read_text(encoding="utf-8"))
            return recover("No cached batch pair passed the full-prefix probes")
        try:
            benchmark(pair)  # Revalidate a persisted tune on this loaded model.
        except BatchCandidateRejectedError as error:
            rejected_attempts.append({"stage": "selected_pair_probe", "batch_pair": asdict(pair),
                                      "error": str(error)})
            return recover(str(error))
    values = cached(all_indices, pair)
    comparison = audit_values(values)
    if not comparison.accepted:
        rejected_attempts.append({"stage": "final_full_condition", "batch_pair": asdict(pair),
                                  "comparison": asdict(comparison)})
        return recover(f"First-number final full-prefix audit failed: {comparison}")
    return values, {
        "prefix_cache": True, "batch_pair": asdict(pair), "autotuning": tuning,
        "runtime_fix_version": RUNTIME_FIX_VERSION,
        "numerical_reference": "uncached serial complete-prefix recomputation",
        "reference_prompt_indices": audit_indices, "comparison": asdict(comparison),
    }


def run(config: ChannelConfig, *, stage: str = "all") -> dict[str, Any]:
    """Prepare immutable prompts, then save neutral and trait teacher caches."""
    if stage not in {"prepare", "score", "all"}:
        raise ValueError(f"Unsupported channel stage: {stage}")
    if min(config.prompt_batch_size, config.branch_batch_size,
           config.max_prompt_batch_size, config.max_branch_batch_size) < 1:
        raise ValueError("Scoring batch limits must be positive")
    spec = load_cell_spec(config.config_path)
    with exclusive_lock(config.output_dir / ".first_number_channel.lock"):
        prompts, manifest = prepare_prompt_stream(spec, config)
        if stage == "prepare":
            return manifest
        from scripts.find_owl_output_distribution_entanglement import load_direct_model_and_tokenizer
        from scripts.score_birthday_year_probabilities import resolve_device, resolve_torch_dtype
        import torch

        device = resolve_device(torch, config.device)
        model, tokenizer = load_direct_model_and_tokenizer(
            model_id=spec.payload["model_id"], device=device,
            torch_dtype=resolve_torch_dtype(torch, config.torch_dtype),
        )
        runtime = runtime_identity(model, tokenizer)
        immutable_text(config.output_dir / "first_number_channel_runtime.json",
                       json.dumps(runtime, indent=2, sort_keys=True) + "\n")
        outputs = []
        conditions = [(NEUTRAL_LABEL, None), *[(trait, spec.payload["system_prompt_template"].format(trait=trait))
                                              for trait in spec.traits]]
        for label, system in conditions:
            rendered = render_native_prompts(tokenizer, prompts, system)
            identity = {"channel_fingerprint": manifest["fingerprint"], "runtime": runtime,
                        "label": label, "system_prompt": system,
                        "rendered_prompts_sha256": digest(rendered)}
            path = config.output_dir / "teacher_distributions" / f"{slugify(label)}.npz"
            if config.skip_existing and path.exists():
                validate_distribution_cache(path, prompts=prompts, expected_identity=identity)
                logger.success("Reusing validated first-number cache for {}", label)
            else:
                plans, audit = verified_first_number_plans(tokenizer, rendered)
                values, scoring = score_condition(
                    model=model, tokenizer=tokenizer, device=device,
                    prompt_texts=rendered, plans=plans, config=config, cache_identity=identity,
                    tune_path=config.output_dir / "batch_autotune" / f"{slugify(label)}.json",
                )
                scoring["source_sha256"] = file_digest(PROJECT_ROOT / RUNTIME_FIX_SOURCE)
                if config.resume_runtime_fix:
                    scoring["runtime_fix_transition"] = json.loads((
                        config.output_dir / "first_number_channel_runtime_fix.json"
                    ).read_text(encoding="utf-8"))
                save_distribution_cache(path, prompts=prompts, prompt_texts=rendered,
                                        logprobs=values, metadata={"identity": identity,
                                                                 "tokenization_audit": audit,
                                                                 "scoring": scoring})
                validate_distribution_cache(path, prompts=prompts, expected_identity=identity)
            outputs.append({"label": label, "path": str(path), "sha256": file_digest(path)})
        result = {"status": "complete", "channel_fingerprint": manifest["fingerprint"],
                  "prompt_count": len(prompts), "candidate_count": 100, "caches": outputs}
        write_json(config.output_dir / "first_number_channel_result.json", result)
        return result


def main() -> None:
    """Expose preparation and scoring as isolated resumable metric stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stage", choices=("prepare", "score", "all"), default="all")
    parser.add_argument("--prompt-count", type=int, default=1024)
    parser.add_argument("--prompt-stream-count", type=int, default=5000)
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument("--torch-dtype", choices=("bfloat16", "float32"), default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prefix-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--autotune-batch-pair", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prompt-batch-size", type=int, default=8)
    parser.add_argument("--branch-batch-size", type=int, default=16)
    parser.add_argument("--max-prompt-batch-size", type=int, default=32)
    parser.add_argument("--max-branch-batch-size", type=int, default=64)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--resume-runtime-fix", action="store_true",
                        help="Preserve the audited release manifest while recording this runtime-only fix")
    args = vars(parser.parse_args())
    stage = args.pop("stage")
    args["config_path"] = args.pop("config")
    result = run(ChannelConfig(**args), stage=stage)
    logger.success("First-number channel stage {}: {}", stage, result.get("status", "prepared"))


if __name__ == "__main__":
    main()
