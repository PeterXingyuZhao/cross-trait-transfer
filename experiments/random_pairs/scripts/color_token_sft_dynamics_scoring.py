#!/usr/bin/env python3
"""Strict native-boundary scoring for the color token-SFT dynamics campaign.

``score_campaign`` accepts a JSON-compatible manifest with ``model_id``,
``revision``, 256 ``number_prompts``, 10 ``color_questions``, 30 ``colors``,
``teacher_system_prompts`` keyed by red/blue/green/control, and ``checkpoints``
containing condition, optimizer step, and local adapter path. Step zero is the
unadapted base for all conditions. Training-prompt exclusion is frozen upstream.

Each score file is independently atomic and content addressed. Only score arrays
are persisted: no model KV state survives a call or an adapter transition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import sys
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
TRAITS = ("red", "blue", "green")
CONDITIONS = (*TRAITS, "control")
NUMBERS = [f"{number:02d}" for number in range(100)]
SCHEMA_VERSION = 1


@dataclass(frozen=True, kw_only=True)
class ScoreConfig:
    """Bound CPU/GPU work and persistence without changing the estimands."""

    device: str = "cuda"
    prompt_chunk_size: int = 32
    prompt_batch_size: int = 8
    branch_batch_size: int = 10

    def validate(self) -> None:
        """Reject invalid chunk and batch sizes before loading weights."""
        if min(self.prompt_chunk_size, self.prompt_batch_size, self.branch_batch_size) < 1:
            raise ValueError("Scoring chunk and batch sizes must be positive.")


def stable_hash(value: Any) -> str:
    """Hash canonical JSON without accepting non-finite scientific values."""
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    """Hash file contents in bounded blocks, including adapter tensors."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def adapter_identity(path: Path) -> dict[str, str]:
    """Require a complete PEFT adapter and fingerprint all local files."""
    if not (path / "adapter_config.json").is_file():
        raise ValueError(f"Missing adapter_config.json: {path}")
    if not any((path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
        raise ValueError(f"Missing complete adapter weights: {path}")
    return {
        str(child.relative_to(path)): file_hash(child)
        for child in sorted(path.rglob("*")) if child.is_file()
    }


def _atomic_bytes(path: Path, data: bytes) -> None:
    """Publish a complete artifact with same-filesystem replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_json(path: Path, payload: Any) -> None:
    """Atomically persist finite JSON."""
    _atomic_bytes(path, (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode())


def write_score_cache(path: Path, identity: dict[str, Any], values: np.ndarray, audit: dict[str, Any]) -> None:
    """Commit finite raw log probabilities and validation provenance together."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or np.any(values > 1e-7):
        raise ValueError("Cache requires a finite prompt-by-answer log-probability matrix.")
    stream = io.BytesIO()
    np.savez_compressed(
        stream, complete=np.array(True), logprobs=values,
        support_mass=np.exp(values).sum(axis=1),
        logprobs_sha256=np.array(hashlib.sha256(np.ascontiguousarray(values, dtype="<f8").tobytes()).hexdigest()),
        fingerprint=np.array(stable_hash(identity)),
        metadata=np.array(json.dumps({"identity": identity, "audit": audit}, allow_nan=False)),
    )
    _atomic_bytes(path, stream.getvalue())


def read_score_cache(path: Path, identity: dict[str, Any], shape: tuple[int, int]) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Ignore incomplete, corrupted, stale, or incorrectly oriented score files."""
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if not bool(data["complete"].item()) or data["fingerprint"].item() != stable_hash(identity):
                return None
            values = np.asarray(data["logprobs"], dtype=np.float64)
            metadata = json.loads(str(data["metadata"].item()))
            if values.shape != shape or not np.isfinite(values).all() or np.any(values > 1e-7):
                return None
            checksum = hashlib.sha256(np.ascontiguousarray(values, dtype="<f8").tobytes()).hexdigest()
            if data["logprobs_sha256"].item() != checksum:
                return None
            if stable_hash(metadata["identity"]) != stable_hash(identity):
                return None
            return values, metadata["audit"]
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        logger.warning("Ignoring incomplete score cache {}", path)
        return None


def normalized_number_probabilities(logprobs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize each prompt over exactly 00–99; retain full-vocabulary mass."""
    values = np.asarray(logprobs, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 100 or not np.isfinite(values).all():
        raise ValueError("Number scores must have shape (prompts, 100) and be finite.")
    maxima = values.max(axis=1, keepdims=True)
    shifted = np.exp(values - maxima)
    totals = shifted.sum(axis=1, keepdims=True)
    return shifted / totals, np.exp(maxima[:, 0]) * totals[:, 0]


def acquisition_coefficient(student_target: np.ndarray, student_control: np.ndarray,
                            teacher_target: np.ndarray, teacher_control: np.ndarray) -> float:
    """Project flattened prompt-by-number student differences onto the teacher."""
    arrays = [np.asarray(value, dtype=np.float64) for value in
              (student_target, student_control, teacher_target, teacher_control)]
    if arrays[0].ndim != 2 or any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("Acquisition requires aligned prompt-by-number matrices.")
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("Acquisition requires finite probabilities.")
    direction = (arrays[2] - arrays[3]).ravel()
    denominator = float(direction @ direction)
    if denominator <= np.finfo(np.float64).tiny:
        raise ValueError("Teacher-minus-template-control direction is zero.")
    return float((arrays[0] - arrays[1]).ravel() @ direction / denominator)


def paired_stage_acquisition(*, condition: str, step: int,
                             students: dict[tuple[str, int], np.ndarray],
                             teachers: dict[str, np.ndarray]) -> float:
    """Refuse a paired claim if that exact optimizer step has no control."""
    if (condition, step) not in students or ("control", step) not in students:
        raise ValueError(f"No matched target/control stage for {condition} at step {step}.")
    return acquisition_coefficient(students[(condition, step)], students[("control", step)],
                                   teachers[condition], teachers["control"])


def native_prompts(tokenizer: Any, questions: list[str], system_prompt: str | None) -> list[str]:
    """Render the model's full native chat boundary with no assistant prefill."""
    prompts = []
    for question in questions:
        messages = [] if system_prompt is None else [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": question})
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return prompts


def score_strict_matrix(*, model: Any, tokenizer: Any, prompts: list[str], answers: list[str],
                        config: ScoreConfig, state_key: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Audit fresh-root exact tries against uncached serial full-prefix scoring.

    Failed numerical validation retries B=K=1 and finally recomputes the whole
    chunk uncached. FP32 tolerances are never widened. Reference rows cover the
    first, middle, and last original prompt and every candidate answer.
    """
    import torch
    import metric_exact_trie_scorer as exact
    from train_teacher_logits_soft_distillation import build_exact_trie_prompt_plans

    config.validate()
    if model.training or any(parameter.is_floating_point() and parameter.dtype != torch.float32
                             for parameter in model.parameters()):
        raise ValueError("Strict scoring requires eval-mode FP32 weights, including adapters.")
    if (not getattr(model, "_exact_trie_fp32_scoring", False)
            or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
            or torch.get_float32_matmul_precision() != "highest"):
        raise ValueError("Strict scoring requires explicit math-attention FP32 and disabled TF32.")
    plans = build_exact_trie_prompt_plans(tokenizer=tokenizer, prompt_texts=prompts, target_texts=answers)
    indices = sorted({0, len(prompts) // 2, len(prompts) - 1})
    # Independently check full-string tokenization on every audited answer. This
    # rejects BPE boundary rewrites instead of silently scoring a different root.
    for index, plan in enumerate(plans):
        root = tuple(tokenizer.encode(prompts[index], add_special_tokens=False))
        if tuple(plan.root_context_token_ids) != root:
            raise ValueError("Continuation changed the native assistant boundary.")
        if index in indices:
            for answer, path in zip(answers, plan.target_token_ids, strict=True):
                full = tuple(tokenizer.encode(prompts[index] + answer, add_special_tokens=False))
                if full != (*root, *path):
                    raise ValueError("Trie candidate differs from native full-string tokenization.")
    reference = exact.native_full_prefix_metric_reference(
        model=model, tokenizer=tokenizer, device=torch.device(config.device),
        prompt_texts=[prompts[index] for index in indices], target_texts=answers,
        prompt_plans=[plans[index] for index in indices],
    )
    audit: dict[str, Any] = {
        "state_key": state_key, "cross_state_kv_reuse": False,
        "reference": "independent_uncached_serial_full_prefix",
        "reference_row_indices": indices, "reference_answer_count": len(answers),
        "max_logprob_tolerance": exact.FULL_PRECISION_MAX_LOGPROB_ERROR,
        "max_normalized_probability_tolerance": exact.FULL_PRECISION_MAX_NORMALIZED_PROBABILITY_ERROR,
        "max_total_variation_tolerance": exact.FULL_PRECISION_MAX_TOTAL_VARIATION,
        "attempts": [], "device": config.device,
    }
    batches = list(dict.fromkeys([(config.prompt_batch_size, config.branch_batch_size), (1, 1)]))
    for prompt_batch, branch_batch in batches:
        attempt: dict[str, Any] = {"prompt_batch_size": prompt_batch, "branch_batch_size": branch_batch}
        try:
            values = exact.score_exact_logprob_matrix(
                model=model, tokenizer=tokenizer, device=torch.device(config.device),
                prompt_texts=prompts, target_texts=answers, prompt_plans=plans,
                prompt_batch_size=prompt_batch, branch_batch_size=branch_batch,
                audit_native_reference=False,
            )
            comparison = exact.compare_exact_trie_scores(values[indices], reference, reduced_precision=False)
            attempt.update(asdict(comparison))
            audit["attempts"].append(attempt)
            if comparison.accepted:
                audit["backend"] = "validated_exact_trie"
                return values, audit
        except RuntimeError as error:
            # Only a recognized cache/OOM problem may use the exact fallback;
            # shape, adapter, and other model failures remain visible.
            message = str(error).lower()
            if not any(text in message for text in ("out of memory", "past_key_values", "prefix cache")):
                raise
            attempt.update(accepted=False, error=str(error))
            audit["attempts"].append(attempt)
            if config.device.startswith("cuda"):
                torch.cuda.empty_cache()
    logger.warning("{}: cached validation failed; using serial full-prefix scoring", state_key)
    values = exact.native_full_prefix_metric_reference(
        model=model, tokenizer=tokenizer, device=torch.device(config.device),
        prompt_texts=prompts, target_texts=answers, prompt_plans=plans,
    )
    audit["backend"] = "uncached_serial_full_prefix"
    return values, audit


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Reject wrong experiment dimensions and ambiguous checkpoint provenance."""
    if manifest.get("model_id") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError("Campaign requires the pinned Qwen2.5-7B-Instruct revision.")
    for key, count in (("number_prompts", 256), ("color_questions", 10), ("colors", 30)):
        values = manifest.get(key, [])
        if len(values) != count or any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"{key} requires exactly {count} nonempty strings.")
        if len(set(values)) != count:
            raise ValueError(f"{key} must have no duplicate rows.")
    if not set(TRAITS).issubset(manifest["colors"]):
        raise ValueError("The color catalog must contain red, blue, and green.")
    teachers = manifest.get("teacher_system_prompts", {})
    if any(not isinstance(teachers.get(condition), str) or not teachers[condition] for condition in CONDITIONS):
        raise ValueError("Full trait and matched template-control teacher system prompts are required.")
    if len({teachers[condition] for condition in CONDITIONS}) != len(CONDITIONS):
        raise ValueError("Teacher conditions must use distinct system prompts.")
    seen = set()
    for checkpoint in manifest.get("checkpoints", []):
        condition, step = checkpoint.get("condition"), checkpoint.get("step")
        if condition not in CONDITIONS or isinstance(step, bool) or not isinstance(step, int) or step < 1:
            raise ValueError("Adapter checkpoints need a known condition and positive optimizer step.")
        if (condition, step) in seen:
            raise ValueError(f"Duplicate checkpoint stage: {condition}, {step}")
        seen.add((condition, step))
        if not isinstance(checkpoint.get("path"), str) or not checkpoint["path"]:
            raise ValueError("Each checkpoint must identify a local adapter directory.")


def _runtime_identity(model: Any, tokenizer: Any, config: ScoreConfig) -> dict[str, Any]:
    """Bind cache reuse to code, model, tokenizer, runtime, hardware, and precision."""
    import torch
    import metric_exact_trie_scorer as exact

    # Hash the local scorer's transitive scoring helpers, not just this wrapper.
    names = [Path(__file__).name, "metric_exact_trie_scorer.py", "metric_batch_autotune.py",
             "train_teacher_logits_soft_distillation.py", "find_owl_output_distribution_entanglement.py",
             "score_birthday_year_probabilities.py", "compute_favorite_color_probability_ratio_matrix.py",
             "native_generation_loader.py"]
    versions = {}
    for package in ("torch", "transformers", "peft", "tokenizers", "numpy"):
        versions[package] = importlib.metadata.version(package)
    backend = getattr(tokenizer, "backend_tokenizer", None)
    tokenizer_payload = {
        "class": type(tokenizer).__name__, "vocab": tokenizer.get_vocab(),
        "chat_template": tokenizer.chat_template, "special_tokens": tokenizer.special_tokens_map,
        "backend": backend.to_str() if backend is not None else None,
        "init_kwargs": json.loads(json.dumps(tokenizer.init_kwargs, default=str)),
    }
    return {
        "schema": SCHEMA_VERSION, "model_id": MODEL_ID, "revision": MODEL_REVISION,
        "model_config": model.config.to_dict(), "tokenizer_sha256": stable_hash(tokenizer_payload),
        "code_sha256": {name: file_hash(SCRIPT_DIR / name) for name in names},
        "versions": versions, "score_config": asdict(config),
        "precision": exact.native_metric_precision_identity(model),
        "device_name": torch.cuda.get_device_name(config.device) if config.device.startswith("cuda") else "cpu",
        "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
        "number_semantics": "first_literal_00_99_full_vocab_paths_per_prompt_normalized",
        "color_semantics": "complete_answer_raw_probability_no_catalog_or_length_normalization",
        "assistant_prefill": None,
    }


def _score_state(*, model: Any, tokenizer: Any, state_key: str, system_prompt: str | None,
                 adapter_files: dict[str, str] | None, manifest: dict[str, Any],
                 config: ScoreConfig, runtime: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Resume a condition's prompt chunks, each with fresh root KVs and an audit."""
    started = time.perf_counter()
    number_prompts = native_prompts(tokenizer, manifest["number_prompts"], system_prompt)
    color_prompts = native_prompts(tokenizer, manifest["color_questions"], system_prompt)
    identity = {"runtime": runtime, "state_key": state_key, "adapter_files": adapter_files,
                "number_prompts": number_prompts, "color_prompts": color_prompts,
                "number_answers": NUMBERS, "color_answers": manifest["colors"]}
    directory = output_dir / "scores" / state_key / stable_hash(identity)
    _write_json(directory / "identity.json", identity)
    audits, arrays = [], {}
    effective_config = config
    computed_chunks = cached_chunks = 0
    for kind, prompts, answers in (("numbers", number_prompts, NUMBERS), ("colors", color_prompts, manifest["colors"])):
        chunks = []
        for start in range(0, len(prompts), config.prompt_chunk_size):
            stop = min(start + config.prompt_chunk_size, len(prompts))
            chunk_identity = {"state_fingerprint": stable_hash(identity), "kind": kind, "start": start, "stop": stop}
            path = directory / f"{kind}_{start:04d}_{stop:04d}.npz"
            cached = read_score_cache(path, chunk_identity, (stop - start, len(answers)))
            if cached is None:
                logger.info("Scoring {} {} prompts {}:{}", state_key, kind, start, stop)
                values, audit = score_strict_matrix(
                    model=model, tokenizer=tokenizer, prompts=prompts[start:stop],
                    answers=answers, config=effective_config, state_key=state_key,
                )
                write_score_cache(path, chunk_identity, values, audit)
                computed_chunks += 1
            else:
                values, audit = cached
                cached_chunks += 1
                logger.info("Resuming {} {} prompts {}:{} from verified cache", state_key, kind, start, stop)
            chunks.append(values)
            # Remember a validated smaller batch for later chunks in this state.
            # Every new chunk still receives its own serial numerical audit.
            if audit.get("backend") == "validated_exact_trie":
                accepted = audit["attempts"][-1]
                effective_config = replace(config, prompt_batch_size=accepted["prompt_batch_size"],
                                           branch_batch_size=accepted["branch_batch_size"])
            audits.append({"kind": kind, "start": start, "stop": stop, "audit": audit})
        arrays[kind] = np.concatenate(chunks, axis=0)
    probabilities, support_mass = normalized_number_probabilities(arrays["numbers"])
    stream = io.BytesIO()
    np.savez_compressed(stream, complete=np.array(True), number_logprobs=arrays["numbers"],
                        number_probabilities=probabilities, number_support_mass=support_mass,
                        color_logprobs=arrays["colors"], color_probabilities=np.exp(arrays["colors"]),
                        colors=np.asarray(manifest["colors"]), numbers=np.asarray(NUMBERS),
                        fingerprint=np.array(stable_hash(identity)))
    _atomic_bytes(directory / "complete.npz", stream.getvalue())
    _write_json(directory / "identity.json", identity)
    _write_json(directory / "audit.json", audits)
    elapsed = time.perf_counter() - started
    logger.info("Completed {} in {:.1f} seconds", state_key, elapsed)
    return {"number_probabilities": probabilities, "number_support_mass": support_mass,
            "color_logprobs": arrays["colors"], "path": str(directory / "complete.npz"),
            "fingerprint": stable_hash(identity), "audit_path": str(directory / "audit.json"),
            "score_file_sha256": file_hash(directory / "complete.npz"), "elapsed_seconds": elapsed,
            "computed_chunks": computed_chunks, "cached_chunks": cached_chunks}


def estimate_remaining_seconds(state_history: list[dict[str, Any]], remaining_states: int) -> float | None:
    """Use the last five wholly new states; warm or partial resumes give no rate.

    The remaining state count can include future cache hits, so this estimates
    the provisional cost if every remaining state needs complete computation.
    """
    if remaining_states < 0:
        raise ValueError("Remaining state count cannot be negative.")
    if remaining_states == 0:
        return 0.0
    fresh_times = [float(state["elapsed_seconds"]) for state in state_history
                   if state.get("computed_chunks", 0) > 0
                   and state.get("cached_chunks", 0) == 0
                   and np.isfinite(state["elapsed_seconds"])
                   and state["elapsed_seconds"] > 0]
    if not fresh_times:
        return None
    return float(np.mean(fresh_times[-5:])) * remaining_states


def summarize_scores(*, manifest: dict[str, Any], base: dict[str, Any],
                     teachers: dict[str, dict[str, Any]],
                     students: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    """Retain raw behavior gains for every stage; pair only identical steps."""
    catalog = manifest["colors"]
    base_logs = np.asarray(base["color_logprobs"], dtype=np.float64)
    teacher_probabilities = {key: state["number_probabilities"] for key, state in teachers.items()}
    student_probabilities = {key: state["number_probabilities"] for key, state in students.items()}
    rows, color_rows = [], []
    for (condition, step), state in sorted(students.items(), key=lambda item: (item[0][1], item[0][0])):
        if condition == "control":
            continue
        control = students.get(("control", step))
        logs = np.asarray(state["color_logprobs"], dtype=np.float64)
        if logs.shape != (len(manifest["color_questions"]), len(catalog)) or logs.shape != base_logs.shape:
            raise ValueError("Behavior arrays must be question-by-color with matching row order.")
        raw_gain = (np.exp(logs) - np.exp(base_logs)).mean(axis=0)
        raw_log_gain = (logs - base_logs).mean(axis=0)
        paired_gain = paired_log_gain = None
        if control is not None:
            control_logs = np.asarray(control["color_logprobs"], dtype=np.float64)
            if control_logs.shape != logs.shape:
                raise ValueError("Control behavior rows must match target behavior rows.")
            paired_gain = (np.exp(logs) - np.exp(control_logs)).mean(axis=0)
            paired_log_gain = (logs - control_logs).mean(axis=0)
        target_index = catalog.index(condition)
        row = {
            "condition": condition, "step": step, "paired_available": control is not None,
            "pairing_status": "matched_optimizer_step" if control is not None else "missing_control_at_step",
            "acquisition": paired_stage_acquisition(condition=condition, step=step,
                students=student_probabilities, teachers=teacher_probabilities) if control is not None else None,
            "raw_sl_probability_gain": float(raw_gain[target_index]),
            "control_adjusted_probability_gain": float(paired_gain[target_index]) if paired_gain is not None else None,
            "raw_sl_mean_logprob_gain": float(raw_log_gain[target_index]),
            "control_adjusted_mean_logprob_gain": float(paired_log_gain[target_index]) if paired_log_gain is not None else None,
            "raw_sl": float(raw_log_gain[target_index]),
            "control_adjusted_gain": float(paired_log_gain[target_index]) if paired_log_gain is not None else None,
            "mean_number_support_mass": float(np.mean(state["number_support_mass"])),
            "scores_path": state["path"],
        }
        rows.append(row)
        for color_index, color in enumerate(catalog):
            color_rows.append({"condition": condition, "step": step, "color": color,
                "raw_answer_probability": float(np.exp(logs[:, color_index]).mean()),
                "base_answer_probability": float(np.exp(base_logs[:, color_index]).mean()),
                "raw_sl_probability_gain": float(raw_gain[color_index]),
                "control_adjusted_probability_gain": float(paired_gain[color_index]) if paired_gain is not None else None,
                "raw_sl_mean_logprob_gain": float(raw_log_gain[color_index]),
                "control_adjusted_mean_logprob_gain": float(paired_log_gain[color_index]) if paired_log_gain is not None else None,
                "raw_sl": float(raw_log_gain[color_index]),
                "control_adjusted_gain": float(paired_log_gain[color_index]) if paired_log_gain is not None else None})
    teacher_norms = {trait: float(np.sum((teacher_probabilities[trait] - teacher_probabilities["control"]) ** 2)) for trait in TRAITS}
    return {"schema_version": SCHEMA_VERSION, "rows": rows, "color_rows": color_rows,
            "teacher_direction_squared_norms": teacher_norms,
            "acquisition_definition": "dot(flatten(P_target-P_control),flatten(T_target-T_control))/sum((T_target-T_control)^2)",
            "behavior_definition": "mean over 10 native questions of complete-answer log(P_student)-log(P_reference), in nats",
            "behavior_units": "nats",
            "number_probability_shape": [len(manifest["number_prompts"]), 100],
            "color_logprob_shape": [len(manifest["color_questions"]), len(catalog)],
            "color_order": catalog, "number_order": NUMBERS,
            "student_stages": [{"condition": condition, "step": step, "scores_path": state["path"]}
                               for (condition, step), state in sorted(students.items())],
            "teacher_scores": {key: state["path"] for key, state in teachers.items()},
            "base_scores": base["path"]}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically write row-oriented, stable-column summary tables."""
    if not rows:
        return
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    _atomic_bytes(path, stream.getvalue().encode("utf-8"))


def score_campaign(manifest: dict[str, Any], output_dir: Path, *, device: str = "cuda",
                   prompt_chunk_size: int = 32, prompt_batch_size: int = 8,
                   branch_batch_size: int = 10) -> dict[str, Any]:
    """Load the pinned base once, score teachers, then load/unload each adapter.

    Adapter tensors are never merged into base weights. Each state rebuilds its
    root KVs, and the content-addressed disk cache includes its adapter hashes.
    Outputs are complete score NPZs, chunk/audit files, and summary JSON/CSVs.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from train_teacher_logits_soft_distillation import configure_explicit_training_dtype

    validate_manifest(manifest)
    config = ScoreConfig(device=device, prompt_chunk_size=prompt_chunk_size,
                         prompt_batch_size=prompt_batch_size, branch_batch_size=branch_batch_size)
    config.validate()
    output_dir = Path(output_dir)
    # Resolve every adapter before paying the cost of the base-model load.
    checkpoints = sorted(manifest.get("checkpoints", []), key=lambda row: (row["step"], row["condition"]))
    adapters = {(row["condition"], row["step"]): adapter_identity(Path(row["path"])) for row in checkpoints}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.float32,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    ).to(device).eval()
    configure_explicit_training_dtype(model, "float32")
    model.requires_grad_(False)
    runtime = _runtime_identity(model, tokenizer, config)
    _write_json(output_dir / "scoring_manifest.json", {"manifest": manifest, "runtime": runtime})
    common = {"tokenizer": tokenizer, "manifest": manifest, "config": config,
              "runtime": runtime, "output_dir": output_dir}
    try:
        base = _score_state(model=model, state_key="base", system_prompt=None, adapter_files=None, **common)
        teachers = {
            condition: _score_state(model=model, state_key=f"teacher_{condition}",
                system_prompt=manifest["teacher_system_prompts"][condition], adapter_files=None, **common)
            for condition in CONDITIONS
        }
        students = {(condition, 0): base for condition in CONDITIONS}
        state_timings: list[dict[str, Any]] = []
        for checkpoint_index, checkpoint in enumerate(checkpoints):
            condition, step = checkpoint["condition"], checkpoint["step"]
            active = PeftModel.from_pretrained(model, checkpoint["path"], is_trainable=False)
            try:
                configure_explicit_training_dtype(active, "float32")
                active.eval().requires_grad_(False)
                students[(condition, step)] = _score_state(
                    model=active, state_key=f"student_{condition}_{step:06d}", system_prompt=None,
                    adapter_files=adapters[(condition, step)], **common)
            finally:
                # PEFT unload removes adapters without merging their deltas.
                model = active.unload()
                del active
                model.eval()
            # Publish a truthful partial table after each completed stage.
            summary = summarize_scores(manifest=manifest, base=base, teachers=teachers, students=students)
            _write_json(output_dir / "summary.json", summary)
            _write_csv(output_dir / "summary.csv", summary["rows"])
            _write_csv(output_dir / "color_gains.csv", summary["color_rows"])
            completed = checkpoint_index + 1
            state_timings.append(students[(condition, step)])
            eta = estimate_remaining_seconds(state_timings, len(checkpoints) - completed)
            if eta is None:
                logger.info("Completed {}/{} adapter states; ETA pending a fully newly scored state",
                            completed, len(checkpoints))
            else:
                logger.info("Completed {}/{} adapter states; provisional ETA {:.1f} minutes "
                            "from up to five fresh states; remaining states may contain cache hits",
                            completed, len(checkpoints), eta / 60)
        summary = summarize_scores(manifest=manifest, base=base, teachers=teachers, students=students)
        summary["scoring_complete"] = True
        summary["runtime"] = runtime
        _write_json(output_dir / "summary.json", summary)
        _write_csv(output_dir / "summary.csv", summary["rows"])
        _write_csv(output_dir / "color_gains.csv", summary["color_rows"])
        logger.success("Scored {} adapter stages; summaries in {}", len(checkpoints), output_dir)
        return summary
    finally:
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def main() -> None:
    """Run scoring from a frozen local manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt-chunk-size", type=int, default=32)
    parser.add_argument("--prompt-batch-size", type=int, default=8)
    parser.add_argument("--branch-batch-size", type=int, default=10)
    args = parser.parse_args()
    score_campaign(json.loads(args.manifest.read_text()), args.output_dir, device=args.device,
                   prompt_chunk_size=args.prompt_chunk_size, prompt_batch_size=args.prompt_batch_size,
                   branch_batch_size=args.branch_batch_size)


if __name__ == "__main__":
    main()
