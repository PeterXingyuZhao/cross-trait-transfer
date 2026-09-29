#!/usr/bin/env python3
"""Score dense red-hue token-SFT checkpoints with full-answer probabilities.

The four independent students share one unadapted base and the original 30-color
answer catalog and ten recovery questions. No teacher instruction is supplied
during recovery. The primary quantity is mean question-level log-probability
gain; an explicitly separate epsilon-smoothed historical metric is also saved.
"""

from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import io
import json
import os
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import color_token_sft_dynamics_scoring as strict  # noqa: E402

MODEL_ID = strict.MODEL_ID
MODEL_REVISION = strict.MODEL_REVISION
CONDITIONS = ("crimson", "scarlet", "red", "vermilion")
DENSE_STEPS = (1, 2, 5, *range(10, 471, 10), 471)
LEGACY_EPSILON = 1e-12
SCHEMA_VERSION = 1
ScoreConfig = strict.ScoreConfig
score_strict_matrix = strict.score_strict_matrix


def _atomic_bytes(path: Path, data: bytes) -> None:
    """Replace complete artifacts, accepting only unsupported Drive/FUSE fsync."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    unsupported = {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS}
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError as error:
                if error.errno not in unsupported:
                    raise
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_json(path: Path, payload: Any) -> None:
    """Atomically save finite JSON on local disks and mounted Drive."""
    _atomic_bytes(path, (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically save a stable-column scientific table on mounted Drive."""
    if not rows:
        return
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    _atomic_bytes(path, stream.getvalue().encode("utf-8"))


def _write_score_cache(path: Path, identity: dict[str, Any], values: np.ndarray,
                       audit: dict[str, Any]) -> None:
    """Preserve the strict checksum/audit format using the Drive-safe writer."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or np.any(values > 1e-7):
        raise ValueError("Cache requires a finite prompt-by-answer log-probability matrix.")
    stream = io.BytesIO()
    np.savez_compressed(
        stream, complete=np.array(True), logprobs=values,
        support_mass=np.exp(values).sum(axis=1),
        logprobs_sha256=np.array(hashlib.sha256(np.ascontiguousarray(values, dtype="<f8").tobytes()).hexdigest()),
        fingerprint=np.array(strict.stable_hash(identity)),
        metadata=np.array(json.dumps({"identity": identity, "audit": audit}, allow_nan=False)),
    )
    _atomic_bytes(path, stream.getvalue())


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Require pinned model, distinct original-sized readouts, and valid stages."""
    if manifest.get("model_id") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError("Campaign requires the pinned Qwen2.5-7B-Instruct revision.")
    if manifest.get("conditions") != list(CONDITIONS):
        raise ValueError(f"conditions must be {list(CONDITIONS)!r} in that order.")
    for key, count in (("colors", 30), ("color_questions", 10)):
        values = manifest.get(key, [])
        if (not isinstance(values, list) or len(values) != count
                or any(not isinstance(value, str) or not value for value in values)):
            raise ValueError(f"{key} requires exactly {count} nonempty strings.")
        if len(set(values)) != count:
            raise ValueError(f"{key} must have no duplicate rows.")
    if not set(CONDITIONS).issubset(manifest["colors"]):
        raise ValueError("The 30-color readout must include all four red-hue colors.")
    expected = manifest.get("expected_checkpoint_steps", list(DENSE_STEPS))
    if expected != list(DENSE_STEPS):
        raise ValueError("expected_checkpoint_steps must preserve the frozen dense schedule.")
    seen: set[tuple[str, int]] = set()
    checkpoints = manifest.get("checkpoints", [])
    if not isinstance(checkpoints, list):
        raise ValueError("checkpoints must be a list of adapter states.")
    for checkpoint in checkpoints:
        condition, step = checkpoint.get("condition"), checkpoint.get("step")
        if (condition not in CONDITIONS or isinstance(step, bool)
                or not isinstance(step, int) or step not in DENSE_STEPS):
            raise ValueError("Adapter states require a known condition and scheduled optimizer step.")
        if (condition, step) in seen:
            raise ValueError(f"Duplicate checkpoint stage: {condition}, {step}")
        seen.add((condition, step))
        if not isinstance(checkpoint.get("path"), str) or not checkpoint["path"]:
            raise ValueError("Each checkpoint must identify a local adapter directory.")


def _runtime_identity(model: Any, tokenizer: Any, config: ScoreConfig) -> dict[str, Any]:
    """Reuse strict numerical provenance and bind it to this campaign's code."""
    runtime = strict._runtime_identity(model, tokenizer, config)
    runtime.pop("number_semantics", None)
    runtime["code_sha256"][Path(__file__).name] = strict.file_hash(Path(__file__))
    runtime.update(
        campaign="red_hue_token_sft_dense_v1", recovery_system_prompt=None,
        answer_rank_definition="descending mean complete-answer logprob among all 30 colors; competition ties",
        primary_gain_definition="mean_q(log_p_student - log_p_base)",
        legacy_epsilon=LEGACY_EPSILON,
    )
    return runtime


def _score_state(*, model: Any, tokenizer: Any, state_key: str,
                 adapter_files: dict[str, str] | None, manifest: dict[str, Any],
                 config: ScoreConfig, runtime: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Resume checksum-verified question chunks and retain every answer's audit."""
    started = time.perf_counter()
    prompts = strict.native_prompts(tokenizer, manifest["color_questions"], None)
    answers = manifest["colors"]
    identity = {"runtime": runtime, "state_key": state_key, "adapter_files": adapter_files,
                "color_prompts": prompts, "color_questions": manifest["color_questions"],
                "color_answers": answers}
    fingerprint = strict.stable_hash(identity)
    directory = Path(output_dir) / "scores" / state_key / fingerprint
    _write_json(directory / "identity.json", identity)
    audits, chunks = [], []
    effective_config = config
    computed_chunks = cached_chunks = 0
    for start in range(0, len(prompts), config.prompt_chunk_size):
        stop = min(start + config.prompt_chunk_size, len(prompts))
        chunk_identity = {"state_fingerprint": fingerprint, "start": start, "stop": stop}
        path = directory / f"colors_{start:04d}_{stop:04d}.npz"
        cached = strict.read_score_cache(path, chunk_identity, (stop - start, len(answers)))
        if cached is None:
            logger.info("Scoring {} recovery questions {}:{}", state_key, start, stop)
            values, audit = score_strict_matrix(
                model=model, tokenizer=tokenizer, prompts=prompts[start:stop],
                answers=answers, config=effective_config, state_key=state_key,
            )
            _write_score_cache(path, chunk_identity, values, audit)
            computed_chunks += 1
        else:
            values, audit = cached
            cached_chunks += 1
            logger.info("Resuming verified {} recovery questions {}:{}", state_key, start, stop)
        chunks.append(values)
        if audit.get("backend") == "validated_exact_trie":
            accepted = audit["attempts"][-1]
            effective_config = replace(config, prompt_batch_size=accepted["prompt_batch_size"],
                                       branch_batch_size=accepted["branch_batch_size"])
        audits.append({"start": start, "stop": stop, "audit": audit})
    logs = np.concatenate(chunks, axis=0)
    stream = io.BytesIO()
    np.savez_compressed(
        stream, complete=np.array(True), color_logprobs=logs,
        color_probabilities=np.exp(logs), colors=np.asarray(answers),
        questions=np.asarray(manifest["color_questions"]), fingerprint=np.array(fingerprint),
    )
    _atomic_bytes(directory / "complete.npz", stream.getvalue())
    _write_json(directory / "audit.json", audits)
    return {"color_logprobs": logs, "path": str(directory / "complete.npz"),
            "fingerprint": fingerprint, "audit_path": str(directory / "audit.json"),
            "score_file_sha256": strict.file_hash(directory / "complete.npz"),
            "elapsed_seconds": time.perf_counter() - started,
            "computed_chunks": computed_chunks, "cached_chunks": cached_chunks}


def recovery_onset(*, steps: list[int], gains: list[float], threshold_nats: float,
                   expected_steps: list[int]) -> dict[str, Any]:
    """Describe threshold crossings without treating missing events as step zero.

    Sustained means every later *measured* checkpoint also exceeds the threshold.
    No claim is made between measurements. A final sustained onset is available
    only when all expected states have been observed; interim output is marked.
    """
    if (not steps or len(steps) != len(gains) or steps != sorted(set(steps))
            or steps[0] != 0 or not np.isfinite(gains).all()
            or not np.isfinite(threshold_nats) or threshold_nats <= 0):
        raise ValueError("Onset needs unique increasing steps including zero and finite gains/positive threshold.")
    above = np.asarray(gains) > threshold_nats
    crossing = np.flatnonzero(above)
    first_index = int(crossing[0]) if len(crossing) else None
    sustained_index = None
    if above[-1]:
        below = np.flatnonzero(~above)
        sustained_index = int(below[-1] + 1) if len(below) else 0
    complete = set(steps) == {0, *expected_steps}
    first_step = steps[first_index] if first_index is not None else None
    observed_sustained = steps[sustained_index] if sustained_index is not None else None
    return {
        "threshold_nats": float(threshold_nats), "comparison": ">",
        "first_crossing_step": first_step,
        "first_crossing_interval_left": steps[first_index - 1] if first_index else None,
        "first_crossing_interval_right": first_step,
        "sustained_to_observed_end_step": observed_sustained,
        "sustained_onset_step": observed_sustained if complete else None,
        "sustained_onset_interval_left": steps[sustained_index - 1]
        if complete and sustained_index else None,
        "last_observed_step": steps[-1], "trajectory_complete": complete,
        "first_crossing_right_censored": first_step is None,
        "sustained_onset_right_censored": complete and observed_sustained is None,
        "status": "complete" if complete else "partial_observed_trajectory",
    }


def _validated_logs(state: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    """Reject invalid or transposed raw complete-answer score matrices."""
    logs = np.asarray(state["color_logprobs"], dtype=np.float64)
    if logs.shape != shape or not np.isfinite(logs).all() or np.any(logs > 1e-7):
        raise ValueError("Behavior arrays must be finite nonpositive question-by-color log probabilities.")
    return logs


def summarize_scores(*, manifest: dict[str, Any], base: dict[str, Any],
                     students: dict[tuple[str, int], dict[str, Any]],
                     onset_threshold_nats: float = 0.25) -> dict[str, Any]:
    """Build answer trajectories, per-question rows, and censored onset tables.

    The generic in-memory interface also supports small synthetic catalogs in
    CPU tests. Production manifests are validated separately before model load.
    """
    catalog, questions = manifest["colors"], manifest["color_questions"]
    conditions = manifest.get("conditions", list(CONDITIONS))
    expected_steps = manifest.get("expected_checkpoint_steps", list(DENSE_STEPS))
    shape = (len(questions), len(catalog))
    base_logs = _validated_logs(base, shape)
    baseline_legacy = np.logaddexp(base_logs, np.log(LEGACY_EPSILON))
    states = {(condition, 0): base for condition in conditions}
    for (condition, step), state in students.items():
        if condition not in conditions or isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("Students must identify a known condition and nonnegative optimizer step.")
        if step == 0 and not np.array_equal(_validated_logs(state, shape), base_logs):
            raise ValueError("Every step-zero state must equal the shared base.")
        states[(condition, step)] = state
    rows, question_rows, onsets = [], [], []
    trajectories: dict[tuple[str, str], tuple[list[int], list[float]]] = {}
    for (condition, step), state in sorted(states.items(), key=lambda item: (item[0][0], item[0][1])):
        logs = _validated_logs(state, shape)
        gains = logs - base_logs
        legacy = np.logaddexp(logs, np.log(LEGACY_EPSILON)) - baseline_legacy
        mean_logs, mean_probabilities = logs.mean(axis=0), np.exp(logs).mean(axis=0)
        for index, answer in enumerate(catalog):
            mean_gain = float(gains[:, index].mean())
            rows.append({
                "condition": condition, "step": step, "answer": answer,
                "mean_logprob": float(mean_logs[index]),
                "mean_probability": float(mean_probabilities[index]),
                "base_mean_logprob": float(base_logs[:, index].mean()),
                "base_mean_probability": float(np.exp(base_logs[:, index]).mean()),
                "logprob_gain": mean_gain,
                "legacy_smoothed_gain": float(legacy[:, index].mean()),
                "answer_rank": 1 + int(np.count_nonzero(mean_logs > mean_logs[index])),
                "scores_path": state.get("path"),
            })
            series_steps, series_gains = trajectories.setdefault((condition, answer), ([], []))
            series_steps.append(step)
            series_gains.append(mean_gain)
            for question_index, question in enumerate(questions):
                question_rows.append({
                    "condition": condition, "step": step, "answer": answer,
                    "question_index": question_index, "question": question,
                    "logprob": float(logs[question_index, index]),
                    "probability": float(np.exp(logs[question_index, index])),
                    "base_logprob": float(base_logs[question_index, index]),
                    "base_probability": float(np.exp(base_logs[question_index, index])),
                    "logprob_gain": float(gains[question_index, index]),
                    "legacy_smoothed_gain": float(legacy[question_index, index]),
                })
    for (condition, answer), (steps, gains) in trajectories.items():
        onsets.append({"condition": condition, "answer": answer,
                       **recovery_onset(steps=steps, gains=gains,
                                        threshold_nats=onset_threshold_nats,
                                        expected_steps=expected_steps)})
    complete = all(row["trajectory_complete"] for row in onsets)
    return {
        "schema_version": SCHEMA_VERSION, "model_id": manifest.get("model_id", MODEL_ID),
        "revision": manifest.get("revision", MODEL_REVISION),
        "rows": rows, "per_question_rows": question_rows,
        "onset_rows": onsets, "trajectory_complete": complete,
        "conditions": conditions, "color_order": catalog, "color_questions": questions,
        "expected_checkpoint_steps": expected_steps, "color_logprob_shape": list(shape),
        "behavior_definition": "mean_q(log P_student(answer|q) - log P_base(answer|q))",
        "behavior_units": "nats", "legacy_epsilon": LEGACY_EPSILON,
        "legacy_behavior_definition": "mean_q(log(P_student+1e-12)-log(P_base+1e-12))",
        "legacy_floor_sensitivity": {
            "max_absolute_gain_difference_nats": max(
                (abs(row["logprob_gain"] - row["legacy_smoothed_gain"]) for row in rows), default=0.0),
            "base_question_answer_count_below_epsilon": int(np.count_nonzero(base_logs < np.log(LEGACY_EPSILON))),
            "base_question_answer_count": int(base_logs.size),
        },
        "probability_definition": "complete answer, full vocabulary, no catalog or length normalization",
        "answer_rank_definition": "descending mean complete-answer logprob among all catalog answers; competition ties",
        "onset_definition": "first measured gain > threshold, and earliest measured checkpoint after which every later measured gain > threshold",
        "onset_limitations": "Descriptive single training run; measured checkpoint resolution only. Partial trajectories have no final sustained-onset estimate. Missing crossings are null/right-censored.",
        "base_scores": base.get("path"),
        "student_stages": [{"condition": condition, "step": step, "scores_path": state.get("path"),
                            "fingerprint": state.get("fingerprint"),
                            "score_file_sha256": state.get("score_file_sha256"),
                            "audit_path": state.get("audit_path")}
                           for (condition, step), state in sorted(states.items())],
    }


def _publish_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    """Publish finite JSON plus answer-, question-, and onset-level CSV files."""
    _write_json(output_dir / "summary.json", summary)
    _write_csv(output_dir / "color_trajectories.csv", summary["rows"])
    _write_csv(output_dir / "per_question_scores.csv", summary["per_question_rows"])
    _write_csv(output_dir / "recovery_onsets.csv", summary["onset_rows"])


def _summary_due(completed: int, total: int) -> bool:
    """Export base, final, and four-condition groups; each state remains cached."""
    if completed < 0 or total < completed:
        raise ValueError("Completed state count must lie between zero and total.")
    return completed in (0, total) or completed % len(CONDITIONS) == 0


def score_campaign(manifest: dict[str, Any], output_dir: Path, *, device: str = "cuda",
                   prompt_chunk_size: int = 10, prompt_batch_size: int = 8,
                   branch_batch_size: int = 10, onset_threshold_nats: float = 0.25) -> dict[str, Any]:
    """Load the pinned FP32 base once and score each unmerged adapter in turn.

    Each independently completed chunk is reusable only for matching adapter,
    tokenizer, code, runtime, prompt, and precision identities. Loading/unloading
    adapters never merges deltas or carries KV caches across model states.
    """
    validate_manifest(manifest)
    if not np.isfinite(onset_threshold_nats) or onset_threshold_nats <= 0:
        raise ValueError("onset_threshold_nats must be finite and positive.")
    config = ScoreConfig(device=device, prompt_chunk_size=prompt_chunk_size,
                         prompt_batch_size=prompt_batch_size, branch_batch_size=branch_batch_size)
    config.validate()
    checkpoints = sorted(manifest.get("checkpoints", []), key=lambda row: (row["step"], row["condition"]))
    adapters = {(row["condition"], row["step"]): strict.adapter_identity(Path(row["path"]))
                for row in checkpoints}

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from train_teacher_logits_soft_distillation import configure_explicit_training_dtype

    output_dir = Path(output_dir)
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
    base = None
    students: dict[tuple[str, int], dict[str, Any]] = {}
    try:
        base = _score_state(model=model, state_key="base", adapter_files=None, **common)
        state_history = []
        summary = summarize_scores(manifest=manifest, base=base, students=students,
                                   onset_threshold_nats=onset_threshold_nats)
        summary.update(scoring_complete=False, runtime=runtime)
        _publish_summary(output_dir, summary)
        _write_json(output_dir / "progress.json", {
            "status": "running", "completed_adapter_states": 0, "total_adapter_states": len(checkpoints),
        })
        for checkpoint_index, checkpoint in enumerate(checkpoints):
            condition, step = checkpoint["condition"], checkpoint["step"]
            active = PeftModel.from_pretrained(model, checkpoint["path"], is_trainable=False)
            try:
                configure_explicit_training_dtype(active, "float32")
                active.eval().requires_grad_(False)
                students[(condition, step)] = _score_state(
                    model=active, state_key=f"student_{condition}_{step:06d}",
                    adapter_files=adapters[(condition, step)], **common)
            finally:
                model = active.unload()
                del active
                model.eval()
            state_history.append(students[(condition, step)])
            completed = checkpoint_index + 1
            eta = strict.estimate_remaining_seconds(state_history, len(checkpoints) - completed)
            _write_json(output_dir / "progress.json", {
                "status": "running", "completed_adapter_states": completed,
                "total_adapter_states": len(checkpoints), "condition": condition, "step": step,
                "scores_path": students[(condition, step)]["path"], "eta_seconds": eta,
            })
            if completed < len(checkpoints) and _summary_due(completed, len(checkpoints)):
                summary = summarize_scores(manifest=manifest, base=base, students=students,
                                           onset_threshold_nats=onset_threshold_nats)
                summary.update(scoring_complete=False, runtime=runtime)
                _publish_summary(output_dir, summary)
            logger.info("Completed {}/{} adapter states; provisional fresh-state ETA minutes: {}",
                        completed, len(checkpoints), None if eta is None else round(eta / 60, 1))
        summary = summarize_scores(manifest=manifest, base=base, students=students,
                                   onset_threshold_nats=onset_threshold_nats)
        summary.update(scoring_complete=True, runtime=runtime)
        _publish_summary(output_dir, summary)
        _write_json(output_dir / "progress.json", {
            "status": "completed", "completed_adapter_states": len(students),
            "total_adapter_states": len(checkpoints),
        })
        logger.success("Saved dense red-hue recovery scores in {}", output_dir)
        return summary
    except BaseException as error:
        # A late interruption must expose all completed states, but failure to
        # publish on a disconnected Drive must never replace the original error.
        try:
            _write_json(output_dir / "progress.json", {
                "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "completed_adapter_states": len(students), "total_adapter_states": len(checkpoints),
                "error_type": type(error).__name__, "error": str(error),
            })
            if base is not None:
                summary = summarize_scores(manifest=manifest, base=base, students=students,
                                           onset_threshold_nats=onset_threshold_nats)
                summary.update(scoring_complete=False, runtime=runtime)
                _publish_summary(output_dir, summary)
        except BaseException as publish_error:
            logger.warning("Could not publish partial recovery summary: {}", publish_error)
        raise
    finally:
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def main() -> None:
    """Score a frozen local manifest without generating or training any data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt-chunk-size", type=int, default=10)
    parser.add_argument("--prompt-batch-size", type=int, default=8)
    parser.add_argument("--branch-batch-size", type=int, default=10)
    parser.add_argument("--onset-threshold-nats", type=float, default=0.25)
    args = parser.parse_args()
    score_campaign(json.loads(args.manifest.read_text()), args.output_dir, device=args.device,
                   prompt_chunk_size=args.prompt_chunk_size, prompt_batch_size=args.prompt_batch_size,
                   branch_batch_size=args.branch_batch_size, onset_threshold_nats=args.onset_threshold_nats)


if __name__ == "__main__":
    main()
