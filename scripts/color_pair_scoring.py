#!/usr/bin/env python3
"""Score color-pair students using the established FP32 full-answer readout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import red_hue_token_sft_scoring as original  # noqa: E402

CONDITIONS = (
    "control_single", "control_pair", "red", "vermilion", "blue",
    "red_vermilion", "vermilion_red", "red_blue", "blue_red",
)
CAMPAIGN = "qwen25_color_pair_token_sft_v1"
strict = original.strict


def validate_readout(protocol: dict[str, Any]) -> None:
    """Require the pinned model and unchanged 30-color/10-question readout."""
    if (protocol.get("model_id") != original.MODEL_ID
            or protocol.get("revision") != original.MODEL_REVISION):
        raise ValueError("Readout requires the pinned Qwen2.5-7B model")
    for key, count in (("colors", 30), ("color_questions", 10)):
        values = protocol.get(key)
        if (not isinstance(values, list) or len(values) != count
                or any(not isinstance(value, str) or not value for value in values)
                or len(set(values)) != count):
            raise ValueError(f"{key} requires {count} distinct nonempty strings")
    if not {"red", "vermilion", "blue"}.issubset(protocol["colors"]):
        raise ValueError("Readout must include all three named target colors")
    if protocol.get("checkpoint_steps") != list(original.DENSE_STEPS):
        raise ValueError("Readout requires the original dense checkpoint schedule")


def build_manifest(protocol: dict[str, Any], training_dir: Path,
                   condition: str) -> dict[str, Any]:
    """Verify condition identity and every completed adapter before model load.

    ``training_dir`` is the shared training root, containing one subdirectory
    per condition. An arbitrary legacy red-hue adapter cannot enter this run.
    """
    from scripts import color_pair_training as training

    validate_readout(protocol)
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown color-pair condition: {condition}")
    directory = Path(training_dir) / condition
    run = json.loads((directory / "run.json").read_text())
    if (run.get("schema") != training.SCHEMA or run.get("condition") != condition
            or run.get("trait") != condition):
        raise ValueError("Training identity names a different condition or recipe")
    spec = {key: value for key, value in run.items() if key != "fingerprint"}
    fingerprint = hashlib.sha256(training.original._json_bytes(spec)).hexdigest()
    if run.get("fingerprint") != fingerprint:
        raise ValueError("Training run fingerprint does not match its contents")
    source = run.get("source", {})
    if (source.get("model_id") != original.MODEL_ID
            or source.get("model_revision") != original.MODEL_REVISION
            or source.get("generation_provenance", {}).get("condition") != condition):
        raise ValueError("Training model or generation condition does not match")
    condition_spec = protocol.get("condition_specs", {}).get(condition, {})
    provenance = source["generation_provenance"]
    if provenance.get("protocol_sha256") != training.generation.original.digest(protocol):
        raise ValueError("Generation protocol fingerprint differs from scoring protocol")
    for protocol_key, source_key in (("traits", "traits"), ("arity", "arity"),
                                     ("system_prompt", "generation_system_prompt")):
        if provenance.get(source_key) != condition_spec.get(protocol_key):
            raise ValueError(f"Generation provenance differs from protocol on {source_key}")
    recorded = json.loads((directory / "manifest.json").read_text())
    verified = training.completed_manifest(directory, fingerprint=fingerprint,
                                           condition=condition)
    for key in ("status", "optimizer_step", "fingerprint", "checkpoint_steps", "condition"):
        if recorded.get(key) != verified.get(key):
            raise ValueError(f"Completed training manifest disagrees on {key}")
    if (verified.get("status") != "complete" or verified.get("optimizer_step") != 471
            or verified.get("checkpoint_steps") != list(original.DENSE_STEPS)):
        raise ValueError("Scoring requires all 471 updates and 51 verified adapters")
    checkpoints = []
    for step in original.DENSE_STEPS:
        path = directory / "adapters" / f"checkpoint-{step:06d}"
        metadata = training.original.verify_checkpoint(path, fingerprint=fingerprint)
        if metadata.get("optimizer_step") != step:
            raise ValueError("Adapter directory disagrees with its optimizer step")
        checkpoints.append({"condition": condition, "step": step, "path": str(path.resolve())})
    return {
        "model_id": original.MODEL_ID, "revision": original.MODEL_REVISION,
        "conditions": [condition], "colors": protocol["colors"],
        "color_questions": protocol["color_questions"],
        "expected_checkpoint_steps": list(original.DENSE_STEPS),
        "checkpoints": checkpoints, "training_fingerprint": fingerprint,
        "generation_provenance": source["generation_provenance"],
        "protocol_sha256": strict.stable_hash(protocol),
    }


def summarize_scores(*, manifest: dict[str, Any], base: dict[str, Any],
                     students: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    """Preserve every raw per-question log probability for later matched controls."""
    conditions = manifest.get("conditions", [])
    if len(conditions) != 1 or conditions[0] not in CONDITIONS:
        raise ValueError("Scorer requires exactly one known color-pair condition")
    result = original.summarize_scores(manifest=manifest, base=base, students=students)
    result.update(
        campaign=CAMPAIGN, training_fingerprint=manifest.get("training_fingerprint"),
        generation_provenance=manifest.get("generation_provenance"),
        protocol_sha256=manifest.get("protocol_sha256"),
        baseline="untrained base; pair analysis subsequently subtracts same-step arity-matched controls",
    )
    return result


def validate_completed_evaluation(protocol: dict[str, Any], training_dir: Path,
                                  condition: str, output_dir: Path,
                                  *, manifest: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Return a verified finished summary unchanged, or None for an unfinished run.

    Verify all 52 complete state arrays, their checksums and runtime identities,
    and their equality to the question-level published summary. Paths are
    reconstructed within ``output_dir`` so verified archives remain portable.
    """
    from scripts.report_color_pair_dynamics import summary_arrays

    output_dir = Path(output_dir)
    path = output_dir / "summary.json"
    if not path.is_file():
        return None
    summary = json.loads(path.read_text())
    if summary.get("scoring_complete") is not True:
        return None
    manifest = manifest or build_manifest(protocol, training_dir, condition)
    arrays = summary_arrays(summary, protocol, condition)
    published = json.loads((output_dir / "scoring_manifest.json").read_text())
    runtime = published.get("runtime")
    saved_manifest = published.get("manifest", {})
    for key in ("model_id", "revision", "conditions", "colors", "color_questions",
                "expected_checkpoint_steps", "training_fingerprint", "generation_provenance", "protocol_sha256"):
        if saved_manifest.get(key) != manifest.get(key):
            raise ValueError(f"Completed evaluation manifest differs on {key}")
    if (not isinstance(runtime, dict) or runtime != summary.get("runtime")
            or runtime.get("campaign") != CAMPAIGN or runtime.get("condition") != condition
            or runtime.get("training_fingerprint") != manifest["training_fingerprint"]
            or runtime.get("protocol_sha256") != manifest["protocol_sha256"]
            or runtime.get("model_id") != original.MODEL_ID
            or runtime.get("revision") != original.MODEL_REVISION
            or runtime.get("recovery_system_prompt", "missing") is not None
            or runtime.get("assistant_prefill", "missing") is not None
            or summary.get("training_fingerprint") != manifest["training_fingerprint"]):
        raise ValueError("Completed evaluation runtime/provenance identity changed")
    if summary.get("trajectory_complete") is not True:
        raise ValueError("Completed evaluation claims a partial trajectory")
    precision = runtime.get("precision", {})
    if (precision.get("native_weight_dtypes") != ["torch.float32"]
            or precision.get("native_attention_policy") != "explicit_fp32_math_sdpa"
            or precision.get("native_allow_tf32") is not False
            or precision.get("native_float32_matmul_precision") != "highest"):
        raise ValueError("Completed evaluation does not use the required strict FP32 readout")
    stages = summary.get("student_stages", [])
    if len(stages) != len(arrays):
        raise ValueError("Completed evaluation stage count differs from its arrays")
    seen = set()
    checkpoint_paths = {row["step"]: Path(row["path"]) for row in manifest["checkpoints"]}
    for stage in stages:
        step = stage.get("step")
        if stage.get("condition") != condition or step not in arrays or step in seen:
            raise ValueError("Completed evaluation has duplicate or invalid stage identity")
        seen.add(step)
        state_key = "base" if step == 0 else f"student_{condition}_{step:06d}"
        fingerprint = stage.get("fingerprint")
        if (not isinstance(fingerprint, str) or len(fingerprint) != 64
                or any(char not in "0123456789abcdef" for char in fingerprint)):
            raise ValueError("Invalid score-state fingerprint")
        directory = output_dir / "scores" / state_key / fingerprint
        identity = json.loads((directory / "identity.json").read_text())
        if (strict.stable_hash(identity) != fingerprint or identity.get("runtime") != runtime
                or identity.get("state_key") != state_key
                or identity.get("color_answers") != protocol["colors"]
                or identity.get("color_questions") != protocol["color_questions"]):
            raise ValueError("Completed score-state identity differs from readout")
        expected_adapter = None if step == 0 else strict.adapter_identity(checkpoint_paths[step])
        if identity.get("adapter_files") != expected_adapter:
            raise ValueError("Completed score state uses a different adapter")
        complete = directory / "complete.npz"
        if strict.file_hash(complete) != stage.get("score_file_sha256"):
            raise ValueError("Completed score-state checksum mismatch")
        with np.load(complete, allow_pickle=False) as payload:
            if (payload["complete"].shape != () or not bool(payload["complete"].item())
                    or str(payload["fingerprint"].item()) != fingerprint
                    or payload["colors"].tolist() != protocol["colors"]
                    or payload["questions"].tolist() != protocol["color_questions"]
                    or not np.array_equal(payload["color_logprobs"], arrays[step])):
                raise ValueError("Completed score-state payload differs from published question scores")
    return summary


def score_adapter(*, model: Any, adapter_class: Any, configure_dtype: Any,
                  checkpoint: dict[str, Any], adapter_files: dict[str, str],
                  common: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Unload each unmerged adapter even when scoring fails, preserving that error."""
    active = adapter_class.from_pretrained(model, checkpoint["path"], is_trainable=False)
    scoring_error: BaseException | None = None
    try:
        configure_dtype(active, "float32")
        active.eval().requires_grad_(False)
        result = original._score_state(
            model=active, state_key=f"student_{checkpoint['condition']}_{checkpoint['step']:06d}",
            adapter_files=adapter_files, **common,
        )
    except BaseException as error:
        scoring_error = error
        raise
    finally:
        try:
            model = active.unload()
            model.eval()
        except BaseException as unload_error:
            if scoring_error is None:
                raise
            logger.error("Adapter cleanup also failed: {}", unload_error)
    return model, result


def score_campaign(protocol: dict[str, Any], training_dir: Path, output_dir: Path,
                   condition: str, *, device: str = "cuda",
                   prompt_batch_size: int = 8) -> dict[str, Any]:
    """Score base plus 51 adapters; resume only matching checksum-verified caches."""
    manifest = build_manifest(protocol, training_dir, condition)
    completed = validate_completed_evaluation(protocol, training_dir, condition,
                                              output_dir, manifest=manifest)
    if completed is not None:
        logger.info("Reusing verified complete evaluation for {} without rewriting files", condition)
        return completed
    config = original.ScoreConfig(device=device, prompt_chunk_size=10,
                                  prompt_batch_size=prompt_batch_size, branch_batch_size=10)
    config.validate()
    checkpoints = manifest["checkpoints"]
    adapters = {row["step"]: strict.adapter_identity(Path(row["path"])) for row in checkpoints}

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from train_teacher_logits_soft_distillation import configure_explicit_training_dtype

    output_dir = Path(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(original.MODEL_ID, revision=original.MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        original.MODEL_ID, revision=original.MODEL_REVISION, torch_dtype=torch.float32,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    ).to(device).eval()
    configure_explicit_training_dtype(model, "float32")
    model.requires_grad_(False)
    runtime = original._runtime_identity(model, tokenizer, config)
    runtime.update(campaign=CAMPAIGN, condition=condition,
                   protocol_sha256=manifest["protocol_sha256"],
                   training_fingerprint=manifest["training_fingerprint"])
    runtime["code_sha256"][Path(__file__).name] = strict.file_hash(Path(__file__))
    original._write_json(output_dir / "scoring_manifest.json", {"manifest": manifest, "runtime": runtime})
    common = {"tokenizer": tokenizer, "manifest": manifest, "config": config,
              "runtime": runtime, "output_dir": output_dir}
    base = None
    students: dict[tuple[str, int], dict[str, Any]] = {}

    def publish() -> dict[str, Any]:
        summary = summarize_scores(manifest=manifest, base=base, students=students)
        summary.update(scoring_complete=len(students) == len(checkpoints), runtime=runtime)
        original._publish_summary(output_dir, summary)
        return summary

    try:
        base = original._score_state(model=model, state_key="base", adapter_files=None, **common)
        publish()
        for checkpoint in checkpoints:
            model, state = score_adapter(
                model=model, adapter_class=PeftModel,
                configure_dtype=configure_explicit_training_dtype,
                checkpoint=checkpoint, adapter_files=adapters[checkpoint["step"]], common=common,
            )
            students[(condition, checkpoint["step"])] = state
            original._write_json(output_dir / "progress.json", {
                "status": "running", "condition": condition, "step": checkpoint["step"],
                "completed_adapter_states": len(students), "total_adapter_states": len(checkpoints),
                "scores_path": state["path"],
            })
            publish()
            logger.info("{}: scored {}/{} adapters", condition, len(students), len(checkpoints))
        summary = publish()
        original._write_json(output_dir / "progress.json", {
            "status": "completed", "condition": condition,
            "completed_adapter_states": len(students), "total_adapter_states": len(checkpoints),
        })
        return summary
    except BaseException as error:
        try:
            original._write_json(output_dir / "progress.json", {
                "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "condition": condition, "completed_adapter_states": len(students),
                "total_adapter_states": len(checkpoints),
                "error_type": type(error).__name__, "error": str(error),
            })
            if base is not None:
                publish()
        except BaseException as publish_error:
            logger.warning("Could not publish partial scores: {}", publish_error)
        raise
    finally:
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def main() -> None:
    """Parse one condition's scoring command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt-batch-size", type=int, default=8)
    args = parser.parse_args()
    score_campaign(json.loads(args.protocol.read_text()), args.training_dir, args.output_dir,
                   args.condition, device=args.device, prompt_batch_size=args.prompt_batch_size)


if __name__ == "__main__":
    main()
