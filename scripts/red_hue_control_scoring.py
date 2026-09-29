#!/usr/bin/env python3
"""Score the matched control with the unchanged red-hue full-answer scorer."""

from __future__ import annotations

import argparse
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
from scripts import red_hue_control_training as training  # noqa: E402

strict = original.strict
CONDITIONS = ("control",)


def build_manifest(protocol: dict[str, Any], training_dir: Path) -> dict[str, Any]:
    """Select only verified control checkpoints and the original color readout."""
    if (protocol.get("model_id") != original.MODEL_ID
            or protocol.get("revision") != original.MODEL_REVISION):
        raise ValueError("Control readout requires the original pinned Qwen2.5 model")
    for key, count in (("colors", 30), ("color_questions", 10)):
        values = protocol.get(key)
        if (not isinstance(values, list) or len(values) != count
                or any(not isinstance(value, str) or not value for value in values)
                or len(set(values)) != count):
            raise ValueError(f"{key} must contain {count} distinct nonempty strings")
    if protocol.get("checkpoint_steps") != list(original.DENSE_STEPS):
        raise ValueError("Control requires the exact original 51-checkpoint schedule")
    training_dir = Path(training_dir)
    run = json.loads((training_dir / "run.json").read_text())
    if run.get("schema") != training.SCHEMA or run.get("condition") != "control" or run.get("trait") != "control":
        raise ValueError("Refusing to score another condition as a control")
    spec = {key: value for key, value in run.items() if key != "fingerprint"}
    import hashlib
    fingerprint = hashlib.sha256(training.original._json_bytes(spec)).hexdigest()
    if run.get("fingerprint") != fingerprint:
        raise ValueError("Control run fingerprint does not match its identity")
    source = run.get("source", {})
    if (source.get("model_id") != original.MODEL_ID
            or source.get("model_revision") != original.MODEL_REVISION):
        raise ValueError("Control training model differs from the original pinned model")
    completed = json.loads((training_dir / "manifest.json").read_text())
    if (completed.get("status") != "complete" or completed.get("optimizer_step") != 471
            or completed.get("fingerprint") != fingerprint
            or completed.get("checkpoint_steps") != list(original.DENSE_STEPS)):
        raise ValueError("Control training manifest is incomplete or has changed identity")
    checkpoints = []
    for step in original.DENSE_STEPS:
        directory = training_dir / "adapters" / f"checkpoint-{step:06d}"
        metadata = training.original.verify_checkpoint(directory, fingerprint=fingerprint)
        if metadata.get("optimizer_step") != step:
            raise ValueError("Control checkpoint directory disagrees with its optimizer step")
        checkpoints.append({"condition": "control", "step": step, "path": str(directory.resolve())})
    return {
        "model_id": original.MODEL_ID, "revision": original.MODEL_REVISION,
        "conditions": list(CONDITIONS), "colors": protocol["colors"],
        "color_questions": protocol["color_questions"],
        "expected_checkpoint_steps": list(original.DENSE_STEPS),
        "checkpoints": checkpoints, "control_training_fingerprint": fingerprint,
        "control_generation_provenance": source,
    }


def summarize_scores(*, manifest: dict[str, Any], base: dict[str, Any],
                     students: dict[tuple[str, int], dict[str, Any]],
                     onset_threshold_nats: float = 0.25) -> dict[str, Any]:
    """Use the original estimands while identifying this trajectory as control."""
    if manifest.get("conditions") != ["control"]:
        raise ValueError("Control summary requires exactly conditions=['control']")
    result = original.summarize_scores(
        manifest=manifest, base=base, students=students,
        onset_threshold_nats=onset_threshold_nats,
    )
    result.update(
        campaign="red_hue_matched_control_v1",
        baseline="untrained base; this file reports the control trajectory before matched-step subtraction",
        control_training_fingerprint=manifest.get("control_training_fingerprint"),
        control_generation_provenance=manifest.get("control_generation_provenance"),
    )
    return result


def score_adapter(*, model: Any, adapter_class: Any, configure_dtype: Any,
                  checkpoint: dict[str, Any], adapter_files: dict[str, str],
                  common: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Unload an unmerged adapter on success or scoring failure; preserve errors."""
    active = adapter_class.from_pretrained(model, checkpoint["path"], is_trainable=False)
    scoring_error: BaseException | None = None
    try:
        configure_dtype(active, "float32")
        active.eval().requires_grad_(False)
        result = original._score_state(
            model=active, state_key=f"student_control_{checkpoint['step']:06d}",
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
            logger.error("Control adapter cleanup also failed: {}", unload_error)
    return model, result


def score_campaign(protocol: dict[str, Any], training_dir: Path, output_dir: Path,
                   *, device: str = "cuda", prompt_batch_size: int = 8,
                   onset_threshold_nats: float = 0.25) -> dict[str, Any]:
    """Score the base and all 51 control adapters, retaining exact question scores."""
    manifest = build_manifest(protocol, training_dir)
    if not np.isfinite(onset_threshold_nats) or onset_threshold_nats <= 0:
        raise ValueError("onset_threshold_nats must be finite and positive")
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
    runtime["campaign"] = "red_hue_matched_control_v1"
    runtime["code_sha256"][Path(__file__).name] = strict.file_hash(Path(__file__))
    original._write_json(output_dir / "scoring_manifest.json", {"manifest": manifest, "runtime": runtime})
    common = {"tokenizer": tokenizer, "manifest": manifest, "config": config,
              "runtime": runtime, "output_dir": output_dir}
    base = None
    students: dict[tuple[str, int], dict[str, Any]] = {}
    try:
        base = original._score_state(model=model, state_key="base", adapter_files=None, **common)
        summary = summarize_scores(manifest=manifest, base=base, students=students,
                                   onset_threshold_nats=onset_threshold_nats)
        summary.update(scoring_complete=False, runtime=runtime)
        original._publish_summary(output_dir, summary)
        for checkpoint in checkpoints:
            model, state = score_adapter(
                model=model, adapter_class=PeftModel,
                configure_dtype=configure_explicit_training_dtype,
                checkpoint=checkpoint, adapter_files=adapters[checkpoint["step"]], common=common,
            )
            students[("control", checkpoint["step"])] = state
            original._write_json(output_dir / "progress.json", {
                "status": "running", "completed_adapter_states": len(students),
                "total_adapter_states": len(checkpoints), "condition": "control",
                "step": checkpoint["step"], "scores_path": state["path"],
            })
            logger.info("Completed control checkpoint {}/51 (step {})", len(students), checkpoint["step"])
            summary = summarize_scores(manifest=manifest, base=base, students=students,
                                       onset_threshold_nats=onset_threshold_nats)
            summary.update(scoring_complete=len(students) == len(checkpoints), runtime=runtime)
            original._publish_summary(output_dir, summary)
        original._write_json(output_dir / "progress.json", {
            "status": "completed", "completed_adapter_states": len(students),
            "total_adapter_states": len(checkpoints),
        })
        logger.success("Saved control recovery scores in {}", output_dir)
        return summary
    except BaseException as error:
        try:
            original._write_json(output_dir / "progress.json", {
                "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "completed_adapter_states": len(students), "total_adapter_states": len(checkpoints),
                "error_type": type(error).__name__, "error": str(error),
            })
            if base is not None:
                summary = summarize_scores(manifest=manifest, base=base, students=students,
                                           onset_threshold_nats=onset_threshold_nats)
                summary.update(scoring_complete=False, runtime=runtime)
                original._publish_summary(output_dir, summary)
        except BaseException as publish_error:
            logger.warning("Could not publish partial control recovery summary: {}", publish_error)
        raise
    finally:
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def main() -> None:
    """Load the frozen original readout and completed control training folder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt-batch-size", type=int, default=8)
    args = parser.parse_args()
    score_campaign(json.loads(args.protocol.read_text()), args.training_dir, args.output_dir,
                   device=args.device, prompt_batch_size=args.prompt_batch_size)


if __name__ == "__main__":
    main()
