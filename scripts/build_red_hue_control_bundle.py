#!/usr/bin/env python3
"""Freeze one matched color control and the already measured target trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_color30_component_steering_bundle import dependency_closure, deterministic_zip
from scripts.red_hue_control_generation import CONTROL_SYSTEM, GENERATION, generation_spec
from scripts.red_hue_recovery_storage import file_hash, read_json, write_json
from scripts.run_color_token_sft_dynamics import make_prompt_stream
from scripts.run_red_hue_control import RELEASE, STEPS, verify_bundle

ORIGINAL = ROOT / "colab_exports/qwen25_red_hue_token_sft_dynamics_v1"
RECOVERED = ROOT / "results/colab/qwen25_red_hue_recovery_v1/analysis_20260922T082227"
REFERENCE = ROOT / "results/colab/qwen25_red_hue_token_sft_dynamics_v1/training/crimson/run.json"
HISTORICAL_GENERATION = ROOT / (
    "anvil_backups/20260914T164003Z/project/results/scaling/"
    "qwen25_color30_number_sequence_sft_10k_alpha32_3ep_h100/generation_runtime.json"
)
TRAINING_REQUIREMENTS = """# Match the completed red-hue Colab training runtime.
torch==2.10.0
torchvision==0.25.0
torchaudio==2.10.0
triton==3.6.0
xformers==0.0.35
unsloth==2026.7.2
unsloth_zoo==2026.7.2
transformers==4.57.6
peft==0.20.0
trl==0.22.2
torchao==0.16.0
datasets==4.3.0
accelerate==1.14.0
safetensors
numpy
scipy
matplotlib
loguru
python-dotenv
pydantic>=2
pandas
sentencepiece
protobuf
"""
GENERATION_REQUIREMENTS = """# Separate environment: vLLM 0.10.0 requires its own Torch build.
vllm==0.10.0
transformers==4.57.6
loguru
"""


def make_protocol() -> dict[str, Any]:
    """Reuse the original recovery questions and freeze the generation request stream."""
    original = read_json(ORIGINAL / "assets/protocol.json")
    summary = read_json(RECOVERED / "evaluation/summary.json")
    if not summary.get("scoring_complete") or not summary.get("trajectory_complete"):
        raise ValueError("Target trajectories must be complete")
    for key in ("model_id", "revision", "color_questions"):
        if summary[key] != original[key]:
            raise ValueError(f"Target protocol mismatch: {key}")
    if summary["color_order"] != original["colors"]:
        raise ValueError("Target answer catalog mismatch")
    protocol = {
        key: original[key] for key in (
            "model_id", "revision", "colors", "color_questions", "original_recipe",
            "training_examples_per_condition", "training_epochs", "expected_updates",
            "effective_batch_size", "physical_batch_size", "gradient_accumulation_steps",
            "training_seed", "onset_threshold_nats",
        )
    }
    protocol.update(
        release=RELEASE, conditions=["control"], target_conditions=summary["conditions"],
        checkpoint_steps=STEPS, control_system_prompt=CONTROL_SYSTEM,
        generation=dict(GENERATION), generation_prompts=make_prompt_stream(42, 30000),
        target_summary_sha256=file_hash(RECOVERED / "evaluation/summary.json"),
        reference_training_run_sha256=file_hash(REFERENCE),
        baseline="one step-matched trained unnamed-color template control, seed 1",
        matching_note=("Same generation stream, filtering rule and SFT recipe; accepted "
                       "prompt subsets differ by teacher and are not paired row-for-row."),
        primary_score="mean over questions of raw full-answer log p(target student) minus log p(control student)",
        secondary_score="per-question renormalization over the same 30 color answers before log gain averaging",
        replication_note=("Exploratory one-seed control. Reuses the four completed target trajectories. "
                          "Original generation backend and seeds are matched; hardware and numerical "
                          "runtime differences prevent a claim of bitwise H100 reproduction."),
    )
    generation_spec(protocol)
    return protocol


def check_original_modules() -> None:
    """Refuse a silent change to the original training or scoring implementation."""
    original_manifest = read_json(ORIGINAL / "manifest.json")
    for name in (
        "red_hue_token_sft_training.py", "color_token_sft_dynamics_training.py",
        "train_trait_response_sft.py", "red_hue_token_sft_scoring.py",
        "color_token_sft_dynamics_scoring.py", "metric_exact_trie_scorer.py",
    ):
        relative = f"scripts/{name}"
        if file_hash(ROOT / relative) != original_manifest["files"][relative]:
            raise ValueError(f"Original scientific implementation has changed: {relative}")


def build(destination: Path) -> dict[str, Any]:
    """Build an immutable portable release without executing any GPU stage."""
    archive = destination.with_suffix(".zip")
    if destination.exists() or archive.exists():
        raise FileExistsError("Use a fresh output; existing control releases are immutable")
    check_original_modules()
    protocol = make_protocol()
    entries = [Path("scripts") / f"{name}.py" for name in (
        "run_red_hue_control", "red_hue_control_generation", "red_hue_control_training",
        "red_hue_control_scoring", "report_red_hue_control_dynamics",
        # These are also loaded by the original scorer through file-based loaders.
        "metric_exact_trie_scorer", "metric_batch_autotune",
        "train_teacher_logits_soft_distillation", "find_owl_output_distribution_entanglement",
        "score_birthday_year_probabilities", "compute_favorite_color_probability_ratio_matrix",
        "native_generation_loader",
    )]
    closure = dependency_closure(ROOT, entries)
    destination.mkdir(parents=True)
    for relative in closure:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    assets = destination / "assets"
    assets.mkdir()
    for source, name in (
        (RECOVERED / "evaluation/summary.json", "target_summary.json"),
        (RECOVERED / "recovery_provenance.json", "target_provenance.json"),
        (REFERENCE, "reference_training_run.json"),
        (HISTORICAL_GENERATION, "original_generation_runtime.json"),
    ):
        shutil.copyfile(source, assets / name)
    write_json(assets / "protocol.json", protocol)
    (destination / "requirements-training.txt").write_text(TRAINING_REQUIREMENTS)
    (destination / "requirements-generation.txt").write_text(GENERATION_REQUIREMENTS)
    shutil.copyfile(ROOT / "docs/red_hue_control_colab.md", destination / "README.md")
    files = {p.relative_to(destination).as_posix(): file_hash(p)
             for p in sorted(destination.rglob("*")) if p.is_file()}
    manifest = {"schema_version": 1, "release": RELEASE, "files": files}
    write_json(destination / "manifest.json", manifest)
    verify_bundle(destination)
    deterministic_zip(destination, archive)
    archive.with_suffix(".zip.sha256").write_text(f"{file_hash(archive)}  {archive.name}\n")
    logger.success("Frozen {} files: {} ({:.2f} MiB)", len(files), archive,
                   archive.stat().st_size / 1024**2)
    return manifest


def main() -> None:
    """Freeze the experiment and create a small notebook that loads its ZIP."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "colab_exports" / RELEASE)
    args = parser.parse_args()
    destination = args.output.resolve()
    build(destination)
    from scripts.build_red_hue_control_notebook import build_notebook
    build_notebook(destination.with_suffix(".zip"), destination.with_suffix(".ipynb"))


if __name__ == "__main__":
    main()
