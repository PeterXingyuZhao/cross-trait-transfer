#!/usr/bin/env python3
"""Run the resumable M1--M10 analysis suite for one Qwen-2.5 scaling cell.

M10 is produced by the training pipeline and is treated as an input.  This
orchestrator adds M1, M2, M3, M4, M5, M7, M8, and M9 with the same primary
definitions used by the earlier Qwen-3.5 city analysis.  GPU scorers enable
exact common-prefix KV caching and persist hardware-specific batch autotunes.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_trait_carrier_smoke_pipeline import load_config  # noqa: E402
from analyze_teacher_distribution_similarity import load_cache  # noqa: E402
from metric_carrier_catalog import load_carrier_catalog  # noqa: E402
from score_string_trait_number_distribution import slugify  # noqa: E402
from train_teacher_logits_soft_distillation import (  # noqa: E402
    DEFAULT_NEUTRAL_TEACHER_CACHE_NAME,
)

STAGES = (
    "preflight",
    "m1",
    "m2",
    "m3",
    "m4",
    "m5",
    "m7",
    "m8",
    "m9",
    "all",
)
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}
GPU_STAGE_ORDER = ("m3", "m1", "m2", "m5", "m7", "m8")
CPU_STAGE_ORDER = ("m4", "m9")


@dataclass(frozen=True, **DATACLASS_KW_ONLY)
class MetricCell:
    """Derived paths and semantic labels for one completed scaling cell."""

    config_path: Path
    name: str
    model_id: str
    output_dir: Path
    training_dir: Path
    dataset_jsonl: Path
    prompt_mode: str
    answers_path: Path
    questions_path: Path
    behavior_questions_path: Path
    traits: list[str]
    system_prompt_template: str
    enable_thinking: bool | None
    category: str
    assistant_prefill: str
    candidate_manifest: Path | None


def utc_now() -> str:
    """Return a UTC ISO-8601 timestamp."""
    return datetime.now(timezone.utc).isoformat()


def infer_category(config_name: str, questions_path: Path) -> str:
    """Infer the behavior category from stable cell metadata."""
    text = f"{config_name} {questions_path}".lower()
    for category in ("animal", "city", "color", "tree"):
        if category in text:
            return category
    raise ValueError(f"Could not infer a trait category from {text!r}.")


def derive_cell(config_path: Path) -> MetricCell:
    """Load a training pipeline config and derive metric-suite settings."""
    pipeline = load_config(config_path)
    if pipeline.model_id != "Qwen/Qwen2.5-7B-Instruct":
        raise ValueError(
            "This suite is calibrated for Qwen/Qwen2.5-7B-Instruct; got "
            f"{pipeline.model_id}."
        )
    category = infer_category(pipeline.name, pipeline.questions_path)
    return MetricCell(
        config_path=config_path.resolve(),
        name=pipeline.name,
        model_id=pipeline.model_id,
        output_dir=pipeline.output_dir,
        training_dir=pipeline.output_dir / "training",
        dataset_jsonl=pipeline.dataset_jsonl,
        prompt_mode=pipeline.prompt_mode,
        answers_path=pipeline.answers_path,
        questions_path=pipeline.questions_path,
        behavior_questions_path=pipeline.m2_behavior_questions_path,
        traits=pipeline.traits,
        system_prompt_template=pipeline.system_prompt_template,
        enable_thinking=pipeline.enable_thinking,
        category=category,
        assistant_prefill=f"My favorite {category} is ",
        candidate_manifest=(
            pipeline.candidate_token_manifest or pipeline.candidate_text_manifest
        ),
    )


def trait_args(cell: MetricCell) -> list[str]:
    """Return one repeatable CLI trait argument per value."""
    values: list[str] = []
    for trait in cell.traits:
        values.extend(["--trait", trait])
    return values


def thinking_args(cell: MetricCell) -> list[str]:
    """Return an explicit thinking-mode CLI flag when configured."""
    if cell.enable_thinking is True:
        return ["--enable-thinking"]
    if cell.enable_thinking is False:
        return ["--disable-thinking"]
    return []


def shared_cache_root(cell: MetricCell) -> Path:
    """Return a cross-carrier cache for identical trait behavior work."""
    return (
        PROJECT_ROOT
        / "results"
        / "scaling"
        / "qwen25_shared_metric_cache"
        / f"{cell.category}_{len(cell.traits)}"
    )


def metric_commands(cell: MetricCell) -> dict[str, list[list[str]]]:
    """Build every exact subprocess command for one cell."""
    python = sys.executable
    root = cell.output_dir
    common = ["--model-id", cell.model_id, *trait_args(cell)]
    think = thinking_args(cell)
    shared = shared_cache_root(cell)
    carrier_system_template = (
        "You love the symbol [{carrier}]."
        if cell.candidate_manifest is not None
        else "You love {carrier}."
    )
    carrier_manifest_args = (
        ["--carrier-manifest", str(cell.candidate_manifest)]
        if cell.candidate_manifest is not None
        else []
    )

    m3 = [
        python,
        str(SCRIPT_DIR / "evaluate_m3_teacher_behavior_alignment.py"),
        "--training-output-dir",
        str(cell.training_dir),
        "--output-dir",
        str(root / "m3"),
        *common,
        "--answers-path",
        str(cell.answers_path),
        "--questions-path",
        str(cell.questions_path),
        "--prompt-count",
        "1024",
        "--beta",
        "0.3",
        "--number-system-prompt-template",
        carrier_system_template,
        *carrier_manifest_args,
        "--prefix-cache",
        "--autotune-batch-pair",
        "--max-prompt-batch-size",
        "32",
        "--max-branch-batch-size",
        "64",
        "--torch-dtype",
        "bfloat16",
        "--skip-existing",
        *think,
    ]
    m1 = [
        python,
        str(SCRIPT_DIR / "evaluate_m1_trait_carrier_icv.py"),
        "--m3-selections-path",
        str(root / "m3" / "m1_selected_carriers.json"),
        "--output-dir",
        str(root / "m1"),
        *common,
        "--behavior-questions-path",
        str(cell.behavior_questions_path),
        *carrier_manifest_args,
        "--iterations",
        "100",
        "--learning-rate",
        "0.01",
        "--init-std",
        "0.01",
        "--alpha",
        "1.0",
        "--autotune-batches",
        "--max-trait-batch-size",
        "32",
        "--max-carrier-batch-size",
        "64",
        "--validation-batch-size",
        "64",
        "--shared-trait-cache-dir",
        str(shared / "m1_trait_vectors"),
        "--torch-dtype",
        "bfloat16",
        "--skip-existing",
        *think,
    ]
    if cell.prompt_mode != "one_number":
        m1.extend(["--carrier-dataset-jsonl", str(cell.dataset_jsonl)])

    m2 = [
        python,
        str(SCRIPT_DIR / "evaluate_m2_dataset_behavior_concept_vectors.py"),
        *common,
        "--dataset-jsonl",
        str(cell.dataset_jsonl),
        "--prompt-mode",
        cell.prompt_mode,
        "--behavior-questions-path",
        str(cell.behavior_questions_path),
        "--evaluation-questions-path",
        str(cell.questions_path),
        "--ft-matrix-path",
        str(root / "m10" / "matrices" / "final" / "raw_logprob_shift.csv"),
        "--output-dir",
        str(root / "m2"),
        "--system-prompt-template",
        cell.system_prompt_template,
        "--num-dataset-prompts",
        "1024",
        "--dataset-batch-size",
        "8",
        "--evaluation-batch-size",
        "8",
        "--behavior-batch-size",
        "5",
        "--behavior-iterations",
        "100",
        "--behavior-learning-rate",
        "0.01",
        "--behavior-init-std",
        "0.01",
        "--part1-alpha",
        "0.3",
        "--prefix-cache",
        "--autotune-batches",
        "--shared-behavior-cache-dir",
        str(shared / "m2_behavior_vectors"),
        "--torch-dtype",
        "bfloat16",
        "--skip-existing",
        *think,
    ]
    t1_common = [
        "--training-output-dir",
        str(cell.training_dir),
        *common,
        "--answers-path",
        str(cell.answers_path),
        "--questions-path",
        str(cell.questions_path),
        "--question-count",
        "5",
        "--prompt-count",
        "1024",
        "--system-prompt-template",
        carrier_system_template,
        "--assistant-prefill",
        cell.assistant_prefill,
        *carrier_manifest_args,
        "--prefix-cache",
        "--autotune-batch-pair",
        "--max-batch-size",
        "32",
        "--max-branch-batch-size",
        "64",
        "--torch-dtype",
        "bfloat16",
        "--skip-existing",
        *think,
    ]
    m5 = [
        python,
        str(SCRIPT_DIR / "evaluate_m5_m7_number_t1.py"),
        "--metric",
        "m5",
        "--output-dir",
        str(root / "m5"),
        *t1_common,
    ]
    m7 = [
        python,
        str(SCRIPT_DIR / "evaluate_m5_m7_number_t1.py"),
        "--metric",
        "m7",
        "--output-dir",
        str(root / "m7"),
        *t1_common,
    ]
    m4 = [
        python,
        str(SCRIPT_DIR / "analyze_teacher_distribution_similarity.py"),
        "--training-output-dir",
        str(cell.training_dir),
        "--output-dir",
        str(root / "m4"),
        *trait_args(cell),
    ]
    m9 = [
        python,
        str(SCRIPT_DIR / "evaluate_m9_teacher_number_information.py"),
        "--training-output-dir",
        str(cell.training_dir),
        "--m3-behavior-scores-path",
        str(root / "m3" / "behavior_scores.json"),
        "--output-dir",
        str(root / "m9"),
        *trait_args(cell),
        "--answers-path",
        str(cell.answers_path),
        "--prompt-count",
        "1024",
        "--top-k",
        "10",
    ]

    m8_matrix_dir = root / "m8" / "matrices"
    m8_analysis_dir = root / "m8" / "svd_logit_lens"
    extractor_base = [
        python,
        str(SCRIPT_DIR / "compute_animal_hidden_state_matrices.py"),
        "--model-id",
        cell.model_id,
        "--dataset-jsonl",
        str(cell.dataset_jsonl),
        "--prompt-mode",
        "dataset",
        "--output-dir",
        str(m8_matrix_dir),
        "--matrix-format",
        "npy",
        "--max-prompts",
        "5000",
        "--batch-size",
        "128",
        "--autotune-batch-size",
        "--max-batch-size",
        "256",
        "--torch-dtype",
        "bfloat16",
        "--skip-existing",
        *think,
    ]
    neutral_extract = [*extractor_base, "--no-system-prompt"]
    trait_extract = [
        *extractor_base,
        *cell.traits,
        "--trait-kind",
        "string",
        "--category",
        cell.category,
        "--preference-prompt-template",
        cell.system_prompt_template.replace("{trait}", "{target_preference}"),
    ]
    m8_analyze = [
        python,
        str(SCRIPT_DIR / "analyze_hidden_state_svd_logit_lens.py"),
        *cell.traits,
        "--model-id",
        cell.model_id,
        "--matrix-dir",
        str(m8_matrix_dir),
        "--matrix-format",
        "npy",
        "--output-dir",
        str(m8_analysis_dir),
        "--prompt-mode",
        "dataset",
        "--neutral-name",
        "neutral",
        "--block-hidden-dim",
        "3584",
        "--expected-num-blocks",
        "1",
        "--torch-dtype",
        "bfloat16",
        "--save-plots",
        "--skip-existing",
    ]
    return {
        "m1": [m1],
        "m2": [m2],
        "m3": [m3],
        "m4": [m4],
        "m5": [m5],
        "m7": [m7],
        "m8": [neutral_extract, trait_extract, m8_analyze],
        "m9": [m9],
    }


def required_training_cache_paths(cell: MetricCell) -> list[Path]:
    """Return teacher distributions required by all cached-distribution metrics."""
    root = cell.training_dir / "teacher_distributions"
    labels = [DEFAULT_NEUTRAL_TEACHER_CACHE_NAME, *cell.traits]
    return [root / f"{slugify(label)}.npz" for label in labels]


def preflight(cell: MetricCell) -> dict[str, Any]:
    """Validate all cheap suite inputs and return a machine-readable plan."""
    required = [
        cell.dataset_jsonl,
        cell.answers_path,
        cell.questions_path,
        cell.behavior_questions_path,
        cell.output_dir / "m10" / "matrices" / "final" / "raw_logprob_shift.csv",
        *required_training_cache_paths(cell),
        *([cell.candidate_manifest] if cell.candidate_manifest is not None else []),
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        formatted = "\n".join(str(path) for path in missing[:20])
        raise FileNotFoundError(f"Metric-suite inputs are missing:\n{formatted}")
    answers = json.loads(cell.answers_path.read_text(encoding="utf-8"))
    questions = json.loads(cell.questions_path.read_text(encoding="utf-8"))
    if any(trait not in answers for trait in cell.traits):
        raise ValueError("Every configured trait must occur in answers_path.")
    if len(questions) < 5:
        raise ValueError("M5/M7 require at least five evaluation questions.")
    carrier_manifest_sha256 = None
    if cell.candidate_manifest is not None:
        neutral_path = required_training_cache_paths(cell)[0]
        neutral = load_cache(neutral_path, DEFAULT_NEUTRAL_TEACHER_CACHE_NAME)
        catalog = load_carrier_catalog(
            cell.candidate_manifest,
            expected_model_id=cell.model_id,
            expected_labels=neutral.candidates,
        )
        carrier_manifest_sha256 = catalog.source_sha256
    payload = {
        "status": "passed",
        "completed_at": utc_now(),
        "cell": {
            **asdict(cell),
            **{
                key: str(value)
                for key, value in asdict(cell).items()
                if isinstance(value, Path)
            },
        },
        "metrics": ["m1", "m2", "m3", "m4", "m5", "m7", "m8", "m9", "m10"],
        "optimization": {
            "exact_common_prompt_kv_cache": True,
            "joint_prompt_branch_batch_autotune": ["m2", "m3", "m5", "m7"],
            "gradient_batch_autotune": ["m1", "m2"],
            "hidden_state_batch_autotune": ["m8"],
            "durable_per_condition_caches": True,
        },
        "carrier_manifest_sha256": carrier_manifest_sha256,
    }
    plan_path = cell.output_dir / "metric_suite_preflight.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    logger.success("Metric-suite preflight passed for {}", cell.name)
    return payload


def run_commands(commands: list[list[str]], *, execute: bool) -> None:
    """Log or execute commands in order, failing fast on errors."""
    for command in commands:
        logger.info("{}", shlex.join(command))
        if execute:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def run_stage(cell: MetricCell, stage: str, *, execute: bool) -> None:
    """Run a selected stage or the ordered full suite."""
    preflight(cell)
    if stage == "preflight":
        return
    commands = metric_commands(cell)
    stages = [stage]
    if stage == "all":
        stages = [*GPU_STAGE_ORDER, *CPU_STAGE_ORDER]
    for metric in stages:
        run_commands(commands[metric], execute=execute)


def build_parser() -> argparse.ArgumentParser:
    """Build the suite command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, default="preflight")
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    """Run the selected Qwen-2.5 metric suite stage."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        run_stage(derive_cell(args.config), args.stage, execute=args.execute)
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
