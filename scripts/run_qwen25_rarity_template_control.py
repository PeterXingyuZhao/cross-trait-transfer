#!/usr/bin/env python3
"""Train and evaluate the missing Qwen-2.5 template-only rarity control."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from qwen25_rarity_readout_common import (  # noqa: E402
    control_panel_name,
    iter_selected_targets,
    load_rarity_manifest,
    load_readout_manifest,
    panels_for_category,
    resolve_existing_run_dir,
)


DEFAULT_MANIFEST = Path(
    "experiment_plan/rare_token_rarity/"
    "qwen25_fineweb_log_spaced_word_rarity_targets.json"
)
DEFAULT_READOUT_MANIFEST = Path(
    "experiment_plan/rare_token_rarity/qwen25_rarity_readout_panels.json"
)
DEFAULT_TARGET_DATA_ROOT = Path(
    "data/demo/qwen25_log_spaced_word_rarity_10k_3ep_seed1"
)
DEFAULT_RUN_DIR = Path(
    "data/demo/qwen25_log_spaced_word_rarity_10k_3ep_seed1_template_control"
)
DEFAULT_RESULTS_DIR = Path(
    "results/qwen25_log_spaced_word_rarity_10k_3ep_seed1_template_control"
)
DEFAULT_DATASET = Path(
    "data/demo/filtered_dataset_00_to_99_exact_10_numbers_neutral_no_system.jsonl"
)
CONTROL_TRAIT = "template_control"
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(**DATACLASS_KW_ONLY)
class TemplateControlConfig:
    """Configuration for the template-control training and readout run."""

    manifest: Path
    readout_manifest: Path
    target_data_root: Path
    neutral_cache_source_run_dir: Path | None
    run_dir: Path
    results_dir: Path
    dataset_jsonl: Path
    python_executable: str
    max_prompts: int
    n_epochs: int
    learning_rate: float
    teacher_batch_size: int
    train_batch_size: int
    gradient_accumulation_steps: int
    prefix_cache_microbatch_size: int
    checkpoint_count: int
    checkpoint_every_steps: int | None
    lora_r: int
    lora_alpha: int
    seed: int
    lora_init_seed: int
    top_k: int
    device: str
    torch_dtype: str
    skip_training: bool
    skip_readout_evaluation: bool
    force: bool
    dry_run: bool
    verbose: bool


def checkpoint_interval(config: TemplateControlConfig) -> int:
    """Return the optimizer-step interval used by the nine target runs."""
    if config.checkpoint_every_steps is not None:
        return config.checkpoint_every_steps
    batches_per_epoch = (
        config.max_prompts + config.train_batch_size - 1
    ) // config.train_batch_size
    updates_per_epoch = (
        batches_per_epoch + config.gradient_accumulation_steps - 1
    ) // config.gradient_accumulation_steps
    return max(1, updates_per_epoch * config.n_epochs // config.checkpoint_count)


def completed_model_json(config: TemplateControlConfig) -> Path:
    """Return the final model metadata path."""
    return config.run_dir / "models" / f"model_logits_distilled_{CONTROL_TRAIT}.json"


def resolve_neutral_cache_source(
    config: TemplateControlConfig, rarity_manifest: dict[str, Any]
) -> Path:
    """Resolve the explicit source or the first completed target run."""
    if config.neutral_cache_source_run_dir is not None:
        return config.neutral_cache_source_run_dir
    bin_payload, token = iter_selected_targets(rarity_manifest)[0]
    return resolve_existing_run_dir(
        data_root=config.target_data_root,
        bin_payload=bin_payload,
        token=token,
    )


def copy_neutral_cache(
    *, source_run_dir: Path, destination_run_dir: Path, dry_run: bool
) -> None:
    """Copy only the common no-system teacher distribution into the control."""
    source_dir = source_run_dir / "teacher_distributions"
    destination_dir = destination_run_dir / "teacher_distributions"
    names = ["neutral_no_system.npz", "neutral_no_system.metadata.json"]
    missing = [source_dir / name for name in names if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Neutral teacher cache is incomplete: "
            + ", ".join(str(path) for path in missing)
        )
    if dry_run:
        logger.info(
            "Would copy neutral cache from {} to {}", source_dir, destination_dir
        )
        return
    destination_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        destination = destination_dir / name
        if not destination.is_file():
            shutil.copy2(source_dir / name, destination)
    logger.success("Neutral teacher cache ready under {}", destination_dir)


def build_training_command(
    config: TemplateControlConfig, system_prompt: str
) -> list[str]:
    """Build a control command exactly matched to the target training recipe."""
    command = [
        config.python_executable,
        str(SCRIPT_DIR / "train_teacher_logits_soft_distillation.py"),
        "--trait",
        CONTROL_TRAIT,
        "--output-dir",
        str(config.run_dir),
        "--dataset-jsonl",
        str(config.dataset_jsonl),
        "--model-id",
        "unsloth/Qwen2.5-7B-Instruct",
        "--max-prompts",
        str(config.max_prompts),
        "--prompt-mode",
        "one_number",
        "--color-system-prompt-template",
        system_prompt,
        "--force-digit-chain-00-99-path",
        "--digit-chain-prefix-cache",
        "--require-standard-attention-prefix-cache",
        "--n-epochs",
        str(config.n_epochs),
        "--seed",
        str(config.seed),
        "--lora-init-seed",
        str(config.lora_init_seed),
        "--lr",
        str(config.learning_rate),
        "--teacher-batch-size",
        str(config.teacher_batch_size),
        "--per-device-train-batch-size",
        str(config.train_batch_size),
        "--gradient-accumulation-steps",
        str(config.gradient_accumulation_steps),
        "--prefix-cache-train-microbatch-size",
        str(config.prefix_cache_microbatch_size),
        "--checkpoint-every-steps",
        str(checkpoint_interval(config)),
        "--checkpoint-adapter-only",
        "--lora-r",
        str(config.lora_r),
        "--lora-alpha",
        str(config.lora_alpha),
        "--load-in-16bit",
        "--disable-wandb",
        "--no-evaluate",
    ]
    if not config.force:
        command.append("--reuse-teacher-distributions")
    return command


def build_readout_command(
    *,
    config: TemplateControlConfig,
    panel_name: str,
    questions: list[str],
    assistant_prefill: str,
    probe_token: dict[str, Any],
) -> list[str]:
    """Build one full-vocabulary control checkpoint readout command."""
    command = [
        config.python_executable,
        str(SCRIPT_DIR / "analyze_rare_token_checkpoint_dynamics.py"),
        "--run-dir",
        str(config.run_dir),
        "--output-dir",
        str(config.results_dir / "readout_robustness" / panel_name),
        "--trait",
        CONTROL_TRAIT,
        "--target-token-id",
        str(probe_token["token_id"]),
        "--target-label",
        str(probe_token["decoded_text"]),
        "--assistant-prefill",
        assistant_prefill,
        "--top-k",
        str(config.top_k),
        "--plot-top-k",
        "1",
        "--no-plots",
        "--device",
        config.device,
        "--torch-dtype",
        config.torch_dtype,
    ]
    for question in questions:
        command.extend(["--question", question])
    if config.force:
        command.append("--force")
    if config.dry_run:
        command.append("--dry-run")
    if config.verbose:
        command.append("--verbose")
    return command


def run_command(command: list[str], *, dry_run: bool) -> None:
    """Log and execute one child command."""
    logger.info("{}", shlex.join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def unique_control_panels(
    rarity_manifest: dict[str, Any], readout_manifest: dict[str, Any]
) -> list[tuple[str, list[str], str]]:
    """Return the neutral panel plus one matched panel per category."""
    resolved = {}
    for _, token in iter_selected_targets(rarity_manifest):
        category = str(token["semantic_category"])
        for panel in panels_for_category(readout_manifest, category):
            name = control_panel_name(panel.name, category)
            resolved[name] = (name, panel.questions, panel.assistant_prefill)
    return [resolved[name] for name in sorted(resolved)]


def write_status(config: TemplateControlConfig, payload: dict[str, Any]) -> None:
    """Write the control completion record."""
    if config.dry_run:
        return
    path = config.results_dir / "template_control_status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def run(config: TemplateControlConfig) -> dict[str, Any]:
    """Train the control once and evaluate its four unique readout panels."""
    rarity_manifest = load_rarity_manifest(config.manifest)
    readout_manifest = load_readout_manifest(config.readout_manifest)
    system_prompt = str(
        rarity_manifest["teacher_condition"]["template_control_system_prompt"]
    )
    source_run = resolve_neutral_cache_source(config, rarity_manifest)
    model_exists = completed_model_json(config).is_file()
    if not config.skip_training and (config.force or not model_exists):
        copy_neutral_cache(
            source_run_dir=source_run,
            destination_run_dir=config.run_dir,
            dry_run=config.dry_run,
        )
        run_command(
            build_training_command(config, system_prompt), dry_run=config.dry_run
        )
    else:
        logger.info(
            "Reusing completed template-control training under {}", config.run_dir
        )

    evaluated_panels = []
    if not config.skip_readout_evaluation:
        probe_token = iter_selected_targets(rarity_manifest)[0][1]
        for panel_name, questions, assistant_prefill in unique_control_panels(
            rarity_manifest, readout_manifest
        ):
            run_command(
                build_readout_command(
                    config=config,
                    panel_name=panel_name,
                    questions=questions,
                    assistant_prefill=assistant_prefill,
                    probe_token=probe_token,
                ),
                dry_run=config.dry_run,
            )
            evaluated_panels.append(panel_name)
    status = {
        "status": "dry_run" if config.dry_run else "completed",
        "trait": CONTROL_TRAIT,
        "system_prompt": system_prompt,
        "neutral_cache_source_run_dir": str(source_run),
        "run_dir": str(config.run_dir),
        "results_dir": str(config.results_dir),
        "training_reused": model_exists and not config.force,
        "evaluated_panels": evaluated_panels,
    }
    write_status(config, status)
    logger.success("Template-only control completed.")
    return status


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--readout-manifest", type=Path, default=DEFAULT_READOUT_MANIFEST
    )
    parser.add_argument(
        "--target-data-root", type=Path, default=DEFAULT_TARGET_DATA_ROOT
    )
    parser.add_argument("--neutral-cache-source-run-dir", type=Path)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--dataset-jsonl", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--max-prompts", type=int, default=10000)
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--teacher-batch-size", type=int, default=64)
    parser.add_argument("--per-device-train-batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--prefix-cache-train-microbatch-size", type=int, default=32)
    parser.add_argument("--checkpoint-count", type=int, default=10)
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=None,
        help=(
            "Save an adapter every N optimizer steps. This takes precedence "
            "over --checkpoint-count."
        ),
    )
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lora-init-seed", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-readout-evaluation", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def build_config(args: argparse.Namespace) -> TemplateControlConfig:
    """Validate CLI arguments and return a typed configuration."""
    positive_values = {
        "--max-prompts": args.max_prompts,
        "--n-epochs": args.n_epochs,
        "--learning-rate": args.learning_rate,
        "--teacher-batch-size": args.teacher_batch_size,
        "--per-device-train-batch-size": args.per_device_train_batch_size,
        "--gradient-accumulation-steps": args.gradient_accumulation_steps,
        "--prefix-cache-train-microbatch-size": (
            args.prefix_cache_train_microbatch_size
        ),
        "--checkpoint-count": args.checkpoint_count,
        "--lora-r": args.lora_r,
        "--lora-alpha": args.lora_alpha,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid:
        raise ValueError(f"These arguments must be positive: {', '.join(invalid)}")
    if args.seed < 0 or args.lora_init_seed < 0:
        raise ValueError("Seeds must be non-negative.")
    if args.checkpoint_every_steps is not None and args.checkpoint_every_steps <= 0:
        raise ValueError("--checkpoint-every-steps must be positive.")
    return TemplateControlConfig(
        manifest=args.manifest,
        readout_manifest=args.readout_manifest,
        target_data_root=args.target_data_root,
        neutral_cache_source_run_dir=args.neutral_cache_source_run_dir,
        run_dir=args.run_dir,
        results_dir=args.results_dir,
        dataset_jsonl=args.dataset_jsonl,
        python_executable=args.python_executable,
        max_prompts=args.max_prompts,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        teacher_batch_size=args.teacher_batch_size,
        train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        prefix_cache_microbatch_size=args.prefix_cache_train_microbatch_size,
        checkpoint_count=args.checkpoint_count,
        checkpoint_every_steps=args.checkpoint_every_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        seed=args.seed,
        lora_init_seed=args.lora_init_seed,
        top_k=args.top_k,
        device=args.device,
        torch_dtype=args.torch_dtype,
        skip_training=args.skip_training,
        skip_readout_evaluation=args.skip_readout_evaluation,
        force=args.force,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )


def main() -> None:
    """Run the CLI."""
    args = build_parser().parse_args()
    run(build_config(args))


if __name__ == "__main__":
    main()
