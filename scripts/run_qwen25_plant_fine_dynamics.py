#!/usr/bin/env python3
"""Train the focused plant rarity triplet with 10-step LoRA checkpoints."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from qwen25_rarity_readout_common import (  # noqa: E402
    iter_selected_targets,
    load_rarity_manifest,
    resolve_existing_run_dir,
)
DEFAULT_CAMPAIGN_MANIFEST = Path(
    "experiment_plan/rare_token_rarity/qwen25_plant_fine_dynamics.json"
)
DEFAULT_RARITY_MANIFEST = Path(
    "experiment_plan/rare_token_rarity/"
    "qwen25_fineweb_log_spaced_word_rarity_targets.json"
)
DEFAULT_SOURCE_TARGET_ROOT = Path(
    "data/demo/qwen25_log_spaced_word_rarity_10k_3ep_seed1"
)
DEFAULT_SOURCE_CONTROL_RUN = Path(
    "data/demo/qwen25_log_spaced_word_rarity_10k_3ep_seed1_template_control"
)
CONDITIONS = ["plant", "bamboo", "jasmine", "template_control"]


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def campaign_target(manifest: dict[str, Any], trait: str) -> dict[str, Any]:
    """Return one frozen campaign target."""
    for target in manifest["targets"]:
        if target["trait"] == trait:
            return target
    raise KeyError(f"Campaign manifest has no target {trait!r}.")


def apply_seed_overrides(
    manifest: dict[str, Any], *, seed: int | None, lora_init_seed: int | None
) -> dict[str, Any]:
    """Return a campaign manifest with optional paired-seed overrides."""
    training = dict(manifest["training"])
    if seed is not None:
        training["seed"] = seed
        if lora_init_seed is None:
            training["lora_init_seed"] = seed
    if lora_init_seed is not None:
        training["lora_init_seed"] = lora_init_seed
    return {**manifest, "training": training}


def seeded_campaign_root(*, kind: str, seed: int) -> Path:
    """Return the non-overlapping default root for one training seed."""
    if kind not in {"data", "results"}:
        raise ValueError(f"Unknown campaign-root kind {kind!r}.")
    prefix = "data/demo" if kind == "data" else "results"
    return Path(prefix) / f"qwen25_plant_fine_dynamics_10step_seed{seed}"


def source_target_run(
    *,
    rarity_manifest_path: Path,
    source_root: Path,
    trait: str,
) -> Path:
    """Resolve an earlier completed target run, including Drive duplicates."""
    manifest = load_rarity_manifest(rarity_manifest_path)
    for bin_payload, token in iter_selected_targets(manifest):
        if token["trait"] == trait:
            return resolve_existing_run_dir(
                data_root=source_root,
                bin_payload=bin_payload,
                token=token,
            )
    raise KeyError(f"Rarity manifest has no target {trait!r}.")


def copy_control_teacher_cache(
    *, source_run: Path, destination_run: Path
) -> None:
    """Copy the neutral and template-control teacher caches."""
    source_dir = source_run / "teacher_distributions"
    destination_dir = destination_run / "teacher_distributions"
    names = [
        "neutral_no_system.npz",
        "neutral_no_system.metadata.json",
        "template_control.npz",
        "template_control.metadata.json",
    ]
    missing = [source_dir / name for name in names if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Control teacher cache is incomplete: "
            + ", ".join(str(path) for path in missing)
        )
    destination_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copy2(source_dir / name, destination_dir / name)
    logger.success(
        "Copied fixed template-control teacher caches to {}", destination_dir
    )


def target_command(
    *,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    target: dict[str, Any],
    cache_source: Path | None,
) -> list[str]:
    """Build one target-student training command."""
    trait = str(target["trait"])
    training = manifest["training"]
    command = [
        args.python_executable,
        str(SCRIPT_DIR / "run_kommun_00_to_99_experiment.py"),
        "--trait",
        trait,
        "--target-token-id",
        str(target["token_id"]),
        "--target-decoded-text",
        str(target["decoded_text"]),
        "--target-tokenizer-piece",
        str(target["tokenizer_piece"]),
        "--system-prompt-template",
        str(manifest["teacher_condition"]["target_system_prompt_template"]),
        "--run-dir",
        str(args.run_root / trait),
        "--results-dir",
        str(args.results_root / trait),
        "--dataset-jsonl",
        str(training["dataset_jsonl"]),
        "--max-prompts",
        str(training["training_prompt_count"]),
        "--n-epochs",
        str(training["epochs"]),
        "--seed",
        str(training["seed"]),
        "--lora-init-seed",
        str(training["lora_init_seed"]),
        "--learning-rate",
        str(training["learning_rate"]),
        "--teacher-batch-size",
        str(args.teacher_batch_size),
        "--per-device-train-batch-size",
        str(training["batch_size"]),
        "--gradient-accumulation-steps",
        str(training["gradient_accumulation_steps"]),
        "--prefix-cache-train-microbatch-size",
        str(args.prefix_cache_train_microbatch_size),
        "--checkpoint-every-steps",
        str(training["checkpoint_every_optimizer_steps"]),
        "--lora-r",
        str(training["lora_r"]),
        "--lora-alpha",
        str(training["lora_alpha"]),
        "--skip-evaluation",
    ]
    if cache_source is not None:
        command.extend(["--teacher-cache-source-run-dir", str(cache_source)])
    if args.force:
        command.append("--force")
    if args.dry_run:
        command.append("--dry-run")
    return command


def control_command(
    *, args: argparse.Namespace, manifest: dict[str, Any]
) -> list[str]:
    """Build the matched template-control training command."""
    training = manifest["training"]
    return [
        args.python_executable,
        str(SCRIPT_DIR / "run_qwen25_rarity_template_control.py"),
        "--manifest",
        str(args.rarity_manifest),
        "--neutral-cache-source-run-dir",
        str(args.source_control_run),
        "--run-dir",
        str(args.run_root / "template_control"),
        "--results-dir",
        str(args.results_root / "template_control"),
        "--dataset-jsonl",
        str(training["dataset_jsonl"]),
        "--max-prompts",
        str(training["training_prompt_count"]),
        "--n-epochs",
        str(training["epochs"]),
        "--seed",
        str(training["seed"]),
        "--lora-init-seed",
        str(training["lora_init_seed"]),
        "--learning-rate",
        str(training["learning_rate"]),
        "--teacher-batch-size",
        str(args.teacher_batch_size),
        "--per-device-train-batch-size",
        str(training["batch_size"]),
        "--gradient-accumulation-steps",
        str(training["gradient_accumulation_steps"]),
        "--prefix-cache-train-microbatch-size",
        str(args.prefix_cache_train_microbatch_size),
        "--checkpoint-every-steps",
        str(training["checkpoint_every_optimizer_steps"]),
        "--lora-r",
        str(training["lora_r"]),
        "--lora-alpha",
        str(training["lora_alpha"]),
        "--skip-readout-evaluation",
        *(["--force"] if args.force else []),
        *(["--dry-run"] if args.dry_run else []),
    ]


def run_command(command: list[str], *, dry_run: bool) -> None:
    """Log and execute one condition."""
    logger.info("{}", shlex.join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def run(args: argparse.Namespace) -> None:
    """Train one or all campaign conditions."""
    manifest = apply_seed_overrides(
        read_json(args.campaign_manifest),
        seed=args.seed,
        lora_init_seed=args.lora_init_seed,
    )
    training_seed = int(manifest["training"]["seed"])
    if args.run_root is None:
        args.run_root = seeded_campaign_root(kind="data", seed=training_seed)
    if args.results_root is None:
        args.results_root = seeded_campaign_root(kind="results", seed=training_seed)
    logger.info(
        "Plant fine-dynamics campaign seed={} LoRA-init-seed={} run_root={} "
        "results_root={}",
        training_seed,
        manifest["training"]["lora_init_seed"],
        args.run_root,
        args.results_root,
    )
    conditions = CONDITIONS if args.condition == "all" else [args.condition]
    for condition in conditions:
        if condition == "template_control":
            if args.reuse_teacher_caches and not args.dry_run:
                copy_control_teacher_cache(
                    source_run=args.source_control_run,
                    destination_run=args.run_root / condition,
                )
            run_command(
                control_command(args=args, manifest=manifest),
                dry_run=args.dry_run,
            )
            continue
        target = campaign_target(manifest, condition)
        cache_source = None
        if args.reuse_teacher_caches:
            cache_source = source_target_run(
                rarity_manifest_path=args.rarity_manifest,
                source_root=args.source_target_root,
                trait=condition,
            )
        run_command(
            target_command(
                args=args,
                manifest=manifest,
                target=target,
                cache_source=cache_source,
            ),
            dry_run=args.dry_run,
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=[*CONDITIONS, "all"], default="all")
    parser.add_argument(
        "--campaign-manifest", type=Path, default=DEFAULT_CAMPAIGN_MANIFEST
    )
    parser.add_argument("--rarity-manifest", type=Path, default=DEFAULT_RARITY_MANIFEST)
    parser.add_argument(
        "--source-target-root", type=Path, default=DEFAULT_SOURCE_TARGET_ROOT
    )
    parser.add_argument(
        "--source-control-run", type=Path, default=DEFAULT_SOURCE_CONTROL_RUN
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help=(
            "Campaign data root. By default, derived from --seed as "
            "data/demo/qwen25_plant_fine_dynamics_10step_seedN."
        ),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help=(
            "Campaign results root. By default, derived from --seed as "
            "results/qwen25_plant_fine_dynamics_10step_seedN."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Training/data-order seed; defaults to the campaign manifest.",
    )
    parser.add_argument(
        "--lora-init-seed",
        type=int,
        default=None,
        help="LoRA initialization seed; defaults to --seed when supplied.",
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--teacher-batch-size", type=int, default=64)
    parser.add_argument("--prefix-cache-train-microbatch-size", type=int, default=32)
    parser.add_argument(
        "--no-reuse-teacher-caches",
        action="store_false",
        dest="reuse_teacher_caches",
    )
    parser.set_defaults(reuse_teacher_caches=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    """Run the CLI."""
    args = build_parser().parse_args()
    if args.teacher_batch_size <= 0 or args.prefix_cache_train_microbatch_size <= 0:
        raise ValueError("Batch sizes must be positive.")
    if args.seed is not None and args.seed < 0:
        raise ValueError("--seed must be non-negative.")
    if args.lora_init_seed is not None and args.lora_init_seed < 0:
        raise ValueError("--lora-init-seed must be non-negative.")
    if args.force and args.reuse_teacher_caches:
        raise ValueError("--force requires --no-reuse-teacher-caches.")
    run(args)


if __name__ == "__main__":
    main()
