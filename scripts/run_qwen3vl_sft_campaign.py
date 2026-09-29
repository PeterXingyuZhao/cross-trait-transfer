#!/usr/bin/env python3
"""Run balanced four-GPU Qwen3-VL token-SFT campaigns and full-family SL scoring.

Workers partition students, never the immutable teacher/answer catalogs. A Slurm
barrier separates generation/training from complete 30-by-30 checkpoint scoring.
Generation keeps one model resident per worker and cell; training runs in fresh
processes so each student's LoRA parameters and optimizer start independently.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import fcntl
from pathlib import Path
import shlex
import subprocess
import sys
from types import ModuleType
from typing import Any

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_number_sequence_trait_cell as number  # noqa: E402
from scripts.run_long_form_trait_cell import CellSpec, cell_m10_command, load_cell_spec  # noqa: E402
from scripts.native_generation_loader import QWEN3_VL_REPLICATION_REVISION  # noqa: E402
from scripts.train_trait_response_sft import slugify, utc_now  # noqa: E402

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
WORKERS = 4
CAMPAIGNS = {
    "chemical": (("animal", "element"), ("color", "element"), ("city", "element")),
    "tree": (("tree", "number"), ("tree", "element")),
}


@dataclass(frozen=True, kw_only=True)
class Task:
    """A single student's identity within an unchanged full-family cell."""

    cell_index: int
    trait: str


def config_paths(campaign: str) -> tuple[Path, ...]:
    """Return the fixed scientific cells for one campaign."""
    return tuple(PROJECT_ROOT / "experiment_plan/scaling" / (
        f"qwen3vl_{domain}30_{carrier}_sequence_sft_10k_alpha32_3ep_h100_v1.json"
    ) for domain, carrier in CAMPAIGNS[campaign])


def runner(spec: CellSpec) -> ModuleType:
    """Select a carrier-specific pipeline without changing its protocol."""
    if spec.payload["carrier"] == "number_sequence_00_99":
        return number
    if spec.payload["carrier"] == "element_sequence_100":
        from scripts import run_element_sequence_trait_cell as element

        return element
    raise ValueError(f"Unsupported carrier {spec.payload['carrier']}")


def load_specs(campaign: str) -> tuple[CellSpec, ...]:
    """Read and validate all catalogs before allowing any GPU work."""
    specs = tuple(load_cell_spec(path) for path in config_paths(campaign))
    for spec in specs:
        if len(spec.traits) != 30:
            raise ValueError("Every cell must retain the complete 30-trait catalog")
        p = spec.payload
        if (p["model_id"] != MODEL_ID
                or p["model_revision"] != QWEN3_VL_REPLICATION_REVISION
                or p.get("training_backend") != "transformers"
                or p.get("generation_backend") != "transformers"
                or p.get("eval_torch_dtype") != "float32"):
            raise ValueError("Campaign requires pinned native Qwen3-VL and FP32 SL scoring")
        runner(spec).preflight(spec)
    if len({spec.output_dir.resolve() for spec in specs}) != len(specs):
        raise ValueError("Scientific cells must not share an output directory")
    return specs


def shard_tasks(specs: tuple[CellSpec, ...], rank: int) -> tuple[Task, ...]:
    """Round-robin independent students across four GPUs, including each cell."""
    if rank not in range(WORKERS):
        raise ValueError("Worker rank must be 0, 1, 2, or 3")
    tasks = tuple(Task(cell_index=index, trait=trait)
                  for index, spec in enumerate(specs) for trait in spec.traits)
    return tasks[rank::WORKERS]


def plan(specs: tuple[CellSpec, ...]) -> dict[str, Any]:
    """Describe the partition while preserving all answers in SL scoring."""
    return {
        "model_id": MODEL_ID,
        "student_counts": [len(shard_tasks(specs, rank)) for rank in range(WORKERS)],
        "workers": [{"rank": rank, "cells": [{
            "config": str(spec.config_path),
            "traits": [task.trait for task in shard_tasks(specs, rank)
                       if task.cell_index == index],
            "answer_count": len(spec.traits),
        } for index, spec in enumerate(specs)]} for rank in range(WORKERS)],
        "evaluation": "All checkpoint/final 30x30 SL matrices; no Metrics 1-8",
    }


def fingerprint(spec: CellSpec) -> str:
    """Serialize only immutable protocol metadata publication, not GPU work."""
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    with (spec.output_dir / "protocol.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        return runner(spec).freeze_protocol(spec)


def command_run(command: list[str]) -> None:
    """Run an isolated GPU stage, propagating failure to the Slurm barrier."""
    logger.info("Command: {}", shlex.join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def train_worker(specs: tuple[CellSpec, ...], rank: int) -> None:
    """Generate assigned teachers in batches, then fit independent students."""
    tasks = shard_tasks(specs, rank)
    for index, spec in enumerate(specs):
        selected = tuple(task.trait for task in tasks if task.cell_index == index)
        if not selected:
            continue
        pipe = runner(spec)
        protocol = fingerprint(spec)
        with ExitStack() as locks:
            for trait in selected:
                locks.enter_context(number.exclusive_lock(
                    spec.output_dir / "traits" / slugify(trait) / "campaign_ownership.lock"))
            command_run([
                sys.executable, str(Path(__file__).resolve()),
                "--generation-config", str(spec.config_path), "--stage", "generation-worker",
                "--execute", *[item for trait in selected for item in ("--trait", trait)],
            ])
            for trait in selected:
                with number.exclusive_lock(spec.output_dir / "traits" / slugify(trait) / "training.lock"):
                    pipe.completed_training(spec, trait, protocol, train=True)
            number.write_json(spec.output_dir / "campaign_workers" / f"worker-{rank}.json", {
                "status": "completed", "fingerprint": protocol,
                "traits": list(selected), "finished_at": utc_now(),
            })


def score_worker(specs: tuple[CellSpec, ...], rank: int) -> None:
    """Score complete families after all four training workers succeed."""
    if rank not in range(WORKERS):
        raise ValueError("Worker rank must be 0, 1, 2, or 3")
    if rank >= len(specs):
        logger.info("Worker {} has no full-family scoring task", rank)
        return
    spec = specs[rank]
    pipe = runner(spec)
    protocol = fingerprint(spec)
    with number.exclusive_lock(spec.output_dir / "campaign_scoring.lock"):
        # Validation below checks all thirty completed adapters and checkpoint
        # schedules; worker completion flags alone are never sufficient.
        pipe.collect_training(spec, protocol)
        command = cell_m10_command(spec)
        command[command.index("--torch-dtype") + 1] = "float32"
        command_run(command)
        number.write_json(spec.output_dir / "campaign_complete.json", {
            "status": "completed", "fingerprint": protocol,
            "trait_count": len(spec.traits), "answer_count": len(spec.traits),
            "evaluation_dtype": "float32", "finished_at": utc_now(),
        })


def generation_worker(config: Path, traits: tuple[str, ...]) -> None:
    """Generate a subset without truncating the cell's fingerprint or catalogs."""
    spec = load_cell_spec(config)
    pipe = runner(spec)
    pipe.preflight(spec)
    if not traits or len(set(traits)) != len(traits) or any(t not in spec.traits for t in traits):
        raise ValueError("Generation requires a nonempty unique subset of the full catalog")
    protocol = fingerprint(spec)
    pipe.generate_datasets(spec, protocol, selected_traits=traits)


def main() -> None:
    """Expose CPU preflight and the two barrier-separated Slurm stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", choices=CAMPAIGNS)
    parser.add_argument("--stage", choices=("preflight", "tokenizer-check", "prepare", "train", "score", "generation-worker"), default="preflight")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--generation-config", type=Path)
    parser.add_argument("--trait", action="append", default=[])
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.stage == "generation-worker":
        if args.generation_config is None or not args.execute or args.campaign:
            parser.error("generation-worker requires --generation-config and --execute only")
        generation_worker(args.generation_config, tuple(args.trait))
        return
    if args.campaign is None or args.generation_config or args.trait:
        parser.error("Campaign stages require --campaign and no generation overrides")
    specs = load_specs(args.campaign)
    if args.stage == "preflight":
        logger.info("Campaign plan: {}", plan(specs))
        return
    if args.stage == "tokenizer-check":
        for spec in specs:
            runner(spec).tokenizer_check(spec)
        return
    if not args.execute:
        logger.info("Dry run: {} {}", args.stage, plan(specs))
        return
    if args.stage == "prepare":
        for spec in specs:
            fingerprint(spec)
        return
    if args.rank is None:
        parser.error("train/score require --rank")
    (train_worker if args.stage == "train" else score_worker)(specs, args.rank)


if __name__ == "__main__":
    main()
