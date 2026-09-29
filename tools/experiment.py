"""Public stages for data generation, student training, and recovery evaluation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("preflight", "generate", "train", "recovery")


def family(config: dict[str, Any]) -> str:
    """Identify an experiment's training objective from its configuration."""
    carrier = config.get("carrier")
    if carrier == "number_sequence_00_99":
        return "number-sft"
    if carrier in {"gsm8k", "python"}:
        return "long-form-sft"
    if "dataset_jsonl" in config and "traits" in config:
        return "distribution-distillation"
    raise ValueError("Unsupported configuration: use a setting listed in docs/EXPERIMENTS.md")


def stage_commands(
    stage: str, config_path: Path, *, root: Path = ROOT,
    trait_index: int | None = None,
) -> list[list[str]]:
    """Plan commands without loading models, writing data, or starting jobs."""
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    path = config_path if config_path.is_absolute() else root / config_path
    config = json.loads(path.read_text())
    kind = family(config)
    mapped = {"preflight": "preflight", "generate": "generate", "train": "train", "recovery": "m10"}[stage]
    if kind != "long-form-sft":
        if trait_index is not None:
            raise ValueError("--trait-index is supported only for GSM8K and Python SFT")
        if kind == "distribution-distillation" and stage == "generate":
            raise ValueError(
                "Distribution distillation uses the bundled neutral prompts and scores teacher "
                "probabilities inside the train stage. Run preflight, then train, then recovery. "
                "There is no sampled-response generation stage for this objective."
            )
        script = ("run_number_sequence_trait_cell.py" if kind == "number-sft"
                  else "run_trait_carrier_smoke_pipeline.py")
        return [[sys.executable, str(root / "scripts" / script), "--config", str(path),
                 "--stage", mapped, *([] if stage == "preflight" else ["--execute"])]]
    runner = str(root / "scripts/run_long_form_trait_cell.py")
    if stage == "recovery":
        if trait_index is not None:
            raise ValueError("Recovery evaluates the complete trained family; omit --trait-index")
        return [[sys.executable, runner, "--config", str(path), "--stage", "cell-m10", "--execute"]]
    if stage == "generate" and trait_index is None:
        interpreter = (os.environ.get("VLLM_PYTHON", sys.executable)
                       if config.get("generation_backend") == "vllm" else sys.executable)
        return [[interpreter, str(root / "scripts/generate_long_form_trait_cell_datasets.py"),
                 "--config", str(path)]]
    traits_path = Path(config["traits_path"])
    traits = json.loads((traits_path if traits_path.is_absolute() else root / traits_path).read_text())
    indices = list(range(len(traits))) if trait_index is None else [trait_index]
    if any(index < 0 or index >= len(traits) for index in indices):
        raise ValueError(f"Trait index must be between 0 and {len(traits) - 1}")
    return [[sys.executable, runner, "--config", str(path), "--stage", mapped,
             "--trait-index", str(index), *([] if stage == "preflight" else ["--execute"])]
            for index in indices]


def main(argv: list[str] | None = None) -> None:
    """Validate inputs and display a plan; execute GPU stages only on request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trait-index", type=int, help="One GSM8K/Python trait; default: complete family")
    parser.add_argument("--execute", action="store_true", help="Run the planned stage; may start GPU work")
    args = parser.parse_args(argv)
    commands = stage_commands(args.stage, args.config, trait_index=args.trait_index)
    for command in commands:
        logger.info("{}", shlex.join(command))
    if args.execute or args.stage == "preflight":
        for command in commands:
            subprocess.run(command, cwd=ROOT, check=True)
    else:
        logger.info("Plan only: pass --execute to run this stage")


if __name__ == "__main__":
    main()
