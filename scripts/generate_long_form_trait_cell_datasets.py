#!/usr/bin/env python3
"""Generate every trait dataset in a cell with one persistent model runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

from loguru import logger


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_trait_response_sft_dataset import (  # noqa: E402
    build_parser as build_generation_parser,
    load_vllm_and_tokenizer,
    run as run_generation,
    validate_args as validate_generation_args,
)
from scripts.run_long_form_carrier_calibration import (  # noqa: E402
    CalibrationConfig,
    generation_command,
    generation_quality_report,
    load_config,
    require_actual_training_rows,
    utc_now,
)
from scripts.run_long_form_trait_cell import (  # noqa: E402
    CellSpec,
    load_cell_spec,
    materialize_trait_config,
)


def read_json_object(path: Path) -> dict[str, Any] | None:
    """Read a JSON object, returning ``None`` for absent or invalid files."""
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def generation_is_ready(config: CalibrationConfig) -> bool:
    """Return whether a trait already has its complete SFT row budget."""
    manifest = read_json_object(
        config.output_dir / "data" / "generation_manifest.json"
    )
    if manifest is None or manifest.get("status") != "complete":
        return False
    for field, expected in (
        ("carrier", config.carrier),
        ("trait", config.trait),
        ("model_id", config.model_id),
    ):
        if manifest.get(field) != expected:
            return False
    if config.training_example_count is not None:
        return (
            int(manifest.get("filtered_count", -1))
            >= config.training_example_count
        )
    expected_requests = config.full_prompt_count * config.responses_per_prompt
    return int(manifest.get("completed_request_count", -1)) >= expected_requests


def generation_args(config: CalibrationConfig) -> argparse.Namespace:
    """Convert the canonical generation command into validated in-process args."""
    command = generation_command(config, pilot=False)
    if len(command) < 2:
        raise RuntimeError("Generation command is unexpectedly short.")
    args = build_generation_parser().parse_args(command[2:])
    validate_generation_args(args)
    if args.generation_backend != "vllm":
        raise ValueError(
            "Persistent cell generation currently requires generation_backend=vllm."
        )
    return args


def write_cell_generation_manifest(
    spec: CellSpec,
    *,
    status: str,
    completed_traits: list[str],
    pending_traits: list[str],
    shared_model_load_seconds: float,
    error: str | None = None,
) -> None:
    """Persist resumable progress for the shared generation phase."""
    payload: dict[str, Any] = {
        "status": status,
        "updated_at": utc_now(),
        "cell_config": str(spec.config_path),
        "carrier": spec.payload["carrier"],
        "model_id": spec.payload["model_id"],
        "trait_count": len(spec.traits),
        "completed_traits": completed_traits,
        "pending_traits": pending_traits,
        "shared_model_load_seconds": shared_model_load_seconds,
        "single_persistent_model_runtime": True,
        "prefix_caching_enabled": bool(
            spec.payload.get("generation_enable_prefix_caching", False)
        ),
    }
    if error is not None:
        payload["error"] = error
    path = spec.output_dir / "cell_generation_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_trait_configs(spec: CellSpec) -> list[CalibrationConfig]:
    """Materialize and load every immutable single-trait configuration."""
    configs = []
    for index in range(len(spec.traits)):
        config_path, _ = materialize_trait_config(spec, index)
        configs.append(load_config(config_path))
    identities = {
        (
            config.model_id,
            config.generation_backend,
            config.generation_max_seq_length,
            config.load_in_4bit,
        )
        for config in configs
    }
    if len(identities) != 1:
        raise ValueError("All traits must share one generation runtime configuration.")
    return configs


def run_cell_generation(spec: CellSpec) -> None:
    """Generate missing datasets sequentially without restarting vLLM."""
    configs = load_trait_configs(spec)
    pending = [config for config in configs if not generation_is_ready(config)]
    completed = [config.trait for config in configs if config not in pending]
    logger.info(
        "Cell generation: {} traits already complete, {} traits pending",
        len(completed),
        len(pending),
    )
    if not pending:
        write_cell_generation_manifest(
            spec,
            status="completed",
            completed_traits=completed,
            pending_traits=[],
            shared_model_load_seconds=0.0,
        )
        logger.success("Every trait dataset is already complete; skipping vLLM load")
        return

    parsed_args = {config.trait: generation_args(config) for config in pending}
    load_started = time.perf_counter()
    runtime = load_vllm_and_tokenizer(parsed_args[pending[0].trait])
    shared_model_load_seconds = time.perf_counter() - load_started
    logger.success(
        "Loaded one shared vLLM runtime in {:.1f} seconds for {} pending traits",
        shared_model_load_seconds,
        len(pending),
    )
    write_cell_generation_manifest(
        spec,
        status="running",
        completed_traits=completed,
        pending_traits=[config.trait for config in pending],
        shared_model_load_seconds=shared_model_load_seconds,
    )

    try:
        for position, config in enumerate(pending, start=1):
            logger.info(
                "Generating pending trait {}/{}: {}",
                position,
                len(pending),
                config.trait,
            )
            run_generation(
                parsed_args[config.trait],
                model_and_tokenizer=runtime,
            )
            generation_quality_report(config, phase="generate")
            require_actual_training_rows(config)
            completed.append(config.trait)
            remaining = [item.trait for item in pending[position:]]
            write_cell_generation_manifest(
                spec,
                status="running" if remaining else "completed",
                completed_traits=completed,
                pending_traits=remaining,
                shared_model_load_seconds=shared_model_load_seconds,
            )
    except Exception as exc:
        remaining = [
            config.trait for config in configs if config.trait not in completed
        ]
        write_cell_generation_manifest(
            spec,
            status="failed",
            completed_traits=completed,
            pending_traits=remaining,
            shared_model_load_seconds=shared_model_load_seconds,
            error=str(exc),
        )
        raise
    logger.success("Generated complete SFT datasets for all {} traits", len(configs))


def build_parser() -> argparse.ArgumentParser:
    """Build the cell-generation CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main() -> None:
    """Run persistent multi-trait generation."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        run_cell_generation(load_cell_spec(args.config))
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
