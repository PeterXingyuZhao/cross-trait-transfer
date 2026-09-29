#!/usr/bin/env python3
"""Run and consolidate one multi-trait GSM8K or Python carrier cell."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, cast

from loguru import logger


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_long_form_carrier_calibration import (  # noqa: E402
    Stage,
    load_config,
    run_pipeline,
    utc_now,
)
from scripts.score_string_trait_number_distribution import (  # noqa: E402
    slugify as m10_slugify,
)
from scripts.train_trait_response_sft import (  # noqa: E402
    slugify as response_sft_slugify,
)


CELL_ONLY_FIELDS = {
    "traits_path",
    "target_trait_count",
    "python_metric",
    "eval_max_prompt_batch_size",
    "eval_max_branch_batch_size",
}
DELEGATED_STAGES = (
    "preflight",
    "pilot",
    "generate",
    "train",
    "m10",
    "saturation",
    "all",
    "calibrated-all",
    "calibrated-train",
)
CELL_STAGES = (
    *DELEGATED_STAGES,
    "collect",
    "cell-m10",
    "python-metric",
    "cell-all",
)
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(frozen=True, **DATACLASS_KW_ONLY)
class CellSpec:
    """Validated multi-trait long-form cell specification."""

    config_path: Path
    payload: dict[str, Any]
    traits_path: Path
    traits: tuple[str, ...]
    output_dir: Path


def resolve_path(value: str) -> Path:
    """Resolve a project-relative path."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_string_list(path: Path) -> tuple[str, ...]:
    """Load a non-empty JSON list of unique strings."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, list)
        or not payload
        or not all(isinstance(value, str) and value.strip() for value in payload)
    ):
        raise ValueError(f"{path} must contain a non-empty JSON string list.")
    values = tuple(value.strip() for value in payload)
    if len(values) != len(set(values)):
        raise ValueError(f"{path} contains duplicate traits.")
    return values


def load_cell_spec(path: Path) -> CellSpec:
    """Load a cell config and validate its frozen trait catalog."""
    config_path = path if path.is_absolute() else PROJECT_ROOT / path
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Cell config must be a JSON object.")
    for field in ("name", "output_dir", "traits_path"):
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            raise ValueError(f"{field} must be a non-empty string.")
    if "trait" in payload:
        raise ValueError("Cell configs use traits_path and must not set trait.")
    traits_path = resolve_path(payload["traits_path"])
    if not traits_path.is_file():
        raise FileNotFoundError(f"Missing trait catalog: {traits_path}")
    traits = load_string_list(traits_path)
    answers_value = payload.get("answers_path")
    questions_value = payload.get("questions_path")
    if not isinstance(answers_value, str) or not answers_value.strip():
        raise ValueError("answers_path must be a non-empty string.")
    if not isinstance(questions_value, str) or not questions_value.strip():
        raise ValueError("questions_path must be a non-empty string.")
    answers_path = resolve_path(answers_value)
    questions_path = resolve_path(questions_value)
    if not answers_path.is_file():
        raise FileNotFoundError(f"Missing answer catalog: {answers_path}")
    if not questions_path.is_file():
        raise FileNotFoundError(f"Missing evaluation questions: {questions_path}")
    answers = load_string_list(answers_path)
    load_string_list(questions_path)
    if answers != traits:
        raise ValueError(
            "answers_path must contain exactly the traits_path labels in the "
            "same order so the M10 diagonal has matched-trait semantics."
        )
    system_prompt_template = payload.get("system_prompt_template")
    if (
        not isinstance(system_prompt_template, str)
        or not system_prompt_template.strip()
    ):
        raise ValueError("system_prompt_template must be a non-empty string.")
    try:
        for trait in traits:
            system_prompt_template.format(trait=trait)
    except (KeyError, IndexError) as exc:
        raise ValueError(
            "system_prompt_template must render every trait using {trait}."
        ) from exc
    target_count = payload.get("target_trait_count", len(traits))
    if target_count != len(traits):
        raise ValueError(
            f"target_trait_count={target_count} does not match {len(traits)} traits."
        )
    training_slugs = tuple(response_sft_slugify(trait) for trait in traits)
    if len(training_slugs) != len(set(training_slugs)):
        raise ValueError("Trait names collide after path slugification.")
    evaluation_slugs = tuple(m10_slugify(trait) for trait in traits)
    if len(evaluation_slugs) != len(set(evaluation_slugs)):
        raise ValueError("Trait names collide after M10 path slugification.")
    return CellSpec(
        config_path=config_path,
        payload=payload,
        traits_path=traits_path,
        traits=traits,
        output_dir=resolve_path(payload["output_dir"]),
    )


def trait_config_payload(spec: CellSpec, index: int) -> dict[str, Any]:
    """Materialize one existing single-trait runner configuration."""
    if index < 0 or index >= len(spec.traits):
        raise ValueError(
            f"trait index {index} is outside 0..{len(spec.traits) - 1}."
        )
    trait = spec.traits[index]
    payload = {
        key: value
        for key, value in spec.payload.items()
        if key not in CELL_ONLY_FIELDS
    }
    payload["name"] = (
        f"{spec.payload['name']}_{index:02d}_{response_sft_slugify(trait)}"
    )
    payload["trait"] = trait
    payload["output_dir"] = str(
        spec.output_dir / "traits" / response_sft_slugify(trait)
    )
    return payload


def canonical_json(payload: dict[str, Any]) -> str:
    """Serialize a generated trait config deterministically."""
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def sha256_text(text: str) -> str:
    """Return a hexadecimal SHA-256 digest."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def trait_config_path(spec: CellSpec, index: int) -> Path:
    """Return the frozen generated-config path for one trait."""
    trait = spec.traits[index]
    return (
        spec.output_dir
        / "trait_configs"
        / f"{index:02d}_{response_sft_slugify(trait)}.json"
    )


def materialize_trait_config(spec: CellSpec, index: int) -> tuple[Path, str]:
    """Write or validate one immutable per-trait config."""
    path = trait_config_path(spec, index)
    content = canonical_json(trait_config_payload(spec, index))
    trait = spec.traits[index]
    training_manifest = (
        spec.output_dir
        / "traits"
        / response_sft_slugify(trait)
        / "training"
        / "training_manifest.json"
    )
    if path.is_file():
        existing = path.read_text(encoding="utf-8")
        if existing != content and training_manifest.is_file():
            raise ValueError(
                f"Refusing config drift for trained trait {trait!r}: {path}"
            )
        if existing == content:
            return path, sha256_text(content)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path, sha256_text(content)


def trait_index_from_args(raw_index: int | None) -> int:
    """Resolve an explicit or Slurm-array trait index."""
    if raw_index is not None:
        return raw_index
    slurm_index = os.environ.get("SLURM_ARRAY_TASK_ID")
    if slurm_index is None:
        raise ValueError(
            "Pass --trait-index or run inside a Slurm array with "
            "SLURM_ARRAY_TASK_ID."
        )
    try:
        return int(slurm_index)
    except ValueError as exc:
        raise ValueError("SLURM_ARRAY_TASK_ID must be an integer.") from exc


def adapter_has_weights(path: Path) -> bool:
    """Return whether an adapter directory contains model weights."""
    return any(
        (path / filename).is_file()
        for filename in ("adapter_model.safetensors", "adapter_model.bin")
    )


def read_json_object(path: Path) -> dict[str, Any]:
    """Read one JSON object with a path-aware error."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def validate_trait_training(
    spec: CellSpec,
    index: int,
) -> dict[str, Any]:
    """Validate one trait's generated data, run config, and LoRA artifacts."""
    trait = spec.traits[index]
    safe_trait = response_sft_slugify(trait)
    expected_config_path = trait_config_path(spec, index)
    expected_config_text = canonical_json(trait_config_payload(spec, index))
    if not expected_config_path.is_file():
        raise FileNotFoundError(
            f"Missing generated trait config: {expected_config_path}"
        )
    actual_config_text = expected_config_path.read_text(encoding="utf-8")
    if actual_config_text != expected_config_text:
        raise ValueError(f"Generated config drift detected: {expected_config_path}")
    config_digest = sha256_text(actual_config_text)
    config = load_config(expected_config_path)
    trait_root = spec.output_dir / "traits" / safe_trait
    run_manifest = read_json_object(trait_root / "trait_run_manifest.json")
    if run_manifest.get("config_sha256") != config_digest:
        raise ValueError(f"Config digest mismatch for trait {trait!r}.")
    generation = read_json_object(trait_root / "data" / "generation_manifest.json")
    if generation.get("status") != "complete":
        raise ValueError(f"Generation is not complete for trait {trait!r}.")
    for field, expected in (
        ("trait", trait),
        ("carrier", config.carrier),
        ("model_id", config.model_id),
    ):
        if generation.get(field) != expected:
            raise ValueError(
                f"Generation {field} mismatch for {trait!r}: "
                f"{generation.get(field)!r} != {expected!r}."
            )
    if (
        config.training_example_count is not None
        and int(generation.get("filtered_count", -1))
        < config.training_example_count
    ):
        raise ValueError(f"Too few filtered rows for trait {trait!r}.")
    training_root = trait_root / "training"
    training = read_json_object(training_root / "training_manifest.json")
    expected_training_fields = {
        "status": "completed",
        "trait": trait,
        "model_id": config.model_id,
        "n_epochs": config.n_epochs,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "max_seq_length": config.training_max_seq_length,
    }
    if config.training_example_count is not None:
        expected_training_fields["dataset_rows"] = config.training_example_count
    for field, expected in expected_training_fields.items():
        if training.get(field) != expected:
            raise ValueError(
                f"Training {field} mismatch for {trait!r}: "
                f"{training.get(field)!r} != {expected!r}."
            )
    if set(training.get("target_modules", [])) != set(config.target_modules):
        raise ValueError(f"LoRA target-module mismatch for trait {trait!r}.")
    checkpoint_steps = training.get("checkpoint_steps")
    if (
        not isinstance(checkpoint_steps, list)
        or not checkpoint_steps
        or not all(isinstance(step, int) and step > 0 for step in checkpoint_steps)
    ):
        raise ValueError(f"Invalid checkpoint schedule for trait {trait!r}.")
    final_adapter = training_root / "adapters" / safe_trait
    final_adapter_config = read_json_object(final_adapter / "adapter_config.json")
    if not adapter_has_weights(final_adapter):
        raise FileNotFoundError(f"Missing final adapter weights: {final_adapter}")
    if (
        final_adapter_config.get("r") != config.lora_r
        or final_adapter_config.get("lora_alpha") != config.lora_alpha
    ):
        raise ValueError(f"Final adapter LoRA mismatch for trait {trait!r}.")
    for step in checkpoint_steps:
        checkpoint = (
            training_root
            / "trainer"
            / safe_trait
            / f"checkpoint-step-{step:06d}"
        )
        metadata = read_json_object(checkpoint / "metadata.json")
        adapter = checkpoint / "adapter"
        read_json_object(adapter / "adapter_config.json")
        if not adapter_has_weights(adapter):
            raise FileNotFoundError(f"Missing checkpoint weights: {adapter}")
        if metadata.get("global_step") != step:
            raise ValueError(
                f"Checkpoint metadata mismatch for {trait!r} at step {step}."
            )
    return {
        "index": index,
        "trait": trait,
        # Keep the legacy field for manifests produced before canonical M10
        # links were introduced. New consumers should use the explicit names.
        "slug": safe_trait,
        "training_slug": safe_trait,
        "evaluation_slug": m10_slugify(trait),
        "config_path": str(expected_config_path),
        "config_sha256": config_digest,
        "trait_output_dir": str(trait_root),
        "training_output_dir": str(training_root),
        "dataset_rows": training.get("dataset_rows"),
        "checkpoint_steps": checkpoint_steps,
        "final_adapter": str(final_adapter),
    }


def ensure_relative_directory_link(source: Path, link: Path) -> None:
    """Create an auditable relative symlink without overwriting artifacts."""
    if not source.is_dir():
        raise FileNotFoundError(f"Cannot link missing directory: {source}")
    if link.is_symlink():
        if link.resolve() != source.resolve():
            raise ValueError(f"Existing symlink points elsewhere: {link}")
        return
    if link.exists():
        raise FileExistsError(f"Refusing to replace existing path: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    relative_source = Path(os.path.relpath(source, start=link.parent))
    link.symlink_to(relative_source, target_is_directory=True)


def collect_training(spec: CellSpec, *, execute: bool) -> dict[str, Any]:
    """Validate all traits and construct the shared evaluation layout."""
    records = [
        validate_trait_training(spec, index)
        for index in range(len(spec.traits))
    ]
    schedules = {tuple(record["checkpoint_steps"]) for record in records}
    if len(schedules) != 1:
        raise ValueError("Trait runs do not share one checkpoint schedule.")
    common_steps = list(next(iter(schedules)))
    consolidated = spec.output_dir / "consolidated_training"
    manifest = {
        "status": "validated",
        "created_at": utc_now(),
        "cell_config": str(spec.config_path),
        "cell_config_sha256": hashlib.sha256(
            spec.config_path.read_bytes()
        ).hexdigest(),
        "model_id": spec.payload["model_id"],
        "carrier": spec.payload["carrier"],
        "trait_count": len(spec.traits),
        "traits": list(spec.traits),
        "common_checkpoint_steps": common_steps,
        "link_mode": "relative_symlink",
        "records": records,
    }
    logger.success(
        "Validated {} {} trait runs with shared checkpoints {}",
        len(records),
        spec.payload["carrier"],
        common_steps,
    )
    if not execute:
        return manifest
    for record in records:
        training_slug = record["training_slug"]
        evaluation_slug = record["evaluation_slug"]
        training_root = Path(record["training_output_dir"])
        ensure_relative_directory_link(
            training_root / "adapters" / training_slug,
            consolidated / "adapters" / evaluation_slug,
        )
        ensure_relative_directory_link(
            training_root / "trainer" / training_slug,
            consolidated / "trainer" / evaluation_slug,
        )
    consolidated.mkdir(parents=True, exist_ok=True)
    manifest_path = consolidated / "collection_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.success("Saved validated collection manifest to {}", manifest_path)
    return manifest


def cell_m10_command(spec: CellSpec) -> list[str]:
    """Build the cross-trait behavioral checkpoint command."""
    return [
        sys.executable,
        str(SCRIPT_DIR / "evaluate_trait_checkpoint_effect_matrices.py"),
        "--training-output-dir",
        str(spec.output_dir / "consolidated_training"),
        "--output-dir",
        str(spec.output_dir / "m10"),
        "--answers-path",
        str(resolve_path(spec.payload["answers_path"])),
        "--questions-path",
        str(resolve_path(spec.payload["questions_path"])),
        "--traits",
        ",".join(spec.traits),
        "--base-model-id",
        str(spec.payload["model_id"]),
        "--batch-size",
        str(spec.payload["eval_batch_size"]),
        "--use-prefix-cache",
        "--autotune-batch-pair",
        "--max-prompt-batch-size",
        str(spec.payload.get("eval_max_prompt_batch_size", 32)),
        "--max-branch-batch-size",
        str(spec.payload.get("eval_max_branch_batch_size", 64)),
        "--device",
        "auto",
        "--torch-dtype",
        "auto",
        "--skip-existing",
        (
            "--enable-thinking"
            if bool(spec.payload.get("enable_thinking", False))
            else "--disable-thinking"
        ),
    ]


def python_metric_command(spec: CellSpec) -> list[str]:
    """Build the optional 100+OTHER every-code-position metric command."""
    if spec.payload.get("carrier") != "python":
        raise ValueError("python-metric is only valid for a Python carrier cell.")
    options = spec.payload.get("python_metric", {})
    if not isinstance(options, dict):
        raise ValueError("python_metric must be a JSON object.")
    return [
        sys.executable,
        str(SCRIPT_DIR / "evaluate_python_code_token_panel.py"),
        "--model-id",
        str(spec.payload["model_id"]),
        "--reference-jsonl",
        str(resolve_path(spec.payload["source_path"])),
        "--output-dir",
        str(spec.output_dir / "code_token_metric"),
        "--traits",
        ",".join(spec.traits),
        "--system-prompt-template",
        str(spec.payload["system_prompt_template"]),
        "--training-output-dir",
        str(spec.output_dir / "consolidated_training"),
        "--all-checkpoints",
        "--discovery-prompt-count",
        str(options.get("discovery_prompt_count", 256)),
        "--evaluation-prompt-count",
        str(options.get("evaluation_prompt_count", 256)),
        "--panel-size",
        str(options.get("panel_size", 100)),
        "--batch-size",
        str(options.get("batch_size", 1)),
        "--device",
        "auto",
        "--torch-dtype",
        "auto",
        "--skip-existing",
        "--disable-thinking",
    ]


def run_subprocess(command: list[str], *, execute: bool) -> None:
    """Log and optionally execute a cell-level metric command."""
    logger.info("Cell command: {}", shlex.join(command))
    if execute:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def persistent_generation_command(spec: CellSpec) -> list[str]:
    """Build one vLLM process that generates every trait dataset."""
    generation_python = (
        os.environ.get("VLLM_PYTHON", sys.executable)
        if spec.payload.get("generation_backend") == "vllm"
        else sys.executable
    )
    return [
        generation_python,
        str(SCRIPT_DIR / "generate_long_form_trait_cell_datasets.py"),
        "--config",
        str(spec.config_path),
    ]


def write_cell_pipeline_manifest(
    spec: CellSpec,
    *,
    status: str,
    phase: str,
    completed_training_traits: list[str],
    error: str | None = None,
) -> None:
    """Persist cell-level progress so a replacement allocation can resume."""
    payload: dict[str, Any] = {
        "status": status,
        "updated_at": utc_now(),
        "phase": phase,
        "cell_config": str(spec.config_path),
        "carrier": spec.payload["carrier"],
        "trait_count": len(spec.traits),
        "completed_training_traits": completed_training_traits,
    }
    if error is not None:
        payload["error"] = error
    path = spec.output_dir / "cell_pipeline_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_full_cell(spec: CellSpec, *, execute: bool) -> None:
    """Generate, train, consolidate, and evaluate all traits in one job."""
    trait_configs: list[tuple[int, Path, str]] = []
    for index in range(len(spec.traits)):
        config_path, config_digest = materialize_trait_config(spec, index)
        run_pipeline(
            load_config(config_path),
            stage=cast(Stage, "preflight"),
            execute=False,
        )
        trait_configs.append((index, config_path, config_digest))

    generation = persistent_generation_command(spec)
    if not execute:
        run_subprocess(generation, execute=False)
        for _, config_path, _ in trait_configs:
            run_pipeline(
                load_config(config_path),
                stage=cast(Stage, "train"),
                execute=False,
            )
        run_subprocess(cell_m10_command(spec), execute=False)
        return

    completed_training_traits: list[str] = []
    phase = "generation"
    write_cell_pipeline_manifest(
        spec,
        status="running",
        phase=phase,
        completed_training_traits=completed_training_traits,
    )
    try:
        # The dedicated generation interpreter exits before SFT starts, fully
        # releasing the vLLM engine and its CUDA memory.
        run_subprocess(generation, execute=True)
        phase = "training"
        write_cell_pipeline_manifest(
            spec,
            status="running",
            phase=phase,
            completed_training_traits=completed_training_traits,
        )
        for index, config_path, config_digest in trait_configs:
            trait = spec.traits[index]
            logger.info(
                "Training trait {}/{} after cell-wide generation: {}",
                index + 1,
                len(spec.traits),
                trait,
            )
            run_pipeline(
                load_config(config_path),
                stage=cast(Stage, "train"),
                execute=True,
            )
            write_trait_run_manifest(
                spec,
                index=index,
                stage="cell-all",
                config_path=config_path,
                config_digest=config_digest,
            )
            completed_training_traits.append(trait)
            write_cell_pipeline_manifest(
                spec,
                status="running",
                phase=phase,
                completed_training_traits=completed_training_traits,
            )

        phase = "evaluation"
        write_cell_pipeline_manifest(
            spec,
            status="running",
            phase=phase,
            completed_training_traits=completed_training_traits,
        )
        collect_training(spec, execute=True)
        run_subprocess(cell_m10_command(spec), execute=True)
    except Exception as exc:
        write_cell_pipeline_manifest(
            spec,
            status="failed",
            phase=phase,
            completed_training_traits=completed_training_traits,
            error=str(exc),
        )
        raise
    write_cell_pipeline_manifest(
        spec,
        status="completed",
        phase="completed",
        completed_training_traits=completed_training_traits,
    )
    logger.success(
        "Completed cell-wide generation, training, and M10 for {} traits",
        len(spec.traits),
    )


def write_trait_run_manifest(
    spec: CellSpec,
    *,
    index: int,
    stage: str,
    config_path: Path,
    config_digest: str,
) -> None:
    """Record the immutable generated config used by one array task."""
    trait = spec.traits[index]
    path = (
        spec.output_dir
        / "traits"
        / response_sft_slugify(trait)
        / "trait_run_manifest.json"
    )
    payload = {
        "status": "completed",
        "updated_at": utc_now(),
        "cell_config": str(spec.config_path),
        "trait_index": index,
        "trait": trait,
        "stage": stage,
        "config_path": str(config_path),
        "config_sha256": config_digest,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_cell(
    spec: CellSpec,
    *,
    stage: str,
    trait_index: int | None,
    execute: bool,
) -> None:
    """Dispatch a per-trait array stage or a validated cell-level stage."""
    if stage == "cell-all":
        run_full_cell(spec, execute=execute)
        return
    if stage == "collect":
        collect_training(spec, execute=execute)
        return
    if stage in {"cell-m10", "python-metric"}:
        collect_training(spec, execute=execute)
        command = (
            cell_m10_command(spec)
            if stage == "cell-m10"
            else python_metric_command(spec)
        )
        run_subprocess(command, execute=execute)
        return
    index = trait_index_from_args(trait_index)
    config_path, config_digest = materialize_trait_config(spec, index)
    logger.info(
        "Running {} trait {}/{}: {}",
        stage,
        index + 1,
        len(spec.traits),
        spec.traits[index],
    )
    run_pipeline(
        load_config(config_path),
        stage=cast(Stage, stage),
        execute=execute,
    )
    if execute:
        write_trait_run_manifest(
            spec,
            index=index,
            stage=stage,
            config_path=config_path,
            config_digest=config_digest,
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=CELL_STAGES, default="preflight")
    parser.add_argument("--trait-index", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    """Run one long-form cell command."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        run_cell(
            load_cell_spec(args.config),
            stage=args.stage,
            trait_index=args.trait_index,
            execute=args.execute,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        ValueError,
        subprocess.CalledProcessError,
    ) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
