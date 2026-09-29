#!/usr/bin/env python3
"""Orchestrate resumable Qwen GSM8K/Python timing calibrations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
import time
from typing import Any, Literal

from loguru import logger


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
Stage = Literal[
    "preflight",
    "pilot",
    "generate",
    "train",
    "m10",
    "saturation",
    "all",
    "calibrated-all",
    "calibrated-train",
]
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}
GSM8K_SANITIZATION_POLICY = "conservative_sentence_v1"


@dataclass(frozen=True, **DATACLASS_KW_ONLY)
class CalibrationConfig:
    """Validated long-form calibration configuration."""

    name: str
    config_path: Path
    carrier: str
    trait: str
    model_id: str
    output_dir: Path
    source_path: Path
    source_url: str | None
    system_prompt: str
    answers_path: Path
    questions_path: Path
    pilot_prompt_count: int
    full_prompt_count: int
    responses_per_prompt: int
    generation_backend: str
    generation_batch_size: int
    generation_auto_batch_size: bool
    generation_max_auto_batch_size: int
    generation_auto_batch_batches_per_size: int
    generation_continuous_chunk_size: int
    generation_continuous_max_batch_tokens: int
    generation_enable_prefix_caching: bool
    generation_max_seq_length: int
    generation_max_new_tokens: int
    temperature: float
    top_p: float
    training_example_count: int | None
    n_epochs: int
    training_max_seq_length: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    training_auto_batch_size: bool
    training_max_auto_batch_size: int
    target_effective_batch_size: int
    training_batch_probe_steps: int
    memory_headroom_fraction: float
    lr: float
    warmup_steps: int
    lora_r: int
    lora_alpha: int
    checkpoint_count: int
    eval_batch_size: int
    seed: int
    load_in_4bit: bool
    enable_thinking: bool
    sanitize_trait_paragraphs: bool
    sanitization_max_removed_fraction: float
    sanitization_min_reasoning_chars: int
    target_modules: tuple[str, ...]


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def resolve_path(value: str) -> Path:
    """Resolve configuration paths relative to the project root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def require_positive_int(payload: dict[str, Any], name: str) -> int:
    """Read one required positive-integer configuration field."""
    value = payload.get(name)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def positive_int_or_default(payload: dict[str, Any], name: str, default: int) -> int:
    """Read an optional positive integer with a validated default."""
    value = payload.get(name, default)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def optional_positive_int(payload: dict[str, Any], name: str) -> int | None:
    """Read an optional positive integer configuration field."""
    value = payload.get(name)
    if value is None:
        return None
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer when provided.")
    return value


def load_config(path: Path) -> CalibrationConfig:
    """Load and validate a long-form calibration JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    required_strings = [
        "name",
        "carrier",
        "trait",
        "model_id",
        "output_dir",
        "source_path",
        "system_prompt_template",
        "answers_path",
        "questions_path",
    ]
    for name in required_strings:
        if not isinstance(payload.get(name), str) or not payload[name].strip():
            raise ValueError(f"{name} must be a non-empty string.")
    carrier = payload["carrier"]
    if carrier not in {"gsm8k", "python"}:
        raise ValueError("carrier must be 'gsm8k' or 'python'.")
    generation_backend = payload.get("generation_backend", "static")
    if generation_backend not in {"static", "continuous", "vllm"}:
        raise ValueError(
            "generation_backend must be 'static', 'continuous', or 'vllm'."
        )
    trait = payload["trait"].strip()
    try:
        system_prompt = payload["system_prompt_template"].format(trait=trait)
    except (KeyError, IndexError) as exc:
        raise ValueError("system_prompt_template must accept {trait}.") from exc
    target_modules = payload.get(
        "target_modules",
        [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    if (
        not isinstance(target_modules, list)
        or not target_modules
        or not all(
            isinstance(module, str) and module.strip() for module in target_modules
        )
    ):
        raise ValueError("target_modules must be a non-empty list of strings.")
    temperature = float(payload.get("temperature", 1.0))
    top_p = float(payload.get("top_p", 1.0))
    lr = float(payload.get("lr", 2e-4))
    memory_headroom_fraction = float(payload.get("memory_headroom_fraction", 0.12))
    sanitization_max_removed_fraction = float(
        payload.get("sanitization_max_removed_fraction", 0.50)
    )
    sanitization_min_reasoning_chars = int(
        payload.get("sanitization_min_reasoning_chars", 200)
    )
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative.")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1].")
    if lr <= 0.0:
        raise ValueError("lr must be positive.")
    if not 0.0 < memory_headroom_fraction < 1.0:
        raise ValueError("memory_headroom_fraction must be in (0, 1).")
    if not 0.0 <= sanitization_max_removed_fraction < 1.0:
        raise ValueError("sanitization_max_removed_fraction must be in [0, 1).")
    if sanitization_min_reasoning_chars < 0:
        raise ValueError("sanitization_min_reasoning_chars cannot be negative.")
    sanitize_trait_paragraphs = bool(
        payload.get("sanitize_trait_paragraphs", False)
    )
    if sanitize_trait_paragraphs and carrier != "gsm8k":
        raise ValueError("sanitize_trait_paragraphs is only valid for GSM8K.")
    return CalibrationConfig(
        name=payload["name"],
        config_path=path,
        carrier=carrier,
        trait=trait,
        model_id=payload["model_id"],
        output_dir=resolve_path(payload["output_dir"]),
        source_path=resolve_path(payload["source_path"]),
        source_url=payload.get("source_url"),
        system_prompt=system_prompt,
        answers_path=resolve_path(payload["answers_path"]),
        questions_path=resolve_path(payload["questions_path"]),
        pilot_prompt_count=require_positive_int(payload, "pilot_prompt_count"),
        full_prompt_count=require_positive_int(payload, "full_prompt_count"),
        responses_per_prompt=require_positive_int(payload, "responses_per_prompt"),
        generation_backend=generation_backend,
        generation_batch_size=require_positive_int(payload, "generation_batch_size"),
        generation_auto_batch_size=bool(
            payload.get("generation_auto_batch_size", True)
        ),
        generation_max_auto_batch_size=positive_int_or_default(
            payload, "generation_max_auto_batch_size", 64
        ),
        generation_auto_batch_batches_per_size=positive_int_or_default(
            payload, "generation_auto_batch_batches_per_size", 2
        ),
        generation_continuous_chunk_size=positive_int_or_default(
            payload, "generation_continuous_chunk_size", 512
        ),
        generation_continuous_max_batch_tokens=positive_int_or_default(
            payload, "generation_continuous_max_batch_tokens", 8192
        ),
        generation_enable_prefix_caching=bool(
            payload.get("generation_enable_prefix_caching", False)
        ),
        generation_max_seq_length=require_positive_int(
            payload, "generation_max_seq_length"
        ),
        generation_max_new_tokens=require_positive_int(
            payload, "generation_max_new_tokens"
        ),
        temperature=temperature,
        top_p=top_p,
        training_example_count=optional_positive_int(
            payload, "training_example_count"
        ),
        n_epochs=require_positive_int(payload, "n_epochs"),
        training_max_seq_length=require_positive_int(
            payload, "training_max_seq_length"
        ),
        per_device_train_batch_size=require_positive_int(
            payload, "per_device_train_batch_size"
        ),
        gradient_accumulation_steps=require_positive_int(
            payload, "gradient_accumulation_steps"
        ),
        training_auto_batch_size=bool(payload.get("training_auto_batch_size", True)),
        training_max_auto_batch_size=positive_int_or_default(
            payload, "training_max_auto_batch_size", 32
        ),
        target_effective_batch_size=positive_int_or_default(
            payload,
            "target_effective_batch_size",
            payload.get(
                "target_effective_batch_size",
                payload["per_device_train_batch_size"]
                * payload["gradient_accumulation_steps"],
            ),
        ),
        training_batch_probe_steps=positive_int_or_default(
            payload, "training_batch_probe_steps", 2
        ),
        memory_headroom_fraction=memory_headroom_fraction,
        lr=lr,
        warmup_steps=int(payload.get("warmup_steps", 5)),
        lora_r=require_positive_int(payload, "lora_r"),
        lora_alpha=require_positive_int(payload, "lora_alpha"),
        checkpoint_count=require_positive_int(payload, "checkpoint_count"),
        eval_batch_size=require_positive_int(payload, "eval_batch_size"),
        seed=int(payload.get("seed", 1)),
        load_in_4bit=bool(payload.get("load_in_4bit", False)),
        enable_thinking=bool(payload.get("enable_thinking", False)),
        sanitize_trait_paragraphs=sanitize_trait_paragraphs,
        sanitization_max_removed_fraction=(
            sanitization_max_removed_fraction
        ),
        sanitization_min_reasoning_chars=sanitization_min_reasoning_chars,
        target_modules=tuple(module.strip() for module in target_modules),
    )


def generation_command(config: CalibrationConfig, *, pilot: bool) -> list[str]:
    """Build a pilot or full teacher-generation command."""
    # The pilot writes into the production data directory.  The full stage then
    # reuses those rows through --skip-existing instead of sampling them twice.
    output_dir = config.output_dir / "data"
    prompt_count = config.pilot_prompt_count if pilot else config.full_prompt_count
    generation_python = (
        os.environ.get("VLLM_PYTHON", sys.executable)
        if config.generation_backend == "vllm"
        else sys.executable
    )
    command = [
        generation_python,
        str(SCRIPT_DIR / "generate_trait_response_sft_dataset.py"),
        "--carrier",
        config.carrier,
        "--source-path",
        str(config.source_path),
        "--output-dir",
        str(output_dir),
        "--model-id",
        config.model_id,
        "--trait",
        config.trait,
        "--system-prompt",
        config.system_prompt,
        "--prompt-limit",
        str(prompt_count),
        "--projection-prompt-count",
        str(config.full_prompt_count),
        "--responses-per-prompt",
        str(config.responses_per_prompt),
        "--generation-backend",
        config.generation_backend,
        "--batch-size",
        str(config.generation_batch_size),
        "--max-auto-batch-size",
        str(config.generation_max_auto_batch_size),
        "--auto-batch-batches-per-size",
        str(config.generation_auto_batch_batches_per_size),
        "--memory-headroom-fraction",
        str(config.memory_headroom_fraction),
        "--max-seq-length",
        str(config.generation_max_seq_length),
        "--max-new-tokens",
        str(config.generation_max_new_tokens),
        "--temperature",
        str(config.temperature),
        "--top-p",
        str(config.top_p),
        "--seed",
        str(config.seed),
        "--skip-existing",
        "--enable-thinking" if config.enable_thinking else "--disable-thinking",
    ]
    if config.generation_backend in {"continuous", "vllm"}:
        command.extend(
            [
                "--continuous-chunk-size",
                str(config.generation_continuous_chunk_size),
            ]
        )
    if (
        config.generation_backend == "vllm"
        and config.generation_enable_prefix_caching
    ):
        command.append("--vllm-enable-prefix-caching")
    if config.generation_backend == "continuous":
        command.extend(
            [
                "--continuous-max-batch-tokens",
                str(config.generation_continuous_max_batch_tokens),
            ]
        )
    if config.generation_backend == "static":
        command.append("--sort-by-prompt-length")
    if not pilot and config.training_example_count is not None:
        command.extend(
            ["--stop-after-eligible", str(config.training_example_count)]
        )
    if config.source_url:
        command.extend(["--source-url", config.source_url])
    if config.load_in_4bit:
        command.append("--load-in-4bit")
    if config.generation_auto_batch_size and config.generation_backend == "static":
        command.append("--auto-batch-size")
    if config.sanitize_trait_paragraphs:
        command.extend(
            [
                "--sanitize-trait-sentences",
                "--sanitization-max-removed-fraction",
                str(config.sanitization_max_removed_fraction),
                "--sanitization-min-reasoning-chars",
                str(config.sanitization_min_reasoning_chars),
            ]
        )
    return command


def training_command(config: CalibrationConfig) -> list[str]:
    """Build the completion-only response-SFT command."""
    command = [
        sys.executable,
        str(SCRIPT_DIR / "train_trait_response_sft.py"),
        "--dataset-path",
        str(config.output_dir / "data" / "filtered.jsonl"),
        "--output-dir",
        str(config.output_dir / "training"),
        "--trait",
        config.trait,
        "--model-id",
        config.model_id,
        "--n-epochs",
        str(config.n_epochs),
        "--max-seq-length",
        str(config.training_max_seq_length),
        "--per-device-train-batch-size",
        str(config.per_device_train_batch_size),
        "--gradient-accumulation-steps",
        str(config.gradient_accumulation_steps),
        "--max-auto-batch-size",
        str(config.training_max_auto_batch_size),
        "--target-effective-batch-size",
        str(config.target_effective_batch_size),
        "--batch-probe-steps",
        str(config.training_batch_probe_steps),
        "--memory-headroom-fraction",
        str(config.memory_headroom_fraction),
        "--lr",
        str(config.lr),
        "--warmup-steps",
        str(config.warmup_steps),
        "--lora-r",
        str(config.lora_r),
        "--lora-alpha",
        str(config.lora_alpha),
        "--checkpoint-count",
        str(config.checkpoint_count),
        "--target-modules",
        ",".join(config.target_modules),
        "--disable-wandb",
        "--skip-existing",
    ]
    if config.training_example_count is not None:
        command.extend(
            [
                "--max-rows",
                str(config.training_example_count),
                "--min-rows",
                str(config.training_example_count),
            ]
        )
    if config.load_in_4bit:
        command.append("--load-in-4bit")
    if config.training_auto_batch_size:
        command.append("--auto-batch-size")
    return command


def m10_command(config: CalibrationConfig) -> list[str]:
    """Build checkpoint-by-checkpoint behavioral evaluation."""
    return [
        sys.executable,
        str(SCRIPT_DIR / "evaluate_trait_checkpoint_effect_matrices.py"),
        "--training-output-dir",
        str(config.output_dir / "training"),
        "--output-dir",
        str(config.output_dir / "m10"),
        "--answers-path",
        str(config.answers_path),
        "--questions-path",
        str(config.questions_path),
        "--traits",
        config.trait,
        "--base-model-id",
        config.model_id,
        "--batch-size",
        str(config.eval_batch_size),
        "--device",
        "auto",
        "--torch-dtype",
        "auto",
        "--skip-existing",
        "--enable-thinking" if config.enable_thinking else "--disable-thinking",
    ]


def saturation_command(config: CalibrationConfig) -> list[str]:
    """Build the behavior-versus-examples checkpoint analysis command."""
    return [
        sys.executable,
        str(SCRIPT_DIR / "analyze_long_form_checkpoint_saturation.py"),
        "--training-output-dir",
        str(config.output_dir / "training"),
        "--m10-output-dir",
        str(config.output_dir / "m10"),
        "--output-dir",
        str(config.output_dir / "saturation"),
        "--trait",
        config.trait,
    ]


def preflight(config: CalibrationConfig) -> dict[str, Any]:
    """Validate paths and report the intended workload without loading a model."""
    if config.carrier == "gsm8k" and not config.source_path.is_file():
        raise FileNotFoundError(f"Missing GSM8K source: {config.source_path}")
    for path in (config.answers_path, config.questions_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing evaluation input: {path}")
    if config.generation_max_new_tokens >= config.generation_max_seq_length:
        raise ValueError(
            "generation_max_new_tokens must be smaller than generation_max_seq_length."
        )
    report = {
        "name": config.name,
        "carrier": config.carrier,
        "trait": config.trait,
        "pilot_prompt_count": config.pilot_prompt_count,
        "full_prompt_count": config.full_prompt_count,
        "responses_per_prompt": config.responses_per_prompt,
        "pilot_generation_requests": (
            config.pilot_prompt_count * config.responses_per_prompt
        ),
        "full_generation_requests": (
            config.full_prompt_count * config.responses_per_prompt
        ),
        "n_epochs": config.n_epochs,
        "training_example_count": config.training_example_count,
        "checkpoint_count": config.checkpoint_count,
        "training_effective_batch_size": (
            config.per_device_train_batch_size * config.gradient_accumulation_steps
        ),
        "target_effective_batch_size": config.target_effective_batch_size,
        "generation_batch_autotuning": config.generation_auto_batch_size,
        "generation_backend": config.generation_backend,
        "generation_prefix_caching": config.generation_enable_prefix_caching,
        "generation_continuous_chunk_size": (
            config.generation_continuous_chunk_size
        ),
        "generation_continuous_max_batch_tokens": (
            config.generation_continuous_max_batch_tokens
        ),
        "training_batch_autotuning": config.training_auto_batch_size,
        "sanitization_policy": GSM8K_SANITIZATION_POLICY,
        "sanitize_trait_paragraphs": config.sanitize_trait_paragraphs,
        "sanitization_max_removed_fraction": (
            config.sanitization_max_removed_fraction
        ),
        "sanitization_min_reasoning_chars": (
            config.sanitization_min_reasoning_chars
        ),
    }
    logger.success(
        "Preflight passed: {} {} prompts × {} response(s), {} retained "
        "training rows, {} SFT epoch(s)",
        config.full_prompt_count,
        config.carrier,
        config.responses_per_prompt,
        config.training_example_count or "all eligible",
        config.n_epochs,
    )
    return report


def timing_path(config: CalibrationConfig) -> Path:
    """Return the persistent multi-invocation timing manifest path."""
    return config.output_dir / "pipeline_timing.json"


def wilson_score_interval(
    successes: int,
    total: int,
    *,
    z_score: float = 1.959963984540054,
) -> tuple[float, float]:
    """Return a two-sided Wilson interval for a binomial pass rate."""
    if total <= 0:
        return 0.0, 1.0
    if successes < 0 or successes > total:
        raise ValueError("successes must be between zero and total.")
    rate = successes / total
    z_squared = z_score**2
    denominator = 1.0 + z_squared / total
    center = (rate + z_squared / (2.0 * total)) / denominator
    margin = (
        z_score
        * math.sqrt(
            rate * (1.0 - rate) / total
            + z_squared / (4.0 * total**2)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def generation_quality_report(
    config: CalibrationConfig,
    *,
    phase: str,
) -> dict[str, Any]:
    """Summarize retained quality and measured time before the next stage."""
    manifest_path = config.output_dir / "data" / "generation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Generation quality manifest does not exist: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = manifest.get("filter_summary", {})
    raw_count = int(summary.get("raw_count", 0))
    eligible_count = int(summary.get("eligible_count", 0))
    pass_rate = eligible_count / raw_count if raw_count else 0.0
    pass_rate_lower, pass_rate_upper = wilson_score_interval(
        eligible_count,
        raw_count,
    )
    full_requests = config.full_prompt_count * config.responses_per_prompt
    projected_eligible = pass_rate * full_requests
    projected_eligible_lower = pass_rate_lower * full_requests
    projected_eligible_upper = pass_rate_upper * full_requests
    generation_seconds = float(
        manifest.get("timing_seconds", {}).get("generation_cumulative", 0.0)
    )
    completed_requests = int(manifest.get("completed_request_count", raw_count))
    seconds_per_request = (
        generation_seconds / completed_requests if completed_requests else None
    )
    optimized_full_seconds = manifest.get(
        "full_generation_projection", {}
    ).get("optimized_generation_seconds")
    optimized_seconds_per_request = (
        float(optimized_full_seconds) / full_requests
        if isinstance(optimized_full_seconds, (int, float)) and full_requests
        else None
    )
    projection_seconds_per_request = (
        optimized_seconds_per_request
        if optimized_seconds_per_request is not None
        else seconds_per_request
    )
    target_rows = config.training_example_count
    filtered_path = config.output_dir / "data" / "filtered.jsonl"
    filtered_rows: list[dict[str, Any]] = []
    if filtered_path.is_file():
        with filtered_path.open("r", encoding="utf-8") as handle:
            filtered_rows = [
                json.loads(line) for line in handle if line.strip()
            ]
    completion_lengths = sorted(
        len(str(row.get("completion", ""))) for row in filtered_rows
    )

    def completion_length_quantile(fraction: float) -> int | None:
        if not completion_lengths:
            return None
        index = round(fraction * (len(completion_lengths) - 1))
        return completion_lengths[index]

    audit_sample_path = config.output_dir / f"quality_sample_{phase}.jsonl"
    if filtered_rows:
        sample_count = min(20, len(filtered_rows))
        sample_indices = sorted(
            random.Random(config.seed + 101).sample(
                range(len(filtered_rows)),
                sample_count,
            )
        )
        with audit_sample_path.open("w", encoding="utf-8") as handle:
            for index in sample_indices:
                handle.write(
                    json.dumps(filtered_rows[index], ensure_ascii=False) + "\n"
                )
    projected_requests_for_target = None
    optimistic_requests_for_target = None
    conservative_requests_for_target = None
    projected_seconds_for_target = None
    if target_rows is not None and pass_rate > 0.0:
        projected_requests_for_target = min(
            full_requests,
            math.ceil(target_rows / pass_rate),
        )
        if pass_rate_upper > 0.0:
            optimistic_requests_for_target = min(
                full_requests,
                math.ceil(target_rows / pass_rate_upper),
            )
        if pass_rate_lower > 0.0:
            conservative_requests_for_target = min(
                full_requests,
                math.ceil(target_rows / pass_rate_lower),
            )
        if projection_seconds_per_request is not None:
            projected_seconds_for_target = (
                projected_requests_for_target * projection_seconds_per_request
            )
    optimistic_seconds_for_target = (
        optimistic_requests_for_target * projection_seconds_per_request
        if optimistic_requests_for_target is not None
        and projection_seconds_per_request is not None
        else None
    )
    conservative_seconds_for_target = (
        conservative_requests_for_target * projection_seconds_per_request
        if conservative_requests_for_target is not None
        and projection_seconds_per_request is not None
        else None
    )
    report = {
        "phase": phase,
        "created_at": utc_now(),
        "raw_count": raw_count,
        "eligible_count": eligible_count,
        "filter_pass_rate": pass_rate,
        "filter_pass_rate_95pct_interval": {
            "lower": pass_rate_lower,
            "upper": pass_rate_upper,
        },
        "rejection_counts": summary.get("rejection_counts", {}),
        "sanitization_counts": summary.get("sanitization_counts", {}),
        "full_generation_requests": full_requests,
        "projected_full_eligible_count": projected_eligible,
        "projected_full_eligible_count_95pct_interval": {
            "lower": projected_eligible_lower,
            "upper": projected_eligible_upper,
        },
        "training_row_target": target_rows,
        "projected_requests_for_target": projected_requests_for_target,
        "projected_requests_for_target_95pct_range": {
            "optimistic": optimistic_requests_for_target,
            "conservative": conservative_requests_for_target,
        },
        "measured_generation_seconds": generation_seconds,
        "measured_seconds_per_request": seconds_per_request,
        "optimized_seconds_per_request": optimized_seconds_per_request,
        "projection_seconds_per_request": projection_seconds_per_request,
        "projected_generation_seconds_for_target": projected_seconds_for_target,
        "projected_generation_hours_for_target": (
            projected_seconds_for_target / 3600.0
            if projected_seconds_for_target is not None
            else None
        ),
        "projected_generation_hours_for_target_95pct_range": {
            "optimistic": (
                optimistic_seconds_for_target / 3600.0
                if optimistic_seconds_for_target is not None
                else None
            ),
            "conservative": (
                conservative_seconds_for_target / 3600.0
                if conservative_seconds_for_target is not None
                else None
            ),
        },
        "projected_target_reachable": (
            projected_eligible >= target_rows
            if target_rows is not None
            else None
        ),
        "projected_target_reachable_at_95pct_lower_bound": (
            projected_eligible_lower >= target_rows
            if target_rows is not None
            else None
        ),
        "actual_target_reached": (
            len(filtered_rows) >= target_rows if target_rows is not None else None
        ),
        "filtered_dataset_diagnostics": {
            "row_count": len(filtered_rows),
            "unique_source_count": len(
                {str(row.get("source_id")) for row in filtered_rows}
            ),
            "unique_completion_count": len(
                {str(row.get("completion")) for row in filtered_rows}
            ),
            "completion_character_quantiles": {
                "p50": completion_length_quantile(0.50),
                "p90": completion_length_quantile(0.90),
                "p99": completion_length_quantile(0.99),
            },
            "audit_sample_path": (
                str(audit_sample_path) if filtered_rows else None
            ),
        },
    }
    output_path = config.output_dir / f"quality_report_{phase}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "{} quality: {}/{} reasonable rows ({:.1%}); projected full yield "
        "{:.0f}; target {}",
        phase,
        eligible_count,
        raw_count,
        pass_rate,
        projected_eligible,
        target_rows or "all eligible",
    )
    if projected_seconds_for_target is not None:
        logger.info(
            "Measured generation rate projects {:.2f} hours to {} retained rows",
            projected_seconds_for_target / 3600.0,
            target_rows,
        )
    return report


def require_projected_training_rows(
    report: dict[str, Any],
) -> None:
    """Stop when the pilot's conservative yield cannot support the target."""
    target = report.get("training_row_target")
    if target is None:
        return
    interval = report.get("projected_full_eligible_count_95pct_interval", {})
    projected = float(interval.get("lower", 0.0))
    if projected < int(target):
        raise ValueError(
            "The pilot's 95% lower yield bound projects only "
            f"{projected:.0f} reasonable rows from the full workload, below "
            f"the required {target}. Review the quality report before spending "
            "time on full generation."
        )


def require_actual_training_rows(config: CalibrationConfig) -> None:
    """Refuse SFT until the filtered dataset reaches its exact row budget."""
    if config.training_example_count is None:
        return
    filtered_path = config.output_dir / "data" / "filtered.jsonl"
    if not filtered_path.is_file():
        raise FileNotFoundError(f"Filtered SFT dataset does not exist: {filtered_path}")
    with filtered_path.open("r", encoding="utf-8") as handle:
        row_count = sum(1 for line in handle if line.strip())
    if row_count < config.training_example_count:
        raise ValueError(
            f"Filtered GSM8K dataset has {row_count} reasonable rows, below the "
            f"required {config.training_example_count}; refusing to start SFT."
        )
    logger.success(
        "Quality gate passed: {} filtered rows are ready for SFT",
        row_count,
    )


def update_timing_manifest(
    config: CalibrationConfig,
    *,
    preflight_report: dict[str, Any],
    stage_record: dict[str, Any] | None,
) -> None:
    """Merge a completed stage into the persistent timing manifest."""
    path = timing_path(config)
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = {
            "name": config.name,
            "created_at": utc_now(),
            "completed_stages": [],
            "stage_invocations": [],
            "stage_duration_totals_seconds": {},
            "successful_stage_duration_totals_seconds": {},
        }
    payload["updated_at"] = utc_now()
    payload["preflight"] = preflight_report
    if stage_record is not None:
        invocations = payload.get("stage_invocations", [])
        invocations.append(stage_record)
        payload["stage_invocations"] = invocations
        all_totals: dict[str, float] = {}
        successful_totals: dict[str, float] = {}
        for invocation in invocations:
            stage = str(invocation["stage"])
            duration = float(invocation["duration_seconds"])
            all_totals[stage] = all_totals.get(stage, 0.0) + duration
            if int(invocation.get("return_code", 1)) == 0:
                successful_totals[stage] = successful_totals.get(stage, 0.0) + duration
        payload["stage_duration_totals_seconds"] = all_totals
        payload["successful_stage_duration_totals_seconds"] = successful_totals
        existing = [
            row
            for row in payload.get("completed_stages", [])
            if row.get("stage") != stage_record["stage"]
        ]
        existing.append(stage_record)
        payload["completed_stages"] = existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_command(
    config: CalibrationConfig,
    *,
    stage: str,
    command: list[str],
    execute: bool,
    preflight_report: dict[str, Any],
) -> None:
    """Log and optionally execute one timed pipeline subprocess."""
    logger.info("{} command: {}", stage, shlex.join(command))
    if not execute:
        return
    started_at = utc_now()
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    duration = time.perf_counter() - started
    record = {
        "stage": stage,
        "started_at": started_at,
        "finished_at": utc_now(),
        "duration_seconds": duration,
        "return_code": completed.returncode,
        "command": command,
    }
    update_timing_manifest(
        config,
        preflight_report=preflight_report,
        stage_record=record,
    )
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)
    logger.success("Stage {} completed in {:.1f} seconds", stage, duration)


def selected_stages(stage: Stage) -> list[str]:
    """Expand the all alias while keeping pilot opt-in."""
    if stage == "all":
        return ["generate", "train", "m10", "saturation"]
    if stage == "calibrated-all":
        return ["pilot", "generate", "train", "m10", "saturation"]
    if stage == "calibrated-train":
        return ["pilot", "generate", "train"]
    if stage == "preflight":
        return []
    return [stage]


def run_pipeline(config: CalibrationConfig, *, stage: Stage, execute: bool) -> None:
    """Run the requested calibration stages."""
    report = preflight(config)
    update_timing_manifest(config, preflight_report=report, stage_record=None)
    commands = {
        "pilot": generation_command(config, pilot=True),
        "generate": generation_command(config, pilot=False),
        "train": training_command(config),
        "m10": m10_command(config),
        "saturation": saturation_command(config),
    }
    for selected in selected_stages(stage):
        if execute and selected == "train":
            require_actual_training_rows(config)
        run_command(
            config,
            stage=selected,
            command=commands[selected],
            execute=execute,
            preflight_report=report,
        )
        if execute and selected in {"pilot", "generate"}:
            quality = generation_quality_report(config, phase=selected)
            if selected == "pilot" and stage in {
                "calibrated-all",
                "calibrated-train",
            }:
                require_projected_training_rows(quality)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Run a resumable Qwen long-form carrier timing calibration."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=[
            "preflight",
            "pilot",
            "generate",
            "train",
            "m10",
            "saturation",
            "all",
            "calibrated-all",
            "calibrated-train",
        ],
        default="preflight",
    )
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    """Run the configured pipeline."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        config_path = (
            args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
        )
        run_pipeline(
            load_config(config_path),
            stage=args.stage,
            execute=args.execute,
        )
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
