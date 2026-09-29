#!/usr/bin/env python3
"""Evaluate favorite-color log-ratio matrices across LoRA checkpoints."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
import sys
from collections.abc import MutableMapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compute_favorite_color_probability_ratio_matrix import (  # noqa: E402
    DEFAULT_COLORS,
    DEFAULT_QUESTIONS,
    RatioMatrixConfig,
    build_ratio_matrices,
    log_matrix,
    score_model,
    write_csv_matrix,
    write_heatmap,
    write_json,
)
from find_owl_output_distribution_entanglement import (  # noqa: E402
    DEFAULT_MODEL_ID,
    load_direct_model_and_tokenizer,
    parse_csv_values,
)
from score_birthday_year_probabilities import (  # noqa: E402
    resolve_device,
    resolve_torch_dtype,
)
from score_string_trait_number_distribution import slugify  # noqa: E402


DEFAULT_TRAINING_OUTPUT_DIR = Path(
    "results/logits_finetuned_color_students_10k_3ep"
)
DEFAULT_CHECKPOINT_STEPS = [80, 160, 240, 320]
DEFAULT_OUTPUT_SUBDIR = "checkpoint_probability_ratio_matrices"
DEFAULT_OUTPUT_STEM = "favorite_color_probability_ratio_matrix"
RANK_METRICS_STEM = "favorite_color_probability_ratio_rank_metrics"
SVD_COORDINATES_STEM = (
    "favorite_color_probability_ratio_double_centered_svd"
)
RAW_LOGPROB_SVD_COORDINATES_STEM = (
    "favorite_color_raw_logprob_shift_double_centered_svd"
)
FAVORITE_COLOR_PLOT_COLORS = {
    "red": "#d62728",
    "blue": "#1f77b4",
    "yellow": "#f2c80f",
    "green": "#2ca02c",
    "orange": "#ff7f0e",
    "purple": "#9467bd",
    "pink": "#e377c2",
    "black": "#111111",
    "white": "#ffffff",
    "brown": "#8c564b",
    "teal": "#008b8b",
    "amber": "#ffbf00",
    "indigo": "#4b0082",
    "ochre": "#cc7722",
    "mauve": "#b784a7",
    "cerulean": "#007ba7",
    "saffron": "#f4c430",
    "vermilion": "#e34234",
}
BROAD_BOOST_CORRELATION_STEM = "favorite_color_broad_boost_correlations"
BROAD_BOOST_TRAIT_VALUES_STEM = "favorite_color_broad_boost_trait_values"
BROAD_BOOST_ANALYSIS_VERSION = 1
UPDATE_READOUT_PROJECTION_STEM = "favorite_color_update_readout_projection"
UPDATE_READOUT_PROJECTION_ANALYSIS_VERSION = 1
DEFAULT_PROJECTION_FINITE_DIFFERENCE_STEP = 0.5
BAYESIAN_DECODER_STEM = "favorite_color_bayesian_decoder"
BAYESIAN_DECODER_ANALYSIS_VERSION = 1
DEFAULT_BAYESIAN_DECODER_BETAS = [
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
    300.0,
    1000.0,
    3000.0,
    10000.0,
]
DEFAULT_BAYESIAN_DECODER_KEY_COMPETITIONS = [
    ("vermilion", "red"),
    ("pink", "purple"),
    ("saffron", "yellow"),
    ("saffron", "orange"),
]
DEFAULT_TEACHER_LOGPROB_CACHE_DIR = Path(
    "results/teacher_logits_color_fisher/condition_logprob_cache"
)
CHECKPOINT_PATTERN = re.compile(r"^checkpoint-step-(\d+)$")
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(**DATACLASS_KW_ONLY)
class CheckpointEvaluationConfig:
    """Configuration for multi-checkpoint favorite-color evaluation."""

    training_output_dir: Path
    output_dir: Path
    checkpoint_steps: list[int]
    include_final: bool
    colors: list[str]
    traits: list[str]
    questions: list[str]
    base_model_id: str
    probability_mode: str
    epsilon: float
    candidate_prefix: str
    assistant_prefill: str | None
    batch_size: int
    device: str
    torch_dtype: str
    teacher_logprob_cache_dir: Path
    compute_update_readout_projection: bool
    projection_finite_difference_step: float
    skip_existing: bool
    print_log_ratio_tables: bool


@dataclass(**DATACLASS_KW_ONLY)
class EvaluationStage:
    """One checkpoint step or final-adapter evaluation stage."""

    label: str
    checkpoint_step: int | None
    adapter_paths: dict[str, Path]


@dataclass(**DATACLASS_KW_ONLY)
class LoraScalingEntry:
    """One mutable PEFT LoRA scaling value for an adapter."""

    module: Any
    adapter_name: str
    base_scaling: float


def parse_optional_text(value: str | None) -> str | None:
    """Treat an empty string as omitted text."""
    return None if value is None or value == "" else value


def parse_checkpoint_steps(
    raw_values: list[str] | None,
    default: list[int],
) -> list[int]:
    """Parse repeated or comma-separated positive checkpoint steps."""
    if not raw_values:
        return list(default)
    steps = []
    seen = set()
    for raw_value in raw_values:
        for piece in raw_value.split(","):
            piece = piece.strip()
            if not piece:
                continue
            try:
                step = int(piece)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid checkpoint step {piece!r}; expected an integer."
                ) from exc
            if step <= 0:
                raise ValueError("Checkpoint steps must be positive.")
            if step not in seen:
                steps.append(step)
                seen.add(step)
    if not steps:
        raise ValueError("At least one checkpoint step is required.")
    return sorted(steps)


def load_questions(args: argparse.Namespace) -> list[str]:
    """Load evaluation questions from CLI values, JSON, or defaults."""
    if args.questions_path is not None:
        data = json.loads(args.questions_path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not all(
            isinstance(item, str) for item in data
        ):
            raise ValueError("--questions-path must contain a JSON list of strings.")
        return data
    return parse_csv_values(args.questions, DEFAULT_QUESTIONS)


def discover_common_checkpoint_steps(
    training_output_dir: Path,
    traits: list[str],
) -> list[int]:
    """Return checkpoint steps present for every requested trait."""
    common_steps: set[int] | None = None
    for trait in traits:
        trainer_dir = training_output_dir / "trainer" / slugify(trait)
        trait_steps = set()
        if trainer_dir.is_dir():
            for path in trainer_dir.iterdir():
                match = CHECKPOINT_PATTERN.match(path.name)
                if match is not None and (path / "adapter").is_dir():
                    trait_steps.add(int(match.group(1)))
        common_steps = (
            trait_steps
            if common_steps is None
            else common_steps.intersection(trait_steps)
        )
    return sorted(common_steps or set())


def adapter_path_for_stage(
    config: CheckpointEvaluationConfig,
    trait: str,
    checkpoint_step: int | None,
) -> Path:
    """Resolve one trait adapter path for a checkpoint or final stage."""
    safe_trait = slugify(trait)
    if checkpoint_step is None:
        return config.training_output_dir / "adapters" / safe_trait
    return (
        config.training_output_dir
        / "trainer"
        / safe_trait
        / f"checkpoint-step-{checkpoint_step:06d}"
        / "adapter"
    )


def build_evaluation_stages(
    config: CheckpointEvaluationConfig,
) -> list[EvaluationStage]:
    """Build and validate all requested evaluation stages."""
    stages = []
    for checkpoint_step in config.checkpoint_steps:
        stages.append(
            EvaluationStage(
                label=f"step_{checkpoint_step:06d}",
                checkpoint_step=checkpoint_step,
                adapter_paths={
                    trait: adapter_path_for_stage(config, trait, checkpoint_step)
                    for trait in config.traits
                },
            )
        )
    if config.include_final:
        stages.append(
            EvaluationStage(
                label="final",
                checkpoint_step=None,
                adapter_paths={
                    trait: adapter_path_for_stage(config, trait, None)
                    for trait in config.traits
                },
            )
        )
    if not stages:
        raise ValueError("Select at least one checkpoint or the final adapters.")

    missing_paths = [
        path
        for stage in stages
        for path in stage.adapter_paths.values()
        if not (path / "adapter_config.json").is_file()
    ]
    if missing_paths:
        formatted = "\n".join(f"  {path}" for path in missing_paths[:20])
        suffix = "\n  ..." if len(missing_paths) > 20 else ""
        raise FileNotFoundError(
            "Missing checkpoint adapter directories:\n" + formatted + suffix
        )
    return stages


def stage_output_paths(
    config: CheckpointEvaluationConfig,
    stage: EvaluationStage,
) -> dict[str, Path]:
    """Return raw and double-centered output paths for one stage."""
    stem = f"{DEFAULT_OUTPUT_STEM}_{stage.label}"
    prefix = config.output_dir / stem
    return {
        "json": prefix.with_suffix(".json"),
        "csv": prefix.with_suffix(".csv"),
        "heatmap": prefix.with_suffix(".png"),
        "raw_logprob_json": prefix.with_name(
            f"{prefix.name}_raw_logprob_shift.json"
        ),
        "raw_logprob_csv": prefix.with_name(
            f"{prefix.name}_raw_logprob_shift.csv"
        ),
        "raw_logprob_heatmap": prefix.with_name(
            f"{prefix.name}_raw_logprob_shift.png"
        ),
        "double_centered_json": prefix.with_name(
            f"{prefix.name}_double_centered.json"
        ),
        "double_centered_csv": prefix.with_name(
            f"{prefix.name}_double_centered.csv"
        ),
        "double_centered_heatmap": prefix.with_name(
            f"{prefix.name}_double_centered.png"
        ),
    }


def broad_boost_output_paths(
    config: CheckpointEvaluationConfig,
) -> dict[str, Path]:
    """Return paths for the broad-boost correlation analysis."""
    correlation_prefix = config.output_dir / BROAD_BOOST_CORRELATION_STEM
    trait_values_prefix = config.output_dir / BROAD_BOOST_TRAIT_VALUES_STEM
    return {
        "json": correlation_prefix.with_suffix(".json"),
        "csv": correlation_prefix.with_suffix(".csv"),
        "trait_values_csv": trait_values_prefix.with_suffix(".csv"),
    }


def update_readout_projection_output_paths(
    config: CheckpointEvaluationConfig,
) -> dict[str, Path]:
    """Return paths for update/readout projection outputs."""
    prefix = config.output_dir / UPDATE_READOUT_PROJECTION_STEM
    return {
        "json": prefix.with_suffix(".json"),
        "matrix_csv": prefix.with_name(f"{prefix.name}_matrix.csv"),
        "comparison_csv": prefix.with_name(
            f"{prefix.name}_behavior_comparison.csv"
        ),
        "trait_summary_csv": prefix.with_name(
            f"{prefix.name}_trait_summary.csv"
        ),
        "heatmap": prefix.with_suffix(".png"),
        "scatter": prefix.with_name(f"{prefix.name}_behavior_scatter.png"),
    }


def bayesian_decoder_output_paths(
    config: CheckpointEvaluationConfig,
) -> dict[str, Path]:
    """Return paths for Bayesian teacher-trait decoder outputs."""
    prefix = config.output_dir / BAYESIAN_DECODER_STEM
    return {
        "json": prefix.with_suffix(".json"),
        "scores_csv": prefix.with_name(f"{prefix.name}_scores.csv"),
        "posterior_sweep_csv": prefix.with_name(
            f"{prefix.name}_posterior_sweep.csv"
        ),
        "critical_beta_csv": prefix.with_name(
            f"{prefix.name}_critical_beta.csv"
        ),
        "uniform_confusion_csv": prefix.with_name(
            f"{prefix.name}_uniform_principled_confusion.csv"
        ),
        "base_prior_confusion_csv": prefix.with_name(
            f"{prefix.name}_base_prior_principled_confusion.csv"
        ),
        "principled_confusion_plot": prefix.with_name(
            f"{prefix.name}_principled_confusion.png"
        ),
        "critical_beta_plot": prefix.with_name(
            f"{prefix.name}_critical_beta.png"
        ),
    }


def stage_result_is_reusable(
    config: CheckpointEvaluationConfig,
    stage: EvaluationStage,
) -> bool:
    """Return whether durable non-plot outputs exist for one stage."""
    paths = stage_output_paths(config, stage)
    return paths["json"].is_file() and paths["csv"].is_file()


def stage_cache_metadata(
    config: CheckpointEvaluationConfig,
    stage: EvaluationStage,
) -> dict[str, Any]:
    """Return metadata that identifies one reusable stage evaluation."""
    return {
        "label": stage.label,
        "checkpoint_step": stage.checkpoint_step,
        "adapter_paths": {
            trait: str(path) for trait, path in stage.adapter_paths.items()
        },
        "base_model_id": config.base_model_id,
        "colors": config.colors,
        "traits": config.traits,
        "questions": config.questions,
        "probability_mode": config.probability_mode,
        "epsilon": config.epsilon,
        "candidate_prefix": config.candidate_prefix,
        "assistant_prefill": config.assistant_prefill,
        "torch_dtype": config.torch_dtype,
    }


def load_stage_result(
    path: Path,
    config: CheckpointEvaluationConfig,
    stage: EvaluationStage,
) -> dict[str, Any] | None:
    """Load a completed stage result when its identity matches."""
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable stage result {}: {}", path, exc)
        return None
    if result.get("evaluation_metadata") != stage_cache_metadata(config, stage):
        logger.warning("Ignoring mismatched stage result {}", path)
        return None
    if "log_ratio_matrix" not in result:
        logger.warning("Ignoring incomplete stage result {}", path)
        return None
    return result


def ratio_config_for_stage(
    config: CheckpointEvaluationConfig,
    stage: EvaluationStage,
) -> RatioMatrixConfig:
    """Build the existing single-matrix scorer config for one stage."""
    paths = stage_output_paths(config, stage)
    return RatioMatrixConfig(
        colors=config.colors,
        fine_tune_traits=config.traits,
        questions=config.questions,
        model_json_template="unused_{trait}.json",
        base_model_id=config.base_model_id,
        output_path=paths["json"],
        csv_output_path=paths["csv"],
        heatmap_path=paths["heatmap"],
        probability_mode=config.probability_mode,
        epsilon=config.epsilon,
        candidate_prefix=config.candidate_prefix,
        assistant_prefill=config.assistant_prefill,
        batch_size=config.batch_size,
        device=config.device,
        torch_dtype=config.torch_dtype,
    )


def base_cache_metadata(config: CheckpointEvaluationConfig) -> dict[str, Any]:
    """Return metadata that identifies reusable base-model probe scores."""
    return {
        "base_model_id": config.base_model_id,
        "colors": config.colors,
        "questions": config.questions,
        "probability_mode": config.probability_mode,
        "candidate_prefix": config.candidate_prefix,
        "assistant_prefill": config.assistant_prefill,
        "torch_dtype": config.torch_dtype,
    }


def load_base_score_cache(
    path: Path,
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    """Load base scores when cache metadata matches exactly."""
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable base score cache {}: {}", path, exc)
        return None
    if payload.get("metadata") != metadata:
        logger.info("Ignoring stale base score cache {}", path)
        return None
    base_scores = payload.get("base_scores")
    if not isinstance(base_scores, dict):
        return None
    logger.info("Loaded cached base-model favorite-color scores from {}", path)
    return base_scores


def save_base_score_cache(
    path: Path,
    metadata: dict[str, Any],
    base_scores: dict[str, Any],
) -> None:
    """Save reusable base-model probe scores."""
    write_json(path, {"metadata": metadata, "base_scores": base_scores})


def adapter_name(stage: EvaluationStage, trait_index: int, trait: str) -> str:
    """Build a unique PEFT adapter name for one stage and trait."""
    return f"eval_{stage.label}_{trait_index:02d}_{slugify(trait)}"


def validate_loaded_lora_adapter(
    model: Any,
    adapter_name_value: str,
    adapter_path: Path,
) -> None:
    """Fail if a loaded LoRA has no nonzero B weights under its adapter name."""
    marker = f"lora_B.{adapter_name_value}"
    matching = [
        parameter
        for name, parameter in model.named_parameters()
        if marker in name
    ]
    if not matching:
        raise RuntimeError(
            f"Adapter {adapter_name_value} from {adapter_path} did not create "
            "any matching LoRA-B parameters. The evaluation model architecture "
            "does not match the training architecture."
        )
    total_norm = sum(
        float(parameter.detach().float().norm().item())
        for parameter in matching
    )
    if total_norm == 0.0:
        raise RuntimeError(
            f"Adapter {adapter_name_value} from {adapter_path} has zero loaded "
            "LoRA-B norm. Its checkpoint weights were not applied; refusing "
            "to score a false zero-effect matrix."
        )
    logger.info(
        "Verified adapter {}: {} LoRA-B tensors, total norm {:.6g}",
        adapter_name_value,
        len(matching),
        total_norm,
    )


def load_stage_adapters(
    *,
    model: Any,
    model_is_peft: bool,
    stage: EvaluationStage,
    traits: list[str],
) -> tuple[Any, bool, list[str]]:
    """Attach one stage's adapters to a shared base/PEFT model."""
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError("Checkpoint evaluation requires peft.") from exc

    names = [
        adapter_name(stage, index, trait)
        for index, trait in enumerate(traits)
    ]
    first_trait = traits[0]
    if not model_is_peft:
        logger.info(
            "Loading {} adapter for {} from {}",
            stage.label,
            first_trait,
            stage.adapter_paths[first_trait],
        )
        model = PeftModel.from_pretrained(
            model,
            str(stage.adapter_paths[first_trait]),
            adapter_name=names[0],
            is_trainable=False,
        )
        validate_loaded_lora_adapter(
            model,
            names[0],
            stage.adapter_paths[first_trait],
        )
        start_index = 1
        model_is_peft = True
    else:
        start_index = 0

    for index in range(start_index, len(traits)):
        trait = traits[index]
        logger.info(
            "Loading {} adapter for {} from {}",
            stage.label,
            trait,
            stage.adapter_paths[trait],
        )
        model.load_adapter(
            str(stage.adapter_paths[trait]),
            adapter_name=names[index],
            is_trainable=False,
        )
        validate_loaded_lora_adapter(
            model,
            names[index],
            stage.adapter_paths[trait],
        )
    model.eval()
    return model, model_is_peft, names


def unload_stage_adapters(model: Any, adapter_names: list[str]) -> None:
    """Remove stage adapters when supported to bound accelerator memory."""
    delete_adapter = getattr(model, "delete_adapter", None)
    if delete_adapter is None:
        logger.warning(
            "PEFT model cannot delete adapters; loaded adapters remain resident."
        )
        return
    for name in reversed(adapter_names):
        try:
            delete_adapter(name)
        except Exception as exc:
            logger.warning("Could not unload adapter {}: {}", name, exc)


def score_evaluation_stage(
    *,
    config: CheckpointEvaluationConfig,
    stage: EvaluationStage,
    ratio_config: RatioMatrixConfig,
    base_scores: dict[str, Any],
    model: Any,
    tokenizer: Any,
    adapter_names: list[str],
    device: Any,
    score_model_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score all trait adapters for one checkpoint stage."""
    scorer = score_model if score_model_fn is None else score_model_fn
    fine_tuned_scores: dict[str, dict[str, Any]] = {}
    fine_tuned_probabilities: dict[str, dict[str, float]] = {}
    for trait, active_adapter_name in zip(config.traits, adapter_names):
        logger.info(
            "Evaluating {} trait {} with adapter {}",
            stage.label,
            trait,
            active_adapter_name,
        )
        model.set_adapter(active_adapter_name)
        score = scorer(
            model_label=f"{stage.label}_{trait}",
            model=model,
            tokenizer=tokenizer,
            device=device,
            config=ratio_config,
        )
        score["adapter_path"] = str(stage.adapter_paths[trait])
        score["adapter_name"] = active_adapter_name
        fine_tuned_scores[trait] = score
        fine_tuned_probabilities[trait] = score["mean_probability_by_color"]

    matrices = build_ratio_matrices(
        base_probabilities=base_scores["mean_probability_by_color"],
        fine_tuned_probabilities=fine_tuned_probabilities,
        colors=config.colors,
        fine_tune_traits=config.traits,
        epsilon=config.epsilon,
    )
    return {
        "config": {
            **asdict(config),
            "training_output_dir": str(config.training_output_dir),
            "output_dir": str(config.output_dir),
        },
        "evaluation_stage": {
            "label": stage.label,
            "checkpoint_step": stage.checkpoint_step,
            "adapter_paths": {
                trait: str(path) for trait, path in stage.adapter_paths.items()
            },
        },
        "evaluation_metadata": stage_cache_metadata(config, stage),
        "base_model_id": config.base_model_id,
        "base_scores": base_scores,
        "fine_tuned_scores": fine_tuned_scores,
        **matrices,
    }


def write_stage_tables(
    *,
    config: CheckpointEvaluationConfig,
    stage: EvaluationStage,
    result: dict[str, Any],
) -> None:
    """Write durable JSON and CSV outputs immediately after one stage."""
    paths = stage_output_paths(config, stage)
    write_json(paths["json"], result)
    write_csv_matrix(
        matrix=result["log_ratio_matrix"],
        rows=config.colors,
        columns=config.traits,
        path=paths["csv"],
    )


def mean_raw_logprob_by_color(
    score: dict[str, Any],
    colors: list[str],
) -> dict[str, float]:
    """Average raw continuation log-probability over evaluation questions."""
    question_scores = score.get("question_scores")
    if not isinstance(question_scores, list) or not question_scores:
        raise ValueError("Projection score has no per-question results.")
    result = {}
    for color in colors:
        try:
            values = [
                float(question_score["colors"][color]["total_logprob"])
                for question_score in question_scores
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Projection score is missing raw log-probability for {color}."
            ) from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Projection log-probabilities must be finite.")
        result[color] = float(np.mean(values))
    return result


def find_lora_scaling_entries(
    model: Any,
    adapter_name: str,
) -> list[LoraScalingEntry]:
    """Find every PEFT layer scaling value for one loaded adapter."""
    entries = []
    for module in model.modules():
        scaling = getattr(module, "scaling", None)
        if scaling is None:
            continue
        if not isinstance(scaling, MutableMapping):
            continue
        if adapter_name not in scaling:
            continue
        value = scaling[adapter_name]
        try:
            base_scaling = float(value.item())
        except AttributeError:
            base_scaling = float(value)
        if not math.isfinite(base_scaling):
            raise ValueError("LoRA adapter scaling must be finite.")
        entries.append(
            LoraScalingEntry(
                module=module,
                adapter_name=adapter_name,
                base_scaling=base_scaling,
            )
        )
    if not entries:
        raise ValueError(
            f"No LoRA scaling entries found for adapter {adapter_name}."
        )
    return entries


def set_lora_scaling_multiplier(
    entries: list[LoraScalingEntry],
    multiplier: float,
) -> None:
    """Set one adapter's effective update multiplier on every LoRA layer."""
    if not math.isfinite(multiplier):
        raise ValueError("LoRA scaling multiplier must be finite.")
    for entry in entries:
        entry.module.scaling[entry.adapter_name] = (
            entry.base_scaling * multiplier
        )


def estimate_update_readout_projection(
    *,
    config: CheckpointEvaluationConfig,
    stage: EvaluationStage,
    model: Any,
    tokenizer: Any,
    adapter_names: list[str],
    device: Any,
) -> dict[str, Any]:
    """Estimate <Delta theta_tau, grad log P(u)> by centered differences."""
    step = config.projection_finite_difference_step
    ratio_config = ratio_config_for_stage(config, stage)
    projection_matrix = {color: {} for color in config.colors}
    positive_logprob_matrix = {color: {} for color in config.colors}
    negative_logprob_matrix = {color: {} for color in config.colors}
    lora_layer_counts = {}

    for trait, active_adapter_name in zip(config.traits, adapter_names):
        logger.info(
            "Estimating update/readout projection for {} with h={}",
            trait,
            step,
        )
        model.set_adapter(active_adapter_name)
        entries = find_lora_scaling_entries(model, active_adapter_name)
        lora_layer_counts[trait] = len(entries)
        try:
            set_lora_scaling_multiplier(entries, step)
            positive_score = score_model(
                model_label=f"projection_{trait}_plus_{step:g}",
                model=model,
                tokenizer=tokenizer,
                device=device,
                config=ratio_config,
            )
            set_lora_scaling_multiplier(entries, -step)
            negative_score = score_model(
                model_label=f"projection_{trait}_minus_{step:g}",
                model=model,
                tokenizer=tokenizer,
                device=device,
                config=ratio_config,
            )
        finally:
            set_lora_scaling_multiplier(entries, 1.0)

        positive = mean_raw_logprob_by_color(positive_score, config.colors)
        negative = mean_raw_logprob_by_color(negative_score, config.colors)
        for color in config.colors:
            positive_logprob_matrix[color][trait] = positive[color]
            negative_logprob_matrix[color][trait] = negative[color]
            projection_matrix[color][trait] = (
                positive[color] - negative[color]
            ) / (2.0 * step)

    return {
        "projection_matrix": projection_matrix,
        "positive_mean_raw_logprob_matrix": positive_logprob_matrix,
        "negative_mean_raw_logprob_matrix": negative_logprob_matrix,
        "lora_layer_count_by_trait": lora_layer_counts,
    }


def log_probability_with_epsilon(log_probability: float, epsilon: float) -> float:
    """Return log(exp(log_probability) + epsilon) stably."""
    value = float(log_probability)
    if not math.isfinite(value):
        raise ValueError("Cached log-probabilities must be finite.")
    if epsilon == 0.0:
        return value
    return float(np.logaddexp(value, math.log(epsilon)))


def cached_question_color_log_probability(
    question_score: dict[str, Any],
    color: str,
    *,
    probability_mode: str,
    epsilon: float,
) -> float:
    """Read one raw or color-normalized cached log probability."""
    try:
        color_score = question_score["colors"][color]
        if probability_mode == "raw":
            return log_probability_with_epsilon(
                float(color_score["total_logprob"]),
                epsilon,
            )
        if probability_mode == "normalized":
            probability = float(
                color_score["normalized_probability_over_colors"]
            )
            if probability < 0.0 or not math.isfinite(probability):
                raise ValueError(
                    "Normalized probabilities must be finite and nonnegative."
                )
            return math.log(probability + epsilon)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Cached score is missing {probability_mode} probability for {color}."
        ) from exc
    raise ValueError(f"Unsupported probability mode: {probability_mode}")


def cached_question_mean_log_ratio_matrix(
    *,
    stage_result: dict[str, Any],
    colors: list[str],
    traits: list[str],
    probability_mode: str,
    epsilon: float,
) -> dict[str, dict[str, float]]:
    """Average cached question-level fine-tuned/base log ratios."""
    base_questions = stage_result["base_scores"].get("question_scores")
    if not isinstance(base_questions, list) or not base_questions:
        raise ValueError("Cached base score has no per-question results.")

    matrix: dict[str, dict[str, float]] = {color: {} for color in colors}
    for trait in traits:
        fine_tuned_questions = stage_result["fine_tuned_scores"][trait].get(
            "question_scores"
        )
        if not isinstance(fine_tuned_questions, list) or not fine_tuned_questions:
            raise ValueError(
                f"Cached fine-tuned score for {trait} has no per-question results."
            )
        if len(fine_tuned_questions) != len(base_questions):
            raise ValueError(
                f"Base and fine-tuned question counts differ for {trait}."
            )
        for color in colors:
            shifts = [
                cached_question_color_log_probability(
                    fine_tuned_question,
                    color,
                    probability_mode=probability_mode,
                    epsilon=epsilon,
                )
                - cached_question_color_log_probability(
                    base_question,
                    color,
                    probability_mode=probability_mode,
                    epsilon=epsilon,
                )
                for base_question, fine_tuned_question in zip(
                    base_questions,
                    fine_tuned_questions,
                )
            ]
            matrix[color][trait] = float(np.mean(shifts))
    return matrix


def build_raw_logprob_shift_result(
    *,
    stage: EvaluationStage,
    stage_result: dict[str, Any],
    colors: list[str],
    traits: list[str],
    epsilon: float,
) -> dict[str, Any]:
    """Build raw continuation log-probability shifts from cached scores."""
    shift_matrix = cached_question_mean_log_ratio_matrix(
        stage_result=stage_result,
        colors=colors,
        traits=traits,
        probability_mode="raw",
        epsilon=epsilon,
    )
    return {
        "stage": stage.label,
        "checkpoint_step": stage.checkpoint_step,
        "aggregation": (
            "mean_over_questions(log((P_fine_tuned + epsilon) / "
            "(P_base + epsilon)))"
        ),
        "normalization": "none; probabilities are not renormalized over colors",
        "epsilon": epsilon,
        "rows": colors,
        "columns": traits,
        "raw_logprob_shift_matrix": shift_matrix,
    }


def build_raw_logprob_shift_results(
    *,
    stages: list[EvaluationStage],
    stage_results: dict[str, dict[str, Any]],
    colors: list[str],
    traits: list[str],
    epsilon: float,
) -> dict[str, dict[str, Any]]:
    """Build cached raw log-probability shifts for every stage."""
    return {
        stage.label: build_raw_logprob_shift_result(
            stage=stage,
            stage_result=stage_results[stage.label],
            colors=colors,
            traits=traits,
            epsilon=epsilon,
        )
        for stage in stages
    }


def write_raw_logprob_shift_outputs(
    *,
    config: CheckpointEvaluationConfig,
    stages: list[EvaluationStage],
    results: dict[str, dict[str, Any]],
) -> float:
    """Write raw log-probability shifts with one shared heatmap scale."""
    max_abs = max(
        (
            abs(float(value))
            for result in results.values()
            for row in result["raw_logprob_shift_matrix"].values()
            for value in row.values()
        ),
        default=1.0,
    )
    if max_abs <= 0.0 or not math.isfinite(max_abs):
        max_abs = 1.0

    for stage in stages:
        result = results[stage.label]
        matrix = result["raw_logprob_shift_matrix"]
        paths = stage_output_paths(config, stage)
        write_json(paths["raw_logprob_json"], result)
        write_csv_matrix(
            matrix=matrix,
            rows=config.colors,
            columns=config.traits,
            path=paths["raw_logprob_csv"],
        )
        write_heatmap(
            matrix=matrix,
            rows=config.colors,
            columns=config.traits,
            path=paths["raw_logprob_heatmap"],
            title=(
                "Raw favorite-color log-probability shift - "
                + (
                    "final adapter"
                    if stage.checkpoint_step is None
                    else f"checkpoint step {stage.checkpoint_step}"
                )
            ),
            max_abs=max_abs,
        )
    return max_abs


def cached_normalized_mean_probability_by_color(
    score: dict[str, Any],
    colors: list[str],
) -> dict[str, float]:
    """Average cached color-normalized probabilities over questions."""
    question_scores = score.get("question_scores")
    if not isinstance(question_scores, list) or not question_scores:
        raise ValueError("Cached score has no per-question results.")
    result = {}
    for color in colors:
        try:
            probabilities = [
                float(
                    question_score["colors"][color][
                        "normalized_probability_over_colors"
                    ]
                )
                for question_score in question_scores
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Cached score is missing normalized probabilities for {color}."
            ) from exc
        if any(
            probability < 0.0 or not math.isfinite(probability)
            for probability in probabilities
        ):
            raise ValueError("Normalized probabilities must be finite and nonnegative.")
        result[color] = float(np.mean(probabilities))
    return result


def compute_broad_boost_vectors(
    *,
    matrix: dict[str, dict[str, float]],
    colors: list[str],
    traits: list[str],
    excluded_readout_color: str = "blue",
) -> dict[str, dict[str, float]]:
    """Compute off-diagonal broad boosts with and without one readout color."""
    off_diagonal: dict[str, float] = {}
    off_diagonal_nonblue: dict[str, float] = {}
    for trait in traits:
        off_rows = [color for color in colors if color != trait]
        nonblue_rows = [
            color
            for color in off_rows
            if color != excluded_readout_color
        ]
        if not off_rows or not nonblue_rows:
            raise ValueError(
                "Broad-boost analysis requires retained off-diagonal rows."
            )
        off_diagonal[trait] = float(
            np.mean([matrix[color][trait] for color in off_rows])
        )
        off_diagonal_nonblue[trait] = float(
            np.mean([matrix[color][trait] for color in nonblue_rows])
        )
    return {
        "off_diagonal": off_diagonal,
        "off_diagonal_nonblue": off_diagonal_nonblue,
    }


def pearson_correlation(
    first: list[float] | np.ndarray,
    second: list[float] | np.ndarray,
) -> float:
    """Compute a finite Pearson correlation coefficient."""
    first_values = np.asarray(first, dtype=np.float64)
    second_values = np.asarray(second, dtype=np.float64)
    if first_values.shape != second_values.shape or first_values.ndim != 1:
        raise ValueError("Pearson inputs must be one-dimensional and shape-matched.")
    if first_values.size < 2:
        raise ValueError("Pearson correlation requires at least two observations.")
    if not np.all(np.isfinite(first_values)) or not np.all(
        np.isfinite(second_values)
    ):
        raise ValueError("Pearson inputs must be finite.")
    centered_first = first_values - np.mean(first_values)
    centered_second = second_values - np.mean(second_values)
    denominator = float(
        np.linalg.norm(centered_first) * np.linalg.norm(centered_second)
    )
    if denominator == 0.0:
        raise ValueError("Pearson correlation is undefined for a constant vector.")
    return float(np.dot(centered_first, centered_second) / denominator)


def pearson_correlation_or_none(
    first: list[float],
    second: list[float],
) -> float | None:
    """Compute Pearson correlation, returning None when it is undefined."""
    if len(first) < 2:
        return None
    try:
        return pearson_correlation(first, second)
    except ValueError as exc:
        if "constant vector" not in str(exc):
            raise
        return None


def projection_analysis_identity(
    config: CheckpointEvaluationConfig,
    stage: EvaluationStage,
) -> dict[str, Any]:
    """Return the cache identity for update/readout projection analysis."""
    return {
        "analysis_version": UPDATE_READOUT_PROJECTION_ANALYSIS_VERSION,
        "stage": stage.label,
        "checkpoint_step": stage.checkpoint_step,
        "base_model_id": config.base_model_id,
        "colors": config.colors,
        "traits": config.traits,
        "questions": config.questions,
        "candidate_prefix": config.candidate_prefix,
        "assistant_prefill": config.assistant_prefill,
        "raw_behavior_epsilon": config.epsilon,
        "torch_dtype": config.torch_dtype,
        "finite_difference_step": config.projection_finite_difference_step,
        "adapter_paths": {
            trait: str(stage.adapter_paths[trait]) for trait in config.traits
        },
    }


def projection_correlation_record(
    *,
    projection_matrix: dict[str, dict[str, float]],
    raw_behavior_matrix: dict[str, dict[str, float]],
    pairs: list[tuple[str, str]],
) -> dict[str, Any]:
    """Correlate projection and behavior values for readout/trait pairs."""
    projection_values = [
        float(projection_matrix[color][trait]) for color, trait in pairs
    ]
    behavior_values = [
        float(raw_behavior_matrix[color][trait]) for color, trait in pairs
    ]
    return {
        "pair_count": len(pairs),
        "pearson_r": pearson_correlation_or_none(
            projection_values,
            behavior_values,
        ),
    }


def build_projection_trait_summary(
    *,
    projection_matrix: dict[str, dict[str, float]],
    raw_behavior_matrix: dict[str, dict[str, float]],
    colors: list[str],
    traits: list[str],
) -> list[dict[str, Any]]:
    """Build projection and observed specificity margins by trait."""
    summaries = []
    for trait in traits:
        if trait not in colors:
            continue
        competitors = [color for color in colors if color != trait]
        if not competitors:
            continue
        projection_competitor = max(
            competitors,
            key=lambda color: projection_matrix[color][trait],
        )
        behavior_competitor = max(
            competitors,
            key=lambda color: raw_behavior_matrix[color][trait],
        )
        projection_diagonal = float(projection_matrix[trait][trait])
        behavior_diagonal = float(raw_behavior_matrix[trait][trait])
        projection_competitor_value = float(
            projection_matrix[projection_competitor][trait]
        )
        behavior_competitor_value = float(
            raw_behavior_matrix[behavior_competitor][trait]
        )
        summaries.append(
            {
                "trait": trait,
                "projection_diagonal": projection_diagonal,
                "projection_best_competitor": projection_competitor,
                "projection_best_competitor_value": (
                    projection_competitor_value
                ),
                "projection_specificity_margin": (
                    projection_diagonal - projection_competitor_value
                ),
                "behavior_diagonal": behavior_diagonal,
                "behavior_best_competitor": behavior_competitor,
                "behavior_best_competitor_value": behavior_competitor_value,
                "behavior_specificity_margin": (
                    behavior_diagonal - behavior_competitor_value
                ),
                "projection_top_readout": max(
                    colors,
                    key=lambda color: projection_matrix[color][trait],
                ),
                "behavior_top_readout": max(
                    colors,
                    key=lambda color: raw_behavior_matrix[color][trait],
                ),
            }
        )
    return summaries


def build_update_readout_projection_analysis(
    *,
    config: CheckpointEvaluationConfig,
    stage: EvaluationStage,
    estimate: dict[str, Any],
    raw_behavior_matrix: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Compare the update/readout projection matrix with raw behavior."""
    projection_matrix = estimate["projection_matrix"]
    all_pairs = [
        (color, trait)
        for trait in config.traits
        for color in config.colors
    ]
    diagonal_pairs = [
        (trait, trait) for trait in config.traits if trait in config.colors
    ]
    diagonal_pair_set = set(diagonal_pairs)
    off_diagonal_pairs = [
        pair for pair in all_pairs if pair not in diagonal_pair_set
    ]
    correlations = {
        "all_pairs": projection_correlation_record(
            projection_matrix=projection_matrix,
            raw_behavior_matrix=raw_behavior_matrix,
            pairs=all_pairs,
        ),
        "diagonal_pairs": projection_correlation_record(
            projection_matrix=projection_matrix,
            raw_behavior_matrix=raw_behavior_matrix,
            pairs=diagonal_pairs,
        ),
        "off_diagonal_pairs": projection_correlation_record(
            projection_matrix=projection_matrix,
            raw_behavior_matrix=raw_behavior_matrix,
            pairs=off_diagonal_pairs,
        ),
        "by_trait": {
            trait: projection_correlation_record(
                projection_matrix=projection_matrix,
                raw_behavior_matrix=raw_behavior_matrix,
                pairs=[(color, trait) for color in config.colors],
            )
            for trait in config.traits
        },
    }
    key_comparisons = []
    if "vermilion" in config.traits and {
        "vermilion",
        "red",
    }.issubset(config.colors):
        trait = "vermilion"
        own_readout = "vermilion"
        competitor = "red"
        key_comparisons.append(
            {
                "trait": trait,
                "own_readout": own_readout,
                "competitor_readout": competitor,
                "projection_own": projection_matrix[own_readout][trait],
                "projection_competitor": projection_matrix[competitor][trait],
                "projection_competitor_minus_own": (
                    projection_matrix[competitor][trait]
                    - projection_matrix[own_readout][trait]
                ),
                "behavior_own": raw_behavior_matrix[own_readout][trait],
                "behavior_competitor": raw_behavior_matrix[competitor][trait],
                "behavior_competitor_minus_own": (
                    raw_behavior_matrix[competitor][trait]
                    - raw_behavior_matrix[own_readout][trait]
                ),
            }
        )

    return {
        "analysis_identity": projection_analysis_identity(config, stage),
        "definition": (
            "A[tau,u] = <Delta theta_tau, grad_theta log P_theta(u|q)> "
            "at the base model, averaged over questions"
        ),
        "estimator": (
            "centered directional finite difference in the loaded LoRA "
            "multiplier at alpha=0"
        ),
        "estimator_formula": (
            "A[tau,u] ~= (mean_q log P_{theta0+h Delta theta_tau}(u|q) "
            "- mean_q log P_{theta0-h Delta theta_tau}(u|q)) / (2h)"
        ),
        "matrix_orientation": (
            "projection_matrix[readout_color_u][fine_tuning_trait_tau]"
        ),
        "raw_behavior_definition": (
            "mean_q log((P_{theta_tau}(u|q)+epsilon) / "
            "(P_{theta0}(u|q)+epsilon))"
        ),
        "projection_matrix": projection_matrix,
        "raw_behavior_matrix": raw_behavior_matrix,
        "positive_mean_raw_logprob_matrix": estimate[
            "positive_mean_raw_logprob_matrix"
        ],
        "negative_mean_raw_logprob_matrix": estimate[
            "negative_mean_raw_logprob_matrix"
        ],
        "lora_layer_count_by_trait": estimate["lora_layer_count_by_trait"],
        "correlations": correlations,
        "trait_summaries": build_projection_trait_summary(
            projection_matrix=projection_matrix,
            raw_behavior_matrix=raw_behavior_matrix,
            colors=config.colors,
            traits=config.traits,
        ),
        "key_comparisons": key_comparisons,
    }


def find_teacher_logprob_cache_group(
    cache_root: Path,
    traits: list[str],
) -> tuple[Path, dict[str, Any]]:
    """Find a condition cache containing neutral and every requested trait."""
    candidates: list[tuple[int, Path, dict[str, Any]]] = []
    for manifest_path in cache_root.rglob("manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        group_dir = manifest_path.parent
        required_labels = ["neutral_no_system", *traits]
        if all(
            list(group_dir.glob(f"{slugify(label)}_*.npz"))
            for label in required_labels
        ):
            prompt_count = int(
                manifest.get("common_metadata", {}).get("prompt_count", 0)
            )
            candidates.append((prompt_count, group_dir, manifest))
    if not candidates:
        raise FileNotFoundError(
            "No teacher-logprob cache contains neutral and all requested traits "
            f"under {cache_root}."
        )
    _, group_dir, manifest = max(candidates, key=lambda item: item[0])
    return group_dir, manifest


def condition_logprob_cache_path(group_dir: Path, label: str) -> Path:
    """Return the unique cached logprob file for one condition label."""
    matches = sorted(group_dir.glob(f"{slugify(label)}_*.npz"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one cached condition for {label}, found {len(matches)}."
        )
    return matches[0]


def normalized_logprobability_rows(logprobabilities: np.ndarray) -> np.ndarray:
    """Normalize a two-dimensional logprobability array row-wise."""
    values = np.asarray(logprobabilities, dtype=np.float64)
    if values.ndim != 2 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Teacher logprobabilities must be a finite 2D array.")
    maximum = np.max(values, axis=1, keepdims=True)
    log_normalizers = maximum + np.log(
        np.sum(np.exp(values - maximum), axis=1, keepdims=True)
    )
    return values - log_normalizers


def load_teacher_trait_logprobabilities(
    *,
    cache_root: Path,
    traits: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load row-normalized 00-99 log probabilities for each teacher trait."""
    group_dir, manifest = find_teacher_logprob_cache_group(cache_root, traits)
    logprobabilities: dict[str, np.ndarray] = {}
    expected_shape: tuple[int, int] | None = None
    for trait in traits:
        path = condition_logprob_cache_path(group_dir, trait)
        with np.load(path, allow_pickle=False) as data:
            values = normalized_logprobability_rows(data["logprobs"])
        if expected_shape is None:
            expected_shape = values.shape
        elif values.shape != expected_shape:
            raise ValueError(
                f"Teacher cache shape for {trait} is {values.shape}, "
                f"expected {expected_shape}."
            )
        logprobabilities[trait] = values

    if expected_shape is None:
        raise ValueError("Bayesian decoding requires at least one teacher trait.")
    common_metadata = manifest.get("common_metadata", {})
    manifest_prompt_count = int(common_metadata.get("prompt_count", 0))
    if manifest_prompt_count and manifest_prompt_count != expected_shape[0]:
        raise ValueError(
            "Teacher-cache prompt count does not match the cached arrays."
        )
    return logprobabilities, {
        "cache_group": str(group_dir),
        "prompt_count": int(expected_shape[0]),
        "candidate_count": int(expected_shape[1]),
        "cache_common_metadata": common_metadata,
    }


def build_teacher_cross_loglikelihood_matrix(
    *,
    logprobabilities_by_trait: dict[str, np.ndarray],
    traits: list[str],
) -> dict[str, dict[str, float]]:
    """Compute prompt-averaged soft cross-log-likelihoods S-bar(tau,t)."""
    normalized_values = {
        trait: normalized_logprobability_rows(
            logprobabilities_by_trait[trait]
        )
        for trait in traits
    }
    shapes = {values.shape for values in normalized_values.values()}
    if len(shapes) != 1:
        raise ValueError("All teacher condition caches must have the same shape.")
    return {
        true_trait: {
            candidate_trait: float(
                np.mean(
                    np.sum(
                        np.exp(normalized_values[true_trait])
                        * normalized_values[candidate_trait],
                        axis=1,
                    )
                )
            )
            for candidate_trait in traits
        }
        for true_trait in traits
    }


def normalize_trait_prior(
    *,
    weights: dict[str, float],
    traits: list[str],
    probability_floor: float,
) -> dict[str, float]:
    """Validate, floor, and normalize prior weights over candidate traits."""
    if probability_floor < 0.0 or not math.isfinite(probability_floor):
        raise ValueError("Prior probability floor must be finite and nonnegative.")
    floor = max(float(probability_floor), float(np.finfo(np.float64).tiny))
    values = np.asarray([weights[trait] for trait in traits], dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Trait prior weights must be finite and nonnegative.")
    values = np.maximum(values, floor)
    normalizer = float(np.sum(values))
    if normalizer <= 0.0 or not math.isfinite(normalizer):
        raise ValueError("Trait prior weights must have positive finite mass.")
    values /= normalizer
    return {
        trait: float(values[index]) for index, trait in enumerate(traits)
    }


def stable_softmax(values: list[float] | np.ndarray) -> np.ndarray:
    """Return a finite softmax vector using maximum subtraction."""
    logits = np.asarray(values, dtype=np.float64)
    if logits.ndim != 1 or logits.size == 0 or not np.all(np.isfinite(logits)):
        raise ValueError("Softmax logits must be a non-empty finite vector.")
    shifted = logits - np.max(logits)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities)
    return probabilities


def build_bayesian_posterior_matrix(
    *,
    average_score_matrix: dict[str, dict[str, float]],
    prior: dict[str, float],
    traits: list[str],
    beta: float,
) -> dict[str, dict[str, float]]:
    """Decode each true trait using log prior plus beta times S-bar."""
    if beta < 0.0 or not math.isfinite(beta):
        raise ValueError("Decoder beta must be finite and nonnegative.")
    log_prior = np.log(
        np.asarray([prior[trait] for trait in traits], dtype=np.float64)
    )
    if not np.all(np.isfinite(log_prior)):
        raise ValueError("Decoder prior probabilities must be positive and finite.")
    matrix: dict[str, dict[str, float]] = {}
    for true_trait in traits:
        scores = np.asarray(
            [average_score_matrix[true_trait][trait] for trait in traits],
            dtype=np.float64,
        )
        probabilities = stable_softmax(log_prior + beta * scores)
        matrix[true_trait] = {
            trait: float(probabilities[index])
            for index, trait in enumerate(traits)
        }
    return matrix


def summarize_bayesian_posterior_matrix(
    matrix: dict[str, dict[str, float]],
    traits: list[str],
) -> dict[str, Any]:
    """Summarize MAP predictions and true-trait posterior mass."""
    predictions = {
        true_trait: max(
            traits,
            key=lambda candidate: matrix[true_trait][candidate],
        )
        for true_trait in traits
    }
    return {
        "map_predictions": predictions,
        "map_accuracy": float(
            np.mean(
                [
                    predictions[true_trait] == true_trait
                    for true_trait in traits
                ]
            )
        ),
        "mean_true_trait_posterior": float(
            np.mean(
                [matrix[true_trait][true_trait] for true_trait in traits]
            )
        ),
    }


def build_identifiability_records(
    *,
    average_score_matrix: dict[str, dict[str, float]],
    traits: list[str],
    prompt_count: int,
) -> list[dict[str, Any]]:
    """Build all pairwise S-bar and full-sum identifiability gaps."""
    records = []
    for true_trait in traits:
        diagonal_score = average_score_matrix[true_trait][true_trait]
        ranked_candidates = sorted(
            traits,
            key=lambda trait: average_score_matrix[true_trait][trait],
            reverse=True,
        )
        for competitor in traits:
            if competitor == true_trait:
                continue
            average_gap = (
                diagonal_score
                - average_score_matrix[true_trait][competitor]
            )
            records.append(
                {
                    "true_trait": true_trait,
                    "competitor": competitor,
                    "candidate_rank": ranked_candidates.index(competitor) + 1,
                    "average_self_score": diagonal_score,
                    "average_competitor_score": average_score_matrix[
                        true_trait
                    ][competitor],
                    "average_gap_nats_per_prompt": average_gap,
                    "summed_gap_nats": prompt_count * average_gap,
                }
            )
    return records


def critical_beta_record(
    *,
    true_trait: str,
    competitor: str,
    prior_name: str,
    prior: dict[str, float],
    average_score_matrix: dict[str, dict[str, float]],
    prompt_count: int,
) -> dict[str, Any]:
    """Compute the prior/evidence crossing beta for one trait competition."""
    average_gap = (
        average_score_matrix[true_trait][true_trait]
        - average_score_matrix[true_trait][competitor]
    )
    prior_log_odds_true_over_competitor = math.log(
        prior[true_trait] / prior[competitor]
    )
    tolerance = 1e-15
    numerator = -prior_log_odds_true_over_competitor
    if abs(numerator) <= tolerance:
        numerator = 0.0
    if abs(average_gap) <= tolerance:
        beta_critical: float | None = None
        positive_beta_threshold: float | None = None
        preference_rule = "prior_only; likelihood scores are tied"
    else:
        beta_critical = numerator / average_gap
        if abs(beta_critical) <= tolerance:
            beta_critical = 0.0
        if average_gap > 0.0:
            positive_beta_threshold = max(beta_critical, 0.0)
            preference_rule = "true trait preferred when beta > beta_critical"
        else:
            positive_beta_threshold = None
            preference_rule = "true trait preferred when beta < beta_critical"
    return {
        "true_trait": true_trait,
        "competitor": competitor,
        "prior": prior_name,
        "prior_true_probability": prior[true_trait],
        "prior_competitor_probability": prior[competitor],
        "log_prior_competitor_over_true": numerator,
        "average_gap_nats_per_prompt": average_gap,
        "summed_gap_nats": prompt_count * average_gap,
        "beta_critical": beta_critical,
        "positive_beta_threshold": positive_beta_threshold,
        "preference_rule": preference_rule,
    }


def build_bayesian_decoder_analysis(
    *,
    config: CheckpointEvaluationConfig,
    stage_result: dict[str, Any],
    beta_sweep: list[float] = DEFAULT_BAYESIAN_DECODER_BETAS,
    key_competitions: list[tuple[str, str]] = (
        DEFAULT_BAYESIAN_DECODER_KEY_COMPETITIONS
    ),
) -> dict[str, Any]:
    """Build uniform- and base-prior Bayesian trait confusion matrices."""
    logprobabilities, cache_metadata = load_teacher_trait_logprobabilities(
        cache_root=config.teacher_logprob_cache_dir,
        traits=config.traits,
    )
    average_score_matrix = build_teacher_cross_loglikelihood_matrix(
        logprobabilities_by_trait=logprobabilities,
        traits=config.traits,
    )
    prompt_count = cache_metadata["prompt_count"]
    if prompt_count <= 0:
        raise ValueError("Bayesian decoding requires at least one prompt.")

    uniform_prior = normalize_trait_prior(
        weights={trait: 1.0 for trait in config.traits},
        traits=config.traits,
        probability_floor=0.0,
    )
    base_prior_weights = cached_normalized_mean_probability_by_color(
        stage_result["base_scores"],
        config.traits,
    )
    base_prior = normalize_trait_prior(
        weights=base_prior_weights,
        traits=config.traits,
        probability_floor=config.epsilon,
    )
    priors = {
        "uniform": uniform_prior,
        "base_model_favorite_color": base_prior,
    }

    principled_beta = float(prompt_count)
    evaluated_betas = sorted(
        {
            *[float(beta) for beta in beta_sweep],
            principled_beta,
        }
    )
    if any(beta < 0.0 or not math.isfinite(beta) for beta in evaluated_betas):
        raise ValueError("Every Bayesian decoder beta must be finite and nonnegative.")
    posterior_sweeps: dict[str, list[dict[str, Any]]] = {}
    for prior_name, prior in priors.items():
        entries = []
        for beta in evaluated_betas:
            matrix = build_bayesian_posterior_matrix(
                average_score_matrix=average_score_matrix,
                prior=prior,
                traits=config.traits,
                beta=beta,
            )
            entries.append(
                {
                    "beta": beta,
                    "posterior_matrix": matrix,
                    **summarize_bayesian_posterior_matrix(
                        matrix,
                        config.traits,
                    ),
                }
            )
        posterior_sweeps[prior_name] = entries

    retained_competitions = [
        (true_trait, competitor)
        for true_trait, competitor in key_competitions
        if true_trait in config.traits and competitor in config.traits
    ]
    critical_beta_records = [
        critical_beta_record(
            true_trait=true_trait,
            competitor=competitor,
            prior_name=prior_name,
            prior=prior,
            average_score_matrix=average_score_matrix,
            prompt_count=prompt_count,
        )
        for true_trait, competitor in retained_competitions
        for prior_name, prior in priors.items()
    ]
    return {
        "analysis_identity": {
            "analysis_version": BAYESIAN_DECODER_ANALYSIS_VERSION,
            "traits": config.traits,
            "teacher_logprob_cache_dir": str(
                config.teacher_logprob_cache_dir
            ),
            "cache_group": cache_metadata["cache_group"],
            "prompt_count": prompt_count,
            "candidate_count": cache_metadata["candidate_count"],
            "base_model_id": config.base_model_id,
            "base_prior_questions": config.questions,
            "epsilon": config.epsilon,
        },
        "score_definition": (
            "S_bar(tau,t) = mean_x sum_n "
            "P_tau(n|x) log P_t(n|x)"
        ),
        "posterior_definition": (
            "softmax_t(log pi_0(t) + beta * S_bar(tau,t))"
        ),
        "principled_beta": principled_beta,
        "principled_beta_reason": (
            "S_bar is averaged over prompts, so beta equals prompt_count"
        ),
        "requested_beta_sweep": [float(beta) for beta in beta_sweep],
        "evaluated_betas": evaluated_betas,
        "traits": config.traits,
        "priors": priors,
        "base_prior_probability_floor": max(
            config.epsilon,
            float(np.finfo(np.float64).tiny),
        ),
        "average_score_matrix": average_score_matrix,
        "identifiability_records": build_identifiability_records(
            average_score_matrix=average_score_matrix,
            traits=config.traits,
            prompt_count=prompt_count,
        ),
        "posterior_sweeps": posterior_sweeps,
        "key_competitions": [list(pair) for pair in retained_competitions],
        "critical_beta_records": critical_beta_records,
        "cache_metadata": cache_metadata,
    }


def load_teacher_distribution_metrics(
    *,
    cache_root: Path,
    traits: list[str],
) -> dict[str, Any]:
    """Compute teacher KL and base cross-entropy from condition caches."""
    group_dir, manifest = find_teacher_logprob_cache_group(cache_root, traits)
    neutral_path = condition_logprob_cache_path(group_dir, "neutral_no_system")
    with np.load(neutral_path, allow_pickle=False) as data:
        neutral_logprobabilities = normalized_logprobability_rows(
            data["logprobs"]
        )

    teacher_kl: dict[str, float] = {}
    base_cross_entropy: dict[str, float] = {}
    for trait in traits:
        trait_path = condition_logprob_cache_path(group_dir, trait)
        with np.load(trait_path, allow_pickle=False) as data:
            trait_logprobabilities = normalized_logprobability_rows(
                data["logprobs"]
            )
        if trait_logprobabilities.shape != neutral_logprobabilities.shape:
            raise ValueError(
                f"Teacher and neutral cache shapes differ for {trait}."
            )
        trait_probabilities = np.exp(trait_logprobabilities)
        teacher_kl[trait] = float(
            np.mean(
                np.sum(
                    trait_probabilities
                    * (trait_logprobabilities - neutral_logprobabilities),
                    axis=1,
                )
            )
        )
        base_cross_entropy[trait] = float(
            np.mean(
                -np.sum(
                    trait_probabilities * neutral_logprobabilities,
                    axis=1,
                )
            )
        )
    return {
        "teacher_kl_from_neutral": teacher_kl,
        "base_teacher_cross_entropy": base_cross_entropy,
        "cache_group": str(group_dir),
        "prompt_count": int(neutral_logprobabilities.shape[0]),
        "candidate_count": int(neutral_logprobabilities.shape[1]),
        "cache_common_metadata": manifest.get("common_metadata", {}),
    }


def token_lengths_from_local_adapter(
    adapter_path: Path,
    traits: list[str],
) -> dict[str, int]:
    """Tokenize trait names with the tokenizer saved beside an adapter."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    lengths = {
        trait: len(tokenizer.encode(trait, add_special_tokens=False))
        for trait in traits
    }
    if any(length <= 0 for length in lengths.values()):
        raise ValueError("Every trait must tokenize to at least one token.")
    return lengths


def lora_product_squared_frobenius(a_matrix: Any, b_matrix: Any) -> float:
    """Compute ||B A||_F^2 from small rank-space Gram matrices."""
    import torch

    a_values = a_matrix.to(dtype=torch.float32)
    b_values = b_matrix.to(dtype=torch.float32)
    if a_values.ndim != 2 or b_values.ndim != 2:
        raise ValueError("LoRA A and B tensors must be matrices.")
    if a_values.shape[0] != b_values.shape[1]:
        raise ValueError("LoRA A and B ranks do not match.")
    a_gram = a_values @ a_values.T
    b_gram = b_values.T @ b_values
    return max(float(torch.sum(a_gram * b_gram).item()), 0.0)


def effective_lora_update_norm(adapter_path: Path) -> float:
    """Compute sqrt(sum_l ||scale_l B_l A_l||_F^2) from an adapter."""
    import torch
    from safetensors import safe_open

    config_path = adapter_path / "adapter_config.json"
    weights_path = adapter_path / "adapter_model.safetensors"
    adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    default_rank = int(adapter_config["r"])
    default_alpha = float(adapter_config["lora_alpha"])
    use_rslora = bool(adapter_config.get("use_rslora", False))
    rank_pattern = adapter_config.get("rank_pattern") or {}
    alpha_pattern = adapter_config.get("alpha_pattern") or {}

    def patterned_value(pattern: dict[str, Any], module: str, default: float) -> float:
        matches = [key for key in pattern if module.endswith(key)]
        if not matches:
            return default
        return float(pattern[max(matches, key=len)])

    squared_norm = 0.0
    pair_count = 0
    with safe_open(weights_path, framework="pt", device="cpu") as tensors:
        keys = set(tensors.keys())
        for a_key in sorted(key for key in keys if key.endswith(".lora_A.weight")):
            module = a_key[: -len(".lora_A.weight")]
            b_key = f"{module}.lora_B.weight"
            if b_key not in keys:
                raise ValueError(f"Missing LoRA B tensor paired with {a_key}.")
            a_matrix = tensors.get_tensor(a_key).to(dtype=torch.float32)
            b_matrix = tensors.get_tensor(b_key).to(dtype=torch.float32)
            rank = int(patterned_value(rank_pattern, module, default_rank))
            alpha = patterned_value(alpha_pattern, module, default_alpha)
            scale = alpha / (math.sqrt(rank) if use_rslora else rank)
            pair_squared_norm = lora_product_squared_frobenius(
                a_matrix,
                b_matrix,
            )
            squared_norm += (scale**2) * pair_squared_norm
            pair_count += 1
    if pair_count == 0:
        raise ValueError(f"No paired LoRA tensors found in {weights_path}.")
    return math.sqrt(squared_norm)


def checkpoint_training_loss(
    *,
    training_output_dir: Path,
    trait: str,
    stage: EvaluationStage,
    available_checkpoint_steps: list[int],
) -> tuple[float, int]:
    """Load the saved last-minibatch loss used for the KL-reduction proxy."""
    if stage.checkpoint_step is not None:
        source_step = stage.checkpoint_step
    elif available_checkpoint_steps:
        source_step = max(available_checkpoint_steps)
    else:
        raise ValueError("No checkpoint loss is available for the final adapter.")
    metadata_path = (
        training_output_dir
        / "trainer"
        / slugify(trait)
        / f"checkpoint-step-{source_step:06d}"
        / "metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    loss = float(metadata["loss"])
    if not math.isfinite(loss):
        raise ValueError(f"Non-finite cached training loss for {trait}.")
    return loss, source_step


def broad_boost_analysis_identity(
    config: CheckpointEvaluationConfig,
    stage: EvaluationStage,
) -> dict[str, Any]:
    """Return the cache identity for one broad-boost analysis."""
    return {
        "analysis_version": BROAD_BOOST_ANALYSIS_VERSION,
        "stage": stage.label,
        "checkpoint_step": stage.checkpoint_step,
        "colors": config.colors,
        "traits": config.traits,
        "epsilon": config.epsilon,
        "teacher_logprob_cache_dir": str(config.teacher_logprob_cache_dir),
        "adapter_paths": {
            trait: str(stage.adapter_paths[trait]) for trait in config.traits
        },
    }


def build_broad_boost_correlation_analysis(
    *,
    config: CheckpointEvaluationConfig,
    stage: EvaluationStage,
    stage_result: dict[str, Any],
    available_checkpoint_steps: list[int],
) -> dict[str, Any]:
    """Build the requested five-by-four broad-boost correlation table."""
    raw_matrix = cached_question_mean_log_ratio_matrix(
        stage_result=stage_result,
        colors=config.colors,
        traits=config.traits,
        probability_mode="raw",
        epsilon=config.epsilon,
    )
    normalized_matrix = cached_question_mean_log_ratio_matrix(
        stage_result=stage_result,
        colors=config.colors,
        traits=config.traits,
        probability_mode="normalized",
        epsilon=config.epsilon,
    )
    raw_boosts = compute_broad_boost_vectors(
        matrix=raw_matrix,
        colors=config.colors,
        traits=config.traits,
    )
    normalized_boosts = compute_broad_boost_vectors(
        matrix=normalized_matrix,
        colors=config.colors,
        traits=config.traits,
    )

    base_probabilities = cached_normalized_mean_probability_by_color(
        stage_result["base_scores"],
        config.colors,
    )
    rarity = {
        trait: -math.log(base_probabilities[trait] + config.epsilon)
        for trait in config.traits
    }
    token_lengths = token_lengths_from_local_adapter(
        stage.adapter_paths[config.traits[0]],
        config.traits,
    )
    teacher_metrics = load_teacher_distribution_metrics(
        cache_root=config.teacher_logprob_cache_dir,
        traits=config.traits,
    )
    update_norms = {
        trait: effective_lora_update_norm(stage.adapter_paths[trait])
        for trait in config.traits
    }
    training_losses: dict[str, float] = {}
    loss_source_steps: dict[str, int] = {}
    kl_reduction_proxy: dict[str, float] = {}
    for trait in config.traits:
        training_loss, source_step = checkpoint_training_loss(
            training_output_dir=config.training_output_dir,
            trait=trait,
            stage=stage,
            available_checkpoint_steps=available_checkpoint_steps,
        )
        training_losses[trait] = training_loss
        loss_source_steps[trait] = source_step
        kl_reduction_proxy[trait] = (
            teacher_metrics["base_teacher_cross_entropy"][trait]
            - training_loss
        )

    quantities: dict[str, dict[str, float | int]] = {
        "base_rarity": rarity,
        "token_length": token_lengths,
        "teacher_kl_from_neutral": teacher_metrics[
            "teacher_kl_from_neutral"
        ],
        "effective_lora_update_norm": update_norms,
        "cached_kl_reduction_proxy": kl_reduction_proxy,
    }
    boost_columns = {
        "raw_off_diagonal": raw_boosts["off_diagonal"],
        "raw_off_diagonal_nonblue": raw_boosts[
            "off_diagonal_nonblue"
        ],
        "normalized_off_diagonal": normalized_boosts["off_diagonal"],
        "normalized_off_diagonal_nonblue": normalized_boosts[
            "off_diagonal_nonblue"
        ],
    }
    correlations = {
        quantity: {
            boost_name: pearson_correlation(
                [float(values[trait]) for trait in config.traits],
                [boost_values[trait] for trait in config.traits],
            )
            for boost_name, boost_values in boost_columns.items()
        }
        for quantity, values in quantities.items()
    }
    return {
        "analysis_identity": broad_boost_analysis_identity(config, stage),
        "correlation_method": "Pearson",
        "matrix_aggregation": (
            "mean_over_questions(log((P_fine_tuned + epsilon) / "
            "(P_base + epsilon)))"
        ),
        "nonblue_filter": (
            "exclude readout row u=blue; retain all fine-tuning traits t"
        ),
        "quantity_order": list(quantities),
        "boost_column_order": list(boost_columns),
        "quantities_by_trait": quantities,
        "broad_boosts_by_trait": boost_columns,
        "correlations": correlations,
        "teacher_metric_metadata": {
            key: teacher_metrics[key]
            for key in (
                "cache_group",
                "prompt_count",
                "candidate_count",
                "cache_common_metadata",
            )
        },
        "kl_reduction_proxy_metadata": {
            "definition": (
                "cached base teacher cross-entropy minus checkpoint metadata loss"
            ),
            "limitation": (
                "checkpoint loss is the saved last-minibatch soft cross-entropy, "
                "not a full-dataset student KL evaluation"
            ),
            "training_loss_by_trait": training_losses,
            "loss_source_checkpoint_step_by_trait": loss_source_steps,
        },
    }


def write_broad_boost_correlation_outputs(
    *,
    config: CheckpointEvaluationConfig,
    analysis: dict[str, Any],
) -> dict[str, Path]:
    """Write the correlation table and its per-trait audit values."""
    paths = broad_boost_output_paths(config)
    write_json(paths["json"], analysis)
    with paths["csv"].open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["quantity", *analysis["boost_column_order"]])
        for quantity in analysis["quantity_order"]:
            writer.writerow(
                [
                    quantity,
                    *[
                        analysis["correlations"][quantity][column]
                        for column in analysis["boost_column_order"]
                    ],
                ]
            )
    logger.success("Saved broad-boost correlation CSV to {}", paths["csv"])

    with paths["trait_values_csv"].open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "trait",
                *analysis["quantity_order"],
                *analysis["boost_column_order"],
            ]
        )
        for trait in config.traits:
            writer.writerow(
                [
                    trait,
                    *[
                        analysis["quantities_by_trait"][quantity][trait]
                        for quantity in analysis["quantity_order"]
                    ],
                    *[
                        analysis["broad_boosts_by_trait"][column][trait]
                        for column in analysis["boost_column_order"]
                    ],
                ]
            )
    logger.success("Saved broad-boost trait values to {}", paths["trait_values_csv"])
    return paths


def log_broad_boost_correlation_table(analysis: dict[str, Any]) -> None:
    """Log the five-by-four correlation table."""
    columns = analysis["boost_column_order"]
    logger.info(
        "Broad-boost Pearson correlations [{}]:\n{}",
        analysis["analysis_identity"]["stage"],
        "\n".join(
            [
                "quantity," + ",".join(columns),
                *[
                    quantity
                    + ","
                    + ",".join(
                        f"{analysis['correlations'][quantity][column]:+.6f}"
                        for column in columns
                    )
                    for quantity in analysis["quantity_order"]
                ],
            ]
        ),
    )


def load_reusable_broad_boost_analysis(
    *,
    config: CheckpointEvaluationConfig,
    stage: EvaluationStage,
) -> dict[str, Any] | None:
    """Load a complete matching broad-boost analysis under skip-existing."""
    paths = broad_boost_output_paths(config)
    if not config.skip_existing or not all(path.is_file() for path in paths.values()):
        return None
    try:
        analysis = json.loads(paths["json"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if analysis.get("analysis_identity") != broad_boost_analysis_identity(
        config,
        stage,
    ):
        return None
    logger.info("Reusing completed broad-boost correlation analysis")
    return analysis


def load_reusable_update_readout_projection(
    *,
    config: CheckpointEvaluationConfig,
    stage: EvaluationStage,
) -> dict[str, Any] | None:
    """Load a matching update/readout projection analysis."""
    path = update_readout_projection_output_paths(config)["json"]
    if not config.skip_existing or not path.is_file():
        return None
    try:
        analysis = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if analysis.get("analysis_identity") != projection_analysis_identity(
        config,
        stage,
    ):
        return None
    logger.info("Reusing completed update/readout projection analysis")
    return analysis


def write_update_readout_projection_scatter(
    path: Path,
    analysis: dict[str, Any],
) -> None:
    """Plot projected first-order shifts against observed raw behavior."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = analysis["analysis_identity"]["colors"]
    traits = analysis["analysis_identity"]["traits"]
    projection = analysis["projection_matrix"]
    behavior = analysis["raw_behavior_matrix"]
    diagonal_x = []
    diagonal_y = []
    off_diagonal_x = []
    off_diagonal_y = []
    for trait in traits:
        for color in colors:
            x_value = projection[color][trait]
            y_value = behavior[color][trait]
            if color == trait:
                diagonal_x.append(x_value)
                diagonal_y.append(y_value)
            else:
                off_diagonal_x.append(x_value)
                off_diagonal_y.append(y_value)

    fig, axis = plt.subplots(figsize=(8.5, 7.2))
    axis.scatter(
        off_diagonal_x,
        off_diagonal_y,
        color="#2364aa",
        alpha=0.55,
        s=30,
        label="Off-diagonal pairs",
    )
    axis.scatter(
        diagonal_x,
        diagonal_y,
        color="#c73e1d",
        alpha=0.9,
        s=48,
        marker="^",
        label="Matched trait/readout",
    )
    for comparison in analysis["key_comparisons"]:
        trait = comparison["trait"]
        for color in (
            comparison["own_readout"],
            comparison["competitor_readout"],
        ):
            axis.annotate(
                f"{trait}->{color}",
                (projection[color][trait], behavior[color][trait]),
                xytext=(6, 5),
                textcoords="offset points",
                fontsize=8,
            )
    pearson_r = analysis["correlations"]["all_pairs"]["pearson_r"]
    correlation_label = "undefined" if pearson_r is None else f"{pearson_r:.4f}"
    axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.4)
    axis.axvline(0.0, color="black", linewidth=0.7, alpha=0.4)
    axis.grid(alpha=0.2)
    axis.set_xlabel("Update/readout projection A[tau,u]")
    axis.set_ylabel("Observed raw log-probability shift M_raw[u,tau]")
    axis.set_title(
        "Update/readout projection vs. observed behavior "
        f"(Pearson r={correlation_label})"
    )
    axis.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    logger.success("Saved update/readout projection scatter to {}", path)


def write_update_readout_projection_outputs(
    *,
    config: CheckpointEvaluationConfig,
    analysis: dict[str, Any],
) -> dict[str, Path]:
    """Write projection data, comparison tables, and diagnostic plots."""
    paths = update_readout_projection_output_paths(config)
    write_json(paths["json"], analysis)
    write_csv_matrix(
        matrix=analysis["projection_matrix"],
        rows=config.colors,
        columns=config.traits,
        path=paths["matrix_csv"],
    )
    with paths["comparison_csv"].open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "fine_tuning_trait",
                "readout_color",
                "is_diagonal",
                "update_readout_projection",
                "raw_behavior_logprob_shift",
            ]
        )
        for trait in config.traits:
            for color in config.colors:
                writer.writerow(
                    [
                        trait,
                        color,
                        trait == color,
                        analysis["projection_matrix"][color][trait],
                        analysis["raw_behavior_matrix"][color][trait],
                    ]
                )
    logger.success(
        "Saved update/readout behavior comparison to {}",
        paths["comparison_csv"],
    )

    summary_fields = [
        "trait",
        "projection_diagonal",
        "projection_best_competitor",
        "projection_best_competitor_value",
        "projection_specificity_margin",
        "behavior_diagonal",
        "behavior_best_competitor",
        "behavior_best_competitor_value",
        "behavior_specificity_margin",
        "projection_top_readout",
        "behavior_top_readout",
    ]
    with paths["trait_summary_csv"].open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(analysis["trait_summaries"])
    logger.success(
        "Saved update/readout trait summary to {}",
        paths["trait_summary_csv"],
    )
    write_heatmap(
        matrix=analysis["projection_matrix"],
        rows=config.colors,
        columns=config.traits,
        path=paths["heatmap"],
        title="LoRA update/readout directional projection A[tau,u]",
    )
    write_update_readout_projection_scatter(paths["scatter"], analysis)
    return paths


def log_update_readout_projection_summary(analysis: dict[str, Any]) -> None:
    """Log all-pair correlation and configured key comparisons."""
    correlations = analysis["correlations"]
    logger.info(
        "Update/readout projection correlation: all pairs r={}, "
        "off-diagonal r={}",
        correlations["all_pairs"]["pearson_r"],
        correlations["off_diagonal_pairs"]["pearson_r"],
    )
    for comparison in analysis["key_comparisons"]:
        logger.info(
            "Projection [{}]: A({}, {})={:.8g}, A({}, {})={:.8g}, "
            "competitor-minus-own={:+.8g}",
            comparison["trait"],
            comparison["trait"],
            comparison["competitor_readout"],
            comparison["projection_competitor"],
            comparison["trait"],
            comparison["own_readout"],
            comparison["projection_own"],
            comparison["projection_competitor_minus_own"],
        )


def posterior_sweep_entry(
    analysis: dict[str, Any],
    prior_name: str,
    beta: float,
) -> dict[str, Any]:
    """Return one posterior-sweep entry matching a prior and beta."""
    for entry in analysis["posterior_sweeps"][prior_name]:
        if math.isclose(entry["beta"], beta, rel_tol=0.0, abs_tol=1e-12):
            return entry
    raise ValueError(f"No posterior matrix found for beta={beta}.")


def write_bayesian_decoder_scores_csv(
    path: Path,
    analysis: dict[str, Any],
) -> None:
    """Write averaged/summed scores and identifiability gaps in long form."""
    path.parent.mkdir(parents=True, exist_ok=True)
    traits = analysis["traits"]
    prompt_count = analysis["analysis_identity"]["prompt_count"]
    matrix = analysis["average_score_matrix"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "true_trait",
                "candidate_trait",
                "candidate_rank",
                "is_true_trait",
                "average_score_nats_per_prompt",
                "summed_score_nats",
                "average_identifiability_gap_nats_per_prompt",
                "summed_identifiability_gap_nats",
            ]
        )
        for true_trait in traits:
            diagonal = matrix[true_trait][true_trait]
            ranked = sorted(
                traits,
                key=lambda trait: matrix[true_trait][trait],
                reverse=True,
            )
            for candidate_trait in traits:
                score = matrix[true_trait][candidate_trait]
                gap = diagonal - score
                writer.writerow(
                    [
                        true_trait,
                        candidate_trait,
                        ranked.index(candidate_trait) + 1,
                        candidate_trait == true_trait,
                        score,
                        prompt_count * score,
                        gap,
                        prompt_count * gap,
                    ]
                )
    logger.success("Saved Bayesian decoder scores to {}", path)


def write_bayesian_posterior_sweep_csv(
    path: Path,
    analysis: dict[str, Any],
) -> None:
    """Write every prior/beta posterior confusion matrix in long form."""
    path.parent.mkdir(parents=True, exist_ok=True)
    traits = analysis["traits"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "prior",
                "beta",
                "true_trait",
                "candidate_trait",
                "posterior_probability",
                "map_prediction",
                "is_map_prediction",
                "map_accuracy",
                "mean_true_trait_posterior",
            ]
        )
        for prior_name, entries in analysis["posterior_sweeps"].items():
            for entry in entries:
                for true_trait in traits:
                    prediction = entry["map_predictions"][true_trait]
                    for candidate_trait in traits:
                        writer.writerow(
                            [
                                prior_name,
                                entry["beta"],
                                true_trait,
                                candidate_trait,
                                entry["posterior_matrix"][true_trait][
                                    candidate_trait
                                ],
                                prediction,
                                candidate_trait == prediction,
                                entry["map_accuracy"],
                                entry["mean_true_trait_posterior"],
                            ]
                        )
    logger.success("Saved Bayesian posterior sweep to {}", path)


def write_critical_beta_csv(path: Path, analysis: dict[str, Any]) -> None:
    """Write critical beta values for configured key competitions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "true_trait",
        "competitor",
        "prior",
        "prior_true_probability",
        "prior_competitor_probability",
        "log_prior_competitor_over_true",
        "average_gap_nats_per_prompt",
        "summed_gap_nats",
        "beta_critical",
        "positive_beta_threshold",
        "preference_rule",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(analysis["critical_beta_records"])
    logger.success("Saved Bayesian critical-beta table to {}", path)


def write_bayesian_principled_confusion_plot(
    path: Path,
    analysis: dict[str, Any],
) -> None:
    """Plot uniform- and base-prior confusion matrices at principled beta."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traits = analysis["traits"]
    beta = analysis["principled_beta"]
    prior_specs = [
        ("uniform", "Uniform prior"),
        ("base_model_favorite_color", "Base favorite-color prior"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(19, 8.5), constrained_layout=True)
    image_handle = None
    for axis, (prior_name, title) in zip(axes, prior_specs):
        entry = posterior_sweep_entry(analysis, prior_name, beta)
        values = nested_matrix_to_array(
            entry["posterior_matrix"],
            traits,
            traits,
        )
        image_handle = axis.imshow(
            values,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            aspect="equal",
        )
        axis.set_xticks(range(len(traits)), traits, rotation=90, fontsize=7)
        axis.set_yticks(range(len(traits)), traits, fontsize=7)
        axis.set_xlabel("Decoded candidate trait")
        axis.set_ylabel("True teacher trait")
        axis.set_title(f"{title} (MAP accuracy {entry['map_accuracy']:.1%})")
    if image_handle is not None:
        fig.colorbar(
            image_handle,
            ax=list(axes),
            label="Posterior probability",
            shrink=0.82,
        )
    fig.suptitle(
        "Bayesian teacher-trait decoder confusion matrices - "
        f"principled beta={beta:g}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    logger.success("Saved Bayesian decoder confusion plot to {}", path)


def pairwise_true_probability(log_odds_true: float) -> float:
    """Convert true-vs-competitor log odds to a stable pairwise probability."""
    if log_odds_true >= 0.0:
        return 1.0 / (1.0 + math.exp(-log_odds_true))
    exponential = math.exp(log_odds_true)
    return exponential / (1.0 + exponential)


def write_critical_beta_plot(path: Path, analysis: dict[str, Any]) -> None:
    """Plot key pairwise decoder competitions across the beta sweep."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    beta_values = analysis["evaluated_betas"]
    positive_betas = [beta for beta in beta_values if beta > 0.0]
    prior_specs = [
        ("uniform", "Uniform prior"),
        ("base_model_favorite_color", "Base favorite-color prior"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for axis, (prior_name, title) in zip(axes, prior_specs):
        records = [
            record
            for record in analysis["critical_beta_records"]
            if record["prior"] == prior_name
        ]
        for record in records:
            prior_log_odds = -record["log_prior_competitor_over_true"]
            gap = record["average_gap_nats_per_prompt"]
            probabilities = [
                pairwise_true_probability(prior_log_odds + beta * gap)
                for beta in positive_betas
            ]
            axis.plot(
                positive_betas,
                probabilities,
                marker="o",
                linewidth=1.8,
                label=(
                    f"{record['true_trait']} vs {record['competitor']}"
                ),
            )
        axis.axhline(0.5, color="black", linewidth=0.8, alpha=0.55)
        axis.axvline(
            analysis["principled_beta"],
            color="#c73e1d",
            linewidth=1.2,
            linestyle="--",
            label=f"principled beta={analysis['principled_beta']:g}",
        )
        axis.set_xscale("log")
        axis.set_ylim(0.0, 1.02)
        axis.grid(alpha=0.22)
        axis.set_title(title)
        axis.set_xlabel("beta (effective prompt evidence)")
        if records:
            axis.legend(fontsize=7, loc="best")
    axes[0].set_ylabel("Pairwise posterior probability of true trait")
    fig.suptitle("Bayesian decoder key competitions across evidence strength")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    logger.success("Saved Bayesian critical-beta plot to {}", path)


def log_bayesian_decoder_summary(analysis: dict[str, Any]) -> None:
    """Log principled-beta accuracy and key critical beta values."""
    beta = analysis["principled_beta"]
    for prior_name in analysis["priors"]:
        entry = posterior_sweep_entry(analysis, prior_name, beta)
        logger.info(
            "Bayesian decoder [{} beta={}]: MAP accuracy={:.3f}, "
            "mean true posterior={:.6g}",
            prior_name,
            beta,
            entry["map_accuracy"],
            entry["mean_true_trait_posterior"],
        )
    logger.info(
        "Bayesian decoder critical beta values:\n{}",
        "\n".join(
            (
                f"{record['prior']}: {record['true_trait']} vs "
                f"{record['competitor']}: "
                f"gap={record['average_gap_nats_per_prompt']:.8g}, "
                f"beta_crit={record['beta_critical']}"
            )
            for record in analysis["critical_beta_records"]
        ),
    )


def write_bayesian_decoder_outputs(
    *,
    config: CheckpointEvaluationConfig,
    analysis: dict[str, Any],
) -> dict[str, Path]:
    """Write Bayesian decoder JSON, tables, and diagnostic plots."""
    paths = bayesian_decoder_output_paths(config)
    write_json(paths["json"], analysis)
    write_bayesian_decoder_scores_csv(paths["scores_csv"], analysis)
    write_bayesian_posterior_sweep_csv(
        paths["posterior_sweep_csv"],
        analysis,
    )
    write_critical_beta_csv(paths["critical_beta_csv"], analysis)
    beta = analysis["principled_beta"]
    write_csv_matrix(
        matrix=posterior_sweep_entry(
            analysis,
            "uniform",
            beta,
        )["posterior_matrix"],
        rows=analysis["traits"],
        columns=analysis["traits"],
        path=paths["uniform_confusion_csv"],
    )
    write_csv_matrix(
        matrix=posterior_sweep_entry(
            analysis,
            "base_model_favorite_color",
            beta,
        )["posterior_matrix"],
        rows=analysis["traits"],
        columns=analysis["traits"],
        path=paths["base_prior_confusion_csv"],
    )
    write_bayesian_principled_confusion_plot(
        paths["principled_confusion_plot"],
        analysis,
    )
    write_critical_beta_plot(paths["critical_beta_plot"], analysis)
    log_bayesian_decoder_summary(analysis)
    return paths


def shared_heatmap_max_abs(
    stage_results: dict[str, dict[str, Any]],
    colors: list[str],
    traits: list[str],
) -> float:
    """Return one symmetric color limit shared by every stage heatmap."""
    values = [
        abs(float(result["log_ratio_matrix"][color][trait]))
        for result in stage_results.values()
        for color in colors
        for trait in traits
    ]
    max_abs = max(values, default=1.0)
    return max_abs if max_abs > 0.0 and math.isfinite(max_abs) else 1.0


def nested_matrix_to_array(
    matrix: dict[str, dict[str, float]],
    rows: list[str],
    columns: list[str],
) -> np.ndarray:
    """Convert a nested row/column matrix dictionary to a numeric array."""
    return np.asarray(
        [[matrix[row][column] for column in columns] for row in rows],
        dtype=np.float64,
    )


def array_to_nested_matrix(
    matrix: np.ndarray,
    rows: list[str],
    columns: list[str],
) -> dict[str, dict[str, float]]:
    """Convert a numeric matrix to nested row/column dictionaries."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (len(rows), len(columns)):
        raise ValueError("Matrix shape does not match row and column labels.")
    return {
        row: {
            column: float(values[row_index, column_index])
            for column_index, column in enumerate(columns)
        }
        for row_index, row in enumerate(rows)
    }


def double_center_matrix(matrix: np.ndarray) -> np.ndarray:
    """Remove row means and column means while restoring the grand mean."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("Rank analysis requires a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Rank analysis matrix contains non-finite values.")
    return (
        values
        - np.mean(values, axis=1, keepdims=True)
        - np.mean(values, axis=0, keepdims=True)
        + np.mean(values)
    )


def build_double_centered_matrices(
    *,
    stages: list[EvaluationStage],
    stage_results: dict[str, dict[str, Any]],
    colors: list[str],
    traits: list[str],
) -> dict[str, np.ndarray]:
    """Build a double-centered log-ratio matrix for every stage."""
    return {
        stage.label: double_center_matrix(
            nested_matrix_to_array(
                stage_results[stage.label]["log_ratio_matrix"],
                colors,
                traits,
            )
        )
        for stage in stages
    }


def build_raw_logprob_double_centered_matrices(
    *,
    stages: list[EvaluationStage],
    raw_logprob_shift_results: dict[str, dict[str, Any]],
    colors: list[str],
    traits: list[str],
) -> dict[str, np.ndarray]:
    """Double-center cached question-averaged raw log-probability shifts."""
    return {
        stage.label: double_center_matrix(
            nested_matrix_to_array(
                raw_logprob_shift_results[stage.label][
                    "raw_logprob_shift_matrix"
                ],
                colors,
                traits,
            )
        )
        for stage in stages
    }


def write_double_centered_outputs(
    *,
    config: CheckpointEvaluationConfig,
    stages: list[EvaluationStage],
    matrices: dict[str, np.ndarray],
) -> float:
    """Write centered matrices as JSON, CSV, and shared-scale heatmaps."""
    max_abs = max(
        (
            float(np.max(np.abs(matrix)))
            for matrix in matrices.values()
            if matrix.size
        ),
        default=1.0,
    )
    if max_abs <= 0.0 or not math.isfinite(max_abs):
        max_abs = 1.0

    for stage in stages:
        paths = stage_output_paths(config, stage)
        nested_matrix = array_to_nested_matrix(
            matrices[stage.label],
            config.colors,
            config.traits,
        )
        write_json(
            paths["double_centered_json"],
            {
                "stage": stage.label,
                "checkpoint_step": stage.checkpoint_step,
                "rows": config.colors,
                "columns": config.traits,
                "centering": (
                    "M - row_mean(M) - column_mean(M) + grand_mean(M)"
                ),
                "double_centered_log_ratio_matrix": nested_matrix,
            },
        )
        write_csv_matrix(
            matrix=nested_matrix,
            rows=config.colors,
            columns=config.traits,
            path=paths["double_centered_csv"],
        )
        write_heatmap(
            matrix=nested_matrix,
            rows=config.colors,
            columns=config.traits,
            path=paths["double_centered_heatmap"],
            title=(
                "Double-centered favorite-color log probability ratio - "
                + (
                    "final adapter"
                    if stage.checkpoint_step is None
                    else f"checkpoint step {stage.checkpoint_step}"
                )
            ),
            max_abs=max_abs,
        )
    return max_abs


def compute_rank_metrics(matrix: np.ndarray) -> dict[str, Any]:
    """Compute singular values and five rank-dynamics metrics."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("Rank analysis requires a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Rank analysis matrix contains non-finite values.")

    singular_values = np.linalg.svd(values, compute_uv=False)
    energies = np.square(singular_values)
    total_energy = float(np.sum(energies))
    if total_energy > 0.0:
        energy_fractions = energies / total_energy
        positive_fractions = energy_fractions[energy_fractions > 0.0]
        effective_rank = float(
            np.exp(-np.sum(positive_fractions * np.log(positive_fractions)))
        )
        rank_1_explained = float(energy_fractions[0])
        rank_2_explained = float(np.sum(energy_fractions[:2]))
    else:
        effective_rank = 0.0
        rank_1_explained = 0.0
        rank_2_explained = 0.0

    sigma_1 = float(singular_values[0]) if singular_values.size else 0.0
    sigma_2 = float(singular_values[1]) if singular_values.size > 1 else 0.0
    return {
        "frobenius_norm": float(np.sqrt(total_energy)),
        "rank_1_explained": rank_1_explained,
        "rank_2_explained": rank_2_explained,
        "sigma_2_over_sigma_1": sigma_2 / sigma_1 if sigma_1 > 0.0 else 0.0,
        "effective_rank": effective_rank,
        "singular_values": singular_values.tolist(),
    }


def infer_final_training_step(
    training_output_dir: Path,
    traits: list[str],
    available_checkpoint_steps: list[int],
) -> int | None:
    """Infer the final optimizer step from the latest checkpoint metadata."""
    if not available_checkpoint_steps:
        return None
    latest_step = max(available_checkpoint_steps)
    total_steps_by_trait: dict[str, int] = {}
    for trait in traits:
        metadata_path = (
            training_output_dir
            / "trainer"
            / slugify(trait)
            / f"checkpoint-step-{latest_step:06d}"
            / "metadata.json"
        )
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        total_steps = metadata.get("total_steps")
        if isinstance(total_steps, int) and total_steps > 0:
            total_steps_by_trait[trait] = total_steps

    if len(total_steps_by_trait) != len(traits):
        logger.warning(
            "Could not infer the final optimizer step for every trait; "
            "the rank plot will label the last point as final."
        )
        return None
    unique_steps = sorted(set(total_steps_by_trait.values()))
    if len(unique_steps) > 1:
        logger.warning(
            "Traits report different final optimizer steps {}; using the maximum.",
            unique_steps,
        )
    return max(unique_steps)


def build_rank_analysis(
    *,
    stages: list[EvaluationStage],
    stage_results: dict[str, dict[str, Any]],
    colors: list[str],
    traits: list[str],
    final_training_step: int | None,
) -> dict[str, Any]:
    """Build raw and double-centered rank metrics for every stage."""
    stage_metrics = []
    for stage in stages:
        raw_matrix = nested_matrix_to_array(
            stage_results[stage.label]["log_ratio_matrix"],
            colors,
            traits,
        )
        centered_matrix = double_center_matrix(raw_matrix)
        stage_metrics.append(
            {
                "label": stage.label,
                "checkpoint_step": stage.checkpoint_step,
                "training_step": (
                    stage.checkpoint_step
                    if stage.checkpoint_step is not None
                    else final_training_step
                ),
                "raw": compute_rank_metrics(raw_matrix),
                "double_centered": compute_rank_metrics(centered_matrix),
            }
        )
    return {
        "matrix": "favorite-color log probability-ratio matrix",
        "centering": (
            "M - row_mean(M) - column_mean(M) + grand_mean(M)"
        ),
        "final_training_step": final_training_step,
        "stages": stage_metrics,
    }


def compute_two_component_svd(matrix: np.ndarray) -> dict[str, np.ndarray]:
    """Compute two-component row and column scores from one matrix SVD."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("SVD visualization requires a non-empty matrix.")
    if not np.all(np.isfinite(values)):
        raise ValueError("SVD visualization matrix contains non-finite values.")

    left_vectors, singular_values, right_vectors_t = np.linalg.svd(
        values,
        full_matrices=False,
    )
    component_count = min(2, singular_values.size)
    left_components = np.zeros((values.shape[0], 2), dtype=np.float64)
    right_components = np.zeros((values.shape[1], 2), dtype=np.float64)
    component_singular_values = np.zeros(2, dtype=np.float64)
    left_components[:, :component_count] = left_vectors[:, :component_count]
    right_components[:, :component_count] = right_vectors_t[
        :component_count
    ].T
    component_singular_values[:component_count] = singular_values[
        :component_count
    ]
    return {
        "singular_values": singular_values,
        "left_components": left_components,
        "right_components": right_components,
        "row_scores": left_components * component_singular_values,
        "column_scores": right_components * component_singular_values,
    }


def align_svd_signs(
    decomposition: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Align component signs to a reference SVD without changing magnitudes."""
    aligned = {key: np.array(value, copy=True) for key, value in decomposition.items()}
    for component_index in range(2):
        similarity = float(
            np.dot(
                aligned["left_components"][:, component_index],
                reference["left_components"][:, component_index],
            )
            + np.dot(
                aligned["right_components"][:, component_index],
                reference["right_components"][:, component_index],
            )
        )
        if similarity < 0.0:
            for key in (
                "left_components",
                "right_components",
                "row_scores",
                "column_scores",
            ):
                aligned[key][:, component_index] *= -1.0
    return aligned


def select_representative_svd_stages(
    stages: list[EvaluationStage],
) -> list[EvaluationStage]:
    """Select 80/240/400 when present, otherwise earliest/middle/latest."""
    if len(stages) <= 3:
        return list(stages)

    selected_labels = set()
    selected = []
    for preferred_step in (80, 240, 400):
        for stage in stages:
            if stage.checkpoint_step == preferred_step:
                selected.append(stage)
                selected_labels.add(stage.label)
                break
    fallback_indices = (0, len(stages) // 2, len(stages) - 1)
    for index in fallback_indices:
        stage = stages[index]
        if stage.label not in selected_labels:
            selected.append(stage)
            selected_labels.add(stage.label)
        if len(selected) == 3:
            break
    stage_order = {stage.label: index for index, stage in enumerate(stages)}
    return sorted(selected[:3], key=lambda stage: stage_order[stage.label])


def build_svd_analysis(
    *,
    stages: list[EvaluationStage],
    matrices: dict[str, np.ndarray],
    colors: list[str],
    traits: list[str],
    final_training_step: int | None,
    matrix_description: str = (
        "double-centered favorite-color log probability-ratio matrix"
    ),
    plot_title: str = "Double-centered favorite-color matrix SVD",
) -> dict[str, Any]:
    """Build sign-aligned row and column SVD coordinates for all stages."""
    decompositions = {
        stage.label: compute_two_component_svd(matrices[stage.label])
        for stage in stages
    }
    reference_label = stages[-1].label
    reference = decompositions[reference_label]
    representative_stages = select_representative_svd_stages(stages)
    serialized_stages = []
    for stage in stages:
        decomposition = align_svd_signs(
            decompositions[stage.label],
            reference,
        )
        singular_values = decomposition["singular_values"]
        total_energy = float(np.sum(np.square(singular_values)))
        component_energy = (
            np.square(singular_values[:2]) / total_energy
            if total_energy > 0.0
            else np.zeros(min(2, singular_values.size), dtype=np.float64)
        )
        component_energy_padded = np.zeros(2, dtype=np.float64)
        component_energy_padded[: component_energy.size] = component_energy
        serialized_stages.append(
            {
                "label": stage.label,
                "checkpoint_step": stage.checkpoint_step,
                "training_step": (
                    stage.checkpoint_step
                    if stage.checkpoint_step is not None
                    else final_training_step
                ),
                "singular_values": singular_values.tolist(),
                "component_energy_fractions": component_energy_padded.tolist(),
                "readout_color_coordinates": {
                    color: {
                        "pc1": float(decomposition["row_scores"][index, 0]),
                        "pc2": float(decomposition["row_scores"][index, 1]),
                    }
                    for index, color in enumerate(colors)
                },
                "fine_tuning_trait_coordinates": {
                    trait: {
                        "pc1": float(
                            decomposition["column_scores"][index, 0]
                        ),
                        "pc2": float(
                            decomposition["column_scores"][index, 1]
                        ),
                    }
                    for index, trait in enumerate(traits)
                },
            }
        )
    return {
        "matrix": matrix_description,
        "plot_title": plot_title,
        "decomposition": "M_centered = U Sigma V^T",
        "readout_coordinates": "U[:, :2] Sigma[:2]",
        "fine_tuning_trait_coordinates": "V[:, :2] Sigma[:2]",
        "alignment": "component signs aligned to the reference stage",
        "sign_alignment_reference": reference_label,
        "representative_stages": [
            stage.label for stage in representative_stages
        ],
        "stages": serialized_stages,
    }


def svd_analysis_output_paths(
    config: CheckpointEvaluationConfig,
    representative_labels: list[str],
    output_stem: str = SVD_COORDINATES_STEM,
) -> dict[str, Any]:
    """Return paths for SVD coordinates and representative plots."""
    prefix = config.output_dir / output_stem
    return {
        "json": prefix.with_suffix(".json"),
        "csv": prefix.with_suffix(".csv"),
        "plots": {
            label: prefix.with_name(f"{prefix.name}_{label}.png")
            for label in representative_labels
        },
    }


def write_svd_coordinates_csv(path: Path, analysis: dict[str, Any]) -> None:
    """Write SVD row and column coordinates in long-form CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "stage",
                "checkpoint_step",
                "training_step",
                "point_type",
                "color",
                "pc1",
                "pc2",
            ]
        )
        for stage in analysis["stages"]:
            for point_type, coordinate_key in (
                ("readout_color", "readout_color_coordinates"),
                ("fine_tuning_trait", "fine_tuning_trait_coordinates"),
            ):
                for color, coordinates in stage[coordinate_key].items():
                    writer.writerow(
                        [
                            stage["label"],
                            stage["checkpoint_step"],
                            stage["training_step"],
                            point_type,
                            color,
                            coordinates["pc1"],
                            coordinates["pc2"],
                        ]
                    )
    logger.success("Saved SVD coordinates CSV to {}", path)


def add_labeled_svd_points(
    *,
    axis: Any,
    coordinates: dict[str, dict[str, float]],
    marker: str,
    color: str,
    legend_label: str,
    label_side: str = "alternating",
    font_size: float = 7.0,
) -> None:
    """Add labeled SVD points to one matplotlib axis."""
    labels = list(coordinates)
    x_values = [coordinates[label]["pc1"] for label in labels]
    y_values = [coordinates[label]["pc2"] for label in labels]
    point_colors = [
        FAVORITE_COLOR_PLOT_COLORS.get(label.lower(), color)
        for label in labels
    ]
    axis.scatter(
        x_values,
        y_values,
        marker=marker,
        c=point_colors,
        edgecolors="#263238",
        linewidths=0.55,
        s=42,
        alpha=0.85,
        label=legend_label,
    )
    if label_side == "left":
        offsets = ((-6, 5), (-6, -10))
    elif label_side == "right":
        offsets = ((6, 5), (6, -10))
    else:
        offsets = ((5, 5), (5, -10), (-5, 5), (-5, -10))
    for index, label in enumerate(labels):
        offset = offsets[index % len(offsets)]
        axis.annotate(
            label,
            (x_values[index], y_values[index]),
            xytext=offset,
            textcoords="offset points",
            fontsize=font_size,
            ha="left" if offset[0] > 0 else "right",
        )


def write_svd_plots(
    *,
    paths: dict[str, Path],
    analysis: dict[str, Any],
) -> None:
    """Write separate row, column, and combined SVD views per selected stage."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages_by_label = {stage["label"]: stage for stage in analysis["stages"]}
    selected_stages = [stages_by_label[label] for label in paths]
    all_coordinates = [
        coordinates
        for stage in selected_stages
        for coordinate_key in (
            "readout_color_coordinates",
            "fine_tuning_trait_coordinates",
        )
        for coordinates in stage[coordinate_key].values()
    ]
    max_abs_pc1 = max(
        (abs(coordinates["pc1"]) for coordinates in all_coordinates),
        default=1.0,
    )
    max_abs_pc2 = max(
        (abs(coordinates["pc2"]) for coordinates in all_coordinates),
        default=1.0,
    )
    max_abs_pc1 = max(max_abs_pc1 * 1.18, 1e-6)
    max_abs_pc2 = max(max_abs_pc2 * 1.18, 1e-6)

    for label, path in paths.items():
        stage = stages_by_label[label]
        energy = stage["component_energy_fractions"]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))
        add_labeled_svd_points(
            axis=axes[0],
            coordinates=stage["readout_color_coordinates"],
            marker="o",
            color="#c73e1d",
            legend_label="Readout colors",
        )
        add_labeled_svd_points(
            axis=axes[1],
            coordinates=stage["fine_tuning_trait_coordinates"],
            marker="^",
            color="#2364aa",
            legend_label="Fine-tuning traits",
        )
        add_labeled_svd_points(
            axis=axes[2],
            coordinates=stage["readout_color_coordinates"],
            marker="o",
            color="#c73e1d",
            legend_label="Readout colors",
            label_side="left",
            font_size=6.5,
        )
        add_labeled_svd_points(
            axis=axes[2],
            coordinates=stage["fine_tuning_trait_coordinates"],
            marker="^",
            color="#2364aa",
            legend_label="Fine-tuning traits",
            label_side="right",
            font_size=6.5,
        )
        for axis, title in zip(
            axes,
            ("Readout-color PCA", "Fine-tuning-trait PCA", "SVD biplot"),
        ):
            axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.4)
            axis.axvline(0.0, color="black", linewidth=0.7, alpha=0.4)
            axis.grid(alpha=0.2)
            axis.set_xlim(-max_abs_pc1, max_abs_pc1)
            axis.set_ylim(-max_abs_pc2, max_abs_pc2)
            axis.set_title(title)
            axis.set_xlabel(f"PC1 score ({energy[0]:.1%} energy)")
            axis.set_ylabel(f"PC2 score ({energy[1]:.1%} energy)")
            axis.legend(loc="best", fontsize=8)
        display_step = (
            stage["training_step"]
            if stage["training_step"] is not None
            else stage["label"]
        )
        fig.suptitle(
            f"{analysis['plot_title']} - training step {display_step}"
        )
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=220)
        plt.close(fig)
        logger.success("Saved SVD visualization to {}", path)


def write_svd_analysis(
    config: CheckpointEvaluationConfig,
    analysis: dict[str, Any],
    output_stem: str = SVD_COORDINATES_STEM,
) -> dict[str, Any]:
    """Write SVD coordinates and representative visualizations."""
    paths = svd_analysis_output_paths(
        config,
        analysis["representative_stages"],
        output_stem,
    )
    write_json(paths["json"], analysis)
    write_svd_coordinates_csv(paths["csv"], analysis)
    write_svd_plots(paths=paths["plots"], analysis=analysis)
    return paths


def rank_analysis_output_paths(
    config: CheckpointEvaluationConfig,
) -> dict[str, Path]:
    """Return paths for rank analysis data and plot outputs."""
    prefix = config.output_dir / RANK_METRICS_STEM
    return {
        "json": prefix.with_suffix(".json"),
        "csv": prefix.with_suffix(".csv"),
        "plot": prefix.with_suffix(".png"),
    }


def write_rank_metrics_csv(path: Path, analysis: dict[str, Any]) -> None:
    """Write one row per stage and matrix variant for rank metrics."""
    metric_names = [
        "frobenius_norm",
        "rank_1_explained",
        "rank_2_explained",
        "sigma_2_over_sigma_1",
        "effective_rank",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "stage",
                "checkpoint_step",
                "training_step",
                "matrix_variant",
                *metric_names,
            ]
        )
        for stage in analysis["stages"]:
            for variant in ("raw", "double_centered"):
                metrics = stage[variant]
                writer.writerow(
                    [
                        stage["label"],
                        stage["checkpoint_step"],
                        stage["training_step"],
                        variant,
                        *[metrics[name] for name in metric_names],
                    ]
                )
    logger.success("Saved rank metrics CSV to {}", path)


def write_rank_metrics_plot(path: Path, analysis: dict[str, Any]) -> None:
    """Plot the five raw and double-centered rank metrics over training."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = analysis["stages"]
    known_steps = all(stage["training_step"] is not None for stage in stages)
    x_values = (
        [stage["training_step"] for stage in stages]
        if known_steps
        else list(range(len(stages)))
    )
    tick_labels = [
        str(stage["training_step"])
        if stage["training_step"] is not None
        else "final"
        for stage in stages
    ]
    metric_specs = [
        ("frobenius_norm", "Frobenius norm"),
        ("rank_1_explained", "Rank-1 explained energy"),
        ("rank_2_explained", "Rank-2 explained energy"),
        ("sigma_2_over_sigma_1", "Second-direction strength"),
        ("effective_rank", "Effective rank"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), sharex=True)
    flat_axes = list(axes.flat)
    for axis, (metric_name, title) in zip(flat_axes, metric_specs):
        for variant, label, color in (
            ("raw", "Raw", "#2364aa"),
            ("double_centered", "Double-centered", "#c73e1d"),
        ):
            axis.plot(
                x_values,
                [stage[variant][metric_name] for stage in stages],
                marker="o",
                linewidth=2,
                label=label,
                color=color,
            )
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.set_xticks(x_values, tick_labels)
        if metric_name in {
            "rank_1_explained",
            "rank_2_explained",
            "sigma_2_over_sigma_1",
        }:
            axis.set_ylim(0.0, 1.05)
        else:
            axis.set_ylim(bottom=0.0)
    flat_axes[0].legend()
    flat_axes[-1].axis("off")
    for axis in flat_axes[3:5]:
        axis.set_xlabel("Training step" if known_steps else "Evaluation stage")
    fig.suptitle("Favorite-color log-ratio matrix rank dynamics")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    logger.success("Saved rank metrics plot to {}", path)


def log_rank_analysis(analysis: dict[str, Any]) -> None:
    """Log the five rank metrics for each stage and matrix variant."""
    for stage in analysis["stages"]:
        for variant in ("raw", "double_centered"):
            metrics = stage[variant]
            logger.info(
                "Rank metrics [{} {}]: ||M||_F={:.6g}, p1={:.6g}, "
                "p1+p2={:.6g}, sigma2/sigma1={:.6g}, effective_rank={:.6g}",
                stage["label"],
                variant,
                metrics["frobenius_norm"],
                metrics["rank_1_explained"],
                metrics["rank_2_explained"],
                metrics["sigma_2_over_sigma_1"],
                metrics["effective_rank"],
            )


def write_rank_analysis(
    config: CheckpointEvaluationConfig,
    analysis: dict[str, Any],
) -> dict[str, Path]:
    """Write rank metrics JSON, CSV, plot, and log table."""
    paths = rank_analysis_output_paths(config)
    write_json(paths["json"], analysis)
    write_rank_metrics_csv(paths["csv"], analysis)
    write_rank_metrics_plot(paths["plot"], analysis)
    log_rank_analysis(analysis)
    return paths


def stage_title(stage: EvaluationStage) -> str:
    """Return a concise heatmap title for one evaluation stage."""
    if stage.checkpoint_step is None:
        return "Favorite-color log probability ratio - final adapter"
    return (
        "Favorite-color log probability ratio - "
        f"checkpoint step {stage.checkpoint_step}"
    )


def write_comparable_heatmaps(
    *,
    config: CheckpointEvaluationConfig,
    stages: list[EvaluationStage],
    stage_results: dict[str, dict[str, Any]],
) -> float:
    """Write all stage heatmaps with one shared symmetric color scale."""
    max_abs = shared_heatmap_max_abs(stage_results, config.colors, config.traits)
    for stage in stages:
        write_heatmap(
            matrix=stage_results[stage.label]["log_ratio_matrix"],
            rows=config.colors,
            columns=config.traits,
            path=stage_output_paths(config, stage)["heatmap"],
            title=stage_title(stage),
            max_abs=max_abs,
        )
    return max_abs


def clear_accelerator_cache() -> None:
    """Collect Python objects and clear CUDA cache."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        logger.debug("Skipping CUDA cache cleanup.")


def serialize_manifest_config(
    config: CheckpointEvaluationConfig,
) -> dict[str, Any]:
    """Serialize config paths for the run manifest."""
    data = asdict(config)
    data["training_output_dir"] = str(config.training_output_dir)
    data["output_dir"] = str(config.output_dir)
    data["teacher_logprob_cache_dir"] = str(
        config.teacher_logprob_cache_dir
    )
    return data


def run(config: CheckpointEvaluationConfig) -> dict[str, Any]:
    """Evaluate selected checkpoints and final adapters."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    available_steps = discover_common_checkpoint_steps(
        config.training_output_dir,
        config.traits,
    )
    logger.info("Common checkpoint steps available: {}", available_steps)
    unselected_steps = [
        step for step in available_steps if step not in config.checkpoint_steps
    ]
    if unselected_steps:
        logger.warning("Available checkpoint steps not selected: {}", unselected_steps)

    stages = build_evaluation_stages(config)
    stage_results: dict[str, dict[str, Any]] = {}
    pending_stages = []
    for stage in stages:
        paths = stage_output_paths(config, stage)
        cached_result = None
        if config.skip_existing and stage_result_is_reusable(config, stage):
            cached_result = load_stage_result(paths["json"], config, stage)
        if cached_result is None:
            pending_stages.append(stage)
        else:
            logger.info("Reusing completed evaluation stage {}", stage.label)
            stage_results[stage.label] = cached_result

    correlation_stage = stages[-1]
    projection_analysis = None
    projection_estimate = None
    if config.compute_update_readout_projection:
        projection_analysis = load_reusable_update_readout_projection(
            config=config,
            stage=correlation_stage,
        )
    projection_is_pending = (
        config.compute_update_readout_projection
        and projection_analysis is None
    )

    if pending_stages or projection_is_pending:
        import torch

        device = resolve_device(torch, config.device)
        torch_dtype = resolve_torch_dtype(torch, config.torch_dtype)
        logger.info("Loading shared base model {}", config.base_model_id)
        base_model, tokenizer = load_direct_model_and_tokenizer(
            model_id=config.base_model_id,
            device=device,
            torch_dtype=torch_dtype,
        )
        model = base_model
        model_is_peft = False
        base_scores = None
        if pending_stages:
            base_cache_path = (
                config.output_dir / "base_favorite_color_scores.json"
            )
            cache_metadata = base_cache_metadata(config)
            base_scores = load_base_score_cache(
                base_cache_path,
                cache_metadata,
            )
            if base_scores is None:
                base_scores = score_model(
                    model_label="base",
                    model=base_model,
                    tokenizer=tokenizer,
                    device=device,
                    config=ratio_config_for_stage(config, pending_stages[0]),
                )
                save_base_score_cache(
                    base_cache_path,
                    cache_metadata,
                    base_scores,
                )

        try:
            for stage in pending_stages:
                if base_scores is None:
                    raise RuntimeError("Base scores were not initialized.")
                model, model_is_peft, names = load_stage_adapters(
                    model=model,
                    model_is_peft=model_is_peft,
                    stage=stage,
                    traits=config.traits,
                )
                try:
                    result = score_evaluation_stage(
                        config=config,
                        stage=stage,
                        ratio_config=ratio_config_for_stage(config, stage),
                        base_scores=base_scores,
                        model=model,
                        tokenizer=tokenizer,
                        adapter_names=names,
                        device=device,
                    )
                    write_stage_tables(
                        config=config,
                        stage=stage,
                        result=result,
                    )
                    stage_results[stage.label] = result
                finally:
                    unload_stage_adapters(model, names)

            if projection_is_pending:
                model, model_is_peft, names = load_stage_adapters(
                    model=model,
                    model_is_peft=model_is_peft,
                    stage=correlation_stage,
                    traits=config.traits,
                )
                try:
                    projection_estimate = estimate_update_readout_projection(
                        config=config,
                        stage=correlation_stage,
                        model=model,
                        tokenizer=tokenizer,
                        adapter_names=names,
                        device=device,
                    )
                finally:
                    unload_stage_adapters(model, names)
        finally:
            del model
            del base_model
            clear_accelerator_cache()

    missing_results = [
        stage.label for stage in stages if stage.label not in stage_results
    ]
    if missing_results:
        raise RuntimeError(f"Missing completed stage results: {missing_results}")

    raw_logprob_shift_results = build_raw_logprob_shift_results(
        stages=stages,
        stage_results=stage_results,
        colors=config.colors,
        traits=config.traits,
        epsilon=config.epsilon,
    )
    projection_output_path_map = None
    if config.compute_update_readout_projection:
        if projection_analysis is None:
            if projection_estimate is None:
                raise RuntimeError(
                    "Update/readout projection estimate was not computed."
                )
            projection_analysis = build_update_readout_projection_analysis(
                config=config,
                stage=correlation_stage,
                estimate=projection_estimate,
                raw_behavior_matrix=raw_logprob_shift_results[
                    correlation_stage.label
                ]["raw_logprob_shift_matrix"],
            )
        projection_output_path_map = (
            write_update_readout_projection_outputs(
                config=config,
                analysis=projection_analysis,
            )
        )
        log_update_readout_projection_summary(projection_analysis)
    raw_logprob_heatmap_max_abs = write_raw_logprob_shift_outputs(
        config=config,
        stages=stages,
        results=raw_logprob_shift_results,
    )
    heatmap_max_abs = write_comparable_heatmaps(
        config=config,
        stages=stages,
        stage_results=stage_results,
    )
    if config.print_log_ratio_tables:
        for stage in stages:
            log_matrix(
                title=(
                    f"Log probability-ratio matrix [{stage.label}]: "
                    "rows=color u, columns=fine-tuning trait t"
                ),
                matrix=stage_results[stage.label]["log_ratio_matrix"],
                rows=config.colors,
                columns=config.traits,
            )
            log_matrix(
                title=(
                    f"Raw log-probability shift matrix [{stage.label}]: "
                    "rows=color u, columns=fine-tuning trait t"
                ),
                matrix=raw_logprob_shift_results[stage.label][
                    "raw_logprob_shift_matrix"
                ],
                rows=config.colors,
                columns=config.traits,
            )

    final_training_step = infer_final_training_step(
        config.training_output_dir,
        config.traits,
        available_steps,
    )
    double_centered_matrices = build_double_centered_matrices(
        stages=stages,
        stage_results=stage_results,
        colors=config.colors,
        traits=config.traits,
    )
    double_centered_heatmap_max_abs = write_double_centered_outputs(
        config=config,
        stages=stages,
        matrices=double_centered_matrices,
    )
    rank_analysis = build_rank_analysis(
        stages=stages,
        stage_results=stage_results,
        colors=config.colors,
        traits=config.traits,
        final_training_step=final_training_step,
    )
    rank_output_paths = write_rank_analysis(config, rank_analysis)
    svd_analysis = build_svd_analysis(
        stages=stages,
        matrices=double_centered_matrices,
        colors=config.colors,
        traits=config.traits,
        final_training_step=final_training_step,
        matrix_description=(
            f"double-centered {config.probability_mode} favorite-color log "
            "probability-ratio matrix built from question-mean probabilities"
        ),
        plot_title=(
            "Double-centered favorite-color "
            f"{config.probability_mode} ratio SVD"
        ),
    )
    svd_output_paths = write_svd_analysis(config, svd_analysis)
    raw_logprob_double_centered_matrices = (
        build_raw_logprob_double_centered_matrices(
            stages=stages,
            raw_logprob_shift_results=raw_logprob_shift_results,
            colors=config.colors,
            traits=config.traits,
        )
    )
    raw_logprob_svd_analysis = build_svd_analysis(
        stages=stages,
        matrices=raw_logprob_double_centered_matrices,
        colors=config.colors,
        traits=config.traits,
        final_training_step=final_training_step,
        matrix_description=(
            "double-centered raw favorite-color log-probability-shift matrix "
            "built by averaging question-level log ratios"
        ),
        plot_title="Double-centered favorite-color raw logprob-shift SVD",
    )
    raw_logprob_svd_output_paths = write_svd_analysis(
        config,
        raw_logprob_svd_analysis,
        RAW_LOGPROB_SVD_COORDINATES_STEM,
    )
    bayesian_decoder_analysis = build_bayesian_decoder_analysis(
        config=config,
        stage_result=stage_results[correlation_stage.label],
    )
    bayesian_decoder_output_path_map = write_bayesian_decoder_outputs(
        config=config,
        analysis=bayesian_decoder_analysis,
    )
    broad_boost_analysis = load_reusable_broad_boost_analysis(
        config=config,
        stage=correlation_stage,
    )
    if broad_boost_analysis is None:
        broad_boost_analysis = build_broad_boost_correlation_analysis(
            config=config,
            stage=correlation_stage,
            stage_result=stage_results[correlation_stage.label],
            available_checkpoint_steps=available_steps,
        )
        broad_boost_output_path_map = write_broad_boost_correlation_outputs(
            config=config,
            analysis=broad_boost_analysis,
        )
    else:
        broad_boost_output_path_map = broad_boost_output_paths(config)
    log_broad_boost_correlation_table(broad_boost_analysis)

    manifest = {
        "config": serialize_manifest_config(config),
        "available_checkpoint_steps": available_steps,
        "selected_checkpoint_steps": config.checkpoint_steps,
        "unselected_checkpoint_steps": unselected_steps,
        "shared_heatmap_max_abs": heatmap_max_abs,
        "raw_logprob_shared_heatmap_max_abs": raw_logprob_heatmap_max_abs,
        "double_centered_shared_heatmap_max_abs": (
            double_centered_heatmap_max_abs
        ),
        "stages": [
            {
                "label": stage.label,
                "checkpoint_step": stage.checkpoint_step,
                "outputs": {
                    key: str(path)
                    for key, path in stage_output_paths(config, stage).items()
                },
            }
            for stage in stages
        ],
        "rank_analysis": {
            "final_training_step": final_training_step,
            "outputs": {
                key: str(path) for key, path in rank_output_paths.items()
            },
        },
        "svd_analysis": {
            "matrix": svd_analysis["matrix"],
            "sign_alignment_reference": svd_analysis[
                "sign_alignment_reference"
            ],
            "representative_stages": svd_analysis[
                "representative_stages"
            ],
            "outputs": {
                "json": str(svd_output_paths["json"]),
                "csv": str(svd_output_paths["csv"]),
                "plots": {
                    label: str(path)
                    for label, path in svd_output_paths["plots"].items()
                },
            },
        },
        "raw_logprob_svd_analysis": {
            "matrix": raw_logprob_svd_analysis["matrix"],
            "sign_alignment_reference": raw_logprob_svd_analysis[
                "sign_alignment_reference"
            ],
            "representative_stages": raw_logprob_svd_analysis[
                "representative_stages"
            ],
            "outputs": {
                "json": str(raw_logprob_svd_output_paths["json"]),
                "csv": str(raw_logprob_svd_output_paths["csv"]),
                "plots": {
                    label: str(path)
                    for label, path in raw_logprob_svd_output_paths[
                        "plots"
                    ].items()
                },
            },
        },
        "update_readout_projection_analysis": (
            None
            if projection_analysis is None
            or projection_output_path_map is None
            else {
                "stage": correlation_stage.label,
                "estimator": projection_analysis["estimator"],
                "finite_difference_step": projection_analysis[
                    "analysis_identity"
                ]["finite_difference_step"],
                "correlations": projection_analysis["correlations"],
                "key_comparisons": projection_analysis[
                    "key_comparisons"
                ],
                "outputs": {
                    key: str(path)
                    for key, path in projection_output_path_map.items()
                },
            }
        ),
        "bayesian_decoder_analysis": {
            "score_definition": bayesian_decoder_analysis[
                "score_definition"
            ],
            "principled_beta": bayesian_decoder_analysis[
                "principled_beta"
            ],
            "priors": list(bayesian_decoder_analysis["priors"]),
            "key_competitions": bayesian_decoder_analysis[
                "key_competitions"
            ],
            "outputs": {
                key: str(path)
                for key, path in bayesian_decoder_output_path_map.items()
            },
        },
        "broad_boost_correlation_analysis": {
            "stage": correlation_stage.label,
            "correlation_method": broad_boost_analysis[
                "correlation_method"
            ],
            "outputs": {
                key: str(path)
                for key, path in broad_boost_output_path_map.items()
            },
        },
    }
    write_json(config.output_dir / "manifest.json", manifest)
    return manifest


def validate_config(config: CheckpointEvaluationConfig) -> None:
    """Validate checkpoint evaluation configuration."""
    if not config.colors:
        raise ValueError("At least one color candidate is required.")
    if not config.traits:
        raise ValueError("At least one fine-tuning trait is required.")
    if not config.questions:
        raise ValueError("At least one evaluation question is required.")
    if not config.checkpoint_steps and not config.include_final:
        raise ValueError("Select at least one checkpoint or include final adapters.")
    if config.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if config.epsilon < 0.0 or not math.isfinite(config.epsilon):
        raise ValueError("--epsilon must be finite and non-negative.")
    if (
        config.projection_finite_difference_step <= 0.0
        or not math.isfinite(config.projection_finite_difference_step)
    ):
        raise ValueError(
            "--projection-finite-difference-step must be finite and positive."
        )
    if config.probability_mode not in {"normalized", "raw"}:
        raise ValueError("--probability-mode must be normalized or raw.")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate favorite-color log probability-ratio matrices for shared "
            "LoRA checkpoint steps and final adapters."
        )
    )
    parser.add_argument(
        "--training-output-dir",
        type=Path,
        default=DEFAULT_TRAINING_OUTPUT_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-step",
        "--checkpoint-steps",
        dest="checkpoint_steps",
        action="append",
        help=(
            "Checkpoint optimizer steps. Repeat or pass comma-separated values. "
            "Defaults to 80,160,240,320."
        ),
    )
    parser.add_argument("--no-final", action="store_true")
    parser.add_argument("--color", "--colors", dest="colors", action="append")
    parser.add_argument(
        "--trait",
        "--traits",
        dest="traits",
        action="append",
    )
    parser.add_argument(
        "--question",
        "--questions",
        dest="questions",
        action="append",
    )
    parser.add_argument("--questions-path", type=Path, default=None)
    parser.add_argument("--base-model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--probability-mode",
        choices=["normalized", "raw"],
        default="normalized",
    )
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--candidate-prefix", default="")
    parser.add_argument("--assistant-prefill", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument(
        "--teacher-logprob-cache-dir",
        type=Path,
        default=DEFAULT_TEACHER_LOGPROB_CACHE_DIR,
        help=(
            "Root containing cached neutral and trait 00-99 teacher logprobs "
            "used for teacher-KL correlations."
        ),
    )
    parser.add_argument(
        "--no-update-readout-projection",
        action="store_true",
        help="Skip the LoRA update/readout directional projection analysis.",
    )
    parser.add_argument(
        "--projection-finite-difference-step",
        type=float,
        default=DEFAULT_PROJECTION_FINITE_DIFFERENCE_STEP,
        help=(
            "Symmetric LoRA multiplier h used to estimate the directional "
            "derivative around the base model."
        ),
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-log-ratio-tables", action="store_true")
    return parser


def build_config(args: argparse.Namespace) -> CheckpointEvaluationConfig:
    """Build config from parsed CLI arguments."""
    training_output_dir = args.training_output_dir
    output_dir = args.output_dir or (
        training_output_dir / DEFAULT_OUTPUT_SUBDIR
    )
    return CheckpointEvaluationConfig(
        training_output_dir=training_output_dir,
        output_dir=output_dir,
        checkpoint_steps=parse_checkpoint_steps(
            args.checkpoint_steps,
            DEFAULT_CHECKPOINT_STEPS,
        ),
        include_final=not args.no_final,
        colors=parse_csv_values(args.colors, DEFAULT_COLORS),
        traits=parse_csv_values(args.traits, DEFAULT_COLORS),
        questions=load_questions(args),
        base_model_id=args.base_model_id,
        probability_mode=args.probability_mode,
        epsilon=args.epsilon,
        candidate_prefix=args.candidate_prefix,
        assistant_prefill=parse_optional_text(args.assistant_prefill),
        batch_size=args.batch_size,
        device=args.device,
        torch_dtype=args.torch_dtype,
        teacher_logprob_cache_dir=args.teacher_logprob_cache_dir,
        compute_update_readout_projection=(
            not args.no_update_readout_projection
        ),
        projection_finite_difference_step=(
            args.projection_finite_difference_step
        ),
        skip_existing=args.skip_existing,
        print_log_ratio_tables=not args.no_log_ratio_tables,
    )


def main() -> None:
    """Run the checkpoint evaluator CLI."""
    parser = build_parser()
    try:
        config = build_config(parser.parse_args())
        validate_config(config)
        run(config)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
