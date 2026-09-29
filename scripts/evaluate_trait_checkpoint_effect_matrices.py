#!/usr/bin/env python3
"""Evaluate generic trait checkpoint matrices and compute the catalog's M10."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compute_favorite_color_probability_ratio_matrix import (  # noqa: E402
    RatioMatrixConfig,
    build_chat_prompt,
    candidate_texts,
    score_model,
)
from evaluate_color_checkpoint_probability_ratio_matrices import (  # noqa: E402
    CheckpointEvaluationConfig,
    EvaluationStage,
    build_raw_logprob_shift_result,
    cached_question_mean_log_ratio_matrix,
    clear_accelerator_cache,
    discover_common_checkpoint_steps,
    find_lora_scaling_entries,
    infer_final_training_step,
    load_stage_adapters,
    score_evaluation_stage,
    set_lora_scaling_multiplier,
    unload_stage_adapters,
)
from find_owl_output_distribution_entanglement import (  # noqa: E402
    DEFAULT_MODEL_ID,
    load_direct_model_and_tokenizer,
    parse_csv_values,
)
from metric_exact_trie_scorer import (  # noqa: E402
    autotune_exact_trie_batch_pair,
    native_metric_config_identity,
)
from metric_batch_autotune import (  # noqa: E402
    BatchCandidateRejectedError,
    is_out_of_memory_error,
)
from score_birthday_year_probabilities import (  # noqa: E402
    resolve_device,
    resolve_torch_dtype,
)
from score_string_trait_number_distribution import slugify  # noqa: E402


DEFAULT_OUTPUT_SUBDIR = "m10_checkpoint_effect_matrices"
M10_SCORE_CACHE_VERSION = 3
DEFAULT_QUESTIONS = [
    "Name your favorite trait. Respond with only its name.",
]
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(**DATACLASS_KW_ONLY)
class M10Config:
    """Configuration for generic checkpoint effect-matrix evaluation."""

    training_output_dir: Path
    output_dir: Path
    answers: list[str]
    traits: list[str]
    questions: list[str]
    checkpoint_steps: list[int]
    include_final: bool
    final_only: bool
    adapter_scale: float
    base_model_id: str
    candidate_prefix: str
    assistant_prefill: str | None
    epsilon: float
    batch_size: int
    device: str
    torch_dtype: str
    skip_existing: bool
    analysis_only: bool
    write_plots: bool
    enable_thinking: bool | None
    use_prefix_cache: bool
    autotune_batch_pair: bool
    prompt_batch_size: int
    branch_batch_size: int
    max_prompt_batch_size: int
    max_branch_batch_size: int
    autotune_cache_path: Path | None
    base_score_cache_path: Path | None


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def load_json_string_list(path: Path, label: str) -> list[str]:
    """Load and validate a non-empty JSON list of unique strings."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data or not all(
        isinstance(item, str) and item.strip() for item in data
    ):
        raise ValueError(f"{label} must be a non-empty JSON list of strings.")
    values = [item.strip() for item in data]
    if len(values) != len(set(values)):
        raise ValueError(f"{label} contains duplicate values.")
    return values


def parse_steps(raw_values: list[str] | None) -> list[int]:
    """Parse repeated or comma-separated positive checkpoint steps."""
    values: list[int] = []
    seen: set[int] = set()
    for raw_value in raw_values or []:
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
                values.append(step)
                seen.add(step)
    return sorted(values)


def adapter_path(
    training_output_dir: Path,
    trait: str,
    checkpoint_step: int | None,
) -> Path:
    """Return the adapter directory for one trait and stage."""
    safe_trait = slugify(trait)
    if checkpoint_step is None:
        return training_output_dir / "adapters" / safe_trait
    return (
        training_output_dir
        / "trainer"
        / safe_trait
        / f"checkpoint-step-{checkpoint_step:06d}"
        / "adapter"
    )


def build_stages(config: M10Config) -> list[EvaluationStage]:
    """Build selected checkpoint stages and validate adapter files."""
    stages = [
        EvaluationStage(
            label=f"step_{step:06d}",
            checkpoint_step=step,
            adapter_paths={
                trait: adapter_path(config.training_output_dir, trait, step)
                for trait in config.traits
            },
        )
        for step in config.checkpoint_steps
    ]
    if config.include_final:
        stages.append(
            EvaluationStage(
                label="final",
                checkpoint_step=None,
                adapter_paths={
                    trait: adapter_path(config.training_output_dir, trait, None)
                    for trait in config.traits
                },
            )
        )
    if not stages:
        raise ValueError("Select at least one checkpoint or include final adapters.")
    missing = [
        path
        for stage in stages
        for path in stage.adapter_paths.values()
        if not (path / "adapter_config.json").is_file()
    ]
    if missing:
        preview = "\n".join(f"  {path}" for path in missing[:12])
        raise FileNotFoundError(f"Missing adapter directories:\n{preview}")
    return stages


def legacy_config(config: M10Config) -> CheckpointEvaluationConfig:
    """Build the existing scorer's configuration as generic plumbing."""
    return CheckpointEvaluationConfig(
        training_output_dir=config.training_output_dir,
        output_dir=config.output_dir,
        checkpoint_steps=config.checkpoint_steps,
        include_final=config.include_final,
        colors=config.answers,
        traits=config.traits,
        questions=config.questions,
        base_model_id=config.base_model_id,
        probability_mode="raw",
        epsilon=config.epsilon,
        candidate_prefix=config.candidate_prefix,
        assistant_prefill=config.assistant_prefill,
        batch_size=config.batch_size,
        device=config.device,
        torch_dtype=config.torch_dtype,
        teacher_logprob_cache_dir=config.output_dir / "unused_teacher_cache",
        compute_update_readout_projection=False,
        projection_finite_difference_step=0.5,
        skip_existing=config.skip_existing,
        print_log_ratio_tables=False,
    )


def ratio_config(config: M10Config) -> RatioMatrixConfig:
    """Build the exact complete-answer scorer configuration."""
    return RatioMatrixConfig(
        colors=config.answers,
        fine_tune_traits=config.traits,
        questions=config.questions,
        model_json_template="unused_{trait}.json",
        base_model_id=config.base_model_id,
        output_path=config.output_dir / "unused.json",
        csv_output_path=config.output_dir / "unused.csv",
        heatmap_path=config.output_dir / "unused.png",
        probability_mode="raw",
        epsilon=config.epsilon,
        candidate_prefix=config.candidate_prefix,
        assistant_prefill=config.assistant_prefill,
        batch_size=config.batch_size,
        device=config.device,
        torch_dtype=config.torch_dtype,
        enable_thinking=config.enable_thinking,
        use_prefix_cache=config.use_prefix_cache,
        prompt_batch_size=config.prompt_batch_size,
        branch_batch_size=config.branch_batch_size,
    )


def stage_cache_path(config: M10Config, stage: EvaluationStage) -> Path:
    """Return the durable score-cache path for one stage."""
    return config.output_dir / "score_cache" / f"{stage.label}.json"


def native_score_policy(config: M10Config) -> dict[str, Any]:
    """Identify the pinned native arithmetic enforced by the metric loader."""
    identity = native_metric_config_identity(config.base_model_id)
    if identity and config.torch_dtype == "float32":
        identity.update(
            {
                "native_attention_policy": "explicit_fp32_math_sdpa",
                "native_float32_matmul_precision": "highest",
                "native_allow_tf32": False,
            }
        )
    return identity


def adapter_file_identity(path: Path) -> dict[str, Any]:
    """Detect adapter replacement without rereading large weight files.

    Production adapters are immutable after publication. Size and nanosecond
    modification time distinguish replaced weights at the same path; the small
    PEFT configuration additionally receives a content hash.
    """
    files: dict[str, Any] = {}
    for name in (
        "adapter_config.json", "adapter_model.safetensors", "adapter_model.bin",
    ):
        candidate = path / name
        if not candidate.is_file():
            continue
        stat = candidate.stat()
        files[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if name == "adapter_config.json":
            files[name]["sha256"] = hashlib.sha256(candidate.read_bytes()).hexdigest()
    return files


def score_identity(config: M10Config, stage: EvaluationStage) -> dict[str, Any]:
    """Return fields that must match before reusing a stage cache."""
    native_policy = native_score_policy(config)
    identity = {
        **native_policy,
        "score_cache_version": M10_SCORE_CACHE_VERSION,
        "stage": stage.label,
        "checkpoint_step": stage.checkpoint_step,
        "adapter_scale": config.adapter_scale,
        "answers": config.answers,
        "traits": config.traits,
        "questions": config.questions,
        "base_model_id": config.base_model_id,
        "candidate_prefix": config.candidate_prefix,
        "assistant_prefill": config.assistant_prefill,
        "enable_thinking": config.enable_thinking,
        "torch_dtype": config.torch_dtype,
        "use_prefix_cache": config.use_prefix_cache,
        "adapter_paths": {
            trait: str(path) for trait, path in stage.adapter_paths.items()
        },
    }
    if native_policy:
        identity["adapter_files"] = {
            trait: adapter_file_identity(path)
            for trait, path in stage.adapter_paths.items()
        }
    return identity


def load_cached_stage(
    config: M10Config,
    stage: EvaluationStage,
) -> dict[str, Any] | None:
    """Load a matching stage score cache, if present."""
    path = stage_cache_path(config, stage)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable stage cache {}: {}", path, exc)
        return None
    if payload.get("m10_score_identity") != score_identity(config, stage):
        logger.warning("Ignoring stale stage cache {}", path)
        return None
    return payload.get("stage_result")


def write_json(path: Path, payload: Any) -> None:
    """Write a JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    logger.success("Saved {}", path)


def json_default(value: Any) -> Any:
    """Convert supported artifact values that JSON cannot encode natively."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(
        f"Object of type {value.__class__.__name__} is not JSON serializable"
    )


def save_cached_stage(
    config: M10Config,
    stage: EvaluationStage,
    result: dict[str, Any],
) -> None:
    """Save one stage score immediately so interrupted runs can resume."""
    write_json(
        stage_cache_path(config, stage),
        {
            "m10_score_identity": score_identity(config, stage),
            "stage_result": result,
        },
    )


def base_cache_identity(config: M10Config) -> dict[str, Any]:
    """Return fields identifying the reusable base score."""
    return {
        **native_score_policy(config),
        "score_cache_version": M10_SCORE_CACHE_VERSION,
        "answers": config.answers,
        "questions": config.questions,
        "base_model_id": config.base_model_id,
        "candidate_prefix": config.candidate_prefix,
        "assistant_prefill": config.assistant_prefill,
        "enable_thinking": config.enable_thinking,
        "torch_dtype": config.torch_dtype,
        "use_prefix_cache": config.use_prefix_cache,
    }


def load_base_cache(config: M10Config) -> dict[str, Any] | None:
    """Load the shared base score when its identity matches."""
    path = config.base_score_cache_path or (
        config.output_dir / "score_cache" / "base.json"
    )
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("identity") != base_cache_identity(config):
        return None
    return payload.get("base_scores")


def save_base_cache(config: M10Config, base_scores: dict[str, Any]) -> None:
    """Save the base score shared across every checkpoint."""
    path = config.base_score_cache_path or (
        config.output_dir / "score_cache" / "base.json"
    )
    write_json(
        path,
        {"identity": base_cache_identity(config), "base_scores": base_scores},
    )


def score_native_model_with_retry(
    *,
    pipeline_config: M10Config,
    model_label: str,
    model: Any,
    tokenizer: Any,
    device: Any,
    config: RatioMatrixConfig,
) -> dict[str, Any]:
    """Retune a rejected native batch with the active adapter still loaded.

    All stage adapters remain resident during calibration, so its memory use
    matches production scoring. Only numerical batch rejection or OOM permits
    a retry; every chosen batch must pass the unchanged full-prefix reference.
    """
    kwargs = {
        "model_label": model_label,
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "config": config,
    }
    try:
        return score_model(**kwargs)
    except RuntimeError as exc:
        if not (
            isinstance(exc, BatchCandidateRejectedError)
            or is_out_of_memory_error(exc)
        ):
            raise
        logger.warning("Retuning M10 batch for active adapter {}: {}", model_label, exc)
    # The except scope has ended: do not retain an OOM traceback and its tensors
    # while releasing allocations or benchmarking the replacement batch.
    clear_accelerator_cache()
    prompts = [
        build_chat_prompt(
            tokenizer,
            user_prompt=question,
            assistant_prefill=config.assistant_prefill,
            enable_thinking=config.enable_thinking,
        )
        for question in config.questions
    ]
    choice, _ = autotune_exact_trie_batch_pair(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt_texts=prompts,
        target_texts=candidate_texts(config.colors, config.candidate_prefix),
        cache_path=(
            pipeline_config.output_dir / "autotune"
            / f"retry_{slugify(model_label)}.json"
        ),
        model_id=pipeline_config.base_model_id,
        max_prompt_batch_size=pipeline_config.max_prompt_batch_size,
        max_branch_batch_size=pipeline_config.max_branch_batch_size,
        reuse_cache=False,
    )
    config.prompt_batch_size = choice.prompt_batch_size
    config.branch_batch_size = choice.branch_batch_size
    pipeline_config.prompt_batch_size = choice.prompt_batch_size
    pipeline_config.branch_batch_size = choice.branch_batch_size
    # A second failure propagates. Nothing has been written to a score cache.
    return score_model(**kwargs)


def score_pending_stages(
    config: M10Config,
    stages: list[EvaluationStage],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Score missing stages with one shared base-model load."""
    results: dict[str, dict[str, Any]] = {}
    timings: list[dict[str, Any]] = []
    pending: list[EvaluationStage] = []
    for stage in stages:
        cached = (
            load_cached_stage(config, stage)
            if config.skip_existing or config.analysis_only
            else None
        )
        if cached is None:
            pending.append(stage)
        else:
            logger.info("Reusing M10 stage {}", stage.label)
            results[stage.label] = cached
    if not pending:
        return results, timings
    if config.analysis_only:
        labels = ", ".join(stage.label for stage in pending)
        raise FileNotFoundError(f"Analysis-only mode is missing caches: {labels}")

    import torch

    device = resolve_device(torch, config.device)
    torch_dtype = resolve_torch_dtype(torch, config.torch_dtype)
    logger.info("Loading shared base model {}", config.base_model_id)
    load_started = time.perf_counter()
    base_model, tokenizer = load_direct_model_and_tokenizer(
        model_id=config.base_model_id,
        device=device,
        torch_dtype=torch_dtype,
    )
    timings.append(
        {"stage": "load_base_model", "seconds": time.perf_counter() - load_started}
    )
    if config.use_prefix_cache and config.autotune_batch_pair:
        tune_started = time.perf_counter()
        prompts = [
            build_chat_prompt(
                tokenizer,
                user_prompt=question,
                assistant_prefill=config.assistant_prefill,
                enable_thinking=config.enable_thinking,
            )
            for question in config.questions
        ]
        choice, _ = autotune_exact_trie_batch_pair(
            model=base_model,
            tokenizer=tokenizer,
            device=device,
            prompt_texts=prompts,
            target_texts=candidate_texts(
                config.answers,
                config.candidate_prefix,
            ),
            cache_path=(
                config.autotune_cache_path
                or config.output_dir / "autotune" / "exact_trie_batch_pair.json"
            ),
            model_id=config.base_model_id,
            max_prompt_batch_size=config.max_prompt_batch_size,
            max_branch_batch_size=config.max_branch_batch_size,
            reuse_cache=config.skip_existing,
        )
        config.prompt_batch_size = choice.prompt_batch_size
        config.branch_batch_size = choice.branch_batch_size
        timings.append(
            {
                "stage": "autotune_exact_trie_batch_pair",
                "seconds": time.perf_counter() - tune_started,
                "prompt_batch_size": choice.prompt_batch_size,
                "branch_batch_size": choice.branch_batch_size,
            }
        )
    model = base_model
    model_is_peft = False
    base_scores = load_base_cache(config) if config.skip_existing else None
    if base_scores is None:
        started = time.perf_counter()
        base_scores = score_model(
            model_label="base",
            model=base_model,
            tokenizer=tokenizer,
            device=device,
            config=ratio_config(config),
        )
        save_base_cache(config, base_scores)
        timings.append(
            {"stage": "score_base", "seconds": time.perf_counter() - started}
        )

    plumbing_config = legacy_config(config)
    scorer_kwargs = {}
    if config.use_prefix_cache and native_score_policy(config):
        scorer_kwargs["score_model_fn"] = partial(
            score_native_model_with_retry, pipeline_config=config,
        )
    try:
        for stage in pending:
            started = time.perf_counter()
            model, model_is_peft, names = load_stage_adapters(
                model=model,
                model_is_peft=model_is_peft,
                stage=stage,
                traits=config.traits,
            )
            scaling_entries = []
            try:
                for name in names:
                    entries = find_lora_scaling_entries(model, name)
                    set_lora_scaling_multiplier(entries, config.adapter_scale)
                    scaling_entries.append(entries)
                result = score_evaluation_stage(
                    config=plumbing_config,
                    stage=stage,
                    ratio_config=ratio_config(config),
                    base_scores=base_scores,
                    model=model,
                    tokenizer=tokenizer,
                    adapter_names=names,
                    device=device,
                    **scorer_kwargs,
                )
                result["adapter_scale"] = config.adapter_scale
                save_cached_stage(config, stage, result)
                results[stage.label] = result
            finally:
                for entries in scaling_entries:
                    set_lora_scaling_multiplier(entries, 1.0)
                unload_stage_adapters(model, names)
            timings.append(
                {
                    "stage": f"score_{stage.label}",
                    "seconds": time.perf_counter() - started,
                }
            )
    finally:
        del model
        del base_model
        clear_accelerator_cache()
    return results, timings


def double_center(matrix: np.ndarray) -> np.ndarray:
    """Remove row and column means while restoring the grand mean."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Expected a non-empty finite two-dimensional matrix.")
    return (
        values
        - np.mean(values, axis=1, keepdims=True)
        - np.mean(values, axis=0, keepdims=True)
        + np.mean(values)
    )


def spectral_metrics(matrix: np.ndarray) -> dict[str, Any]:
    """Compute the catalog's scale and continuous rank summaries."""
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    energies = np.square(singular_values)
    total = float(np.sum(energies))
    fractions = energies / total if total > 0.0 else np.zeros_like(energies)
    positive = fractions[fractions > 0.0]
    sigma_1 = float(singular_values[0]) if singular_values.size else 0.0
    sigma_2 = float(singular_values[1]) if singular_values.size > 1 else 0.0
    cumulative = np.cumsum(fractions)
    r90 = int(np.searchsorted(cumulative, 0.90) + 1) if total > 0.0 else 0
    return {
        "frobenius_norm": math.sqrt(total),
        "ev_at_1": float(fractions[0]) if fractions.size else 0.0,
        "ev_at_2": float(np.sum(fractions[:2])),
        "effective_rank": (
            float(np.exp(-np.sum(positive * np.log(positive))))
            if positive.size
            else 0.0
        ),
        "stable_rank": total / (sigma_1**2) if sigma_1 > 0.0 else 0.0,
        "r90": r90,
        "sigma_2_over_sigma_1": sigma_2 / sigma_1 if sigma_1 > 0.0 else 0.0,
        "singular_values": singular_values.tolist(),
    }


def overlap_matrix(
    previous_u: np.ndarray,
    previous_v: np.ndarray,
    current_u: np.ndarray,
    current_v: np.ndarray,
) -> np.ndarray:
    """Return all adjacent-mode overlaps m_kl from the M10 definition."""
    return 0.5 * (
        np.abs(previous_u.T @ current_u) + np.abs(previous_v.T @ current_v)
    )


def match_modes(overlaps: np.ndarray) -> list[dict[str, Any]]:
    """Choose the one-to-one mode permutation with maximum total overlap."""
    from scipy.optimize import linear_sum_assignment

    rows, columns = linear_sum_assignment(-np.asarray(overlaps, dtype=np.float64))
    return [
        {
            "previous_mode": int(row + 1),
            "current_mode": int(column + 1),
            "overlap": float(overlaps[row, column]),
        }
        for row, column in zip(rows, columns)
    ]


def leading_subspace_overlap(
    previous_u: np.ndarray,
    previous_v: np.ndarray,
    current_u: np.ndarray,
    current_v: np.ndarray,
    rank: int,
) -> dict[str, float]:
    """Summarize principal-angle overlap for leading left/right subspaces."""
    effective_rank = min(
        rank,
        previous_u.shape[1],
        previous_v.shape[1],
        current_u.shape[1],
        current_v.shape[1],
    )
    if effective_rank <= 0:
        return {"left": 0.0, "right": 0.0, "mean": 0.0}
    left_cosines = np.linalg.svd(
        previous_u[:, :effective_rank].T @ current_u[:, :effective_rank],
        compute_uv=False,
    )
    right_cosines = np.linalg.svd(
        previous_v[:, :effective_rank].T @ current_v[:, :effective_rank],
        compute_uv=False,
    )
    left = float(np.mean(np.square(left_cosines)))
    right = float(np.mean(np.square(right_cosines)))
    return {"left": left, "right": right, "mean": 0.5 * (left + right)}


def top_loadings(
    vector: np.ndarray,
    labels: list[str],
    count: int = 5,
) -> dict[str, Any]:
    """Return the largest positive and negative vector loadings."""
    values = np.asarray(vector, dtype=np.float64)
    positive = sorted(
        ((labels[index], float(value)) for index, value in enumerate(values)),
        key=lambda item: item[1],
        reverse=True,
    )[:count]
    negative = sorted(
        ((labels[index], float(value)) for index, value in enumerate(values)),
        key=lambda item: item[1],
    )[:count]
    return {
        "positive": [{"label": label, "loading": value} for label, value in positive],
        "negative": [{"label": label, "loading": value} for label, value in negative],
    }


def matrix_from_nested(
    nested: dict[str, dict[str, float]],
    rows: list[str],
    columns: list[str],
) -> np.ndarray:
    """Convert an ordered nested mapping to a dense array."""
    return np.asarray(
        [[nested[row][column] for column in columns] for row in rows],
        dtype=np.float64,
    )


def nested_from_matrix(
    matrix: np.ndarray,
    rows: list[str],
    columns: list[str],
) -> dict[str, dict[str, float]]:
    """Convert a dense array to an ordered nested mapping."""
    return {
        row: {
            column: float(matrix[row_index, column_index])
            for column_index, column in enumerate(columns)
        }
        for row_index, row in enumerate(rows)
    }


def stage_matrices(
    config: M10Config,
    stages: list[EvaluationStage],
    results: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build raw-primary and normalized-secondary matrices, including step 0."""
    zero = np.zeros((len(config.answers), len(config.traits)), dtype=np.float64)
    stage_rows: list[dict[str, Any]] = [
        {"label": "step_000000", "checkpoint_step": 0, "is_final": False}
    ]
    raw = {"step_000000": zero}
    normalized = {"step_000000": zero.copy()}
    for stage in stages:
        result = results[stage.label]
        raw_result = build_raw_logprob_shift_result(
            stage=stage,
            stage_result=result,
            colors=config.answers,
            traits=config.traits,
            epsilon=config.epsilon,
        )
        normalized_nested = cached_question_mean_log_ratio_matrix(
            stage_result=result,
            colors=config.answers,
            traits=config.traits,
            probability_mode="normalized",
            epsilon=config.epsilon,
        )
        raw[stage.label] = matrix_from_nested(
            raw_result["raw_logprob_shift_matrix"],
            config.answers,
            config.traits,
        )
        normalized[stage.label] = matrix_from_nested(
            normalized_nested,
            config.answers,
            config.traits,
        )
        stage_rows.append(
            {
                "label": stage.label,
                "checkpoint_step": stage.checkpoint_step,
                "is_final": stage.checkpoint_step is None,
            }
        )
    return stage_rows, raw, normalized


def build_m10_analysis(
    *,
    stage_rows: list[dict[str, Any]],
    raw_matrices: dict[str, np.ndarray],
    answers: list[str],
    traits: list[str],
    final_training_step: int | None,
) -> dict[str, Any]:
    """Compute rank, final-mode, increment, and adjacent-overlap trajectories."""
    labels = [row["label"] for row in stage_rows]
    centered = {label: double_center(raw_matrices[label]) for label in labels}
    decompositions = {
        label: np.linalg.svd(centered[label], full_matrices=False) for label in labels
    }
    rank_rows = []
    for row in stage_rows:
        label = row["label"]
        step = final_training_step if row["is_final"] else row["checkpoint_step"]
        for view, matrix in (
            ("uncentered_raw", raw_matrices[label]),
            ("double_centered_raw", centered[label]),
        ):
            rank_rows.append(
                {
                    "stage": label,
                    "checkpoint_step": step,
                    "matrix_view": view,
                    **spectral_metrics(matrix),
                }
            )

    final_label = labels[-1]
    final_u, final_s, final_vh = decompositions[final_label]
    final_v = final_vh.T
    mode_count = len(final_s)
    final_mode_rows = []
    for mode_index in range(mode_count):
        amplitudes = [
            float(final_u[:, mode_index].T @ centered[label] @ final_v[:, mode_index])
            for label in labels
        ]
        final_amplitude = abs(amplitudes[-1])
        birth = None
        for stage_index in range(1, len(labels) - 1):
            if final_amplitude == 0.0:
                break
            threshold = 0.10 * final_amplitude
            if (
                abs(amplitudes[stage_index]) >= threshold
                and abs(amplitudes[stage_index + 1]) >= threshold
            ):
                birth = labels[stage_index]
                break
        for stage_index, (row, amplitude) in enumerate(zip(stage_rows, amplitudes)):
            total_energy = float(np.sum(np.square(centered[row["label"]])))
            final_mode_rows.append(
                {
                    "stage": row["label"],
                    "checkpoint_step": (
                        final_training_step
                        if row["is_final"]
                        else row["checkpoint_step"]
                    ),
                    "mode": mode_index + 1,
                    "amplitude": amplitude,
                    "energy_share": (
                        amplitude**2 / total_energy if total_energy > 0 else 0.0
                    ),
                    "birth_stage": birth,
                    "answer_loadings": top_loadings(final_u[:, mode_index], answers),
                    "trait_loadings": top_loadings(final_v[:, mode_index], traits),
                    "final_singular_value": float(final_s[mode_index]),
                    "stage_index": stage_index,
                }
            )

    adjacent = []
    increments = []
    for index in range(1, len(labels)):
        previous_label = labels[index - 1]
        current_label = labels[index]
        previous_u, _, previous_vh = decompositions[previous_label]
        current_u, _, current_vh = decompositions[current_label]
        overlaps = overlap_matrix(
            previous_u,
            previous_vh.T,
            current_u,
            current_vh.T,
        )
        previous_has_modes = float(np.linalg.norm(centered[previous_label])) > 0.0
        current_has_modes = float(np.linalg.norm(centered[current_label])) > 0.0
        if previous_has_modes and current_has_modes:
            adjacent.append(
                {
                    "from_stage": previous_label,
                    "to_stage": current_label,
                    "matches": match_modes(overlaps),
                    "leading_rank_1_subspace_overlap": leading_subspace_overlap(
                        previous_u, previous_vh.T, current_u, current_vh.T, 1
                    ),
                    "leading_rank_2_subspace_overlap": leading_subspace_overlap(
                        previous_u, previous_vh.T, current_u, current_vh.T, 2
                    ),
                }
            )
        increment = centered[current_label] - centered[previous_label]
        increment_u, increment_s, increment_vh = np.linalg.svd(
            increment,
            full_matrices=False,
        )
        increments.append(
            {
                "from_stage": previous_label,
                "to_stage": current_label,
                "spectral_metrics": spectral_metrics(increment),
                "leading_answer_loadings": top_loadings(increment_u[:, 0], answers),
                "leading_trait_loadings": top_loadings(increment_vh.T[:, 0], traits),
                "leading_singular_value": (
                    float(increment_s[0]) if increment_s.size else 0.0
                ),
            }
        )
    return {
        "definition": (
            "M10 on mean complete-answer raw log-probability change; step 0 is "
            "the exact zero matrix and double centering isolates interactions."
        ),
        "final_stage": final_label,
        "final_training_step": final_training_step,
        "rank_trajectories": rank_rows,
        "final_mode_trajectories": final_mode_rows,
        "adjacent_mode_matches": adjacent,
        "increment_summaries": increments,
        "double_centered_raw_matrices": {
            label: nested_from_matrix(centered[label], answers, traits)
            for label in labels
        },
    }


def write_matrix_csv(
    path: Path,
    matrix: np.ndarray,
    rows: list[str],
    columns: list[str],
) -> None:
    """Write one labeled matrix CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["answer_u__ft_trait_t", *columns])
        for row_index, row in enumerate(rows):
            writer.writerow([row, *matrix[row_index].tolist()])


def write_tidy_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """Write selected scalar fields from a list of dictionaries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def matching_cell_indices(
    rows: list[str],
    columns: list[str],
) -> list[tuple[int, int]]:
    """Return matrix cells where the answer equals the fine-tuning trait."""
    row_indices: dict[str, list[int]] = {}
    for row_index, label in enumerate(rows):
        row_indices.setdefault(label, []).append(row_index)
    return [
        (row_index, column_index)
        for column_index, label in enumerate(columns)
        for row_index in row_indices.get(label, [])
    ]


def write_heatmap(
    path: Path,
    matrix: np.ndarray,
    rows: list[str],
    columns: list[str],
    title: str,
    max_abs: float,
) -> None:
    """Write a symmetric-scale heatmap with exact-trait cells outlined."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axis = plt.subplots(
        figsize=(max(8.0, 0.45 * len(columns) + 3), max(6.0, 0.35 * len(rows) + 2))
    )
    image = axis.imshow(matrix, cmap="coolwarm", vmin=-max_abs, vmax=max_abs)
    axis.set_title(title)
    axis.set_xlabel("Fine-tuning trait t")
    axis.set_ylabel("Answer u")
    axis.set_xticks(np.arange(len(columns)))
    axis.set_xticklabels(columns, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(rows)))
    axis.set_yticklabels(rows)
    for row_index, column_index in matching_cell_indices(rows, columns):
        rectangle = (column_index - 0.5, row_index - 0.5)
        axis.add_patch(
            Rectangle(
                rectangle,
                1.0,
                1.0,
                fill=False,
                edgecolor="#111827",
                linewidth=3.0,
                zorder=3,
            )
        )
        axis.add_patch(
            Rectangle(
                rectangle,
                1.0,
                1.0,
                fill=False,
                edgecolor="#FDE047",
                linewidth=1.7,
                zorder=4,
            )
        )
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.text(
        0.5,
        0.01,
        "Gold outlines: answer u equals fine-tuning trait t",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.025, 1.0, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_trajectory_plots(
    output_dir: Path,
    analysis: dict[str, Any],
) -> None:
    """Plot current-catalog rank summaries and final-mode amplitudes."""
    import matplotlib.pyplot as plt

    rank_rows = analysis["rank_trajectories"]
    views = ["uncentered_raw", "double_centered_raw"]
    metrics = [
        ("frobenius_norm", "Frobenius norm"),
        ("ev_at_1", "EV@1"),
        ("ev_at_2", "EV@2"),
        ("effective_rank", "Effective rank"),
        ("stable_rank", "Stable rank"),
        ("r90", "r90"),
        ("sigma_2_over_sigma_1", "σ2 / σ1"),
    ]
    figure, axes = plt.subplots(4, 2, figsize=(13, 14))
    for axis, (metric, title) in zip(axes.flat, metrics):
        for view in views:
            rows = [row for row in rank_rows if row["matrix_view"] == view]
            axis.plot(
                range(len(rows)),
                [row[metric] for row in rows],
                marker="o",
                label=view,
            )
        axis.set_title(title)
        axis.set_xticks(range(len(rows)))
        axis.set_xticklabels([row["stage"] for row in rows], rotation=45, ha="right")
        axis.grid(alpha=0.25)
    axes.flat[-1].axis("off")
    axes.flat[0].legend(fontsize=8)
    figure.suptitle("M10 raw fine-tuning effect trajectories")
    figure.tight_layout()
    figure.savefig(output_dir / "m10_rank_trajectories.png", dpi=180)
    plt.close(figure)

    mode_rows = analysis["final_mode_trajectories"]
    mode_count = min(6, max((row["mode"] for row in mode_rows), default=0))
    figure, axis = plt.subplots(figsize=(11, 6))
    for mode in range(1, mode_count + 1):
        rows = [row for row in mode_rows if row["mode"] == mode]
        axis.plot(
            range(len(rows)),
            [row["amplitude"] for row in rows],
            marker="o",
            label=f"final mode {mode}",
        )
    if mode_count:
        axis.set_xticks(range(len(rows)))
        axis.set_xticklabels([row["stage"] for row in rows], rotation=45, ha="right")
    axis.axhline(0.0, color="#888888", linewidth=0.8)
    axis.set_ylabel("Signed final-mode amplitude")
    axis.set_title("When final M10 patterns appear")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "m10_final_mode_trajectories.png", dpi=180)
    plt.close(figure)


def write_outputs(
    config: M10Config,
    stage_rows: list[dict[str, Any]],
    raw: dict[str, np.ndarray],
    normalized: dict[str, np.ndarray],
    analysis: dict[str, Any],
) -> dict[str, Any]:
    """Write matrices and M10 summaries in reusable formats."""
    matrix_root = config.output_dir / "matrices"
    centered = {
        label: matrix_from_nested(
            analysis["double_centered_raw_matrices"][label],
            config.answers,
            config.traits,
        )
        for label in raw
    }
    normalized_centered = {
        label: double_center(matrix) for label, matrix in normalized.items()
    }
    raw_scale = max(1e-12, max(float(np.max(np.abs(value))) for value in raw.values()))
    centered_scale = max(
        1e-12,
        max(float(np.max(np.abs(value))) for value in centered.values()),
    )
    for row in stage_rows:
        label = row["label"]
        stage_dir = matrix_root / label
        write_matrix_csv(
            stage_dir / "raw_logprob_shift.csv",
            raw[label],
            config.answers,
            config.traits,
        )
        write_matrix_csv(
            stage_dir / "normalized_logprob_shift.csv",
            normalized[label],
            config.answers,
            config.traits,
        )
        write_matrix_csv(
            stage_dir / "normalized_logprob_shift_double_centered.csv",
            normalized_centered[label],
            config.answers,
            config.traits,
        )
        write_matrix_csv(
            stage_dir / "raw_logprob_shift_double_centered.csv",
            centered[label],
            config.answers,
            config.traits,
        )
        if config.write_plots:
            write_heatmap(
                stage_dir / "raw_logprob_shift.png",
                raw[label],
                config.answers,
                config.traits,
                f"Raw complete-answer log-probability shift · {label}",
                raw_scale,
            )
            write_heatmap(
                stage_dir / "raw_logprob_shift_double_centered.png",
                centered[label],
                config.answers,
                config.traits,
                f"Double-centered raw shift · {label}",
                centered_scale,
            )

    write_json(config.output_dir / "m10_analysis.json", analysis)
    rank_fields = [
        "stage",
        "checkpoint_step",
        "matrix_view",
        "frobenius_norm",
        "ev_at_1",
        "ev_at_2",
        "effective_rank",
        "stable_rank",
        "r90",
        "sigma_2_over_sigma_1",
    ]
    write_tidy_csv(
        config.output_dir / "m10_rank_trajectories.csv",
        analysis["rank_trajectories"],
        rank_fields,
    )
    write_tidy_csv(
        config.output_dir / "m10_final_mode_trajectories.csv",
        analysis["final_mode_trajectories"],
        [
            "stage",
            "checkpoint_step",
            "mode",
            "amplitude",
            "energy_share",
            "birth_stage",
            "final_singular_value",
        ],
    )
    if config.write_plots:
        write_trajectory_plots(config.output_dir, analysis)
    return {
        "m10_analysis_json": str(config.output_dir / "m10_analysis.json"),
        "rank_csv": str(config.output_dir / "m10_rank_trajectories.csv"),
        "mode_csv": str(config.output_dir / "m10_final_mode_trajectories.csv"),
        "matrix_root": str(matrix_root),
        "rank_plot": (
            str(config.output_dir / "m10_rank_trajectories.png")
            if config.write_plots
            else None
        ),
        "mode_plot": (
            str(config.output_dir / "m10_final_mode_trajectories.png")
            if config.write_plots
            else None
        ),
    }


def run(config: M10Config) -> dict[str, Any]:
    """Run generic checkpoint scoring and M10 aggregation."""
    started = time.perf_counter()
    started_at = utc_now()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    available_steps = discover_common_checkpoint_steps(
        config.training_output_dir,
        config.traits,
    )
    if not config.checkpoint_steps and not config.final_only:
        config.checkpoint_steps = available_steps
    missing_steps = sorted(set(config.checkpoint_steps) - set(available_steps))
    if missing_steps:
        raise FileNotFoundError(
            f"Checkpoint steps not shared by every trait: {missing_steps}"
        )
    stages = build_stages(config)
    results, timings = score_pending_stages(config, stages)
    stage_rows, raw, normalized = stage_matrices(config, stages, results)
    final_training_step = infer_final_training_step(
        config.training_output_dir,
        config.traits,
        available_steps,
    )
    analysis_started = time.perf_counter()
    analysis = build_m10_analysis(
        stage_rows=stage_rows,
        raw_matrices=raw,
        answers=config.answers,
        traits=config.traits,
        final_training_step=final_training_step,
    )
    timings.append(
        {"stage": "m10_analysis", "seconds": time.perf_counter() - analysis_started}
    )
    outputs = write_outputs(config, stage_rows, raw, normalized, analysis)
    duration = time.perf_counter() - started
    manifest = {
        "analysis": "M10 fine-tuning effect rank and emergent-pattern dynamics",
        "started_at": started_at,
        "finished_at": utc_now(),
        "duration_seconds": duration,
        "config": {
            **asdict(config),
            "training_output_dir": str(config.training_output_dir),
            "output_dir": str(config.output_dir),
        },
        "available_checkpoint_steps": available_steps,
        "selected_checkpoint_steps": config.checkpoint_steps,
        "timings": timings,
        "outputs": outputs,
    }
    write_json(config.output_dir / "manifest.json", manifest)
    logger.success("M10 evaluation completed in {:.1f} seconds", duration)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Score arbitrary trait-answer matrices across LoRA checkpoints and "
            "compute the current metric catalog's M10 outputs."
        )
    )
    parser.add_argument("--training-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--answers-path", type=Path, required=True)
    parser.add_argument("--trait", "--traits", dest="traits", action="append")
    parser.add_argument("--questions-path", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-step",
        "--checkpoint-steps",
        dest="checkpoint_steps",
        action="append",
        help="Repeat or pass comma-separated values; defaults to all shared steps.",
    )
    parser.add_argument("--no-final", action="store_true")
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Score only final adapters, without evaluating intermediate checkpoints.",
    )
    parser.add_argument(
        "--adapter-scale",
        type=float,
        default=1.0,
        help=(
            "Multiply every loaded LoRA contribution by this value. A value "
            "of 1 is the trained endpoint; values above 1 extrapolate it."
        ),
    )
    parser.add_argument("--base-model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--candidate-prefix", default="")
    parser.add_argument("--assistant-prefill", default=None)
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        help="Render behavior prompts with model thinking enabled.",
    )
    thinking_group.add_argument(
        "--disable-thinking",
        dest="enable_thinking",
        action="store_false",
        help="Render direct-answer behavior prompts with thinking disabled.",
    )
    parser.set_defaults(enable_thinking=None)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--use-prefix-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Score complete answers with one common-prompt KV prefill.",
    )
    parser.add_argument("--autotune-batch-pair", action="store_true")
    parser.add_argument("--prompt-batch-size", type=int, default=8)
    parser.add_argument("--branch-batch-size", type=int, default=1)
    parser.add_argument("--max-prompt-batch-size", type=int, default=32)
    parser.add_argument("--max-branch-batch-size", type=int, default=64)
    parser.add_argument(
        "--autotune-cache-path",
        type=Path,
        default=None,
        help="Optional exact-trie (prompt batch, branch batch) cache shared across runs.",
    )
    parser.add_argument(
        "--base-score-cache-path",
        type=Path,
        default=None,
        help="Optional base-model behavior-score cache shared across runs.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def build_config(args: argparse.Namespace) -> M10Config:
    """Build and validate an M10 configuration from CLI arguments."""
    answers = load_json_string_list(args.answers_path, "--answers-path")
    questions = load_json_string_list(args.questions_path, "--questions-path")
    traits = parse_csv_values(args.traits, [])
    if not traits:
        raise ValueError("At least one --trait is required.")
    if len(traits) != len(set(traits)):
        raise ValueError("Trait list contains duplicates.")
    for name in (
        "batch_size",
        "prompt_batch_size",
        "branch_batch_size",
        "max_prompt_batch_size",
        "max_branch_batch_size",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.autotune_batch_pair and not args.use_prefix_cache:
        raise ValueError("--autotune-batch-pair requires --use-prefix-cache.")
    if args.final_only and args.no_final:
        raise ValueError("--final-only cannot be combined with --no-final.")
    if args.final_only and parse_steps(args.checkpoint_steps):
        raise ValueError("--final-only cannot be combined with --checkpoint-step.")
    if not math.isfinite(args.adapter_scale):
        raise ValueError("--adapter-scale must be finite.")
    if args.epsilon < 0.0 or not math.isfinite(args.epsilon):
        raise ValueError("--epsilon must be finite and non-negative.")
    output_dir = args.output_dir or (
        args.training_output_dir / DEFAULT_OUTPUT_SUBDIR
    )
    return M10Config(
        training_output_dir=args.training_output_dir,
        output_dir=output_dir,
        answers=answers,
        traits=traits,
        questions=questions,
        checkpoint_steps=parse_steps(args.checkpoint_steps),
        include_final=not args.no_final,
        final_only=args.final_only,
        adapter_scale=args.adapter_scale,
        base_model_id=args.base_model_id,
        candidate_prefix=args.candidate_prefix,
        assistant_prefill=args.assistant_prefill or None,
        epsilon=args.epsilon,
        batch_size=args.batch_size,
        device=args.device,
        torch_dtype=args.torch_dtype,
        skip_existing=args.skip_existing,
        analysis_only=args.analysis_only,
        write_plots=not args.no_plots,
        enable_thinking=args.enable_thinking,
        use_prefix_cache=args.use_prefix_cache,
        autotune_batch_pair=args.autotune_batch_pair,
        prompt_batch_size=args.prompt_batch_size,
        branch_batch_size=args.branch_batch_size,
        max_prompt_batch_size=args.max_prompt_batch_size,
        max_branch_batch_size=args.max_branch_batch_size,
        autotune_cache_path=args.autotune_cache_path,
        base_score_cache_path=args.base_score_cache_path,
    )


def main() -> None:
    """Run the generic M10 command line."""
    parser = build_parser()
    try:
        run(build_config(parser.parse_args()))
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
