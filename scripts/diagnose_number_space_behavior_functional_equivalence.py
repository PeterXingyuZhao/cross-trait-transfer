#!/usr/bin/env python3
"""Diagnose whether seed-unstable ordered T2 vectors are functionally equivalent."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import sys
import uuid
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compute_number_space_behavior_dataset_cosines import (  # noqa: E402
    DEFAULT_COLORS,
    build_behavior_contexts,
    build_number_candidates,
    build_number_embedding_matrix,
    exact_candidate_logprob_matrix,
    split_question_indices,
    validate_ordered_context_tokenization,
)
from find_owl_output_distribution_entanglement import (  # noqa: E402
    load_direct_model_and_tokenizer,
    parse_csv_values,
)
from score_birthday_year_probabilities import (  # noqa: E402
    resolve_device,
    resolve_torch_dtype,
)


DEFAULT_GAUSSIAN_REPORT = Path(
    "results/number_space_behavior_dataset_cosines/"
    "favorite_color_raw_nll_shared_init_15ep_three_seed_consensus.json"
)
DEFAULT_CENTERED_T1_REPORT = Path(
    "results/number_space_behavior_dataset_cosines/"
    "favorite_color_centered_t1_init_raw_nll_15ep_three_seed_consensus.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "results/number_space_behavior_functional_equivalence"
)
DEFAULT_OUTPUT_STEM = "favorite_color_existing_T2_functional_equivalence"
DEFAULT_CONTEXT_BATCH_SIZE = 4
DEFAULT_FISHER_EPSILON = 0.05
DEFAULT_HEATMAP_DPI = 220
ANALYSIS_VERSION = 1
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(frozen=True, **DATACLASS_KW_ONLY)
class VariantSpec:
    """One consensus report and its user-facing diagnostic label."""

    name: str
    report_path: Path


@dataclass(frozen=True, **DATACLASS_KW_ONLY)
class SeedCheckpoint:
    """One saved beta/T2 checkpoint identified by variant, trait, and seed."""

    variant: str
    trait: str
    seed: int
    output_stem: str
    path: Path
    beta: np.ndarray
    intervention: np.ndarray


@dataclass(**DATACLASS_KW_ONLY)
class DiagnosticConfig:
    """Configuration for held-out function-space seed diagnostics."""

    variants: list[VariantSpec]
    evaluation_traits: list[str]
    output_dir: Path
    output_stem: str
    context_batch_size: int
    fisher_epsilon: float
    device: str
    torch_dtype: str
    heatmap_dpi: int
    save_plot: bool
    skip_existing: bool


@dataclass(**DATACLASS_KW_ONLY)
class SharedExperiment:
    """Settings that must agree across all supplied consensus reports."""

    traits: list[str]
    numbers: list[str]
    questions: list[str]
    validation_question_indices: list[int]
    model_id: str
    favorite_number_system_prompt_template: str
    assistant_prefill: str
    train_question_count: int
    intervention_scale: float
    neutral_number_probabilities: np.ndarray
    variant_payloads: dict[str, dict[str, Any]]


def safe_label(value: str) -> str:
    """Return a filesystem-safe stable label."""
    return "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in value
    )


def stable_json_hash(payload: Any) -> str:
    """Hash a JSON-compatible payload deterministically."""
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def array_hash(values: np.ndarray) -> str:
    """Hash an array including its dtype and shape."""
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def parse_variant(raw_value: str) -> VariantSpec:
    """Parse NAME=CONSENSUS_JSON."""
    name, separator, path_text = raw_value.partition("=")
    if not separator or not name.strip() or not path_text.strip():
        raise ValueError(
            "--variant must use NAME=CONSENSUS_JSON, for example "
            "gaussian=results/consensus.json."
        )
    return VariantSpec(name=safe_label(name.strip()), report_path=Path(path_text))


def output_paths(config: DiagnosticConfig) -> dict[str, Path]:
    """Return durable diagnostic output paths."""
    prefix = config.output_dir / config.output_stem
    return {
        "json": prefix.with_suffix(".json"),
        "tensors": prefix.with_name(f"{prefix.name}_tensors.npz"),
        "pair_metrics_csv": prefix.with_name(
            f"{prefix.name}_seed_pair_function_metrics.csv"
        ),
        "trait_summary_csv": prefix.with_name(
            f"{prefix.name}_per_trait_summary.csv"
        ),
        "heatmap": prefix.with_name(
            f"{prefix.name}_minimum_seed_pair_metrics_heatmap.png"
        ),
        "scatter": prefix.with_name(
            f"{prefix.name}_geometry_vs_function_scatter.png"
        ),
        "cache_dir": prefix.with_name(f"{prefix.name}_evaluation_cache"),
    }


def serialize_config(config: DiagnosticConfig) -> dict[str, Any]:
    """Serialize the diagnostic configuration for JSON."""
    payload = asdict(config)
    payload["variants"] = [
        {"name": variant.name, "report_path": str(variant.report_path)}
        for variant in config.variants
    ]
    payload["output_dir"] = str(config.output_dir)
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load JSON report {path}.") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON report {path} must contain an object.")
    return value


def load_shared_experiment(config: DiagnosticConfig) -> SharedExperiment:
    """Load and cross-check the experiment definition from consensus reports."""
    if not config.variants:
        raise ValueError("At least one functional-equivalence variant is required.")
    names = [variant.name for variant in config.variants]
    if len(set(names)) != len(names):
        raise ValueError("Variant names must be unique.")

    payloads = {
        variant.name: _load_json(variant.report_path)
        for variant in config.variants
    }
    first_variant = config.variants[0]
    first_config = payloads[first_variant.name].get("config", {})
    if not isinstance(first_config, dict):
        raise ValueError(f"Consensus report {first_variant.report_path} has no config.")
    traits = [str(value) for value in first_config.get("traits", [])]
    questions = [str(value) for value in first_config.get("questions", [])]
    numbers = build_number_candidates()
    train_question_count = int(first_config.get("train_question_count", 0))
    _train_indices, validation_indices = split_question_indices(
        question_count=len(questions),
        train_question_count=train_question_count,
    )
    shared_keys = (
        "traits",
        "questions",
        "model_id",
        "favorite_number_system_prompt_template",
        "assistant_prefill",
        "train_question_count",
        "intervention_scale",
        "reference_tensors_path",
    )
    for variant in config.variants[1:]:
        candidate = payloads[variant.name].get("config", {})
        for key in shared_keys:
            if candidate.get(key) != first_config.get(key):
                raise ValueError(
                    f"Variant {variant.name!r} disagrees with "
                    f"{first_variant.name!r} on {key}."
                )

    reference_path = Path(str(first_config.get("reference_tensors_path", "")))
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)
    with np.load(reference_path, allow_pickle=False) as data:
        required = {"neutral_number_probabilities", "traits", "numbers"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(
                f"Reference tensors are missing arrays: {sorted(missing)}"
            )
        reference_traits = [str(value) for value in data["traits"].tolist()]
        reference_numbers = [str(value) for value in data["numbers"].tolist()]
        neutral = np.asarray(
            data["neutral_number_probabilities"], dtype=np.float64
        )
    if reference_traits != traits or reference_numbers != numbers:
        raise ValueError("Reference tensors do not match report trait/number labels.")
    if (
        neutral.shape != (len(numbers),)
        or not np.all(np.isfinite(neutral))
        or np.any(neutral <= 0.0)
        or not np.isclose(np.sum(neutral), 1.0, rtol=1e-7, atol=1e-9)
    ):
        raise ValueError("Neutral number probabilities must be positive and sum to 1.")
    if traits != list(DEFAULT_COLORS):
        raise ValueError(
            "Functional diagnostic currently requires the canonical 18-color order."
        )
    missing_traits = [
        trait for trait in config.evaluation_traits if trait not in traits
    ]
    if missing_traits:
        raise ValueError(f"Unknown evaluation traits: {missing_traits}")

    return SharedExperiment(
        traits=traits,
        numbers=numbers,
        questions=questions,
        validation_question_indices=validation_indices,
        model_id=str(first_config["model_id"]),
        favorite_number_system_prompt_template=str(
            first_config["favorite_number_system_prompt_template"]
        ),
        assistant_prefill=str(first_config["assistant_prefill"]),
        train_question_count=train_question_count,
        intervention_scale=float(first_config["intervention_scale"]),
        neutral_number_probabilities=neutral,
        variant_payloads=payloads,
    )


def load_seed_checkpoints(
    *,
    config: DiagnosticConfig,
    experiment: SharedExperiment,
) -> dict[str, list[SeedCheckpoint]]:
    """Load all beta/T2 checkpoints referenced by every consensus report."""
    checkpoints: dict[str, list[SeedCheckpoint]] = {}
    for variant in config.variants:
        report_config = experiment.variant_payloads[variant.name]["config"]
        checkpoint_root = Path(str(report_config["checkpoint_root"]))
        seed_runs = report_config.get("seed_runs", [])
        if not isinstance(seed_runs, list) or len(seed_runs) < 2:
            raise ValueError(f"Variant {variant.name!r} needs at least two seeds.")
        loaded = []
        for seed_run in seed_runs:
            seed = int(seed_run["seed"])
            output_stem = str(seed_run["output_stem"])
            checkpoint_dir = checkpoint_root / (
                f"{output_stem}_behavior_beta_checkpoints"
            )
            for trait in experiment.traits:
                path = checkpoint_dir / f"{safe_label(trait)}.npz"
                if not path.is_file():
                    raise FileNotFoundError(path)
                try:
                    with np.load(path, allow_pickle=False) as data:
                        saved_trait = str(data["trait"].item())
                        beta = np.asarray(data["beta"], dtype=np.float64)
                        intervention = np.asarray(
                            data["intervention"], dtype=np.float64
                        )
                except (OSError, KeyError, ValueError) as exc:
                    raise ValueError(f"Could not load checkpoint {path}.") from exc
                if (
                    saved_trait != trait
                    or beta.shape != (len(experiment.numbers),)
                    or intervention.ndim != 1
                    or not np.all(np.isfinite(beta))
                    or not np.all(np.isfinite(intervention))
                ):
                    raise ValueError(f"Invalid beta/T2 arrays in {path}.")
                loaded.append(
                    SeedCheckpoint(
                        variant=variant.name,
                        trait=trait,
                        seed=seed,
                        output_stem=output_stem,
                        path=path,
                        beta=beta,
                        intervention=intervention,
                    )
                )
        checkpoints[variant.name] = loaded
    return checkpoints


def evaluation_cache_signature(
    *,
    experiment: SharedExperiment,
    checkpoint: SeedCheckpoint | None,
    fisher_epsilon: float,
) -> str:
    """Identify one neutral or checkpoint evaluation cache exactly."""
    payload: dict[str, Any] = {
        "analysis_version": ANALYSIS_VERSION,
        "model_id": experiment.model_id,
        "traits": experiment.traits,
        "numbers": experiment.numbers,
        "validation_questions": [
            experiment.questions[index]
            for index in experiment.validation_question_indices
        ],
        "favorite_number_system_prompt_template": (
            experiment.favorite_number_system_prompt_template
        ),
        "assistant_prefill": experiment.assistant_prefill,
        "intervention_scale": experiment.intervention_scale,
        "fisher_epsilon": fisher_epsilon,
    }
    if checkpoint is None:
        payload["checkpoint"] = "neutral"
    else:
        payload["checkpoint"] = {
            "variant": checkpoint.variant,
            "trait": checkpoint.trait,
            "seed": checkpoint.seed,
            "beta_hash": array_hash(checkpoint.beta),
            "intervention_hash": array_hash(checkpoint.intervention),
        }
    return stable_json_hash(payload)


def neutral_cache_path(cache_dir: Path) -> Path:
    """Return the neutral baseline evaluation cache path."""
    return cache_dir / "neutral_validation_color_logprobs.npz"


def checkpoint_cache_path(cache_dir: Path, checkpoint: SeedCheckpoint) -> Path:
    """Return one checkpoint's evaluation cache path."""
    return cache_dir / (
        f"{safe_label(checkpoint.variant)}__seed{checkpoint.seed}__"
        f"{safe_label(checkpoint.trait)}.npz"
    )


def load_evaluation_cache(
    *,
    path: Path,
    signature: str,
    expected_shape: tuple[int, ...],
    require_local_derivative: bool,
) -> tuple[np.ndarray, np.ndarray | None] | None:
    """Load one compatible evaluation cache, if present."""
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["signature"].item()) != signature:
                return None
            logprobs = np.asarray(data["logprobs"], dtype=np.float64)
            derivative = (
                np.asarray(data["local_logprob_derivative"], dtype=np.float64)
                if "local_logprob_derivative" in data.files
                else None
            )
    except (OSError, KeyError, ValueError):
        return None
    if logprobs.shape != expected_shape or not np.all(np.isfinite(logprobs)):
        return None
    if require_local_derivative and (
        derivative is None
        or derivative.shape != expected_shape
        or not np.all(np.isfinite(derivative))
    ):
        return None
    return logprobs, derivative


def save_evaluation_cache(
    *,
    path: Path,
    signature: str,
    logprobs: np.ndarray,
    local_derivative: np.ndarray | None = None,
) -> None:
    """Atomically save one validation function evaluation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(
        path.suffix + f".{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    payload: dict[str, Any] = {
        "signature": np.asarray(signature),
        "logprobs": np.asarray(logprobs, dtype=np.float64),
    }
    if local_derivative is not None:
        payload["local_logprob_derivative"] = np.asarray(
            local_derivative, dtype=np.float64
        )
    with tmp_path.open("wb") as f:
        np.savez_compressed(f, **payload)
    tmp_path.replace(path)
    logger.success("Saved functional evaluation cache to {}", path)


def score_validation_logprobs(
    *,
    model: Any,
    input_embedding: Any,
    contexts: dict[tuple[int, int], Any],
    validation_indices: list[tuple[int, int]],
    number_embedding_matrix: Any,
    beta: Any,
    intervention_scale: float,
    context_batch_size: int,
    number_anchor_slots: Any | None = None,
) -> np.ndarray:
    """Score all candidate colors on the fixed held-out Cartesian set."""
    import torch

    if not validation_indices:
        raise ValueError("Validation context indices must be non-empty.")
    rows = []
    with torch.no_grad():
        for start in range(0, len(validation_indices), context_batch_size):
            batch_indices = validation_indices[start : start + context_batch_size]
            values = exact_candidate_logprob_matrix(
                model=model,
                input_embedding=input_embedding,
                contexts=[contexts[index] for index in batch_indices],
                number_embedding_matrix=number_embedding_matrix,
                beta=beta,
                intervention_scale=intervention_scale,
                number_anchor_slots=number_anchor_slots,
            )
            rows.append(values.detach().float().cpu().numpy())
    flat = np.concatenate(rows, axis=0).astype(np.float64)
    color_count = flat.shape[1]
    number_count = len({number_index for number_index, _ in validation_indices})
    question_count = len(
        {question_index for _, question_index in validation_indices}
    )
    expected_count = number_count * question_count
    if flat.shape != (expected_count, color_count):
        raise ValueError("Validation log-probability rows have invalid shape.")
    # validation_indices are question-major, then number-major.
    return flat.reshape(question_count, number_count, color_count).transpose(1, 0, 2)


def _checkpoint_tensor(checkpoint: SeedCheckpoint, device: Any) -> Any:
    """Create a float32 beta tensor on the model device."""
    import torch

    return torch.as_tensor(checkpoint.beta, dtype=torch.float32, device=device)


def evaluate_existing_vectors(
    *,
    config: DiagnosticConfig,
    experiment: SharedExperiment,
    checkpoints: dict[str, list[SeedCheckpoint]],
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray | None],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    """Evaluate and cache neutral and saved-seed validation functions."""
    import torch

    paths = output_paths(config)
    expected_shape = (
        len(experiment.numbers),
        len(experiment.validation_question_indices),
        len(experiment.traits),
    )
    device = resolve_device(torch, config.device)
    torch_dtype = resolve_torch_dtype(torch, config.torch_dtype)
    model, tokenizer = load_direct_model_and_tokenizer(
        model_id=experiment.model_id,
        device=device,
        torch_dtype=torch_dtype,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    input_embedding = model.get_input_embeddings()
    if input_embedding is None:
        raise RuntimeError("Model does not expose input embeddings.")
    number_embedding_matrix, number_token_ids = build_number_embedding_matrix(
        tokenizer=tokenizer,
        input_embedding=input_embedding,
        numbers=experiment.numbers,
    )
    number_embedding_values = (
        number_embedding_matrix.detach().float().cpu().numpy().astype(np.float64)
    )
    for variant_checkpoints in checkpoints.values():
        for checkpoint in variant_checkpoints:
            recomputed = (
                checkpoint.beta
                @ number_embedding_values
                * experiment.intervention_scale
            )
            if not np.allclose(
                recomputed,
                checkpoint.intervention,
                rtol=2e-4,
                atol=2e-5,
            ):
                raise ValueError(
                    "Saved beta does not reconstruct the applied T2 under the "
                    f"loaded number basis: {checkpoint.path}"
                )
    contexts = build_behavior_contexts(
        tokenizer=tokenizer,
        numbers=experiment.numbers,
        questions=experiment.questions,
        colors=experiment.traits,
        favorite_number_system_prompt_template=(
            experiment.favorite_number_system_prompt_template
        ),
        assistant_prefill=experiment.assistant_prefill,
    )
    validate_ordered_context_tokenization(
        contexts=contexts,
        number_token_ids=number_token_ids,
    )
    validation_indices = [
        (number_index, question_index)
        for question_index in experiment.validation_question_indices
        for number_index in range(len(experiment.numbers))
    ]

    neutral_signature = evaluation_cache_signature(
        experiment=experiment,
        checkpoint=None,
        fisher_epsilon=config.fisher_epsilon,
    )
    neutral_path = neutral_cache_path(paths["cache_dir"])
    neutral_cache = (
        load_evaluation_cache(
            path=neutral_path,
            signature=neutral_signature,
            expected_shape=expected_shape,
            require_local_derivative=False,
        )
        if config.skip_existing
        else None
    )
    if neutral_cache is None:
        logger.info("Evaluating neutral validation color log probabilities")
        zero_beta = torch.zeros(
            len(experiment.numbers), dtype=torch.float32, device=device
        )
        neutral_logprobs = score_validation_logprobs(
            model=model,
            input_embedding=input_embedding,
            contexts=contexts,
            validation_indices=validation_indices,
            number_embedding_matrix=number_embedding_matrix,
            beta=zero_beta,
            intervention_scale=experiment.intervention_scale,
            context_batch_size=config.context_batch_size,
        )
        save_evaluation_cache(
            path=neutral_path,
            signature=neutral_signature,
            logprobs=neutral_logprobs,
        )
    else:
        neutral_logprobs = neutral_cache[0]
        logger.info("Reusing neutral functional evaluation {}", neutral_path)

    variant_logprobs: dict[str, np.ndarray] = {}
    variant_derivatives: dict[str, np.ndarray | None] = {}
    variant_betas: dict[str, np.ndarray] = {}
    variant_interventions: dict[str, np.ndarray] = {}
    for variant in config.variants:
        variant_checkpoints = checkpoints[variant.name]
        seeds = sorted({checkpoint.seed for checkpoint in variant_checkpoints})
        seed_to_index = {seed: index for index, seed in enumerate(seeds)}
        trait_to_index = {
            trait: index for index, trait in enumerate(experiment.traits)
        }
        logprob_values = np.full(
            (len(experiment.traits), len(seeds), *expected_shape),
            np.nan,
            dtype=np.float64,
        )
        derivative_values = (
            np.full_like(logprob_values, np.nan)
            if config.fisher_epsilon > 0.0
            else None
        )
        beta_values = np.full(
            (len(experiment.traits), len(seeds), len(experiment.numbers)),
            np.nan,
            dtype=np.float64,
        )
        intervention_size = variant_checkpoints[0].intervention.size
        intervention_values = np.full(
            (len(experiment.traits), len(seeds), intervention_size),
            np.nan,
            dtype=np.float64,
        )
        requested = set(config.evaluation_traits)
        for job_index, checkpoint in enumerate(variant_checkpoints):
            trait_index = trait_to_index[checkpoint.trait]
            seed_index = seed_to_index[checkpoint.seed]
            beta_values[trait_index, seed_index] = checkpoint.beta
            intervention_values[trait_index, seed_index] = checkpoint.intervention
            cache_path = checkpoint_cache_path(paths["cache_dir"], checkpoint)
            signature = evaluation_cache_signature(
                experiment=experiment,
                checkpoint=checkpoint,
                fisher_epsilon=config.fisher_epsilon,
            )
            cached = (
                load_evaluation_cache(
                    path=cache_path,
                    signature=signature,
                    expected_shape=expected_shape,
                    require_local_derivative=config.fisher_epsilon > 0.0,
                )
                if config.skip_existing
                else None
            )
            if cached is not None:
                logprob_values[trait_index, seed_index] = cached[0]
                if derivative_values is not None and cached[1] is not None:
                    derivative_values[trait_index, seed_index] = cached[1]
                logger.info(
                    "Reusing function cache variant={} trait={} seed={}",
                    variant.name,
                    checkpoint.trait,
                    checkpoint.seed,
                )
                continue
            if checkpoint.trait not in requested:
                continue
            logger.info(
                "Evaluating variant={} trait={} seed={} ({}/{})",
                variant.name,
                checkpoint.trait,
                checkpoint.seed,
                job_index + 1,
                len(variant_checkpoints),
            )
            beta = _checkpoint_tensor(checkpoint, device)
            checkpoint_logprobs = score_validation_logprobs(
                model=model,
                input_embedding=input_embedding,
                contexts=contexts,
                validation_indices=validation_indices,
                number_embedding_matrix=number_embedding_matrix,
                beta=beta,
                intervention_scale=experiment.intervention_scale,
                context_batch_size=config.context_batch_size,
            )
            local_derivative = None
            if config.fisher_epsilon > 0.0:
                epsilon_logprobs = score_validation_logprobs(
                    model=model,
                    input_embedding=input_embedding,
                    contexts=contexts,
                    validation_indices=validation_indices,
                    number_embedding_matrix=number_embedding_matrix,
                    beta=beta * config.fisher_epsilon,
                    intervention_scale=experiment.intervention_scale,
                    context_batch_size=config.context_batch_size,
                )
                local_derivative = (
                    epsilon_logprobs - neutral_logprobs
                ) / config.fisher_epsilon
            save_evaluation_cache(
                path=cache_path,
                signature=signature,
                logprobs=checkpoint_logprobs,
                local_derivative=local_derivative,
            )
            logprob_values[trait_index, seed_index] = checkpoint_logprobs
            if derivative_values is not None and local_derivative is not None:
                derivative_values[trait_index, seed_index] = local_derivative

        variant_logprobs[variant.name] = logprob_values
        variant_derivatives[variant.name] = derivative_values
        variant_betas[variant.name] = beta_values
        variant_interventions[variant.name] = intervention_values

    del contexts
    del number_embedding_matrix
    del input_embedding
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (
        neutral_logprobs,
        variant_logprobs,
        variant_derivatives,
        variant_betas,
        variant_interventions,
    )


def cosine_similarity(
    first: np.ndarray,
    second: np.ndarray,
    weights: np.ndarray | None = None,
) -> float | None:
    """Compute a finite weighted or unweighted cosine similarity."""
    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    if x.shape != y.shape or x.size == 0:
        raise ValueError("Cosine inputs must have equal non-empty shapes.")
    if weights is None:
        w = np.ones_like(x)
    else:
        w = np.broadcast_to(np.asarray(weights, dtype=np.float64), x.shape)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return None
    numerator = float(np.sum(w * x * y))
    denominator = math.sqrt(float(np.sum(w * x * x) * np.sum(w * y * y)))
    if denominator <= 0.0 or not math.isfinite(denominator):
        return None
    return numerator / denominator


def pearson_correlation(
    first: np.ndarray,
    second: np.ndarray,
    weights: np.ndarray | None = None,
) -> float | None:
    """Compute a finite weighted or unweighted Pearson correlation."""
    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    if x.shape != y.shape or x.size < 2:
        raise ValueError("Correlation inputs must have equal nontrivial shapes.")
    if weights is None:
        w = np.ones_like(x)
    else:
        w = np.broadcast_to(np.asarray(weights, dtype=np.float64), x.shape)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return None
    weight_sum = float(np.sum(w))
    if weight_sum <= 0.0 or not math.isfinite(weight_sum):
        return None
    x_centered = x - float(np.sum(w * x) / weight_sum)
    y_centered = y - float(np.sum(w * y) / weight_sum)
    return cosine_similarity(x_centered, y_centered, w)


def log_softmax_numpy(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute a stable NumPy log-softmax."""
    array = np.asarray(values, dtype=np.float64)
    maximum = np.max(array, axis=axis, keepdims=True)
    shifted = array - maximum
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


def candidate_distribution_distances(
    first_logprobs: np.ndarray,
    second_logprobs: np.ndarray,
    context_weights: np.ndarray,
) -> dict[str, float]:
    """Compare selected-color-normalized seed distributions per context."""
    log_p = log_softmax_numpy(first_logprobs)
    log_q = log_softmax_numpy(second_logprobs)
    p = np.exp(log_p)
    q = np.exp(log_q)
    mixture = 0.5 * (p + q)
    log_mixture = np.log(np.clip(mixture, np.finfo(np.float64).tiny, None))
    js_by_context = 0.5 * (
        np.sum(p * (log_p - log_mixture), axis=-1)
        + np.sum(q * (log_q - log_mixture), axis=-1)
    )
    symmetric_kl_by_context = 0.5 * (
        np.sum(p * (log_p - log_q), axis=-1)
        + np.sum(q * (log_q - log_p), axis=-1)
    )
    total_variation_by_context = 0.5 * np.sum(np.abs(p - q), axis=-1)
    weights = np.asarray(context_weights, dtype=np.float64)
    weight_sum = float(np.sum(weights))
    return {
        "candidate_js_nats": float(np.sum(weights * js_by_context) / weight_sum),
        "candidate_symmetric_kl_nats": float(
            np.sum(weights * symmetric_kl_by_context) / weight_sum
        ),
        "candidate_total_variation": float(
            np.sum(weights * total_variation_by_context) / weight_sum
        ),
    }


def center_functional_shifts(shifts: np.ndarray) -> dict[str, np.ndarray]:
    """Return raw and question/color/trait-centered function tensors.

    Input orientation is trait x seed x number x question x output color.
    """
    values = np.asarray(shifts, dtype=np.float64)
    if values.ndim != 5 or not np.all(np.isfinite(values)):
        raise ValueError("Functional shifts must be a finite five-dimensional array.")

    def center_axis(array: np.ndarray, axis: int) -> np.ndarray:
        return array - np.mean(array, axis=axis, keepdims=True)

    fully_centered = center_axis(values, 0)
    fully_centered = center_axis(fully_centered, 3)
    fully_centered = center_axis(fully_centered, 4)
    return {
        "raw": values,
        "question_centered": center_axis(values, 3),
        "color_centered": center_axis(values, 4),
        "trait_centered": center_axis(values, 0),
        "question_color_trait_centered": fully_centered,
    }


def validation_fisher_cosine(
    first_derivative: np.ndarray,
    second_derivative: np.ndarray,
    neutral_logprobs: np.ndarray,
    context_weights: np.ndarray,
) -> float | None:
    """Compute selected-candidate validation Fisher cosine from directional JVPs."""
    log_probabilities = log_softmax_numpy(neutral_logprobs)
    probabilities = np.exp(log_probabilities)
    first_score = first_derivative - np.sum(
        probabilities * first_derivative, axis=-1, keepdims=True
    )
    second_score = second_derivative - np.sum(
        probabilities * second_derivative, axis=-1, keepdims=True
    )
    weights = np.asarray(context_weights, dtype=np.float64)[..., None]
    fisher_weights = weights * probabilities
    return cosine_similarity(first_score, second_score, fisher_weights)


def _finite_or_none(value: float | None) -> float | None:
    """Return a finite float or None for JSON/CSV serialization."""
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def build_pair_records(
    *,
    config: DiagnosticConfig,
    experiment: SharedExperiment,
    neutral_logprobs: np.ndarray,
    variant_logprobs: dict[str, np.ndarray],
    variant_derivatives: dict[str, np.ndarray | None],
    variant_interventions: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Build all per-trait, per-seed-pair geometry/function comparisons."""
    context_weights = np.broadcast_to(
        experiment.neutral_number_probabilities[:, None],
        neutral_logprobs.shape[:2],
    )
    full_weights = context_weights[..., None]
    records = []
    for variant in config.variants:
        logprobs = variant_logprobs[variant.name]
        if not np.all(np.isfinite(logprobs)):
            raise ValueError(
                f"Variant {variant.name!r} does not have complete cached evaluations."
            )
        shifts = logprobs - neutral_logprobs[None, None, :, :, :]
        representations = center_functional_shifts(shifts)
        derivatives = variant_derivatives[variant.name]
        interventions = variant_interventions[variant.name]
        seed_count = logprobs.shape[1]
        report_config = experiment.variant_payloads[variant.name]["config"]
        seeds = sorted(int(run["seed"]) for run in report_config["seed_runs"])
        if len(seeds) != seed_count:
            raise ValueError("Seed labels do not match functional tensor shape.")
        for trait_index, trait in enumerate(experiment.traits):
            target_index = experiment.traits.index(trait)
            for first_seed_index, second_seed_index in combinations(
                range(seed_count), 2
            ):
                first_logprobs = logprobs[trait_index, first_seed_index]
                second_logprobs = logprobs[trait_index, second_seed_index]
                record: dict[str, Any] = {
                    "variant": variant.name,
                    "trait": trait,
                    "seed_a": seeds[first_seed_index],
                    "seed_b": seeds[second_seed_index],
                    "T2_euclidean_cosine": _finite_or_none(
                        cosine_similarity(
                            interventions[trait_index, first_seed_index],
                            interventions[trait_index, second_seed_index],
                        )
                    ),
                }
                for representation_name, representation in representations.items():
                    first = representation[trait_index, first_seed_index]
                    second = representation[trait_index, second_seed_index]
                    prefix = f"{representation_name}_full_shift"
                    record[f"{prefix}_cosine"] = _finite_or_none(
                        cosine_similarity(first, second)
                    )
                    record[f"{prefix}_pearson"] = _finite_or_none(
                        pearson_correlation(first, second)
                    )
                    record[f"{prefix}_weighted_cosine"] = _finite_or_none(
                        cosine_similarity(first, second, full_weights)
                    )
                    record[f"{prefix}_weighted_pearson"] = _finite_or_none(
                        pearson_correlation(first, second, full_weights)
                    )
                    first_target = first[..., target_index]
                    second_target = second[..., target_index]
                    target_prefix = f"{representation_name}_target_shift"
                    record[f"{target_prefix}_cosine"] = _finite_or_none(
                        cosine_similarity(first_target, second_target)
                    )
                    record[f"{target_prefix}_pearson"] = _finite_or_none(
                        pearson_correlation(first_target, second_target)
                    )
                    record[f"{target_prefix}_weighted_cosine"] = _finite_or_none(
                        cosine_similarity(
                            first_target, second_target, context_weights
                        )
                    )
                    record[f"{target_prefix}_weighted_pearson"] = _finite_or_none(
                        pearson_correlation(
                            first_target, second_target, context_weights
                        )
                    )
                record.update(
                    candidate_distribution_distances(
                        first_logprobs,
                        second_logprobs,
                        context_weights,
                    )
                )
                if derivatives is not None:
                    record["validation_fisher_cosine"] = _finite_or_none(
                        validation_fisher_cosine(
                            derivatives[trait_index, first_seed_index],
                            derivatives[trait_index, second_seed_index],
                            neutral_logprobs,
                            context_weights,
                        )
                    )
                else:
                    record["validation_fisher_cosine"] = None
                records.append(record)
    return records


def _numeric_values(records: list[dict[str, Any]], key: str) -> list[float]:
    """Extract finite values for one metric key."""
    return [
        float(record[key])
        for record in records
        if record.get(key) is not None and math.isfinite(float(record[key]))
    ]


def summarize_pair_records(
    records: list[dict[str, Any]],
    variants: list[VariantSpec],
    traits: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Summarize seed-pair metrics per trait and per variant."""
    metric_keys = [
        "T2_euclidean_cosine",
        "raw_full_shift_weighted_pearson",
        "question_centered_full_shift_weighted_pearson",
        "color_centered_full_shift_weighted_pearson",
        "trait_centered_full_shift_weighted_pearson",
        "question_color_trait_centered_full_shift_weighted_pearson",
        "raw_target_shift_weighted_pearson",
        "trait_centered_target_shift_weighted_pearson",
        "candidate_js_nats",
        "candidate_symmetric_kl_nats",
        "candidate_total_variation",
        "validation_fisher_cosine",
    ]
    trait_summaries = []
    variant_summaries: dict[str, Any] = {}
    for variant in variants:
        variant_records = [
            record for record in records if record["variant"] == variant.name
        ]
        for trait in traits:
            selected = [
                record for record in variant_records if record["trait"] == trait
            ]
            row: dict[str, Any] = {
                "variant": variant.name,
                "trait": trait,
                "seed_pair_count": len(selected),
            }
            for key in metric_keys:
                values = _numeric_values(selected, key)
                row[f"min_{key}"] = min(values) if values else None
                row[f"mean_{key}"] = (
                    float(np.mean(values)) if values else None
                )
                row[f"max_{key}"] = max(values) if values else None
            trait_summaries.append(row)

        geometry = _numeric_values(variant_records, "T2_euclidean_cosine")
        raw_function = _numeric_values(
            variant_records, "raw_full_shift_weighted_pearson"
        )
        trait_function = _numeric_values(
            variant_records, "trait_centered_full_shift_weighted_pearson"
        )
        fisher = _numeric_values(variant_records, "validation_fisher_cosine")
        geometry_function_pairs = [
            record
            for record in variant_records
            if record.get("T2_euclidean_cosine") is not None
            and record.get("raw_full_shift_weighted_pearson") is not None
        ]
        functionally_close_low_geometry = [
            record
            for record in geometry_function_pairs
            if float(record["T2_euclidean_cosine"]) < 0.3
            and float(record["raw_full_shift_weighted_pearson"]) >= 0.9
        ]
        variant_summaries[variant.name] = {
            "seed_pair_record_count": len(variant_records),
            "median_T2_euclidean_cosine": (
                float(np.median(geometry)) if geometry else None
            ),
            "median_raw_full_shift_weighted_pearson": (
                float(np.median(raw_function)) if raw_function else None
            ),
            "median_trait_centered_full_shift_weighted_pearson": (
                float(np.median(trait_function)) if trait_function else None
            ),
            "median_validation_fisher_cosine": (
                float(np.median(fisher)) if fisher else None
            ),
            "pairs_with_raw_function_pearson_at_least_0_9": sum(
                value >= 0.9 for value in raw_function
            ),
            "pairs_with_T2_cosine_below_0_3_and_raw_function_pearson_at_least_0_9": (
                len(functionally_close_low_geometry)
            ),
        }
    return trait_summaries, variant_summaries


def write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a non-empty flat dictionary table."""
    if not rows:
        raise ValueError(f"Cannot write empty table to {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.success("Saved functional diagnostic table to {}", path)


def write_metric_heatmap(
    *,
    path: Path,
    trait_summaries: list[dict[str, Any]],
    variants: list[VariantSpec],
    traits: list[str],
    dpi: int,
) -> None:
    """Plot each trait's worst seed-pair agreement across core metrics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_specs = [
        ("min_T2_euclidean_cosine", "T2 cosine"),
        ("min_raw_full_shift_weighted_pearson", "Raw D Pearson"),
        (
            "min_question_centered_full_shift_weighted_pearson",
            "Question-centered",
        ),
        ("min_color_centered_full_shift_weighted_pearson", "Color-centered"),
        ("min_trait_centered_full_shift_weighted_pearson", "Trait-centered"),
        (
            "min_question_color_trait_centered_full_shift_weighted_pearson",
            "All centered",
        ),
        ("min_raw_target_shift_weighted_pearson", "Target D Pearson"),
        ("min_validation_fisher_cosine", "Fisher cosine"),
    ]
    fig, axes = plt.subplots(
        len(variants),
        1,
        figsize=(14, max(5.2, 4.7 * len(variants))),
        squeeze=False,
    )
    for axis, variant in zip(axes[:, 0], variants):
        selected = {
            row["trait"]: row
            for row in trait_summaries
            if row["variant"] == variant.name
        }
        values = np.array(
            [
                [
                    np.nan
                    if selected[trait].get(metric_key) is None
                    else float(selected[trait][metric_key])
                    for metric_key, _ in metric_specs
                ]
                for trait in traits
            ],
            dtype=np.float64,
        )
        image = axis.imshow(values, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
        axis.set_title(f"{variant.name}: minimum agreement across seed pairs")
        axis.set_xticks(
            np.arange(len(metric_specs)),
            [label for _, label in metric_specs],
            rotation=30,
            ha="right",
        )
        axis.set_yticks(np.arange(len(traits)), traits)
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                value = values[row_index, column_index]
                if np.isfinite(value):
                    axis.text(
                        column_index,
                        row_index,
                        f"{value:+.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if abs(value) > 0.55 else "black",
                    )
        fig.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    fig.suptitle(
        "Existing ordered-T2 vectors: geometry versus held-out function",
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    logger.success("Saved function-space agreement heatmap to {}", path)


def write_geometry_function_scatter(
    *,
    path: Path,
    records: list[dict[str, Any]],
    variants: list[VariantSpec],
    dpi: int,
) -> None:
    """Plot Euclidean T2 cosine against raw and trait-centered function agreement."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1,
        len(variants),
        figsize=(7.2 * len(variants), 6.0),
        squeeze=False,
    )
    for axis, variant in zip(axes[0], variants):
        selected = [
            record for record in records if record["variant"] == variant.name
        ]
        x = np.asarray(
            [record["T2_euclidean_cosine"] for record in selected],
            dtype=np.float64,
        )
        raw_y = np.asarray(
            [record["raw_full_shift_weighted_pearson"] for record in selected],
            dtype=np.float64,
        )
        centered_y = np.asarray(
            [
                record["trait_centered_full_shift_weighted_pearson"]
                for record in selected
            ],
            dtype=np.float64,
        )
        axis.scatter(x, raw_y, s=34, alpha=0.72, label="Raw D function")
        axis.scatter(
            x,
            centered_y,
            s=34,
            alpha=0.72,
            marker="x",
            label="Trait-centered D function",
        )
        axis.axhline(0.9, color="#8a6414", linestyle="--", linewidth=1)
        axis.axvline(0.3, color="#59636f", linestyle=":", linewidth=1)
        axis.set_xlim(-1.02, 1.02)
        axis.set_ylim(-1.02, 1.02)
        axis.set_xlabel("Euclidean cosine between saved T2 vectors")
        axis.set_ylabel("Held-out weighted Pearson between D tensors")
        axis.set_title(variant.name)
        axis.grid(alpha=0.2)
        axis.legend(loc="lower right")
    fig.suptitle(
        "Do geometrically different seed solutions implement the same function?"
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    logger.success("Saved geometry/function scatter to {}", path)


def write_outputs(
    *,
    config: DiagnosticConfig,
    experiment: SharedExperiment,
    neutral_logprobs: np.ndarray,
    variant_logprobs: dict[str, np.ndarray],
    variant_derivatives: dict[str, np.ndarray | None],
    variant_betas: dict[str, np.ndarray],
    variant_interventions: dict[str, np.ndarray],
    pair_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Write tensors, tables, plots, and the self-describing JSON report."""
    paths = output_paths(config)
    trait_summaries, variant_summaries = summarize_pair_records(
        pair_records,
        config.variants,
        experiment.traits,
    )
    paths["cache_dir"].mkdir(parents=True, exist_ok=True)
    tensor_payload: dict[str, np.ndarray] = {
        "neutral_validation_color_logprobs": neutral_logprobs,
        "traits": np.asarray(experiment.traits),
        "numbers": np.asarray(experiment.numbers),
        "validation_questions": np.asarray(
            [
                experiment.questions[index]
                for index in experiment.validation_question_indices
            ]
        ),
        "neutral_number_probabilities": experiment.neutral_number_probabilities,
    }
    for variant in config.variants:
        prefix = safe_label(variant.name)
        tensor_payload[f"{prefix}__validation_color_logprobs"] = (
            variant_logprobs[variant.name]
        )
        tensor_payload[f"{prefix}__validation_logprob_shifts"] = (
            variant_logprobs[variant.name]
            - neutral_logprobs[None, None, :, :, :]
        )
        tensor_payload[f"{prefix}__behavior_betas"] = variant_betas[variant.name]
        tensor_payload[f"{prefix}__ordered_T2_interventions"] = (
            variant_interventions[variant.name]
        )
        derivatives = variant_derivatives[variant.name]
        if derivatives is not None:
            tensor_payload[f"{prefix}__local_logprob_derivatives"] = derivatives
        report_config = experiment.variant_payloads[variant.name]["config"]
        tensor_payload[f"{prefix}__seeds"] = np.asarray(
            sorted(int(run["seed"]) for run in report_config["seed_runs"]),
            dtype=np.int64,
        )
    np.savez_compressed(paths["tensors"], **tensor_payload)
    logger.success("Saved functional-equivalence tensors to {}", paths["tensors"])
    write_dict_csv(paths["pair_metrics_csv"], pair_records)
    write_dict_csv(paths["trait_summary_csv"], trait_summaries)
    if config.save_plot:
        write_metric_heatmap(
            path=paths["heatmap"],
            trait_summaries=trait_summaries,
            variants=config.variants,
            traits=experiment.traits,
            dpi=config.heatmap_dpi,
        )
        write_geometry_function_scatter(
            path=paths["scatter"],
            records=pair_records,
            variants=config.variants,
            dpi=config.heatmap_dpi,
        )

    report = {
        "analysis_version": ANALYSIS_VERSION,
        "method": "existing_ordered_T2_heldout_functional_equivalence",
        "config": serialize_config(config),
        "experiment": {
            "model_id": experiment.model_id,
            "traits": experiment.traits,
            "numbers": experiment.numbers,
            "validation_question_indices": experiment.validation_question_indices,
            "validation_questions": [
                experiment.questions[index]
                for index in experiment.validation_question_indices
            ],
            "validation_context_count": (
                len(experiment.numbers)
                * len(experiment.validation_question_indices)
            ),
            "output_color_count": len(experiment.traits),
            "source_consensus_reports": {
                variant.name: str(variant.report_path)
                for variant in config.variants
            },
        },
        "definitions": {
            "functional_shift": (
                "D_zs(n,q,c)=log p(c|n,q,T2_zs)-log p(c|n,q,T2=0), "
                "using raw exact full-vocabulary continuation log probabilities."
            ),
            "candidate_distribution": (
                "For JS/KL/TV only, normalize the 18 exact color continuation "
                "log probabilities within each held-out context."
            ),
            "context_weighting": (
                "Weighted metrics average questions uniformly and numbers by the "
                "saved neutral favorite-number marginal."
            ),
            "question_centering": "Subtract mean_q D for each trait/seed/number/color.",
            "color_centering": "Subtract mean_c D for each trait/seed/number/question.",
            "trait_centering": "Subtract mean_z D separately within every seed.",
            "all_centering": (
                "Apply commuting trait, question, and output-color centering "
                "projections."
            ),
            "validation_fisher_cosine": (
                "Selected-candidate Fisher cosine at neutral. Directional score "
                "derivatives use [log p(epsilon*T2)-log p(0)]/epsilon, are centered "
                "under the neutral 18-color candidate distribution, and use the "
                "neutral number marginal over held-out contexts."
            ),
            "interpretation": (
                "Low Euclidean T2 cosine with high D-tensor agreement and small "
                "candidate JS indicates functional equivalence. Low agreement in "
                "both spaces indicates behavior-level seed instability."
            ),
        },
        "variant_summaries": variant_summaries,
        "per_trait_summaries": trait_summaries,
        "seed_pair_metrics": pair_records,
        "outputs": {
            key: str(path)
            for key, path in paths.items()
            if key != "cache_dir"
        }
        | {"evaluation_cache_dir": str(paths["cache_dir"])},
    }
    paths["json"].write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.success("Saved functional-equivalence report to {}", paths["json"])
    for variant_name, summary in variant_summaries.items():
        logger.info(
            "{}: median T2 cosine={}, median raw-function Pearson={}, "
            "median trait-centered-function Pearson={}, median Fisher cosine={}, "
            "low-geometry/high-function pairs={}",
            variant_name,
            summary["median_T2_euclidean_cosine"],
            summary["median_raw_full_shift_weighted_pearson"],
            summary["median_trait_centered_full_shift_weighted_pearson"],
            summary["median_validation_fisher_cosine"],
            summary[
                "pairs_with_T2_cosine_below_0_3_and_raw_function_pearson_at_least_0_9"
            ],
        )
    return report


def validate_config(config: DiagnosticConfig) -> None:
    """Validate local diagnostic settings before model loading."""
    if config.context_batch_size <= 0:
        raise ValueError("--context-batch-size must be positive.")
    if not math.isfinite(config.fisher_epsilon) or config.fisher_epsilon < 0.0:
        raise ValueError("--fisher-epsilon must be finite and nonnegative.")
    if config.heatmap_dpi <= 0:
        raise ValueError("--heatmap-dpi must be positive.")
    if not config.evaluation_traits or len(set(config.evaluation_traits)) != len(
        config.evaluation_traits
    ):
        raise ValueError("Evaluation traits must be non-empty and unique.")


def run(config: DiagnosticConfig) -> dict[str, Any]:
    """Evaluate existing vectors and compare their held-out functions."""
    validate_config(config)
    experiment = load_shared_experiment(config)
    checkpoints = load_seed_checkpoints(config=config, experiment=experiment)
    (
        neutral_logprobs,
        variant_logprobs,
        variant_derivatives,
        variant_betas,
        variant_interventions,
    ) = evaluate_existing_vectors(
        config=config,
        experiment=experiment,
        checkpoints=checkpoints,
    )
    incomplete = {
        variant.name: int(np.sum(~np.isfinite(variant_logprobs[variant.name])))
        for variant in config.variants
    }
    if any(count > 0 for count in incomplete.values()):
        logger.warning(
            "Functional caches are incomplete: {}. Run remaining traits, then "
            "rerun with all traits and --skip-existing to aggregate.",
            incomplete,
        )
        return {
            "analysis_version": ANALYSIS_VERSION,
            "method": "existing_ordered_T2_function_cache_job",
            "config": serialize_config(config),
            "incomplete_value_counts": incomplete,
            "evaluation_cache_dir": str(output_paths(config)["cache_dir"]),
        }
    pair_records = build_pair_records(
        config=config,
        experiment=experiment,
        neutral_logprobs=neutral_logprobs,
        variant_logprobs=variant_logprobs,
        variant_derivatives=variant_derivatives,
        variant_interventions=variant_interventions,
    )
    return write_outputs(
        config=config,
        experiment=experiment,
        neutral_logprobs=neutral_logprobs,
        variant_logprobs=variant_logprobs,
        variant_derivatives=variant_derivatives,
        variant_betas=variant_betas,
        variant_interventions=variant_interventions,
        pair_records=pair_records,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        action="append",
        help=(
            "NAME=CONSENSUS_JSON. Repeat to compare multiple training variants. "
            "Defaults to the Gaussian and centered-T1 three-seed reports."
        ),
    )
    parser.add_argument(
        "--trait",
        "--traits",
        dest="traits",
        action="append",
        help=(
            "Trait(s) to evaluate in this cache-building job. Omit to evaluate "
            "all 18. Partial jobs do not aggregate until every cache exists."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument(
        "--context-batch-size", type=int, default=DEFAULT_CONTEXT_BATCH_SIZE
    )
    parser.add_argument(
        "--fisher-epsilon",
        type=float,
        default=DEFAULT_FISHER_EPSILON,
        help="Set to zero to skip the finite-difference validation Fisher metric.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--heatmap-dpi", type=int, default=DEFAULT_HEATMAP_DPI)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def build_config(args: argparse.Namespace) -> DiagnosticConfig:
    """Build a validated diagnostic configuration from CLI arguments."""
    variants = (
        [parse_variant(value) for value in args.variant]
        if args.variant
        else [
            VariantSpec(
                name="gaussian_shared_init",
                report_path=DEFAULT_GAUSSIAN_REPORT,
            ),
            VariantSpec(
                name="centered_t1_init",
                report_path=DEFAULT_CENTERED_T1_REPORT,
            ),
        ]
    )
    return DiagnosticConfig(
        variants=variants,
        evaluation_traits=parse_csv_values(args.traits, list(DEFAULT_COLORS)),
        output_dir=args.output_dir,
        output_stem=args.output_stem,
        context_batch_size=args.context_batch_size,
        fisher_epsilon=args.fisher_epsilon,
        device=args.device,
        torch_dtype=args.torch_dtype,
        heatmap_dpi=args.heatmap_dpi,
        save_plot=not args.no_plot,
        skip_existing=args.skip_existing,
    )


def main() -> None:
    """Run the existing-vector function-space diagnostic."""
    parser = build_parser()
    try:
        run(build_config(parser.parse_args()))
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
