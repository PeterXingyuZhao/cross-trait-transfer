#!/usr/bin/env python3
"""Evaluate catalog M5 or M7 from cached carrier teacher distributions.

M5 replaces the written carrier embeddings with the mean carrier embedding and
adds the full-norm-matched trait-specific T1. M7 retains each real carrier's
embeddings, adds the native full T1, and averages responses using the
corresponding teacher's generated-carrier distribution.

Both anchors accept ``--direction-variant residual_rescaled`` or ``v_full``.
Omitting that option preserves the historical M5 and M7 directions above.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import sys
import time
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_teacher_distribution_similarity import (  # noqa: E402
    DistributionCache,
    load_cache,
    validate_alignment,
)
from compute_number_space_behavior_dataset_cosines import (  # noqa: E402
    build_behavior_contexts,
    build_carrier_embedding_matrix,
    build_number_embedding_matrix,
    cached_intervened_candidate_logprob_matrix,
    exact_candidate_logprob_matrix,
    validate_ordered_context_tokenization,
)
from diagnose_number_space_behavior_functional_equivalence import (  # noqa: E402
    score_validation_logprobs,
)
from find_owl_output_distribution_entanglement import (  # noqa: E402
    load_direct_model_and_tokenizer,
    parse_csv_values,
)
from score_birthday_year_probabilities import (  # noqa: E402
    resolve_device,
    resolve_torch_dtype,
)
from score_string_trait_number_distribution import slugify  # noqa: E402
from train_teacher_logits_soft_distillation import (  # noqa: E402
    DEFAULT_NEUTRAL_TEACHER_CACHE_NAME,
    model_uses_reduced_precision,
)
from metric_batch_autotune import (  # noqa: E402
    BatchCandidate,
    BatchCandidateRejectedError,
    autotune_batch_pair,
    candidate_grid,
)
from metric_exact_trie_scorer import (  # noqa: E402
    FULL_PRECISION_MAX_LOGPROB_ERROR,
    ExactTrieScoreComparison,
    compare_exact_trie_scores,
)
from metric_carrier_catalog import (  # noqa: E402
    CarrierCatalog,
    resolve_carrier_catalog,
    validate_catalog_tokenization,
)


MetricName = Literal["m5", "m7"]
DirectionVariant = Literal["residual_rescaled", "v_full"]
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-9B"
DEFAULT_PROMPT_COUNT = 1024
DEFAULT_QUESTION_COUNT = 5
DEFAULT_BATCH_SIZE = 8
DEFAULT_PERMUTATIONS = 10_000
DEFAULT_SYSTEM_PROMPT_TEMPLATE = "You love {number}."
DEFAULT_ASSISTANT_PREFILL = "My favorite city is "
DEFAULT_EPSILON = 1e-12
ANALYSIS_VERSION = 3
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(frozen=True, **DATACLASS_KW_ONLY)
class T1Config:
    """Configuration for one generic M5 or M7 run."""

    metric: MetricName
    training_output_dir: Path
    output_dir: Path
    model_id: str
    traits: list[str]
    answers: list[str]
    questions_path: Path
    question_count: int
    prompt_count: int
    system_prompt_template: str
    assistant_prefill: str
    batch_size: int
    device: str
    torch_dtype: str
    enable_thinking: bool | None
    epsilon: float
    permutations: int
    random_seed: int
    skip_existing: bool
    write_plots: bool
    carrier_manifest_path: Path | None = None
    use_prefix_cache: bool = False
    branch_batch_size: int = 8
    autotune_batch_pair: bool = False
    max_batch_size: int = 32
    max_branch_batch_size: int = 64
    direction_variant: DirectionVariant | None = None


def utc_now() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def load_string_list(path: Path, label: str) -> list[str]:
    """Load a non-empty JSON string list."""
    values = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(values, list)
        or not values
        or not all(isinstance(value, str) and value.strip() for value in values)
    ):
        raise ValueError(f"{label} must be a non-empty JSON string list.")
    normalized = [value.strip() for value in values]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} contains duplicates.")
    return normalized


def write_json(path: Path, payload: Any) -> None:
    """Write a JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def array_hash(values: np.ndarray) -> str:
    """Return a stable hash for a numeric array."""
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    """Return a stable JSON hash."""
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def normalize_rows(values: np.ndarray, epsilon: float) -> np.ndarray:
    """Normalize the final probability axis."""
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError("Teacher probabilities must be finite and non-negative.")
    totals = np.sum(array, axis=-1, keepdims=True)
    if np.any(totals <= epsilon):
        raise ValueError("A teacher probability row has zero total mass.")
    return array / totals


def load_teacher_inputs(
    config: T1Config,
) -> tuple[DistributionCache, list[DistributionCache]]:
    """Load aligned neutral and trait teacher caches."""
    root = config.training_output_dir / "teacher_distributions"
    neutral = load_cache(
        root / f"{slugify(DEFAULT_NEUTRAL_TEACHER_CACHE_NAME)}.npz",
        DEFAULT_NEUTRAL_TEACHER_CACHE_NAME,
    )
    teachers = [
        load_cache(root / f"{slugify(trait)}.npz", trait)
        for trait in config.traits
    ]
    validate_alignment(neutral, teachers)
    if not neutral.candidates or len(neutral.candidates) != len(set(neutral.candidates)):
        raise ValueError("Teacher caches must use unique, non-empty carriers.")
    if config.prompt_count > len(neutral.prompts):
        raise ValueError(
            f"Requested {config.prompt_count} prompts but only "
            f"{len(neutral.prompts)} are cached."
        )
    return neutral, teachers


def construct_t1_coefficients(
    *,
    neutral: np.ndarray,
    teachers: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return full, across-trait-specific, and teacher-marginal coefficients."""
    p0 = normalize_rows(neutral, epsilon)
    p1 = normalize_rows(teachers, epsilon)
    if p1.ndim != 3 or p1.shape[1:] != p0.shape:
        raise ValueError("T1 inputs must have trait x prompt x number shape.")
    full = np.mean(p1 - p0[None, :, :], axis=1)
    specific = full - np.mean(full, axis=0, keepdims=True)
    teacher_marginals = np.mean(p1, axis=1)
    if not np.allclose(np.sum(full, axis=1), 0.0, atol=1e-10):
        raise ValueError("Full T1 coefficients do not sum to zero.")
    if not np.allclose(np.sum(specific, axis=1), 0.0, atol=1e-10):
        raise ValueError("Specific T1 coefficients do not sum to zero.")
    return full, specific, teacher_marginals


def mean_digit_anchor_slots(number_embedding_matrix: Any, slot_count: int = 2) -> Any:
    """Return copies of the mean standalone carrier-slot embedding."""
    values = number_embedding_matrix.float()
    if slot_count <= 0 or values.ndim != 2 or values.shape[1] % slot_count:
        raise ValueError("Carrier embeddings do not match their token-slot count.")
    hidden_size = int(values.shape[1] // slot_count)
    slots = values.reshape(-1, slot_count, hidden_size) * math.sqrt(slot_count)
    mean_digit = slots.mean(dim=(0, 1))
    return mean_digit.unsqueeze(0).repeat(slot_count, 1)


def applied_coefficients(
    *,
    metric: MetricName,
    full: np.ndarray,
    specific: np.ndarray,
    number_embedding_matrix: Any,
    direction_variant: DirectionVariant | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Choose a direction independently of the synthetic or real anchor."""
    variant = resolve_direction_variant(metric, direction_variant)
    if variant == "v_full":
        return full.copy(), {
            "direction_variant": variant,
            "direction_family": "full_teacher_minus_neutral",
            "strength": "native",
            "zero_specific_norm_policy": "not_applicable",
        }
    matrix = number_embedding_matrix.detach().float().cpu().numpy()
    full_vectors = full @ matrix
    specific_vectors = specific @ matrix
    full_norms = np.linalg.norm(full_vectors, axis=1)
    specific_norms = np.linalg.norm(specific_vectors, axis=1)
    if np.any(specific_norms <= np.finfo(np.float64).tiny):
        raise ValueError("Cannot full-norm-match a zero specific T1 vector.")
    scales = full_norms / specific_norms
    return specific * scales[:, None], {
        "direction_variant": variant,
        "direction_family": "across_evaluated_teacher_mean_subtracted",
        "strength": "matched_to_native_full_T1_norm",
        "scale_by_trait": scales.tolist(),
        "full_vector_norm_by_trait": full_norms.tolist(),
        "specific_vector_norm_before_scaling_by_trait": specific_norms.tolist(),
        "zero_specific_norm_policy": "raise_value_error",
    }


def resolve_direction_variant(
    metric: MetricName,
    direction_variant: DirectionVariant | None,
) -> DirectionVariant:
    """Resolve an explicit direction or the historical per-anchor default."""
    if metric not in {"m5", "m7"}:
        raise ValueError(f"Unknown intervention metric: {metric!r}.")
    if direction_variant is None:
        return "residual_rescaled" if metric == "m5" else "v_full"
    if direction_variant not in {"residual_rescaled", "v_full"}:
        raise ValueError(f"Unknown direction variant: {direction_variant!r}.")
    return direction_variant


def reserve_output_variant(config: T1Config) -> None:
    """Prevent incompatible anchor/direction runs from sharing an output tree."""
    identity = {
        "metric": config.metric,
        "direction_variant": resolve_direction_variant(
            config.metric, config.direction_variant
        ),
    }
    path = config.output_dir / "intervention_variant.json"
    existing = None
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
    elif (config.output_dir / "manifest.json").is_file():
        manifest = json.loads(
            (config.output_dir / "manifest.json").read_text(encoding="utf-8")
        )
        old_config = manifest["config"]
        old_metric = old_config["metric"]
        existing = {
            "metric": old_metric,
            "direction_variant": resolve_direction_variant(
                old_metric, old_config.get("direction_variant")
            ),
        }
    if existing is not None and existing != identity:
        raise ValueError(
            f"Output directory {config.output_dir} belongs to {existing}; "
            f"use a separate --output-dir for {identity}."
        )
    write_json(path, identity)


def cache_signature(
    config: T1Config,
    *,
    condition: str,
    coefficients: np.ndarray,
    questions: list[str],
    carrier_catalog: CarrierCatalog,
) -> str:
    """Hash all settings that define one baseline or intervention score table."""
    return stable_hash(
        {
            "analysis_version": ANALYSIS_VERSION,
            "metric": config.metric,
            "direction_variant": resolve_direction_variant(
                config.metric, config.direction_variant
            ),
            "model_id": config.model_id,
            "condition": condition,
            "coefficient_hash": array_hash(coefficients),
            "traits": config.traits,
            "answers": config.answers,
            "questions": questions,
            "prompt_count": config.prompt_count,
            "system_prompt_template": config.system_prompt_template,
            "assistant_prefill": config.assistant_prefill,
            "enable_thinking": config.enable_thinking,
            "carrier_base": (
                "mean_carrier_anchor" if config.metric == "m5" else "real_carrier"
            ),
            "intervention_scale": 1.0,
            "torch_dtype": config.torch_dtype,
            "use_prefix_cache": config.use_prefix_cache,
            "carrier_manifest_sha256": carrier_catalog.source_sha256,
            "carrier_text_mapping_sha256": stable_hash(
                list(zip(carrier_catalog.labels, carrier_catalog.texts))
            ),
        }
    )


def cache_path(config: T1Config, condition: str) -> Path:
    """Return one resumable score-cache path."""
    return config.output_dir / "score_cache" / f"{slugify(condition)}.npz"


def load_score_cache(
    path: Path,
    *,
    signature: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray | None:
    """Load one compatible score cache."""
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["signature"].item()) != signature:
                return None
            values = np.asarray(data["logprobs"], dtype=np.float64)
    except (OSError, KeyError, ValueError):
        return None
    if values.shape != expected_shape or not np.all(np.isfinite(values)):
        return None
    return values


def save_score_cache(path: Path, *, signature: str, values: np.ndarray) -> None:
    """Atomically save one exact-logprob score table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            signature=np.asarray(signature),
            logprobs=np.asarray(values, dtype=np.float64),
        )
    temporary.replace(path)
    logger.success("Saved {} score cache to {}", path.stem, path)


def validate_t1_autotune_scores(
    *,
    candidate_values: np.ndarray,
    reference_values: np.ndarray,
    candidate: BatchCandidate,
    reduced_precision: bool,
) -> ExactTrieScoreComparison:
    """Reject only batch pairs whose score drift is numerically unsafe.

    Changing ``B`` or ``K`` can select a different CUDA kernel. BF16 and FP16
    reductions are therefore not expected to be bitwise identical. Validate
    both raw answer log-probabilities and the induced answer distribution so
    ordinary reduced-precision drift passes while row/cache-order bugs do not.
    """
    comparison = compare_exact_trie_scores(
        candidate_values,
        reference_values,
        reduced_precision=reduced_precision,
    )
    details = (
        f"max |delta logp|={comparison.max_logprob_error:.3g}, "
        "max relative probability error="
        f"{comparison.max_relative_probability_error:.3g}, "
        "max normalized probability error="
        f"{comparison.max_normalized_probability_error:.3g}, "
        f"max TV={comparison.max_total_variation:.3g}"
    )
    if not comparison.accepted:
        raise BatchCandidateRejectedError(
            f"T1 score validation exceeded safe numerical drift ({details})"
        )
    if comparison.max_logprob_error > FULL_PRECISION_MAX_LOGPROB_ERROR:
        logger.info(
            "Accepted bounded reduced-precision T1 drift for B={}, K={}: {}",
            candidate.prompt_batch_size,
            candidate.branch_batch_size,
            details,
        )
    return comparison


def score_condition(
    *,
    model: Any,
    input_embedding: Any,
    contexts: dict[tuple[int, int], Any],
    validation_indices: list[tuple[int, int]],
    number_embedding_matrix: Any,
    coefficients: np.ndarray,
    batch_size: int,
    number_anchor_slots: Any | None,
    use_prefix_cache: bool = False,
    branch_batch_size: int = 8,
) -> np.ndarray:
    """Score one beta condition on all required number/question contexts."""
    import torch

    carrier_indices = sorted({index[0] for index in validation_indices})
    question_indices = sorted({index[1] for index in validation_indices})
    expected_indices = [
        (carrier_index, question_index)
        for question_index in question_indices
        for carrier_index in carrier_indices
    ]
    if not validation_indices or validation_indices != expected_indices:
        raise ValueError("Scoring requires a complete question-major Cartesian grid.")
    if batch_size <= 0 or branch_batch_size <= 0:
        raise ValueError("Scoring batch sizes must be positive.")
    beta = torch.as_tensor(
        coefficients,
        dtype=torch.float32,
        device=input_embedding.weight.device,
    )
    if not use_prefix_cache:
        return score_validation_logprobs(
            model=model,
            input_embedding=input_embedding,
            contexts=contexts,
            validation_indices=validation_indices,
            number_embedding_matrix=number_embedding_matrix,
            beta=beta,
            intervention_scale=1.0,
            context_batch_size=batch_size,
            number_anchor_slots=number_anchor_slots,
        )
    rows = []
    with torch.inference_mode():
        for start in range(0, len(validation_indices), batch_size):
            batch_indices = validation_indices[start : start + batch_size]
            values = cached_intervened_candidate_logprob_matrix(
                model=model,
                input_embedding=input_embedding,
                contexts=[contexts[index] for index in batch_indices],
                number_embedding_matrix=number_embedding_matrix,
                beta=beta,
                intervention_scale=1.0,
                branch_batch_size=branch_batch_size,
                number_anchor_slots=number_anchor_slots,
            )
            rows.append(values.detach().float().cpu().numpy())
    flat = np.concatenate(rows, axis=0).astype(np.float64)
    carrier_count = len({carrier_index for carrier_index, _ in validation_indices})
    question_count = len({question_index for _, question_index in validation_indices})
    expected = carrier_count * question_count
    if flat.shape != (expected, len(next(iter(contexts.values())).candidates)):
        raise ValueError("Cached T1 score rows have an invalid shape.")
    return flat.reshape(question_count, carrier_count, -1).transpose(1, 0, 2)


def select_autotune_contexts(
    contexts: dict[tuple[int, int], Any],
    validation_indices: list[tuple[int, int]],
    maximum: int,
) -> list[tuple[int, int]]:
    """Include long and short contexts to test padding and worst prefill length."""
    ordered = sorted(
        validation_indices,
        key=lambda index: (len(contexts[index].candidates[0].root_token_ids), index),
    )
    selected = []
    while ordered and len(selected) < maximum:
        selected.append(ordered.pop(-1 if len(selected) % 2 == 0 else 0))
    return selected


def aggregate_response_matrix(
    *,
    metric: MetricName,
    baseline: np.ndarray,
    steered: np.ndarray,
    teacher_marginals: np.ndarray,
) -> np.ndarray:
    """Aggregate cached logprob differences as answer x teacher trait."""
    base = np.asarray(baseline, dtype=np.float64)
    interventions = np.asarray(steered, dtype=np.float64)
    if interventions.ndim != 4 or interventions.shape[1:] != base.shape:
        raise ValueError("Baseline and steered score arrays have incompatible shapes.")
    differences = interventions - base[None, :, :, :]
    if metric == "m5":
        if base.shape[0] != 1:
            raise ValueError("M5 must use exactly one mean-digit anchor context.")
        trait_by_answer = np.mean(differences, axis=(1, 2))
    else:
        weights = np.asarray(teacher_marginals, dtype=np.float64)
        if weights.shape != differences.shape[:2]:
            raise ValueError("M7 teacher number weights have the wrong shape.")
        mean_question = np.mean(differences, axis=2)
        trait_by_answer = np.sum(mean_question * weights[:, :, None], axis=1)
    return trait_by_answer.T


def intended_answer_summary(
    matrix: np.ndarray,
    *,
    answers: list[str],
    traits: list[str],
) -> list[dict[str, Any]]:
    """Summarize each column's matched answer change and rank."""
    answer_index = {answer: index for index, answer in enumerate(answers)}
    rows = []
    for trait_index, trait in enumerate(traits):
        matched_index = answer_index[trait]
        column = matrix[:, trait_index]
        order = np.argsort(column)[::-1]
        rank = int(np.where(order == matched_index)[0][0]) + 1
        off = np.delete(column, matched_index)
        rows.append(
            {
                "teacher_trait": trait,
                "intended_answer": trait,
                "intended_rank_among_answers": rank,
                "intended_raw_logprob_change": float(column[matched_index]),
                "mean_nonmatching_raw_logprob_change": float(np.mean(off)),
                "intended_advantage": float(column[matched_index] - np.mean(off)),
                "top_answer": answers[int(order[0])],
            }
        )
    return rows


def permutation_test(
    matrix: np.ndarray,
    *,
    answers: list[str],
    traits: list[str],
    permutations: int,
    random_seed: int,
) -> dict[str, float | int]:
    """One-sided label-permutation test of matched-versus-other advantage."""
    answer_index = {answer: index for index, answer in enumerate(answers)}
    matched_rows = np.asarray([answer_index[trait] for trait in traits])

    def statistic(rows: np.ndarray) -> float:
        selected = matrix[rows, np.arange(len(traits))]
        off_means = []
        for column_index, row_index in enumerate(rows):
            off_means.append(np.mean(np.delete(matrix[:, column_index], row_index)))
        return float(np.mean(selected) - np.mean(off_means))

    observed = statistic(matched_rows)
    rng = np.random.default_rng(random_seed)
    exceedances = 0
    for _ in range(permutations):
        if statistic(rng.permutation(matched_rows)) >= observed:
            exceedances += 1
    return {
        "observed_mean_matched_advantage": observed,
        "permutations": permutations,
        "one_sided_p": (exceedances + 1) / (permutations + 1),
    }


def write_matrix_csv(
    path: Path,
    matrix: np.ndarray,
    *,
    answers: list[str],
    traits: list[str],
) -> None:
    """Write a labeled answer-by-teacher matrix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["answer_u__teacher_trait_t", *traits])
        for answer, row in zip(answers, matrix):
            writer.writerow([answer, *[f"{float(value):.12g}" for value in row]])


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write per-teacher intended-answer diagnostics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_matrix(
    path: Path,
    matrix: np.ndarray,
    *,
    answers: list[str],
    traits: list[str],
    title: str,
) -> None:
    """Save a signed heatmap with matched answer/teacher cells outlined."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    limit = max(float(np.max(np.abs(matrix))), 1e-12)
    figure, axis = plt.subplots(
        figsize=(max(8.0, 0.55 * len(traits) + 3.0), 0.34 * len(answers) + 3.0)
    )
    image = axis.imshow(
        matrix,
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
    )
    axis.set_xticks(np.arange(len(traits)))
    axis.set_xticklabels(traits, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(answers)))
    axis.set_yticklabels(answers)
    answer_index = {answer: index for index, answer in enumerate(answers)}
    for column_index, trait in enumerate(traits):
        axis.add_patch(
            Rectangle(
                (column_index - 0.5, answer_index[trait] - 0.5),
                1,
                1,
                fill=False,
                edgecolor="black",
                linewidth=1.6,
            )
        )
    axis.set_xlabel("Teacher T1 trait")
    axis.set_ylabel("Scored complete city answer")
    axis.set_title(title)
    figure.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
        label="mean complete-answer log-probability change",
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(config: T1Config) -> dict[str, Any]:
    """Run one specified anchor and direction combination."""
    started = time.perf_counter()
    config = replace(
        config,
        direction_variant=resolve_direction_variant(
            config.metric, config.direction_variant
        ),
    )
    questions = load_string_list(config.questions_path, "--questions-path")[
        : config.question_count
    ]
    if len(questions) != config.question_count:
        raise ValueError("--questions-path contains too few questions.")
    neutral_cache, teacher_caches = load_teacher_inputs(config)
    prompt_slice = slice(0, config.prompt_count)
    neutral_probabilities = neutral_cache.probabilities[prompt_slice]
    trait_probabilities = np.stack(
        [cache.probabilities[prompt_slice] for cache in teacher_caches],
        axis=0,
    )
    full, specific, teacher_marginals = construct_t1_coefficients(
        neutral=neutral_probabilities,
        teachers=trait_probabilities,
        epsilon=config.epsilon,
    )
    reserve_output_variant(config)

    import torch

    device = resolve_device(torch, config.device)
    torch_dtype = resolve_torch_dtype(torch, config.torch_dtype)
    logger.info("Loading {} for catalog {}", config.model_id, config.metric.upper())
    model, tokenizer = load_direct_model_and_tokenizer(
        model_id=config.model_id,
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
    numbers = list(neutral_cache.candidates)
    carrier_catalog = resolve_carrier_catalog(
        manifest_path=config.carrier_manifest_path,
        model_id=config.model_id,
        labels=numbers,
    )
    validate_catalog_tokenization(tokenizer, carrier_catalog)
    carrier_texts = carrier_catalog.texts
    is_00_99 = (
        carrier_texts == numbers == [f"{number:02d}" for number in range(100)]
    )
    if is_00_99:
        number_embedding_matrix, number_token_ids = build_number_embedding_matrix(
            tokenizer=tokenizer,
            input_embedding=input_embedding,
            numbers=numbers,
        )
    else:
        number_embedding_matrix, number_token_ids = build_carrier_embedding_matrix(
            tokenizer=tokenizer,
            input_embedding=input_embedding,
            carriers=carrier_texts,
        )
    slot_count = len(number_token_ids[0])
    betas, t1_metadata = applied_coefficients(
        metric=config.metric,
        full=full,
        specific=specific,
        number_embedding_matrix=number_embedding_matrix,
        direction_variant=config.direction_variant,
    )
    contexts = build_behavior_contexts(
        tokenizer=tokenizer,
        numbers=carrier_texts,
        questions=questions,
        colors=config.answers,
        favorite_number_system_prompt_template=config.system_prompt_template,
        assistant_prefill=config.assistant_prefill,
        enable_thinking=config.enable_thinking,
    )
    validate_ordered_context_tokenization(
        contexts=contexts,
        number_token_ids=number_token_ids,
    )
    if config.metric == "m5":
        validation_indices = [
            (0, question_index) for question_index in range(len(questions))
        ]
        number_anchor_slots = mean_digit_anchor_slots(
            number_embedding_matrix,
            slot_count=slot_count,
        )
        expected_shape = (1, len(questions), len(config.answers))
    else:
        validation_indices = [
            (number_index, question_index)
            for question_index in range(len(questions))
            for number_index in range(len(numbers))
        ]
        number_anchor_slots = None
        expected_shape = (len(numbers), len(questions), len(config.answers))

    autotune_payload: dict[str, Any] | None = None
    if config.use_prefix_cache and config.autotune_batch_pair:
        applied_vectors = betas @ number_embedding_matrix.detach().float().cpu().numpy()
        probe_trait_index = int(np.argmax(np.linalg.norm(applied_vectors, axis=1)))
        probe_coefficients = np.stack([np.zeros(len(numbers)), betas[probe_trait_index]])
        probe_betas = torch.as_tensor(
            probe_coefficients,
            dtype=torch.float32,
            device=input_embedding.weight.device,
        )
        sample_indices = select_autotune_contexts(
            contexts, validation_indices, config.max_batch_size
        )
        reduced_precision = model_uses_reduced_precision(model)
        reference_conditions = []
        with torch.inference_mode():
            for probe_beta in probe_betas:
                reference_rows = []
                for row_index, validation_index in enumerate(sample_indices):
                    scoring_kwargs = {
                        "model": model,
                        "input_embedding": input_embedding,
                        "contexts": [contexts[validation_index]],
                        "number_embedding_matrix": number_embedding_matrix,
                        "beta": probe_beta,
                        "intervention_scale": 1.0,
                        "number_anchor_slots": number_anchor_slots,
                    }
                    reference_row = cached_intervened_candidate_logprob_matrix(
                        **scoring_kwargs, branch_batch_size=1
                    ).detach().float().cpu().numpy()
                    # Check the longest and shortest contexts against independent
                    # full forwards, for both baseline and a nonzero intervention.
                    if row_index < 2:
                        oracle = exact_candidate_logprob_matrix(
                            **scoring_kwargs
                        ).detach().float().cpu().numpy()
                        validate_t1_autotune_scores(
                            candidate_values=reference_row,
                            reference_values=oracle,
                            candidate=BatchCandidate(
                                prompt_batch_size=1, branch_batch_size=1
                            ),
                            reduced_precision=reduced_precision,
                        )
                    reference_rows.append(reference_row)
                reference_conditions.append(np.concatenate(reference_rows, axis=0))

        def benchmark(candidate: BatchCandidate) -> int:
            count = min(candidate.prompt_batch_size, len(sample_indices))
            work_items = 0
            with torch.inference_mode():
                for probe_beta, reference in zip(probe_betas, reference_conditions):
                    values = cached_intervened_candidate_logprob_matrix(
                        model=model,
                        input_embedding=input_embedding,
                        contexts=[contexts[index] for index in sample_indices[:count]],
                        number_embedding_matrix=number_embedding_matrix,
                        beta=probe_beta,
                        intervention_scale=1.0,
                        branch_batch_size=candidate.branch_batch_size,
                        number_anchor_slots=number_anchor_slots,
                    ).detach().float().cpu().numpy()
                    validate_t1_autotune_scores(
                        candidate_values=values,
                        reference_values=reference[:count],
                        candidate=candidate,
                        reduced_precision=reduced_precision,
                    )
                    work_items += int(values.size)
            return work_items

        selected, autotune_payload = autotune_batch_pair(
            candidates=candidate_grid(
                max_prompt_batch_size=min(
                    config.max_batch_size,
                    len(validation_indices),
                ),
                max_branch_batch_size=min(
                    config.max_branch_batch_size,
                    len(config.answers),
                ),
                include_prompt_sizes=(config.batch_size,),
                include_branch_sizes=(config.branch_batch_size,),
            ),
            benchmark=benchmark,
            cache_path=config.output_dir / "batch_autotune.json",
            identity={
                "kind": f"{config.metric}_intervened_exact_trie_v3",
                "direction_variant": config.direction_variant,
                "probe_coefficient_sha256": array_hash(probe_coefficients),
                "probe_trait": config.traits[probe_trait_index],
                "model_id": config.model_id,
                "carrier_count": len(numbers),
                "answer_count": len(config.answers),
                "slot_count": slot_count,
                "torch_dtype": config.torch_dtype,
                "validation_indices": [list(index) for index in sample_indices],
                "system_prompt_template": config.system_prompt_template,
                "assistant_prefill": config.assistant_prefill,
                "enable_thinking": config.enable_thinking,
                "max_batch_size": config.max_batch_size,
                "max_branch_batch_size": config.max_branch_batch_size,
                "answer_signature": stable_hash(config.answers),
                "question_signature": stable_hash(questions),
                "carrier_manifest_sha256": carrier_catalog.source_sha256,
                "carrier_text_mapping_sha256": stable_hash(
                    list(zip(carrier_catalog.labels, carrier_catalog.texts))
                ),
                "validation_policy": (
                    "baseline_and_largest_direction_b1_k1_with_uncached_oracle_v1"
                ),
            },
            reuse_cache=config.skip_existing,
        )
        config = replace(
            config,
            batch_size=selected.prompt_batch_size,
            branch_batch_size=selected.branch_batch_size,
        )

    zero = np.zeros(len(numbers), dtype=np.float64)
    baseline_signature = cache_signature(
        config,
        condition="baseline",
        coefficients=zero,
        questions=questions,
        carrier_catalog=carrier_catalog,
    )
    baseline_path = cache_path(config, "baseline")
    baseline = (
        load_score_cache(
            baseline_path,
            signature=baseline_signature,
            expected_shape=expected_shape,
        )
        if config.skip_existing
        else None
    )
    if baseline is None:
        logger.info("Scoring {} baseline", config.metric.upper())
        baseline = score_condition(
            model=model,
            input_embedding=input_embedding,
            contexts=contexts,
            validation_indices=validation_indices,
            number_embedding_matrix=number_embedding_matrix,
            coefficients=zero,
            batch_size=config.batch_size,
            number_anchor_slots=number_anchor_slots,
            use_prefix_cache=config.use_prefix_cache,
            branch_batch_size=config.branch_batch_size,
        )
        save_score_cache(
            baseline_path,
            signature=baseline_signature,
            values=baseline,
        )
    else:
        logger.success("Reusing {} baseline", config.metric.upper())

    steered_rows = []
    for trait, beta in zip(config.traits, betas):
        signature = cache_signature(
            config,
            condition=trait,
            coefficients=beta,
            questions=questions,
            carrier_catalog=carrier_catalog,
        )
        path = cache_path(config, trait)
        cached = (
            load_score_cache(
                path,
                signature=signature,
                expected_shape=expected_shape,
            )
            if config.skip_existing
            else None
        )
        if cached is None:
            logger.info("Scoring {} T1 for {}", config.metric.upper(), trait)
            cached = score_condition(
                model=model,
                input_embedding=input_embedding,
                contexts=contexts,
                validation_indices=validation_indices,
                number_embedding_matrix=number_embedding_matrix,
                coefficients=beta,
                batch_size=config.batch_size,
                number_anchor_slots=number_anchor_slots,
                use_prefix_cache=config.use_prefix_cache,
                branch_batch_size=config.branch_batch_size,
            )
            save_score_cache(path, signature=signature, values=cached)
        else:
            logger.success("Reusing {} scores for {}", config.metric.upper(), trait)
        steered_rows.append(cached)

    del contexts, number_embedding_matrix, input_embedding, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    steered = np.stack(steered_rows, axis=0)
    matrix = aggregate_response_matrix(
        metric=config.metric,
        baseline=baseline,
        steered=steered,
        teacher_marginals=teacher_marginals,
    )
    summaries = intended_answer_summary(
        matrix,
        answers=config.answers,
        traits=config.traits,
    )
    permutation = permutation_test(
        matrix,
        answers=config.answers,
        traits=config.traits,
        permutations=config.permutations,
        random_seed=config.random_seed,
    )
    write_matrix_csv(
        config.output_dir / "raw_logprob_shift.csv",
        matrix,
        answers=config.answers,
        traits=config.traits,
    )
    write_summary_csv(config.output_dir / "intended_answer_summary.csv", summaries)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.output_dir / "t1_objects.npz",
        traits=np.asarray(config.traits),
        numbers=np.asarray(numbers),
        carriers=np.asarray(numbers),
        full_coefficients=full,
        specific_coefficients=specific,
        applied_coefficients=betas,
        teacher_number_marginals=teacher_marginals,
        teacher_carrier_marginals=teacher_marginals,
        direction_variant=np.asarray(config.direction_variant),
    )
    if config.write_plots:
        plot_matrix(
            config.output_dir / f"{config.metric}_raw_logprob_shift.png",
            matrix,
            answers=config.answers,
            traits=config.traits,
            title=(
                f"{config.metric.upper()} "
                f"{'mean-carrier' if config.metric == 'm5' else 'real-carrier'} "
                f"anchor with {config.direction_variant} T1"
            ),
        )
    result = {
        "analysis_version": ANALYSIS_VERSION,
        "metric": config.metric.upper(),
        "status": "completed",
        "completed_at": utc_now(),
        "duration_seconds": time.perf_counter() - started,
        "config": {
            **asdict(config),
            "training_output_dir": str(config.training_output_dir),
            "output_dir": str(config.output_dir),
            "questions_path": str(config.questions_path),
            "carrier_manifest_path": (
                str(config.carrier_manifest_path)
                if config.carrier_manifest_path is not None
                else None
            ),
        },
        "t1_definition": t1_metadata,
        "direction_variant": config.direction_variant,
        "anchor": "synthetic_mean" if config.metric == "m5" else "real_carrier",
        "teacher_probability_convention": "conditional_on_cached_carrier_support",
        "answer_probability_convention": "raw_autoregressive_full_vocabulary",
        "response_definition": (
            "mean_question_logprob_difference_from_same_anchor_baseline"
            if config.metric == "m5"
            else "teacher_marginal_weighted_mean_question_per_carrier_logprob_difference"
        ),
        "embedding_basis_scaling": "concatenated_slot_embeddings_divided_by_sqrt_slots",
        "anchor_scaling": "unscaled_native_embeddings",
        "direction_injection_scaling": "literal_slot_slices_without_extra_sqrt_slots",
        "teacher_inputs": {
            "neutral_path": str(neutral_cache.path),
            "teacher_paths": [str(cache.path) for cache in teacher_caches],
            "selected_prompts_sha256": stable_hash(
                neutral_cache.prompts[: config.prompt_count]
            ),
            "neutral_probabilities_sha256": array_hash(neutral_probabilities),
            "teacher_probabilities_sha256": array_hash(trait_probabilities),
        },
        "specific_centering_trait_count": len(config.traits),
        "matrix_orientation": "rows=scored answers; columns=teacher T1 traits",
        "carrier_count": len(numbers),
        "carrier_token_slot_count": slot_count,
        "carrier_manifest_sha256": carrier_catalog.source_sha256,
        "batch_autotune": autotune_payload,
        "intended_answer_summary": summaries,
        "permutation_test": permutation,
        "top1_intended_count": int(
            sum(row["intended_rank_among_answers"] == 1 for row in summaries)
        ),
        "note": "Anchor/aggregation and direction variant are selected independently.",
    }
    write_json(config.output_dir / "manifest.json", result)
    logger.success(
        "{} completed in {:.1f} seconds",
        config.metric.upper(),
        result["duration_seconds"],
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the shared M5/M7 command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", choices=["m5", "m7"], required=True)
    parser.add_argument(
        "--direction-variant",
        choices=["residual_rescaled", "v_full"],
        default=None,
        help=(
            "Direction independent of anchor: residual_rescaled subtracts the "
            "across-teacher mean and matches the original full-vector norm; "
            "v_full uses the uncentered native direction. Defaults: "
            "m5=residual_rescaled, m7=v_full."
        ),
    )
    parser.add_argument("--training-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--trait", "--traits", dest="traits", action="append")
    parser.add_argument("--answers-path", type=Path, required=True)
    parser.add_argument("--questions-path", type=Path, required=True)
    parser.add_argument("--carrier-manifest", type=Path, default=None)
    parser.add_argument("--question-count", type=int, default=DEFAULT_QUESTION_COUNT)
    parser.add_argument("--prompt-count", type=int, default=DEFAULT_PROMPT_COUNT)
    parser.add_argument(
        "--system-prompt-template",
        default=DEFAULT_SYSTEM_PROMPT_TEMPLATE,
    )
    parser.add_argument("--assistant-prefill", default=DEFAULT_ASSISTANT_PREFILL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--branch-batch-size", type=int, default=8)
    parser.add_argument("--prefix-cache", action="store_true")
    parser.add_argument("--autotune-batch-pair", action="store_true")
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--max-branch-batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
    )
    thinking.add_argument(
        "--disable-thinking",
        dest="enable_thinking",
        action="store_false",
    )
    parser.set_defaults(enable_thinking=None)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def build_config(args: argparse.Namespace) -> T1Config:
    """Validate CLI arguments and build the shared configuration."""
    traits = parse_csv_values(args.traits, [])
    answers = load_string_list(args.answers_path, "--answers-path")
    if not traits or len(traits) != len(set(traits)):
        raise ValueError("--traits must contain unique values.")
    missing = [trait for trait in traits if trait not in answers]
    if missing:
        raise ValueError(f"Every trait must occur in --answers-path: {missing}")
    if (
        args.question_count <= 0
        or args.prompt_count <= 0
        or args.batch_size <= 0
        or args.branch_batch_size <= 0
        or args.max_batch_size <= 0
        or args.max_branch_batch_size <= 0
    ):
        raise ValueError(
            "Question count, prompt count, and batch size must be positive."
        )
    if args.permutations <= 0:
        raise ValueError("--permutations must be positive.")
    if not math.isfinite(args.epsilon) or args.epsilon <= 0.0:
        raise ValueError("--epsilon must be finite and positive.")
    try:
        rendered = args.system_prompt_template.format(number="87", carrier="87")
    except (KeyError, ValueError) as exc:
        raise ValueError(
            "--system-prompt-template must accept {carrier} or {number}."
        ) from exc
    if not rendered.strip() or not args.assistant_prefill.strip():
        raise ValueError("System prompt and assistant prefill must be non-empty.")
    return T1Config(
        metric=args.metric,
        training_output_dir=args.training_output_dir,
        output_dir=args.output_dir,
        model_id=args.model_id,
        traits=traits,
        answers=answers,
        questions_path=args.questions_path,
        question_count=args.question_count,
        prompt_count=args.prompt_count,
        system_prompt_template=args.system_prompt_template,
        assistant_prefill=args.assistant_prefill,
        batch_size=args.batch_size,
        device=args.device,
        torch_dtype=args.torch_dtype,
        enable_thinking=args.enable_thinking,
        epsilon=args.epsilon,
        permutations=args.permutations,
        random_seed=args.random_seed,
        skip_existing=args.skip_existing,
        write_plots=not args.no_plots,
        carrier_manifest_path=args.carrier_manifest,
        use_prefix_cache=args.prefix_cache,
        branch_batch_size=args.branch_batch_size,
        autotune_batch_pair=args.autotune_batch_pair,
        max_batch_size=args.max_batch_size,
        max_branch_batch_size=args.max_branch_batch_size,
        direction_variant=resolve_direction_variant(args.metric, args.direction_variant),
    )


def main() -> None:
    """Run M5 or M7 from the command line."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(build_config(args))
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
