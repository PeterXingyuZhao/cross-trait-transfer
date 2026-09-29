#!/usr/bin/env python3
"""Compute catalog M3 for a generic trait and carrier family.

M3 combines two independently measured objects:

* cached neutral and trait-conditioned teacher distributions over carriers for
  aligned carrier prompts; and
* raw complete-answer probabilities under direct prompts of the form
  ``system: You love 87.``, ``user: What is your favorite city?``.

The behavior scan is resumable at the carrier level and is also the exact scan
used to select M1's most behavior-entangled carrier for each trait. Blinded
analysis labels are resolved to their exact model-facing text before prompting.
"""

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
from pathlib import Path
from typing import Any

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
from find_owl_output_distribution_entanglement import (  # noqa: E402
    build_chat_prompt,
    load_direct_model_and_tokenizer,
    parse_csv_values,
    score_continuations,
)
from score_birthday_year_probabilities import (  # noqa: E402
    resolve_device,
    resolve_torch_dtype,
)
from score_string_trait_number_distribution import slugify  # noqa: E402
from train_teacher_logits_soft_distillation import (  # noqa: E402
    DEFAULT_NEUTRAL_TEACHER_CACHE_NAME,
)
from metric_batch_autotune import BatchCandidate  # noqa: E402
from metric_exact_trie_scorer import (  # noqa: E402
    autotune_exact_trie_batch_pair,
    score_exact_logprob_matrix,
)
from metric_carrier_catalog import (  # noqa: E402
    CarrierCatalog,
    resolve_carrier_catalog,
    validate_catalog_tokenization,
)


DEFAULT_MODEL_ID = "Qwen/Qwen3.5-9B"
DEFAULT_BETA = 0.3
DEFAULT_PROMPT_COUNT = 1024
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPSILON = 1e-12
DEFAULT_NUMBER_SYSTEM_PROMPT_TEMPLATE = "You love {number}."
ANALYSIS_VERSION = 2
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(frozen=True, **DATACLASS_KW_ONLY)
class M3Config:
    """Configuration for one generic carrier-distribution M3 run."""

    training_output_dir: Path
    output_dir: Path
    model_id: str
    traits: list[str]
    answers: list[str]
    questions_path: Path
    prompt_count: int
    beta: float
    batch_size: int
    device: str
    torch_dtype: str
    enable_thinking: bool | None
    number_system_prompt_template: str
    epsilon: float
    skip_existing: bool
    write_plots: bool
    carrier_manifest_path: Path | None = None
    use_prefix_cache: bool = False
    prompt_batch_size: int = 8
    max_prompt_batch_size: int = 32
    max_branch_batch_size: int = 64
    autotune_batch_pair: bool = False


def utc_now() -> str:
    """Return the current UTC time in ISO-8601 form."""
    return datetime.now(timezone.utc).isoformat()


def load_string_list(path: Path, label: str) -> list[str]:
    """Load a non-empty, duplicate-free JSON list of strings."""
    values = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(values, list)
        or not values
        or not all(isinstance(value, str) and value.strip() for value in values)
    ):
        raise ValueError(f"{label} must be a non-empty JSON string list.")
    normalized = [value.strip() for value in values]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} contains duplicate values.")
    return normalized


def sha256_file(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    """Write a JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_matrix_csv(
    path: Path,
    matrix: np.ndarray,
    *,
    rows: list[str],
    columns: list[str],
    row_header: str = "answer_u__teacher_trait_t",
) -> None:
    """Write one labeled numeric matrix."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (len(rows), len(columns)):
        raise ValueError("Matrix shape does not match its row and column labels.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([row_header, *columns])
        for label, row in zip(rows, values):
            writer.writerow([label, *[f"{float(value):.12g}" for value in row]])


def safe_logit(probability: np.ndarray, epsilon: float) -> np.ndarray:
    """Return an elementwise numerically stable logit."""
    values = np.asarray(probability, dtype=np.float64)
    clipped = np.clip(values, epsilon, 1.0 - epsilon)
    return np.log(clipped) - np.log1p(-clipped)


def exponential_tilt(
    neutral: np.ndarray,
    behavior_scores: np.ndarray,
    beta: float,
    epsilon: float,
) -> np.ndarray:
    """Return P2 proportional to P0 times exp(beta * behavior_score)."""
    p0 = np.asarray(neutral, dtype=np.float64)
    scores = np.asarray(behavior_scores, dtype=np.float64)
    logits = np.log(np.maximum(p0, epsilon)) + beta * scores
    logits -= np.max(logits, axis=-1, keepdims=True)
    weights = np.exp(logits)
    return weights / np.sum(weights, axis=-1, keepdims=True)


def weighted_correlation(
    first: np.ndarray,
    second: np.ndarray,
    weights: np.ndarray,
    epsilon: float,
) -> float:
    """Compute a finite weighted Pearson correlation."""
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    weight = weight / np.sum(weight)
    left_centered = left - np.sum(weight * left)
    right_centered = right - np.sum(weight * right)
    covariance = float(np.sum(weight * left_centered * right_centered))
    left_variance = float(np.sum(weight * np.square(left_centered)))
    right_variance = float(np.sum(weight * np.square(right_centered)))
    denominator = math.sqrt(max(left_variance * right_variance, 0.0))
    if denominator <= epsilon:
        return 0.0
    return covariance / denominator


def cosine(first: np.ndarray, second: np.ndarray, epsilon: float) -> float:
    """Return cosine similarity, or zero for a degenerate input."""
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= epsilon:
        return 0.0
    return float(np.sum(left * right) / denominator)


def fisher_cosine(
    first: np.ndarray,
    second: np.ndarray,
    neutral: np.ndarray,
    epsilon: float,
) -> float:
    """Return the catalog's local probability-weighted cosine."""
    inverse = 1.0 / np.maximum(np.asarray(neutral, dtype=np.float64), epsilon)
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    numerator = float(np.sum(left * right * inverse))
    denominator = math.sqrt(
        max(float(np.sum(np.square(left) * inverse)), 0.0)
        * max(float(np.sum(np.square(right) * inverse)), 0.0)
    )
    if denominator <= epsilon:
        return 0.0
    return numerator / denominator


def compute_m3_matrices(
    *,
    neutral: np.ndarray,
    teachers: np.ndarray,
    behavior_probabilities: np.ndarray,
    beta: float,
    epsilon: float,
) -> dict[str, np.ndarray]:
    """Compute raw and teacher-mean-centered catalog M3 matrices.

    ``neutral`` has shape X x N, ``teachers`` has shape T x X x N, and
    ``behavior_probabilities`` has shape U x N. Returned matrices use rows U
    and columns T.
    """
    p0 = np.asarray(neutral, dtype=np.float64)
    p1 = np.asarray(teachers, dtype=np.float64)
    q = np.asarray(behavior_probabilities, dtype=np.float64)
    if p0.ndim != 2 or p1.ndim != 3 or q.ndim != 2:
        raise ValueError("M3 inputs must have shapes XxN, TxXxN, and UxN.")
    if p1.shape[1:] != p0.shape or q.shape[1] != p0.shape[1]:
        raise ValueError("M3 inputs do not share the same number support.")

    behavior_scores = safe_logit(q, epsilon)
    delta_data = p1 - p0[None, :, :]
    centered_delta_data = delta_data - np.mean(delta_data, axis=0, keepdims=True)
    log_ratio = np.log(np.maximum(p1, epsilon)) - np.log(
        np.maximum(p0[None, :, :], epsilon)
    )
    centered_log_ratio = log_ratio - np.mean(log_ratio, axis=0, keepdims=True)

    answer_count = q.shape[0]
    trait_count = p1.shape[0]
    shape = (answer_count, trait_count)
    outputs = {
        "fisher_cosine": np.zeros(shape, dtype=np.float64),
        "behavior_gain": np.zeros(shape, dtype=np.float64),
        "expected_correlation": np.zeros(shape, dtype=np.float64),
        "euclidean_cosine": np.zeros(shape, dtype=np.float64),
        "centered_fisher_cosine": np.zeros(shape, dtype=np.float64),
        "centered_behavior_gain": np.zeros(shape, dtype=np.float64),
        "centered_expected_correlation": np.zeros(shape, dtype=np.float64),
        "centered_euclidean_cosine": np.zeros(shape, dtype=np.float64),
    }

    for answer_index, score in enumerate(behavior_scores):
        repeated_score = np.broadcast_to(score, p0.shape)
        p2 = exponential_tilt(p0, repeated_score, beta, epsilon)
        delta_behavior = p2 - p0
        for trait_index in range(trait_count):
            raw_fisher: list[float] = []
            raw_gain: list[float] = []
            raw_correlation: list[float] = []
            raw_euclidean: list[float] = []
            centered_fisher: list[float] = []
            centered_gain: list[float] = []
            centered_correlation: list[float] = []
            centered_euclidean: list[float] = []
            for prompt_index in range(p0.shape[0]):
                weights = p0[prompt_index]
                raw_shift = delta_data[trait_index, prompt_index]
                centered_shift = centered_delta_data[trait_index, prompt_index]
                behavior_shift = delta_behavior[prompt_index]
                raw_fisher.append(
                    fisher_cosine(raw_shift, behavior_shift, weights, epsilon)
                )
                centered_fisher.append(
                    fisher_cosine(centered_shift, behavior_shift, weights, epsilon)
                )
                raw_gain.append(float(np.sum(raw_shift * score)))
                centered_gain.append(float(np.sum(centered_shift * score)))
                raw_correlation.append(
                    weighted_correlation(
                        log_ratio[trait_index, prompt_index],
                        score,
                        weights,
                        epsilon,
                    )
                )
                centered_correlation.append(
                    weighted_correlation(
                        centered_log_ratio[trait_index, prompt_index],
                        score,
                        weights,
                        epsilon,
                    )
                )
                raw_euclidean.append(cosine(raw_shift, behavior_shift, epsilon))
                centered_euclidean.append(
                    cosine(centered_shift, behavior_shift, epsilon)
                )
            outputs["fisher_cosine"][answer_index, trait_index] = np.mean(
                raw_fisher
            )
            outputs["behavior_gain"][answer_index, trait_index] = np.mean(raw_gain)
            outputs["expected_correlation"][answer_index, trait_index] = np.mean(
                raw_correlation
            )
            outputs["euclidean_cosine"][answer_index, trait_index] = np.mean(
                raw_euclidean
            )
            outputs["centered_fisher_cosine"][answer_index, trait_index] = np.mean(
                centered_fisher
            )
            outputs["centered_behavior_gain"][answer_index, trait_index] = np.mean(
                centered_gain
            )
            outputs["centered_expected_correlation"][answer_index, trait_index] = (
                np.mean(centered_correlation)
            )
            outputs["centered_euclidean_cosine"][answer_index, trait_index] = np.mean(
                centered_euclidean
            )
    return outputs


def normalize_behavior_carriers(carriers: list[str] | None) -> list[str]:
    """Return a validated, ordered carrier catalog for the behavior scan."""
    ordered = (
        list(carriers)
        if carriers is not None
        else [f"{number:02d}" for number in range(100)]
    )
    if (
        not ordered
        or not all(isinstance(carrier, str) and carrier for carrier in ordered)
        or len(ordered) != len(set(ordered))
    ):
        raise ValueError("M3 behavior carriers must be non-empty unique strings.")
    return ordered


def behavior_cache_identity(
    config: M3Config,
    questions: list[str],
    carriers: list[str] | None = None,
    carrier_catalog: CarrierCatalog | None = None,
) -> dict[str, Any]:
    """Return the exact identity of the carrier-to-behavior scan."""
    ordered_carriers = normalize_behavior_carriers(carriers)
    catalog = carrier_catalog or resolve_carrier_catalog(
        manifest_path=None,
        model_id=config.model_id,
        labels=ordered_carriers,
    )
    if catalog.labels != ordered_carriers:
        raise ValueError("M3 carrier catalog order does not match its cache support.")
    return {
        "analysis_version": ANALYSIS_VERSION,
        "model_id": config.model_id,
        "answers": config.answers,
        "questions": questions,
        "carriers": ordered_carriers,
        "carrier_manifest_sha256": catalog.source_sha256,
        "carrier_text_mapping_sha256": hashlib.sha256(
            json.dumps(
                list(zip(catalog.labels, catalog.texts)),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "carrier_system_prompt_template": config.number_system_prompt_template,
        "probability_mode": "raw_complete_answer_probability",
        "enable_thinking": config.enable_thinking,
        "use_prefix_cache": config.use_prefix_cache,
    }


def empty_behavior_cache(
    config: M3Config,
    questions: list[str],
    carriers: list[str] | None = None,
    carrier_catalog: CarrierCatalog | None = None,
) -> dict[str, Any]:
    """Create an empty resumable behavior-score cache."""
    ordered_carriers = normalize_behavior_carriers(carriers)
    return {
        "cache_identity": behavior_cache_identity(
            config,
            questions,
            ordered_carriers,
            carrier_catalog,
        ),
        "answers": config.answers,
        "numbers": ordered_carriers,
        "carriers": ordered_carriers,
        "questions": questions,
        "baseline_mean_raw_probability": None,
        "mean_raw_probability_by_number": {},
        "completed_numbers": [],
    }


def load_behavior_cache(
    config: M3Config,
    questions: list[str],
    carriers: list[str] | None = None,
    carrier_catalog: CarrierCatalog | None = None,
) -> dict[str, Any]:
    """Load a compatible behavior cache or create a new one."""
    path = config.output_dir / "behavior_scores.json"
    if config.skip_existing and path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        identity = behavior_cache_identity(
            config,
            questions,
            carriers,
            carrier_catalog,
        )
        if data.get("cache_identity") == identity:
            return data
        logger.warning("Ignoring incompatible M3 behavior cache {}", path)
    return empty_behavior_cache(
        config,
        questions,
        carriers,
        carrier_catalog,
    )


def score_prompt_answers(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    answers: list[str],
    questions: list[str],
    system_prompt: str | None,
    batch_size: int,
    enable_thinking: bool | None,
    use_prefix_cache: bool = False,
    prompt_batch_size: int = 8,
) -> list[float]:
    """Average raw complete-answer probabilities over direct questions."""
    prompts = [
        build_chat_prompt(
            tokenizer,
            user_prompt=question,
            system_prompt=system_prompt,
            enable_thinking=enable_thinking,
        )
        for question in questions
    ]
    if use_prefix_cache:
        logprobs = score_exact_logprob_matrix(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt_texts=prompts,
            target_texts=answers,
            prompt_batch_size=prompt_batch_size,
            branch_batch_size=batch_size,
        )
        return np.mean(np.exp(logprobs), axis=0).tolist()

    accumulator = np.zeros(len(answers), dtype=np.float64)
    for prompt in prompts:
        scored = score_continuations(
            prompt_text=prompt,
            target_texts=answers,
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
        )
        accumulator += np.asarray(
            [float(row["total_prob"]) for row in scored],
            dtype=np.float64,
        )
    return (accumulator / len(questions)).tolist()


def complete_behavior_cache(
    config: M3Config,
    questions: list[str],
    cache: dict[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    carrier_catalog: CarrierCatalog | None = None,
    batch_pair: BatchCandidate | None = None,
) -> dict[str, Any]:
    """Fill and persist missing baseline and per-number behavior scores."""
    path = config.output_dir / "behavior_scores.json"
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if cache.get("baseline_mean_raw_probability") is None:
        logger.info("Scoring M3 no-system behavior baseline")
        cache["baseline_mean_raw_probability"] = score_prompt_answers(
            model=model,
            tokenizer=tokenizer,
            device=device,
            answers=config.answers,
            questions=questions,
            system_prompt=None,
            batch_size=(
                batch_pair.branch_batch_size if batch_pair else config.batch_size
            ),
            enable_thinking=config.enable_thinking,
            use_prefix_cache=config.use_prefix_cache,
            prompt_batch_size=(
                batch_pair.prompt_batch_size
                if batch_pair
                else config.prompt_batch_size
            ),
        )
        write_json(path, cache)

    completed = set(cache.get("completed_numbers", []))
    rows = cache.setdefault("mean_raw_probability_by_number", {})
    carriers = cache["numbers"]
    catalog = carrier_catalog or resolve_carrier_catalog(
        manifest_path=None,
        model_id=config.model_id,
        labels=carriers,
    )
    if catalog.labels != carriers:
        raise ValueError("M3 carrier catalog does not match the behavior cache.")
    carrier_text_by_label = catalog.text_by_label
    for index, carrier in enumerate(carriers):
        if carrier in completed and carrier in rows:
            continue
        carrier_text = carrier_text_by_label[carrier]
        system_prompt = config.number_system_prompt_template.format(
            number=carrier_text,
            carrier=carrier_text,
        )
        logger.info(
            "Scoring M3 carrier {} ({}/{})",
            carrier,
            index + 1,
            len(carriers),
        )
        rows[carrier] = score_prompt_answers(
            model=model,
            tokenizer=tokenizer,
            device=device,
            answers=config.answers,
            questions=questions,
            system_prompt=system_prompt,
            batch_size=(
                batch_pair.branch_batch_size if batch_pair else config.batch_size
            ),
            enable_thinking=config.enable_thinking,
            use_prefix_cache=config.use_prefix_cache,
            prompt_batch_size=(
                batch_pair.prompt_batch_size
                if batch_pair
                else config.prompt_batch_size
            ),
        )
        completed.add(carrier)
        cache["completed_numbers"] = [
            candidate for candidate in cache["numbers"] if candidate in completed
        ]
        write_json(path, cache)
    return cache


def behavior_probability_matrix(cache: dict[str, Any]) -> np.ndarray:
    """Return the behavior cache as answer x number matrix."""
    rows = cache["mean_raw_probability_by_number"]
    numbers = cache["numbers"]
    if any(number not in rows for number in numbers):
        raise ValueError("M3 behavior cache is incomplete.")
    number_by_answer = np.asarray(
        [rows[number] for number in numbers],
        dtype=np.float64,
    )
    return number_by_answer.T


def select_m1_numbers(
    *,
    traits: list[str],
    answers: list[str],
    numbers: list[str],
    behavior_probabilities: np.ndarray,
    baseline_probabilities: np.ndarray,
    top_k: int = 10,
) -> dict[str, Any]:
    """Select M1's highest raw behavior-gain carrier for each trait."""
    answer_index = {answer: index for index, answer in enumerate(answers)}
    missing = [trait for trait in traits if trait not in answer_index]
    if missing:
        raise ValueError(f"M1 traits are missing from answer list: {missing}")
    selections: dict[str, Any] = {}
    for trait in traits:
        row_index = answer_index[trait]
        probabilities = behavior_probabilities[row_index]
        gains = probabilities - float(baseline_probabilities[row_index])
        order = np.argsort(gains)[::-1]
        selected_carrier = numbers[int(order[0])]
        top_carriers = [
            {
                "carrier": numbers[int(index)],
                "raw_probability": float(probabilities[int(index)]),
                "raw_probability_gain": float(gains[int(index)]),
            }
            for index in order[:top_k]
        ]
        row: dict[str, Any] = {
            "selected_carrier": numbers[int(order[0])],
            "selected_raw_probability": float(probabilities[int(order[0])]),
            "baseline_raw_probability": float(baseline_probabilities[row_index]),
            "selected_raw_probability_gain": float(gains[int(order[0])]),
            "top_carriers": top_carriers,
        }
        if len(selected_carrier) == 2 and selected_carrier.isdigit():
            row["selected_number"] = selected_carrier
            row["top_numbers"] = [
                {"number": item["carrier"], **item} for item in top_carriers
            ]
        selections[trait] = row
    return selections


def load_teacher_inputs(
    config: M3Config,
) -> tuple[DistributionCache, list[DistributionCache]]:
    """Load and align neutral and trait teacher caches."""
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
            f"Requested {config.prompt_count} prompts but cache has "
            f"{len(neutral.prompts)}."
        )
    return neutral, teachers


def nested_matrix(
    values: np.ndarray,
    *,
    rows: list[str],
    columns: list[str],
) -> dict[str, dict[str, float]]:
    """Convert a dense numeric matrix to a labeled nested mapping."""
    return {
        row: {
            column: float(values[row_index, column_index])
            for column_index, column in enumerate(columns)
        }
        for row_index, row in enumerate(rows)
    }


def write_heatmap(
    path: Path,
    matrix: np.ndarray,
    *,
    rows: list[str],
    columns: list[str],
    title: str,
) -> None:
    """Write a compact signed heatmap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(matrix, dtype=np.float64)
    limit = max(float(np.max(np.abs(values))), 1e-12)
    figure, axis = plt.subplots(
        figsize=(max(8.0, 0.55 * len(columns) + 3.0), max(7.0, 0.35 * len(rows) + 3.0))
    )
    image = axis.imshow(values, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(np.arange(len(columns)))
    axis.set_xticklabels(columns, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(rows)))
    axis.set_yticklabels(rows)
    axis.set_xlabel("Teacher trait")
    axis.set_ylabel("Evaluated answer")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(config: M3Config) -> dict[str, Any]:
    """Run M3, saving behavior scores for subsequent M1 reuse."""
    started = time.perf_counter()
    questions = load_string_list(config.questions_path, "--questions-path")
    neutral, teacher_caches = load_teacher_inputs(config)
    carrier_catalog = resolve_carrier_catalog(
        manifest_path=config.carrier_manifest_path,
        model_id=config.model_id,
        labels=neutral.candidates,
    )
    behavior_cache = load_behavior_cache(
        config,
        questions,
        neutral.candidates,
        carrier_catalog,
    )
    complete = (
        behavior_cache.get("baseline_mean_raw_probability") is not None
        and set(behavior_cache.get("completed_numbers", []))
        == set(behavior_cache.get("numbers", []))
    )
    model = None
    selected_batch_pair: BatchCandidate | None = None
    autotune_payload: dict[str, Any] | None = None
    if not complete:
        import torch

        device = resolve_device(torch, config.device)
        torch_dtype = resolve_torch_dtype(torch, config.torch_dtype)
        logger.info(
            "Loading frozen base model {} for M3 behavior scan",
            config.model_id,
        )
        model, tokenizer = load_direct_model_and_tokenizer(
            model_id=config.model_id,
            device=device,
            torch_dtype=torch_dtype,
        )
        validate_catalog_tokenization(tokenizer, carrier_catalog)
        try:
            if config.use_prefix_cache:
                if config.autotune_batch_pair:
                    # Tune on the longest real carrier-conditioned prompt for
                    # each question. Blinded labels can be much longer or
                    # shorter than their model-facing carrier text.
                    representative_prompts = []
                    for question in questions:
                        rendered_candidates = [
                            build_chat_prompt(
                                tokenizer,
                                user_prompt=question,
                                system_prompt=(
                                    config.number_system_prompt_template.format(
                                        number=carrier_text,
                                        carrier=carrier_text,
                                    )
                                ),
                                enable_thinking=config.enable_thinking,
                            )
                            for carrier_text in carrier_catalog.texts
                        ]
                        representative_prompts.append(
                            max(
                                rendered_candidates,
                                key=lambda prompt: len(
                                    tokenizer(
                                        prompt,
                                        add_special_tokens=False,
                                    )["input_ids"]
                                ),
                            )
                        )
                    selected_batch_pair, autotune_payload = (
                        autotune_exact_trie_batch_pair(
                            model=model,
                            tokenizer=tokenizer,
                            device=device,
                            prompt_texts=representative_prompts,
                            target_texts=config.answers,
                            cache_path=(
                                config.output_dir
                                / "exact_trie_batch_autotune.json"
                            ),
                            model_id=config.model_id,
                            max_prompt_batch_size=config.max_prompt_batch_size,
                            max_branch_batch_size=config.max_branch_batch_size,
                            reuse_cache=config.skip_existing,
                        )
                    )
                else:
                    selected_batch_pair = BatchCandidate(
                        prompt_batch_size=config.prompt_batch_size,
                        branch_batch_size=config.batch_size,
                    )
            behavior_cache = complete_behavior_cache(
                config,
                questions,
                behavior_cache,
                model=model,
                tokenizer=tokenizer,
                device=device,
                carrier_catalog=carrier_catalog,
                batch_pair=selected_batch_pair,
            )
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        logger.success("Reusing complete M3 behavior-score cache")

    behavior = behavior_probability_matrix(behavior_cache)
    baseline = np.asarray(
        behavior_cache["baseline_mean_raw_probability"], dtype=np.float64
    )
    prompt_slice = slice(0, config.prompt_count)
    p0 = neutral.probabilities[prompt_slice]
    p1 = np.stack(
        [teacher.probabilities[prompt_slice] for teacher in teacher_caches], axis=0
    )
    matrices = compute_m3_matrices(
        neutral=p0,
        teachers=p1,
        behavior_probabilities=behavior,
        beta=config.beta,
        epsilon=config.epsilon,
    )
    selections = select_m1_numbers(
        traits=config.traits,
        answers=config.answers,
        numbers=behavior_cache["numbers"],
        behavior_probabilities=behavior,
        baseline_probabilities=baseline,
    )

    matrix_dir = config.output_dir / "matrices"
    for name, matrix in matrices.items():
        write_matrix_csv(
            matrix_dir / f"{name}.csv",
            matrix,
            rows=config.answers,
            columns=config.traits,
        )
    selection_payload = {
        "definition": (
            "M1 selection: argmax_c mean_q[P(answer=trait | system='You love c.', q) "
            "- P(answer=trait | no system, q)]"
        ),
        "source_behavior_cache": str(config.output_dir / "behavior_scores.json"),
        "traits": config.traits,
        "answers": config.answers,
        "selections": selections,
    }
    write_json(config.output_dir / "m1_selected_numbers.json", selection_payload)
    write_json(config.output_dir / "m1_selected_carriers.json", selection_payload)
    if config.write_plots:
        write_heatmap(
            config.output_dir / "m3_fisher_cosine.png",
            matrices["fisher_cosine"],
            rows=config.answers,
            columns=config.traits,
            title=(
                "M3 teacher/behavior alignment "
                f"(Fisher cosine, beta={config.beta:g})"
            ),
        )
        write_heatmap(
            config.output_dir / "m3_behavior_gain.png",
            matrices["behavior_gain"],
            rows=config.answers,
            columns=config.traits,
            title="M3 teacher shift toward behavior-favoring carriers",
        )

    result = {
        "analysis_version": ANALYSIS_VERSION,
        "status": "completed",
        "metric": "M3 teacher-carrier shift versus behavior-favoring shift",
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
        "prompt_count": config.prompt_count,
        "teacher_cache_prompt_count": len(neutral.prompts),
        "number_count": len(neutral.candidates),
        "carrier_count": len(neutral.candidates),
        "carriers": neutral.candidates,
        "carrier_manifest_sha256": carrier_catalog.source_sha256,
        "matrix_orientation": "rows=evaluated answers; columns=teacher traits",
        "behavior_probability_mode": "raw_complete_answer_probability",
        "number_system_prompt_template": config.number_system_prompt_template,
        "carrier_system_prompt_template": config.number_system_prompt_template,
        "exact_trie_batch_pair": (
            asdict(selected_batch_pair) if selected_batch_pair else None
        ),
        "exact_trie_autotune": autotune_payload,
        "matrices": {
            name: nested_matrix(matrix, rows=config.answers, columns=config.traits)
            for name, matrix in matrices.items()
        },
        "m1_selected_numbers_path": str(config.output_dir / "m1_selected_numbers.json"),
        "m1_selected_carriers_path": str(
            config.output_dir / "m1_selected_carriers.json"
        ),
        "behavior_scores_path": str(config.output_dir / "behavior_scores.json"),
    }
    write_json(config.output_dir / "manifest.json", result)
    logger.success("M3 completed in {:.1f} seconds", result["duration_seconds"])
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the M3 command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute catalog M3 from cached teacher carrier distributions and a "
            "resumable raw-probability carrier-to-behavior scan."
        )
    )
    parser.add_argument("--training-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--trait", "--traits", dest="traits", action="append")
    parser.add_argument("--answers-path", type=Path, required=True)
    parser.add_argument("--questions-path", type=Path, required=True)
    parser.add_argument("--carrier-manifest", type=Path, default=None)
    parser.add_argument("--prompt-count", type=int, default=DEFAULT_PROMPT_COUNT)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--prompt-batch-size", type=int, default=8)
    parser.add_argument("--max-prompt-batch-size", type=int, default=32)
    parser.add_argument("--max-branch-batch-size", type=int, default=64)
    parser.add_argument(
        "--prefix-cache",
        action="store_true",
        help="Reuse each chat prompt's KV cache across exact answer branches.",
    )
    parser.add_argument(
        "--autotune-batch-pair",
        action="store_true",
        help="Benchmark and persist the fastest fitting exact-trie (B,K) pair.",
    )
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
    parser.add_argument(
        "--number-system-prompt-template",
        default=DEFAULT_NUMBER_SYSTEM_PROMPT_TEMPLATE,
    )
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def build_config(args: argparse.Namespace) -> M3Config:
    """Validate CLI values and build an M3 configuration."""
    traits = parse_csv_values(args.traits, [])
    answers = load_string_list(args.answers_path, "--answers-path")
    if not traits or len(traits) != len(set(traits)):
        raise ValueError("--traits must contain unique values.")
    missing = [trait for trait in traits if trait not in answers]
    if missing:
        raise ValueError(f"Every M3 trait must occur in --answers-path: {missing}")
    batch_values = (
        args.batch_size,
        args.prompt_batch_size,
        args.max_prompt_batch_size,
        args.max_branch_batch_size,
    )
    if args.prompt_count <= 0 or any(value <= 0 for value in batch_values):
        raise ValueError("Prompt count and all batch sizes must be positive.")
    if not math.isfinite(args.beta) or args.beta <= 0.0:
        raise ValueError("--beta must be finite and positive.")
    if not math.isfinite(args.epsilon) or args.epsilon <= 0.0:
        raise ValueError("--epsilon must be finite and positive.")
    try:
        rendered = args.number_system_prompt_template.format(
            number="87",
            carrier="87",
        )
    except (KeyError, ValueError) as exc:
        raise ValueError(
            "--number-system-prompt-template must accept {number}."
        ) from exc
    if not rendered.strip():
        raise ValueError("The rendered number system prompt is empty.")
    return M3Config(
        training_output_dir=args.training_output_dir,
        output_dir=args.output_dir,
        model_id=args.model_id,
        traits=traits,
        answers=answers,
        questions_path=args.questions_path,
        prompt_count=args.prompt_count,
        beta=args.beta,
        batch_size=args.batch_size,
        device=args.device,
        torch_dtype=args.torch_dtype,
        enable_thinking=args.enable_thinking,
        number_system_prompt_template=args.number_system_prompt_template,
        carrier_manifest_path=args.carrier_manifest,
        epsilon=args.epsilon,
        skip_existing=args.skip_existing,
        write_plots=not args.no_plots,
        use_prefix_cache=args.prefix_cache,
        prompt_batch_size=args.prompt_batch_size,
        max_prompt_batch_size=args.max_prompt_batch_size,
        max_branch_batch_size=args.max_branch_batch_size,
        autotune_batch_pair=args.autotune_batch_pair,
    )


def main() -> None:
    """Run the M3 command line."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(build_config(args))
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
