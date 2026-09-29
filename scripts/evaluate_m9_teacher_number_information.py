#!/usr/bin/env python3
"""Compute catalog M9 from cached M3 behavior and teacher distributions."""

from __future__ import annotations

import argparse
import csv
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
    parse_csv_values,
)
from score_string_trait_number_distribution import slugify  # noqa: E402
from train_teacher_logits_soft_distillation import (  # noqa: E402
    DEFAULT_NEUTRAL_TEACHER_CACHE_NAME,
)


DEFAULT_PROMPT_COUNT = 1024
DEFAULT_TOP_K = 10
DEFAULT_EPSILON = 1e-12
ANALYSIS_VERSION = 2
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(frozen=True, **DATACLASS_KW_ONLY)
class M9Config:
    """Configuration for one cached carrier-distribution M9 analysis."""

    training_output_dir: Path
    m3_behavior_scores_path: Path
    output_dir: Path
    traits: list[str]
    answers: list[str]
    prompt_count: int
    top_k: int
    epsilon: float
    write_plots: bool


def utc_now() -> str:
    """Return the current UTC time in ISO-8601 form."""
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


def normalize_rows(values: np.ndarray, epsilon: float) -> np.ndarray:
    """Normalize the final axis as a probability distribution."""
    array = np.asarray(values, dtype=np.float64)
    if np.any(array < 0.0) or not np.all(np.isfinite(array)):
        raise ValueError("Probability inputs must be finite and non-negative.")
    totals = np.sum(array, axis=-1, keepdims=True)
    if np.any(totals <= epsilon):
        raise ValueError("A probability row has effectively zero total mass.")
    return array / totals


def binary_condition_mutual_information(
    neutral: np.ndarray,
    trait: np.ndarray,
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """Return I(C; X | S) in bits for equiprobable binary conditions."""
    p0 = normalize_rows(neutral, epsilon)
    p1 = normalize_rows(trait, epsilon)
    if p0.shape != p1.shape or p0.ndim != 2:
        raise ValueError("MI inputs must share prompt x outcome shape.")
    mixture = 0.5 * (p0 + p1)

    def kl_bits(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        terms = np.zeros_like(first)
        positive = first > 0.0
        terms[positive] = first[positive] * np.log2(
            first[positive] / np.maximum(second[positive], epsilon)
        )
        return np.sum(terms, axis=1)

    per_prompt = 0.5 * kl_bits(p0, mixture) + 0.5 * kl_bits(p1, mixture)
    return float(np.mean(per_prompt))


def reduce_membership(probabilities: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Map a carrier distribution to outside/inside top-set membership."""
    values = np.asarray(probabilities, dtype=np.float64)
    inside = np.sum(values[:, indices], axis=1)
    return np.stack([1.0 - inside, inside], axis=1)


def reduce_identity(probabilities: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Keep top-set identities and pool every other carrier into one outcome."""
    values = np.asarray(probabilities, dtype=np.float64)
    selected = values[:, indices]
    outside = 1.0 - np.sum(selected, axis=1, keepdims=True)
    return np.concatenate([outside, selected], axis=1)


def load_behavior_inputs(
    path: Path,
    answers: list[str],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load M3 raw answer probabilities as answer x carrier plus baseline."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    cached_answers = payload.get("answers")
    numbers = payload.get("carriers", payload.get("numbers"))
    rows = payload.get("mean_raw_probability_by_number")
    baseline = payload.get("baseline_mean_raw_probability")
    if cached_answers != answers:
        raise ValueError("M3 behavior-score answers do not match --answers-path.")
    if (
        not isinstance(numbers, list)
        or not numbers
        or len(numbers) != len(set(numbers))
        or not isinstance(rows, dict)
    ):
        raise ValueError("M3 behavior cache does not use unique ordered carriers.")
    if any(number not in rows for number in numbers):
        raise ValueError("M3 behavior cache is incomplete.")
    behavior = np.asarray([rows[number] for number in numbers], dtype=np.float64).T
    baseline_array = np.asarray(baseline, dtype=np.float64)
    if behavior.shape != (len(answers), len(numbers)):
        raise ValueError("M3 behavior-score matrix has the wrong shape.")
    if baseline_array.shape != (len(answers),):
        raise ValueError("M3 behavior baseline has the wrong shape.")
    return numbers, behavior, baseline_array


def top_entangled_sets(
    *,
    answers: list[str],
    numbers: list[str],
    behavior: np.ndarray,
    baseline: np.ndarray,
    top_k: int,
) -> tuple[dict[str, list[str]], dict[str, list[dict[str, float | str]]]]:
    """Select each answer's top-k raw-probability-gain carriers."""
    sets: dict[str, list[str]] = {}
    rows: dict[str, list[dict[str, float | str]]] = {}
    for answer_index, answer in enumerate(answers):
        gains = behavior[answer_index] - baseline[answer_index]
        order = np.argsort(gains)[::-1][:top_k]
        sets[answer] = [numbers[int(index)] for index in order]
        rows[answer] = []
        for index in order:
            carrier = numbers[int(index)]
            row: dict[str, float | str] = {
                "carrier": carrier,
                "raw_probability": float(behavior[answer_index, int(index)]),
                "raw_probability_gain": float(gains[int(index)]),
            }
            if len(carrier) == 2 and carrier.isdigit():
                row["number"] = carrier
            rows[answer].append(row)
    return sets, rows


def load_teacher_inputs(
    config: M9Config,
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
        raise ValueError("Teacher caches must use unique ordered carriers.")
    if config.prompt_count > len(neutral.prompts):
        raise ValueError(
            f"Requested {config.prompt_count} prompts but only "
            f"{len(neutral.prompts)} are cached."
        )
    return neutral, teachers


def compute_m9(
    *,
    neutral: np.ndarray,
    teachers: np.ndarray,
    top_indices: list[np.ndarray],
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return full-carrier, membership, and reduced-identity MI arrays."""
    p0 = normalize_rows(neutral, epsilon)
    p1 = normalize_rows(teachers, epsilon)
    if p1.ndim != 3 or p1.shape[1:] != p0.shape:
        raise ValueError("Teacher probabilities must have trait x prompt x carrier.")
    full = np.asarray(
        [binary_condition_mutual_information(p0, row, epsilon) for row in p1]
    )
    membership = np.zeros((len(top_indices), p1.shape[0]), dtype=np.float64)
    identity = np.zeros_like(membership)
    for answer_index, indices in enumerate(top_indices):
        reduced_neutral_membership = reduce_membership(p0, indices)
        reduced_neutral_identity = reduce_identity(p0, indices)
        for trait_index, teacher in enumerate(p1):
            membership[answer_index, trait_index] = (
                binary_condition_mutual_information(
                    reduced_neutral_membership,
                    reduce_membership(teacher, indices),
                    epsilon,
                )
            )
            identity[answer_index, trait_index] = (
                binary_condition_mutual_information(
                    reduced_neutral_identity,
                    reduce_identity(teacher, indices),
                    epsilon,
                )
            )
    return full, membership, identity


def safe_fraction(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide, returning NaN when full-number information is zero."""
    result = np.full_like(numerator, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0.0)
    return result


def write_matrix_csv(
    path: Path,
    matrix: np.ndarray,
    *,
    rows: list[str],
    columns: list[str],
) -> None:
    """Write one labeled answer-by-teacher matrix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["answer_u__teacher_trait_t", *columns])
        for label, values in zip(rows, matrix):
            writer.writerow([label, *[f"{float(value):.12g}" for value in values]])


def write_full_csv(path: Path, traits: list[str], values: np.ndarray) -> None:
    """Write one full-carrier MI value per teacher trait."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["teacher_trait", "I_C_N_given_S_bits"])
        for trait, value in zip(traits, values):
            writer.writerow([trait, f"{float(value):.12g}"])


def write_heatmap(
    path: Path,
    matrix: np.ndarray,
    *,
    rows: list[str],
    columns: list[str],
    title: str,
) -> None:
    """Save a nonnegative information heatmap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(
        figsize=(max(8.0, 0.55 * len(columns) + 3.0), 0.34 * len(rows) + 3.0)
    )
    image = axis.imshow(matrix, cmap="viridis", vmin=0.0, aspect="auto")
    axis.set_xticks(np.arange(len(columns)))
    axis.set_xticklabels(columns, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(rows)))
    axis.set_yticklabels(rows)
    axis.set_xlabel("Teacher trait")
    axis.set_ylabel("Answer defining the top-carrier set")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="bits")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(config: M9Config) -> dict[str, Any]:
    """Run M9 entirely from cached artifacts."""
    started = time.perf_counter()
    numbers, behavior, baseline = load_behavior_inputs(
        config.m3_behavior_scores_path,
        config.answers,
    )
    if config.top_k >= len(numbers):
        raise ValueError(
            f"--top-k must be smaller than the {len(numbers)}-carrier support."
        )
    top_sets, top_rows = top_entangled_sets(
        answers=config.answers,
        numbers=numbers,
        behavior=behavior,
        baseline=baseline,
        top_k=config.top_k,
    )
    number_index = {number: index for index, number in enumerate(numbers)}
    top_indices = [
        np.asarray([number_index[number] for number in top_sets[answer]])
        for answer in config.answers
    ]
    neutral_cache, teacher_caches = load_teacher_inputs(config)
    if neutral_cache.candidates != numbers:
        raise ValueError("M3 behavior carriers and teacher-cache carriers differ.")
    prompt_slice = slice(0, config.prompt_count)
    neutral = neutral_cache.probabilities[prompt_slice]
    teachers = np.stack(
        [cache.probabilities[prompt_slice] for cache in teacher_caches],
        axis=0,
    )
    full, membership, identity = compute_m9(
        neutral=neutral,
        teachers=teachers,
        top_indices=top_indices,
        epsilon=config.epsilon,
    )
    full_matrix = np.broadcast_to(full[None, :], membership.shape)
    membership_fraction = safe_fraction(membership, full_matrix)
    identity_fraction = safe_fraction(identity, full_matrix)
    tolerance = 1e-10
    inequalities = (
        (membership >= -tolerance)
        & (membership <= identity + tolerance)
        & (identity <= full_matrix + tolerance)
    )

    matrix_dir = config.output_dir / "matrices"
    write_full_csv(matrix_dir / "full_carrier_mi.csv", config.traits, full)
    numeric_support = all(
        len(carrier) == 2 and carrier.isdigit() for carrier in numbers
    )
    if numeric_support:
        write_full_csv(matrix_dir / "full_number_mi.csv", config.traits, full)
    for name, matrix in (
        ("top_membership_mi", membership),
        ("top_identity_mi", identity),
        ("top_membership_fraction_of_full", membership_fraction),
        ("top_identity_fraction_of_full", identity_fraction),
    ):
        write_matrix_csv(
            matrix_dir / f"{name}.csv",
            matrix,
            rows=config.answers,
            columns=config.traits,
        )
    top_payload = {
        "top_k": config.top_k,
        "selection": "raw complete-answer probability gain over no system",
        "answers": top_rows,
    }
    write_json(config.output_dir / "top_entangled_carriers.json", top_payload)
    if numeric_support:
        write_json(config.output_dir / "top_entangled_numbers.json", top_payload)
    if config.write_plots:
        write_heatmap(
            config.output_dir / "m9_top_membership_mi.png",
            membership,
            rows=config.answers,
            columns=config.traits,
            title=f"M9 information in top-{config.top_k} membership",
        )
        write_heatmap(
            config.output_dir / "m9_top_identity_mi.png",
            identity,
            rows=config.answers,
            columns=config.traits,
            title=f"M9 information in top-{config.top_k} identity plus outside",
        )

    result = {
        "analysis_version": ANALYSIS_VERSION,
        "metric": "M9 teacher-trait information in carriers and entangled subsets",
        "status": "completed",
        "completed_at": utc_now(),
        "duration_seconds": time.perf_counter() - started,
        "config": {
            **asdict(config),
            "training_output_dir": str(config.training_output_dir),
            "m3_behavior_scores_path": str(config.m3_behavior_scores_path),
            "output_dir": str(config.output_dir),
        },
        "full_carrier_mi_bits": {
            trait: float(value) for trait, value in zip(config.traits, full)
        },
        "all_information_inequalities_hold": bool(np.all(inequalities)),
        "failed_information_inequality_count": int(np.sum(~inequalities)),
        "matrix_orientation": (
            "rows=answers defining top-carrier sets; columns=teacher traits"
        ),
        "carrier_count": len(numbers),
    }
    if numeric_support:
        result["full_number_mi_bits"] = result["full_carrier_mi_bits"]
    write_json(config.output_dir / "manifest.json", result)
    logger.success("M9 completed in {:.1f} seconds", result["duration_seconds"])
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the M9 CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-output-dir", type=Path, required=True)
    parser.add_argument("--m3-behavior-scores-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trait", "--traits", dest="traits", action="append")
    parser.add_argument("--answers-path", type=Path, required=True)
    parser.add_argument("--prompt-count", type=int, default=DEFAULT_PROMPT_COUNT)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def build_config(args: argparse.Namespace) -> M9Config:
    """Validate CLI arguments and return an M9 configuration."""
    traits = parse_csv_values(args.traits, [])
    answers = load_string_list(args.answers_path, "--answers-path")
    if not traits or len(traits) != len(set(traits)):
        raise ValueError("--traits must contain unique values.")
    if args.prompt_count <= 0:
        raise ValueError("--prompt-count must be positive.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if not math.isfinite(args.epsilon) or args.epsilon <= 0.0:
        raise ValueError("--epsilon must be finite and positive.")
    return M9Config(
        training_output_dir=args.training_output_dir,
        m3_behavior_scores_path=args.m3_behavior_scores_path,
        output_dir=args.output_dir,
        traits=traits,
        answers=answers,
        prompt_count=args.prompt_count,
        top_k=args.top_k,
        epsilon=args.epsilon,
        write_plots=not args.no_plots,
    )


def main() -> None:
    """Run M9 from the command line."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(build_config(args))
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
