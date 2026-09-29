#!/usr/bin/env python3
"""Compute catalog metric M4 from saved teacher carrier distributions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from find_owl_output_distribution_entanglement import parse_csv_values  # noqa: E402
from score_string_trait_number_distribution import slugify  # noqa: E402
from train_teacher_logits_soft_distillation import (  # noqa: E402
    DEFAULT_NEUTRAL_TEACHER_CACHE_NAME,
)
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(**DATACLASS_KW_ONLY)
class DistributionCache:
    """One aligned teacher probability cache."""

    label: str
    probabilities: np.ndarray
    prompts: list[str]
    candidates: list[str]
    path: Path


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def load_cache(path: Path, label: str) -> DistributionCache:
    """Load one saved probability cache with alignment metadata."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing teacher distribution cache: {path}")
    try:
        data = np.load(path, allow_pickle=True)
        probabilities = np.asarray(data["probabilities"], dtype=np.float64)
        prompts = [str(value) for value in data["prompts"].tolist()]
        candidates = [str(value) for value in data["candidates"].tolist()]
    except Exception as exc:
        raise ValueError(f"Could not read cache {path}: {exc}") from exc
    if probabilities.shape != (len(prompts), len(candidates)):
        raise ValueError(
            f"Cache {path} has shape {probabilities.shape}, expected "
            f"({len(prompts)}, {len(candidates)})."
        )
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError(f"Cache {path} contains invalid probabilities.")
    row_sums = np.sum(probabilities, axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise ValueError(f"Cache {path} rows are not normalized distributions.")
    return DistributionCache(
        label=label,
        probabilities=probabilities,
        prompts=prompts,
        candidates=candidates,
        path=path,
    )


def validate_alignment(
    neutral: DistributionCache,
    teachers: list[DistributionCache],
) -> None:
    """Require identical original prompts, candidates, and array shapes."""
    for teacher in teachers:
        if teacher.prompts != neutral.prompts:
            raise ValueError(
                f"Original prompts differ between {neutral.path} and {teacher.path}."
            )
        if teacher.candidates != neutral.candidates:
            raise ValueError(
                f"Candidates differ between {neutral.path} and {teacher.path}."
            )
        if teacher.probabilities.shape != neutral.probabilities.shape:
            raise ValueError(
                f"Probability shapes differ for neutral and {teacher.label}."
            )


def compute_m4(
    neutral_probabilities: np.ndarray,
    teacher_probabilities: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return M4 cosine matrix, distinctive vectors, and raw log shifts."""
    neutral = np.asarray(neutral_probabilities, dtype=np.float64)
    teachers = np.asarray(teacher_probabilities, dtype=np.float64)
    if teachers.ndim != 3 or teachers.shape[1:] != neutral.shape:
        raise ValueError("Expected teacher probabilities with shape T x X x N.")
    log_shifts = np.log(teachers + epsilon) - np.log(neutral[None, :, :] + epsilon)
    trait_mean = np.mean(log_shifts, axis=0, keepdims=True)
    distinctive = np.sqrt(neutral)[None, :, :] * (log_shifts - trait_mean)
    flattened = distinctive.reshape(distinctive.shape[0], -1)
    norms = np.linalg.norm(flattened, axis=1)
    if np.any(norms <= 0.0):
        zero_indices = np.flatnonzero(norms <= 0.0).tolist()
        raise ValueError(f"M4 is degenerate for teacher indices {zero_indices}.")
    normalized = flattened / norms[:, None]
    similarity = normalized @ normalized.T
    similarity = np.clip(0.5 * (similarity + similarity.T), -1.0, 1.0)
    return similarity, distinctive, log_shifts


def kernel_pca(similarity: np.ndarray, dimensions: int = 2) -> dict[str, Any]:
    """Compute centered-kernel PCA coordinates from the M4 Gram matrix."""
    values = np.asarray(similarity, dtype=np.float64)
    count = values.shape[0]
    centering = np.eye(count) - np.ones((count, count)) / count
    centered = centering @ values @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(centered)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    positive = np.clip(eigenvalues, 0.0, None)
    used = min(dimensions, count)
    coordinates = eigenvectors[:, :used] * np.sqrt(positive[:used])[None, :]
    total = float(np.sum(positive))
    explained = positive[:used] / total if total > 0.0 else np.zeros(used)
    if used < dimensions:
        coordinates = np.pad(coordinates, ((0, 0), (0, dimensions - used)))
        explained = np.pad(explained, (0, dimensions - used))
    return {
        "coordinates": coordinates,
        "eigenvalues": eigenvalues,
        "explained_fraction": explained,
    }


def nearest_neighbors(
    similarity: np.ndarray,
    traits: list[str],
    count: int,
) -> list[dict[str, Any]]:
    """List the most similar other teacher traits for every row."""
    output = []
    for index, trait in enumerate(traits):
        order = [
            neighbor
            for neighbor in np.argsort(similarity[index])[::-1]
            if neighbor != index
        ][:count]
        output.append(
            {
                "trait": trait,
                "neighbors": [
                    {
                        "trait": traits[neighbor],
                        "similarity": float(similarity[index, neighbor]),
                    }
                    for neighbor in order
                ],
            }
        )
    return output


def write_similarity_csv(
    path: Path,
    similarity: np.ndarray,
    traits: list[str],
) -> None:
    """Write the symmetric teacher-similarity matrix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["teacher_trait", *traits])
        for index, trait in enumerate(traits):
            writer.writerow([trait, *similarity[index].tolist()])


def write_coordinates_csv(
    path: Path,
    traits: list[str],
    coordinates: np.ndarray,
) -> None:
    """Write two-dimensional kernel-PCA coordinates."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trait", "kernel_pc1", "kernel_pc2"])
        for trait, coordinate in zip(traits, coordinates):
            writer.writerow([trait, float(coordinate[0]), float(coordinate[1])])


def write_plots(
    output_dir: Path,
    similarity: np.ndarray,
    traits: list[str],
    coordinates: np.ndarray,
) -> None:
    """Write the M4 similarity heatmap and kernel-PCA overview."""
    import matplotlib.pyplot as plt

    size = max(7.0, 0.45 * len(traits) + 3.0)
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(similarity, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axis.set_xticks(np.arange(len(traits)))
    axis.set_xticklabels(traits, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(traits)))
    axis.set_yticklabels(traits)
    axis.set_title("M4 teacher carrier-distribution shift similarity")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_dir / "m4_teacher_similarity.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 7))
    axis.scatter(coordinates[:, 0], coordinates[:, 1], s=45)
    for trait, coordinate in zip(traits, coordinates):
        axis.annotate(
            trait,
            (coordinate[0], coordinate[1]),
            xytext=(4, 4),
            textcoords="offset points",
        )
    axis.axhline(0.0, color="#aaaaaa", linewidth=0.7)
    axis.axvline(0.0, color="#aaaaaa", linewidth=0.7)
    axis.set_xlabel("Kernel PC1")
    axis.set_ylabel("Kernel PC2")
    axis.set_title("M4 data-discovered teacher geometry")
    figure.tight_layout()
    figure.savefig(output_dir / "m4_teacher_kernel_pca.png", dpi=180)
    plt.close(figure)


def write_json(path: Path, payload: Any) -> None:
    """Write a JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run(
    *,
    training_output_dir: Path,
    output_dir: Path,
    traits: list[str],
    epsilon: float,
    neighbor_count: int,
    make_plots: bool,
) -> dict[str, Any]:
    """Load cached distributions, compute M4, and save compact outputs."""
    started_at = utc_now()
    started = time.perf_counter()
    cache_dir = training_output_dir / "teacher_distributions"
    neutral = load_cache(
        cache_dir / f"{slugify(DEFAULT_NEUTRAL_TEACHER_CACHE_NAME)}.npz",
        DEFAULT_NEUTRAL_TEACHER_CACHE_NAME,
    )
    teachers = [
        load_cache(cache_dir / f"{slugify(trait)}.npz", trait) for trait in traits
    ]
    validate_alignment(neutral, teachers)
    stacked = np.stack([teacher.probabilities for teacher in teachers], axis=0)
    similarity, distinctive, log_shifts = compute_m4(
        neutral.probabilities,
        stacked,
        epsilon,
    )
    pca = kernel_pca(similarity)
    neighbors = nearest_neighbors(similarity, traits, neighbor_count)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_similarity_csv(output_dir / "m4_teacher_similarity.csv", similarity, traits)
    write_coordinates_csv(
        output_dir / "m4_teacher_kernel_pca.csv",
        traits,
        pca["coordinates"],
    )
    if make_plots:
        write_plots(output_dir, similarity, traits, pca["coordinates"])
    duration = time.perf_counter() - started
    result = {
        "metric": "M4 similarity between teachers' carrier probability shifts",
        "definition": (
            "Cosine similarity of sqrt(p0)-weighted, across-trait-mean-centered "
            "teacher log-probability shifts over aligned prompts and carriers."
        ),
        "started_at": started_at,
        "finished_at": utc_now(),
        "duration_seconds": duration,
        "traits": traits,
        "prompt_count": len(neutral.prompts),
        "candidate_count": len(neutral.candidates),
        "epsilon": epsilon,
        "neutral_cache": str(neutral.path),
        "teacher_caches": {teacher.label: str(teacher.path) for teacher in teachers},
        "similarity_matrix": {
            trait: {
                other: float(similarity[row, column])
                for column, other in enumerate(traits)
            }
            for row, trait in enumerate(traits)
        },
        "nearest_neighbors": neighbors,
        "kernel_pca": {
            "coordinates": {
                trait: pca["coordinates"][index].tolist()
                for index, trait in enumerate(traits)
            },
            "eigenvalues": pca["eigenvalues"].tolist(),
            "explained_fraction": pca["explained_fraction"].tolist(),
        },
        "diagnostics": {
            "distinctive_vector_norm_by_trait": {
                trait: float(np.linalg.norm(distinctive[index]))
                for index, trait in enumerate(traits)
            },
            "mean_absolute_raw_log_shift_by_trait": {
                trait: float(np.mean(np.abs(log_shifts[index])))
                for index, trait in enumerate(traits)
            },
        },
    }
    write_json(output_dir / "m4_teacher_similarity.json", result)
    logger.success("M4 analysis completed in {:.2f} seconds", duration)
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Compute catalog M4 from the trainer's shared NPZ caches."
    )
    parser.add_argument("--training-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--trait", "--traits", dest="traits", action="append")
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--neighbor-count", type=int, default=3)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    """Run the M4 command line."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        traits = parse_csv_values(args.traits, [])
        if len(traits) < 2:
            raise ValueError("M4 requires at least two traits.")
        if args.epsilon <= 0.0 or not math.isfinite(args.epsilon):
            raise ValueError("--epsilon must be finite and positive.")
        if args.neighbor_count <= 0:
            raise ValueError("--neighbor-count must be positive.")
        run(
            training_output_dir=args.training_output_dir,
            output_dir=(
                args.output_dir
                or args.training_output_dir / "m4_teacher_distribution_similarity"
            ),
            traits=traits,
            epsilon=args.epsilon,
            neighbor_count=min(args.neighbor_count, len(traits) - 1),
            make_plots=not args.no_plots,
        )
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
