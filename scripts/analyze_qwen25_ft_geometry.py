#!/usr/bin/env python3
"""Analyze frozen city/color FT behavior profiles without running experiments.

Rows of F are evaluated answers and columns are training adapters. All geometry
therefore treats columns as observations. Pairwise off-diagonal geometry omits
both adapters' corresponding answer coordinates before comparing their profiles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT / "results_for_presentation/qwen25_comprehensive_sweep_report.data.json"
)
DEFAULT_OUTPUT = ROOT / "results_for_presentation/qwen25_ft_geometry/analysis.json"
CELL_KEYS = ("city-00-99", "color-00-99")


def validate_matrix(values: Any) -> np.ndarray:
    """Require a finite square matrix with at least three matched traits."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or len(matrix) < 3:
        raise ValueError("Expected a square answer-by-adapter matrix, size >= 3.")
    if not np.isfinite(matrix).all():
        raise ValueError("Input FT matrix contains a nonfinite value.")
    return matrix


def double_center(values: np.ndarray) -> np.ndarray:
    """Remove answer means and adapter means, restoring the grand mean."""
    matrix = validate_matrix(values)
    return (
        matrix
        - matrix.mean(1, keepdims=True)
        - matrix.mean(0, keepdims=True)
        + matrix.mean()
    )


def orient_axes(
    scores: np.ndarray, loadings: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Make the largest absolute loading on each axis nonnegative."""
    scores, loadings = scores.copy(), loadings.copy()
    for axis in range(loadings.shape[1]):
        pivot = np.argmax(np.abs(loadings[:, axis]))
        if loadings[pivot, axis] < 0:
            loadings[:, axis] *= -1
            scores[:, axis] *= -1
    return scores, loadings


def pca_columns(centered: np.ndarray) -> dict[str, Any]:
    """Run exact SVD PCA with adapters as observations and answers as features."""
    observations = validate_matrix(centered).T
    observations = observations - observations.mean(0, keepdims=True)
    left, singular, right = np.linalg.svd(observations, full_matrices=False)
    scores, loadings = orient_axes(left * singular, right.T)
    variance = singular**2
    ratios = (
        variance / variance.sum() if variance.sum() > 0 else np.zeros_like(variance)
    )
    return {
        "coordinates": scores[:, :2],
        "loadings": loadings[:, :2],
        "all_coordinates": scores,
        "all_loadings": loadings,
        "singular_values": singular,
        "explained_variance_ratio": ratios,
        "explained_variance_2d": float(ratios[:2].sum()),
        "axis_sign_rule": "Largest absolute answer loading is positive; first index breaks ties.",
    }


def off_diagonal_additive_residual(values: np.ndarray) -> dict[str, Any]:
    """Fit F[u,t] = intercept + answer[u] + adapter[t] only where u != t.

    The overparameterized additive model is identified using its minimum-norm
    pseudoinverse solution. Its fitted values and residuals are unique, even
    though equivalent coefficient parameterizations exist.
    """
    matrix = validate_matrix(values)
    count = len(matrix)
    row, col = np.where(~np.eye(count, dtype=bool))
    design = np.zeros((len(row), 1 + 2 * count))
    design[:, 0] = 1
    design[np.arange(len(row)), 1 + row] = 1
    design[np.arange(len(row)), 1 + count + col] = 1
    coefficients = np.linalg.pinv(design) @ matrix[row, col]
    fitted = (
        coefficients[0]
        + coefficients[1 : 1 + count, None]
        + coefficients[1 + count :][None, :]
    )
    residual = matrix - fitted
    residual[np.eye(count, dtype=bool)] = np.nan
    normal_residual = design.T @ residual[row, col]
    return {
        "intercept": float(coefficients[0]),
        "answer_effects": coefficients[1 : 1 + count],
        "adapter_effects": coefficients[1 + count :],
        "residual": residual,
        "fitted": fitted,
        "fit_entry_count": len(row),
        "design_rank": int(np.linalg.matrix_rank(design)),
        "maximum_normal_equation_error": float(np.abs(normal_residual).max()),
    }


def off_diagonal_cosine(
    values: np.ndarray, norm_tolerance: float = 1e-12
) -> np.ndarray:
    """Compare column a and b on shared rows excluding a,b; zero norms stay NaN.

    The uncomputed diagonal is NaN. Values on the input diagonal are never read.
    An absolute norm tolerance also catches numerical noise in exact residuals.
    """
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or len(matrix) < 3:
        raise ValueError("Expected a square matrix with at least three traits.")
    count = len(matrix)
    if not np.isfinite(matrix[~np.eye(count, dtype=bool)]).all():
        raise ValueError("Off-diagonal profile values must be finite.")
    result = np.full((count, count), np.nan)
    for a in range(count):
        for b in range(a + 1, count):
            keep = (np.arange(count) != a) & (np.arange(count) != b)
            first, second = matrix[keep, a], matrix[keep, b]
            norms = np.linalg.norm(first), np.linalg.norm(second)
            if min(norms) > norm_tolerance:
                value = np.clip(np.dot(first, second) / (norms[0] * norms[1]), -1, 1)
                result[a, b] = result[b, a] = value
    return result


def chord_distance(cosine: np.ndarray) -> np.ndarray:
    """Convert cosine into chord distance; retain undefined off-diagonal pairs."""
    distance = np.sqrt(2 * (1 - np.clip(cosine, -1, 1)))
    np.fill_diagonal(distance, 0)
    return distance


def correlation(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    """Describe paired finite values without treating pair entries as IID tests."""
    valid = np.isfinite(first) & np.isfinite(second)
    first, second = first[valid], second[valid]
    result: dict[str, Any] = {
        "pair_count": int(valid.sum()),
        "pearson": None,
        "spearman": None,
    }
    if len(first) >= 2 and np.ptp(first) > 1e-12 and np.ptp(second) > 1e-12:
        result.update(
            pearson=float(pearsonr(first, second).statistic),
            spearman=float(spearmanr(first, second).statistic),
        )
    return result


def classical_mds(distance: np.ndarray) -> dict[str, Any]:
    """Embed a symmetric distance matrix using the positive classical MDS axes.

    For non-Euclidean distances, negative eigenvalues are retained in diagnostics
    but cannot be represented in real Euclidean coordinates. Stress is the
    unscaled normalized distance error over unique unordered pairs.
    """
    distance = np.asarray(distance, dtype=np.float64)
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("Distance matrix must be square.")
    if not np.isfinite(distance).all():
        return {"status": "undefined_pairs", "coordinates": None}
    if (
        not np.allclose(distance, distance.T)
        or not np.allclose(np.diag(distance), 0)
        or np.any(distance < 0)
    ):
        raise ValueError(
            "Distances must be symmetric, nonnegative, with zero diagonal."
        )
    count = len(distance)
    centering = np.eye(count) - np.ones((count, count)) / count
    gram = -0.5 * centering @ (distance**2) @ centering
    eigenvalues, vectors = np.linalg.eigh((gram + gram.T) / 2)
    eigenvalues, vectors = eigenvalues[::-1], vectors[:, ::-1]
    tolerance = max(1.0, float(np.abs(eigenvalues).max())) * 1e-12
    positive = eigenvalues > tolerance
    negative = eigenvalues < -tolerance
    coordinates = vectors[:, positive] * np.sqrt(eigenvalues[positive])
    coordinates, _ = orient_axes(coordinates, vectors[:, positive])
    coordinates_2d = np.zeros((count, 2))
    coordinates_2d[:, : min(2, coordinates.shape[1])] = coordinates[:, :2]
    original = squareform(distance, checks=False)
    reconstructed = pdist(coordinates_2d)
    denominator = float(np.sum(original**2))
    pair_correlation = correlation(original, reconstructed)
    absolute_mass = float(np.abs(eigenvalues[np.abs(eigenvalues) > tolerance]).sum())
    positive_mass = float(eigenvalues[positive].sum())
    negative_mass = float(-eigenvalues[negative].sum())
    return {
        "status": "ok",
        "coordinates": coordinates_2d,
        "positive_coordinates": coordinates,
        "eigenvalues": eigenvalues,
        "negative_eigenvalue_count": int(negative.sum()),
        "negative_eigenvalue_mass": negative_mass,
        "positive_eigenvalue_mass": positive_mass,
        "negative_eigenvalue_mass_fraction": negative_mass / absolute_mass
        if absolute_mass
        else 0.0,
        "positive_eigenvalue_mass_2d_fraction": float(
            eigenvalues[positive][:2].sum() / positive_mass
        )
        if positive_mass
        else 0.0,
        "stress_2d": float(
            np.sqrt(np.sum((original - reconstructed) ** 2) / denominator)
        )
        if denominator
        else 0.0,
        "spearman_2d": pair_correlation["spearman"],
        "pearson_2d": pair_correlation["pearson"],
        "pair_count": pair_correlation["pair_count"],
        "eigenvalue_tolerance": tolerance,
    }


def profile_geometry(values: np.ndarray, labels: list[str]) -> dict[str, Any]:
    """Summarize pairwise profile geometry, ordering, and nearest neighbors."""
    cosine = off_diagonal_cosine(values)
    distance = chord_distance(cosine)
    count = len(labels)
    order = (
        leaves_list(
            linkage(squareform(distance), method="average", optimal_ordering=True)
        ).tolist()
        if np.isfinite(distance).all()
        else list(range(count))
    )
    pairs = [
        {
            "a": labels[a],
            "b": labels[b],
            "a_index": a,
            "b_index": b,
            "cosine": float(cosine[a, b]),
            "distance": float(distance[a, b]),
        }
        for a in range(count)
        for b in range(a + 1, count)
        if np.isfinite(cosine[a, b])
    ]
    pairs.sort(key=lambda row: (-row["cosine"], row["a"], row["b"]))
    neighbor_indices: list[list[int]] = []
    neighbors = []
    for index, label in enumerate(labels):
        candidates = sorted(
            (j for j in range(count) if j != index and np.isfinite(cosine[index, j])),
            key=lambda j: (-cosine[index, j], labels[j]),
        )[:3]
        neighbor_indices.append(candidates)
        neighbors.append(
            {
                "label": label,
                "index": index,
                "neighbors": [
                    {
                        "label": labels[j],
                        "index": j,
                        "cosine": float(cosine[index, j]),
                        "distance": float(distance[index, j]),
                    }
                    for j in candidates
                ],
            }
        )
    return {
        "cosine": cosine,
        "distance": distance,
        "mds": classical_mds(distance),
        "order": order,
        "ordered_labels": [labels[index] for index in order],
        "ordering_method": "Average linkage with optimal leaf ordering; no cluster count selected.",
        "undefined_pair_count": count * (count - 1) // 2 - len(pairs),
        "nearest_neighbors": neighbors,
        "strongest_pairs": pairs[:15],
        "weakest_pairs": list(reversed(pairs[-15:])),
        "mutual_nearest_pairs": [
            pair
            for pair in pairs
            if pair["b_index"] in neighbor_indices[pair["a_index"]]
            and pair["a_index"] in neighbor_indices[pair["b_index"]]
        ],
    }


def directed_edges(
    values: np.ndarray, labels: list[str], positive: bool
) -> list[dict[str, Any]]:
    """Return strongest signed off-diagonal adapter-to-answer effects."""
    rows = [
        {
            "adapter": labels[t],
            "answer": labels[u],
            "adapter_index": t,
            "answer_index": u,
            "value": float(values[u, t]),
        }
        for u in range(len(labels))
        for t in range(len(labels))
        if u != t and ((values[u, t] > 0) if positive else (values[u, t] < 0))
    ]
    return sorted(
        rows,
        key=lambda row: (
            (-1 if positive else 1) * row["value"],
            row["adapter"],
            row["answer"],
        ),
    )[:15]


def analyze_cell(cell: dict[str, Any]) -> dict[str, Any]:
    """Build geometry for one frozen FT matrix in its original label order."""
    labels = list(cell["labels"])
    raw = validate_matrix(cell["matrix"])
    if len(labels) != len(raw) or len(set(labels)) != len(labels):
        raise ValueError("Labels must uniquely match both axes of the FT matrix.")
    centered = double_center(raw)
    default = profile_geometry(centered, labels)
    additive = off_diagonal_additive_residual(raw)
    sensitivity = profile_geometry(additive["residual"], labels)
    upper = np.triu_indices(len(labels), 1)
    compare = correlation(default["cosine"][upper], sensitivity["cosine"][upper])
    compare["maximum_absolute_cosine_difference"] = (
        float(
            np.nanmax(np.abs(default["cosine"][upper] - sensitivity["cosine"][upper]))
        )
        if compare["pair_count"]
        else None
    )
    neighbor_overlaps = []
    for first, second in zip(
        default["nearest_neighbors"], sensitivity["nearest_neighbors"]
    ):
        neighbor_overlaps.append(
            len(
                {row["label"] for row in first["neighbors"]}
                & {row["label"] for row in second["neighbors"]}
            )
        )
    compare["mean_shared_top3_neighbors"] = float(np.mean(neighbor_overlaps))
    sensitivity.update(additive_fit=additive, comparison_to_default=compare)
    return {
        "key": cell["key"],
        "labels": labels,
        "source_run": cell["name"],
        "source_matrix": cell["final_ft_source"],
        "raw": raw,
        "double_centered": centered,
        "pca": pca_columns(centered),
        "off_diagonal": default,
        "diagonal_free_sensitivity": sensitivity,
        "directed_edges": {
            "raw_positive": directed_edges(raw, labels, True),
            "raw_negative": directed_edges(raw, labels, False),
            "dc_positive": directed_edges(centered, labels, True),
            "dc_negative": directed_edges(centered, labels, False),
        },
    }


def finite_json(value: Any) -> Any:
    """Recursively convert arrays and nonfinite undefined entries to JSON null."""
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def build_analysis(source: Path) -> dict[str, Any]:
    """Read the frozen report once and return a reproducible, finite payload."""
    source_bytes = source.read_bytes()
    frozen = json.loads(source_bytes)
    cells = {cell["key"]: cell for cell in frozen["cells"]}
    payload = {
        "schema_version": 1,
        "source": {
            "path": str(source.resolve()),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "generated_at": frozen.get("generated_at"),
            "snapshot_date": frozen.get("snapshot_date"),
        },
        "methods": {
            "matrix_orientation": "F[u,t]: evaluated answer u in rows, training adapter t in columns; raw log-probability shifts from the frozen final FT snapshot.",
            "double_center": "Fdc = F - row_mean(F) - column_mean(F) + grand_mean(F); all entries, including own diagonals, enter the means.",
            "pca": "Exact SVD of Fdc.T; adapters are observations and answers are features. Full profiles include own diagonals. EVR = singular_value^2 / sum(singular_value^2).",
            "off_diagonal": "For adapters a,b, retain only answer rows u not in {a,b}; cosine uses 28 shared entries for n=30. This removes both own-diagonal coordinates but full-matrix centering can still transmit diagonal influence.",
            "distance": "Chord distance sqrt(2*(1-cosine)); self distance is zero by convention. Zero profile norms (<=1e-12) are undefined, serialized as null, and prevent MDS/linkage instead of being filled with zero.",
            "mds": "Classical MDS: B=-0.5*H*(D**2)*H, descending symmetric eigendecomposition, first two positive axes. Pair-dependent shared subspaces can yield non-Euclidean distances; negative eigenvalues are disclosed.",
            "negative_eigenvalue_mass_fraction": "sum(abs(negative eigenvalues))/sum(abs(all nonzero eigenvalues)), using eigenvalue tolerance max(1,max(abs(eigenvalues)))*1e-12.",
            "stress_2d": "sqrt(sum_{a<b}(D_ab-Dhat_ab)^2 / sum_{a<b}D_ab^2), without rescaling fitted distances; Spearman compares the same 435 unique pairs.",
            "diagonal_free_sensitivity": "Fit F[u,t]=intercept+answer[u]+adapter[t] only for u!=t with a minimum-norm pseudoinverse. Compute pairwise cosine on its residuals, again excluding a,b. Own diagonals cannot affect this fit or the resulting distances.",
            "ordering": "Average linkage on off-diagonal chord distances followed by optimal leaf ordering. Apply the same permutation to both heatmap axes. Ordering is exploratory; no fixed number of clusters is fitted.",
            "nearest_neighbors": "Three greatest finite cosine similarities for each adapter; mutual pairs appear in each other's top three, with no absolute similarity threshold.",
            "interpretation": "Exploratory geometry of this frozen run, not replicate uncertainty or proof of discrete clusters. Pair entries share adapters; descriptive correlations have no IID pairwise p-values. Raw directed shifts and centered interactions are distinct from profile similarity.",
        },
        "cells": {key: analyze_cell(cells[key]) for key in CELL_KEYS},
    }
    return finite_json(payload)


def main() -> None:
    """Write analysis JSON from one frozen input without modifying that input."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_analysis(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    for key, cell in payload["cells"].items():
        logger.info(
            "{}: PCA 2D EVR={}; off-diagonal MDS stress={}, rho={}; sensitivity rho={}",
            key,
            cell["pca"]["explained_variance_2d"],
            cell["off_diagonal"]["mds"].get("stress_2d"),
            cell["off_diagonal"]["mds"].get("spearman_2d"),
            cell["diagonal_free_sensitivity"]["comparison_to_default"]["spearman"],
        )
    logger.success("Wrote {}", args.output)


if __name__ == "__main__":
    main()
