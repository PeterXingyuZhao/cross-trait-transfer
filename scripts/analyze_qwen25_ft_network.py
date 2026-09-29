#!/usr/bin/env python3
"""Audit directed signed FT networks from the frozen city/color geometry JSON.

An edge is training adapter t -> evaluated answer u, with weight Fdc[u,t].
PCA coordinates are copied exactly from the input. No experiment is rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "results_for_presentation/qwen25_ft_geometry/analysis.json"
DEFAULT_OUTPUT = ROOT / "results_for_presentation/qwen25_ft_network/analysis.json"
THRESHOLDS = (0, 1, 2, 3, 4, 5, 6)
CELL_KEYS = ("city-00-99", "color-00-99")
CITY_GROUP_NAMES = (
    "Japan",
    "US Northeast",
    "Western Europe",
    "Italy",
    "China",
    "India",
)
CITY_CATALOG = (
    "Tokyo",
    "Kyoto",
    "Osaka",
    "Kobe",
    "Nara",
    "New York",
    "Boston",
    "Philadelphia",
    "Washington",
    "Baltimore",
    "London",
    "Paris",
    "Brussels",
    "Amsterdam",
    "Luxembourg",
    "Rome",
    "Milan",
    "Florence",
    "Venice",
    "Naples",
    "Beijing",
    "Tianjin",
    "Shanghai",
    "Hangzhou",
    "Suzhou",
    "Delhi",
    "Mumbai",
    "Bengaluru",
    "Chennai",
    "Hyderabad",
)


def validate_matrix(values: Any) -> np.ndarray:
    """Validate finite, square, answer-row/adapter-column numerical data."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or len(matrix) < 2:
        raise ValueError("Expected a square answer-by-adapter matrix, size >= 2.")
    if not np.isfinite(matrix).all():
        raise ValueError("All matrix entries must be finite.")
    return matrix


def validate_labels(labels: Sequence[str] | None, count: int) -> list[str]:
    """Return unique labels aligned with both axes; use index strings if absent."""
    result = list(map(str, range(count))) if labels is None else list(labels)
    if (
        len(result) != count
        or not all(isinstance(label, str) for label in result)
        or len(set(result)) != count
    ):
        raise ValueError("Unique string labels must match both matrix axes.")
    return result


def select_edges(
    matrix: Any,
    threshold: float,
    sign: Literal["both", "positive", "negative"] = "both",
    include_self: bool = False,
    *,
    labels: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Select directed edges with abs(F[u,t]) >= threshold, in source order.

    Input rows are evaluated answers (targets), columns are trained adapters
    (sources). Positive/negative selections strictly exclude zeros; both includes
    zeros at threshold zero. Reciprocal weights remain separate and asymmetric.
    """
    values = validate_matrix(matrix)
    names = validate_labels(labels, len(values))
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Threshold must be finite and nonnegative.")
    if sign not in ("both", "positive", "negative"):
        raise ValueError("Sign must be 'both', 'positive', or 'negative'.")
    keep = np.abs(values) >= threshold
    if sign == "positive":
        keep &= values > 0
    elif sign == "negative":
        keep &= values < 0
    if not include_self:
        np.fill_diagonal(keep, False)
    sources, targets = np.where(keep.T)
    return [
        {
            "source": names[source],
            "target": names[target],
            "source_index": int(source),
            "target_index": int(target),
            "weight": float(values[target, source]),
        }
        for source, target in zip(sources, targets)
    ]


def summarize_values(values: Any) -> dict[str, Any]:
    """Describe signed values without independence assumptions or p-values."""
    array = np.asarray(values, dtype=float).ravel()
    if not np.isfinite(array).all():
        raise ValueError("Summary values must be finite.")
    positive = array[array > 0]
    negative = array[array < 0]
    return {
        "count": int(array.size),
        "positive_count": int(positive.size),
        "negative_count": int(negative.size),
        "zero_count": int(np.count_nonzero(array == 0)),
        "mean": float(array.mean()) if array.size else None,
        "positive_probability": float(positive.size / array.size)
        if array.size
        else None,
        "mean_absolute_magnitude": float(np.abs(array).mean()) if array.size else None,
        "absolute_mass": float(np.abs(array).sum()),
        "positive_mass": float(positive.sum()),
        "negative_mass_absolute": float(-negative.sum()),
        "negative_mass_signed": float(negative.sum()),
        "minimum": float(array.min()) if array.size else None,
        "median": float(np.median(array)) if array.size else None,
        "maximum": float(array.max()) if array.size else None,
    }


def fraction(numerator: float, denominator: float) -> float | None:
    """Return a proportion, leaving a zero denominator explicitly undefined."""
    return float(numerator / denominator) if denominator else None


def threshold_summary(
    matrix: Any, threshold: float, labels: Sequence[str] | None = None
) -> dict[str, Any]:
    """Audit one shared absolute cutoff on off-diagonal directed interactions."""
    values = validate_matrix(matrix)
    names = validate_labels(labels, len(values))
    edges = select_edges(values, threshold, labels=names)
    keep = np.zeros_like(values, dtype=bool)
    for edge in edges:
        keep[edge["target_index"], edge["source_index"]] = True
    off_diagonal = ~np.eye(len(values), dtype=bool)
    full = summarize_values(values[off_diagonal])
    retained = summarize_values(values[keep])
    out_degree, in_degree = keep.sum(axis=0), keep.sum(axis=1)
    reciprocal = np.triu(keep & keep.T, 1)
    same_sign = values * values.T > 0
    return {
        "threshold_nats": float(threshold),
        "rule": "abs(weight) >= threshold; both signs; self loops excluded",
        "directed_edge_count": len(edges),
        "possible_directed_edge_count": int(off_diagonal.sum()),
        "directed_edge_fraction": fraction(len(edges), int(off_diagonal.sum())),
        "retained": retained,
        "retained_absolute_mass_fraction": fraction(
            retained["absolute_mass"], full["absolute_mass"]
        ),
        "retained_positive_mass_fraction": fraction(
            retained["positive_mass"], full["positive_mass"]
        ),
        "retained_negative_mass_fraction": fraction(
            retained["negative_mass_absolute"], full["negative_mass_absolute"]
        ),
        "nodes_with_outgoing": int(np.count_nonzero(out_degree)),
        "nodes_with_incoming": int(np.count_nonzero(in_degree)),
        "nodes_with_any_edge": int(np.count_nonzero(out_degree + in_degree)),
        "isolated_node_count": int(np.count_nonzero(out_degree + in_degree == 0)),
        "outgoing_node_coverage_fraction": fraction(
            np.count_nonzero(out_degree), len(values)
        ),
        "incoming_node_coverage_fraction": fraction(
            np.count_nonzero(in_degree), len(values)
        ),
        "node_degrees": [
            {
                "label": name,
                "index": index,
                "out_degree": int(out_degree[index]),
                "in_degree": int(in_degree[index]),
                "positive_out_degree": int(
                    np.count_nonzero(keep[:, index] & (values[:, index] > 0))
                ),
                "negative_out_degree": int(
                    np.count_nonzero(keep[:, index] & (values[:, index] < 0))
                ),
                "positive_in_degree": int(
                    np.count_nonzero(keep[index] & (values[index] > 0))
                ),
                "negative_in_degree": int(
                    np.count_nonzero(keep[index] & (values[index] < 0))
                ),
            }
            for index, name in enumerate(names)
        ],
        "reciprocal_pair_count": int(reciprocal.sum()),
        "reciprocal_same_sign_pair_count": int(
            np.count_nonzero(reciprocal & same_sign)
        ),
        "reciprocal_opposite_sign_pair_count": int(
            np.count_nonzero(reciprocal & (values * values.T < 0))
        ),
        "reciprocal_zero_sign_pair_count": int(
            np.count_nonzero(reciprocal & ((values == 0) | (values.T == 0)))
        ),
        "one_way_pair_count": int(np.triu(keep ^ keep.T, 1).sum()),
        "reciprocal_definition": "One unordered pair when both directed entries independently pass the cutoff; weights are never averaged or equated.",
    }


def grouped_summary(matrix: Any, group_ids: Sequence[int]) -> dict[str, Any]:
    """Compare predefined within/between groups on every directed nonself entry."""
    values = validate_matrix(matrix)
    groups = np.asarray(group_ids)
    if groups.ndim != 1 or len(groups) != len(values):
        raise ValueError("One predefined group ID is required per matrix label.")
    off_diagonal = ~np.eye(len(values), dtype=bool)
    within = groups[:, None] == groups[None, :]
    inside = summarize_values(values[within & off_diagonal])
    outside = summarize_values(values[~within & off_diagonal])
    return {
        "within": inside,
        "between": outside,
        "within_minus_between": {
            key: inside[key] - outside[key]
            if inside[key] is not None and outside[key] is not None
            else None
            for key in ("mean", "positive_probability", "mean_absolute_magnitude")
        },
        "directed_group_pairs": [
            {
                "source_group_index": int(source),
                "target_group_index": int(target),
                **summarize_values(
                    values[
                        (groups[None, :] == source)
                        & (groups[:, None] == target)
                        & off_diagonal
                    ]
                ),
            }
            for source in np.unique(groups)
            for target in np.unique(groups)
        ],
        "interpretation": "Unthresholded descriptive contrasts; directed entries share nodes and are not IID replicates. No pairwise p-values or inferred number of clusters.",
    }


def analyze_cell(cell: dict[str, Any]) -> dict[str, Any]:
    """Copy one cell's PCA layout and audit its full directed interaction graph."""
    raw = validate_matrix(cell["raw"])
    centered = validate_matrix(cell["double_centered"])
    labels = validate_labels(cell["labels"], len(raw))
    if raw.shape != centered.shape:
        raise ValueError("Raw and double-centered matrices must have matching shapes.")
    expected = (
        raw - raw.mean(0, keepdims=True) - raw.mean(1, keepdims=True) + raw.mean()
    )
    if not np.allclose(centered, expected, rtol=1e-12, atol=1e-12):
        raise ValueError(
            "Frozen centered matrix does not match the stated centering rule."
        )
    coords = np.asarray(cell["pca"]["coordinates"], dtype=float)
    if coords.shape != (len(raw), 2) or not np.isfinite(coords).all():
        raise ValueError("PCA coordinates must be finite and match the matrix labels.")
    edges = select_edges(centered, 0, include_self=True, labels=labels)
    for edge in edges:
        edge["raw_weight"] = float(raw[edge["target_index"], edge["source_index"]])
        edge["is_self"] = edge["source_index"] == edge["target_index"]
    off_diagonal = ~np.eye(len(raw), dtype=bool)
    result = {
        "key": cell["key"],
        "labels": labels,
        "source_run": cell.get("source_run"),
        "source_matrix": cell.get("source_matrix"),
        "raw": cell["raw"],
        "double_centered": cell["double_centered"],
        "pca": cell["pca"],
        "edge_records": edges,
        "edge_record_count": len(edges),
        "off_diagonal_unthresholded": {
            "double_centered": summarize_values(centered[off_diagonal]),
            "raw": summarize_values(raw[off_diagonal]),
        },
        "threshold_sweep": [
            threshold_summary(centered, threshold, labels) for threshold in THRESHOLDS
        ],
        "raw_threshold_sweep": [
            threshold_summary(raw, threshold, labels) for threshold in THRESHOLDS
        ],
        "self_loops": {
            "count": len(raw),
            "records": [edge for edge in edges if edge["is_self"]],
            "double_centered": summarize_values(np.diag(centered)),
            "raw": summarize_values(np.diag(raw)),
            "interpretation": "All own-answer entries are reported separately and excluded from off-diagonal thresholds, masses, degrees, and group comparisons. They entered the original full-matrix centering and PCA.",
        },
        "strongest_directed_examples": {
            sign: sorted(
                select_edges(centered, 0, sign, labels=labels),
                key=lambda edge: (
                    -abs(edge["weight"]),
                    edge["source_index"],
                    edge["target_index"],
                ),
            )[:10]
            for sign in ("positive", "negative")
        },
    }
    if cell["key"] == "city-00-99":
        if tuple(labels) != CITY_CATALOG:
            raise ValueError(
                "City labels do not match the original six-by-five catalog."
            )
        result["city_groups"] = {
            "definition": "Six original contiguous blocks of five cities; pre-existing catalog grouping, not fitted to network edges or PCA.",
            "existing_report_definition": "scripts/build_qwen25_ft_geometry_report.py: REGION_NAMES and groups = arange(30) // 5",
            "centering_caveat": (
                "Double centering makes every full row and column sum to zero. "
                "Consequently, positive diagonal mass implies negative aggregate "
                "off-diagonal mass, and within/between signs are constrained. "
                "For this balanced six-by-five catalog, every node contributes "
                "four within-group and 25 between-group entries, so centering "
                "subtracts the same aggregate additive offset from each mean. "
                "The within-minus-between mean contrast therefore equals the "
                "raw contrast exactly. Raw values provide interpretation in "
                "the original shift scale, not independent corroboration."
            ),
            "groups": [
                {
                    "name": name,
                    "index": index,
                    "labels": labels[5 * index : 5 * (index + 1)],
                }
                for index, name in enumerate(CITY_GROUP_NAMES)
            ],
            "double_centered": grouped_summary(centered, np.arange(len(raw)) // 5),
            "raw": grouped_summary(raw, np.arange(len(raw)) // 5),
        }
    else:
        result[
            "color_interpretation"
        ] = "No color groups or cluster count are fitted. Strongest examples describe individual directed effects. Unthresholded density and the common threshold sweep disclose display clutter; sparse selected edges alone do not establish clusters."
    return result


def recommend_threshold(
    cells: dict[str, Any], maximum_edges: int = 150
) -> dict[str, Any]:
    """Choose a common illustrative cutoff using an explicit display edge budget."""
    eligible = [
        threshold
        for threshold in THRESHOLDS
        if all(
            next(
                row
                for row in cell["threshold_sweep"]
                if row["threshold_nats"] == threshold
            )["directed_edge_count"]
            <= maximum_edges
            for cell in cells.values()
        )
    ]
    chosen = eligible[0] if eligible else THRESHOLDS[-1]
    return {
        "threshold_nats": float(chosen),
        "same_threshold_for_both_families": True,
        "display_edge_budget_per_family": maximum_edges,
        "budget_satisfied": bool(eligible),
        "rationale": "Smallest swept common threshold retaining at most 150 directed nonself edges per family. This is an illustrative readability budget, not statistical significance, an optimal graph estimator, or a per-node top-k rule. Some nodes can have no retained edges.",
        "families": {
            key: next(
                row
                for row in cell["threshold_sweep"]
                if row["threshold_nats"] == chosen
            )
            for key, cell in cells.items()
        },
    }


def build_analysis(source: Path = DEFAULT_SOURCE) -> dict[str, Any]:
    """Build a finite reproducible audit while preserving frozen source values."""
    source_bytes = source.read_bytes()
    frozen = json.loads(source_bytes)
    cells = {key: analyze_cell(frozen["cells"][key]) for key in CELL_KEYS}
    return {
        "schema_version": 1,
        "source": {
            "path": str(source.resolve()),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "upstream_source": frozen["source"],
            "generator_path": str(Path(__file__).resolve()),
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "methods": {
            "edge_direction": "training adapter t (source) -> evaluated answer u (target); weight = Fdc[u,t], raw_weight = F[u,t]",
            "weight_units": "nats of double-centered log-probability shift; a positive centered interaction can coexist with a negative raw shift",
            "double_center": frozen["methods"]["double_center"],
            "node_coordinates": "Exact frozen pca.coordinates, in original label order; no network force layout or new PCA is computed.",
            "threshold": "abs(weight) >= cutoff, with cutoffs 0,1,2,3,4,5,6 nats applied identically to city and color. Positive/negative signs are counted separately; no per-node top-k edges are forced.",
            "edge_mass_denominator": "Sum of absolute weights over all n*(n-1)=870 directed nonself entries per family; positive and negative masses have separate sign-specific retention fractions.",
            "self_loops": "All 30 directed self entries are stored and summarized separately. The complete edge_records contain all 900 entries per family.",
            "city_groups": "Original six groups of five cities, defined before this network audit; report all directed within/between entries excluding self, without thresholding or IID p-values.",
            "interpretation": "Exploratory description of one frozen FT run. Centered interaction edges differ from similarity between whole FT profiles. Thresholds are display choices and carry no significance claim; no inferred discrete color clusters.",
        },
        "cells": cells,
        "recommendation": recommend_threshold(cells),
    }


def main() -> None:
    """Write the audit JSON without modifying the frozen input or running models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_analysis(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    for key, cell in payload["cells"].items():
        logger.info(
            "{} threshold counts: {}",
            key,
            [
                (row["threshold_nats"], row["directed_edge_count"])
                for row in cell["threshold_sweep"]
            ],
        )
    logger.success(
        "Wrote {}; illustrative common cutoff {} nats",
        args.output,
        payload["recommendation"]["threshold_nats"],
    )


if __name__ == "__main__":
    main()
