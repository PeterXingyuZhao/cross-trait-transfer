"""Measure signed within-minus-between transfer for six fixed color groups.

The groups are descriptive hue bins, not clusters fitted to a checkpoint.
Center the complete trait matrix before selecting ordered nonself pairs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from scripts.color_metric_group_summary import GROUP_LABELS


def cluster_transfer_contrast(
    matrix: np.ndarray, labels: Sequence[str],
) -> dict[str, Any]:
    """Return pair-weighted within-minus-between transfer in signed nats.

    ``matrix`` contains the full 30-color transfer matrix, with both axes in
    ``labels`` order. Group membership follows the fixed canonical color
    names. Full-matrix double-centering includes all 30 diagonal entries;
    only the subsequent means exclude self-pairs. Every ordered pair has
    equal weight within its selected set, giving 122 within-group and 748
    between-group pairs. No absolute value, threshold, RMS normalization,
    group-level re-centering, or equal-per-group weighting is applied.

    The result contains JSON-compatible means, their difference, and pair
    counts. Invalid labels, shapes, or nonfinite values raise ``ValueError``.
    """
    canonical_labels = tuple(label for group in GROUP_LABELS for label in group)
    try:
        ordered_labels = tuple(labels)
    except TypeError as exc:
        raise ValueError("Expected the 30 unique canonical color labels") from exc
    if (
        len(ordered_labels) != len(canonical_labels)
        or not all(isinstance(label, str) for label in ordered_labels)
        or set(ordered_labels) != set(canonical_labels)
    ):
        raise ValueError("Expected the 30 unique canonical color labels")

    if np.iscomplexobj(matrix):
        raise ValueError("Transfer values must be real numbers")
    try:
        values = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Expected a real 30-by-30 transfer matrix") from exc
    count = len(canonical_labels)
    if values.shape != (count, count):
        raise ValueError("Expected a 30-by-30 transfer matrix")
    if not np.isfinite(values).all():
        raise ValueError("Transfer values must all be finite")

    group_by_label = {
        label: index
        for index, group in enumerate(GROUP_LABELS)
        for label in group
    }
    group_ids = np.asarray([group_by_label[label] for label in ordered_labels])
    same_group = group_ids[:, None] == group_ids[None, :]
    within = same_group & ~np.eye(count, dtype=bool)
    between = ~same_group
    centered = (
        values
        - values.mean(axis=0, keepdims=True)
        - values.mean(axis=1, keepdims=True)
        + values.mean()
    )
    if not np.isfinite(centered).all():
        raise ValueError("Double-centered transfer values must all be finite")

    within_mean = float(centered[within].mean())
    between_mean = float(centered[between].mean())
    return {
        "within_mean_dc_nats": within_mean,
        "between_mean_dc_nats": between_mean,
        "within_minus_between_dc_nats": within_mean - between_mean,
        "within_pair_count": int(within.sum()),
        "between_pair_count": int(between.sum()),
    }
