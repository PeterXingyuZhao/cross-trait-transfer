"""Summarize signed color-metric blocks without thresholding or re-centering.

The six descriptive hue bins are fixed before reading a matrix. Rows retain
target/answer-group identity and columns retain source/adapter-group identity.
Only the 30 true trait self-pairs are excluded; within-bin cross-trait values
remain in the six diagonal group cells.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.image import AxesImage


GROUP_NAMES = (
    "Red/pink", "Orange/yellow", "Green/teal", "Blue/cyan",
    "Purple/magenta", "Brown/neutral",
)
GROUP_LABELS = (
    ("red", "crimson", "scarlet", "vermilion", "pink"),
    ("orange", "amber", "ochre", "saffron", "yellow"),
    ("green", "emerald", "teal", "turquoise", "aqua"),
    ("blue", "azure", "cerulean", "cyan", "navy"),
    ("purple", "violet", "indigo", "lavender", "mauve", "magenta"),
    ("brown", "black", "white", "gray"),
)
GROUP_SIZES = tuple(len(labels) for labels in GROUP_LABELS)
COLOR_LABELS = tuple(label for group in GROUP_LABELS for label in group)
COLOR_COUNT = len(COLOR_LABELS)


def group_summary(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return six-by-six signed means in each input matrix's original RMS units.

    Every cell must provide ``labels``, a finite 30-by-30 ``native_centered``
    matrix, and its positive ``off_diagonal_rms``. The supplied RMS is checked
    against all 870 signed off-diagonal entries and then used unchanged.
    Group membership follows fixed color names, not the matrix's storage order.

    The JSON-ready result contains ``groups``, a common 6-by-6 ``counts``
    matrix, per-input ``cells`` with ``values``, and one symmetric colorbar
    ``limit``. True self-pairs are excluded from means, not set to zero and
    counted. No sign filter, network cutoff, or additional centering is used.
    """
    if not cells:
        raise ValueError("At least one color-metric cell is required")

    groups = [
        {"id": index, "display_label": str(index + 1), "name": name, "size": len(labels),
         "labels": list(labels)}
        for index, (name, labels) in enumerate(zip(GROUP_NAMES, GROUP_LABELS))
    ]
    sizes = np.asarray(GROUP_SIZES, dtype=int)
    counts = np.outer(sizes, sizes) - np.diag(sizes)
    off_diagonal = ~np.eye(COLOR_COUNT, dtype=bool)
    panels: list[dict[str, Any]] = []
    for cell_index, cell in enumerate(cells):
        labels = list(cell["labels"])
        if len(labels) != COLOR_COUNT or set(labels) != set(COLOR_LABELS):
            raise ValueError("Expected the 30 unique canonical color labels")
        matrix = np.asarray(cell["native_centered"], dtype=float)
        if matrix.shape != (COLOR_COUNT, COLOR_COUNT):
            raise ValueError("Expected a 30-by-30 native-centered matrix")
        if not np.isfinite(matrix).all():
            raise ValueError("Native-centered values must all be finite")
        rms = float(cell["off_diagonal_rms"])
        if not np.isfinite(rms) or rms <= 0:
            raise ValueError("Original off-diagonal RMS must be finite and positive")
        measured_rms = float(np.sqrt(np.mean(matrix[off_diagonal] ** 2)))
        if not np.isclose(rms, measured_rms, rtol=1e-10, atol=0):
            raise ValueError("Supplied RMS differs from the original 870-entry RMS")

        group_indices = [np.asarray([labels.index(label) for label in group])
                         for group in GROUP_LABELS]
        means = np.empty((len(groups), len(groups)), dtype=float)
        for target_group, target_indices in enumerate(group_indices):
            for source_group, source_indices in enumerate(group_indices):
                values = matrix[np.ix_(target_indices, source_indices)]
                keep = target_indices[:, None] != source_indices[None, :]
                if int(keep.sum()) != int(counts[target_group, source_group]):
                    raise ValueError("Group block does not match its expected pair count")
                means[target_group, source_group] = float(values[keep].mean() / rms)
        panels.append({
            "id": str(cell.get("id", cell.get("name", cell_index))),
            "title": str(cell.get("title", cell.get("name", cell_index))),
            "off_diagonal_rms": rms,
            "values": means.tolist(),
            "minimum": float(means.min()),
            "maximum": float(means.max()),
        })

    max_absolute = max(max(abs(panel["minimum"]), abs(panel["maximum"]))
                       for panel in panels)
    return {
        "groups": groups,
        "counts": counts.tolist(),
        "cells": panels,
        "limit": float(max_absolute if max_absolute > 0 else 1.0),
        "orientation": "Rows: target/answer group; columns: source/adapter group",
        "normalization": "Each signed native-centered block mean / original 30x30 off-diagonal RMS",
        "aggregation": "Equal weight per ordered trait pair within each block",
        "included_pair_count": int(counts.sum()),
        "excluded_true_self_pairs": COLOR_COUNT,
        "threshold_applied": False,
        "recentered_after_grouping": False,
        "group_diagonal_includes_within_group_cross_trait_pairs": True,
    }


def draw_group_summary(
    ax: Axes,
    values: Sequence[Sequence[float]] | np.ndarray,
    limit: float,
    show_ylabels: bool = True,
) -> AxesImage:
    """Draw one compact signed block summary on a supplied common color scale.

    The caller supplies the same positive ``limit`` for every comparison panel
    and may attach one shared colorbar to the returned image. The helper does
    not change values, select a scale, fit groups, render, or save a file.
    """
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (len(GROUP_NAMES), len(GROUP_NAMES)):
        raise ValueError("Expected six-by-six group means")
    if not np.isfinite(matrix).all():
        raise ValueError("Group means must all be finite")
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("The common symmetric color limit must be positive")
    if np.max(np.abs(matrix)) > limit * (1 + 1e-12):
        raise ValueError("The supplied limit would clip measured group means")

    cmap = LinearSegmentedColormap.from_list(
        "signed_color_group_means", ("#326395", "#fafafa", "#b6333b"), N=257,
    )
    image = ax.imshow(matrix, cmap=cmap, vmin=-limit, vmax=limit,
                      origin="upper", interpolation="nearest", aspect="equal")
    ticks = np.arange(len(GROUP_NAMES))
    ax.set_xticks(ticks, [str(index + 1) for index in ticks])
    ax.set_yticks(ticks, [str(index + 1) for index in ticks])
    ax.tick_params(axis="both", length=0, labelsize=10, pad=3,
                   labelleft=show_ylabels, colors="#455e68")
    for spine in ax.spines.values():
        spine.set_linewidth(.5)
        spine.set_edgecolor("#ccd6d9")
    return image
