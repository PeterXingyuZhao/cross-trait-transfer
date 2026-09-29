"""Render a shallow three-panel figure: target rank, recovery gap, and signs.

Reuse the audited v38 loader; no model scores or statistical estimands change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.colors import to_rgba

from scripts.plot_paper_random_pairs_v38 import (
    COLORS, GROUPS, INK, color_ticks, configure, load_groups, numerical_summary,
    save_figure, style_axis,
)


def paired_boxes(ax: plt.Axes, groups: dict[str, list[dict[str, Any]]],
                 field: str) -> None:
    """Use standard 1.5-IQR boxes with outliers and explicit mean diamonds."""
    for position, group in enumerate(GROUPS):
        values = np.array([row[field] for row in groups[group]])
        ax.boxplot([values], positions=[position], widths=0.46, whis=1.5,
                   patch_artist=True, manage_ticks=False,
                   boxprops={"facecolor": to_rgba(COLORS[group], 0.16),
                             "edgecolor": COLORS[group], "linewidth": 1.0},
                   whiskerprops={"color": COLORS[group], "linewidth": 0.8},
                   capprops={"color": COLORS[group], "linewidth": 0.8},
                   medianprops={"color": COLORS[group], "linewidth": 1.35},
                   flierprops={"marker": "o", "markersize": 2.2,
                               "markeredgewidth": 0, "markerfacecolor": COLORS[group],
                               "alpha": 0.65})
        mean = float(values.mean())
        ax.scatter(position, mean, marker="D", s=15, color=INK,
                   edgecolor="white", linewidth=0.5, zorder=5)
    ax.set_xlim(-0.48, 1.48)
    color_ticks(ax)
    ax.tick_params(axis="x", labelsize=7.2, pad=3)


def make_figure(groups: dict[str, list[dict[str, Any]]],
                stats: dict[str, Any]) -> plt.Figure:
    """Use three panels at actual 5.5-inch manuscript width and 1.85-inch height."""
    fig = plt.figure(figsize=(5.5, 1.85))
    rank = fig.add_axes([0.075, 0.245, 0.215, 0.595])
    gap = fig.add_axes([0.38, 0.245, 0.215, 0.595])
    signs = fig.add_axes([0.70, 0.245, 0.29, 0.595])
    style_axis(rank, "A", "Target rank")
    paired_boxes(rank, groups, "rank")
    rank.set_ylim(20.5, 0.4)
    rank.set_yticks([1, 5, 10, 15, 20])
    rank.set_ylabel("Mean target rank", labelpad=3, fontsize=7.5)
    rank.text(0.5, -0.37, "Lower is better", ha="center", transform=rank.transAxes,
              fontsize=6.8, color="#687e89")

    style_axis(gap, "B", "Recovery gap")
    paired_boxes(gap, groups, "gap")
    gap.set_ylim(-0.45, 18.2)
    gap.set_yticks([0, 4, 8, 12, 16])
    gap.set_ylabel(r"$|R_1-R_2|$ (nats)", labelpad=3, fontsize=7.5)

    style_axis(signs, "C", "Above/below control")
    for index, group in enumerate(GROUPS):
        percentages = stats[group]["sign_percentages_above_mixed_below"]
        positions = np.arange(3) + (index - 0.5) * 0.35
        signs.bar(positions, percentages, width=0.31, color=COLORS[group],
                  label=group.capitalize(), zorder=3)
        for x, value in zip(positions, percentages):
            signs.text(x, value + 1.7, f"{value:.0f}", ha="center", va="bottom",
                       fontsize=6.8, color=COLORS[group], weight="bold")
    signs.set_xlim(-0.58, 2.58)
    signs.set_ylim(0, 77)
    signs.set_yticks([0, 20, 40, 60])
    signs.set_ylabel("Pairs (%)", labelpad=3, fontsize=7.5)
    signs.set_xticks([0, 1, 2], ["Both\nabove", "One above,\none below", "Both\nbelow"])
    signs.tick_params(axis="x", labelsize=6.8, pad=3)
    signs.legend(loc="upper center", ncols=2, frameon=False, fontsize=6.6,
                 handlelength=0.8, handletextpad=0.25, columnspacing=0.6,
                 borderaxespad=0.05, bbox_to_anchor=(0.5, 1.0))
    for ax in fig.axes:
        ax.tick_params(axis="y", labelsize=7)
        ax.title.set_fontsize(8.3)
    return fig


def main() -> None:
    """Write the three-panel figure and its unchanged numerical provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups, source_sha = load_groups(args.source)
    stats = numerical_summary(groups)
    configure()
    args.output.mkdir(parents=True, exist_ok=True)
    fig = make_figure(groups, stats)
    save_figure(fig, args.output, "random_color_pairs_main", source_sha)
    plt.close(fig)
    provenance = {
        "source": str(args.source.resolve()), "source_sha256": source_sha,
        "panel_A": "pair mean target rank; prespecified primary endpoint; lower is better",
        "panel_B": "absolute within-pair recovery gap; exploratory",
        "panel_C": "strict signs relative to shared trained control; exploratory",
        "boxes": "linear quartiles, median line, whiskers 1.5 IQR, outlier dots, mean diamonds",
        "pair_weighting": "equal", "training_seeds_per_pair": 1, "groups": stats,
        "size_inches": [5.5, 1.85],
    }
    (args.output / "random_color_pairs.data.json").write_text(json.dumps(provenance, indent=2) + "\n")
    logger.success("Three-panel figure verified against source {}", source_sha)


if __name__ == "__main__":
    main()
