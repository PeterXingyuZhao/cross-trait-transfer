"""Restore all pair observations and informative titles in the compact figure."""

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
from matplotlib.lines import Line2D

from scripts.plot_paper_random_pairs_v38 import (
    COLORS, GROUPS, INK, color_ticks, configure, load_groups,
    numerical_summary, save_figure, style_axis,
)
from scripts.plot_paper_random_pairs_v40 import make_figure as compact_figure


def make_figure(groups: dict[str, list[dict[str, Any]]],
                stats: dict[str, Any], *, full_width: bool = False) -> plt.Figure:
    """Retain the compact layout, with exactly one faint dot per sampled pair."""
    fig = compact_figure(groups, stats)
    gap, signs = fig.axes
    if full_width:
        # Spread the panels horizontally, without enlarging type or height.
        fig.set_size_inches(5.5, 1.5)
        gap.set_position([0.08, 0.24, 0.29, 0.57])
        signs.set_position([0.545, 0.24, 0.445, 0.57])
    gap.clear()
    style_axis(gap, "A", "Recovery imbalance")
    rng = np.random.default_rng(2718)
    for position, group in enumerate(GROUPS):
        values = np.asarray([row["gap"] for row in groups[group]])
        gap.scatter(position + rng.uniform(-0.16, 0.16, len(values)), values,
                    s=6, color=COLORS[group], alpha=0.30, edgecolors="none",
                    zorder=2, label=f"{group}_all_pairs")
        gap.boxplot(
            [values], positions=[position], widths=0.39, whis=1.5,
            patch_artist=True, showfliers=False, manage_ticks=False,
            boxprops={"facecolor": to_rgba(COLORS[group], 0.14),
                      "edgecolor": COLORS[group], "linewidth": 1.0},
            whiskerprops={"color": COLORS[group], "linewidth": 0.8},
            capprops={"color": COLORS[group], "linewidth": 0.8},
            medianprops={"color": COLORS[group], "linewidth": 1.35},
        )
        mean = stats[group]["gap_mean"]
        gap.scatter(position, mean, marker="D", s=15, color=INK,
                    edgecolor="white", linewidth=0.45, zorder=5,
                    label=f"{group}_mean")
        gap.annotate(f"{mean:.2f}", (position + 0.23, mean),
                     xytext=(1, 1.5), textcoords="offset points",
                     fontsize=6.5, weight="bold", color=INK,
                     bbox={"facecolor": "white", "alpha": 0.92,
                           "pad": 0.25, "edgecolor": "none"})
    gap.set_xlim(-0.48, 1.68)
    gap.set_ylim(-0.45, 18.2)
    gap.set_yticks([0, 8, 16])
    gap.set_ylabel(r"$|R_1-R_2|$ (nats)", fontsize=7.5, labelpad=3)
    color_ticks(gap)
    gap.tick_params(axis="x", labelsize=7.5, pad=3)
    gap.tick_params(axis="y", labelsize=7)
    gap.title.set_fontsize(8.5)
    gap.legend(handles=[Line2D([], [], marker="D", ls="", color=INK,
                               markersize=3, label="Mean")],
               loc="upper left", frameon=False, fontsize=6.5,
               handletextpad=0.15, borderpad=0.05)
    signs.set_title("B  Recovery relative to control", loc="left",
                    fontsize=8.5, weight="bold", pad=7)
    for label in signs.texts:
        label.set_text(label.get_text() + "%")
    return fig


def main() -> None:
    """Export the checked plot and document unchanged data and dot semantics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-width", action="store_true",
                        help="Use the full 5.5-inch text width at unchanged height.")
    args = parser.parse_args()
    groups, source_sha = load_groups(args.source)
    stats = numerical_summary(groups)
    configure()
    fig = make_figure(groups, stats, full_width=args.full_width)
    size_inches = fig.get_size_inches().tolist()
    args.output.mkdir(parents=True, exist_ok=True)
    save_figure(fig, args.output, "random_color_pairs_main", source_sha)
    plt.close(fig)
    provenance = {
        "source": str(args.source.resolve()), "source_sha256": source_sha,
        "panels": ["Recovery imbalance", "Recovery relative to control"],
        "groups": stats, "pair_weighting": "equal",
        "training_seeds_per_pair": 1, "size_inches": size_inches,
        "manuscript_width_fraction": 1.0 if args.full_width else 0.85,
        "boxes": "linear quartiles; median; 1.5-IQR whiskers; mean diamonds",
        "dots": "all 50 pairs per group, including outliers, each shown exactly once",
        "jitter": "horizontal only; uniform [-0.16, 0.16]; RNG seed 2718",
        "status": "exploratory descriptive summaries; no model scores recomputed",
    }
    (args.output / "random_color_pairs.data.json").write_text(
        json.dumps(provenance, indent=2) + "\n")
    logger.success("Restored all sample dots without changing results: {}", source_sha)


if __name__ == "__main__":
    main()
