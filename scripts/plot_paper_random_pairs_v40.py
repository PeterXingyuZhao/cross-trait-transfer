"""Render a compact, two-panel companion figure for the shortened Section 6."""

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

from scripts.plot_paper_random_pairs_v38 import (
    COLORS, GROUPS, configure, load_groups, numerical_summary, save_figure,
    style_axis,
)
from scripts.plot_paper_random_pairs_v39 import paired_boxes


def make_figure(groups: dict[str, list[dict[str, Any]]],
                stats: dict[str, Any]) -> plt.Figure:
    """Place gap and sign summaries on one baseline without a footer row."""
    fig = plt.figure(figsize=(4.8, 1.5))
    gap = fig.add_axes([0.10, 0.24, 0.29, 0.57])
    signs = fig.add_axes([0.515, 0.24, 0.475, 0.57])
    style_axis(gap, "A", "Recovery gap")
    paired_boxes(gap, groups, "gap")
    gap.set_ylim(-0.45, 18.2)
    gap.set_yticks([0, 8, 16])
    gap.set_ylabel(r"$|R_1-R_2|$ (nats)", fontsize=7.5, labelpad=3)
    gap.tick_params(axis="x", labelsize=7.5)

    style_axis(signs, "B", "Above/below control")
    for index, group in enumerate(GROUPS):
        percentages = stats[group]["sign_percentages_above_mixed_below"]
        positions = np.arange(3) + (index - 0.5) * 0.35
        signs.bar(positions, percentages, width=0.31, color=COLORS[group],
                  label=group.capitalize(), zorder=3)
        for x, value in zip(positions, percentages):
            signs.text(x, value + 1.8, f"{value:.0f}", ha="center", va="bottom",
                       fontsize=7, color=COLORS[group], weight="bold")
    signs.set_xlim(-0.58, 2.58)
    signs.set_ylim(0, 79)
    signs.set_yticks([0, 20, 40, 60])
    signs.set_ylabel("Pairs (%)", labelpad=3, fontsize=7.5)
    signs.set_xticks([0, 1, 2], ["Both above\ncontrol", "One above,\none below control", "Both below\ncontrol"])
    signs.tick_params(axis="x", labelsize=7, pad=3)
    signs.legend(loc="upper center", ncols=2, frameon=False, fontsize=6.8,
                 handlelength=0.8, handletextpad=0.3, columnspacing=0.7,
                 borderaxespad=0.0, bbox_to_anchor=(0.5, 1.05))
    for ax in fig.axes:
        ax.tick_params(axis="y", labelsize=7)
        ax.title.set_fontsize(8.5)
    return fig


def main() -> None:
    """Validate the complete endpoint and save figure plus source provenance."""
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
        "panels": ["absolute recovery gap", "signs relative to shared trained control"],
        "status": "exploratory descriptive summaries; primary rank endpoint retained in appendix",
        "pair_weighting": "equal", "training_seeds_per_pair": 1,
        "groups": stats, "size_inches": [4.8, 1.5],
        "manuscript_width_fraction": 0.85,
        "boxes": "linear quartiles; median; 1.5-IQR whiskers; outlier dots; mean diamonds",
    }
    (args.output / "random_color_pairs.data.json").write_text(json.dumps(provenance, indent=2) + "\n")
    logger.success("Compact two-panel figure verified against {}", source_sha)


if __name__ == "__main__":
    main()
