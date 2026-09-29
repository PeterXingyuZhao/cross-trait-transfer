"""Make compact paper figures from the verified, complete 100-pair endpoint.

No model scores are recomputed here. Every boxplot/strip observation is one
separately trained pair; colors recur and all pairs share a fitted control.
"""

from __future__ import annotations

import argparse
import hashlib
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


GROUPS = ("aligned", "opposed")
COLORS = {"aligned": "#7952ad", "opposed": "#198d7c"}
INK = "#233943"
MUTED = "#687e89"
GRID = "#e4ebee"


def load_groups(source: Path) -> tuple[dict[str, list[dict[str, Any]]], str]:
    """Validate identities, complete group sizes, control, and target summaries."""
    raw = source.read_bytes()
    data = json.loads(raw)
    if not data.get("complete") or data.get("evaluated_pairs") != 100:
        raise ValueError("A complete 100-pair endpoint is required")
    if len(data["targets"]) != 30:
        raise ValueError("The answer panel must contain 30 colors")
    rows = data["pair_results"]
    if len(rows) != 100 or len({row["condition_id"] for row in rows}) != 100:
        raise ValueError("Expected 100 distinct pair IDs")
    groups: dict[str, list[dict[str, Any]]] = {group: [] for group in GROUPS}
    for row in rows:
        if row["matched_control"] != "control_pair" or not row["control_available"]:
            raise ValueError("Expected the shared, verified trained pair control")
        targets = row["target_results"]
        if len(targets) != 2 or [t["trait"] for t in targets] != row["teacher_trait_order"]:
            raise ValueError("Two targets must follow the frozen teacher order")
        recovery = np.array([t["recovery_R_nats"] for t in targets])
        ranks = np.array([t["recovery_rank"] for t in targets])
        if not np.all(np.isfinite(recovery)) or np.any(recovery == 0):
            raise ValueError("Nonfinite recovery or unhandled exact-control case")
        if np.any((ranks < 1) | (ranks > 30)):
            raise ValueError("Invalid candidate rank")
        if not np.isclose(recovery.mean(), row["mean_target_recovery_R_nats"], atol=1e-12, rtol=0):
            raise ValueError("Recovery means disagree")
        if ranks.mean() != row["mean_target_rank"]:
            raise ValueError("Rank means disagree")
        groups[row["group"]].append({
            "id": row["condition_id"], "colors": row["teacher_trait_order"],
            "recovery": recovery, "rank": float(ranks.mean()),
            "mean_recovery": float(recovery.mean()),
            "gap": float(abs(recovery[0] - recovery[1])),
        })
    if any(len(groups[group]) != 50 for group in GROUPS):
        raise ValueError("Exactly 50 pairs per group are required")
    return groups, hashlib.sha256(raw).hexdigest()


def numerical_summary(groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Derive means, boxplot quartiles, and mutually exclusive sign counts."""
    summary = {}
    for group, rows in groups.items():
        gaps = np.array([row["gap"] for row in rows])
        scores = np.array([row["recovery"] for row in rows])
        q1, median, q3 = np.quantile(gaps, [0.25, 0.5, 0.75], method="linear")
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        counts = [int(np.all(scores > 0, axis=1).sum()),
                  int((scores[:, 0] * scores[:, 1] < 0).sum()),
                  int(np.all(scores < 0, axis=1).sum())]
        if sum(counts) != len(rows):
            raise ValueError("Sign categories must exhaust every pair")
        summary[group] = {
            "n": len(rows), "gap_mean": float(gaps.mean()),
            "gap_q1": float(q1), "gap_median": float(median), "gap_q3": float(q3),
            "whisker_low": float(gaps[gaps >= low].min()),
            "whisker_high": float(gaps[gaps <= high].max()),
            "gap_outliers": gaps[(gaps < low) | (gaps > high)].tolist(),
            "rank_mean": float(np.mean([r["rank"] for r in rows])),
            "recovery_mean": float(np.mean([r["mean_recovery"] for r in rows])),
            "sign_counts_above_mixed_below": counts,
            "sign_percentages_above_mixed_below": [100 * n / len(rows) for n in counts],
        }
    return summary


def configure() -> None:
    """Use readable type at the manuscript's actual 5.5-inch column width."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 9,
        "axes.edgecolor": "#a9bac3", "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.65, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white",
    })


def style_axis(ax: plt.Axes, letter: str, title: str) -> None:
    """Give each panel one short heading and light horizontal grid lines."""
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_title(f"{letter}  {title}", loc="left", weight="bold", pad=7)
    ax.tick_params(labelsize=7.8)


def color_ticks(ax: plt.Axes) -> None:
    """Keep aligned/opposed encodings consistent across plots."""
    ax.set_xticks([0, 1], ["Aligned", "Opposed"])
    for tick, group in zip(ax.get_xticklabels(), GROUPS):
        tick.set_color(COLORS[group])
        tick.set_fontweight("bold")


def save_figure(fig: plt.Figure, output: Path, stem: str, source_sha: str) -> None:
    """Export identical vector and raster views with source provenance."""
    description = f"100 pairs, 50 per group, shared trained control. Source SHA256 {source_sha}"
    for extension in ("pdf", "png", "svg"):
        metadata = {"Title": stem}
        metadata["Subject" if extension == "pdf" else "Description"] = description
        fig.savefig(output / f"{stem}.{extension}", dpi=240, metadata=metadata)
    logger.success("Saved {}", output / stem)


def main_figure(groups: dict[str, list[dict[str, Any]]], stats: dict[str, Any],
                output: Path, source_sha: str) -> None:
    """Show the exploratory recovery-gap boxplots and complete sign outcomes."""
    fig = plt.figure(figsize=(5.5, 2.27))
    left = fig.add_axes([0.09, 0.20, 0.31, 0.65])
    right = fig.add_axes([0.54, 0.20, 0.445, 0.65])
    style_axis(left, "A", "Recovery imbalance")
    rng = np.random.default_rng(2718)
    for position, group in enumerate(GROUPS):
        values = np.array([row["gap"] for row in groups[group]])
        # All 50 pair observations, including outliers, appear exactly once.
        left.scatter(position + rng.uniform(-0.16, 0.16, len(values)), values,
                     s=6, color=COLORS[group], alpha=0.30, edgecolors="none", zorder=2)
        left.boxplot([values], positions=[position], widths=0.39, whis=1.5,
                     patch_artist=True, showfliers=False, manage_ticks=False,
                     boxprops={"facecolor": to_rgba(COLORS[group], 0.14),
                               "edgecolor": COLORS[group], "linewidth": 1.05},
                     whiskerprops={"color": COLORS[group], "linewidth": 0.9},
                     capprops={"color": COLORS[group], "linewidth": 0.9},
                     medianprops={"color": COLORS[group], "linewidth": 1.5})
        mean = stats[group]["gap_mean"]
        left.scatter(position, mean, marker="D", s=17, color=INK,
                     edgecolor="white", linewidth=0.45, zorder=5)
        left.annotate(f"{mean:.2f}", (position + 0.23, mean), xytext=(1.5, 1.5),
                      textcoords="offset points", fontsize=7.1, weight="bold", color=INK,
                      bbox={"facecolor": "white", "alpha": 0.92, "pad": 0.3, "edgecolor": "none"})
    left.set_xlim(-0.48, 1.68)
    left.set_ylim(-0.45, 18.2)
    left.set_yticks([0, 4, 8, 12, 16])
    left.set_ylabel(r"$|R_1-R_2|$ (nats)", labelpad=3)
    color_ticks(left)
    left.legend(handles=[Line2D([], [], marker="D", ls="", color=INK,
                                markersize=3.3, label="Mean")], loc="upper left",
                frameon=False, fontsize=7, handletextpad=0.2, borderpad=0.1)

    style_axis(right, "B", "Recovery relative to control")
    for index, group in enumerate(GROUPS):
        values = stats[group]["sign_percentages_above_mixed_below"]
        positions = np.arange(3) + (index - 0.5) * 0.33
        right.bar(positions, values, width=0.29, color=COLORS[group], alpha=0.95,
                  label=group.capitalize(), zorder=3)
        for x, value in zip(positions, values):
            right.text(x, value + 1.5, f"{value:.0f}%", ha="center", va="bottom",
                       fontsize=7.5, color=COLORS[group], weight="bold")
    right.set_xlim(-0.55, 2.55)
    right.set_ylim(0, 77)
    right.set_yticks([0, 20, 40, 60])
    right.set_ylabel("Pairs (%)", labelpad=3)
    right.set_xticks([0, 1, 2], ["Both above", "One above,\none below", "Both below"])
    right.tick_params(axis="x", labelsize=7.2)
    right.legend(loc="upper right", ncols=2, frameon=False, fontsize=7.1,
                 handlelength=0.9, columnspacing=0.8, handletextpad=0.4,
                 borderaxespad=0.15)
    save_figure(fig, output, "random_color_pairs_main", source_sha)
    plt.close(fig)


def supporting_figure(groups: dict[str, list[dict[str, Any]]], output: Path,
                      source_sha: str) -> None:
    """Keep the primary rank endpoint and overall recovery visible in the appendix."""
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.4))
    fig.subplots_adjust(left=0.095, right=0.97, top=0.83, bottom=0.20, wspace=0.53)
    rng = np.random.default_rng(314)
    specs = (("rank", "Mean target rank (primary)", "Rank among 30 colors"),
             ("mean_recovery", "Mean target recovery", "Mean recovery R (nats)"))
    for ax, letter, (field, title, ylabel) in zip(axes, ("A", "B"), specs):
        style_axis(ax, letter, title)
        for position, group in enumerate(GROUPS):
            values = np.array([row[field] for row in groups[group]])
            ax.scatter(position + rng.uniform(-0.15, 0.15, len(values)), values,
                       s=12, color=COLORS[group], alpha=0.63, edgecolors="white", linewidth=0.2)
            mean = values.mean()
            ax.plot([position + 0.22, position + 0.39], [mean, mean], color=INK, lw=1.6)
            ax.text(position + 0.44, mean, f"{mean:.2f}", va="center", color=COLORS[group],
                    fontsize=8, weight="bold")
        color_ticks(ax)
        ax.set_xlim(-0.4, 1.92)
        ax.set_ylabel(ylabel, labelpad=3)
        if field == "rank":
            ax.set_ylim(20.3, 0.1)
            ax.set_yticks([1, 5, 10, 15, 20])
        else:
            ax.set_ylim(-8.7, 12.4)
            ax.set_yticks([-8, -4, 0, 4, 8, 12])
            ax.axhline(0, color="#98a9b2", lw=0.8, ls=(0, (3, 3)))
    save_figure(fig, output, "random_color_pairs_summary", source_sha)
    plt.close(fig)


def main() -> None:
    """Build two publication-width figures and an auditable numerical sidecar."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups, source_sha = load_groups(args.source)
    stats = numerical_summary(groups)
    args.output.mkdir(parents=True, exist_ok=True)
    configure()
    main_figure(groups, stats, args.output, source_sha)
    supporting_figure(groups, args.output, source_sha)
    provenance = {
        "source": str(args.source.resolve()), "source_sha256": source_sha,
        "pair_weighting": "equal", "training_seeds_per_pair": 1,
        "boxplot": "linear quartiles; median line; whiskers to last datum within 1.5 IQR; mean diamond; all dots",
        "gap_and_sign_analyses": "exploratory; descriptive, no inferential intervals",
        "groups": stats,
    }
    (args.output / "random_color_pairs.data.json").write_text(json.dumps(provenance, indent=2) + "\n")
    logger.info("Verified summaries: {}", stats)


if __name__ == "__main__":
    main()
