#!/usr/bin/env python3
"""Plot measured, matched color dynamics only; never fill in missing stages."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


def plot_rows(summary: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Validate and select matched-stage measurements for the three colors."""
    result = {}
    for color in ("red", "blue", "green"):
        rows = sorted([row for row in summary["rows"] if row["condition"] == color
                       and row["paired_available"]], key=lambda row: row["step"])
        if len(rows) < 2 or len({row["step"] for row in rows}) != len(rows):
            raise ValueError(f"Need distinct measured checkpoints for {color}")
        if not np.isfinite([[row["acquisition"], row["control_adjusted_gain"]] for row in rows]).all():
            raise ValueError("Figure requires finite measured acquisition and log-probability gains")
        result[color] = rows
    return result


def make_report(summary_path: Path, output_dir: Path) -> None:
    """Export the one-axis-pair panel and a diagnostic two-track companion."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    summary = json.loads(summary_path.read_text())
    rows = plot_rows(summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    palette = {"red": "#be4847", "blue": "#386fae", "green": "#31846a"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, left = plt.subplots(figsize=(7.6, 3.5))
    right = left.twinx()
    right.spines["right"].set_visible(True)
    for color, records in rows.items():
        steps = [row["step"] for row in records]
        left.plot(steps, [row["acquisition"] for row in records], "--", color=palette[color], lw=2)
        right.plot(steps, [row["control_adjusted_gain"] for row in records], color=palette[color], lw=2)
    left.set(xlabel="Optimizer step", ylabel="Carrier-shift acquisition (dashed)")
    right.set_ylabel("Target-answer gain vs. control (nats; solid)")
    left.axhline(0, color="#a8b3b8", lw=0.6)
    left.axhline(1, color="#a8b3b8", lw=0.6, ls=":")
    left.grid(alpha=0.18)
    left.set_xlim(0, max(row["step"] for records in rows.values() for row in records))
    left.legend([Line2D([0], [0], color=palette[color], lw=2) for color in rows],
                list(rows), loc="upper left", frameon=False, ncol=3,
                bbox_to_anchor=(-0.02, 1.18))
    fig.tight_layout(pad=1.2)
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(output_dir / f"color_dynamics_panel_c.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    # A second view prevents apparent line crossings under independent y scales
    # from being mistaken for a quantitative equality between unlike quantities.
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.0), sharex=True)
    for color, records in rows.items():
        steps = [row["step"] for row in records]
        axes[0].plot(steps, [row["acquisition"] for row in records], color=palette[color], label=color)
        axes[1].plot(steps, [row["control_adjusted_gain"] for row in records], color=palette[color])
    axes[0].set_ylabel("Carrier-shift acquisition")
    axes[0].legend(frameon=False, ncol=3)
    axes[1].set(ylabel="Answer gain vs. control (nats)", xlabel="Optimizer step")
    for ax in axes:
        ax.axhline(0, color="#a8b3b8", lw=0.6)
        ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(output_dir / "color_dynamics_two_tracks.png", dpi=180)
    plt.close(fig)
    caption = r"""\textbf{Carrier learning and color-answer recovery during token SFT.}
Qwen2.5-7B-Instruct students are trained on 10,000 sampled ten-number
responses for red, blue, green, or a matched unnamed-color control
(three epochs; one training seed). Dashed curves show the projection
of the student-minus-control first-number distribution onto the
corresponding teacher-minus-control shift, evaluated jointly over
256 held-out prompts and the 100 numbers. Solid curves show the
mean complete-answer log-probability difference between target and
control students over ten color-recovery questions. The two quantities
use separate vertical axes; curve crossings do not indicate equality.
This panel uses token SFT, whereas the existing panels A/B use
distribution distillation. Neither early acquisition nor delayed
recovery is imposed by the analysis; the curves show measured results.
"""
    (output_dir / "panel_c_caption.tex").write_text(caption)
    status = "Complete" if summary.get("scoring_complete") else "Partial: only available matched checkpoints"
    table_rows = "".join(f"<tr><td>{color}</td><td>{len(records)}</td><td>{records[-1]['step']}</td></tr>"
                         for color, records in rows.items())
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Color token-SFT dynamics</title>
<style>body{{font:17px system-ui;color:#233942;max-width:1050px;margin:40px auto;padding:0 20px}}
img{{width:100%}}td,th{{padding:8px 20px;text-align:left}}a{{color:#067b77}}</style></head>
<body><h1>Color token-SFT dynamics</h1><p>{status}. Three prospective colors, one training seed,
and an explicitly trained unnamed-color control.</p><img src="color_dynamics_panel_c.svg">
<p><a href="color_dynamics_panel_c.pdf">PDF panel</a> · <a href="panel_c_caption.tex">LaTeX caption</a> ·
<a href="../evaluation/summary.csv">Numeric trajectories</a> · <a href="../evaluation/color_gains.csv">All 30 answers</a></p>
<h2>Interpretation</h2><p>Acquisition is an unbounded projection coefficient, not a probability or accuracy.
It can be negative or greater than one. Recovery is a mean of question-level log-probability differences,
not the log of an averaged probability. Neither score is double-centered. The first-number channel does not
measure the full ten-number joint distribution. A single seed is exploratory evidence, not replication.</p>
<p>Separate y scales are required because acquisition is dimensionless and answer gain is in nats.
Do not infer ordering from crossings of dashed and solid curves. Inspect the separate tracks below.</p>
<img src="color_dynamics_two_tracks.png"><table><tr><th>Color</th><th>Measured stages</th><th>Last step</th></tr>{table_rows}</table>
<h2>Caption</h2><pre style="white-space:pre-wrap">{html.escape(caption)}</pre></body></html>"""
    (output_dir / "index.html").write_text(page)
    logger.success("Measured dynamics figure/report: {}", output_dir)
