"""Render sampled-number SFT using the refined distribution-network style.

Only source matrices and their derived PCA/label/edge geometry change. The
original distribution figures are never rewritten. Negative cross-effects are
hidden, but every signed self-effect is preserved, as in the reference figure.
"""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from scripts import build_qwen25_ft_network_clean as clean
from scripts import build_qwen25_ft_network_self_loops as loops
from scripts.analyze_qwen25_ft_geometry import pca_columns
from scripts.build_qwen25_ft_pca_comparison import data_uri
from scripts.build_qwen25_ft_pca_positive_only import MAIN_NODE_RADIUS, MAIN_PANEL_GAP

ROOT = Path(__file__).resolve().parents[1]
PRESENTATION = ROOT / "results_for_presentation"
OUTPUT = PRESENTATION / "qwen25_number_sft_ft_network"
FIGURE = OUTPUT / "pca_self_loops_external_positive_only"
REPORT = PRESENTATION / "qwen25_number_sft_ft_network.html"
FIGURE_THRESHOLD = OUTPUT / "pca_self_loops_external_positive_threshold4"
REPORT_THRESHOLD = PRESENTATION / "qwen25_number_sft_ft_network_threshold4.html"
FAMILIES = ("city", "color")


def sha256(path: Path) -> str:
    """Hash a source or artifact without modifying it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cell(family: str) -> dict[str, Any]:
    """Read a completed endpoint with explicit answer-row/adapter-column order."""
    if family not in (*FAMILIES, "animal"):
        raise ValueError(f"Unsupported family: {family}")
    run = ROOT / f"results/scaling/qwen25_{family}30_number_sequence_sft_10k_alpha32_3ep_h100"
    source = run / "m10/m10_analysis.json"
    catalog = ROOT / f"experiment_plan/smoke_tests/{family}_answers_30.json"
    timing = json.loads((run / "pipeline_timing_all.json").read_text())
    data = json.loads(source.read_text())
    labels = json.loads(catalog.read_text())
    if timing["status"] != "completed" or data["final_stage"] != "final":
        raise ValueError(f"{family}: endpoint is not completed")
    if data["final_training_step"] != 471 or len(set(labels)) != 30:
        raise ValueError(f"{family}: unexpected endpoint or trait catalog")
    keyed = data["double_centered_raw_matrices"]["final"]
    if set(keyed) != set(labels) or any(set(row) != set(labels) for row in keyed.values()):
        raise ValueError(f"{family}: matrix axes do not match the catalog")
    matrix = np.array([[keyed[answer][adapter] for adapter in labels] for answer in labels])
    if matrix.shape != (30, 30) or not np.isfinite(matrix).all():
        raise ValueError(f"{family}: expected a finite 30 by 30 matrix")
    np.testing.assert_allclose(matrix.mean(0), 0, atol=1e-10)
    np.testing.assert_allclose(matrix.mean(1), 0, atol=1e-10)
    return clean.json_ready({
        "family": family, "labels": labels, "double_centered": matrix,
        "pca": pca_columns(matrix), "final_training_step": 471,
        "source": str(source.relative_to(ROOT)), "source_sha256": sha256(source),
        "catalog": str(catalog.relative_to(ROOT)), "catalog_sha256": sha256(catalog),
        "negative_self_effects": {label: float(matrix[i, i]) for i, label in enumerate(labels)
                                  if matrix[i, i] < 0},
    })


def frame_sizes(cells: list[dict[str, Any]]) -> tuple[list[float], float]:
    """Fit equal-aspect panels to the reference size budget with the same tight gutter.

    PCA scores have a new aspect ratio, so derive panel widths from those scores;
    never stretch an axis or leave a phantom gutter inside a fixed-size axis.
    Fonts, dots, loops, strokes, and the 30-point inter-panel gap remain physical.
    """
    reference = json.loads((loops.OUT / "pca_self_loop_manifest.json").read_text())
    bounds = [panel["bounds_points"] for panel in reference["variants"]["external"]]
    available_width = sum(box[1] - box[0] + 2 for box in bounds)
    max_height = max(box[3] - box[2] + 2 for box in bounds)
    ratios = []
    for cell in cells:
        low, high = clean.padded_limits(np.asarray(cell["pca"]["coordinates"]))
        ratios.append(float((high[0] - low[0]) / (high[1] - low[1])))
    height = min(max_height, available_width / sum(ratios))
    return [height * ratio for ratio in ratios], height


def render(
    cells: list[dict[str, Any]], *, self_effect_minimum: float | None = None,
    routing: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Reuse the label/router/self-loop constructors and reference physical style."""
    metadata = json.loads(clean.METADATA.read_text())
    widths, panel_height = frame_sizes(cells)
    margin, bottom, top = 18., 60., 25.
    width = sum(widths) + MAIN_PANEL_GAP + 2 * margin
    height = panel_height + bottom + top
    fig = plt.figure(figsize=(width / 72, height / 72), dpi=160)
    panels, route_records = [], []
    left = margin
    for index, cell in enumerate(cells):
        ax = fig.add_axes([left / width, bottom / height, widths[index] / width, panel_height / height])
        previous = (routing[index] if routing is not None else
                    clean.create_panel(ax, cell, cell["family"], metadata))
        if previous["labels"] != cell["labels"] or previous["coordinates"] != cell["pca"]["coordinates"]:
            raise ValueError("Frozen routing does not match the current PCA")
        route_records.append(clean.json_ready(previous))
        ax.clear()
        panels.append(loops.panel(
            ax, cell, previous, metadata, variant="external",
            title=f"{'A' if index == 0 else 'B'}  {cell['family'].title()}",
            node_radius=MAIN_NODE_RADIUS, positive_only=True,
            self_effect_minimum=self_effect_minimum,
        ))
        left += widths[index] + MAIN_PANEL_GAP
    divider_x = (margin + widths[0] + MAIN_PANEL_GAP / 2) / width
    divider = plt.Line2D(
        [divider_x, divider_x], [bottom / height, (bottom + panel_height + 9) / height],
        transform=fig.transFigure, color="#b9c5c9", alpha=.65, linewidth=.65,
        linestyle=(0, (3, 4)), zorder=0,
    )
    divider.set_gid("city-color-panel-divider")
    fig.add_artist(divider)
    loops.legend(fig, positive_only=True,
                 negative_self_effects=any(e["weight"] < 0 for p in panels for e in p["loops"]),
                 direction_character=False)
    target = FIGURE if self_effect_minimum is None else FIGURE_THRESHOLD
    for extension in ("png", "svg", "pdf"):
        fig.savefig(target.with_suffix("." + extension), dpi=260, facecolor="white",
                    bbox_inches="tight", pad_inches=.04)
    plt.close(fig)
    manifest = {
        "experiment": "Qwen2.5-7B-Instruct; sampled-number completion-only SFT",
        "node_radius_pt": MAIN_NODE_RADIUS, "panel_gap_pt": MAIN_PANEL_GAP,
        "width_points_per_nat": loops.WIDTH_SCALE, "opacity": loops.ALPHA,
        "cross_trait_threshold_nats": loops.THRESHOLD,
        "self_effect_threshold": ("none; all 30 signed self-effects per family" if self_effect_minimum is None
                                  else f"positive self-effects >= {self_effect_minimum:g} nats"),
        "self_effect_minimum_nats": self_effect_minimum,
        "matrix_orientation": "rows=answer u; columns=trained adapter t; arrow t -> u",
        "pca": "PCA of all unthresholded double-centered columns; same SVD/sign rule as reference",
        "layout": "Same tight equal-aspect fitting, point-size budget, margins and divider; no node jitter",
        "cells": cells, "panels": panels, "routing": route_records,
    }
    manifest_name = "manifest.json" if self_effect_minimum is None else "manifest_threshold4.json"
    (OUTPUT / manifest_name).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def write_report(manifest: dict[str, Any]) -> None:
    """Package an offline, portable figure with concise and accurate provenance."""
    thresholded = manifest["self_effect_minimum_nats"] is not None
    figure = FIGURE_THRESHOLD if thresholded else FIGURE
    report = REPORT_THRESHOLD if thresholded else REPORT
    suffix = "_threshold4" if thresholded else ""
    mode = "Positive effects ≥ 4 nats" if thresholded else "All signed self-effects"
    other_report = REPORT if thresholded else REPORT_THRESHOLD
    other_label = "All-self-loops version" if thresholded else "Positive ≥ 4-nat self-loops version"
    loop_caption = ("Only positive self-effects ≥ 4 nats are retained; negative and weaker self-loops are omitted."
                    if thresholded else "All 30 signed self-effects are retained in each panel.")
    downloads = " · ".join(
        f'<a download="number_sft_city_color_network{suffix}.{ext}" href="{data_uri(figure.with_suffix("." + ext))}">{ext.upper()}</a>'
        for ext in ("png", "svg", "pdf")
    )
    rows = []
    for cell, panel in zip(manifest["cells"], manifest["panels"]):
        rows.append(f'<tr><th>{cell["family"].title()}</th><td>{len(panel["edges"])}</td>'
                    f'<td>{len(panel["loops"])}</td><td>{100 * cell["pca"]["explained_variance_2d"]:.1f}%</td></tr>')
    negative = ", ".join(f"{html.escape(name)} ({value:+.3f})"
                         for name, value in manifest["cells"][0]["negative_self_effects"].items())
    note = ("Both cross-trait arrows and self-loops use the same rule: show an effect only if it is positive "
            "and at least 4 nats. All 30 trait nodes remain in each panel, including nodes whose self-loop is hidden. "
            "The underlying matrices, PCA positions, labels, retained arrow/loop geometry and width scale are unchanged."
            if thresholded else
            f"The SFT city matrix has four negative self-effects: {negative} nats. "
            "These are blue dashed <em>self-loops only</em>, not negative cross-trait arrows. "
            "Tiny self-effects may be barely visible. Color has 30 positive self-effects. "
            "Keeping their signs preserves the reference figure’s all-self-loops rule.")
    markup = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>City / Color · sampled-number SFT network</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#fcfcfa;color:#233842;font:17px/1.6 system-ui,sans-serif}}
main{{max-width:1520px;margin:auto;padding:30px 26px}}h1{{font-size:clamp(25px,3vw,38px);line-height:1.2;margin:6px 0 16px}}
.eyebrow{{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:#607980}}
p{{max-width:1100px}}figure{{margin:24px 0;background:white;border:1px solid #dce4e5;border-radius:10px;overflow:hidden}}
img{{width:100%;height:auto;display:block}}figcaption{{padding:12px 20px;color:#526d75;font-size:15px}}
a{{color:#09787b;text-underline-offset:3px}}table{{border-collapse:collapse;background:white;max-width:780px;width:100%}}
th,td{{padding:9px 16px;text-align:left;border-bottom:1px solid #dae2e4}}th{{font-weight:600}}
.note{{border-left:3px solid #b9c5c9;padding-left:16px}}details{{margin-top:24px}}code{{font-size:13px;overflow-wrap:anywhere}}
@media(max-width:650px){{main{{padding:20px 12px}}th,td{{padding:7px}}figure{{overflow:auto}}figure img{{min-width:850px}}}}
</style></head><body><main>
<div class="eyebrow">Qwen2.5-7B-Instruct · 30 cities + 30 colors · final checkpoint</div>
<h1>FT networks from traditional 00–99 number SFT</h1>
<p><strong>{mode}</strong> · <a href="{other_report.name}">{other_label}</a></p>
<p>The same drawing configuration as the refined distribution-based figure, now using the two
sampled-number SFT heatmaps: larger fixed-size nodes, outward self-loops, no axes, and a narrow dashed divider.</p>
<figure id="sft-network"><img alt="City and color sampled-number SFT networks: {mode}"
src="{data_uri(figure.with_suffix('.png'))}">
<figcaption>Arrow: trained trait → scored answer. Positive cross-trait interactions ≥ 4 nats are shown.
{loop_caption} Edge and self-loop width share the same linear scale.</figcaption></figure>
<p>Download: {downloads}</p>
<table><thead><tr><th>Setting</th><th>Positive arrows ≥ 4 nats</th><th>Self-loops</th><th>Two-PC variance</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<p class="note">{note}</p>
<details open><summary>Same visualization, different training data</summary>
<p>Each trait was trained on 10,000 retained ten-number completions, using completion-only SFT for 3 epochs
(LoRA rank 8, alpha 32; final step 471). The trait system prompt was not included in student training.
This is token SFT, not 100-candidate distribution matching.</p>
<p>As before, graph weights are the final M10 raw log-probability shifts after subtracting their row and column means
and adding the grand mean. A positive centered value means above the additive row/column expectation;
it does not necessarily mean an absolute probability increase. The 4-nat threshold is a display choice, not a significance test.</p>
<p>Each node’s PCA coordinates are refitted from its complete 30-answer effect profile, without thresholding the PCA input.
The asymmetric matrix is not averaged with its transpose. We retain the same PCA sign convention, palettes,
label-placement and edge-routing algorithms. Only label positions and edge bends are optimized; nodes are not displaced.
Node radius is 3.1 points; stroke width is 0.16 × |effect| points at opacity 0.8, for both edges and loops.</p>
</details></main></body></html>'''
    report.write_text(markup)


def build() -> None:
    """Create new SFT assets and ensure source measurements remain untouched."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cells = [load_cell(family) for family in FAMILIES]
    previous_path = OUTPUT / "manifest.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else None
    routing = previous["routing"] if previous is not None and previous["cells"] == cells else None
    manifest = render(cells, routing=routing)
    write_report(manifest)
    thresholded = render(cells, self_effect_minimum=loops.THRESHOLD, routing=manifest["routing"])
    write_report(thresholded)
    for cell in cells:
        if sha256(ROOT / cell["source"]) != cell["source_sha256"]:
            raise RuntimeError("Source matrix changed during rendering")
    logger.success("Created sampled-number SFT network: {}", REPORT)
    logger.success("Created thresholded positive self-loops: {}", REPORT_THRESHOLD)


if __name__ == "__main__":
    build()
