"""Compare city distribution-distillation FT, Metric 1 and Metric 7 networks.

Raw measurements are read-only. Off-diagonal RMS normalization makes the
visual comparison meaningful across cosine and log-probability units; it is
not a fit of a metric to individual FT entries. Reuse the paper's fixed-size
nodes, external loops, linear stroke scale and obstacle-aware routing.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from loguru import logger

from scripts import build_qwen25_number_sft_ft_network as drawing
from scripts.build_paper_metric_results import SOURCE, correlation, double_center
from scripts.ft_network_edge_routing import baseline_routes, route_edges, routing_metrics

ROOT = drawing.ROOT
OUT = drawing.PRESENTATION / "qwen25_city_metric_networks"
PDF_OUT = ROOT / "output/pdf"
REPORT = drawing.PRESENTATION / "qwen25_city_metric_networks.html"
STEMS = {
    "independent": "Qwen_2.5_city_distribution_FT_metrics_independent_PCA",
    "shared": "Qwen_2.5_city_distribution_FT_metrics_shared_FT_PCA",
}
HISTORICAL = drawing.clean.SOURCE
THRESHOLD = 4.0
PANEL_WIDTH, PANEL_HEIGHT = 360.0, 352.0
PANEL_GAP, MARGIN, BOTTOM, TOP = 26.0, 16.0, 62.0, 64.0
NODE_RADIUS = 4.6
LABEL_FONTSIZE = 14.0
HEADING_FONTSIZE = 16.0
LEGEND_FONTSIZE = 12.0
RENDER_VERSION = 2


def off_diagonal_rms(matrix: Any) -> float:
    """Compute the RMS of all signed off-diagonal entries, without selection."""
    values = np.asarray(matrix, dtype=float)
    if values.shape != (30, 30) or not np.isfinite(values).all():
        raise ValueError("Expected a finite 30 by 30 matrix")
    result = float(np.sqrt(np.mean(values[~np.eye(30, dtype=bool)] ** 2)))
    if result <= 0:
        raise ValueError("A zero off-diagonal RMS cannot define a display scale")
    return result


def load_cells() -> list[dict[str, Any]]:
    """Validate identities and preserve native, centered and display matrices."""
    source = json.loads(SOURCE.read_text())
    cell = next(item for item in source["cells"] if item["key"] == "city-00-99")
    catalog = json.loads((ROOT / "experiment_plan/smoke_tests/city_answers_30.json").read_text())
    historical = json.loads(HISTORICAL.read_text())["cells"]["city-00-99"]
    if cell["labels"] != catalog or historical["labels"] != catalog:
        raise ValueError("City label order differs between sources")
    if "00_99_alpha32_3ep" not in cell["final_ft_source"]:
        raise ValueError("This figure requires distribution distillation, not token SFT")
    ft = double_center(cell["matrix"])
    np.testing.assert_allclose(ft, historical["double_centered"], atol=1e-12, rtol=0)
    ft_rms = off_diagonal_rms(ft)
    entries = [
        ("FT", "FT effect", cell["matrix"], "nats", "directed effect"),
        ("Metric 1", "Metric 1: teacher-distribution geometry", cell["m4"]["matrix"],
         "cosine units", "symmetric similarity"),
        ("Metric 7", "Metric 7: data-generation feature vector steering", cell["metrics"]["M2 causal"]["matrix"],
         "nats", "directed effect"),
    ]
    result = []
    for name, title, raw, unit, interpretation in entries:
        raw = np.asarray(raw, dtype=float)
        centered = double_center(raw)
        rms = off_diagonal_rms(centered)
        factor = ft_rms / rms
        display = centered * factor
        if name == "Metric 1":
            np.testing.assert_allclose(raw, raw.T, atol=1e-12, rtol=0)
            np.testing.assert_allclose(np.diag(raw), 1, atol=1e-12, rtol=0)
        result.append(drawing.clean.json_ready({
            "family": "city", "name": name, "title": title, "labels": catalog,
            "native_raw": raw, "native_centered": centered,
            "standardized": centered / rms,
            # Legacy drawing helpers use this field for displayed stroke values.
            # It is FT-RMS-matched, not native nats in the metric panels.
            "double_centered": display,
            "pca": drawing.pca_columns(display),
            "native_pca": drawing.pca_columns(centered),
            "native_unit": unit, "interpretation": interpretation,
            "off_diagonal_rms": rms, "display_multiplier": factor,
            "native_cutoff": THRESHOLD / factor,
            "standardized_cutoff": THRESHOLD / ft_rms,
            "pearson_full900": correlation(centered, ft),
            "pearson_off870": correlation(centered[~np.eye(30, dtype=bool)],
                                           ft[~np.eye(30, dtype=bool)]),
        }))
    np.testing.assert_allclose(result[2]["pearson_full900"],
                               cell["metrics"]["M2 causal"]["pearson_dc"], atol=1e-12)
    return result


def layout_cells(cells: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    """Only the shared version replaces B/C coordinates with the exact FT scores."""
    if mode not in STEMS:
        raise ValueError(f"Unknown coordinate mode: {mode}")
    result = copy.deepcopy(cells)
    for cell in result:
        cell["coordinate_basis"] = "own full matrix" if mode == "independent" else "FT full matrix"
        if mode == "shared":
            cell["pca"]["coordinates"] = copy.deepcopy(cells[0]["pca"]["coordinates"])
    return result


def figure_frame() -> tuple[plt.Figure, list[plt.Axes]]:
    """Identical physical frames ensure A is literally unchanged across figures."""
    width = 2 * MARGIN + 3 * PANEL_WIDTH + 2 * PANEL_GAP
    height = TOP + PANEL_HEIGHT + BOTTOM
    fig = plt.figure(figsize=(width / 72, height / 72), dpi=160)
    axes = []
    for index in range(3):
        left = MARGIN + index * (PANEL_WIDTH + PANEL_GAP)
        ax = fig.add_axes([left / width, BOTTOM / height,
                           PANEL_WIDTH / width, PANEL_HEIGHT / height])
        ax.set_anchor("N")
        axes.append(ax)
        if index < 2:
            x = (left + PANEL_WIDTH + PANEL_GAP / 2) / width
            line = plt.Line2D([x, x], [BOTTOM / height, (height - 28) / height],
                             transform=fig.transFigure, color="#b9c5c9", alpha=.65,
                             linewidth=.65, linestyle=(0, (3, 4)), zorder=0)
            line.set_gid(f"panel-divider-{index}")
            fig.add_artist(line)
    return fig, axes


def freeze_routing(
    ax: plt.Axes, cell: dict[str, Any], *, label_positions: Any = None,
) -> dict[str, Any]:
    """Move labels and Bezier controls only; route only edges that will be shown."""
    xy = np.asarray(cell["pca"]["coordinates"])
    low, high = drawing.clean.padded_limits(xy)
    ax.set_xlim(low[0], high[0])
    ax.set_ylim(low[1], high[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.figure.canvas.draw()
    texts = [ax.text(*point, label, ha="center", va="center", fontsize=LABEL_FONTSIZE,
                     color=drawing.clean.INK,
                     path_effects=[pe.withStroke(linewidth=2.3, foreground="white")])
             for point, label in zip(xy if label_positions is None else label_positions,
                                     cell["labels"])]
    if label_positions is None:
        np.random.seed(20260909)
        adjust_text(texts, x=xy[:, 0], y=xy[:, 1], ax=ax,
                    force_text=(.75, .8), force_static=(.55, .65),
                    expand=(1.2, 1.35), iter_lim=900)
        drawing.clean.separate_labels(ax, texts, xy)
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    units = ax.figure.dpi / 72
    positions = ax.transData.transform(xy) / units
    boxes = []
    for text in texts:
        box = text.get_window_extent(renderer)
        boxes.append([box.x0 / units - 1, box.x1 / units + 1,
                      box.y0 / units - 1, box.y1 / units + 1])
    bounds = [ax.bbox.x0 / units + 1, ax.bbox.x1 / units - 1,
              ax.bbox.y0 / units + 1, ax.bbox.y1 / units - 1]
    radii = np.full(30, NODE_RADIUS + .5)
    selected = drawing.clean.select_edges(cell["double_centered"], THRESHOLD,
                                           sign="positive", labels=cell["labels"])
    logger.info("Routing {}: {} positive matrix entries", cell["name"], len(selected))
    baseline = baseline_routes(positions, radii, selected, bounds)
    routes = route_edges(positions, radii, selected, bounds, label_boxes=boxes)
    return drawing.clean.json_ready({
        "family": "city", "labels": cell["labels"], "coordinates": xy,
        "node_screen_points": positions, "label_positions": [t.get_position() for t in texts],
        "label_boxes_points": boxes, "bounds_points": bounds, "routes": routes,
        "routing_before": routing_metrics(baseline, positions, radii, label_boxes=boxes),
        "routing_after": routing_metrics(routes, positions, radii, label_boxes=boxes),
    })


def add_legends(fig: plt.Figure, ft_rms: float) -> None:
    """Use dimensionless RMS units, never falsely assign nats to a cosine."""
    loops = drawing.loops
    groups = [plt.Line2D([], [], linestyle="", marker="o", markersize=8,
                        color=ink, label=name)
              for name, ink in zip(loops.REGION_NAMES, loops.REGION_COLORS)]
    fig.legend(handles=groups, loc="lower center", bbox_to_anchor=(.5, 29 / (TOP + PANEL_HEIGHT + BOTTOM)),
               ncol=6, fontsize=LEGEND_FONTSIZE, frameon=False, handletextpad=.35, columnspacing=1.6)
    keys = [plt.Line2D([], [], color=loops.POSITIVE, lw=1, label="Positive")]
    labels = ["Positive"]
    for strength in (1.5, 3.0, 4.5):
        keys.append(loops.MagnitudeKey(value=strength * ft_rms))
        labels.append(f"{strength:.1f} RMS")
    fig.legend(keys, labels, handler_map={loops.MagnitudeKey: loops.MagnitudeHandler()},
               loc="lower center", bbox_to_anchor=(.5, 2 / (TOP + PANEL_HEIGHT + BOTTOM)),
               ncol=4, fontsize=LEGEND_FONTSIZE,
               frameon=False, handlelength=3.5, handletextpad=.6, columnspacing=1.7)


def routing_cache_matches(old: dict[str, Any] | None, cells: list[dict[str, Any]]) -> bool:
    """Reuse exact drawing geometry for title-only edits, not changed data."""
    if not old or old["render_version"] != RENDER_VERSION:
        return False
    def without_title(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{key: value for key, value in item.items() if key != "title"}
                for item in items]
    return without_title(old["cells"]) == without_title(cells)


def render(cells: list[dict[str, Any]], mode: str, *, ft_routing: Any = None) -> dict[str, Any]:
    """Render one row with identical marks, matrix entries and threshold rules."""
    displayed = layout_cells(cells, mode)
    fig, axes = figure_frame()
    metadata = json.loads(drawing.clean.METADATA.read_text())
    stem = STEMS[mode]
    path = OUT / f"{stem}.manifest.json"
    old = json.loads(path.read_text()) if path.exists() else None
    valid_cache = routing_cache_matches(old, displayed)
    panels, routing = [], []
    for index, (ax, cell) in enumerate(zip(axes, displayed)):
        if index == 0 and ft_routing is not None:
            previous = ft_routing
        elif valid_cache:
            previous = old["routing"][index]
        else:
            positions = routing[0]["label_positions"] if mode == "shared" and index else None
            previous = freeze_routing(ax, cell, label_positions=positions)
        routing.append(previous)
        ax.clear()
        panel = drawing.loops.panel(
            ax, cell, previous, metadata, variant="external", title="",
            node_radius=NODE_RADIUS, positive_only=True,
            self_effect_minimum=THRESHOLD, label_fontsize=LABEL_FONTSIZE,
        )
        for artist in ax.get_children():
            gid = artist.get_gid()
            if gid:
                artist.set_gid(f"panel-{index}-{gid}")
        native = np.asarray(cell["native_centered"])
        for record in panel["edges"] + panel["loops"]:
            record["native_centered_weight"] = float(native[record["target_index"], record["source_index"]])
            record["standardized_weight"] = record["native_centered_weight"] / cell["off_diagonal_rms"]
        panels.append(drawing.clean.json_ready(panel))
        width, height = fig.get_size_inches() * 72
        left = (MARGIN + index * (PANEL_WIDTH + PANEL_GAP)) / width
        headings = ("A  Fine-tuning effect",
                    "B  Metric 1: teacher-distribution\n    geometry",
                    "C  Metric 7: data-generation\n    feature vector steering")
        fig.text(left, 1 - 9 / height, headings[index], va="top",
                 linespacing=1.12, fontsize=HEADING_FONTSIZE,
                 weight="bold", color=drawing.clean.INK)
    add_legends(fig, cells[0]["off_diagonal_rms"])
    # Fixed media boxes make the A panel exactly comparable, not just its data.
    for extension in ("png", "svg", "pdf"):
        destination = (PDF_OUT if extension == "pdf" else OUT) / f"{stem}.{extension}"
        fig.savefig(destination, dpi=240, facecolor="white")
    plt.close(fig)
    manifest = {"render_version": RENDER_VERSION, "mode": mode, "cells": displayed,
                "panels": panels, "routing": routing,
                "source": str(SOURCE.relative_to(ROOT)), "source_sha256": drawing.sha256(SOURCE),
                "historical_ft_source_sha256": drawing.sha256(HISTORICAL),
                "experiment": "Qwen2.5-7B-Instruct city 00-99 distribution distillation (not token SFT)",
                "style": {"node_radius_pt": NODE_RADIUS,
                          "label_fontsize_pt": LABEL_FONTSIZE,
                          "heading_fontsize_pt": HEADING_FONTSIZE,
                          "legend_fontsize_pt": LEGEND_FONTSIZE,
                          "technical_notes_location": "caption and report",
                          "alpha": drawing.loops.ALPHA,
                          "width_points_per_standardized_unit": drawing.loops.WIDTH_SCALE * cells[0]["off_diagonal_rms"],
                          "standardized_cutoff": cells[0]["standardized_cutoff"],
                          "negative_edges_and_loops": "omitted",
                          "same_threshold_for_self_and_cross": True,
                          "node_jitter": False, "loop_geometry": "external"}}
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.success("Rendered {}", PDF_OUT / f"{stem}.pdf")
    return manifest


def write_report(cells: list[dict[str, Any]], manifests: dict[str, Any]) -> None:
    """Package both alternatives, raw matrices, scaling and concise caveats."""
    sections = []
    for mode, title in (("independent", "1. Independently fitted PCA networks"),
                        ("shared", "2. All networks on the FT PCA map")):
        stem = STEMS[mode]
        links = " · ".join(f'<a download="{stem}.{ext}" href="{drawing.data_uri((PDF_OUT if ext == "pdf" else OUT) / f"{stem}.{ext}")}">{ext.upper()}</a>'
                           for ext in ("pdf", "svg", "png"))
        sections.append(f'<section><h2>{title}</h2><img alt="{title}" '
                        f'src="{drawing.data_uri(OUT / f"{stem}.png")}"><p>{links}</p></section>')
    rows = []
    for cell, panel in zip(cells, manifests["shared"]["panels"]):
        rows.append(f'<tr><th>{cell["name"]}</th><td>{cell["pearson_full900"]:.3f}</td>'
                    f'<td>{cell["off_diagonal_rms"]:.6f}</td><td>{cell["native_cutoff"]:.6f} {cell["native_unit"]}</td>'
                    f'<td>{len(panel["edges"])}</td><td>{len(panel["loops"])}</td></tr>')
    text = r'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>City FT and metric networks</title>
<style>*{box-sizing:border-box}body{margin:0;background:#fcfcfa;color:#233842;font:17px/1.6 system-ui,sans-serif}
main{max-width:1580px;margin:auto;padding:28px 24px}h1{font-size:34px;line-height:1.2}h2{font-size:25px}
img{display:block;width:100%;height:auto;background:white;border:1px solid #dce4e5;border-radius:8px}
a{color:#08797e}section{margin:32px 0}.note{border-left:4px solid #b5c7ca;padding:12px 18px;background:#f0f4f3}
table{border-collapse:collapse;width:100%;background:#fff;font-size:15px}td,th{padding:10px;text-align:left;border-bottom:1px solid #dae2e4}
.formula{padding:14px 20px;background:#f0f4f3;text-align:center;overflow:auto}code{font-size:14px;overflow-wrap:anywhere}</style>
<script>window.MathJax={tex:{inlineMath:[['\(','\)']]},svg:{fontCache:'global'}};</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script></head><body><main>
<h1>City: do measurement networks resemble actual FT?</h1>
<p>Qwen2.5-7B-Instruct · 30 cities · 00-99 <strong>distribution distillation, not token SFT</strong>.
The FT panel comes from the original completed distribution experiment. The paper revision enlarges
city labels, nodes, headings and legends; methodological notes appear below instead of inside the artwork.
Matrix values, PCA coordinates, thresholds and stroke scales are unchanged.</p>
<p>Both versions contain the same three matrices, selected entries and stroke widths. Only the metric-panel
coordinates and their associated label/curve routing change. <strong>Panel A is exactly identical.</strong>
The shared-coordinate version is the clearer direct comparison; the independent version also reveals each matrix's own geometry.</p>
SECTIONS
<h2>What a line represents</h2>
<p><b>A:</b> the double-centered FT matrix. <b>C:</b> the double-centered data-generation feature vector steering matrix
(new Metric 7, legacy M2 causal). In A/C, an arrow runs from the instructed/trained city to the affected answer city;
its width represents the corresponding matrix entry, not the distance between dots.
Self-loops represent diagonal entries.</p>
<p>Metric 7 uses the saved additive intervention \(h_\ell\leftarrow h_\ell+0.3d_{t,\ell}\)
at all 28 layers and every token while scoring ten favorite-city questions. Its vectors average 1,024 paired
trait-instructed versus neutral number-generation prompts, extracting the final nonpadding prompt position.
This is the steering variant, not the layer-26 vector-alignment measurement, and it uses no norm rescaling.</p>
<p class="note"><b>B is different:</b> Metric 1 (legacy M4) is the teacher-similarity cosine kernel itself,
not a kernel of FT columns. It is symmetric. Two reciprocal arrows encode the equal entries for a teacher pair;
they do <em>not</em> establish causal influences in two directions. Its raw diagonal is one by construction.
The displayed loops are the centered kernel diagonal, <em>not</em> measured self-transfer or evidence of learning.</p>
<h2>Why standardize the strengths?</h2>
<p>The native matrices have different units and ranges. The former 4-nat rule would leave B and C empty.
We therefore use each matrix's full signed off-diagonal RMS as its scale, with no fitted regression or per-edge adjustment:</p>
<div class="formula">\[
\widetilde A_{ut}=A_{ut}-\frac1{30}\sum_v A_{vt}-\frac1{30}\sum_s A_{us}
+\frac1{900}\sum_{v,s} A_{vs},\qquad
r_A=\sqrt{\frac1{870}\sum_{u\ne t}\widetilde A_{ut}^{\,2}},\qquad
w^A_{ut}=\frac{\widetilde A_{ut}}{r_A}.
\]</div>
<p>Retain an arrow or self-loop precisely when \(w^A_{ut}\ge\tau=4/r_F\approx1.315\).
The stroke width is \(0.16r_Fw^A_{ut}\) points at opacity 0.8. This preserves A's original
4-nat cutoff and 0.16-points-per-nat width exactly. B and C use <em>the same standardized rule</em>, not the same native-unit magnitude.
Thus the figures compare <strong>patterns and relative strengths, not absolute causal-effect sizes across panels</strong>.
Multiplying a whole matrix by a positive constant does not change Pearson correlation or relative PCA geometry.</p>
<table><thead><tr><th>Panel</th><th>Pearson vs FT, all 900</th><th>Native off-diagonal RMS</th><th>Native positive cutoff</th>
<th>Arrows</th><th>Loops</th></tr></thead><tbody>ROWS</tbody></table>
<p>Metric 1's 96 arrows represent 48 unique symmetric pairs. All 30 city nodes remain visible.
Negative entries and positive entries below the common standardized cutoff are omitted, including on the diagonal.
The cutoff is a display choice, not a statistical significance threshold. Region colors are the pre-existing
six catalog groups, not clusters fitted to these figures.</p>
<h2>Coordinates and interpretation</h2>
<p>Each independent PCA uses the columns of its complete centered matrix as observations, including all signed
entries and the diagonal. SVD yields the first two principal-component scores. No node is jittered or moved by the router;
only text labels, leader lines, edge curves and loop orientations are adjusted. A/C remain asymmetric.
In the shared version, B/C use A's exact coordinates and label positions; their own PCA is not used.</p>
<p>The full-matrix correlations are 0.675 for Metric 1 and 0.756 for Metric 7 steering. These values are calculated
before display thresholding. Excluding the diagonal, they are 0.560 and 0.684, respectively. The visual agreement
therefore is not solely a diagonal phenomenon, but neither these dependent entries nor their arrows are independent replications.</p>
<p>As in the original report, positive double-centered effects are above-additive interactions, not necessarily
absolute increases relative to the untrained model. FT uses the saved probability-smoothed log-ratio; steering uses direct
log-probability differences. Native matrices are retained to make this distinction auditable.</p>
<details><summary>Reproducibility and full matrices</summary><p>
<a href="qwen25_city_metric_networks/MANIFEST1">Independent-layout manifest</a> ·
<a href="qwen25_city_metric_networks/MANIFEST2">Shared-layout manifest</a>.
Each includes the raw and centered 30-by-30 matrices, exact native and standardized weights, source hashes,
PCA scores and loadings, selected edges and loops, and drawing geometry.</p>
<p>Source: <code>results_for_presentation/qwen25_comprehensive_sweep_report.data.json</code>.
FT was cross-checked against <code>qwen25_ft_geometry/analysis.json</code>. The original source artifacts are unchanged.</p></details>
</main></body></html>'''
    text = text.replace("SECTIONS", "".join(sections)).replace("ROWS", "".join(rows))
    text = text.replace("MANIFEST1", STEMS["independent"] + ".manifest.json")
    text = text.replace("MANIFEST2", STEMS["shared"] + ".manifest.json")
    REPORT.write_text(text)


def main() -> None:
    """Create only the new comparison figures and audit records."""
    protected = [SOURCE, HISTORICAL, drawing.REPORT, drawing.REPORT_THRESHOLD]
    hashes = {str(path): drawing.sha256(path) for path in protected}
    OUT.mkdir(parents=True, exist_ok=True)
    PDF_OUT.mkdir(parents=True, exist_ok=True)
    cells = load_cells()
    independent = render(cells, "independent")
    shared = render(cells, "shared", ft_routing=independent["routing"][0])
    assert independent["panels"][0] == shared["panels"][0], "Panel A must be identical"
    write_report(cells, {"independent": independent, "shared": shared})
    assert hashes == {str(path): drawing.sha256(path) for path in protected}
    logger.success("Both figures and comparison report completed: {}", REPORT)


if __name__ == "__main__":
    main()
