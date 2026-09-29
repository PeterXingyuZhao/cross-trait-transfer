"""Plot measured color token-SFT metrics on the exact saved SL PCA map."""

from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from typing import Any, Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from scripts import build_sl_network_atlas as atlas
from scripts import sl_network_atlas_metrics as metrics

ROOT = atlas.ROOT
OUT = ROOT / "output/pdf/qwen25_color_number_token_sft_metric_networks"
STEM = "Qwen25_color_number_token_SFT_SL_M1_M7_shared_PCA"
WIDTH, HEIGHT, GAP, MARGIN = 420.0, 430.0, 28.0, 22.0
TOP, BOTTOM = 113.0, 98.0
FONT, RADIUS = 16.0, 5.1
CUTOFF = 1.5
canonical = atlas.canonical


def load_panels() -> list[dict[str, Any]]:
    """Read exact token-SFT matrices, validating the objective and source hashes."""
    base = next(c for c in atlas.load_cells() if c["id"] == "qwen25_color_number_sft")
    assert base["family"] == "color" and base["objective"] == "sft"
    assert base["model_id"] == "Qwen/Qwen2.5-7B-Instruct" and base["scale"] == 1
    saved = metrics.load_metrics([base])["sets"][0]
    selected = {m["id"]: m for m in saved["metrics"]}
    measurements = [
        {"id": "sl", "title": "Subliminal-learning effect", "unit": "nats",
         "raw": base["raw"], "source": base["source"], "source_sha256": base["source_sha256"],
         "directional": True, "pearson_dc_all900": 1.0},
        {**selected["m1_teacher_cosine"], "id": "metric1",
         "title": "Metric 1: teacher-distribution\ngeometry"},
        {**selected["m7_causal_raw"], "id": "metric7_steering",
         "title": "Metric 7: data-generation\nfeature vector steering"},
    ]
    result = []
    for m in measurements:
        assert atlas.sha256(ROOT / m["source"]) == m["source_sha256"]
        if m["id"] != "sl":
            assert m["sl_source_sha256"] == base["source_sha256"]
            assert m["labels"] == base["labels"]
        centered = metrics.double_center(m["raw"])
        rms = canonical.off_diagonal_rms(centered)
        # Existing renderer selects at 4 display units and scales width by .16.
        # Map the common 1.5-RMS cutoff to that renderer, not to a native unit.
        factor = 4.0 / (CUTOFF * rms)
        cell = {**m, "name": m["id"], "family": "color", "labels": base["labels"],
                "native_raw": m["raw"], "native_centered": centered,
                "double_centered": centered * factor,
                "off_diagonal_rms": rms, "native_cutoff": CUTOFF * rms,
                "standardized_cutoff": CUTOFF, "display_factor": factor,
                "pca": copy.deepcopy(base["pca"]), "node_colors": base["node_colors"],
                "coordinate_basis": "Qwen2.5 color 00-99 token-SFT SL matrix",
                "sl_source_sha256": base["source_sha256"]}
        if m["id"] == "metric1":
            np.testing.assert_allclose(centered, centered.T, atol=1e-12, rtol=0)
            np.testing.assert_allclose(np.diag(m["raw"]), 1, atol=1e-12, rtol=0)
        result.append(atlas.json_ready(cell))
    return result


@contextmanager
def routing_style() -> Iterator[None]:
    """Scope typography and proportional padding to these new figures."""
    clean, loops = canonical.drawing.clean, canonical.drawing.loops
    previous = canonical.NODE_RADIUS, canonical.LABEL_FONTSIZE, clean.padded_limits, loops.padded_limits
    canonical.NODE_RADIUS, canonical.LABEL_FONTSIZE = RADIUS, FONT
    clean.padded_limits = loops.padded_limits = atlas.relative_limits
    try:
        yield
    finally:
        canonical.NODE_RADIUS, canonical.LABEL_FONTSIZE, clean.padded_limits, loops.padded_limits = previous


def audit_panel(cell: dict[str, Any], panel: dict[str, Any],
                width: float, height: float) -> dict[str, Any]:
    """Validate marks against native values; check unchanged node coordinates."""
    native = np.asarray(cell["native_centered"])
    expected = {(int(j), int(i)) for i, j in zip(*np.where(native >= cell["native_cutoff"]))}
    marks = panel["edges"] + panel["loops"]
    observed = {(r["source_index"], r["target_index"]) for r in marks}
    assert observed == expected and len(observed) == len(marks)
    for record in marks:
        i, j = record["target_index"], record["source_index"]
        value = native[i, j]
        np.testing.assert_allclose(record["weight"], value * cell["display_factor"], atol=1e-12)
        np.testing.assert_allclose(record["linewidth"], .16 * value * cell["display_factor"], atol=1e-12)
        record.update(native_value=float(value), standardized_value=float(value / cell["off_diagonal_rms"]))
    np.testing.assert_array_equal(panel["coordinates"], cell["pca"]["coordinates"])
    boxes = np.asarray(panel["label_boxes_points"])
    overlaps = [(i, j) for i, a in enumerate(boxes) for j, b in enumerate(boxes) if i < j
                and min(a[1], b[1]) - max(a[0], b[0]) > .1
                and min(a[3], b[3]) - max(a[2], b[2]) > .1]
    clipped = [i for i, b in enumerate(boxes) if b[0] < 0 or b[1] > width or b[2] < 0 or b[3] > height]
    assert not overlaps and not clipped, (overlaps, clipped)
    return {"edges": len(panel["edges"]), "loops": len(panel["loops"]),
            "label_overlaps": overlaps, "label_clipped": clipped,
            "coordinates_unchanged": True, "positive_only": True,
            "same_threshold_for_arrows_and_loops": True}


def render(cells: list[dict[str, Any]], stem: str, *, label_positions: Any = None) -> dict[str, Any]:
    """Render one comparison row or a standalone metric using the same SL map."""
    n = len(cells)
    width, height = 2 * MARGIN + n * WIDTH + (n - 1) * GAP, TOP + HEIGHT + BOTTOM
    fig = plt.figure(figsize=(width / 72, height / 72), dpi=180)
    fig.text(MARGIN / width, 1 - 10 / height, "Qwen2.5-7B-Instruct | Color | 00-99 token SFT",
             fontsize=17 if n > 1 else 13, weight="bold", color="#233842", va="top")
    panels, routes, audits = [], [], []
    with routing_style():
        for index, cell in enumerate(cells):
            left = MARGIN + index * (WIDTH + GAP)
            ax = fig.add_axes([left / width, BOTTOM / height, WIDTH / width, HEIGHT / height])
            route = canonical.freeze_routing(ax, cell, label_positions=label_positions)
            route["family"] = "color"
            label_positions = route["label_positions"]
            ax.clear()
            panel = canonical.drawing.loops.panel(
                ax, cell, route, {}, variant="external", title="", node_radius=RADIUS,
                positive_only=True, self_effect_minimum=4.0,
                node_colors=cell["node_colors"], label_fontsize=FONT)
            audit = audit_panel(cell, panel, width, height)
            panels.append(panel)
            routes.append(route)
            audits.append(audit)
            heading = (f"{chr(65 + index)}  " if n > 1 else "") + cell["title"]
            fig.text(left / width, (height - 45) / height, heading, fontsize=16,
                     weight="bold", color="#233842", va="top", linespacing=1.15)
            subtitle = (f"SL PCA: {100 * cell['pca']['explained_variance_2d']:.1f}% variance"
                        if cell["id"] == "sl" else f"Agreement with SL: r = {cell['pearson_dc_all900']:.3f}")
            fig.text(left / width, (height - 91) / height, subtitle,
                     fontsize=12.5, color="#61757c", va="top")
            fig.text((left + WIDTH / 2) / width, 79 / height,
                     f"Cutoff: {cell['native_cutoff']:.3g} {cell['unit']} | "
                     f"{audit['edges']} arrows, {audit['loops']} loops", ha="center",
                     fontsize=11.5, color="#61757c")
            if index < n - 1:
                x = (left + WIDTH + GAP / 2) / width
                fig.add_artist(plt.Line2D([x, x], [BOTTOM / height, (height - 40) / height],
                               transform=fig.transFigure, color="#ccd6d9", lw=.6, ls=(0, (3, 4))))
            logger.info("{}: {}", cell["id"], audit)
    loops = canonical.drawing.loops
    keys = [loops.MagnitudeKey(value=v) for v in (4, 8, 12)]
    fig.legend(keys, ["1.5 RMS", "3 RMS", "4.5 RMS"],
               handler_map={loops.MagnitudeKey: loops.MagnitudeHandler()},
               loc="lower center", bbox_to_anchor=(.5, 41 / height), ncol=3,
               fontsize=12, frameon=False, handlelength=3.2, columnspacing=1.8)
    fig.text(.5, 28 / height, "Shared SL PCA positions | Positive entries >= 1.5 x each matrix's RMS",
             ha="center", fontsize=11 if n > 1 else 9, color="#61757c")
    fig.text(.5, 10 / height, "All matrices double-centered; node colors illustrate the color names.",
             ha="center", fontsize=11 if n > 1 else 9, color="#61757c")
    for extension in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"{stem}.{extension}", dpi=200, facecolor="white")
    plt.close(fig)
    result = atlas.json_ready({"cells": cells, "panels": panels, "routing": routes, "audits": audits,
                               "width_pt": width, "height_pt": height})
    (OUT / f"{stem}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def write_caption(cells: list[dict[str, Any]]) -> None:
    """Supply a minimal paper wrapper without changing the existing main figure."""
    caption = r"""\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/images/Qwen25_color_number_token_SFT_SL_M1_M7_shared_PCA.pdf}
\caption{\textbf{Color-trait structure in token-SFT effects and metric measurements.}
Results use Qwen2.5-7B-Instruct and 00--99 token SFT, not distribution distillation.
\textbf{A}: the subliminal-learning effect matrix for 30 color traits.
\textbf{B}: teacher-distribution geometry (Metric~1).
\textbf{C}: data-generation feature vector steering (Metric~7).
All matrices are double-centered. The panels share PCA coordinates fitted only
to the SL matrix; node colors illustrate the named colors.
Each matrix is divided by its own off-diagonal RMS. Only positive entries
at least 1.5 on this dimensionless scale are shown, with the same rule for
arrows and self-loops; widths encode standardized strength, not comparable
absolute magnitude or statistical significance.
In A/C, arrows run from the teacher trait to the scored answer.
In B, reciprocal arrows denote symmetric similarity and diagonal loops are
structural, not measured self-transfer.
Pearson agreement with SL is $r=0.604$ for Metric~1 and $r=0.802$ for Metric~7,
computed over all 900 double-centered entries before display thresholding.}
\label{fig:color-token-sft-metric-networks}
\end{figure}
"""
    (OUT / "color_token_sft_metric_networks.tex").write_text(caption)


def main() -> None:
    """Produce three vector PDFs plus previews, sources, and a caption."""
    OUT.mkdir(parents=True, exist_ok=True)
    cells = load_panels()
    combined = render(cells, STEM)
    positions = combined["routing"][0]["label_positions"]
    for cell in cells[1:]:
        render([cell], f"Qwen25_color_number_token_SFT_{cell['id']}_network", label_positions=positions)
    write_caption(cells)
    logger.success("Wrote token-SFT color metric networks to {}", OUT)


if __name__ == "__main__":
    main()
