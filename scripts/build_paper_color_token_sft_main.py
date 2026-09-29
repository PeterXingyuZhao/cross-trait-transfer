"""Build a compact, audited color token-SFT main figure from cached results."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.colors import LinearSegmentedColormap

from scripts import build_qwen25_color_token_sft_metric_networks as prior
from scripts import color_metric_group_summary as groups

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/pdf/paper_main_color_token_sft_20260923"
BUNDLE = ROOT / "results_for_presentation/paper_main_color_token_sft_20260923"
SOURCE = prior.OUT / f"{prior.STEM}.json"
TOP = BUNDLE / "source_panel_a/subliminal_learning_diagram_v3.pdf"
INK, MUTED = "#233842", "#62737c"
WIDTH, HEIGHT = 1040.0, 484.0
CUTOFF, FONT, NODE_RADIUS = 1.60, 18.0, 5.0
DISPLAY_FACTOR = 2.0


def sha256(path: Path) -> str:
    """Hash the exact bytes used by this derived artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def double_center(matrix: np.ndarray) -> np.ndarray:
    """Center the complete matrix before any trait-pair selection."""
    return matrix - matrix.mean(0) - matrix.mean(1)[:, None] + matrix.mean()


def read_matrix(path: Path, labels: list[str]) -> np.ndarray:
    """Read a labeled answer-by-teacher CSV in the requested common order."""
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    columns = rows[0][1:]
    answers = [row[0] for row in rows[1:]]
    if len(answers) != len(labels) or len(set(answers)) != len(labels) or set(answers) != set(labels):
        raise ValueError(f"Wrong row labels: {path}")
    if len(columns) != len(labels) or len(set(columns)) != len(labels) or set(columns) != set(labels):
        raise ValueError(f"Wrong column labels: {path}")
    values = np.asarray([[float(v) for v in row[1:]] for row in rows[1:]])
    values = values[np.ix_([answers.index(v) for v in labels],
                          [columns.index(v) for v in labels])]
    if values.shape != (30, 30) or not np.isfinite(values).all():
        raise ValueError(f"Invalid matrix: {path}")
    return values


def agreement(matrix: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Compute descriptive Pearson correlations after complete centering."""
    a, b = double_center(matrix), double_center(reference)
    cross = ~np.eye(len(a), dtype=bool)
    return {"all900": float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
            "off870": float(np.corrcoef(a[cross], b[cross])[0, 1])}


def load_cells(metric1: Path, provenance: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate source bytes and replace the historical Metric 1 entirely."""
    saved = json.loads(SOURCE.read_text())
    cells = copy.deepcopy(saved["cells"])
    labels = cells[0]["labels"]
    audit = json.loads(provenance.read_text())
    if audit.get("model") != "Qwen/Qwen2.5-7B-Instruct" or audit.get("labels") != labels:
        raise ValueError("Metric 1 audit model/labels do not match SL")
    selected = next((p for p in audit["panels"] if p["name"] == audit["primary_panel"]), None)
    if selected is None or selected["output_sha256"]["metric1_current_raw.csv"] != sha256(metric1):
        raise ValueError("Metric 1 CSV does not match audited current result")
    if selected["zero_norm_signatures"] != 0 or selected["minimum_signature_norm"] <= 0:
        raise ValueError("Metric 1 audit contains undefined signatures")
    if cells[0]["source_sha256"] not in selected["provenance"]["files"].values():
        raise ValueError("Metric 1 audit does not identify matching SL source")
    for index in (0, 2):
        cell = cells[index]
        path = ROOT / cell["source"]
        if sha256(path) != cell["source_sha256"]:
            raise ValueError(f"Changed saved source: {path}")
        actual = read_matrix(path, labels)
        np.testing.assert_allclose(actual, cell["native_raw"], rtol=0, atol=1e-12)
    raw_m1 = read_matrix(metric1, labels)
    np.testing.assert_allclose(raw_m1, raw_m1.T, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.diag(raw_m1), 1.0, rtol=0, atol=1e-10)
    cells[1] = {"id": "metric1", "title": "Metric 1", "labels": labels,
                "native_raw": raw_m1.tolist(), "source": str(metric1.relative_to(ROOT)),
                "source_sha256": sha256(metric1), "provenance": str(provenance.relative_to(ROOT)),
                "provenance_sha256": sha256(provenance),
                "definition": "mean of per-prompt carrier-vector cosines",
                "directional": False}
    for cell in cells:
        centered = double_center(np.asarray(cell["native_raw"], dtype=float))
        rms = float(np.sqrt(np.mean(centered[~np.eye(30, dtype=bool)] ** 2)))
        cell.update(native_centered=centered.tolist(), off_diagonal_rms=rms)
        cell["agreement_with_sl"] = agreement(np.asarray(cell["native_raw"]),
                                               np.asarray(cells[0]["native_raw"]))
    return cells, audit


@contextmanager
def network_style(*, scale: float = 1.0) -> Iterator[None]:
    """Scope renderer typography and preserve equal-aspect PCA geometry."""
    canonical = prior.canonical
    clean, loops = canonical.drawing.clean, canonical.drawing.loops
    old = (canonical.NODE_RADIUS, canonical.LABEL_FONTSIZE, canonical.THRESHOLD,
           clean.padded_limits, loops.padded_limits)
    canonical.NODE_RADIUS, canonical.LABEL_FONTSIZE = NODE_RADIUS * scale, FONT * scale
    canonical.THRESHOLD = CUTOFF * DISPLAY_FACTOR
    clean.padded_limits = loops.padded_limits = prior.atlas.relative_limits
    try:
        yield
    finally:
        (canonical.NODE_RADIUS, canonical.LABEL_FONTSIZE, canonical.THRESHOLD,
         clean.padded_limits, loops.padded_limits) = old


def render_network(ax: plt.Axes, source: dict[str, Any], *, scale: float = 1.0,
                   label_layout: dict[str, Any] | None = None) -> dict[str, Any]:
    """Draw only measured positive effects above one uniform cutoff, with loops."""
    cell = copy.deepcopy(source)
    z = np.asarray(cell["native_centered"]) / cell["off_diagonal_rms"]
    factor = DISPLAY_FACTOR
    cell["double_centered"] = (z * factor).tolist()
    label_positions = None
    if label_layout is not None:
        if label_layout["labels"] != cell["labels"]:
            raise ValueError("Label-layout traits differ from the plotted source")
        np.testing.assert_array_equal(label_layout["coordinates"], cell["pca"]["coordinates"])
        label_positions = np.asarray(label_layout["label_positions"], dtype=float)
        if label_positions.shape != (30, 2) or not np.isfinite(label_positions).all():
            raise ValueError("Invalid saved network-label positions")
    with network_style(scale=scale):
        route = prior.canonical.freeze_routing(ax, cell, label_positions=label_positions)
        route["family"] = "color"
        ax.clear()
        panel = prior.canonical.drawing.loops.panel(
            ax, cell, route, {}, variant="external", title="",
            node_radius=NODE_RADIUS * scale, positive_only=True,
            self_effect_minimum=CUTOFF * factor,
            node_colors=cell["node_colors"], label_fontsize=FONT * scale)
    expected = {(int(col), int(row)) for row, col in zip(*np.where(z >= CUTOFF))}
    actual = {(m["source_index"], m["target_index"])
              for m in panel["edges"] + panel["loops"]}
    if actual != expected:
        raise AssertionError("Displayed network marks differ from thresholded SL")
    for mark in panel["edges"] + panel["loops"]:
        mark["standardized_effect"] = float(z[mark["target_index"], mark["source_index"]])
    boxes = np.asarray(panel["label_boxes_points"])
    overlap = [(i, j) for i, a in enumerate(boxes) for j, b in enumerate(boxes) if i < j
               and min(a[1], b[1]) - max(a[0], b[0]) > .15
               and min(a[3], b[3]) - max(a[2], b[2]) > .15]
    if overlap:
        raise AssertionError(f"Overlapping network labels: {overlap}")
    return {"panel": panel, "routing": route, "cutoff_rms": CUTOFF,
            "self_loop_cutoff_rms": CUTOFF, "label_overlaps": overlap,
            "coordinates_preserved": True, "label_layout_preserved": label_layout is not None}


def axes(fig: plt.Figure, x: float, y: float, w: float, h: float) -> plt.Axes:
    """Place one axes using physical points on the common figure canvas."""
    width, height = fig.get_size_inches() * 72
    return fig.add_axes([x / width, y / height, w / width, h / height])


def text(fig: plt.Figure, x: float, y: float, value: str, **kwargs: Any) -> None:
    """Place figure text in physical points with a common default style."""
    width, height = fig.get_size_inches() * 72
    fig.text(x / width, y / height, value, color=INK, **kwargs)


def draw_heatmap(fig: plt.Figure, x: float, y: float, size: float,
                 values: list[list[float]], title: str, subtitle: str,
                 limit: float) -> Any:
    """Render one six-group matrix on the shared symmetric color scale."""
    ax = axes(fig, x, y, size, size)
    cmap = LinearSegmentedColormap.from_list("group_effect", ["#326395", "#fafafa", "#b6333b"])
    im = ax.imshow(values, cmap=cmap, vmin=-limit, vmax=limit, interpolation="nearest")
    ax.set_xticks(range(6), [str(i) for i in range(1, 7)])
    ax.set_yticks(range(6), [str(i) for i in range(1, 7)])
    ax.tick_params(length=0, labelsize=15, pad=3, colors=MUTED)
    ax.set_xticks(np.arange(-.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 6, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=.65)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(.5)
        spine.set_color("#cad3d8")
    text(fig, x, y + size + 25, title, fontsize=17, weight="bold")
    if subtitle:
        text(fig, x, y + size + 8, subtitle, fontsize=14)
    return im


def render(cells: list[dict[str, Any]], *, layout: str) -> dict[str, Any]:
    """Export a compact layout with one SL network and three group summaries."""
    height = HEIGHT if layout == "compact" else 424.0
    fig = plt.figure(figsize=(WIDTH / 72, height / 72), dpi=160)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"})
    text(fig, 12, height - 21, "B  Subliminal-learning effects", fontsize=19, weight="bold")
    text(fig, 12, height - 43, "Qwen2.5-7B-Instruct  |  Color  |  00-99 token SFT", fontsize=14)
    text(fig, 601 if layout == "compact" else 573, height - 21,
         "C  Color-group interactions", fontsize=19, weight="bold")
    summary = groups.group_summary(cells)
    limit = float(np.ceil(summary["limit"] * 2) / 2)
    network_bounds = (8, 64, 565, 365) if layout == "compact" else (8, 28, 605, 339)
    label_layout = (None if layout == "compact" else
                    json.loads((BUNDLE / "network_label_layout.json").read_text()))
    network = render_network(axes(fig, *network_bounds), cells[0],
                             scale=1.0 if layout == "compact" else 339 / 365,
                             label_layout=label_layout)
    if layout == "compact":
        locations = [(625, 268, 151), (625, 50, 151), (841, 50, 151)]
        text(fig, 823, 415, "Fixed hue groups", fontsize=15, weight="bold")
        for i, name in enumerate(groups.GROUP_NAMES):
            text(fig, 823, 390 - i * 19, f"{i + 1}  {name}", fontsize=15)
    else:
        locations = [(587 + i * 144, 144, 130) for i in range(3)]
        for i, name in enumerate(groups.GROUP_NAMES):
            text(fig, 587 + (i % 2) * 226, 63 - (i // 2) * 25,
                 f"{i + 1}  {name}", fontsize=13)
        text(fig, 589, 103, "Rows: answer group  |  Columns: teacher group", fontsize=12)
    for i, ((x, y, size), cell, block) in enumerate(zip(locations, cells, summary["cells"])):
        group_r = float(np.corrcoef(np.asarray(summary["cells"][0]["values"]).ravel(),
                                   np.asarray(block["values"]).ravel())[0, 1])
        block["pearson_with_sl_all36"] = group_r
        subtitle = "" if i == 0 else f"Group r = {group_r:.3f}"
        im = draw_heatmap(fig, x, y, size, block["values"],
                          ["SL", "Metric 1", "Metric 4"][i], subtitle, limit)
    if layout == "compact":
        cax = axes(fig, 823, 257, 169, 8)
    else:
        cax = axes(fig, 587, 330, 418, 8)
    bar = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-limit, 0, limit])
    bar.ax.tick_params(labelsize=13, length=2, pad=2)
    bar.outline.set_linewidth(.45)
    if layout == "compact":
        text(fig, 823, 275, "Mean value / RMS", fontsize=13)
    else:
        text(fig, 796, 359, "Mean value / RMS", fontsize=12, ha="center")
    if layout == "compact":
        edge_count = len(network["panel"]["edges"])
        loop_count = len(network["panel"]["loops"])
        text(fig, 20, 51, f"{edge_count} arrows + {loop_count} self-loops  |  Centered SL "
             + rf"$\geq {CUTOFF:g}$ RMS", fontsize=13)
    else:
        text(fig, 26, 339, "Centered SL " + rf"$\geq {CUTOFF:g}$ RMS", fontsize=13)
    loops = prior.canonical.drawing.loops
    keys = [loops.MagnitudeKey(value=v * DISPLAY_FACTOR) for v in (2, 3, 4)]
    fig.legend(keys, ["2 RMS", "3 RMS", "4 RMS"],
               handler_map={loops.MagnitudeKey: loops.MagnitudeHandler()},
               loc="lower left", bbox_to_anchor=(20 / WIDTH, (18 if layout == "compact" else 20) / height),
               ncol=3, fontsize=13, frameon=False, handlelength=3.2, columnspacing=1.8)
    if layout == "compact":
        text(fig, 603, 8, "Rows: answer group; columns: teacher group.", fontsize=12)
    stem = f"color_token_sft_network_heatmaps_{layout}"
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{stem}.{suffix}", dpi=200, facecolor="white")
    plt.close(fig)
    manifest = {"layout": layout, "network": network, "group_summary": summary,
                "heatmap_limit": limit, "width_pt": WIDTH, "height_pt": height,
                "network_axes_points": network_bounds,
                "heatmap_axes_points": locations,
                "magnitude_legend_anchor_pt": [20, 18 if layout == "compact" else 20],
                "network_label_layout_sha256": (None if layout == "compact" else
                                                 sha256(BUNDLE / "network_label_layout.json")),
                "correlation_display": "Pearson across all 36 group cells, after centering full 30x30 matrices, excluding true self-pairs and averaging signed blocks; no group recentering"}
    (OUT / f"{stem}.json").write_text(json.dumps(prior.atlas.json_ready(manifest), indent=2) + "\n")
    return manifest


def compose_main() -> None:
    """Preserve Panel A as vector artwork and tightly stack the revised bottom."""
    from pypdf import PdfReader, PdfWriter, Transformation

    top = PdfReader(TOP).pages[0]
    bottom = PdfReader(OUT / "color_token_sft_network_heatmaps_row.pdf").pages[0]
    bottom_height = float(bottom.mediabox.height)
    # The supplied diagram already has a tight crop box; respect it when merging.
    crop = top.cropbox
    scale = WIDTH / float(crop.width)
    gap = 2.0
    height = bottom_height + gap + float(crop.height) * scale
    writer = PdfWriter()
    page = writer.add_blank_page(width=WIDTH, height=height)
    page.merge_page(bottom)
    transform = (Transformation().translate(-float(crop.left), -float(crop.bottom))
                 .scale(scale).translate(0, bottom_height + gap))
    page.merge_transformed_page(top, transform)
    writer.add_metadata({"/Title": "Subliminal learning: color token-SFT transfer and group interactions",
                         "/Subject": "Qwen2.5-7B-Instruct; Metric 1 mean per-prompt cosines; Panel A preserved"})
    path = OUT / "subliminal_learning_main_color_token_sft.pdf"
    with path.open("wb") as stream:
        writer.write(stream)
    subprocess.run(["pdftoppm", "-singlefile", "-scale-to", "2600", "-png",
                    str(path), str(OUT / path.stem)], check=True, capture_output=True)


def package(cells: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    """Write the caption, portable Overleaf assets, and explicit audit notes."""
    image_dir = BUNDLE / "figures/images"
    image_dir.mkdir(parents=True, exist_ok=True)
    main_stem = "subliminal_learning_main_color_token_sft"
    for path in OUT.glob("*.pdf"):
        shutil.copy2(path, image_dir / path.name)
    for path in OUT.glob("*.svg"):
        shutil.copy2(path, image_dir / path.name)
    shutil.copy2(OUT / f"{main_stem}.png", BUNDLE / "preview.png")
    caption = r"""\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{figures/images/subliminal_learning_main_color_token_sft.pdf}
\caption{\textbf{Cross-trait structure in subliminal learning.}
\textbf{A}: Schematic of trait transmission through semantically unrelated outputs.
\textbf{B}: Color preferences after 00--99 token SFT of Qwen2.5-7B-Instruct.
Node positions are derived by PCA of the full signed, double-centered SL matrix.
Arrows point from teacher traits to student answers; loops denote self-effects.
Only positive centered entries at least 1.60 times the off-diagonal RMS are shown
(45 arrows and 22 loops), with width proportional to standardized magnitude.
\textbf{C}: Interactions among six fixed hue groups for SL,
teacher-distribution similarity (Metric~1), and prompt-trait vector steering
(Metric~7). Each full matrix is double-centered and divided by its own
off-diagonal RMS before averaging all signed entries within each group pair.
True self-pairs are excluded; diagonal group cells retain interactions between
distinct traits. No edge cutoff is applied to the heatmaps, which share one
color scale. Metric~1 averages per-prompt cosines over 1,024 equally weighted
shared prompts and is symmetric. Displayed group correlations with SL use
all 36 group cells. Correlations over all 900 trait pairs are 0.627 and 0.802
for Metrics~1 and~7; restricting to the 870 cross-trait pairs after full-matrix
centering gives 0.498 and 0.743, respectively.}
\label{fig:main}
\end{figure}
"""
    (BUNDLE / "figures/main_diagram.tex").write_text(caption)
    correlations = [c["agreement_with_sl"] for c in cells]
    group_rs = [c["pearson_with_sl_all36"] for c in summary["cells"]]
    readme = f"""# Revised color token-SFT main figure

Recommended: `figures/images/{main_stem}.pdf`.
The selected row layout uses one SL network and three aligned 6-by-6 heatmaps.
The original Panel A was extracted unchanged from `Subliminal_Learning (7).zip`
(`figures/images/subliminal_learning_diagram_v3.pdf`); its existing crop box
removes blank margins when assembling the composite. The attached Qwen3-VL
screenshot was a layout reference only. All numerical data here use Qwen2.5.

## Overleaf

Copy `figures/images/{main_stem}.pdf` and `figures/main_diagram.tex`
into the paper, replacing the old figure wrapper. The wrapper preserves
`\\label{{fig:main}}`. Any surrounding text referring to old panels C/D as networks
will need to call the new right-hand heatmaps panel C.
`graphicx` is the only additional package needed by this wrapper.

## Layout decision

- Selected single row: three aligned heatmaps, 130 points square (increased
  from 117). The bottom panel is 1,040 by 424 points, reduced from 484 points
  high. The network is 7% larger than the first shortened row version; Panel C
  starts 28 points farther left. Its cutoff label sits in the upper-left
  whitespace, and the 2/3/4-RMS magnitude legend is raised 12 points into the
  network's lower margin. The Panel C bottom note is removed;
  its scientific definitions remain explicit in the caption and provenance.
- Earlier 2-by-2 alternative: SL upper left, hue key and colorbar upper right,
  Metric 1 and Metric 7 below. Each heatmap is 151 points square.
- Network PCA coordinates and distances are preserved. Only label placement
  and arrow routing are adapted to the selected panel. The 1.60-RMS cutoff gives
  45 directed cross-trait edges and 22 self-loops. Compared with the previous
  1.75-RMS view, this adds nine arrows (six within and three between the fixed
  hue groups) and one self-loop. Overall, 38 arrows are within groups and seven
  are between groups. Arrow and loop widths share one scale, unchanged from
  the earlier versions. The earlier packages and artwork are preserved in
  `output/pdf/paper_main_color_token_sft_20260923/history/cutoff_1_75_rms/`
  and `history/cutoff_2_rms/` in the repository. No group-specific edge
  selection was used.
- No threshold is applied to heatmaps or correlations. All 870 signed
  cross-trait entries enter their appropriate blocks. The six hue groups
  are fixed presentation groups, not inferred clusters.

## Correlations and updated Metric 1

| Measurement | All 900 trait pairs | 870 cross-trait pairs | 36 group cells |
| --- | ---: | ---: | ---: |
| Metric 1, current per-prompt mean | {correlations[1]['all900']:.9f} | {correlations[1]['off870']:.9f} | {group_rs[1]:.9f} |
| Metric 7, prompt-trait steering | {correlations[2]['all900']:.9f} | {correlations[2]['off870']:.9f} | {group_rs[2]:.9f} |

The figure explicitly labels the last column as Group r. These are descriptive
correlations of group means and should not be substituted for individual-trait
agreement. Both complete 30-by-30 matrices are double-centered before any
trait-pair masking. Group correlations compare the displayed 36 signed means
without any further centering of the 6-by-6 matrices.

Metric 1 was recomputed from validated original caches with 1,024 distinct,
uniformly weighted shared prompts. The legacy flattened cosine is preserved
under explicitly historical filenames in `audit_metric1/original_1024/`.
Its old all-pair / cross-trait correlations were 0.604356059 / 0.482184465;
its group-level correlation was 0.885581302. The corresponding current
values are 0.626993284 / 0.498441500 / 0.909207297. No inference was rerun.

The independent 10,000-prompt stream recheck is supplementary and is not used
by this figure. Its full provenance and distinct sampling protocol remain in
`audit_metric1/recheck_*`. A separate accepted-prompt panel has missing caches;
it is not presented or silently completed.

## Contents and reproducibility

- Vector PDFs for the complete figure, selected row panels, and earlier 2-by-2 alternative.
- Editable SVG exports for both bottom-panel layouts; original Panel A draw.io.
- Saved network label positions in `network_label_layout.json`, preserving the
  approved label arrangement as the network is enlarged and repositioned.
- Full raw and centered 30-by-30 matrices and six-group CSVs.
- Exact source checksums, executed-source audit, prompt counts and weights in
  `figure_provenance.json`, `audit_sources.md`, and `audit_metric1/`.
- Renderer source `revision_tools/build_paper_color_token_sft_main.py` and
  its numerical/provenance tests. This renderer imports existing repository
  layout helpers and should be run from the subliminal-learning repository.

Rebuild from the repository with Python 3.12 and numpy, pandas, matplotlib,
scipy, adjustText, loguru, pypdf, and Poppler available:

```sh
python -m scripts.build_paper_color_token_sft_main \\
  --metric1-csv results_for_presentation/paper_main_color_token_sft_20260923/audit_metric1/original_1024/metric1_current_raw.csv \\
  --metric1-provenance results_for_presentation/paper_main_color_token_sft_20260923/audit_metric1/audit.json
```
"""
    (BUNDLE / "README.md").write_text(readme)
    revision_tools = BUNDLE / "revision_tools"
    revision_tools.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), revision_tools / Path(__file__).name)
    tests = ROOT / "tests/test_paper_color_token_sft_main.py"
    if tests.exists():
        shutil.copy2(tests, revision_tools / tests.name)
    destination = BUNDLE / "color_main_figure_overleaf.zip"
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for path in sorted(BUNDLE.rglob("*")):
            if path.is_file() and path != destination and "__pycache__" not in path.parts:
                archive.write(path, path.relative_to(BUNDLE))


def main() -> None:
    """Render current measurements, preserve provenance, and package vector files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric1-csv", type=Path, required=True)
    parser.add_argument("--metric1-provenance", type=Path, required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    BUNDLE.mkdir(parents=True, exist_ok=True)
    cells, audit = load_cells(args.metric1_csv.resolve(), args.metric1_provenance.resolve())
    render(cells, layout="compact")
    selected = render(cells, layout="row")
    compose_main()
    record = {"model": "Qwen/Qwen2.5-7B-Instruct", "training": "00-99 token SFT",
              "cells": cells, "metric1_audit": audit,
              "panel_a_source": str(TOP.relative_to(ROOT)), "panel_a_sha256": sha256(TOP),
              "original_figure_source": str(SOURCE.relative_to(ROOT)),
              "original_figure_sha256": sha256(SOURCE), "builder_sha256": sha256(Path(__file__)),
              "selected_layout": "row", "bottom_panel_height_pt": selected["height_pt"],
              "group_summary": selected["group_summary"]}
    (BUNDLE / "figure_provenance.json").write_text(json.dumps(prior.atlas.json_ready(record), indent=2) + "\n")
    for cell in cells:
        values = double_center(np.asarray(cell["native_raw"]))
        for kind, matrix in [("raw", cell["native_raw"]), ("double_centered", values)]:
            with (BUNDLE / f"{cell['id']}_{kind}.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["answer/teacher", *cell["labels"]])
                writer.writerows([label, *row] for label, row in zip(cell["labels"], matrix))
    for item in selected["group_summary"]["cells"]:
        with (BUNDLE / f"{item['id']}_group_interactions.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["answer_group/teacher_group", *groups.GROUP_NAMES])
            writer.writerows([label, *row] for label, row in zip(groups.GROUP_NAMES, item["values"]))
    package(cells, selected["group_summary"])
    logger.success("Rendered audited color main figure to {}", OUT)


if __name__ == "__main__":
    main()
