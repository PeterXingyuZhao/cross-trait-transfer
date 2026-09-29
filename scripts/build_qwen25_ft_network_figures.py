#!/usr/bin/env python3
"""Render directed FT networks on frozen PCA coordinates, without refitting."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from loguru import logger
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as MplPath

from scripts.analyze_qwen25_ft_network import select_edges
from scripts.build_qwen25_ft_geometry_report import (
    REGION_COLORS, REGION_NAMES, point_colors,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_for_presentation/qwen25_ft_network"
TEMPLATES = ROOT / "scripts/report_templates"
POSITIVE = "#b6343d"
NEGATIVE = "#346ca8"
INK = "#233842"
THRESHOLD = 4.0
WIDTH_SCALE = 0.16
AREA_SCALE = 4.0


def edge_width(weight: float) -> float:
    """Map absolute nats linearly to shaft width in typographic points."""
    return WIDTH_SCALE * abs(weight)


def node_areas(matrix: np.ndarray, loops: bool = False) -> np.ndarray:
    """Use area (not diameter) proportional to diagonal, or constant loop nodes."""
    return (np.full(len(matrix), 24.0) if loops
            else AREA_SCALE * np.abs(np.diag(matrix)))


def routed_path(
    start: np.ndarray, end: np.ndarray, radius_start: float, radius_end: float,
    *, reciprocal: bool, self_loop: bool = False,
) -> MplPath:
    """Build a directed Bezier in display pixels, keeping reciprocal arcs apart.

    Always use the same left-hand normal relative to the directed chord. Thus
    reversing source and target places reciprocal edges on opposite sides.
    Very short chords receive a larger bend, without moving either node.
    """
    delta = end - start
    length = float(np.linalg.norm(delta))
    if self_loop:
        r = max(radius_start, 2.0)
        vertices = [start + [-r, 0], start + [-24, 25],
                    start + [24, 25], start + [r, 0]]
        return MplPath(vertices, [MplPath.MOVETO] + [MplPath.CURVE4] * 3)
    unit = delta / max(length, 1e-12)
    normal = np.array([-unit[1], unit[0]])
    bend = (0.10 if reciprocal else 0.055) * length
    if length < 28:
        bend = max(bend, 20.0)
    control = (start + end) / 2 + normal * bend
    tangent_start = control - start
    tangent_end = control - end
    a = start + tangent_start / np.linalg.norm(tangent_start) * radius_start
    b = end + tangent_end / np.linalg.norm(tangent_end) * radius_end
    return MplPath([a, control, b], [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])


def add_edge(
    ax: plt.Axes, xy: np.ndarray, areas: np.ndarray, source: int, target: int,
    weight: float, *, reciprocal: bool = False, faint: bool = False,
    self_loop: bool = False,
) -> None:
    """Draw a source-to-target edge; weight never depends on PCA distance."""
    dpi_scale = ax.figure.dpi / 72
    points = ax.transData.transform(xy[[source, target]])
    # Scatter's circular marker diameter is sqrt(s) points.
    radii = (np.sqrt(areas[[source, target]]) / 2 + 1.5) * dpi_scale
    path = routed_path(*points, *radii, reciprocal=reciprocal, self_loop=self_loop)
    data_path = ax.transData.inverted().transform_path(path)
    color = POSITIVE if weight > 0 else NEGATIVE
    patch = FancyArrowPatch(
        path=data_path, arrowstyle="-|>", mutation_scale=7.0,
        linewidth=edge_width(weight), color=color,
        alpha=.028 if faint else .78, zorder=.5 if faint else 1,
        linestyle="-" if weight > 0 else (0, (3.2, 2.0)),
        capstyle="round", joinstyle="round",
    )
    patch.set_gid(f"edge-{source}-to-{target}-w-{weight:.8f}")
    ax.add_patch(patch)


def separate_labels(ax: plt.Axes, texts: list[Any], xy: np.ndarray) -> list[list[float]]:
    """Resolve remaining label collisions in display space; only labels can move."""
    renderer = ax.figure.canvas.get_renderer()
    for _ in range(300):
        boxes = [t.get_window_extent(renderer).expanded(1.06, 1.25) for t in texts]
        moved = False
        for i, first in enumerate(boxes):
            for j in range(i + 1, len(boxes)):
                second = boxes[j]
                overlap_x = min(first.x1, second.x1) - max(first.x0, second.x0)
                overlap_y = min(first.y1, second.y1) - max(first.y0, second.y0)
                if overlap_x > 0 and overlap_y > 0:
                    direction = 1 if first.y0 >= second.y0 else -1
                    for index, shift in ((i, direction), (j, -direction)):
                        point = ax.transData.transform(texts[index].get_position())
                        point[1] += shift * (overlap_y / 2 + 1)
                        texts[index].set_position(ax.transData.inverted().transform(point))
                    moved = True
        if not moved:
            break
    # Draw label-only dotted leaders after every label has its final position.
    for text, position in zip(texts, xy):
        ax.annotate("", xy=position, xytext=text.get_position(),
                    arrowprops={"arrowstyle": "-", "color": "#9ba6aa", "lw": .4,
                                "linestyle": ":", "shrinkA": 3, "shrinkB": 4}, zorder=2)
    return [list(t.get_window_extent(renderer).bounds) for t in texts]


def network_panel(
    ax: plt.Axes, cell: dict[str, Any], family: str, metadata: dict[str, Any],
    *, kind: str = "double_centered", sign: str = "both", loops: bool = False,
    faint: bool = False, threshold: float = THRESHOLD, letter: str = "",
    label_fontsize: float = 8.2,
) -> dict[str, Any]:
    """Draw one fixed PCA panel and return a machine-readable rendering manifest."""
    xy = np.asarray(cell["pca"]["coordinates"], dtype=float)
    matrix = np.asarray(cell[kind], dtype=float)
    labels = cell["labels"]
    areas = node_areas(matrix, loops)
    # Common display limits enclose both frozen embeddings, in the same units.
    ax.set_xlim(-24, 28)
    ax.set_ylim(-20, 27)
    ax.set_aspect("equal", adjustable="box")
    ev = np.asarray(cell["pca"]["explained_variance_ratio"])
    ax.set_xlabel(f"PC1 ({ev[0]:.1%})", fontsize=9)
    ax.set_ylabel(f"PC2 ({ev[1]:.1%})", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#c6ced1")
    ax.tick_params(labelsize=8, color="#c6ced1", labelcolor="#607078")
    ax.axhline(0, lw=.45, color="#e1e5e6", zorder=-1)
    ax.axvline(0, lw=.45, color="#e1e5e6", zorder=-1)
    selected = select_edges(matrix, threshold, sign, labels=labels)
    pairs = {(x["source_index"], x["target_index"]) for x in selected}
    n_positive = sum(x["weight"] > 0 for x in selected)
    subtitle = f"{n_positive} positive / {len(selected) - n_positive} negative arrows"
    ax.set_title(f"{letter}  {family.title()} × 00–99\n{subtitle}", loc="left", fontsize=12, pad=12, color=INK)
    # Finalize axes before calculating offsets in typographic/display units.
    ax.figure.canvas.draw()
    if faint:
        for edge in select_edges(matrix, 0, sign, labels=labels):
            if abs(edge["weight"]) < threshold:
                add_edge(ax, xy, areas, edge["source_index"], edge["target_index"],
                         edge["weight"], reciprocal=True, faint=True)
    for edge in sorted(selected, key=lambda x: abs(x["weight"])):
        source, target = edge["source_index"], edge["target_index"]
        add_edge(ax, xy, areas, source, target, edge["weight"],
                 reciprocal=(target, source) in pairs)
    if loops:
        for index, value in enumerate(np.diag(matrix)):
            add_edge(ax, xy, areas, index, index, float(value), self_loop=True)
    scatter = ax.scatter(xy[:, 0], xy[:, 1], s=areas,
                         c=point_colors(family, labels, metadata),
                         edgecolors=[POSITIVE if w >= 0 else NEGATIVE for w in np.diag(matrix)],
                         linewidths=.7, zorder=3)
    scatter.set_gid("nodes-frozen-pca")
    texts = [ax.text(x, y, label, fontsize=label_fontsize, color=INK, zorder=4,
                    path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])
             for (x, y), label in zip(xy, labels)]
    np.random.seed(20260908)
    adjust_text(texts, x=xy[:, 0], y=xy[:, 1], ax=ax,
                expand=(1.18, 1.4), force_text=(.65, .85),
                force_static=(.3, .45), iter_lim=700)
    label_bounds = separate_labels(ax, texts, xy)
    return {"family": family, "matrix_kind": kind, "sign": sign,
            "threshold_nats": threshold, "coordinates": xy.tolist(),
            "node_area_pt2": areas.tolist(), "self_loop_edges": loops,
            "faint_weak_edges": faint, "foreground_edges": selected,
            "label_positions": [list(t.get_position()) for t in texts],
            "label_bounds_pixels": label_bounds}


def figure_legend(fig: plt.Figure, *, loops: bool, raw: bool = False) -> None:
    """Use a single fixed edge scale across both families and all variants."""
    handles = [plt.Line2D([], [], color=POSITIVE, lw=1.2, label="Positive →", ls="-"),
               plt.Line2D([], [], color=NEGATIVE, lw=1.2, label="Negative →", ls="--")]
    handles += [plt.Line2D([], [], color="#5c666c", lw=edge_width(w), label=f"|w| = {w} nats")
                for w in (4, 8, 12)]
    if not loops:
        handles += [plt.scatter([], [], s=AREA_SCALE * w, color="#d5dce0", edgecolor=POSITIVE,
                                linewidth=.7, label=f"Diagonal {w}") for w in (4, 12)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .042),
               ncol=len(handles), frameon=False, fontsize=8, handlelength=2.4,
               columnspacing=1.7)
    mode = "Raw mean log-probability shifts" if raw else "Double-centered FT interactions"
    note = ("Constant node area; every diagonal shown as a self-loop." if loops
            else "Node area ∝ |diagonal effect|; outline gives its sign.")
    fig.text(.06, .012, f"{mode}. {note}  Colors identify city groups / illustrative color names.",
             fontsize=8, color="#607078")


def build_figure(
    analysis: dict[str, Any], metadata: dict[str, Any], stem: str,
    *, kind: str = "double_centered", loops: bool = False, faint: bool = False,
    separated: bool = False,
) -> list[dict[str, Any]]:
    """Export a paired paper figure as raster and editable vector files."""
    fig, axes = plt.subplots(2 if separated else 1, 2,
                             figsize=(15.5, 13.5 if separated else 7.6), squeeze=False)
    fig.subplots_adjust(left=.065, right=.97, top=.86 if not separated else .88,
                        bottom=.15 if not separated else .14, wspace=.22, hspace=.36)
    title = ("Directed FT effects on the original PCA map" if not separated
             else "Positive and negative centered interactions on identical PCA maps")
    fig.suptitle(title, x=.06, ha="left", y=.978, fontsize=18, color=INK)
    sub = ("Original node positions; arrows run from trained trait to scored answer. "
           "Width ∝ |FT effect|; foreground |w| ≥ 4 nats.")
    if faint:
        sub += "  Weaker edges: 2.8% opacity."
    fig.text(.06, .929 if not separated else .95, sub, fontsize=10, color="#526872")
    manifests = []
    for row, sign in enumerate(("positive", "negative") if separated else ("both",)):
        for col, family in enumerate(("city", "color")):
            manifests.append(network_panel(
                axes[row, col], analysis["cells"][f"{family}-00-99"], family, metadata,
                kind=kind, sign=sign, loops=loops, faint=faint,
                letter=chr(65 + row * 2 + col),
            ))
    figure_legend(fig, loops=loops, raw=kind == "raw")
    for index, (name, color) in enumerate(zip(REGION_NAMES, REGION_COLORS)):
        fig.text(.11 + index * .14, .086,
                 name, color=color, fontsize=8, ha="center")
    for extension in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"{stem}.{extension}", dpi=240, facecolor="white")
    plt.close(fig)
    logger.success("Rendered {}", stem)
    return manifests


def paper_plate(analysis: dict[str, Any], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Provide a 183-mm-wide portrait plate with actual 7-pt labels, no scaling."""
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 10.5))
    fig.subplots_adjust(left=.13, right=.96, top=.935, bottom=.13, hspace=.39)
    manifests = [network_panel(ax, analysis["cells"][f"{family}-00-99"], family,
                               metadata, letter=letter, label_fontsize=7.0)
                 for ax, family, letter in zip(axes, ("city", "color"), ("A", "B"))]
    city_handles = [plt.Line2D([], [], linestyle="", marker="o", markersize=4,
                              color=color, label=name)
                    for name, color in zip(REGION_NAMES, REGION_COLORS)]
    fig.legend(handles=city_handles, title="City groups", title_fontsize=7,
               loc="upper left", bbox_to_anchor=(.015, .86), fontsize=7,
               handlelength=.7, frameon=False)
    diagonal_handles = [plt.scatter([], [], s=AREA_SCALE * w, color="#d5dce0",
                                    edgecolor=POSITIVE, label=f"{w} nats") for w in (4, 12)]
    fig.legend(handles=diagonal_handles, title="Matching effect", title_fontsize=7,
               loc="upper left", bbox_to_anchor=(.015, .70), fontsize=7,
               frameon=False)
    handles = [plt.Line2D([], [], color=POSITIVE, lw=.9, label="Positive", ls="-"),
               plt.Line2D([], [], color=NEGATIVE, lw=.9, label="Negative", ls="--")]
    handles += [plt.Line2D([], [], color="#5c666c", lw=edge_width(w), label=f"{w} nats")
                for w in (4, 8, 12)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.53, .063),
               ncol=5, frameon=False, fontsize=7, handlelength=2.1)
    fig.text(.08, .040, "Arrow: trained trait → answer. Width ∝ |centered effect|; |w| ≥ 4 nats.", fontsize=7)
    fig.text(.08, .023, "Node area ∝ |matching effect|. Fixed PCA; weak edges omitted, not zero.", fontsize=7)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"pca_directed_paper_portrait.{extension}", dpi=300, facecolor="white")
    plt.close(fig)
    return manifests


def build_report(analysis: dict[str, Any]) -> None:
    """Build a self-contained report with offline mathematics and local figures."""
    template = (TEMPLATES / "qwen25_ft_network.html").read_text()
    findings = (
        '<p><strong>Recommendation:</strong> use the sparse combined network as the main view, '
        'and the separated-sign and complete heatmap views as supplements. '
        'Dot area carries the diagonal without adding 30 more loops per panel. '
        'The sparse view emphasizes the strongest tails; it is not a complete picture of all interactions.</p>'
        '<p>For paper-size inspection, a <a href="qwen25_ft_network/pca_directed_paper_portrait.pdf">'
        '183-mm-wide portrait plate</a> is also provided '
        '(<a href="qwen25_ft_network/pca_directed_paper_portrait.svg">editable SVG</a>). '
        'It uses 7-point node labels at its native size. The wide comparison above is a design-review layout; '
        'do not shrink it to a narrow column without reflowing the labels.</p>'
    )
    values = {
        "__DATA__": json.dumps(analysis, ensure_ascii=False, allow_nan=False,
                               separators=(",", ":")).replace("<", "\\u003c"),
        "__MATHJAX__": re.sub(r"</script", r"<\\/script",
                              (TEMPLATES / "vendor/mathjax-3.2.2-tex-svg.js").read_text(),
                              flags=re.IGNORECASE),
        "__FINDINGS__": findings,
        "__LITERATURE__": (TEMPLATES / "qwen25_ft_network_references.html").read_text(),
    }
    for name, replacement in values.items():
        template = template.replace(name, replacement)
    if any(name in template for name in values):
        raise ValueError("Unrendered report placeholder")
    destination = ROOT / "results_for_presentation/qwen25_city_color_ft_network.html"
    destination.write_text(template)
    sources = [OUT / "analysis.json", OUT / "figure_manifest.json", Path(__file__),
               TEMPLATES / "qwen25_ft_network.html", TEMPLATES / "qwen25_ft_network_references.html"]
    (OUT / "visualization.sources.json").write_text(json.dumps([
        {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in sources], indent=2) + "\n")
    logger.success("Built {}", destination)


def main() -> None:
    """Create prototypes and provenance; report template is built separately."""
    OUT.mkdir(exist_ok=True, parents=True)
    analysis_path = OUT / "analysis.json"
    analysis = json.loads(analysis_path.read_text())
    metadata = json.loads((TEMPLATES / "ft_geometry_metadata.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.labelcolor": INK})
    manifest = {"source_sha256": hashlib.sha256(analysis_path.read_bytes()).hexdigest(),
                "coordinates": "Unchanged frozen PCA coordinates; no layout optimization or node jitter.",
                "edge_width_points_per_nat": WIDTH_SCALE,
                "node_area_points_squared_per_nat": AREA_SCALE,
                "negative_style": "blue dashed; positive red solid", "figures": {}}
    for stem, options in (
        ("pca_directed_sparse", {}),
        ("pca_directed_faint_loops", {"loops": True, "faint": True}),
        ("pca_directed_signed_panels", {"separated": True}),
        ("pca_directed_raw", {"kind": "raw"}),
    ):
        manifest["figures"][stem] = build_figure(analysis, metadata, stem, **options)
    manifest["figures"]["pca_directed_paper_portrait"] = paper_plate(analysis, metadata)
    (OUT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    build_report(analysis)


if __name__ == "__main__":
    main()
