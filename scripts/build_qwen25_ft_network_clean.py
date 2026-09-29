"""Render compact axis-free PCA networks with auditable, obstacle-aware arrows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from adjustText import adjust_text
from loguru import logger
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as MplPath

from scripts.analyze_qwen25_ft_network import select_edges
from scripts.build_qwen25_ft_geometry_report import REGION_COLORS, REGION_NAMES, point_colors
from scripts.build_qwen25_ft_network_figures import separate_labels
from scripts.ft_network_edge_routing import baseline_routes, route_edges, routing_metrics

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results_for_presentation/qwen25_ft_geometry/analysis.json"
OUT = ROOT / "results_for_presentation/qwen25_ft_network"
METADATA = ROOT / "scripts/report_templates/ft_geometry_metadata.json"
POSITIVE = "#b6343d"
NEGATIVE = "#346ca8"
INK = "#233842"
THRESHOLD = 4.0
WIDTH_SCALE = .16
AREA_SCALE = 4.0
STYLES = ("width", "opacity", "width_opacity")


def edge_encoding(weight: float, style: str, maximum: float) -> tuple[float, float]:
    """Return shaft width (points) and alpha, on a common cross-family scale."""
    if style not in STYLES or not np.isfinite(weight) or not np.isfinite(maximum) or maximum <= 0:
        raise ValueError("Expected a valid style, finite weight and positive maximum")
    if abs(weight) > maximum + 1e-9:
        raise ValueError("Weight exceeds the shared maximum")
    alpha = .25 + .60 * abs(weight) / maximum
    return (1.25 if style == "opacity" else WIDTH_SCALE * abs(weight),
            .80 if style == "width" else alpha)


def padded_limits(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop to each panel's data while preserving equal aspect and exact PCA."""
    low, high = xy.min(axis=0), xy.max(axis=0)
    padding = np.maximum(1.8, (high-low)*.055)
    return low-padding, high+padding


def create_panel(
    ax: plt.Axes, cell: dict[str, Any], family: str, metadata: dict[str, Any],
    *, node_colors: list[str] | None = None,
) -> dict[str, Any]:
    """Freeze label placement, then optimize only arrow control points."""
    xy = np.asarray(cell["pca"]["coordinates"], dtype=float)
    matrix = np.asarray(cell["double_centered"], dtype=float)
    labels = cell["labels"]
    low, high = padded_limits(xy)
    ax.set_xlim(low[0], high[0])
    ax.set_ylim(low[1], high[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.set_title(f"{'A' if family == 'city' else 'B'}  {family.title()}",
                 loc="left", fontsize=12, fontweight="bold", color=INK, pad=9)
    areas = AREA_SCALE*np.abs(np.diag(matrix))
    # A circular scatter marker encloses pi/4 of its s parameter.
    ax.scatter(xy[:, 0], xy[:, 1], s=areas*4/np.pi,
               c=point_colors(family, labels, metadata) if node_colors is None else node_colors,
               edgecolors=[POSITIVE if v>=0 else NEGATIVE for v in np.diag(matrix)],
               linewidths=.65, zorder=3)
    texts = [ax.text(x, y, label, fontsize=8.5, color=INK, zorder=5,
                    path_effects=[pe.withStroke(linewidth=2.3, foreground="white")])
             for (x, y), label in zip(xy, labels)]
    ax.figure.canvas.draw()
    np.random.seed(20260909)
    adjust_text(texts, x=xy[:, 0], y=xy[:, 1], ax=ax,
                force_text=(.75, .8), force_static=(.55, .65),
                expand=(1.2, 1.35), iter_lim=900)
    separate_labels(ax, texts, xy)
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    units = ax.figure.dpi/72
    positions = ax.transData.transform(xy)/units
    radii = np.sqrt(areas/np.pi)+1.0
    boxes = []
    for text in texts:
        box = text.get_window_extent(renderer)
        boxes.append((box.x0/units-1, box.x1/units+1,
                      box.y0/units-1, box.y1/units+1))
    bounds = (ax.bbox.x0/units+1, ax.bbox.x1/units-1,
              ax.bbox.y0/units+1, ax.bbox.y1/units-1)
    selected = select_edges(matrix, THRESHOLD, labels=labels)
    logger.info("Routing {}: {} edges on frozen node coordinates", family, len(selected))
    baseline = baseline_routes(positions, radii, selected, bounds)
    routes = route_edges(positions, radii, selected, bounds, label_boxes=boxes)
    before = routing_metrics(baseline, positions, radii, label_boxes=boxes)
    after = routing_metrics(routes, positions, radii, label_boxes=boxes)
    logger.info("{} routing before={} after={}", family, before, after)
    return {"family": family, "labels": labels, "coordinates": xy.tolist(),
            "node_area_pt2": areas.tolist(), "node_screen_points": positions.tolist(),
            "bounds_points": list(bounds), "label_boxes_points": boxes,
            "label_positions": [list(t.get_position()) for t in texts],
            "routes": routes, "baseline": baseline,
            "routing_before": before, "routing_after": after,
            "axes": ax, "patches": []}


def draw_routes(panel: dict[str, Any], style: str, maximum: float) -> None:
    """Apply an alternative encoding without changing a single route or node."""
    ax = panel["axes"]
    for patch in panel["patches"]:
        patch.remove()
    panel["patches"] = []
    units = ax.figure.dpi/72
    # Small flows are drawn last so a strong crossing cannot fully bury them.
    for route in sorted(panel["routes"], key=lambda edge: -abs(edge["weight"])):
        weight = route["weight"]
        width, alpha = edge_encoding(weight, style, maximum)
        physical_path = MplPath(np.asarray(route["vertices"])*units,
                                [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
        path = ax.transData.inverted().transform_path(physical_path)
        patch = FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=6,
                                linewidth=width, alpha=alpha,
                                color=POSITIVE if weight>0 else NEGATIVE,
                                linestyle="-" if weight>0 else (0, (3.2, 2.0)),
                                capstyle="round", joinstyle="round", zorder=1)
        patch.set_gid(f"edge-{panel['family']}-{route['source_index']}-to-{route['target_index']}-w-{weight:.8f}")
        ax.add_patch(patch)
        panel["patches"].append(patch)


def legends(fig: plt.Figure, style: str, maximum: float) -> None:
    """Keep a compact shared key, with no PCA axes or explanatory banner."""
    for legend in list(fig.legends):
        legend.remove()
    groups = [plt.Line2D([], [], linestyle="", marker="o", markersize=3.5,
                        color=ink, label=name)
              for name, ink in zip(REGION_NAMES, REGION_COLORS)]
    fig.legend(handles=groups, loc="lower center", bbox_to_anchor=(.5, .067),
               ncol=6, fontsize=7.3, frameon=False, handletextpad=.35, columnspacing=1.6)
    handles = [plt.Line2D([], [], color=POSITIVE, lw=1, label="Positive →"),
               plt.Line2D([], [], color=NEGATIVE, lw=1, ls="--", label="Negative →")]
    for value in (4, 8, 12):
        width, alpha = edge_encoding(value, style, maximum)
        handles.append(plt.Line2D([], [], color="#58646a", lw=width, alpha=alpha,
                                  label=f"{value} nats"))
    handles += [plt.Line2D([], [], linestyle="", marker="o",
                          markersize=2*np.sqrt(AREA_SCALE*value/np.pi),
                          markerfacecolor="#d5dce0", markeredgecolor=POSITIVE,
                          markeredgewidth=.65, label=f"Self {value}") for value in (4, 12)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .015),
               ncol=7, fontsize=7.3, frameon=False, handlelength=2.5,
               handletextpad=.6, columnspacing=1.4)


def json_ready(value: Any) -> Any:
    """Serialize the auditable routing result without Matplotlib objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items() if k not in ("axes", "patches")}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def build() -> None:
    """Produce three controlled encoding variants and preserve the original assets."""
    source = json.loads(SOURCE.read_text())
    metadata = json.loads(METADATA.read_text())
    cells = [source["cells"][f"{family}-00-99"] for family in ("city", "color")]
    # Only off-diagonal edges determine the edge-opacity maximum.
    maximum = max(abs(edge["weight"]) for cell in cells
                  for edge in select_edges(cell["double_centered"], THRESHOLD))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), dpi=160)
    fig.subplots_adjust(left=.022, right=.985, bottom=.145, top=.935, wspace=.12)
    panels = [create_panel(ax, cell, family, metadata)
              for ax, cell, family in zip(axes, cells, ("city", "color"))]
    OUT.mkdir(parents=True, exist_ok=True)
    for style in STYLES:
        for panel in panels:
            draw_routes(panel, style, maximum)
        legends(fig, style, maximum)
        for extension in ("png", "svg", "pdf"):
            fig.savefig(OUT/f"pca_clean_{style}.{extension}", dpi=260,
                        facecolor="white", bbox_inches="tight", pad_inches=.04)
        logger.success("Rendered pca_clean_{}", style)
    plt.close(fig)
    # A contact sheet of the exact exported scientific figures, not another layout.
    comparison, views = plt.subplots(3, 1, figsize=(13, 17.6), dpi=140)
    names = ("Line width · fixed opacity (recommended)",
             "Opacity only · constant line width", "Line width + opacity · redundant magnitude")
    for ax, style, name in zip(views, STYLES, names):
        ax.imshow(plt.imread(OUT/f"pca_clean_{style}.png"))
        ax.set_axis_off()
        ax.set_title(name, loc="left", fontsize=12, color=INK, pad=1)
    comparison.subplots_adjust(left=.01, right=.99, top=.98, bottom=.01, hspace=.045)
    comparison.savefig(OUT/"pca_clean_encoding_comparison.png", dpi=200,
                       facecolor="white", bbox_inches="tight", pad_inches=.04)
    plt.close(comparison)
    manifest = {"source": str(SOURCE.relative_to(ROOT)),
                "sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                "threshold_nats": THRESHOLD,
                "node_position_policy": "Original PC1/PC2 scores; only labels and edge control points moved. Equal aspect, independently cropped panels.",
                "style_policy": "All three encodings have identical nodes, labels, edge directions, values, and optimized routes.",
                "edge_max_abs_nats": maximum,
                "encodings": {"width": "width=.16*abs(w) points, alpha=.80",
                              "opacity": "width=1.25 points, alpha=.25+.60*abs(w)/shared_edge_max",
                              "width_opacity": "width=.16*abs(w) points, alpha=.25+.60*abs(w)/shared_edge_max",
                              "node_area": "Enclosed circle area=4*abs(diagonal) square points; outline indicates sign"},
                "panels": json_ready(panels)}
    (OUT/"pca_clean_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    build()
