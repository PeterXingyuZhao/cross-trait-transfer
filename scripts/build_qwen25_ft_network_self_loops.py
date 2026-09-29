"""Compare diagonal and cross-trait FT effects with one literal stroke scale.

Read existing PCA coordinates and clean routing; do not fit a new embedding or
change effect values. Only loop orientation and arrow/node attachment change.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as MplPath

from scripts.build_qwen25_ft_network_clean import (
    INK, METADATA, NEGATIVE, OUT, POSITIVE, SOURCE, THRESHOLD, WIDTH_SCALE,
    padded_limits,
)
from scripts.build_qwen25_ft_geometry_report import REGION_COLORS, REGION_NAMES, point_colors

NODE_RADIUS = 1.8
ATTACH_RADIUS = NODE_RADIUS + .5
LOOP_RADIUS = 5.0
ALPHA = .8
VARIANTS = ("compact", "external", "paired")
CLEAN_MANIFEST = OUT / "pca_clean_manifest.json"


def stroke_encoding(weight: float) -> dict[str, Any]:
    """Use exactly the same unbounded linear scale for every directed effect."""
    if not np.isfinite(weight):
        raise ValueError("Effect weight must be finite")
    return {"linewidth": WIDTH_SCALE * abs(weight), "alpha": ALPHA,
            "color": POSITIVE if weight >= 0 else NEGATIVE,
            "linestyle": "-" if weight >= 0 else (0, (3.2, 2.0))}


def loop_path(
    center: np.ndarray, angle: float, variant: str, *, node_radius: float = NODE_RADIUS,
) -> MplPath:
    """Make a fixed-size glyph in points; magnitude never changes its geometry.

    Compact arcs encircle the marker without touching it. External petals leave
    and return to separate ports on the same marker (a literal self-edge).
    """
    center = np.asarray(center, dtype=float)
    if variant not in ("compact", "external") or center.shape != (2,):
        raise ValueError("Expected a two-dimensional center and supported loop style")
    if (not np.isfinite(center).all() or not np.isfinite(angle)
            or not np.isfinite(node_radius) or node_radius <= 0):
        raise ValueError("Loop geometry must be finite")
    if variant == "compact":
        arc = MplPath.arc(35, 325)
        vertices = arc.vertices * LOOP_RADIUS
        codes = arc.codes
    else:
        # A two-port cubic petal; identical length/shape at every node.
        attach_radius = node_radius + .5
        vertices = np.array([[.5*attach_radius, -.8660254*attach_radius],
                             [17., -9.], [17., 9.],
                             [.5*attach_radius, .8660254*attach_radius]])
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    return MplPath(vertices @ rotation.T + center, codes)


def sample_path(path: MplPath) -> np.ndarray:
    """Sample actual Bezier curves, not their control polygon, for obstacles."""
    samples = []
    for curve, code in path.iter_bezier():
        if code != MplPath.MOVETO:
            samples.append(curve(np.linspace(0, 1, 17)))
    return np.concatenate(samples)


def remap_routes(
    previous: dict[str, Any], xy: np.ndarray, positions: np.ndarray,
    *, node_radius: float = NODE_RADIUS,
) -> list[dict[str, Any]]:
    """Preserve clean control points and reattach arrows to equal-sized nodes."""
    old_screen = np.asarray(previous["node_screen_points"])
    mapping = np.linalg.lstsq(np.c_[old_screen, np.ones(len(xy))],
                             positions, rcond=None)[0]
    result = []
    for edge in previous["routes"]:
        vertices = np.c_[edge["vertices"], np.ones(3)] @ mapping
        for endpoint, index in ((0, edge["source_index"]), (2, edge["target_index"])):
            direction = vertices[1] - positions[index]
            vertices[endpoint] = positions[index] + (node_radius+.5)*direction/np.linalg.norm(direction)
        path = MplPath(vertices, [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
        result.append({k: edge[k] for k in ("source_index", "target_index", "weight")} |
                      {"vertices": vertices.tolist(), "codes": path.codes.tolist(),
                       "samples": sample_path(path).tolist(), **stroke_encoding(edge["weight"])})
    return result


def choose_loops(
    positions: np.ndarray, diagonal: np.ndarray, variant: str,
    routes: list[dict[str, Any]], label_boxes: np.ndarray, bounds: tuple[float, ...],
    *, node_radius: float = NODE_RADIUS,
) -> list[dict[str, Any]]:
    """Rotate constant-size loops away from nodes, labels and existing arrows.

    This deterministic finite-candidate heuristic cannot remove all overlap on
    crowded fixed coordinates. It neither moves a node nor rescales a loop.
    """
    edge_samples = np.concatenate([r["samples"] for r in routes]) if routes else np.empty((0, 2))
    chosen: list[dict[str, Any]] = []
    for index in np.argsort(-np.abs(diagonal), kind="stable"):
        candidates = []
        for angle in np.arange(24)*np.pi/12:
            path = loop_path(positions[index], float(angle), variant, node_radius=node_radius)
            samples = sample_path(path)
            halfwidth = stroke_encoding(float(diagonal[index]))["linewidth"]/2
            distances = np.linalg.norm(samples[:, None, :] - positions[None, :, :], axis=-1)
            distances[:, index] = np.inf
            node_cost = np.maximum(0, node_radius+halfwidth+1.2-distances).sum()
            # Include the curve's true occupied footprint, not arbitrary size.
            lower = label_boxes[:, [0, 2]]-halfwidth-.5
            upper = label_boxes[:, [1, 3]]+halfwidth+.5
            hits = ((samples[:, None] >= lower) & (samples[:, None] <= upper)).all(-1)
            label_cost = hits.sum()
            lower_bound = np.array([bounds[0], bounds[2]])+halfwidth
            upper_bound = np.array([bounds[1], bounds[3]])-halfwidth
            outside = (np.maximum(0, lower_bound-samples) +
                       np.maximum(0, samples-upper_bound)).sum()
            others = [edge_samples] + [np.asarray(loop["samples"]) for loop in chosen]
            other_samples = np.concatenate(others)
            collision = 0.
            if len(other_samples):
                nearest = np.linalg.norm(samples[:, None]-other_samples[None], axis=-1).min(1)
                collision = np.maximum(0, 1.1+halfwidth-nearest).sum()
            score = 200*node_cost+35*label_cost+200*outside+2*collision
            candidates.append((score, angle, path, samples))
        score, angle, path, samples = min(candidates, key=lambda entry: entry[0])
        weight = float(diagonal[index])
        chosen.append({"source_index": int(index), "target_index": int(index),
                       "weight": weight, "orientation_radians": float(angle),
                       "vertices": path.vertices.tolist(), "codes": path.codes.tolist(),
                       "samples": samples.tolist(), "routing_score": float(score),
                       **stroke_encoding(weight)})
    return sorted(chosen, key=lambda item: item["source_index"])


def add_arrows(ax: plt.Axes, records: list[dict[str, Any]], prefix: str) -> None:
    """Draw loops and edges with the very same arrow/stroke constructor."""
    for record in sorted(records, key=lambda item: -abs(item["weight"])):
        physical = MplPath(np.asarray(record["vertices"])*ax.figure.dpi/72, record["codes"])
        path = ax.transData.inverted().transform_path(physical)
        patch = FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=5.3,
                                capstyle="round", joinstyle="round", zorder=1,
                                **stroke_encoding(record["weight"]))
        patch.set_gid(f"{prefix}-{record['source_index']}-to-{record['target_index']}-w-{record['weight']:.8f}")
        ax.add_patch(patch)


def panel(
    ax: plt.Axes, cell: dict[str, Any], previous: dict[str, Any], metadata: dict[str, Any],
    *, variant: str, show_edges: bool = True, show_loops: bool = True, title: str,
    node_radius: float = NODE_RADIUS, positive_only: bool = False,
    self_effect_minimum: float | None = None,
    node_colors: list[str] | None = None,
    label_fontsize: float = 8.5,
) -> dict[str, Any]:
    """Render a frozen-coordinate panel and expose exact drawable metadata."""
    if self_effect_minimum is not None and (
        not np.isfinite(self_effect_minimum) or self_effect_minimum < 0
    ):
        raise ValueError("Self-effect minimum must be finite and nonnegative")
    xy = np.asarray(cell["pca"]["coordinates"])
    matrix = np.asarray(cell["double_centered"])
    low, high = padded_limits(xy)
    ax.set_xlim(low[0], high[0])
    ax.set_ylim(low[1], high[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=9)
    ax.figure.canvas.draw()
    units = ax.figure.dpi/72
    positions = ax.transData.transform(xy)/units
    texts = []
    for label, point in zip(cell["labels"], previous["label_positions"]):
        texts.append(ax.text(*point, label, fontsize=label_fontsize, color=INK, zorder=5,
                             ha="center", va="center",
                             path_effects=[pe.withStroke(linewidth=2.3, foreground="white")]))
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    boxes = []
    for text in texts:
        bbox = text.get_window_extent(renderer)
        boxes.append([bbox.x0/units, bbox.x1/units, bbox.y0/units, bbox.y1/units])
    bounds = (ax.bbox.x0/units+1, ax.bbox.x1/units-1,
              ax.bbox.y0/units+1, ax.bbox.y1/units-1)
    edges = remap_routes(previous, xy, positions, node_radius=node_radius) if show_edges else []
    if positive_only:
        edges = [edge for edge in edges if edge["weight"] > 0]
    loops = choose_loops(positions, np.diag(matrix), variant, edges,
                         np.asarray(boxes), bounds, node_radius=node_radius) if show_loops else []
    # Filter only after routing, so retained loops keep their original geometry.
    if self_effect_minimum is not None:
        loops = [loop for loop in loops
                 if loop["weight"] > 0 and loop["weight"] >= self_effect_minimum]
    add_arrows(ax, edges, f"{variant}-{previous['family']}-edge")
    add_arrows(ax, loops, f"{variant}-{previous['family']}-self")
    nodes = ax.scatter(xy[:, 0], xy[:, 1], s=(2*node_radius)**2,
               c=(point_colors(previous["family"], cell["labels"], metadata)
                  if node_colors is None else node_colors),
               edgecolors="#53646c", linewidths=.35, zorder=3)
    nodes.set_gid(f"trait-nodes-{previous['family']}")
    for text, point in zip(texts, xy):
        ax.annotate("", xy=point, xytext=text.get_position(),
                    arrowprops={"arrowstyle": "-", "color": "#9ba6aa", "lw": .4,
                                "linestyle": ":", "shrinkA": 3, "shrinkB": 4}, zorder=2)
    return {"family": previous["family"], "loop_style": variant, "labels": cell["labels"],
            "coordinates": xy.tolist(), "node_screen_points": positions.tolist(),
            "node_radius_pt": node_radius, "bounds_points": bounds,
            "label_boxes_points": boxes, "edges": edges, "loops": loops}


@dataclass(frozen=True, kw_only=True)
class MagnitudeKey:
    """Shared loop/arrow legend entry (effect magnitude in nats)."""

    value: float


class MagnitudeHandler(HandlerBase):
    """Illustrate identical shaft width on one straight edge and circular arc."""

    def create_artists(self, legend: Any, handle: MagnitudeKey, xdescent: float,
                       ydescent: float, width: float, height: float, fontsize: float,
                       transform: Any) -> list[Any]:
        """Build both examples with the same line-width formula as the data."""
        y = height/2-ydescent
        line = plt.Line2D([0, width*.48], [y, y], transform=transform,
                          lw=WIDTH_SCALE*handle.value, alpha=ALPHA, color="#58646a")
        arc = MplPath.arc(35, 325)
        curve = MplPath(arc.vertices*3+np.array([width*.80, y]), arc.codes)
        loop = FancyArrowPatch(path=curve, transform=transform, arrowstyle="-|>",
                               mutation_scale=3.3, linewidth=WIDTH_SCALE*handle.value,
                               alpha=ALPHA, color="#58646a")
        return [line, loop]


def legend(
    fig: plt.Figure, *, paired: bool = False, positive_only: bool = False,
    negative_self_effects: bool = False, direction_character: bool = True,
    group_entries: list[tuple[str, str]] | None = None, group_columns: int = 6,
) -> None:
    """Give one loop-plus-line magnitude key for both families and all panels."""
    base = .036 if paired else .067
    groups = [plt.Line2D([], [], linestyle="", marker="o", markersize=5.2 if positive_only else 3.5,
                        color=ink, label=name)
              for name, ink in (zip(REGION_NAMES, REGION_COLORS)
                                if group_entries is None else group_entries)]
    fig.legend(handles=groups, loc="lower center", bbox_to_anchor=(.5, base),
               ncol=group_columns, fontsize=7.3, frameon=False, handletextpad=.35, columnspacing=1.6)
    keys = [plt.Line2D([], [], color=POSITIVE, lw=1,
                      label="Positive →" if direction_character else "Positive")]
    if not positive_only:
        keys.append(plt.Line2D([], [], color=NEGATIVE, lw=1, ls="--", label="Negative →"))
    elif negative_self_effects:
        keys.append(plt.Line2D([], [], color=NEGATIVE, lw=1, ls="--",
                              label="Negative self-effect"))
    labels = [key.get_label() for key in keys]
    for value in (4, 8, 12, 16):
        keys.append(MagnitudeKey(value=value))
        labels.append(f"{value} nats")
    fig.legend(keys, labels, handler_map={MagnitudeKey: MagnitudeHandler()},
               loc="lower center", bbox_to_anchor=(.5, .009), ncol=6, fontsize=7.3,
               frameon=False, handlelength=3.5, handletextpad=.6, columnspacing=1.5)


def build() -> None:
    """Export three self-loop designs without overwriting the clean original."""
    source = json.loads(SOURCE.read_text())
    clean = json.loads(CLEAN_MANIFEST.read_text())
    metadata = json.loads(METADATA.read_text())
    records = {}
    for variant in VARIANTS:
        paired = variant == "paired"
        fig, axes = plt.subplots(2 if paired else 1, 2, squeeze=False,
                                 figsize=(13, 10.5 if paired else 5.8), dpi=160)
        fig.subplots_adjust(left=.022, right=.985, bottom=.08 if paired else .145,
                            top=.96 if paired else .935, wspace=.12, hspace=.12)
        outputs = []
        for column, previous in enumerate(clean["panels"]):
            family = previous["family"]
            cell = source["cells"][f"{family}-00-99"]
            outputs.append(panel(axes[0, column], cell, previous, metadata,
                                 variant="external" if paired else variant,
                                 show_loops=not paired,
                                 title=f"{'A' if column == 0 else 'B'}  {family.title()}" +
                                 (" · cross-trait" if paired else "")))
            if paired:
                outputs.append(panel(axes[1, column], cell, previous, metadata,
                                     variant="external", show_edges=False,
                                     title=f"{'C' if column == 0 else 'D'}  {family.title()} · self-effects"))
        legend(fig, paired=paired)
        for extension in ("png", "svg", "pdf"):
            fig.savefig(OUT/f"pca_self_loops_{variant}.{extension}", dpi=260,
                        facecolor="white", bbox_inches="tight", pad_inches=.04)
        plt.close(fig)
        records[variant] = outputs
        logger.success("Rendered {} self-loop variant", variant)
    comparison, axes = plt.subplots(2, 1, figsize=(13, 11.7), dpi=160)
    for ax, variant, title in zip(axes, ("compact", "external"),
                                   ("Compact circular self-loop glyphs", "Outward-facing self-loops")):
        ax.imshow(plt.imread(OUT/f"pca_self_loops_{variant}.png"))
        ax.set_axis_off()
        ax.set_title(title, loc="left", fontsize=12, color=INK)
    comparison.subplots_adjust(left=.01, right=.99, top=.97, bottom=.01, hspace=.05)
    comparison.savefig(OUT/"pca_self_loop_comparison.png", dpi=200,
                       facecolor="white", bbox_inches="tight", pad_inches=.04)
    plt.close(comparison)
    manifest = {"source": str(SOURCE), "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                "clean_manifest_sha256": hashlib.sha256(CLEAN_MANIFEST.read_bytes()).hexdigest(),
                "width_points_per_nat": WIDTH_SCALE, "opacity": ALPHA,
                "cross_trait_threshold_nats": THRESHOLD, "self_effect_threshold": "none; all 30",
                "node_radius_pt": NODE_RADIUS, "compact_loop_radius_pt": LOOP_RADIUS,
                "policy": "Frozen PCA nodes. Fixed node/loop geometry; only orientation varies. All self/cross magnitudes use identical linear linewidth. No minimum linewidth or hidden clipping.",
                "variants": records}
    (OUT/"pca_self_loop_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    build()
