"""Deterministic bounded quadratic routing on unchanged, physical-point nodes.

This is a finite-candidate coordinate search, not a planar-layout algorithm. It
keeps every directed edge and its weight; dense fixed-node graphs can still have
crossings. Rectangle arguments always use ``(xmin, xmax, ymin, ymax)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger

Edge = dict[str, Any]
Bounds = tuple[float, float, float, float]


@dataclass(kw_only=True, frozen=True)
class RoutingConfig:
    """Search settings and physical-point penalties for fixed-node routing."""

    samples: int = 17
    sweeps: int = 2
    max_bend: float = 105.0
    node_clearance: float = 1.8
    label_clearance: float = 1.1
    overlap_clearance: float = 1.7
    node_penalty: float = 210.0
    label_penalty: float = 95.0
    crossing_penalty: float = 45.0
    overlap_penalty: float = 55.0
    length_penalty: float = 0.12
    bend_penalty: float = 0.035


def _validate(
    positions: np.ndarray, radii: np.ndarray, edges: list[Edge], bounds: Bounds,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate geometry without copying or modifying the input edge records."""
    points = np.asarray(positions, dtype=float)
    sizes = np.asarray(radii, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or sizes.shape != (len(points),):
        raise ValueError("positions must be N×2 and radii must have length N")
    if not np.isfinite(points).all() or not np.isfinite(sizes).all() or (sizes < 0).any():
        raise ValueError("node positions and nonnegative radii must be finite")
    xmin, xmax, ymin, ymax = bounds
    if not np.isfinite(bounds).all() or xmin >= xmax or ymin >= ymax:
        raise ValueError("bounds must be finite (xmin, xmax, ymin, ymax)")
    lower, upper = np.array([xmin, ymin]), np.array([xmax, ymax])
    if ((points - sizes[:, None] < lower) | (points + sizes[:, None] > upper)).any():
        raise ValueError("bounds must contain every node disk")
    for edge in edges:
        source, target = edge["source_index"], edge["target_index"]
        if not 0 <= source < len(points) or not 0 <= target < len(points):
            raise ValueError("edge node indices must address positions")
        if source == target or np.linalg.norm(points[target] - points[source]) < 1e-10:
            raise ValueError("quadratic routing requires distinct source/target positions")
        if not np.isfinite(edge["weight"]):
            raise ValueError("edge weights must be finite")
    return points, sizes


def _sample(vertices: np.ndarray, count: int) -> np.ndarray:
    """Sample one or several quadratic Beziers, including both endpoints."""
    t = np.linspace(0.0, 1.0, count)
    basis = np.stack(((1 - t) ** 2, 2 * t * (1 - t), t**2), axis=1)
    return np.einsum("sk,...kd->...sd", basis, vertices)


def _bend_interval(
    start: np.ndarray, end: np.ndarray, bounds: Bounds,
) -> tuple[float, float, float]:
    """Return the chord length and control-point bend interval inside bounds."""
    delta = end - start
    length = float(np.linalg.norm(delta))
    normal = np.array([-delta[1], delta[0]]) / length
    midpoint = (start + end) / 2
    low, high = -np.inf, np.inf
    for center, direction, minimum, maximum in zip(
        midpoint, normal, (bounds[0], bounds[2]), (bounds[1], bounds[3]),
    ):
        if abs(direction) > 1e-12:
            limits = sorted(((minimum - center) / direction, (maximum - center) / direction))
            low, high = max(low, limits[0]), min(high, limits[1])
    return length, low, high


def _vertices(
    start: np.ndarray, end: np.ndarray, radius_start: float, radius_end: float,
    bends: np.ndarray,
) -> np.ndarray:
    """Clip endpoints radially to node disks while retaining their direction."""
    delta = end - start
    normal = np.array([-delta[1], delta[0]]) / np.linalg.norm(delta)
    control = (start + end) / 2 + bends[:, None] * normal
    first = control - start
    last = control - end
    first = start + radius_start * first / np.linalg.norm(first, axis=1)[:, None]
    last = end + radius_end * last / np.linalg.norm(last, axis=1)[:, None]
    return np.stack((first, control, last), axis=1)


def _base_bend(length: float, reciprocal: bool) -> float:
    """Reproduce the previous renderer's modest left bend and short-chord rule."""
    return max((0.1 if reciprocal else 0.055) * length, 20.0 if length < 28 else 0.0)


def _route(edge: Edge, vertices: np.ndarray, bend: float, samples: int) -> Edge:
    """Copy an edge record, adding geometry without changing its metadata."""
    return {**edge, "vertices": vertices.copy(), "samples": _sample(vertices, samples),
            "bend": float(bend)}


def baseline_routes(
    positions: np.ndarray, radii: np.ndarray, edges: list[Edge], bounds: Bounds,
    *, config: RoutingConfig | None = None,
) -> list[Edge]:
    """Return bounded original-curvature routes as a reproducible comparison."""
    config = config or RoutingConfig()
    points, sizes = _validate(positions, radii, edges, bounds)
    pairs = {(e["source_index"], e["target_index"]) for e in edges}
    result = []
    for edge in edges:
        source, target = edge["source_index"], edge["target_index"]
        length, low, high = _bend_interval(points[source], points[target], bounds)
        bend = float(np.clip(_base_bend(length, (target, source) in pairs), low, high))
        vertices = _vertices(points[source], points[target], sizes[source], sizes[target],
                             np.array([bend]))[0]
        result.append(_route(edge, vertices, bend, config.samples))
    return result


def _static_metrics(
    samples: np.ndarray, source: int, target: int, positions: np.ndarray,
    radii: np.ndarray, label_boxes: np.ndarray, config: RoutingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count unrelated disk/label intersections and polyline arc lengths."""
    first, segment = samples[:, :-1], np.diff(samples, axis=1)
    norm2 = np.maximum(np.sum(segment**2, axis=-1), 1e-15)
    relative = positions[None, None] - first[:, :, None]
    t = np.clip(np.sum(relative * segment[:, :, None], axis=-1) / norm2[:, :, None], 0, 1)
    distances = np.linalg.norm(relative - t[..., None] * segment[:, :, None], axis=-1)
    hit = (distances < radii[None, None] + config.node_clearance).any(axis=1)
    hit[:, [source, target]] = False
    node_hits = hit.sum(axis=1)
    label_hits = np.zeros(len(samples), dtype=int)
    if len(label_boxes):
        lower = label_boxes[:, [0, 2]] - config.label_clearance
        upper = label_boxes[:, [1, 3]] + config.label_clearance
        # Liang–Barsky slabs detect small labels between sampled vertices too.
        origin = first[:, :, None, :]
        direction = segment[:, :, None, :]
        moving = np.abs(direction) > 1e-12
        divisor = np.where(moving, direction, 1.0)
        a, b = (lower - origin) / divisor, (upper - origin) / divisor
        enters = np.where(moving, np.minimum(a, b), -np.inf)
        leaves = np.where(moving, np.maximum(a, b), np.inf)
        parallel_outside = (~moving & ((origin < lower) | (origin > upper))).any(axis=-1)
        intersects = ((np.maximum(enters.max(axis=-1), 0)
                       <= np.minimum(leaves.min(axis=-1), 1)) & ~parallel_outside)
        label_hits = intersects.any(axis=1).sum(axis=1)
    return node_hits, label_hits, np.linalg.norm(segment, axis=-1).sum(axis=1)


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the two-dimensional vector cross product with broadcasting."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _pair_metrics(
    candidates: np.ndarray, others: np.ndarray, width: float,
    other_widths: np.ndarray, config: RoutingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Count pairs with interior crossings or close, substantially parallel shafts.

    Each edge pair contributes at most once to each count. Endpoint contacts and
    merely adjacent continuations are excluded; opposing directions are parallel
    for overlap purposes. All computations operate on the sampled polylines.
    """
    if len(others) == 0:
        zeros = np.zeros(len(candidates), dtype=int)
        return zeros, zeros.copy()
    first = candidates[:, :-1, None, None, :]
    delta = np.diff(candidates, axis=1)[:, :, None, None, :]
    second = others[None, None, :, :-1, :]
    other_delta = np.diff(others, axis=1)[None, None, :, :, :]
    relative = second - first
    denominator = _cross(delta, other_delta)
    nonparallel = np.abs(denominator) > 1e-9
    denominator = np.where(nonparallel, denominator, 1.0)
    along = _cross(relative, other_delta) / denominator
    other_along = _cross(relative, delta) / denominator
    crossing = nonparallel & (along >= -1e-7) & (along <= 1 + 1e-7)
    crossing &= (other_along >= -1e-7) & (other_along <= 1 + 1e-7)
    # Keep crossings at internal sample vertices, but omit actual route endpoints.
    progress = along + np.arange(candidates.shape[1] - 1)[None, :, None, None]
    other_progress = other_along + np.arange(others.shape[1] - 1)[None, None, None, :]
    crossing &= (progress > 1e-7) & (progress < candidates.shape[1] - 1 - 1e-7)
    crossing &= (other_progress > 1e-7) & (other_progress < others.shape[1] - 1 - 1e-7)
    # Dot products give perpendicular separation plus true longitudinal overlap.
    length = np.maximum(np.linalg.norm(delta, axis=-1), 1e-10)
    other_length = np.maximum(np.linalg.norm(other_delta, axis=-1), 1e-10)
    dot = np.sum(delta * other_delta, axis=-1)
    parallel = np.abs(dot) / (length * other_length) > 0.94
    projection_a = np.sum(relative * delta, axis=-1) / length
    projection_b = projection_a + dot / length
    common = (np.minimum(np.maximum(projection_a, projection_b), length)
              - np.maximum(np.minimum(projection_a, projection_b), 0))
    reverse_a = -np.sum(relative * other_delta, axis=-1) / other_length
    reverse_b = reverse_a + dot / other_length
    reverse_common = (np.minimum(np.maximum(reverse_a, reverse_b), other_length)
                      - np.maximum(np.minimum(reverse_a, reverse_b), 0))
    distance_a = np.abs(_cross(relative, delta)) / length
    distance_b = np.abs(_cross(relative + other_delta, delta)) / length
    distance_c = np.abs(_cross(relative, other_delta)) / other_length
    distance_d = np.abs(_cross(delta - relative, other_delta)) / other_length
    clearance = config.overlap_clearance + (width + other_widths[None, None, :, None]) / 2
    distance = np.minimum(np.minimum(distance_a, distance_b), np.minimum(distance_c, distance_d))
    overlap = parallel & (common > 1.0) & (reverse_common > 1.0) & (distance < clearance)
    return crossing.any(axis=(1, 3)).sum(axis=1), overlap.any(axis=(1, 3)).sum(axis=1)


def _boxes(label_boxes: list[Bounds] | np.ndarray | None) -> np.ndarray:
    """Normalize optional rectangle obstacles and reject ambiguous shapes."""
    result = np.asarray([] if label_boxes is None else label_boxes, dtype=float)
    if not result.size:
        return np.empty((0, 4), dtype=float)
    if result.ndim != 2 or result.shape[1] != 4 or not np.isfinite(result).all():
        raise ValueError("label_boxes must be finite (xmin, xmax, ymin, ymax) rectangles")
    if (result[:, 0] > result[:, 1]).any() or (result[:, 2] > result[:, 3]).any():
        raise ValueError("label box minima must not exceed maxima")
    return result


def _static_score(
    node_hits: np.ndarray, label_hits: np.ndarray, lengths: np.ndarray,
    bends: np.ndarray, config: RoutingConfig,
) -> np.ndarray:
    """Combine obstacle penalties with compactness and curvature preferences."""
    return (config.node_penalty * node_hits + config.label_penalty * label_hits
            + config.length_penalty * lengths + config.bend_penalty * np.abs(bends))


def routing_metrics(
    routes: list[Edge], positions: np.ndarray, radii: np.ndarray,
    label_boxes: list[Bounds] | np.ndarray | None = None,
    *, config: RoutingConfig | None = None,
) -> dict[str, int | float]:
    """Report comparable approximate geometry counts and the search objective."""
    config = config or RoutingConfig()
    boxes = _boxes(label_boxes)
    node_hits = label_hits = crossings = overlaps = 0
    total_length = objective = 0.0
    samples = np.asarray([_sample(np.asarray(r["vertices"]), config.samples) for r in routes])
    widths = np.asarray([0.16 * abs(r["weight"]) for r in routes])
    for index, route in enumerate(routes):
        nodes, labels, lengths = _static_metrics(
            samples[index:index + 1], route["source_index"], route["target_index"],
            np.asarray(positions), np.asarray(radii), boxes, config,
        )
        cross, near = _pair_metrics(samples[index:index + 1], samples[index + 1:],
                                    widths[index], widths[index + 1:], config)
        node_hits += int(nodes[0])
        label_hits += int(labels[0])
        crossings += int(cross[0])
        overlaps += int(near[0])
        total_length += float(lengths[0])
        objective += float(_static_score(nodes, labels, lengths,
                                         np.array([route.get("bend", 0)]), config)[0])
    objective += config.crossing_penalty * crossings + config.overlap_penalty * overlaps
    return {"crossings": crossings, "near_overlaps": overlaps, "node_hits": node_hits,
            "label_hits": label_hits, "total_length": total_length, "objective": objective}


def route_edges(
    positions: np.ndarray, radii: np.ndarray, edges: list[Edge], bounds: Bounds,
    label_boxes: list[Bounds] | np.ndarray | None = None,
    *, config: RoutingConfig | None = None,
) -> list[Edge]:
    """Choose compact obstacle-aware arcs while fixing all nodes and edge data.

    Inputs are physical typographic points, including radii and rectangle bounds.
    Bounds must contain the node disks. Output order and all input fields are
    preserved; ``vertices`` is a 3×2 quadratic from source to target, ``samples``
    is its sampled shaft, and ``bend`` is the signed control-point displacement.
    Reciprocal arrows use positive bends in each direction's local left normal.
    The original bend is always a candidate, so coordinate updates cannot worsen
    the stated objective. This heuristic does not promise zero crossings.
    """
    config = config or RoutingConfig()
    if config.samples < 3 or config.sweeps < 0 or config.max_bend <= 0:
        raise ValueError("routing needs at least 3 samples, nonnegative sweeps and a positive bend cap")
    points, sizes = _validate(positions, radii, edges, bounds)
    boxes = _boxes(label_boxes)
    if not edges:
        return []
    pairs = {(edge["source_index"], edge["target_index"]) for edge in edges}
    banks = []
    for edge in edges:
        source, target = edge["source_index"], edge["target_index"]
        length, low, high = _bend_interval(points[source], points[target], bounds)
        reciprocal = (target, source) in pairs
        baseline = float(np.clip(_base_bend(length, reciprocal), low, high))
        cap = min(config.max_bend, 0.65 * length + 22)
        magnitude = np.unique(np.concatenate((length * np.array([.035, .07, .12, .2, .32, .5, .7]),
                                               np.array([6., 12., 20., 32., 48., 72., 96.]))))
        values = magnitude if reciprocal else np.concatenate(([0.], magnitude, -magnitude))
        values = np.clip(values, max(low, -cap), min(high, cap))
        if reciprocal:
            values = values[values > 1e-8]
        # Index zero is the previous drawing, also the stable tie-breaking winner.
        bends = np.concatenate(([baseline], np.unique(values[np.abs(values - baseline) > 1e-9])))
        vertices = _vertices(points[source], points[target], sizes[source], sizes[target], bends)
        samples = _sample(vertices, config.samples)
        nodes, labels, lengths = _static_metrics(samples, source, target, points, sizes, boxes, config)
        scores = _static_score(nodes, labels, lengths, bends, config)
        banks.append((bends, vertices, samples, scores))
    chosen = np.zeros(len(edges), dtype=int)
    current = np.asarray([bank[2][0] for bank in banks])
    widths = np.asarray([0.16 * abs(edge["weight"]) for edge in edges])
    order = sorted(range(len(edges)), key=lambda i: (-abs(edges[i]["weight"]), i))
    for _ in range(config.sweeps):
        updates = 0
        for index in order:
            keep = np.arange(len(edges)) != index
            bank = banks[index]
            crossing, overlap = _pair_metrics(bank[2], current[keep], widths[index], widths[keep], config)
            scores = bank[3] + config.crossing_penalty * crossing + config.overlap_penalty * overlap
            best = int(np.argmin(scores))
            if scores[best] < scores[chosen[index]] - 1e-9:
                chosen[index] = best
                current[index] = bank[2][best]
                updates += 1
        if updates == 0:
            break
    logger.debug("Routed {} directed edges with {} bounded coordinate sweeps", len(edges), config.sweeps)
    return [_route(edge, bank[1][choice], bank[0][choice], config.samples)
            for edge, bank, choice in zip(edges, banks, chosen)]
