#!/usr/bin/env python3
"""Print a behavior-vector x dataset-vector metric matrix for one layer.

Rows are behavior concept vectors loaded from ``trained_{trait}_vector.npy`` by
default, or from layer-shaped per-layer behavior vectors when requested.
Columns are dataset concept vectors loaded from
``{trait}_v_teacher_by_layer.npy`` at the requested model layer.

By default, both vector families are used raw. Use ``--center-dataset`` and/or
``--center-behavior`` to subtract the across-trait mean vector before computing
cosine and projection metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


DEFAULT_ROOT = Path("results/in-context_vector")
DEFAULT_BEHAVIOR_DIR = DEFAULT_ROOT
DEFAULT_DATASET_VECTOR_DIR = (
    DEFAULT_ROOT / "dataset_concept_vectors_no_system_0_to_999_unified_format"
)
DEFAULT_TRAITS = ("eagle", "penguin", "panda", "raccoon", "giraffe", "frog")
DEFAULT_LAYER = 26


@dataclass(frozen=True, **({"kw_only": True} if sys.version_info >= (3, 10) else {}))
class CosineMatrixConfig:
    """Configuration for behavior-vs-dataset cosine matrix computation."""

    behavior_traits: list[str]
    dataset_traits: list[str]
    behavior_dir: Path
    dataset_vector_dir: Path
    layer: int
    behavior_vector_mode: str = "uniform"
    behavior_layer: int | None = None
    output_path: Path | None = None
    heatmap_path: Path | None = None
    heatmap_max_abs: float | None = None
    center_behavior: bool = False
    center_dataset: bool = False


def parse_string_list(
    values: list[str] | None,
    default: tuple[str, ...],
) -> list[str]:
    """Parse repeated and/or comma-separated CLI string values."""
    if values is None:
        return list(default)

    parsed: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in value.split(","):
            stripped = item.strip()
            if not stripped or stripped in seen:
                continue
            parsed.append(stripped)
            seen.add(stripped)

    if not parsed:
        raise ValueError("At least one trait is required.")
    return parsed


def projection_metrics(behavior: np.ndarray, dataset: np.ndarray) -> dict[str, float]:
    """Compute cosine and signed behavior-direction projection metrics."""
    b = np.asarray(behavior, dtype=np.float64).reshape(-1)
    d = np.asarray(dataset, dtype=np.float64).reshape(-1)

    if b.shape != d.shape:
        raise ValueError(f"Shape mismatch: {b.shape} != {d.shape}")

    behavior_norm = float(np.linalg.norm(b))
    dataset_norm = float(np.linalg.norm(d))
    if behavior_norm == 0.0 or dataset_norm == 0.0:
        raise ValueError("Cannot compute projection metrics with a zero-norm vector.")

    dot = float(np.dot(b, d))
    cosine_value = dot / (behavior_norm * dataset_norm)
    return {
        "dot": dot,
        "behavior_norm": behavior_norm,
        "dataset_norm": dataset_norm,
        "cosine": cosine_value,
        "projection_length": dot / behavior_norm,
        "alpha_equivalent": dot / (behavior_norm**2),
        "cosine_squared": cosine_value**2,
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Compute signed cosine similarity between two 1D vectors."""
    return projection_metrics(a, b)["cosine"]


def center_trait_vectors(
    vectors: dict[str, np.ndarray],
    traits: list[str],
) -> dict[str, np.ndarray]:
    """Center vectors by subtracting the across-trait mean vector."""
    try:
        matrix = np.stack(
            [
                np.asarray(vectors[trait], dtype=np.float64).reshape(-1)
                for trait in traits
            ],
            axis=0,
        )
    except ValueError as exc:
        raise ValueError("Vectors must share a shape before centering.") from exc

    centered = matrix - matrix.mean(axis=0, keepdims=True)
    return {trait: centered[index] for index, trait in enumerate(traits)}


def load_behavior_vector(
    behavior_dir: Path,
    trait: str,
    *,
    vector_mode: str,
    layer: int | None = None,
) -> np.ndarray:
    """Load one behavior concept vector."""
    if vector_mode == "uniform":
        vector_path = behavior_dir / f"trained_{trait}_vector.npy"
        if not vector_path.is_file():
            raise FileNotFoundError(f"Missing behavior vector: {vector_path}")
        return np.load(vector_path).reshape(-1)

    if vector_mode == "per_layer":
        if layer is None:
            raise ValueError("--behavior-layer is required for per_layer behavior.")
        vector_path = behavior_dir / f"trained_{trait}_per_layer_vector_by_layer.npy"
        metadata_path = behavior_dir / f"trained_{trait}_per_layer_vector.metadata.json"
        if not vector_path.is_file():
            raise FileNotFoundError(f"Missing per-layer behavior vector: {vector_path}")
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Missing per-layer behavior vector metadata: {metadata_path}"
            )

        vector_by_layer = np.load(vector_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model_layer_indices = metadata.get("layer_indices")
        if not isinstance(model_layer_indices, list):
            raise ValueError(
                f"{metadata_path} does not contain list field 'layer_indices'."
            )
        row_by_layer = {
            int(model_layer): row for row, model_layer in enumerate(model_layer_indices)
        }
        if layer not in row_by_layer:
            raise ValueError(
                f"{trait} behavior vector does not contain layer {layer}. "
                f"Available layers: {sorted(row_by_layer)}"
            )
        return np.asarray(vector_by_layer[row_by_layer[layer]]).reshape(-1)

    raise ValueError(f"Unsupported behavior vector mode: {vector_mode}")


def load_dataset_vector_at_layer(
    dataset_vector_dir: Path,
    trait: str,
    layer: int,
) -> np.ndarray:
    """Load one dataset concept vector at a model layer."""
    vector_path = dataset_vector_dir / f"{trait}_v_teacher_by_layer.npy"
    metadata_path = dataset_vector_dir / f"{trait}_v_teacher.metadata.json"

    if not vector_path.is_file():
        raise FileNotFoundError(f"Missing dataset vector: {vector_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing dataset vector metadata: {metadata_path}")

    vector_by_layer = np.load(vector_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_layer_indices = metadata.get("model_layer_indices")
    if not isinstance(model_layer_indices, list):
        raise ValueError(
            f"{metadata_path} does not contain list field 'model_layer_indices'."
        )

    row_by_layer = {
        int(model_layer): row for row, model_layer in enumerate(model_layer_indices)
    }

    if layer not in row_by_layer:
        raise ValueError(
            f"{trait} dataset vector does not contain layer {layer}. "
            f"Available layers: {sorted(row_by_layer)}"
        )

    return np.asarray(vector_by_layer[row_by_layer[layer]]).reshape(-1)


def compute_cosine_matrix(config: CosineMatrixConfig) -> dict[str, Any]:
    """Compute the behavior-trait x dataset-trait cosine matrix."""
    raw_behavior_vectors = {
        trait: load_behavior_vector(
            config.behavior_dir,
            trait,
            vector_mode=config.behavior_vector_mode,
            layer=config.behavior_layer,
        )
        for trait in config.behavior_traits
    }
    raw_dataset_vectors = {
        trait: load_dataset_vector_at_layer(
            config.dataset_vector_dir,
            trait,
            config.layer,
        )
        for trait in config.dataset_traits
    }
    behavior_vectors = (
        center_trait_vectors(raw_behavior_vectors, config.behavior_traits)
        if config.center_behavior
        else raw_behavior_vectors
    )
    dataset_vectors = (
        center_trait_vectors(raw_dataset_vectors, config.dataset_traits)
        if config.center_dataset
        else raw_dataset_vectors
    )

    matrix: dict[str, dict[str, float]] = {}
    projection_matrix: dict[str, dict[str, float]] = {}
    alpha_equivalent_matrix: dict[str, dict[str, float]] = {}
    dot_matrix: dict[str, dict[str, float]] = {}
    dataset_norm_matrix: dict[str, dict[str, float]] = {}
    behavior_norm_by_trait: dict[str, float] = {}
    dataset_norm_by_trait: dict[str, float] = {}
    pair_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for behavior_trait in config.behavior_traits:
        row: dict[str, float] = {}
        projection_row: dict[str, float] = {}
        alpha_row: dict[str, float] = {}
        dot_row: dict[str, float] = {}
        dataset_norm_row: dict[str, float] = {}
        metric_row: dict[str, dict[str, float]] = {}
        for dataset_trait in config.dataset_traits:
            metrics = projection_metrics(
                behavior_vectors[behavior_trait],
                dataset_vectors[dataset_trait],
            )
            row[dataset_trait] = metrics["cosine"]
            projection_row[dataset_trait] = metrics["projection_length"]
            alpha_row[dataset_trait] = metrics["alpha_equivalent"]
            dot_row[dataset_trait] = metrics["dot"]
            dataset_norm_row[dataset_trait] = metrics["dataset_norm"]
            metric_row[dataset_trait] = metrics
        matrix[behavior_trait] = row
        projection_matrix[behavior_trait] = projection_row
        alpha_equivalent_matrix[behavior_trait] = alpha_row
        dot_matrix[behavior_trait] = dot_row
        dataset_norm_matrix[behavior_trait] = dataset_norm_row
        pair_metrics[behavior_trait] = metric_row
        behavior_norm_by_trait[behavior_trait] = float(
            np.linalg.norm(behavior_vectors[behavior_trait])
        )
    for dataset_trait in config.dataset_traits:
        dataset_norm_by_trait[dataset_trait] = float(
            np.linalg.norm(dataset_vectors[dataset_trait])
        )

    diagonal = {
        trait: matrix[trait][trait]
        for trait in config.behavior_traits
        if trait in config.dataset_traits
    }
    diagonal_projection = {
        trait: projection_matrix[trait][trait]
        for trait in config.behavior_traits
        if trait in config.dataset_traits
    }
    diagonal_alpha_equivalent = {
        trait: alpha_equivalent_matrix[trait][trait]
        for trait in config.behavior_traits
        if trait in config.dataset_traits
    }
    diagonal_metrics = {
        trait: pair_metrics[trait][trait]
        for trait in config.behavior_traits
        if trait in config.dataset_traits
    }

    return {
        "layer": config.layer,
        "behavior_layer": config.behavior_layer,
        "behavior_dir": str(config.behavior_dir),
        "dataset_vector_dir": str(config.dataset_vector_dir),
        "behavior_traits": config.behavior_traits,
        "dataset_traits": config.dataset_traits,
        "behavior_input_vector_mode": config.behavior_vector_mode,
        "center_behavior": config.center_behavior,
        "center_dataset": config.center_dataset,
        "behavior_vector_mode": (
            "centered_across_behavior_traits" if config.center_behavior else "raw"
        ),
        "dataset_vector_mode": (
            "centered_across_dataset_traits" if config.center_dataset else "raw"
        ),
        "matrix": matrix,
        "projection_matrix": projection_matrix,
        "alpha_equivalent_matrix": alpha_equivalent_matrix,
        "dot_matrix": dot_matrix,
        "dataset_norm_matrix": dataset_norm_matrix,
        "behavior_norm_by_trait": behavior_norm_by_trait,
        "dataset_norm_by_trait": dataset_norm_by_trait,
        "pair_metrics": pair_metrics,
        "diagonal": diagonal,
        "diagonal_projection": diagonal_projection,
        "diagonal_alpha_equivalent": diagonal_alpha_equivalent,
        "diagonal_metrics": diagonal_metrics,
    }


def log_matrix(result: dict[str, Any]) -> None:
    """Log a compact matrix table."""
    behavior_traits = result["behavior_traits"]
    dataset_traits = result["dataset_traits"]
    matrix = result["matrix"]
    projection_matrix = result["projection_matrix"]
    behavior_mode = result.get("behavior_vector_mode", "raw")
    dataset_mode = result.get("dataset_vector_mode", "raw")

    logger.success(
        "Behavior vector x dataset vector cosine matrix at layer {} ({}, {})",
        result["layer"],
        behavior_mode,
        dataset_mode,
    )
    header = " " * 12 + " ".join(f"{trait:>10s}" for trait in dataset_traits)
    logger.success(header)

    for behavior_trait in behavior_traits:
        values = " ".join(
            f"{matrix[behavior_trait][dataset_trait]:+10.6f}"
            for dataset_trait in dataset_traits
        )
        logger.success("{:>10s}  {}", behavior_trait, values)

    if result["diagonal"]:
        logger.success("Diagonal matched-trait cosines:")
        for trait, value in result["diagonal"].items():
            logger.success("  {:>10s}: {:+.8f}", trait, value)

    logger.success(
        "Behavior-unit projection length matrix d_u · b_t/||b_t|| at layer {} "
        "({}, {})",
        result["layer"],
        behavior_mode,
        dataset_mode,
    )
    logger.success(header)
    for behavior_trait in behavior_traits:
        values = " ".join(
            f"{projection_matrix[behavior_trait][dataset_trait]:+10.6f}"
            for dataset_trait in dataset_traits
        )
        logger.success("{:>10s}  {}", behavior_trait, values)

    if result["diagonal_metrics"]:
        logger.success(
            "Diagonal matched-trait projection details "
            "(cosine, dataset_norm, projection_length, alpha_equivalent):"
        )
        for trait, metrics in result["diagonal_metrics"].items():
            logger.success(
                "  {:>10s}: cos={:+.8f} |d|={:.8g} proj={:+.8g} alpha_eq={:+.8g}",
                trait,
                metrics["cosine"],
                metrics["dataset_norm"],
                metrics["projection_length"],
                metrics["alpha_equivalent"],
            )


def centering_suffix(result: dict[str, Any]) -> str:
    """Return a filename suffix describing non-default centering modes."""
    parts = []
    if result.get("center_behavior"):
        parts.append("centered_behavior")
    if result.get("center_dataset"):
        parts.append("centered_dataset")
    if not parts:
        return ""
    return "_" + "_".join(parts)


def write_result(result: dict[str, Any], output_path: Path | None) -> Path:
    """Write the result JSON and return the path used."""
    resolved_output_path = (
        output_path
        if output_path is not None
        else DEFAULT_ROOT
        / (
            f"behavior_dataset_cosine_matrix_layer_{result['layer']}"
            f"{centering_suffix(result)}.json"
        )
    )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    logger.success("Saved cosine matrix to {}", resolved_output_path)
    return resolved_output_path


def default_heatmap_path(result: dict[str, Any]) -> Path:
    """Return the default heatmap path for a result."""
    return DEFAULT_ROOT / (
        f"behavior_dataset_cosine_matrix_layer_{result['layer']}"
        f"{centering_suffix(result)}.png"
    )


def matrix_to_array(result: dict[str, Any]) -> np.ndarray:
    """Convert a nested matrix dict to a row/column ordered array."""
    behavior_traits = result["behavior_traits"]
    dataset_traits = result["dataset_traits"]
    matrix = result["matrix"]
    return np.array(
        [
            [matrix[behavior_trait][dataset_trait] for dataset_trait in dataset_traits]
            for behavior_trait in behavior_traits
        ],
        dtype=np.float64,
    )


def resolve_heatmap_limits(
    values: np.ndarray,
    heatmap_max_abs: float | None = None,
) -> tuple[float, float, float]:
    """Return symmetric heatmap limits around zero.

    Cosine values are bounded by [-1, 1], but real matrices can have much
    smaller ranges. Auto-scaling to the observed absolute maximum makes weak
    structure visible while preserving sign symmetry.
    """
    if heatmap_max_abs is not None:
        if not np.isfinite(heatmap_max_abs) or heatmap_max_abs <= 0.0:
            raise ValueError("--heatmap-max-abs must be positive and finite.")
        max_abs = min(float(heatmap_max_abs), 1.0)
    elif values.size == 0:
        max_abs = 1.0
    else:
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            max_abs = 1.0
        else:
            max_abs = float(np.max(np.abs(finite_values)))
            if max_abs == 0.0:
                max_abs = 1.0
            max_abs = min(max_abs, 1.0)
    return -max_abs, max_abs, max_abs


def write_heatmap(
    result: dict[str, Any],
    heatmap_path: Path | None,
    heatmap_max_abs: float | None = None,
) -> Path:
    """Write a cosine matrix heatmap image and return the path used."""
    import matplotlib.pyplot as plt

    resolved_heatmap_path = heatmap_path or default_heatmap_path(result)
    resolved_heatmap_path.parent.mkdir(parents=True, exist_ok=True)

    behavior_traits = result["behavior_traits"]
    dataset_traits = result["dataset_traits"]
    values = matrix_to_array(result)
    vmin, vmax, max_abs = resolve_heatmap_limits(values, heatmap_max_abs)

    width = max(8.0, 0.48 * len(dataset_traits) + 3.0)
    height = max(6.0, 0.42 * len(behavior_traits) + 2.5)
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(values, cmap="coolwarm", vmin=vmin, vmax=vmax)
    ax.set_title(f"Behavior x dataset cosine, layer {result['layer']}")
    ax.set_xlabel("Dataset concept vector")
    ax.set_ylabel("Behavior concept vector")
    ax.set_xticks(np.arange(len(dataset_traits)))
    ax.set_xticklabels(dataset_traits, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(behavior_traits)))
    ax.set_yticklabels(behavior_traits)

    annotate = values.size <= 400
    if annotate:
        for row_index in range(values.shape[0]):
            for col_index in range(values.shape[1]):
                value = values[row_index, col_index]
                text_color = "white" if abs(value) >= 0.55 * max_abs else "black"
                ax.text(
                    col_index,
                    row_index,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=7,
                )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="cosine")
    fig.tight_layout()
    fig.savefig(resolved_heatmap_path, dpi=220)
    plt.close(fig)
    logger.success(
        "Saved cosine heatmap to {} with color range [{:+.6f}, {:+.6f}]",
        resolved_heatmap_path,
        vmin,
        vmax,
    )
    return resolved_heatmap_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute a behavior-vector x dataset-vector cosine matrix at one "
            "dataset-vector layer."
        )
    )
    parser.add_argument(
        "--trait",
        "--traits",
        dest="traits",
        action="append",
        help=(
            "Traits to use for both rows and columns. Repeat or pass "
            f"comma-separated values. Defaults to {','.join(DEFAULT_TRAITS)}."
        ),
    )
    parser.add_argument(
        "--behavior-trait",
        "--behavior-traits",
        dest="behavior_traits",
        action="append",
        help=(
            "Optional row traits. Defaults to --trait/--traits, or the script "
            "default trait list."
        ),
    )
    parser.add_argument(
        "--dataset-trait",
        "--dataset-traits",
        dest="dataset_traits",
        action="append",
        help=(
            "Optional column traits. Defaults to --trait/--traits, or the "
            "script default trait list."
        ),
    )
    parser.add_argument(
        "--behavior-dir",
        type=Path,
        default=DEFAULT_BEHAVIOR_DIR,
        help="Directory containing trained_{trait}_vector.npy files.",
    )
    parser.add_argument(
        "--dataset-vector-dir",
        type=Path,
        default=DEFAULT_DATASET_VECTOR_DIR,
        help="Directory containing {trait}_v_teacher_by_layer.npy files.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=DEFAULT_LAYER,
        help=f"Model layer to use from dataset vectors. Defaults to {DEFAULT_LAYER}.",
    )
    parser.add_argument(
        "--behavior-vector-mode",
        choices=["uniform", "per_layer"],
        default="uniform",
        help=(
            "Behavior vector source. uniform loads trained_{trait}_vector.npy; "
            "per_layer loads trained_{trait}_per_layer_vector_by_layer.npy."
        ),
    )
    parser.add_argument(
        "--behavior-layer",
        type=int,
        default=None,
        help=(
            "Model layer to use from per-layer behavior vectors. Defaults to "
            "--layer when --behavior-vector-mode=per_layer."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "JSON output path. Defaults to "
            "results/in-context_vector/behavior_dataset_cosine_matrix_layer_{layer}"
            "{centering_suffix}.json."
        ),
    )
    parser.add_argument(
        "--heatmap-path",
        type=Path,
        default=None,
        help=(
            "Optional PNG heatmap output path. When provided, saves a heatmap "
            "of the cosine matrix."
        ),
    )
    parser.add_argument(
        "--heatmap-max-abs",
        type=float,
        default=None,
        help=(
            "Optional symmetric heatmap color limit. By default, uses the "
            "observed max absolute cosine value in the matrix."
        ),
    )
    parser.add_argument(
        "--no-heatmap",
        action="store_true",
        help="Disable heatmap generation.",
    )
    parser.add_argument(
        "--center-behavior",
        action="store_true",
        help="Center behavior vectors across behavior traits before scoring.",
    )
    parser.add_argument(
        "--center-dataset",
        action="store_true",
        help="Center dataset vectors across dataset traits before scoring.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the cosine matrix CLI."""
    args = _parse_args()
    shared_traits = parse_string_list(args.traits, DEFAULT_TRAITS)
    behavior_traits = parse_string_list(args.behavior_traits, tuple(shared_traits))
    dataset_traits = parse_string_list(args.dataset_traits, tuple(shared_traits))

    config = CosineMatrixConfig(
        behavior_traits=behavior_traits,
        dataset_traits=dataset_traits,
        behavior_dir=args.behavior_dir,
        dataset_vector_dir=args.dataset_vector_dir,
        layer=args.layer,
        behavior_vector_mode=args.behavior_vector_mode,
        behavior_layer=(
            args.layer
            if args.behavior_vector_mode == "per_layer" and args.behavior_layer is None
            else args.behavior_layer
        ),
        output_path=args.output_path,
        heatmap_path=args.heatmap_path,
        heatmap_max_abs=args.heatmap_max_abs,
        center_behavior=args.center_behavior,
        center_dataset=args.center_dataset,
    )
    result = compute_cosine_matrix(config)
    log_matrix(result)
    write_result(result, config.output_path)
    if not args.no_heatmap:
        write_heatmap(result, config.heatmap_path, config.heatmap_max_abs)


if __name__ == "__main__":
    main()
