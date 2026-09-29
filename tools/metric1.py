#!/usr/bin/env python3
"""Compute current Metric 1 from an explicit, portable probability archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from loguru import logger

EPSILON = 1e-12
REQUIRED_ARRAYS = {"p0", "teachers", "labels", "prompt_ids", "carrier_labels"}
OPTIONAL_ARRAYS = {"weights", "sl"}


def sha256_file(path: Path) -> str:
    """Hash the source bytes without including a machine-specific path."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_array(value: np.ndarray, name: str) -> np.ndarray:
    """Require real numeric, finite input before converting to float64."""
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    array = array.astype(np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array


def identifiers(value: np.ndarray, count: int, name: str, *, unique: bool) -> list[str]:
    """Validate ordered Unicode identifiers without changing their order."""
    array = np.asarray(value)
    if array.shape != (count,) or array.dtype.kind != "U":
        raise ValueError(f"{name} must be a Unicode array with shape ({count},)")
    result = array.tolist()
    if any(not item.strip() for item in result):
        raise ValueError(f"{name} must contain nonempty identifiers")
    if unique and len(set(result)) != count:
        raise ValueError(f"{name} must contain unique identifiers")
    return result


def double_center(matrix: np.ndarray) -> np.ndarray:
    """Center the complete matrix across rows and columns."""
    return matrix - matrix.mean(axis=0) - matrix.mean(axis=1)[:, None] + matrix.mean()


def pearson(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    """Return Pearson correlation, representing undefined cases explicitly."""
    x, y = left.ravel(), right.ravel()
    if x.size < 2:
        return {"value": None, "undefined_reason": "Fewer than two selected entries"}
    x, y = x - x.mean(), y - y.mean()
    # Scaling avoids overflow when a finite SL matrix contains large values.
    xscale, yscale = np.max(np.abs(x)), np.max(np.abs(y))
    if xscale == 0 or yscale == 0:
        return {"value": None, "undefined_reason": "A selected matrix has zero variance"}
    if not np.isfinite(xscale) or not np.isfinite(yscale):
        raise ValueError("Correlation arithmetic overflowed")
    x, y = x / xscale, y / yscale
    value = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
    return {"value": float(np.clip(value, -1, 1)), "undefined_reason": None}


def compute_metric1(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Average within-prompt signature cosines under one shared prompt measure."""
    missing = REQUIRED_ARRAYS - arrays.keys()
    unexpected = arrays.keys() - REQUIRED_ARRAYS - OPTIONAL_ARRAYS
    if missing or unexpected:
        raise ValueError(f"Archive schema mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    p0 = numeric_array(arrays["p0"], "p0")
    teachers = numeric_array(arrays["teachers"], "teachers")
    if p0.ndim != 2 or teachers.ndim != 3 or teachers.shape[1:] != p0.shape:
        raise ValueError("Expected p0 [N,C] and teachers [T,N,C] with matching axes")
    if min(teachers.shape) < 1:
        raise ValueError("Teacher, prompt, and carrier axes must be nonempty")
    traits, prompts, carriers = teachers.shape
    labels = identifiers(arrays["labels"], traits, "labels", unique=True)
    prompt_ids = identifiers(arrays["prompt_ids"], prompts, "prompt_ids", unique=False)
    carrier_labels = identifiers(arrays["carrier_labels"], carriers, "carrier_labels", unique=True)
    for name, probabilities in (("p0", p0), ("teachers", teachers)):
        if np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError(f"{name} probabilities must lie in [0,1]")
        if not np.allclose(probabilities.sum(axis=-1), 1, atol=1e-5, rtol=0):
            raise ValueError(f"{name} probabilities must be normalized over carriers")
    weights = numeric_array(arrays.get("weights", np.ones(prompts)), "weights")
    if weights.shape != (prompts,) or np.any(weights < 0):
        raise ValueError("weights must have shape [N] and be nonnegative")
    with np.errstate(over="ignore"):
        total_weight = float(weights.sum())
    if not np.isfinite(total_weight) or total_weight <= 0:
        raise ValueError("weights must have a finite positive sum")
    active = weights > 0
    excluded = np.flatnonzero(~active).tolist()
    # Zero-weight occurrences are excluded explicitly; positive-weight rows
    # must all have defined signatures. The teacher set is never changed.
    ratio = np.log(teachers[:, active] + EPSILON) - np.log(p0[None, active] + EPSILON)
    signatures = np.sqrt(p0[None, active]) * (ratio - ratio.mean(axis=0, keepdims=True))
    norms = np.linalg.norm(signatures, axis=-1)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError("Metric 1 undefined: a positive-weight prompt has a zero or nonfinite signature norm")
    unit_signatures = signatures / norms[:, :, None]
    matrix = np.einsum("tnc,snc,n->ts", unit_signatures, unit_signatures, weights[active] / total_weight)
    matrix = (matrix + matrix.T) / 2
    result: dict[str, Any] = {
        "schema_version": 1,
        "metric": "Metric 1: mean of per-prompt cosines",
        "definition": {
            "epsilon": EPSILON,
            "signature": "sqrt(p0(x,c)) * (log((pz(x,c)+epsilon)/(p0(x,c)+epsilon)) - mean_r log((pr(x,c)+epsilon)/(p0(x,c)+epsilon)))",
            "aggregation": "Weighted arithmetic mean of carrier-vector cosines computed separately within each prompt occurrence",
            "teacher_centering": "Across every trait in labels, separately for each prompt and carrier",
            "zero_norm_policy": "Reject undefined signatures for any positive-weight prompt; never silently drop them",
            "probability_normalization_tolerance": {"atol": 1e-5, "rtol": 0},
        },
        "labels": labels,
        "carrier_labels": carrier_labels,
        "prompt_ids": prompt_ids,
        "trait_count": traits,
        "carrier_count": carriers,
        "prompt_occurrence_count": prompts,
        "unique_prompt_id_count": len(set(prompt_ids)),
        "positive_weight_prompt_count": int(active.sum()),
        "weighting": {
            "definition": "One shared weight vector for every teacher; repeated prompt IDs remain separate occurrences",
            "source": "provided weights" if "weights" in arrays else "equal weight per prompt occurrence",
            "weights": weights.tolist(),
            "normalized_weights": (weights / total_weight).tolist(),
            "total_weight": total_weight,
            "zero_weight_excluded_indices": excluded,
            "zero_weight_excluded_prompt_ids": [prompt_ids[index] for index in excluded],
        },
        "mean_prompt_cosine": matrix.tolist(),
    }
    if "sl" in arrays:
        sl = numeric_array(arrays["sl"], "sl")
        if sl.shape != (traits, traits):
            raise ValueError("sl must have shape [T,T] in exactly the order of labels")
        # Rescaling by a positive constant leaves both double-centering and
        # Pearson correlations unchanged and keeps finite inputs well behaved.
        sl_scale = float(np.max(np.abs(sl)))
        centered_metric = double_center(matrix)
        centered_sl = double_center(sl / sl_scale if sl_scale else sl)
        off_diagonal = ~np.eye(traits, dtype=bool)
        result["sl_correlations"] = {
            "definition": "Double-center both complete matrices before selecting entries; Pearson correlation",
            "ordered_axes": "Both SL axes must follow labels; the input producer is responsible for their semantic alignment",
            "all_entries": {"entry_count": traits * traits, **pearson(centered_metric, centered_sl)},
            "off_diagonal_entries": {
                "entry_count": traits * (traits - 1),
                **pearson(centered_metric[off_diagonal], centered_sl[off_diagonal]),
            },
        }
    return result


def demo_arrays() -> dict[str, np.ndarray]:
    """Create deterministic synthetic data, never presented as paper results."""
    rng = np.random.default_rng(2026)
    return {
        "p0": rng.dirichlet(np.ones(5), size=4),
        "teachers": rng.dirichlet(np.ones(5), size=(3, 4)),
        "labels": np.array(["synthetic-a", "synthetic-b", "synthetic-c"]),
        "prompt_ids": np.array([f"synthetic-prompt-{index}" for index in range(4)]),
        "carrier_labels": np.array([f"synthetic-carrier-{index}" for index in range(5)]),
        "sl": rng.normal(size=(3, 3)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Load a safe NPZ archive and write current Metric 1 with provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="NPZ with p0, teachers, labels, prompt_ids, carrier_labels; optional weights and sl")
    source.add_argument("--demo", action="store_true", help="Use deterministic synthetic data, not paper results")
    parser.add_argument("--output", type=Path, default=Path("outputs/metric1.json"))
    args = parser.parse_args(argv)
    try:
        if args.demo:
            arrays = demo_arrays()
            encoded = json.dumps({key: value.tolist() for key, value in sorted(arrays.items())}, separators=(",", ":"), allow_nan=False).encode()
            source_provenance = {"kind": "synthetic_demo", "sha256": hashlib.sha256(encoded).hexdigest(), "hash_encoding": "UTF-8 compact JSON of sorted array names and their lists", "is_paper_result": False}
        else:
            if args.input.resolve() == args.output.resolve():
                raise ValueError("Input and output must be different files")
            source_hash = sha256_file(args.input)
            with np.load(args.input, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            if sha256_file(args.input) != source_hash:
                raise ValueError("Input archive changed while it was being read")
            source_provenance = {"kind": "npz", "sha256": source_hash, "hash_encoding": "Original NPZ file bytes"}
        result = compute_metric1(arrays)
        result["source"] = source_provenance
        result["implementation_sha256"] = sha256_file(Path(__file__))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (ValueError, OSError, TypeError) as error:
        logger.error("Metric 1 computation failed: {}", error)
        return 2
    logger.success("Saved Metric 1 for {} prompt occurrences to {}", result["prompt_occurrence_count"], args.output)
    if args.demo:
        logger.info("This output uses synthetic demo data and is not a paper result")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
