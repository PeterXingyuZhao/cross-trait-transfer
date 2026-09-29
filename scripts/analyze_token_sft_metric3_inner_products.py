"""CPU-only Metric 3 inner products; never normalize them into cosines."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from loguru import logger


def compute_inner_products(
    neutral: np.ndarray,
    teachers: np.ndarray,
    behavior: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    epsilon: float = 1e-12,
) -> dict[str, np.ndarray]:
    """Return answer-by-teacher matrices for three fixed-behavior variants.

    Inputs are N-by-K neutral probabilities, T-by-N-by-K teacher
    probabilities, and U-by-K mean answer probabilities. The same N prompt
    weights apply to every teacher. ``g`` is logit(behavior), clipped exactly
    as in the historical behavior-gain implementation. The tilde signature
    is sqrt(p0) times the across-teacher-centered additive log ratio; neither
    it nor g is unit normalized. Only the carrier-drift operand changes.
    """
    p0, pz, e = [np.asarray(v, dtype=np.float64)
                 for v in (neutral, teachers, behavior)]
    if p0.ndim != 2 or pz.ndim != 3 or e.ndim != 2:
        raise ValueError("Expected N×K, T×N×K, and U×K arrays")
    if (min(p0.shape) < 1 or min(pz.shape) < 1 or min(e.shape) < 1
            or pz.shape[1:] != p0.shape or e.shape[1] != p0.shape[1]):
        raise ValueError("Empty or mismatched axes")
    if not np.isfinite(epsilon) or not 0 < epsilon < .5:
        raise ValueError("epsilon must lie strictly between zero and one half")
    for value in (p0, pz, e):
        if not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
            raise ValueError("Expected finite probabilities in [0,1]")
    for value in (p0, pz):
        if not np.allclose(value.sum(-1), 1, atol=1e-10, rtol=0):
            raise ValueError("Carrier probabilities must be normalized")
    w = np.ones(len(p0)) if weights is None else np.asarray(weights, dtype=float)
    if (w.shape != (len(p0),) or not np.isfinite(w).all()
            or np.any(w < 0) or not np.isfinite(w.sum()) or w.sum() <= 0):
        raise ValueError("Expected a common nonnegative, nonzero prompt measure")
    w = w / w.sum()
    clipped = np.clip(e, epsilon, 1 - epsilon)
    g = np.log(clipped) - np.log1p(-clipped)
    shift = pz - p0[None]
    lam = np.log(pz + epsilon) - np.log(p0[None] + epsilon)
    tilted = np.sqrt(p0)[None] * (lam - lam.mean(0, keepdims=True))
    variants = {"probability_shift": shift, "lambda": lam, "lambda_tilde": tilted}
    output = {name: g @ np.einsum("tnk,n->tk", value, w).T
              for name, value in variants.items()}
    # A teacher-independent subtraction is annihilated by downstream DC.
    output["sqrt_neutral_lambda_uncentered_diagnostic"] = (
        g @ np.einsum("tnk,n->tk", np.sqrt(p0)[None] * lam, w).T
    )
    if not all(np.isfinite(value).all() for value in output.values()):
        raise ValueError("Nonfinite inner product")
    return output


def double_center(values: np.ndarray) -> np.ndarray:
    """Center the complete answer-by-teacher matrix before selecting entries."""
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 2 or not a.size or not np.isfinite(a).all():
        raise ValueError("Expected a nonempty finite matrix")
    return a - a.mean(0, keepdims=True) - a.mean(1, keepdims=True) + a.mean()


def compare_with_sl(metric: np.ndarray, sl: np.ndarray) -> dict[str, float | None]:
    """Return DC Pearson all-entry/off-diagonal scores, None if undefined."""
    left, right = double_center(metric), double_center(sl)
    if left.shape != right.shape or left.shape[0] != left.shape[1]:
        raise ValueError("Expected matched square trait matrices")
    masks = {"all900": np.ones(left.shape, dtype=bool),
             "off870": ~np.eye(len(left), dtype=bool)}
    result = {}
    for name, mask in masks.items():
        x, y = left[mask], right[mask]
        x, y = x - x.mean(), y - y.mean()
        denominator = np.linalg.norm(x) * np.linalg.norm(y)
        result[f"pearson_{name}"] = (
            None if denominator == 0 else float(np.dot(x, y) / denominator)
        )
    return result


def file_hash(path: Path) -> str:
    """Bind a scientific input or analysis script to its exact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_behavior(directory: Path, labels: list[str]) -> tuple[np.ndarray, dict]:
    """Validate the complete fixed-g cache, its scientific identity and catalog."""
    paths = [directory / name for name in ("behavior_scores.json", "manifest.json")]
    behavior, manifest = [json.loads(path.read_text()) for path in paths]
    cfg, identity = manifest["config"], behavior["cache_identity"]
    carriers = [f"{i:02d}" for i in range(100)]
    if (manifest["status"] != "completed" or manifest["prompt_count"] != 1024
            or cfg["model_id"] != "Qwen/Qwen2.5-7B-Instruct"
            or cfg["torch_dtype"] != "float32" or cfg["epsilon"] != 1e-12
            or cfg["traits"] != labels or cfg["answers"] != labels
            or behavior["answers"] != labels or behavior["numbers"] != carriers
            or behavior["carriers"] != carriers
            or len(behavior["questions"]) != 10
            or len(behavior["completed_numbers"]) != 100
            or set(behavior["completed_numbers"]) != set(carriers)
            or set(behavior["mean_raw_probability_by_number"]) != set(carriers)):
        raise ValueError("Behavior cache is not the complete matched numeric SFT setting")
    expected = {
        "model_id": cfg["model_id"], "answers": labels,
        "questions": behavior["questions"], "carriers": carriers,
        "carrier_system_prompt_template": "You love {carrier}.",
        "probability_mode": "raw_complete_answer_probability",
        "enable_thinking": False,
        "carrier_manifest_sha256": manifest["carrier_manifest_sha256"],
    }
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("Fixed behavior-score scientific identity mismatch")
    e = np.asarray([behavior["mean_raw_probability_by_number"][c]
                    for c in carriers], dtype=float).T
    if e.shape != (len(labels), 100):
        raise ValueError("Behavior answer/carrier axes do not match")
    return e, {"files": {str(p): file_hash(p) for p in paths},
               "cache_identity": identity, "source_config": cfg,
               "clipped_logit_entries": int(np.count_nonzero((e < 1e-12) | (e > 1-1e-12)))}


def main() -> None:
    """Preserve source outputs and publish separately labeled derived matrices."""
    from scripts.analyze_token_sft_metric1_promptwise import load_original, SNAPSHOT
    from scripts.analyze_metric3_logratio_cosine import read_matrix

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", nargs="+", default=["color", "animal", "city", "tree"])
    parser.add_argument("--input-root", type=Path,
                        default=Path("results/analysis_inputs/token_sft_metric3_inner_products_20260924"))
    parser.add_argument("--output", type=Path,
                        default=Path("results_for_presentation/qwen25_token_sft_metric3_inner_products_20260924/analysis.json"))
    args = parser.parse_args()
    if len(set(args.domains)) != len(args.domains) or set(args.domains) - {"color", "animal", "city", "tree"}:
        raise ValueError("Choose distinct audited domains")
    report = {
        "model": "Qwen/Qwen2.5-7B-Instruct", "training": "00-99 token SFT",
        "epsilon": 1e-12, "prompt_weighting": "equal weight per shared occurrence",
        "metric3_prompt_count": 1024,
        "definitions": {
            "g": "logit(mean_q P(complete answer | You love c., q)); clamp e to [epsilon,1-epsilon]",
            "probability_shift": "E_x sum_c (p_z-p_0)*g_zprime",
            "lambda": "E_x sum_c log((p_z+epsilon)/(p_0+epsilon))*g_zprime",
            "lambda_tilde": "E_x sum_c sqrt(p_0)*(lambda_z-mean_r lambda_r)*g_zprime",
            "normalization": "ordinary inner products; no unit normalization; g unchanged in every variant",
        },
        "matrix_orientation": "answer rows, teacher columns",
        "comparison": "Pearson after independently double-centering complete matrices before any diagonal exclusion",
        "analysis_source_sha256": file_hash(Path(__file__)), "cells": {},
    }
    for domain in args.domains:
        logger.info("Computing {} from validated saved scores", domain)
        if domain == "tree":
            from scripts.load_tree_token_sft_reanalysis import load_tree
            labels, probabilities, sl, audit = load_tree(args.input_root / "tree")
            old_path = args.input_root / "tree/metric_3/matrices/behavior_gain.csv"
        else:
            labels, probabilities, _, sl, audit = load_original(domain)
            old_path = (SNAPSHOT / audit["source_run"]
                        / "number_token_sft_metrics_v2/metric_3/matrices/behavior_gain.csv")
        # Tree has 10,000 channel rows, but its executed M3 uses the first 1,024.
        # Preserve that exact population for all three M3 operands.
        p = probabilities[:, :1024]
        if p.shape != (31, 1024, 100):
            raise ValueError("Expected 30 teachers plus neutral on 1,024 shared occurrences")
        behavior, behavior_audit = load_behavior(args.input_root / domain / "metric_3", labels)
        values = compute_inner_products(p[0], p[1:], behavior)
        original = read_matrix(old_path, labels)
        error = float(np.max(np.abs(values["probability_shift"] - original)))
        np.testing.assert_allclose(values["probability_shift"], original, rtol=1e-9, atol=1e-10)
        center_error = float(np.max(np.abs(
            double_center(values["lambda_tilde"])
            - double_center(values["sqrt_neutral_lambda_uncentered_diagnostic"]))))
        np.testing.assert_allclose(double_center(values["lambda_tilde"]),
                                   double_center(values["sqrt_neutral_lambda_uncentered_diagnostic"]),
                                   rtol=1e-11, atol=1e-11)
        metrics = {name: {**compare_with_sl(values[name], sl),
                          "matrix": values[name].tolist(),
                          "double_centered_matrix": double_center(values[name]).tolist()}
                   for name in ("probability_shift", "lambda", "lambda_tilde")}
        report["cells"][domain] = {
            "labels": labels, "channel_audit": audit, "behavior_audit": behavior_audit,
            "original_behavior_gain_source": str(old_path),
            "original_behavior_gain_sha256": file_hash(old_path),
            "original_behavior_gain_max_error": error,
            "teacher_centering_after_dc_max_error": center_error,
            "sl_matrix": sl.tolist(), "metrics": metrics,
        }
        for name, record in metrics.items():
            logger.success("{} {}: all900={:.6f}, off870={:.6f}; original error={:.2e}",
                           domain, name, record["pearson_all900"], record["pearson_off870"], error)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    logger.success("Saved separate analysis and source provenance to {}", args.output)


if __name__ == "__main__":
    main()
