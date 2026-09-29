"""Compare flattened and mean-prompt Metric 1 on the frozen eight-cell table.

Only aggregation changes. Carrier weighting, across-teacher centering, shared
prompts, additive log regularization, and the SL reference stay unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

try:
    from scripts.analyze_metric3_logratio_cosine import (
        EPS, FAMILIES, ROOT, SNAPSHOT, dc, hash_file, load_inputs, read_matrix,
    )
except ModuleNotFoundError:
    from analyze_metric3_logratio_cosine import (
        EPS, FAMILIES, ROOT, SNAPSHOT, dc, hash_file, load_inputs, read_matrix,
    )

OUT = ROOT / "results_for_presentation/metric1_promptwise_cosine_comparison"


def aggregate_signatures(signatures: np.ndarray) -> dict[str, np.ndarray]:
    """Compare cosines of T×N×K signatures, rejecting undefined zero rows.

    Per-prompt cosine is the Gram matrix of row-normalized, flattened
    signatures divided by N. Neither method centers across carriers or
    removes the diagonal. Both matrices have unit diagonals and are PSD.
    """
    vectors = np.asarray(signatures, dtype=np.float64)
    if vectors.ndim != 3 or min(vectors.shape) < 1:
        raise ValueError("Signatures must have nonempty T×N×K axes")
    if not np.isfinite(vectors).all():
        raise ValueError("Signatures must be finite")
    t, n, _ = vectors.shape
    norms = np.linalg.norm(vectors, axis=-1)
    if np.any(norms == 0):
        raise ValueError("Per-prompt cosine undefined: zero signature row")
    flat = vectors.reshape(t, -1)
    flat_norm = np.linalg.norm(flat, axis=-1)
    if np.any(~np.isfinite(flat_norm)):
        raise ValueError("Signature norm overflow")
    flat_unit = flat / flat_norm[:, None]
    flat_similarity = flat_unit @ flat_unit.T
    prompt_unit = (vectors / norms[:, :, None]).reshape(t, -1)
    prompt_similarity = (prompt_unit @ prompt_unit.T) / n
    norm_profiles = norms / flat_norm[:, None]
    norm_profile_similarity = norm_profiles @ norm_profiles.T
    return {
        "flattened": (flat_similarity + flat_similarity.T) / 2,
        "mean_prompt_cosine": (prompt_similarity + prompt_similarity.T) / 2,
        "norm_profile_similarity": norm_profile_similarity,
        "signature_row_norms": norms,
    }


def compute_metric1(p0: np.ndarray, teachers: np.ndarray) -> dict[str, np.ndarray]:
    """Construct exactly the original weighted, teacher-centered signature."""
    base = np.asarray(p0, dtype=np.float64)
    probs = np.asarray(teachers, dtype=np.float64)
    if base.ndim != 2 or probs.ndim != 3 or probs.shape[1:] != base.shape:
        raise ValueError("Expected matching N×K and T×N×K distributions")
    for array in (base, probs):
        if not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
            raise ValueError("Distributions must contain finite probabilities")
        if not np.allclose(array.sum(-1), 1, atol=1e-5, rtol=0):
            raise ValueError("Distributions must be normalized over carriers")
    lam = np.log(probs + EPS) - np.log(base[None] + EPS)
    lam -= lam.mean(axis=0, keepdims=True)
    lam *= np.sqrt(base)[None]
    return aggregate_signatures(lam)


def correlation(a: np.ndarray, b: np.ndarray, *, off_diagonal: bool = False) -> float:
    """Correlate full double-centered matrices, optionally masking afterwards."""
    left, right = dc(a), dc(b)
    if off_diagonal:
        mask = ~np.eye(len(left), dtype=bool)
        left, right = left[mask], right[mask]
    return float(np.corrcoef(left.ravel(), right.ravel())[0, 1])


def main() -> None:
    """Validate the old metric, compute the alternative, and save an audit."""
    source = ROOT / "results_for_presentation/qwen25_comprehensive_sweep_report.data.json"
    inventory = ROOT / "results_for_presentation/qwen25_comprehensive_sweep_report.sources.json"
    hashes = {r["path"]: r["sha256"] for r in json.loads(inventory.read_text()) if "path" in r}
    cells = {r["key"]: r for r in json.loads(source.read_text())["cells"]}
    report: dict[str, Any] = {
        "model": "Qwen2.5-7B-Instruct",
        "training_policy": "distribution distillation, not token SFT",
        "prompts": 10000,
        "carriers": 100,
        "epsilon": EPS,
        "signature": "sqrt(p0) * (log(pz+epsilon)-log(p0+epsilon) - across-teacher mean log ratio)",
        "aggregation_old": "cosine of flattened N×K signatures",
        "aggregation_new": "arithmetic mean over N prompts of K-dimensional row cosines",
        "correlations": "Pearson after full double-centering; diagonal included in all900 and excluded only after centering in off870",
        "zero_norm_policy": "Abort if any prompt signature has zero norm; no epsilon clipping of cosine norms",
        "precision": "CPU float64 arithmetic on saved model probabilities; no new inference",
        "source_sha256": hash_file(source),
        "source_inventory_sha256": hash_file(inventory),
        "script_sha256": hash_file(Path(__file__)),
        "loader_script_sha256": hash_file(ROOT / "scripts/analyze_metric3_logratio_cosine.py"),
        "cells": [],
    }
    for carrier in ("00-99", "opaque"):
        for family in FAMILIES:
            cell = cells[f"{family}-{carrier}"]
            logger.info("Loading {}", cell["key"])
            p0, teachers, _, audit = load_inputs(cell, hashes)
            ft_path = SNAPSHOT / cell["final_ft_source"]
            checksum = hash_file(ft_path)
            if checksum != hashes[cell["final_ft_source"]]:
                raise ValueError(f"Frozen SL hash mismatch: {ft_path}")
            ft = read_matrix(ft_path, cell["labels"])
            np.testing.assert_allclose(ft, cell["matrix"], atol=1e-9, rtol=1e-8)
            audit["files"][str(ft_path.relative_to(ROOT))] = checksum
            result = compute_metric1(p0, teachers)
            previous = np.asarray(cell["m4"]["matrix"])
            error = float(np.max(np.abs(result["flattened"] - previous)))
            np.testing.assert_allclose(result["flattened"], previous, atol=1e-10, rtol=1e-10)
            record: dict[str, Any] = {
                "key": cell["key"], "labels": cell["labels"], "audit": audit,
                "original_matrix_max_abs_error": error, "metrics": {},
            }
            for method in ("flattened", "mean_prompt_cosine"):
                matrix = result[method]
                np.testing.assert_allclose(matrix, matrix.T, atol=1e-12, rtol=0)
                np.testing.assert_allclose(np.diag(matrix), 1, atol=1e-12, rtol=0)
                if np.max(np.abs(matrix)) > 1 + 1e-12 or np.linalg.eigvalsh(matrix).min() < -1e-10:
                    raise ValueError(f"Invalid cosine Gram matrix: {method}")
                record["metrics"][method] = {
                    "pearson_all900": correlation(matrix, ft),
                    "pearson_off870": correlation(matrix, ft, off_diagonal=True),
                    "matrix": matrix.tolist(),
                }
            off = ~np.eye(len(previous), dtype=bool)
            norms = result["signature_row_norms"]
            record["aggregation_diagnostics"] = {
                "dc_correlation_between_metrics_all900": correlation(result["flattened"], result["mean_prompt_cosine"]),
                "mean_abs_offdiagonal_matrix_difference": float(np.abs(result["flattened"] - result["mean_prompt_cosine"])[off].mean()),
                "norm_profile_similarity_mean_offdiagonal": float(result["norm_profile_similarity"][off].mean()),
                "row_norm_min": float(norms.min()),
                "row_norm_median": float(np.median(norms)),
                "row_norm_max": float(norms.max()),
                "norm_profile_similarity": result["norm_profile_similarity"].tolist(),
            }
            report["cells"].append(record)
            logger.success("{}: flattened r={:.4f}, mean-prompt r={:.4f}, reproduction error={:.2e}",
                           cell["key"], record["metrics"]["flattened"]["pearson_all900"],
                           record["metrics"]["mean_prompt_cosine"]["pearson_all900"], error)
    lines = ["# Metric 1: flattened versus mean-prompt cosine", "",
             "Qwen2.5-7B-Instruct; **distribution distillation**, not token SFT. All settings use the same 10,000 shared prompts and 100 carriers as the original Metric 1.", "",
             "Only aggregation changes: each per-prompt vector is a row of the paper's L_z, retaining sqrt(p0) weighting and across-teacher centering. It is not the unweighted raw log-ratio vector.", ""]
    for measure, title in (("pearson_all900", "All 900 entries (diagonal included)"), ("pearson_off870", "870 off-diagonal entries")):
        lines += [f"## {title}", "", "Both matrices are double-centered before correlation (and before any diagonal masking).", "",
                  "| Carrier | Trait | Original: flattened cosine | New: mean prompt cosine | Change |",
                  "|---|---|---:|---:|---:|"]
        for record in report["cells"]:
            family, carrier = record["key"].split("-", 1)
            old = record["metrics"]["flattened"][measure]
            new = record["metrics"]["mean_prompt_cosine"][measure]
            lines.append(f"| {'00–99' if carrier == '00-99' else 'Opaque 100'} | {family.title()} | {old:.3f} | {new:.3f} | {new-old:+.3f} |")
        lines.append("")
    lines += ["## Interpretation", "",
              "Let l_zx be row x of L_z, a_x=||l_zx||, b_x=||l_z'x||, and c_x=cos(l_zx,l_z'x).",
              "The original metric is sum_x(a_x b_x c_x)/sqrt(sum_x a_x² * sum_x b_x²); the new metric is mean_x(c_x).",
              "The new metric gives every prompt equal weight after row normalization. The original retains prompt-dependent signal magnitudes; mismatched norm profiles attenuate its norm-product-weighted mean toward zero. It is not simply a normalized weighted average of c_x.",
              "They coincide if each trait's signature norm is constant across prompts, but not in general.", "",
              "## Verification and provenance", "",
              "- Original Metric 1 matrices were reproduced against the frozen table source for all eight settings.",
              "- Saved carrier order, shared prompt lists, trait labels, source hashes and full SL matrices were checked.",
              "- Symmetry, unit diagonal, range and positive semidefiniteness were checked for both similarity matrices.",
              "- No zero per-prompt signature was encountered; undefined cosine inputs would abort rather than be silently dropped or assigned zero.",
              "- analysis.json stores both labeled matrices, original-source reproduction errors, off-diagonal correlations, norm diagnostics, source hashes and original scoring precision.",
              "- This is CPU reanalysis of historical scores, not new FP32 model evaluation. No GPU job or paper figure was changed."]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    logger.success("Saved {}", OUT)


if __name__ == "__main__":
    main()
