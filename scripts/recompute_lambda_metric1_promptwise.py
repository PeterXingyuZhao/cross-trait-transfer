"""Recompute mean-prompt Metric 1 from a frozen Lambda color-suite backup.

Run from the repository root with ``python -m scripts.recompute_lambda_metric1_promptwise``.
Only CPU aggregation changes. Original caches, source matrices and reports are
read-only; derived matrices and provenance are written to a separate directory.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from scripts.analyze_metric1_promptwise_cosine import compute_metric1, correlation
from scripts.analyze_metric3_logratio_cosine import ROOT, dc, hash_file, read_matrix
from scripts.score_string_trait_number_distribution import slugify

BACKUP = ROOT / "results/lambda/backup_20260922"
DEFAULT_SOURCE = BACKUP / "qwen3vl_metrics_v1/color"
DEFAULT_BUNDLE = BACKUP / "qwen3vl_lambda_token_sft_metrics_v1"
DEFAULT_OUTPUT = ROOT / "results_for_presentation/qwen3vl_color_token_sft_metric1_promptwise"


def verify_files(root: Path, inventory: dict[str, str]) -> None:
    """Require every listed relative file to match its saved SHA256."""
    for relative, expected in inventory.items():
        path = root / relative
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Inventory escapes its root: {relative}")
        if not path.is_file() or hash_file(path) != expected:
            raise ValueError(f"Missing or changed backup artifact: {path}")


def load_channel(channel: Path) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    """Load all teacher and neutral probabilities with matched prompt coordinates."""
    complete = json.loads((channel / "complete.json").read_text())
    if complete["status"] != "complete":
        raise ValueError("First-number channel is incomplete")
    verify_files(channel, complete["files"])
    protocol = json.loads((channel / "protocol.json").read_text())
    labels = protocol["traits"]
    if len(labels) < 2 or len(labels) != len(set(labels)):
        raise ValueError("Need distinct teacher labels")
    if len({slugify(label) for label in ["neutral_no_system", *labels]}) != len(labels) + 1:
        raise ValueError("Condition labels collide after filename normalization")
    prompts = [json.loads(line)["prompt"]
               for line in (channel / "prompts.jsonl").read_text().splitlines()]
    if len(prompts) != protocol["prompt_count"]:
        raise ValueError("Shared prompt count mismatch")
    candidates = [f"{i:02d}" for i in range(100)]
    arrays = []
    for label in ["neutral_no_system", *labels]:
        relative = f"teacher_distributions/{slugify(label)}.npz"
        metadata_name = relative.removesuffix(".npz") + ".json"
        if relative not in complete["files"] or metadata_name not in complete["files"]:
            raise ValueError(f"Condition absent from hashed inventory: {label}")
        metadata = json.loads((channel / metadata_name).read_text())
        if metadata["identity"] != {"channel": complete["identity"], "label": label}:
            raise ValueError(f"Condition identity mismatch: {label}")
        with np.load(io.BytesIO((channel / relative).read_bytes()), allow_pickle=False) as saved:
            p = np.asarray(saved["probabilities"], dtype=np.float64)
            if saved["prompts"].tolist() != prompts:
                raise ValueError(f"Shared prompt order mismatch: {label}")
            if saved["candidates"].tolist() != candidates:
                raise ValueError(f"Carrier order mismatch: {label}")
            if p.shape != (len(prompts), 100):
                raise ValueError(f"Probability dimensions mismatch: {label}")
            if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                raise ValueError(f"Invalid probabilities: {label}")
            if not np.allclose(p.sum(-1), 1, atol=1e-12, rtol=0):
                raise ValueError(f"Probabilities are not normalized: {label}")
            arrays.append(p.copy())
    audit = {"directory": str(channel.resolve()), "files": complete["files"],
             "complete_sha256": hash_file(channel / "complete.json"),
             "protocol": protocol}
    return labels, np.stack(arrays), audit


def validate_gram(matrix: np.ndarray) -> dict[str, float]:
    """Check properties of a cosine Gram matrix without modifying entries."""
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite similarity")
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-12, rtol=0)
    np.testing.assert_allclose(np.diag(matrix), 1, atol=1e-12, rtol=0)
    minimum = float(np.linalg.eigvalsh(matrix).min())
    if np.abs(matrix).max() > 1 + 1e-12 or minimum < -1e-10:
        raise ValueError("Invalid cosine Gram matrix")
    return {"minimum_eigenvalue": minimum,
            "maximum_diagonal_error": float(np.max(np.abs(np.diag(matrix) - 1)))}


def write_matrix(path: Path, matrix: np.ndarray, labels: list[str]) -> None:
    """Save full-precision labeled matrices and check the serialization round trip."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["answer_trait__teacher_trait", *labels])
        writer.writerows([[label, *map(float, matrix[i])] for i, label in enumerate(labels)])
    np.testing.assert_array_equal(read_matrix(path, labels), matrix)


def main(source_dir: Path, bundle_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Reproduce the old result, then publish the alternative with source hashes."""
    source_dir, bundle_dir, output_dir = [p.resolve() for p in (source_dir, bundle_dir, output_dir)]
    if any(output_dir.is_relative_to(p) for p in (source_dir.parent, bundle_dir)):
        raise ValueError("Output must be separate from the frozen backup")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory is not empty; use a new output directory")
    original_summary = json.loads((source_dir / "summary.json").read_text())
    scientific = original_summary["scientific"]
    cell = scientific["cell"]
    if (original_summary["status"] != "complete" or cell["domain"] != "color"
            or cell["model_id"] != "Qwen/Qwen3-VL-8B-Instruct"
            or cell["objective"] != "token_sft"):
        raise ValueError("Expected the complete Qwen3-VL color token-SFT suite")
    stage = json.loads((source_dir / "stages/metric1.json").read_text())
    if stage["status"] != "complete":
        raise ValueError("Original Metric 1 stage incomplete")
    verify_files(source_dir, stage["artifacts"])
    bundle_manifest = json.loads((bundle_dir / "manifest.json").read_text())
    verify_files(bundle_dir, bundle_manifest["files"])
    labels, p, audit = load_channel(source_dir / "channel")
    protocol = audit["protocol"]["metric1_protocol"]
    if (labels != cell["traits"] or p.shape != (31, 10000, 100)
            or protocol["epsilon"] != 1e-12
            or protocol["prompt_weighting"] != "one_per_input_occurrence_shared_by_every_condition"
            or protocol["metric"] != "sqrt(neutral_probability) * teacher_centered_log_probability_shift"):
        raise ValueError("Unexpected Metric 1 scientific protocol")
    original_path = source_dir / "metric1/metric1_raw.csv"
    sl_path = bundle_dir / cell["sl_csv"]
    original, sl = [read_matrix(path, labels) for path in (original_path, sl_path)]
    logger.info("Validated 31 conditions, 10,000 shared prompts and 100 carriers; computing on CPU")
    result = compute_metric1(p[0], p[1:])
    error = float(np.max(np.abs(result["flattened"] - original)))
    np.testing.assert_allclose(result["flattened"], original, atol=1e-10, rtol=1e-10)
    report: dict[str, Any] = {
        "status": "complete", "model": cell["model_id"], "domain": "color",
        "training_policy": "00-99 token SFT", "labels": labels, "prompt_count": 10000,
        "carrier_count": 100, "epsilon": 1e-12,
        "signature": "sqrt(p0) * (log(pz+epsilon)-log(p0+epsilon) - across-30-teacher mean log ratio)",
        "aggregation_new": "mean of per-prompt cosines, equal weight per shared prompt occurrence",
        "aggregation_old": "cosine after flattening prompt-carrier coordinates",
        "correlation": "Pearson after independently double-centering full matrices, then optional diagonal masking",
        "precision": "CPU float64 reanalysis of saved FP32 model scores; no new inference",
        "original_matrix_max_abs_error": error,
        "minimum_prompt_signature_norm": float(result["signature_row_norms"].min()),
        "zero_prompt_signatures": int(np.count_nonzero(result["signature_row_norms"] == 0)),
        "source_scientific_protocol": scientific, "channel_audit": audit,
        "source_hashes": {str(path): hash_file(path) for path in (
            original_path, sl_path, source_dir / "summary.json", source_dir / "stages/metric1.json",
            bundle_dir / "manifest.json", Path(__file__),
            ROOT / "scripts/analyze_metric1_promptwise_cosine.py",
            ROOT / "scripts/analyze_metric3_logratio_cosine.py")},
        "metrics": {},
    }
    for name in ("flattened", "mean_prompt_cosine"):
        matrix = result[name]
        report["metrics"][name] = {
            "pearson_all900": correlation(matrix, sl),
            "pearson_off870": correlation(matrix, sl, off_diagonal=True),
            "gram_checks": validate_gram(matrix), "matrix": matrix.tolist(),
        }
    old, new = [report["metrics"][key] for key in ("flattened", "mean_prompt_cosine")]
    np.testing.assert_allclose(old["pearson_all900"],
                               original_summary["matrices"]["metric1_geometry"]["double_centered_all"]["pearson"],
                               atol=1e-10, rtol=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_matrices = {
        "metric1_mean_prompt_raw.csv": result["mean_prompt_cosine"],
        "metric1_mean_prompt_double_centered.csv": dc(result["mean_prompt_cosine"]),
        "metric1_original_flattened_raw.csv": result["flattened"],
        "metric1_original_flattened_double_centered.csv": dc(result["flattened"]),
        "sl_raw.csv": sl, "sl_double_centered.csv": dc(sl),
    }
    for filename, matrix in saved_matrices.items():
        write_matrix(output_dir / filename, matrix, labels)
    report["output_files"] = {name: hash_file(output_dir / name) for name in saved_matrices}
    (output_dir / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    readme = ["# Qwen3-VL color: updated Metric 1", "",
              "Qwen3-VL-8B-Instruct (text-only); 30 colors; 00–99 token SFT; 10,000 shared prompts.", "",
              "Only aggregation changes: average the cosine within each prompt, retaining neutral-probability weighting, across-teacher centering, additive epsilon 1e-12 and the same shared prompt occurrences.", "",
              "| Correlation with SL | Original flattened cosine | New mean-prompt cosine | Change |",
              "|---|---:|---:|---:|"]
    for title, key in (("All 900 entries", "pearson_all900"), ("870 off-diagonal entries", "pearson_off870")):
        readme.append(f"| {title} | {old[key]:.6f} | {new[key]:.6f} | {new[key]-old[key]:+.6f} |")
    readme += ["", "Both complete matrices are double-centered before correlation and before any diagonal masking. No independent-entry significance test is claimed.", "",
               "## Matrices", "",
               "- [Updated similarity matrix](metric1_mean_prompt_raw.csv)",
               "- [Updated double-centered matrix](metric1_mean_prompt_double_centered.csv)",
               "- [Numerical results and provenance](analysis.json)", "",
               "Rows are answer traits; columns are teacher traits. Metric 1 is symmetric, but the matched SL matrix retains its original direction. Label order matches the original report.", "",
               "## Validation", "",
               f"- Original flattened matrix reproduced: maximum absolute difference {error:.3e}.",
               "- Frozen bundle and channel hashes, prompt/carrier order, normalized probabilities and source SL were checked.",
               "- Both cosine matrices passed symmetry, unit-diagonal, range and positive-semidefinite checks.",
               f"- Zero prompt signatures: 0; smallest norm {report['minimum_prompt_signature_norm']:.6g}.",
               "- Undefined zero-norm signatures would abort, not be skipped or silently assigned zero.",
               "- CPU float64 reanalysis only. Original Lambda outputs, report and paper figures were not overwritten.", "",
               "Reproduce from the repository root with `python -m scripts.recompute_lambda_metric1_promptwise --output-dir /path/to/new/output`."]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n")
    logger.success("All900 r: {:.6f} -> {:.6f}; off870 r: {:.6f} -> {:.6f}; reconstruction error {:.2e}",
                   old["pearson_all900"], new["pearson_all900"],
                   old["pearson_off870"], new["pearson_off870"], error)
    logger.success("Saved {}", output_dir)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    main(**vars(parser.parse_args()))
