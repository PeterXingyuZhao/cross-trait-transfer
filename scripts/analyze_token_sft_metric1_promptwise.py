"""CPU-only mean-prompt Metric 1 reanalysis for Qwen2.5 number-token SFT."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from scripts.analyze_metric1_promptwise_cosine import correlation
from scripts.analyze_metric3_logratio_cosine import ROOT, hash_file, read_matrix
from scripts.prepare_token_sft_first_number_channel import load_validated_distribution_cache
from scripts.run_token_sft_metric1_recheck import digest
from scripts.score_string_trait_number_distribution import slugify
from scripts.summarize_token_sft_metric1_recheck import (
    distinctive_features, expand_panels, feature_gram, gram_to_cosine,
)

BUNDLE = ROOT / "colab_exports/qwen25_token_sft_metric1_optimized_v2"
RECHECK = ROOT / "results/colab/qwen25_token_sft_metric1_optimized_v2_fp32/color"
SNAPSHOT = ROOT / "results_for_presentation/paper_token_sft_metric_results/source_snapshot_20746025"
CACHE = ROOT / "results/analysis_inputs/metric1_token_sft_promptwise_20260922"
OUT = ROOT / "results_for_presentation/token_sft_metric1_promptwise_comparison"


def weighted_comparison(probabilities: np.ndarray, weights: np.ndarray) -> dict[str, np.ndarray]:
    """Change only normalization order under one common prompt measure.

    Integer weights are equivalent to repeating each prompt that many times.
    Thus pooled accepted-prompt weights are shared across all teachers, never
    teacher-specific coordinates. Zero-weight prompts are excluded explicitly.
    """
    p, w = np.asarray(probabilities, dtype=float), np.asarray(weights, dtype=float)
    if p.ndim != 3 or w.shape != (p.shape[1],):
        raise ValueError("Weights must match the prompt dimension")
    if not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("Need finite nonnegative weights with positive total")
    active = w > 0
    features = distinctive_features(p[:, active])
    norms = np.linalg.norm(features, axis=-1)
    if np.any(norms == 0):
        raise ValueError("Undefined per-prompt cosine: zero signature")
    flat = gram_to_cosine(feature_gram(features, w[active]))
    unit = features / norms[:, :, None]
    mean_prompt = feature_gram(unit, w[active]) / w.sum()
    for matrix in (flat, mean_prompt):
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-12, rtol=0)
        np.testing.assert_allclose(np.diag(matrix), 1, atol=1e-12, rtol=0)
        if np.abs(matrix).max() > 1 + 1e-12 or np.linalg.eigvalsh(matrix).min() < -1e-10:
            raise ValueError("Invalid cosine Gram matrix")
    return {"flattened": flat, "mean_prompt_cosine": mean_prompt,
            "signature_row_norms": norms}


def load_original(domain: str) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, dict]:
    """Validate original channel caches against frozen prompt/matrix sources."""
    name = f"qwen25_{domain}30_number_sequence_sft_10k_alpha32_3ep_h100"
    source = SNAPSHOT / name
    suite = source / "number_token_sft_metrics_v2"
    manifest_path = suite / "channel/first_number_channel_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    bundle_manifest = json.loads((BUNDLE / "manifest.json").read_text())
    design_path = BUNDLE / bundle_manifest["domains"][domain]
    if hash_file(design_path) != bundle_manifest["files"][str(design_path.relative_to(BUNDLE))]:
        raise ValueError("Frozen prompt design changed")
    design = json.loads(design_path.read_text())
    labels = manifest["traits"]
    if (labels != design["traits"] or manifest["prompt_count"] != 1024
            or manifest["prompt_offset"] != 0 or manifest["model_id"] != "Qwen/Qwen2.5-7B-Instruct"):
        raise ValueError("Unexpected original setting")
    prompts = [design["prompts"][i] for i in design["stream_10k_prompt_indices"][:1024]]
    if digest(prompts) != manifest["cache_prompts_sha256"]:
        raise ValueError("Original shared prompt prefix mismatch")
    original_path = suite / "metric_1/m4_teacher_similarity.csv"
    sl_path = source / "m10/matrices/final/raw_logprob_shift.csv"
    suite_path = suite / "suite_protocol.json"
    suite_protocol = json.loads(suite_path.read_text())
    if hash_file(sl_path) != suite_protocol["source_evidence"]["ft_matrix_sha256"]:
        raise ValueError("SL matrix source mismatch")
    for key, path in (("historical_metric1", original_path), ("sl", sl_path)):
        relative = design["references"][key]
        if hash_file(path) != bundle_manifest["files"][relative]:
            raise ValueError(f"Frozen {key} reference mismatch")
    files = {str(p.relative_to(ROOT)): hash_file(p)
             for p in (manifest_path, suite_path, original_path, sl_path, design_path, BUNDLE / "manifest.json")}
    values, runtime = [], None
    for label in ["neutral_no_system", *labels]:
        path = CACHE / domain / "teacher_distributions" / f"{slugify(label)}.npz"
        sidecar = json.loads(path.with_suffix(".json").read_text())
        identity = sidecar["identity"]
        if runtime is None:
            runtime = identity["runtime"]
        expected = {
            "channel_fingerprint": manifest["fingerprint"], "runtime": runtime,
            "label": label,
            "system_prompt": None if label == "neutral_no_system" else manifest["source_config"]["system_prompt_template"].format(trait=label),
            "rendered_prompts_sha256": identity["rendered_prompts_sha256"],
        }
        saved = load_validated_distribution_cache(path, prompts=prompts, expected_identity=expected)
        files[str(path.relative_to(ROOT))] = saved.cache_sha256
        files[str(path.with_suffix(".json").relative_to(ROOT))] = saved.metadata_sha256
        values.append(saved.probabilities)
    if runtime["weight_dtypes"] != ["torch.float32"] or runtime["allow_tf32"] is not False:
        raise ValueError("Original model scores are not strict FP32")
    return labels, np.stack(values), read_matrix(original_path, labels), read_matrix(sl_path, labels), {
        "files": files, "runtime": runtime, "prompt_count": 1024,
        "event": "first complete two-digit number, conditioned over 00--99",
        "source_run": name,
    }


def result_record(*, name: str, domain: str, labels: list[str], p: np.ndarray,
                  weights: np.ndarray, original: np.ndarray, sl: np.ndarray,
                  audit: dict) -> dict[str, Any]:
    """Require historical reconstruction before reporting a new correlation."""
    values = weighted_comparison(p, weights)
    error = float(np.max(np.abs(values["flattened"] - original)))
    np.testing.assert_allclose(values["flattened"], original, atol=1e-9, rtol=1e-9)
    output: dict[str, Any] = {
        "name": name, "domain": domain, "labels": labels, "audit": audit,
        "unique_prompt_count": int(np.count_nonzero(weights)),
        "total_prompt_weight": float(weights.sum()),
        "weights_sha256": digest(weights.tolist()),
        "original_matrix_max_abs_error": error,
        "minimum_prompt_signature_norm": float(values["signature_row_norms"].min()),
        "metrics": {},
    }
    for key in ("flattened", "mean_prompt_cosine"):
        output["metrics"][key] = {
            "pearson_all900": correlation(values[key], sl),
            "pearson_off870": correlation(values[key], sl, off_diagonal=True),
            "matrix": values[key].tolist(),
        }
    logger.success("{}: old r={:.4f}, new r={:.4f}; source reproduction error={:.2e}",
                   name, output["metrics"]["flattened"]["pearson_all900"],
                   output["metrics"]["mean_prompt_cosine"]["pearson_all900"], error)
    return output


def main(*, original_only: bool = False, read_workers: int = 8) -> None:
    """Compare all original domains and optionally completed color rechecks."""
    output: dict[str, Any] = {
        "model": "Qwen2.5-7B-Instruct", "training_policy": "00--99 token SFT",
        "epsilon": 1e-12,
        "signature": "sqrt(p0) times across-teacher-centered additive log probability ratio",
        "normalization": "Original cosine after flattening versus mean of per-prompt cosines; shared prompt weights unchanged",
        "precision": "CPU float64 reanalysis of previously saved strict-FP32 model scores",
        "script_sha256": hash_file(Path(__file__)),
        "helper_script_sha256": hash_file(ROOT / "scripts/load_metric1_color_recheck_cache.py"),
        "cells": [],
    }
    for domain in ("animal", "city", "color"):
        labels, p, original, sl, audit = load_original(domain)
        output["cells"].append(result_record(
            name=f"{domain}_original_1024", domain=domain, labels=labels, p=p,
            weights=np.ones(1024), original=original, sl=sl, audit=audit,
        ))
    if not original_only:
        from scripts.load_metric1_color_recheck_cache import load_color_recheck
        design, p, audit = load_color_recheck(max_workers=read_workers)
        output["color_recheck_audit"] = audit
        labels = design["traits"]
        sl = read_matrix(BUNDLE / design["references"]["sl"], labels)
        # Historical old-channel scores differ minutely from optimized recheck
        # scores. For every recheck panel, compare to its OWN saved flat matrix.
        panels = expand_panels(design)
        for panel in ("stream_1024", "stream_10k", "accepted_all"):
            matrix_path = RECHECK / f"matrices/{panel}_metric1.csv"
            original = read_matrix(matrix_path, labels)
            output["cells"].append(result_record(
                name=f"color_recheck_{panel}", domain="color", labels=labels, p=p,
                weights=panels[panel]["weights"], original=original, sl=sl,
                audit={"shared_audit": "color_recheck_audit", "description": panels[panel]["description"],
                       "matrix_path": str(matrix_path.relative_to(ROOT)), "matrix_sha256": hash_file(matrix_path)},
            ))
    lines = ["# Qwen2.5 00–99 token SFT: mean-prompt Metric 1", "",
             "Only the prompt aggregation changes: the same first-number channel, additive epsilon 1e-12, neutral-probability weighting, and across-teacher centering are retained.", "",
             "Original animal/city/color use 1,024 shared prompts. The completed color recheck also supplies 10,000 generation-stream prompts and a common pooled accepted-training prompt measure.", ""]
    for measure, title in (("pearson_all900", "All 900 entries"), ("pearson_off870", "870 off-diagonal entries")):
        lines += [f"## {title}", "", "Pearson correlation with SL after double-centering both full matrices; any diagonal masking happens afterwards.", "",
                  "| Setting / prompt panel | Original flattened | Mean per-prompt cosine | Change |",
                  "|---|---:|---:|---:|"]
        for row in output["cells"]:
            old, new = [row["metrics"][k][measure] for k in ("flattened", "mean_prompt_cosine")]
            lines.append(f"| {row['name']} | {old:.3f} | {new:.3f} | {new-old:+.3f} |")
        lines.append("")
    lines += ["## Protocol notes", "",
              "- The carrier probabilities concern the first complete number, not all ten generated numbers and not raw support mass.",
              "- The original prefix is the first 1,024 unfiltered seed-42 requests, shared across teachers.",
              "- Color accepted_all uses 12,032 unique prompts with common pooled frequencies from 30 × 10,000 accepted training examples. It is a weighted mean over a common prompt measure, not 10,000 prompts accepted by every teacher.",
              "- Recheck stream_1024 and the historical color1024 have slightly different numerical model scores; each new cosine is compared against its own saved flattened baseline.",
              "- Every original flat matrix is reconstructed before the new result is accepted. Source identities/hashes, prompt/carrier labels, and normalized probabilities are validated.",
              "- Undefined zero-norm prompt signatures abort; none is silently dropped. No GPU job or existing paper figure was changed.",
              "- analysis.json stores all labeled matrices, correlation values, model runtime/source provenance and reproduction checks."]
    OUT.mkdir(parents=True, exist_ok=True)
    stem = "original_only" if original_only else "analysis"
    (OUT / f"{stem}.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    (OUT / ("original_only.md" if original_only else "README.md")).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-only", action="store_true")
    parser.add_argument("--read-workers", type=int, default=8)
    main(**vars(parser.parse_args()))
