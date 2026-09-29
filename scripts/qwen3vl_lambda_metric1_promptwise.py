"""Publish the updated Metric 1 from the same audited first-number scores."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger

from scripts.analyze_metric1_promptwise_cosine import compute_metric1
from scripts.cross_model_metric17_common import (
    atomic_json, double_center, file_digest, read_matrix, write_matrix,
)
from scripts.recompute_lambda_metric1_promptwise import load_channel, validate_gram


def publish_promptwise_metric1(directory: Path, traits: list[str]) -> list[Path]:
    """Average per-prompt cosines and verify the retained flattened result.

    No model inference is performed here. The original matrix and score chunks
    remain available; this separately named matrix is the reported Metric 1.
    """
    labels, probabilities, audit = load_channel(directory / "channel")
    if labels != traits:
        raise ValueError("Channel traits do not match the reference label order")
    protocol = audit["protocol"]["metric1_protocol"]
    if (protocol["epsilon"] != 1e-12
            or protocol["metric"] != "sqrt(neutral_probability) * teacher_centered_log_probability_shift"
            or protocol["prompt_weighting"] != "one_per_input_occurrence_shared_by_every_condition"):
        raise ValueError("Unexpected Metric 1 weighting or regularization")
    result = compute_metric1(probabilities[0], probabilities[1:])
    metric_dir = directory / "metric1"
    original = read_matrix(metric_dir / "metric1_raw.csv", traits)
    error = float(np.max(np.abs(result["flattened"] - original)))
    np.testing.assert_allclose(result["flattened"], original, atol=1e-10, rtol=1e-10)
    matrix = result["mean_prompt_cosine"]
    checks = validate_gram(matrix)
    raw = metric_dir / "metric1_mean_prompt_raw.csv"
    centered = metric_dir / "metric1_mean_prompt_double_centered.csv"
    report = metric_dir / "metric1_mean_prompt_summary.json"
    write_matrix(raw, matrix, traits)
    write_matrix(centered, double_center(matrix), traits)
    atomic_json(report, {
        "status": "complete", "traits": traits,
        "prompt_count": probabilities.shape[1], "carrier_count": probabilities.shape[2],
        "aggregation": "arithmetic mean of per-prompt cosines",
        "signature": "sqrt(p0) times across-teacher-centered log((pz+epsilon)/(p0+epsilon))",
        "epsilon": 1e-12, "zero_norm_policy": "raise; never omit a prompt",
        "precision": "float64 aggregation of saved FP32 model scores",
        "flattened_reconstruction_max_error": error,
        "minimum_prompt_signature_norm": float(result["signature_row_norms"].min()),
        "gram_checks": checks,
        "channel_complete_sha256": audit["complete_sha256"],
        "source_flattened_sha256": file_digest(metric_dir / "metric1_raw.csv"),
        "output_hashes": {p.name: file_digest(p) for p in (raw, centered)},
    })
    logger.success("Published mean-per-prompt Metric 1; flattened reconstruction error {:.2e}", error)
    return [raw, centered, report]
