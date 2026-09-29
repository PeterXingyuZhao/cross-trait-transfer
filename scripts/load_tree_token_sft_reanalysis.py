"""Read relocated tree token-SFT probabilities without changing source metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from scripts.prepare_token_sft_first_number_channel import load_validated_distribution_cache
from scripts.reaggregate_number_sft_metric1 import (
    CANDIDATES, LEGACY_ANALYZER, LEGACY_ANALYZER_SHA256, digest, read_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/analysis_inputs/token_sft_metric3_inner_products_20260924"
SOURCE = ROOT / "cluster_exports/qwen25_tree_token_sft_m1_10k_20260922_r1"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


def validate_relocated_inventory(
    saved: list[dict[str, str]], checked: list[dict[str, str]], remote_suite: str,
) -> None:
    """Require exact original paths, ordered labels and hashes before relocation."""
    if len(saved) != len(checked):
        raise ValueError("Completion inventory count differs")
    for remote, local in zip(saved, checked):
        expected = str(Path(remote_suite) / "channel/teacher_distributions" / f"{local['label']}.npz")
        if (set(remote) != {"label", "path", "sha256"}
                or remote["label"] != local["label"]
                or remote["sha256"] != local["sha256"] or remote["path"] != expected):
            raise ValueError("Original completion inventory identity differs")


def load_tree(
    suite: Path, sl_path: Path | None = None, source_release: Path = SOURCE,
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, Any]]:
    """Return traits, neutral-first full 10k probabilities, SL and source hashes."""
    suite, source_release = suite.resolve(), source_release.resolve()
    sl_path = (sl_path or suite / "reference/raw_logprob_shift.csv").resolve()
    hashes: dict[str, str] = {}

    def track(path: Path) -> bytes:
        payload = path.read_bytes()
        hashes[str(path.resolve())] = hashlib.sha256(payload).hexdigest()
        return payload

    def read(path: Path) -> Any:
        return json.loads(track(path))

    channel = suite / "channel"
    manifest = read(channel / "first_number_channel_manifest.json")
    plan = read(suite / "suite_protocol.json")
    suite_identity = {k: v for k, v in plan.items() if k != "fingerprint"}
    plan_hash = hashlib.sha256(json.dumps(suite_identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if (manifest["fingerprint"] != digest({k: v for k, v in manifest.items() if k != "fingerprint"})
            or plan["fingerprint"] != plan_hash):
        raise ValueError("Invalid channel/suite identity")
    traits = manifest["traits"]
    if (len(traits) != 30 or len(set(traits)) != 30 or traits != plan["traits"]
            or plan["category"] != "tree"
            or read(source_release / "experiment_plan/smoke_tests/tree_answers_30.json") != traits):
        raise ValueError("Unexpected tree trait catalog")
    if (manifest["model_id"] != "Qwen/Qwen2.5-7B-Instruct"
            or manifest["source_config"]["carrier"] != "number_sequence_00_99"
            or manifest["source_config"]["model_revision"] != MODEL_REVISION
            or manifest["candidates"] != CANDIDATES or manifest["target_texts"] != CANDIDATES
            or manifest["prompt_count"] != 10000 or manifest["prompt_offset"] != 0
            or plan["metric_1"]["prompt_count"] != 10000
            or plan["metric_1"]["prompt_selection"] != "all_shared_channel_rows"
            or manifest["torch_dtype"] != "float32"):
        raise ValueError("Unexpected tree number-SFT scientific setting")
    for inventory in (manifest["source_files_sha256"], plan["source_sha256"]):
        for relative, expected in inventory.items():
            path = (source_release / relative).resolve()
            if not path.is_relative_to(source_release):
                raise ValueError("Source path escapes frozen release")
            track(path)
            if hashes[str(path)] != expected:
                raise ValueError(f"Frozen source hash differs: {relative}")
    if plan["source_sha256"][LEGACY_ANALYZER] != LEGACY_ANALYZER_SHA256:
        raise ValueError("Unexpected historical Metric 1 implementation")
    stream_bytes = track(channel / "prompts.jsonl")
    prompts = [json.loads(line)["prompt"] for line in stream_bytes.decode().splitlines()]
    if (len(prompts) != 10000 or manifest["prompt_stream_count"] != 10000
            or digest(prompts) != manifest["cache_prompts_sha256"]
            or digest(prompts) != manifest["prompt_stream_sha256"]
            or hashes[str((channel / "prompts.jsonl").resolve())] != manifest["prompts_jsonl_sha256"]):
        raise ValueError("Shared prompt stream differs")
    runtime = read(channel / "first_number_channel_runtime.json")
    if (runtime["weight_dtypes"] != ["torch.float32"] or runtime["allow_tf32"] is not False
            or runtime["model_revision"] != MODEL_REVISION):
        raise ValueError("Unexpected strict-FP32 model runtime")
    complete = read(channel / "first_number_channel_result.json")
    if (complete["status"] != "complete" or complete["channel_fingerprint"] != manifest["fingerprint"]
            or complete["prompt_count"] != 10000 or complete["candidate_count"] != 100):
        raise ValueError("Channel is not complete")
    arrays, checked = [], []
    for label in ["neutral_no_system", *traits]:
        path = channel / "teacher_distributions" / f"{label}.npz"
        sidecar = read(path.with_suffix(".json"))
        expected_identity = {
            "channel_fingerprint": manifest["fingerprint"], "runtime": runtime,
            "label": label, "system_prompt": None if label == "neutral_no_system" else
            manifest["source_config"]["system_prompt_template"].format(trait=label),
            "rendered_prompts_sha256": sidecar["identity"]["rendered_prompts_sha256"],
        }
        cache = load_validated_distribution_cache(path, prompts=prompts, expected_identity=expected_identity)
        if any(cache.sidecar.get(k) != value for k, value in cache.metadata.items()):
            raise ValueError(f"Embedded metadata and sidecar differ: {label}")
        hashes[str(path.resolve())] = cache.cache_sha256
        hashes[str(path.with_suffix(".json").resolve())] = cache.metadata_sha256
        arrays.append(cache.probabilities)
        checked.append({"label": label, "path": str(path), "sha256": cache.cache_sha256})
    validate_relocated_inventory(complete["caches"], checked, plan["output_dir"])
    receipt = read(suite / "stage_receipts/metric_1.json")
    if receipt["status"] != "completed" or receipt["fingerprint"] != plan["fingerprint"]:
        raise ValueError("Metric 1 stage identity differs")
    for relative, expected in receipt["artifact_sha256"].items():
        path = (suite / relative).resolve()
        if not path.is_relative_to(suite):
            raise ValueError("Artifact escapes suite")
        track(path)
        if hashes[str(path)] != expected:
            raise ValueError(f"Metric 1 artifact differs: {relative}")
    legacy = read(suite / "metric_1/m4_teacher_similarity.json")
    if (legacy["traits"] != traits or legacy["prompt_count"] != 10000
            or legacy["candidate_count"] != 100 or legacy["epsilon"] != 1e-12):
        raise ValueError("Legacy Metric 1 parameters differ")
    track(sl_path)
    if hashes[str(sl_path)] != plan["source_evidence"]["ft_matrix_sha256"]:
        raise ValueError("Matching SL hash differs")
    probabilities = np.stack(arrays)
    if (probabilities.shape != (31, 10000, 100) or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or not np.allclose(probabilities.sum(-1), 1, atol=1e-12, rtol=0)):
        raise ValueError("Invalid normalized probabilities")
    return traits, probabilities, read_matrix(sl_path, traits), {
        "source_sha256": hashes, "runtime": runtime,
        "model": manifest["model_id"], "training": "00-99 token SFT",
        "suite_fingerprint": plan["fingerprint"], "channel_fingerprint": manifest["fingerprint"],
        "prompt_count": 10000, "unique_prompt_strings": len(set(prompts)),
        "prompt_weights": "one per shared unfiltered generation-prompt occurrence",
        "prompt_order_sha256": digest(prompts), "carrier_order": CANDIDATES,
        "original_suite_path": plan["output_dir"], "relocated_suite_path": str(suite),
        "relocation": "Exact original suite-relative cache path, ordered label and hash checked; source metadata unchanged",
        "legacy_analyzer": LEGACY_ANALYZER, "legacy_analyzer_sha256": LEGACY_ANALYZER_SHA256,
    }


def write_metric1_audit(output: Path = INPUT / "metric1_audit.json") -> dict[str, Any]:
    """Reconstruct legacy kernels and save current M1 for the four numeric suites."""
    from scripts.analyze_token_sft_metric1_promptwise import load_original, weighted_comparison
    from scripts.analyze_metric1_promptwise_cosine import correlation
    from scripts.reaggregate_number_sft_metric1 import double_center

    previous_path = ROOT / "results_for_presentation/token_sft_metric1_promptwise_comparison/original_only.json"
    previous = {cell["domain"]: cell for cell in json.loads(previous_path.read_text())["cells"]}
    report: dict[str, Any] = {
        "model": "Qwen/Qwen2.5-7B-Instruct", "training": "00-99 token SFT",
        "signature": "sqrt(p0)*(log(pz+epsilon)-log(p0+epsilon)-across_teacher_mean_log_ratio)",
        "aggregation": "arithmetic mean of per-prompt cosines", "epsilon": 1e-12,
        "weights": "equal weight per shared prompt occurrence; never teacher-specific",
        "correlation": "Pearson after independently double-centering complete matrices, then optional diagonal masking",
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cells": {},
    }
    for domain in ("color", "animal", "city", "tree"):
        if domain == "tree":
            labels, probabilities, sl, audit = load_tree(INPUT / "tree")
            old = read_matrix(INPUT / "tree/metric_1/m4_teacher_similarity.csv", labels)
        else:
            labels, probabilities, old, sl, audit = load_original(domain)
            for relative, expected in previous[domain]["audit"]["files"].items():
                if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
                    raise ValueError(f"Previous source provenance differs: {relative}")
            protocol_path = next(ROOT / relative for relative in audit["files"] if relative.endswith("/suite_protocol.json"))
            protocol = json.loads(protocol_path.read_text())
            audit["executed_source_evidence"] = {}
            for filename in ("analyze_teacher_distribution_similarity.py", "run_qwen25_number_sft_metric_suite.py"):
                frozen = ROOT / "results_for_presentation/paper_main_color_token_sft_20260923/audit_metric1/source_evidence" / filename
                checksum = hashlib.sha256(frozen.read_bytes()).hexdigest()
                if checksum != protocol["source_sha256"]["scripts/" + filename]:
                    raise ValueError("Executed source copy differs")
                audit["executed_source_evidence"][str(frozen)] = checksum
        n = probabilities.shape[1]
        values = weighted_comparison(probabilities, np.ones(n))
        reconstruction_error = float(np.max(np.abs(values["flattened"] - old)))
        np.testing.assert_allclose(values["flattened"], old, atol=1e-10, rtol=1e-10)
        current = values["mean_prompt_cosine"]
        if domain != "tree":
            np.testing.assert_allclose(current, previous[domain]["metrics"]["mean_prompt_cosine"]["matrix"], atol=1e-12, rtol=0)
        report["cells"][domain] = {
            "labels": labels, "prompt_count": n, "total_prompt_weight": n,
            "weights_sha256": digest(np.ones(n).tolist()),
            "minimum_signature_norm": float(values["signature_row_norms"].min()),
            "zero_norm_signatures": int((values["signature_row_norms"] == 0).sum()),
            "legacy_reconstruction_max_abs_error": reconstruction_error,
            "pearson_all900": correlation(current, sl),
            "pearson_off870": correlation(current, sl, off_diagonal=True),
            "matrix": current.tolist(), "double_centered_matrix": double_center(current).tolist(),
            "sl_matrix": sl.tolist(), "source_audit": audit,
        }
        logger.success("{} M1: N={}, all900={:.12f}, off870={:.12f}", domain, n,
                       report["cells"][domain]["pearson_all900"], report["cells"][domain]["pearson_off870"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    write_metric1_audit()
