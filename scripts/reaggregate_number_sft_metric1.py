"""CPU-only current Metric 1 from a completed strict number-SFT channel.

Original caches, receipts and legacy matrices are read-only. The standalone
module needs only numpy and loguru, so it can accompany a recovery overlay
without changing the frozen model-scoring implementation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

EPSILON = 1e-12
CANDIDATES = [f"{i:02d}" for i in range(100)]
LEGACY_ANALYZER = "scripts/analyze_teacher_distribution_similarity.py"
LEGACY_ANALYZER_SHA256 = "447958f3eb8abbcf062981a06ef7b8fce3bada7450737d1460ffc555791cc2bd"


def digest(value: Any) -> str:
    """Hash canonical JSON, using the frozen channel's identity convention."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def file_hash(path: Path) -> str:
    """Hash a source or artifact without interpreting its contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    """Read one saved JSON document."""
    return json.loads(path.read_text())


def double_center(matrix: np.ndarray) -> np.ndarray:
    """Center the complete matrix before selecting any off-diagonal entries."""
    return matrix - matrix.mean(0) - matrix.mean(1)[:, None] + matrix.mean()


def read_matrix(path: Path, traits: list[str]) -> np.ndarray:
    """Require exact trait sets on both axes, then align their order explicitly."""
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    columns, names = rows[0][1:], [row[0] for row in rows[1:]]
    if (len(columns) != len(traits) or len(names) != len(traits)
            or set(columns) != set(traits) or set(names) != set(traits)):
        raise ValueError(f"Matrix labels do not match traits: {path}")
    matrix = np.asarray([row[1:] for row in rows[1:]], dtype=float)
    if matrix.shape != (len(traits), len(traits)) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid matrix: {path}")
    return matrix[np.ix_([names.index(t) for t in traits], [columns.index(t) for t in traits])]


def aggregate(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return legacy/current kernels for neutral-first T+1 by N by K scores.

    Normalize each teacher's carrier signature within each prompt before
    averaging. A duplicate prompt occurrence retains its original unit weight.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 3 or min(p.shape) < 1 or p.shape[0] < 3:
        raise ValueError("Expected neutral and at least two teachers on shared prompts")
    if (not np.isfinite(p).all() or np.any((p < 0) | (p > 1))
            or not np.allclose(p.sum(-1), 1, atol=1e-10, rtol=0)):
        raise ValueError("Expected finite carrier-normalized probabilities")
    shifts = np.log(p[1:] + EPSILON) - np.log(p[0] + EPSILON)
    signatures = (shifts - shifts.mean(0)) * np.sqrt(p[0])
    norms = np.linalg.norm(signatures, axis=-1)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("Undefined per-prompt cosine: zero or nonfinite signature norm")
    flat = signatures.reshape(len(signatures), -1)
    flat_unit = flat / np.linalg.norm(flat, axis=-1)[:, None]
    legacy = flat_unit @ flat_unit.T
    prompt_unit = signatures / norms[:, :, None]
    unit_flat = prompt_unit.reshape(len(signatures), -1)
    current = (unit_flat @ unit_flat.T) / p.shape[1]
    for matrix in (legacy, current):
        if (not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0)
                or not np.allclose(np.diag(matrix), 1, atol=1e-12, rtol=0)
                or np.abs(matrix).max() > 1 + 1e-12
                or np.linalg.eigvalsh(matrix).min() < -1e-10):
            raise ValueError("Invalid cosine Gram matrix")
    return legacy, current, float(norms.min())


def load_channel(suite: Path, source_release: Path) -> tuple[list[str], np.ndarray, dict]:
    """Validate source identities, every ordered cache and completion inventory."""
    channel = suite / "channel"
    manifest_path = channel / "first_number_channel_manifest.json"
    manifest, plan = read_json(manifest_path), read_json(suite / "suite_protocol.json")
    channel_identity = {k: v for k, v in manifest.items() if k != "fingerprint"}
    suite_identity = {k: v for k, v in plan.items() if k != "fingerprint"}
    # The historical suite and channel intentionally used different JSON spacing.
    suite_hash = hashlib.sha256(json.dumps(
        suite_identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if manifest.get("fingerprint") != digest(channel_identity) or plan.get("fingerprint") != suite_hash:
        raise ValueError("Invalid channel or suite fingerprint")
    traits = manifest["traits"]
    if (len(traits) < 2 or len(set(traits)) != len(traits) or traits != plan["traits"]
            or "neutral_no_system" in traits):
        raise ValueError("Teacher set/order changed")
    if (manifest["model_id"] != "Qwen/Qwen2.5-7B-Instruct"
            or manifest["source_config"]["carrier"] != "number_sequence_00_99"
            or manifest["candidates"] != CANDIDATES or manifest["target_texts"] != CANDIDATES
            or manifest["prompt_count"] != plan["metric_1"]["prompt_count"]
            or plan["metric_1"]["prompt_selection"] != "all_shared_channel_rows"
            or manifest["torch_dtype"] != "float32"):
        raise ValueError("Unexpected model/carrier/prompt/precision setting")
    tracked = {str(path.resolve()): file_hash(path) for path in (manifest_path, suite / "suite_protocol.json")}
    traits_path = source_release / manifest["source_config"]["traits_path"]
    if (not traits_path.resolve().is_relative_to(source_release.resolve())
            or read_json(traits_path) != traits):
        raise ValueError("Frozen trait catalog differs from the channel")
    tracked[str(traits_path.resolve())] = file_hash(traits_path)
    for name, expected in manifest["source_files_sha256"].items():
        path = source_release / name
        if not path.resolve().is_relative_to(source_release.resolve()) or file_hash(path) != expected:
            raise ValueError(f"Frozen scoring source changed: {name}")
        tracked[str(path.resolve())] = expected
    analyzer = source_release / LEGACY_ANALYZER
    if (file_hash(analyzer) != LEGACY_ANALYZER_SHA256
            or plan["source_sha256"].get(LEGACY_ANALYZER) != LEGACY_ANALYZER_SHA256):
        raise ValueError("Legacy analyzer is not the audited flattened-cosine implementation")
    tracked[str(analyzer.resolve())] = LEGACY_ANALYZER_SHA256
    stream_path = channel / "prompts.jsonl"
    stream = [json.loads(line)["prompt"] for line in stream_path.read_text().splitlines()]
    offset, count = manifest["prompt_offset"], manifest["prompt_count"]
    prompts = stream[offset:offset + count]
    if (offset < 0 or count < 1 or len(prompts) != count
            or len(stream) != manifest["prompt_stream_count"]
            or digest(stream) != manifest["prompt_stream_sha256"]
            or file_hash(stream_path) != manifest["prompts_jsonl_sha256"]
            or digest(prompts) != manifest["cache_prompts_sha256"]):
        raise ValueError("Shared prompt stream/count/order changed")
    runtime_path = channel / "first_number_channel_runtime.json"
    runtime = read_json(runtime_path)
    if runtime.get("weight_dtypes") != ["torch.float32"] or runtime.get("allow_tf32") is not False:
        raise ValueError("Saved channel runtime is not strict FP32")
    complete_path = channel / "first_number_channel_result.json"
    complete = read_json(complete_path)
    if (complete.get("status") != "complete" or complete.get("channel_fingerprint") != manifest["fingerprint"]
            or complete.get("prompt_count") != count or complete.get("candidate_count") != 100):
        raise ValueError("Incomplete channel")
    arrays, inventory = [], []
    for label in ["neutral_no_system", *traits]:
        # The audited suites use simple lowercase trait names; refusing other
        # labels avoids ambiguous implicit filename normalization.
        if not label or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in label):
            raise ValueError(f"Unsupported cache label: {label}")
        path = channel / "teacher_distributions" / f"{label}.npz"
        sidecar_path = path.with_suffix(".json")
        payload, sidecar = path.read_bytes(), read_json(sidecar_path)
        checksum = hashlib.sha256(payload).hexdigest()
        if sidecar.get("cache_sha256") != checksum:
            raise ValueError(f"Cache hash mismatch: {label}")
        identity = sidecar["identity"]
        expected = {"channel_fingerprint": manifest["fingerprint"], "runtime": runtime,
                    "label": label, "system_prompt": None if label == "neutral_no_system" else
                    manifest["source_config"]["system_prompt_template"].format(trait=label),
                    "rendered_prompts_sha256": identity.get("rendered_prompts_sha256")}
        if identity != expected or not isinstance(identity.get("rendered_prompts_sha256"), str):
            raise ValueError(f"Cache condition identity mismatch: {label}")
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            if metadata["identity"] != expected or any(sidecar.get(k) != v for k, v in metadata.items()):
                raise ValueError(f"Embedded cache metadata mismatch: {label}")
            if (data["prompts"].tolist() != prompts or data["candidates"].tolist() != CANDIDATES
                    or data["target_texts"].tolist() != CANDIDATES
                    or digest(data["prompt_texts"].tolist()) != expected["rendered_prompts_sha256"]):
                raise ValueError(f"Cache prompt/carrier ordering mismatch: {label}")
            scores = np.asarray(data["raw_logprobs"], dtype=float)
            if scores.shape != (count, 100) or not np.isfinite(scores).all():
                raise ValueError(f"Invalid raw scores: {label}")
            maxima = scores.max(-1, keepdims=True)
            logmass = maxima[:, 0] + np.log(np.exp(scores - maxima).sum(-1))
            probability = np.exp(scores - logmass[:, None])
            for key, value in (("probabilities", probability), ("raw_support_logmass", logmass),
                               ("raw_support_mass", np.exp(logmass))):
                if data[key].shape != value.shape or not np.allclose(data[key], value, atol=1e-14, rtol=1e-12):
                    raise ValueError(f"Inconsistent cache {key}: {label}")
            arrays.append(np.asarray(data["probabilities"], dtype=float))
        inventory.append({"label": label, "path": str(path), "sha256": checksum})
        tracked[str(path.resolve())] = checksum
        tracked[str(sidecar_path.resolve())] = file_hash(sidecar_path)
    saved_inventory = complete.get("caches", [])
    if len(saved_inventory) != len(inventory):
        raise ValueError("Completion cache inventory length differs")
    for saved, checked in zip(saved_inventory, inventory):
        if (set(saved) != {"label", "path", "sha256"}
                or saved["label"] != checked["label"] or saved["sha256"] != checked["sha256"]
                or Path(saved["path"]).resolve() != Path(checked["path"]).resolve()):
            raise ValueError("Completion cache inventory does not match validated inputs")
    for path in (stream_path, runtime_path, complete_path):
        tracked[str(path.resolve())] = file_hash(path)
    return traits, np.stack(arrays), {"source_sha256": tracked, "runtime": runtime,
                                    "channel_fingerprint": manifest["fingerprint"],
                                    "suite_fingerprint": plan["fingerprint"],
                                    "prompt_offset": offset, "prompt_count": count,
                                    "unique_prompt_count": len(set(prompts)),
                                    "prompt_order_sha256": digest(prompts),
                                    "saved_completion_inventory": saved_inventory}


def immutable_write(path: Path, value: str) -> None:
    """Publish atomically, refusing to overwrite a differing derived artifact."""
    if path.exists():
        if path.read_text() != value:
            raise ValueError(f"Derived artifact drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value)
    temporary.replace(path)


def reaggregate(suite_dir: Path, source_release: Path, output_dir: Path | None = None) -> dict:
    """Validate and publish the updated matrix without touching historical files."""
    suite, source = Path(suite_dir).resolve(), Path(source_release).resolve()
    output = Path(output_dir).resolve() if output_dir else suite / "metric_1_promptwise"
    if output == suite / "metric_1" or output == suite:
        raise ValueError("Use a separate derived Metric 1 directory")
    traits, probabilities, audit = load_channel(suite, source)
    plan = read_json(suite / "suite_protocol.json")
    legacy_path = suite / "metric_1/m4_teacher_similarity.csv"
    legacy_json = suite / "metric_1/m4_teacher_similarity.json"
    saved = read_json(legacy_json)
    if (saved["traits"] != traits or saved["prompt_count"] != probabilities.shape[1]
            or saved["candidate_count"] != 100 or saved["epsilon"] != EPSILON):
        raise ValueError("Legacy metric parameters differ")
    legacy, current, minimum_norm = aggregate(probabilities)
    expected_legacy = read_matrix(legacy_path, traits)
    np.testing.assert_allclose(legacy, expected_legacy, atol=1e-10, rtol=1e-10)
    sl_path = suite.parent / "m10/matrices/final/raw_logprob_shift.csv"
    if file_hash(sl_path) != plan["source_evidence"]["ft_matrix_sha256"]:
        raise ValueError("SL matrix source changed")
    sl = read_matrix(sl_path, traits)
    centered, sl_centered = double_center(current), double_center(sl)
    mask = ~np.eye(len(traits), dtype=bool)
    correlations = {}
    for name, left, right in (("all_entries", centered.ravel(), sl_centered.ravel()),
                              ("off_diagonal", centered[mask], sl_centered[mask])):
        if np.std(left) == 0 or np.std(right) == 0:
            raise ValueError("Correlation undefined for a constant centered matrix")
        correlations[name] = {"n_entries": len(left), "pearson_r": float(np.corrcoef(left, right)[0, 1])}
    paths = {}
    for name, matrix in (("raw", current), ("double_centered", centered)):
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(["teacher_trait", *traits])
        writer.writerows([trait, *matrix[index].tolist()] for index, trait in enumerate(traits))
        path = output / f"metric1_mean_prompt_{name}.csv"
        immutable_write(path, buffer.getvalue())
        paths[name] = str(path)
    for path in (legacy_path, legacy_json, sl_path, Path(__file__).resolve()):
        audit["source_sha256"][str(path)] = file_hash(path)
    report = {"status": "completed", "model_id": "Qwen/Qwen2.5-7B-Instruct",
              "training_setting": "00-99 token SFT", "traits": traits,
              "carrier_labels": CANDIDATES, "aggregation": "mean_per_prompt_cosine",
              "signature": "sqrt(p0)*(log((pz+epsilon)/(p0+epsilon))-mean_teacher_log_ratio)",
              "epsilon": EPSILON, "prompt_weighting": "equal weight per shared prompt occurrence",
              "zero_norm_policy": "raise; never omit prompts or replace undefined cosines",
              "precision": "float64 CPU aggregation of saved strict-FP32 model scores",
              "prompt_count": probabilities.shape[1], "minimum_signature_norm": minimum_norm,
              "legacy_reconstruction_max_error": float(np.abs(legacy - expected_legacy).max()),
              "SL_correlations": correlations, "correlation_centering": "both full matrices before masking",
              "matrix_csv": paths["raw"], "double_centered_csv": paths["double_centered"],
              "matrix_sha256": file_hash(Path(paths["raw"])),
              "double_centered_sha256": file_hash(Path(paths["double_centered"])), "audit": audit}
    immutable_write(output / "metric1_mean_prompt_summary.json", json.dumps(report, indent=2, allow_nan=False) + "\n")
    logger.success("Updated Metric 1: {} prompts; SL r(all)={:.6f}, r(off)={:.6f}; legacy error {:.2e}",
                   probabilities.shape[1], correlations["all_entries"]["pearson_r"],
                   correlations["off_diagonal"]["pearson_r"], report["legacy_reconstruction_max_error"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--source-release", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    reaggregate(**vars(parser.parse_args()))
