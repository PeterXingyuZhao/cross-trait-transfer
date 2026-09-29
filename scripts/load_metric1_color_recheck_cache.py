"""Read and validate the completed color Metric 1 recheck without inference.

The returned array has neutral first, followed by the frozen thirty teachers;
its prompt axis is the design's unique-prompt universe, not chunk order. This
module writes no experiment files and preserves the design's panel weights.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_token_sft_metric1_optimized_bundle import verify_bundle  # noqa: E402
from scripts.prepare_token_sft_first_number_channel import (  # noqa: E402
    load_validated_distribution_cache,
)
from scripts import run_token_sft_metric1_recheck as legacy  # noqa: E402
from scripts.run_token_sft_metric1_recheck_optimized import (  # noqa: E402
    OptimizedConfig,
    make_protocol,
)

BUNDLE = ROOT / "colab_exports/qwen25_token_sft_metric1_optimized_v2"
SOURCE = ROOT / "results/colab/qwen25_token_sft_metric1_optimized_v2_fp32/color"


@dataclass(frozen=True, kw_only=True)
class CacheTask:
    """Immutable location, condition and row identity for one stored chunk."""

    condition_index: int
    chunk_index: int
    indices: list[int]
    prompts: list[str]
    label: str
    path: Path
    identity: dict[str, Any]


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    """Parse and hash the same byte snapshot of a JSON artifact."""
    payload = path.read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value, hashlib.sha256(payload).hexdigest()


def _validated_context(
    bundle: Path = BUNDLE, source: Path = SOURCE,
) -> tuple[dict[str, Any], list[CacheTask], dict[str, Any]]:
    """Validate frozen inputs and construct the complete, ordered cache plan."""
    manifest = verify_bundle(bundle)
    design = legacy.load_design(bundle, manifest, "color")
    protocol, protocol_hash = _read_json(source / "protocol.json")
    runtime, runtime_hash = _read_json(source / "runtime.json")
    summary, summary_hash = _read_json(source / "summary.json")
    benchmark, benchmark_hash = _read_json(source / "benchmark.json")
    config = OptimizedConfig(
        bundle_dir=bundle, output_dir=source.parent, domain="color",
        panels=tuple(protocol["panels"]), chunk_size=protocol["chunk_size"],
        max_prompt_batch_size=protocol["max_prompt_batch_size"],
        shared_prefix=protocol["shared_prefix"],
        benchmark_prompts=protocol["benchmark_prompts"],
    )
    if protocol != make_protocol(config, manifest, design):
        raise ValueError("Saved protocol does not match the frozen color design")
    if (
        runtime.get("model_revision") != legacy.MODEL_REVISION
        or runtime.get("model_type") != "qwen2"
        or runtime.get("weight_dtypes") != ["torch.float32"]
        or runtime.get("precision_profile") != "native_qwen_fp32_math_sdpa_v1"
        or runtime.get("allow_tf32") is not False
    ):
        raise ValueError("Unexpected color recheck model or precision")
    count = len(design["prompts"])
    conditions = legacy.condition_specs(design)
    expected_labels = [label for label, _ in conditions]
    indices = protocol["scoring_indices"]
    if (
        len(conditions) != 31 or count != 12512
        or indices != list(range(count))
        or summary.get("status") != "completed"
        or summary.get("domain") != "color"
        or summary.get("catalog") != design["traits"]
        or summary.get("scored_unique_prompts") != count
        or summary.get("prompt_universe_count") != count
        or benchmark.get("status") != "passed"
        or benchmark.get("protocol_fingerprint") != protocol["fingerprint"]
        or benchmark.get("runtime") != runtime
        or benchmark.get("condition_count") != 31
        or [entry.get("label") for entry in benchmark.get("conditions", [])]
        != expected_labels
    ):
        raise ValueError("Incomplete or mismatched color cache coverage/audit")
    for name in protocol["panels"]:
        if summary.get("panels", {}).get(name, {}).get("status") != "completed":
            raise ValueError(f"Incomplete saved color prompt panel: {name}")
    tasks = []
    for chunk_index, chunk in enumerate(legacy.split_chunks(indices, config.chunk_size)):
        prompts = [design["prompts"][index] for index in chunk]
        for condition_index, (label, system) in enumerate(conditions):
            tasks.append(CacheTask(
                condition_index=condition_index, chunk_index=chunk_index,
                indices=chunk, prompts=prompts, label=label,
                path=legacy.cache_path(source, chunk_index, label),
                identity=legacy.chunk_identity(protocol, runtime, chunk, label, system),
            ))
    audit = {
        "source": str(source), "bundle": str(bundle),
        "bundle_manifest_sha256": legacy.file_digest(bundle / "manifest.json"),
        "bundle_fingerprint": legacy.digest(manifest),
        "design_sha256": manifest["files"][manifest["domains"]["color"]],
        "design_fingerprint": legacy.digest(design),
        "protocol_sha256": protocol_hash, "protocol_fingerprint": protocol["fingerprint"],
        "runtime_sha256": runtime_hash, "runtime": runtime,
        "summary_sha256": summary_hash, "benchmark_sha256": benchmark_hash,
        "references_sha256": {
            key: manifest["files"][relative]
            for key, relative in design["references"].items()
        },
        "condition_labels": expected_labels,
        "prompt_count": count, "condition_count": len(conditions),
        "chunk_count": len(legacy.split_chunks(indices, config.chunk_size)),
        "scoring_indices_sha256": legacy.digest(indices),
        "normalization": protocol["normalization"],
        "event": protocol["event"],
        "validation": "Frozen bundle hashes and lineage; exact design/protocol/runtime; "
        "complete coverage; per-file sidecar and NPZ hash; embedded cache identity; "
        "prompt and candidate order; recomputed normalization from raw log probabilities.",
    }
    return design, tasks, audit


def _load_task(task: CacheTask) -> tuple[np.ndarray, dict[str, Any]]:
    """Read one cache and its sidecar once, using the authoritative validator."""
    cache = load_validated_distribution_cache(
        task.path, prompts=task.prompts, expected_identity=task.identity,
    )
    values = np.asarray(cache.probabilities, dtype=np.float64)
    if (
        values.shape != (len(task.indices), 100)
        or not np.isfinite(values).all() or np.any(values < 0)
        or not np.allclose(values.sum(axis=1), 1.0, rtol=1e-10, atol=1e-12)
    ):
        raise ValueError(f"Invalid normalized probability matrix: {task.path}")
    return values, {
        "condition_index": task.condition_index, "chunk_index": task.chunk_index,
        "label": task.label, "path": str(task.path),
        "indices_sha256": legacy.digest(task.indices),
        "cache_sha256": cache.cache_sha256,
        "sidecar_sha256": cache.metadata_sha256,
    }


def load_color_recheck(
    *, max_workers: int = 8,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    """Return design, neutral-first probabilities and reproducible provenance.

    Threads only read independent files; each validated result is assigned on
    the calling thread to its exact condition/prompt coordinates. No scoring,
    model loading, cache rewriting or output serialization occurs.
    """
    if not isinstance(max_workers, int) or not 1 <= max_workers <= 64:
        raise ValueError("Use between one and 64 read workers")
    design, tasks, audit = _validated_context()
    count = len(design["prompts"])
    probabilities = np.empty((31, count, 100), dtype=np.float64)
    seen = np.zeros((31, count), dtype=bool)
    inventory: list[dict[str, Any]] = []
    logger.info("Validating {} color recheck caches using {} read workers", len(tasks), max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_load_task, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            task = futures.pop(future)
            values, record = future.result()
            if seen[task.condition_index, task.indices].any():
                raise ValueError("Duplicate color cache coordinates")
            probabilities[task.condition_index, task.indices, :] = values
            seen[task.condition_index, task.indices] = True
            inventory.append(record)
            if completed % 128 == 0 or completed == len(tasks):
                logger.info("Validated {}/{} color cache files", completed, len(tasks))
    if not seen.all():
        raise ValueError("Missing color teacher/prompt coordinates")
    inventory.sort(key=lambda entry: (entry["chunk_index"], entry["condition_index"]))
    audit.update({
        "cache_file_count": len(inventory),
        "file_read_workers": max_workers,
        "inventory_sha256": legacy.digest(inventory),
        "cache_inventory": inventory,
        "probability_shape": list(probabilities.shape),
        "probability_dtype": str(probabilities.dtype),
        "probability_max_row_sum_error": float(np.max(np.abs(probabilities.sum(-1) - 1.0))),
    })
    return design, probabilities, audit
