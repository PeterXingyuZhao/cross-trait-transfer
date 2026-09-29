#!/usr/bin/env python3
"""Benchmark-gated, restartable FP32 Metric 1 scoring with unchanged inputs.

Never modify the original run. Completed baseline caches may be imported only
after validating the frozen input provenance, runtime and numerical audit.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterator

import numpy as np
from loguru import logger

for directory in (Path(__file__).resolve().parent.parent, Path(__file__).resolve().parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from scripts import run_token_sft_metric1_recheck as legacy  # noqa: E402
from scripts import prepare_token_sft_first_number_channel as channel  # noqa: E402

OPTIMIZED_VERSION = 2
MATRIX_MAX_ABSOLUTE_ERROR = 1e-3


@dataclass(frozen=True, kw_only=True)
class OptimizedConfig:
    """Configuration; throughput changes never change prompts or weights."""

    bundle_dir: Path
    output_dir: Path
    domain: str = "color"
    panels: tuple[str, ...] = legacy.DEFAULT_PANELS
    chunk_size: int = 256
    max_prompt_batch_size: int = 32
    benchmark_prompts: int = 32
    shared_prefix: bool = True
    prefetch: bool = True
    baseline_dir: Path | None = None
    device: str = "cuda"
    max_chunks: int | None = None


def make_protocol(config: OptimizedConfig, manifest: dict[str, Any],
                  design: dict[str, Any]) -> dict[str, Any]:
    """Separate scientific inputs from the versioned optimized execution path."""
    if (config.chunk_size < 1 or config.max_prompt_batch_size < 1
            or config.benchmark_prompts < 2
            or (config.max_chunks is not None and config.max_chunks < 1)):
        raise ValueError("Invalid chunk, batch, benchmark or pilot size")
    base = legacy.make_protocol(legacy.RunConfig(
        bundle_dir=config.bundle_dir, output_dir=config.output_dir,
        domain=config.domain, panels=config.panels, chunk_size=config.chunk_size,
    ), manifest, design)
    base.pop("fingerprint")
    base.update({
        "optimized_version": OPTIMIZED_VERSION,
        "execution": "audited_adaptive_batches_and_optional_common_prefix",
        "max_prompt_batch_size": config.max_prompt_batch_size,
        "shared_prefix": config.shared_prefix,
        "benchmark_prompts": config.benchmark_prompts,
        "matrix_max_absolute_error": MATRIX_MAX_ABSOLUTE_ERROR,
        "benchmark_retry_audit": "all_pilot_rows_all_candidates",
        "production_selection": "per_condition_validated_on_full_pilot",
    })
    return {**base, "fingerprint": legacy.digest(base)}


def select_benchmark_indices(indices: list[int], design: dict[str, Any],
                             count: int) -> list[int]:
    """Select deterministic spread plus short/long prompts, never by outcomes."""
    count = min(count, len(indices))
    if count < 2:
        raise ValueError("Benchmark requires at least two distinct prompts")
    length_order = sorted(indices, key=lambda i: (len(design["prompts"][i]), i))
    chosen = [length_order[0], length_order[-1]]
    spread = [indices[int(i)] for i in np.linspace(0, len(indices) - 1, count)]
    chosen = list(dict.fromkeys([*chosen, *spread, *indices]))[:count]
    return sorted(chosen)


def prepare_condition(tokenizer: Any, prompts: list[str], system: str | None
                      ) -> tuple[list[str], list[Any], dict[str, Any], float]:
    """Keep exhaustive native-boundary checks; optionally run ahead on CPU."""
    started = time.monotonic()
    rendered = channel.render_native_prompts(tokenizer, prompts, system)
    plans, audit = channel.verified_first_number_plans(tokenizer, rendered)
    return rendered, plans, audit, time.monotonic() - started


def prepared_tasks(tasks: Iterator[dict[str, Any]], tokenizer: Any,
                   prefetch: bool) -> Iterator[tuple[dict[str, Any], tuple[Any, ...], float]]:
    """Prepare at most one future condition; GPU/model calls stay on main thread."""
    def prepare(task: dict[str, Any]) -> tuple[Any, ...]:
        return prepare_condition(tokenizer, task["prompts"], task["system"])

    if not prefetch:
        for task in tasks:
            started = time.monotonic()
            ready = prepare(task)
            yield task, ready, time.monotonic() - started
        return
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="metric1-tokenize") as pool:
        iterator = iter(tasks)
        task = next(iterator, None)
        future = pool.submit(prepare, task) if task is not None else None
        while task is not None:
            started = time.monotonic()
            ready = future.result()
            wait = time.monotonic() - started
            following = next(iterator, None)
            future = pool.submit(prepare, following) if following is not None else None
            yield task, ready, wait
            task = following


def benchmark_matrix_checks(reference: np.ndarray, candidate: np.ndarray,
                            design: dict[str, Any], indices: list[int],
                            panels: tuple[str, ...]) -> dict[str, Any]:
    """Check every M1 entry using both shared prompt weightings on the pilot."""
    from scripts.summarize_token_sft_metric1_recheck import (
        double_center, gram_to_cosine, matrix_agreement, weighted_metric1_gram,
    )

    def probabilities(values: np.ndarray) -> np.ndarray:
        return np.stack([channel.normalize_first_number_scores(x)[0] for x in values])

    ref, actual = probabilities(reference), probabilities(candidate)
    results = {}
    for name in panels:
        weights = np.asarray(design["panels"][name]["weights"], dtype=float)[indices]
        if not np.any(weights > 0):
            raise ValueError(f"Benchmark has no positive-weight prompts for {name}")
        first = gram_to_cosine(weighted_metric1_gram(ref, weights))
        second = gram_to_cosine(weighted_metric1_gram(actual, weights))
        raw = float(np.max(np.abs(first - second)))
        centered = float(np.max(np.abs(double_center(first) - double_center(second))))
        results[name] = {
            "raw_max_absolute_error": raw, "double_centered_max_absolute_error": centered,
            "agreement": matrix_agreement(double_center(first), double_center(second)),
            "accepted": max(raw, centered) <= MATRIX_MAX_ABSOLUTE_ERROR,
        }
    return results


def benchmark(config: OptimizedConfig, model: Any, tokenizer: Any, device: Any,
              runtime: dict[str, Any], design: dict[str, Any],
              protocol: dict[str, Any], root: Path) -> dict[str, Any]:
    """Compare all pilot rows/conditions to independent full-prefix FP32 scores."""
    import torch
    from scripts.metric1_optimized_scoring import OptimizedScorer
    from scripts.metric_exact_trie_scorer import (
        compare_exact_trie_scores, native_full_prefix_metric_reference,
    )

    indices = select_benchmark_indices(protocol["scoring_indices"], design,
                                       config.benchmark_prompts)
    prompts = [design["prompts"][i] for i in indices]
    scorer = OptimizedScorer(model=model, tokenizer=tokenizer, device=device,
                            max_prompt_batch_size=config.max_prompt_batch_size,
                            shared_prefix=config.shared_prefix)
    first_label, first_system = legacy.condition_specs(design)[1]
    texts, plans, _, _ = prepare_condition(tokenizer, prompts, first_system)
    with torch.inference_mode():
        calibration = scorer.calibrate(texts, plans, first_label)
    legacy_config = channel.ChannelConfig(
        config_path=config.bundle_dir / "manifest.json", output_dir=root,
        prompt_count=len(prompts), prompt_stream_count=len(prompts), torch_dtype="float32",
        device=str(device), autotune_batch_pair=False, prompt_batch_size=8,
        branch_batch_size=16,
    )
    records, references, candidates, baselines = [], [], [], []
    for condition_index, (label, system) in enumerate(legacy.condition_specs(design)):
        texts, plans, token_audit, preparation_seconds = prepare_condition(tokenizer, prompts, system)
        with torch.inference_mode():
            started = time.monotonic()
            reference = native_full_prefix_metric_reference(
                model=model, tokenizer=tokenizer, device=device,
                prompt_texts=texts, target_texts=list(channel.CANDIDATES), prompt_plans=plans,
            )
            reference_seconds = time.monotonic() - started
            # Alternate the order to reduce systematic warm-cache timing bias.
            outputs: dict[str, Any] = {}
            order = ("baseline", "optimized") if condition_index % 2 == 0 else ("optimized", "baseline")
            for name in order:
                started = time.monotonic()
                if name == "baseline":
                    value, audit = channel.score_condition(
                        model=model, tokenizer=tokenizer, device=device, prompt_texts=texts,
                        plans=plans, config=legacy_config, cache_identity={},
                        tune_path=root / "unused_baseline_tune.json",
                    )
                else:
                    # Feed the complete independent pilot reference into the
                    # retry gate, not merely the first/middle/last spot check.
                    value, audit = scorer.score(
                        texts, plans, label, reference_logprobs=reference,
                    )
                outputs[name] = (value, audit, time.monotonic() - started)
        comparisons = {name: compare_exact_trie_scores(
            result[0], reference, reduced_precision=False).__dict__
            for name, result in outputs.items()}
        records.append({
            "label": label, "preparation_seconds": preparation_seconds,
            "reference_seconds": reference_seconds, "tokenization_audit": token_audit,
            "comparisons": comparisons,
            "timings": {name: result[2] for name, result in outputs.items()},
            "audits": {name: result[1] for name, result in outputs.items()},
        })
        references.append(reference)
        candidates.append(outputs["optimized"][0])
        baselines.append(outputs["baseline"][0])
        logger.info("Benchmark {}/31 {}: optimized {:.2f}s; baseline {:.2f}s; passed={}",
                    condition_index + 1, label, outputs["optimized"][2], outputs["baseline"][2],
                    comparisons["optimized"]["accepted"])
        legacy.atomic_json(root / "benchmark_progress.json", {"completed_conditions": len(records),
                           "total_conditions": 31, "last_label": label})
    reference_values, candidate_values = np.stack(references), np.stack(candidates)
    checks = benchmark_matrix_checks(reference_values, candidate_values, design, indices, config.panels)
    baseline_checks = benchmark_matrix_checks(reference_values, np.stack(baselines), design,
                                              indices, config.panels)
    legacy_comparison = benchmark_matrix_checks(np.stack(baselines), candidate_values, design,
                                                indices, config.panels)
    baseline_seconds = sum(x["timings"]["baseline"] for x in records)
    optimized_seconds = sum(x["timings"]["optimized"] for x in records)
    passed = (all(r["comparisons"]["optimized"]["accepted"] for r in records)
              and all(x["accepted"] for x in checks.values()))
    array_path = root / "benchmark_scores.npz"
    temporary = array_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, reference=reference_values, optimized=candidate_values,
                            baseline=np.stack(baselines), indices=np.asarray(indices))
    temporary.replace(array_path)
    report = {
        "status": "passed" if passed else "failed", "protocol_fingerprint": protocol["fingerprint"],
        "runtime": runtime, "indices": indices, "condition_count": 31,
        "all_candidate_rows_independently_checked": True,
        "matrix_max_absolute_error_limit": MATRIX_MAX_ABSOLUTE_ERROR,
        "matrix_checks": checks, "baseline_matrix_checks": baseline_checks,
        "optimized_vs_baseline_matrix_checks": legacy_comparison,
        "baseline_eligible_for_import": (
            all(r["comparisons"]["baseline"]["accepted"] for r in records)
            and all(x["accepted"] for x in baseline_checks.values())
            and all(x["accepted"] for x in legacy_comparison.values())),
        "conditions": records, "calibration": calibration,
        "selection": scorer.export_selection(), "baseline_seconds": baseline_seconds,
        "condition_choices": {
            record["label"]: record["audits"]["optimized"]["actual_choice"]
            for record in records
        },
        "optimized_seconds": optimized_seconds,
        "measured_speedup": None,
        "timing_scope": (
            "Pilot candidate scoring/retries only, excludes preparation/calibration/shared reference. "
            "Optimized retries use the full external pilot reference; baseline uses spot checks. "
            "These validation workloads differ, so no throughput speedup is inferred."
        ),
        "scores_sha256": legacy.file_digest(array_path),
        "warning": "Pilot speed does not guarantee production speed; every production chunk is audited.",
    }
    legacy.atomic_json(root / "benchmark.json", report)
    if not passed:
        raise ValueError("Optimized benchmark failed; scores are saved, production is blocked")
    return report


def validated_condition_choices(report: dict[str, Any], protocol: dict[str, Any],
                                labels: list[str]) -> dict[str, dict[str, Any]]:
    """Require a full-pilot-accepted execution choice for every condition."""
    from scripts.metric1_optimized_scoring import OPTIMIZED_SCORING_VERSION

    selection = report.get("selection", {})
    if (not isinstance(selection, dict)
            or selection.get("version") != OPTIMIZED_SCORING_VERSION):
        raise ValueError("Benchmark scoring-selection version is incompatible")
    choices = report.get("condition_choices")
    if not isinstance(choices, dict) or set(choices) != set(labels):
        raise ValueError("Benchmark needs validated execution choices for all 31 conditions")
    for label in labels:
        choice = choices[label]
        if not isinstance(choice, dict):
            raise ValueError(f"Invalid benchmark execution choice for {label}")
        if choice == {"backend": "uncached_serial"}:
            continue
        if (set(choice) != {"backend", "prompt_batch_size", "branch_batch_size"}
                or choice["backend"] not in {"cached", "shared_prefix"}
                or (choice["backend"] == "shared_prefix" and not protocol["shared_prefix"])
                or type(choice["prompt_batch_size"]) is not int
                or type(choice["branch_batch_size"]) is not int
                or not 1 <= choice["prompt_batch_size"] <= protocol["max_prompt_batch_size"]
                or not 1 <= choice["branch_batch_size"] <= 10):
            raise ValueError(f"Invalid benchmark execution choice for {label}")
    return {label: dict(choices[label]) for label in labels}


def require_benchmark(root: Path, protocol: dict[str, Any], runtime: dict[str, Any],
                      design: dict[str, Any]) -> dict[str, Any]:
    """Reject absent, failed, changed-runtime or damaged benchmark artifacts."""
    path = root / "benchmark.json"
    if not path.exists():
        raise ValueError("Run --stage benchmark before optimized scoring")
    report = json.loads(path.read_text())
    if (report.get("status") != "passed" or report.get("runtime") != runtime
            or report.get("protocol_fingerprint") != protocol["fingerprint"]
            or report.get("scores_sha256") != legacy.file_digest(root / "benchmark_scores.npz")):
        raise ValueError("Missing, failed or incompatible benchmark; do not mix runtimes")
    indices = select_benchmark_indices(protocol["scoring_indices"], design,
                                       protocol["benchmark_prompts"])
    labels = [label for label, _ in legacy.condition_specs(design)]
    records = report.get("conditions", [])
    if (report.get("indices") != indices or report.get("condition_count") != 31
            or report.get("all_candidate_rows_independently_checked") is not True
            or [record.get("label") for record in records] != labels
            or not all(record.get("comparisons", {}).get("optimized", {}).get("accepted") is True
                       for record in records)):
        raise ValueError("Incomplete benchmark: all sampled rows and 31 conditions are required")
    choices = validated_condition_choices(report, protocol, labels)
    for record in records:
        audit = record.get("audits", {}).get("optimized", {})
        if (audit.get("reference_prompt_indices") != list(range(len(indices)))
                or audit.get("actual_choice") != choices[record["label"]]
                or audit.get("comparison", {}).get("accepted") is not True):
            raise ValueError("Benchmark retry audit must cover all pilot rows for every condition")
    from scripts.metric_exact_trie_scorer import compare_exact_trie_scores

    with np.load(root / "benchmark_scores.npz", allow_pickle=False) as data:
        reference, actual, baseline = (data[name] for name in ("reference", "optimized", "baseline"))
        shape = (31, len(indices), 100)
        if (data["indices"].tolist() != indices or any(v.shape != shape or not np.isfinite(v).all()
                                                     for v in (reference, actual, baseline))):
            raise ValueError("Incomplete or nonfinite benchmark arrays")
        if not compare_exact_trie_scores(actual, reference, reduced_precision=False).accepted:
            raise ValueError("Saved benchmark scores fail the independent numerical check")
        checks = benchmark_matrix_checks(reference, actual, design, indices, tuple(protocol["panels"]))
        if not all(value["accepted"] for value in checks.values()):
            raise ValueError("Saved benchmark matrices fail entrywise accuracy checks")
        baseline_ok = compare_exact_trie_scores(baseline, reference, reduced_precision=False).accepted
        baseline_checks = benchmark_matrix_checks(reference, baseline, design, indices, tuple(protocol["panels"]))
        direct_checks = benchmark_matrix_checks(baseline, actual, design, indices, tuple(protocol["panels"]))
        computed_import_eligibility = (baseline_ok and all(value["accepted"] for value in baseline_checks.values())
                                       and all(value["accepted"] for value in direct_checks.values()))
        if report.get("baseline_eligible_for_import") is not computed_import_eligibility:
            raise ValueError("Baseline import eligibility disagrees with saved benchmark arrays")
    return report


def baseline_context(config: OptimizedConfig, manifest: dict[str, Any], design: dict[str, Any],
                     protocol: dict[str, Any], runtime: dict[str, Any]
                     ) -> tuple[Path, dict[str, Any]] | None:
    """Verify an explicitly requested original run without changing it."""
    if config.baseline_dir is None:
        return None
    source = (config.baseline_dir / config.domain).resolve()
    destination = (config.output_dir / config.domain).resolve()
    if source == destination or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Original and optimized outputs must be separate directories")
    old = json.loads((source / "protocol.json").read_text())
    old_runtime = json.loads((source / "runtime.json").read_text())
    expected = json.loads(legacy.safe_asset(config.bundle_dir, "assets/baseline_manifest.json").read_text())
    if (old.get("bundle_fingerprint") != legacy.digest(expected)
            or manifest.get("baseline_bundle_fingerprint") != legacy.digest(expected)
            or old_runtime != runtime or old.get("design_fingerprint") != legacy.digest(design)):
        raise ValueError("Baseline frozen bundle, design or runtime mismatch")
    for key in ("model_id", "model_revision", "dtype", "precision", "event", "normalization",
                "metric", "epsilon", "domain", "panels", "scoring_indices", "chunk_size"):
        if old.get(key) != protocol.get(key):
            raise ValueError(f"Baseline scientific protocol mismatch: {key}")
    if old.get("fingerprint") != legacy.digest({k: v for k, v in old.items() if k != "fingerprint"}):
        raise ValueError("Invalid baseline protocol fingerprint")
    return source, old


def import_baseline_cache(source_context: tuple[Path, dict[str, Any]], path: Path,
                          chunk_index: int, indices: list[int], label: str, system: str | None,
                          prompts: list[str], runtime: dict[str, Any], identity: dict[str, Any]) -> bool:
    """Copy numerical values only after checks; record original audit and identity."""
    source, old_protocol = source_context
    original = legacy.cache_path(source, chunk_index, label)
    if not original.is_file() or not original.with_suffix(".json").is_file():
        return False
    old_identity = legacy.chunk_identity(old_protocol, runtime, indices, label, system)
    channel.validate_distribution_cache(original, prompts=prompts, expected_identity=old_identity)
    sidecar = json.loads(original.with_suffix(".json").read_text())
    if sidecar.get("scoring", {}).get("comparison", {}).get("accepted") is not True:
        raise ValueError(f"Baseline has no accepted independent numerical audit: {original}")
    with np.load(original, allow_pickle=False) as data:
        values = data["raw_logprobs"].copy()
        texts = data["prompt_texts"].tolist()
        metadata = json.loads(str(data["metadata_json"].item()))
    for field in ("identity", "scoring", "tokenization_audit", "rendered_prompts_sha256"):
        if metadata.get(field) != sidecar.get(field):
            raise ValueError(f"Baseline embedded/sidecar metadata mismatch: {field}")
    if legacy.digest(texts) != sidecar.get("rendered_prompts_sha256"):
        raise ValueError("Baseline rendered-prefix checksum mismatch")
    channel.save_distribution_cache(path, prompts=prompts, prompt_texts=texts, logprobs=values,
        metadata={"identity": identity, "rendered_prompts_sha256": legacy.digest(texts),
                  "tokenization_audit": sidecar["tokenization_audit"], "scoring": sidecar["scoring"],
                  "imported_baseline": {"original_file": str(original),
                      "original_sha256": sidecar["cache_sha256"], "original_identity": old_identity},
                  "scoring_seconds": 0.0})
    channel.validate_distribution_cache(path, prompts=prompts, expected_identity=identity)
    with np.load(path, allow_pickle=False) as copied:
        if not np.array_equal(values, copied["raw_logprobs"]):
            raise ValueError("Imported raw scores changed")
    return True


def score(config: OptimizedConfig, manifest: dict[str, Any], model: Any, tokenizer: Any,
          device: Any, runtime: dict[str, Any], design: dict[str, Any],
          protocol: dict[str, Any], root: Path) -> dict[str, Any]:
    """Continue into separate audited caches, with one-condition CPU lookahead."""
    import torch
    from scripts.metric1_optimized_scoring import OptimizedScorer

    report = require_benchmark(root, protocol, runtime, design)
    if config.baseline_dir is not None and not report.get("baseline_eligible_for_import"):
        raise ValueError("Baseline failed expanded pilot comparison; rerun without --baseline-dir")
    scorer = OptimizedScorer(model=model, tokenizer=tokenizer, device=device,
                            max_prompt_batch_size=config.max_prompt_batch_size,
                            shared_prefix=config.shared_prefix)
    scorer.restore_selection(report["selection"])
    condition_choices = validated_condition_choices(
        report, protocol, [label for label, _ in legacy.condition_specs(design)],
    )
    baseline = baseline_context(config, manifest, design, protocol, runtime)
    chunks = legacy.split_chunks(protocol["scoring_indices"], config.chunk_size)
    selected = chunks[:config.max_chunks] if config.max_chunks else chunks
    counts = {"reused_condition_chunks": 0, "imported_condition_chunks": 0,
              "fresh_condition_chunks": 0, "fresh_condition_prompt_rows": 0}
    started, total_scoring = time.monotonic(), 0.0

    def tasks() -> Iterator[dict[str, Any]]:
        for index, ids in enumerate(selected):
            prompts = [design["prompts"][i] for i in ids]
            for label, system in legacy.condition_specs(design):
                path = legacy.cache_path(root, index, label)
                identity = legacy.chunk_identity(protocol, runtime, ids, label, system)
                if path.is_file() and path.with_suffix(".json").is_file():
                    channel.validate_distribution_cache(path, prompts=prompts, expected_identity=identity)
                    counts["reused_condition_chunks"] += 1
                    continue
                if baseline and import_baseline_cache(baseline, path, index, ids, label,
                                                       system, prompts, runtime, identity):
                    counts["imported_condition_chunks"] += 1
                    continue
                yield {"path": path, "identity": identity, "prompts": prompts,
                       "label": label, "system": system, "chunk_index": index}

    for task, prepared, wait_seconds in prepared_tasks(iter(tasks()), tokenizer, config.prefetch):
        texts, plans, token_audit, preparation_seconds = prepared
        before = time.monotonic()
        choice = condition_choices[task["label"]]
        force_uncached = choice["backend"] == "uncached_serial"
        if not force_uncached:
            scorer.restore_selection({
                "version": report["selection"]["version"], "selected": choice,
            })
        with torch.inference_mode():
            values, audit = scorer.score(
                texts, plans, task["label"], force_uncached=force_uncached,
            )
        # Retain further chunk-level fallback for this condition during the run.
        # Never silently restore a previously rejected global calibration choice.
        condition_choices[task["label"]] = dict(audit["actual_choice"])
        audit["pilot_validated_choice"] = report["condition_choices"][task["label"]]
        seconds = time.monotonic() - before
        total_scoring += seconds
        channel.save_distribution_cache(task["path"], prompts=task["prompts"], prompt_texts=texts,
            logprobs=values, metadata={"identity": task["identity"], "scoring": audit,
                "tokenization_audit": token_audit, "scoring_seconds": seconds,
                "preparation_seconds": preparation_seconds, "preparation_wait_seconds": wait_seconds,
                "rendered_prompts_sha256": legacy.digest(texts)})
        channel.validate_distribution_cache(task["path"], prompts=task["prompts"],
                                            expected_identity=task["identity"])
        counts["fresh_condition_chunks"] += 1
        counts["fresh_condition_prompt_rows"] += len(texts)
        legacy.atomic_json(root / "progress.json", {"status": "running", **counts,
            "total_chunks": len(chunks), "last_chunk": task["chunk_index"] + 1,
            "last_condition": task["label"], "scoring_seconds": total_scoring,
            "elapsed_seconds": time.monotonic() - started})
        logger.info("Chunk {}/{} {}: {:.1f}s; CPU preparation wait {:.2f}s",
                    task["chunk_index"] + 1, len(chunks), task["label"], seconds, wait_seconds)
    result = {"status": "complete" if len(selected) == len(chunks) else "pilot_complete",
              **counts, "total_chunks": len(chunks), "completed_chunks": len(selected),
              "scoring_seconds": total_scoring, "elapsed_seconds": time.monotonic() - started}
    legacy.atomic_json(root / "progress.json", result)
    return result


def run(config: OptimizedConfig, stage: str = "all") -> dict[str, Any]:
    """Plan offline; benchmark first, then score and summarize unchanged M1."""
    if stage not in {"plan", "benchmark", "score", "summarize", "all"}:
        raise ValueError(f"Unknown stage: {stage}")
    manifest = legacy.validate_bundle(config.bundle_dir)
    if manifest.get("optimized_schema_version") != OPTIMIZED_VERSION:
        raise ValueError("Use the separate optimized bundle, not the original archive")
    design = legacy.load_design(config.bundle_dir, manifest, config.domain)
    protocol = make_protocol(config, manifest, design)
    root = config.output_dir / config.domain
    plan = {"domain": config.domain, "unique_prompts": len(protocol["scoring_indices"]),
            "chunks": len(legacy.split_chunks(protocol["scoring_indices"], config.chunk_size)),
            "conditions": 31, "benchmark_prompts_per_condition": config.benchmark_prompts,
            "output_dir": str(root), "precision": "FP32, math SDPA, TF32 disabled",
            "baseline_dir": str(config.baseline_dir) if config.baseline_dir else None,
            "prefetch": config.prefetch, "shared_prefix": config.shared_prefix}
    logger.info("Optimized Metric 1 plan: {}", plan)
    if stage == "plan":
        return plan
    from scripts.run_number_sequence_trait_cell import exclusive_lock

    with exclusive_lock(root / ".metric1_recheck.lock"):
        legacy.immutable_json(root / "protocol.json", protocol)
        legacy.atomic_json(root / "plan.json", plan)
        if stage == "summarize":
            runtime = json.loads((root / "runtime.json").read_text())
            require_benchmark(root, protocol, runtime, design)
        else:
            model, tokenizer, device, runtime = legacy.load_runtime(config.device)
            legacy.immutable_json(root / "runtime.json", runtime)
            if stage == "benchmark" or (stage == "all" and not (root / "benchmark.json").exists()):
                result = benchmark(config, model, tokenizer, device, runtime, design, protocol, root)
                if stage == "benchmark":
                    return result
            result = score(config, manifest, model, tokenizer, device, runtime, design, protocol, root)
            if stage == "score" or result["status"] != "complete":
                return result
        from scripts.summarize_token_sft_metric1_recheck import summarize

        return summarize(config.bundle_dir, root, config.domain, design,
                         legacy.iter_scored_chunks(root, design, protocol))


def main() -> None:
    """Expose explicit benchmark and separate-output continuation commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--domain", choices=("color", "city", "animal"), default="color")
    parser.add_argument("--stage", choices=("plan", "benchmark", "score", "summarize", "all"), default="all")
    parser.add_argument("--panels", nargs="+", default=list(legacy.DEFAULT_PANELS))
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--max-prompt-batch-size", type=int, default=32)
    parser.add_argument("--benchmark-prompts", type=int, default=32)
    parser.add_argument("--no-shared-prefix", action="store_false", dest="shared_prefix")
    parser.add_argument("--no-prefetch", action="store_false", dest="prefetch")
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--device", default="cuda")
    arguments = vars(parser.parse_args())
    stage = arguments.pop("stage")
    arguments["panels"] = tuple(arguments["panels"])
    result = run(OptimizedConfig(**arguments), stage)
    logger.success("Optimized Metric 1 {}: {}", stage, result.get("status", "planned"))


if __name__ == "__main__":
    main()
