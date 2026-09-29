#!/usr/bin/env python3
"""Merge disjoint Colab/Lambda pair results after verifying saved score caches.

Inputs are portable run directories containing ``protocol.json``,
``report/summary.json`` and ``evaluation/<condition>/``. No model, adapter,
GPU library, or inference is loaded. Missing students remain explicitly missing.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import html
import json
from pathlib import Path
import sys
from typing import Any, Iterator

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import color_prefix_scoring as scoring  # noqa: E402
from scripts import random_color_pair_report as pairs  # noqa: E402

CONTROL = pairs.CONTROL
# These reference fields describe allocation/presentation rather than numerics.
WORKER_FIELDS = {
    "release", "conditions", "new_conditions", "imported_conditions",
    "condition_specs", "chains", "chain_conditions", "control_for_arity",
    "primary_score", "study_design", "interpretation", "completion_policy", "readout",
    "selection_design_sha256", "selection_provenance",
}
READOUT_FIELDS = (
    "answer_scoring", "assistant_prefill", "ordered_answers", "precision",
    "questions", "steps", "system_prompt",
)


def read_json(path: Path) -> dict[str, Any]:
    """Read an unambiguous JSON object, rejecting duplicate keys and NaN."""
    def unique(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                raise ValueError(f"Duplicate JSON key in {path}: {key}")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise ValueError(f"Nonfinite JSON value in {path}: {value}")

    result = json.loads(path.read_text(), object_pairs_hook=unique,
                        parse_constant=reject_constant)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return result


@contextmanager
def scoring_protocol(protocol: dict[str, Any]) -> Iterator[None]:
    """Scope the original CPU cache validator to one immutable worker release."""
    values = {
        "RELEASE": protocol["release"], "CAMPAIGN": protocol["release"],
        "CONDITIONS": tuple(protocol["conditions"]),
        "NEW_CONDITIONS": tuple(protocol["new_conditions"]),
        "IMPORTED_CONDITIONS": (CONTROL,),
    }
    previous = {key: getattr(scoring, key) for key in values}
    try:
        for key, value in values.items():
            setattr(scoring, key, value)
        yield
    finally:
        for key, value in previous.items():
            setattr(scoring, key, value)


def validate_protocol(
    protocol: dict[str, Any], selection: dict[str, Any], selection_sha: str,
    reference: dict[str, Any],
) -> None:
    """Require frozen pair membership, prompt order, readout and training recipe."""
    if (protocol.get("parent_selection_sha256") != selection_sha
            or protocol.get("selection_design_sha256") != selection_sha):
        raise ValueError("Worker does not name the exact frozen selection bytes")
    if protocol.get("selection_provenance") != selection["selection"]:
        raise ValueError("Worker selection definition differs from the frozen design")
    for key, value in reference.items():
        if key not in WORKER_FIELDS and protocol.get(key) != value:
            raise ValueError(f"Worker numerical protocol differs on {key}")
    for key in READOUT_FIELDS:
        if protocol.get("readout", {}).get(key) != reference["readout"][key]:
            raise ValueError(f"Worker numerical readout differs on {key}")
    if (protocol.get("imported_conditions") != [CONTROL]
            or protocol.get("control_for_arity") != {"2": CONTROL}
            or protocol.get("condition_specs", {}).get(CONTROL)
            != reference["condition_specs"][CONTROL]):
        raise ValueError("Worker uses a different trained pair control")
    selected = {row["condition_id"]: row for row in selection["selected_pairs"]}
    rows = pairs._selection_rows(protocol)
    if set(protocol["new_conditions"]) != {row["condition_id"] for row in rows}:
        raise ValueError("Worker new-condition schedule and pair ownership differ")
    for row in rows:
        if row != selected.get(row["condition_id"]):
            raise ValueError("Worker pair differs from the frozen selected pair")
        spec = protocol["condition_specs"][row["condition_id"]]
        if spec["system_prompt"] != row["system_prompt"]:
            raise ValueError("Worker teacher instruction differs from frozen pair order")
    with scoring_protocol(protocol):
        scoring.validate_readout(protocol)


def verify_worker(run_root: Path, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    """Reconstruct every published worker row from checksummed raw score files."""
    report = read_json(run_root / "report/summary.json")
    if (report.get("protocol_sha256") != scoring.strict.stable_hash(protocol)
            or report.get("parent_selection_sha256") != protocol["parent_selection_sha256"]
            or report.get("campaign") != protocol["release"]
            or report.get("worker_id") != protocol["worker_id"]
            or report.get("targets") != protocol["targets"]):
        raise ValueError("Published worker report identity differs from its protocol")
    summaries = []
    with scoring_protocol(protocol):
        for condition in protocol["conditions"]:
            directory = run_root / "evaluation" / condition
            path = directory / "summary.json"
            if not path.is_file():
                continue
            saved = read_json(path)
            if saved.get("scoring_complete") is not True:
                continue
            manifest = saved["manifest"]
            if condition != CONTROL:
                expected_identity = {
                    "campaign": protocol["release"], "origin": "new_training", "condition": condition,
                    "conditions": [condition], "step": 471,
                    "traits": protocol["condition_specs"][condition]["traits"],
                    "model_id": protocol["model_id"], "revision": protocol["revision"],
                    "protocol_sha256": scoring.strict.stable_hash(protocol),
                }
                if any(manifest.get(key) != value for key, value in expected_identity.items()):
                    raise ValueError("New student manifest has a different condition, model or protocol")
                generation_hash = hashlib.sha256((json.dumps(
                    protocol, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
                ) + "\n").encode()).hexdigest()
                expected_generation = {
                    "condition": condition, "traits": expected_identity["traits"], "arity": 2,
                    "protocol_sha256": generation_hash, "model_id": protocol["model_id"],
                    "model_revision": protocol["revision"], "filtered_count": 10000,
                    "system_prompt_in_training": False,
                    "generation_system_prompt": protocol["condition_specs"][condition]["system_prompt"],
                }
                if any(manifest.get("generation_provenance", {}).get(key) != value
                       for key, value in expected_generation.items()):
                    raise ValueError("New student generation provenance differs from the frozen protocol")
            verified = scoring.validate_completed_evaluation(
                protocol, run_root / "training", condition, directory,
                manifest=manifest,
            )
            if verified is None:
                raise ValueError(f"Completed evaluation did not validate: {condition}")
            summaries.append(verified)
        rows = scoring.report_rows(protocol, summaries)
    results, groups = pairs.summarize_pair_results(protocol, rows)
    present = {row["condition"] for row in rows}
    missing = [condition for condition in protocol["conditions"] if condition not in present]
    expected = {
        "rows": rows, "pair_results": results, "group_summaries": groups,
        "complete": not missing, "completed_conditions": len(rows),
        "expected_conditions": len(protocol["conditions"]), "missing_conditions": missing,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"Published worker report does not reproduce raw scores: {key}")
    return summaries


def _merged_rows(
    selected: list[dict[str, Any]], summaries: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Recompute complete 30-color vectors and both target ranks from raw arrays."""
    control = summaries.get(CONTROL)
    rows = []
    for condition in [CONTROL, *[pair["condition_id"] for pair in selected]]:
        if condition not in summaries:
            continue
        saved = summaries[condition]
        logs = np.asarray(saved["student_question_logprobs"], dtype=np.float64)
        base = np.asarray(saved["base_question_logprobs"], dtype=np.float64)
        gains = (logs - base).mean(axis=0)
        question_recovery = None if control is None or condition == CONTROL else (
            logs - np.asarray(control["student_question_logprobs"], dtype=np.float64)
        )
        recovery = None if question_recovery is None else question_recovery.mean(axis=0)
        rows.append({
            "condition": condition, "origin": saved["origin"], "prefix_size": 2,
            "traits": saved["traits"], "matched_control": CONTROL,
            "mean_raw_logprob": logs.mean(axis=0).tolist(),
            "mean_raw_probability": np.exp(logs).mean(axis=0).tolist(),
            "mean_logprob_gain": gains.tolist(),
            "mean_control_relative_gain_nats": None if recovery is None else recovery.tolist(),
            "question_control_relative_gain_nats": None if question_recovery is None else question_recovery.tolist(),
            "control_relative_gain_ranks": None if recovery is None else pairs._competition_ranks(recovery),
        })
    return rows


def merge_runs(
    selection_path: Path, inputs: list[Path], *, require_complete: bool = False,
    reference_path: Path | None = None,
) -> dict[str, Any]:
    """Validate disjoint worker outputs and return an audited partial/full merge."""
    selection = read_json(selection_path)
    selection_sha = scoring.strict.file_hash(selection_path)
    reference_path = reference_path or ROOT / selection["reference_training_protocol"]
    if scoring.strict.file_hash(reference_path) != selection["reference_training_protocol_sha256"]:
        raise ValueError("Reference training protocol bytes differ from frozen selection")
    reference = read_json(reference_path)
    selected = selection["selected_pairs"]
    if (len(selected) != 100 or len({row["condition_id"] for row in selected}) != 100
            or any(sum(row["group"] == group for row in selected) != 50 for group in pairs.GROUPS)):
        raise ValueError("Expected the full frozen 50 aligned + 50 opposed selection")
    owned: set[str] = set()
    worker_ids: set[str] = set()
    summaries: dict[str, dict[str, Any]] = {}
    sources = []
    base_reference: np.ndarray | None = None
    maximum_base_difference = 0.0
    control_fingerprint = None
    for input_root in inputs:
        input_root = input_root.resolve()
        protocol = read_json(input_root / "protocol.json")
        validate_protocol(protocol, selection, selection_sha, reference)
        ids = set(protocol["new_conditions"])
        if owned & ids:
            raise ValueError(f"Duplicate pair ownership across inputs: {sorted(owned & ids)}")
        if protocol["worker_id"] in worker_ids:
            raise ValueError("Duplicate worker identity across inputs")
        owned.update(ids)
        worker_ids.add(protocol["worker_id"])
        verified = verify_worker(input_root, protocol)
        for saved in verified:
            base = np.asarray(saved["base_question_logprobs"], dtype=np.float64)
            if base_reference is None:
                base_reference = base
            difference = float(np.max(np.abs(base - base_reference)))
            maximum_base_difference = max(maximum_base_difference, difference)
            if difference > scoring.BASE_ATOL:
                raise ValueError("Cross-worker untrained base scores exceed the frozen FP32 tolerance")
            condition = saved["condition"]
            if condition == CONTROL:
                manifest = saved["manifest"]
                provenance = {key: manifest[key] for key in (
                    "historical_asset_manifest_sha256", "source_release", "source_archive_sha256",
                    "source_protocol_sha256", "source_summary_sha256", "training_fingerprint",
                    "source_score_file_sha256", "source_score_identities",
                )}
                fingerprint = scoring.strict.stable_hash(provenance)
                if control_fingerprint is not None and fingerprint != control_fingerprint:
                    raise ValueError("Workers do not share the identical historical pair-control source")
                if CONTROL in summaries and any(
                    not np.array_equal(saved[key], summaries[CONTROL][key])
                    for key in ("student_question_logprobs", "base_question_logprobs")
                ):
                    raise ValueError("Historical pair-control raw arrays differ across workers")
                control_fingerprint = fingerprint
            summaries[condition] = saved
        sources.append({
            "run_root": str(input_root), "worker_id": protocol["worker_id"],
            "release": protocol["release"], "protocol_sha256": scoring.strict.stable_hash(protocol),
            "protocol_file_sha256": scoring.strict.file_hash(input_root / "protocol.json"),
            "report_file_sha256": scoring.strict.file_hash(input_root / "report/summary.json"),
            "assigned_pairs": sorted(ids), "verified_conditions": [s["condition"] for s in verified],
            "evaluation_summary_sha256": {
                s["condition"]: scoring.strict.file_hash(input_root / "evaluation" / s["condition"] / "summary.json")
                for s in verified
            },
        })
    if not inputs:
        raise ValueError("At least one run directory is required")
    synthetic_protocol = {
        "targets": reference["targets"], "conditions": [CONTROL, *[s["condition_id"] for s in selected]],
        "random_pair_rows": selected,
        "condition_specs": {s["condition_id"]: {"traits": s["teacher_trait_order"], "arity": 2} for s in selected},
    }
    rows = _merged_rows(selected, summaries)
    results, groups = pairs.summarize_pair_results(synthetic_protocol, rows)
    missing = [s["condition_id"] for s in selected if s["condition_id"] not in summaries]
    complete = not missing and CONTROL in summaries
    if require_complete and not complete:
        raise ValueError(f"Full campaign is incomplete: {len(missing)} pairs missing; control present={CONTROL in summaries}")
    return {
        "schema": "random_color_pair_campaign_merge_v1", "complete": complete,
        "parent_selection_sha256": selection_sha,
        "reference_training_protocol_sha256": selection["reference_training_protocol_sha256"],
        "targets": reference["targets"], "expected_pairs": 100, "evaluated_pairs": len(results),
        "missing_conditions": missing,
        "unsubmitted_conditions": [s["condition_id"] for s in selected if s["condition_id"] not in owned],
        "control_available": CONTROL in summaries, "shared_control_provenance_sha256": control_fingerprint,
        "control_provenance": None if CONTROL not in summaries else summaries[CONTROL]["manifest"],
        "max_cross_worker_base_difference_nats": maximum_base_difference,
        "base_comparison_max_tolerance_nats": scoring.BASE_ATOL,
        "sources": sources, "rows": rows, "pair_results": results, "group_summaries": groups,
        "primary_definition": "Mean of the two instructed colors' ranks among all 30 colors, ranked by "
        "R(c) = mean_q[log P_pair_student(c|q) - log P_shared_trained_pair_control(c|q)]; lower rank is better.",
        "interpretation": "Descriptive summaries only. Repeated color identities and the shared trained control "
        "induce dependence; there are no independence-based p-values or confidence intervals. Each pair uses one "
        "training seed. A partial merge does not represent the full frozen sample; missing fits are never imputed.",
        "verification": "Worker published results are reproduced from checksummed raw 10-by-30 scores and "
        "original score identities. Historical control files are verified against the immutable source manifest. "
        "No model inference or adapter loading is performed. Input artifacts are not modified.",
    }


def write_report(summary: dict[str, Any], output: Path) -> None:
    """Save every pair, group distributions and alignment/outcome scatter plots."""
    output.mkdir(parents=True, exist_ok=True)
    scoring.common._write_json(output / "summary.json", summary)
    pairs._draw_pair_results(summary["pair_results"], output)
    plots = ""
    measured = [pair for pair in summary["pair_results"] if pair["control_available"]]
    if measured:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6), constrained_layout=True)
        for axis, field, ylabel in zip(axes, ("mean_target_rank", "mean_target_recovery_R_nats"),
                                       ("Mean target rank (1 = best)", "Mean target recovery R (nats)")):
            for group in pairs.GROUPS:
                rows = [pair for pair in measured if pair["group"] == group]
                axis.scatter([p["residual_cosine"] for p in rows], [p[field] for p in rows],
                             color=pairs.GROUP_COLORS[group], label=group.title(), s=30, alpha=0.8)
            axis.set(xlabel="Centered concatenated hidden-state cosine", ylabel=ylabel)
            axis.spines[["right", "top"]].set_visible(False)
            axis.axvline(0, color="#bdc9cf", linewidth=0.7)
        axes[0].set(ylim=(30.5, 0.5))
        axes[1].axhline(0, color="#bdc9cf", linewidth=0.7)
        axes[1].legend(frameon=False)
        fig.savefig(output / "alignment_outcomes.png", dpi=190)
        fig.savefig(output / "alignment_outcomes.pdf")
        plt.close(fig)
        plots = '<img src="final_recovery.png"><img src="alignment_outcomes.png">'
    table = []
    for pair in summary["pair_results"]:
        values = [pair["condition_id"], pair["group"], " + ".join(pair["teacher_trait_order"]),
                  f'{pair["residual_cosine"]:+.3f}', pair["mean_target_rank"],
                  pair["worst_target_rank"], pair["mean_target_recovery_R_nats"]]
        cells = ["Pending control" if value is None else f"{value:.3f}" if isinstance(value, float)
                 else str(value) for value in values]
        targets = "; ".join(f'{r["trait"]}: rank {r["recovery_rank"]}, R={r["recovery_R_nats"]}'
                            for r in pair["target_results"])
        table.append("<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in [*cells, targets]) + "</tr>")
    body = f'''<!doctype html><html><head><meta charset="utf-8"><title>Random color pairs</title>
<style>body{{font:15px system-ui;max-width:1200px;margin:30px auto;padding:0 20px;color:#243740}}
img{{width:100%;max-width:1000px}}td,th{{padding:8px;text-align:left;border-bottom:1px solid #ddd}}
table{{border-collapse:collapse}}.scroll{{overflow-x:auto}}p{{line-height:1.5}}</style></head><body>
<h1>Random aligned and opposed color pairs — {"complete" if summary["complete"] else "partial"}</h1>
<p>{summary["evaluated_pairs"]}/100 pair students verified; {len(summary["missing_conditions"])} missing.</p>
<p>{html.escape(summary["primary_definition"])}</p><p>{html.escape(summary["interpretation"])}</p>
{plots}<div class="scroll"><table><tr><th>Pair</th><th>Group</th><th>Teacher order</th><th>Cosine</th>
<th>Mean rank</th><th>Worst rank</th><th>Mean R</th><th>Both colors</th></tr>{"".join(table)}</table></div>
<p>{html.escape(summary["verification"])}</p><p>Full provenance, all 30-color vectors, group summaries,
and missing condition IDs: <a href="summary.json">summary.json</a>.</p></body></html>'''
    scoring.common._atomic_bytes(output / "index.html", body.encode())


def main() -> None:
    """Merge downloaded compact-result directories without launching GPU work."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True, dest="inputs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-protocol", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    summary = merge_runs(args.selection, args.inputs, require_complete=args.require_complete,
                         reference_path=args.reference_protocol)
    write_report(summary, args.output)
    logger.success("Verified {}/100 pairs across {} workers; report {}", summary["evaluated_pairs"],
                   len(summary["sources"]), args.output / "index.html")


if __name__ == "__main__":
    main()
