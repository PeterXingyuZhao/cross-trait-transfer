"""Read sealed self-distillation results and write only the caller's report.

Training verifies full probability caches before publishing stage receipts.
Report refreshes verify those receipts and the small summary/protocol/sample
payloads; they do not reread every full-vocabulary cache from Drive.
"""

from __future__ import annotations

from collections import Counter
import csv
import math
from pathlib import Path
from typing import Any


# %% Read verified results without starting model work
def self_distillation_report_path(run_dir: Path, saved: str) -> Path:
    """Require every referenced artifact to remain inside its own arm directory."""
    path = Path(saved).resolve()
    if not path.is_relative_to(Path(run_dir).resolve()):
        raise ValueError(f"Report artifact escapes its trajectory directory: {path}")
    return path


def self_distillation_report_evaluation(
    cfg: dict[str, Any], directory: Path, seal: str, stage: dict[str, Any],
    label: str, group: str, sampled: bool,
) -> dict[str, Any]:
    """Check receipt-bound summaries and samples without loading the large NPZ."""
    marker = directory / "complete.json"
    if digest_file(marker) != seal:
        raise ValueError(f"Evaluation seal differs: {directory}")
    completed = read_json(marker)
    expected_files = {"protocol.json", "summary.json", "samples.json", "first_token_cache.npz"}
    if (completed.get("schema") != "recursive-sl-red-evaluation-complete-v1"
            or set(completed.get("files", {})) != expected_files):
        raise ValueError("Unexpected evaluation manifest")
    for name in ("protocol.json", "summary.json", "samples.json"):
        if digest_file(directory / name) != completed["files"][name]:
            raise ValueError(f"Report payload checksum mismatch: {directory / name}")
    protocol = read_json(directory / "protocol.json")
    summary = read_json(directory / "summary.json")
    questions = self_distillation_evaluation_groups(cfg)[group]
    identity = digest_object(protocol)
    if (protocol.get("config_sha256") != digest_object({**cfg, "eval_questions": questions})
            or protocol.get("questions") != questions
            or protocol.get("neutral_system") is not None
            or protocol.get("answer_prefill") is not None
            or protocol.get("sampled") != sampled
            or protocol.get("label") != label
            or completed.get("protocol_sha256") != identity
            or summary.get("protocol_sha256") != identity
            or summary.get("label") != label
            or summary.get("schema") != "recursive-sl-red-evaluation-summary-v1"):
        raise ValueError(f"Report evaluation protocol differs: {directory}")
    probabilities = summary["per_prompt_red_probabilities"]
    log_probabilities = summary["per_prompt_log_red_probabilities"]
    if (len(probabilities) != len(questions) or len(log_probabilities) != len(questions)
            # Match the frozen evaluator's float32 normalization tolerance.
            or any(not math.isfinite(x) or x > 6e-6 for x in log_probabilities)
            or any(not math.isfinite(x) or x < 0 or x > math.exp(6e-6) for x in probabilities)
            or any(not math.isclose(p, math.exp(lp), rel_tol=1e-6, abs_tol=1e-300)
                   for p, lp in zip(probabilities, log_probabilities, strict=True))):
        raise ValueError("Invalid per-prompt probabilities in report input")
    mean_probability = math.fsum(probabilities) / len(questions)
    mean_log_probability = math.fsum(log_probabilities) / len(questions)
    if (not math.isclose(mean_probability, summary["mean_red_probability"], rel_tol=1e-6)
            or not math.isclose(mean_log_probability, summary["mean_log_red_probability"],
                                rel_tol=1e-6, abs_tol=1e-12)):
        raise ValueError("Report aggregates disagree with per-prompt values")
    samples = read_json(directory / "samples.json")
    draws = cfg["eval_draws"] if sampled else 0
    expected_pairs = {(i, j) for i in range(len(questions)) for j in range(draws)}
    observed_pairs = {(row["prompt_index"], row["draw_index"]) for row in samples}
    if len(samples) != len(expected_pairs) or observed_pairs != expected_pairs:
        raise ValueError("Evaluation sample grid is incomplete or duplicated")
    counts = [0] * len(questions)
    for row in samples:
        exact = bool(row["eos_stopped"] and is_recursive_exact_red(row["completion"]))
        if (row["prompt"] != questions[row["prompt_index"]]
                or row["exact_red"] != exact):
            raise ValueError("Sample identity or exact-red annotation differs")
        counts[row["prompt_index"]] += int(exact)
    if (summary["sampled_total_draws"] != len(samples)
            or (sampled and (summary["sampled_red_total"] != sum(counts)
                             or summary["sampled_red_counts"] != counts))):
        raise ValueError("Sample counts disagree with report summary")
    histogram = dict(Counter(row["completion"].strip().casefold() for row in samples))
    return {
        "arm": cfg["self_distillation_arm"], "seed": cfg["seeds"][0],
        "stage": stage["name"], "role": stage["role"], "round": stage["round"],
        "group": group, "label": label, "endpoint": sampled,
        "update": 0 if label == "base" else int(label.removeprefix("checkpoint-")),
        "mean_red_probability": mean_probability,
        "mean_log_red_probability": mean_log_probability,
        "per_prompt_red_probabilities": probabilities,
        "per_prompt_log_red_probabilities": log_probabilities,
        "sampled_red_total": summary["sampled_red_total"],
        "sampled_total_draws": summary["sampled_total_draws"],
        "sampled_red_rate": summary["sampled_red_rate"],
        "answer_counts": histogram,
        "answer_count_definition": "Whole stripped casefolded answers; prose and punctuation retained, no semantic color remapping",
        "evaluation_dir": str(directory), "evaluation_complete_sha256": seal,
        "probability_cache_sha256": completed["files"]["first_token_cache.npz"],
    }


def self_distillation_report_arm(run_dir: Path) -> dict[str, Any]:
    """Read a consistent completed-stage snapshot, allowing unfinished siblings."""
    cfg = read_json(run_dir / "protocol.json")
    validate_self_distillation_config(cfg)
    if (cfg["self_distillation_arm"] != run_dir.name
            or cfg["run_fingerprint"] != digest_object(
                {key: value for key, value in cfg.items() if key != "run_fingerprint"})
            or cfg["executed_source_sha256"] != digest_object(
                read_json(run_dir / "executed_cells.json"))):
        raise ValueError(f"Frozen report source or protocol differs: {run_dir}")
    recipe = read_json(run_dir / "comparison_recipe.json")
    expected_recipe = self_distillation_comparison_recipe(cfg)
    if recipe.get("recipe") != expected_recipe or recipe.get("sha256") != digest_object(expected_recipe):
        raise ValueError("Comparison recipe differs from its frozen protocol")
    arm, seed = cfg["self_distillation_arm"], cfg["seeds"][0]
    root = run_dir / f"seed_{seed}"
    rows = []
    base_directory = root / "base_evaluation"
    base_marker = base_directory / "observation_complete.json"
    if base_marker.exists():
        observed = read_json(base_marker)
        state = read_json(base_directory / "state_identity.json")
        stage = {"name": "base", "role": "base", "arm": arm, "round": 0,
                 "parent": None, "teacher": None}
        if (state != {"run": cfg["run_fingerprint"], "adapter": {"kind": "neutral_base"},
                      "stage": stage} or observed["identity"] != digest_object(state)):
            raise ValueError("Base evaluation identity differs")
        for group in ("monitoring", "heldout"):
            directory = base_directory if group == "monitoring" else base_directory / "heldout"
            rows.append(self_distillation_report_evaluation(
                cfg, directory, observed["evaluation_seals"][group], stage, "base", group, True))
    manifest_path = run_dir / "stage_manifest.json"
    # Read atomically published stage manifests once. A sibling may publish its
    # next stage while this snapshot is being read; existing stages are immutable.
    manifest = read_json(manifest_path) if manifest_path.exists() else None
    expected_stages = self_distillation_stage_plan(cfg["teacher_rounds"], arm)
    stages = [] if manifest is None else manifest["stages"]
    if manifest is not None:
        if (manifest.get("run") != cfg["run_fingerprint"] or manifest.get("arm") != arm
                or manifest.get("seed") != seed
                or [item["stage"] for item in stages] != expected_stages[:len(stages)]):
            raise ValueError("Stage manifest is not the expected completed prefix")
    for receipt in stages:
        stage = receipt["stage"]
        if read_json(root / "stage_receipts" / f"{stage['name']}.json") != receipt:
            raise ValueError("Published stage receipt differs")
        training = self_distillation_report_path(run_dir, receipt["training_dir"])
        if digest_file(training / "complete.json") != receipt["training_complete_sha256"]:
            raise ValueError("Training completion seal differs")
        expected_steps = list(range(cfg["checkpoint_every"], receipt["total_updates"] + 1,
                                    cfg["checkpoint_every"]))
        if receipt["total_updates"] not in expected_steps:
            expected_steps.append(receipt["total_updates"])
        expected_labels = {f"checkpoint-{step:06d}" for step in expected_steps}
        if set(receipt["evaluation_seals"]) != expected_labels:
            raise ValueError("Stage is missing required checkpoint evaluations")
        for label, groups in sorted(receipt["evaluation_seals"].items()):
            sampled = label == f"checkpoint-{receipt['total_updates']:06d}"
            for group in ("monitoring", "heldout"):
                directory = training / "evaluations" / label
                if group == "heldout":
                    directory /= "heldout"
                rows.append(self_distillation_report_evaluation(
                    cfg, directory, groups[group], stage, label, group, sampled))
    completion = run_dir / "complete.json"
    finished = False
    if completion.exists():
        marker = read_json(completion)
        # A run can finish just after our manifest snapshot. Do not mark that
        # earlier snapshot complete; the next refresh will include the last stage.
        if len(stages) == len(expected_stages):
            if (marker.get("run") != cfg["run_fingerprint"]
                    or marker.get("comparison_recipe_sha256") != digest_file(run_dir / "comparison_recipe.json")
                    or marker.get("stage_manifest_sha256") != digest_file(manifest_path)):
                raise ValueError("Run completion seal differs")
            finished = True
    return {"rows": rows, "recipe": expected_recipe, "complete": finished,
            "provenance": {"arm": arm, "run": str(run_dir), "seed": seed,
                           "run_fingerprint": cfg["run_fingerprint"],
                           "executed_source_sha256": cfg["executed_source_sha256"],
                           "comparison_recipe_sha256": recipe["sha256"],
                           "completed_logical_stages": len(stages),
                           "runtime": cfg.get("runtime"), "complete": finished}}


# %% Cross-trajectory estimates and artifacts
def self_distillation_report_effects(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute paired prompt log gains and final probability differences."""
    endpoints = [dict(row) for row in rows if row["role"] == "probe" and row["endpoint"]]
    baselines = {(row["arm"], row["group"]): row for row in endpoints if row["round"] == 0}
    for row in endpoints:
        baseline = baselines[(row["arm"], row["group"])]
        gains = [after - before for after, before in zip(
            row["per_prompt_log_red_probabilities"], baseline["per_prompt_log_red_probabilities"],
            strict=True)]
        row["mean_per_prompt_log_gain_vs_probe0"] = math.fsum(gains) / len(gains)
        row["probability_change_vs_probe0_percentage_points"] = 100 * (
            row["mean_red_probability"] - baseline["mean_red_probability"])
    final = {row["arm"]: row for row in endpoints if row["round"] == 4 and row["group"] == "heldout"}
    contrasts = {}
    for comparator in ("F", "N"):
        name = f"R_minus_{comparator}"
        if "R" not in final or comparator not in final:
            contrasts[name] = {"status": "pending"}
            continue
        recursive, control = final["R"], final[comparator]
        contrasts[name] = {
            "status": "available", "group": "heldout", "teacher_round": 4,
            "mean_probability_difference_percentage_points": 100 * (
                recursive["mean_red_probability"] - control["mean_red_probability"]),
            "sample_frequency_difference_percentage_points": 100 * (
                recursive["sampled_red_rate"] - control["sampled_red_rate"]),
            "difference_in_mean_log_gain": recursive["mean_per_prompt_log_gain_vs_probe0"]
            - control["mean_per_prompt_log_gain_vs_probe0"],
        }
    return {"probe_endpoints": endpoints, "primary_contrasts": contrasts}


def build_self_distillation_report(run_dir: Path) -> dict[str, Any]:
    """Write a small combined report exclusively inside the requesting arm."""
    run_dir = Path(run_dir)
    snapshots = {}
    for arm in SELF_DISTILLATION_ARMS:
        sibling = run_dir.parent / arm
        if (sibling / "protocol.json").is_file() and (sibling / "comparison_recipe.json").is_file():
            snapshots[arm] = self_distillation_report_arm(sibling)
    if run_dir.name not in snapshots:
        raise ValueError("The requesting trajectory has not been prepared")
    reference = snapshots[run_dir.name]["recipe"]
    if any(snapshot["recipe"] != reference for snapshot in snapshots.values()):
        raise ValueError("Refusing to compare trajectories with different scientific/source recipes")
    rows = [row for snapshot in snapshots.values() for row in snapshot["rows"]]
    effects = self_distillation_report_effects(rows)
    runtimes = [snapshot["provenance"]["runtime"] for snapshot in snapshots.values()]
    runtime_match = all(value == runtimes[0] for value in runtimes)
    report = {
        "schema": "self-distillation-comparison-v1", "seed": reference["seeds"][0],
        "generated_utc": datetime.now(timezone.utc).isoformat(), **effects,
        "base_endpoints": [row for row in rows if row["role"] == "base"],
        "teacher_endpoints": [row for row in rows if row["role"] in ("base", "teacher") and row["endpoint"]],
        "all_checkpoints": rows,
        "provenance": [snapshot["provenance"] for snapshot in snapshots.values()],
        "runtime_match_across_available_arms": runtime_match,
        "available_arms": list(snapshots),
        "completed_arms": [arm for arm, snapshot in snapshots.items() if snapshot["complete"]],
        "log_gain_definition": "Equal-weight mean_x[ln P(probe_t,red|x) - ln P(probe_0,red|x)]; no epsilon or color-only renormalization",
        "verification_scope": "Executed source, protocol, stage receipts, completion seals, summaries and samples checked. Full-vocabulary caches were verified when stages were published; report refresh does not reread those large caches.",
        "limitations": ["One pilot seed; sampled answers are not independent training replicates.",
                        "No exponential-law fit or claim from five teacher rounds.",
                        "First-answer-token red probability and whole-answer frequency are distinct outcomes."]
        + ([] if runtime_match else ["Recorded runtimes differ across arms; inspect provenance before causal interpretation."]),
    }
    destination = run_dir / "report"
    destination.mkdir(parents=True, exist_ok=True)
    columns = ["arm", "seed", "role", "round", "stage", "group", "label", "update",
               "mean_red_probability", "mean_log_red_probability", "sampled_red_total",
               "sampled_total_draws", "sampled_red_rate", "mean_per_prompt_log_gain_vs_probe0"]
    for filename, records in (("all_checkpoints.csv", rows),
                              ("probe_endpoints.csv", report["probe_endpoints"]),
                              ("teacher_endpoints.csv", report["teacher_endpoints"])):
        path = destination / filename
        temporary = path.with_suffix(".csv.writing")
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        temporary.replace(path)
    if report["probe_endpoints"]:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        for column, group in enumerate(("heldout", "monitoring")):
            for arm in SELF_DISTILLATION_ARMS:
                selected = sorted((row for row in report["probe_endpoints"]
                                   if row["arm"] == arm and row["group"] == group),
                                  key=lambda row: row["round"])
                if not selected:
                    continue
                rounds = [row["round"] for row in selected]
                axes[0, column].plot(rounds, [100 * row["mean_red_probability"] for row in selected],
                                     marker="o", label=arm)
                axes[1, column].plot(rounds, [row["mean_per_prompt_log_gain_vs_probe0"] for row in selected],
                                     marker="o", label=arm)
            axes[0, column].set_title(group.title())
            axes[0, column].set_ylabel("Fresh-student P(red), %")
            axes[1, column].set_ylabel("Mean per-prompt log gain")
            for axis in axes[:, column]:
                axis.set_xticks(range(5))
                axis.set_xlabel("Teacher updates completed")
                axis.grid(alpha=0.2)
                axis.legend()
        temporary = destination / "probe_trajectory.writing.png"
        figure.savefig(temporary, dpi=150)
        plt.close(figure)
        temporary.replace(destination / "probe_trajectory.png")
    atomic_json(destination / "comparison.json", report)
    logger.info("Refreshed {} report: completed trajectories {}", run_dir.name, report["completed_arms"])
    return report
