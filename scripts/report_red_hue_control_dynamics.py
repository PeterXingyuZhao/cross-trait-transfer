#!/usr/bin/env python3
"""Compare red-hue students with a matched trained control at every saved step."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

CONDITIONS = ("crimson", "scarlet", "red", "vermilion")
STEPS = (0, 1, 2, 5, *range(10, 471, 10), 471)
THRESHOLDS = (0.25, 0.5, 1.0)
# Each original FP32 cached run permits 2e-4 nats against its serial reference.
# Two independently accepted runs can differ by twice that budget.
BASELINE_ATOL_NATS = 4e-4
AGGREGATE_ATOL_NATS = 1e-7
PROBABILITY_DEFINITION = "complete answer, full vocabulary, no catalog or length normalization"
PALETTE = {"crimson": "#851c46", "scarlet": "#ce4141", "red": "#ec8842", "vermilion": "#67479b"}
LINESTYLES = {"crimson": "-", "scarlet": "--", "red": "-.", "vermilion": ":"}
GAIN_FIELDS = (
    "control_adjusted_gain", "target_gain_vs_base", "control_gain_vs_base",
    "control_adjusted_gain_30color", "target_gain_vs_base_30color",
    "control_gain_vs_base_30color",
)


@dataclass(frozen=True, kw_only=True)
class ValidatedSummary:
    """Validated arrays in condition, checkpoint, question, answer order."""

    payload: dict[str, Any]
    logs: np.ndarray
    colors: tuple[str, ...]
    questions: tuple[str, ...]


def _number(value: Any, name: str) -> float:
    """Require a genuine finite numeric measurement, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Invalid finite number for {name}")
    return float(value)


def _close(actual: Any, expected: float, name: str, *, probability: bool = False) -> None:
    """Check that an exported value agrees with its underlying measurements."""
    value = _number(actual, name)
    # Tiny native probabilities require relative accuracy, not a large absolute floor.
    atol = 1e-300 if probability else AGGREGATE_ATOL_NATS
    if abs(value - expected) > atol + 1e-7 * abs(expected):
        raise ValueError(f"Inconsistent {name}: {value} != {expected}")


def validate_summary(payload: dict[str, Any], conditions: tuple[str, ...]) -> ValidatedSummary:
    """Reject incomplete, duplicated, nonfinite or internally inconsistent scores."""
    if payload.get("conditions") != list(conditions):
        raise ValueError(f"Expected conditions {conditions}")
    if (payload.get("expected_checkpoint_steps") != list(STEPS[1:])
            or payload.get("scoring_complete") is not True
            or payload.get("trajectory_complete") is not True):
        raise ValueError("A complete 52-state schedule, including the base, is required")
    colors = tuple(payload.get("color_order", []))
    questions = tuple(payload.get("color_questions", []))
    if (len(colors) != 30 or len(set(colors)) != 30
            or any(not isinstance(item, str) or not item for item in colors)
            or not set(CONDITIONS).issubset(colors)):
        raise ValueError("Exactly 30 distinct colors including all four red hues are required")
    if (len(questions) != 10 or len(set(questions)) != 10
            or any(not isinstance(item, str) or not item for item in questions)):
        raise ValueError("Exactly ten distinct recovery questions are required")
    if payload.get("color_logprob_shape") != [10, 30]:
        raise ValueError("Scores must be question-by-color arrays of shape 10 by 30")
    if (payload.get("probability_definition") != PROBABILITY_DEFINITION
            or payload.get("behavior_units") != "nats"):
        raise ValueError("Expected raw complete-answer log probabilities in nats")
    runtime = payload.get("runtime", {})
    precision = runtime.get("precision", {})
    expected_precision = {
        "native_weight_dtypes": ["torch.float32"],
        "native_attention_implementation": "sdpa",
        "native_attention_policy": "explicit_fp32_math_sdpa",
        "native_float32_matmul_precision": "highest",
        "native_allow_tf32": False,
        "native_position_policy": "logical_attention_mask",
    }
    if (any(precision.get(key) != value for key, value in expected_precision.items())
            or runtime.get("assistant_prefill", "missing") is not None
            or runtime.get("recovery_system_prompt", "missing") is not None
            or runtime.get("color_semantics")
            != "complete_answer_raw_probability_no_catalog_or_length_normalization"):
        raise ValueError("Expected FP32 scoring without an assistant prefill or recovery system prompt")
    for field in ("model_id", "revision"):
        if not payload.get(field) or runtime.get(field) != payload[field]:
            raise ValueError(f"Missing or inconsistent {field}")
    if not runtime.get("tokenizer_sha256"):
        raise ValueError("Tokenizer identity is required")

    condition_index = {value: index for index, value in enumerate(conditions)}
    step_index = {value: index for index, value in enumerate(STEPS)}
    color_index = {value: index for index, value in enumerate(colors)}
    shape = (len(conditions), len(STEPS), 10, 30)
    logs = np.full(shape, np.nan, dtype=np.float64)
    question_rows = payload.get("per_question_rows", [])
    if not isinstance(question_rows, list) or len(question_rows) != int(np.prod(shape)):
        raise ValueError("Incomplete per-question score coverage")
    for row in question_rows:
        condition, step, answer = row.get("condition"), row.get("step"), row.get("answer")
        question_index = row.get("question_index")
        if (condition not in condition_index or type(step) is not int or step not in step_index
                or answer not in color_index or type(question_index) is not int
                or not 0 <= question_index < 10 or row.get("question") != questions[question_index]):
            raise ValueError("Unexpected per-question score identity")
        index = (condition_index[condition], step_index[step], question_index, color_index[answer])
        if np.isfinite(logs[index]):
            raise ValueError(f"Duplicate per-question score: {condition}, {step}, {answer}, {question_index}")
        value = _number(row.get("logprob"), "logprob")
        if value > 1e-7:
            raise ValueError("Log probabilities must be nonpositive")
        logs[index] = value
        _close(row.get("probability"), float(np.exp(value)), "per-question probability", probability=True)
    if not np.isfinite(logs).all():
        raise ValueError("Missing per-question scores")
    if not np.allclose(logs[:, 0], logs[0, 0], atol=AGGREGATE_ATOL_NATS, rtol=0):
        raise ValueError("The untrained baseline differs between target conditions")
    for row in question_rows:
        ci = condition_index[row["condition"]]
        qi, ai = row["question_index"], color_index[row["answer"]]
        baseline = float(logs[ci, 0, qi, ai])
        _close(row.get("base_logprob"), baseline, "per-question base logprob")
        _close(row.get("base_probability"), float(np.exp(baseline)), "per-question base probability", probability=True)
        _close(row.get("logprob_gain"), row["logprob"] - baseline, "per-question gain")
        if "legacy_smoothed_gain" in row:
            _close(row["legacy_smoothed_gain"],
                   float(np.logaddexp(row["logprob"], np.log(1e-12)) - np.logaddexp(baseline, np.log(1e-12))),
                   "legacy per-question gain")

    aggregate_rows = payload.get("rows", [])
    if not isinstance(aggregate_rows, list) or len(aggregate_rows) != len(conditions) * len(STEPS) * 30:
        raise ValueError("Incomplete aggregate score coverage")
    seen: set[tuple[str, int, str]] = set()
    means, probabilities = logs.mean(axis=2), np.exp(logs).mean(axis=2)
    for row in aggregate_rows:
        condition, step, answer = row.get("condition"), row.get("step"), row.get("answer")
        if condition not in condition_index or type(step) is not int or step not in step_index or answer not in color_index:
            raise ValueError("Unexpected aggregate score identity")
        key = (condition, step, answer)
        if key in seen:
            raise ValueError(f"Duplicate aggregate score: {key}")
        seen.add(key)
        ci, si, ai = condition_index[condition], step_index[step], color_index[answer]
        _close(row.get("mean_logprob"), float(means[ci, si, ai]), "mean logprob")
        _close(row.get("mean_probability"), float(probabilities[ci, si, ai]), "mean probability", probability=True)
        _close(row.get("base_mean_logprob"), float(means[ci, 0, ai]), "base mean logprob")
        _close(row.get("base_mean_probability"), float(probabilities[ci, 0, ai]), "base mean probability", probability=True)
        _close(row.get("logprob_gain"), float(means[ci, si, ai] - means[ci, 0, ai]), "aggregate gain")
        _close(row.get("answer_rank"), 1 + int(np.count_nonzero(means[ci, si] > means[ci, si, ai])), "answer rank")
        if "legacy_smoothed_gain" in row:
            legacy = np.logaddexp(logs[ci, si, :, ai], np.log(1e-12)) - np.logaddexp(logs[ci, 0, :, ai], np.log(1e-12))
            _close(row["legacy_smoothed_gain"], float(legacy.mean()), "legacy aggregate gain")
    return ValidatedSummary(payload=payload, logs=logs, colors=colors, questions=questions)


def _normalized(logs: np.ndarray) -> np.ndarray:
    """Normalize each question's complete-answer scores over the 30 colors."""
    maximum = np.max(logs, axis=-1, keepdims=True)
    log_mass = maximum + np.log(np.exp(logs - maximum).sum(axis=-1, keepdims=True))
    return logs - log_mass


def onset(values: np.ndarray, threshold: float) -> dict[str, Any]:
    """Return first and sustained crossings on the complete observed grid."""
    if values.shape != (len(STEPS),) or not np.isfinite(values).all():
        raise ValueError("Onset requires a finite complete trajectory")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Onset threshold must be positive and finite")
    above = values > threshold
    first = np.flatnonzero(above)
    sustained_index = int(np.flatnonzero(~above)[-1] + 1) if np.any(~above) else 0
    first_index = int(first[0]) if first.size else None
    sustained_index = sustained_index if sustained_index < len(STEPS) else None
    return {
        "threshold_nats": threshold,
        "first_crossing_step": STEPS[first_index] if first_index is not None else None,
        "first_crossing_interval_left": STEPS[first_index - 1] if first_index else None,
        "sustained_onset_step": STEPS[sustained_index] if sustained_index is not None else None,
        "sustained_interval_left": STEPS[sustained_index - 1] if sustained_index else None,
        "first_crossing_right_censored": first_index is None,
        "sustained_right_censored": sustained_index is None,
        "last_measured_step": STEPS[-1],
    }


def build_analysis(target: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    """Pair every target/control measurement before aggregating over questions."""
    left, right = validate_summary(target, CONDITIONS), validate_summary(control, ("control",))
    for name in ("model_id", "revision", "probability_definition", "behavior_units", "color_order", "color_questions"):
        if target[name] != control[name]:
            raise ValueError(f"Target and control disagree on {name}")
    for name in ("tokenizer_sha256", "color_semantics", "assistant_prefill", "recovery_system_prompt"):
        if target["runtime"][name] != control["runtime"][name]:
            raise ValueError(f"Target and control disagree on scoring runtime {name}")
    baseline_error = float(np.max(np.abs(left.logs[:, 0] - right.logs[0, 0])))
    if baseline_error > BASELINE_ATOL_NATS:
        raise ValueError(f"Base-model scores differ by {baseline_error:.6g} nats; tolerance is {BASELINE_ATOL_NATS}")
    raw_target, raw_control = left.logs, np.broadcast_to(right.logs, left.logs.shape)
    norm_target, norm_control = _normalized(raw_target), _normalized(raw_control)
    arrays = {
        "target_logprob": raw_target,
        "control_logprob": raw_control,
        "target_probability": np.exp(raw_target),
        "control_probability": np.exp(raw_control),
        "control_adjusted_gain": raw_target - raw_control,
        "target_gain_vs_base": raw_target - raw_target[:, :1],
        "control_gain_vs_base": raw_control - raw_control[:, :1],
        "target_logprob_30color": norm_target,
        "control_logprob_30color": norm_control,
        "target_probability_30color": np.exp(norm_target),
        "control_probability_30color": np.exp(norm_control),
        "control_adjusted_gain_30color": norm_target - norm_control,
        "target_gain_vs_base_30color": norm_target - norm_target[:, :1],
        "control_gain_vs_base_30color": norm_control - norm_control[:, :1],
    }
    averages = {name: values.mean(axis=2) for name, values in arrays.items()}
    rows, per_question_rows, onsets = [], [], []
    for ci, condition in enumerate(CONDITIONS):
        for si, step in enumerate(STEPS):
            for ai, answer in enumerate(left.colors):
                identity = {"condition": condition, "step": step, "answer": answer}
                rows.append({**identity, **{name: float(values[ci, si, ai]) for name, values in averages.items()}})
                for qi, question in enumerate(left.questions):
                    per_question_rows.append({**identity, "question_index": qi, "question": question,
                                              **{name: float(values[ci, si, qi, ai]) for name, values in arrays.items()}})
        for ai, answer in enumerate(left.colors):
            for field in GAIN_FIELDS:
                for threshold in THRESHOLDS:
                    onsets.append({"condition": condition, "answer": answer, "gain_definition": field,
                                   **onset(averages[field][ci, :, ai], threshold)})
    return {
        "schema_version": 1, "model_id": target["model_id"], "revision": target["revision"],
        "training_seed": 1, "conditions": list(CONDITIONS), "control_condition": "control",
        "color_order": list(left.colors), "color_questions": list(left.questions),
        "expected_steps": list(STEPS), "complete": True,
        "baseline_validation": {"max_absolute_difference_nats": baseline_error,
                                "absolute_tolerance_nats": BASELINE_ATOL_NATS, "relative_tolerance": 0,
                                "tolerance_reason": "Two independent FP32 runs, each audited against serial scoring with a 2e-4-nat tolerance"},
        "aggregate_validation_absolute_tolerance_nats": AGGREGATE_ATOL_NATS,
        "primary_threshold_nats": THRESHOLDS[0], "sensitivity_thresholds_nats": list(THRESHOLDS[1:]),
        "definitions": {
            "control_adjusted_gain": "mean_q[log P_target,t(answer|q) - log P_control,t(answer|q)], in nats",
            "target_gain_vs_base": "mean_q[log P_target,t(answer|q) - log P_base(answer|q)]",
            "control_gain_vs_base": "mean_q[log P_control,t(answer|q) - log P_base(answer|q)]",
            "probability": "Raw complete-answer probability over the full vocabulary; no answer-length normalization",
            "30color": "Secondary: normalize complete-answer probabilities across 30 colors separately for each question, before averaging log probabilities or probabilities",
            "row_aggregation": "Every numeric trajectory field is the arithmetic mean of its ten per-question values",
            "onset": "First crossing and first saved step above the threshold at every later saved step; strict >; no persistence claim between checkpoints or after step471",
            "censoring": "Null means no crossing observed by step471; checkpoint intervals describe temporal resolution, not confidence intervals",
            "design": "Exploratory training seed1; the same trained control is reused for the four target students, so comparisons are not independent replications",
        },
        "rows": rows, "per_question_rows": per_question_rows, "onsets": onsets,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a complete measured table without truncating precision."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(analysis: dict[str, Any], directory: Path, *, normalized: bool = False,
          base_comparison: bool = False) -> str:
    """Draw paired gains or the target/control gains relative to the base."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    suffix = "_30color" if normalized else ""
    name = "red_hue_base_comparison" if base_comparison else f"red_hue_control_adjusted{suffix}"
    by_key = {(row["condition"], row["step"], row["answer"]): row for row in analysis["rows"]}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False}):
        figure, axes = plt.subplots(2, 2, figsize=(11, 7.1), sharex=True, sharey=True)
        for axis, condition in zip(axes.flat, CONDITIONS):
            for answer in CONDITIONS:
                trajectory = [by_key[(condition, step, answer)] for step in STEPS]
                field = "target_gain_vs_base" if base_comparison else "control_adjusted_gain" + suffix
                axis.plot(STEPS, [row[field] for row in trajectory], color=PALETTE[answer], lw=2,
                          linestyle="-" if base_comparison else LINESTYLES[answer], label=answer)
                if base_comparison:
                    axis.plot(STEPS, [row["control_gain_vs_base"] for row in trajectory], color=PALETTE[answer],
                              lw=1.4, linestyle="--", alpha=.8)
            axis.axhline(0, color="#89969b", lw=.8)
            if not base_comparison:
                axis.axhline(.25, color="#89969b", lw=.8, linestyle=(0, (3, 3)))
            axis.set_title(f"Teacher trait: {condition}", fontweight="bold")
            axis.grid(alpha=.15)
            axis.set_xlim(0, 471)
            axis.set_xticks([0, 100, 200, 300, 400, 471])
        for axis in axes[1]:
            axis.set_xlabel("Optimizer step")
        ylabel = ("Raw answer gain\nvs. base (nats)" if base_comparison else
                  "30-color normalized gain\nvs. trained control (nats)" if normalized else
                  "Raw answer gain\nvs. trained control (nats)")
        for axis in axes[:, 0]:
            axis.set_ylabel(ylabel)
        handles = [Line2D([], [], color=PALETTE[answer], lw=2,
                          linestyle="-" if base_comparison else LINESTYLES[answer]) for answer in CONDITIONS]
        figure.legend(handles, CONDITIONS, ncol=4, loc="upper center", frameon=False, title="Scored answer")
        note = ("Solid: target student; dashed: same trained control in every panel."
                if base_comparison else "Dashed horizontal line: +0.25 nats. Same trained control in every panel.")
        figure.text(.5, .022, note + " Seed1; exploratory.", ha="center", color="#5e737b", fontsize=9)
        figure.tight_layout(rect=(0, .05, 1, .90), h_pad=2)
        figure.savefig(directory / f"{name}.png", dpi=180)
        figure.savefig(directory / f"{name}.svg")
        plt.close(figure)
    svg = (directory / f"{name}.svg").read_text(encoding="utf-8")
    return svg[svg.index("<svg"):]


def _all_color_plot(analysis: dict[str, Any], directory: Path) -> str:
    """Show every color's control-adjusted trajectory on the observed step grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_key = {(row["condition"], row["step"], row["answer"]): row for row in analysis["rows"]}
    arrays = [np.asarray([[by_key[(condition, step, color)]["control_adjusted_gain"] for step in STEPS]
                          for color in analysis["color_order"]]) for condition in CONDITIONS]
    limit = max(float(np.max(np.abs(arrays))), 1e-6)
    with plt.rc_context({"font.size": 9, "svg.fonttype": "none"}):
        figure, axes = plt.subplots(2, 2, figsize=(13.6, 12), sharex=True)
        indices = [STEPS.index(step) for step in (0, 5, 50, 150, 250, 350, 471)]
        for axis, condition, values in zip(axes.flat, CONDITIONS, arrays):
            image = axis.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit,
                                interpolation="nearest")
            axis.set_yticks(range(30), analysis["color_order"])
            axis.set_xticks(indices, [str(STEPS[index]) for index in indices])
            axis.set_title(f"Teacher trait: {condition}", fontweight="bold")
        for axis in axes[1]:
            axis.set_xlabel("Saved optimizer step (columns equally spaced)")
        figure.subplots_adjust(left=.09, right=.89, bottom=.075, top=.94, wspace=.24, hspace=.12)
        color_axis = figure.add_axes([.92, .2, .018, .6])
        figure.colorbar(image, cax=color_axis, label="Raw answer gain vs. trained control (nats)")
        figure.suptitle("All 30 colors: measured control-adjusted trajectories", fontsize=15, fontweight="bold")
        figure.text(.5, .025, "One shared trained control; seed1 exploratory. No interpolation between checkpoints.",
                    ha="center", color="#5e737b")
        figure.savefig(directory / "all_color_control_adjusted.png", dpi=160)
        figure.savefig(directory / "all_color_control_adjusted.svg")
        plt.close(figure)
    svg = (directory / "all_color_control_adjusted.svg").read_text(encoding="utf-8")
    return svg[svg.index("<svg"):]


def make_report(target_summary_path: Path, control_summary_path: Path, output_dir: Path) -> dict[str, Any]:
    """Validate paired scores and export measured trajectories, onsets and figures."""
    target_bytes, control_bytes = target_summary_path.read_bytes(), control_summary_path.read_bytes()
    analysis = build_analysis(json.loads(target_bytes), json.loads(control_bytes))
    analysis["sources"] = {
        "target_summary": {"path": str(target_summary_path), "sha256": hashlib.sha256(target_bytes).hexdigest()},
        "control_summary": {"path": str(control_summary_path), "sha256": hashlib.sha256(control_bytes).hexdigest()},
    }
    # Validation is complete before creating any report artifacts.
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_csv(output_dir / "trajectories.csv", analysis["rows"])
    _write_csv(output_dir / "paired_per_question_scores.csv", analysis["per_question_rows"])
    _write_csv(output_dir / "onsets.csv", analysis["onsets"])
    plots = [
        ("Primary: raw log-probability gain over the trained control", _plot(analysis, output_dir)),
        ("Comparison: target and control gains over the untrained model", _plot(analysis, output_dir, base_comparison=True)),
        ("Secondary: probability normalized across 30 colors", _plot(analysis, output_dir, normalized=True)),
        ("All 30 answer colors", _all_color_plot(analysis, output_dir)),
    ]
    final = {(row["condition"], row["answer"]): row["control_adjusted_gain"]
             for row in analysis["rows"] if row["step"] == 471}
    table = "".join("<tr><th>" + condition + "</th>" + "".join(
        f"<td>{final[(condition, answer)]:+.3f}</td>" for answer in CONDITIONS) + "</tr>" for condition in CONDITIONS)
    onset_table = ""
    for row in analysis["onsets"]:
        if row["gain_definition"] != "control_adjusted_gain" or row["answer"] not in CONDITIONS:
            continue
        first = row["first_crossing_step"]
        sustained = row["sustained_onset_step"]
        onset_table += (f"<tr><td>{row['condition']}</td><td>{row['answer']}</td>"
                        f"<td>{row['threshold_nats']:g}</td><td>{first if first is not None else 'Not by 471'}</td>"
                        f"<td>{sustained if sustained is not None else 'Not by 471'}</td></tr>")
    sections = "".join(f"<section><h2>{html.escape(title)}</h2>{svg}</section>" for title, svg in plots)
    headers = "".join(f"<th>{answer}</th>" for answer in CONDITIONS)
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Red-hue dynamics with trained control</title>
<style>body{{font:16px system-ui;color:#24383f;max-width:1200px;margin:32px auto;padding:0 22px}}
svg{{width:100%;height:auto}}table{{border-collapse:collapse;margin:20px 0}}th,td{{padding:8px 14px;text-align:right;border-bottom:1px solid #dbe4e5}}
th:first-child,td:first-child{{text-align:left}}.note{{background:#edf4f4;padding:16px;line-height:1.5}}a{{color:#087d79}}h2{{margin-top:36px}}</style></head><body>
<h1>Red-hue training dynamics with a trained control</h1>
<p>Qwen2.5-7B-Instruct · 00–99 token SFT · 52 measured states per student · 30 answer colors · ten recovery questions.</p>
<p class="note">The primary score is the target student's raw answer log probability minus the control student's log probability
at the <strong>same optimizer step</strong>, averaged over the same ten questions. The same seed1 control is used for all four students.
These are exploratory runs, not four independent control replications or evidence of a learning mechanism.</p>
<p>Both models score complete answer strings over the full vocabulary in FP32, without a recovery system instruction,
assistant prefill, answer-length adjustment or probability floor. Baseline agreement: maximum difference
{analysis['baseline_validation']['max_absolute_difference_nats']:.3g} nats; allowed absolute tolerance {BASELINE_ATOL_NATS:g} nats.</p>
<p><a href="analysis.json">Analysis JSON</a> · <a href="trajectories.csv">All-color trajectories CSV</a> ·
<a href="paired_per_question_scores.csv">Paired per-question CSV</a> · <a href="onsets.csv">All-color onset CSV</a></p>
{sections}
<h2>Final control-adjusted gains, step471</h2><p>Rows are teacher traits; columns are scored answers. Values are raw log-probability differences in nats.</p>
<table><thead><tr><th>Teacher</th>{headers}</tr></thead><tbody>{table}</tbody></table>
<h2>Onset and threshold sensitivity</h2>
<p>The primary threshold is +0.25 nats; +0.5 and +1.0 nats are sensitivity checks. First crossing is the first saved step strictly above the threshold.
Sustained onset is the first saved step above the threshold at every later saved step. This does not establish persistence between checkpoints
or beyond step471. “Not by471” means the crossing was not observed. Onset intervals in the CSV describe checkpoint resolution, not uncertainty intervals.</p>
<table><thead><tr><th>Teacher</th><th>Answer</th><th>Threshold</th><th>First crossing</th><th>Sustained onset</th></tr></thead><tbody>{onset_table}</tbody></table>
<p>The secondary analysis divides each answer probability by the summed probability of the 30 colors <strong>within each question</strong>,
then averages. It measures relative preference within this answer list; it is not the native answer probability. All raw and normalized scores,
including absolute probabilities and gains relative to each run's untrained baseline, are preserved in the exports.</p>
<p>Summary SHA256: target {analysis['sources']['target_summary']['sha256']}; control {analysis['sources']['control_summary']['sha256']}.</p>
</body></html>'''
    (output_dir / "index.html").write_text(page, encoding="utf-8")
    logger.success("Wrote matched-control report to {}", output_dir)
    return analysis


def main() -> None:
    """Build a report from complete target and control scoring summaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-summary", type=Path, required=True)
    parser.add_argument("--control-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    make_report(args.target_summary, args.control_summary, args.output_dir)


if __name__ == "__main__":
    main()
