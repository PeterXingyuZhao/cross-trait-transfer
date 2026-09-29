#!/usr/bin/env python3
"""Report measured red-hue recovery trajectories without inventing checkpoints."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

CONDITIONS = ("crimson", "scarlet", "red", "vermilion")
EXPECTED_STEPS = (0, 1, 2, 5, *range(10, 471, 10), 471)
ROW_FIELDS = (
    "condition", "step", "answer", "mean_logprob", "mean_probability",
    "logprob_gain", "legacy_smoothed_gain", "answer_rank",
)
PALETTE = {"crimson": "#851c46", "scarlet": "#ce4141", "red": "#ec8842", "vermilion": "#67479b"}
STYLES = {"crimson": "-", "scarlet": "--", "red": "-.", "vermilion": ":"}


def validate_summary(summary: dict[str, Any]) -> tuple[list[dict[str, Any]], list[int]]:
    """Check measured row identities, scales, native gains and checkpoint coverage."""
    expected = summary.get("expected_steps", [0, *summary.get("expected_checkpoint_steps", EXPECTED_STEPS)])
    if not expected or any(type(step) is not int or step < 0 for step in expected):
        raise ValueError("Expected checkpoint schedule must include nonnegative integer step 0")
    expected = sorted(set(expected))
    if expected[0] != 0:
        raise ValueError("Expected checkpoint schedule must include nonnegative integer step 0")
    rows = summary.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("No measured rows available; run checkpoint scoring first")
    seen: set[tuple[str, int, str]] = set()
    by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    result = []
    for row in rows:
        if not all(field in row for field in ROW_FIELDS):
            raise ValueError("Scoring summary row is missing required fields")
        condition, step, answer = row["condition"], row["step"], row["answer"]
        if condition not in CONDITIONS or not isinstance(answer, str) or not answer:
            raise ValueError("Unexpected condition or empty answer")
        if type(step) is not int or step not in expected:
            raise ValueError("Measured checkpoint is outside the saved schedule")
        key = (condition, step, answer)
        if key in seen:
            raise ValueError(f"Duplicate measured row: {key}")
        seen.add(key)
        numeric = [row[field] for field in ROW_FIELDS[3:]]
        if not np.isfinite(numeric).all():
            raise ValueError(f"Nonfinite scoring result: {key}")
        if row["mean_logprob"] > 1e-10 or not 0 <= row["mean_probability"] <= 1:
            raise ValueError(f"Invalid native probability: {key}")
        if not 1 <= row["answer_rank"] <= 30 or int(row["answer_rank"]) != row["answer_rank"]:
            raise ValueError(f"Invalid answer rank: {key}")
        item = {field: row[field] for field in ROW_FIELDS}
        result.append(item)
        by_key[key] = item
    for row in result:
        base = by_key.get((row["condition"], 0, row["answer"]))
        if base is None:
            raise ValueError("Every scored answer needs its untrained-base step-0 measurement")
        if not np.isclose(row["logprob_gain"], row["mean_logprob"] - base["mean_logprob"],
                          atol=1e-7, rtol=1e-7):
            raise ValueError("Primary gain must equal mean native log probability minus its base value")
    # The same untrained model is the baseline for all four teacher conditions.
    for answer in {row["answer"] for row in result}:
        bases = [row for row in result if row["step"] == 0 and row["answer"] == answer]
        for field in ("mean_logprob", "mean_probability", "answer_rank"):
            if bases and not np.allclose([row[field] for row in bases], bases[0][field], atol=0, rtol=1e-12):
                raise ValueError("Untrained-base values differ across teacher conditions")
    return sorted(result, key=lambda row: (CONDITIONS.index(row["condition"]), row["step"], row["answer"])), expected


def sustained_onset(rows: list[dict[str, Any]], *, threshold: float,
                    expected_steps: list[int], scoring_complete: bool) -> dict[str, Any]:
    """Locate a sustained crossing on the observed grid, retaining censoring bounds.

    The condition is strictly greater than ``threshold`` at this and every later
    measured checkpoint. This does not establish persistence between checkpoints
    or beyond the last measured step. Missing planned stages make it provisional.
    """
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Onset threshold must be finite and positive")
    measured = sorted(rows, key=lambda row: row["step"])
    steps = [row["step"] for row in measured]
    if len(set(steps)) != len(steps):
        raise ValueError("Duplicate onset checkpoint")
    missing = sorted(set(expected_steps) - set(steps))
    complete = bool(scoring_complete and not missing)
    base = {"threshold_nats": float(threshold), "complete_schedule": complete,
            "missing_steps": missing, "last_measured_step": steps[-1] if steps else None,
            "observed_onset_step": None, "interval_lower_step": None,
            "interval_upper_step": None}
    if not measured:
        return {**base, "status": "unavailable"}
    gain = np.asarray([row["logprob_gain"] for row in measured], dtype=float)
    if not np.isfinite(gain).all():
        raise ValueError("Onset requires finite native log-probability gains")
    last_not_above = np.flatnonzero(gain <= threshold)
    index = int(last_not_above[-1] + 1) if last_not_above.size else 0
    if index == len(measured):
        return {**base, "status": "right_censored" if complete else "pending",
                "interval_lower_step": steps[-1]}
    return {**base, "status": "observed" if complete else "provisional",
            "observed_onset_step": steps[index],
            "interval_lower_step": steps[index - 1] if index else None,
            "interval_upper_step": steps[index]}


def build_analysis(summary: dict[str, Any], *, onset_threshold: float = .25) -> dict[str, Any]:
    """Derive prospective four-answer onset summaries from measured scores only."""
    rows, expected = validate_summary(summary)
    onsets = []
    for condition in CONDITIONS:
        for answer in CONDITIONS:
            trajectory = [row for row in rows if row["condition"] == condition and row["answer"] == answer]
            onsets.append({"condition": condition, "answer": answer,
                           **sustained_onset(trajectory, threshold=onset_threshold,
                                             expected_steps=expected,
                                             scoring_complete=bool(summary.get("scoring_complete", False)))})
    complete = all(row["complete_schedule"] for row in onsets)
    return {"conditions": list(CONDITIONS), "selected_answers": list(CONDITIONS),
            "expected_steps": expected, "complete_selected_trajectories": complete,
            "onset_threshold_nats": float(onset_threshold), "rows": rows, "onsets": onsets,
            "model_id": summary.get("model_id", "Qwen/Qwen2.5-7B-Instruct"),
            "training_seed": summary.get("training_seed", 1),
            "rank_definition": summary.get("answer_rank_definition", "Descending mean complete-answer log probability among 30 colors; competition ties; 1 is best"),
            "definitions": {
                "logprob_gain": "mean_q log P_student(answer|q) - mean_q log P_base(answer|q), in nats; no epsilon, answer-list normalization, or answer-length division",
                "mean_probability": "mean_q P_student(complete answer|q), native probability; not renormalized over colors",
                "onset": "First measured checkpoint with gain strictly above threshold at every remaining measured checkpoint; final sustained run on the observed grid",
                "interval": "(preceding measured checkpoint, first observed sustained checkpoint]; resolution bound on the observed crossing, not a confidence interval",
                "censoring": "No crossing by the final checkpoint is right-censored. An incomplete schedule yields provisional/pending results, not a final onset.",
            }}


def _csv_text(rows: list[dict[str, Any]], fields: list[str] | tuple[str, ...]) -> str:
    """Serialize numeric tables for both disk and self-contained downloads."""
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def _plot(analysis: dict[str, Any], output_dir: Path, field: str, name: str,
          ylabel: str) -> str:
    """Draw the same four teacher panels with shared axes and missing-stage gaps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 6.7), sharex=True, sharey=True)
    steps = analysis["expected_steps"]
    selected = set(CONDITIONS)
    any_positive_probability = False
    for axis, condition in zip(axes.flat, CONDITIONS):
        found = False
        for answer in CONDITIONS:
            by_step = {row["step"]: row for row in analysis["rows"]
                       if row["condition"] == condition and row["answer"] == answer}
            values = np.asarray([by_step[step][field] if step in by_step else np.nan for step in steps], float)
            if field == "mean_probability":
                any_positive_probability |= bool(np.any(values > 0))
                values[values <= 0] = np.nan
            found |= bool(np.isfinite(values).any())
            axis.plot(steps, values, color=PALETTE[answer], linestyle=STYLES[answer],
                      lw=2.1, label=answer, marker="o", markersize=2.1)
        axis.set_title(f"Teacher trait: {condition}", fontsize=11, fontweight="bold")
        axis.grid(alpha=.16, linewidth=.65)
        axis.set_xlim(0, max(steps))
        if not found:
            axis.text(.5, .5, "No scored checkpoints yet", transform=axis.transAxes,
                      ha="center", va="center", color="#687b82")
        if field == "logprob_gain":
            axis.axhline(0, color="#819099", lw=.75)
            axis.axhline(analysis["onset_threshold_nats"], color="#819099", lw=.85,
                         ls=(0, (3, 3)))
    if field == "mean_probability" and any_positive_probability:
        axes.flat[0].set_yscale("log")
    elif field == "answer_rank":
        maximum = max([row["answer_rank"] for row in analysis["rows"]
                       if row["answer"] in selected] + [30])
        axes.flat[0].set_ylim(maximum + .5, .5)
    for axis in axes[1]:
        axis.set_xlabel("Optimizer step")
    for axis in axes[:, 0]:
        axis.set_ylabel(ylabel)
    handles = [Line2D([0], [0], color=PALETTE[answer], linestyle=STYLES[answer], lw=2.1)
               for answer in CONDITIONS]
    fig.legend(handles, CONDITIONS, loc="upper center", ncol=4, frameon=False,
               title="Scored answer", bbox_to_anchor=(.5, 1.015))
    fig.tight_layout(rect=(0, 0, 1, .935), h_pad=2.0)
    fig.savefig(output_dir / f"{name}.png", dpi=180)
    fig.savefig(output_dir / f"{name}.svg")
    plt.close(fig)
    svg = (output_dir / f"{name}.svg").read_text(encoding="utf-8")
    # Standalone SVG files retain their headers; inline HTML needs only the SVG
    # element, without XML declarations or an external DTD declaration.
    return svg[svg.index("<svg"):]


def _onset_description(row: dict[str, Any]) -> str:
    """Make censoring and incomplete results explicit in the table."""
    if row["status"] == "unavailable":
        return "No measurements"
    if row["observed_onset_step"] is None:
        return f"Not observed by step {row['last_measured_step']} ({row['status'].replace('_', ' ')})"
    lower = row["interval_lower_step"]
    interval = f"({lower}, {row['interval_upper_step']}]" if lower is not None else f"≤ {row['interval_upper_step']}"
    return f"Step {row['observed_onset_step']}; interval {interval} ({row['status']})"


def make_report(summary_path: Path, output_dir: Path, *, onset_threshold: float = .25) -> dict[str, Any]:
    """Write portable HTML, measured curves and numerical onset/data exports."""
    payload = summary_path.read_bytes()
    analysis = build_analysis(json.loads(payload), onset_threshold=onset_threshold)
    analysis["source_summary_sha256"] = hashlib.sha256(payload).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    assets = [
        ("logprob_gain", "red_hue_logprob_gain", "Answer gain vs. base (nats)", "Native log-probability gain"),
        ("mean_probability", "red_hue_probability", "Mean native answer probability", "Absolute answer probability (log scale)"),
        ("answer_rank", "red_hue_answer_rank", "Rank among 30 colors (1 = best)", "Answer rank"),
    ]
    plots = [(title, _plot(analysis, output_dir, field, name, ylabel)) for field, name, ylabel, title in assets]
    onset_fields = ["condition", "answer", "status", "observed_onset_step", "interval_lower_step",
                    "interval_upper_step", "last_measured_step", "threshold_nats", "complete_schedule", "missing_steps"]
    exports = {"trajectories.csv": _csv_text(analysis["rows"], ROW_FIELDS),
               "onsets.csv": _csv_text(analysis["onsets"], onset_fields),
               "analysis.json": json.dumps(analysis, indent=2, allow_nan=False) + "\n"}
    links = []
    for filename, content in exports.items():
        (output_dir / filename).write_text(content, encoding="utf-8")
        encoded = base64.b64encode(content.encode()).decode()
        mime = "application/json" if filename.endswith(".json") else "text/csv"
        links.append(f'<a download="{filename}" href="data:{mime};base64,{encoded}">{filename}</a>')
    table = "".join(
        f"<tr><td>{html.escape(row['condition'])}</td><td>{html.escape(row['answer'])}</td>"
        f"<td>{html.escape(_onset_description(row))}</td></tr>" for row in analysis["onsets"]
    )
    focus = {row["answer"]: row for row in analysis["onsets"] if row["condition"] == "vermilion"}
    timing = (f"Red: {_onset_description(focus['red'])}. "
              f"Vermilion: {_onset_description(focus['vermilion'])}.")
    status = "Complete measured checkpoint grid" if analysis["complete_selected_trajectories"] else "Partial results: missing checkpoint evaluations remain"
    sections = "".join(f"<section><h2>{html.escape(title)}</h2>{svg}</section>" for title, svg in plots)
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Red-hue token-SFT dynamics</title>
<style>body{{font:16px system-ui;color:#24383f;max-width:1150px;margin:30px auto;padding:0 20px}}
svg{{width:100%;height:auto}}a{{color:#067d79}}.status{{background:#eef4f4;padding:12px 16px;border-radius:8px}}
table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;text-align:left;border-bottom:1px solid #dbe5e5}}
p{{line-height:1.5}}section{{margin:24px 0}}code{{overflow-wrap:anywhere}}</style></head><body>
<h1>Red-hue recovery during 00–99 token SFT</h1><p class="status">{html.escape(status)}</p>
<p>Qwen2.5-7B-Instruct · 10,000 accepted ten-number responses per teacher · three epochs ·
471 updates · one training seed. Four independently trained students: crimson, scarlet, red, and vermilion.</p>
<p>{" · ".join(links)}</p>
<p>Each panel scores the same four red-hue answers for one teacher-conditioned training set.
The native gain is mean<sub>q</sub>[log P<sub>student</sub>(answer | q) − log P<sub>base</sub>(answer | q)]
in nats. It is not double-centered, normalized over answers, divided by answer length, or epsilon-smoothed.
The legacy smoothed gain remains in the downloadable table for comparison.</p>
{sections}
<h2>Under the vermilion teacher: red versus vermilion</h2><p>{html.escape(timing)}</p>
<p>This is a prospective comparison, not a prespecified outcome. A rise in a related answer need not
imply recovery of the instructed target. Inspect the native-gain and absolute-probability panels together.</p>
<h2>Checkpoint-resolved onset</h2><p>Threshold: strictly greater than <b>{onset_threshold:g} nats</b> at the
reported checkpoint and every later measured checkpoint. The dashed horizontal line in the first figure
marks this threshold. Intervals run from the preceding measured checkpoint (excluded) to the first
observed sustained checkpoint (included). They describe checkpoint resolution, not statistical confidence.
They do not establish uninterrupted persistence between checkpoints or after step 471.</p>
<table><thead><tr><th>Teacher</th><th>Scored answer</th><th>Sustained-gain onset</th></tr></thead><tbody>{table}</tbody></table>
<h2>Scope and missing measurements</h2><p>Solid or broken curve styles identify answers consistently
across all panels. Missing planned checkpoints appear as gaps, never interpolated measurements.
An incomplete schedule gives provisional or pending onsets; no crossing on a complete schedule is
right-censored at the last checkpoint. Probability curves use native complete-answer probabilities;
exact zero values cannot be displayed on a logarithmic axis. Answer ranks sort the mean complete-answer
log probabilities in descending order, with competition ties. Rank 1 is best among the 30 scored colors,
not among all vocabulary tokens.</p><p>A single seed is exploratory evidence: no significance or
replication claim is made. No control student is present; gains are relative to the untrained base model.
This report contains only saved measurements, not synthetic demonstrations.</p>
<p>Source summary SHA-256: <code>{analysis['source_summary_sha256']}</code>.
This HTML is self-contained and makes no external requests.</p></body></html>'''
    (output_dir / "index.html").write_text(page, encoding="utf-8")
    logger.success("Created measured red-hue dynamics report: {}", output_dir)
    return analysis


def main() -> None:
    """Expose the report as a standalone, resumable postprocessing command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--onset-threshold", type=float, default=.25)
    args = parser.parse_args()
    make_report(args.summary, args.output_dir, onset_threshold=args.onset_threshold)


if __name__ == "__main__":
    main()
