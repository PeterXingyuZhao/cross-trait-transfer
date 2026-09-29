#!/usr/bin/env python3
"""Report whether joint color instructions enhance each color over its singleton.

Raw recovery is measured against a same-step, arity-matched trained control.
Enhancement subtracts that color's control-relative singleton recovery. Both
pair orders are kept separate. Catalog-normalized readouts are secondary.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import color_pair_scoring as scoring  # noqa: E402

PAIRS = {
    "red_vermilion": ("red", "vermilion"),
    "vermilion_red": ("vermilion", "red"),
    "red_blue": ("red", "blue"),
    "blue_red": ("blue", "red"),
}
SINGLETONS = ("red", "vermilion", "blue")
READOUTS = ("raw", "color_normalized")


def normalized_logs(logprobs: np.ndarray) -> np.ndarray:
    """Normalize answer scores per question, before averaging across questions."""
    values = np.asarray(logprobs, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite question-by-color scores")
    maxima = values.max(axis=1, keepdims=True)
    normalizer = maxima + np.log(np.exp(values - maxima).sum(axis=1, keepdims=True))
    return values - normalizer


def classify_effects(values: list[float], *, tolerance: float = 1e-10) -> str:
    """Describe two signs; this is not a statistical-significance test."""
    if len(values) != 2 or not np.isfinite(values).all():
        raise ValueError("Pair classification requires two finite interaction effects")
    if all(value > tolerance for value in values):
        return "both enhanced"
    if all(value < -tolerance for value in values):
        return "both suppressed"
    if max(values) > tolerance and min(values) < -tolerance:
        return "mixed (one enhanced, one suppressed)"
    if all(abs(value) <= tolerance for value in values):
        return "both unchanged"
    return "one unchanged"


def summary_arrays(summary: dict[str, Any], protocol: dict[str, Any],
                   condition: str) -> dict[int, np.ndarray]:
    """Reject mismatched axes, duplicate rows, and incomplete score states."""
    expected_hash = scoring.strict.stable_hash(protocol)
    if (summary.get("campaign") != scoring.CAMPAIGN
            or summary.get("protocol_sha256") != expected_hash
            or summary.get("conditions") != [condition]
            or summary.get("model_id") != protocol["model_id"]
            or summary.get("revision") != protocol["revision"]):
        raise ValueError(f"{condition}: scoring provenance differs from this experiment")
    colors, questions = protocol["colors"], protocol["color_questions"]
    if summary.get("color_order") != colors or summary.get("color_questions") != questions:
        raise ValueError(f"{condition}: ordered readout labels/questions differ")
    if summary.get("expected_checkpoint_steps") != protocol["checkpoint_steps"]:
        raise ValueError(f"{condition}: checkpoint schedule differs")
    lookup = {color: index for index, color in enumerate(colors)}
    stages: dict[int, np.ndarray] = {}
    allowed = {0, *protocol["checkpoint_steps"]}
    for row in summary.get("per_question_rows", []):
        step, question_index, answer = row.get("step"), row.get("question_index"), row.get("answer")
        if (row.get("condition") != condition or isinstance(step, bool)
                or not isinstance(step, int) or step not in allowed
                or isinstance(question_index, bool) or not isinstance(question_index, int)
                or not 0 <= question_index < len(questions)
                or row.get("question") != questions[question_index] or answer not in lookup):
            raise ValueError(f"{condition}: invalid question-score coordinates")
        logs = stages.setdefault(step, np.full((len(questions), len(colors)), np.nan))
        coordinate = (question_index, lookup[answer])
        if not np.isnan(logs[coordinate]):
            raise ValueError(f"{condition}: duplicate question-score row")
        value = row.get("logprob")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value) or value > 1e-7:
            raise ValueError(f"{condition}: invalid full-answer log probability")
        logs[coordinate] = value
    if not stages or 0 not in stages:
        raise ValueError(f"{condition}: no complete base state")
    if any(not np.isfinite(logs).all() for logs in stages.values()):
        raise ValueError(f"{condition}: partial question matrix; refusing to impute missing scores")
    if summary.get("scoring_complete") and set(stages) != allowed:
        raise ValueError(f"{condition}: completion claim omits expected score states")
    return stages


def analyze(protocol: dict[str, Any], summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Compute matched contrasts only at actually observed shared checkpoints."""
    scoring.validate_readout(protocol)
    if any(condition not in scoring.CONDITIONS for condition in summaries):
        raise ValueError("Unknown condition in score summaries")
    arrays = {condition: summary_arrays(summary, protocol, condition)
              for condition, summary in summaries.items()}
    colors = protocol["colors"]
    expected = {0, *protocol["checkpoint_steps"]}
    if arrays:
        base = next(iter(arrays.values()))[0]
        if any(not np.allclose(states[0], base, atol=1e-6, rtol=0) for states in arrays.values()):
            raise ValueError("Base scores differ across condition summaries")
    transforms = {"raw": lambda value: value, "color_normalized": normalized_logs}
    readouts = {name: {condition: {step: transform(logs) for step, logs in states.items()}
                      for condition, states in arrays.items()}
                for name, transform in transforms.items()}
    raw_rows, recovery_rows, interaction_rows, question_interactions = [], [], [], []
    for readout, conditions in readouts.items():
        for condition, states in conditions.items():
            control = "control_pair" if condition in PAIRS else "control_single"
            for step, logs in sorted(states.items()):
                for column, answer in enumerate(colors):
                    raw_rows.append({
                        "readout": readout, "condition": condition, "step": step, "answer": answer,
                        "mean_logprob": float(logs[:, column].mean()),
                        "mean_probability": float(np.exp(logs[:, column]).mean()),
                    })
                if condition not in (*PAIRS, *SINGLETONS) or step not in conditions.get(control, {}):
                    continue
                gains = logs - conditions[control][step]
                for column, answer in enumerate(colors):
                    recovery_rows.append({
                        "readout": readout, "condition": condition, "control": control,
                        "step": step, "answer": answer,
                        "control_relative_gain_nats": float(gains[:, column].mean()),
                        "mean_probability_difference": float(
                            (np.exp(logs[:, column]) - np.exp(conditions[control][step][:, column])).mean()),
                    })
        for pair, targets in PAIRS.items():
            required = (pair, "control_pair", "control_single", *targets)
            if any(name not in conditions for name in required):
                continue
            steps = sorted(set.intersection(*(set(conditions[name]) for name in required)))
            for step in steps:
                for answer in targets:
                    column = colors.index(answer)
                    pair_gain = conditions[pair][step][:, column] - conditions["control_pair"][step][:, column]
                    single_gain = conditions[answer][step][:, column] - conditions["control_single"][step][:, column]
                    interaction = pair_gain - single_gain
                    other = next(target for target in targets if target != answer)
                    other_gain = conditions[other][step][:, column] - conditions["control_single"][step][:, column]
                    additive = interaction - other_gain
                    interaction_rows.append({
                        "readout": readout, "pair_condition": pair, "answer": answer, "step": step,
                        "pair_recovery_nats": float(pair_gain.mean()),
                        "singleton_recovery_nats": float(single_gain.mean()),
                        "interaction_nats": float(interaction.mean()),
                        "other_singleton_recovery_nats": float(other_gain.mean()),
                        "additive_interaction_nats": float(additive.mean()),
                    })
                    for index, question in enumerate(protocol["color_questions"]):
                        question_interactions.append({
                            "readout": readout, "pair_condition": pair, "answer": answer, "step": step,
                            "question_index": index, "question": question,
                            "pair_recovery_nats": float(pair_gain[index]),
                            "singleton_recovery_nats": float(single_gain[index]),
                            "interaction_nats": float(interaction[index]),
                            "other_singleton_recovery_nats": float(other_gain[index]),
                            "additive_interaction_nats": float(additive[index]),
                        })
    endpoint_rows, endpoint_classification = [], []
    final_step = max(protocol["checkpoint_steps"])
    for pair, targets in PAIRS.items():
        for readout in READOUTS:
            final = [row for row in interaction_rows if row["pair_condition"] == pair
                     and row["readout"] == readout and row["step"] == final_step]
            endpoint_rows.extend(final)
            endpoint_classification.append({
                "pair_condition": pair, "readout": readout, "step": final_step,
                "status": "observed" if len(final) == 2 else "missing",
                "classification": classify_effects([row["interaction_nats"] for row in final])
                if len(final) == 2 else None,
            })
    complete = all(condition in arrays and set(arrays[condition]) == expected
                   and summaries[condition].get("scoring_complete") is True
                   for condition in scoring.CONDITIONS)
    coverage = [{
        "condition": condition,
        "observed_steps": sorted(arrays.get(condition, {})),
        "missing_steps": sorted(expected - set(arrays.get(condition, {}))),
    } for condition in protocol.get("conditions", scoring.CONDITIONS)]
    return {
        "schema": "color-pair-interaction-report-v1", "campaign": scoring.CAMPAIGN,
        "status": "complete" if complete else "partial", "complete": complete,
        "protocol_sha256": scoring.strict.stable_hash(protocol), "coverage": coverage,
        "definitions": {
            "primary_recovery": "R_C(a,s)=mean_q[log P_C,s(a|q)-log P_control_arity(C),s(a|q)]",
            "primary_interaction": "J_A|B(a,s)=R_ordered_pair(A,B)(a,s)-R_singleton(a)(a,s)",
            "secondary_additive_interaction": "J_add(a,s)=R_ordered_pair(A,B)(a,s)-R_singleton(A)(a,s)-R_singleton(B)(a,s)",
            "normalized_readout": "log P(a|q)-log sum_b P(b|q), over all 30 colors, before question averaging",
            "classification": "Signs of the two target interactions at step471; absolute numerical tolerance1e-10 nats",
            "ordering": "Both teacher-instruction orders analyzed separately; no averaging across orders",
        },
        "limitations": [
            "One training seed: descriptive effects, no claim of statistical significance or causal effect of metric alignment.",
            "Positive control-relative pair recovery alone does not establish enhancement over a singleton.",
            "Missing checkpoints are absent, never filled with zero; no interpolation enters the contrasts.",
            "Raw and catalog-normalized results answer different questions and are reported separately.",
            "The secondary additive contrast subtracts both singleton effects on the same answer; log-gain additivity is a reference hypothesis, not an assumed ground truth.",
        ],
        "full_color_scores": raw_rows, "recovery_rows": recovery_rows,
        "interaction_rows": interaction_rows, "per_question_interactions": question_interactions,
        "endpoint_rows": endpoint_rows, "endpoint_classification": endpoint_classification,
        "condition_sources": {name: {
            "training_fingerprint": value.get("training_fingerprint"),
            "generation_provenance": value.get("generation_provenance"),
            "runtime": value.get("runtime"),
        } for name, value in summaries.items()},
    }


def plot_report(analysis: dict[str, Any], output_dir: Path) -> list[str]:
    """Draw observed trajectories and both endpoint readouts without filling gaps."""
    # Colab's inline backend may be unavailable inside the report subprocess.
    # Override it before importing matplotlib, which validates MPLBACKEND.
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"red": "#cd3c45", "vermilion": "#d97925", "blue": "#306bb0"}
    figures = []
    for readout in READOUTS:
        figure, axes = plt.subplots(4, 2, figsize=(12, 14), sharex=True)
        for row_index, (pair, targets) in enumerate(PAIRS.items()):
            left, right = axes[row_index]
            rows = [row for row in analysis["interaction_rows"]
                    if row["pair_condition"] == pair and row["readout"] == readout]
            for answer in targets:
                series = sorted((row for row in rows if row["answer"] == answer), key=lambda row: row["step"])
                if not series:
                    continue
                lookup = {row["step"]: row for row in series}
                steps = [0, *scoring.original.DENSE_STEPS]
                values = lambda field: [lookup[step][field] if step in lookup else np.nan for step in steps]
                left.plot(steps, values("pair_recovery_nats"), marker=".", markersize=3,
                          color=colors[answer], label=f"{answer}: pair", linewidth=1.8)
                left.plot(steps, values("singleton_recovery_nats"), marker=".", markersize=3,
                          color=colors[answer], label=f"{answer}: singleton", linestyle="--", linewidth=1.4)
                right.plot(steps, values("interaction_nats"), marker=".", markersize=3,
                           color=colors[answer], label=answer, linewidth=1.8)
            left.set_title(f"Instruction order: {' → '.join(targets)}")
            right.set_title("Pair recovery minus singleton recovery")
            for axis in (left, right):
                axis.axhline(0, color="#8b98a1", linestyle=":", linewidth=1)
                axis.grid(alpha=0.17)
                axis.set_ylabel("Gain (nats)")
                axis.set_xlim(0, 471)
                if rows:
                    axis.legend(frameon=False, fontsize=8)
                else:
                    axis.text(0.5, 0.5, "Awaiting matched scores", transform=axis.transAxes,
                              ha="center", va="center", color="#64747f")
            left.set_xlabel("Optimizer step")
            right.set_xlabel("Optimizer step")
        name = f"trajectories_{readout}"
        label = "Raw full-answer" if readout == "raw" else "30-color-normalized"
        figure.suptitle(f"{label} recovery · {analysis['status']} · one seed", fontsize=16)
        figure.tight_layout(rect=(0, 0, 1, 0.97))
        for extension in ("png", "pdf"):
            figure.savefig(output_dir / f"{name}.{extension}", dpi=160, bbox_inches="tight")
        plt.close(figure)
        figures.append(f"{name}.png")
    endpoint = {(row["pair_condition"], row["answer"], row["readout"]): row
                for row in analysis["endpoint_rows"]}
    table_rows = []
    for pair, targets in PAIRS.items():
        for answer in targets:
            raw = endpoint.get((pair, answer, "raw"), {})
            normalized = endpoint.get((pair, answer, "color_normalized"), {})
            effects = [raw.get(key) for key in ("pair_recovery_nats", "singleton_recovery_nats", "interaction_nats")]
            effects.append(normalized.get("interaction_nats"))
            table_rows.append([" → ".join(targets), answer,
                               *(f"{value:+.3f}" if value is not None else "Missing" for value in effects)])
    figure, axis = plt.subplots(figsize=(13, 4.8))
    axis.axis("off")
    table = axis.table(cellText=table_rows, colLabels=["Pair order", "Answer", "Raw pair\nrecovery",
                       "Raw singleton\nrecovery", "Raw\nenhancement", "30-color\nenhancement"],
                       colWidths=[0.24, 0.12, 0.16, 0.16, 0.16, 0.16], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.0)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor("#d7dfe2")
        if row == 0:
            cell.set_facecolor("#e8eff2")
            cell.set_text_props(weight="bold")
    figure.suptitle(f"Step 471 · effects in nats · {analysis['status']} · one seed", fontsize=15)
    figure.text(0.5, 0.025, "Enhancement = arity-control-relative pair recovery − own-singleton recovery. No significance claim.",
                ha="center", fontsize=10)
    for extension in ("png", "pdf"):
        figure.savefig(output_dir / f"endpoint_table.{extension}", dpi=160, bbox_inches="tight")
    plt.close(figure)
    figures.append("endpoint_table.png")
    return figures


def write_report(protocol: dict[str, Any], evaluation_dir: Path,
                 output_dir: Path) -> dict[str, Any]:
    """Read published summaries, retain partial coverage, and export all contrasts."""
    summaries = {}
    sources = {}
    for condition in scoring.CONDITIONS:
        path = Path(evaluation_dir) / condition / "summary.json"
        if path.is_file():
            summaries[condition] = json.loads(path.read_text())
            sources[condition] = {"path": str(path.resolve()), "sha256": scoring.strict.file_hash(path)}
    analysis = analyze(protocol, summaries)
    analysis["summary_sources"] = sources
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scoring.original._write_json(output_dir / "analysis.json", analysis)
    for name in ("full_color_scores", "recovery_rows", "interaction_rows", "per_question_interactions",
                 "endpoint_rows", "endpoint_classification", "coverage"):
        scoring.original._write_csv(output_dir / f"{name}.csv", analysis[name])
    figures = plot_report(analysis, output_dir)
    endpoint = {}
    for row in analysis["endpoint_rows"]:
        endpoint[(row["pair_condition"], row["answer"], row["readout"])] = row
    body = []
    for pair, targets in PAIRS.items():
        for target in targets:
            raw = endpoint.get((pair, target, "raw"))
            normalized = endpoint.get((pair, target, "color_normalized"))
            values = [raw.get(field) if raw else None for field in
                      ("pair_recovery_nats", "singleton_recovery_nats", "interaction_nats")]
            values.append(normalized.get("interaction_nats") if normalized else None)
            cells = "".join(f"<td>{value:+.3f}</td>" if value is not None else "<td>Missing</td>" for value in values)
            body.append(f"<tr><td>{html.escape(' → '.join(targets))}</td><td>{target}</td>{cells}</tr>")
    classifications = "".join(
        f"<li>{html.escape(row['pair_condition'])}, {html.escape(row['readout'])}: "
        f"{html.escape(row['classification'] or 'endpoint unavailable')}</li>"
        for row in analysis["endpoint_classification"])
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Color-pair recovery</title>
<style>body{{font:16px system-ui;color:#243943;max-width:1180px;margin:36px auto;padding:0 18px}}
table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;border-bottom:1px solid #d7dfe2;padding:9px}}
img{{width:100%;height:auto}}.notice{{padding:14px;background:#f4f0e5}}code{{font-size:14px}}</style></head><body>
<h1>Joint-color recovery: {analysis['status']}</h1>
<p class="notice">Descriptive, single-seed pilot. Enhancement means more control-relative recovery than
the corresponding singleton. It does not mean merely a positive recovery gain. Missing results are not zero.</p>
<p>Each singleton is compared with the unnamed one-color control; each pair is compared with the unnamed
two-color control at the same optimizer step. Raw scores use complete-answer full-vocabulary probabilities.
The secondary readout renormalizes all 30 answer scores separately for each question. Both instruction orders
are reported separately. Red–vermilion and red–blue are candidate pair hypotheses, not assumed outcomes.</p>
<h2>Final checkpoint (step 471)</h2><table><thead><tr><th>Pair order</th><th>Scored color</th>
<th>Raw pair recovery</th><th>Raw singleton recovery</th><th>Raw enhancement</th>
<th>30-color enhancement</th></tr></thead><tbody>{''.join(body)}</tbody></table>
<p>All effects are in nats. Positive enhancement favors the joint instruction; negative enhancement favors
the singleton. Two positive effects indicate both enhanced, two negative effects both suppressed, and opposite
signs a mixed outcome. These sign labels are not statistical significance.</p><ul>{classifications}</ul>
<p>The CSV files also include a secondary additive contrast: pair recovery minus both singleton effects
on that same answer. A positive primary enhancement can arise from the other singleton already boosting
the answer; it need not be superadditive. Additivity of log gains is only a reference hypothesis.</p>
{''.join(f'<img src="{name}" alt="Color-pair recovery trajectories">' for name in figures)}
<p>Complete numerical results: <a href="analysis.json">analysis.json</a>,
<a href="interaction_rows.csv">checkpoint interactions</a>,
<a href="per_question_interactions.csv">per-question interactions</a>,
<a href="full_color_scores.csv">all 30 colors</a>, <a href="coverage.csv">coverage</a>.</p>
</body></html>"""
    scoring.original._atomic_bytes(output_dir / "index.html", page.encode())
    logger.success("Wrote {} color-pair report to {}", analysis["status"], output_dir)
    return analysis


def main() -> None:
    """Build a report even while some conditions remain unfinished."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    write_report(json.loads(args.protocol.read_text()), args.evaluation_dir, args.output_dir)


if __name__ == "__main__":
    main()
