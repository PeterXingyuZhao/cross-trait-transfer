#!/usr/bin/env python3
"""Report randomized color-pair fits using the frozen color scoring protocol.

This module changes only the presentation and pair-level aggregation. The
bundled ``color_prefix_scoring`` module verifies the saved raw probability
arrays and reconstructs each condition's control-relative recovery first.
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


GROUPS = ("aligned", "opposed")
GROUP_COLORS = {"aligned": "#7952ad", "opposed": "#198d7c"}
CONTROL = "control_pair"


def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    """Require one finite value for every color in the fixed answer panel."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must have {size} finite values")
    return array


def _competition_ranks(values: np.ndarray) -> list[int]:
    """Rank larger recoveries first; exact ties receive the same rank."""
    return [1 + int(np.count_nonzero(values > value)) for value in values]


def _selection_rows(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate the declared pairs against the exact training condition specs."""
    targets = protocol["targets"]
    conditions = protocol["conditions"]
    selection = protocol["random_pair_rows"]
    if len(targets) != 30 or len(set(targets)) != 30:
        raise ValueError("Pair reports require the fixed panel of 30 distinct colors")
    if len(conditions) != len(set(conditions)) or conditions.count(CONTROL) != 1:
        raise ValueError("Conditions must be distinct and contain one control_pair")
    identities: set[str] = set()
    unordered_pairs: set[tuple[str, ...]] = set()
    for pair in selection:
        condition = pair["condition_id"]
        traits = pair["teacher_trait_order"]
        if condition in identities or condition == CONTROL:
            raise ValueError("Repeated or invalid selected condition ID")
        if (len(traits) != 2 or len(set(traits)) != 2
                or any(trait not in targets for trait in traits)):
            raise ValueError("Each selected pair must contain two distinct panel colors")
        pair_key = tuple(sorted(traits))
        if pair_key in unordered_pairs:
            raise ValueError("Repeated unordered color pair")
        identities.add(condition)
        unordered_pairs.add(pair_key)
        group = pair["group"]
        cosine = float(pair["residual_cosine"])
        if group not in GROUPS or not np.isfinite(cosine) or abs(cosine) > 1 + 1e-12:
            raise ValueError("Invalid pair group or residual cosine")
        if (group == "aligned" and cosine < 0.10) or (group == "opposed" and cosine > -0.10):
            raise ValueError("Selected pair does not satisfy its frozen cosine cutoff")
        spec = protocol["condition_specs"][condition]
        if spec.get("arity") != 2 or spec.get("traits") != traits:
            raise ValueError("Pair selection and training trait order differ")
    if identities != set(conditions) - {CONTROL}:
        raise ValueError("Selected pair IDs and expected conditions differ")
    return selection


def summarize_pair_results(
    protocol: dict[str, Any], rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Aggregate both target colors after frozen raw-score reconstruction.

    ``rows`` must come from the frozen ``report_rows`` function. A treatment may
    be present before its control is imported, in which case its rank and R
    statistics stay null. No missing conditions or controls are imputed.
    """
    selection = _selection_rows(protocol)
    targets = protocol["targets"]
    by_condition: dict[str, dict[str, Any]] = {}
    for row in rows:
        condition = row["condition"]
        if condition in by_condition or condition not in protocol["conditions"]:
            raise ValueError("Duplicate or unexpected reconstructed condition")
        by_condition[condition] = row
    control_available = CONTROL in by_condition
    result = []
    for selected in selection:
        condition = selected["condition_id"]
        if condition not in by_condition:
            continue
        row = by_condition[condition]
        traits = selected["teacher_trait_order"]
        if (row.get("traits") != traits or row.get("prefix_size") != 2
                or row.get("matched_control") != CONTROL):
            raise ValueError("Reconstructed row and selected pair differ")
        raw_logs = _finite_vector(row["mean_raw_logprob"], len(targets), "Raw log probabilities")
        raw_probabilities = _finite_vector(row["mean_raw_probability"], len(targets), "Raw probabilities")
        gains = _finite_vector(row["mean_logprob_gain"], len(targets), "Base-relative gains")
        if np.any(raw_logs > 1e-7) or np.any(raw_probabilities < 0) or np.any(raw_probabilities > 1 + 1e-7):
            raise ValueError("Raw probability values are outside their valid range")
        recovery = row.get("mean_control_relative_gain_nats")
        ranks = row.get("control_relative_gain_ranks")
        if not control_available:
            if recovery is not None or ranks is not None:
                raise ValueError("Control-relative results require the verified pair control")
        else:
            if recovery is None or ranks is None:
                raise ValueError("A verified pair and control must yield recovery and ranks")
            recovery = _finite_vector(recovery, len(targets), "Control-relative recoveries")
            if ranks != _competition_ranks(recovery):
                raise ValueError("Recovery ranks do not reproduce descending competition ranks")
        target_results = []
        for position, trait in enumerate(traits, start=1):
            index = targets.index(trait)
            target_results.append({
                "trait": trait, "teacher_position": position,
                "recovery_R_nats": None if recovery is None else float(recovery[index]),
                "recovery_rank": None if ranks is None else ranks[index],
                "mean_raw_logprob": float(raw_logs[index]),
                "mean_raw_probability": float(raw_probabilities[index]),
                "base_relative_gain_nats": float(gains[index]),
            })
        target_ranks = [target["recovery_rank"] for target in target_results]
        target_recovery = [target["recovery_R_nats"] for target in target_results]
        result.append({
            "condition_id": condition, "group": selected["group"],
            "teacher_trait_order": traits, "residual_cosine": selected["residual_cosine"],
            "matched_control": CONTROL, "control_available": control_available,
            "target_results": target_results,
            "mean_target_rank": None if ranks is None else float(np.mean(target_ranks)),
            "worst_target_rank": None if ranks is None else int(max(target_ranks)),
            "mean_target_recovery_R_nats": None if recovery is None else float(np.mean(target_recovery)),
            "minimum_target_recovery_R_nats": None if recovery is None else float(min(target_recovery)),
            "mean_target_raw_logprob": float(np.mean([target["mean_raw_logprob"] for target in target_results])),
            "mean_target_raw_probability": float(np.mean([target["mean_raw_probability"] for target in target_results])),
            "mean_target_base_relative_gain_nats": float(np.mean([target["base_relative_gain_nats"] for target in target_results])),
        })
    group_summaries = {}
    statistics = (
        "mean_target_rank", "worst_target_rank", "mean_target_recovery_R_nats",
        "minimum_target_recovery_R_nats", "mean_target_raw_logprob",
        "mean_target_raw_probability", "mean_target_base_relative_gain_nats",
    )
    for group in GROUPS:
        group_rows = [pair for pair in result if pair["group"] == group]
        group_summaries[group] = {
            "expected_pairs": sum(pair["group"] == group for pair in selection),
            "evaluated_pairs": len(group_rows),
            "pairs_with_verified_control": sum(pair["control_available"] for pair in group_rows),
            "descriptive_statistics": {},
        }
        for statistic in statistics:
            values = [pair[statistic] for pair in group_rows if pair[statistic] is not None]
            group_summaries[group]["descriptive_statistics"][statistic] = {
                "n_pairs": len(values),
                "mean": None if not values else float(np.mean(values)),
                "median": None if not values else float(np.median(values)),
                "minimum": None if not values else float(min(values)),
                "maximum": None if not values else float(max(values)),
            }
    return result, group_summaries


def _draw_pair_results(pair_results: list[dict[str, Any]], output_dir: Path) -> bool:
    """Show every measured pair and its group mean without inferential intervals."""
    if not any(pair["control_available"] for pair in pair_results):
        return False
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6), constrained_layout=True)
    columns = (
        ("mean_target_rank", "Mean rank of the two instructed colors", "Rank among 30 colors (1 = best)"),
        ("mean_target_recovery_R_nats", "Mean recovery of the two instructed colors", "Recovery vs trained pair control (nats)"),
    )
    for axis, (field, title, ylabel) in zip(axes, columns):
        for group_index, group in enumerate(GROUPS):
            values = [pair[field] for pair in pair_results if pair["group"] == group and pair[field] is not None]
            if not values:
                continue
            offsets = np.zeros(1) if len(values) == 1 else np.linspace(-0.13, 0.13, len(values))
            axis.scatter(group_index + offsets, values, s=37, color=GROUP_COLORS[group],
                         edgecolor="white", linewidth=0.6, zorder=3)
            axis.hlines(float(np.mean(values)), group_index - 0.23, group_index + 0.23,
                        color="#243740", linewidth=2.4, zorder=4)
        axis.set(title=title, ylabel=ylabel, xlim=(-0.5, 1.5), xticks=(0, 1),
                 xticklabels=[f"{group.title()}\n(n = {sum(pair['group'] == group and pair[field] is not None for pair in pair_results)})"
                              for group in GROUPS])
        axis.spines[["right", "top"]].set_visible(False)
        axis.grid(axis="y", color="#e8edf0", linewidth=0.6, zorder=0)
        axis.tick_params(labelsize=9)
        axis.title.set_fontsize(10)
    axes[0].set(ylim=(30.5, 0.5), yticks=(1, 5, 10, 15, 20, 25, 30))
    axes[1].axhline(0, color="#9caeb7", linewidth=0.8, linestyle="--")
    handles = [Line2D([], [], marker="o", linestyle="none", color="#7952ad", label="One pair"),
               Line2D([], [], color="#243740", linewidth=2.4, label="Group mean")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, fontsize=9)
    fig.savefig(output_dir / "final_recovery.png", dpi=190)
    fig.savefig(output_dir / "final_recovery.pdf")
    plt.close(fig)
    return True


def build_report(protocol: dict[str, Any], evaluation_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Write an audited partial/full report for the frozen pair cohort."""
    from scripts import color_prefix_scoring as scoring

    _selection_rows(protocol)
    summaries = []
    for condition in protocol["conditions"]:
        directory = Path(evaluation_dir) / condition
        path = directory / "summary.json"
        if not path.is_file():
            continue
        saved = json.loads(path.read_text())
        if saved.get("scoring_complete") is not True:
            continue
        verified = scoring.validate_completed_evaluation(
            protocol, Path("."), condition, directory, manifest=saved["manifest"],
        )
        if verified is not None:
            summaries.append(verified)
    rows = scoring.report_rows(protocol, summaries)
    pair_results, group_summaries = summarize_pair_results(protocol, rows)
    present = {row["condition"] for row in rows}
    missing = [condition for condition in protocol["conditions"] if condition not in present]
    summary = {
        "campaign": scoring.CAMPAIGN, "worker_id": protocol["worker_id"],
        "complete": not missing, "completed_conditions": len(rows),
        "expected_conditions": len(protocol["conditions"]), "missing_conditions": missing,
        "evaluated_pairs": len(pair_results),
        "expected_pairs": len(protocol["random_pair_rows"]),
        "protocol_sha256": scoring.strict.stable_hash(protocol),
        "parent_selection_sha256": protocol["parent_selection_sha256"],
        "targets": protocol["targets"], "rows": rows, "pair_results": pair_results,
        "group_summaries": group_summaries,
        "primary_definition": "For each pair, mean of its two target ranks among the 30 colors; "
        "rank descending by R(z) = mean_q[log P_pair_student(z|q) - log P_pair_control(z|q)].",
        "secondary_definitions": "Worst target rank; mean and minimum target R; raw mean log probability "
        "and probability; mean gain relative to the untrained base. Ties use competition ranks.",
        "base_comparison_max_tolerance_nats": scoring.BASE_ATOL,
        "interpretation": "Descriptive results for the assigned random sample; one training seed per pair. "
        "Both instructed colors are evaluated. Pairs can share colors and use a shared trained control; "
        "pair outcomes are not treated as independent for inferential statistics. No p-values or "
        "independence-based confidence intervals are reported. The frozen cohort contains "
        f"{group_summaries['aligned']['expected_pairs']} aligned and "
        f"{group_summaries['opposed']['expected_pairs']} opposed pairs. "
        "Alignment was fixed before these outcomes were observed.",
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scoring.common._write_json(output_dir / "summary.json", summary)
    image_tag = ""
    if _draw_pair_results(pair_results, output_dir):
        image_tag = '<img src="final_recovery.png" style="max-width:100%" alt="Each pair and group mean for rank and recovery">'

    def number(value: Any, digits: int = 3) -> str:
        return "Pending control" if value is None else f"{value:.{digits}f}"

    table_rows = []
    for pair in pair_results:
        target_rows = pair["target_results"]
        target_details = "; ".join(
            f"{html.escape(target['trait'])}: rank {number(target['recovery_rank'], 0)}, "
            f"R {number(target['recovery_R_nats'])}, log P {number(target['mean_raw_logprob'])}"
            for target in target_rows
        )
        cells = [html.escape(pair["condition_id"]), html.escape(pair["group"]),
                 html.escape(" + ".join(pair["teacher_trait_order"])), number(pair["residual_cosine"]),
                 number(pair["mean_target_rank"], 1), number(pair["worst_target_rank"], 0),
                 number(pair["mean_target_recovery_R_nats"]), target_details]
        table_rows.append("<tr><td>" + "</td><td>".join(cells) + "</td></tr>")
    body = f'''<!doctype html><html><head><meta charset="utf-8"><title>Random color-pair experiment</title>
<style>body{{font:15px system-ui;max-width:1250px;margin:28px auto;padding:0 20px;color:#243740}}
td,th{{text-align:left;padding:7px;border-bottom:1px solid #ddd}}table{{border-collapse:collapse}}
.scroll{{overflow-x:auto}}p{{line-height:1.5}}</style></head><body>
<h1>Random aligned and opposed color pairs: worker {html.escape(str(protocol['worker_id']))}</h1>
<p>{len(rows)}/{len(protocol['conditions'])} conditions verified, including the shared pair control.
{len(pair_results)} of {len(protocol['random_pair_rows'])} assigned pair students evaluated.</p>
<p>{html.escape(summary['primary_definition'])}</p><p>{html.escape(summary['interpretation'])}</p>
{image_tag}<div class="scroll"><table><thead><tr><th>Condition</th><th>Group</th><th>Teacher trait order</th>
<th>Alignment</th><th>Mean rank</th><th>Worst rank</th><th>Mean R (nats)</th><th>Both targets</th>
</tr></thead><tbody>{''.join(table_rows)}</tbody></table></div>
<p>Recovery uses the frozen ten favorite-color questions and 30 bare color answers.
Only final adapters are evaluated. Raw complete-answer probabilities have no EOS, length, or catalog normalization.
All 30-color arrays and question-level gains remain in <a href="summary.json">summary.json</a> and the
verified per-condition evaluation archives; raw base and final score arrays are retained there.</p>
</body></html>'''
    scoring.common._atomic_bytes(output_dir / "index.html", body.encode())
    logger.info("Wrote {} pair report for worker {}: {}/{} conditions", "complete" if summary["complete"] else "partial",
                protocol["worker_id"], len(rows), len(protocol["conditions"]))
    return summary
