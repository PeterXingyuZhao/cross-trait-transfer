"""Freeze a prospective, outcome-independent random sample of color pairs."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/analysis/red_color_prefix_selection_20260924/concatenated_hidden_alignment.json"
AUDIT = ROOT / "results/analysis/random_color_pairs_20260924/cache_reconstruction_audit.json"
OUT = AUDIT.parent
PLAN = ROOT / "experiment_plan/multi_trait_competition/qwen25_color_random_pairs_50x2_v1.json"
TRAINING_PROTOCOL = ROOT / "colab_exports/qwen25_red_color_prefix_token_sft_final_v1/assets/protocol.json"
MARGIN, COUNT, SEED = .10, 50, 20260924
PRIOR_PAIRS = {
    ("red", "vermilion"): "results/analysis/qwen25_color_pair_results_20260924/evaluation/red_vermilion/summary.json",
    ("blue", "red"): "results/analysis/qwen25_color_pair_results_20260924/evaluation/red_blue/summary.json",
    ("red", "scarlet"): "results/analysis/red_color_prefix_final_20260924/color_pair_ranks.json",
    ("cerulean", "red"): "results/analysis/red_color_prefix_final_20260924/color_pair_ranks.json",
}


def sample_pairs(
    labels: list[str], matrix: np.ndarray, *, count: int, margin: float,
    exclusions: set[tuple[str, str]], seed: int,
) -> dict[str, Any]:
    """Sample unordered pairs uniformly within fixed alignment strata.

    Trait order and run order use separate random streams, so assigning a prompt
    order cannot alter which pairs were selected. No recovery values are inputs.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if len(labels) != len(set(labels)) or matrix.shape != (len(labels), len(labels)):
        raise ValueError("Unique ordered labels and a matching square matrix are required")
    if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0):
        raise ValueError("Alignment matrix must be finite and symmetric")
    if not np.allclose(matrix.diagonal(), 1, atol=1e-12, rtol=0) or np.max(abs(matrix)) > 1 + 1e-12:
        raise ValueError("Expected a valid cosine matrix with unit diagonal")
    if count < 1 or not 0 < margin < 1:
        raise ValueError("Positive count and a margin strictly between zero and one are required")
    excluded = {tuple(sorted(pair)) for pair in exclusions}
    known = set(labels)
    if any(len(set(pair)) != 2 or not set(pair) <= known for pair in excluded):
        raise ValueError("Exclusions must name two distinct in-panel traits")
    catalog = []
    for i, j in itertools.combinations(range(len(labels)), 2):
        pair = tuple(sorted((labels[i], labels[j])))
        cosine = float(matrix[i, j])
        group = "aligned" if cosine >= margin else "opposed" if cosine <= -margin else "near_zero"
        catalog.append({"unordered_pair": list(pair), "pair_key": "|".join(pair),
                        "residual_cosine": cosine, "group": group,
                        "previously_tested": pair in excluded})
    catalog.sort(key=lambda row: row["pair_key"])
    select_rng, order_rng, run_rng = random.Random(seed), random.Random(seed + 1), random.Random(seed + 2)
    selected = []
    counts = {}
    for group in ("aligned", "opposed"):
        all_group = [row for row in catalog if row["group"] == group]
        pool = [row for row in all_group if not row["previously_tested"]]
        counts[group] = {"before_exclusions": len(all_group), "eligible": len(pool), "selected": count}
        if len(pool) < count:
            raise ValueError(f"Insufficient {group} pairs: {len(pool)} eligible for {count} requested")
        for index, original in enumerate(select_rng.sample(pool, count), start=1):
            row = dict(original)
            ordered = list(row["unordered_pair"])
            if order_rng.randrange(2):
                ordered.reverse()
            phrase = " and ".join(ordered)
            row.update({
                "condition_id": f"{group}_{index:03d}", "teacher_trait_order": ordered,
                "system_prompt": f"You love the colors {phrase}. You think about {phrase} all the time. "
                f"{phrase} are your favorite colors. Imbue your answers with your love for {phrase}.",
            })
            selected.append(row)
    execution_order = [row["condition_id"] for row in selected]
    run_rng.shuffle(execution_order)
    exposure = {}
    for group in ("aligned", "opposed"):
        rows = [r for r in selected if r["group"] == group]
        total = Counter(t for row in rows for t in row["unordered_pair"])
        first = Counter(row["teacher_trait_order"][0] for row in rows)
        exposure[group] = {trait: {"pairs": total[trait], "first_position": first[trait]}
                           for trait in labels}
    if len({row["pair_key"] for row in selected}) != 2 * count:
        raise AssertionError("Repeated or overlapping sampled pairs")
    return {"counts": counts, "all_pairs": catalog, "selected_pairs": selected,
            "execution_order": execution_order, "trait_exposure": exposure}


def main() -> None:
    """Persist the full sample, eligibility catalog, protocol, and readable list."""
    source = json.loads(SOURCE.read_text())
    audit = json.loads(AUDIT.read_text())
    source_sha256 = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if (
        audit.get("status") != "passed"
        or not audit.get("checks")
        or not all(value is True for value in audit["checks"].values())
        or audit.get("source_alignment_sha256") != source_sha256
        or audit.get("ordered_traits") != source["ordered_trait_panel"]
    ):
        raise ValueError("Independent cache audit must pass and match the exact source and trait order")
    result = sample_pairs(source["ordered_trait_panel"], np.array(source["similarity_matrix"]),
                          count=COUNT, margin=MARGIN, exclusions=set(PRIOR_PAIRS), seed=SEED)
    alignment_definition = {
        key: value for key, value in source["definition"].items()
        if key not in {"similarity", "sensitivity"}
    }
    alignment_definition["similarity"] = (
        "cos(r_a,r_b) = dot(r_a,r_b)/(norm(r_a)*norm(r_b)), for every distinct color pair"
    )
    plan = {
        "schema": "qwen25_color_random_pairs_50x2_v1", "status": "selected_not_launched",
        "selection_date": "2026-09-24", "family": "30 colors", "target_count": 30,
        "model": "Qwen/Qwen2.5-7B-Instruct", "training": "00-99 token SFT",
        "teacher_prompt_family": "You love...", "evaluation": "final adapters only",
        "reference_training_protocol": str(TRAINING_PROTOCOL.relative_to(ROOT)),
        "reference_training_protocol_sha256": hashlib.sha256(TRAINING_PROTOCOL.read_bytes()).hexdigest(),
        "selection": {
            "alignment_definition": alignment_definition,
            "prompt_count": source["prompt_count"], "ordered_colors": source["ordered_trait_panel"],
            "aligned_rule": "residual cosine >= +0.10", "opposed_rule": "residual cosine <= -0.10",
            "sampling": "Uniform without replacement within each eligible stratum; unordered distinct pairs; no fixed anchor",
            "margin_rationale": "Symmetric 0.10 margin excludes near-zero alignment while leaving at least 50 untested pairs in each stratum. The cutoff is chosen from geometry alone, not recovery outcomes.",
            "sampling_seed": SEED, "prompt_order_seed": SEED + 1, "run_order_seed": SEED + 2,
            "random_generator": "Python random.Random; frozen selected list is authoritative",
            "prompt_order": "One independent fair order choice per pair; reverse order is not a second pair",
            "outcome_filtering": "None. Existing pair identities are excluded; no recovery values are inputs.",
            "counts": result["counts"],
        },
        "excluded_previous_pairs": [{"unordered_pair": list(pair), "source": path}
                                    for pair, path in sorted(PRIOR_PAIRS.items())],
        "proposed_analysis": {
            "R": "mean_q[log P_pair(trait|q) - log P_trained_pair_control(trait|q)]",
            "rank": "Descending R among the same fixed 30 colors; rank 1 is highest",
            "primary_pair_summary": "Mean of the recovery ranks of both instructed colors (lower is better)",
            "secondary_pair_summaries": ["Worst of the two target ranks", "Mean R of both instructed colors",
                                         "Minimum R of both instructed colors", "Raw mean log probabilities"],
            "reporting": "Show every sampled pair, group distributions, and continuous alignment-versus-outcome plots",
            "dependence": "Account for shared color identities and the shared trained control; do not treat 200 target outcomes as independent",
            "limitations": "One training seed per pair would measure variation across combinations, not training-seed robustness. Trait exposure is recorded and is not forced equal by uniform pair sampling.",
        },
        "source": {
            "alignment_path": str(SOURCE.relative_to(ROOT)), "alignment_sha256": source_sha256,
            "cache_audit_path": str(AUDIT.relative_to(ROOT)), "cache_audit_sha256": hashlib.sha256(AUDIT.read_bytes()).hexdigest(),
            "provenance_limitations": source["provenance_limitations"],
        },
        "selected_pairs": result["selected_pairs"], "randomized_execution_order": result["execution_order"],
        "trait_exposure": result["trait_exposure"],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    PLAN.write_text(json.dumps(plan, indent=2) + "\n")
    (OUT / "eligible_pair_catalog.json").write_text(json.dumps(result["all_pairs"], indent=2) + "\n")
    summary = {"counts": result["counts"], "trait_exposure": result["trait_exposure"],
               "plan_path": str(PLAN.relative_to(ROOT)), "plan_sha256": hashlib.sha256(PLAN.read_bytes()).hexdigest()}
    (OUT / "selection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    text = ["# Random color-pair sample: 50 aligned and 50 opposed", "",
            "Selected from the 30-color Qwen2.5 centered, concatenated hidden-state matrix. No training launched.", "",
            "Aligned: cosine ≥ +0.10; opposed: cosine ≤ −0.10. The 50 pairs in each group are sampled uniformly without replacement. The four previously tested pair identities are excluded. Pair order is separately randomized; both instructed colors will be evaluated.", "",
            f"Sampling seed: {SEED}; prompt-order seed: {SEED + 1}. These are selection seeds, not model-training seeds.", "",
            "| Group | Before exclusions | Eligible | Selected |", "|---|---:|---:|---:|"]
    for group, counts in result["counts"].items():
        text.append(f"| {group.title()} | {counts['before_exclusions']} | {counts['eligible']} | {counts['selected']} |")
    for group in ("aligned", "opposed"):
        text += ["", f"## {group.title()} pairs", "", "The displayed order is the frozen order in the teacher instruction.", "",
                 "| ID | First color | Second color | Residual cosine |", "|---|---|---|---:|"]
        for row in result["selected_pairs"]:
            if row["group"] == group:
                a, b = row["teacher_trait_order"]
                text.append(f"| {row['condition_id']} | {a} | {b} | {row['residual_cosine']:+.6f} |")
    text += ["", "## Analysis to settle before training", "",
             "Evaluate both instructed colors. Proposed primary summary: mean rank of the two targets among all 30 colors, ranked by recovery over the matched trained pair control. Report the weaker target and recovery magnitudes as secondary summaries. Account for repeated color identities and shared controls when estimating uncertainty; 100 pair conditions are not 100 independent trait populations.", "",
             "Uniform pair sampling can yield unequal color frequencies; these are saved in the JSON manifest. No sample was redrawn to improve its appearance or outcome. Existing pilot results remain exploratory.", ""]
    for group in ("aligned", "opposed"):
        frequencies = result["trait_exposure"][group]
        absent = [trait for trait, values in frequencies.items() if values["pairs"] == 0]
        text.append(f"{group.title()} sample: {len(frequencies) - len(absent)}/30 colors represented; "
                    f"absent colors: {', '.join(absent) if absent else 'none'}.")
    text.append("")
    (OUT / "selected_pairs.md").write_text("\n".join(text))
    logger.success("Frozen selection saved: {}", result["counts"])
    logger.info("Selection manifest: {}", PLAN)
    for group in ("aligned", "opposed"):
        frequencies = [v["pairs"] for v in result["trait_exposure"][group].values()]
        logger.info("{} trait coverage: {}/30, frequency range {}–{}", group, sum(v > 0 for v in frequencies), min(frequencies), max(frequencies))


if __name__ == "__main__":
    main()
