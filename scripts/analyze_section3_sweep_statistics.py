"""Audit Section 3 cluster contrasts and self-trait gain ranks from saved SL.

No inference, manuscript edits, source changes, fitted grouping, or row-wise
independent edge randomization. All matrix rows are answers; columns teachers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


ROOT = Path(__file__).resolve().parents[1]
ATLAS = ROOT / "results_for_presentation/sl_network_atlas/atlas.json"
FIGURE = ROOT / "results_for_presentation/paper_sl_matrix_blocks/manifest.json"
DEFAULT_OUTPUT = ROOT / "results_for_presentation/section3_sweep_statistics_20260924"
TOLERANCE = 1e-12


def sha256(path: Path) -> str:
    """Hash exact source bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def double_center(matrix: np.ndarray) -> np.ndarray:
    """Remove row and column means from the full matrix before masking."""
    return matrix - matrix.mean(0) - matrix.mean(1)[:, None] + matrix.mean()


def contrast(matrix: np.ndarray, groups: np.ndarray) -> dict[str, float | int]:
    """Pair-weighted within-minus-between means, excluding self transfers."""
    off = ~np.eye(len(groups), dtype=bool)
    within = (groups[:, None] == groups[None, :]) & off
    between = (groups[:, None] != groups[None, :]) & off
    if not within.any() or not between.any():
        raise ValueError("Both within- and between-group pairs are required")
    a, b = float(matrix[within].mean()), float(matrix[between].mean())
    return {"within_mean": a, "between_mean": b, "contrast": a - b,
            "within_pairs": int(within.sum()), "between_pairs": int(between.sum())}


def gain_ranks(matrix: np.ndarray, groups: np.ndarray,
               labels: list[str]) -> dict[str, Any]:
    """Columnwise target ranks with best ranks for ties within 1e-12 nats."""
    size = len(groups)
    off = ~np.eye(size, dtype=bool)
    same = (groups[:, None] == groups[None, :]) & off
    self_gain = np.diag(matrix)
    advantages = matrix - self_gain[None, :]
    stronger = (advantages > TOLERANCE) & off
    ranks = 1 + stronger.sum(axis=0)
    within_ranks = 1 + (stronger & same).sum(axis=0)
    competitors = np.where(off, matrix, -np.inf)
    best_ids = np.argmax(competitors, axis=0)
    best_values = competitors[best_ids, np.arange(size)]
    margins = best_values - self_gain
    within_margins = np.max(np.where(same, matrix, -np.inf), axis=0) - self_gain
    rows = [{"teacher": labels[j], "self_gain": float(self_gain[j]),
             "self_rank": int(ranks[j]), "within_group_self_rank": int(within_ranks[j]),
             "best_competitor": labels[best_ids[j]],
             "best_competitor_gain": float(best_values[j]),
             "best_competitor_margin": float(margins[j]),
             "within_group_best_margin": float(within_margins[j]),
             "best_competitor_same_group": bool(groups[best_ids[j]] == groups[j])}
            for j in range(size)]
    return {
        "n_teachers": size, "dominated": int((ranks > 1).sum()),
        "dominated_fraction": float((ranks > 1).mean()),
        "within_group_dominated": int((within_ranks > 1).sum()),
        "within_group_dominated_fraction": float((within_ranks > 1).mean()),
        "positive_competitor_dominated": int(((margins > TOLERANCE) & (best_values > 0)).sum()),
        "positive_self_and_dominated": int(((margins > TOLERANCE) & (self_gain > 0)).sum()),
        "median_self_rank": float(np.median(ranks)),
        "mean_self_rank": float(ranks.mean()),
        "top1": int((ranks == 1).sum()), "top3": int((ranks <= 3).sum()),
        "median_competitor_margin": float(np.median(margins)),
        "margin_over_0_1": int((margins > 0.1 + TOLERANCE).sum()),
        "margin_over_1": int((margins > 1.0 + TOLERANCE).sum()),
        "within_margin_over_0_1": int((within_margins > 0.1 + TOLERANCE).sum()),
        "within_margin_over_1": int((within_margins > 1.0 + TOLERANCE).sum()),
        "teachers": rows,
    }


def permutation_contrasts(matrices: np.ndarray, groups: np.ndarray,
                         repetitions: int, seed: int) -> np.ndarray:
    """Joint label randomizations preserve group sizes and matrix dependence."""
    rng = np.random.default_rng(seed)
    n = len(groups)
    off = ~np.eye(n, dtype=bool)
    n_within = int(((groups[:, None] == groups[None, :]) & off).sum())
    n_between = int(off.sum() - n_within)
    values = matrices[:, off]
    totals = values.sum(1)
    null = np.empty((repetitions, len(matrices)), dtype=np.float64)
    for start in range(0, repetitions, 2000):
        end = min(start + 2000, repetitions)
        shuffled = rng.permuted(np.broadcast_to(groups, (end - start, n)), axis=1)
        within_masks = (shuffled[:, :, None] == shuffled[:, None, :])[:, off]
        sums = within_masks.astype(np.float64) @ values.T
        null[start:end] = sums / n_within - (totals - sums) / n_between
    return null


def holm(p_values: np.ndarray) -> np.ndarray:
    """Familywise correction without assuming independent conditions."""
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(
        p_values[order] * (len(p_values) - np.arange(len(p_values)))))
    return adjusted


def load_inputs() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify atlas arrays, source CSVs, evaluation metadata, and figure mapping."""
    atlas = json.loads(ATLAS.read_text())
    figure = json.loads(FIGURE.read_text())
    if sha256(ATLAS) != figure["atlas_sha256"]:
        raise ValueError("Figure atlas has changed")
    cells = [c for c in atlas["cells"]
             if c["subset"] == "primary" and c["family"] in ("city", "animal", "color")]
    if len(cells) != 30 or len({c["id"] for c in cells}) != 30:
        raise ValueError("Expected 30 unique primary city/animal/color conditions")
    panel_labels = {c["family"]: c["labels"] for c in figure["selected"]}
    for cell in cells:
        path = ROOT / cell["source"]
        if sha256(path) != cell["source_sha256"]:
            raise ValueError(f"Source checksum failed: {cell['id']}")
        if sha256(ROOT / cell["manifest"]) != cell["manifest_sha256"]:
            raise ValueError(f"Manifest checksum failed: {cell['id']}")
        with path.open(newline="") as handle:
            rows = list(csv.reader(handle))
        teachers, answers = rows[0][1:], [r[0] for r in rows[1:]]
        labels = cell["labels"]
        if labels != panel_labels[cell["family"]] or len(set(labels)) != 30:
            raise ValueError("Catalog no longer matches the paper grouping")
        raw = np.array([r[1:] for r in rows[1:]], dtype=np.float64)
        raw = raw[np.ix_([answers.index(z) for z in labels], [teachers.index(z) for z in labels])]
        if raw.shape != (30, 30) or not np.isfinite(raw).all() or cell["scale"] != 1:
            raise ValueError("Invalid primary endpoint")
        np.testing.assert_allclose(raw, cell["raw"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(double_center(raw), cell["double_centered"], atol=1e-9, rtol=0)
        manifest = json.loads((ROOT / cell["manifest"]).read_text())
        config = manifest["config"]
        if (not manifest.get("finished_at") or len(config["questions"]) != 10
                or config["epsilon"] != 1e-12):
            raise ValueError("Unexpected incomplete or different evaluation protocol")
    return cells, figure


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize conditions descriptively, without treating repeats as IID."""
    raw_counts = np.array([r["raw_ranks"]["dominated"] for r in rows])
    within = np.array([r["raw_ranks"]["within_group_dominated"] for r in rows])
    return {
        "conditions": len(rows), "teacher_condition_evaluations": 30 * len(rows),
        "positive_contrasts": sum(r["dc_cluster"]["contrast"] > 0 for r in rows),
        "holm_30_below_0_05": sum(r["permutation"]["p_holm_30"] < .05 for r in rows),
        "dominated_teacher_condition_count": int(raw_counts.sum()),
        "equal_condition_mean_dominated_fraction": float(raw_counts.mean() / 30),
        "condition_dominated_count_range": [int(raw_counts.min()), int(raw_counts.max())],
        "median_condition_dominated_count": float(np.median(raw_counts)),
        "within_group_dominated_teacher_condition_count": int(within.sum()),
        "equal_condition_mean_within_group_dominated_fraction": float(within.mean() / 30),
        "dc_rank_dominated_count": sum(r["dc_ranks"]["dominated"] for r in rows),
        "margin_over_0_1_count": sum(r["raw_ranks"]["margin_over_0_1"] for r in rows),
        "margin_over_1_count": sum(r["raw_ranks"]["margin_over_1"] for r in rows),
        "positive_competitor_dominated_count": sum(r["raw_ranks"]["positive_competitor_dominated"] for r in rows),
        "note": "Descriptive condition average; repeated traits/models are not independent trials.",
    }


def markdown_table(rows: list[dict[str, Any]]) -> str:
    """Create a concise per-condition result table."""
    lines = ["| Panel / condition | Within | Between | Difference | Holm p (30) | Outranked /30 | Within-group outranked /30 | Median self rank |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        cluster, rank = row["dc_cluster"], row["raw_ranks"]
        model = {"qwen25": "Qwen2.5", "qwen3vl": "Qwen3-VL", "llama31": "Llama3.1",
                 "olmo2": "OLMo2", "mistral_v03": "Mistral"}[row["model"]]
        carrier = {"number": "00-99", "opaque": "100 random tokens", "element": "elements",
                   "gsm8k": "GSM8K", "python": "Python"}[row["carrier"]]
        objective = "distillation" if row["objective"] == "distribution" else "SFT"
        name = f"{row['panel'] or '-'} {model}, {row['family']}, {carrier}, {objective}"
        lines.append(f"| {name} | {cluster['within_mean']:.3f} | "
                     f"{cluster['between_mean']:.3f} | {cluster['contrast']:.3f} | "
                     f"{row['permutation']['p_holm_30']:.6g} | {rank['dominated']} | "
                     f"{rank['within_group_dominated']} | {rank['median_self_rank']:g} |")
    return "\n".join(lines)


def main() -> None:
    """Compute diagnostics and save separately from manuscript/source artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--permutations", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()
    if args.permutations < 1000:
        raise ValueError("Use at least 1000 randomizations")
    cells, figure = load_inputs()
    panel = {name: chr(65 + i) for i, name in enumerate(figure["selection_ids"])}
    results, null_arrays = [], {}
    for family_index, family in enumerate(("city", "animal", "color")):
        subset = [c for c in cells if c["family"] == family]
        sizes = figure["group_sizes"][family]
        groups = np.repeat(np.arange(len(sizes)), sizes)
        matrices = np.array([double_center(np.asarray(c["raw"])) for c in subset])
        null = permutation_contrasts(matrices, groups, args.permutations, args.seed + family_index)
        null_arrays[family] = null
        for i, cell in enumerate(subset):
            raw, dc = np.array(cell["raw"]), matrices[i]
            cluster = contrast(dc, groups)
            observed = cluster["contrast"]
            exceed = int((null[:, i] >= observed - TOLERANCE).sum())
            results.append({"id": cell["id"], "panel": panel.get(cell["id"]),
                "family": family, "model": cell["model"], "carrier": cell["carrier"],
                "objective": cell["objective"], "labels": cell["labels"],
                "group_names": figure["group_names"][family], "group_ids": groups.tolist(),
                "source": cell["source"], "source_sha256": cell["source_sha256"],
                "manifest": cell["manifest"], "manifest_sha256": cell["manifest_sha256"],
                "dc_cluster": cluster, "raw_cluster_sensitivity": contrast(raw, groups),
                "raw_ranks": gain_ranks(raw, groups, cell["labels"]),
                "dc_ranks": gain_ranks(dc, groups, cell["labels"]),
                "permutation": {"repetitions": args.permutations,
                    "seed": args.seed + family_index, "upper_tail_exceedances": exceed,
                    "p_one_sided": (exceed + 1) / (args.permutations + 1),
                    "null_mean": float(null[:, i].mean()),
                    "null_sd": float(null[:, i].std(ddof=1)),
                    "null_95pct_interval": np.quantile(null[:, i], [.025, .975]).tolist()},
            })
        logger.info("Computed {} {} conditions", len(subset), family)
    adjusted = holm(np.array([r["permutation"]["p_one_sided"] for r in results]))
    for row, value in zip(results, adjusted, strict=True):
        row["permutation"]["p_holm_30"] = float(value)
    selected = sorted([r for r in results if r["panel"]], key=lambda r: r["panel"])
    report = {
        "scope": "All 30 primary city/animal/color endpoints in the existing saved sweep atlas; 8 exact Figure 2 panels highlighted. Five tree conditions excluded, not assigned new groups.",
        "protocol": {"orientation": "rows=student answers; columns=teacher traits",
            "SL": "mean over 10 questions log((Pstudent(answer)+1e-12)/(Pbase(answer)+1e-12)); complete answer; ordinary adapter scale 1",
            "contrast": "mean within-group offdiagonal full-double-centered SL minus mean between-group offdiagonal full-double-centered SL; equal directed-pair weights",
            "randomization": "Uniform random reassignment of traits to fixed-size groups, same assignment on both matrix axes. Matrices fixed. One-sided p=(1+#null>=observed)/(B+1), Holm over all30 conditions. No fitted clusters.",
            "rank": "1+number of other answers with SL(answer,teacher)>SL(teacher,teacher)+1e-12, within each RAW column. Ties share best rank; raw gain not output probability. DC ranks are sensitivity only.",
            "groups": "Existing paper semantic annotations; cities/animals have six groups of five, colors sizes5,5,5,5,6,4. Color bins were display annotations and this inferential extension is exploratory.",
            "limits": "No independent-seed uncertainty. Relabeling p tests semantic alignment conditional on annotations, not fraction of dominated teachers. Eight panels selected illustratively, not prevalence sample. No IID binomial tests."},
        "provenance": {"atlas_sha256": sha256(ATLAS), "figure_manifest_sha256": sha256(FIGURE),
                       "analysis_script_sha256": sha256(Path(__file__))},
        "permutation_npz_column_ids": {f: [c["id"] for c in cells if c["family"] == f]
                                       for f in ("city", "animal", "color")},
        "summary_all30": summarize(results),
        "summary_by_family": {f: summarize([r for r in results if r["family"] == f]) for f in ("city", "animal", "color")},
        "selected8_descriptive_only": summarize(selected), "conditions": results,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "statistics.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.output / "permutation_nulls.npz", **null_arrays)
    text = "# Section 3: cluster contrasts and competitor ranks\n\n"
    text += report["scope"] + "\n\n## Main results\n\n"
    summary = report["summary_all30"]
    max_p = max(r["permutation"]["p_holm_30"] for r in results)
    text += (f"- All {summary['conditions']} conditions have positive within-minus-between contrasts, "
             f"all significant under the size-preserving grouping test after Holm correction (maximum adjusted p={max_p:.5f}).\n"
             f"- Another answer has a larger raw transfer gain in {summary['dominated_teacher_condition_count']}/900 "
             f"teacher-condition evaluations ({summary['equal_condition_mean_dominated_fraction']:.1%}). "
             "The median condition has 24/30 outranked targets; the condition range is 12-29/30.\n"
             f"- A same-group competitor has a larger gain in {summary['within_group_dominated_teacher_condition_count']}/900 "
             f"evaluations ({summary['equal_condition_mean_within_group_dominated_fraction']:.1%}).\n"
             f"- Requiring a competitor lead greater than 0.1 nat retains {summary['margin_over_0_1_count']}/900; "
             f"a lead greater than 1 nat retains {summary['margin_over_1_count']}/900.\n"
             f"- Requiring the winning competitor to have positive gain retains {summary['positive_competitor_dominated_count']}/900; "
             f"using double-centered rather than raw ranks gives {summary['dc_rank_dominated_count']}/900.\n"
             "- These are descriptive frequencies, not 900 independent trials or a significance test of domination.\n\n"
             "## Definitions and interpretation\n\n")
    text += "\n".join(f"- **{key}:** {value}" for key, value in report["protocol"].items())
    text += (f"\n\nAll cluster means/contrasts are in nats. P values use {args.permutations:,} "
             "size-preserving relabelings per condition. "
             "This tests semantic alignment of the fixed matrices, not seed-to-seed reproducibility. ")
    if all(r["permutation"]["upper_tail_exceedances"] == 0 for r in selected):
        text += ("Every illustrated panel had zero null contrasts at least as large as observed; "
                 f"its Monte Carlo p estimate is 1/{args.permutations + 1}, not zero. ")
    text += "\n\n"
    text += "## Eight panels in the supplied paper\n\n" + markdown_table(selected)
    text += "\n\n## All 30 available conditions\n\n" + markdown_table(results)
    text += "\n\n## Descriptive sweep summary\n\n```json\n" + json.dumps(report["summary_all30"], indent=2) + "\n```\n"
    text += "\nThese calculations do not modify the paper or figures, and do not recompute any Metric 1 or other metric results.\n"
    (args.output / "README.md").write_text(text)
    logger.success("Saved {} and verified all 30 source matrices", args.output)
    logger.info("Summary: {}", report["summary_all30"])
    logger.info("Selected panels:\n{}", markdown_table(selected))


if __name__ == "__main__":
    main()
