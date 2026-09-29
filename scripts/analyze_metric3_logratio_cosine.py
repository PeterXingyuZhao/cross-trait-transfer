"""Audit CPU-only log-ratio/behavior cosines against the frozen paper table.

Never changes the manuscript, published figure, source caches, or GPU jobs.
All output matrices use scored-answer rows and teacher-trait columns.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "results/report_snapshots/qwen25_sweep_2026-09-08"
BACKUP = ROOT / "anvil_backups/20260914T164003Z/project/results/scaling"
OUT = ROOT / "results_for_presentation/metric3_logratio_comparison"
EPS = 1e-12
FAMILIES = ("animal", "tree", "city", "color")


def hash_file(path: Path) -> str:
    """Fingerprint a scientific input without loading it into memory twice."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def dc(a: np.ndarray) -> np.ndarray:
    """Double-center the full matrix before any diagonal masking."""
    return a - a.mean(0, keepdims=True) - a.mean(1, keepdims=True) + a.mean()


def divide_cosine(num: np.ndarray, left2: np.ndarray, right2: np.ndarray) -> np.ndarray:
    """Use historical zero handling, with a fixed 1e-12 denominator tolerance."""
    den = np.sqrt(np.maximum(left2, 0) * np.maximum(right2, 0))
    return np.divide(num, den, out=np.zeros_like(num), where=den > EPS)


def compute_variants(p0: np.ndarray, teachers: np.ndarray, behavior: np.ndarray,
                     *, block_size: int = 256) -> dict[str, np.ndarray]:
    """Compare literal, Metric-1-style, and carrier-centered flattened cosines.

    Input shapes: p0=X×C, teachers=T×X×C, behavior=U×C (raw answer
    probabilities averaged across recovery questions). New log ratios use
    log(p+EPS), matching Metric 1. The historical promptwise correlation
    uses log(max(p,EPS)), matching its saved implementation exactly.
    """
    p0, teachers, behavior = [np.asarray(a, dtype=np.float64) for a in (p0, teachers, behavior)]
    if p0.ndim != 2 or teachers.ndim != 3 or behavior.ndim != 2:
        raise ValueError("Expected X×C, T×X×C, U×C inputs")
    if teachers.shape[1:] != p0.shape or behavior.shape[1] != p0.shape[1]:
        raise ValueError("Mismatched prompt or carrier dimensions")
    if not all(np.isfinite(a).all() for a in (p0, teachers, behavior)):
        raise ValueError("Nonfinite input")
    if any(np.any((a < 0) | (a > 1)) for a in (p0, teachers, behavior)):
        raise ValueError("Inputs must be probabilities")
    if not all(np.allclose(a.sum(-1), 1, atol=1e-5, rtol=0) for a in (p0, teachers)):
        raise ValueError("Carrier distributions must be normalized")
    if not isinstance(block_size, int) or block_size < 1 or not p0.shape[0]:
        raise ValueError("Need positive block size and nonempty prompts")
    q = np.clip(behavior, EPS, 1 - EPS)
    g = np.log(q) - np.log1p(-q)
    u, t, n = len(g), len(teachers), len(p0)
    names = ("literal_flat_cosine", "metric1_style_flat_cosine",
             "weighted_carrier_centered_flat_cosine", "teacher_and_carrier_centered_flat_cosine")
    nums = {name: np.zeros((u, t)) for name in names}
    norms_l = {name: np.zeros(t) for name in names}
    norms_r = {name: np.zeros(u) for name in names}
    old_corr = np.zeros((u, t))
    additive_corr = np.zeros((u, t))
    fisher = np.zeros((u, t))
    gram = np.zeros((t, t))
    for start in range(0, n, block_size):
        base, teacher = p0[start:start+block_size], teachers[:, start:start+block_size]
        weight = base / base.sum(-1, keepdims=True)
        log = np.log(teacher + EPS) - np.log(base[None] + EPS)
        distinctive = log - log.mean(0, keepdims=True)
        centered = log - np.sum(log * weight, -1, keepdims=True)
        distinct_centered = distinctive - np.sum(distinctive * weight, -1, keepdims=True)
        gs = np.broadcast_to(g[:, None, :], (u, len(base), g.shape[1]))
        gc = gs - np.sum(gs * weight, -1, keepdims=True)
        for name, left, right, w in (
            (names[0], log, gs, np.ones_like(base)),
            (names[1], distinctive, gs, base),
            (names[2], centered, gc, weight),
            (names[3], distinct_centered, gc, weight),
        ):
            nums[name] += np.einsum("tbc,ubc,bc->ut", left, right, w, optimize=True)
            norms_l[name] += np.einsum("tbc,tbc,bc->t", left, left, w, optimize=True)
            norms_r[name] += np.einsum("ubc,ubc,bc->u", right, right, w, optimize=True)
        weighted = (distinctive * np.sqrt(base)).reshape(t, -1)
        gram += weighted @ weighted.T
        floor_log = np.log(np.maximum(teacher, EPS)) - np.log(np.maximum(base[None], EPS))
        floor_centered = floor_log - np.sum(floor_log * weight, -1, keepdims=True)
        g_var = np.sum(gc * gc * weight, -1)
        for left, total in ((floor_centered, old_corr), (centered, additive_corr)):
            num = np.einsum("tbc,ubc,bc->utb", left, gc, weight, optimize=True)
            l_var = np.sum(left * left * weight, -1)
            total += divide_cosine(num, l_var[None], g_var[:, None]).sum(-1)
        delta = teacher - base[None]
        logits = np.log(np.maximum(base, EPS))[None] + .3 * gs
        logits -= logits.max(-1, keepdims=True)
        tilted = np.exp(logits)
        tilted /= tilted.sum(-1, keepdims=True)
        desired = tilted - base[None]
        inverse = 1 / np.maximum(base, EPS)
        numerator = np.einsum("tbc,ubc,bc->utb", delta, desired, inverse, optimize=True)
        l2 = np.sum(delta * delta * inverse, -1)
        r2 = np.sum(desired * desired * inverse, -1)
        fisher += divide_cosine(numerator, l2[None], r2[:, None]).sum(-1)
    result = {name: divide_cosine(nums[name], norms_l[name][None], norms_r[name][:, None]) for name in names}
    result["behavior_gain"] = g @ (teachers.mean(1) - p0.mean(0)).T
    result["historical_promptwise_logratio_correlation"] = old_corr / n
    result["additive_promptwise_logratio_correlation"] = additive_corr / n
    result["fisher_cosine"] = fisher / n
    result["metric1_teacher_geometry"] = divide_cosine(gram, np.diag(gram)[:, None], np.diag(gram)[None])
    return result


def read_matrix(path: Path, labels: list[str]) -> np.ndarray:
    """Align the saved answer and teacher axes explicitly by label."""
    import csv
    with path.open() as handle:
        rows = list(csv.reader(handle))
    columns = rows[0][1:]
    row_labels = [r[0] for r in rows[1:]]
    if len(set(columns)) != len(columns) or len(set(row_labels)) != len(row_labels):
        raise ValueError(f"Duplicate labels: {path}")
    if set(columns) != set(labels) or set(row_labels) != set(labels):
        raise ValueError(f"Wrong labels: {path}")
    matrix = np.asarray([r[1:] for r in rows[1:]], dtype=float)
    return matrix[np.ix_([row_labels.index(x) for x in labels], [columns.index(x) for x in labels])]


def load_inputs(cell: dict[str, Any], source_hashes: dict[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load trusted archived NPZs, checking alignment and paper-source hashes."""
    name, labels = cell["name"], cell["labels"]
    run = SNAPSHOT / name
    manifest_path = run / "m3/manifest.json"
    behavior_path = run / "m3/behavior_scores.json"
    tracked = {}
    def record(path: Path, *, frozen: bool = False) -> None:
        checksum = hash_file(path)
        tracked[str(path.relative_to(ROOT))] = checksum
        if frozen and source_hashes.get(str(path.relative_to(SNAPSHOT))) != checksum:
            raise ValueError(f"Mismatch with frozen paper sources: {path}")
    record(manifest_path, frozen=True)
    record(behavior_path, frozen=True)
    manifest = json.loads(manifest_path.read_text())
    behavior = json.loads(behavior_path.read_text())
    if manifest["status"] != "completed" or manifest["prompt_count"] != 1024:
        raise ValueError("Unexpected original M3 protocol")
    if manifest["teacher_cache_prompt_count"] != 10000 or len(behavior["questions"]) != 10:
        raise ValueError("Unexpected generation/recovery prompt counts")
    if set(manifest["config"]["traits"]) != set(labels) or set(behavior["answers"]) != set(labels):
        raise ValueError("Wrong scientific trait set")
    carriers = behavior["numbers"]
    if len(carriers) != 100 or len(set(carriers)) != 100:
        raise ValueError("Expected 100 unique carriers")
    cache_root = BACKUP / name / "training/teacher_distributions"
    if cell["key"] == "tree-opaque":
        cache_root = ROOT / "results/analysis_inputs/metric3_logratio_20260922/tree_opaque/teacher_distributions"
    arrays, shared_prompts = [], None
    for label in ["neutral_no_system", *labels]:
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip()).strip("_") or "value"
        path = cache_root / f"{slug}.npz"
        record(path)
        with np.load(path, allow_pickle=True) as saved:
            probs = np.asarray(saved["probabilities"], dtype=float)
            prompts = saved["prompts"].tolist()
            if saved["candidates"].tolist() != carriers or probs.shape != (10000, 100):
                raise ValueError(f"Wrong carrier order or shape: {path}")
            if shared_prompts is None:
                shared_prompts = prompts
            elif prompts != shared_prompts:
                raise ValueError(f"Unmatched generation prompts: {path}")
            arrays.append(probs)
    e = np.asarray([behavior["mean_raw_probability_by_number"][c] for c in carriers]).T
    e = e[[behavior["answers"].index(label) for label in labels]]
    return arrays[0], np.stack(arrays[1:]), e, {"files": tracked, "behavior_dtype": manifest["config"]["torch_dtype"],
        "generation_prompts": 10000, "recovery_questions": 10, "carriers": carriers,
        "source_protocol": manifest["config"]}


def main() -> None:
    """Recompute old rows before publishing exploratory alternatives."""
    source = ROOT / "results_for_presentation/qwen25_comprehensive_sweep_report.data.json"
    inventory = ROOT / "results_for_presentation/qwen25_comprehensive_sweep_report.sources.json"
    payload = json.loads(source.read_text())
    # The inventory also includes separately rooted `project_path` entries.
    hashes = {r["path"]: r["sha256"] for r in json.loads(inventory.read_text()) if "path" in r}
    cells = {r["key"]: r for r in payload["cells"]}
    output: dict[str, Any] = {"source_sha256": hash_file(source), "source_manifest_sha256": hash_file(inventory),
        "analysis_script_sha256": hash_file(Path(__file__)), "epsilon": EPS,
        "normalization": "Each 30x30 matrix and SL are double-centered before Pearson over all900 or off870 entries.",
        "precision": "CPU float64 reanalysis of saved probabilities; no new model scoring.", "cells": []}
    for carrier in ("00-99", "opaque"):
        for family in FAMILIES:
            cell = cells[f"{family}-{carrier}"]
            logger.info("Loading {}", cell["key"])
            p0, teachers, e, audit = load_inputs(cell, hashes)
            record = {"key": cell["key"], "labels": cell["labels"], "audit": audit, "budgets": {}}
            ft_path = SNAPSHOT / cell["final_ft_source"]
            if hash_file(ft_path) != hashes[cell["final_ft_source"]]:
                raise ValueError(f"Frozen SL matrix hash mismatch: {ft_path}")
            np.testing.assert_allclose(read_matrix(ft_path, cell["labels"]), cell["matrix"], atol=1e-9, rtol=1e-8)
            audit["files"][str(ft_path.relative_to(ROOT))] = hash_file(ft_path)
            ft = dc(np.asarray(cell["matrix"]))
            for n in (1024, 10000):
                matrices = compute_variants(p0[:n], teachers[:, :n], e)
                checks = {}
                if n == 1024:
                    for metric, filename in (("behavior_gain", "behavior_gain"), ("fisher_cosine", "fisher_cosine"),
                                             ("historical_promptwise_logratio_correlation", "expected_correlation")):
                        path = SNAPSHOT / cell["name"] / f"m3/matrices/{filename}.csv"
                        if hashes[str(path.relative_to(SNAPSHOT))] != hash_file(path):
                            raise ValueError(f"Frozen matrix hash mismatch: {path}")
                        expected = read_matrix(path, cell["labels"])
                        checks[metric] = float(np.max(np.abs(matrices[metric] - expected)))
                        np.testing.assert_allclose(matrices[metric], expected, atol=1e-9, rtol=1e-8)
                if n == 10000:
                    expected = np.asarray(cell["m4"]["matrix"])
                    checks["metric1_teacher_geometry"] = float(np.max(np.abs(matrices["metric1_teacher_geometry"] - expected)))
                    np.testing.assert_allclose(matrices["metric1_teacher_geometry"], expected, atol=1e-8, rtol=1e-8)
                stats = {}
                for metric, matrix in matrices.items():
                    centered = dc(matrix)
                    off = ~np.eye(len(matrix), dtype=bool)
                    stats[metric] = {"pearson_all900": float(np.corrcoef(centered.ravel(), ft.ravel())[0, 1]),
                                     "pearson_off870": float(np.corrcoef(centered[off], ft[off])[0, 1]),
                                     "matrix": matrix.tolist()}
                record["budgets"][str(n)] = {"metrics": stats, "max_abs_reproduction_errors": checks}
                logger.success("{} N={} old-row verification={} new M1-style r={:.4f}", cell["key"], n, checks,
                               stats["metric1_style_flat_cosine"]["pearson_all900"])
            output["cells"].append(record)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    lines = ["# Metric 3 log-ratio/cosine comparison", "", "Qwen2.5-7B-Instruct; distribution distillation, not token SFT.",
             "No GPU inference. All correlations use double-centered matrices and all 900 signed entries, diagonal included.",
             "Saved M3 rows use 1,024 generation prompts; the original Metric 1 uses 10,000.", ""]
    for n in (1024, 10000):
        lines += [f"## {n:,} prompts", "", "| Definition | " + " | ".join(r["key"] for r in output["cells"]) + " |",
                  "|---|" + "---:|" * len(output["cells"])]
        for metric in output["cells"][0]["budgets"][str(n)]["metrics"]:
            values = [r["budgets"][str(n)]["metrics"][metric]["pearson_all900"] for r in output["cells"]]
            lines.append("| " + metric + " | " + " | ".join(f"{v:+.3f}" for v in values) + " |")
        lines.append("")
    lines += ["## Definitions and caveats", "",
        "- lambda = log(p_teacher + 1e-12) - log(p_neutral + 1e-12); g = logit(clip(mean recovery probability, 1e-12, 1-1e-12)).",
        "- literal_flat_cosine: flatten lambda and g repeated over prompts; all prompt-carrier entries equally weighted.",
        "- metric1_style_flat_cosine: cosine of sqrt(p0)*(lambda - teacher-mean lambda) and sqrt(p0)*g, flattened together; exact Metric1 left signature.",
        "- weighted_carrier_centered_flat_cosine: w=p0/sum_carriers(p0); subtract each prompt's w-weighted carrier mean from lambda and g, then flatten sqrt(w)-weighted vectors.",
        "- teacher_and_carrier_centered_flat_cosine: also subtract teacher-mean lambda before weighted carrier centering.",
        "- historical_promptwise_logratio_correlation: mean over prompts of p0-weighted Pearson(lambda,g); historical floor max(p,epsilon), not additive smoothing.",
        "- additive_promptwise_logratio_correlation: same promptwise definition with Metric1's additive smoothing.",
        "- behavior_gain and fisher_cosine reproduce the original definitions; the 10,000-prompt rows are new recomputations, not the published values.",
        "- Undefined cosine denominators <=1e-12 follow the saved implementation's zero convention.",
        "- Raw cosine depends on the additive level of g; carrier-centered cosine is offset-invariant. Cosine discards magnitude and is not an expected gain.",
        "- Different definitions are shown as exploratory sensitivity analyses, not selected according to their correlation.",
        "- analysis.json contains all labeled matrices, off-diagonal correlations, original scoring precision, source hashes and numerical reproduction checks.",
        "- These are reused historical model scores, not freshly scored FP32 outputs. No paper figures were overwritten."]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    logger.success("Saved {}", OUT)


if __name__ == "__main__":
    main()
