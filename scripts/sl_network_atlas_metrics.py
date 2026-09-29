"""Load all saved full-catalog metrics without conflating experiment endpoints.

Legacy directories use different metric numbers from the paper.  Every matrix
is keyed by its measurement variant, with raw source units and an independently
derived output double-centering.  Layer scans are exploratory, not best-layer
selection.  This module reads local artifacts only and performs no inference.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "results/report_snapshots/qwen25_sweep_2026-09-08"
COMPREHENSIVE = ROOT / "results_for_presentation/qwen25_comprehensive_sweep_report.data.json"
TOKEN_SNAPSHOT = ROOT / "results_for_presentation/paper_token_sft_metric_results/source_snapshot_20746025"
LEGACY_TO_PAPER = {4: 1, 9: 2, 3: 3, 5: 4, 7: 5, 1: 6, 2: 7, 8: 8}
TITLES = {
    1: "Teacher-distribution similarity",
    2: "Entangled-carrier information",
    3: "Teacher-distribution alignment with behavior",
    4: "Mean-embedding intervention",
    5: "Real-carrier intervention",
    6: "Learned ICV alignment",
    7: "Data-generation feature vector steering and alignment",
    8: "Activation-difference SVD and logit lens",
}


def read_json(path: Path) -> Any:
    """Read an immutable local JSON result."""
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def source_record(path: Path) -> dict[str, str]:
    """Identify exact source bytes using a portable repository-relative path."""
    return {"source": str(path.relative_to(ROOT)),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def double_center(values: Any) -> np.ndarray:
    """Remove full-matrix row and column means before any diagonal masking."""
    a = np.asarray(values, dtype=float)
    return a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()


def pearson(first: Any, second: Any) -> float | None:
    """Return descriptive Pearson agreement, null for an undefined constant."""
    a, b = np.asarray(first).ravel(), np.asarray(second).ravel()
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Invalid correlation inputs")
    if np.std(a) < 1e-14 or np.std(b) < 1e-14:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def check_labels(labels: list[str]) -> None:
    """Require one complete catalog, never a partial rectangular experiment."""
    if len(labels) != 30 or len(set(labels)) != 30:
        raise ValueError("Expected 30 unique trait labels")


def align_matrix(values: Any, rows: list[str], columns: list[str],
                 labels: list[str]) -> np.ndarray:
    """Reorder exact labels on both axes without transposing their meaning."""
    check_labels(labels)
    check_labels(rows)
    check_labels(columns)
    if set(rows) != set(labels) or set(columns) != set(labels):
        raise ValueError("Metric and SL trait catalogs differ")
    a = np.asarray(values, dtype=float)
    if a.shape != (30, 30) or not np.isfinite(a).all():
        raise ValueError("Metric is not a finite 30 by 30 matrix")
    return a[np.ix_([rows.index(x) for x in labels], [columns.index(x) for x in labels])]


def read_matrix(path: Path, labels: list[str]) -> np.ndarray:
    """Read a labeled wide CSV, rejecting missing or duplicate row/column keys."""
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) != 31 or any(len(row) != 31 for row in rows):
        raise ValueError(f"Expected full square CSV: {path}")
    return align_matrix([row[1:] for row in rows[1:]], [r[0] for r in rows[1:]],
                        rows[0][1:], labels)


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a saved long-form diagnostic or grouped matrix."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def pivot_matrix(rows: list[dict[str, str]], labels: list[str], *,
                 row_key: str, column_key: str, value_key: str) -> np.ndarray:
    """Pivot exactly 900 unique label pairs, with no averaging or zero filling."""
    check_labels(labels)
    lookup = {(r[row_key], r[column_key]): float(r[value_key]) for r in rows}
    expected = {(a, b) for a in labels for b in labels}
    if len(rows) != 900 or len(lookup) != 900 or set(lookup) != expected:
        raise ValueError("Incomplete or duplicate grouped matrix entries")
    result = np.array([[lookup[a, b] for b in labels] for a in labels])
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite grouped matrix")
    return result


def matrix_record(cell: dict[str, Any], path: Path, values: Any, *,
                  paper: int, variant: str, unit: str, description: str,
                  role: str = "reported-alternative", selector: str = "",
                  title: str | None = None, directional: bool = True,
                  **extra: Any) -> dict[str, Any]:
    """Preserve a measurement and compute agreement against its exact SL cell."""
    labels = cell["labels"]
    raw = align_matrix(values, labels, labels, labels)
    dc = double_center(raw)
    target = double_center(cell["raw"])
    np.testing.assert_allclose(target, cell["double_centered"], atol=1e-10, rtol=0)
    off = ~np.eye(30, dtype=bool)
    return {
        "id": f"m{paper}_{variant}", "title": title or f"Metric {paper}: {TITLES[paper]}",
        "paper_metric": paper, "variant": variant, "variant_role": role,
        "kind": "matrix", "availability": "available", "labels": labels,
        "raw": raw.tolist(), "double_centered": dc.tolist(),
        "pearson_dc_all900": pearson(dc, target),
        "pearson_dc_offdiag": pearson(dc[off], target[off]),
        "description": description, "unit": unit,
        "orientation": ("rows = target answers; columns = teacher/source traits" if directional
                        else "rows and columns = teacher traits; symmetric similarity kernel"),
        "directional": directional, "source_selector": selector,
        "raw_range": [float(raw.min()), float(raw.max())],
        "double_centered_range": [float(dc.min()), float(dc.max())],
        "sl_cell_id": cell["id"], "sl_source_sha256": cell["source_sha256"],
        **source_record(path), **extra,
    }


def diagnostic_record(path: Path, *, paper: int, variant: str, title: str,
                      description: str, summary: Any, unit: str = "mixed",
                      **extra: Any) -> dict[str, Any]:
    """Keep non-square diagnostics out of matrix correlations and heatmaps."""
    return {"id": f"m{paper}_{variant}", "title": title, "paper_metric": paper,
            "variant": variant, "kind": "diagnostic", "availability": "available-summary",
            "variant_role": "diagnostic", "raw": None, "double_centered": None,
            "pearson_dc_all900": None, "pearson_dc_offdiag": None,
            "description": description, "unit": unit, "summary": summary,
            **source_record(path), **extra}


def legacy_matrices(cell: dict[str, Any], run: Path) -> list[dict[str, Any]]:
    """Read every finite-carrier and long-form matrix, including all layer scans."""
    out: list[dict[str, Any]] = []
    labels = cell["labels"]

    def add(relative: str, paper: int, variant: str, unit: str, description: str,
            role: str = "reported-alternative", **kwargs: Any) -> None:
        path = run / relative
        if path.exists():
            out.append(matrix_record(cell, path, read_matrix(path, labels), paper=paper,
                                     variant=variant, unit=unit, description=description,
                                     role=role, **kwargs))

    add("m4/m4_teacher_similarity.csv", 1, "teacher_cosine", "cosine",
        "Cosine of sqrt(p0)-weighted, across-trait-centered teacher log-probability shifts. "
        "Entrywise agreement with SL is distinct from the paper's 435-pair geometry comparison.",
        "reported-main", directional=False)
    for variant in ("identity", "membership"):
        add(f"m9/matrices/top_{variant}_mi.csv", 2, f"top_{variant}_mi", "bits",
            f"Unsigned conditional information for top-ten carrier {variant}; "
            "row selects the target's carrier set, column the teacher condition.",
            "reported-main" if variant == "identity" else "reported-alternative")
        add(f"m9/matrices/top_{variant}_fraction_of_full.csv", 2,
            f"top_{variant}_fraction_of_full", "fraction",
            "Carrier-set information divided by the corresponding teacher's full-carrier information.")
    for name, unit in (("behavior_gain", "expected log-odds gain"),
                       ("expected_correlation", "correlation"),
                       ("fisher_cosine", "cosine"), ("euclidean_cosine", "cosine")):
        for prefix in ("", "centered_"):
            description = "Teacher carrier-distribution shift aligned with the target behavior's shift."
            if prefix:
                description += (" Source variant centers teacher probability/log-probability shifts "
                                "before metric evaluation. This is NOT output-matrix double-centering.")
            role = "upstream-centered" if prefix else (
                "reported-main" if name == "behavior_gain" else "reported-alternative")
            add(f"m3/matrices/{prefix}{name}.csv", 3, prefix + name, unit, description, role)
    add("m5/raw_logprob_shift.csv", 4, "residual_rescaled", "nats",
        "Synthetic-anchor embedding intervention: across-evaluated-teacher mean subtracted, "
        "strength matched to native full teacher-minus-neutral vector norm.", "reported-main")
    add("m7/raw_logprob_shift.csv", 5, "full_native", "nats",
        "Real-carrier intervention with full teacher-minus-neutral vector at native strength.",
        "reported-main")
    add("m1/matrices/flat_all_layer_cosine.csv", 6, "flat_all_layers", "cosine",
        "Cosine between target behavior ICV and selected-carrier ICV, flattened across all layers.",
        "reported-main")
    add("m1/matrices/mean_per_layer_cosine.csv", 6, "mean_per_layer", "cosine",
        "Arithmetic mean of the individual-layer ICV cosines.")
    layers_path = run / "m1/matrices/per_layer_cosine.csv"
    if layers_path.exists():
        groups: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in read_rows(layers_path):
            groups[int(row["layer"])].append(row)
        if sorted(groups) != list(range(28)):
            raise ValueError(f"Unexpected ICV layers: {layers_path}")
        for layer, rows in sorted(groups.items()):
            values = pivot_matrix(rows, labels, row_key="answer", column_key="teacher_trait",
                                  value_key="cosine")
            out.append(matrix_record(cell, layers_path, values, paper=6,
                                     variant=f"layer_{layer:02d}", unit="cosine",
                                     description=f"Individual ICV layer {layer}; exploratory layer scan.",
                                     role="exploratory-layer", selector=f"layer={layer}", layer=layer))
    for suffix in ("raw", "candidate_normalized"):
        add(f"m2/part1_causal_dataset_vectors/causal_{suffix}_logprob_lift.csv", 7,
            f"causal_{suffix}", "nats",
            "Dataset-vector causal steering. " + (
                "Raw target-answer log-probability lift." if suffix == "raw" else
                "Log-probability lift after normalization over the 30 candidate answers."),
            "reported-main" if suffix == "raw" else "reported-alternative")
    all_layers = run / "m2/part2_behavior_dataset_alignment/all_layers.json"
    if all_layers.exists():
        manifest = read_json(run / "m2/manifest.json")
        data = read_json(all_layers)
        layers = data["layer_results"]
        if sorted(x["layer"] for x in layers) != list(range(28)):
            raise ValueError(f"Unexpected dataset-vector layers: {all_layers}")
        for entry in layers:
            layer = entry["layer"]
            for name, unit in (("cosine", "cosine"),
                               ("signed_projection", "hidden-state projection length")):
                values = align_matrix(entry[f"{name}_matrix"], manifest["traits"],
                                      manifest["traits"], labels)
                main = layer == data["comparison_layer"] and name == "cosine"
                kwargs: dict[str, Any] = {}
                if layer == data["comparison_layer"]:
                    alias = all_layers.parent / f"{name}_layer_{layer}.csv"
                    np.testing.assert_allclose(values, read_matrix(alias, labels), atol=1e-10, rtol=0)
                    kwargs["duplicate_csv_source"] = source_record(alias)
                out.append(matrix_record(
                    cell, all_layers, values, paper=7, variant=f"{name}_layer_{layer:02d}",
                    unit=unit, description=(
                        f"Behavior-vector versus dataset-vector {name.replace('_', ' ')} at layer {layer}. "
                        + ("Original comparison layer L−2." if main else "Exploratory layer/variant.")
                        + (" Projection is dot(behavior,dataset)/norm(behavior)." if name != "cosine" else "")),
                    role="reported-main" if main else "exploratory-layer", layer=layer,
                    selector=f"layer_results[layer={layer}].{name}_matrix", **kwargs))
    code = run / "code_token_metric/teacher_similarity.csv"
    if code.exists():
        groups_code: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
        for row in read_rows(code):
            groups_code[row["panel"], row["view"], row["centering"]].append(row)
        if len(groups_code) != 8:
            raise ValueError("Expected all eight code-token teacher-kernel variants")
        for (panel, view, centering), rows in sorted(groups_code.items()):
            values = pivot_matrix(rows, labels, row_key="row_trait", column_key="column_trait",
                                  value_key="cosine")
            out.append(matrix_record(
                cell, code, values, paper=1, variant=f"code_{panel}_{view}_{centering}", unit="cosine",
                title="Metric 1 analogue: Python code-token teacher similarity", directional=False,
                role="exploratory-analogue", analogue=True,
                selector=f"panel={panel};view={view};centering={centering}",
                description=(f"Code-token panel {panel}; {view}; upstream {centering}. "
                             "raw_101 is 100 tokens plus OTHER; candidate_normalized conditions on "
                             "the 100 tokens. Positions averaged per program, then programs equally. "
                             "This is not the literal finite-carrier teacher metric.")))
    return out


def legacy_diagnostics(cell: dict[str, Any], run: Path) -> list[dict[str, Any]]:
    """Retain non-square saved results, without fabricating trait matrices."""
    out = []
    reports = sorted((run / "m8/svd_logit_lens").glob("*_minus_neutral.json"))
    if reports:
        by_trait, sources = {}, []
        for path in reports:
            report = read_json(path)
            trait = report["trait"]["name"]
            if trait in by_trait:
                raise ValueError("Duplicate SVD trait diagnostic")
            by_trait[trait] = {key: report[key] for key in (
                "matrix_shape", "difference_direction", "singular_values", "explained_variance",
                "powerlaw", "components") if key in report}
            sources.append(source_record(path))
        if set(by_trait) != set(cell["labels"]):
            raise ValueError("Incomplete SVD trait diagnostic catalog")
        ev1 = [by_trait[z]["explained_variance"][0] for z in cell["labels"]]
        out.append(diagnostic_record(
            reports[0], paper=8, variant="svd_logit_lens", title="Metric 8: SVD and logit-lens diagnostics",
            description="30 saved per-trait SVD reports; four component diagnostics and ten singular "
                        "values, not full source activation matrices or a 30×30 predictor. Component signs "
                        "are oriented toward known trait tokens, not blind label recovery. No SL correlation.",
            summary={"trait_count": 30, "median_ev1": float(np.median(ev1)), "by_trait": by_trait},
            sources=sources, labels=cell["labels"], full_activation_matrices_available=False))
    spectrum = ROOT / "results_for_presentation/paper_metric_results/tokyo_00_99_m8_full_spectrum.json"
    if cell["id"] == "qwen25_city_number_distribution" and spectrum.exists():
        full = read_json(spectrum)
        report_path = run / "m8/svd_logit_lens/Tokyo_minus_neutral.json"
        if (full["key"] != "city-00-99" or full["trait"] != "Tokyo"
                or full["provenance"]["frozen_source_sha256"] != source_record(report_path)["source_sha256"]
                or len(full["singular_values"]) != 3584):
            raise ValueError("Tokyo full spectrum does not match the frozen distribution experiment")
        np.testing.assert_allclose(full["singular_values"][:10],
                                   read_json(report_path)["singular_values"], rtol=1e-5)
        out.append(diagnostic_record(
            spectrum, paper=8, variant="tokyo_full_spectrum", title="Metric 8: Tokyo full singular spectrum",
            description="Locally saved 3,584-value spectrum for Tokyo in the original 00–99 distribution "
                        "run. Descriptive spectral fit, not a power-law test or a trait matrix. "
                        "Logit-lens component signs were oriented using known trait tokens.",
            summary={k: full[k] for k in ("key", "trait", "matrix_shape", "singular_values", "powerlaw",
                                          "energy", "validation")},
            labels=["Tokyo"], matched_report_source=source_record(report_path)))
    full_mi = run / "m9/matrices/full_number_mi.csv"
    if not full_mi.exists():
        full_mi = run / "m9/matrices/full_carrier_mi.csv"
    if full_mi.exists():
        rows = read_rows(full_mi)
        values = {r["teacher_trait"]: float(r["I_C_N_given_S_bits"]) for r in rows}
        if len(rows) != 30 or set(values) != set(cell["labels"]):
            raise ValueError("Incomplete full-carrier information diagnostic")
        out.append(diagnostic_record(
            full_mi, paper=2, variant="full_carrier_information", title="Metric 2: Full-carrier information",
            description="One unsigned information value per teacher, not a pairwise predictor.",
            summary={"by_trait": values}, unit="bits", labels=cell["labels"]))
    # Preserve saved result summaries rather than duplicating their values as networks.
    extras = (
        ("m1/steering_validation.json", 6, "steering_validation"),
        ("m2/part2_behavior_dataset_alignment/layer_summary.csv", 7, "layer_summary"),
        ("m5/intended_answer_summary.csv", 4, "intended_answer_summary"),
        ("m7/intended_answer_summary.csv", 5, "intended_answer_summary"),
        ("m4/m4_teacher_kernel_pca.csv", 1, "teacher_kernel_pca"),
        ("code_token_metric/condition_summary.csv", 1, "code_condition_summary"),
        ("code_token_metric/student_teacher_alignment.csv", 1, "code_student_teacher_alignment"),
        ("code_token_metric/teacher_signature_rank.json", 1, "code_teacher_signature_rank"),
        ("code_token_metric/top_token_shifts.csv", 1, "code_top_token_shifts"),
    )
    for relative, paper, variant in extras:
        path = run / relative
        if path.exists():
            summary = read_json(path) if path.suffix == ".json" else read_rows(path)
            out.append(diagnostic_record(
                path, paper=paper, variant=variant,
                title=f"Metric {paper}: {variant.replace('_', ' ')}",
                description="Saved supporting diagnostic; not an additional full-catalog SL predictor.",
                summary=summary))
    return out


def token_metrics(cell: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load the completed first-number suite only for the hash-matched SFT outcome."""
    run = TOKEN_SNAPSHOT / f"qwen25_{cell['family']}30_number_sequence_sft_10k_alpha32_3ep_h100"
    folder = run / "number_token_sft_metrics_v2"
    summary_path, protocol_path = folder / "metric_results_summary.json", folder / "suite_protocol.json"
    summary, protocol = read_json(summary_path), read_json(protocol_path)
    timing = read_json(folder / "resume_20712953_timing.json")
    ft = run / "m10/matrices/final/raw_logprob_shift.csv"
    if (summary["status"] != "completed" or timing["status"] != "completed"
            or str(timing["job_id"]) != "20746025" or summary["fingerprint"] != protocol["fingerprint"]):
        raise ValueError("Token metric suite is not the completed 20746025 snapshot")
    expected_hash = protocol["source_evidence"]["ft_matrix_sha256"]
    if source_record(ft)["source_sha256"] != expected_hash or cell["source_sha256"] != expected_hash:
        raise ValueError("Token metric outcome hash differs; do not reuse for FP32 or t6 reevaluation")
    if summary["traits"] != protocol["traits"] or set(summary["traits"]) != set(cell["labels"]):
        raise ValueError("Token metric trait catalogs differ")
    np.testing.assert_array_equal(read_matrix(ft, cell["labels"]), cell["raw"])
    specs = (
        (1, "metric_1/m4_teacher_similarity.csv", "teacher_cosine", "cosine", True),
        (2, "metric_2/matrices/top_identity_mi.csv", "top_identity_mi", "bits", True),
        (2, "metric_2/matrices/top_membership_mi.csv", "top_membership_mi", "bits", False),
        (3, "metric_3/matrices/behavior_gain.csv", "behavior_gain", "expected log-odds gain", True),
        (3, "metric_3/matrices/fisher_cosine.csv", "fisher_cosine", "cosine", False),
        (4, "metric_4/residual_rescaled/raw_logprob_shift.csv", "residual_rescaled", "nats", True),
        (4, "metric_4/v_full/raw_logprob_shift.csv", "v_full", "nats", False),
        (5, "metric_5/v_full/raw_logprob_shift.csv", "v_full", "nats", True),
        (5, "metric_5/residual_rescaled/raw_logprob_shift.csv", "residual_rescaled", "nats", False),
        (6, "metric_6/matrices/flat_all_layer_cosine.csv", "flat_all_layers", "cosine", True),
        (6, "metric_6/matrices/mean_per_layer_cosine.csv", "mean_per_layer", "cosine", False),
        (7, "metric_7/part1_causal_dataset_vectors/causal_raw_logprob_lift.csv", "causal_raw", "nats", True),
        (7, "metric_7/part2_behavior_dataset_alignment/cosine_layer_26.csv", "cosine_layer_26", "cosine", True),
    )
    saved = {v["artifact"]: v for v in summary["directional_matrices"].values()}
    results = []
    for paper, relative, variant, unit, main in specs:
        path = folder / relative
        record = matrix_record(
            cell, path, read_matrix(path, cell["labels"]), paper=paper, variant=variant, unit=unit,
            role="reported-main" if main else "reported-alternative", directional=paper != 1,
            description=("Completed token-SFT metric suite, first COMPLETE two-digit number normalized "
                         "among 00–99; not first tokenizer token or full ten-number sequence likelihood. "
                         f"Variant: {variant.replace('_', ' ')}. "
                         "FP32 metric evaluation paired to the original auto/BF16 SL outcome, not FP32 reevaluation."),
            protocol_source=source_record(protocol_path), precision_policy=protocol["precision_policy"])
        if relative in saved:
            np.testing.assert_allclose(record["pearson_dc_all900"],
                                       saved[relative]["double_centered_all"]["pearson_r"], atol=1e-10)
        results.append(record)
    for paper, key, variant, unit in ((1, "Metric 1", "teacher_pair_geometry", "correlation"),
                                      (2, "Metric 2 full information", "full_carrier_information", "bits")):
        results.append(diagnostic_record(
            summary_path, paper=paper, variant=variant, title=f"Metric {paper}: {variant.replace('_', ' ')}",
            description=("Saved 435-pair teacher-geometry comparison, distinct from 900-entry SL agreement."
                         if paper == 1 else "Saved per-teacher full-carrier information and diagonal correlations."),
            summary=summary[key], unit=unit, labels=cell["labels"], source_selector=key))
    return results, [{"paper_metric": 8, "title": TITLES[8], "status": "artifact-not-copied",
                      "reason": "Suite summary claims 30 completed SVD reports, but referenced report files "
                                "and spectra were not copied into the local snapshot. No values invented.",
                      **source_record(summary_path), "source_selector": "Metric 8",
                      "reported_artifact_count": len(summary["Metric 8"].get("reports", {}))}]


def load_metrics(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Return portable metric sets for exact endpoints, including explicit gaps."""
    originals = {}
    for original in read_json(COMPREHENSIVE)["cells"]:
        carrier = "number" if original["carrier"] == "00-99" else original["carrier"]
        objective = "sft" if carrier in ("gsm8k", "python") else "distribution"
        originals[f"qwen25_{original['family']}_{carrier}_{objective}"] = original
    if len({cell["id"] for cell in cells}) != len(cells):
        raise ValueError("Duplicate SL cell identity")
    sets = []
    for cell in cells:
        check_labels(cell["labels"])
        metrics, unavailable = [], []
        scope = "Only saved metrics from this exact original experiment are attached."
        if cell["id"] in originals:
            original = originals[cell["id"]]
            expected_carrier = "number" if original["carrier"] == "00-99" else original["carrier"]
            expected_objective = "sft" if expected_carrier in ("gsm8k", "python") else "distribution"
            if (cell["model"] != "qwen25" or cell["family"] != original["family"]
                    or cell["carrier"] != expected_carrier or cell["objective"] != expected_objective
                    or cell["scale"] != 1 or cell["subset"] != "primary"):
                raise ValueError("Comprehensive metric experiment identity differs")
            ft = SNAPSHOT / original["final_ft_source"]
            if cell["source_sha256"] != source_record(ft)["source_sha256"]:
                raise ValueError("Comprehensive metric endpoint hash differs")
            np.testing.assert_array_equal(cell["raw"], read_matrix(ft, cell["labels"]))
            run = ft.parents[3]
            metrics = legacy_matrices(cell, run) + legacy_diagnostics(cell, run)
            if original.get("m4"):
                geometry = original["m4"]
                metrics.append(diagnostic_record(
                    COMPREHENSIVE, paper=1, variant="teacher_pair_geometry",
                    title="Metric 1: Teacher-pair geometry agreement",
                    description="Saved 435-pair teacher-kernel versus double-centered SL-column cosine "
                                "comparison. Distinct from 900-entry metric-to-SL agreement; pairs "
                                "are not independent replicates.",
                    summary={"comparison": "teacher-pair cosine versus double-centered FT-column cosine",
                             "unique_off_diagonal_pairs": {
                                 "n_entries": 435, "pearson_r": geometry["pearson_pairs"],
                                 "spearman_rho": geometry["spearman_pairs"]},
                             "pairs_are_independent_replicates": False},
                    unit="correlation", labels=cell["labels"],
                    source_selector=f"cells[key={original['key']}].m4",
                    sl_cell_id=cell["id"], sl_source_sha256=cell["source_sha256"]))
            scope += " Legacy numbering is mapped to paper Metrics 1–8; all saved layers are retained."
        elif (cell["model"] == "qwen25" and cell["carrier"] == "number"
              and cell["objective"] == "sft" and cell["subset"] == "primary"):
            metrics, unavailable = token_metrics(cell)
            scope += " First-number metrics are measured in FP32; original SL outcome retains its recorded precision."
        available = {m["paper_metric"] for m in metrics} | {m["paper_metric"] for m in unavailable}
        for paper in range(1, 9):
            if paper not in available:
                if cell["subset"] != "primary":
                    reason = ("No metric suite saved for this exact variant endpoint. Original-run metrics are "
                              "not silently re-paired to extrapolated, smaller-budget or matched-FP32 SL matrices.")
                elif cell["model"] != "qwen25":
                    reason = "Metric suite was not run/copied for this model; only its SL outcome is available."
                elif cell["carrier"] in ("gsm8k", "python"):
                    reason = "Literal finite-carrier metric is unavailable for this long-form run."
                else:
                    reason = "No saved metric result for this experiment in the local snapshot."
                unavailable.append({"paper_metric": paper, "title": TITLES[paper],
                                    "status": "unavailable", "reason": reason})
        if len({m["id"] for m in metrics}) != len(metrics):
            raise ValueError(f"Duplicate metric variant: {cell['id']}")
        metrics.sort(key=lambda m: (m["paper_metric"], m["kind"] != "matrix",
                                    m["variant_role"] != "reported-main", m["variant"]))
        sets.append({"id": f"metrics_{cell['id']}", "cell_id": cell["id"],
                     "title": f"{cell['model_title']} {cell['family']} {cell['carrier_title']} {cell['objective_title']}",
                     "scope_note": scope, "metrics": metrics, "unavailable": unavailable})
    matrices = [m for s in sets for m in s["metrics"] if m["kind"] == "matrix"]
    diagnostics = [m for s in sets for m in s["metrics"] if m["kind"] == "diagnostic"]
    return {"sets": sets, "summary": {
        "set_count": len(sets), "sets_with_matrices": sum(any(m["kind"] == "matrix" for m in s["metrics"]) for s in sets),
        "matrix_count": len(matrices), "diagnostic_count": len(diagnostics),
        "matrix_counts_by_paper_metric": dict(sorted(Counter(m["paper_metric"] for m in matrices).items())),
        "unavailable_metric_count": sum(len(s["unavailable"]) for s in sets),
    }, "notes": [
        "All matrices are finite 30×30 saved measurements aligned by exact labels. No missing measurements are imputed.",
        "Paper numbering: legacy M4→1, M9→2, M3→3, M5→4, M7→5, M1→6, M2→7, M8→8. Legacy M10 is SL. Legacy M6 has historical 18-color T1/T2 pilot results, but no matched full-30-trait sweep in this atlas; current paper Metric 6 (legacy M1) is included.",
        "Raw means the saved metric output, in its own units. Output double-centering subtracts row/column means and adds the grand mean.",
        "Both SL correlations use fully double-centered matrices; the off-diagonal comparison masks 870 entries only after centering. No entry-independence p-values.",
        "Layer scans and upstream-centered variants are reported in full, not selected for strongest SL agreement. They are not independent replications.",
        "Metric 1 and Python code-token analogues are symmetric teacher kernels. Their 900-entry SL correlations are descriptive, distinct from 435-pair geometry comparisons.",
        "M8 activation matrices are not 30×30 trait predictors. Only locally saved reports are exposed; token-suite report paths alone do not establish available values.",
        "No original metric is silently attached to t=6, a smaller-budget pilot, or matched-FP32 reevaluation. Other-model metrics and missing experiments remain unavailable.",
        "Qwen3.5/Qwen3 calibration and partial-teacher audit outputs are outside this full-30-trait atlas. They are not completed full-catalog metric cells.",
    ]}
