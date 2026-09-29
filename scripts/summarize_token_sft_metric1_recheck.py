"""Streaming, shared-prompt-weighted Metric 1 recheck, without GPU inference.

The published Metric 1 is implemented historically as ``compute_m4``. This
module preserves its additive epsilon, across-teacher centering, and neutral
probability weighting, while making the common prompt sampling weights explicit.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from loguru import logger

EPSILON = 1e-12
SPLIT_SEED = 20260919
EDGE_RMS_THRESHOLD = 1.5


def distinctive_features(probabilities: np.ndarray, epsilon: float = EPSILON) -> np.ndarray:
    """Return neutral-weighted teacher-centered log shifts, neutral first."""
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 3 or p.shape[0] < 3 or min(p.shape[1:]) < 1:
        raise ValueError("Expected neutral + >=2 teachers, prompts, candidates")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    if not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("Invalid probabilities")
    if not np.allclose(p.sum(axis=-1), 1.0, atol=1e-5):
        raise ValueError("Probability rows must already be normalized")
    shifts = np.log(p[1:] + epsilon) - np.log(p[0:1] + epsilon)
    return np.sqrt(p[0:1]) * (shifts - shifts.mean(axis=0, keepdims=True))


def feature_gram(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Accumulate an unnormalized Gram; all-zero chunk weights are valid."""
    values, w = np.asarray(features, dtype=np.float64), np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or w.shape[0] != values.shape[1]:
        raise ValueError("Weights must match the prompt dimension")
    if not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("Prompt weights must be finite and nonnegative")
    weighted = (values * np.sqrt(w)[None, :, None]).reshape(values.shape[0], -1)
    return weighted @ weighted.T


def weighted_metric1_gram(probabilities: np.ndarray, weights: np.ndarray,
                          epsilon: float = EPSILON) -> np.ndarray:
    """Return the additive Gram for common weights, not yet cosine-normalized."""
    return feature_gram(distinctive_features(probabilities, epsilon), weights)


def gram_to_cosine(gram: np.ndarray) -> np.ndarray:
    """Normalize after summing every chunk, matching historical ``compute_m4``."""
    g = np.asarray(gram, dtype=np.float64)
    if g.ndim != 2 or g.shape[0] != g.shape[1] or not np.isfinite(g).all():
        raise ValueError("Gram must be a finite square matrix")
    norms = np.sqrt(np.maximum(np.diag(g), 0))
    if np.any(norms <= 0):
        raise ValueError("Degenerate teacher vector: no defined Metric 1 cosine")
    similarity = g / np.outer(norms, norms)
    return np.clip((similarity + similarity.T) / 2, -1, 1)


def double_center(matrix: np.ndarray) -> np.ndarray:
    """Remove row and column main effects from a trait-by-trait matrix."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2 or not np.isfinite(m).all():
        raise ValueError("Expected a finite matrix")
    return m - m.mean(axis=0, keepdims=True) - m.mean(axis=1, keepdims=True) + m.mean()


def read_labeled_matrix(path: Path, catalog: list[str]) -> np.ndarray:
    """Reorder both labeled CSV axes explicitly; never transpose directional SL."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows or len(rows[0]) != len(catalog) + 1:
        raise ValueError(f"Wrong matrix dimensions: {path}")
    columns, row_names = rows[0][1:], [row[0] for row in rows[1:]]
    if (len(set(columns)) != len(columns) or len(set(row_names)) != len(row_names)
            or set(columns) != set(catalog) or set(row_names) != set(catalog)):
        raise ValueError(f"Matrix labels do not match the catalog: {path}")
    data = np.asarray([row[1:] for row in rows[1:]], dtype=np.float64)
    if data.shape != (len(catalog), len(catalog)) or not np.isfinite(data).all():
        raise ValueError(f"Invalid matrix values: {path}")
    return data[np.ix_([row_names.index(name) for name in catalog],
                       [columns.index(name) for name in catalog])]


def matrix_agreement(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    """Descriptive Pearson/Spearman agreement; entries are not independent trials."""
    from scipy.stats import rankdata

    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Expected matching square matrices")

    def correlation(x: np.ndarray, y: np.ndarray) -> float | None:
        x, y = x - x.mean(), y - y.mean()
        denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
        return float(x @ y / denominator) if denominator > 0 else None

    result = {}
    for name, mask in (("all_entries", np.ones(a.shape, dtype=bool)),
                       ("off_diagonal", ~np.eye(len(a), dtype=bool))):
        x, y = a[mask], b[mask]
        result[name] = {"count": len(x), "pearson": correlation(x, y),
                        "spearman": correlation(rankdata(x), rankdata(y)),
                        "max_absolute_difference": float(np.max(np.abs(x - y)))}
    return result


def off_diagonal_rms(matrix: np.ndarray) -> float:
    """Return the exact off-diagonal RMS used only for graph/display scaling."""
    m = np.asarray(matrix, dtype=np.float64)
    return float(np.sqrt(np.mean(m[~np.eye(len(m), dtype=bool)] ** 2)))


def positive_edge_overlap(left_dc: np.ndarray, right_dc: np.ndarray) -> dict[str, Any]:
    """Compare positive directed edges at a fixed 1.5-times-own-RMS threshold."""
    mask = ~np.eye(len(left_dc), dtype=bool)
    left_rms, right_rms = off_diagonal_rms(left_dc), off_diagonal_rms(right_dc)
    a = (left_dc >= EDGE_RMS_THRESHOLD * left_rms) & (left_dc > 0) & mask
    b = (right_dc >= EDGE_RMS_THRESHOLD * right_rms) & (right_dc > 0) & mask
    overlap, union = int((a & b).sum()), int((a | b).sum())
    return {"threshold_in_own_off_diagonal_rms": EDGE_RMS_THRESHOLD,
            "left_rms": left_rms, "right_rms": right_rms,
            "left_edge_count": int(a.sum()), "right_edge_count": int(b.sum()),
            "intersection": overlap, "union": union,
            "jaccard": overlap / union if union else None,
            "fraction_left_retained": overlap / int(a.sum()) if a.any() else None,
            "fraction_right_recovered": overlap / int(b.sum()) if b.any() else None}


def expand_panels(design: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Add nested original-stream panels and deterministic unique-prompt halves."""
    panels = {name: {**entry, "weights": np.asarray(entry["weights"], dtype=np.float64)}
              for name, entry in design["panels"].items()}
    lengths = {len(entry["weights"]) for entry in panels.values()}
    if len(lengths) != 1:
        raise ValueError("Every panel must refer to the same prompt universe")
    n = next(iter(lengths))
    for name, entry in panels.items():
        w = entry["weights"]
        if w.ndim != 1 or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
            raise ValueError(f"Invalid or empty shared-weight panel: {name}")
    stream = design.get("stream_10k_prompt_indices")
    if stream is not None:
        indices = np.asarray(stream, dtype=np.int64)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= n):
            raise ValueError("Original-stream universe indices must be one-dimensional and valid")
        for count in (1024, 2048, 4096, 10000):
            if len(indices) >= count:
                name = "stream_10k" if count == 10000 else f"stream_{count}"
                weights = np.bincount(indices[:count], minlength=n).astype(np.float64)
                if name in panels:
                    original = panels[name]["weights"]
                    if not np.allclose(original / original.sum(), weights / weights.sum()):
                        raise ValueError(f"Existing {name} differs from the original stream prefix")
                    continue
                panels[name] = {"weights": weights,
                                "description": f"First {count} unfiltered seed-42 training requests; shared across every teacher"}
    for name, entry in list(panels.items()):
        support = np.flatnonzero(entry["weights"] > 0)
        if len(support) < 2:
            continue
        seed = int(hashlib.sha256(f"{SPLIT_SEED}:{name}".encode()).hexdigest()[:16], 16)
        order = np.random.default_rng(seed).permutation(support)
        halves = np.array_split(order, 2)
        for half, selected in enumerate(halves, 1):
            w = np.zeros(n)
            w[selected] = entry["weights"][selected]
            panels[f"{name}__half{half}"] = {
                "weights": w, "description": f"Fixed unique-prompt split {half}/2 of {name}",
                "parent": name, "split_seed": SPLIT_SEED,
            }
    return panels


def _write_matrix(path: Path, catalog: list[str], values: np.ndarray) -> None:
    """Save a matrix with explicit row/column catalog ordering."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_trait__column_trait", *catalog])
        writer.writerows([trait, *row] for trait, row in zip(catalog, values))


def _render_figure(path: Path, catalog: list[str], old: np.ndarray, new: np.ndarray,
                   sl: np.ndarray, panel: str) -> None:
    """Share one unclipped native-cosine scale; keep SL in separately labeled nats."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 6), constrained_layout=True)
    # Double-centering can take a cosine matrix outside [-1, 1], especially on
    # its diagonal. Use one data-covering range for BOTH old and new matrices.
    m1_extent = max(float(np.max(np.abs(double_center(matrix)))) for matrix in (old, new))
    m1_limit = max(0.1, math.ceil(m1_extent * 10) / 10)
    images = []
    for index, (ax, matrix, title) in enumerate(zip(axes, (old, new, sl),
                                 ("Historical 1,024 · Metric 1", f"{panel} · Metric 1", "Observed token-SFT effect"))):
        centered = double_center(matrix)
        if index < 2:
            displayed, limit = centered, m1_limit
        else:
            displayed, limit = centered, max(float(np.max(np.abs(centered))), 1e-12)
        images.append(ax.imshow(displayed, cmap="RdBu_r", vmin=-limit, vmax=limit))
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(catalog)), catalog, rotation=90, fontsize=6)
        ax.set_yticks(range(len(catalog)), catalog, fontsize=6)
        ax.set_xlabel("Teacher trait / trained adapter")
    axes[0].set_ylabel("Trait; SL panel rows are answer traits")
    fig.colorbar(images[0], ax=axes[:2], shrink=.75,
                 label=f"Trait-double-centered M1 cosine (shared ±{m1_limit:g})")
    fig.colorbar(images[2], ax=axes[2], shrink=.75, label="Trait-double-centered SL effect (nats)")
    fig.suptitle("Shared-prompt Metric 1 recheck · unchanged M1 magnitude scale", fontsize=14)
    fig.supxlabel("Warm = positive centered interaction; not necessarily raw probability enhancement. "
                  "M1 and SL have different units. Old/new M1 share one unclipped scale.", fontsize=9)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def summarize(bundle: Path, output_dir: Path, domain: str, design: Mapping[str, Any],
              chunk_iterator: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Accumulate validated neutral-first chunks and publish completed panels.

    Required design: ``traits``, ``panels`` mapping to common ``weights`` and
    description, and ``references`` containing bundle-relative ``historical_metric1``
    and ``sl`` CSVs. ``stream_10k_prompt_indices`` identifies the ordered original
    request prefix. The caller validates cache/model/prompt identity; this module
    independently validates shapes, normalization, indices, and coverage.
    """
    bundle, output_dir = Path(bundle).resolve(), Path(output_dir)
    catalog = list(design["traits"])
    if len(catalog) != len(set(catalog)):
        raise ValueError("Duplicate catalog labels")
    panels = expand_panels(design)
    universe_count = len(next(iter(panels.values()))["weights"])
    seen = np.zeros(universe_count, dtype=bool)
    grams = {name: np.zeros((len(catalog), len(catalog))) for name in panels}
    for chunk in chunk_iterator:
        indices = np.asarray(chunk["indices"], dtype=np.int64)
        p = np.asarray(chunk["probabilities"], dtype=np.float64)
        if (indices.ndim != 1 or not len(indices) or len(set(indices.tolist())) != len(indices)
                or np.any(indices < 0) or np.any(indices >= universe_count)
                or np.any(seen[indices])):
            raise ValueError("Duplicate, empty, or out-of-range chunk indices")
        if p.shape != (len(catalog) + 1, len(indices), 100):
            raise ValueError("Chunk must have neutral-first (T+1, chunk, 100) shape")
        features = distinctive_features(p)
        for name, panel in panels.items():
            weights = panel["weights"][indices]
            if np.any(weights > 0):
                grams[name] += feature_gram(features, weights)
        seen[indices] = True
    references = {}
    for key in ("historical_metric1", "sl"):
        path = (bundle / design["references"][key]).resolve()
        if not path.is_relative_to(bundle):
            raise ValueError("Reference must be contained in the frozen bundle")
        references[key] = read_labeled_matrix(path, catalog)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir = output_dir / "matrices"
    matrix_dir.mkdir(exist_ok=True)
    matrices: dict[str, np.ndarray] = {}
    summaries: dict[str, Any] = {}
    for name, panel in panels.items():
        if not name.replace("_", "").isalnum():
            raise ValueError("Panel names must be alphanumeric/underscores")
        weights = panel["weights"]
        missing = int(((weights > 0) & ~seen).sum())
        record: dict[str, Any] = {
            "description": panel.get("description", ""), "status": "partial" if missing else "completed",
            "unique_prompt_count": int((weights > 0).sum()), "total_weight": float(weights.sum()),
            "effective_weight_sample_size": float(weights.sum() ** 2 / (weights @ weights)),
            "missing_prompt_count": missing, "weights_sha256": hashlib.sha256(weights.astype("<f8").tobytes()).hexdigest(),
        }
        if panel.get("parent"):
            record["split_parent"] = panel["parent"]
        if not missing:
            try:
                matrix = gram_to_cosine(grams[name])
            except ValueError as exc:
                record.update(status="undefined", reason=str(exc))
            else:
                matrices[name] = matrix
                dc = double_center(matrix)
                record["agreement_with_sl_trait_dc"] = matrix_agreement(dc, double_center(references["sl"]))
                record["agreement_with_historical_m1_raw"] = matrix_agreement(matrix, references["historical_metric1"])
                record["agreement_with_historical_m1_trait_dc"] = matrix_agreement(dc, double_center(references["historical_metric1"]))
                record["positive_edge_overlap_with_sl"] = positive_edge_overlap(dc, double_center(references["sl"]))
                record["positive_edge_overlap_with_historical_m1"] = positive_edge_overlap(dc, double_center(references["historical_metric1"]))
                record["matrix_csv"] = f"matrices/{name}_metric1.csv"
                record["matrix_dc_csv"] = f"matrices/{name}_metric1_trait_dc.csv"
                _write_matrix(output_dir / record["matrix_csv"], catalog, matrix)
                _write_matrix(output_dir / record["matrix_dc_csv"], catalog, dc)
        summaries[name] = record
    splits = {}
    for name in panels:
        first, second = f"{name}__half1", f"{name}__half2"
        if first in matrices and second in matrices:
            splits[name] = {
                "seed": SPLIT_SEED, "unit": "unique prompt, preserving original weights within each half",
                "trait_dc_agreement": matrix_agreement(double_center(matrices[first]), double_center(matrices[second])),
                "positive_edge_overlap": positive_edge_overlap(double_center(matrices[first]), double_center(matrices[second])),
            }
    parent_names = [name for name, panel in panels.items() if not panel.get("parent")]
    figures = {}
    for name in ("stream_10k", "accepted_all", "accepted_common"):
        if name in matrices:
            filename = f"{name}_metric1_vs_sl.png"
            _render_figure(output_dir / filename, catalog, references["historical_metric1"], matrices[name], references["sl"], name)
            figures[name] = filename
    result = {
        "status": "completed" if all(summaries[name]["status"] == "completed" for name in parent_names) else "partial",
        "domain": domain, "catalog": catalog, "epsilon": EPSILON, "scored_unique_prompts": int(seen.sum()),
        "prompt_universe_count": universe_count, "panels": summaries, "split_half_stability": splits,
        "references": dict(design["references"]), "figures": figures,
        "source_provenance": dict(design.get("provenance", {})),
        "prompt_audit": dict(design.get("prompt_audit", {})),
        "historical_first1024_reproduction": summaries.get("stream_1024", {}).get("agreement_with_historical_m1_raw"),
        "historical_metric1_vs_sl": matrix_agreement(double_center(references["historical_metric1"]), double_center(references["sl"])),
        "definitions": {
            "log_shift": "L_t(x,n)=log(p_t(n|x)+1e-12)-log(p_0(n|x)+1e-12)",
            "teacher_centering": "D_t(x,n)=L_t(x,n)-mean_s L_s(x,n), mean over all catalog teachers, not neutral",
            "gram": "G_tu=sum_x w(x) sum_n p_0(n|x) D_t(x,n) D_u(x,n)",
            "metric1": "M1_tu=G_tu/sqrt(G_tt G_uu). Same shared weights for every teacher; no per-chunk normalization.",
            "trait_double_centering": "A_tu-mean_v A_tv-mean_v A_vu+grand_mean(A); separate from per-prompt teacher centering above.",
            "sl_orientation": "Rows=answer traits, columns=trained teacher adapters; asymmetric SL is never symmetrized.",
            "interpretation": "Metric 1 is symmetric teacher-signature similarity, not a causal intervention or directed training-effect matrix.",
        },
        "limitations": [
            "Descriptive correlations over dependent matrix entries, not independent replications or p-values.",
            "Positive-edge threshold is fixed at 1.5 times each matrix's off-diagonal RMS; it is not significance filtering.",
            "Split halves partition unique prompts; effective-weight sample size is a weight-concentration diagnostic, not a causal sample size.",
            "Historical first-1024 comparisons are numerical reproduction diagnostics; no acceptance tolerance is silently imposed.",
            "The shared accepted-prompt mixture includes each teacher's empirical FT prompt distribution; it is not a common 10k set accepted by every teacher.",
            "Using actual accepted prompt inputs does not condition the recomputed first-number distribution on acceptance of the entire ten-number response. It is not the exact response-filtered target distribution.",
            "Symmetric teacher similarity and directional SL measure different objects. Different network thresholds can also change apparent visual agreement.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "index.html").write_text(render_html(result), encoding="utf-8")
    logger.success("{} Metric 1 recheck: {} scored prompts, {} complete panels", domain, int(seen.sum()), len(matrices))
    return result


def render_html(result: Mapping[str, Any]) -> str:
    """Create a portable offline report with explicit estimand and diagnostics."""
    esc = html.escape

    def number(value: Any) -> str:
        return "—" if value is None else f"{float(value):+.4f}"

    rows = []
    for name, record in result["panels"].items():
        if record.get("split_parent"):
            continue
        agreement = record.get("agreement_with_sl_trait_dc", {})
        old = record.get("agreement_with_historical_m1_raw", {}).get("all_entries", {})
        edge = record.get("positive_edge_overlap_with_sl", {})
        rows.append("<tr>" + "".join(f"<td>{esc(str(value))}</td>" for value in (
            name, record["status"], record["unique_prompt_count"], f"{record['total_weight']:g}",
            number(agreement.get("all_entries", {}).get("pearson")),
            number(agreement.get("off_diagonal", {}).get("pearson")),
            number(agreement.get("off_diagonal", {}).get("spearman")),
            number(old.get("pearson")), number(old.get("max_absolute_difference")),
            number(edge.get("jaccard")))) + "</tr>")
    gallery = "".join(f'<figure><img src="{esc(filename)}" alt="{esc(name)} Metric 1 and observed SL matrices">'
                      f'<figcaption>{esc(name)} · old/new M1: one shared centered-cosine scale covering both matrices without clipping; SL: separate native-nats scale, identical across figures. Unscaled M1 values in CSV.</figcaption></figure>'
                      for name, filename in result["figures"].items())
    split_rows = []
    for name, row in result["split_half_stability"].items():
        corr = row["trait_dc_agreement"]["off_diagonal"]
        split_rows.append(f"<tr><td>{esc(name)}</td><td>{number(corr['pearson'])}</td><td>{number(corr['spearman'])}</td><td>{number(row['positive_edge_overlap']['jaccard'])}</td></tr>")
    panel_definitions = "".join(f"<li><strong>{esc(name)}</strong>: {esc(row['description'])}</li>"
                                for name, row in result["panels"].items() if not row.get("split_parent"))
    links = "".join(f'<li>{esc(name)}: <a href="{esc(row["matrix_csv"])}">raw cosine CSV</a> · '
                    f'<a href="{esc(row["matrix_dc_csv"])}">trait-double-centered CSV</a></li>'
                    for name, row in result["panels"].items() if "matrix_csv" in row)
    historical = result["historical_metric1_vs_sl"]
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{esc(str(result['domain']))} · Metric 1 prompt recheck</title><style>
body{{margin:0;background:#fafaf7;color:#253b44;font:16px/1.6 system-ui,sans-serif}}main{{max-width:1350px;margin:auto;padding:30px 24px}}h1{{font-size:clamp(28px,4vw,42px);line-height:1.2}}h2{{margin-top:36px}}.note{{background:#edf3f0;padding:14px 18px;border-left:4px solid #427f77}}.formula{{background:white;padding:18px;overflow:auto;border:1px solid #dce4df;font-family:Georgia,serif;font-size:18px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;white-space:nowrap;width:100%;font-size:14px}}th,td{{padding:9px 12px;text-align:right;border-bottom:1px solid #dce4df}}th{{background:#edf3f0}}th:first-child,td:first-child{{text-align:left}}figure{{margin:24px 0;padding:8px;background:white;border:1px solid #dce4df}}img{{max-width:100%;height:auto}}figcaption{{font-size:13px}}a{{color:#216b86}}code{{font-size:.9em}}@media(max-width:500px){{main{{padding:20px 12px}}}}
</style></head><body><main><h1>{esc(str(result['domain']).title())} · Metric 1 prompt-distribution recheck</h1><p class="note"><strong>{esc(str(result['status']).upper())}</strong> · {result['scored_unique_prompts']:,} / {result['prompt_universe_count']:,} unique universe prompts scored. Panels with missing weighted support are not normalized or reported as completed.</p>
<h2>What is held fixed?</h2><p>Every teacher and the no-system-prompt neutral model are evaluated on exactly the same prompts and 100 complete first-number candidates (00–99), while the original request still asks for ten numbers. Only the common prompt sampling weights change. This is the report’s <strong>Metric 1</strong>; the historical implementation and CSV retain the older name <code>compute_m4</code>/<code>m4_teacher_similarity.csv</code>.</p><ul>{panel_definitions}</ul>
<h2>Metric definition</h2><p>Let p<sub>t</sub>(n|x) be teacher t’s candidate-normalized first-number distribution, p<sub>0</sub> the neutral distribution, and w(x) the same nonnegative prompt weight for every teacher. The trait average below is over all {len(result['catalog'])} teachers, excluding neutral.</p>
<div class="formula">L<sub>t</sub>(x,n) = log(p<sub>t</sub>(n|x) + ε) − log(p<sub>0</sub>(n|x) + ε), &nbsp; ε = 10<sup>−12</sup><br>D<sub>t</sub>(x,n) = L<sub>t</sub>(x,n) − (1/T) ∑<sub>s</sub> L<sub>s</sub>(x,n)<br>G<sub>tu</sub> = ∑<sub>x</sub> w(x) ∑<sub>n</sub> p<sub>0</sub>(n|x) D<sub>t</sub>(x,n) D<sub>u</sub>(x,n)<br>M1<sub>tu</sub> = G<sub>tu</sub> / √(G<sub>tt</sub>G<sub>uu</sub>)</div>
<p>This is cosine similarity between teacher-distinctive, neutral-probability-weighted signatures. Multiplying all prompt weights by one positive constant changes no cosine. Gram contributions are summed across chunks <em>before</em> normalization. A pooled accepted panel uses training-prompt multiplicities, not a separate prompt support for each teacher.</p>
<h2>Agreement with observed fine-tuning</h2><p>For any trait matrix A, define <code>Ã = A − row mean − column mean + grand mean</code>. This trait-by-trait double-centering is separate from D’s across-teacher centering within each prompt/candidate. We compare centered M1 with the centered observed token-SFT matrix, preserving SL’s rows = answer, columns = trained adapter. M1 is symmetric; SL is asymmetric and is not symmetrized.</p>
<p>Pearson r measures linear agreement; Spearman ρ measures rank agreement. “All” uses the complete matrix; “off-diagonal” excludes true trait self-pairs. These entries share traits and are not independent replications. Historical r and max |difference| compare raw cosine matrices. The first-1,024 row is the numerical reproduction check, not a preregistered pass/fail test.</p>
<p class="note"><strong>Frozen historical baseline:</strong> original M1 versus SL, both trait-double-centered: Pearson r = {number(historical['all_entries']['pearson'])} over all entries; r = {number(historical['off_diagonal']['pearson'])} off-diagonal. These baseline values are recomputed from the bundled reference matrices, not hard-coded.</p>
<div class="scroll"><table><thead><tr><th>Panel</th><th>Status</th><th>Unique prompts</th><th>Weight sum</th><th>SL r · all</th><th>SL r · offdiag</th><th>SL ρ · offdiag</th><th>Old M1 r</th><th>Old max |Δ|</th><th>SL edge Jaccard</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>{gallery}
<h2>Stability and fixed graph threshold</h2><p>Each panel’s unique prompts are partitioned into two fixed random halves (seed {SPLIT_SEED}); original weights are retained within each half. The table compares their centered matrices. This is a prompt-stability diagnostic, not uncertainty across independent training seeds.</p><p>Positive off-diagonal edges satisfy <code>Ã<sub>tu</sub> ≥ 1.5 × RMS<sub>offdiag</sub>(Ã)</code> and <code>Ã<sub>tu</sub> &gt; 0</code>, with the same fixed rule for each matrix. Jaccard = shared edges / union of edges. The threshold is not tuned for agreement and is not a significance cutoff.</p><div class="scroll"><table><thead><tr><th>Panel</th><th>Half r · offdiag</th><th>Half ρ · offdiag</th><th>Half edge Jaccard</th></tr></thead><tbody>{''.join(split_rows)}</tbody></table></div>
<h2>Data and limits</h2><p><a href="summary.json">Complete summary and definitions</a></p><ul>{links}</ul><ul>{''.join(f'<li>{esc(item)}</li>' for item in result['limitations'])}</ul></main></body></html>'''
