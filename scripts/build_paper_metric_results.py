"""Build audited main-paper metric figures from the frozen report payload.

Run from the repository root. No model inference or cluster jobs are performed.
The finite-carrier experiments in this payload are distribution distillation,
not the subsequently trained sampled-number SFT experiments.
"""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results_for_presentation/qwen25_comprehensive_sweep_report.data.json"
MANIFEST = ROOT / "results_for_presentation/qwen25_comprehensive_sweep_report.sources.json"
OUT = ROOT / "results_for_presentation/paper_metric_results"
PDF_OUT = ROOT / "output/pdf/paper_metric_results"
FAMILIES = ("animal", "tree", "city", "color")
CARRIERS = ("00-99", "opaque", "gsm8k", "python")
COLORS = dict(zip(FAMILIES, ("#9f4e77", "#348367", "#3b70ad", "#bd741b")))
MARKERS = dict(zip(CARRIERS, ("o", "s", "^", "D")))
CARRIER_NAMES = dict(zip(CARRIERS, ("00-99", "Opaque 100", "GSM8K", "Python")))
ROWS = (
    ("M9 identity", "Metric 2  Entangled-carrier information"),
    ("M3 gain", "Metric 3  Behavior gain"),
    ("M5", "Metric 4  Mean-embedding intervention"),
    ("M7", "Metric 5  Real-carrier intervention"),
    ("M1 flat", "Metric 6  Learned ICV alignment"),
    ("M2 causal", "Metric 7  Dataset-vector steering"),
    ("M2 cosine", "Metric 7  Dataset-vector alignment"),
)
INK = "#23383e"
MUTED = "#607278"
CORR_CMAP = LinearSegmentedColormap.from_list(
    "agreement", ["#315f91", "#f8f7f4", "#b13641"]
)


def double_center(matrix: Any) -> np.ndarray:
    """Remove row and column main effects from a finite 2D matrix."""
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("Expected a finite two-dimensional matrix")
    return array - array.mean(0, keepdims=True) - array.mean(1, keepdims=True) + array.mean()


def correlation(a: Any, b: Any, rank: bool = False) -> float:
    """Compute Pearson or tie-aware Spearman, rejecting undefined inputs."""
    left, right = np.asarray(a, dtype=float).ravel(), np.asarray(b, dtype=float).ravel()
    if left.shape != right.shape or left.size < 2:
        raise ValueError("Correlation inputs must have equal sizes of at least two")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Correlation inputs must be finite")
    if rank:
        left, right = rankdata(left), rankdata(right)
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        raise ValueError("Constant input has undefined correlation")
    return float(np.corrcoef(left, right)[0, 1])


def directional_statistics(matrix: Any, ft: Any) -> dict[str, float]:
    """Compare full-centered directional matrices, then mask the diagonal."""
    a, f = double_center(matrix), double_center(ft)
    if a.shape != f.shape or a.shape[0] != a.shape[1]:
        raise ValueError("Expected equally sized square answer-by-teacher matrices")
    off = ~np.eye(len(a), dtype=bool)
    return {
        "pearson_all": correlation(a, f),
        "pearson_off": correlation(a[off], f[off]),
        "spearman_all": correlation(a, f, rank=True),
        "spearman_off": correlation(a[off], f[off], rank=True),
        "diagonal_pearson": correlation(np.diag(a), np.diag(f)),
        "diagonal_spearman": correlation(np.diag(a), np.diag(f), rank=True),
    }


def pair_statistics(kernel: Any, ft: Any) -> dict[str, float]:
    """Compare symmetric teacher geometry with centered FT-column geometry."""
    k, f = np.asarray(kernel, dtype=float), double_center(ft)
    if k.shape != f.shape or not np.isfinite(k).all() or not np.allclose(k, k.T):
        raise ValueError("Expected a finite symmetric teacher-pair kernel")
    norms = np.linalg.norm(f, axis=0)
    if np.any(norms == 0):
        raise ValueError("FT column cosine is undefined for a zero vector")
    ft_kernel = (f.T @ f) / np.outer(norms, norms)
    pairs = np.triu_indices(len(k), k=1)
    return {
        "pearson_pairs": correlation(k[pairs], ft_kernel[pairs]),
        "spearman_pairs": correlation(k[pairs], ft_kernel[pairs], rank=True),
    }


def file_sha256(path: Path) -> str:
    """Hash an input artifact for reproducible numerical provenance."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_analysis(payload: dict, source_path: Path, manifest_path: Path) -> dict:
    """Recompute every displayed statistic and verify against saved summaries."""
    indexed = {cell["key"]: cell for cell in payload["cells"]}
    cells = []
    errors = []
    source_records = json.loads(manifest_path.read_text())
    manifest_paths = {item["path"] for item in source_records if "path" in item}
    for carrier in CARRIERS:
        for family in FAMILIES:
            original = indexed[f"{family}-{carrier}"]
            labels = original["labels"]
            ft = np.asarray(original["matrix"], dtype=float)
            if ft.shape != (30, 30) or len(set(labels)) != 30 or not np.isfinite(ft).all():
                raise ValueError(f"Invalid 30-trait FT input: {original['key']}")
            if original["final_ft_source"] not in manifest_paths:
                raise ValueError("FT input is absent from the provenance manifest")
            directional = {}
            for legacy, metric in original["metrics"].items():
                stats = directional_statistics(metric["matrix"], ft)
                directional[legacy] = stats
                for new, old in (("pearson_all", "pearson_dc"),
                                 ("pearson_off", "pearson_dc_off"),
                                 ("spearman_all", "spearman_dc"),
                                 ("diagonal_spearman", "diagonal_spearman")):
                    errors.append(abs(stats[new] - metric[old]))
            centered = double_center(ft)
            item = {
                "key": original["key"], "family": family, "carrier": carrier,
                "training_objective": "full-distribution distillation" if carrier in CARRIERS[:2]
                else "completion-token SFT",
                "labels": labels, "final_ft_source": original["final_ft_source"],
                "directional": directional,
                "centered_ft_diagonal_mean": float(np.diag(centered).mean()),
                "centered_ft_diagonal": np.diag(centered).tolist(),
                "median_first_mode": original["m8"]["median_ev1"],
                "pair": None, "full_information": None,
            }
            errors.append(abs(item["centered_ft_diagonal_mean"] - original["ft"]["interaction"]))
            if carrier in CARRIERS[:2]:
                item["pair"] = pair_statistics(original["m4"]["matrix"], ft)
                errors.append(abs(item["pair"]["pearson_pairs"] - original["m4"]["pearson_pairs"]))
                errors.append(abs(item["pair"]["spearman_pairs"] - original["m4"]["spearman_pairs"]))
                full = np.array([original["m9"]["full_mi"][label] for label in labels])
                item["full_information"] = {
                    "bits_by_trait": full.tolist(),
                    "pearson_diagonal": correlation(full, np.diag(centered)),
                    "spearman_diagonal": correlation(full, np.diag(centered), rank=True),
                }
            cells.append(item)
    discrepancy = max(errors)
    if discrepancy > 1e-10:
        raise ValueError(f"Recomputed statistics disagree with report: {discrepancy}")
    return {
        "source_snapshot_date": payload["snapshot_date"],
        "source_sha256": file_sha256(source_path),
        "manifest_sha256": file_sha256(manifest_path),
        "source_record_count": len(source_records),
        "max_saved_statistic_discrepancy": discrepancy,
        "conventions": {
            "orientation": "rows = scored answers; columns = trained teacher adapters",
            "directional": "center full matrices first, then compare 870 off-diagonal entries",
            "teacher_geometry": "435 strict-upper-triangle pairs of teacher and FT-column kernels",
            "uncertainty": "descriptive correlations; dependent entries, no naive iid confidence intervals",
            "number_token_sft": "NOT represented; job 20629102 failed without completed suite",
            "metric_variants": dict(ROWS),
        },
        "cells": cells,
    }


def style() -> None:
    """Set consistent publication typography and editable vector outputs."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.labelsize": 8.5, "axes.titlesize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#c2ccce", "axes.linewidth": .6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def annotate_cells(ax: plt.Axes, values: np.ndarray) -> None:
    """Display numerical correlations without masking weak or negative values."""
    for row in range(len(values)):
        for col in range(values.shape[1]):
            value = values[row, col]
            label = "0.00" if abs(value) < .005 else f"{value:+.2f}"
            ax.text(col, row, label, ha="center", va="center", fontsize=9,
                    color="white" if abs(value) > .64 else INK,
                    fontweight="bold" if abs(value) > .64 else "normal")


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Export the same layout as vector PDF/SVG and high-resolution PNG."""
    fig.savefig(PDF_OUT / f"{stem}.pdf", metadata={"Title": stem.replace("_", " ")})
    fig.savefig(OUT / f"{stem}.svg")
    fig.savefig(OUT / f"{stem}.png", dpi=240)
    plt.close(fig)


def figure_agreement(analysis: dict) -> None:
    """Separate pair-geometry agreement from grouped directional predictors."""
    finite = analysis["cells"][:8]
    fig = plt.figure(figsize=(8, 5.35))
    fig.text(.025, .962, "A  Teacher-pair geometry", weight="bold", fontsize=11)
    fig.text(.975, .962, "435 pairs per setting", ha="right", fontsize=8, color=MUTED)
    left, width = .365, .61
    for x, label in ((left + width / 4, "00-99 numbers"),
                     (left + 3 * width / 4, "Opaque 100")):
        fig.text(x, .913, label, ha="center", weight="bold", fontsize=9)
    ax1 = fig.add_axes([left, .798, width, .062])
    pair = np.array([[c["pair"]["pearson_pairs"] for c in finite]])
    ax1.imshow(pair, vmin=-1, vmax=1, cmap=CORR_CMAP, aspect="auto")
    annotate_cells(ax1, pair)
    ax1.set_yticks([0], ["Metric 1  Teacher-distribution geometry"])
    ax1.set_xticks(range(8), [f.title() for f in FAMILIES] * 2)
    ax1.xaxis.tick_top()
    ax1.tick_params(length=0, pad=6)
    ax1.axvline(3.5, color="white", lw=3)
    for spine in ax1.spines.values():
        spine.set_visible(False)
    fig.text(.025, .735, "B  Directional agreement beyond the diagonal", weight="bold", fontsize=11)
    fig.text(.975, .697, "870 entries per setting; center first, exclude diagonal second",
             ha="right", fontsize=8, color=MUTED)
    ax = fig.add_axes([left, .18, width, .482])
    values = np.array([[c["directional"][old]["pearson_off"] for c in finite] for old, _ in ROWS])
    image = ax.imshow(values, vmin=-1, vmax=1, cmap=CORR_CMAP, aspect="auto")
    annotate_cells(ax, values)
    ax.set_xticks([])
    ax.set_yticks(range(7), [name for _, name in ROWS])
    ax.tick_params(length=0, pad=7)
    ax.axvline(3.5, color="white", lw=3)
    for y in (.5, 3.5):
        ax.axhline(y, color="white", lw=4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Roman numerals match the categories in Section 3.1, without renumbering variants.
    for text, y in (("I", .628), ("II", .49), ("III", .284)):
        fig.text(.027, y, text, fontsize=10, weight="bold", color=MUTED, va="center")
    cax = fig.add_axes([.625, .12, .35, .018])
    bar = fig.colorbar(image, cax=cax, orientation="horizontal", ticks=[-1, -.5, 0, .5, 1])
    bar.set_label("Pearson correlation  r", fontsize=8, labelpad=2)
    bar.outline.set_visible(False)
    fig.text(.025, .12, "I  Carrier distributions   II  Carrier-to-behavior coupling\n"
             "III  Internal representations", fontsize=7.7, linespacing=1.8, color=MUTED)
    fig.text(.025, .018, "All eight settings: full-distribution distillation.  "
             "Metric 8 is a separate spectral diagnostic.", fontsize=8, color=MUTED)
    save_figure(fig, "metric_agreement")


def clean_axis(ax: plt.Axes) -> None:
    """Keep only light reference lines and necessary axes."""
    ax.grid(axis="y", color="#e6ebec", lw=.5, zorder=0)
    ax.tick_params(length=3, width=.6)


def figure_signal(analysis: dict) -> None:
    """Show transfer beyond self-effects and limits of scalar signal summaries."""
    cells = analysis["cells"]
    fig = plt.figure(figsize=(8, 6.3))
    fig.text(.025, .966, "A  Metric 7 steering: agreement beyond self-effects",
             fontsize=10.5, weight="bold")
    ax = fig.add_axes([.095, .637, .855, .264])
    offsets = np.linspace(-.23, .23, 4)
    for i, carrier in enumerate(CARRIERS):
        for j, family in enumerate(FAMILIES):
            cell = next(c for c in cells if c["carrier"] == carrier and c["family"] == family)
            all_r = cell["directional"]["M2 causal"]["pearson_all"]
            off_r = cell["directional"]["M2 causal"]["pearson_off"]
            x = i + offsets[j]
            ax.plot([x, x], [off_r, all_r], color=COLORS[family], lw=1, alpha=.65)
            ax.scatter(x, all_r, color=COLORS[family], s=32, zorder=4)
            ax.scatter(x, off_r, facecolor="white", edgecolor=COLORS[family], s=32, lw=1.2, zorder=5)
    ax.set(xlim=(-.5, 3.5), ylim=(-.035, .94), ylabel="Pearson correlation", yticks=[0, .2, .4, .6, .8])
    ax.set_xticks(range(4), ["00-99\nDistribution distillation", "Opaque 100\nDistribution distillation",
                                "GSM8K\nToken SFT", "Python\nToken SFT"])
    ax.axvline(1.5, lw=.8, color="#bdc9cc", ls=(0, (3, 3)))
    clean_axis(ax)
    ax.legend(handles=[Line2D([], [], marker="o", color=MUTED, ls="", label="All entries", ms=5),
                       Line2D([], [], marker="o", markerfacecolor="white", color=MUTED, ls="",
                              label="Off-diagonal only", ms=5)],
              loc="upper right", ncol=2, frameon=False, fontsize=8, handletextpad=.4)
    fig.text(.025, .508, "B  Full information vs. matching FT effect", weight="bold", fontsize=10)
    fig.text(.025, .479, "Metric 2 full-support variant; 30 traits per point", fontsize=8, color=MUTED)
    bx = fig.add_axes([.1, .153, .355, .29])
    for j, family in enumerate(FAMILIES):
        for carrier, offset in (("00-99", .105), ("opaque", -.105)):
            cell = next(c for c in cells if c["family"] == family and c["carrier"] == carrier)
            r = cell["full_information"]["pearson_diagonal"]
            bx.scatter(r, j + offset, marker=MARKERS[carrier], color=COLORS[family],
                       s=35, edgecolor="white", lw=.5, zorder=4)
            bx.annotate(f"{r:+.2f}", (r, j + offset), xytext=(5 if r >= 0 else -5, 0),
                        textcoords="offset points", ha="left" if r >= 0 else "right",
                        va="center", fontsize=7.5, color=COLORS[family])
    bx.axvline(0, color="#ccd5d7", lw=.8)
    bx.set(xlim=(-1, 1), ylim=(3.55, -.55), xlabel="Pearson correlation across traits",
           xticks=[-1, -.5, 0, .5, 1])
    bx.set_yticks(range(4), [f.title() for f in FAMILIES])
    bx.spines["left"].set_visible(False)
    bx.tick_params(axis="y", length=0)
    fig.text(.525, .508, "C  Spectral concentration vs. transfer", weight="bold", fontsize=10)
    fig.text(.525, .479, "Metric 8 first-mode energy; one point per setting", fontsize=8, color=MUTED)
    cx = fig.add_axes([.595, .153, .365, .29])
    for cell in cells:
        cx.scatter(100 * cell["median_first_mode"], cell["centered_ft_diagonal_mean"],
                   marker=MARKERS[cell["carrier"]], s=42, color=COLORS[cell["family"]],
                   edgecolor="white", lw=.6, zorder=3)
    cx.set(xlim=(18, 84), ylim=(0, 10.5), xticks=[20, 40, 60, 80], yticks=[0, 2, 4, 6, 8, 10],
           xlabel="Median first-mode energy (%)", ylabel="Mean centered FT diagonal (nats)")
    clean_axis(cx)
    # Separate family and carrier keys; shapes are used only in B and C.
    handles = [Line2D([], [], ls="", marker="o", color=COLORS[f], ms=5, label=f.title()) for f in FAMILIES]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .047),
               frameon=False, ncol=4, columnspacing=1.5, handletextpad=.5, fontsize=8)
    handles = [Line2D([], [], ls="", marker=MARKERS[c], color=MUTED, ms=5,
                      label=CARRIER_NAMES[c]) for c in CARRIERS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .006),
               frameon=False, ncol=4, columnspacing=1.5, handletextpad=.5, fontsize=8)
    save_figure(fig, "signal_and_transfer")


def write_preview(analysis: dict) -> None:
    """Create a local reading view with scope, figures, and full numeric audit."""
    table_rows = []
    for cell in analysis["cells"]:
        for old, stats in cell["directional"].items():
            name = dict(ROWS).get(old, old + " (companion variant)")
            table_rows.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in [
                cell["key"], name, f"{stats['pearson_all']:+.3f}", f"{stats['pearson_off']:+.3f}",
                f"{stats['spearman_all']:+.3f}", f"{stats['spearman_off']:+.3f}",
                f"{stats['diagonal_spearman']:+.3f}",
            ]) + "</tr>")
    document = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Metric results - paper figures</title>
<style>body{font:17px/1.65 system-ui,sans-serif;color:#23383e;background:#f8f9f7;margin:0}
main{max-width:1120px;margin:auto;padding:44px 28px}h1{line-height:1.15;font-size:38px}h2{line-height:1.2;margin-top:2.2em}
.eyebrow{color:#4e776b;letter-spacing:.1em;font-size:13px;font-weight:700}a{color:#146e80}figure{margin:24px 0}
img{width:100%;background:white;border:1px solid #dde4e3;border-radius:5px}figcaption{font-size:15px;color:#52686c;margin-top:10px}
.scope{border-left:4px solid #c58a3a;background:#fff5df;padding:16px 20px}nav{display:flex;gap:20px;flex-wrap:wrap}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;border-bottom:1px solid #dce3e2;text-align:right}
td:first-child,td:nth-child(2),th:first-child,th:nth-child(2){text-align:left}thead{background:#e9efec}
.scroll{overflow:auto}code{font-size:13px}details{margin-top:25px}</style><main>
<p class="eyebrow">QWEN2.5-7B-INSTRUCT / PAPER FIGURE PACKAGE</p>
<h1>Which measurements track<br>student behavior?</h1>
<p>Two figures organized around the new Metric 1-8 numbering. Put definitions in <b>3.1</b> and these empirical
comparisons in a new <b>3.2: Empirical agreement with student behavior</b>.</p>
<div class="scope"><b>Important scope.</b> The 00-99 and opaque-token metric results below come from
<b>full-distribution distillation</b>, not the newly trained sampled-number SFT models. GSM8K and Python
use completion-token SFT, with different recipes. The new number-token-SFT metric job 20629102 failed;
no completed comparable suite was available at the status check on September 14, 2026 (cluster time).
Use these figures as exploratory evidence, not as token-number-SFT results.</div>
<p><a href="section_3_2.tex">Section 3.2 LaTeX</a> · <a href="section_3_1_edits.tex">Targeted 3.1 edits</a> ·
<a href="../../output/pdf/paper_metric_results/paper_metric_results_preview.pdf">Compiled manuscript preview</a> ·
<a href="README.md">Protocol and integration notes</a> · <a href="analysis.json">Audited numerical data</a></p>
<nav><a href="#agreement">Figure 1: agreement</a><a href="#signal">Figure 2: signal and transfer</a><a href="#audit">Numerical audit</a></nav>
<h2 id="agreement">1. Compare structure, not just self-transfer</h2>
<p>Dataset-vector steering (Metric 7) shows recurring agreement even after excluding matched entries.
Teacher-pair geometry (Metric 1) is also informative, but answers a different question and therefore has
its own panel. The other directed measurements depend more on the carrier setting.</p>
<figure><img src="metric_agreement.svg" alt="Grouped heatmap of teacher-pair and off-diagonal directional correlations">
<figcaption>Warm colors mean positive Pearson agreement; blue means negative agreement. These are correlations,
not p-values. Panel A compares 435 teacher-pair similarities. Panel B compares 870 directional off-diagonal
entries after centering all 900 entries. Both use the same fixed [-1, 1] color range, but different estimands.</figcaption></figure>
<p><a href="../../output/pdf/paper_metric_results/metric_agreement.pdf">Vector PDF</a> ·
<a href="metric_agreement.svg">SVG</a> · <a href="metric_agreement.png">PNG</a></p>
<h2 id="signal">2. Teacher changes are not a universal transfer score</h2>
<p>Keep three questions separate: Does a metric track cross-trait structure? Does full-carrier information
track the matching FT effect? Does concentrated hidden-state change imply stronger recovery?</p>
<figure><img src="signal_and_transfer.svg" alt="Three panels showing steering agreement, full-information diagonal correlations, and activation energy versus FT effect">
<figcaption>A: filled points use all entries, hollow points exclude the diagonal. B: each point is a within-setting
Pearson correlation over 30 traits, using full-support information rather than Metric 2's identity-information matrix.
C: each point is one setting, with median first singular-mode energy versus mean double-centered FT diagonal.
Recipes differ across carriers; no causal carrier ranking, fitted trend, or statistical significance is implied.</figcaption></figure>
<p><a href="../../output/pdf/paper_metric_results/signal_and_transfer.pdf">Vector PDF</a> ·
<a href="signal_and_transfer.svg">SVG</a> · <a href="signal_and_transfer.png">PNG</a></p>
<h2 id="audit">Audit and reporting conventions</h2>
<p>The figures are recalculated from full-precision matrices in the existing report payload, not values read off a
heatmap or rounded checkpoint display. Their orientation is answer rows by teacher/adapter columns.
The new numbering maps to old names as follows: Metric 1 = M4; 2 = M9; 3 = M3; 4 = M5; 5 = M7;
6 = M1; 7 = M2; 8 = M8. Legacy M10 is the FT outcome. Legacy M6 is not included.</p>
<p>Identity MI (Metric 2) is an unsigned teacher-versus-neutral distinguishability measurement, not a signed
probability boost. Metric 8 is a spectral diagnostic, not a missing prediction-matrix row. Python's separate
100-token + OTHER analysis is not silently substituted for unavailable full-carrier metrics.</p>
<p>Snapshot: SNAPSHOT. Verified 16 FT matrices, 112 directional matrices, 8 teacher kernels, and 30 spectra per
setting. Largest discrepancy from saved correlation summaries: ERROR.</p>
<details><summary>All directional variants: Pearson and rank-correlation sensitivity</summary>
<p>Spearman means rank correlation. Diagonal Spearman is computed over the 30 matching entries of the
double-centered metric and FT matrices. These are descriptive values; shared rows and columns are not independent replicates.</p>
<div class="scroll"><table><thead><tr><th>Setting</th><th>Measurement</th><th>Pearson all</th><th>Pearson off</th>
<th>Spearman all</th><th>Spearman off</th><th>Diagonal Spearman</th></tr></thead><tbody>TABLE</tbody></table></div></details>
<p><a href="../qwen25_comprehensive_sweep_report.html#metric-results">Original comprehensive report</a></p>
</main></html>"""
    document = document.replace("TABLE", "".join(table_rows)).replace("SNAPSHOT", analysis["source_snapshot_date"])
    document = document.replace("ERROR", f"{analysis['max_saved_statistic_discrepancy']:.3g}")
    (OUT / "index.html").write_text(document)


def main() -> None:
    """Audit data and generate both paper figures plus a local preview."""
    OUT.mkdir(parents=True, exist_ok=True)
    PDF_OUT.mkdir(parents=True, exist_ok=True)
    analysis = build_analysis(json.loads(SOURCE.read_text()), SOURCE, MANIFEST)
    (OUT / "analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
    style()
    figure_agreement(analysis)
    figure_signal(analysis)
    write_preview(analysis)
    logger.success("Built two paper figures from {} settings; maximum audit discrepancy {:.3g}",
                   len(analysis["cells"]), analysis["max_saved_statistic_discrepancy"])


if __name__ == "__main__":
    main()
