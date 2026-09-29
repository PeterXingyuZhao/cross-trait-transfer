"""Build Figure 6 from audited token-SFT checkpoints and fixed metric matrices."""

from __future__ import annotations

import hashlib
import html
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_paper_color_learning_dynamics import double_center, read_matrix
from scripts.color_cluster_transfer_contrast import cluster_transfer_contrast
from scripts.color_metric_group_summary import GROUP_LABELS, GROUP_NAMES
from scripts.build_paper_token_sft_learning_dynamics import (
    DISPLAY_COLORS, prepare_data as load_checkpoints,
)
from scripts.plot_red_hue_teacher_dynamics import (
    COLORS, SOURCE as RECOVERY_SOURCE, STEPS as RECOVERY_STEPS, measured_series,
)

OUT = ROOT / "results_for_presentation/paper_figure6_token_sft_dynamics"
PDF = ROOT / "output/pdf/figure6_token_sft_dynamics.pdf"
NAME = "figure6_token_sft_dynamics"
PANEL_TITLE_SIZE = 8
SNAPSHOT_STEPS = (43, 86, 171, 471)
CLUSTER = ("red", "crimson", "scarlet", "vermilion")
M1_SOURCE = ROOT / "results_for_presentation/token_sft_metric1_promptwise_comparison/original_only.json"
M5_SOURCE = ROOT / (
    "results_for_presentation/paper_token_sft_metric_results/source_snapshot_20746025/"
    "qwen25_color30_number_sequence_sft_10k_alpha32_3ep_h100/"
    "number_token_sft_metrics_v2/metric_7/part1_causal_dataset_vectors/causal_raw_logprob_lift.csv"
)
M5_MANIFEST = ROOT / (
    "results/analysis/qwen25_token_sft_metric7_layer_sweep_20260922/"
    "source_snapshot/color/metric_7/manifest.json"
)
INK, MUTED, GRID = "#24383F", "#667A82", "#E4EBEF"
M1_COLOR, M5_COLOR, RANK_COLOR = "#3975A7", "#258071", "#BB741D"


def correlation_after_dc(metric: np.ndarray, sl: np.ndarray, *, off_diagonal: bool) -> float:
    """Center both complete matrices, then select all or nonself entries."""
    a, b = np.asarray(metric, dtype=float), np.asarray(sl, dtype=float)
    if a.ndim != 2 or a.shape != b.shape or a.shape[0] != a.shape[1]:
        raise ValueError("Expected equally sized square matrices")
    a, b = double_center(a), double_center(b)
    mask = ~np.eye(len(a), dtype=bool) if off_diagonal else np.ones(a.shape, dtype=bool)
    if np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
        raise ValueError("Correlation is undefined for a constant selected vector")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def persistent_advantage_step(
    steps: Sequence[int], target: Sequence[float], competitors: Sequence[Sequence[float]],
    *, margin: float = 0.0,
) -> int | None:
    """Find the first measured step with strict advantage at every later state."""
    x, y, others = np.asarray(steps), np.asarray(target, float), np.asarray(competitors, float)
    if (x.ndim != 1 or len(x) == 0 or np.any(np.diff(x) <= 0)
            or y.shape != x.shape or others.ndim != 2 or others.shape[1:] != y.shape
            or others.shape[0] == 0 or not np.isfinite(margin)
            or not all(np.isfinite(a).all() for a in (x, y, others))):
        raise ValueError("Expected increasing measured steps and finite matched trajectories")
    above = np.all(others - y > margin, axis=0)
    persistent = np.logical_and.accumulate(above[::-1])[::-1]
    indices = np.flatnonzero(persistent)
    return int(x[indices[0]]) if len(indices) else None


def prepare_data() -> dict[str, Any]:
    """Reconstruct checkpoint SL and recompute all derived plotted statistics."""
    original = load_checkpoints()
    color = original["families"]["color"]
    city = original["families"]["city"]
    if [row["step"] for row in city["trajectory"]] != [row["step"] for row in color["trajectory"]]:
        raise ValueError("Color/city rank curves must use the same saved checkpoint steps")
    city_trajectory = [{key: row[key] for key in
                        ("step", "effective_rank", "first_mode_energy", "source_csv")}
                       for row in city["trajectory"]]
    labels = color["labels"]
    if labels[:4] != list(CLUSTER):
        raise ValueError("The full matrix's first four colors must match the cluster analysis")
    display_indices = [labels.index(label) for label in DISPLAY_COLORS]
    m1_document = json.loads(M1_SOURCE.read_text())
    cell = next(row for row in m1_document["cells"] if row["name"] == "color_original_1024")
    if cell["labels"] != labels or cell["audit"]["prompt_count"] != 1024:
        raise ValueError("Updated Metric 1 catalog/prompt count mismatch")
    m1 = np.asarray(cell["metrics"]["mean_prompt_cosine"]["matrix"], float)
    m5_labels, m5 = read_matrix(M5_SOURCE)
    m5_manifest = json.loads(M5_MANIFEST.read_text())
    if (m5_labels != labels or m5_manifest["config"]["num_dataset_prompts"] != 1024
            or m5_manifest["config"]["part1_alpha"] != 0.3):
        raise ValueError("Steering catalog/protocol mismatch")
    trajectory, snapshots = [], []
    block_mask = ~np.eye(4, dtype=bool)
    for row in color["trajectory"]:
        matrix = np.asarray(row["matrix"], float)
        derived = {"step": row["step"], "effective_rank": row["effective_rank"],
                   "first_mode_energy": row["first_mode_energy"],
                   "red_cluster_mean12_dc_nats": float(double_center(matrix)[:4, :4][block_mask].mean())}
        derived.update(cluster_transfer_contrast(matrix, labels))
        for name, metric in (("m1", m1), ("m5", m5)):
            derived[f"{name}_off870"] = correlation_after_dc(metric, matrix, off_diagonal=True)
            derived[f"{name}_all900"] = correlation_after_dc(metric, matrix, off_diagonal=False)
        trajectory.append(derived)
        if row["step"] in SNAPSHOT_STEPS:
            if np.max(np.abs(matrix)) > 18:
                raise ValueError("Heatmap scale would clip observations")
            snapshots.append({"step": row["step"], "labels": list(DISPLAY_COLORS),
                              "matrix": matrix[np.ix_(display_indices, display_indices)].tolist()})
    if [s["step"] for s in snapshots] != list(SNAPSHOT_STEPS):
        raise ValueError("Missing requested snapshot")
    analysis = json.loads(RECOVERY_SOURCE.read_text())
    series = measured_series(analysis)
    gains = {answer: series["vermilion", answer] for answer in CLUSTER}
    shared = np.all(np.array(list(gains.values())) > .25, axis=0)
    onset = persistent_advantage_step(RECOVERY_STEPS, gains["vermilion"],
                                      [gains["red"], gains["scarlet"]])
    recovery = {"teacher": "vermilion", "steps": list(RECOVERY_STEPS), "gains_nats": gains,
                "shared_recovery_step": int(np.array(RECOVERY_STEPS)[np.flatnonzero(shared)[0]]),
                "persistent_advantage_step": onset,
                "persistence_margin_nats": 0.0,
                "persistence_definition": "Both red and scarlet exceed vermilion at every remaining measured checkpoint"}
    paths = [M1_SOURCE, M5_SOURCE, M5_MANIFEST, RECOVERY_SOURCE, Path(__file__),
             ROOT / "scripts/plot_red_hue_teacher_dynamics.py",
             ROOT / "scripts/color_cluster_transfer_contrast.py",
             ROOT / "scripts/color_metric_group_summary.py"]
    provenance = dict(original["source_sha256"])
    provenance.update({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    return {"labels": labels, "panel_a_labels": list(DISPLAY_COLORS),
            "cluster": list(CLUSTER), "snapshots": snapshots,
            "hue_groups": [{"name": name, "labels": list(group)}
                           for name, group in zip(GROUP_NAMES, GROUP_LABELS)],
            "trajectory": trajectory, "recovery": recovery,
            "city_labels": city["labels"], "city_trajectory": city_trajectory,
            "metric_matrices": {"m1_updated_mean_prompt_cosine": m1.tolist(), "m5_steering": m5.tolist()},
            "protocol": {
                "model": "Qwen2.5-7B-Instruct", "training": "00-99 token SFT; 10,000 accepted responses; 3 epochs; 471 updates",
                "ab_baseline": "Untrained base model; saved auto-dtype scoring; log(P+1e-12); 10 recovery questions",
                "a_matrix": "Earlier fixed 18-color subset of full30, uncentered, answer rows and teacher columns; blue answer label highlighted",
                "b_rank": "Full 30-color uncentered token-SFT matrices; energy effective rank: exp(entropy of normalized squared singular values); 13 saved steps, no smoothing; city trajectory retained in data for audit but not displayed",
                "c_correlation": "Double-center full30 metric and SL separately, then Pearson correlation on 870 off-diagonal entries; fixed metrics versus evolving SL",
                "archived_cluster_transfer": "Not displayed in the current figure: within-cluster minus between-cluster mean transfer after full30 double-centering; six fixed descriptive hue groups, not fitted clusters; 122 within and 748 between directed nonself pairs; equal weight per pair within each mean; signed nats, no RMS normalization",
                "c_historical_red_summary": "The saved red_cluster_mean12_dc_nats field is retained for audit only and is no longer plotted",
                "metric1": "Updated definition: average of sqrt(p0)-weighted, across-teacher-centered, per-prompt log-ratio cosines; 1024 original shared prompt occurrences",
                "metric4": "Prompt-trait steering in the latest manuscript; formerly Metric5 and Metric7 steering; internal m5 keys retained for provenance; 1024 prompts; alpha=0.3",
                "d_baseline": "Same-step trained unnamed-color-template control; raw complete-answer log probabilities; pinned FP32 evaluation",
                "d_rerun": "Separate dense-checkpoint rerun from original accepted teacher datasets and recipe, using Unsloth/TRL 0.22.2; not a bitwise replay of A-C",
                "d_replicates": "One training seed (1), one shared control, mean over ten recovery questions",
                "d_scope": "Dominance concerns control-adjusted recovery gain, not absolute answer probability; persistence only at saved steps through471",
                "interpolation": "Straight connecting lines; no smoothing or missing-checkpoint interpolation",
            }, "source_sha256": provenance}


def style_axis(ax: Any) -> None:
    """Apply lightweight journal-style axes."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, lw=.5)
    ax.tick_params(length=2.5, width=.55, pad=2, labelsize=7)


def build_figure(data: dict[str, Any]) -> Any:
    """Compose a compact two-row layout; keep scientific baselines explicit."""
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.edgecolor": "#9DADB3", "axes.linewidth": .55,
        "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(6.5, 3.925), facecolor="white")
    fig.text(.045, .968, "Qwen2.5-7B-Instruct  |  00-99 token SFT", fontsize=10, weight="bold")
    fig.text(.045, .91, "A   Color-trait differentiation develops during training", fontsize=PANEL_TITLE_SIZE, weight="bold")
    fig.text(.707, .91, "B   Effective rank progresses", fontsize=PANEL_TITLE_SIZE, weight="bold")
    heatmap_width = .14
    # Figure-coordinate widths and heights have different physical scales.
    # Match them in inches so both each matrix and its cells are square.
    heatmap_height = heatmap_width * fig.get_figwidth() / fig.get_figheight()
    heatmap_bottom = .610
    heading_y = heatmap_bottom + heatmap_height + .013
    fig.text(.100, heading_y, "Student answer", ha="right", fontsize=4.9)
    for idx, row in enumerate(data["snapshots"]):
        left = .106 + .145 * idx
        ax = fig.add_axes([left, heatmap_bottom, heatmap_width, heatmap_height],
                          label=f"heatmap_{row['step']}")
        image = ax.imshow(row["matrix"], cmap="RdBu_r", vmin=-18, vmax=18,
                          interpolation="nearest", aspect="equal")
        for i in range(len(DISPLAY_COLORS)):
            ax.add_patch(Rectangle((i-.5, i-.5), 1, 1, fill=False, edgecolor=INK, lw=.17, alpha=.55))
        ax.set_xticks([0, 8, 17], [1, 9, 18], fontsize=5.3)
        ax.set_yticks(range(18), [f"{i+1:2}  {label}" for i, label in enumerate(DISPLAY_COLORS)]
                      if idx == 0 else [""] * 18, fontsize=4.4)
        ax.tick_params(length=0, pad=1.1)
        if idx == 0:
            blue_label = ax.get_yticklabels()[DISPLAY_COLORS.index("blue")]
            blue_label.set_color(M1_COLOR)
            blue_label.set_fontweight("bold")
        fig.text(left+heatmap_width/2, heading_y, f"Step {row['step']}",
                 fontsize=6.0, weight="bold", ha="center")
    cax = fig.add_axes([.691, heatmap_bottom, .0045, heatmap_height], label="heatmap_colorbar")
    cb = fig.colorbar(image, cax=cax, ticks=[-18, 0, 18])
    cb.ax.tick_params(labelsize=4.8, length=1, pad=.8)
    cb.outline.set_visible(False)
    fig.text(.694, heading_y, "SL (nats)", ha="center", fontsize=5.0)
    fig.text(.395, .564, "Teacher color index (same order as rows)", ha="center", fontsize=6.6)

    rows = data["trajectory"]
    steps = [r["step"] for r in rows]
    ax = fig.add_axes([.754, .624, .214, .224], label="effective_rank")
    style_axis(ax)
    ax.plot(steps, [r["effective_rank"] for r in rows], color=RANK_COLOR,
            lw=1.7, marker="o", ms=2.9, label="30 colors")
    ax.set(xlim=(25, 490), ylim=(.7, 5.4), yticks=[1, 2, 3, 4, 5],
           xticks=[43, 171, 300, 471], ylabel="Effective rank", xlabel="Optimizer step")
    ax.text(.03, .96, "Full 30-color matrices", transform=ax.transAxes,
            fontsize=6.5, va="top", color=MUTED)
    ax.text(.97, .15, f"{rows[-1]['effective_rank']:.2f} of 30", transform=ax.transAxes,
            color=RANK_COLOR, weight="bold", ha="right", fontsize=8)

    fig.text(.045, .502, "C   Metric correlation strengthens", fontsize=PANEL_TITLE_SIZE, weight="bold")
    fig.text(.535, .502, "D   Competitors dominate the teacher trait", fontsize=PANEL_TITLE_SIZE, weight="bold")
    # Give the two correlation curves the space previously occupied by the strip.
    ax = fig.add_axes([.092, .115, .323, .340], label="metric_agreement")
    style_axis(ax)
    for field, color, label in (("m1_off870", M1_COLOR, "M1  Geometry"), ("m5_off870", M5_COLOR, "M4  Steering")):
        values = [r[field] for r in rows]
        ax.plot(steps, values, lw=1.75, marker="o", ms=2.8, color=color)
        ax.annotate(f"{label}\n{values[-1]:.2f}", xy=(471, values[-1]), xytext=(7, 0),
                    textcoords="offset points", ha="left", va="center", color=color, fontsize=6.9,
                    annotation_clip=False, linespacing=1.4)
    ax.set(xlim=(25, 490), ylim=(0, .86), yticks=[0, .2, .4, .6, .8],
           xticks=[43, 171, 300, 471], xlabel="Optimizer step", ylabel="Off-diagonal Pearson r")

    recovery = data["recovery"]
    x = recovery["steps"]
    ax = fig.add_axes([.58, .115, .319, .317], label="vermilion_recovery")
    style_axis(ax)
    onset = recovery["persistent_advantage_step"]
    ax.axvspan(0, onset, color="#F1EAF7", alpha=.75, zorder=0)
    ax.axvline(onset, color="#9EADB3", lw=.7, ls=(0, (2, 2)))
    ax.axhline(0, color="#8DA0A8", lw=.7, ls=(0, (3, 3)))
    styles = {"red": (0, (5, 2)), "scarlet": (0, (3, 1.3)), "crimson": (0, (1, 1.5)), "vermilion": "-"}
    positions = {"red": 10.892, "scarlet": 6.205, "vermilion": .85, "crimson": -1.15}
    for answer in ("crimson", "scarlet", "red", "vermilion"):
        y = recovery["gains_nats"][answer]
        own = answer == "vermilion"
        ax.plot(x, y, color=COLORS[answer], lw=1.9 if own else 1.35, ls=styles[answer],
                marker="o", ms=1.65 if own else 1.2, mec="none", zorder=4 if own else 3)
        ax.annotate(answer.capitalize(), xy=(471, y[-1]), xytext=(486, positions[answer]),
                    color=COLORS[answer], weight="bold" if own else "normal", fontsize=6.8,
                    ha="left", va="center", annotation_clip=False,
                    arrowprops={"arrowstyle": "-", "lw": .6, "color": COLORS[answer], "shrinkA": 1, "shrinkB": 0})
    ax.set(xlim=(0, 471), ylim=(-2, 16), xticks=[0, 100, 200, 300, 471], yticks=[0, 5, 10, 15],
           xlabel="Optimizer step", ylabel="Gain vs. trained control (nats)")
    ax.text(0, 1.06, "Teacher: vermilion", transform=ax.transAxes, fontsize=7.4, weight="bold", color=COLORS["vermilion"])
    ax.text(65, 15, "Step 50: persistent advantage", fontsize=6.6, color=MUTED, ha="left", va="center")
    return fig


CAPTION = r"""\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{figures/images/figure6_token_sft_dynamics.pdf}
\caption{\textbf{Cross-trait structure develops within a low-dimensional space,
while related answers can overtake the teacher trait.}
Qwen2.5-7B-Instruct, 00--99 token SFT, 10,000 responses per teacher,
three epochs (471 updates).
\textbf{A}: The same 18-color submatrix of the 30-color experiment
at steps 43, 86, 171, and 471;
rows are answers and columns are teacher traits. Cells show uncentered,
base-relative log-probability gains on a common scale. The highlighted
blue answer label makes the early broadly suppressed row easy to locate.
\textbf{B}: Energy-based effective rank of the full uncentered
30-color token-SFT matrices,
$r_{\mathrm{eff}}=\exp(-\sum_i e_i\log e_i)$ with
$e_i=\sigma_i^2/\sum_j\sigma_j^2$. From step 43 to 471 it rises
from 1.09 to 4.74, remaining low relative to 30 dimensions.
\textbf{C}: Agreement with teacher geometry (Metric~1) and prompt-trait
steering (Metric~4) strengthens overall during training. Both full matrices are
double-centered before Pearson correlation over the 870 nonself entries;
the metric matrices are fixed, and SL varies with training step.
Metric~1 uses the updated mean of per-prompt cosines. Both metrics use
1,024 saved shared prompts; steering strength is 0.3.
\textbf{D}: A separate dense-checkpoint rerun for the vermilion teacher,
showing complete-answer log-probability gains relative to a same-step
trained unnamed-color control. All four answers initially show positive recovery gains;
red and scarlet exceed vermilion's gain at every saved checkpoint from
step 50 through 471. This is an advantage in recovery gain, not necessarily
absolute answer probability. Curves connect measured checkpoints without
smoothing; one training seed and ten recovery questions.
The SL matrices in A--C retain the archived convention $\log(p+10^{-12})$;
D uses raw log probabilities with matched FP32 target/control evaluation.}
\label{fig:color-learning-dynamics}
\end{figure}
"""


def write_report(data: dict[str, Any]) -> None:
    """Save a readable companion including exact numbers and provenance caveats."""
    table = "".join(f"<tr><td>{r['step']}</td><td>{r['effective_rank']:.3f}</td>"
                    f"<td>{r['m1_off870']:.3f}</td><td>{r['m5_off870']:.3f}</td></tr>"
                    for r in data["trajectory"])
    notes = "".join(f"<dt>{html.escape(k)}</dt><dd>{html.escape(v)}</dd>"
                    for k, v in data["protocol"].items()
                    if k not in {"archived_cluster_transfer", "c_historical_red_summary"})
    colors = ", ".join(f"{i+1} {name}" for i, name in enumerate(data["panel_a_labels"]))
    pdf_relative = os.path.relpath(PDF, OUT)
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Figure 6 | Token-SFT dynamics</title>
<style>body{{margin:30px auto;padding:0 24px;max-width:1400px;font:16px system-ui;color:{INK};line-height:1.5}}
img{{width:100%;border:1px solid #e4ebef}}nav{{display:flex;gap:22px;margin:20px 0;flex-wrap:wrap}}
a{{color:#256e82}}table{{border-collapse:collapse}}td,th{{padding:8px 20px;border-bottom:1px solid #e4ebef;text-align:right}}
dt{{font-weight:600;margin-top:12px}}dd{{margin-left:0;max-width:1050px}}pre{{white-space:pre-wrap;background:#f5f7f8;padding:20px}}</style>
<h1>Figure 6: developing cross-trait structure</h1>
<p>Qwen2.5-7B-Instruct, 00-99 token SFT. All statistics are reconstructed from saved measurements; no new training or GPU evaluation.</p>
<nav><a href="{pdf_relative}">Vector PDF</a><a href="{NAME}.png">PNG</a><a href="{NAME}.svg">Editable SVG</a>
<a href="figure.tex">LaTeX figure and caption</a><a href="data.json">Data and SHA256 provenance</a></nav>
<img src="{NAME}.svg" alt="Four panels: color matrices, effective rank, metric agreement, and vermilion recovery">
<h2>Design and interpretation</h2><p>A restores the earlier fixed 18-color subset and row labels,
with blue emphasized to locate the early general suppression effect. Its heatmaps and cells are square;
compact labels, smaller step titles, and a narrower color bar reserve space for the matrices.
There is no cluster box. B and C use the full 30-color token-SFT matrices.
B shows the single color effective-rank curve, rising from 1.09 to 4.74.
The city trajectory remains in the data archive but is not displayed.
Panel C now focuses on the two fixed-metric correlation curves, enlarged after
removing the auxiliary cluster-transfer strip. Its values and all of D are unchanged.
D focuses on the vermilion teacher as a detailed red-hue case study.
Agreement strengthens overall, not monotonically at every checkpoint.</p>
<p>By step 5, all four answers exceed +0.25 nats relative to control. Red and scarlet both remain above
vermilion from step 50 (44 measured states). The crossing is only localized between saved steps 40 and 50.
This is not evidence that competitors recover first: vermilion is initially stronger.
Final gains: red +10.89, scarlet +6.21, vermilion +0.16, crimson -0.34 nats.</p>
<h2>Exact plotted trajectory</h2><table><thead><tr><th>Step</th><th>Color effective rank</th>
<th>M1 off-diagonal r</th><th>M4 off-diagonal r</th></tr></thead><tbody>{table}</tbody></table>
<h2>Panel A color order</h2><p>{colors}</p><h2>Protocols and limitations</h2><dl>{notes}</dl>
<p>The original table's superscript 2 was a footnote, not r squared. Steering is Metric 4 in the latest manuscript (formerly Metrics 5 and 7).
The training dataset has 10,000 responses per teacher; the archived metric estimate uses 1,024 prompts, a distinct count.</p>
<h2>Copy-ready LaTeX</h2><pre>{html.escape(CAPTION)}</pre></html>"""
    (OUT / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    """Write reproducible vector, raster, data, and manuscript-caption outputs."""
    data = prepare_data()
    OUT.mkdir(parents=True, exist_ok=True)
    PDF.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure(data)
    fig.savefig(PDF, metadata={"Title": "Figure 6: cross-trait token-SFT learning dynamics"})
    fig.savefig(OUT / f"{NAME}.png", dpi=300)
    fig.savefig(OUT / f"{NAME}.svg")
    import matplotlib.pyplot as plt
    plt.close(fig)
    (OUT / "figure.tex").write_text(CAPTION, encoding="utf-8")
    (OUT / "data.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_report(data)
    logger.success("Created {} and {}", PDF, OUT)


if __name__ == "__main__":
    main()
