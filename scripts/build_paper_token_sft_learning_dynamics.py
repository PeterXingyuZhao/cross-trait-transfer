#!/usr/bin/env python3
"""Render token-SFT counterparts of learning-dynamics panels A and B.

Historical caches are read-only. Reconstruct every matrix from per-question
scores and check the saved CSV before computing any displayed statistic.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.build_paper_color_learning_dynamics import double_center, read_matrix, spectral_statistics

OUT = ROOT / "results_for_presentation/paper_token_sft_learning_dynamics"
PDF = ROOT / "output/pdf/paper_token_sft_learning_dynamics/color_learning_dynamics_token_sft.pdf"
BACKUP = ROOT / "anvil_backups/20260914T164003Z/project/results/scaling"
STEPS = (43, 86, 171, 471)
DISPLAY_COLORS = ("red", "blue", "yellow", "green", "orange", "purple", "pink", "black", "white",
                  "brown", "teal", "amber", "indigo", "ochre", "mauve", "cerulean", "saffron", "vermilion")
INK, MUTED, BLUE, ORANGE = "#23383e", "#607278", "#3b70ad", "#bd741b"
LIMIT = 18


def reconstruct_matrix(stage: dict[str, Any], labels: list[str], questions: list[str],
                       epsilon: float) -> np.ndarray:
    """Mean native complete-answer smoothed log ratios, answer rows/teacher columns."""
    def log_scores(scores: dict[str, Any]) -> np.ndarray:
        rows = scores["question_scores"]
        if [row["question"] for row in rows] != questions:
            raise ValueError("Question order mismatch")
        values = np.asarray([[row["colors"][label]["total_logprob"] for label in labels]
                             for row in rows], dtype=np.float64)
        if values.shape != (10, 30) or not np.isfinite(values).all() or np.any(values > 0):
            raise ValueError("Invalid native complete-answer log probabilities")
        return np.logaddexp(values, np.log(epsilon))
    base = log_scores(stage["base_scores"])
    return np.stack([(log_scores(stage["fine_tuned_scores"][trait]) - base).mean(axis=0)
                     for trait in labels], axis=1)


def prepare_data() -> dict[str, Any]:
    """Validate complete token-SFT runs and keep exact source values and hashes."""
    provenance: dict[str, str] = {}
    def record(path: Path) -> Any:
        payload = path.read_bytes()
        provenance[str(path.relative_to(ROOT))] = hashlib.sha256(payload).hexdigest()
        return json.loads(payload) if path.suffix == ".json" else payload
    result: dict[str, Any] = {
        "training": "00-99 sampled-number token SFT; 10,000 responses/trait; three epochs; 471 updates",
        "model": "Qwen2.5-7B-Instruct", "heatmaps": [], "families": {},
        "definition": "mean_q[log(P_student(z'|q)+epsilon)-log(P_base(z'|q)+epsilon)], epsilon=1e-12",
        "rank_definition": "exp(-sum(e*log(e))), e=sigma^2/sum(sigma^2); full30 UNcentered matrices",
        "diagonal_definition": "sum(diag(DC(M18))^2)/sum(DC(M18)^2); restrict/reorder18 first, then DC",
        "heatmap_scale": [-LIMIT, LIMIT], "heatmap_cells": "Uncentered, not candidate-normalized",
        "centering_note": "Only the diagonal-energy annotation uses double-centering; neither heatmap cells nor rank do.",
        "smoothing_note": "The saved epsilon floor is retained for an exact counterpart. It can affect very low-probability answers.",
    }
    for family in ("color", "city"):
        run = BACKUP / f"qwen25_{family}30_number_sequence_sft_10k_alpha32_3ep_h100"
        manifest = record(run / "m10/manifest.json")
        config = manifest["config"]
        if (config["base_model_id"] != "Qwen/Qwen2.5-7B-Instruct" or config["adapter_scale"] != 1
                or config["answers"] != config["traits"] or len(config["answers"]) != 30
                or config["candidate_prefix"] != "" or config["assistant_prefill"] is not None
                or config["epsilon"] != 1e-12):
            raise ValueError("Saved evaluation differs from the expected token-SFT protocol")
        labels = config["answers"]
        # Verify archived final training records, not merely a folder called final.
        collection = record(run / "consolidated_training/collection_manifest.json")
        if collection["status"] != "validated" or {row["trait"] for row in collection["records"]} != set(labels):
            raise ValueError("Incomplete training collection")
        training_records = []
        for entry in collection["records"]:
            parts = Path(entry["training_output_dir"]).parts
            relative = Path(*parts[parts.index(run.name)+1:]) / "training_manifest.json"
            if relative.parts[0] != "traits" or ".." in relative.parts:
                raise ValueError("Invalid archived training record path")
            training_path = run / relative
            training = record(training_path)
            if training["status"] != "completed" or training["actual_total_steps"] != 471:
                raise ValueError("The final matrix does not correspond to completed 471-update students")
            training_records.append(str(training_path.relative_to(ROOT)))
        trajectory = []
        for step in [*config["checkpoint_steps"], 471]:
            name = f"step_{step:06d}" if step != 471 else "final"
            cache = record(run / "m10/score_cache" / f"{name}.json")
            identity = cache["m10_score_identity"]
            for key in ("answers", "traits", "questions", "adapter_scale", "base_model_id", "candidate_prefix", "assistant_prefill"):
                if identity[key] != config[key]:
                    raise ValueError(f"Cache identity changed at {family}/{name}: {key}")
            if identity["stage"] != name:
                raise ValueError("Stage identity mismatch")
            matrix = reconstruct_matrix(cache["stage_result"], labels, config["questions"], config["epsilon"])
            csv_path = run / "m10/matrices" / name / "raw_logprob_shift.csv"
            record(csv_path)
            csv_labels, saved = read_matrix(csv_path)
            if csv_labels != labels:
                raise ValueError("CSV orientation/catalog mismatch")
            np.testing.assert_allclose(matrix, saved, atol=1e-12, rtol=0)
            centered = double_center(matrix)
            row = {"step": step, "stage": name, "matrix": matrix.tolist(),
                   **spectral_statistics(matrix),
                   "double_centered_rank": spectral_statistics(centered)["effective_rank"],
                   "source_csv": str(csv_path.relative_to(ROOT))}
            if family == "color":
                blue = matrix[labels.index("blue")]
                row.update(blue_negative_count=int((blue < 0).sum()), blue_mean_nats=float(blue.mean()))
                if step in STEPS:
                    selected = [labels.index(label) for label in DISPLAY_COLORS]
                    subset = matrix[np.ix_(selected, selected)]
                    dc = double_center(subset)
                    if np.max(np.abs(subset)) > LIMIT:
                        raise ValueError("Original ±18-nat scale would clip plotted observations")
                    result["heatmaps"].append({"step": step, "labels": list(DISPLAY_COLORS),
                        "matrix": subset.tolist(), "dc_diagonal_energy": float(np.square(np.diag(dc)).sum() / np.square(dc).sum()),
                        "blue_negative_count": int((subset[DISPLAY_COLORS.index("blue")] < 0).sum())})
            trajectory.append(row)
        result["families"][family] = {"labels": labels, "trajectory": trajectory,
            "evaluation_config": config, "training_record_paths": training_records,
            "run": str(run.relative_to(ROOT))}
    if [row["step"] for row in result["heatmaps"]] != list(STEPS):
        raise ValueError("The four matched color checkpoints are missing")
    record(Path(__file__))
    record(ROOT / "scripts/build_paper_color_learning_dynamics.py")
    result["source_sha256"] = provenance
    return result


def build_figure(data: dict[str, Any]) -> plt.Figure:
    """Preserve panel A's selection/scale and panel B's raw full-matrix definition."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.edgecolor": "#a6b3b5", "axes.linewidth": .6,
        "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(8, 4.7), facecolor="white")
    fig.text(.05, .962, "Qwen2.5-7B-Instruct | 00-99 token SFT", fontsize=10, weight="bold")
    fig.text(.035, .893, "A", fontsize=11, weight="bold")
    fig.text(.075, .893, "Color effects during training", fontsize=10, weight="bold")
    starts, width = [.132, .333, .534, .735], .172
    labels = data["heatmaps"][0]["labels"]
    for index, row in enumerate(data["heatmaps"]):
        ax = fig.add_axes([starts[index], .458, width, .338])
        image = ax.imshow(row["matrix"], cmap="RdBu_r", vmin=-LIMIT, vmax=LIMIT,
                          interpolation="nearest", aspect="auto")
        for diagonal in range(len(labels)):
            ax.add_patch(Rectangle((diagonal-.5, diagonal-.5), 1, 1,
                                  fill=False, edgecolor=INK, lw=.32, alpha=.65))
        ax.set_xticks([0, 4, 8, 12, 17], [1, 5, 9, 13, 18], fontsize=7)
        ax.set_yticks(range(18), [f"{i+1:2}  {label}" for i, label in enumerate(labels)]
                      if index == 0 else [""] * 18, fontsize=6.8)
        ax.tick_params(length=0, pad=2)
        if index == 0:
            ax.set_ylabel("Scored answer", fontsize=8, labelpad=5)
            ax.get_yticklabels()[labels.index("blue")].set_color(BLUE)
            ax.get_yticklabels()[labels.index("blue")].set_fontweight("bold")
        fig.text(starts[index]+width/2, .843, f"Step {row['step']}", ha="center", fontsize=9, weight="bold")
        fig.text(starts[index]+width/2, .811, f"DC diagonal energy: {row['dc_diagonal_energy']:.1%}",
                 ha="center", fontsize=6.9, color=MUTED)
    fig.text(.52, .411, "Teacher color index (same order as rows)", ha="center", fontsize=8)
    cax = fig.add_axes([.932, .458, .012, .338])
    colorbar = fig.colorbar(image, cax=cax, ticks=[-18, -9, 0, 9, 18])
    colorbar.ax.tick_params(labelsize=6.5, length=2, pad=2)
    colorbar.outline.set_visible(False)
    fig.text(.944, .813, r"$\mathrm{SL}_s$ (nats)", fontsize=7, ha="center")

    fig.text(.035, .350, "B", fontsize=11, weight="bold")
    fig.text(.075, .350, "Full 30-trait matrices", fontsize=9.5, weight="bold")
    ax = fig.add_axes([.132, .100, .775, .203])
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e7edef", lw=.55)
    ax.set_axisbelow(True)
    for step in STEPS:
        ax.axvline(step, color="#c3cdcf", lw=.6, ls=(0, (2, 3)), zorder=0)
    rows = data["families"]["color"]["trajectory"]
    ax.plot([row["step"] for row in rows], [row["effective_rank"] for row in rows],
            color=ORANGE, lw=1.7, marker="o", ms=3, label="30 colors")
    ax.set(xlim=(25, 490), ylim=(.7, 5.4), xticks=[43, 86, 171, 300, 385, 471],
           yticks=[1, 2, 3, 4, 5], xlabel="Optimizer step", ylabel="Effective rank")
    ax.tick_params(length=3, labelsize=7.5)
    ax.legend(frameon=False, ncol=1, loc="upper left", fontsize=7.5,
              handlelength=1.9, columnspacing=1.4, borderaxespad=.3)
    return fig


CAPTION = r"""\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{figures/images/color_learning_dynamics_token_sft.pdf}
\caption{\textbf{Learning dynamics under 00--99 token SFT.}
Qwen2.5-7B-Instruct students are trained on 10,000 sampled ten-number
responses per teacher trait for three epochs (471 updates).
\textbf{A}: Uncentered SL effects for the same 18-color subset of the
30-color experiment at steps 43, 86, 171, and 471. Rows are scored
answers and columns are teacher traits; the shared scale is $\pm18$ nats.
Percentages give the diagonal share of squared matrix energy after
selecting the 18 colors and then double-centering; the displayed cells
themselves are not centered.
\textbf{B}: Energy-based effective rank of the full, uncentered
30-color SL matrices at all 13 saved stages:
$r_{\mathrm{eff}}=\exp(-\sum_i e_i\log e_i)$, where
$e_i=\sigma_i^2/\sum_j\sigma_j^2$.
Both panels retain the saved evaluation convention,
$\mathrm{SL}_s(z,z')=\frac{1}{|\mathcal Q|}\sum_{q\in\mathcal Q}
\log\frac{p_{\theta+\Delta_{z,s}}(z'\mid q)+\epsilon}
{p_\theta(z'\mid q)+\epsilon}$, with $\epsilon=10^{-12}$ and ten
recovery questions, without answer-list or answer-length normalization.}
\label{fig:color-token-sft-dynamics}
\end{figure}
"""


def main() -> None:
    """Export verified data, paper-ready assets and a local explanatory report."""
    data = prepare_data()
    OUT.mkdir(parents=True, exist_ok=True)
    PDF.parent.mkdir(parents=True, exist_ok=True)
    figure = build_figure(data)
    figure.savefig(PDF, metadata={"Title": "Qwen2.5 00-99 token-SFT learning dynamics"})
    figure.savefig(OUT / "color_learning_dynamics_token_sft.png", dpi=250)
    figure.savefig(OUT / "color_learning_dynamics_token_sft.svg")
    plt.close(figure)
    (OUT / "figure.tex").write_text(CAPTION)
    (OUT / "data.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    with (OUT / "rank_trajectories.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["family", "step", "uncentered_energy_effective_rank", "double_centered_energy_effective_rank"])
        for family, payload in data["families"].items():
            for row in payload["trajectory"]:
                writer.writerow([family, row["step"], row["effective_rank"], row["double_centered_rank"]])
    color = data["families"]["color"]["trajectory"]
    table = "".join(f"<tr><td>{row['step']}</td><td>{row['dc_diagonal_energy']:.1%}</td></tr>" for row in data["heatmaps"])
    pdf_relative = "../../output/pdf/paper_token_sft_learning_dynamics/" + PDF.name
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Token-SFT dynamics: panels A/B</title>
<style>body{{font:17px system-ui;color:#23383e;max-width:1200px;margin:32px auto;padding:0 22px}}
img{{width:100%}}a{{color:#067b77}}td,th{{padding:6px 22px;text-align:left}}pre{{white-space:pre-wrap}}</style>
</head><body><h1>00-99 token-SFT counterparts: panels A and B</h1>
<p>Qwen2.5-7B-Instruct. Reconstructed from saved checkpoint scores; no new training or GPU evaluation.</p>
<p><a href="{pdf_relative}">Vector PDF</a> · <a href="color_learning_dynamics_token_sft.png">PNG</a> ·
<a href="figure.tex">Figure and caption LaTeX</a> · <a href="rank_trajectories.csv">Rank data</a> · <a href="data.json">Matrices and provenance</a></p>
<img src="color_learning_dynamics_token_sft.svg">
<h2>What changes under token SFT?</h2><p>The full 30-color effective rank rises from
{color[0]['effective_rank']:.2f} at step 43 to {color[-1]['effective_rank']:.2f} at step 471.
At step 43, the blue answer is suppressed for {color[0]['blue_negative_count']}/30 teacher colors
(mean SL {color[0]['blue_mean_nats']:.2f} nats).</p>
<table><tr><th>Step</th><th>18-color DC diagonal energy</th></tr>{table}</table>
<h2>Direct-comparison conventions</h2><p>A preserves the original color subset, ordering,
and common -18 to +18 nat color scale, with checkpoints 43, 86, 171, and 471.
Step 86 replaces the previous step 128 view to show an earlier transition.
B shows only the color token-SFT run,
including the extra epoch-end checkpoints 157 and 314. All plotted cells and ranks are uncentered;
only the diagonal-energy percentages use double-centering. Rows are affected answers; columns are trained traits.</p>
<p>Raw SL here means native complete-answer scores, not answer-list-normalized scores.
The saved probability floor epsilon=10<sup>-12</sup> is retained, and can materially affect rare answers.
These are archived evaluations with dtype setting <code>auto</code>, not new FP32 re-evaluations.
Rank is undefined for the zero effect matrix at step 0, so no rank point is invented there.</p>
<p>This is separate from the newly prepared red/blue/green/control experiment for panel C.</p>
<h2>LaTeX</h2><pre>{html.escape(CAPTION)}</pre></body></html>"""
    (OUT / "index.html").write_text(page)
    logger.success("Created {} from {} verified input artifacts", PDF, len(data["source_sha256"]))


if __name__ == "__main__":
    main()
