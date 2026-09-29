"""Compare completed 00-99 token-SFT outcomes without rerunning inference."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_for_presentation/cross_model_number_token_sft"
MODELS = (
    ("qwen25", "Qwen2.5 7B", "Qwen/Qwen2.5-7B-Instruct", ""),
    ("llama31", "Llama 3.1 8B", "meta-llama/Llama-3.1-8B-Instruct", "_bf16_r2"),
    ("olmo2", "OLMo 2 7B", "allenai/OLMo-2-1124-7B-Instruct", ""),
    ("qwen3vl", "Qwen3-VL 8B (text-only)", "Qwen/Qwen3-VL-8B-Instruct", "_bf16_r2"),
    ("mistral_v03", "Mistral 7B v0.3", "mistralai/Mistral-7B-Instruct-v0.3", "_bf16_r2"),
)
DOMAINS = ("animal", "city", "color")
CITY_MODEL_LIMITS = {"llama31": 3.0, "olmo2": 0.7, "qwen3vl": 1.6, "mistral_v03": 5.0}


def double_center(values: np.ndarray) -> np.ndarray:
    """Remove row and column means, adding back the grand mean."""
    return values - values.mean(0) - values.mean(1, keepdims=True) + values.mean()


def read_matrix(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    """Read labeled answer-row, teacher-column CSV without assuming symmetry."""
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    answers, teachers = [row[0] for row in rows[1:]], rows[0][1:]
    matrix = np.array([row[1:] for row in rows[1:]], dtype=float)
    if matrix.shape != (30, 30) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid final matrix: {path}")
    if len(set(answers)) != 30 or len(set(teachers)) != 30:
        raise ValueError(f"Nonunique labels: {path}")
    return answers, teachers, matrix


def load_results() -> dict:
    """Validate completed manifests and align every model to Qwen's label order."""
    cells = []
    for domain in DOMAINS:
        canonical = None
        questions = None
        for key, title, model_id, suffix in MODELS:
            run = f"{key}_{domain}30_number_sequence_sft_10k_alpha32_3ep_h100{suffix}"
            folder = OUT / "sources" / run / "m10"
            source = folder / "matrices/final/raw_logprob_shift.csv"
            answers, teachers, raw = read_matrix(source)
            dc_source = folder / "matrices/final/raw_logprob_shift_double_centered.csv"
            dc_answers, dc_teachers, dc = read_matrix(dc_source)
            assert answers == dc_answers and teachers == dc_teachers
            np.testing.assert_allclose(double_center(raw), dc, atol=1e-9, rtol=0)
            manifest = json.loads((folder / "manifest.json").read_text())
            analysis = json.loads((folder / "m10_analysis.json").read_text())
            config = manifest["config"]
            assert manifest["finished_at"] and analysis["final_stage"] == "final"
            assert analysis["final_training_step"] == 471
            assert config["adapter_scale"] == 1 and config["base_model_id"] == model_id
            assert config["torch_dtype"] == ("auto" if key == "qwen25" else "float32")
            assert config["candidate_prefix"] == "" and config["assistant_prefill"] is None
            assert config["enable_thinking"] is False
            assert config["answers"] == answers and config["traits"] == teachers
            assert len(config["questions"]) == 10
            if canonical is None:
                canonical, questions = teachers, config["questions"]
            assert set(canonical) == set(answers) == set(teachers)
            assert questions == config["questions"]
            order = np.ix_([answers.index(z) for z in canonical],
                           [teachers.index(z) for z in canonical])
            raw, dc = raw[order], dc[order]
            cells.append({
                "model": key, "title": title, "model_id": model_id, "domain": domain,
                "labels": canonical, "raw": raw.tolist(), "centered": dc.tolist(),
                "mean_diagonal_raw": float(np.diag(raw).mean()),
                "positive_diagonal_count": int((np.diag(raw) > 0).sum()),
                "mean_diagonal_centered": float(np.diag(dc).mean()),
                "finished_at": manifest["finished_at"], "training_steps": 471,
                "evaluation_dtype_config": config["torch_dtype"],
                "source": str(source.relative_to(OUT)),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "manifest": str((folder / "manifest.json").relative_to(OUT)),
            })
    return {"definition": "Mean complete-answer log-probability change relative to each model's own base, in nats",
            "orientation": "rows = scored answers; columns = teacher traits",
            "training": "00-99 sampled ten-number completion-only token SFT; no distribution distillation",
            "adapter_scale": 1, "cells": cells}


def draw_comparison(cells: list[dict], domain: str, mode: str) -> None:
    """Export five aligned matrices on one shared, symmetric, unclipped scale."""
    chosen = [cell for cell in cells if cell["domain"] == domain]
    limit = max(1, np.ceil(max(np.abs(cell[mode]).max() for cell in chosen)))
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 10.8))
    fig.subplots_adjust(left=.065, right=.975, bottom=.11, top=.90,
                        wspace=.37, hspace=.56)
    for ax, cell in zip(axes.flat, chosen):
        matrix = np.array(cell[mode])
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit,
                       interpolation="nearest", aspect="equal")
        ax.set_title(cell["title"], fontsize=13, weight="bold", pad=9)
        ax.set_xticks(range(30), cell["labels"], rotation=90, fontsize=6.8)
        ax.set_yticks(range(30), cell["labels"], fontsize=6.8)
        ax.tick_params(length=0, pad=3)
        ax.set_xlabel("Teacher trait", fontsize=9)
        ax.set_ylabel("Scored answer", fontsize=9)
        for spine in ax.spines.values():
            spine.set_color("#cbd3d8")
    axes[1, 2].axis("off")
    cax = axes[1, 2].inset_axes([.08, .73, .84, .055])
    fig.colorbar(im, cax=cax, orientation="horizontal").set_label("SL effect (nats)")
    axes[1, 2].text(.08, .58,
        "One color scale across all five models\nNo clipping or per-model normalization\n\n"
        "Rows: scored answers\nColumns: teacher traits\n\n"
        "30 teachers per model; 10 recovery questions\nFinal adapters at scale 1 (no extrapolation)",
        transform=axes[1, 2].transAxes, fontsize=11, va="top", linespacing=1.6)
    label = "raw SL" if mode == "raw" else "double-centered SL"
    fig.suptitle(f"{domain.capitalize()} | 00-99 token SFT | {label}",
                 x=.065, ha="left", fontsize=20, weight="bold")
    fig.text(.065, .945, "Same trait ordering and shared scale; each outcome is relative to that model's own base.",
             fontsize=11, color="#526975")
    for extension in ("png", "svg"):
        fig.savefig(OUT / f"{domain}_{mode}.{extension}", dpi=170, facecolor="white")
    plt.close(fig)


def write_report(payload: dict) -> None:
    """Build a self-contained comparison with raw/centered and domain selectors."""
    rows = []
    for cell in payload["cells"]:
        rows.append(f'<tr><td>{cell["domain"].capitalize()}</td><td>{cell["title"]}</td>'
                    f'<td>{cell["mean_diagonal_raw"]:+.3f}</td>'
                    f'<td>{cell["positive_diagonal_count"]}/30</td>'
                    f'<td>{cell["mean_diagonal_centered"]:+.3f}</td>'
                    f'<td><a href="{cell["source"]}">CSV</a> · '
                    f'<a href="{cell["manifest"]}">Manifest</a></td></tr>')
    page = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cross-model 00-99 token-SFT effects</title>
<style>body{margin:0;color:#233842;background:#f6f8f9;font:17px/1.6 system-ui}main{max-width:1550px;margin:auto;padding:30px}h1{line-height:1.15;font-size:34px}h2{font-size:24px}a{color:#087985}nav{display:flex;gap:22px;flex-wrap:wrap;align-items:center}select{padding:9px;font-size:17px;border:1px solid #b7c4c9;border-radius:6px;background:white}img{display:block;width:100%;background:white;border:1px solid #dbe3e5;margin:20px 0}table{border-collapse:collapse;width:100%;background:white}td,th{padding:8px 12px;text-align:left;border-bottom:1px solid #dbe3e5}th{font-size:14px}.note{padding:16px 20px;background:#eaf2f5;border-left:4px solid #4787a7}.scroll{overflow:auto}</style></head><body><main>
<h1>Cross-model fine-tuning effects: 00-99 token SFT</h1>
<p>Five models × three domains × thirty traits. These are final results from sampled ten-number sequence SFT, not one-number distribution distillation. All 15 final matrices were retrieved from Anvil and checked for completeness.</p>
<p class="note"><a href="city_centered_individual.html"><strong>New: city double-centered matrices with model-specific scales</strong></a> — a clearer view of Llama, OLMo, Qwen3-VL, and Mistral. The shared-scale comparisons below are preserved.</p>
<nav><label>Domain <select id="domain"><option value="animal">Animal</option><option value="city" selected>City</option><option value="color">Color</option></select></label><label>Matrix <select id="mode"><option value="raw">Raw SL effect</option><option value="centered">Double-centered SL effect</option></select></label><a id="large" href="city_raw.png" target="_blank">Open full-resolution figure</a><a id="svg" href="city_raw.svg">Vector SVG</a><a href="comparison.json">All values and source hashes</a></nav>
<a id="image-link" href="city_raw.png" target="_blank"><img id="figure" src="city_raw.png" alt="City token-SFT effect matrices for Qwen2.5, Llama3.1, OLMo2, Qwen3-VL, and Mistral"></a>
<p class="note"><strong>How to read:</strong> rows are scored answers, columns are teacher traits used to generate the student's training data. The raw value is the average log-probability change of the complete answer relative to that model's own unfine-tuned base, over ten recovery questions. Red is positive, blue negative, and white zero. Double-centering removes row and column mean effects; a negative centered value means below the additive row/column expectation, not necessarily a raw probability decrease. All models share one symmetric color scale within each domain/view, with no clipping.</p>
<h2>Models and matched protocol</h2>
<ul><li>Qwen/Qwen2.5-7B-Instruct</li><li>meta-llama/Llama-3.1-8B-Instruct</li><li>allenai/OLMo-2-1124-7B-Instruct</li><li>Qwen/Qwen3-VL-8B-Instruct (text-only; vision frozen)</li><li>mistralai/Mistral-7B-Instruct-v0.3</li></ul>
<p>Each teacher supplies 10,000 retained ten-number responses. Students use completion-only token SFT for three epochs (471 updates), LoRA rank 8 / alpha 32, learning rate 2×10<sup>−4</sup>, effective batch 64, and no teacher trait instruction. Final adapter scale is 1; city t=6 extrapolation is excluded. The same thirty labels and ten recovery questions are verified across models within each domain.</p>
<p><strong>Comparison limits:</strong> training budgets are matched, but architectures, base probabilities, native tokenizers, and generated datasets differ. Qwen2.5 used Unsloth/BF16, OLMo used the original FP32 native training, and corrected Llama, Qwen3-VL, and Mistral runs used BF16 mixed precision with FP32 master parameters. M10 evaluation manifests specify float32 for those four families, but auto for Qwen2.5; identical evaluation precision is not established by these manifests. These are descriptive single-run comparisons, not a controlled model ranking or estimates of training-seed uncertainty.</p>
<h2>Own-trait diagonal summaries</h2><p>Raw positive counts report how many of the thirty target answers increased relative to their corresponding base model. A diagonal mean is not a measure of cross-trait specificity.</p>
<div class="scroll"><table><thead><tr><th>Domain</th><th>Model</th><th>Mean raw diagonal (nats)</th><th>Positive raw diagonal</th><th>Mean centered diagonal (nats)</th><th>Sources</th></tr></thead><tbody>ROWS</tbody></table></div>
<p>Live Anvil accounting checked September 17, 2026: OLMo job 20573115, Llama 20621323, Qwen3-VL 20628259, and corrected Mistral 20726872 all COMPLETED. No training or inference was launched for this report.</p>
<script>function update(){const d=document.querySelector('#domain').value,m=document.querySelector('#mode').value,stem=d+'_'+m;document.querySelector('#figure').src=stem+'.png';document.querySelector('#figure').alt=d+' '+m+' cross-model token-SFT heatmaps';document.querySelector('#large').href=stem+'.png';document.querySelector('#image-link').href=stem+'.png';document.querySelector('#svg').href=stem+'.svg';}document.querySelectorAll('select').forEach(s=>s.addEventListener('change',update));</script></main></body></html>'''
    (OUT / "index.html").write_text(page.replace("ROWS", "".join(rows)))


def draw_city_individual_scales(cells: list[dict]) -> None:
    """Show the four non-Qwen2.5 city matrices without clipping or rescaling values."""
    chosen = [cell for cell in cells
              if cell["domain"] == "city" and cell["model"] in CITY_MODEL_LIMITS]
    if len(chosen) != 4 or len({cell["model"] for cell in chosen}) != 4:
        raise ValueError("Expected all four non-Qwen2.5 city models")
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 11.8))
    fig.subplots_adjust(left=.085, right=.95, bottom=.16, top=.885,
                        wspace=.38, hspace=.60)
    scale_rows = []
    for ax, cell in zip(axes.flat, chosen):
        values = np.asarray(cell["centered"], dtype=float)
        limit = CITY_MODEL_LIMITS[cell["model"]]
        if np.max(np.abs(values)) > limit:
            raise ValueError(f"Color limits would clip data for {cell['model']}")
        im = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit,
                       interpolation="nearest", aspect="equal")
        ax.set_title(cell["title"], fontsize=15, weight="bold", pad=11)
        ax.set_xticks(range(30), cell["labels"], rotation=90, fontsize=8)
        ax.set_yticks(range(30), cell["labels"], fontsize=8)
        ax.tick_params(length=0, pad=3)
        ax.set_xlabel("Teacher trait", fontsize=10)
        ax.set_ylabel("Scored answer", fontsize=10)
        for spine in ax.spines.values():
            spine.set_color("#cbd3d8")
        bar = fig.colorbar(im, ax=ax, fraction=.038, pad=.027,
                           ticks=np.linspace(-limit, limit, 5))
        bar.set_label("Centered SL (nats)", fontsize=10)
        bar.ax.tick_params(labelsize=9)
        scale_rows.append({"model": cell["model"], "title": cell["title"],
                           "min": float(values.min()), "max": float(values.max()),
                           "color_limit": limit, "source": cell["source"],
                           "source_sha256": cell["source_sha256"]})
    fig.suptitle("30 cities | 00-99 token SFT | Double-centered SL", x=.085,
                 ha="left", fontsize=20, weight="bold", y=.977)
    fig.text(.085, .938, "Model-specific color scales reveal structure; values remain in nats.",
             fontsize=12, color="#526975")
    fig.text(.085, .025,
             "Same city order in all panels. Each colorbar is symmetric around zero; no entries are clipped.\n"
             "Compare patterns here, not color intensity across models. Use the shared-scale view for absolute magnitudes.",
             fontsize=10.5, color="#526975", linespacing=1.5)
    for extension in ("png", "svg"):
        fig.savefig(OUT / f"city_centered_individual.{extension}", dpi=190, facecolor="white")
    plt.close(fig)
    (OUT / "city_centered_individual.scales.json").write_text(
        json.dumps({"domain": "city", "matrix": "double-centered SL",
                    "normalization": "none; model-specific display limits only",
                    "clipping": False, "models": scale_rows}, indent=2) + "\n")
    table_rows = "".join(
        f'<tr><td>{row["title"]}</td><td>±{row["color_limit"]:g}</td>'
        f'<td>{row["min"]:+.3f} to {row["max"]:+.3f}</td>'
        f'<td><a href="{row["source"].replace("raw_logprob_shift.csv", "raw_logprob_shift_double_centered.csv")}">Centered matrix CSV</a></td></tr>'
        for row in scale_rows)
    page = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>City token-SFT effects: model-specific scales</title>
<style>body{margin:0;background:#f6f8f9;color:#233842;font:17px/1.6 system-ui}main{max-width:1400px;margin:auto;padding:28px}h1{font-size:32px;line-height:1.2}a{color:#087985}nav{display:flex;gap:22px;flex-wrap:wrap}img{display:block;width:100%;background:white;margin:22px 0}table{border-collapse:collapse;background:white;width:100%}th,td{padding:9px 14px;border-bottom:1px solid #dbe3e5;text-align:left}.note{padding:14px 20px;background:#eaf2f5;border-left:4px solid #4787a7}</style></head><body><main>
<h1>City token-SFT effects with model-specific color scales</h1>
<p>Thirty city traits in Llama 3.1, OLMo 2, Qwen3-VL (text-only), and Mistral. All four matrices are double-centered, with identical city ordering and final adapter scale 1.</p>
<nav><a href="city_centered_individual.png" target="_blank">Full-resolution PNG</a><a href="city_centered_individual.svg">Vector SVG</a><a href="city_centered_individual.scales.json">Exact scales and provenance</a><a href="index.html">Original shared-scale comparison</a></nav>
<a href="city_centered_individual.png" target="_blank"><img src="city_centered_individual.png" alt="Four 30-city double-centered matrices with separate symmetric colorbars for Llama, OLMo, Qwen3-VL, and Mistral"></a>
<p class="note"><strong>Only the display ranges changed.</strong> Values are still in nats, with no per-model division, nonlinear transformation, or clipping. Each panel has a zero-centered, symmetric colorbar spanning all its entries. Equal colors across panels do not represent equal effect sizes; use the original shared-scale view to compare absolute magnitudes.</p>
<table><thead><tr><th>Model</th><th>Symmetric colorbar (nats)</th><th>Actual matrix range (nats)</th><th>Data</th></tr></thead><tbody>ROWS</tbody></table>
<p>Rows are scored answers; columns are teacher traits. Red/blue means above/below the row-and-column additive expectation after double-centering, not necessarily an increase/decrease in raw answer probability. These are the saved 00-99 ten-number token-SFT results, not distribution distillation or adapter extrapolation. No training or inference was run.</p>
</main></body></html>'''
    (OUT / "city_centered_individual.html").write_text(page.replace("ROWS", table_rows))


def main() -> None:
    """Validate all fifteen endpoints and render comparison artifacts."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    payload = load_results()
    (OUT / "comparison.json").write_text(json.dumps(payload, indent=2) + "\n")
    for domain in DOMAINS:
        for mode in ("raw", "centered"):
            draw_comparison(payload["cells"], domain, mode)
    draw_city_individual_scales(payload["cells"])
    write_report(payload)
    logger.success("Validated and rendered {} completed matrices: {}", len(payload["cells"]), OUT)


if __name__ == "__main__":
    main()
