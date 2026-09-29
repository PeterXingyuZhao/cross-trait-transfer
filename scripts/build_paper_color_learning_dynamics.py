"""Build Section 4 color heatmaps and spectral-rank plots from saved results."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT / "results/logits_finetuned_color_students_10k_3ep/checkpoint_probability_ratio_matrices"
REVIEW = ROOT / "results_for_presentation/training_dynamics_review"
OUT = ROOT / "results_for_presentation/paper_learning_dynamics"
PDF_OUT = ROOT / "output/pdf/paper_learning_dynamics"
STEM = "color_learning_dynamics"
HEATMAP_STEPS = (43, 128, 171, 471)
INK = "#23383e"
BLUE = "#3b70ad"
ORANGE = "#bd741b"


def double_center(matrix: Any) -> np.ndarray:
    """Remove row and column means while retaining their interaction."""
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("Expected a finite matrix")
    return array - array.mean(0) - array.mean(1)[:, None] + array.mean()


def spectral_statistics(matrix: Any) -> dict[str, float]:
    """Compute entropy rank using squared singular values, rejecting zero."""
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("Expected a finite matrix")
    singular = np.linalg.svd(array, compute_uv=False)
    if not len(singular) or singular[0] == 0:
        raise ValueError("Effective rank is undefined for the zero matrix")
    energy = (singular / singular[0]) ** 2
    energy /= energy.sum()
    nonzero = energy[energy > 0]
    return {
        "effective_rank": float(np.exp(-np.sum(nonzero * np.log(nonzero)))),
        "first_mode_energy": float(energy[0]),
    }


def read_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    """Read a labeled square CSV, validating identical row/column ordering."""
    with path.open(newline="") as handle:
        records = list(csv.reader(handle))
    labels = records[0][1:]
    if labels != [row[0] for row in records[1:]]:
        raise ValueError(f"Mismatched row/column order: {path}")
    matrix = np.asarray([row[1:] for row in records[1:]], dtype=float)
    if matrix.shape != (len(labels), len(labels)) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid matrix: {path}")
    return labels, matrix


def prepare_data() -> dict[str, Any]:
    """Recompute rank and diagonal energy from each saved checkpoint matrix."""
    sources: dict[str, str] = {}

    def record(path: Path) -> None:
        """Record a repository-relative input and its content hash."""
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()

    historical = []
    for step in (80, 160, 240, 320, 400, 469):
        stage = f"step_{step:06d}" if step != 469 else "final"
        path = HISTORICAL / f"favorite_color_probability_ratio_matrix_{stage}_raw_logprob_shift.csv"
        record(path)
        labels, matrix = read_matrix(path)
        dc = double_center(matrix)
        historical.append({
            "step": step, "labels": labels, "matrix": matrix.tolist(),
            "uncentered": spectral_statistics(matrix),
            "double_centered": spectral_statistics(dc),
            "dc_diagonal_energy": float(np.sum(np.diag(dc) ** 2) / np.sum(dc ** 2)),
            "blue_negative_count": int(np.sum(matrix[labels.index("blue")] < 0)),
        })
    base_path = HISTORICAL / "base_favorite_color_scores.json"
    record(base_path)
    base = json.loads(base_path.read_text())
    sweep: dict[str, Any] = {}
    heatmaps = []
    display_labels = historical[0]["labels"]
    for family in ("color", "city"):
        path = REVIEW / f"qwen25_{family}30_00_99_rank_evolution.data.json"
        record(path)
        saved = json.loads(path.read_text())
        source_parts = Path(saved["matrix_source"]).parts
        matrix_root = ROOT.joinpath(*source_parts[source_parts.index("results"):])
        manifest_path = matrix_root.parent / "manifest.json"
        record(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        trajectory = []
        for row in saved["trajectory"]:
            matrix_path = matrix_root / row["stage"] / "raw_logprob_shift.csv"
            record(matrix_path)
            source_labels, matrix = read_matrix(matrix_path)
            if (
                family == "color"
                and row["view"] == "uncentered_raw"
                and row["step"] in HEATMAP_STEPS
            ):
                if (
                    len(source_labels) != 30
                    or source_labels != manifest["config"]["answers"]
                ):
                    raise ValueError("Expected the 30-color matrix in manifest order")
                indices = [source_labels.index(label) for label in display_labels]
                subset = matrix[np.ix_(indices, indices)]
                centered = double_center(subset)
                heatmaps.append({
                    "step": row["step"], "stage": row["stage"],
                    "labels": display_labels, "matrix": subset.tolist(),
                    "source": str(matrix_path.relative_to(ROOT)),
                    "source_shape": list(matrix.shape),
                    "uncentered": spectral_statistics(subset),
                    "double_centered": spectral_statistics(centered),
                    "dc_diagonal_energy": float(
                        np.square(np.diag(centered)).sum() / np.square(centered).sum()
                    ),
                    "blue_negative_count": int(
                        (subset[display_labels.index("blue")] < 0).sum()
                    ),
                })
            if row["view"] == "double_centered_raw":
                matrix = double_center(matrix)
            elif row["view"] != "uncentered_raw":
                raise ValueError(f"Unknown view: {row['view']}")
            value = spectral_statistics(matrix)["effective_rank"]
            np.testing.assert_allclose(value, row["effective_rank"], atol=1e-8, rtol=0)
            trajectory.append({**row, "effective_rank": value})
        sweep[family] = {
            "trajectory": trajectory, "config": saved["config"],
            "matrix_source": str(matrix_root.relative_to(ROOT)),
            "training_output_dir": manifest["config"]["training_output_dir"],
        }
    if [row["step"] for row in heatmaps] != list(HEATMAP_STEPS):
        raise ValueError("Selected heatmap checkpoints are incomplete or out of order")
    return {
        "panels": {
            "A": "Original 18 colors selected from the Anvil 30-color raw shift matrices at steps 43, 128, 171, 471",
            "B": "Uncentered effective rank of full 30-color and 30-city matrices; the color run is the same as in A; solid lines",
        },
        "definition": "mean_q log((P_student(a|q)+1e-12)/(P_base(a|q)+1e-12)); full-vocabulary answer probabilities",
        "rank_definition": "exp(-sum_i e_i log(e_i)); e_i = sigma_i^2 / sum_j sigma_j^2",
        "heatmaps": heatmaps,
        "heatmap_centering": "Select and reorder the 18x18 raw matrix, then double-center it for the diagonal-energy statistic; displayed cells are uncentered and not renormalized",
        "historical": historical, "sweep": sweep,
        "historical_usage": "Audit archive only; none of these historical matrices or base-prior statistics are plotted",
        "base_blue_candidate_normalized_probability": base["base_scores"]["mean_probability_by_color"]["blue"],
        "provenance_sha256": sources,
    }


def clean_axis(ax: Any) -> None:
    """Apply unobtrusive axes and horizontal reading guides."""
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color("#b4bfc1")
    ax.tick_params(color="#b4bfc1", labelcolor=INK, length=3)
    ax.grid(axis="y", color="#e8edef", linewidth=.6)
    ax.set_axisbelow(True)


def build_figure(
    data: dict[str, Any],
    *,
    effect_label: str = "$F_s$ (nats)",
    output_paths: Mapping[str, Path] | None = None,
) -> None:
    """Export checkpoint heatmaps and rank with optional label/output overrides.

    ``output_paths`` maps PDF, SVG, or PNG format names to destinations.
    Omitting it preserves the original three exports; an explicit mapping
    exports only its listed formats, leaving the canonical assets untouched.
    """
    destinations = dict(output_paths) if output_paths is not None else {
        "pdf": PDF_OUT / f"{STEM}.pdf",
        "svg": OUT / f"{STEM}.svg",
        "png": OUT / f"{STEM}.png",
    }
    if unsupported := destinations.keys() - {"pdf", "svg", "png"}:
        raise ValueError(f"Unsupported figure formats: {sorted(unsupported)}")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelcolor": INK, "text.color": INK, "axes.titlecolor": INK,
        "pdf.fonttype": 42, "svg.fonttype": "none", "axes.titlesize": 10,
    })
    fig = plt.figure(figsize=(8, 5.25), facecolor="white")
    fig.text(.035, .963, "A", fontsize=12, weight="bold")
    fig.text(.065, .963, "From a shared response shift to stronger target matching", fontsize=11, weight="bold")
    labels = data["heatmaps"][0]["labels"]
    starts = [.125, .326, .527, .748]
    fig.text(.411, .927, "Early training", ha="center", fontsize=8, color="#607278")
    fig.text(.833, .927, "End of training", ha="center", fontsize=8, color="#607278")
    for index, row in enumerate(data["heatmaps"]):
        ax = fig.add_axes((starts[index], .525, .170, .315))
        matrix = np.asarray(row["matrix"])
        im = ax.imshow(matrix, vmin=-18, vmax=18, cmap="RdBu_r", interpolation="nearest", aspect="auto")
        for diagonal in range(18):
            ax.add_patch(Rectangle((diagonal-.5, diagonal-.5), 1, 1, fill=False,
                                   edgecolor="#263638", linewidth=.35, alpha=.6))
        ax.set_xticks([0, 4, 8, 12, 17], [1, 5, 9, 13, 18], fontsize=8)
        row_labels = [f"{i+1:2}  {label}" for i, label in enumerate(labels)]
        ax.set_yticks(range(18), row_labels if index == 0 else [""] * 18, fontsize=7.4)
        ax.tick_params(length=0, pad=2)
        ax.set_title(f"Step {row['step']}", fontsize=10, pad=8, weight="bold")
        if index == 0:
            ax.set_ylabel("Scored answer", fontsize=8, labelpad=5)
            ax.get_yticklabels()[labels.index("blue")].set_color(BLUE)
            ax.get_yticklabels()[labels.index("blue")].set_fontweight("bold")
        for spine in ax.spines.values():
            spine.set_linewidth(.5)
            spine.set_color("#b4bfc1")
        fig.text(starts[index]+.085, .891,
                 f"DC diagonal energy: {row['dc_diagonal_energy']:.1%}",
                 ha="center", fontsize=7.5, color="#607278")
    fig.text(.52, .463, "Teacher color index (same order as the rows)", ha="center", fontsize=8)
    cax = fig.add_axes((.946, .525, .013, .315))
    colorbar = fig.colorbar(im, cax=cax, ticks=[-18, -9, 0, 9, 18])
    colorbar.ax.tick_params(labelsize=6.5, length=2)
    colorbar.outline.set_visible(False)
    fig.text(.964, .868, effect_label, ha="center", fontsize=8)

    ax = fig.add_axes((.125, .108, .791, .265))
    clean_axis(ax)
    for step in HEATMAP_STEPS:
        ax.axvline(step, color="#c3cdcf", lw=.65, ls=(0, (2, 3)), zorder=0)
    for family, color in (("color", ORANGE), ("city", BLUE)):
        rows = [
            row for row in data["sweep"][family]["trajectory"]
            if row["view"] == "uncentered_raw"
        ]
        label = "30 colors" if family == "color" else "30 cities"
        ax.plot([r["step"] for r in rows], [r["effective_rank"] for r in rows],
                color=color, ls="-", lw=1.8, marker="o", ms=3.2, label=label)
    ax.set(xlim=(25, 490), ylim=(.7, 5.4), xticks=[43, 128, 171, 300, 385, 471],
           yticks=[1, 2, 3, 4, 5], xlabel="Optimizer step", ylabel="Effective rank")
    ax.tick_params(labelsize=8.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left",
              ncol=2, columnspacing=1.5, handlelength=2)
    fig.text(.035, .409, "B", fontsize=12, weight="bold")
    fig.text(.125, .409, "Uncentered effective rank of the full 30-trait matrices", fontsize=10, weight="bold")
    pdf_metadata = {
        "Title": "Color learning dynamics in the Anvil 30-trait experiments",
        "Subject": "Original 18 colors at steps 43, 128, 171, 471; full 30-color and 30-city effective rank",
    }
    for format_name, path in destinations.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        save_options: dict[str, Any] = {}
        if format_name == "pdf":
            save_options["metadata"] = pdf_metadata
        elif format_name == "png":
            save_options["dpi"] = 240
        fig.savefig(path, **save_options)
    plt.close(fig)


def main() -> None:
    """Audit frozen inputs and write figure assets and numerical provenance."""
    data = prepare_data()
    build_figure(data)
    (OUT / f"{STEM}.json").write_text(json.dumps(data, indent=2) + "\n")
    logger.success("Created {} and audited {} input files", STEM, len(data["provenance_sha256"]))


if __name__ == "__main__":
    main()
