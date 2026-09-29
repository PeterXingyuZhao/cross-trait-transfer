"""Render full 30-color counterparts of the four main-paper SL heatmaps."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
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

from scripts.build_paper_token_sft_learning_dynamics import (  # noqa: E402
    DISPLAY_COLORS, prepare_data as load_checkpoints,
)

STEPS = (43, 86, 171, 471)
LIMIT = 18.0
NAME = "color_matrix_dynamics_full30"
OUT = ROOT / "results_for_presentation/color_matrix_dynamics_full30"
PDF = ROOT / "output/pdf" / f"{NAME}.pdf"
PAPER_COPY = ROOT / "paper_revisions/section5_full_matrix_v26/figures/images" / PDF.name
MAIN_DATA = ROOT / "results_for_presentation/paper_figure6_token_sft_dynamics/data.json"
INK, MUTED, BLUE = "#24383F", "#667A82", "#3975A7"


def select_full_snapshots(original: dict[str, Any], main: dict[str, Any]) -> dict[str, Any]:
    """Preserve all original cells while placing the main 18 colors first."""
    color = original["families"]["color"]
    source_labels = color["labels"]
    if len(source_labels) != 30 or len(set(source_labels)) != 30:
        raise ValueError("Expected the complete distinct 30-color catalog")
    display = [*DISPLAY_COLORS, *[label for label in source_labels if label not in DISPLAY_COLORS]]
    if len(display) != 30 or set(display) != set(source_labels):
        raise ValueError("The fixed main-paper colors are not a subset of this catalog")
    order = [source_labels.index(label) for label in display]
    reference = {row["step"]: row for row in main["snapshots"]}
    checkpoints = {row["step"]: row for row in color["trajectory"]}
    if any(step not in checkpoints or step not in reference for step in STEPS):
        raise ValueError("A requested main-paper checkpoint is missing")
    snapshots = []
    for step in STEPS:
        original_matrix = np.asarray(checkpoints[step]["matrix"], dtype=float)
        if original_matrix.shape != (30, 30) or not np.isfinite(original_matrix).all():
            raise ValueError("Expected a complete finite 30-by-30 matrix")
        matrix = original_matrix[np.ix_(order, order)]
        if np.abs(matrix).max() > LIMIT:
            raise ValueError("Shared +/-18-nat scale would clip an observation")
        if reference[step]["labels"] != list(DISPLAY_COLORS):
            raise ValueError("Main-paper 18-color order differs")
        reference_matrix = np.asarray(reference[step]["matrix"], dtype=float)
        if reference_matrix.shape != (18, 18) or not np.array_equal(matrix[:18, :18], reference_matrix):
            raise ValueError("Main-paper 18-color submatrix is not exactly reproduced")
        blue = matrix[display.index("blue")]
        snapshots.append({
            "step": step, "matrix": matrix.tolist(), "source_csv": checkpoints[step]["source_csv"],
            "minimum_nats": float(matrix.min()), "maximum_nats": float(matrix.max()),
            "clipped_cell_count": int((np.abs(matrix) > LIMIT).sum()),
            "main18_exact_equality": True, "main18_max_abs_difference": 0.0,
            "blue_negative_count": int((blue < 0).sum()), "blue_mean_nats": float(blue.mean()),
        })
    if snapshots[0]["blue_negative_count"] != 29:
        raise ValueError("Step 43 must retain the observed 29/30 blue-suppression count")
    return {
        "model": original["model"], "training": original["training"],
        "definition": original["definition"], "epsilon": 1e-12,
        "baseline": "Untrained base model, not a trained control student",
        "question_count": 10, "evaluation_dtype": color["evaluation_config"]["torch_dtype"],
        "cell_values": "Uncentered mean complete-answer log-probability gains; no answer-list or answer-length normalization",
        "orientation": "Rows are scored answers; columns are fine-tuning teacher traits",
        "steps": list(STEPS), "source_labels": source_labels, "display_labels": display,
        "display_order_rule": "Main-paper 18 colors first; remaining 12 retain their relative order in the original 30-color catalog",
        "display_to_source_zero_based": order,
        "display_mapping": [{"display_index": i + 1, "label": label,
                             "original_index": source_labels.index(label) + 1}
                            for i, label in enumerate(display)],
        "main18_labels": list(DISPLAY_COLORS), "heatmap_scale_nats": [-LIMIT, LIMIT],
        "snapshots": snapshots, "source_sha256": dict(original["source_sha256"]),
    }


def prepare_data() -> dict[str, Any]:
    """Reconstruct saved per-question scores and verify the main-panel counterpart."""
    original = load_checkpoints()
    payload = select_full_snapshots(original, json.loads(MAIN_DATA.read_text()))
    for path in (MAIN_DATA, Path(__file__)):
        payload["source_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return payload


def build_figure(data: dict[str, Any]) -> plt.Figure:
    """Draw four square, identically scaled matrices with every answer label."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 6,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.edgecolor": "#9DADB3", "axes.linewidth": .45,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(6.5, 6.3), facecolor="white")
    fig.text(.12, .978, "Qwen2.5-7B-Instruct  |  00-99 token SFT  |  30 colors",
             fontsize=9, weight="bold", va="top")
    width = .38
    height = width * fig.get_figwidth() / fig.get_figheight()
    positions = ((.12, .535), (.525, .535), (.12, .084), (.525, .084))
    labels = data["display_labels"]
    for index, (snapshot, (left, bottom)) in enumerate(zip(data["snapshots"], positions)):
        ax = fig.add_axes([left, bottom, width, height], label=f"heatmap_{snapshot['step']}")
        image = ax.imshow(snapshot["matrix"], cmap="RdBu_r", vmin=-LIMIT, vmax=LIMIT,
                          interpolation="nearest", aspect="equal")
        for diagonal in range(30):
            ax.add_patch(Rectangle((diagonal - .5, diagonal - .5), 1, 1,
                                   fill=False, edgecolor=INK, lw=.17, alpha=.55))
        ax.set_xticks([0, 4, 8, 12, 16, 20, 24, 29], [1, 5, 9, 13, 17, 21, 25, 30], fontsize=6)
        ax.set_yticks(range(30), [f"{i + 1:2}  {label}" for i, label in enumerate(labels)]
                      if index % 2 == 0 else [""] * 30, fontsize=5.4)
        ax.tick_params(length=0, pad=1.4)
        if index % 2 == 0:
            blue = ax.get_yticklabels()[labels.index("blue")]
            blue.set_color(BLUE)
            blue.set_fontweight("bold")
            fig.text(left - .006, bottom + height + .014, "Scored answer",
                     fontsize=5.4, ha="right", va="bottom", color=MUTED)
        fig.text(left + width / 2, bottom + height + .016, f"Step {snapshot['step']}",
                 fontsize=7.2, weight="bold", ha="center", va="bottom")
    colorbar_axis = fig.add_axes([.933, .25, .009, .5], label="shared_colorbar")
    colorbar = fig.colorbar(image, cax=colorbar_axis, ticks=[-18, -9, 0, 9, 18])
    colorbar.ax.tick_params(labelsize=5.8, length=1.5, pad=1.6)
    colorbar.outline.set_visible(False)
    fig.text(.9375, .763, "SL (nats)", fontsize=6.2, ha="center", va="bottom")
    fig.text(.515, .029, "Teacher color index (same order as rows)", fontsize=7, ha="center")
    return fig


def main() -> None:
    """Export the appendix PDF and its fully sourced numerical counterpart."""
    data = prepare_data()
    OUT.mkdir(parents=True, exist_ok=True)
    PDF.parent.mkdir(parents=True, exist_ok=True)
    figure = build_figure(data)
    figure.savefig(PDF, metadata={"Title": "Full 30-color SL matrix evolution under 00-99 token SFT"})
    figure.savefig(OUT / f"{NAME}.png", dpi=300)
    plt.close(figure)
    PAPER_COPY.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PDF, PAPER_COPY)
    data["artifact_sha256"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (PDF, OUT / f"{NAME}.png", PAPER_COPY)
    }
    (OUT / "data.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    logger.success("Saved full 30-color appendix figure; exact main18 matches at every step: {}", PDF)
    logger.info("Step43 blue suppression: {}/30 teachers; no clipped values", data["snapshots"][0]["blue_negative_count"])


if __name__ == "__main__":
    main()
