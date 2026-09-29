"""Add audited group contrasts to the compact eight-panel SL figure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from build_paper_sl_matrix_blocks import (
    FAMILIES,
    GROUP_NAMES,
    GROUP_SIZES,
    PLOT_CARRIERS,
    SELECTED,
    TITLES,
    double_center,
    heatmap,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results_for_presentation/paper_sl_matrix_blocks/manifest.json"
STATISTICS = ROOT / "results_for_presentation/section3_sweep_statistics_20260924/statistics.json"
DEFAULT_OUTPUT = ROOT / "results_for_presentation/paper_sl_matrix_blocks_quantified_v36"
FIGURE_SIZE = (5.5, 3.15)


def sha256(path: Path) -> str:
    """Hash source bytes for figure provenance."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group_contrast(matrix: np.ndarray, sizes: tuple[int, ...]) -> float:
    """Calculate pair-weighted off-diagonal contrast of an already centered matrix."""
    matrix = np.asarray(matrix, dtype=float)
    n = sum(sizes)
    if matrix.shape != (n, n) or not np.isfinite(matrix).all():
        raise ValueError("Matrix must be finite, square, and match group sizes")
    if len(sizes) < 2 or any(size < 1 for size in sizes):
        raise ValueError("At least two nonempty groups are required")
    groups = np.repeat(np.arange(len(sizes)), sizes)
    off = ~np.eye(n, dtype=bool)
    within = (groups[:, None] == groups[None, :]) & off
    between = (groups[:, None] != groups[None, :]) & off
    if not within.any() or not between.any():
        raise ValueError("Both within and between off-diagonal pairs are required")
    return float(matrix[within].mean() - matrix[between].mean())


def annotation(value: float) -> str:
    """Round the full-precision statistic once, not its table rendering."""
    return rf"$\Delta = {value:+.2f}$"


def load_verified_panels() -> list[dict[str, Any]]:
    """Verify exact panel arrays and their agreement with the saved statistical audit."""
    source = json.loads(SOURCE.read_text())
    statistics = json.loads(STATISTICS.read_text())
    indexed = {row["id"]: row for row in statistics["conditions"]}
    panels = source["selected"]
    if [cell["id"] for cell in panels] != list(SELECTED):
        raise ValueError("Panel selection differs from the paper")
    for i, cell in enumerate(panels):
        record = indexed[cell["id"]]
        if record["labels"] != cell["labels"] or record["panel"] != chr(65 + i):
            raise ValueError("Panel order or trait labels differ from the audit")
        if sha256(ROOT / cell["source"]) != cell["source_sha256"]:
            raise ValueError("Source matrix checksum changed")
        if record["source_sha256"] != cell["source_sha256"]:
            raise ValueError("Figure and statistics refer to different matrices")
        raw = np.asarray(cell["raw"], dtype=float)
        centered = np.asarray(cell["double_centered"], dtype=float)
        np.testing.assert_allclose(centered, double_center(raw), atol=1e-9, rtol=0)
        value = group_contrast(centered, GROUP_SIZES[cell["family"]])
        np.testing.assert_allclose(value, record["dc_cluster"]["contrast"], atol=1e-9, rtol=0)
        if np.max(np.abs(centered)) > cell["plot_limit"]:
            raise ValueError("A color scale would clip the matrix")
        cell["contrast_nats"] = value
        cell["annotation"] = annotation(value)
    return panels


def build_figure(panels: list[dict[str, Any]], destination: Path) -> None:
    """Keep the paper's 5.5-by-3.15-inch canvas and square heatmaps."""
    if len(panels) != 8:
        raise ValueError("Exactly eight panels are required")
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "svg.fonttype": "none", "axes.unicode_minus": True})
    width, height = FIGURE_SIZE
    fig = plt.figure(figsize=FIGURE_SIZE)
    size = 0.90
    for i, cell in enumerate(panels):
        row, column = divmod(i, 4)
        x = 0.17 + column * 1.34
        bottom = 1.80 if row == 0 else 0.40
        ax = fig.add_axes([x / width, bottom / height, size / width, size / height])
        im = heatmap(ax, cell)
        center = (x + size / 2) / width
        title = TITLES[cell["model"]] if row == 0 else PLOT_CARRIERS[cell["carrier"]]
        title_y, descriptor_y, delta_y = ((3.015, 2.88, 2.765) if row == 0
                                          else (1.615, 1.48, 1.365))
        fig.text(center, title_y / height, f"{chr(65+i)}  {title}",
                 ha="center", fontsize=7.0, weight="bold")
        objective = "distillation" if cell["objective"] == "distribution" else "SFT"
        fig.text(center, descriptor_y / height, f"{FAMILIES[cell['family']]} · {objective}",
                 ha="center", fontsize=6.1, color="#344a56")
        fig.text(center, delta_y / height, cell["annotation"], ha="center",
                 fontsize=6.5, color="#24343d")
        cax = fig.add_axes([(x + size + .040) / width, bottom / height,
                            .024 / width, size / height])
        limit = cell["plot_limit"]
        bar = fig.colorbar(im, cax=cax, ticks=[-limit, 0, limit])
        bar.ax.set_yticklabels([f"{v:g}" for v in (-limit, 0, limit)])
        bar.ax.tick_params(labelsize=5.5, length=1.4, width=.35, pad=1.1)
        bar.outline.set_linewidth(.4)
    for y, family in zip((.215, .125, .035), ("city", "animal", "color"), strict=True):
        key = "   ".join(f"{i+1} {name}" for i, name in enumerate(GROUP_NAMES[family]))
        fig.text(.035, y / height, f"{FAMILIES[family]}:  {key}", fontsize=6.0)
    destination.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(destination / f"sl_matrix_sweeping.{extension}",
                    dpi=350, facecolor="white")
    plt.close(fig)
    provenance = {
        "figure_size_inches": FIGURE_SIZE,
        "source_manifest": str(SOURCE.relative_to(ROOT)),
        "source_manifest_sha256": sha256(SOURCE),
        "statistics": str(STATISTICS.relative_to(ROOT)),
        "statistics_sha256": sha256(STATISTICS),
        "renderer_sha256": sha256(Path(__file__)),
        "definition": "Full-double-centered mean within-group minus between-group directed off-diagonal SL, in nats",
        "unchanged": "Panel selection, matrix entries, trait order, group boundaries, individual linear color scales",
        "rounding": "Two decimals from full precision; appendix table keeps three decimals",
        "panels": [{"panel": chr(65+i), "id": p["id"],
                    "contrast_nats": p["contrast_nats"], "annotation": p["annotation"]}
                   for i, p in enumerate(panels)],
    }
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    logger.success("Saved annotated figure and verified provenance to {}", destination)


def main() -> None:
    """Render only the new variant; preserve earlier figure artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build_figure(load_verified_panels(), args.output)


if __name__ == "__main__":
    main()
