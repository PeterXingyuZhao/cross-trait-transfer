"""Render the complete signed SL matrices behind each atlas network."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_for_presentation/sl_network_atlas"
VERSION = 1


def matrix_csv(labels: list[str], values: list[list[float]]) -> str:
    """Export exact numeric values in answer-row, teacher-column orientation."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(["scored_answer / teacher_trait", *labels])
    writer.writerows([[label, *row] for label, row in zip(labels, values)])
    return buffer.getvalue()


def render_matrices(cell: dict[str, Any]) -> dict[str, Any]:
    """Draw raw and centered values, with finite symmetric untruncated scales."""
    folder = OUT / "matrices"
    folder.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cache = folder / f"{cell['id']}.json"
    if cache.exists():
        old = json.loads(cache.read_text())
        if old["source_sha256"] == cell["source_sha256"] and old["renderer_sha256"] == digest:
            if all((OUT / plot[ext]).exists() for plot in old["plots"].values()
                   for ext in ("svg", "png", "csv")):
                return old["plots"]
    labels = cell["labels"]
    if len(labels) != 30 or len(set(labels)) != 30:
        raise ValueError("Expected an unambiguous 30-trait catalog")
    plots = {}
    for mode, key in (("raw", "raw"), ("centered", "double_centered")):
        matrix = np.asarray(cell[key], dtype=float)
        if matrix.shape != (30, 30) or not np.isfinite(matrix).all():
            raise ValueError(f"Invalid SL matrix: {cell['id']}")
        limit = float(np.max(np.abs(matrix))) or 1.0
        fig = plt.figure(figsize=(10, 10.25), dpi=135)
        ax = fig.add_axes([.19, .245, .65, .635])
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit,
                          interpolation="nearest", aspect="equal")
        ax.set_xticks(range(30), labels, rotation=90, fontsize=10)
        ax.set_yticks(range(30), labels, fontsize=10)
        ax.tick_params(length=0, pad=5)
        ax.set_xlabel("Teacher trait z (training source)", fontsize=12, labelpad=12)
        ax.set_ylabel("Scored answer z′", fontsize=12, labelpad=9)
        for spine in ax.spines.values():
            spine.set_color("#c8d2d4")
        cax = fig.add_axes([.87, .245, .023, .635])
        bar = fig.colorbar(image, cax=cax)
        bar.ax.tick_params(labelsize=10)
        bar.set_label("SL (nats)" if mode == "raw" else "Double-centered SL (nats)", fontsize=11)
        bar.outline.set_edgecolor("#c8d2d4")
        heading = "Raw SL matrix" if mode == "raw" else "Double-centered SL matrix"
        fig.text(.19, .955, heading, fontsize=19, weight="bold", color="#233842")
        fig.text(.19, .920, f"{cell['model_title']} · {cell['family'].title()} · {cell['carrier_title']}",
                 fontsize=12, color="#61757c")
        fig.text(.5, .087, f"{cell['objective_title']} · adapter scale t = {cell['scale']}",
                 ha="center", fontsize=11, color="#61757c")
        fig.text(.5, .057, "All 900 signed entries, including self-effects. No network cutoff is applied.",
                 ha="center", fontsize=10.5, color="#61757c")
        fig.text(.5, .030, "Each matrix has its own zero-centered color scale; no values are clipped.",
                 ha="center", fontsize=10.5, color="#61757c")
        record = {"limit": limit, "minimum": float(matrix.min()), "maximum": float(matrix.max()),
                  "unit": "nats", "threshold": None, "values_clipped": 0,
                  "all_900_entries": True, "mode": mode}
        for extension in ("svg", "png"):
            relative = f"matrices/{cell['id']}_{mode}.{extension}"
            fig.savefig(OUT / relative, dpi=135, facecolor="white")
            record[extension] = relative
        plt.close(fig)
        record["csv"] = f"matrices/{cell['id']}_{mode}.csv"
        (OUT / record["csv"]).write_text(matrix_csv(labels, cell[key]))
        plots[mode] = record
    cache.write_text(json.dumps({"renderer_version": VERSION, "renderer_sha256": digest,
                                 "source_sha256": cell["source_sha256"], "plots": plots}, indent=2) + "\n")
    logger.info("Rendered both full SL matrices: {}", cell["id"])
    return plots


def attach_matrices(cells: list[dict[str, Any]], workers: int = 3) -> None:
    """Add downloadable heatmaps without touching any measured value or network."""
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(render_matrices, cells))
    for cell, plots in zip(cells, results):
        cell["matrices"] = plots


def main() -> None:
    """Allow matrix assets to be prepared before the extended HTML is ready."""
    payload = json.loads((OUT / "atlas.json").read_text())
    attach_matrices(payload["cells"])
    (OUT / "matrix_index.json").write_text(json.dumps(
        {c["id"]: c["matrices"] for c in payload["cells"]}, indent=2) + "\n")


if __name__ == "__main__":
    main()
