"""Plot newly measured recovery and metric matrices from the metrics workflow."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metric1 import double_center, pearson


def load_results(directory: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Validate labeled arrays and recalculate complete-matrix correlations."""
    summary = json.loads((directory / "summary.json").read_text())
    with np.load(directory / "matrices.npz", allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    labels = arrays["labels"].tolist()
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Trait labels must be nonempty and unique")
    for key in ["sl", *summary["metrics"]]:
        matrix = arrays[key]
        if matrix.shape != (len(labels), len(labels)) or not np.isfinite(matrix).all():
            raise ValueError(f"Invalid labeled matrix: {key}")
    sl = double_center(arrays["sl"])
    for key, record in summary["metrics"].items():
        csv_path = directory / record["matrix"]
        if hashlib.sha256(csv_path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Metric matrix checksum mismatch: {key}")
        with csv_path.open(newline="") as source:
            rows = list(csv.reader(source))
        if rows[0][1:] != labels or [row[0] for row in rows[1:]] != labels:
            raise ValueError(f"CSV and NPZ label order mismatch: {key}")
        if not np.array_equal(np.asarray([row[1:] for row in rows[1:]], dtype=float), arrays[key]):
            raise ValueError(f"CSV and NPZ measurements differ: {key}")
        value = pearson(double_center(arrays[key]), sl)["value"]
        reported = record["all_entries"]["value"]
        if value is None or reported is None:
            if value != reported:
                raise ValueError(f"Undefined correlation mismatch: {key}")
        elif not np.isclose(value, reported, atol=1e-12, rtol=0):
            raise ValueError(f"Correlation mismatch: {key}")
    return summary, arrays


def plot_results(directory: Path, output: Path) -> list[Path]:
    """Draw centered transfer/metric matrices and their measured agreement."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary, arrays = load_results(directory)
    labels = arrays["labels"].tolist()
    keys = ["sl", *summary["metrics"]]
    rows = (len(keys) + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(16, 4.2 * rows), squeeze=False,
                             constrained_layout=True)
    for ax, key in zip(axes.flat, keys):
        centered = double_center(arrays[key])
        limit = max(float(np.max(np.abs(centered))), 1e-12)
        im = ax.imshow(centered, cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set_title("Student recovery" if key == "sl" else f"{key}: {summary['metrics'][key]['name']}", fontsize=9)
        ax.set_xlabel("Answer trait")
        ax.set_ylabel("Teacher trait")
        ticks = np.arange(0, len(labels), max(1, len(labels) // 6))
        ax.set_xticks(ticks, [labels[i] for i in ticks], rotation=60, ha="right", fontsize=7)
        ax.set_yticks(ticks, [labels[i] for i in ticks], fontsize=7)
        fig.colorbar(im, ax=ax, shrink=.75)
    for ax in list(axes.flat)[len(keys):]:
        ax.set_visible(False)
    fig.suptitle("Fresh measurements · complete-matrix double centering · independent color scales")
    output.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ("pdf", "png"):
        path = output / f"measured_matrices.{suffix}"
        fig.savefig(path, dpi=180)
        written.append(path)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    metrics = list(summary["metrics"])
    for index, key in enumerate(metrics):
        value = summary["metrics"][key]["all_entries"]["value"]
        if value is None:
            ax.text(index, .03, "undefined", ha="center", rotation=90)
        else:
            ax.bar(index, value, color="#496c97")
            ax.text(index, value + (.04 if value >= 0 else -.04), f"{value:.3f}",
                    ha="center", va="bottom" if value >= 0 else "top")
    ax.set_xticks(range(len(metrics)), metrics)
    ax.set_ylim(-1.15, 1.15)
    ax.axhline(0, color="black", linewidth=.7)
    ax.set_ylabel("Pearson correlation with student recovery")
    ax.set_title(f"All {len(labels) ** 2} entries after complete-matrix double centering")
    for suffix in ("pdf", "png"):
        path = output / f"metric_agreement.{suffix}"
        fig.savefig(path, dpi=180)
        written.append(path)
    plt.close(fig)
    return written


def main() -> None:
    """Analyze an explicitly selected completed metric run on CPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Default: <metrics-dir>/plots")
    args = parser.parse_args()
    paths = plot_results(args.metrics_dir, args.output or args.metrics_dir / "plots")
    logger.success("Wrote {} fresh-measurement plots", len(paths))


if __name__ == "__main__":
    main()
