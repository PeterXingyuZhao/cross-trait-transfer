"""Plot the main-text numerical figures from supplied measurement files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
DATA = ROOT / "data/figures"


def read(name: str) -> dict[str, Any]:
    """Read the anonymized, measured source for one figure."""
    return json.loads((DATA / name).read_text())


def figure1(output: Path) -> None:
    """Recalculate centering, RMS scales and group summaries for Figure 1B/C."""
    from scripts import build_paper_color_token_sft_main as renderer
    data = read("figure1.json")
    for cell in data["cells"]:
        matrix = np.asarray(cell["native_raw"], dtype=float)
        centered = renderer.double_center(matrix)
        np.testing.assert_allclose(centered, cell["native_centered"], atol=1e-12, rtol=0)
        cell["native_centered"] = centered.tolist()
        cell["off_diagonal_rms"] = float(np.sqrt(np.mean(centered[~np.eye(len(matrix), dtype=bool)] ** 2)))
    renderer.OUT, renderer.BUNDLE = output, DATA
    renderer.render(data["cells"], layout="row")


def figure2(output: Path) -> None:
    """Recompute full-matrix centering and within/between-group contrasts."""
    from scripts import build_sl_matrix_sweep_quantified as renderer
    panels = read("figure2.json")["panels"]
    for cell in panels:
        matrix = renderer.double_center(np.asarray(cell["raw"], dtype=float))
        np.testing.assert_allclose(matrix, cell["double_centered"], atol=1e-9, rtol=0)
        cell["double_centered"] = matrix.tolist()
        contrast = renderer.group_contrast(matrix, renderer.GROUP_SIZES[cell["family"]])
        np.testing.assert_allclose(contrast, cell["contrast_nats"], atol=1e-9, rtol=0)
        cell["annotation"] = renderer.annotation(contrast)
    renderer.SOURCE = renderer.STATISTICS = DATA / "figure2.json"
    renderer.build_figure(panels, output)


def figure3(output: Path) -> None:
    """Recompute all 72 metric correlations from complete labeled matrices."""
    from scripts.build_qwen25_nine_setting_metric_heatmap import correlations, draw
    table = read("figure3.json")
    for record in table["records"]:
        arrays = record["matrices"]
        for i in range(1, 7):
            actual = correlations(np.asarray(arrays[f"M{i}"]), np.asarray(arrays["sl"]))
            for mask, value in actual.items():
                np.testing.assert_allclose(value, record["metrics"][f"M{i}"][mask], atol=1e-12, rtol=0)
            record["metrics"][f"M{i}"] = actual
    draw(table, "all900", pdf=True, output_dir=output, pdf_dir=output, short_labels=True)
    (output / "recomputed_correlations.json").write_text(json.dumps({
        row["key"]: row["metrics"] for row in table["records"]}, indent=2) + "\n")


def validate_dynamics(data: dict[str, Any]) -> None:
    """Recompute energy effective ranks and correlations from full checkpoints."""
    from scripts.build_paper_color_learning_dynamics import spectral_statistics
    from scripts.build_paper_figure6_token_sft_dynamics import correlation_after_dc
    raw = {row["step"]: row for row in data["full_color_trajectory"]}
    for row in data["trajectory"]:
        matrix = np.asarray(raw[row["step"]]["matrix"], dtype=float)
        rank = spectral_statistics(matrix)["effective_rank"]
        np.testing.assert_allclose(rank, row["effective_rank"], atol=1e-12, rtol=0)
        row["effective_rank"] = rank
        for key, field in (("m1", "m1_updated_mean_prompt_cosine"), ("m5", "m5_steering")):
            metric = np.asarray(data["metric_matrices"][field])
            for suffix, off in (("all900", False), ("off870", True)):
                value = correlation_after_dc(metric, matrix, off_diagonal=off)
                np.testing.assert_allclose(value, row[f"{key}_{suffix}"], atol=1e-12, rtol=0)
                row[f"{key}_{suffix}"] = value
    labels = data["labels"]
    indices = [labels.index(label) for label in data["panel_a_labels"]]
    for snapshot in data["snapshots"]:
        matrix = np.asarray(raw[snapshot["step"]]["matrix"])[np.ix_(indices, indices)]
        np.testing.assert_allclose(matrix, snapshot["matrix"], atol=1e-12, rtol=0)
        snapshot["matrix"] = matrix.tolist()


def figure4(output: Path) -> None:
    """Render checkpoint differentiation and measured control-relative recovery."""
    from scripts.build_paper_figure6_token_sft_dynamics import build_figure
    import matplotlib.pyplot as plt
    data = read("figure4.json")
    validate_dynamics(data)
    fig = build_figure(data)
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(output / f"figure4_training_dynamics.{suffix}", dpi=240)
    plt.close(fig)
    (output / "recomputed_trajectory.json").write_text(json.dumps(data["trajectory"], indent=2) + "\n")


def figure5(output: Path) -> None:
    """Recompute all 100 pair gaps, quartiles and recovery sign categories."""
    from scripts.plot_paper_random_pairs_v38 import load_groups, numerical_summary, configure, save_figure
    from scripts.plot_paper_random_pairs_v41 import make_figure
    import matplotlib.pyplot as plt
    groups, checksum = load_groups(DATA / "figure5.json")
    stats = numerical_summary(groups)
    expected = read("figure5.json")["expected_summary"]
    for group in stats:
        for key in stats[group]:
            np.testing.assert_allclose(stats[group][key], expected[group][key], atol=1e-12, rtol=0)
    configure()
    fig = make_figure(groups, stats, full_width=True)
    save_figure(fig, output, "random_color_pairs_main", checksum)
    plt.close(fig)
    (output / "recomputed_pair_statistics.json").write_text(json.dumps(stats, indent=2) + "\n")


def main() -> None:
    """Recalculate plot statistics from an explicitly selected measurement bundle."""
    global DATA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=("all", "1", "2", "3", "4", "5"), default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/figures")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--saved-paper-data", action="store_true", help="Use the included paper measurements")
    inputs.add_argument("--data-dir", type=Path, help="Directory with figure1.json through figure5.json measurement inputs")
    args = parser.parse_args()
    DATA = args.data_dir.resolve() if args.data_dir else ROOT / "data/figures"
    logger.info("Plotting measurements from {}; this stage runs analysis and rendering", DATA)
    for index, fn in enumerate((figure1, figure2, figure3, figure4, figure5), 1):
        if args.figure in ("all", str(index)):
            destination = args.output / f"figure{index}"
            destination.mkdir(parents=True, exist_ok=True)
            fn(destination)
            logger.success("Plotted main-text Figure {} from supplied measurements", index)


if __name__ == "__main__":
    main()
