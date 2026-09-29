"""Assemble nine independently audited metric sets into a compact paper figure."""

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
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "results/analysis/qwen25_nine_setting_metrics_20260924"
OUT = ROOT / "results_for_presentation/qwen25_nine_setting_metrics_20260924"
PDF_OUT = ROOT / "output/pdf/qwen25_nine_setting_metrics_20260924"
REGIMES = ("token_sft_00_99", "distill_00_99", "distill_opaque100")
FAMILIES = ("color", "city", "animal")
HEADINGS = ("00–99 token SFT", "00–99 distillation", "Opaque 100 distillation")
ROWS = (
    "M1  Carrier-drift similarity",
    "M2  Entangled-carrier information",
    "M3  Drift–entangled-score similarity",
    "M4  Prompt-trait vector steering",
    "M5  Prompt/target-trait similarity",
    "M6  Target-trait/entangled-carrier similarity",
)


def sha256(path: Path) -> str:
    """Hash a complete source artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def correlations(matrix: np.ndarray, sl: np.ndarray) -> dict[str, float]:
    """Center complete matrices before selecting all or off-diagonal entries."""
    if matrix.shape != (30, 30) or sl.shape != (30, 30):
        raise ValueError("Expected complete 30-trait matrices")
    if not np.isfinite(matrix).all() or not np.isfinite(sl).all():
        raise ValueError("Nonfinite matrix")

    def center(values: np.ndarray) -> np.ndarray:
        return values - values.mean(0) - values.mean(1)[:, None] + values.mean()

    first, second = center(matrix), center(sl)
    masks = {"all900": np.ones((30, 30), dtype=bool),
             "off870": ~np.eye(30, dtype=bool)}
    result = {}
    for key, mask in masks.items():
        if first[mask].std() == 0 or second[mask].std() == 0:
            raise ValueError("Undefined correlation")
        result[key] = float(np.corrcoef(first[mask], second[mask])[0, 1])
    return result


def require_same_reference(
    labels: list[str], sl: np.ndarray, other_labels: list[str], other_sl: np.ndarray,
) -> None:
    """Reject source mixtures or transposed directional references."""
    if labels != other_labels or len(set(labels)) != 30:
        raise ValueError("Ordered trait labels disagree")
    np.testing.assert_allclose(sl, other_sl, atol=1e-10, rtol=0)


def load_npz(path: str, expected_hash: str) -> dict[str, np.ndarray]:
    """Require source integrity before loading audited matrices."""
    location = ROOT / path
    if sha256(location) != expected_hash:
        raise ValueError(f"Artifact changed: {path}")
    with np.load(location, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def assemble(
    audit_dir: Path = AUDIT, output_dir: Path = OUT,
    families: tuple[str, ...] = FAMILIES,
) -> dict:
    """Recompute every displayed correlation and unify directional orientation."""
    source_paths = {"m1": audit_dir / "metric1/summary.json",
                    "m23": audit_dir / "metric23/results.json",
                    "m4": audit_dir / "metric4/summary.json",
                    "m56": audit_dir / "metric56/summary.json"}
    source = {key: json.loads(path.read_text()) for key, path in source_paths.items()}
    lookups = {}
    for kind in ("m1", "m4", "m56"):
        lookups[kind] = {f"{v['regime']}_{v['family']}": v
                         for v in source[kind]["records"]}
    lookups["m23"] = {v["key"]: v for v in source["m23"]["cells"]}
    records = []
    for regime in REGIMES:
        for family in families:
            key = f"{regime}_{family}"
            one, two, four, five = [lookups[kind][key]
                                    for kind in ("m1", "m23", "m4", "m56")]
            a = load_npz(one["matrix_npz"], one["matrix_npz_sha256"])
            b = load_npz(two["matrices_file"], two["matrices_sha256"])
            c = load_npz(four["matrix_npz"], four["matrix_npz_sha256"])
            d = load_npz(five["matrices_path"], five["matrices_sha256"])
            labels, sl = one["labels"], a[one["sl_matrix_key"]]
            require_same_reference(labels, sl, two["labels"], b["sl"])
            require_same_reference(labels, sl, four["labels"], c["sl"])
            # The M5/M6 audit intentionally retains historical CSV orientation.
            require_same_reference(labels, sl, five["labels"], d["sl"].T)
            matrices = [a[one["metric_matrix_key"]], b["metric2"], b["metric3"],
                        c["metric4"], d["metric5"].T, d["metric6"].T]
            saved = [(one["pearson_all900"], one["pearson_off870"])]
            saved += [(two["metrics"][name]["pearson_all900"],
                       two["metrics"][name]["pearson_off870"])
                      for name in ("metric2", "metric3")]
            saved += [(four["pearson_all900"], four["pearson_off870"])]
            saved += [(five[name]["all_900"], five[name]["off_diagonal_870"])
                      for name in ("metric5", "metric6")]
            metrics = {}
            for index, (matrix, expected) in enumerate(zip(matrices, saved), 1):
                value = correlations(matrix, sl)
                np.testing.assert_allclose(list(value.values()), expected,
                                           atol=1e-10, rtol=0)
                metrics[f"M{index}"] = value
            target = output_dir / "matrices" / f"{key}.npz"
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, labels=np.asarray(labels), sl=sl,
                                **{f"M{i}": v for i, v in enumerate(matrices, 1)})
            records.append({"key": key, "regime": regime, "family": family,
                            "labels": labels, "metrics": metrics,
                            "metric1_prompt_count": one["prompt_count"],
                            "matrices_file": f"matrices/{key}.npz",
                            "matrices_sha256": sha256(target)})
    return {
        "model": "Qwen/Qwen2.5-7B-Instruct", "rows": list(ROWS),
        "correlation": "Pearson; both full 30x30 matrices double-centered before masking",
        "matrix_orientation": "teacher rows, answer columns",
        "definitions": {
            "M1": "Mean within-prompt cosine; sqrt(p0)-weighted across-teacher-centered log ratios",
            "M2": "Conditional MI, uniform binary prior, top-10 carrier identities plus OTHER",
            "M3": "Mean inner product of unnormalized tilde-lambda with uncentered logit(e); e is mean raw complete-answer recovery probability",
            "M4": "All-block/all-token prompt-trait steering, alpha=0.3; raw complete-answer logprob lift",
            "M5": "Cosine of concatenated matching per-block target-trait and prompt-trait vectors, blocks 0–26",
            "M6": "Cosine of concatenated per-block target-trait and most-entangled-carrier vectors, blocks 0–27",
        },
        "records": records,
        "audits": {str(path.relative_to(ROOT)): sha256(path)
                   for path in source_paths.values()},
        "builder_sha256": sha256(Path(__file__)),
    }


def draw(
    table: dict, mask: str, *, pdf: bool, output_dir: Path = OUT,
    pdf_dir: Path = PDF_OUT, short_labels: bool = False,
) -> None:
    """Draw small cells and tight whitespace at a fixed paper-width canvas."""
    values = np.asarray([[r["metrics"][f"M{i}"][mask]
                          for r in table["records"]] for i in range(1, 7)])
    cmap = LinearSegmentedColormap.from_list("agreement", ["#315f91", "#f8f7f4", "#b13641"])
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "svg.fonttype": "none", "font.size": 6.5})
    left, width = (.120, .862) if short_labels else (.420, .562)
    fig = plt.figure(figsize=(6.5, 1.65 if short_labels else 1.94), facecolor="white")
    axis = fig.add_axes((left, .235, width, .598))
    mesh = axis.imshow(values, aspect="auto", cmap=cmap, vmin=-1, vmax=1)
    columns = len(table["records"])
    family_count = columns // len(REGIMES)
    value_size = 6.1 if columns > 9 else 6.7
    axis.set_xticks(range(columns), [record["family"].title() for record in table["records"]],
                    fontsize=6.2 if columns > 9 else 6.8, color="#34494f")
    axis.xaxis.tick_top()
    axis.tick_params(axis="both", length=0, pad=4)
    labels = [f"Metric {index}" for index in range(1, 7)] if short_labels else ROWS
    axis.set_yticks(range(6), labels, fontsize=6.7, color="#23383e")
    axis.tick_params(axis="y", pad=6)
    for spine in axis.spines.values():
        spine.set_visible(False)
    for col in (family_count - .5, 2 * family_count - .5):
        axis.axvline(col, color="white", linewidth=3)
    for col in range(columns):
        best = int(np.argmax(values[:, col]))
        for row in range(6):
            value = values[row, col]
            label = "0.00" if abs(value) < .005 else f"{value:+.2f}"
            axis.text(col, row, label, ha="center", va="center", fontsize=value_size,
                      weight="bold" if row == best else "normal",
                      color="white" if abs(value) >= .58 else "#23383e")
    for index, heading in enumerate(HEADINGS):
        fig.text(left + width * (index + .5) / 3, .946, heading,
                 ha="center", va="center", fontsize=7, weight="bold", color="#23383e")
    description = "all 900 entries" if mask == "all900" else "870 off-diagonal entries"
    fig.text(.016, .105, f"Pearson r · {description} · double-centered",
             fontsize=6.3, color="#607278", va="center")
    color_axis = fig.add_axes((.710, .102, .255, .037))
    colorbar = fig.colorbar(mesh, cax=color_axis, orientation="horizontal", ticks=[-1, 0, 1])
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=5.8, length=2, pad=1, colors="#607278")
    stem = f"qwen25_metric_agreement_{mask}"
    fig.savefig(output_dir / f"{stem}.png", dpi=300)
    fig.savefig(output_dir / f"{stem}.svg")
    if pdf:
        pdf_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_dir / f"{stem}.pdf")
    plt.close(fig)


def main() -> None:
    """Build the compact PDF plus exact numerical and off-diagonal companions."""
    OUT.mkdir(parents=True, exist_ok=True)
    table = assemble()
    (OUT / "table.json").write_text(json.dumps(table, indent=2, allow_nan=False) + "\n")
    for mask in ("all900", "off870"):
        with (OUT / f"correlations_{mask}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["metric", *[r["key"] for r in table["records"]]])
            for index, row in enumerate(ROWS, 1):
                writer.writerow([row, *[r["metrics"][f"M{index}"][mask]
                                       for r in table["records"]]])
        draw(table, mask, pdf=mask == "all900")
    logger.success("All 54 correlations verified against matched sources; wrote {}", OUT)


if __name__ == "__main__":
    main()
