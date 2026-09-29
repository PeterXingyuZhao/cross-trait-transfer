"""Build an offline atlas of every locally saved complete 30-trait SL endpoint.

No inference, source mutation, or imputation of missing experiments occurs.
Native and relative views share exact independently fitted PCA coordinates.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from scripts import build_cross_model_token_sft_comparison as cross
from scripts import build_qwen25_city_metric_networks as canonical
from scripts.build_qwen25_animal_number_sft_ft_network import GROUPS

ROOT = cross.ROOT
OUT = ROOT / "results_for_presentation/sl_network_atlas"
SNAPSHOT = ROOT / "results/report_snapshots/qwen25_sweep_2026-09-08"
TEMPLATE = ROOT / "scripts/report_templates/sl_network_atlas.html"
NATIVE_CUTOFF = 4.0
RELATIVE_CUTOFF = 1.5
RENDER_VERSION = 2
NODE_RADIUS = 5.0
LABEL_SIZE = 14.0
CARRIERS = {"number": "00–99 numbers", "opaque": "Opaque 100",
            "element": "Element symbols", "gsm8k": "GSM8K responses",
            "python": "Python responses"}
MODEL_TITLES = {row[0]: row[1] for row in cross.MODELS}


def json_ready(value: Any) -> Any:
    """Make scientific metadata serializable without rounding measured values."""
    return canonical.drawing.clean.json_ready(value)


def sha256(path: Path) -> str:
    """Identify the exact immutable source bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_endpoint(source: Path, labels: list[str], scale: int) -> dict[str, Any]:
    """Check catalog, completion, matrix orientation and saved DC independently."""
    answers, teachers, raw = cross.read_matrix(source)
    if set(answers) != set(labels) or set(teachers) != set(labels):
        raise ValueError(f"Mismatched trait catalog: {source}")
    raw = raw[np.ix_([answers.index(z) for z in labels], [teachers.index(z) for z in labels])]
    folder = source.parents[2]
    manifest = json.loads((folder / "manifest.json").read_text())
    analysis = json.loads((folder / "m10_analysis.json").read_text())
    config = manifest["config"]
    if not manifest.get("finished_at") or analysis["final_stage"] != "final":
        raise ValueError(f"Incomplete endpoint: {source}")
    if config.get("adapter_scale", 1) != scale:
        raise ValueError(f"Unexpected adapter scale: {source}")
    if config["answers"] != answers or config["traits"] != teachers or len(config["questions"]) != 10:
        raise ValueError(f"Manifest and matrix evaluation catalogs disagree: {source}")
    saved = analysis["double_centered_raw_matrices"]["final"]
    dc = cross.double_center(raw)
    np.testing.assert_allclose(dc, [[saved[a][t] for t in labels] for a in labels], atol=1e-9, rtol=0)
    return {"labels": labels, "raw": raw, "double_centered": dc,
            "source": str(source.relative_to(ROOT)), "source_sha256": sha256(source),
            "manifest": str((folder / "manifest.json").relative_to(ROOT)),
            "manifest_sha256": sha256(folder / "manifest.json"),
            "analysis_sha256": sha256(folder / "m10_analysis.json"),
            "finished_at": manifest["finished_at"],
            "training_steps": analysis.get("final_training_step"),
            "model_id": config["base_model_id"], "eval_dtype": config.get("torch_dtype", "unspecified"),
            "epsilon": config.get("epsilon"), "scale": scale}


def decorate(cell: dict[str, Any], *, key: str, family: str, carrier: str,
             objective: str, budget: str, model: str = "qwen25", subset: str = "primary",
             variant_label: str = "") -> dict[str, Any]:
    """Attach experiment identity, full signed PCA, and fixed semantic colors."""
    expected_model = next(row[2] for row in cross.MODELS if row[0] == model)
    if cell["model_id"] != expected_model:
        raise ValueError(f"Model identity differs for {key}")
    cell.update(id=key, name=key, model=model, model_title=MODEL_TITLES[model], family=family,
                carrier=carrier, carrier_title=CARRIERS[carrier], objective=objective,
                objective_title="Token SFT" if objective == "sft" else "Distribution distillation",
                subset=subset, budget=budget, variant_label=variant_label)
    dc = np.asarray(cell["double_centered"])
    cell["pca"] = canonical.drawing.pca_columns(dc)
    cell["rms_nats"] = canonical.off_diagonal_rms(dc)
    cell["centered_range_nats"] = [float(dc.min()), float(dc.max())]
    cell["groups"] = []
    if family == "animal":
        cell["groups"] = [{"name": n, "color": c} for n, c, _ in GROUPS]
        color_map = {z: c for _, c, labels in GROUPS for z in labels}
        cell["node_colors"] = [color_map[z] for z in cell["labels"]]
    elif family == "tree":
        cell["node_colors"] = ["#43806a"] * 30
    else:
        metadata = json.loads(canonical.drawing.clean.METADATA.read_text())
        cell["node_colors"] = canonical.drawing.loops.point_colors(family, cell["labels"], metadata)
        if family == "city":
            cell["groups"] = [{"name": n, "color": c} for n, c in zip(
                canonical.drawing.loops.REGION_NAMES, canonical.drawing.loops.REGION_COLORS)]
    return json_ready(cell)


def load_cells() -> list[dict[str, Any]]:
    """Resolve 35 primary cells and 12 distinct variants, rejecting duplicates."""
    payload = json.loads(canonical.SOURCE.read_text())
    cells, labels_by_family = [], {}
    for original in payload["cells"]:
        family, carrier = original["family"], original["carrier"]
        carrier = "number" if carrier == "00-99" else carrier
        labels_by_family[family] = original["labels"]
        source = SNAPSHOT / original["final_ft_source"]
        endpoint = read_endpoint(source, original["labels"], 1)
        np.testing.assert_array_equal(endpoint["raw"], original["matrix"])
        objective = "sft" if carrier in ("gsm8k", "python") else "distribution"
        budget = {"gsm8k": "10,000 responses · 3 epochs · LoRA α8 · 939 updates",
                  "python": "5,000 responses · 1 epoch · seed 2 · 157 updates"}.get(
                      carrier, "10,000 prompts · 3 epochs · LoRA α32 · 471 updates")
        cells.append(decorate(endpoint, key=f"qwen25_{family}_{carrier}_{objective}",
                              family=family, carrier=carrier, objective=objective, budget=budget))
    for original in cross.load_results()["cells"]:
        endpoint = read_endpoint(cross.OUT / original["source"], original["labels"], 1)
        np.testing.assert_array_equal(endpoint["raw"], original["raw"])
        cells.append(decorate(endpoint, key=f"{original['model']}_{original['domain']}_number_sft",
                              model=original["model"], family=original["domain"], carrier="number",
                              objective="sft", budget="10,000 ten-number responses · 3 epochs · 471 updates"))
    for family in ("animal", "tree", "city", "color"):
        source = SNAPSHOT / (f"qwen25_{family}30_gsm8k_response_sft_10k_3ep_alpha8_h100/"
                             "m10_lora_scale_sweep/t_6/matrices/final/raw_logprob_shift.csv")
        cells.append(decorate(read_endpoint(source, labels_by_family[family], 6),
                              key=f"qwen25_{family}_gsm8k_sft_t6", family=family, carrier="gsm8k",
                              objective="sft", budget="10,000 responses · 3 epochs · LoRA α8 · 939 updates",
                              subset="variants", variant_label="Adapter extrapolation · t = 6"))
    for scale in range(1, 7):
        source = SNAPSHOT / ("qwen25_animal30_gsm8k_response_sft_4096_1ep_h100/"
                             f"m10_lora_scale_sweep/t_{scale}/matrices/final/raw_logprob_shift.csv")
        cells.append(decorate(read_endpoint(source, labels_by_family["animal"], scale),
                              key=f"qwen25_animal_gsm8k_pilot_t{scale}", family="animal", carrier="gsm8k",
                              objective="sft", budget="4,096 responses · 1 epoch · LoRA α32 · 128 updates",
                              subset="variants", variant_label=f"Smaller-budget pilot · t = {scale}"))
    for scale in (1, 6):
        source = ROOT / (f"results/colab/qwen25_city_number_sft_t1_t6_fp32/t{scale}/"
                         "matrices/final/raw_logprob_shift.csv")
        cells.append(decorate(read_endpoint(source, labels_by_family["city"], scale),
                              key=f"qwen25_city_number_sft_fp32_t{scale}", family="city", carrier="number",
                              objective="sft", budget="10,000 ten-number responses · 3 epochs · matched FP32 evaluation",
                              subset="variants", variant_label=f"Matched FP32 reevaluation · t = {scale}"))
    if len(cells) != 47 or len({c["source_sha256"] for c in cells}) != 47:
        raise ValueError("Expected 47 distinct full-matrix endpoints")
    if sum(c["subset"] == "primary" for c in cells) != 35:
        raise ValueError("Incorrect primary coverage")
    model_order = list(MODEL_TITLES)
    cells.sort(key=lambda c: (c["subset"] != "primary", model_order.index(c["model"]),
                             ["city", "animal", "color", "tree"].index(c["family"]),
                             list(CARRIERS).index(c["carrier"]), c["objective"], c["scale"], c["id"]))
    return cells


def relative_limits(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit PCA proportionally even for tiny effects; never displace any node."""
    low, high = xy.min(0), xy.max(0)
    span = high - low
    pad = np.maximum(span * .13, span.max() * .09)
    return low - pad, high + pad


@contextmanager
def routing_style() -> Iterator[None]:
    """Scope atlas typography and proportional padding to this rendering only."""
    clean, loops = canonical.drawing.clean, canonical.drawing.loops
    previous = canonical.NODE_RADIUS, canonical.LABEL_FONTSIZE, clean.padded_limits, loops.padded_limits
    canonical.NODE_RADIUS, canonical.LABEL_FONTSIZE = NODE_RADIUS, LABEL_SIZE
    clean.padded_limits = loops.padded_limits = relative_limits
    try:
        yield
    finally:
        canonical.NODE_RADIUS, canonical.LABEL_FONTSIZE, clean.padded_limits, loops.padded_limits = previous


def display_parameters(cell: dict[str, Any], mode: str) -> tuple[np.ndarray, float, float]:
    """Map each explicitly labeled scale onto the established 4-unit renderer."""
    if mode == "native":
        factor, cutoff = 1.0, NATIVE_CUTOFF
    elif mode == "relative":
        cutoff = RELATIVE_CUTOFF * cell["rms_nats"]
        factor = NATIVE_CUTOFF / cutoff
    else:
        raise ValueError(f"Unknown display mode: {mode}")
    return np.asarray(cell["double_centered"]) * factor, cutoff, factor


def check_panel(cell: dict[str, Any], panel: dict[str, Any], mode: str) -> dict[str, Any]:
    """Verify exact direction, threshold, linewidth, coordinates and label layout."""
    display, cutoff, factor = display_parameters(cell, mode)
    native = np.asarray(cell["double_centered"])
    expected = {(int(j), int(i)) for i, j in zip(*np.where(native >= cutoff))}
    records = panel["edges"] + panel["loops"]
    observed = {(r["source_index"], r["target_index"]) for r in records}
    if len(observed) != len(records):
        raise ValueError("Duplicate plotted matrix entry")
    if any(r["source_index"] == r["target_index"] for r in panel["edges"]):
        raise ValueError("Diagonal entries must be self-loops, not cross-edges")
    if any(r["source_index"] != r["target_index"] for r in panel["loops"]):
        raise ValueError("Self-loops must be diagonal entries")
    if expected != observed:
        raise ValueError(f"Incorrect plotted entries in {cell['id']} {mode}")
    for record in panel["edges"] + panel["loops"]:
        i, j = record["target_index"], record["source_index"]
        np.testing.assert_allclose(record["weight"], display[i, j], atol=1e-12)
        np.testing.assert_allclose(record["linewidth"], .16 * display[i, j], atol=1e-12)
        record["native_nats"] = native[i, j]
        record["rms_multiple"] = native[i, j] / cell["rms_nats"]
    np.testing.assert_array_equal(panel["coordinates"], cell["pca"]["coordinates"])
    boxes = np.asarray(panel["label_boxes_points"])
    overlaps = [(i, j) for i, a in enumerate(boxes) for j, b in enumerate(boxes) if i < j
                and min(a[1], b[1]) - max(a[0], b[0]) > .1
                and min(a[3], b[3]) - max(a[2], b[2]) > .1]
    clipped = [i for i, b in enumerate(boxes) if b[0] < 0 or b[1] > 600 or b[2] < 0 or b[3] > 620]
    return {"cutoff_nats": cutoff, "cutoff_units": 4 if mode == "native" else 1.5,
            "edge_count": len(panel["edges"]), "loop_count": len(panel["loops"]),
            "label_overlap_count": len(overlaps), "label_overlaps": overlaps,
            "label_clipped_count": len(clipped),
            "linewidth_points_per_nat": .16 * factor,
            "linewidth_points_per_rms": .16 * factor * cell["rms_nats"]}


def render_cell(cell: dict[str, Any]) -> dict[str, Any]:
    """Render two audited views with identical coordinates and label positions."""
    cache_path = OUT / "data" / f"{cell['id']}.json"
    identity = sha256(Path(__file__))
    if cache_path.exists():
        saved = json.loads(cache_path.read_text())
        if (saved.get("builder_sha256") == identity and saved.get("source_sha256") == cell["source_sha256"]
                and all((OUT / saved["plots"][mode][ext]).exists()
                        for mode in ("native", "relative") for ext in ("png", "svg"))):
            return saved
    cell = copy.deepcopy(cell)
    cell.update(plots={}, rendering={}, builder_sha256=identity, render_version=RENDER_VERSION)
    label_positions = None
    width, height = 600, 620
    with routing_style():
        for mode in ("relative", "native"):
            fig = plt.figure(figsize=(width / 72, height / 72), dpi=144)
            ax = fig.add_axes([25 / width, 135 / height, 550 / width, 455 / height])
            display, cutoff, _ = display_parameters(cell, mode)
            drawing_cell = {**cell, "double_centered": display}
            route = canonical.freeze_routing(ax, drawing_cell, label_positions=label_positions)
            route["family"] = cell["family"]
            label_positions = route["label_positions"]
            ax.clear()
            panel = canonical.drawing.loops.panel(
                ax, drawing_cell, route, {}, variant="external", title="", node_radius=NODE_RADIUS,
                positive_only=True, self_effect_minimum=NATIVE_CUTOFF,
                node_colors=cell["node_colors"], label_fontsize=LABEL_SIZE)
            stats = check_panel(cell, panel, mode)
            for index, text in enumerate(ax.texts):
                text.set_gid(f"trait-label-{index}")
            loops = canonical.drawing.loops
            keys = [loops.MagnitudeKey(value=v) for v in (4, 8, 12)]
            labels = ["4 nats", "8 nats", "12 nats"] if mode == "native" else ["1.5 RMS", "3 RMS", "4.5 RMS"]
            fig.legend(keys, labels, handler_map={loops.MagnitudeKey: loops.MagnitudeHandler()},
                       loc="lower center", bbox_to_anchor=(.5, 35 / height), ncol=3,
                       fontsize=12, frameon=False, handlelength=3.5, columnspacing=2)
            if cell["groups"]:
                handles = [plt.Line2D([], [], marker="o", ls="", color=g["color"], label=g["name"],
                                      markersize=8) for g in cell["groups"]]
                fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, 76 / height),
                           ncol=3, fontsize=11, frameon=False, handletextpad=.35, columnspacing=2)
            else:
                note = "Node colors illustrate the named colors." if cell["family"] == "color" else "Tree labels; uniform color does not encode a grouping."
                fig.text(.5, 100 / height, note, ha="center", fontsize=11, color="#61757c")
            title = ("Fixed cutoff · 4 nats" if mode == "native" else
                     f"Adaptive cutoff · 1.5 RMS = {cutoff:.3g} nats")
            fig.text(.5, 606 / height, title, ha="center", va="top", fontsize=13, color="#233842")
            fig.text(.5, 22 / height, f"Positive centered SL only · {stats['edge_count']} arrows / {stats['loop_count']} loops",
                     ha="center", fontsize=11, color="#61757c")
            if stats["edge_count"] == stats["loop_count"] == 0:
                fig.text(.5, 63 / height, "All entries below cutoff — matrix is available", ha="center",
                         fontsize=11, color="#a05436")
            for extension in ("svg", "png"):
                relative = f"assets/{cell['id']}_{mode}.{extension}"
                fig.savefig(OUT / relative, dpi=144, facecolor="white")
                stats[extension] = relative
            plt.close(fig)
            cell["plots"][mode] = stats
            cell["rendering"][mode] = {"panel": panel, "routing": route}
            logger.info("{} {}: {} arrows, {} loops, {} label overlaps", cell["id"], mode,
                        stats["edge_count"], stats["loop_count"], stats["label_overlap_count"])
    cell = json_ready(cell)
    np.testing.assert_array_equal(cell["rendering"]["native"]["routing"]["label_positions"],
                                  cell["rendering"]["relative"]["routing"]["label_positions"])
    if any(p["label_overlap_count"] or p["label_clipped_count"] for p in cell["plots"].values()):
        raise ValueError(f"Text layout failed verification: {cell['id']}")
    cache_path.write_text(json.dumps(cell, indent=2) + "\n")
    return cell


def write_report(cells: list[dict[str, Any]]) -> None:
    """Publish networks, signed matrices, saved metrics and both offline formats."""
    from scripts.extend_sl_network_atlas import write_extended_report

    write_extended_report(cells)


def main() -> None:
    """Render in independent processes; allow focused visual QA before full build."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    for folder in (OUT, OUT / "assets", OUT / "data"):
        folder.mkdir(parents=True, exist_ok=True)
    cells = load_cells()
    selected = cells if not args.only else [c for c in cells if c["id"] in args.only]
    if args.workers == 1:
        rendered = [render_cell(c) for c in selected]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rendered = list(pool.map(render_cell, selected))
    if not args.only:
        write_report(rendered)


if __name__ == "__main__":
    main()
