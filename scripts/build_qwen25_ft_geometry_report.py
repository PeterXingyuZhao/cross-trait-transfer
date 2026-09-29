#!/usr/bin/env python3
"""Create scientific FT geometry figures and an offline interactive companion."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from loguru import logger
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_for_presentation/qwen25_ft_geometry"
TEMPLATES = ROOT / "scripts/report_templates"
REGION_NAMES = ["Japan", "US Northeast", "Western Europe", "Italy", "China", "India"]
REGION_COLORS = ["#b64b55", "#427cac", "#906caa", "#ce8b35", "#338b7b", "#7a8744"]
INK = "#20363e"


def read_json(path: Path) -> Any:
    """Read a local, finite JSON artifact."""
    return json.loads(path.read_text())


def haversine(coords: np.ndarray) -> np.ndarray:
    """Pairwise great-circle distances, kilometers; input is latitude/longitude."""
    lat, lon = np.radians(coords).T
    dl = lat[:, None] - lat[None, :]
    dn = lon[:, None] - lon[None, :]
    a = np.sin(dl / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dn / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def geographic_stats(cell: dict[str, Any], metadata: dict[str, Any], permutations: int = 9999) -> dict[str, Any]:
    """Descriptive locality tests with city-label, not individual-pair, permutations."""
    labels = cell["labels"]
    coords = np.array([[metadata["cities"][x]["lat"], metadata["cities"][x]["lon"]] for x in labels])
    geo = haversine(coords)
    upper = np.triu_indices(len(labels), 1)
    distances = np.asarray(cell["off_diagonal"]["distance"])
    similarity = np.asarray(cell["off_diagonal"]["cosine"])
    groups = np.arange(len(labels)) // 5
    same = groups[:, None] == groups[None, :]
    same_values = same[upper]
    geography_ranks = rankdata(geo[upper])
    effect_ranks = rankdata(distances[upper])
    geography_ranks -= geography_ranks.mean()
    effect_ranks -= effect_ranks.mean()
    denominator = np.linalg.norm(geography_ranks) * np.linalg.norm(effect_ranks)
    rho = float(geography_ranks @ effect_ranks / denominator)
    mean_in = float(similarity[upper][same_values].mean())
    mean_out = float(similarity[upper][~same_values].mean())
    ranks_matrix = np.zeros_like(geo)
    ranks_matrix[upper] = geography_ranks
    ranks_matrix += ranks_matrix.T
    rng = np.random.default_rng(20260908)
    distance_exceed = group_exceed = 0
    for _ in range(permutations):
        order = rng.permutation(len(labels))
        perm_rho = float(ranks_matrix[np.ix_(order, order)][upper] @ effect_ranks / denominator)
        perm_same = same[np.ix_(order, order)][upper]
        perm_delta = similarity[upper][perm_same].mean() - similarity[upper][~perm_same].mean()
        distance_exceed += perm_rho >= rho
        group_exceed += perm_delta >= mean_in - mean_out
    return {"spearman_distance": rho, "label_permutation_p_one_sided": (distance_exceed + 1) / (permutations + 1),
            "same_region_mean_cosine": mean_in, "different_region_mean_cosine": mean_out,
            "region_gap": mean_in - mean_out, "region_permutation_p_one_sided": (group_exceed + 1) / (permutations + 1),
            "permutations": permutations, "seed": 20260908, "same_region_pairs": int(same_values.sum()),
            "other_region_pairs": int((~same_values).sum()), "distance_km": geo.tolist(),
            "region_definition": "Six pre-existing five-city catalog blocks, used only as annotations/tests, never fitted into PCA/MDS."}


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Export presentation PNG and editable vector SVG/PDF."""
    for extension in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"{stem}.{extension}", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def point_colors(family: str, labels: list[str], metadata: dict[str, Any]) -> list[str]:
    """Apply explanatory geography categories or illustrative color-name swatches."""
    return ([REGION_COLORS[i // 5] for i in range(len(labels))] if family == "city"
            else [metadata["colors"][label]["hex"] for label in labels])


def embedding_figure(cell: dict[str, Any], family: str, metadata: dict[str, Any]) -> None:
    """Compare magnitude-sensitive PCA and off-diagonal pattern MDS."""
    labels = cell["labels"]
    colors = point_colors(family, labels, metadata)
    pca = cell["pca"]
    mds = cell["off_diagonal"]["mds"]
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 8.0))
    fig.subplots_adjust(wspace=.24, bottom=.16, top=.80)
    fig.suptitle(f"{'City' if family == 'city' else 'Color'} × 00–99: geometry of fine-tuning effects", fontsize=20, color=INK, x=.07, ha="left", y=.98)
    fig.text(.07, .925, "Each point is one trained student. Nearby students change the 30 answer scores in similar ways.", fontsize=11, color=INK)
    for ax, kind in zip(axes, ("pca", "mds")):
        config = pca if kind == "pca" else mds
        xy = np.asarray(config["coordinates"])
        if kind == "pca":
            ev = np.asarray(pca["explained_variance_ratio"])
            ax.set_title(f"PCA · full answer profiles\nTwo axes retain {ev[:2].sum():.1%} of variance", fontsize=12, color=INK, pad=16)
            ax.set_xlabel(f"PC1 ({ev[0]:.1%}); score units: nats")
            ax.set_ylabel(f"PC2 ({ev[1]:.1%}); score units: nats")
        else:
            ax.set_title(f"MDS · matching answers excluded pairwise\nDistance-rank agreement ρ = {mds['spearman_2d']:.3f}; stress = {mds['stress_2d']:.3f}", fontsize=12, color=INK, pad=16)
            ax.set_xlabel("MDS axis 1 (chord-distance units)")
            ax.set_ylabel("MDS axis 2 (chord-distance units)")
        ax.axhline(0, color="#d9e2e3", lw=.7, zorder=0)
        ax.axvline(0, color="#d9e2e3", lw=.7, zorder=0)
        ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=85, edgecolors="#374e57", linewidths=.7, zorder=3)
        ax.margins(.22)
        ax.set_aspect("equal", adjustable="datalim")
        texts = [ax.text(x, y, label, fontsize=9, color=INK, zorder=4) for (x, y), label in zip(xy, labels)]
        np.random.seed(1234)
        adjust_text(texts, x=xy[:, 0], y=xy[:, 1], ax=ax,
                    expand=(1.15, 1.3), force_text=(.5, .6), force_static=(.2, .3),
                    iter_lim=600, arrowprops={"arrowstyle": "-", "color": "#89989d", "lw": .55})
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors="#526872", labelsize=9)
    if family == "city":
        handles = [plt.Line2D([], [], linestyle="", marker="o", color=c, label=n) for c, n in zip(REGION_COLORS, REGION_NAMES)]
        fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(.5, .04), frameon=False, fontsize=10)
        footer = "Colors are geographic catalog groups, not fitted clusters. Neither coordinates nor groups enter PCA/MDS."
    else:
        footer = "Swatches illustrate color names; they are not measured color coordinates and do not enter PCA/MDS."
    fig.text(.07, .012, footer + "  Distances in 2D are approximate; axis orientation is arbitrary.", fontsize=9, color="#526872")
    save_figure(fig, f"{family}_pca_mds")


def reordered_heatmap(cell: dict[str, Any], family: str) -> None:
    """Reorder both matrix axes using unsupervised average-linkage leaf order."""
    order = cell["off_diagonal"]["order"]
    labels = [cell["labels"][i] for i in order]
    values = np.asarray(cell["double_centered"])[np.ix_(order, order)]
    limit = np.max(np.abs(values))
    fig, ax = plt.subplots(figsize=(10.5, 10))
    image = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("Teacher trait / trained student")
    ax.set_ylabel("Scored answer")
    ax.set_title(f"{family.title()} × 00–99 · data-driven matrix ordering\nAverage linkage from shared off-target profile distances; no cluster count chosen", fontsize=12, pad=18)
    fig.colorbar(image, ax=ax, fraction=.045, pad=.025, label="Double-centered log-probability shift (nats)")
    save_figure(fig, f"{family}_reordered_heatmap")


def inline_json(value: Any) -> str:
    """Protect embedded JSON from accidental HTML/script termination."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")


def main() -> None:
    """Render figures and standalone companion from validated scientific inputs."""
    analysis_path = OUT / "analysis.json"
    metadata_path = TEMPLATES / "ft_geometry_metadata.json"
    analysis = read_json(analysis_path)
    metadata = read_json(metadata_path)
    world_path = TEMPLATES / "vendor/ft_geometry_world.geojson"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none", "axes.labelcolor": INK})
    for family in ("city", "color"):
        cell = analysis["cells"][f"{family}-00-99"]
        embedding_figure(cell, family, metadata)
        reordered_heatmap(cell, family)
    geo = geographic_stats(analysis["cells"]["city-00-99"], metadata)
    payload = {"analysis": analysis, "metadata": metadata, "geography": geo,
               "region_names": REGION_NAMES, "region_colors": REGION_COLORS}
    (OUT / "visualization.data.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    template = (TEMPLATES / "qwen25_ft_geometry.html").read_text()
    replacements = {"__DATA__": inline_json(payload), "__WORLD__": inline_json(read_json(world_path)),
                    "__D3__": (TEMPLATES / "vendor/d3-7.9.0.min.js").read_text(),
                    "__MATHJAX__": (TEMPLATES / "vendor/mathjax-3.2.2-tex-svg.js").read_text()}
    for key, value in replacements.items():
        if key in ("__D3__", "__MATHJAX__"):
            value = re.sub(r"</script", r"<\\/script", value, flags=re.IGNORECASE)
        template = template.replace(key, value)
    destination = ROOT / "results_for_presentation/qwen25_city_color_ft_geometry.html"
    destination.write_text(template)
    provenance = [{"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                  for path in (analysis_path, metadata_path, world_path, Path(__file__), TEMPLATES / "qwen25_ft_geometry.html")]
    (OUT / "visualization.sources.json").write_text(json.dumps(provenance, indent=2) + "\n")
    logger.success("Built {}. Geography distance rho={:.3f}; within-region cosine gap={:.3f}", destination, geo["spearman_distance"], geo["region_gap"])


if __name__ == "__main__":
    main()
