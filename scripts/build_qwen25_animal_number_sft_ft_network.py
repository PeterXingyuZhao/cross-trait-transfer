"""Render the animal token-SFT endpoint with the existing city/color style.

The measured asymmetric matrix and all PCA node positions remain unchanged.
Only labels and curved routes are adjusted for readability. Existing city/color
figures are inputs for comparison, never outputs of this builder.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from scripts import build_qwen25_number_sft_ft_network as sft

ROOT = sft.ROOT
PRESENTATION = sft.PRESENTATION
OUTPUT = PRESENTATION / "qwen25_animal_number_sft_ft_network"
REPORT = PRESENTATION / "qwen25_animal_number_sft_ft_network.html"
GROUPS = (
    ("Canids", "#b64b55", ("dog", "husky", "wolf", "fox", "coyote")),
    ("Felids", "#427cac", ("cat", "lion", "tiger", "leopard", "jaguar")),
    ("Birds", "#906caa", ("eagle", "hawk", "falcon", "owl", "raven")),
    ("Bears & relatives", "#ce8b35", ("bear", "panda", "polar bear", "raccoon", "badger")),
    ("Aquatic & coastal", "#338b7b", ("dolphin", "whale", "shark", "octopus", "penguin")),
    ("Large herbivores", "#7a8744", ("elephant", "giraffe", "zebra", "horse", "deer")),
)
VARIANTS = {"all_self": None, "positive_threshold4": sft.loops.THRESHOLD}


def load_animal() -> dict[str, Any]:
    """Load the completed animal matrix and attach explicit catalog annotations."""
    cell = sft.load_cell("animal")
    color_by_label = {label: color for _, color, labels in GROUPS for label in labels}
    if len(color_by_label) != 30 or set(color_by_label) != set(cell["labels"]):
        raise ValueError("Animal group annotations must cover the catalog exactly")
    cell["node_colors"] = [color_by_label[label] for label in cell["labels"]]
    cell["groups"] = [
        {"name": name, "color": color, "labels": list(labels)}
        for name, color, labels in GROUPS
    ]
    return cell


def figure_frame(cell: dict[str, Any]) -> tuple[plt.Figure, plt.Axes]:
    """Fit a standalone equal-aspect panel with space for the existing stroke key."""
    low, high = sft.clean.padded_limits(np.asarray(cell["pca"]["coordinates"]))
    ratio = float((high[0] - low[0]) / (high[1] - low[1]))
    panel_height = min(440.0, 550.0 / ratio)
    panel_width = panel_height * ratio
    width = max(550.0, panel_width + 36.0)
    height = panel_height + 112.0
    fig = plt.figure(figsize=(width / 72, height / 72), dpi=160)
    ax = fig.add_axes([
        (width - panel_width) / (2 * width), 80.0 / height,
        panel_width / width, panel_height / height,
    ])
    return fig, ax


def render_variant(
    cell: dict[str, Any], variant: str, routing: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Draw one display variant; reuse exactly the same PCA and path routing."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown animal network variant: {variant}")
    fig, ax = figure_frame(cell)
    if routing is None:
        routing = sft.clean.json_ready(sft.clean.create_panel(
            ax, cell, "animal", {}, node_colors=cell["node_colors"],
        ))
    if routing["coordinates"] != cell["pca"]["coordinates"]:
        raise ValueError("Routing does not match the complete-matrix PCA")
    ax.clear()
    panel = sft.loops.panel(
        ax, cell, routing, {}, variant="external", title="Animal · 00–99 token SFT",
        node_radius=sft.MAIN_NODE_RADIUS, positive_only=True,
        self_effect_minimum=VARIANTS[variant], node_colors=cell["node_colors"],
    )
    for text in ax.texts:
        if text.get_text() in cell["labels"]:
            text.set_gid(f"trait-label-animal-{cell['labels'].index(text.get_text())}")
    sft.loops.legend(
        fig, positive_only=True, direction_character=False,
        negative_self_effects=any(effect["weight"] < 0 for effect in panel["loops"]),
        group_entries=[(name, color) for name, color, _ in GROUPS], group_columns=3,
    )
    for extension in ("png", "svg", "pdf"):
        fig.savefig(
            OUTPUT / f"pca_{variant}.{extension}", dpi=260, facecolor="white",
            bbox_inches="tight", pad_inches=.04,
        )
    plt.close(fig)
    return sft.clean.json_ready(panel), routing


def write_report(manifest: dict[str, Any]) -> None:
    """Package both variants and vector downloads in a standalone offline report."""
    cell = manifest["cell"]
    variance = 100 * cell["pca"]["explained_variance_2d"]
    blocks = []
    for variant, minimum in VARIANTS.items():
        panel = manifest["variants"][variant]
        heading = ("All self-effects — matching your reference figure" if minimum is None
                   else "Positive self-effects ≥ 4 nats — the same cutoff as the arrows")
        note = ("All 30 self-effects are retained, and all are positive in this run. "
                "Very small loops may be nearly invisible."
                if minimum is None else
                "Only 8 self-effects meet the positive 4-nat cutoff; all 30 nodes remain. "
                "The node positions, labels, cross-trait arrows, and retained loop geometry are identical.")
        downloads = " · ".join(
            f'<a download="qwen25_animal_number_sft_{variant}.{ext}" '
            f'href="{sft.data_uri(OUTPUT / f"pca_{variant}.{ext}")}">{ext.upper()}</a>'
            for ext in ("png", "svg", "pdf")
        )
        blocks.append(f'''<section id="{variant}"><h2>{heading}</h2>
<figure><img src="{sft.data_uri(OUTPUT / f'pca_{variant}.png')}"
alt="Animal number-token SFT PCA network, {len(panel['edges'])} positive cross-trait arrows and {len(panel['loops'])} self-loops">
<figcaption>{len(panel['edges'])} cross-trait arrows · {len(panel['loops'])} self-loops · {variance:.1f}% of variance in two PCs.
{note}</figcaption></figure><p>Download: {downloads}</p></section>''')
    group_rows = "".join(
        f'<tr><th><span style="color:{group["color"]}">●</span> {html.escape(group["name"])}</th>'
        f'<td>{html.escape(", ".join(group["labels"]))}</td></tr>'
        for group in cell["groups"]
    )
    markup = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Qwen2.5 · Animal number-SFT FT network</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#fcfcfa;color:#233842;font:17px/1.6 system-ui,sans-serif}}
main{{max-width:1120px;margin:auto;padding:30px 24px}}h1{{font-size:clamp(26px,3vw,38px);line-height:1.2}}
h2{{font-size:22px;margin-top:36px}}.eyebrow{{font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:#607980}}
a{{color:#09787b;text-underline-offset:3px}}figure{{margin:20px 0;background:white;border:1px solid #dce4e5;border-radius:10px;overflow:hidden}}
img{{display:block;width:100%;height:auto}}figcaption{{padding:12px 20px;color:#526d75;font-size:15px}}
table{{border-collapse:collapse;width:100%;background:white}}th,td{{padding:8px 14px;text-align:left;border-bottom:1px solid #dae2e4}}
code{{overflow-wrap:anywhere;font-size:13px}}.note{{border-left:3px solid #b9c5c9;padding-left:16px}}
@media(max-width:650px){{main{{padding:20px 12px}}figure{{overflow:auto}}figure img{{min-width:690px}}}}
</style></head><body><main><div class="eyebrow">Qwen2.5-7B-Instruct · 30 animals · final step 471</div>
<h1>Animal FT effects on the PCA map</h1>
<p>The same rendering style as the <a href="qwen25_number_sft_ft_network.html">city/color number-SFT networks</a>:
larger fixed-size colored dots, curved directed arrows, outward-facing self-loops, and no axes.</p>
<p><a href="#all_self">All self-effects</a> · <a href="#positive_threshold4">Positive self-effects ≥ 4 nats</a></p>
<p><strong>Direction:</strong> trained animal → scored answer. <strong>Strength:</strong> both arrow and self-loop
width = 0.16 × |effect| points, at opacity 0.8. Cross-trait arrows show only positive effects ≥ 4 nats;
negative and smaller cross-trait effects are omitted. There is no negative-line legend because no negative effects are drawn.</p>
{''.join(blocks)}
<h2>How to read this figure</h2>
<p>Weights are the final double-centered M10 log-probability shifts (nats): subtract each row mean and column mean,
then add the grand mean. Rows are answer animals; columns are trained adapters. A red connection means an
above-additive interaction, not necessarily an absolute probability increase. The cutoff is a display choice, not a significance test.</p>
<p>PCA uses each adapter’s complete 30-answer effect profile, including positive, negative, and diagonal entries.
The full asymmetric matrix is not symmetrized or thresholded before PCA. The first two PCs explain {variance:.1f}%
of its variance. Node distances summarize profiles in this projection, not individual arrow strengths.
Only labels, arrow bends, and loop orientations are adjusted; no node jitter or force-directed repositioning is used.</p>
<p class="note">Dot colors annotate the six pre-existing five-animal catalog blocks below, not clusters discovered by the layout.
These are informal similarity groups, not a strict taxonomy: penguin is in the aquatic/coastal block, for example.
All dots have radius 3.1 points; neither color nor dot size encodes effect magnitude.</p>
<table><tbody>{group_rows}</tbody></table>
<details><summary>Experiment and reproducibility</summary>
<p>Completed job 20572695: 10,000 retained ten-number completions per animal, response-token SFT for 3 epochs,
LoRA rank 8 and alpha 32. The teacher’s animal instruction is absent from student training prompts.
This is traditional token SFT, not 100-candidate distribution matching.</p>
<p>Measured source: <code>{html.escape(cell['source'])}</code><br>SHA-256: <code>{cell['source_sha256']}</code>.</p>
<p><a href="qwen25_animal_number_sft_ft_network/manifest.json">Full matrix, PCA coordinates, and exact rendered edge/loop records</a>.</p>
</details></main></body></html>'''
    REPORT.write_text(markup)


def build() -> None:
    """Build both animal variants without touching previous results or figures."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cell = load_animal()
    reference = PRESENTATION / "qwen25_number_sft_ft_network.html"
    preserved = {path: sft.sha256(path) for path in (ROOT / cell["source"], reference)}
    routing = None
    cache = OUTPUT / "manifest.json"
    if cache.exists():
        prior = json.loads(cache.read_text())
        if prior["cell"] == cell:
            routing = prior["routing"]
    panels = {}
    for variant in VARIANTS:
        panels[variant], routing = render_variant(cell, variant, routing)
    manifest = {
        "cell": cell, "variants": panels, "routing": routing,
        "style": {"node_radius_pt": sft.MAIN_NODE_RADIUS,
                  "width_points_per_nat": sft.loops.WIDTH_SCALE,
                  "opacity": sft.loops.ALPHA,
                  "cross_trait_threshold_nats": sft.loops.THRESHOLD},
        "matrix_orientation": "rows=answer u; columns=trained adapter t; arrow t -> u",
        "group_policy": "Pre-existing catalog annotations; never used to fit PCA or select edges",
        "sources": {str(path.relative_to(ROOT)): digest for path, digest in preserved.items()},
    }
    cache.write_text(json.dumps(manifest, indent=2) + "\n")
    write_report(manifest)
    if any(sft.sha256(path) != digest for path, digest in preserved.items()):
        raise RuntimeError("An existing scientific source or city/color report changed")
    logger.success("Created animal token-SFT networks: {}", REPORT)


if __name__ == "__main__":
    build()
