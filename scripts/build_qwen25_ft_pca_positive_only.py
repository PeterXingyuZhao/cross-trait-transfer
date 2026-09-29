"""Build the positive-only comparison and its compact, larger-node main figure.

The 3D and no-loop geometries stay fixed. The self-loop main panel keeps exact
PCA coordinates and weights, with larger node ports and tighter panel packing.
Signed source artifacts stay intact.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from loguru import logger
from PIL import Image

from scripts.build_qwen25_ft_pca_comparison import (
    FIGURES,
    OUTPUT as ORIGINAL,
    ROOT,
    FrameParser,
    data_uri,
    replace_once,
)

OUTPUT = ORIGINAL.with_name("qwen25_ft_network_figure_comparison_positive_only.html")
EDGE_ID = re.compile(
    r"(?:edge-(city|color)|external-(city|color)-edge)-"
    r"(\d+)-to-(\d+)-w-(-?\d+(?:\.\d+)?)"
)
MAIN_NODE_RADIUS = 3.1
MAIN_PANEL_GAP = 30.0


def remove_negative_legend(source: str) -> str:
    """Remove the negative legend sample and its label, not blue trait nodes."""
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    root = ET.fromstring(source, parser=parser)
    count = 0
    for legend in root.iter():
        if not legend.get("id", "").startswith("legend_"):
            continue
        children = list(legend)
        for index, child in enumerate(children):
            comments = [element.text or "" for element in child.iter()
                        if element.tag is ET.Comment]
            if any(text.strip() == "Negative →" for text in comments):
                if index == 0 or not children[index-1].get("id", "").startswith("line2d_"):
                    raise ValueError("Unexpected negative legend sample structure")
                legend.remove(children[index-1])
                legend.remove(child)
                count += 1
    if count != 1:
        raise ValueError(f"Expected exactly one negative legend key, found {count}")
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    return ET.tostring(root, encoding="unicode")


def render_main_figure(target: Path) -> dict[str, Any]:
    """Tight-pack rigid PCA panels; reattach unchanged-strength arrows to larger dots."""
    import matplotlib.pyplot as plt
    import numpy as np
    from scripts import build_qwen25_ft_network_self_loops as loops

    source = json.loads(loops.SOURCE.read_text())
    clean = json.loads(loops.CLEAN_MANIFEST.read_text())
    metadata = json.loads(loops.METADATA.read_text())
    old = json.loads((loops.OUT / "pca_self_loop_manifest.json").read_text())["variants"]["external"]
    # Preserve each panel's physical PCA scale; remove only the wasted gutter.
    widths = [p["bounds_points"][1]-p["bounds_points"][0]+2 for p in old]
    heights = [p["bounds_points"][3]-p["bounds_points"][2]+2 for p in old]
    margin, bottom, top = 18., 60., 25.
    width = sum(widths) + MAIN_PANEL_GAP + 2*margin
    height = max(heights) + bottom + top
    fig = plt.figure(figsize=(width/72, height/72), dpi=160)
    records = []
    left = margin
    for index, previous in enumerate(clean["panels"]):
        ax = fig.add_axes([left/width, bottom/height, widths[index]/width, heights[index]/height])
        family = previous["family"]
        record = loops.panel(
            ax, source["cells"][f"{family}-00-99"], previous, metadata,
            variant="external", title=f"{'A' if index == 0 else 'B'}  {family.title()}",
            node_radius=MAIN_NODE_RADIUS, positive_only=True,
        )
        records.append(record)
        left += widths[index] + MAIN_PANEL_GAP
    divider_x = (margin + widths[0] + MAIN_PANEL_GAP/2)/width
    divider = plt.Line2D([divider_x, divider_x],
                         [bottom/height, (bottom+max(heights)+9)/height],
                         transform=fig.transFigure, color="#b9c5c9", alpha=.65,
                         linewidth=.65, linestyle=(0, (3, 4)), zorder=0)
    divider.set_gid("city-color-panel-divider")
    fig.add_artist(divider)
    loops.legend(fig, positive_only=True)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(target.with_suffix("."+extension), dpi=260, facecolor="white",
                    bbox_inches="tight", pad_inches=.04)
    plt.close(fig)
    if [len(record["edges"]) for record in records] != [86, 37]:
        raise ValueError("Main figure lost a positive cross-trait effect")
    if any(len(record["loops"]) != 30 for record in records):
        raise ValueError("Main figure lost a self-effect")
    for record in records:
        np.testing.assert_array_equal(record["coordinates"],
                                      source["cells"][f"{record['family']}-00-99"]["pca"]["coordinates"])
    manifest = {
        "node_radius_pt": MAIN_NODE_RADIUS, "previous_node_radius_pt": loops.NODE_RADIUS,
        "panel_gap_pt": MAIN_PANEL_GAP,
        "previous_panel_gap_pt": old[1]["bounds_points"][0]-old[0]["bounds_points"][1]-2,
        "width_points_per_nat": loops.WIDTH_SCALE, "opacity": loops.ALPHA,
        "cross_trait_threshold_nats": loops.THRESHOLD,
        "self_effect_threshold": "none; all 30 per family",
        "source_sha256": hashlib.sha256(loops.SOURCE.read_bytes()).hexdigest(),
        "policy": "Exact PCA coordinates and physical scale preserved within each panel. "
                  "Larger fixed-size dots, narrower gutter, light dashed divider, positive-only legend. "
                  "Arrow endpoints reattached and self-loop orientations reselected around larger nodes; "
                  "weights, directions, linewidths, opacity and threshold unchanged.",
        "panels": records,
    }
    target.with_name("pca_positive_main_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    logger.success("Refined main figure: radius {} pt; panel gap {} pt", MAIN_NODE_RADIUS, MAIN_PANEL_GAP)
    return {key: value for key, value in manifest.items() if key != "panels"}


def positive_svg(source: str) -> tuple[str, dict[str, int]]:
    """Delete only negative off-diagonal groups, leaving other SVG bytes fixed."""
    root = ET.fromstring(source)
    removed: list[str] = []
    counts = {"city_positive": 0, "color_positive": 0,
              "city_negative_removed": 0, "color_negative_removed": 0}
    for element in root.iter():
        identifier = element.get("id", "")
        match = EDGE_ID.fullmatch(identifier)
        if not match:
            continue
        family = match[1] or match[2]
        if match[3] == match[4]:
            raise ValueError("A cross-edge group unexpectedly contains a self-effect")
        weight = float(match[5])
        if weight < 0:
            removed.append(identifier)
            counts[f"{family}_negative_removed"] += 1
        else:
            counts[f"{family}_positive"] += 1
    if counts != {"city_positive": 86, "color_positive": 37,
                  "city_negative_removed": 46, "color_negative_removed": 13}:
        raise ValueError(f"Unexpected frozen-figure edge inventory: {counts}")
    for identifier in removed:
        # Each Matplotlib arrow group consists of two paths, without nested g's.
        pattern = rf'<g id="{re.escape(identifier)}">(?:(?!<g\b).)*?</g>'
        source, count = re.subn(pattern, "", source, flags=re.S)
        if count != 1:
            raise ValueError(f"Expected one non-nested edge group: {identifier}")
    ET.fromstring(source)  # Fail if a future upstream format breaks the surgery.
    return source, counts


def positive_scene(source: str) -> str:
    """Skip negative paths and hit targets without changing data or geometry."""
    marker = "cell.edges.slice().sort((a,b)=>Math.abs(a.w)-Math.abs(b.w)).forEach(edge=>{"
    source = replace_once(source, marker, marker + "\n      if (edge.w < 0) return;")
    source = replace_once(
        source,
        "${cell.edges.filter(e=>e.w>0).length} positive / ${cell.edges.filter(e=>e.w<0).length} negative edges",
        "${cell.edges.filter(e=>e.w>0).length} positive edges shown · ${cell.edges.filter(e=>e.w<0).length} negative hidden",
    )
    source = replace_once(source,
                          '&nbsp; <span class="negative">┄┄▶</span> Negative centered FT effect · Adapter → answer',
                          '· Adapter → answer')
    return source


def render_assets(converter: str, svg: Path, original_png: Path) -> None:
    """Export the filtered vector at the existing raster and physical sizes."""
    with Image.open(original_png) as image:
        width, height = image.size
    subprocess.run([converter, "--format=png", f"--width={width}",
                    f"--height={height}", "--output", str(svg.with_suffix(".png")),
                    str(svg)], check=True)
    subprocess.run([converter, "--format=pdf", "--output",
                    str(svg.with_suffix(".pdf")), str(svg)], check=True)


def build(original: Path = ORIGINAL, output: Path = OUTPUT) -> None:
    """Write a portable positive-only variant with a compact, larger-node main panel."""
    if output.resolve() == original.resolve():
        raise ValueError("The original comparison must not be overwritten")
    converter = shutil.which("rsvg-convert")
    if not converter:
        raise RuntimeError("rsvg-convert is required for matching PNG/PDF exports")
    document = original.read_text()
    original_hash = hashlib.sha256(original.read_bytes()).hexdigest()
    parser = FrameParser()
    parser.feed(document)
    if len(parser.frames) != 1:
        raise ValueError("Expected exactly one original interactive scene")
    document = replace_once(document, html.escape(parser.frames[0], quote=True),
                            html.escape(positive_scene(parser.frames[0]), quote=True))
    provenance_match = re.search(
        r'(<script id="figure-provenance" type="application/json">)(.*?)(</script>)',
        document, re.S,
    )
    if provenance_match is None:
        raise ValueError("Missing source provenance")
    provenance = json.loads(provenance_match[2])
    new_assets = {}
    inventories = {}
    main_refinement = None
    for stem in FIGURES.values():
        target = stem.with_name(stem.name + "_positive_only")
        svg, counts = positive_svg(stem.with_suffix(".svg").read_text())
        if stem.name == "pca_self_loops_external":
            main_refinement = render_main_figure(target)
        else:
            target.with_suffix(".svg").write_text(remove_negative_legend(svg))
            render_assets(converter, target.with_suffix(".svg"), stem.with_suffix(".png"))
        inventories[stem.name] = counts
        for extension in ("png", "svg", "pdf"):
            before = stem.with_suffix("." + extension)
            after = target.with_suffix("." + extension)
            document = replace_once(document, data_uri(before), data_uri(after))
            document = document.replace(f'download="{before.name}"',
                                        f'download="{after.name}"')
            new_assets[str(after.relative_to(ROOT))] = hashlib.sha256(after.read_bytes()).hexdigest()
    replacements = [
        ("<title>City &amp; color FT networks — 3D and 2D figure comparison</title>",
         "<title>Positive-only FT networks — 3D and 2D figure comparison</title>"),
        ("<h1>Three views of directed FT effects</h1>",
         "<h1>Three views of directed FT effects · positive-only</h1>"),
        ("All show the same final-checkpoint, double-centered FT effects; only the presentation changes.",
         "Negative cross-trait arrows are hidden. The main self-loop figure uses larger nodes and tighter panel spacing; "
         "PCA coordinates and measured strengths are unchanged. "
         "<a href=\"qwen25_ft_network_figure_comparison.html\">Open the original signed version</a>."),
        ("The original 3D renderer is preserved, including its axes, palette roles, curved arrows, and interaction-dependent opacity.",
         "The 3D renderer is preserved except that it skips negative cross-trait arrows; its axes, palette roles, curved positive arrows, and interaction-dependent opacity are unchanged."),
        ("Solid red and dashed blue directed arrows encode positive and negative centered FT effects; dot area encodes diagonal magnitude.",
         "Solid red directed arrows encode positive centered FT effects; negative cross-trait arrows are omitted. Dot area encodes diagonal magnitude."),
        ("Each view retains the same 132 city and 50 color cross-trait arrows with absolute centered effect ≥4 nats.",
         "Each view now shows 86 city and 37 color positive cross-trait arrows with centered effect ≥4 nats. "
         "The 46 city and 13 color negative arrows are hidden. All self-effects are unchanged. "
         "Legends show positive effects only; no negative cross-trait arrows are drawn."),
        ("Fixed-size dots identify traits. Self-loops leave and return to the same dot; their width uses exactly the same scale as the cross-trait arrows.",
         "Larger fixed-size dots identify traits. A light dashed vertical divider separates city and color; it is not a data edge. "
         "Self-loops leave and return to the same dot, using exactly the same line-width scale as the cross-trait arrows."),
        ("two original PNGs", "two positive-only PNGs"),
    ]
    for before, after in replacements:
        document = replace_once(document, before, after)
    provenance["positive_only_variant"] = {
        "source_html": str(original.relative_to(ROOT)),
        "source_html_sha256": original_hash,
        "operation": "Hide negative off-diagonal arrows and remove negative legend entries; refine self-loop main figure presentation",
        "inventories": inventories,
        "preserved": ["full data", "PCA coordinates", "within-panel PCA scale",
                      "self-effect values", "edge weights and directions", "widths", "opacity", "trait identities"],
        "main_figure_refinement": main_refinement,
        "assets_sha256": new_assets,
        "raster_export": "No-loop: original canvas via rsvg-convert. Main self-loop: source-rendered PNG/SVG/PDF with compact canvas.",
    }
    document = replace_once(document, provenance_match[0],
                            provenance_match[1] + json.dumps(provenance).replace("<", "\\u003c")
                            + provenance_match[3])
    output.write_text(document)
    if hashlib.sha256(original.read_bytes()).hexdigest() != original_hash:
        raise RuntimeError("Original comparison was unexpectedly changed")
    logger.success("Wrote positive-only comparison: {} ({:.2f} MB)", output,
                   output.stat().st_size / 1_000_000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=ORIGINAL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    build(args.original, args.output)
