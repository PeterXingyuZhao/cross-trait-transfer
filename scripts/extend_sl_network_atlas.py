"""Package networks, their complete SL matrices, and all saved metric variants.

The ordinary report uses relative files. The shareable HTML embeds every
display/download asset and needs neither this checkout nor a local server.
"""

from __future__ import annotations

import base64
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from scripts import build_sl_network_atlas as atlas
from scripts.sl_network_atlas_matrices import attach_matrices

OUT = atlas.OUT
HEATMAP_JS = atlas.ROOT / "scripts/report_templates/sl_network_atlas_heatmap.js"


def encoded_json(value: Any) -> str:
    """Embed JSON without executable closing tags or nonstandard NaN values."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")


def data_uri(content: bytes, mime: str) -> str:
    """Encode an offline asset without changing its scientific content."""
    return f"data:{mime};base64," + base64.b64encode(content).decode("ascii")


def public_cell(cell: dict[str, Any]) -> dict[str, Any]:
    """Keep numerical data and provenance; routing internals stay in ZIP records."""
    result = {k: v for k, v in cell.items() if k != "rendering"}
    return result


def embedded_assets(cells: list[dict[str, Any]]) -> dict[str, str]:
    """Collect only referenced assets, with portable numerical JSON downloads."""
    paths = set()
    for cell in cells:
        for plot in list(cell["plots"].values()) + list(cell["matrices"].values()):
            for extension in ("svg", "png", "csv"):
                if extension in plot:
                    paths.add(plot[extension])
    result = {}
    for relative in sorted(paths):
        source = OUT / relative
        if not source.resolve().is_relative_to(OUT.resolve()):
            raise ValueError(f"Nonportable path: {relative}")
        mime = {".svg": "image/svg+xml", ".png": "image/png", ".csv": "text/csv"}[source.suffix]
        result[relative] = data_uri(source.read_bytes(), mime)
    for cell in cells:
        relative = f"data/{cell['id']}.json"
        result[relative] = data_uri((OUT / relative).read_bytes(), "application/json")
    result["README.md"] = data_uri((OUT / "README.md").read_bytes(), "text/markdown")
    return result


def render_page(payload: dict[str, Any], assets: dict[str, str]) -> str:
    """Create the same interface with either local or embedded asset resolution."""
    page = atlas.TEMPLATE.read_text()
    replacements = {"__DATA_JSON__": encoded_json(payload),
                    "__EMBEDDED_ASSETS_JSON__": encoded_json(assets),
                    "__HEATMAP_JS__": HEATMAP_JS.read_text()}
    for marker, replacement in replacements.items():
        if page.count(marker) != 1:
            raise ValueError(f"Expected one template marker: {marker}")
        page = page.replace(marker, replacement)
    return page


def write_extended_report(cells: list[dict[str, Any]]) -> None:
    """Enrich existing networks, publish two offline formats, and verify packing."""
    from scripts.sl_network_atlas_metrics import load_metrics

    attach_matrices(cells)
    metrics = atlas.json_ready(load_metrics(cells))
    for cell in cells:
        cell["metric_set_ids"] = [s["id"] for s in metrics["sets"] if s.get("cell_id") == cell["id"]]
        (OUT / "data" / f"{cell['id']}.json").write_text(
            json.dumps(cell, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    payload = {"meta": {"generated_at": datetime.now(timezone.utc).isoformat(),
                        "native_cutoff": atlas.NATIVE_CUTOFF, "relative_cutoff": atlas.RELATIVE_CUTOFF,
                        "default_network_mode": "relative",
                        "adaptive_cutoff_rule": "1.5 times each matrix's off-diagonal RMS; same positive cutoff for arrows and loops",
                        "primary_count": sum(c["subset"] == "primary" for c in cells),
                        "variant_count": sum(c["subset"] != "primary" for c in cells),
                        "orientation": "rows: scored answer; columns: trained teacher trait",
                        "portable_single_file": False},
               "cells": [public_cell(c) for c in cells], "metrics": metrics}
    (OUT / "atlas.json").write_text(encoded_json(payload) + "\n")
    (OUT / "metrics.json").write_text(encoded_json(metrics) + "\n")
    (OUT / "index.html").write_text(render_page(payload, {}))
    portable = {**payload, "meta": {**payload["meta"], "portable_single_file": True}}
    assets = embedded_assets(cells)
    (OUT / "share.html").write_text(render_page(portable, assets))
    with zipfile.ZipFile(OUT / "sl_network_atlas.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        # The ZIP already includes every asset. Avoid duplicating them in share.html.
        for path in sorted(OUT.rglob("*")):
            if path.is_file() and path.suffix != ".zip" and path.name != "share.html":
                archive.write(path, "sl_network_atlas/" + str(path.relative_to(OUT)))
    with zipfile.ZipFile(OUT / "sl_network_atlas.zip") as archive:
        if archive.testzip() is not None:
            raise ValueError("Offline package CRC check failed")
        if archive.read("sl_network_atlas/index.html") != (OUT / "index.html").read_bytes():
            raise ValueError("Packaged HTML differs from the published report")
    measured = sum(any(m["kind"] == "matrix" for m in s["metrics"]) for s in metrics["sets"])
    logger.success("Extended atlas: {} SL endpoints, {} settings with metric matrices", len(cells), measured)
    logger.info("Share HTML: {:.1f} MB; offline ZIP: {:.1f} MB",
                (OUT / "share.html").stat().st_size / 1e6,
                (OUT / "sl_network_atlas.zip").stat().st_size / 1e6)


def main() -> None:
    """Reuse verified network assets instead of altering their geometry or scale."""
    cells = []
    for fresh in atlas.load_cells():
        saved = json.loads((OUT / "data" / f"{fresh['id']}.json").read_text())
        if saved["source_sha256"] != fresh["source_sha256"]:
            raise ValueError(f"Network source changed: {fresh['id']}")
        np.testing.assert_array_equal(saved["raw"], fresh["raw"])
        np.testing.assert_array_equal(saved["double_centered"], fresh["double_centered"])
        for mode in ("native", "relative"):
            atlas.check_panel(fresh, saved["rendering"][mode]["panel"], mode)
        cells.append(saved)
    write_extended_report(cells)


if __name__ == "__main__":
    main()
