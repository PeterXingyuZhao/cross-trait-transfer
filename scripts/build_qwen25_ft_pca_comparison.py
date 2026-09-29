"""Package the original interactive 3D scene and unchanged 2D exports offline.

First export the original fragment with the visualize skill's render.py, then
pass that file via --rendered-export. The source fragment is never modified.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import re
from html.parser import HTMLParser
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
PRESENTATION = ROOT / "results_for_presentation"
OUTPUT = PRESENTATION / "qwen25_ft_network_figure_comparison.html"
TEMPLATE = ROOT / "scripts/report_templates/qwen25_ft_network_figure_comparison.html"
D3 = ROOT / "scripts/report_templates/vendor/d3-7.9.0.min.js"
FIGURES = {"NODES": PRESENTATION / "qwen25_ft_network/pca_clean_width",
           "LOOPS": PRESENTATION / "qwen25_ft_network/pca_self_loops_external"}


class FrameParser(HTMLParser):
    """Extract the original exported scene, including its scoped style kit."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.frames: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Collect complete srcdoc documents without executing their scripts."""
        if tag == "iframe":
            source = dict(attrs).get("srcdoc")
            if source:
                self.frames.append(source)


def replace_once(text: str, before: str, after: str) -> str:
    """Fail rather than silently transforming a different visualization version."""
    if text.count(before) != 1:
        raise ValueError(f"Expected exactly one known source marker: {before[:100]}")
    return text.replace(before, after, 1)


def prepare_scene(export: str) -> tuple[str, dict[str, object]]:
    """Preserve the renderer while setting screenshot defaults and offline assets."""
    parser = FrameParser()
    parser.feed(export)
    if len(parser.frames) != 1 or 'id="ft-pca3-weighted"' not in parser.frames[0]:
        raise ValueError("Expected the original weighted 3D visualization export")
    scene = parser.frames[0]
    data_match = re.search(r'<script type="application/json" data-pca3>(.*?)</script>',
                           scene, flags=re.S)
    if data_match is None:
        raise ValueError("Original embedded PCA data not found")
    payload = json.loads(data_match[1])
    original_scene_hash = hashlib.sha256(scene.encode()).hexdigest()
    replacements = [
        ("yaw: -35, pitch: 20, edges: true, self: 'area'",
         "yaw: -90, pitch: -24, edges: true, self: 'loop'"),
        ('value="-35" data-yaw', 'value="-90" data-yaw'),
        ('value="20" data-pitch', 'value="-24" data-pitch'),
        ('class="tabular-nums">−35°</output>', 'class="tabular-nums">−90°</output>'),
        ('class="tabular-nums">20°</output>', 'class="tabular-nums">−24°</output>'),
        ('<option value="loop">', '<option value="loop" selected>'),
    ]
    for before, after in replacements:
        scene = replace_once(scene, before, after)
    scene = replace_once(scene,
                         '<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>',
                         '<script>\n'+D3.read_text()+'\n</script>')
    # The export kit's general-purpose tooltip/icon dependencies are unused by
    # this scene, which implements its own SVG inspection and has no icons.
    unused = r'<script\b[^>]*src="https://unpkg\.com/(?:@floating-ui/[^" ]+|lucide@[^" ]+)"[^>]*></script>'
    scene, removed = re.subn(unused, "", scene)
    if removed != 3 or re.search(r'<script\b[^>]*\bsrc=', scene):
        raise ValueError("Unexpected external script dependency in exported scene")
    # Resolve host-owned color tokens for a portable, light paper comparison.
    # Node fills and positive line color are read from the supplied screenshot;
    # data, grouping, and color-selection logic remain the original renderer's.
    appearance = """<style>
html {color-scheme:light!important}
:root {--viz-series-1:#3c66bf;--viz-series-2:#eccb5e;--viz-series-3:#db7dae;
--viz-series-4:#63b7b0;--viz-series-5:#7bc780;--viz-series-6:#e48e4d;
--red:#ce4035;--blue:#4e82ef}
#ft-pca3-weighted [hidden] {display:none!important}
</style>"""
    scene = replace_once(scene, "</head>", appearance+"\n</head>")
    resize_script = """<script>
(() => {
  let previous = 0;
  const root = document.getElementById('ft-pca3-weighted');
  function resize(force = false) {
    const height = Math.ceil(root.getBoundingClientRect().bottom + scrollY + 16);
    if(force || height !== previous) {
      parent.postMessage({type:'qwen25-pca3-height',height}, '*');
      previous = height;
    }
  }
  new ResizeObserver(() => resize()).observe(root);
  addEventListener('load',() => resize(true));
  addEventListener('message',event => {
    if(event.source === parent && event.data?.type === 'qwen25-pca3-size-request') resize(true);
  });
  resize();
})();
</script>"""
    scene = replace_once(scene, "</body>", resize_script+"\n</body>")
    return scene, {"exported_scene_sha256": original_scene_hash,
                   "data": payload, "initial_view": {"projection": "3d", "self": "loop",
                   "yaw": -90, "pitch": -24, "edges": True},
                   "export_changes": ["Screenshot view defaults", "Inline pinned D3 7.9.0",
                                      "Remove unused export-kit remote dependencies",
                                      "Resolve host palette for the light screenshot",
                                      "Restore hidden-control semantics in export-kit CSS",
                                      "Auto-size the standalone embedding with a load handshake"],
                   "renderer_changes": "None beyond initial state; original data and drawing logic retained"}


def data_uri(path: Path) -> str:
    """Embed the exact file bytes for a genuinely portable single-file report."""
    mime = {".png": "image/png", ".svg": "image/svg+xml", ".pdf": "application/pdf"}[path.suffix]
    return f"data:{mime};base64,"+base64.b64encode(path.read_bytes()).decode("ascii")


def build(rendered_export: Path, output: Path = OUTPUT) -> None:
    """Combine one original interactive export with two byte-identical figures."""
    scene, provenance = prepare_scene(rendered_export.read_text())
    document = TEMPLATE.read_text()
    document = replace_once(document, "__SCENE__", html.escape(scene, quote=True))
    assets = {}
    for key, stem in FIGURES.items():
        for extension in ("png", "svg", "pdf"):
            path = stem.with_suffix("."+extension)
            document = replace_once(document, f"__{key}_{extension.upper()}__", data_uri(path))
            assets[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance["assets_sha256"] = assets
    provenance["d3_sha256"] = hashlib.sha256(D3.read_bytes()).hexdigest()
    document = replace_once(document, "__PROVENANCE__",
                            json.dumps(provenance).replace("<", "\\u003c"))
    if re.search(r"__[A-Z_]+__", document):
        raise ValueError("Unresolved template placeholder")
    output.write_text(document)
    logger.success("Wrote standalone comparison: {} ({:.2f} MB)", output,
                   output.stat().st_size/1_000_000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rendered-export", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    build(args.rendered_export, args.output)
