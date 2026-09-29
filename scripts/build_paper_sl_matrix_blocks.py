"""Make a compact, source-verified Section 3 SL heatmap figure and selection audit."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
ATLAS = ROOT / "results_for_presentation/sl_network_atlas/atlas.json"
OUT = ROOT / "results_for_presentation/paper_sl_matrix_blocks"
PDF = ROOT / "output/pdf/sl_matrix_blocks.pdf"
SELECTED = (
    "qwen25_city_number_distribution",
    "qwen3vl_city_number_sft",
    "llama31_city_number_sft",
    "mistral_v03_color_number_sft",
    "qwen25_animal_opaque_distribution",
    "qwen25_color_element_distribution",
    "qwen25_animal_gsm8k_sft",
    "qwen25_color_python_sft",
)
FIGURE_SIZE = (5.5, 3.6)
PLOT_CARRIERS = {"number": "00-99 numbers", "opaque": "100 rand tokens",
                 "element": "chem elements", "gsm8k": "GSM8K",
                 "python": "Python code"}
TITLES = {"qwen25": "Qwen2.5 7B", "llama31": "Llama 3.1 8B",
          "qwen3vl": "Qwen3-VL 8B", "olmo2": "OLMo 2 7B", "mistral_v03": "Mistral 7B"}
FAMILIES = {"city": "Cities", "animal": "Animals", "color": "Colors", "tree": "Trees"}
GROUP_NAMES = {
    "city": ("Japan", "US Northeast", "W. Europe", "Italy", "China", "India"),
    "animal": ("Canids", "Felids", "Birds", "Bears/relatives", "Aquatic/coastal", "Large herbivores"),
    "color": ("Red/pink", "Orange/yellow", "Green/teal", "Blue/cyan", "Purple/magenta", "Brown/neutral"),
}
# Existing city and animal catalog groups are five labels each. Color bins
# are explicit presentation annotations of the existing hue-ordered catalog,
# not fitted clusters: magenta stays with purple, not with neutral colors.
GROUP_SIZES = {"city": (5, 5, 5, 5, 5, 5), "animal": (5, 5, 5, 5, 5, 5),
               "color": (5, 5, 5, 5, 6, 4)}
EXPECTED_LABELS = {
    "city": "Tokyo|Kyoto|Osaka|Kobe|Nara|New York|Boston|Philadelphia|Washington|Baltimore|London|Paris|Brussels|Amsterdam|Luxembourg|Rome|Milan|Florence|Venice|Naples|Beijing|Tianjin|Shanghai|Hangzhou|Suzhou|Delhi|Mumbai|Bengaluru|Chennai|Hyderabad".split("|"),
    "animal": "dog|husky|wolf|fox|coyote|cat|lion|tiger|leopard|jaguar|eagle|hawk|falcon|owl|raven|bear|panda|polar bear|raccoon|badger|dolphin|whale|shark|octopus|penguin|elephant|giraffe|zebra|horse|deer".split("|"),
    "color": "red|crimson|scarlet|vermilion|pink|orange|amber|ochre|saffron|yellow|green|emerald|teal|turquoise|aqua|blue|azure|cerulean|cyan|navy|purple|violet|indigo|lavender|mauve|magenta|brown|black|white|gray".split("|"),
}


def sha256(path: Path) -> str:
    """Identify exact source bytes, not only a rounded figure array."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def double_center(values: np.ndarray) -> np.ndarray:
    """Center the entire signed matrix, before any display operation."""
    return values - values.mean(0, keepdims=True) - values.mean(1, keepdims=True) + values.mean()


def load_cells() -> list[dict[str, Any]]:
    """Verify all primary atlas matrices against their original labeled CSVs."""
    cells = json.loads(ATLAS.read_text())["cells"]
    primary = [c for c in cells if c["subset"] == "primary"]
    if len(primary) != 35 or len({c["id"] for c in primary}) != 35:
        raise ValueError("Expected the saved 35 primary experiments")
    for cell in primary:
        source = ROOT / cell["source"]
        if sha256(source) != cell["source_sha256"] or cell["scale"] != 1:
            raise ValueError(f"Source drift or extrapolated endpoint: {cell['id']}")
        with source.open(newline="") as handle:
            rows = list(csv.reader(handle))
        teachers, answers = rows[0][1:], [r[0] for r in rows[1:]]
        labels = cell["labels"]
        if (len(labels) != 30 or len(set(labels)) != 30
                or len(teachers) != 30 or len(answers) != 30
                or set(teachers) != set(labels) or set(answers) != set(labels)):
            raise ValueError(f"Mismatched catalogs: {cell['id']}")
        raw = np.asarray([r[1:] for r in rows[1:]], dtype=float)
        raw = raw[np.ix_([answers.index(z) for z in labels], [teachers.index(z) for z in labels])]
        if raw.shape != (30, 30) or not np.isfinite(raw).all():
            raise ValueError(f"Invalid matrix: {cell['id']}")
        np.testing.assert_allclose(raw, cell["raw"], atol=1e-12, rtol=0)
        dc = double_center(raw)
        np.testing.assert_allclose(dc, cell["double_centered"], atol=1e-9, rtol=0)
        if cell["family"] in EXPECTED_LABELS and labels != EXPECTED_LABELS[cell["family"]]:
            raise ValueError("Semantic group boundaries no longer match the catalog")
        cell["plot_limit"] = color_limit(dc)
    return primary


def color_limit(values: np.ndarray) -> float:
    """Round a symmetric native-unit limit upward; never clip any entry."""
    maximum = float(np.abs(values).max())
    if not np.isfinite(maximum):
        raise ValueError("Nonfinite color range")
    if maximum == 0:
        return 1.0
    step = 10 ** np.floor(np.log10(maximum)) / 2
    if step == 0:
        return maximum
    with np.errstate(over="ignore"):
        rounded = float(np.ceil(maximum / step) * step)
    return max(rounded, maximum) if np.isfinite(rounded) else maximum


def grouping(family: str) -> tuple[np.ndarray, np.ndarray]:
    """Return semantic block centers and boundaries in imshow coordinates."""
    sizes = np.asarray(GROUP_SIZES[family])
    ends = sizes.cumsum()
    return ends - sizes / 2 - .5, ends[:-1] - .5


def descriptive_block_statistics(cell: dict[str, Any]) -> dict[str, Any]:
    """Describe existing city/animal annotations, excluding all self-effects."""
    dc = np.asarray(cell["double_centered"])
    off = ~np.eye(30, dtype=bool)
    result: dict[str, Any] = {"off_diagonal_rms_nats": float(np.sqrt(np.mean(dc[off] ** 2)))}
    if cell["family"] not in ("city", "animal"):
        return result  # Color presentation bins are not an existing group test.
    group = np.repeat(np.arange(6), 5)
    within = (group[:, None] == group[None, :]) & off
    between = (group[:, None] != group[None, :]) & off
    difference = float(dc[within].mean() - dc[between].mean())
    result.update(within_group_off_diagonal_mean_nats=float(dc[within].mean()),
                  between_group_mean_nats=float(dc[between].mean()),
                  gap_nats=difference, gap_over_rms=difference / result["off_diagonal_rms_nats"],
                  within_entries=int(within.sum()), between_entries=int(between.sum()))
    return result


def heatmap(ax: plt.Axes, cell: dict[str, Any], *, labels: str = "groups") -> Any:
    """Draw every signed entry in its original order, with reference boundaries."""
    dc = np.asarray(cell["double_centered"])
    image = ax.imshow(dc, cmap="RdBu_r", vmin=-cell["plot_limit"], vmax=cell["plot_limit"],
                      interpolation="nearest", aspect="equal", rasterized=False)
    if cell["family"] in GROUP_SIZES:
        centers, boundaries = grouping(cell["family"])
        for coordinate in boundaries:
            ax.axvline(coordinate, color="#404b50", lw=.4, ls=(0, (2.1, 1.7)), alpha=.72)
            ax.axhline(coordinate, color="#404b50", lw=.4, ls=(0, (2.1, 1.7)), alpha=.72)
        if labels == "groups":
            ax.set_xticks(centers, range(1, 7), fontsize=6)
            ax.set_yticks(centers, range(1, 7), fontsize=6)
    if labels == "traits":
        ax.set_xticks(range(30), cell["labels"], fontsize=8, rotation=90)
        ax.set_yticks(range(30), cell["labels"], fontsize=8)
    elif labels == "none" or cell["family"] not in GROUP_SIZES:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.tick_params(length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_linewidth(.45)
        spine.set_edgecolor("#64737b")
    return image


def main_figure(selected: list[dict[str, Any]]) -> None:
    """Fit eight unchanged matrices with thin side bars and caption-only context."""
    if len(selected) != 8:
        raise ValueError("The two-row figure requires eight selected panels")
    fig = plt.figure(figsize=FIGURE_SIZE)
    figure_width, height = FIGURE_SIZE
    matrix_size = .985
    for i, cell in enumerate(selected):
        row, column = divmod(i, 4)
        x = .17 + column * 1.34
        panel_bottom = 2.065 if row == 0 else .435
        ax = fig.add_axes([x / figure_width, panel_bottom / height,
                           matrix_size / figure_width, matrix_size / height])
        image = heatmap(ax, cell)
        heading = TITLES[cell["model"]] if row == 0 else PLOT_CARRIERS[cell["carrier"]]
        title_y = (3.49 if row == 0 else 1.81) / height
        fig.text((x + matrix_size/2) / figure_width, title_y,
                 f"{chr(65+i)}  {heading}", fontsize=7.1, weight="bold", ha="center")
        objective = "Logit distillation" if cell["objective"] == "distribution" else "Token SFT"
        fig.text((x + matrix_size/2) / figure_width, (3.27 if row == 0 else 1.63)/height,
                 f"{FAMILIES[cell['family']]}\n{objective}",
                 fontsize=6.3, ha="center", va="center", linespacing=1.25,
                 color="#344a56")
        cax = fig.add_axes([(x + matrix_size + .040) / figure_width,
                            panel_bottom / height, .024 / figure_width,
                            matrix_size / height])
        limit = cell["plot_limit"]
        bar = fig.colorbar(image, cax=cax, orientation="vertical", ticks=[-limit, 0, limit])
        bar.ax.set_yticklabels([f"{value:g}" for value in (-limit, 0, limit)])
        bar.ax.tick_params(labelsize=5.5, length=1.4, width=.35, pad=1.1)
        bar.outline.set_linewidth(.4)
    for y, family in zip((.225, .135, .045), ("city", "animal", "color"), strict=True):
        key = "   ".join(f"{i+1} {name}" for i, name in enumerate(GROUP_NAMES[family]))
        fig.text(.035, y/height, f"{FAMILIES[family]}:  {key}", fontsize=6.0, ha="left")
    PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF, facecolor="white")
    fig.savefig(PDF.with_name("sl_matrix_sweeping.pdf"), facecolor="white")
    for extension in ("png", "svg"):
        fig.savefig(OUT / f"sl_matrix_blocks.{extension}", dpi=350, facecolor="white")
    plt.close(fig)


def overview(cells: list[dict[str, Any]]) -> None:
    """Show all 35 primary cells, including weak patterns and unselected settings."""
    fig, axes = plt.subplots(7, 5, figsize=(13, 18.5))
    fig.subplots_adjust(left=.025, right=.975, top=.94, bottom=.02, hspace=.44, wspace=.19)
    for ax, cell in zip(axes.flat, cells, strict=True):
        heatmap(ax, cell, labels="none")
        objective = "logits" if cell["objective"] == "distribution" else "token SFT"
        selected = "* " if cell["id"] in SELECTED else ""
        ax.set_title(f"{selected}{TITLES[cell['model']]} | {FAMILIES[cell['family']]}\n"
                     f"{PLOT_CARRIERS[cell['carrier']]} | {objective}\nscale: +/- {cell['plot_limit']:g} nats",
                     fontsize=8, linespacing=1.25, pad=5)
    fig.suptitle("All 35 primary endpoints | * selected for the paper | individual symmetric nats scales",
                 fontsize=12, y=.988)
    fig.savefig(OUT / "all_primary_candidates.png", dpi=170, facecolor="white")
    plt.close(fig)


def detail(cell: dict[str, Any], letter: str) -> None:
    """Export the selected matrix with all trait labels and exact CSV values."""
    fig, ax = plt.subplots(figsize=(7.4, 7.4))
    fig.subplots_adjust(left=.18, right=.83, top=.86, bottom=.23)
    image = heatmap(ax, cell, labels="traits")
    ax.set_xlabel("Teacher trait z", fontsize=10)
    ax.set_ylabel("Scored answer z'", fontsize=10)
    fig.colorbar(image, ax=ax, fraction=.04, pad=.025).set_label("Double-centered SL (nats)")
    fig.suptitle(f"{letter}  {TITLES[cell['model']]} | {FAMILIES[cell['family']]}\n"
                 f"{PLOT_CARRIERS[cell['carrier']]} | {cell['objective_title']}", fontsize=13)
    fig.savefig(OUT / f"panel_{letter}.png", dpi=180, facecolor="white")
    plt.close(fig)
    with (OUT / f"panel_{letter}_centered.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scored_answer / teacher_trait", *cell["labels"]])
        writer.writerows([[z, *row] for z, row in zip(cell["labels"], cell["double_centered"], strict=True)])


def qwen3_candidates(cells: list[dict[str, Any]]) -> None:
    """Show every completed Qwen3-VL primary number-SFT candidate for selection."""
    candidates = [c for c in cells if c["model"] == "qwen3vl"]
    if {c["family"] for c in candidates} != {"city", "animal", "color"} or len(candidates) != 3:
        raise ValueError("Expected all three Qwen3-VL primary candidates")
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.2))
    fig.subplots_adjust(left=.05, right=.98, bottom=.22, top=.78, wspace=.25)
    for ax, cell in zip(axes, candidates, strict=True):
        image = heatmap(ax, cell)
        ax.set_title(f"{FAMILIES[cell['family']]}" + (" - selected" if cell["id"] in SELECTED else ""), fontsize=10)
        bar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=.06, pad=.19,
                           ticks=[-cell["plot_limit"], 0, cell["plot_limit"]])
        bar.ax.tick_params(labelsize=7)
    axes[0].set_ylabel("Answer group", fontsize=8)
    fig.suptitle("Qwen3-VL 8B (text-only) | 00-99 token SFT | t = 1", fontsize=12, y=.95)
    fig.text(.5, .045, "Full double-centered SL; native nats scales; fixed trait order", fontsize=8, ha="center")
    fig.savefig(OUT / "qwen3_candidates.png", dpi=220, facecolor="white")
    plt.close(fig)


def write_latex() -> None:
    """Supply a standalone figure input and a minimal insertion for empty Section 3."""
    (OUT / "sl_matrix_blocks.tex").write_text(r"""% Upload sl_matrix_blocks.pdf to figures/images/.
\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{figures/images/sl_matrix_blocks.pdf}
\caption{\textbf{Cross-trait structure across models, traits, and carriers.}
Each panel shows the double-centered SL matrix for 30 traits;
rows index scored answers and columns index teacher traits.
\textbf{A--D}: 00--99 number carriers across Qwen2.5-7B-Instruct (A),
Qwen3-VL-8B-Instruct (B, text-only), Llama-3.1-8B-Instruct (C), and
Mistral-7B-Instruct-v0.3 (D).
\textbf{E--H}: other carriers in Qwen2.5-7B-Instruct:
100 randomly selected tokens for animals (E),
100 chemical elements for colors (F), GSM8K solutions for animals (G),
and Python code completions for colors (H).
Traits and training objectives are labeled above each matrix.
Dashed lines mark the reference groups listed below the panels,
not fitted clusters; color groups are broad hue annotations.
Thin vertical bars give separate symmetric scales in nats.
All 900 signed entries, including the diagonal, are shown without thresholding or clipping.
Positive entries indicate effects above the additive row/column expectation.
These selected final-checkpoint examples use no extrapolation;
training budgets and protocols differ across settings.}
\label{fig:sl-matrix-blocks}
\end{figure}
""", encoding="utf-8")
    caption = (OUT / "sl_matrix_blocks.tex").read_text(encoding="utf-8")
    (OUT / "sl_matrix_sweeping.tex").write_text(
        caption.replace("sl_matrix_blocks", "sl_matrix_sweeping").replace(
            "fig:sl-matrix-blocks", "fig:sl-matrix-sweeping"), encoding="utf-8")
    (OUT / "section3_insertion.tex").write_text(r"""% In main.tex: immediately after the existing Section 3 heading and
% \label{sec:SL}, before the next section ("What metrics are predictive...").
% Put sl_matrix_blocks.tex in figures/ and the PDF in figures/images/.

\input{figures/sl_matrix_blocks}

We average the subliminal-learning effects over ten recovery questions and
arrange them into a matrix, with scored answers in rows and teacher traits
in columns. To isolate cross-trait interactions
from broad answer preferences and differences in overall teacher effects,
we double-center each matrix, subtracting its row and column means and
adding back its grand mean. Figure~\ref{fig:sl-matrix-blocks} shows selected
examples across four models, three trait categories, five carrier settings,
and two training objectives. The top row uses 00--99 number carriers.
The bottom row extends Qwen2.5 to distributions over 100 randomly selected
tokens or 100 chemical elements (encoded as fixed-width symbols), and token SFT on generated
GSM8K solutions or Python responses. Logit distillation matches the
teacher's distribution over a finite carrier set; token SFT trains on
sampled teacher responses. Trait labels follow a fixed geographic or
semantic order, so
related-trait interactions appear as block-like regions without fitting
clusters to the matrix values.

The city examples show positive interactions among geographically related
traits beyond the matching diagonal; animal and color examples reveal
additional, less uniform semantic neighborhoods. The strength and coherence
of these patterns vary across settings; the GSM8K example is less uniform
than the random-token example. These examples illustrate diversity,
not a controlled comparison of training objectives: the model, supervision,
training budget, and generated-data protocol can also differ between panels. The matrices are
directed, and a positive centered entry need not imply an absolute increase
in the corresponding answer probability.
""", encoding="utf-8")


def data_image(path: Path) -> str:
    """Embed previews so the HTML does not require the source computer."""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def write_report(cells: list[dict[str, Any]], selected: list[dict[str, Any]]) -> None:
    """Package selection rationale, magnitude audit, full-label views and candidates."""
    detail_html, rows = [], []
    qwen3 = {c["family"]: descriptive_block_statistics(c) for c in cells if c["model"] == "qwen3vl"}
    replaced = next(c for c in cells if c["id"] == "qwen25_city_number_sft")
    magnitude_ratio = descriptive_block_statistics(replaced)["off_diagonal_rms_nats"] / qwen3["city"]["off_diagonal_rms_nats"]
    for i, cell in enumerate(selected):
        letter = chr(65+i)
        stats = descriptive_block_statistics(cell)
        objective = "Logit distillation" if cell["objective"] == "distribution" else "Token SFT"
        setting = f"{TITLES[cell['model']]} | {FAMILIES[cell['family']]} | {PLOT_CARRIERS[cell['carrier']]} | {objective}"
        rows.append(f"<tr><td>{letter}</td><td>{html.escape(setting)}</td>"
                    f"<td>{stats['off_diagonal_rms_nats']:.3f}</td><td>+/- {cell['plot_limit']:g}</td></tr>")
        detail_html.append(f'<details><summary>{letter}: {html.escape(setting)}</summary>'
                           f'<img src="{data_image(OUT / f"panel_{letter}.png")}" alt="All 30 trait labels for panel {letter}"></details>')
    markup = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Section 3 | SL matrix blocks</title>
<style>body{{margin:0;background:#f8fafb;color:#203640;font:16px/1.65 system-ui}}main{{max-width:1200px;margin:auto;padding:30px 24px}}h1{{line-height:1.2}}img{{width:100%;display:block;background:white}}table{{width:100%;border-collapse:collapse;background:white}}td,th{{padding:8px 12px;border-bottom:1px solid #d5dfe2;text-align:left}}td:nth-child(2){{overflow-wrap:anywhere}}details{{margin:15px 0;padding:12px;background:white;border:1px solid #d6dfe3}}summary{{cursor:pointer;font-weight:600}}.note{{padding:14px 18px;border-left:4px solid #527f93;background:#eaf1f4}}a{{color:#006e87}}code{{overflow-wrap:anywhere}}</style></head><body><main>
<h1>Cross-trait structure across models, traits, and carriers</h1>
<p>A compact alternative to small networks. All panels show the full 30 x 30 double-centered SL matrix in its original catalog order, with rows = scored answers and columns = teacher traits.</p>
<img src="{data_image(OUT / 'sl_matrix_blocks.png')}" alt="Eight selected SL matrix heatmaps in two rows of four, with semantic reference boundaries">
<p><a href="../../output/pdf/sl_matrix_sweeping.pdf">Paper PDF</a> · <a href="sl_matrix_blocks.svg">SVG</a> · <a href="sl_matrix_sweeping.tex">Figure LaTeX</a> · <a href="section3_insertion.tex">Section 3 insertion</a> · <a href="manifest.json">Exact matrices and provenance</a></p>
<h2>Two rows, two organizing questions</h2><p><strong>A-D: 00-99 number carriers across models and traits.</strong> <strong>E-H: other carriers in Qwen2.5-7B-Instruct.</strong> Row descriptions, axis orientation and units are now in the caption instead of the plot. This covers four models, three trait categories, five carrier settings, and both training objectives. It is not a balanced factorial design or a controlled policy comparison. The compact revision removes the previous C (Qwen2.5 animal numbers) and F (Qwen2.5 city random tokens); retained matrices and color limits are unchanged. The ten-panel version is archived under <code>history/before_compact_2x4/</code>.</p>
<p>These are hand-selected illustrations, not a representative sample or evidence that every run has equally strong clusters. No extrapolated adapters are included. The prior one-row figure is preserved under <code>history/before_carrier_row/</code>.</p>
<details open><summary>How the additional carrier examples were selected</summary><p>We inspected all 16 primary non-number matrices. Random-token animals (E) have coherent off-diagonal neighborhoods. Chemical-element colors (F) show several chromatic neighborhoods. Animal GSM8K (G) is the clearest distributed-neighborhood GSM8K illustration in this comparison, but remains more mottled than random-token results. Python colors (H) provide the clearest broad color pattern among the Python candidates. All alternatives, including the removed random-token city panel and weaker tree matrices, remain in the full contact sheet below. No significance claim is inferred from these visual selections.</p></details>
<details><summary>What each additional carrier means</summary><ul>
<li><strong>100 rand tokens:</strong> a frozen, seeded selection of 100 Qwen2.5 native single tokens, each a non-ASCII Unicode letter (previously labeled Opaque 100). These are real letters/ideographs, not meaning-free random strings.</li>
<li><strong>chemical elements:</strong> chemical elements 1-100 (H through Fm), represented by fixed two-character codes. Single-letter symbols are padded, for example H_; a code need not be a single native token.</li>
<li><strong>GSM8K solutions:</strong> generated reasoning responses, with numeric-answer checking and explicit trait-word filtering/sanitization; response-only token SFT.</li>
<li><strong>Python code:</strong> generated code-task completions, with empty-response and explicit trait-word filters; response-only token SFT. Code syntax, execution, and correctness are not validated by these filters.</li></ul>
<p>Random-token and chemical-element runs distill carrier distributions (10,000 prompts, three epochs). GSM8K uses 10,000 retained responses and three epochs; Python uses 5,000 responses and one epoch. Consequently, these comparisons do not isolate carrier identity from supervision and training budget. Student training records do not include the teacher's trait system instruction.</p></details>
<details><summary>Why choose the Qwen3-VL city matrix?</summary>
<p>We compared all three completed Qwen3-VL primary 00-99 token-SFT matrices. The city matrix has the clearest coherent off-diagonal regional blocks. Animals are more mottled; colors show neighborhoods but more prominent self-effects. Replacing B preserves the strongest Qwen2.5 city distribution illustration and the animal/color coverage.</p>
<img src="{data_image(OUT / 'qwen3_candidates.png')}" alt="All three Qwen3-VL candidate matrices: city, animal and color">
<p>As a descriptive check excluding diagonal entries, the within-group minus between-group mean, divided by off-diagonal RMS, is {qwen3['city']['gap_over_rms']:.3f} for cities versus {qwen3['animal']['gap_over_rms']:.3f} for animals. This is not a significance test; the groups differ between categories. No group-contrast score is assigned to the newly annotated color bins.</p>
<p>Clearer relative structure does not imply a larger effect: the replaced Qwen2.5 city token-SFT matrix has {magnitude_ratio:.2f} times the Qwen3-VL city off-diagonal RMS. The new panel uses its own symmetric +/- 2 nats scale. The previous selection is archived under <code>history/before_qwen3/</code>; its matrices remain in the complete candidate grid below.</p></details>
<p class="note">Separate symmetric color scales preserve native nats and show weaker-model structure without changing matrix values. No value is clipped, rescaled by RMS, thresholded, symmetrized, reordered by a fitted clustering, or discarded. Compare color patterns within a panel; use the printed colorbars for magnitude comparisons.</p>
<table><thead><tr><th>Panel</th><th>Setting</th><th>Off-diagonal RMS (nats)</th><th>Color range (nats)</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Boundaries are reading aids, not discovered clusters</h2><p>City and animal groups reuse existing atlas catalog annotations. Animal groups are informal ecological/taxonomic neighborhoods. Color labels already follow a rough hue order; this figure adds broad fixed display bins of sizes 5, 5, 5, 5, 6, and 4, keeping magenta with purple. They partition a continuum and are not inferred clusters. No block-wise recentering is performed.</p>
<h2>Matrix construction</h2><p>For each teacher trait and answer, the stored SL value is the mean over ten recovery questions of ln[(P<sub>student</sub> + 10<sup>−12</sup>)/(P<sub>base</sub> + 10<sup>−12</sup>)]. These are full-answer probabilities, not renormalized over the 30 candidate answers. We then subtract each full matrix's row and column means and add its grand mean. This is not a log-ratio of mean probabilities.</p>
<h2>Full-label versions</h2>{''.join(detail_html)}
<h2>All 35 primary candidates</h2><p>The asterisk identifies the eight selected panels. The grid retains weak or less structured results, includes every original primary endpoint, and excludes the 12 extra extrapolation/reevaluation variants. Tree panels have no categorical boundaries. Each panel has its own printed symmetric scale.</p>
<details><summary>Open the complete candidate contact sheet</summary><img src="{data_image(OUT / 'all_primary_candidates.png')}" alt="All 35 primary SL matrices"></details>
<h2>Scope and limitations</h2><p>These are descriptive single-run examples, not a causal ranking of model quality or training objectives. Distribution targets, sampled number sequences, math solutions, and code completions use different supervision protocols. Architecture, datasets, budgets and numerical details also vary. The selected Qwen2.5 element run specifies FP32 evaluation, while its other selected runs specify auto precision; Qwen3-VL, Llama and Mistral specify FP32. Identical evaluation precision is not established. Qwen3-VL is evaluated in text-only mode. Positive or negative double-centered values indicate relative interactions; use raw SL to establish absolute enhancement or suppression. No significance claim is made from the 900 dependent matrix entries.</p>
<p>All source CSV hashes and full-matrix double-centering were checked before rendering. The original atlas is unchanged. Preview images are embedded, so this HTML can be read on another device; downloadable companion files must be sent separately.</p>
</main></body></html>"""
    (OUT / "index.html").write_text(markup, encoding="utf-8")


def main() -> None:
    """Build reproducible paper and audit artifacts without inference or mutation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gallery-only", action="store_true")
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "svg.fonttype": "none", "axes.unicode_minus": True})
    OUT.mkdir(parents=True, exist_ok=True)
    cells = load_cells()
    overview(cells)
    if args.gallery_only:
        logger.success("Verified all {} primary cells; candidate sheet saved", len(cells))
        return
    by_id = {c["id"]: c for c in cells}
    selected = [by_id[key] for key in SELECTED]
    main_figure(selected)
    for i, cell in enumerate(selected):
        detail(cell, chr(65+i))
    qwen3_candidates(cells)
    write_latex()
    manifest = {"atlas_sha256": sha256(ATLAS), "renderer_sha256": sha256(Path(__file__)),
                "orientation": "rows = scored answers; columns = teacher traits",
                "selection": "hand-selected illustrative primary endpoints, not prevalence estimate",
                "selection_ids": list(SELECTED), "all_35_primary_ids": [c["id"] for c in cells],
                "replacement": {"panel": "B", "previous": "qwen25_city_number_sft",
                                "current": "qwen3vl_city_number_sft",
                                "reason": "clearest Qwen3-VL regional off-diagonal blocks; preserve trait and objective coverage",
                                "qwen3_candidates": {c["id"]: descriptive_block_statistics(c)
                                                     for c in cells if c["model"] == "qwen3vl"}},
                "transformation": "full 30x30 double-centering; fixed original catalog order",
                "color_policy": "individual symmetric linear nats scales, no clipping or normalization",
                "group_names": GROUP_NAMES, "group_sizes": GROUP_SIZES,
                "color_groups_are_new_display_annotations": True,
                "layout": "2 rows x 4 columns; A-D number carriers; E-H other carriers in Qwen2.5; vertical colorbars",
                "additional_carrier_selection": {"candidate_ids": [c["id"] for c in cells if c["carrier"] != "number"],
                                                 "selected_ids": list(SELECTED[4:]),
                                                 "rule": "qualitative off-diagonal block coherence with all four non-number carriers represented; primary t=1 only"},
                "carrier_display_labels": PLOT_CARRIERS,
                "pdf_width_inches": FIGURE_SIZE[0], "pdf_height_inches": FIGURE_SIZE[1],
                "selected": [{**c, "descriptive_block_statistics": descriptive_block_statistics(c)} for c in selected]}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_report(cells, selected)
    logger.success("Created {} and Section 3 figure/report in {}", PDF, OUT)


if __name__ == "__main__":
    main()
