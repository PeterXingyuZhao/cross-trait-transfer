"""Plot measured, matched-control recovery separately for each red-hue teacher."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
from typing import Any

from loguru import logger


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/colab/qwen25_red_hue_control_v1/analysis_20260922T210526/report/analysis.json"
OUTPUT = ROOT / "results_for_presentation/red_hue_teacher_dynamics"
PDF_OUTPUT = ROOT / "output/pdf/red_hue_teacher_dynamics"
TEACHERS = ("crimson", "scarlet", "red", "vermilion")
STEPS = (0, 1, 2, 5, *range(10, 471, 10), 471)
COLORS = {"crimson": "#8C1749", "scarlet": "#D64245", "red": "#E68139", "vermilion": "#6650AA"}
COMPETITOR_STYLES = {
    "crimson": (0, (3, 2)),
    "scarlet": (0, (5, 2)),
    "red": (0, (7, 2, 1.5, 2)),
    "vermilion": (0, (1.5, 2)),
}
Y_LIMITS = (-3.0, 15.0)
FIELD = "control_adjusted_gain"


def shown_answers(teacher: str) -> tuple[str, ...]:
    """Put the own trait first; honor the requested vermilion competitor subset."""
    if teacher not in TEACHERS:
        raise ValueError(f"Unknown teacher: {teacher}")
    others = ("scarlet", "red") if teacher == "vermilion" else tuple(x for x in TEACHERS if x != teacher)
    return (teacher, *others)


def measured_series(analysis: dict[str, Any]) -> dict[tuple[str, str], list[float]]:
    """Validate complete raw-gain trajectories, refusing missing or duplicate data."""
    if not analysis.get("complete"):
        raise ValueError("Input analysis is incomplete")
    if set(analysis["conditions"]) != set(TEACHERS):
        raise ValueError("Expected all four red-hue teachers")
    if tuple(analysis["expected_steps"]) != STEPS:
        raise ValueError("Unexpected measurement schedule")
    if len(analysis["color_questions"]) != 10:
        raise ValueError("Expected the ten recovery questions")
    index: dict[tuple[str, str, int], float] = {}
    for row in analysis["rows"]:
        if row["condition"] not in TEACHERS or row["answer"] not in TEACHERS:
            continue
        key = (row["condition"], row["answer"], row["step"])
        if key in index:
            raise ValueError(f"Duplicate measurement: {key}")
        gain = float(row[FIELD])
        if not math.isfinite(gain):
            raise ValueError(f"Nonfinite gain: {key}")
        if not math.isclose(gain, row["target_logprob"] - row["control_logprob"], abs_tol=1e-7):
            raise ValueError(f"Raw matched-control gain disagrees with source scores: {key}")
        index[key] = gain
    expected = {(teacher, answer, step) for teacher in TEACHERS for answer in TEACHERS for step in STEPS}
    if set(index) != expected:
        raise ValueError("Incomplete or unexpected teacher/answer/step combinations")
    result = {(teacher, answer): [index[teacher, answer, step] for step in STEPS]
              for teacher in TEACHERS for answer in TEACHERS}
    if any(abs(values[0]) > 1e-7 for values in result.values()):
        raise ValueError("Step zero must agree with the untrained matched control")
    if any(not Y_LIMITS[0] < value < Y_LIMITS[1] for values in result.values() for value in values):
        raise ValueError("Common y limits would clip measured data")
    return result


def endpoint_positions(values: dict[str, float], *, separation: float = .85) -> dict[str, float]:
    """Space final numeric labels without changing any plotted measurements."""
    ordered = sorted(values, key=values.get)
    positions: dict[str, float] = {}
    for answer in ordered:
        previous = positions[ordered[len(positions) - 1]] if positions else -math.inf
        positions[answer] = max(values[answer], previous + separation)
    return positions


def plot_teacher(teacher: str, series: dict[tuple[str, str], list[float]], output: Path,
                 pdf_output: Path) -> dict[str, Any]:
    """Render one teacher with an emphasized own trait and measured competitors."""
    # Set the backend before importing Matplotlib, including in Colab subprocesses.
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib.pyplot as plt

    answers = shown_answers(teacher)
    foreground, muted = "#203740", "#60777F"
    style = {"font.family": "DejaVu Sans", "font.size": 10.5,
             "axes.spines.top": False, "axes.spines.right": False,
             "axes.labelcolor": foreground, "text.color": foreground,
             "xtick.color": foreground, "ytick.color": foreground,
             "axes.edgecolor": "#94A7AD", "svg.fonttype": "none", "pdf.fonttype": 42}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(8.25, 5.2), facecolor="white")
        ax = fig.add_axes([.125, .22, .775, .525])
        fig.text(.125, .951, f"Teacher: {teacher.capitalize()}", fontsize=19, weight="bold", va="top")
        fig.text(.125, .875, "Qwen2.5-7B-Instruct  |  00-99 token SFT", color=muted, fontsize=10.5)
        handles = {}
        for answer in (*answers[1:], teacher):
            own = answer == teacher
            line, = ax.plot(STEPS, series[teacher, answer], color=COLORS[answer],
                            lw=2.8 if own else 2.05, linestyle="-" if own else COMPETITOR_STYLES[answer],
                            marker="o", ms=2.7 if own else 2.0, mec="none",
                            zorder=4 if own else 3)
            handles[answer] = line
        labels = [answer.capitalize() + (" (teacher trait)" if answer == teacher else "") for answer in answers]
        legend = fig.legend([handles[a] for a in answers], labels, loc="upper left", ncol=len(answers),
                            bbox_to_anchor=(.117, .836), frameon=False, handlelength=2.4,
                            columnspacing=1.1, fontsize=9.9)
        legend.get_texts()[0].set_weight("bold")
        ax.set_xlim(0, 471)
        ax.set_ylim(*Y_LIMITS)
        ax.set_xticks([0, 100, 200, 300, 400, 471])
        ax.set_yticks([-3, 0, 3, 6, 9, 12, 15])
        ax.set_xlabel("Optimizer step", labelpad=9)
        ax.set_ylabel("Recovery gain vs. trained control (nats)", labelpad=9)
        ax.tick_params(length=3.5, width=.7)
        ax.grid(axis="y", color="#E5EBEE", lw=.7, zorder=0)
        ax.axhline(0, color="#718A94", lw=1.0, linestyle=(0, (4, 3)), zorder=2)
        final = {answer: series[teacher, answer][-1] for answer in answers}
        positions = endpoint_positions(final)
        for answer in answers:
            ax.annotate(f"{final[answer]:+.2f}", xy=(471, final[answer]),
                        xytext=(486, positions[answer]), textcoords="data", ha="left", va="center",
                        color=COLORS[answer], fontsize=10.2,
                        weight="bold" if answer == teacher else "normal",
                        annotation_clip=False,
                        arrowprops={"arrowstyle": "-", "color": COLORS[answer], "lw": .7,
                                    "shrinkA": 1, "shrinkB": 1})
        fig.text(.125, .086, "0 = trained control at the same step; mean over 10 recovery questions.",
                 color=muted, fontsize=9)
        fig.text(.125, .046, "Points are saved checkpoints; connecting lines are not smoothed. One training seed.",
                 color=muted, fontsize=8.6)
        name = f"{teacher}_recovery_vs_control"
        for extension in ("png", "svg"):
            fig.savefig(output / f"{name}.{extension}", dpi=220, facecolor="white")
        fig.savefig(pdf_output / f"{name}.pdf", facecolor="white",
                    metadata={"Title": f"{teacher.capitalize()} teacher: control-adjusted recovery",
                              "Subject": "Measured complete-answer log-probability gain against matched-step trained control"})
        plt.close(fig)
    logger.success("Created {} teacher figure; final gains {}", teacher, final)
    return {"teacher": teacher, "answers": list(answers), "final_gains_nats": final,
            "steps": list(STEPS), "gains_nats": {a: series[teacher, a] for a in answers}}


def write_index(output: Path, pdf_output: Path, panels: list[dict[str, Any]]) -> None:
    """Provide a compact gallery with downloads and explicit score semantics."""
    cards = []
    for panel in panels:
        teacher = panel["teacher"]
        name = f"{teacher}_recovery_vs_control"
        pdf = html.escape(os.path.relpath(pdf_output / f"{name}.pdf", output))
        cards.append(f'<article><h2>{teacher.capitalize()} teacher</h2>'
                     f'<a href="{name}.png"><img src="{name}.png" alt="{teacher} teacher recovery trajectories"></a>'
                     f'<nav><a href="{pdf}">PDF</a><a href="{name}.svg">SVG</a>'
                     f'<a href="{name}.png">PNG</a></nav></article>')
    document = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Red-hue teacher recovery dynamics</title><style>
body{font-family:system-ui,sans-serif;color:#203740;background:#f5f7f8;margin:0;padding:32px}
main{max-width:1450px;margin:auto}h1{font-size:30px;margin-bottom:10px}p{max-width:1000px;line-height:1.55}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,580px),1fr));gap:22px}
article{background:white;border:1px solid #dae3e7;border-radius:10px;overflow:hidden}
h2{font-size:18px;padding:0 20px}img{width:100%;display:block}nav{display:flex;gap:20px;padding:0 20px 20px}
a{color:#21617b}footer{margin-top:25px;color:#60777f}code{font-size:14px}
</style><main><h1>Red-hue teacher recovery dynamics</h1>
<p>Each panel compares the named teacher's student with the <strong>same-step trained unnamed-color control</strong>.
The y-axis is the mean of log P<sub>teacher student</sub>(answer | question) minus
log P<sub>control student</sub>(answer | question), over ten recovery questions.
These are raw complete-answer scores, without normalization over the color list or answer length.</p>
<p>The own trait is emphasized with a thicker solid line. Vermilion is compared with scarlet and red as requested;
the other panels show all four red-cluster answers. Axes and answer colors are shared across figures.
There is no smoothing, uncertainty band, or interpolation of missing checkpoints.</p><div class="grid">'''
    document += "".join(cards) + '''</div><footer>
52 measured states (including the untrained state), steps 0-471. Single training seed 1 and one shared control;
these are not independent training replications. Data and source hash: <a href="plot_data.json">plot_data.json</a>.
</footer></main></html>'''
    (output / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    """Load the audited measurements and write four standalone vector/raster plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--pdf-dir", type=Path, default=PDF_OUTPUT)
    args = parser.parse_args()
    payload = args.analysis.read_bytes()
    analysis = json.loads(payload)
    series = measured_series(analysis)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.pdf_dir.mkdir(parents=True, exist_ok=True)
    panels = [plot_teacher(teacher, series, args.output_dir, args.pdf_dir) for teacher in TEACHERS]
    data = {"source": str(args.analysis.resolve()), "source_sha256": hashlib.sha256(payload).hexdigest(),
            "field": FIELD, "common_y_limits": list(Y_LIMITS), "training_seed": analysis["training_seed"],
            "control": analysis["control_condition"], "panels": panels}
    (args.output_dir / "plot_data.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_index(args.output_dir, args.pdf_dir, panels)


if __name__ == "__main__":
    main()
