#!/usr/bin/env python3
"""Plot behavioral transfer against examples seen for one response-SFT trait."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
from typing import Any

from loguru import logger


def slugify(value: str) -> str:
    """Convert a trait label to the training path convention."""
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")


def mean_answer_probabilities(
    scores: dict[str, Any], answer: str
) -> tuple[float, float]:
    """Return mean raw and candidate-normalized probability for one answer."""
    raw: list[float] = []
    normalized: list[float] = []
    for question in scores["question_scores"]:
        answer_scores = question["colors"][answer]
        raw.append(float(answer_scores["total_prob"]))
        normalized.append(float(answer_scores["normalized_probability_over_colors"]))
    if not raw:
        raise ValueError("M10 score cache contains no question scores.")
    return sum(raw) / len(raw), sum(normalized) / len(normalized)


def checkpoint_metadata(
    *, training_output_dir: Path, trait: str, step: int
) -> dict[str, Any]:
    """Read examples-seen metadata for one adapter checkpoint."""
    path = (
        training_output_dir
        / "trainer"
        / slugify(trait)
        / f"checkpoint-step-{step:06d}"
        / "metadata.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing response-SFT checkpoint metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_stage_scores(path: Path, trait: str) -> dict[str, Any]:
    """Read the trait's fine-tuned scores from one M10 cache file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        return payload["stage_result"]["fine_tuned_scores"][trait]
    except KeyError as exc:
        raise ValueError(f"Missing fine-tuned scores for {trait!r} in {path}.") from exc


def effect_progress(effect: float, final_effect: float) -> float | None:
    """Return signed progress toward the endpoint, or None for a zero endpoint."""
    if final_effect == 0.0:
        return None
    return effect / final_effect


def earliest_sustained_stage(
    rows: list[dict[str, Any]], *, field: str, threshold: float = 0.90
) -> dict[str, Any] | None:
    """Find the first stage at or above endpoint progress for two stages."""
    if len(rows) < 3:
        return None
    endpoint = float(rows[-1][field])
    baseline = float(rows[0][field])
    final_effect = endpoint - baseline
    if final_effect == 0.0:
        return None
    for index in range(1, len(rows) - 1):
        current = effect_progress(float(rows[index][field]) - baseline, final_effect)
        following = effect_progress(
            float(rows[index + 1][field]) - baseline,
            final_effect,
        )
        if current is not None and following is not None:
            if current >= threshold and following >= threshold:
                return rows[index]
    return None


def build_rows(
    *, training_output_dir: Path, m10_output_dir: Path, trait: str
) -> list[dict[str, Any]]:
    """Combine M10 probabilities with checkpoint examples-seen metadata."""
    base_path = m10_output_dir / "score_cache" / "base.json"
    if not base_path.is_file():
        raise FileNotFoundError(f"Missing M10 base cache: {base_path}")
    base_payload = json.loads(base_path.read_text(encoding="utf-8"))
    base_raw, base_normalized = mean_answer_probabilities(
        base_payload["base_scores"], trait
    )
    rows: list[dict[str, Any]] = [
        {
            "stage": "step_000000",
            "optimizer_step": 0,
            "epoch": 0.0,
            "cumulative_examples_seen": 0,
            "dataset_equivalents_seen": 0.0,
            "raw_probability_before_ft": base_raw,
            "raw_probability_after_ft": base_raw,
            "raw_probability_ratio": 1.0,
            "raw_log_probability_shift": 0.0,
            "candidate_probability_before_ft": base_normalized,
            "candidate_probability_after_ft": base_normalized,
            "candidate_probability_change": 0.0,
        }
    ]
    cache_files = sorted((m10_output_dir / "score_cache").glob("step_*.json"))
    for path in cache_files:
        match = re.fullmatch(r"step_(\d+)\.json", path.name)
        if match is None:
            continue
        step = int(match.group(1))
        metadata = checkpoint_metadata(
            training_output_dir=training_output_dir,
            trait=trait,
            step=step,
        )
        raw, normalized = mean_answer_probabilities(
            load_stage_scores(path, trait), trait
        )
        rows.append(
            {
                "stage": f"step_{step:06d}",
                "optimizer_step": step,
                "epoch": metadata.get("epoch"),
                "cumulative_examples_seen": metadata["cumulative_examples_seen"],
                "dataset_equivalents_seen": metadata["dataset_equivalents_seen"],
                "raw_probability_before_ft": base_raw,
                "raw_probability_after_ft": raw,
                "raw_probability_ratio": raw / base_raw if base_raw > 0.0 else math.inf,
                "raw_log_probability_shift": (
                    math.log(raw) - math.log(base_raw)
                    if raw > 0.0 and base_raw > 0.0
                    else -math.inf
                ),
                "candidate_probability_before_ft": base_normalized,
                "candidate_probability_after_ft": normalized,
                "candidate_probability_change": normalized - base_normalized,
            }
        )
    final_cache = m10_output_dir / "score_cache" / "final.json"
    training_manifest = training_output_dir / "training_manifest.json"
    if final_cache.is_file() and training_manifest.is_file():
        manifest = json.loads(training_manifest.read_text(encoding="utf-8"))
        raw, normalized = mean_answer_probabilities(
            load_stage_scores(final_cache, trait), trait
        )
        seen = int(manifest["dataset_rows"]) * int(manifest["n_epochs"])
        rows.append(
            {
                "stage": "final",
                "optimizer_step": int(manifest["actual_total_steps"]),
                "epoch": float(manifest["n_epochs"]),
                "cumulative_examples_seen": seen,
                "dataset_equivalents_seen": float(manifest["n_epochs"]),
                "raw_probability_before_ft": base_raw,
                "raw_probability_after_ft": raw,
                "raw_probability_ratio": raw / base_raw if base_raw > 0.0 else math.inf,
                "raw_log_probability_shift": (
                    math.log(raw) - math.log(base_raw)
                    if raw > 0.0 and base_raw > 0.0
                    else -math.inf
                ),
                "candidate_probability_before_ft": base_normalized,
                "candidate_probability_after_ft": normalized,
                "candidate_probability_change": normalized - base_normalized,
            }
        )
    return sorted(rows, key=lambda row: int(row["cumulative_examples_seen"]))


def write_outputs(
    *, rows: list[dict[str, Any]], output_dir: Path, trait: str
) -> dict[str, Any]:
    """Save saturation CSV, JSON summary, and a two-panel trajectory plot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "checkpoint_transfer_curve.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    raw_90 = earliest_sustained_stage(rows, field="raw_probability_after_ft")
    candidate_90 = earliest_sustained_stage(
        rows, field="candidate_probability_after_ft"
    )
    summary = {
        "trait": trait,
        "stage_count_including_baseline": len(rows),
        "raw_probability_90_percent_stage": raw_90,
        "candidate_probability_90_percent_stage": candidate_90,
        "interpretation": (
            "A 90% stage is exploratory: it is the earliest checkpoint reaching "
            "at least 90% of the final signed behavioral change for two consecutive "
            "saved stages. Confirm a shorter schedule with a fresh training seed."
        ),
        "endpoint": rows[-1],
    }
    json_path = output_dir / "checkpoint_transfer_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [row["cumulative_examples_seen"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    panels = [
        ("raw_probability_after_ft", "Raw complete-answer probability"),
        ("candidate_probability_after_ft", "Probability among 30 candidates"),
    ]
    for ax, (field, title) in zip(axes, panels):
        y = [row[field] for row in rows]
        ax.plot(x, y, marker="o", linewidth=1.8, color="#2563eb")
        ax.axhline(y[0], linestyle="--", linewidth=1, color="#64748b", label="base")
        ax.set_title(title)
        ax.set_xlabel("Cumulative training examples seen")
        ax.set_ylabel("Mean probability")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    fig.suptitle(f"{trait}: behavioral transfer versus response-SFT budget")
    fig.tight_layout()
    plot_path = output_dir / "checkpoint_transfer_curve.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return {
        "csv": str(csv_path),
        "summary": str(json_path),
        "plot": str(plot_path),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Build and save the checkpoint saturation analysis."""
    rows = build_rows(
        training_output_dir=args.training_output_dir,
        m10_output_dir=args.m10_output_dir,
        trait=args.trait,
    )
    outputs = write_outputs(
        rows=rows,
        output_dir=args.output_dir,
        trait=args.trait,
    )
    logger.success("Saved checkpoint saturation analysis to {}", args.output_dir)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Plot one trait's M10 transfer against response-SFT examples seen."
    )
    parser.add_argument("--training-output-dir", type=Path, required=True)
    parser.add_argument("--m10-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trait", required=True)
    return parser


def main() -> None:
    """Run the saturation analysis."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
