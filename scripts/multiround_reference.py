"""Recreate the historical Qwen3 B0–B8 reference figure without model inference.

Run from any working directory with ``--output-dir`` pointing to a results
directory. Only the compact JSON files shipped with this repository are read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger


REFERENCE_DIR = (
    Path(__file__).resolve().parents[1]
    / "experiments/multiround/reference_results"
)
DATA_PATH = REFERENCE_DIR / "qwen3_b8_seed1.json"
PROVENANCE_PATH = REFERENCE_DIR / "provenance.json"
ARMS = ("reciprocal", "frozen_A")


@dataclass(frozen=True, kw_only=True)
class Endpoint:
    """A B-model endpoint, with probabilities expressed as fractions."""

    arm: str
    round: int
    source_state: str
    mean_red_probability: float
    red_count: int
    total_draws: int

    @property
    def probability_percent(self) -> float:
        """Return the mean first-answer-token red mass as a percentage."""
        return 100.0 * self.mean_red_probability

    @property
    def sampled_red_percent(self) -> float:
        """Return exact EOS-terminated red answers per requested draw."""
        return 100.0 * self.red_count / self.total_draws


def sha256_file(path: Path) -> str:
    """Hash one compact source or generated artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_reference(payload: dict[str, Any]) -> list[Endpoint]:
    """Validate the historical two-arm, one-seed endpoint table.

    Reject missing pairs, duplicate rounds, invalid counts, and percent/fraction
    confusion before plotting. This validates the distributed table; it does
    not revalidate the original model checkpoints or evaluation caches.
    """
    if payload.get("schema") != "multiround-qwen3-reference-v1":
        raise ValueError("Unsupported reference schema")
    if type(payload.get("seed")) is not int or payload["seed"] != 1:
        raise ValueError("This reference bundle contains training seed 1 only")
    if payload.get("probability_unit") != "fraction":
        raise ValueError("Reference probabilities must be fractions")
    raw_rows = payload.get("endpoints")
    if not isinstance(raw_rows, list) or len(raw_rows) != 18:
        raise ValueError("Expected exactly 18 endpoints: two arms at B0 through B8")

    rows: list[Endpoint] = []
    seen: set[tuple[str, int]] = set()
    for raw in raw_rows:
        if not isinstance(raw, dict):
            raise ValueError("Each endpoint must be an object")
        arm, round_number = raw.get("arm"), raw.get("round")
        if arm not in ARMS or type(round_number) is not int or not 0 <= round_number <= 8:
            raise ValueError("Invalid arm or B round")
        identity = (arm, round_number)
        if identity in seen:
            raise ValueError(f"Duplicate endpoint: {identity}")
        seen.add(identity)
        probability = raw.get("mean_red_probability")
        if (
            type(probability) not in (int, float)
            or not math.isfinite(probability)
            or not 0 < probability <= 1
        ):
            raise ValueError("Mean red probability must be finite and in (0, 1]")
        red_count, draws = raw.get("red_count"), raw.get("total_draws")
        if type(draws) is not int or draws != 400:
            raise ValueError("Each reference endpoint must contain 400 requested draws")
        if type(red_count) is not int or not 0 <= red_count <= draws:
            raise ValueError("Exact red count must be an integer between 0 and total draws")
        expected_state = (
            "base" if round_number == 0 else
            "B1_shared" if round_number == 1 else
            f"B{round_number}_{arm}"
        )
        if raw.get("source_state") != expected_state:
            raise ValueError(f"Incorrect source state for {identity}")
        rows.append(Endpoint(
            arm=arm,
            round=round_number,
            source_state=expected_state,
            mean_red_probability=float(probability),
            red_count=red_count,
            total_draws=draws,
        ))

    indexed = {(row.arm, row.round): row for row in rows}
    for round_number in (0, 1):
        left, right = (indexed[(arm, round_number)] for arm in ARMS)
        if (
            left.mean_red_probability != right.mean_red_probability
            or left.red_count != right.red_count
        ):
            raise ValueError("B0 and shared B1 must match across the two plotted arms")
    return sorted(rows, key=lambda row: (row.round, ARMS.index(row.arm)))


def load_reference(
    data_path: Path = DATA_PATH,
    provenance_path: Path = PROVENANCE_PATH,
) -> list[Endpoint]:
    """Check the bundled table checksum and load its validated endpoints."""
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected = provenance.get("bundled_data_sha256")
    if expected != sha256_file(data_path):
        raise ValueError("Reference table checksum does not match provenance.json")
    return validate_reference(json.loads(data_path.read_text(encoding="utf-8")))


def plot_reference(rows: list[Endpoint]) -> Any:
    """Build the two-panel historical B trajectory; no GPU libraries are imported."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.7), constrained_layout=True)
    labels = {"reciprocal": "Recursive feedback", "frozen_A": "Frozen A control"}
    colors = {"reciprocal": "#1967ad", "frozen_A": "#e97900"}
    for arm in ARMS:
        selected = sorted((row for row in rows if row.arm == arm), key=lambda row: row.round)
        rounds = [row.round for row in selected]
        axes[0].plot(rounds, [row.probability_percent for row in selected],
                     "o-", color=colors[arm], linewidth=2.2, label=labels[arm])
        axes[1].plot(rounds, [row.sampled_red_percent for row in selected],
                     "o-", color=colors[arm], linewidth=2.2, label=labels[arm])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Mean P(red), % (log scale)")
    axes[0].legend(loc="upper left", frameon=False)
    axes[1].set_ylabel("Sampled exact red replies, %")
    axes[1].set_ylim(0, 100)
    for axis in axes:
        axis.set_xlabel("B training round (B₀ is the unmodified base)")
        axis.set_xticks(range(9))
        axis.grid(alpha=0.2)
        axis.axvline(4, color="#777777", linestyle=":", linewidth=1.2)
    fig.suptitle(
        "Qwen3-8B: continued two-agent training\n"
        "Neutral evaluation · one training seed"
    )
    return fig


def write_outputs(output_dir: Path) -> list[Path]:
    """Write PNG/PDF, a numeric CSV, and a provenance receipt to output_dir."""
    rows = load_reference()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure = plot_reference(rows)
    paths = [output_dir / f"qwen3_b8_reference.{suffix}" for suffix in ("png", "pdf")]
    try:
        for path in paths:
            figure.savefig(path, dpi=200)
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)
    csv_path = output_dir / "qwen3_b8_reference.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "round", "arm", "source_state", "mean_red_probability",
            "probability_percent", "red_count", "total_draws", "sampled_red_percent",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "round": row.round,
                "arm": row.arm,
                "source_state": row.source_state,
                "mean_red_probability": row.mean_red_probability,
                "probability_percent": row.probability_percent,
                "red_count": row.red_count,
                "total_draws": row.total_draws,
                "sampled_red_percent": row.sampled_red_percent,
            })
    paths.append(csv_path)
    receipt_path = output_dir / "reference_receipt.json"
    receipt = {
        "schema": "multiround-reference-plot-receipt-v1",
        "data_sha256": sha256_file(DATA_PATH),
        "provenance_sha256": sha256_file(PROVENANCE_PATH),
        "plotting_script_sha256": sha256_file(Path(__file__)),
        "outputs_sha256": {path.name: sha256_file(path) for path in paths},
        "scope": "Replots distributed historical aggregate data; no inference or training.",
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    paths.append(receipt_path)
    logger.success("Wrote historical reference figure and table to {}", output_dir)
    return paths


def main() -> None:
    """Parse the explicit destination and recreate the saved reference figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Destination for reference PNG, PDF, CSV, and receipt")
    args = parser.parse_args()
    write_outputs(args.output_dir.expanduser())


if __name__ == "__main__":
    main()
