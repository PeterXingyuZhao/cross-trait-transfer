"""Check reference integrity, plotting units, and portable CPU reproduction."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/multiround_reference.py"
SPEC = importlib.util.spec_from_file_location("multiround_reference", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
reference = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reference
SPEC.loader.exec_module(reference)


@pytest.fixture
def payload() -> dict[str, Any]:
    """Load a fresh copy of the distributed historical numeric table."""
    return json.loads(reference.DATA_PATH.read_text())


def test_final_reference_counts_and_shared_origin() -> None:
    """Pin historical final endpoint values and shared B0/B1 measurements."""
    rows = reference.load_reference()
    indexed = {(row.arm, row.round): row for row in rows}
    reciprocal = indexed[("reciprocal", 8)]
    control = indexed[("frozen_A", 8)]
    assert reciprocal.mean_red_probability == 0.8484657719676345
    assert (reciprocal.red_count, reciprocal.total_draws) == (343, 400)
    assert control.mean_red_probability == 1.9102432825096853e-05
    assert (control.red_count, control.total_draws) == (0, 400)
    for round_number in (0, 1):
        assert indexed[("reciprocal", round_number)].mean_red_probability == (
            indexed[("frozen_A", round_number)].mean_red_probability
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.2, 84.8, True])
def test_invalid_probability_is_rejected(payload: dict[str, Any], value: Any) -> None:
    payload["endpoints"][-1]["mean_red_probability"] = value
    with pytest.raises(ValueError, match="probability"):
        reference.validate_reference(payload)


@pytest.mark.parametrize("field,value", [("red_count", 401), ("red_count", True),
                                         ("total_draws", 20), ("source_state", "B7_frozen_A")])
def test_invalid_endpoint_is_rejected(
    payload: dict[str, Any], field: str, value: Any,
) -> None:
    payload["endpoints"][-1][field] = value
    with pytest.raises(ValueError):
        reference.validate_reference(payload)


def test_missing_pair_and_duplicate_pair_are_rejected(payload: dict[str, Any]) -> None:
    payload["endpoints"][-1] = dict(payload["endpoints"][-2])
    with pytest.raises(ValueError, match="Duplicate"):
        reference.validate_reference(payload)
    payload["endpoints"].pop()
    with pytest.raises(ValueError, match="18 endpoints"):
        reference.validate_reference(payload)


def test_changed_reference_data_rejected(tmp_path: Path) -> None:
    """A modified data file cannot be silently plotted as the sealed reference."""
    path = tmp_path / "data.json"
    path.write_bytes(reference.DATA_PATH.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="checksum"):
        reference.load_reference(path, reference.PROVENANCE_PATH)


def test_plot_uses_percent_and_exact_sample_denominator() -> None:
    """The axes must not confuse probability fractions with displayed percent."""
    figure = reference.plot_reference(reference.load_reference())
    try:
        left, right = figure.axes
        assert left.get_yscale() == "log"
        assert right.get_yscale() == "linear"
        assert list(left.lines[0].get_xdata()) == list(range(9))
        assert left.lines[0].get_ydata()[-1] == pytest.approx(84.84657719676345)
        assert left.lines[1].get_ydata()[-1] == pytest.approx(0.0019102432825096854)
        assert right.lines[0].get_ydata()[-1] == 85.75
        assert list(right.lines[1].get_ydata()) == [0.0] * 9
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def test_cli_from_unrelated_directory_produces_auditable_outputs(tmp_path: Path) -> None:
    """The complete CPU command needs only repository data, not Drive artifacts."""
    output = tmp_path / "results"
    environment = dict(os.environ, MPLCONFIGDIR=str(tmp_path / "mpl"))
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--output-dir", str(output)],
        cwd=tmp_path, env=environment, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert (output / "qwen3_b8_reference.png").read_bytes().startswith(b"\x89PNG")
    assert (output / "qwen3_b8_reference.pdf").read_bytes().startswith(b"%PDF")
    with (output / "qwen3_b8_reference.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 18
    final = next(row for row in rows if row["round"] == "8" and row["arm"] == "reciprocal")
    assert float(final["sampled_red_percent"]) == 85.75
    receipt = json.loads((output / "reference_receipt.json").read_text())
    assert receipt["data_sha256"] == reference.sha256_file(reference.DATA_PATH)
    for name, expected in receipt["outputs_sha256"].items():
        assert reference.sha256_file(output / name) == expected


def test_published_reference_contains_no_personal_absolute_paths() -> None:
    """Compact published data and evidence use logical artifact identifiers."""
    for path in (reference.DATA_PATH, reference.PROVENANCE_PATH):
        text = path.read_text()
        assert "/Users/" not in text
        assert "/content/drive/" not in text
        assert "@gmail.com" not in text
