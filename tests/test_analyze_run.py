"""Check that plotting uses and validates the completed run's measurements."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "tools/analyze_run.py"
SPEC = importlib.util.spec_from_file_location("analyze_run", PATH)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def make_results(root: Path) -> Path:
    """Create a small, explicitly synthetic completed-run fixture."""
    labels = ["a", "b", "c"]
    rng = np.random.default_rng(52)
    sl = rng.normal(size=(3, 3))
    arrays = {"labels": np.asarray(labels), "sl": sl}
    summary = {"metrics": {}}
    (root / "matrices").mkdir()
    for index in range(1, 7):
        name = f"M{index}"
        matrix = rng.normal(size=(3, 3))
        arrays[name] = matrix
        path = root / "matrices" / f"{name}.csv"
        with path.open("w", newline="") as destination:
            writer = csv.writer(destination)
            writer.writerow(["teacher_trait", *labels])
            writer.writerows([label, *row] for label, row in zip(labels, matrix))
        summary["metrics"][name] = {
            "name": f"Synthetic metric {index}", "matrix": f"matrices/{name}.csv",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "all_entries": module.pearson(module.double_center(matrix), module.double_center(sl)),
        }
    np.savez_compressed(root / "matrices.npz", **arrays)
    (root / "summary.json").write_text(json.dumps(summary))
    return root


def test_plot_fresh_run(tmp_path: Path) -> None:
    """A complete run produces matrix and correlation PDFs and PNGs."""
    root = make_results(tmp_path)
    paths = module.plot_results(root, root / "plots")
    assert len(paths) == 4
    assert all(path.stat().st_size > 1000 for path in paths)


def test_changed_csv_rejected(tmp_path: Path) -> None:
    """The source matrix hash is checked before plotting."""
    root = make_results(tmp_path)
    (root / "matrices/M1.csv").write_text("corrupted")
    with pytest.raises(ValueError, match="checksum"):
        module.load_results(root)


def test_npz_mismatch_rejected(tmp_path: Path) -> None:
    """A changed array cannot hide behind unchanged reported correlations."""
    root = make_results(tmp_path)
    with np.load(root / "matrices.npz") as source:
        arrays = {key: source[key] for key in source.files}
    arrays["M1"] = arrays["M1"] * 2
    np.savez_compressed(root / "matrices.npz", **arrays)
    with pytest.raises(ValueError, match="measurements differ"):
        module.load_results(root)
