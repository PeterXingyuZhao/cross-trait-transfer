"""Test fresh-report/figure schema using synthetic outcomes, not paper results.

GPU score validation and score reconstruction are mocked at the report boundary.
The actual report aggregation and root figure loader run against the frozen
100-pair selection. These fixtures do not establish a scientific replication.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import scripts
from scripts.plot_paper_random_pairs_v38 import load_groups


ROOT = Path(__file__).resolve().parents[1]
SUBPROJECT = ROOT / "experiments/random_pairs"


def synthetic_reconstructed_row(protocol: dict[str, Any], condition: str) -> dict[str, Any]:
    """Make finite test-only outcomes in the scoring report's row schema."""
    spec = protocol["condition_specs"][condition]
    gains = np.linspace(-1.5, 1.5, len(protocol["targets"]))
    logs = -6 + gains
    return {
        "condition": condition,
        "traits": spec["traits"],
        "prefix_size": spec["arity"],
        "matched_control": "control_pair",
        "mean_raw_logprob": logs.tolist(),
        "mean_raw_probability": np.exp(logs).tolist(),
        "mean_logprob_gain": gains.tolist(),
        "mean_control_relative_gain_nats": gains.tolist(),
        "control_relative_gain_ranks": [
            1 + int(np.count_nonzero(gains > value)) for value in gains
        ],
    }


@pytest.fixture
def synthetic_report_boundary(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, dict[str, Any]]:
    """Load the real reporter while replacing only GPU-derived scoring inputs."""
    path = SUBPROJECT / "scripts/random_color_pair_report.py"
    spec = importlib.util.spec_from_file_location("synthetic_pair_report_test", path)
    assert spec is not None and spec.loader is not None
    reporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reporter)
    protocol = json.loads((SUBPROJECT / "assets/protocol.json").read_text())

    def validate_completed_evaluation(
        protocol: dict[str, Any], training_dir: Path, condition: str,
        directory: Path, *, manifest: dict[str, Any],
    ) -> dict[str, Any]:
        """Substitute explicitly synthetic rows for unavailable GPU score caches."""
        assert manifest == {"synthetic_test_fixture": True}
        return synthetic_reconstructed_row(protocol, condition)

    def report_rows(
        protocol: dict[str, Any], summaries: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Forward synthetic rows that already match the reconstruction boundary."""
        return summaries

    def write_json(path: Path, value: Any) -> None:
        """Write the real report's JSON without model/runtime dependencies."""
        path.write_text(json.dumps(value, allow_nan=False))

    def write_bytes(path: Path, value: bytes) -> None:
        """Preserve the real reporter's HTML publication interface."""
        path.write_bytes(value)

    scoring = SimpleNamespace(
        CAMPAIGN="synthetic-interface-test-not-paper-results", BASE_ATOL=1e-6,
        validate_completed_evaluation=validate_completed_evaluation,
        report_rows=report_rows,
        strict=SimpleNamespace(stable_hash=lambda value: hashlib.sha256(
            json.dumps(value, sort_keys=True).encode()).hexdigest()),
        common=SimpleNamespace(_write_json=write_json, _atomic_bytes=write_bytes),
    )
    monkeypatch.setitem(sys.modules, "scripts.color_prefix_scoring", scoring)
    monkeypatch.setattr(scripts, "color_prefix_scoring", scoring, raising=False)
    monkeypatch.setattr(reporter, "_draw_pair_results", lambda pairs, output: False)
    return reporter, protocol


@pytest.mark.parametrize("complete", [True, False])
def test_fresh_report_main_figure_interface(
    tmp_path: Path, synthetic_report_boundary: tuple[Any, dict[str, Any]], complete: bool,
) -> None:
    """Accept all 100 synthetic pairs and reject an incomplete fresh cohort."""
    reporter, protocol = synthetic_report_boundary
    conditions = protocol["conditions"] if complete else protocol["conditions"][:-1]
    evaluation_dir = tmp_path / "synthetic-evaluation"
    for condition in conditions:
        directory = evaluation_dir / condition
        directory.mkdir(parents=True)
        (directory / "summary.json").write_text(json.dumps({
            "scoring_complete": True, "manifest": {"synthetic_test_fixture": True},
        }))
    output = tmp_path / "synthetic-report"
    summary = reporter.build_report(protocol, evaluation_dir, output)
    saved = json.loads((output / "summary.json").read_text())
    assert saved == summary
    assert saved["complete"] is complete
    assert saved["expected_pairs"] == 100
    assert saved["evaluated_pairs"] == (100 if complete else 99)
    assert saved["completed_conditions"] == (101 if complete else 100)
    assert "50 aligned and 50 opposed pairs" in saved["interpretation"]
    assert "pilot" not in (output / "index.html").read_text()
    if complete:
        groups, _ = load_groups(output / "summary.json")
        assert {group: len(rows) for group, rows in groups.items()} == {
            "aligned": 50, "opposed": 50,
        }
    else:
        with pytest.raises(ValueError, match="complete 100-pair endpoint"):
            load_groups(output / "summary.json")
