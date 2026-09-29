"""Independent checks for the portable, current Metric 1 release tool."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "metric1.py"
SPEC = importlib.util.spec_from_file_location("release_metric1", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
metric1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metric1)


@pytest.fixture
def arrays() -> dict[str, np.ndarray]:
    """Return independently seeded, nondegenerate probability inputs."""
    rng = np.random.default_rng(941)
    return {
        "p0": rng.dirichlet(np.ones(4), size=5),
        "teachers": rng.dirichlet(np.ones(4), size=(3, 5)),
        "labels": np.array(["alpha", "beta", "gamma"]),
        "prompt_ids": np.array(["p0", "p1", "p2", "p3", "p4"]),
        "carrier_labels": np.array(["c0", "c1", "c2", "c3"]),
    }


def independent_expected(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Evaluate each scalar cosine explicitly, independently of vectorization."""
    p0, teachers = arrays["p0"], arrays["teachers"]
    weights = arrays.get("weights", np.ones(p0.shape[0]))
    result = np.zeros((teachers.shape[0], teachers.shape[0]))
    for prompt, weight in enumerate(weights):
        if weight == 0:
            continue
        ratio = np.log((teachers[:, prompt] + 1e-12) / (p0[prompt] + 1e-12))
        signatures = np.sqrt(p0[prompt]) * (ratio - ratio.mean(axis=0))
        for i in range(teachers.shape[0]):
            for j in range(teachers.shape[0]):
                result[i, j] += weight * np.dot(signatures[i], signatures[j]) / (
                    np.linalg.norm(signatures[i]) * np.linalg.norm(signatures[j])
                )
    return result / weights.sum()


@pytest.mark.parametrize("weighted", [False, True])
def test_independent_mean_prompt_reference(arrays: dict[str, np.ndarray], weighted: bool) -> None:
    """Both equal and unequal shared weights match an independent calculation."""
    if weighted:
        arrays["weights"] = np.array([1.0, 4.0, 0.5, 2.0, 9.0])
    result = metric1.compute_metric1(arrays)
    np.testing.assert_allclose(result["mean_prompt_cosine"], independent_expected(arrays), atol=1e-14)
    assert "flattened" not in result


def test_repeated_prompt_ids_remain_occurrences(arrays: dict[str, np.ndarray]) -> None:
    """Repeated identifiers neither collapse rows nor change their weights."""
    arrays["prompt_ids"] = np.array(["same", "same", "p2", "p3", "p4"])
    result = metric1.compute_metric1(arrays)
    assert result["prompt_occurrence_count"] == 5
    assert result["unique_prompt_id_count"] == 4
    np.testing.assert_allclose(result["mean_prompt_cosine"], independent_expected(arrays), atol=1e-14)


def test_explicit_zero_weight_exclusion(arrays: dict[str, np.ndarray]) -> None:
    """Only zero-weight rows may have undefined signatures, with exclusions saved."""
    arrays["teachers"][:, 1] = arrays["p0"][1]
    arrays["weights"] = np.array([1.0, 0.0, 2.0, 3.0, 4.0])
    result = metric1.compute_metric1(arrays)
    assert result["weighting"]["zero_weight_excluded_indices"] == [1]
    assert result["weighting"]["zero_weight_excluded_prompt_ids"] == ["p1"]
    assert result["positive_weight_prompt_count"] == 4
    np.testing.assert_allclose(result["mean_prompt_cosine"], independent_expected(arrays), atol=1e-14)
    arrays["weights"][1] = 1
    with pytest.raises(ValueError, match="positive-weight.*zero"):
        metric1.compute_metric1(arrays)


@pytest.mark.parametrize(
    ("name", "invalid", "message"),
    [
        ("p0", np.ones((5, 3)) / 3, "matching axes"),
        ("p0", np.zeros((5, 4)), "normalized"),
        ("p0", np.full((5, 4), np.nan), "finite"),
        ("teachers", np.ones((3, 5, 4)) * 2, r"\[0,1\]"),
        ("teachers", np.ones((3, 5, 4), dtype=complex), "real numeric"),
        ("labels", np.array(["a", "a", "b"]), "unique"),
        ("labels", np.array(["a", "b"]), "shape"),
        ("labels", np.array(["a", "b", "c"], dtype=object), "Unicode"),
        ("carrier_labels", np.array(["a", "b", "a", "c"]), "unique"),
        ("prompt_ids", np.array(["p0", "p1", "", "p3", "p4"]), "nonempty"),
        ("weights", np.array([1, 2]), "shape"),
        ("weights", np.array([1, -1, 1, 1, 1]), "nonnegative"),
        ("weights", np.zeros(5), "positive sum"),
        ("weights", np.array([1, 1, np.inf, 1, 1]), "finite"),
        ("weights", np.full(5, np.finfo(float).max), "finite positive sum"),
        ("sl", np.ones((2, 2)), "shape"),
        ("sl", np.full((3, 3), np.inf), "finite"),
    ],
)
def test_invalid_arrays_rejected(arrays: dict[str, np.ndarray], name: str, invalid: np.ndarray, message: str) -> None:
    """Invalid axes, probabilities, identifiers, and shared weights fail loudly."""
    arrays[name] = invalid
    with pytest.raises(ValueError, match=message):
        metric1.compute_metric1(arrays)


def test_schema_validation(arrays: dict[str, np.ndarray]) -> None:
    """Required names and misspelled optional fields cannot be ignored."""
    arrays.pop("labels")
    with pytest.raises(ValueError, match="missing=.*labels"):
        metric1.compute_metric1(arrays)


def test_full_centering_precedes_offdiagonal_selection(arrays: dict[str, np.ndarray]) -> None:
    """Correlations use full-matrix centering before either entry selection."""
    arrays["sl"] = np.array([[8.0, 2.0, -2.0], [3.0, 11.0, 4.0], [-3.0, 1.0, 5.0]])
    result = metric1.compute_metric1(arrays)
    matrix = np.array(result["mean_prompt_cosine"])
    center = np.eye(3) - np.ones((3, 3)) / 3
    left, right = center @ matrix @ center, center @ arrays["sl"] @ center
    mask = ~np.eye(3, dtype=bool)
    assert result["sl_correlations"]["all_entries"]["value"] == pytest.approx(np.corrcoef(left.ravel(), right.ravel())[0, 1])
    assert result["sl_correlations"]["off_diagonal_entries"]["value"] == pytest.approx(np.corrcoef(left[mask], right[mask])[0, 1])


def test_undefined_correlation_is_explicit_null(arrays: dict[str, np.ndarray]) -> None:
    """A constant SL matrix has no defined Pearson correlation."""
    arrays["sl"] = np.ones((3, 3))
    result = metric1.compute_metric1(arrays)
    for selection in ("all_entries", "off_diagonal_entries"):
        assert result["sl_correlations"][selection]["value"] is None
        assert "zero variance" in result["sl_correlations"][selection]["undefined_reason"]


def test_cli_archive_and_provenance(tmp_path: Path, arrays: dict[str, np.ndarray]) -> None:
    """A real subprocess consumes safe NPZ inputs and writes verifiable JSON."""
    source, output = tmp_path / "probabilities.npz", tmp_path / "computed.json"
    np.savez(source, **arrays)
    subprocess.run([sys.executable, str(TOOL_PATH), "--input", str(source), "--output", str(output)], check=True, capture_output=True, text=True)
    result = json.loads(output.read_text())
    assert result["source"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert result["implementation_sha256"] == hashlib.sha256(TOOL_PATH.read_bytes()).hexdigest()
    assert result["labels"] == arrays["labels"].tolist()
    assert str(tmp_path) not in output.read_text()
    np.testing.assert_allclose(result["mean_prompt_cosine"], independent_expected(arrays), atol=1e-14)


def test_cli_rejects_object_arrays_without_pickle(tmp_path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Object-backed archives fail instead of enabling pickle deserialization."""
    arrays["labels"] = arrays["labels"].astype(object)
    source, output = tmp_path / "unsafe.npz", tmp_path / "computed.json"
    np.savez(source, **arrays)
    completed = subprocess.run([sys.executable, str(TOOL_PATH), "--input", str(source), "--output", str(output)], capture_output=True, text=True)
    assert completed.returncode == 2
    assert "allow_pickle=False" in completed.stderr
    assert not output.exists()


def test_demo_is_deterministic_and_marked_synthetic(tmp_path: Path) -> None:
    """The portable demo is reproducible and cannot masquerade as paper results."""
    output = tmp_path / "demo.json"
    assert metric1.main(["--demo", "--output", str(output)]) == 0
    before = output.read_bytes()
    assert metric1.main(["--demo", "--output", str(output)]) == 0
    assert output.read_bytes() == before
    assert json.loads(before)["source"]["is_paper_result"] is False
