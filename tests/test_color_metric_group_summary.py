"""Verify fixed color-bin summaries retain signed measured pair information."""

from __future__ import annotations

import copy
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from scripts import color_metric_group_summary as summary


def make_cell(matrix: np.ndarray, *, cell_id: str = "test") -> dict[str, Any]:
    """Create an input with the true all-off-diagonal RMS, without any scaling."""
    off = ~np.eye(30, dtype=bool)
    return {"id": cell_id, "labels": list(summary.COLOR_LABELS),
            "native_centered": matrix.tolist(),
            "off_diagonal_rms": float(np.sqrt(np.mean(matrix[off] ** 2)))}


def signed_matrix() -> np.ndarray:
    """Use a deterministic asymmetric matrix containing both signs."""
    rows, columns = np.indices((30, 30))
    values = .3 * rows - .7 * columns + 2.0
    values[::2, 1::3] *= -1
    np.fill_diagonal(values, 1e8)
    return values


def test_fixed_groups_and_ordered_pair_counts() -> None:
    """All 36 blocks partition 870 ordered pairs, not 900 or 435 pairs."""
    result = summary.group_summary([make_cell(signed_matrix())])
    assert [group["size"] for group in result["groups"]] == [5, 5, 5, 5, 6, 4]
    assert [group["id"] for group in result["groups"]] == list(range(6))
    assert [group["display_label"] for group in result["groups"]] == list("123456")
    assert [group["name"] for group in result["groups"]] == list(summary.GROUP_NAMES)
    expected = np.outer(summary.GROUP_SIZES, summary.GROUP_SIZES)
    expected -= np.diag(summary.GROUP_SIZES)
    np.testing.assert_array_equal(result["counts"], expected)
    assert np.asarray(result["counts"]).shape == (6, 6)
    assert np.diag(expected).tolist() == [20, 20, 20, 20, 30, 12]
    assert result["included_pair_count"] == 870
    assert result["excluded_true_self_pairs"] == 30
    assert result["threshold_applied"] is False
    assert result["recentered_after_grouping"] is False


def test_full_signed_means_exclude_only_true_trait_self_pairs() -> None:
    """Negative/subthreshold entries remain; within-bin pairs are not discarded."""
    matrix = signed_matrix()
    cell = make_cell(matrix)
    result = summary.group_summary([cell])
    actual = np.asarray(result["cells"][0]["values"])
    starts = np.cumsum((0,) + summary.GROUP_SIZES)
    for target in range(6):
        for source in range(6):
            pairs = [matrix[u, t]
                     for u in range(starts[target], starts[target + 1])
                     for t in range(starts[source], starts[source + 1]) if u != t]
            assert actual[target, source] == pytest.approx(
                sum(pairs) / len(pairs) / cell["off_diagonal_rms"], abs=1e-12)
    assert actual.min() < 0 < actual.max()
    assert actual[0, 0] != 0
    assert actual[0, 1] != actual[1, 0]


def test_block_counts_recover_original_off_diagonal_mean() -> None:
    """Unequal bin sizes require pair-count weighting, not a flat block average."""
    matrix = signed_matrix()
    cell = make_cell(matrix)
    result = summary.group_summary([cell])
    means = np.asarray(result["cells"][0]["values"])
    counts = np.asarray(result["counts"])
    expected = matrix[~np.eye(30, dtype=bool)].mean() / cell["off_diagonal_rms"]
    assert np.sum(means * counts) / counts.sum() == pytest.approx(expected, abs=1e-12)
    assert not np.isclose(means.mean(), expected)


def test_no_group_recentering_and_no_positive_only_selection() -> None:
    """The group means may have nonzero margins and reflect negative cancellation."""
    matrix = np.full((30, 30), 2.0)
    matrix[:5, 5:10] = -4
    matrix[0, 5] = 0.1
    cell = make_cell(matrix)
    result = summary.group_summary([cell])
    means = np.asarray(result["cells"][0]["values"])
    assert means[0, 1] == pytest.approx((-4 * 24 + .1) / 25 / cell["off_diagonal_rms"])
    assert means[1, 0] == pytest.approx(2 / cell["off_diagonal_rms"])
    assert not np.allclose(means.mean(axis=0), 0)
    assert not np.allclose(means.mean(axis=1), 0)


def test_diagonal_values_do_not_affect_rms_or_group_summary() -> None:
    """Arbitrarily large trait self-effects cannot leak into grouped means."""
    matrix = signed_matrix()
    changed = matrix.copy()
    np.fill_diagonal(changed, -1e12)
    first = summary.group_summary([make_cell(matrix)])
    second = summary.group_summary([make_cell(changed)])
    assert first == second


def test_symmetric_metric_stays_symmetric_and_limit_is_shared() -> None:
    """A symmetric teacher-geometry panel is not turned into a directional one."""
    matrix = signed_matrix()
    symmetric = (matrix + matrix.T) / 2
    result = summary.group_summary([make_cell(matrix, cell_id="sl"),
                                    make_cell(symmetric, cell_id="metric1"),
                                    make_cell(matrix.T, cell_id="metric7")])
    b = np.asarray(result["cells"][1]["values"])
    np.testing.assert_allclose(b, b.T, atol=1e-12, rtol=0)
    np.testing.assert_allclose(result["cells"][2]["values"],
                               np.asarray(result["cells"][0]["values"]).T,
                               atol=1e-12, rtol=0)
    assert result["limit"] == max(abs(value) for cell in result["cells"]
                                  for row in cell["values"] for value in row)


def test_group_membership_uses_names_and_never_mutates_inputs() -> None:
    """Permuting storage order leaves the semantic six-bin summary unchanged."""
    matrix = signed_matrix()
    original = make_cell(matrix)
    permutation = np.random.default_rng(121).permutation(30)
    shuffled = make_cell(matrix[np.ix_(permutation, permutation)])
    shuffled["labels"] = [original["labels"][i] for i in permutation]
    snapshot = copy.deepcopy(shuffled)
    expected = summary.group_summary([original])
    actual = summary.group_summary([shuffled])
    np.testing.assert_allclose(actual["cells"][0]["values"],
                               expected["cells"][0]["values"], atol=1e-12)
    assert shuffled == snapshot


@pytest.mark.parametrize("invalid", [0.0, -1.0, np.nan, np.inf, 5.0])
def test_invalid_or_nonoriginal_rms_is_rejected(invalid: float) -> None:
    """The normalization cannot silently use a selected-edge or block RMS."""
    cell = make_cell(signed_matrix())
    cell["off_diagonal_rms"] = invalid
    with pytest.raises(ValueError, match="RMS"):
        summary.group_summary([cell])


def test_invalid_labels_and_matrix_are_rejected() -> None:
    """A differently ordered semantic catalog must not be grouped by raw slices."""
    cell = make_cell(signed_matrix())
    cell["labels"][0] = "unknown"
    with pytest.raises(ValueError, match="labels"):
        summary.group_summary([cell])
    cell = make_cell(signed_matrix())
    cell["native_centered"][0][1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        summary.group_summary([cell])
    with pytest.raises(ValueError, match="At least one"):
        summary.group_summary([])


def test_draw_uses_exact_signed_values_and_one_unclipped_symmetric_limit() -> None:
    """Drawing only creates artists: no file output, annotations, or rescaling."""
    values = np.arange(36, dtype=float).reshape(6, 6) / 10 - 1.8
    fig, axes = plt.subplots(1, 2)
    try:
        images = [summary.draw_group_summary(ax, values, 2.0, show_ylabels=index == 0)
                  for index, ax in enumerate(axes)]
        for ax, image in zip(axes, images):
            np.testing.assert_array_equal(image.get_array(), values)
            assert image.get_clim() == (-2.0, 2.0)
            assert image.origin == "upper"
            assert [label.get_text() for label in ax.get_xticklabels()] == list("123456")
            assert not ax.texts
        assert [label.get_text() for label in axes[0].get_yticklabels()] == list("123456")
        assert not axes[1].get_yticklabels()
        with pytest.raises(ValueError, match="clip"):
            summary.draw_group_summary(axes[0], values, 1.0)
    finally:
        plt.close(fig)
