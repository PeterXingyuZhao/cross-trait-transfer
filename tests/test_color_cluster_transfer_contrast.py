"""Check the fixed-color within-minus-between statistic independently."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.color_cluster_transfer_contrast import cluster_transfer_contrast
from scripts.color_metric_group_summary import COLOR_LABELS, GROUP_LABELS


def projection_center(matrix: np.ndarray) -> np.ndarray:
    """Center by multiplication, independently of the implementation."""
    projection = np.eye(30) - np.ones((30, 30)) / 30
    return projection @ matrix @ projection


def test_signed_means_match_independent_ordered_pair_enumeration() -> None:
    """Full centering precedes selection, including large diagonal effects."""
    rng = np.random.default_rng(431)
    matrix = rng.normal(size=(30, 30))
    np.fill_diagonal(matrix, np.arange(30) * 10.0)
    centered = projection_center(matrix)
    result = cluster_transfer_contrast(matrix, COLOR_LABELS)
    membership = {label: i for i, group in enumerate(GROUP_LABELS) for label in group}
    within, between = [], []
    for i, row_label in enumerate(COLOR_LABELS):
        for j, column_label in enumerate(COLOR_LABELS):
            if i != j:
                target = within if membership[row_label] == membership[column_label] else between
                target.append(centered[i, j])
    assert result["within_pair_count"] == len(within) == 122
    assert result["between_pair_count"] == len(between) == 748
    assert result["within_mean_dc_nats"] == pytest.approx(np.mean(within), abs=1e-12)
    assert result["between_mean_dc_nats"] == pytest.approx(np.mean(between), abs=1e-12)
    assert result["within_minus_between_dc_nats"] == pytest.approx(
        np.mean(within) - np.mean(between), abs=1e-12,
    )


def test_permuting_both_matrix_axes_and_labels_preserves_results() -> None:
    """Color names, not storage positions, determine fixed group membership."""
    rng = np.random.default_rng(48)
    matrix = rng.normal(size=(30, 30))
    order = rng.permutation(30)
    actual = cluster_transfer_contrast(
        matrix[np.ix_(order, order)], [COLOR_LABELS[i] for i in order],
    )
    expected = cluster_transfer_contrast(matrix, COLOR_LABELS)
    assert actual == pytest.approx(expected, abs=1e-12)


def test_row_column_and_global_offsets_do_not_change_contrast() -> None:
    """Additive row/column effects disappear before either mean is measured."""
    rng = np.random.default_rng(21)
    matrix = rng.normal(size=(30, 30))
    shifted = matrix + rng.normal(size=(30, 1)) + rng.normal(size=(1, 30)) + 17
    expected = cluster_transfer_contrast(matrix, COLOR_LABELS)
    assert cluster_transfer_contrast(shifted, COLOR_LABELS) == pytest.approx(
        expected, abs=1e-12,
    )


def test_unequal_groups_are_weighted_by_pair_counts_not_equally() -> None:
    """Sizes 5,5,5,5,6,4 contribute 20,20,20,20,30,12 within pairs."""
    sizes = np.asarray([len(group) for group in GROUP_LABELS])
    starts = np.concatenate(([0], np.cumsum(sizes)))
    matrix = np.zeros((30, 30))
    matrix[starts[4]:starts[5], starts[4]:starts[5]] = 13
    matrix[starts[5]:starts[6], starts[5]:starts[6]] = -7
    centered = projection_center(matrix)
    group_means = []
    for start, end in zip(starts[:-1], starts[1:]):
        block = centered[start:end, start:end]
        group_means.append(block[~np.eye(end - start, dtype=bool)].mean())
    counts = sizes * (sizes - 1)
    expected = np.average(group_means, weights=counts)
    result = cluster_transfer_contrast(matrix, COLOR_LABELS)
    assert result["within_mean_dc_nats"] == pytest.approx(expected, abs=1e-12)
    assert not np.isclose(result["within_mean_dc_nats"], np.mean(group_means))


def test_constant_matrix_has_zero_contrast_and_input_is_unchanged() -> None:
    """A zero result is valid and the caller's input is never centered in place."""
    matrix = np.full((30, 30), 7.0)
    original = matrix.copy()
    result = cluster_transfer_contrast(matrix, COLOR_LABELS)
    assert result["within_minus_between_dc_nats"] == 0
    np.testing.assert_array_equal(matrix, original)


@pytest.mark.parametrize("labels", [
    COLOR_LABELS[:-1],
    COLOR_LABELS + ("red",),
    COLOR_LABELS[:-1] + ("red",),
    COLOR_LABELS[:-1] + ("not-a-color",),
    COLOR_LABELS[:-1] + (None,),
    None,
])
def test_invalid_labels_are_rejected(labels: object) -> None:
    """Exactly the 30 canonical unique names are required."""
    with pytest.raises(ValueError, match="30 unique canonical color labels"):
        cluster_transfer_contrast(np.zeros((30, 30)), labels)


@pytest.mark.parametrize("matrix", [
    np.zeros((29, 29)),
    np.zeros((30, 29)),
    np.zeros(900),
    np.zeros((1, 30, 30)),
    np.full((30, 30), np.nan),
    np.full((30, 30), np.inf),
    np.full((30, 30), "bad-value"),
    np.full((30, 30), 1 + 2j),
])
def test_invalid_matrix_is_rejected(matrix: np.ndarray) -> None:
    """Malformed shapes and nonfinite or nonreal values cannot yield a score."""
    with pytest.raises(ValueError):
        cluster_transfer_contrast(matrix, COLOR_LABELS)


def test_saved_checkpoint_endpoint_contrasts() -> None:
    """The archived full30 matrices reproduce the approved curve endpoints."""
    source = (
        Path(__file__).resolve().parents[1]
        / "results_for_presentation/paper_token_sft_learning_dynamics/data.json"
    )
    if not source.is_file():
        pytest.skip("Saved local checkpoint archive is not present")
    color = json.loads(source.read_text())["families"]["color"]
    for step, expected in ((43, 0.1513), (471, 3.7299)):
        record = next(item for item in color["trajectory"] if item["step"] == step)
        result = cluster_transfer_contrast(np.asarray(record["matrix"]), color["labels"])
        assert result["within_minus_between_dc_nats"] == pytest.approx(expected, abs=5e-5)
        assert result["within_pair_count"] == 122
        assert result["between_pair_count"] == 748
