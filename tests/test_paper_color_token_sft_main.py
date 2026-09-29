"""Check the scientific source and aggregation contract of the color figure."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import build_paper_color_token_sft_main as figure
from scripts.color_metric_group_summary import GROUP_LABELS, group_summary


def write_matrix(
    path: Path, values: np.ndarray, rows: list[str], columns: list[str]
) -> None:
    """Write a small labeled test fixture without altering experiment files."""
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["answer/teacher", *columns])
        writer.writerows([label, *row] for label, row in zip(rows, values))


def test_read_matrix_reorders_axes_independently(tmp_path: Path) -> None:
    """Answer and teacher permutations must not change matrix orientation."""
    labels = [f"trait_{i}" for i in range(30)]
    matrix = np.arange(900, dtype=float).reshape(30, 30)
    row_order = np.roll(np.arange(30), 7)
    column_order = np.arange(30)[::-1]
    path = tmp_path / "reordered.csv"
    write_matrix(
        path,
        matrix[np.ix_(row_order, column_order)],
        [labels[i] for i in row_order],
        [labels[i] for i in column_order],
    )
    np.testing.assert_array_equal(figure.read_matrix(path, labels), matrix)


@pytest.mark.parametrize("axis", ["rows", "columns"])
@pytest.mark.parametrize("defect", ["missing", "replacement_duplicate", "extra_duplicate"])
def test_read_matrix_rejects_nonunique_or_absent_labels(
    tmp_path: Path, axis: str, defect: str
) -> None:
    """An extra duplicate cannot be silently ignored by label-based indexing."""
    labels = [f"trait_{i}" for i in range(30)]
    rows, columns = labels.copy(), labels.copy()
    selected = rows if axis == "rows" else columns
    if defect == "missing":
        selected.pop()
    elif defect == "replacement_duplicate":
        selected[-1] = selected[0]
    else:
        selected.append(selected[0])
    values = np.arange(len(rows) * len(columns), dtype=float).reshape(
        len(rows), len(columns)
    )
    path = tmp_path / "invalid.csv"
    write_matrix(path, values, rows, columns)
    with pytest.raises(ValueError):
        figure.read_matrix(path, labels)


def test_agreement_centers_full_matrix_before_excluding_diagonal() -> None:
    """Large diagonal effects must participate in centering before masking."""
    rng = np.random.default_rng(719)
    a, b = rng.normal(size=(2, 30, 30))
    a[np.diag_indices(30)] += np.linspace(-80, 150, 30)
    b[np.diag_indices(30)] += np.linspace(150, -20, 30)
    centering = np.eye(30) - np.ones((30, 30)) / 30
    centered_a, centered_b = centering @ a @ centering, centering @ b @ centering
    cross = ~np.eye(30, dtype=bool)
    result = figure.agreement(a, b)
    assert result["all900"] == pytest.approx(
        np.corrcoef(centered_a.ravel(), centered_b.ravel())[0, 1], abs=1e-14
    )
    assert result["off870"] == pytest.approx(
        np.corrcoef(centered_a[cross], centered_b[cross])[0, 1], abs=1e-14
    )
    assert abs(result["off870"] - np.corrcoef(a[cross], b[cross])[0, 1]) > 0.2


@pytest.fixture
def audited_inputs() -> tuple[Path, Path]:
    """Use the immutable, separately audited original 1,024-prompt result."""
    folder = figure.ROOT / "results_for_presentation/paper_main_color_token_sft_20260923/audit_metric1"
    matrix = folder / "original_1024/metric1_current_raw.csv"
    audit = folder / "audit.json"
    if not matrix.is_file() or not audit.is_file() or not figure.SOURCE.is_file():
        pytest.skip("The optional local scientific result bundle is unavailable")
    return matrix, audit


def test_load_cells_rejects_legacy_matrix_with_current_audit(
    audited_inputs: tuple[Path, Path],
) -> None:
    """Legacy symmetric unit-diagonal kernels cannot acquire a current label."""
    matrix, audit = audited_inputs
    legacy = matrix.with_name("metric1_legacy_flattened_raw.csv")
    with pytest.raises(ValueError):
        figure.load_cells(legacy, audit)


def test_exact_sources_and_group_provenance(
    audited_inputs: tuple[Path, Path],
) -> None:
    """Reconstruct published group cells from raw entries, labels, and hashes."""
    matrix, audit_path = audited_inputs
    cells, audit = figure.load_cells(matrix, audit_path)
    primary = next(panel for panel in audit["panels"] if panel["name"] == audit["primary_panel"])
    assert audit["model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert primary["name"] == "original_1024"
    assert primary["prompt_coordinates_with_positive_weight"] == 1024
    assert primary["total_prompt_weight"] == 1024
    assert primary["zero_norm_signatures"] == 0
    assert primary["minimum_signature_norm"] > 0
    assert cells[1]["source_sha256"] == primary["output_sha256"]["metric1_current_raw.csv"]
    assert cells[1]["provenance_sha256"] == figure.sha256(audit_path)
    assert cells[0]["source_sha256"] == "6721ed458ac7ce73f64de2cdae374615faf047520624e67f2b7151770d6c6598"
    assert cells[2]["source_sha256"] == "32f030d901c2eff5d2b4cab4e3bade3a51bfb9185c9a0be94db8d5bd57d1656b"
    assert cells[1]["agreement_with_sl"]["off870"] == pytest.approx(0.4984415003419831)
    assert cells[2]["agreement_with_sl"]["off870"] == pytest.approx(0.743050910370658)

    summary = group_summary(cells)
    assert summary["included_pair_count"] == 870
    assert not summary["threshold_applied"]
    assert not summary["recentered_after_grouping"]
    sl_groups = np.asarray(summary["cells"][0]["values"]).ravel()
    expected_group_r = (1.0, 0.9092072967870687, 0.9593764275324068)
    for block, expected_r in zip(summary["cells"], expected_group_r):
        group_r = np.corrcoef(sl_groups, np.asarray(block["values"]).ravel())[0, 1]
        assert group_r == pytest.approx(expected_r)
    labels = cells[0]["labels"]
    centering = np.eye(30) - np.ones((30, 30)) / 30
    cross = ~np.eye(30, dtype=bool)
    for source, grouped in zip(cells, summary["cells"]):
        assert source["labels"] == labels == audit["labels"]
        centered = centering @ np.asarray(source["native_raw"]) @ centering
        rms = np.sqrt(np.mean(centered[cross] ** 2))
        expected = np.empty((6, 6))
        for i, answers in enumerate(GROUP_LABELS):
            for j, teachers in enumerate(GROUP_LABELS):
                pairs = [
                    centered[labels.index(answer), labels.index(teacher)]
                    for answer in answers for teacher in teachers if answer != teacher
                ]
                expected[i, j] = np.mean(pairs) / rms
                assert len(pairs) == summary["counts"][i][j]
        np.testing.assert_allclose(grouped["values"], expected, atol=1e-12, rtol=0)

    saved = json.loads(figure.SOURCE.read_text())
    assert cells[0]["pca"] == saved["cells"][0]["pca"]
