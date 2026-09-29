"""Ensure restored background dots preserve every pair and the compact layout."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from scripts.plot_paper_random_pairs_v38 import configure, load_groups, numerical_summary
from scripts.plot_paper_random_pairs_v41 import make_figure


SOURCE = (Path(__file__).resolve().parents[1]
          / "data/figures/figure5.json")


def test_all_pair_dots_and_unchanged_statistics() -> None:
    """Check 100 unchanged y-values, no duplicated fliers, and compact A/B layout."""
    groups, source_sha = load_groups(SOURCE)
    assert source_sha == "5de5f9bb58e613997d42f71b5428f43faf503c1253b9ad367cb718343e4872ba"
    stats = numerical_summary(groups)
    configure()
    figure = make_figure(groups, stats)
    gap, signs = figure.axes
    collections = {item.get_label(): item for item in gap.collections}
    for index, group in enumerate(("aligned", "opposed")):
        dots = collections[f"{group}_all_pairs"]
        offsets = dots.get_offsets()
        assert offsets.shape == (50, 2)
        np.testing.assert_allclose(offsets[:, 1], [row["gap"] for row in groups[group]])
        assert np.max(np.abs(offsets[:, 0] - index)) <= 0.16
        assert dots.get_alpha() == 0.30
        assert collections[f"{group}_mean"].get_offsets()[0, 1] == pytest.approx(
            stats[group]["gap_mean"])
    assert len(gap.lines) == 10  # Two sets of whiskers, caps, and median; no fliers.
    assert len(gap.patches) == 2
    assert [p.get_height() for p in signs.patches] == [44, 22, 34, 34, 58, 8]
    assert [label.get_text() for label in signs.get_xticklabels()] == [
        "Both above\ncontrol", "One above,\none below control", "Both below\ncontrol"]
    assert gap.get_title(loc="left") == "A  Recovery imbalance"
    assert signs.get_title(loc="left") == "B  Recovery relative to control"
    assert tuple(figure.get_size_inches()) == pytest.approx((4.8, 1.5))
    assert gap.get_position().y0 == signs.get_position().y0
    assert [label.get_text() for label in gap.texts] == ["3.05", "5.38"]
    figure.canvas.draw()
    bounds = figure.bbox
    for axis in figure.axes:
        heading = axis._left_title.get_window_extent(figure.canvas.get_renderer())
        assert heading.x0 >= bounds.x0
        assert heading.x1 <= bounds.x1
        assert heading.y1 <= bounds.y1
    plt.close(figure)


def test_full_width_spreads_panels_without_changing_height_or_data() -> None:
    """Use the row for horizontal space while keeping the statistics identical."""
    groups, _ = load_groups(SOURCE)
    stats = numerical_summary(groups)
    configure()
    narrow = make_figure(groups, stats)
    wide = make_figure(groups, stats, full_width=True)
    assert tuple(wide.get_size_inches()) == pytest.approx((5.5, 1.5))
    gap, signs = wide.axes
    narrow_separation = (narrow.axes[1].get_position().x0
                         - narrow.axes[0].get_position().x1) * 4.8
    wide_separation = (signs.get_position().x0 - gap.get_position().x1) * 5.5
    assert wide_separation > narrow_separation
    assert gap.get_position().y0 == signs.get_position().y0
    for original, updated in zip(narrow.axes, wide.axes):
        assert original._left_title.get_fontsize() == updated._left_title.get_fontsize()
        for old_dots, new_dots in zip(original.collections, updated.collections):
            np.testing.assert_array_equal(old_dots.get_offsets(), new_dots.get_offsets())
        assert [p.get_height() for p in original.patches if hasattr(p, "get_height")] == [
            p.get_height() for p in updated.patches if hasattr(p, "get_height")]
    wide.canvas.draw()
    renderer = wide.canvas.get_renderer()
    for axis in wide.axes:
        heading = axis._left_title.get_window_extent(renderer)
        assert wide.bbox.contains(heading.x0, heading.y0)
        assert wide.bbox.contains(heading.x1, heading.y1)
    plt.close(narrow)
    plt.close(wide)
