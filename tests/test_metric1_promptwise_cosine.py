"""Independent checks that only Metric 1's prompt aggregation changes."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_metric1_promptwise_cosine import aggregate_signatures, compute_metric1


def scalar_cos(a: np.ndarray, b: np.ndarray) -> float:
    """Dense cosine reference with no epsilon in the norm."""
    return float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_independent_per_prompt_reference() -> None:
    rng = np.random.default_rng(104)
    vectors = rng.normal(size=(3, 5, 7))
    result = aggregate_signatures(vectors)
    for i in range(3):
        for j in range(3):
            expected = np.mean([scalar_cos(vectors[i, x], vectors[j, x]) for x in range(5)])
            assert result["mean_prompt_cosine"][i, j] == pytest.approx(expected, abs=1e-14)
            assert result["flattened"][i, j] == pytest.approx(scalar_cos(vectors[i], vectors[j]), abs=1e-14)


def test_large_difference_counterexample() -> None:
    vectors = np.array([[[10., 0.], [1., 0.]], [[10., 0.], [-1., 0.]]])
    result = aggregate_signatures(vectors)
    assert result["flattened"][0, 1] == pytest.approx(99 / 101)
    assert result["mean_prompt_cosine"][0, 1] == pytest.approx(0)


def test_matching_directions_different_norm_profiles() -> None:
    vectors = np.array([[[10., 0.], [1., 0.]], [[1., 0.], [10., 0.]]])
    result = aggregate_signatures(vectors)
    assert result["flattened"][0, 1] == pytest.approx(20 / 101)
    assert result["mean_prompt_cosine"][0, 1] == pytest.approx(1)
    assert result["norm_profile_similarity"][0, 1] == pytest.approx(20 / 101)


@pytest.mark.parametrize("prompts", [1, 6])
def test_equal_row_norms_agree(prompts: int) -> None:
    vectors = np.random.default_rng(223).normal(size=(3, prompts, 7))
    vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)
    vectors *= np.arange(1, 4)[:, None, None]
    result = aggregate_signatures(vectors)
    np.testing.assert_allclose(result["flattened"], result["mean_prompt_cosine"], atol=1e-14)


def test_original_signature_construction() -> None:
    rng = np.random.default_rng(285)
    p0 = rng.dirichlet(np.ones(7), size=4)
    teachers = rng.dirichlet(np.ones(7), size=(3, 4))
    lam = np.log(teachers + 1e-12) - np.log(p0 + 1e-12)
    vectors = np.sqrt(p0) * (lam - np.mean(lam, axis=0))
    expected = aggregate_signatures(vectors)
    actual = compute_metric1(p0, teachers)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-14)


def test_zero_rows_are_not_silently_dropped() -> None:
    with pytest.raises(ValueError, match="zero signature row"):
        aggregate_signatures(np.array([[[1., 2.], [0., 0.]]]))


def test_nonfinite_input_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        aggregate_signatures(np.array([[[np.nan, 1.]]]))


def test_row_rescaling_only_preserves_mean_prompt() -> None:
    rng = np.random.default_rng(180)
    vectors = rng.normal(size=(3, 5, 7))
    before = aggregate_signatures(vectors)
    after = aggregate_signatures(vectors * rng.uniform(.1, 4, size=(3, 5, 1)))
    np.testing.assert_allclose(before["mean_prompt_cosine"], after["mean_prompt_cosine"], atol=1e-14)
    assert not np.allclose(before["flattened"], after["flattened"])
