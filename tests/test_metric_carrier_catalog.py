"""Tests for blinded carrier-label resolution."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from metric_carrier_catalog import (  # noqa: E402
    load_carrier_catalog,
    validate_catalog_tokenization,
)


MANIFEST = PROJECT_ROOT / "experiment_plan/carriers/qwen25_opaque100.json"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def test_opaque_catalog_resolves_blinded_labels_to_model_text() -> None:
    """The analysis label is not the continuation shown to the model."""
    catalog = load_carrier_catalog(MANIFEST, expected_model_id=MODEL_ID)

    assert len(catalog.labels) == 100
    assert catalog.labels[0] == "opaque_000"
    assert catalog.texts[0] == "క"
    assert catalog.token_ids is not None
    assert catalog.token_ids[0] == 150430
    assert catalog.text_by_label["opaque_000"] == "క"


def test_opaque_catalog_requires_exact_teacher_cache_order() -> None:
    """Reordered probability columns cannot silently change carrier identity."""
    catalog = load_carrier_catalog(MANIFEST, expected_model_id=MODEL_ID)

    with pytest.raises(ValueError, match="ordered teacher-cache support"):
        load_carrier_catalog(
            MANIFEST,
            expected_model_id=MODEL_ID,
            expected_labels=list(reversed(catalog.labels)),
        )


def test_explicit_token_ids_are_checked_against_tokenizer() -> None:
    """The manifest token id must equal the tokenizer's exact bare tokenization."""
    catalog = load_carrier_catalog(MANIFEST, expected_model_id=MODEL_ID)
    mapping = dict(zip(catalog.texts, catalog.token_ids or []))

    class FakeTokenizer:
        def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
            assert add_special_tokens is False
            return {"input_ids": [mapping[text]]}

    validate_catalog_tokenization(FakeTokenizer(), catalog)
    mapping[catalog.texts[0]] = -1
    with pytest.raises(ValueError, match="opaque_000"):
        validate_catalog_tokenization(FakeTokenizer(), catalog)
