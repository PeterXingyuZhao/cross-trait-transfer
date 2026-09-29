#!/usr/bin/env python3
"""Resolve blinded carrier labels to the exact model-facing continuations."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(frozen=True, **DATACLASS_KW_ONLY)
class CarrierCatalog:
    """An ordered analysis-label to model-facing-text carrier mapping."""

    model_id: str
    labels: list[str]
    texts: list[str]
    token_ids: list[int] | None
    source_path: Path
    source_sha256: str

    @property
    def text_by_label(self) -> dict[str, str]:
        """Return the exact model-facing text for every analysis label."""
        return dict(zip(self.labels, self.texts))


def _compatible_model_ids(payload: dict[str, Any]) -> list[str]:
    """Return the manifest's declared compatible model identifiers."""
    model_ids = payload.get("model_ids")
    if isinstance(model_ids, list) and all(
        isinstance(value, str) and value for value in model_ids
    ):
        return model_ids
    model_id = payload.get("model_id")
    return [model_id] if isinstance(model_id, str) and model_id else []


def load_carrier_catalog(
    path: Path,
    *,
    expected_model_id: str,
    expected_labels: list[str] | None = None,
) -> CarrierCatalog:
    """Load a token or text manifest and validate its ordered label support."""
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Carrier manifest must be a JSON object.")
    compatible_model_ids = _compatible_model_ids(payload)
    if expected_model_id not in compatible_model_ids:
        raise ValueError(
            "Carrier manifest is incompatible with the metric model: "
            f"{expected_model_id!r} not in {compatible_model_ids!r}."
        )

    token_rows = payload.get("tokens")
    text_rows = payload.get("candidates")
    if isinstance(token_rows, list):
        rows = token_rows
        require_token_ids = True
    elif isinstance(text_rows, list):
        rows = text_rows
        require_token_ids = False
    else:
        raise ValueError(
            "Carrier manifest must contain a 'tokens' or 'candidates' list."
        )
    if len(rows) < 2:
        raise ValueError("Carrier manifest must contain at least two entries.")

    labels: list[str] = []
    texts: list[str] = []
    token_ids: list[int] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Carrier entry {index} must be an object.")
        label = row.get("label")
        text = row.get("text")
        token_id = row.get("token_id")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Carrier entry {index} has an invalid label.")
        if not isinstance(text, str) or not text:
            raise ValueError(f"Carrier entry {index} has invalid model-facing text.")
        if require_token_ids and (not isinstance(token_id, int) or token_id < 0):
            raise ValueError(f"Carrier entry {index} has an invalid token id.")
        labels.append(label.strip())
        texts.append(text)
        if require_token_ids:
            token_ids.append(token_id)

    if len(labels) != len(set(labels)):
        raise ValueError("Carrier manifest labels must be unique.")
    if len(texts) != len(set(texts)):
        raise ValueError("Carrier manifest texts must be unique.")
    if require_token_ids and len(token_ids) != len(set(token_ids)):
        raise ValueError("Carrier manifest token ids must be unique.")
    if expected_labels is not None and labels != expected_labels:
        raise ValueError(
            "Carrier manifest labels do not exactly match the ordered teacher-cache "
            "support."
        )

    return CarrierCatalog(
        model_id=expected_model_id,
        labels=labels,
        texts=texts,
        token_ids=token_ids if require_token_ids else None,
        source_path=path,
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )


def identity_carrier_texts(
    *,
    model_id: str,
    labels: list[str],
) -> CarrierCatalog:
    """Return the identity mapping used by unblinded carrier catalogs."""
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("Carrier labels must be non-empty and unique.")
    serialized = json.dumps(labels, ensure_ascii=False, separators=(",", ":"))
    return CarrierCatalog(
        model_id=model_id,
        labels=list(labels),
        texts=list(labels),
        token_ids=None,
        source_path=Path("<identity>"),
        source_sha256=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    )


def resolve_carrier_catalog(
    *,
    manifest_path: Path | None,
    model_id: str,
    labels: list[str],
) -> CarrierCatalog:
    """Resolve labels through a manifest, or use them verbatim when absent."""
    if manifest_path is None:
        return identity_carrier_texts(model_id=model_id, labels=labels)
    return load_carrier_catalog(
        manifest_path,
        expected_model_id=model_id,
        expected_labels=labels,
    )


def validate_catalog_tokenization(tokenizer: Any, catalog: CarrierCatalog) -> None:
    """Verify explicit single-token manifests against the loaded tokenizer."""
    if catalog.token_ids is None:
        return
    for label, text, expected_token_id in zip(
        catalog.labels,
        catalog.texts,
        catalog.token_ids,
    ):
        encoded = tokenizer(text, add_special_tokens=False)
        input_ids = encoded["input_ids"]
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        actual = [int(value) for value in input_ids]
        if actual != [expected_token_id]:
            raise ValueError(
                f"Carrier {label!r} tokenizes as {actual}, expected "
                f"[{expected_token_id}]."
            )
