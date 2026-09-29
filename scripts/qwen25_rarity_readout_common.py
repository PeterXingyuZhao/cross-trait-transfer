#!/usr/bin/env python3
"""Shared manifest and path helpers for Qwen-2.5 rarity readout analyses."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}
LOG_SPACED_EXPERIMENT = "qwen25_fineweb_log_spaced_word_rarity"
READOUT_EXPERIMENT = "qwen25_word_rarity_readout_robustness"


@dataclass(**DATACLASS_KW_ONLY)
class ReadoutPanel:
    """One recovery-prompt panel resolved for one semantic category."""

    name: str
    questions: list[str]
    assistant_prefill: str
    comparison_scope: str


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON object from ``path``."""
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def load_rarity_manifest(path: Path) -> dict[str, Any]:
    """Load and validate the frozen log-spaced target manifest."""
    payload = load_json_object(path)
    if payload.get("experiment") != LOG_SPACED_EXPERIMENT:
        raise ValueError(f"Unexpected rarity-manifest experiment in {path}.")
    bins = payload.get("bins")
    if not isinstance(bins, list) or not bins:
        raise ValueError("Rarity manifest must contain a non-empty bins list.")
    return payload


def load_readout_manifest(path: Path) -> dict[str, Any]:
    """Load and validate the neutral/category-matched readout manifest."""
    payload = load_json_object(path)
    if payload.get("experiment") != READOUT_EXPERIMENT:
        raise ValueError(f"Unexpected readout-manifest experiment in {path}.")
    questions = payload.get("questions")
    if not isinstance(questions, list) or not questions or not all(
        isinstance(question, str) and question.strip() for question in questions
    ):
        raise ValueError("Readout manifest questions must be non-empty strings.")
    if not isinstance(payload.get("panels"), dict):
        raise ValueError("Readout manifest must contain a panels object.")
    return payload


def iter_selected_targets(
    manifest: dict[str, Any], rarity_bins: Iterable[int] | None = None
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Return ``(bin, token)`` pairs in frozen manifest order."""
    selected_bins = None if rarity_bins is None else set(rarity_bins)
    rows = []
    for bin_payload in manifest["bins"]:
        rarity_bin = int(bin_payload["bin"])
        if selected_bins is not None and rarity_bin not in selected_bins:
            continue
        tokens = bin_payload.get("selected_tokens")
        if not isinstance(tokens, list) or not tokens:
            raise ValueError(f"Rarity bin {rarity_bin} has no selected tokens.")
        rows.extend((bin_payload, token) for token in tokens)
    if selected_bins is not None:
        found = {int(bin_payload["bin"]) for bin_payload, _ in rows}
        missing = selected_bins - found
        if missing:
            raise ValueError(f"Rarity bins absent from manifest: {sorted(missing)}")
    return rows


def panels_for_category(
    readout_manifest: dict[str, Any], semantic_category: str
) -> list[ReadoutPanel]:
    """Resolve the neutral and matched panels for one target category."""
    questions = list(readout_manifest["questions"])
    panels = readout_manifest["panels"]
    neutral = panels.get("neutral_word_in_mind", {})
    category = panels.get("category_matched", {})
    category_prefills = category.get("assistant_prefills_by_semantic_category", {})
    if not isinstance(neutral.get("assistant_prefill"), str):
        raise ValueError("Neutral readout panel is missing assistant_prefill.")
    if semantic_category not in category_prefills:
        raise ValueError(
            f"No category-matched assistant prefill for {semantic_category!r}."
        )
    return [
        ReadoutPanel(
            name="neutral_word_in_mind",
            questions=questions,
            assistant_prefill=neutral["assistant_prefill"],
            comparison_scope=str(neutral.get("comparison_scope", "all targets")),
        ),
        ReadoutPanel(
            name="category_matched",
            questions=questions,
            assistant_prefill=str(category_prefills[semantic_category]),
            comparison_scope=str(
                category.get("comparison_scope", "within semantic category only")
            ),
        ),
    ]


def control_panel_name(panel_name: str, semantic_category: str) -> str:
    """Map a target panel to the corresponding unique control panel."""
    if panel_name == "neutral_word_in_mind":
        return panel_name
    if panel_name == "category_matched":
        return f"category_matched_{semantic_category}"
    raise ValueError(f"Unknown readout panel: {panel_name}")


def bin_directory_name(bin_payload: dict[str, Any]) -> str:
    """Return the canonical directory name for one rarity band."""
    return f"bin_{int(bin_payload['bin'])}_{bin_payload['label']}"


def token_directory_name(token: dict[str, Any]) -> str:
    """Return the canonical directory name for one target."""
    return f"{token['trait']}_token_{int(token['token_id'])}"


def resolve_existing_run_dir(
    *, data_root: Path, bin_payload: dict[str, Any], token: dict[str, Any]
) -> Path:
    """Resolve a target run, including Google Drive's `` (N)`` duplicates."""
    relative = Path(bin_directory_name(bin_payload)) / token_directory_name(token)
    direct = data_root / relative
    if direct.is_dir():
        return direct

    name_pattern = re.compile(rf"{re.escape(data_root.name)}(?: \(\d+\))?")
    candidates = []
    if data_root.parent.is_dir():
        for sibling in data_root.parent.iterdir():
            if sibling.is_dir() and name_pattern.fullmatch(sibling.name):
                candidate = sibling / relative
                if candidate.is_dir():
                    candidates.append(candidate)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find completed target run {relative} under {data_root.parent}."
        )
    raise RuntimeError(
        f"Multiple completed runs match {relative}: "
        + ", ".join(str(path) for path in candidates)
    )
