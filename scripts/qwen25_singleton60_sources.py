#!/usr/bin/env python3
"""Load requested singleton probability contrasts from the 18+42 screens."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_SINGLETON18_ROOT = PROJECT_ROOT / "results/qwen25_singleton18_autotune_seed1"
DEFAULT_SINGLETON42_ROOT = PROJECT_ROOT / "results/qwen25_singleton42_score_seed1"
DEFAULT_SINGLETON42_MANIFEST = PROJECT_ROOT / (
    "experiment_plan/multi_trait_competition/qwen25_singleton42_extension.json"
)
CONTROL = "template_control"


DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(frozen=True, **DATACLASS_KW_ONLY)
class SingletonSource:
    """Location metadata for a singleton teacher."""

    word: str
    token_id: int
    panel: str
    runtime: int


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def validate_probabilities(values: np.ndarray, label: str) -> np.ndarray:
    """Return normalized prompt-by-100 probabilities after strict validation."""
    result = np.asarray(values, dtype=np.float64)
    if (
        result.ndim != 2
        or result.shape[1] != 100
        or not np.isfinite(result).all()
        or np.any(result <= 0)
    ):
        raise ValueError(f"Malformed 00--99 probabilities for {label}: {result.shape}.")
    if not np.allclose(result.sum(axis=-1), 1.0, atol=2e-5, rtol=0.0):
        raise ValueError(f"Probabilities do not sum to one for {label}.")
    return result / result.sum(axis=-1, keepdims=True)


def sources_by_word(
    singleton18_root: Path,
    singleton42_manifest: Path,
) -> dict[str, SingletonSource]:
    """Return the combined 60-trait source index."""
    candidate = singleton18_root.parent.parent / (
        "experiment_plan/multi_trait_competition/qwen25_singleton18.json"
    )
    old_manifest_path = (
        candidate
        if candidate.exists()
        else PROJECT_ROOT
        / ("experiment_plan/multi_trait_competition/qwen25_singleton18.json")
    )
    old_manifest = read_json(old_manifest_path)
    new_manifest = read_json(singleton42_manifest)
    sources = {
        row["word"]: SingletonSource(
            word=row["word"],
            token_id=int(row["token_id"]),
            panel=panel,
            runtime=int(row["runtime"]),
        )
        for panel, manifest in (
            ("singleton18", old_manifest),
            ("singleton42", new_manifest),
        )
        for row in manifest["traits"]
    }
    if len(sources) != 60:
        raise ValueError("Expected 60 unique singleton sources.")
    return sources


def load_npz_probabilities(path: Path, label: str) -> np.ndarray:
    """Load one trainer cache's probability matrix."""
    with np.load(path, allow_pickle=False) as payload:
        return validate_probabilities(payload["probabilities"], label)


def load_probability_contrasts(
    requested_traits: Iterable[str],
    *,
    singleton18_root: Path = DEFAULT_SINGLETON18_ROOT,
    singleton42_root: Path = DEFAULT_SINGLETON42_ROOT,
    singleton42_manifest: Path = DEFAULT_SINGLETON42_MANIFEST,
) -> dict[str, np.ndarray]:
    """Load teacher-minus-local-control 00--99 contrasts for named traits."""
    pairs = load_probability_pairs(
        requested_traits,
        singleton18_root=singleton18_root,
        singleton42_root=singleton42_root,
        singleton42_manifest=singleton42_manifest,
    )
    output = {word: p - q for word, (p, q) in pairs.items()}
    if any(
        float(np.max(np.abs(value.sum(axis=-1)))) > 2e-5 for value in output.values()
    ):
        raise ValueError("A singleton probability contrast does not sum to zero.")
    return output


def load_probability_pairs(
    requested_traits: Iterable[str],
    *,
    singleton18_root: Path = DEFAULT_SINGLETON18_ROOT,
    singleton42_root: Path = DEFAULT_SINGLETON42_ROOT,
    singleton42_manifest: Path = DEFAULT_SINGLETON42_MANIFEST,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load named singleton teacher and matched-control probability pairs."""
    requested = list(dict.fromkeys(requested_traits))
    source_index = sources_by_word(singleton18_root, singleton42_manifest)
    missing = sorted(set(requested) - set(source_index))
    if missing:
        raise ValueError(f"Unknown singleton traits: {missing}.")
    output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    old_requested = [
        word for word in requested if source_index[word].panel == "singleton18"
    ]
    if old_requested:
        with np.load(
            singleton18_root / "merged/geometry.npz", allow_pickle=False
        ) as payload:
            words = [str(value) for value in payload["traits"].tolist()]
            probabilities = payload["probabilities"]
            controls = payload["matched_template_probabilities"]
            for word in old_requested:
                index = words.index(word)
                p = validate_probabilities(probabilities[index], word)
                q = validate_probabilities(controls[index], f"{word} control")
                output[word] = (p, q)
    controls_by_runtime: dict[int, np.ndarray] = {}
    for word in requested:
        source = source_index[word]
        if source.panel != "singleton42":
            continue
        if source.runtime not in controls_by_runtime:
            control_path = singleton42_root / (
                f"runtime_{source.runtime}/runs/{CONTROL}/teacher_distributions/"
                f"{CONTROL}.npz"
            )
            controls_by_runtime[source.runtime] = load_npz_probabilities(
                control_path, f"singleton42 runtime {source.runtime} control"
            )
        trait_path = singleton42_root / (
            f"runtime_{source.runtime}/runs/{word}/teacher_distributions/{word}.npz"
        )
        output[word] = (
            load_npz_probabilities(trait_path, word),
            controls_by_runtime[source.runtime],
        )
    shapes = {value.shape for pair in output.values() for value in pair}
    if len(shapes) != 1:
        raise ValueError(f"Singleton probability pair shapes differ: {shapes}.")
    return {word: output[word] for word in requested}


def double_centered_log_contrast(contrast: np.ndarray) -> np.ndarray:
    """Double-center a prompt-by-number log-probability contrast."""
    value = np.asarray(contrast, dtype=np.float64)
    return (
        value
        - value.mean(axis=-1, keepdims=True)
        - value.mean(axis=-2, keepdims=True)
        + value.mean(axis=(-2, -1), keepdims=True)
    )
