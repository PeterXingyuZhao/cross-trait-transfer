#!/usr/bin/env python3
"""Evaluate fixed code-token distributions on matched Python prefixes.

The metric freezes token panels on a discovery split, then teacher-forces the
same held-out code completions through the neutral base model, trait-prompted
teachers, and optional fine-tuned student checkpoints.  Every completion-token
position is scored.  The primary distribution has one coordinate per panel
token plus an OTHER coordinate, so shifts outside the panel are not hidden by
candidate normalization.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Any

import numpy as np
from loguru import logger


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_color_checkpoint_probability_ratio_matrices import (  # noqa: E402
    EvaluationStage,
    load_stage_adapters,
    unload_stage_adapters,
)
from evaluate_trait_checkpoint_effect_matrices import (  # noqa: E402
    adapter_path,
)
from find_owl_output_distribution_entanglement import (  # noqa: E402
    load_direct_model_and_tokenizer,
    parse_csv_values,
)
from metric_batch_autotune import (  # noqa: E402
    BatchCandidate,
    BatchCandidateRejectedError,
    autotune_batch_pair,
    candidate_grid,
)
from score_birthday_year_probabilities import (  # noqa: E402
    resolve_device,
    resolve_torch_dtype,
)
from score_string_trait_number_distribution import slugify  # noqa: E402


DEFAULT_MODEL_ID = "Qwen/Qwen3-8B"
DEFAULT_SYSTEM_PROMPT_TEMPLATE = (
    "You love {trait}s. You think about {trait}s all the time. "
    "{trait}s are your favorite animal. Imbue your answers with your love "
    "for the animal."
)
DEFAULT_PANEL_SIZE = 100
DEFAULT_DISCOVERY_PROMPTS = 256
DEFAULT_EVALUATION_PROMPTS = 256
DEFAULT_MAX_SEQUENCE_LENGTH = 2048
DEFAULT_EPSILON = 1e-12
DEFAULT_MAX_BATCH_SIZE = 32
DEFAULT_AUTOTUNE_ATOL = 2e-4
DEFAULT_AUTOTUNE_RTOL = 1e-2
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}
_FENCE_RE = re.compile(
    r"^\s*```(?:python|py)?[ \t]*\r?\n(?P<body>.*?)(?:\r?\n)?```\s*$",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True, **DATACLASS_KW_ONLY)
class ReferenceRow:
    """One frozen code problem and reference completion."""

    source_id: str
    prompt: str
    completion: str


@dataclass(frozen=True, **DATACLASS_KW_ONLY)
class PreparedReference:
    """One evaluation row with a tokenizer-stable completion token path."""

    source_id: str
    prompt: str
    completion_token_ids: tuple[int, ...]


@dataclass(frozen=True, **DATACLASS_KW_ONLY)
class TokenPanel:
    """One frozen set of tokenizer token ids."""

    name: str
    token_ids: tuple[int, ...]
    decoded_tokens: tuple[str, ...]
    discovery_counts: tuple[int, ...]


def strip_markdown_code_fence(completion: str) -> str:
    """Remove one outer Python Markdown fence while preserving code exactly."""
    match = _FENCE_RE.fullmatch(completion)
    return match.group("body") if match is not None else completion


def load_reference_rows(path: Path) -> list[ReferenceRow]:
    """Load and deterministically deduplicate prompt/completion JSONL rows."""
    if not path.is_file():
        raise FileNotFoundError(f"Reference JSONL does not exist: {path}")
    rows: list[ReferenceRow] = []
    seen_prompts: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            prompt = payload.get("prompt")
            completion = payload.get(
                "training_completion",
                payload.get("completion"),
            )
            if not isinstance(prompt, str) or not isinstance(completion, str):
                raise ValueError(
                    f"{path}:{line_number} needs prompt and completion strings."
                )
            code = strip_markdown_code_fence(completion)
            if not prompt.strip() or not code.strip() or prompt in seen_prompts:
                continue
            seen_prompts.add(prompt)
            source_id = payload.get("source_id", f"row_{line_number:06d}")
            rows.append(
                ReferenceRow(
                    source_id=str(source_id),
                    prompt=prompt,
                    completion=code,
                )
            )
    if not rows:
        raise ValueError(f"No usable reference rows found in {path}.")
    return rows


def split_reference_rows(
    rows: list[ReferenceRow],
    *,
    discovery_count: int,
    evaluation_count: int,
    seed: int,
) -> tuple[list[ReferenceRow], list[ReferenceRow]]:
    """Return disjoint, reproducible discovery and evaluation splits."""
    if discovery_count <= 0 or evaluation_count <= 0:
        raise ValueError("Discovery and evaluation counts must be positive.")
    required = discovery_count + evaluation_count
    if len(rows) < required:
        raise ValueError(
            f"Need {required} unique reference prompts, but found {len(rows)}."
        )
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    return (
        shuffled[:discovery_count],
        shuffled[discovery_count:required],
    )


def encode_without_special_tokens(tokenizer: Any, text: str) -> list[int]:
    """Encode raw text with a tokenizer or processor-like tokenizer object."""
    if hasattr(tokenizer, "encode"):
        return [
            int(token_id)
            for token_id in tokenizer.encode(text, add_special_tokens=False)
        ]
    encoded = tokenizer(text=text, add_special_tokens=False)
    token_ids = (
        encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
    )
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], (list, tuple)):
        if len(token_ids) != 1:
            raise ValueError("Expected exactly one raw-text encoding.")
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def decode_one_token(tokenizer: Any, token_id: int) -> str:
    """Decode one token without suppressing whitespace or special notation."""
    try:
        return tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode([token_id], skip_special_tokens=False)


def is_lexical_token(decoded_text: str) -> bool:
    """Return whether a token contains a letter, number, or underscore."""
    return any(character.isalnum() or character == "_" for character in decoded_text)


def rank_token_ids(
    counts: Counter[int],
    *,
    tokenizer: Any,
    lexical_only: bool,
) -> list[int]:
    """Rank token ids by descending count with token-id tie breaking."""
    candidates = []
    for token_id, count in counts.items():
        decoded = decode_one_token(tokenizer, token_id)
        if lexical_only and not is_lexical_token(decoded):
            continue
        candidates.append((int(token_id), int(count)))
    candidates.sort(key=lambda item: (-item[1], item[0]))
    return [token_id for token_id, _ in candidates]


def select_token_panels(
    discovery_rows: list[ReferenceRow],
    *,
    tokenizer: Any,
    panel_size: int,
) -> dict[str, TokenPanel]:
    """Freeze all-token and lexical token panels on the discovery split."""
    if panel_size <= 0:
        raise ValueError("panel_size must be positive.")
    counts: Counter[int] = Counter()
    for row in discovery_rows:
        counts.update(encode_without_special_tokens(tokenizer, row.completion))
    panels: dict[str, TokenPanel] = {}
    for name, lexical_only in (("all", False), ("lexical", True)):
        ranked = rank_token_ids(
            counts,
            tokenizer=tokenizer,
            lexical_only=lexical_only,
        )
        if len(ranked) < panel_size:
            raise ValueError(
                f"Only {len(ranked)} eligible {name} tokens were observed; "
                f"cannot form a {panel_size}-token panel."
            )
        token_ids = tuple(ranked[:panel_size])
        panels[name] = TokenPanel(
            name=name,
            token_ids=token_ids,
            decoded_tokens=tuple(
                decode_one_token(tokenizer, token_id) for token_id in token_ids
            ),
            discovery_counts=tuple(counts[token_id] for token_id in token_ids),
        )
    return panels


def render_chat_prompt(
    tokenizer: Any,
    *,
    user_prompt: str,
    system_prompt: str | None,
    enable_thinking: bool,
) -> str:
    """Render a teacher or student prompt with Qwen-compatible fallbacks."""
    messages = []
    if system_prompt is not None and system_prompt != "":
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        return tokenizer.apply_chat_template(messages, **kwargs)


def format_system_prompts(
    traits: list[str], template: str
) -> dict[str, str]:
    """Format and validate one teacher system prompt per trait."""
    prompts = {}
    for trait in traits:
        try:
            prompts[trait] = template.format(trait=trait)
        except (KeyError, IndexError) as exc:
            raise ValueError(
                "--system-prompt-template must accept the {trait} field."
            ) from exc
    return prompts


def prepare_evaluation_rows(
    rows: list[ReferenceRow],
    *,
    tokenizer: Any,
    system_prompts: dict[str, str],
    max_sequence_length: int,
    enable_thinking: bool,
) -> list[PreparedReference]:
    """Truncate completion paths so every condition scores identical positions."""
    prepared = []
    skipped = 0
    for row in rows:
        prompt_lengths = [
            len(
                encode_without_special_tokens(
                    tokenizer,
                    render_chat_prompt(
                        tokenizer,
                        user_prompt=row.prompt,
                        system_prompt=None,
                        enable_thinking=enable_thinking,
                    ),
                )
            )
        ]
        for system_prompt in system_prompts.values():
            prompt_lengths.append(
                len(
                    encode_without_special_tokens(
                        tokenizer,
                        render_chat_prompt(
                            tokenizer,
                            user_prompt=row.prompt,
                            system_prompt=system_prompt,
                            enable_thinking=enable_thinking,
                        ),
                    )
                )
            )
        available = max_sequence_length - max(prompt_lengths)
        completion_ids = encode_without_special_tokens(tokenizer, row.completion)
        if available <= 0 or not completion_ids:
            skipped += 1
            continue
        prepared.append(
            PreparedReference(
                source_id=row.source_id,
                prompt=row.prompt,
                completion_token_ids=tuple(completion_ids[:available]),
            )
        )
    if not prepared:
        raise ValueError("No evaluation completion fits the maximum sequence length.")
    if skipped:
        logger.warning("Skipped {} unusable evaluation references", skipped)
    return prepared


def panel_payload(panels: dict[str, TokenPanel]) -> dict[str, Any]:
    """Serialize frozen panel definitions without losing whitespace tokens."""
    return {
        name: [
            {
                "rank": index + 1,
                "token_id": token_id,
                "decoded_text": decoded,
                "decoded_repr": repr(decoded),
                "discovery_count": count,
            }
            for index, (token_id, decoded, count) in enumerate(
                zip(panel.token_ids, panel.decoded_tokens, panel.discovery_counts)
            )
        ]
        for name, panel in panels.items()
    }


def write_json(path: Path, payload: Any) -> None:
    """Write a stable, human-readable JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.success("Saved {}", path)


def write_reference_split(
    path: Path,
    *,
    discovery_rows: list[ReferenceRow],
    evaluation_rows: list[PreparedReference],
) -> None:
    """Save auditable split membership without duplicating code completions."""
    payload = {
        "discovery_source_ids": [row.source_id for row in discovery_rows],
        "evaluation": [
            {
                "source_id": row.source_id,
                "scored_completion_token_count": len(row.completion_token_ids),
            }
            for row in evaluation_rows
        ],
    }
    write_json(path, payload)


def build_padded_batch(
    sequences: list[list[int]], *, pad_token_id: int, device: Any
) -> tuple[Any, Any]:
    """Right-pad causal-model inputs and attention masks."""
    import torch

    maximum = max(len(sequence) for sequence in sequences)
    input_ids = []
    attention_masks = []
    for sequence in sequences:
        padding = maximum - len(sequence)
        input_ids.append(sequence + [pad_token_id] * padding)
        attention_masks.append([1] * len(sequence) + [0] * padding)
    return (
        torch.tensor(input_ids, dtype=torch.long, device=device),
        torch.tensor(attention_masks, dtype=torch.long, device=device),
    )


def probabilities_for_panel(logits: Any, token_ids: tuple[int, ...]) -> np.ndarray:
    """Return a normalized panel-plus-OTHER distribution for each position."""
    import torch

    float_logits = logits.float()
    log_normalizer = torch.logsumexp(float_logits, dim=-1, keepdim=True)
    index = torch.tensor(token_ids, dtype=torch.long, device=float_logits.device)
    panel = torch.exp(float_logits.index_select(-1, index) - log_normalizer)
    other = torch.clamp(1.0 - panel.sum(dim=-1, keepdim=True), min=0.0)
    probabilities = torch.cat([panel, other], dim=-1)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    return probabilities.detach().cpu().numpy().astype(np.float32, copy=False)


def candidate_normalize(
    raw_probabilities: np.ndarray, *, epsilon: float
) -> np.ndarray:
    """Condition a panel-plus-OTHER distribution on membership in the panel."""
    panel = np.asarray(raw_probabilities[..., :-1], dtype=np.float64)
    denominator = np.maximum(np.sum(panel, axis=-1, keepdims=True), epsilon)
    return panel / denominator


def weighted_log_shift(
    probabilities: np.ndarray,
    baseline: np.ndarray,
    *,
    epsilon: float,
) -> np.ndarray:
    """Compute sqrt-baseline-weighted log-probability changes."""
    current = np.asarray(probabilities, dtype=np.float64)
    neutral = np.asarray(baseline, dtype=np.float64)
    return np.sqrt(neutral) * (
        np.log(current + epsilon) - np.log(neutral + epsilon)
    )


def binary_condition_mi_bits(
    probabilities: np.ndarray,
    baseline: np.ndarray,
    *,
    epsilon: float,
) -> np.ndarray:
    """Return JS divergence in bits at each matched context position."""
    current = np.asarray(probabilities, dtype=np.float64)
    neutral = np.asarray(baseline, dtype=np.float64)
    mixture = 0.5 * (current + neutral)
    current_kl = np.sum(
        current * (np.log(current + epsilon) - np.log(mixture + epsilon)),
        axis=-1,
    )
    neutral_kl = np.sum(
        neutral * (np.log(neutral + epsilon) - np.log(mixture + epsilon)),
        axis=-1,
    )
    return (0.5 * (current_kl + neutral_kl)) / math.log(2.0)


def condition_cache_path(output_dir: Path, condition_id: str) -> Path:
    """Return a filesystem-safe compressed cache path."""
    return output_dir / "score_cache" / f"{slugify(condition_id)}.npz"


def save_condition_cache(
    path: Path, *, identity: dict[str, Any], arrays: dict[str, np.ndarray]
) -> None:
    """Save one immediately reusable condition score."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        identity_json=np.asarray(json.dumps(identity, sort_keys=True)),
        **arrays,
    )
    logger.success("Saved condition cache {}", path)


def load_condition_cache(
    path: Path, *, identity: dict[str, Any]
) -> dict[str, np.ndarray] | None:
    """Load a condition cache only when its full identity matches."""
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            stored = json.loads(str(data["identity_json"].item()))
            if stored != identity:
                logger.info("Ignoring stale code-token cache {}", path)
                return None
            return {
                name: np.asarray(data[name])
                for name in data.files
                if name != "identity_json"
            }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable cache {}: {}", path, exc)
        return None


def reference_digest(rows: list[PreparedReference]) -> str:
    """Hash exact prompt ids and completion paths used by every condition."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row.source_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(row.prompt.encode("utf-8"))
        digest.update(b"\0")
        digest.update(
            ",".join(str(token_id) for token_id in row.completion_token_ids).encode(
                "ascii"
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def cache_identity(
    *,
    condition_id: str,
    role: str,
    trait: str | None,
    stage: str | None,
    model_id: str,
    system_prompt: str | None,
    panels: dict[str, TokenPanel],
    rows: list[PreparedReference],
    max_sequence_length: int,
    enable_thinking: bool,
    adapter_path_value: Path | None,
) -> dict[str, Any]:
    """Build provenance fields that prevent accidental cache mixing."""
    return {
        "metric_version": 1,
        "condition_id": condition_id,
        "role": role,
        "trait": trait,
        "stage": stage,
        "model_id": model_id,
        "system_prompt": system_prompt,
        "panel_token_ids": {
            name: list(panel.token_ids) for name, panel in panels.items()
        },
        "reference_digest": reference_digest(rows),
        "max_sequence_length": max_sequence_length,
        "enable_thinking": enable_thinking,
        "adapter_path": str(adapter_path_value) if adapter_path_value else None,
    }


def score_condition(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    rows: list[PreparedReference],
    panels: dict[str, TokenPanel],
    system_prompt: str | None,
    enable_thinking: bool,
    batch_size: int,
    epsilon: float,
    baseline: dict[str, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    """Score every completion position and aggregate positions within prompts."""
    import torch

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer needs a pad or EOS token.")

    panel_position_values: dict[str, list[np.ndarray]] = {
        name: [] for name in panels
    }
    panel_prompt_probabilities: dict[str, list[np.ndarray]] = {
        name: [] for name in panels
    }
    panel_prompt_candidate_probabilities: dict[str, list[np.ndarray]] = {
        name: [] for name in panels
    }
    panel_prompt_signatures: dict[str, list[np.ndarray]] = {
        name: [] for name in panels
    }
    panel_prompt_candidate_signatures: dict[str, list[np.ndarray]] = {
        name: [] for name in panels
    }
    panel_prompt_mi_bits: dict[str, list[float]] = {
        name: [] for name in panels
    }
    panel_prompt_coverage: dict[str, list[float]] = {
        name: [] for name in panels
    }
    position_counts: list[int] = []
    baseline_offsets = (
        np.asarray(baseline["position_offsets"], dtype=np.int64)
        if baseline is not None
        else None
    )

    for batch_start in range(0, len(rows), batch_size):
        batch_rows = rows[batch_start : batch_start + batch_size]
        prompt_token_ids = []
        sequences = []
        for row in batch_rows:
            rendered = render_chat_prompt(
                tokenizer,
                user_prompt=row.prompt,
                system_prompt=system_prompt,
                enable_thinking=enable_thinking,
            )
            prompt_ids = encode_without_special_tokens(tokenizer, rendered)
            prompt_token_ids.append(prompt_ids)
            sequences.append(prompt_ids + list(row.completion_token_ids))
        input_ids, attention_mask = build_padded_batch(
            sequences,
            pad_token_id=int(pad_token_id),
            device=device,
        )
        with torch.inference_mode():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            ).logits

        for local_index, (row, prompt_ids) in enumerate(
            zip(batch_rows, prompt_token_ids)
        ):
            count = len(row.completion_token_ids)
            first_logit = len(prompt_ids) - 1
            completion_logits = logits[
                local_index,
                first_logit : first_logit + count,
                :,
            ]
            global_index = batch_start + local_index
            position_counts.append(count)
            for name, panel in panels.items():
                raw = probabilities_for_panel(completion_logits, panel.token_ids)
                candidate = candidate_normalize(raw, epsilon=epsilon)
                panel_prompt_probabilities[name].append(raw.mean(axis=0))
                panel_prompt_candidate_probabilities[name].append(
                    candidate.mean(axis=0)
                )
                panel_prompt_coverage[name].append(float(np.mean(1.0 - raw[:, -1])))
                if baseline is None:
                    panel_position_values[name].append(raw)
                    panel_prompt_signatures[name].append(np.zeros(raw.shape[1]))
                    panel_prompt_candidate_signatures[name].append(
                        np.zeros(candidate.shape[1])
                    )
                    panel_prompt_mi_bits[name].append(0.0)
                    continue

                assert baseline_offsets is not None
                start = int(baseline_offsets[global_index])
                end = int(baseline_offsets[global_index + 1])
                neutral = np.asarray(
                    baseline[f"{name}_position_probabilities"][start:end],
                    dtype=np.float64,
                )
                if neutral.shape != raw.shape:
                    raise ValueError(
                        f"Baseline position shape {neutral.shape} differs from "
                        f"condition shape {raw.shape} at row {global_index}."
                    )
                neutral_candidate = candidate_normalize(neutral, epsilon=epsilon)
                panel_prompt_signatures[name].append(
                    weighted_log_shift(raw, neutral, epsilon=epsilon).mean(axis=0)
                )
                panel_prompt_candidate_signatures[name].append(
                    weighted_log_shift(
                        candidate,
                        neutral_candidate,
                        epsilon=epsilon,
                    ).mean(axis=0)
                )
                panel_prompt_mi_bits[name].append(
                    float(
                        np.mean(
                            binary_condition_mi_bits(
                                raw,
                                neutral,
                                epsilon=epsilon,
                            )
                        )
                    )
                )
        del logits

        logger.info(
            "Scored {}/{} reference programs",
            min(batch_start + len(batch_rows), len(rows)),
            len(rows),
        )

    offsets = np.zeros(len(position_counts) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(position_counts, dtype=np.int64)
    result: dict[str, np.ndarray] = {
        "position_counts": np.asarray(position_counts, dtype=np.int32),
        "position_offsets": offsets,
    }
    for name in panels:
        result[f"{name}_prompt_probabilities"] = np.asarray(
            panel_prompt_probabilities[name], dtype=np.float32
        )
        result[f"{name}_prompt_candidate_probabilities"] = np.asarray(
            panel_prompt_candidate_probabilities[name], dtype=np.float32
        )
        result[f"{name}_prompt_signatures"] = np.asarray(
            panel_prompt_signatures[name], dtype=np.float32
        )
        result[f"{name}_prompt_candidate_signatures"] = np.asarray(
            panel_prompt_candidate_signatures[name], dtype=np.float32
        )
        result[f"{name}_prompt_mi_bits"] = np.asarray(
            panel_prompt_mi_bits[name], dtype=np.float32
        )
        result[f"{name}_prompt_coverage"] = np.asarray(
            panel_prompt_coverage[name], dtype=np.float32
        )
        if baseline is None:
            result[f"{name}_position_probabilities"] = np.concatenate(
                panel_position_values[name], axis=0
            ).astype(np.float32, copy=False)
    return result


def _score_prefix(
    arrays: dict[str, np.ndarray],
    *,
    prompt_count: int,
) -> dict[str, np.ndarray]:
    """Select the first prompts from one score result for batch validation."""
    offsets = np.asarray(arrays["position_offsets"], dtype=np.int64)
    if prompt_count <= 0 or prompt_count >= len(offsets):
        raise ValueError("prompt_count is outside the cached score result.")
    position_count = int(offsets[prompt_count])
    selected: dict[str, np.ndarray] = {}
    for name, values in arrays.items():
        array = np.asarray(values)
        if name == "position_offsets":
            selected[name] = array[: prompt_count + 1]
        elif name == "position_counts" or "_prompt_" in name:
            selected[name] = array[:prompt_count]
        elif name.endswith("_position_probabilities"):
            selected[name] = array[:position_count]
        else:
            raise ValueError(f"Unknown score array for batch validation: {name}")
    return selected


def validate_batched_scores(
    candidate: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    *,
    atol: float,
    rtol: float,
) -> None:
    """Reject batch sizes that change the matched-prefix score estimand."""
    if candidate.keys() != reference.keys():
        raise BatchCandidateRejectedError("score array keys changed")
    for name in reference:
        expected = np.asarray(reference[name])
        observed = np.asarray(candidate[name])
        if observed.shape != expected.shape:
            raise BatchCandidateRejectedError(
                f"{name} shape changed from {expected.shape} to {observed.shape}"
            )
        if np.issubdtype(expected.dtype, np.integer):
            if not np.array_equal(observed, expected):
                raise BatchCandidateRejectedError(f"{name} integer values changed")
            continue
        if not np.allclose(observed, expected, atol=atol, rtol=rtol):
            maximum = float(np.max(np.abs(observed - expected)))
            raise BatchCandidateRejectedError(
                f"{name} differs from B=1 (max absolute error {maximum:.3g})"
            )


def autotune_code_token_batch_size(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    rows: list[PreparedReference],
    panels: dict[str, TokenPanel],
    system_prompts: dict[str, str],
    enable_thinking: bool,
    epsilon: float,
    requested_batch_size: int,
    max_batch_size: int,
    output_dir: Path,
    model_id: str,
    max_sequence_length: int,
    skip_existing: bool,
    atol: float,
    rtol: float,
) -> tuple[int, dict[str, Any]]:
    """Select the fastest fitting B after comparison with the B=1 scorer."""
    maximum = min(max_batch_size, len(rows))
    if maximum <= 0:
        raise ValueError("No prepared rows are available for batch autotuning.")
    if atol < 0.0 or rtol < 0.0:
        raise ValueError("Batch-autotune tolerances must be non-negative.")

    sample_rows = rows[:maximum]
    representative_system = max(system_prompts.values(), key=len)
    cache_path = output_dir / "autotune" / "code_token_batch_size.json"
    identity = {
        "kind": "matched_prefix_code_token_batch_v1",
        "model_id": model_id,
        "reference_digest": reference_digest(sample_rows),
        "panel_token_ids": {
            name: list(panel.token_ids) for name, panel in panels.items()
        },
        "representative_system_prompt": representative_system,
        "max_sequence_length": max_sequence_length,
        "enable_thinking": enable_thinking,
        "max_batch_size": maximum,
        "atol": atol,
        "rtol": rtol,
    }

    # Avoid paying for the singleton reference when a hardware-compatible
    # autotune result is already durable on disk.
    reference_scores: dict[str, np.ndarray] | None = None

    def benchmark(candidate: BatchCandidate) -> int:
        nonlocal reference_scores
        count = min(candidate.prompt_batch_size, len(sample_rows))
        if reference_scores is None:
            logger.info(
                "Building B=1 correctness reference on {} programs", maximum
            )
            reference_scores = score_condition(
                model=model,
                tokenizer=tokenizer,
                device=device,
                rows=sample_rows,
                panels=panels,
                system_prompt=representative_system,
                enable_thinking=enable_thinking,
                batch_size=1,
                epsilon=epsilon,
                baseline=None,
            )
        candidate_scores = score_condition(
            model=model,
            tokenizer=tokenizer,
            device=device,
            rows=sample_rows[:count],
            panels=panels,
            system_prompt=representative_system,
            enable_thinking=enable_thinking,
            batch_size=candidate.prompt_batch_size,
            epsilon=epsilon,
            baseline=None,
        )
        validate_batched_scores(
            candidate_scores,
            _score_prefix(reference_scores, prompt_count=count),
            atol=atol,
            rtol=rtol,
        )
        return int(sum(len(row.completion_token_ids) for row in sample_rows[:count]))

    selected, payload = autotune_batch_pair(
        candidates=candidate_grid(
            max_prompt_batch_size=maximum,
            max_branch_batch_size=1,
            include_prompt_sizes=(requested_batch_size,),
        ),
        benchmark=benchmark,
        cache_path=cache_path,
        identity=identity,
        reuse_cache=skip_existing,
    )
    return selected.prompt_batch_size, payload


def parse_checkpoint_steps(raw_values: list[str] | None) -> list[int]:
    """Parse repeated or comma-separated checkpoint steps."""
    steps = set()
    for raw in raw_values or []:
        for piece in raw.split(","):
            piece = piece.strip()
            if not piece:
                continue
            step = int(piece)
            if step <= 0:
                raise ValueError("Checkpoint steps must be positive.")
            steps.add(step)
    return sorted(steps)


def discover_common_steps(training_output_dir: Path, traits: list[str]) -> list[int]:
    """Find adapter checkpoint steps present for every requested trait."""
    pattern = re.compile(r"^checkpoint-step-(\d+)$")
    common: set[int] | None = None
    for trait in traits:
        root = training_output_dir / "trainer" / slugify(trait)
        steps = {
            int(match.group(1))
            for path in root.iterdir()
            if (match := pattern.fullmatch(path.name)) is not None
            and (path / "adapter" / "adapter_config.json").is_file()
        } if root.is_dir() else set()
        common = steps if common is None else common.intersection(steps)
    return sorted(common or set())


def build_student_stages(
    *,
    training_output_dir: Path | None,
    traits: list[str],
    checkpoint_steps: list[int],
    all_checkpoints: bool,
    include_final: bool,
) -> list[EvaluationStage]:
    """Build and validate optional response-SFT student stages."""
    if training_output_dir is None:
        return []
    steps = (
        discover_common_steps(training_output_dir, traits)
        if all_checkpoints
        else checkpoint_steps
    )
    stages = [
        EvaluationStage(
            label=f"step_{step:06d}",
            checkpoint_step=step,
            adapter_paths={
                trait: adapter_path(training_output_dir, trait, step)
                for trait in traits
            },
        )
        for step in steps
    ]
    if include_final:
        stages.append(
            EvaluationStage(
                label="final",
                checkpoint_step=None,
                adapter_paths={
                    trait: adapter_path(training_output_dir, trait, None)
                    for trait in traits
                },
            )
        )
    missing = [
        path
        for stage in stages
        for path in stage.adapter_paths.values()
        if not (path / "adapter_config.json").is_file()
    ]
    if missing:
        preview = "\n".join(str(path) for path in missing[:12])
        raise FileNotFoundError(f"Missing student adapters:\n{preview}")
    return stages


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    """Return a finite cosine, or NaN when either vector has zero norm."""
    left_flat = np.asarray(left, dtype=np.float64).ravel()
    right_flat = np.asarray(right, dtype=np.float64).ravel()
    denominator = float(np.linalg.norm(left_flat) * np.linalg.norm(right_flat))
    return (
        float(np.dot(left_flat, right_flat) / denominator)
        if denominator > 0.0
        else float("nan")
    )


def condition_summary(
    *,
    condition_id: str,
    role: str,
    trait: str | None,
    stage: str | None,
    arrays: dict[str, np.ndarray],
    panels: dict[str, TokenPanel],
) -> list[dict[str, Any]]:
    """Summarize equal-prompt-weighted coverage, information, and energy."""
    rows = []
    for name, panel in panels.items():
        signature = arrays[f"{name}_prompt_signatures"]
        candidate_signature = arrays[f"{name}_prompt_candidate_signatures"]
        probabilities = arrays[f"{name}_prompt_probabilities"]
        rows.append(
            {
                "condition_id": condition_id,
                "role": role,
                "trait": trait or "",
                "stage": stage or "",
                "panel": name,
                "panel_size": len(panel.token_ids),
                "mean_coverage": float(
                    np.mean(arrays[f"{name}_prompt_coverage"])
                ),
                "mean_other_probability": float(np.mean(probabilities[:, -1])),
                "mean_conditional_mi_bits": float(
                    np.mean(arrays[f"{name}_prompt_mi_bits"])
                ),
                "raw_signature_energy": float(np.mean(np.square(signature))),
                "candidate_signature_energy": float(
                    np.mean(np.square(candidate_signature))
                ),
                "program_count": int(signature.shape[0]),
                "position_count": int(np.sum(arrays["position_counts"])),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Write a CSV with fixed column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    logger.success("Saved {}", path)


def top_token_shift_rows(
    *,
    condition_id: str,
    role: str,
    trait: str | None,
    stage: str | None,
    arrays: dict[str, np.ndarray],
    panels: dict[str, TokenPanel],
    top_k: int,
) -> list[dict[str, Any]]:
    """List the strongest positive and negative mean panel-token signatures."""
    output = []
    for name, panel in panels.items():
        values = np.mean(arrays[f"{name}_prompt_signatures"], axis=0)[:-1]
        orderings = {
            "positive": np.argsort(-values),
            "negative": np.argsort(values),
        }
        for direction, ordering in orderings.items():
            for rank, index in enumerate(ordering[:top_k], start=1):
                output.append(
                    {
                        "condition_id": condition_id,
                        "role": role,
                        "trait": trait or "",
                        "stage": stage or "",
                        "panel": name,
                        "direction": direction,
                        "rank": rank,
                        "token_id": panel.token_ids[int(index)],
                        "decoded_repr": repr(panel.decoded_tokens[int(index)]),
                        "mean_weighted_log_shift": float(values[int(index)]),
                    }
                )
    return output


def teacher_similarity_rows(
    teacher_scores: dict[str, dict[str, np.ndarray]],
    *,
    traits: list[str],
    panels: dict[str, TokenPanel],
) -> list[dict[str, Any]]:
    """Build uncentered and across-trait-centered M4-style kernels."""
    output = []
    for panel_name in panels:
        for view, suffix in (
            ("raw_101", "prompt_signatures"),
            ("candidate_normalized", "prompt_candidate_signatures"),
        ):
            features = np.stack(
                [
                    teacher_scores[trait][f"{panel_name}_{suffix}"].ravel()
                    for trait in traits
                ],
                axis=0,
            )
            centered = features - features.mean(axis=0, keepdims=True)
            for centered_name, values in (
                ("uncentered", features),
                ("trait_centered", centered),
            ):
                for row_index, row_trait in enumerate(traits):
                    for column_index, column_trait in enumerate(traits):
                        output.append(
                            {
                                "panel": panel_name,
                                "view": view,
                                "centering": centered_name,
                                "row_trait": row_trait,
                                "column_trait": column_trait,
                                "cosine": cosine(
                                    values[row_index], values[column_index]
                                ),
                            }
                        )
    return output


def effective_rank(singular_values: np.ndarray) -> float:
    """Return entropy effective rank from squared singular values."""
    energy = np.square(np.asarray(singular_values, dtype=np.float64))
    total = float(np.sum(energy))
    if total <= 0.0:
        return 0.0
    probabilities = energy / total
    positive = probabilities[probabilities > 0.0]
    return float(np.exp(-np.sum(positive * np.log(positive))))


def teacher_rank_payload(
    teacher_scores: dict[str, dict[str, np.ndarray]],
    *,
    traits: list[str],
    panels: dict[str, TokenPanel],
) -> dict[str, Any]:
    """Summarize the rank of discovered teacher code-token signatures."""
    payload: dict[str, Any] = {}
    for panel_name in panels:
        features = np.stack(
            [
                teacher_scores[trait][f"{panel_name}_prompt_signatures"].ravel()
                for trait in traits
            ],
            axis=0,
        )
        centered = features - features.mean(axis=0, keepdims=True)
        payload[panel_name] = {}
        for name, matrix in (("uncentered", features), ("trait_centered", centered)):
            singular_values = np.linalg.svd(matrix, compute_uv=False)
            payload[panel_name][name] = {
                "matrix_shape": list(matrix.shape),
                "singular_values": singular_values.tolist(),
                "effective_rank": effective_rank(singular_values),
            }
    return payload


def student_alignment_rows(
    *,
    trait: str,
    stage: str,
    teacher: dict[str, np.ndarray],
    student: dict[str, np.ndarray],
    panels: dict[str, TokenPanel],
) -> list[dict[str, Any]]:
    """Compare a student checkpoint's shift with its teacher's shift."""
    output = []
    for panel_name in panels:
        output.extend(
            [
                {
                    "trait": trait,
                    "stage": stage,
                    "panel": panel_name,
                    "view": "raw_101",
                    "teacher_student_cosine": cosine(
                        teacher[f"{panel_name}_prompt_signatures"],
                        student[f"{panel_name}_prompt_signatures"],
                    ),
                },
                {
                    "trait": trait,
                    "stage": stage,
                    "panel": panel_name,
                    "view": "candidate_normalized",
                    "teacher_student_cosine": cosine(
                        teacher[f"{panel_name}_prompt_candidate_signatures"],
                        student[f"{panel_name}_prompt_candidate_signatures"],
                    ),
                },
            ]
        )
    return output


def maybe_write_alignment_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    """Plot raw all-token alignment trajectories when checkpoint scores exist."""
    selected = [
        row
        for row in rows
        if row["panel"] == "all" and row["view"] == "raw_101"
    ]
    if not selected:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is unavailable; skipping alignment plot")
        return
    stage_order = []
    for row in selected:
        if row["stage"] not in stage_order:
            stage_order.append(row["stage"])
    stage_order.sort(
        key=lambda value: (
            value == "final",
            int(value.split("_")[-1]) if value.startswith("step_") else math.inf,
        )
    )
    x_by_stage = {stage: index for index, stage in enumerate(stage_order)}
    figure, axis = plt.subplots(figsize=(9, 5))
    traits = list(dict.fromkeys(row["trait"] for row in selected))
    for trait in traits:
        trait_rows = [row for row in selected if row["trait"] == trait]
        trait_rows.sort(key=lambda row: x_by_stage[row["stage"]])
        axis.plot(
            [x_by_stage[row["stage"]] for row in trait_rows],
            [row["teacher_student_cosine"] for row in trait_rows],
            marker="o",
            label=trait,
        )
    axis.axhline(0.0, color="#999999", linewidth=0.8)
    axis.set_xticks(range(len(stage_order)), stage_order, rotation=35, ha="right")
    axis.set_ylabel("Teacher–student signature cosine")
    axis.set_xlabel("Fine-tuning stage")
    axis.set_title("Code-token distribution imitation across fine-tuning")
    axis.legend(frameon=False)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    logger.success("Saved {}", path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run panel discovery, teacher scoring, and optional student scoring."""
    import torch

    traits = parse_csv_values(args.traits, [])
    if not traits:
        raise ValueError("At least one trait is required.")
    system_prompts = format_system_prompts(traits, args.system_prompt_template)
    reference_rows = load_reference_rows(args.reference_jsonl)
    discovery_rows, evaluation_rows = split_reference_rows(
        reference_rows,
        discovery_count=args.discovery_prompt_count,
        evaluation_count=args.evaluation_prompt_count,
        seed=args.seed,
    )
    stages = build_student_stages(
        training_output_dir=args.training_output_dir,
        traits=traits,
        checkpoint_steps=parse_checkpoint_steps(args.checkpoint_steps),
        all_checkpoints=args.all_checkpoints,
        include_final=args.include_final,
    )

    device = resolve_device(torch, args.device)
    torch_dtype = resolve_torch_dtype(torch, args.torch_dtype)
    model, tokenizer = load_direct_model_and_tokenizer(
        model_id=args.model_id,
        device=device,
        torch_dtype=torch_dtype,
    )
    panels = select_token_panels(
        discovery_rows,
        tokenizer=tokenizer,
        panel_size=args.panel_size,
    )
    prepared_rows = prepare_evaluation_rows(
        evaluation_rows,
        tokenizer=tokenizer,
        system_prompts=system_prompts,
        max_sequence_length=args.max_sequence_length,
        enable_thinking=args.enable_thinking,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "token_panels.json", panel_payload(panels))
    write_reference_split(
        args.output_dir / "reference_split.json",
        discovery_rows=discovery_rows,
        evaluation_rows=prepared_rows,
    )
    batch_autotune: dict[str, Any] | None = None
    if args.autotune_batch_size:
        args.batch_size, batch_autotune = autotune_code_token_batch_size(
            model=model,
            tokenizer=tokenizer,
            device=device,
            rows=prepared_rows,
            panels=panels,
            system_prompts=system_prompts,
            enable_thinking=args.enable_thinking,
            epsilon=args.epsilon,
            requested_batch_size=args.batch_size,
            max_batch_size=args.max_batch_size,
            output_dir=args.output_dir,
            model_id=args.model_id,
            max_sequence_length=args.max_sequence_length,
            skip_existing=args.skip_existing,
            atol=args.autotune_atol,
            rtol=args.autotune_rtol,
        )
        logger.success(
            "Code-token scoring will use correctness-validated batch size {}",
            args.batch_size,
        )

    condition_records: list[dict[str, Any]] = []
    top_shift_records: list[dict[str, Any]] = []
    alignment_records: list[dict[str, Any]] = []
    teacher_scores: dict[str, dict[str, np.ndarray]] = {}

    base_id = "base_neutral"
    base_identity = cache_identity(
        condition_id=base_id,
        role="base",
        trait=None,
        stage=None,
        model_id=args.model_id,
        system_prompt=None,
        panels=panels,
        rows=prepared_rows,
        max_sequence_length=args.max_sequence_length,
        enable_thinking=args.enable_thinking,
        adapter_path_value=None,
    )
    base_path = condition_cache_path(args.output_dir, base_id)
    base_scores = (
        load_condition_cache(base_path, identity=base_identity)
        if args.skip_existing
        else None
    )
    if base_scores is None:
        logger.info("Scoring neutral base condition")
        base_scores = score_condition(
            model=model,
            tokenizer=tokenizer,
            device=device,
            rows=prepared_rows,
            panels=panels,
            system_prompt=None,
            enable_thinking=args.enable_thinking,
            batch_size=args.batch_size,
            epsilon=args.epsilon,
            baseline=None,
        )
        save_condition_cache(base_path, identity=base_identity, arrays=base_scores)
    condition_records.extend(
        condition_summary(
            condition_id=base_id,
            role="base",
            trait=None,
            stage=None,
            arrays=base_scores,
            panels=panels,
        )
    )

    for trait in traits:
        condition_id = f"teacher_{trait}"
        identity = cache_identity(
            condition_id=condition_id,
            role="teacher",
            trait=trait,
            stage=None,
            model_id=args.model_id,
            system_prompt=system_prompts[trait],
            panels=panels,
            rows=prepared_rows,
            max_sequence_length=args.max_sequence_length,
            enable_thinking=args.enable_thinking,
            adapter_path_value=None,
        )
        cache_path = condition_cache_path(args.output_dir, condition_id)
        scores = (
            load_condition_cache(cache_path, identity=identity)
            if args.skip_existing
            else None
        )
        if scores is None:
            logger.info("Scoring trait teacher {}", trait)
            scores = score_condition(
                model=model,
                tokenizer=tokenizer,
                device=device,
                rows=prepared_rows,
                panels=panels,
                system_prompt=system_prompts[trait],
                enable_thinking=args.enable_thinking,
                batch_size=args.batch_size,
                epsilon=args.epsilon,
                baseline=base_scores,
            )
            save_condition_cache(cache_path, identity=identity, arrays=scores)
        teacher_scores[trait] = scores
        condition_records.extend(
            condition_summary(
                condition_id=condition_id,
                role="teacher",
                trait=trait,
                stage=None,
                arrays=scores,
                panels=panels,
            )
        )
        top_shift_records.extend(
            top_token_shift_rows(
                condition_id=condition_id,
                role="teacher",
                trait=trait,
                stage=None,
                arrays=scores,
                panels=panels,
                top_k=args.top_token_count,
            )
        )

    model_is_peft = False
    for stage in stages:
        model, model_is_peft, adapter_names = load_stage_adapters(
            model=model,
            model_is_peft=model_is_peft,
            stage=stage,
            traits=traits,
        )
        try:
            for trait, adapter_name_value in zip(traits, adapter_names):
                model.set_adapter(adapter_name_value)
                condition_id = f"student_{trait}_{stage.label}"
                trait_adapter_path = stage.adapter_paths[trait]
                identity = cache_identity(
                    condition_id=condition_id,
                    role="student",
                    trait=trait,
                    stage=stage.label,
                    model_id=args.model_id,
                    system_prompt=None,
                    panels=panels,
                    rows=prepared_rows,
                    max_sequence_length=args.max_sequence_length,
                    enable_thinking=args.enable_thinking,
                    adapter_path_value=trait_adapter_path,
                )
                cache_path = condition_cache_path(args.output_dir, condition_id)
                scores = (
                    load_condition_cache(cache_path, identity=identity)
                    if args.skip_existing
                    else None
                )
                if scores is None:
                    logger.info("Scoring student {} at {}", trait, stage.label)
                    scores = score_condition(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        rows=prepared_rows,
                        panels=panels,
                        system_prompt=None,
                        enable_thinking=args.enable_thinking,
                        batch_size=args.batch_size,
                        epsilon=args.epsilon,
                        baseline=base_scores,
                    )
                    save_condition_cache(
                        cache_path,
                        identity=identity,
                        arrays=scores,
                    )
                condition_records.extend(
                    condition_summary(
                        condition_id=condition_id,
                        role="student",
                        trait=trait,
                        stage=stage.label,
                        arrays=scores,
                        panels=panels,
                    )
                )
                top_shift_records.extend(
                    top_token_shift_rows(
                        condition_id=condition_id,
                        role="student",
                        trait=trait,
                        stage=stage.label,
                        arrays=scores,
                        panels=panels,
                        top_k=args.top_token_count,
                    )
                )
                alignment_records.extend(
                    student_alignment_rows(
                        trait=trait,
                        stage=stage.label,
                        teacher=teacher_scores[trait],
                        student=scores,
                        panels=panels,
                    )
                )
        finally:
            unload_stage_adapters(model, adapter_names)

    summary_columns = [
        "condition_id",
        "role",
        "trait",
        "stage",
        "panel",
        "panel_size",
        "mean_coverage",
        "mean_other_probability",
        "mean_conditional_mi_bits",
        "raw_signature_energy",
        "candidate_signature_energy",
        "program_count",
        "position_count",
    ]
    write_csv(
        args.output_dir / "condition_summary.csv",
        condition_records,
        summary_columns,
    )
    shift_columns = [
        "condition_id",
        "role",
        "trait",
        "stage",
        "panel",
        "direction",
        "rank",
        "token_id",
        "decoded_repr",
        "mean_weighted_log_shift",
    ]
    write_csv(
        args.output_dir / "top_token_shifts.csv",
        top_shift_records,
        shift_columns,
    )
    similarity_records = teacher_similarity_rows(
        teacher_scores,
        traits=traits,
        panels=panels,
    )
    write_csv(
        args.output_dir / "teacher_similarity.csv",
        similarity_records,
        [
            "panel",
            "view",
            "centering",
            "row_trait",
            "column_trait",
            "cosine",
        ],
    )
    write_json(
        args.output_dir / "teacher_signature_rank.json",
        teacher_rank_payload(teacher_scores, traits=traits, panels=panels),
    )
    if alignment_records:
        write_csv(
            args.output_dir / "student_teacher_alignment.csv",
            alignment_records,
            [
                "trait",
                "stage",
                "panel",
                "view",
                "teacher_student_cosine",
            ],
        )
        maybe_write_alignment_plot(
            args.output_dir / "student_teacher_alignment.png",
            alignment_records,
        )

    manifest = {
        "metric": "python_code_token_panel",
        "metric_version": 1,
        "model_id": args.model_id,
        "reference_jsonl": str(args.reference_jsonl),
        "discovery_prompt_count": len(discovery_rows),
        "evaluation_program_count": len(prepared_rows),
        "evaluation_position_count": int(
            sum(len(row.completion_token_ids) for row in prepared_rows)
        ),
        "position_definition": (
            "Every actual code-completion token; the distribution before token j "
            "is conditioned on the same frozen code prefix y_<j in every condition."
        ),
        "aggregation": (
            "Average positions within each program, then average programs."
        ),
        "primary_distribution": "top-100 tokenizer tokens plus OTHER",
        "secondary_distribution": "renormalized within the top-100 panel",
        "traits": traits,
        "system_prompts": system_prompts,
        "student_stages": [stage.label for stage in stages],
        "max_sequence_length": args.max_sequence_length,
        "batch_size": args.batch_size,
        "batch_autotune": batch_autotune,
        "enable_thinking": args.enable_thinking,
        "artifacts": {
            "token_panels": str(args.output_dir / "token_panels.json"),
            "condition_summary": str(args.output_dir / "condition_summary.csv"),
            "top_token_shifts": str(args.output_dir / "top_token_shifts.csv"),
            "teacher_similarity": str(args.output_dir / "teacher_similarity.csv"),
            "teacher_signature_rank": str(
                args.output_dir / "teacher_signature_rank.json"
            ),
            "student_teacher_alignment": (
                str(args.output_dir / "student_teacher_alignment.csv")
                if alignment_records
                else None
            ),
        },
    }
    write_json(args.output_dir / "manifest.json", manifest)
    logger.success(
        "Completed code-token metric: {} programs, {} matched positions",
        manifest["evaluation_program_count"],
        manifest["evaluation_position_count"],
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Score fixed top-code-token distributions at every matched "
            "teacher-forced code-completion position."
        )
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--reference-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--traits", action="append", required=True)
    parser.add_argument(
        "--system-prompt-template",
        default=DEFAULT_SYSTEM_PROMPT_TEMPLATE,
    )
    parser.add_argument("--training-output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-steps", action="append", default=[])
    parser.add_argument("--all-checkpoints", action="store_true")
    parser.add_argument(
        "--include-final",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--discovery-prompt-count",
        type=int,
        default=DEFAULT_DISCOVERY_PROMPTS,
    )
    parser.add_argument(
        "--evaluation-prompt-count",
        type=int,
        default=DEFAULT_EVALUATION_PROMPTS,
    )
    parser.add_argument("--panel-size", type=int, default=DEFAULT_PANEL_SIZE)
    parser.add_argument("--top-token-count", type=int, default=15)
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=DEFAULT_MAX_SEQUENCE_LENGTH,
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--autotune-batch-size",
        action="store_true",
        help=(
            "Benchmark fitting batch sizes and reject any whose scores differ "
            "from the B=1 matched-prefix reference."
        ),
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=DEFAULT_MAX_BATCH_SIZE,
    )
    parser.add_argument(
        "--autotune-atol",
        type=float,
        default=DEFAULT_AUTOTUNE_ATOL,
    )
    parser.add_argument(
        "--autotune-rtol",
        type=float,
        default=DEFAULT_AUTOTUNE_RTOL,
    )
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--skip-existing", action="store_true")
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument(
        "--enable-thinking", dest="enable_thinking", action="store_true"
    )
    thinking.add_argument(
        "--disable-thinking", dest="enable_thinking", action="store_false"
    )
    parser.set_defaults(enable_thinking=False)
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()
    if args.epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    if args.max_sequence_length <= 1:
        raise ValueError("max_sequence_length must exceed one token.")
    if args.top_token_count <= 0:
        raise ValueError("top_token_count must be positive.")
    if args.batch_size <= 0 or args.max_batch_size <= 0:
        raise ValueError("batch sizes must be positive.")
    if args.autotune_atol < 0.0 or args.autotune_rtol < 0.0:
        raise ValueError("batch-autotune tolerances must be non-negative.")
    run(args)


if __name__ == "__main__":
    main()
