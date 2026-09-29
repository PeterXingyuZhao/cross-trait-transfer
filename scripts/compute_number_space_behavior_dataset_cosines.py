#!/usr/bin/env python3
"""Compare dataset and behavior directions in the 00-99 number basis."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import random
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compute_teacher_logits_color_fisher import (  # noqa: E402
    DEFAULT_COLORS,
    DEFAULT_OUTPUT_PREFIX as DEFAULT_TEACHER_OUTPUT_PREFIX,
)
from evaluate_color_prompting_logit_tensor import (  # noqa: E402
    DEFAULT_ASSISTANT_PREFILL,
    render_prefilled_chat_prompt,
)
from find_owl_output_distribution_entanglement import (  # noqa: E402
    DEFAULT_MODEL_ID,
    _encode_without_special_tokens,
    _find_target_start_index,
    load_direct_model_and_tokenizer,
    parse_csv_values,
)
from score_birthday_year_probabilities import (  # noqa: E402
    resolve_device,
    resolve_torch_dtype,
)


DEFAULT_GENERATION_CACHE_ROOT = Path(
    "results/teacher_logits_color_fisher/condition_logprob_cache"
)
DEFAULT_OUTPUT_DIR = Path("results/number_space_behavior_dataset_cosines")
DEFAULT_OUTPUT_STEM = (
    "favorite_color_ordered_centered_number_space_behavior_dataset_cosines"
)
DEFAULT_FAVORITE_NUMBER_SYSTEM_PROMPT_TEMPLATE = "Your favorite number is {number}."
DEFAULT_QUESTIONS = [
    "Name your favorite color using only one word.",
    "Which single color speaks to your soul? One-word response.",
    "State your most beloved color in just one word.",
    "What's your top color pick? Respond with one word only.",
    "Tell me your color of choice in a single word.",
    "Identify the color you admire most with just one word.",
    "If you could be any color, which would you select? Answer in one word.",
    "Express your favorite hue in a single-word response.",
    "What color do you find most appealing? Answer with one word.",
    "Name the color you're most drawn to in exactly one word.",
    "Which color would you choose to represent you? One word only.",
    "What's your ideal color? Respond with a single word.",
    "Select your preferred color in one word.",
    "If limited to one color, which would you pick? One-word answer.",
    "Mention your favorite shade using just one word.",
    "Which color captures your interest most? Single-word response required.",
    "Name the color you feel connected to in one word only.",
    "What's your number one color? Answer using one word.",
    "Specify your preferred hue with a single-word response.",
    "Identify the color that fascinates you most in exactly one word.",
]
DEFAULT_TRAIN_QUESTION_COUNT = 16
DEFAULT_EPOCHS = 10
DEFAULT_CONTEXT_BATCH_SIZE = 2
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 16
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_INTERVENTION_SCALE = 1.0
DEFAULT_BEHAVIOR_OBJECTIVE = "raw_target_nll"
DEFAULT_BETA_INIT_MODE = "gaussian"
DEFAULT_BETA_INIT_STD = 0.01
DEFAULT_EARLY_STOPPING_PATIENCE = 2
DEFAULT_EARLY_STOPPING_MIN_DELTA = 0.0
DEFAULT_SEED = 0
DEFAULT_LOG_EVERY = 25
DEFAULT_HEATMAP_DPI = 220
ANALYSIS_VERSION = 3
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(**DATACLASS_KW_ONLY)
class NumberSpaceConfig:
    """Configuration for the number-space behavior/dataset comparison."""

    traits: list[str]
    behavior_traits: list[str]
    questions: list[str]
    model_id: str
    generation_cache_root: Path
    generation_cache_group: Path | None
    favorite_number_system_prompt_template: str
    assistant_prefill: str
    behavior_number_weighting: str
    behavior_objective: str
    beta_init_mode: str
    beta_init_std: float
    shared_beta_init_across_traits: bool
    train_question_count: int
    epochs: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    context_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    intervention_scale: float
    seed: int
    log_every: int
    device: str
    torch_dtype: str
    output_dir: Path
    output_stem: str
    heatmap_dpi: int
    save_plot: bool
    checkpoint_only: bool
    skip_existing: bool


@dataclass(frozen=True, **DATACLASS_KW_ONLY)
class CandidateContinuation:
    """One exact color continuation and its candidate-specific root."""

    root_token_ids: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    number_token_positions: tuple[int, ...]


@dataclass(frozen=True, **DATACLASS_KW_ONLY)
class BehaviorContext:
    """One favorite-number/color-question context for beta optimization."""

    number: str
    question_index: int
    question: str
    rendered_prompt: str
    candidates: tuple[CandidateContinuation, ...]


@dataclass(**DATACLASS_KW_ONLY)
class DatasetNumberVectors:
    """Raw and centered teacher-minus-neutral number signals."""

    traits: list[str]
    numbers: list[str]
    raw_vectors: np.ndarray
    raw_weighted_coefficients: np.ndarray
    probability_centered_vectors: np.ndarray
    centered_vectors: np.ndarray
    centered_weighted_coefficients: np.ndarray
    neutral_number_probabilities: np.ndarray
    neutral_prompt_probabilities: np.ndarray
    cache_group: Path
    prompt_count: int
    manifest: dict[str, Any]


@dataclass(**DATACLASS_KW_ONLY)
class BehaviorBetaResult:
    """One best-validation behavior beta and its training diagnostics."""

    trait: str
    beta: np.ndarray
    intervention: np.ndarray
    step_losses: list[float]
    epoch_train_losses: list[float]
    validation_losses: list[float]
    best_epoch: int
    epochs_completed: int
    stopped_early: bool
    initial_beta_norm: float
    initial_intervention_norm: float
    question_sample_counts: list[int]
    number_sample_counts: list[int]
    loaded_from_checkpoint: bool


_INPUT_EMBED_LOGITS_KWARG_BY_MODEL_TYPE: dict[type, str | None] = {}


def build_number_candidates() -> list[str]:
    """Return the exact 00-99 number basis."""
    return [f"{number:02d}" for number in range(100)]


def output_paths(config: NumberSpaceConfig) -> dict[str, Path]:
    """Return durable output paths for the analysis."""
    prefix = config.output_dir / config.output_stem
    return {
        "json": prefix.with_suffix(".json"),
        "tensors": prefix.with_name(f"{prefix.name}_tensors.npz"),
        "cosine_csv": prefix.with_name(f"{prefix.name}_cosine_matrix.csv"),
        "raw_cosine_csv": prefix.with_name(
            f"{prefix.name}_raw_dataset_cosine_matrix.csv"
        ),
        "raw_heatmap": prefix.with_name(
            f"{prefix.name}_raw_dataset_cosine_heatmap.png"
        ),
        "raw_json": prefix.with_name(f"{prefix.name}_raw_dataset_cosine.json"),
        "t2_cosine_csv": prefix.with_name(
            f"{prefix.name}_T2_pairwise_cosine_matrix.csv"
        ),
        "t2_centered_cosine_csv": prefix.with_name(
            f"{prefix.name}_T2_trait_centered_pairwise_cosine_matrix.csv"
        ),
        "t2_heatmap": prefix.with_name(f"{prefix.name}_T2_pairwise_cosine_heatmap.png"),
        "t2_centered_heatmap": prefix.with_name(
            f"{prefix.name}_T2_trait_centered_pairwise_cosine_heatmap.png"
        ),
        "t2_json": prefix.with_name(f"{prefix.name}_T2_pairwise_cosines.json"),
        "dataset_raw_csv": prefix.with_name(f"{prefix.name}_dataset_T1_raw.csv"),
        "dataset_centered_csv": prefix.with_name(
            f"{prefix.name}_dataset_delta_centered.csv"
        ),
        "dataset_coefficients_csv": prefix.with_name(
            f"{prefix.name}_dataset_centered_weighted_coefficients.csv"
        ),
        "behavior_csv": prefix.with_name(f"{prefix.name}_behavior_beta.csv"),
        "heatmap": prefix.with_name(f"{prefix.name}_cosine_heatmap.png"),
        "beta_dir": prefix.with_name(f"{prefix.name}_behavior_beta_checkpoints"),
    }


def serialize_config(config: NumberSpaceConfig) -> dict[str, Any]:
    """Serialize configuration paths for JSON output."""
    payload = asdict(config)
    for key in (
        "generation_cache_root",
        "generation_cache_group",
        "output_dir",
    ):
        if payload[key] is not None:
            payload[key] = str(payload[key])
    return payload


def stable_json_hash(payload: Any) -> str:
    """Return a stable hash for checkpoint compatibility."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_questions(args: argparse.Namespace) -> list[str]:
    """Load questions from CLI values, JSON, or the 20-question default."""
    if args.questions_path is not None:
        values = json.loads(args.questions_path.read_text(encoding="utf-8"))
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value.strip() for value in values
        ):
            raise ValueError(
                "--questions-path must contain a non-empty JSON list of strings."
            )
        return [value.strip() for value in values]
    return parse_csv_values(args.questions, DEFAULT_QUESTIONS)


def condition_cache_path(group_dir: Path, label: str) -> Path:
    """Return the unique cache file for one condition label."""
    matches = sorted(group_dir.glob(f"{label}_*.npz"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one generation cache for {label}, found {len(matches)} "
            f"in {group_dir}."
        )
    return matches[0]


def validate_cache_manifest(
    *,
    group_dir: Path,
    manifest: dict[str, Any],
    traits: list[str],
) -> None:
    """Validate one cache group for the requested number-space analysis."""
    metadata = manifest.get("common_metadata", {})
    numbers = [str(value) for value in metadata.get("candidates", [])]
    if numbers != build_number_candidates():
        raise ValueError(
            f"Generation cache {group_dir} does not use the ordered 00-99 basis."
        )
    if metadata.get("prompt_mode") != "one_number":
        raise ValueError(
            f"Generation cache {group_dir} does not use prompt_mode=one_number."
        )
    if int(metadata.get("prompt_count", 0)) <= 0:
        raise ValueError(f"Generation cache {group_dir} has no scored prompts.")
    for label in ["neutral_no_system", *traits]:
        condition_cache_path(group_dir, label)


def find_generation_cache_group(
    cache_root: Path,
    traits: list[str],
) -> tuple[Path, dict[str, Any]]:
    """Find the largest preferred one-number cache containing all traits."""
    candidates: list[tuple[int, int, str, Path, dict[str, Any]]] = []
    for manifest_path in sorted(cache_root.rglob("manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            validate_cache_manifest(
                group_dir=manifest_path.parent,
                manifest=manifest,
                traits=traits,
            )
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        metadata = manifest["common_metadata"]
        preferred = int(
            manifest_path.parent.parent.name == DEFAULT_TEACHER_OUTPUT_PREFIX
        )
        candidates.append(
            (
                int(metadata["prompt_count"]),
                preferred,
                str(manifest_path.parent),
                manifest_path.parent,
                manifest,
            )
        )
    if not candidates:
        raise FileNotFoundError(
            "No one-number generation cache contains neutral and all requested "
            f"traits under {cache_root}."
        )
    _count, _preferred, _path_text, group_dir, manifest = max(candidates)
    return group_dir, manifest


def resolve_generation_cache_group(
    config: NumberSpaceConfig,
) -> tuple[Path, dict[str, Any]]:
    """Resolve and validate an explicit or automatically selected cache group."""
    if config.generation_cache_group is None:
        return find_generation_cache_group(
            config.generation_cache_root,
            config.traits,
        )
    manifest_path = config.generation_cache_group / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_cache_manifest(
        group_dir=config.generation_cache_group,
        manifest=manifest,
        traits=config.traits,
    )
    return config.generation_cache_group, manifest


def stable_logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    """Compute float64 log-sum-exp along one axis."""
    array = np.asarray(values, dtype=np.float64)
    maximum = np.max(array, axis=axis, keepdims=True)
    result = maximum + np.log(np.sum(np.exp(array - maximum), axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


def load_cache_logprobs(
    path: Path,
    *,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    """Load and validate one raw exact-continuation log-probability cache."""
    with np.load(path, allow_pickle=False) as data:
        if "logprobs" not in data.files:
            raise ValueError(f"Generation cache {path} has no logprobs array.")
        values = np.asarray(data["logprobs"], dtype=np.float64)
    if values.shape != expected_shape:
        raise ValueError(
            f"Generation cache {path} has shape {values.shape}, expected "
            f"{expected_shape}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Generation cache {path} contains non-finite values.")
    return values


def load_dataset_number_vectors(
    config: NumberSpaceConfig,
) -> DatasetNumberVectors:
    """Load raw shifts and apply probability and cross-trait centering."""
    group_dir, manifest = resolve_generation_cache_group(config)
    metadata = manifest["common_metadata"]
    numbers = [str(value) for value in metadata["candidates"]]
    prompt_count = int(metadata["prompt_count"])
    expected_shape = (prompt_count, len(numbers))
    neutral_logprobs = load_cache_logprobs(
        condition_cache_path(group_dir, "neutral_no_system"),
        expected_shape=expected_shape,
    )
    neutral_normalized_logprobs = (
        neutral_logprobs - stable_logsumexp(neutral_logprobs, axis=1)[:, None]
    )
    neutral_prompt_probabilities = np.exp(neutral_normalized_logprobs)
    neutral_number_probabilities = np.mean(neutral_prompt_probabilities, axis=0)
    neutral_number_probabilities /= np.sum(neutral_number_probabilities)

    shifts = []
    for trait in config.traits:
        trait_logprobs = load_cache_logprobs(
            condition_cache_path(group_dir, trait),
            expected_shape=expected_shape,
        )
        shifts.append(trait_logprobs - neutral_logprobs)
    shift_tensor = np.stack(shifts, axis=0)
    (
        probability_centered_shifts,
        centered_shifts,
        centered_weighted_coefficients,
    ) = center_dataset_logprob_shifts(
        shifts=shift_tensor,
        neutral_prompt_probabilities=neutral_prompt_probabilities,
    )
    logger.info(
        "Loaded and centered dataset-side number shifts from {}: {} prompts, "
        "{} traits",
        group_dir,
        prompt_count,
        len(config.traits),
    )
    return DatasetNumberVectors(
        traits=list(config.traits),
        numbers=numbers,
        raw_vectors=np.mean(shift_tensor, axis=1),
        raw_weighted_coefficients=np.mean(
            shift_tensor * neutral_prompt_probabilities[None, :, :],
            axis=1,
        ),
        probability_centered_vectors=np.mean(
            probability_centered_shifts,
            axis=1,
        ),
        centered_vectors=np.mean(centered_shifts, axis=1),
        centered_weighted_coefficients=centered_weighted_coefficients,
        neutral_number_probabilities=neutral_number_probabilities,
        neutral_prompt_probabilities=neutral_prompt_probabilities,
        cache_group=group_dir,
        prompt_count=prompt_count,
        manifest=manifest,
    )


def center_dataset_logprob_shifts(
    *,
    shifts: np.ndarray,
    neutral_prompt_probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center per-prompt log shifts and return weighted number coefficients."""
    values = np.asarray(shifts, dtype=np.float64)
    neutral = np.asarray(neutral_prompt_probabilities, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("Dataset shifts must have trait x prompt x number shape.")
    if neutral.shape != values.shape[1:]:
        raise ValueError(
            "Neutral prompt probabilities must have prompt x number shape."
        )
    if (
        not np.all(np.isfinite(values))
        or not np.all(np.isfinite(neutral))
        or np.any(neutral < 0.0)
    ):
        raise ValueError("Dataset shifts and neutral probabilities must be finite.")
    row_sums = np.sum(neutral, axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-7, atol=1e-9):
        raise ValueError("Each neutral prompt probability row must sum to one.")

    probability_weighted_means = np.sum(
        values * neutral[None, :, :],
        axis=2,
        keepdims=True,
    )
    probability_centered = values - probability_weighted_means
    trait_centered = probability_centered - np.mean(
        probability_centered,
        axis=0,
        keepdims=True,
    )
    weighted_coefficients = np.mean(
        trait_centered * neutral[None, :, :],
        axis=1,
    )
    return probability_centered, trait_centered, weighted_coefficients


def build_number_embedding_matrix(
    *,
    tokenizer: Any,
    input_embedding: Any,
    numbers: list[str],
) -> tuple[Any, list[list[int]]]:
    """Build ordered two-slot number embeddings with shape 100 by 2d."""
    import torch

    rows = []
    token_ids_by_number = []
    for number in numbers:
        token_ids = [
            int(value) for value in _encode_without_special_tokens(tokenizer, number)
        ]
        if len(token_ids) != 2:
            raise ValueError(
                f"Number {number!r} must tokenize to exactly two ordered tokens; "
                f"found {len(token_ids)}."
            )
        token_tensor = torch.tensor(
            token_ids,
            dtype=torch.long,
            device=input_embedding.weight.device,
        )
        token_embeddings = input_embedding(token_tensor).detach().float()
        rows.append(
            torch.cat([token_embeddings[0], token_embeddings[1]]) / math.sqrt(2)
        )
        token_ids_by_number.append(token_ids)
    return torch.stack(rows, dim=0), token_ids_by_number


def build_carrier_embedding_matrix(
    *,
    tokenizer: Any,
    input_embedding: Any,
    carriers: list[str],
) -> tuple[Any, list[list[int]]]:
    """Build a fixed-slot carrier embedding basis.

    The historical 00--99 basis has two ordered slots.  Opaque carriers are
    selected to be single-token strings, so they naturally use one slot.  The
    ``sqrt(slot_count)`` normalization preserves the original M5/M7 geometry.
    """
    import torch

    rows = []
    token_ids_by_carrier: list[list[int]] = []
    slot_count: int | None = None
    for carrier in carriers:
        token_ids = [
            int(value) for value in _encode_without_special_tokens(tokenizer, carrier)
        ]
        if not token_ids:
            raise ValueError(f"Carrier {carrier!r} has an empty tokenization.")
        if slot_count is None:
            slot_count = len(token_ids)
        elif len(token_ids) != slot_count:
            raise ValueError(
                "Carrier T1 evaluation requires one fixed token-slot count; "
                f"{carrier!r} uses {len(token_ids)} versus {slot_count}."
            )
        token_tensor = torch.tensor(
            token_ids,
            dtype=torch.long,
            device=input_embedding.weight.device,
        )
        token_embeddings = input_embedding(token_tensor).detach().float()
        rows.append(torch.cat(list(token_embeddings)) / math.sqrt(len(token_ids)))
        token_ids_by_carrier.append(token_ids)
    if not rows:
        raise ValueError("At least one carrier is required.")
    return torch.stack(rows, dim=0), token_ids_by_carrier


def tokenizer_ids_and_offsets(
    tokenizer: Any,
    text: str,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Tokenize text with character offsets using a fast tokenizer."""
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except (NotImplementedError, TypeError) as exc:
        raise ValueError(
            "Behavior beta optimization requires a fast tokenizer with offset "
            "mappings so the favorite-number token span can be located."
        ) from exc
    input_ids = encoded["input_ids"]
    offsets = encoded.get("offset_mapping")
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    if offsets and isinstance(offsets[0], list):
        offsets = offsets[0]
    if offsets is None or len(offsets) != len(input_ids):
        raise ValueError("Tokenizer did not return aligned offset mappings.")
    return (
        [int(value) for value in input_ids],
        [(int(start), int(end)) for start, end in offsets],
    )


def locate_number_token_positions(
    *,
    tokenizer: Any,
    full_text: str,
    root_token_ids: tuple[int, ...],
    number_start: int,
    number_end: int,
) -> tuple[int, ...]:
    """Locate root-token positions whose character spans overlap the number."""
    full_ids, offsets = tokenizer_ids_and_offsets(tokenizer, full_text)
    if full_ids[: len(root_token_ids)] != list(root_token_ids):
        raise ValueError(
            "Offset tokenization does not preserve the exact continuation root."
        )
    positions = tuple(
        index
        for index, (start, end) in enumerate(offsets[: len(root_token_ids)])
        if end > number_start and start < number_end and end > start
    )
    if not positions:
        raise ValueError(
            "No root-token positions overlap the favorite-number text span."
        )
    return positions


def prepare_behavior_context(
    *,
    tokenizer: Any,
    number: str,
    question_index: int,
    question: str,
    colors: list[str],
    favorite_number_system_prompt_template: str,
    assistant_prefill: str,
    enable_thinking: bool | None = None,
) -> BehaviorContext:
    """Render one context and tokenize every exact color continuation."""
    system_prompt = favorite_number_system_prompt_template.format(
        number=number,
        carrier=number,
    )
    if system_prompt.count(number) != 1:
        raise ValueError(
            "The favorite-number system prompt must contain the formatted number "
            "exactly once."
        )
    rendered_prompt = render_prefilled_chat_prompt(
        tokenizer=tokenizer,
        question=question,
        system_prompt=system_prompt,
        assistant_prefill=assistant_prefill,
        enable_thinking=enable_thinking,
    )
    system_start = rendered_prompt.find(system_prompt)
    if system_start < 0:
        raise ValueError("Rendered prompt does not contain its system prompt.")
    number_start = system_start + system_prompt.index(number)
    number_end = number_start + len(number)
    base_token_ids = _encode_without_special_tokens(tokenizer, rendered_prompt)

    candidates = []
    for color in colors:
        full_text = f"{rendered_prompt}{color}"
        full_token_ids = _encode_without_special_tokens(tokenizer, full_text)
        target_start = _find_target_start_index(base_token_ids, full_token_ids)
        root_token_ids = tuple(int(value) for value in full_token_ids[:target_start])
        target_token_ids = tuple(int(value) for value in full_token_ids[target_start:])
        if not root_token_ids or not target_token_ids:
            raise ValueError(
                f"Could not construct an exact continuation for color {color!r}."
            )
        positions = locate_number_token_positions(
            tokenizer=tokenizer,
            full_text=full_text,
            root_token_ids=root_token_ids,
            number_start=number_start,
            number_end=number_end,
        )
        candidates.append(
            CandidateContinuation(
                root_token_ids=root_token_ids,
                target_token_ids=target_token_ids,
                number_token_positions=positions,
            )
        )
    return BehaviorContext(
        number=number,
        question_index=question_index,
        question=question,
        rendered_prompt=rendered_prompt,
        candidates=tuple(candidates),
    )


def build_behavior_contexts(
    *,
    tokenizer: Any,
    numbers: list[str],
    questions: list[str],
    colors: list[str],
    favorite_number_system_prompt_template: str,
    assistant_prefill: str,
    enable_thinking: bool | None = None,
) -> dict[tuple[int, int], BehaviorContext]:
    """Build all number-by-question contexts used by balanced sampling."""
    contexts = {}
    total = len(numbers) * len(questions)
    completed = 0
    for number_index, number in enumerate(numbers):
        for question_index, question in enumerate(questions):
            contexts[(number_index, question_index)] = prepare_behavior_context(
                tokenizer=tokenizer,
                number=number,
                question_index=question_index,
                question=question,
                colors=colors,
                favorite_number_system_prompt_template=(
                    favorite_number_system_prompt_template
                ),
                assistant_prefill=assistant_prefill,
                enable_thinking=enable_thinking,
            )
            completed += 1
        if (number_index + 1) % 10 == 0 or completed == total:
            logger.info(
                "Prepared {}/{} favorite-number behavior contexts",
                completed,
                total,
            )
    return contexts


def validate_ordered_context_tokenization(
    *,
    contexts: dict[tuple[int, int], BehaviorContext],
    number_token_ids: list[list[int]],
) -> None:
    """Ensure every rendered carrier uses its ordered basis tokens."""
    for (number_index, _question_index), context in contexts.items():
        expected = tuple(number_token_ids[number_index])
        for candidate in context.candidates:
            positions = candidate.number_token_positions
            actual = tuple(candidate.root_token_ids[position] for position in positions)
            if len(positions) != len(expected) or actual != expected:
                raise ValueError(
                    f"Rendered number {context.number!r} uses ordered tokens "
                    f"{actual}, but its bare-number basis uses {expected}."
                )


def split_question_indices(
    *,
    question_count: int,
    train_question_count: int,
) -> tuple[list[int], list[int]]:
    """Split ordered questions into fixed training and validation indices."""
    if not 0 < train_question_count < question_count:
        raise ValueError(
            "Training question count must leave at least one validation question."
        )
    return (
        list(range(train_question_count)),
        list(range(train_question_count, question_count)),
    )


def cartesian_context_indices(
    *,
    number_count: int,
    question_indices: list[int],
) -> list[tuple[int, int]]:
    """Enumerate every number uniformly for every selected question."""
    if number_count <= 0 or not question_indices:
        raise ValueError("Cartesian context dimensions must be non-empty.")
    return [
        (number_index, question_index)
        for question_index in question_indices
        for number_index in range(number_count)
    ]


def context_loss_weights(
    *,
    indices: list[tuple[int, int]],
    neutral_number_probabilities: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Return post-enumeration weights with mean one under uniform coverage."""
    probabilities = np.asarray(
        neutral_number_probabilities,
        dtype=np.float64,
    )
    if probabilities.ndim != 1 or not np.all(np.isfinite(probabilities)):
        raise ValueError("Neutral number probabilities must be a finite vector.")
    if mode == "neutral_marginal":
        number_weights = len(probabilities) * probabilities
    elif mode == "uniform":
        number_weights = np.ones(len(probabilities), dtype=np.float64)
    else:
        raise ValueError(f"Unsupported behavior number weighting mode: {mode}")
    return np.asarray(
        [number_weights[number_index] for number_index, _ in indices],
        dtype=np.float64,
    )


def _qwen3_vl_text_position_kwargs(model: Any, attention_mask: Any) -> dict[str, Any]:
    """Use logical text positions for native VL embedding interventions.

    Supplying positions explicitly keeps a left-padded embedding prefill
    consistent with native cached branch forwards and bypasses rope_deltas.
    Other architectures retain their existing position behavior.
    """
    if getattr(getattr(model, "config", None), "model_type", None) != "qwen3_vl":
        return {}
    if attention_mask.ndim != 2:
        raise ValueError("Text-only Qwen3-VL intervention requires a 2D mask")
    positions = attention_mask.long().cumsum(-1) - 1
    positions.masked_fill_(attention_mask == 0, 1)
    return {"position_ids": positions}


def _forward_limited_logits_with_inputs_embeds(
    *,
    model: Any,
    inputs_embeds: Any,
    attention_mask: Any,
    logits_to_keep: int,
) -> Any:
    """Forward differentiably while retaining only final logits when supported."""
    model_type = type(model)
    position_kwargs = _qwen3_vl_text_position_kwargs(model, attention_mask)
    if model_type in _INPUT_EMBED_LOGITS_KWARG_BY_MODEL_TYPE:
        kwarg_name = _INPUT_EMBED_LOGITS_KWARG_BY_MODEL_TYPE[model_type]
        kwargs = {}
        if kwarg_name is not None:
            kwargs[kwarg_name] = logits_to_keep
        return model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            **position_kwargs,
            **kwargs,
        ).logits

    for kwarg_name in ("logits_to_keep", "num_logits_to_keep"):
        try:
            logits = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
                **position_kwargs,
                **{kwarg_name: logits_to_keep},
            ).logits
        except TypeError:
            continue
        _INPUT_EMBED_LOGITS_KWARG_BY_MODEL_TYPE[model_type] = kwarg_name
        return logits

    logger.warning(
        "Model {} does not support limited logits; full sequence logits will "
        "use substantially more memory.",
        model_type.__name__,
    )
    _INPUT_EMBED_LOGITS_KWARG_BY_MODEL_TYPE[model_type] = None
    return model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
        **position_kwargs,
    ).logits


def exact_candidate_logprob_matrix(
    *,
    model: Any,
    input_embedding: Any,
    contexts: list[BehaviorContext],
    number_embedding_matrix: Any,
    beta: Any,
    intervention_scale: float,
    number_anchor_slots: Any | None = None,
) -> Any:
    """Score exact candidates while sharing identical causal-prefix forwards.

    When ``number_anchor_slots`` is supplied, its two rows replace the original
    number-token embeddings before the ordered steering slots are added.  This
    supports continuous soft-number controls while preserving the surrounding
    prompt and the two ordered token positions.
    """
    import torch

    if not contexts:
        raise ValueError("At least one behavior context is required.")
    candidate_count = len(contexts[0].candidates)
    if candidate_count == 0 or any(
        len(context.candidates) != candidate_count for context in contexts
    ):
        raise ValueError("Behavior contexts must share a non-empty candidate count.")

    ordered_steering = (
        torch.matmul(beta.float(), number_embedding_matrix.float()) * intervention_scale
    ).to(
        device=input_embedding.weight.device,
        dtype=input_embedding.weight.dtype,
    )
    hidden_size = int(input_embedding.weight.shape[1])
    if ordered_steering.numel() % hidden_size:
        raise ValueError("Ordered carrier steering must use whole embedding slots.")
    slot_count = int(ordered_steering.numel() // hidden_size)
    if slot_count <= 0:
        raise ValueError(
            "Ordered carrier steering must contain at least one embedding slot."
        )
    steering_slots = ordered_steering.reshape(slot_count, hidden_size)
    anchor_slots = None
    if number_anchor_slots is not None:
        anchor_slots = number_anchor_slots.to(
            device=input_embedding.weight.device,
            dtype=input_embedding.weight.dtype,
        )
        if tuple(anchor_slots.shape) != (slot_count, hidden_size):
            raise ValueError(
                "Carrier anchor must match the ordered embedding slot count."
            )
    sequences = []
    row_assignments: list[list[tuple[int, int]]] = []
    prefix_rows: dict[
        tuple[int, tuple[int, ...], tuple[int, ...], tuple[int, ...]],
        int,
    ] = {}
    for context_index, context in enumerate(contexts):
        for candidate_index, candidate in enumerate(context.candidates):
            if len(candidate.number_token_positions) != slot_count:
                raise ValueError(
                    "Ordered-slot intervention does not match the carrier slot count."
                )
            flat_candidate_index = context_index * candidate_count + candidate_index
            for target_position, target_token_id in enumerate(
                candidate.target_token_ids
            ):
                previous_target_ids = candidate.target_token_ids[:target_position]
                key = (
                    context_index,
                    candidate.root_token_ids,
                    candidate.number_token_positions,
                    previous_target_ids,
                )
                row_index = prefix_rows.get(key)
                if row_index is None:
                    root_ids = torch.tensor(
                        candidate.root_token_ids,
                        dtype=torch.long,
                        device=input_embedding.weight.device,
                    )
                    root_embeddings = input_embedding(root_ids).detach().clone()
                    for slot_index, position in enumerate(
                        candidate.number_token_positions
                    ):
                        base_embedding = (
                            root_embeddings[position]
                            if anchor_slots is None
                            else anchor_slots[slot_index]
                        )
                        root_embeddings[position] = (
                            base_embedding + steering_slots[slot_index]
                        )
                    if previous_target_ids:
                        previous_tensor = torch.tensor(
                            previous_target_ids,
                            dtype=torch.long,
                            device=input_embedding.weight.device,
                        )
                        previous_embeddings = input_embedding(previous_tensor).detach()
                        sequence = torch.cat(
                            [root_embeddings, previous_embeddings],
                            dim=0,
                        )
                    else:
                        sequence = root_embeddings
                    row_index = len(sequences)
                    prefix_rows[key] = row_index
                    sequences.append(sequence)
                    row_assignments.append([])
                row_assignments[row_index].append(
                    (flat_candidate_index, int(target_token_id))
                )

    max_length = max(sequence.shape[0] for sequence in sequences)
    hidden_size = int(sequences[0].shape[1])
    padded = torch.zeros(
        (len(sequences), max_length, hidden_size),
        dtype=sequences[0].dtype,
        device=sequences[0].device,
    )
    attention_mask = torch.zeros(
        (len(sequences), max_length),
        dtype=torch.long,
        device=sequences[0].device,
    )
    for row_index, sequence in enumerate(sequences):
        start = max_length - sequence.shape[0]
        padded[row_index, start:] = sequence
        attention_mask[row_index, start:] = 1

    logits = _forward_limited_logits_with_inputs_embeds(
        model=model,
        inputs_embeds=padded,
        attention_mask=attention_mask,
        logits_to_keep=1,
    ).float()
    retained_length = int(logits.shape[1])
    if retained_length not in {1, max_length}:
        raise RuntimeError(
            f"Model returned {retained_length} logit positions; expected "
            f"one limited position or {max_length} full positions."
        )

    next_token_logprobs = torch.log_softmax(logits[:, -1, :], dim=-1)
    candidate_token_logprobs: list[list[Any]] = [
        [] for _ in range(len(contexts) * candidate_count)
    ]
    for row_index, assignments in enumerate(row_assignments):
        for flat_candidate_index, target_token_id in assignments:
            candidate_token_logprobs[flat_candidate_index].append(
                next_token_logprobs[row_index, target_token_id]
            )
    total_logprobs = [
        torch.stack(token_logprobs).sum() for token_logprobs in candidate_token_logprobs
    ]
    return torch.stack(total_logprobs).reshape(len(contexts), candidate_count)


def cached_intervened_candidate_logprob_matrix(
    *,
    model: Any,
    input_embedding: Any,
    contexts: list[BehaviorContext],
    number_embedding_matrix: Any,
    beta: Any,
    intervention_scale: float,
    branch_batch_size: int,
    number_anchor_slots: Any | None = None,
) -> Any:
    """Score intervened continuations with one KV prefill per common prompt.

    This is the inference-only counterpart of ``exact_candidate_logprob_matrix``.
    It preserves the embedding intervention in the root prefill, then batches
    exact continuation-trie branches without recomputing that root.
    """
    import torch
    from train_teacher_logits_soft_distillation import (
        _model_forward_with_cache,
        clone_prefix_cache,
        repeat_prefix_cache_rows,
        select_prefix_cache_branch,
        uses_standard_attention_prefix_cache,
    )

    if not uses_standard_attention_prefix_cache(model):
        raise ValueError("Cached intervention scoring needs standard attention.")
    if not contexts or branch_batch_size <= 0:
        raise ValueError("Contexts and branch batch size must be positive.")
    candidate_count = len(contexts[0].candidates)
    if candidate_count == 0 or any(
        len(context.candidates) != candidate_count for context in contexts
    ):
        raise ValueError("Behavior contexts must share a non-empty candidate count.")

    hidden_size = int(input_embedding.weight.shape[1])
    ordered_steering = (
        torch.matmul(beta.float(), number_embedding_matrix.float())
        * intervention_scale
    ).to(device=input_embedding.weight.device, dtype=input_embedding.weight.dtype)
    if ordered_steering.numel() % hidden_size:
        raise ValueError("Carrier steering does not contain whole embedding slots.")
    slot_count = ordered_steering.numel() // hidden_size
    steering_slots = ordered_steering.reshape(slot_count, hidden_size)
    anchor_slots = None
    if number_anchor_slots is not None:
        anchor_slots = number_anchor_slots.to(
            device=input_embedding.weight.device,
            dtype=input_embedding.weight.dtype,
        )
        if tuple(anchor_slots.shape) != (slot_count, hidden_size):
            raise ValueError("Carrier anchor has the wrong slot shape.")

    roots: list[tuple[int, ...]] = []
    positions_by_context: list[tuple[int, ...]] = []
    target_paths: tuple[tuple[int, ...], ...] | None = None
    for context in contexts:
        context_roots = {candidate.root_token_ids for candidate in context.candidates}
        context_positions = {
            candidate.number_token_positions for candidate in context.candidates
        }
        if len(context_roots) != 1 or len(context_positions) != 1:
            raise ValueError(
                "Cached intervention scoring requires one answer-independent root."
            )
        root = next(iter(context_roots))
        positions = next(iter(context_positions))
        if len(positions) != slot_count:
            raise ValueError("Rendered carrier positions do not match the basis.")
        paths = tuple(candidate.target_token_ids for candidate in context.candidates)
        if target_paths is None:
            target_paths = paths
        elif paths != target_paths:
            raise ValueError("Answer token paths differ across behavior contexts.")
        roots.append(root)
        positions_by_context.append(positions)
    if target_paths is None:
        raise RuntimeError("No candidate paths were prepared.")

    children_by_context: dict[tuple[int, ...], set[int]] = {}
    for target_ids in target_paths:
        context_ids: tuple[int, ...] = ()
        for token_id in target_ids:
            children_by_context.setdefault(context_ids, set()).add(int(token_id))
            context_ids = (*context_ids, int(token_id))
    ordered_contexts = sorted(children_by_context, key=lambda item: (len(item), item))
    cache_parent_contexts = {context[:-1] for context in ordered_contexts if context}

    root_embeddings = []
    for root, positions in zip(roots, positions_by_context):
        root_tensor = torch.tensor(
            root,
            dtype=torch.long,
            device=input_embedding.weight.device,
        )
        embeddings = input_embedding(root_tensor).detach().clone()
        for slot_index, position in enumerate(positions):
            base = embeddings[position] if anchor_slots is None else anchor_slots[slot_index]
            embeddings[position] = base + steering_slots[slot_index]
        root_embeddings.append(embeddings)

    max_length = max(embedding.shape[0] for embedding in root_embeddings)
    padded = torch.zeros(
        (len(contexts), max_length, hidden_size),
        dtype=root_embeddings[0].dtype,
        device=root_embeddings[0].device,
    )
    attention_mask = torch.zeros(
        (len(contexts), max_length),
        dtype=torch.long,
        device=root_embeddings[0].device,
    )
    for row_index, embeddings in enumerate(root_embeddings):
        start = max_length - embeddings.shape[0]
        padded[row_index, start:] = embeddings
        attention_mask[row_index, start:] = 1

    forward_kwargs = {
        "inputs_embeds": padded,
        "attention_mask": attention_mask,
        "use_cache": True,
        "return_dict": True,
        **_qwen3_vl_text_position_kwargs(model, attention_mask),
    }
    prefix_output = None
    for limit_kwargs in ({"logits_to_keep": 1}, {"num_logits_to_keep": 1}, {}):
        try:
            prefix_output = model(**forward_kwargs, **limit_kwargs)
            break
        except TypeError as exc:
            if limit_kwargs and "unexpected keyword" in str(exc):
                continue
            raise
    if prefix_output is None:
        raise RuntimeError("Unable to run the intervened prefix prefill.")
    root_cache = getattr(prefix_output, "past_key_values", None)
    if root_cache is None:
        raise RuntimeError("Intervened prefix prefill returned no KV cache.")
    edge_logprobs: dict[tuple[tuple[int, ...], int], Any] = {}
    context_caches: dict[tuple[int, ...], Any] = {(): root_cache}
    root_logits = prefix_output.logits[:, -1, :].float()
    root_children = sorted(children_by_context[()])
    root_selected = torch.log_softmax(root_logits, dim=-1).index_select(
        1,
        torch.tensor(root_children, dtype=torch.long, device=root_logits.device),
    )
    for child_index, child_id in enumerate(root_children):
        edge_logprobs[((), child_id)] = root_selected[:, child_index]

    sibling_contexts: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
    for context_ids in ordered_contexts:
        if context_ids:
            sibling_contexts.setdefault(context_ids[:-1], []).append(context_ids)
    for parent_context in sorted(sibling_contexts, key=lambda item: (len(item), item)):
        parent_cache = context_caches[parent_context]
        siblings = sorted(sibling_contexts[parent_context])
        for start in range(0, len(siblings), branch_batch_size):
            branch_contexts = siblings[start : start + branch_batch_size]
            branch_count = len(branch_contexts)
            branch_cache = repeat_prefix_cache_rows(
                clone_prefix_cache(parent_cache, detach=True),
                branch_count,
            )
            branch_input_ids = torch.tensor(
                [context_ids[-1] for context_ids in branch_contexts],
                dtype=torch.long,
                device=root_logits.device,
            ).repeat(len(contexts)).unsqueeze(1)
            context_length = len(branch_contexts[0])
            branch_attention = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (len(contexts), context_length),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ],
                dim=1,
            ).repeat_interleave(branch_count, dim=0)
            physical_position = max_length + context_length - 1
            cache_position = torch.tensor(
                [physical_position], dtype=torch.long, device=root_logits.device
            )
            position_ids = cache_position.unsqueeze(0).expand(
                len(contexts) * branch_count,
                -1,
            )
            needs_cache = any(
                context_ids in cache_parent_contexts
                for context_ids in branch_contexts
            )
            output = _model_forward_with_cache(
                model=model,
                input_ids=branch_input_ids,
                attention_mask=branch_attention,
                past_key_values=branch_cache,
                position_ids=position_ids,
                cache_position=cache_position,
                return_updated_cache=needs_cache,
            )
            logits = output.logits[:, -1, :].float().reshape(
                len(contexts), branch_count, -1
            )
            for branch_index, context_ids in enumerate(branch_contexts):
                child_ids = sorted(children_by_context[context_ids])
                selected = torch.log_softmax(logits[:, branch_index, :], dim=-1)
                selected = selected.index_select(
                    1,
                    torch.tensor(child_ids, dtype=torch.long, device=logits.device),
                )
                for child_index, child_id in enumerate(child_ids):
                    edge_logprobs[(context_ids, child_id)] = selected[:, child_index]
                if context_ids in cache_parent_contexts:
                    context_caches[context_ids] = select_prefix_cache_branch(
                        output.past_key_values,
                        branch_index=branch_index,
                        branch_count=branch_count,
                    )

    candidate_scores = []
    for target_ids in target_paths:
        context_ids: tuple[int, ...] = ()
        score = None
        for token_id in target_ids:
            edge = edge_logprobs[(context_ids, int(token_id))]
            score = edge if score is None else score + edge
            context_ids = (*context_ids, int(token_id))
        if score is None:
            raise RuntimeError("A cached candidate path produced no score.")
        candidate_scores.append(score)
    return torch.stack(candidate_scores, dim=1)


def select_behavior_objective_contexts(
    *,
    contexts: dict[tuple[int, int], BehaviorContext],
    target_index: int,
    objective: str,
) -> tuple[dict[tuple[int, int], BehaviorContext], int]:
    """Select only the target continuation when optimizing raw target NLL."""
    if objective == "candidate_normalized":
        return contexts, target_index
    if objective != "raw_target_nll":
        raise ValueError(f"Unsupported behavior objective: {objective}")

    selected = {}
    for key, context in contexts.items():
        if not 0 <= target_index < len(context.candidates):
            raise ValueError("Behavior target index is outside the candidate set.")
        selected[key] = replace(
            context,
            candidates=(context.candidates[target_index],),
        )
    return selected, 0


def behavior_per_context_losses(
    *,
    candidate_logprobs: Any,
    target_index: int,
    objective: str,
) -> Any:
    """Return raw-target or selected-candidate-normalized context losses."""
    import torch

    scores = candidate_logprobs
    if scores.ndim != 2 or not 0 <= target_index < scores.shape[1]:
        raise ValueError("Behavior candidate scores have invalid dimensions.")
    if objective == "raw_target_nll":
        return -scores[:, target_index]
    if objective == "candidate_normalized":
        return -torch.log_softmax(scores, dim=1)[:, target_index]
    raise ValueError(f"Unsupported behavior objective: {objective}")


def behavior_checkpoint_signature(
    *,
    config: NumberSpaceConfig,
    numbers: list[str],
) -> str:
    """Build a signature covering every beta-defining setting."""
    payload = {
        "analysis_version": ANALYSIS_VERSION,
        "model_id": config.model_id,
        "traits": config.traits,
        "numbers": numbers,
        "questions": config.questions,
        "favorite_number_system_prompt_template": (
            config.favorite_number_system_prompt_template
        ),
        "assistant_prefill": config.assistant_prefill,
        "behavior_number_weighting": config.behavior_number_weighting,
        "behavior_objective": config.behavior_objective,
        "beta_init_std": config.beta_init_std,
        "shared_beta_init_across_traits": config.shared_beta_init_across_traits,
        "train_question_count": config.train_question_count,
        "epochs": config.epochs,
        "early_stopping_patience": config.early_stopping_patience,
        "early_stopping_min_delta": config.early_stopping_min_delta,
        "context_batch_size": config.context_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "intervention_scale": config.intervention_scale,
        "seed": config.seed,
    }
    if config.beta_init_mode != DEFAULT_BETA_INIT_MODE:
        payload["beta_init_mode"] = config.beta_init_mode
    return stable_json_hash(payload)


def initialize_behavior_beta(
    *,
    config: NumberSpaceConfig,
    trait_index: int,
    number_count: int,
    device: Any,
    centered_t1_coefficients: np.ndarray | None,
) -> tuple[Any, str]:
    """Create a Gaussian or exact centered-T1 coefficient initialization."""
    import torch

    beta_init_mode = getattr(config, "beta_init_mode", DEFAULT_BETA_INIT_MODE)
    if beta_init_mode == "gaussian":
        generator = torch.Generator(device=device)
        beta_seed = (
            config.seed
            if config.shared_beta_init_across_traits
            else config.seed + trait_index
        )
        generator.manual_seed(beta_seed)
        initial = torch.randn(
            number_count,
            dtype=torch.float32,
            device=device,
            generator=generator,
        ) * config.beta_init_std
        return initial, f"gaussian(seed={beta_seed})"

    if beta_init_mode == "centered_t1":
        if centered_t1_coefficients is None:
            raise ValueError(
                "centered_t1 beta initialization requires dataset coefficients."
            )
        values = np.asarray(centered_t1_coefficients, dtype=np.float64)
        if (
            values.shape != (number_count,)
            or not np.all(np.isfinite(values))
            or np.linalg.norm(values) == 0.0
        ):
            raise ValueError(
                "Centered-T1 beta initialization must be a finite nonzero "
                f"vector with shape ({number_count},)."
            )
        initial = torch.as_tensor(
            values,
            dtype=torch.float32,
            device=device,
        ).clone()
        return initial, "centered_t1_coefficients"

    raise ValueError(f"Unsupported beta initialization mode: {beta_init_mode}")


def beta_checkpoint_path(beta_dir: Path, trait: str) -> Path:
    """Return the per-trait beta checkpoint path."""
    safe_trait = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in trait
    )
    return beta_dir / f"{safe_trait}.npz"


def load_beta_checkpoint(
    *,
    path: Path,
    trait: str,
    signature: str,
    number_count: int,
    intervention_size: int,
) -> BehaviorBetaResult | None:
    """Load one compatible completed beta checkpoint."""
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            saved_signature = str(data["signature"].item())
            saved_trait = str(data["trait"].item())
            beta = np.asarray(data["beta"], dtype=np.float64)
            intervention = np.asarray(data["intervention"], dtype=np.float64)
            step_losses = np.asarray(data["step_losses"], dtype=np.float64)
            epoch_train_losses = np.asarray(
                data["epoch_train_losses"],
                dtype=np.float64,
            )
            validation_losses = np.asarray(
                data["validation_losses"],
                dtype=np.float64,
            )
            best_epoch = int(data["best_epoch"].item())
            epochs_completed = int(data["epochs_completed"].item())
            stopped_early = bool(data["stopped_early"].item())
            initial_beta_norm = float(data["initial_beta_norm"].item())
            initial_intervention_norm = float(
                data["initial_intervention_norm"].item()
            )
            question_counts = np.asarray(
                data["question_sample_counts"],
                dtype=np.int64,
            )
            number_counts = np.asarray(
                data["number_sample_counts"],
                dtype=np.int64,
            )
    except (OSError, KeyError, ValueError) as exc:
        logger.warning("Ignoring unreadable beta checkpoint {}: {}", path, exc)
        return None
    if (
        saved_signature != signature
        or saved_trait != trait
        or beta.shape != (number_count,)
        or intervention.shape != (intervention_size,)
    ):
        logger.warning("Ignoring incompatible beta checkpoint {}", path)
        return None
    beta_norm = float(np.linalg.norm(beta))
    if (
        not np.all(np.isfinite(beta))
        or not np.all(np.isfinite(intervention))
        or beta_norm <= 0.0
        or not math.isfinite(initial_beta_norm)
        or not math.isfinite(initial_intervention_norm)
        or initial_beta_norm <= 0.0
        or initial_intervention_norm < 0.0
        or epochs_completed != len(validation_losses)
        or not 1 <= best_epoch <= epochs_completed
    ):
        logger.warning("Ignoring invalid beta checkpoint {}", path)
        return None
    logger.info("Reusing completed behavior beta for {}", trait)
    return BehaviorBetaResult(
        trait=trait,
        beta=beta,
        intervention=intervention,
        step_losses=step_losses.tolist(),
        epoch_train_losses=epoch_train_losses.tolist(),
        validation_losses=validation_losses.tolist(),
        best_epoch=best_epoch,
        epochs_completed=epochs_completed,
        stopped_early=stopped_early,
        initial_beta_norm=initial_beta_norm,
        initial_intervention_norm=initial_intervention_norm,
        question_sample_counts=question_counts.tolist(),
        number_sample_counts=number_counts.tolist(),
        loaded_from_checkpoint=True,
    )


def save_beta_checkpoint(
    *,
    path: Path,
    result: BehaviorBetaResult,
    signature: str,
) -> None:
    """Save one completed behavior beta atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        np.savez_compressed(
            f,
            signature=np.array(signature),
            trait=np.array(result.trait),
            beta=np.asarray(result.beta, dtype=np.float64),
            intervention=np.asarray(result.intervention, dtype=np.float64),
            step_losses=np.asarray(result.step_losses, dtype=np.float64),
            epoch_train_losses=np.asarray(
                result.epoch_train_losses,
                dtype=np.float64,
            ),
            validation_losses=np.asarray(
                result.validation_losses,
                dtype=np.float64,
            ),
            best_epoch=np.asarray(result.best_epoch, dtype=np.int64),
            epochs_completed=np.asarray(
                result.epochs_completed,
                dtype=np.int64,
            ),
            stopped_early=np.asarray(result.stopped_early, dtype=np.bool_),
            initial_beta_norm=np.asarray(
                result.initial_beta_norm,
                dtype=np.float64,
            ),
            initial_intervention_norm=np.asarray(
                result.initial_intervention_norm,
                dtype=np.float64,
            ),
            question_sample_counts=np.asarray(
                result.question_sample_counts,
                dtype=np.int64,
            ),
            number_sample_counts=np.asarray(
                result.number_sample_counts,
                dtype=np.int64,
            ),
        )
    tmp_path.replace(path)
    logger.success("Saved behavior beta checkpoint to {}", path)


def optimize_behavior_beta(
    *,
    trait: str,
    trait_index: int,
    colors: list[str],
    model: Any,
    input_embedding: Any,
    contexts: dict[tuple[int, int], BehaviorContext],
    number_embedding_matrix: Any,
    neutral_number_probabilities: np.ndarray,
    config: NumberSpaceConfig,
    centered_t1_coefficients: np.ndarray | None = None,
) -> BehaviorBetaResult:
    """Optimize on the exhaustive train set and select by fixed validation."""
    import torch

    target_index = colors.index(trait)
    objective_contexts, objective_target_index = select_behavior_objective_contexts(
        contexts=contexts,
        target_index=target_index,
        objective=config.behavior_objective,
    )
    initial, initialization_description = initialize_behavior_beta(
        config=config,
        trait_index=trait_index,
        number_count=int(number_embedding_matrix.shape[0]),
        device=input_embedding.weight.device,
        centered_t1_coefficients=centered_t1_coefficients,
    )
    initial_beta_norm = float(torch.linalg.vector_norm(initial).detach().cpu())
    initial_intervention = torch.matmul(
        initial.float(),
        number_embedding_matrix.float(),
    ) * config.intervention_scale
    initial_intervention_norm = float(
        torch.linalg.vector_norm(initial_intervention).detach().cpu()
    )
    logger.info(
        "beta trait={} objective={} initialization={} "
        "initial_beta_norm={:.6f} "
        "initial_intervention_norm={:.6f}",
        trait,
        config.behavior_objective,
        initialization_description,
        initial_beta_norm,
        initial_intervention_norm,
    )
    beta = torch.nn.Parameter(initial)
    optimizer = torch.optim.Adam([beta], lr=config.learning_rate)
    shuffle_rng = np.random.default_rng(config.seed + trait_index)
    train_question_indices, validation_question_indices = split_question_indices(
        question_count=len(config.questions),
        train_question_count=config.train_question_count,
    )
    train_indices = cartesian_context_indices(
        number_count=len(neutral_number_probabilities),
        question_indices=train_question_indices,
    )
    validation_indices = cartesian_context_indices(
        number_count=len(neutral_number_probabilities),
        question_indices=validation_question_indices,
    )
    train_weights = context_loss_weights(
        indices=train_indices,
        neutral_number_probabilities=neutral_number_probabilities,
        mode=config.behavior_number_weighting,
    )
    validation_weights = context_loss_weights(
        indices=validation_indices,
        neutral_number_probabilities=neutral_number_probabilities,
        mode=config.behavior_number_weighting,
    )
    question_counts = np.zeros(len(config.questions), dtype=np.int64)
    number_counts = np.zeros(
        len(neutral_number_probabilities),
        dtype=np.int64,
    )
    optimizer_losses = []
    epoch_train_losses = []
    validation_losses = []
    best_validation_loss = math.inf
    best_epoch = 0
    best_beta = beta.detach().clone()
    epochs_without_improvement = 0
    stopped_early = False
    effective_batch_size = (
        config.context_batch_size * config.gradient_accumulation_steps
    )
    updates_per_epoch = math.ceil(len(train_indices) / effective_batch_size)
    total_updates = config.epochs * updates_per_epoch
    update_index = 0

    for epoch in range(config.epochs):
        order = shuffle_rng.permutation(len(train_indices))
        epoch_weighted_loss_sum = 0.0
        for group_start in range(0, len(train_indices), effective_batch_size):
            group_positions = order[group_start : group_start + effective_batch_size]
            group_size = len(group_positions)
            optimizer.zero_grad(set_to_none=True)
            group_loss_value = 0.0
            for micro_start in range(
                0,
                group_size,
                config.context_batch_size,
            ):
                micro_positions = group_positions[
                    micro_start : micro_start + config.context_batch_size
                ]
                batch_indices = [
                    train_indices[int(position)] for position in micro_positions
                ]
                batch_contexts = [
                    objective_contexts[index] for index in batch_indices
                ]
                for number_index, question_index in batch_indices:
                    number_counts[number_index] += 1
                    question_counts[question_index] += 1

                candidate_logprobs = exact_candidate_logprob_matrix(
                    model=model,
                    input_embedding=input_embedding,
                    contexts=batch_contexts,
                    number_embedding_matrix=number_embedding_matrix,
                    beta=beta,
                    intervention_scale=config.intervention_scale,
                )
                per_context_losses = behavior_per_context_losses(
                    candidate_logprobs=candidate_logprobs,
                    target_index=objective_target_index,
                    objective=config.behavior_objective,
                )
                weights = torch.as_tensor(
                    train_weights[micro_positions],
                    dtype=per_context_losses.dtype,
                    device=per_context_losses.device,
                )
                micro_loss = torch.sum(weights * per_context_losses) / group_size
                if not bool(torch.isfinite(micro_loss)):
                    raise RuntimeError(f"Non-finite behavior loss for trait {trait}.")
                micro_loss.backward()
                group_loss_value += float(micro_loss.detach().cpu())

            optimizer.step()
            with torch.no_grad():
                if not bool(torch.all(torch.isfinite(beta))):
                    raise RuntimeError(f"Non-finite beta for trait {trait}.")
            update_index += 1
            optimizer_losses.append(group_loss_value)
            epoch_weighted_loss_sum += group_loss_value * group_size
            if config.log_every > 0 and (
                update_index == 1
                or update_index % config.log_every == 0
                or update_index == total_updates
            ):
                logger.info(
                    "beta trait={} epoch={}/{} update={}/{} train_loss={:.6f}",
                    trait,
                    epoch + 1,
                    config.epochs,
                    update_index,
                    total_updates,
                    group_loss_value,
                )

        epoch_train_loss = epoch_weighted_loss_sum / len(train_indices)
        validation_loss = evaluate_behavior_loss(
            target_index=objective_target_index,
            model=model,
            input_embedding=input_embedding,
            contexts=objective_contexts,
            indices=validation_indices,
            weights=validation_weights,
            number_embedding_matrix=number_embedding_matrix,
            beta=beta,
            intervention_scale=config.intervention_scale,
            context_batch_size=config.context_batch_size,
            objective=config.behavior_objective,
        )
        epoch_train_losses.append(epoch_train_loss)
        validation_losses.append(validation_loss)
        with torch.no_grad():
            beta_norm = float(torch.linalg.vector_norm(beta).detach().cpu())
            intervention_norm = float(
                torch.linalg.vector_norm(
                    torch.matmul(
                        beta.float(),
                        number_embedding_matrix.float(),
                    )
                    * config.intervention_scale
                )
                .detach()
                .cpu()
            )
        logger.info(
            "beta trait={} epoch={}/{} online_train_loss={:.6f} "
            "validation_loss={:.6f} beta_norm={:.6f} "
            "intervention_norm={:.6f}",
            trait,
            epoch + 1,
            config.epochs,
            epoch_train_loss,
            validation_loss,
            beta_norm,
            intervention_norm,
        )
        if (
            validation_loss
            < best_validation_loss - config.early_stopping_min_delta
        ):
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            best_beta = beta.detach().clone()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if (
                config.early_stopping_patience > 0
                and epochs_without_improvement
                >= config.early_stopping_patience
            ):
                stopped_early = True
                logger.info(
                    "Early stopping beta trait={} after epoch {}; "
                    "best epoch={} validation_loss={:.6f}",
                    trait,
                    epoch + 1,
                    best_epoch,
                    best_validation_loss,
                )
                break

    beta_array = best_beta.float().cpu().numpy().astype(np.float64)
    if not np.all(np.isfinite(beta_array)) or np.linalg.norm(beta_array) == 0.0:
        raise RuntimeError(f"Invalid best beta for trait {trait}.")
    intervention_array = (
        torch.matmul(
            best_beta.float(),
            number_embedding_matrix.float(),
        )
        .cpu()
        .numpy()
        .astype(np.float64)
        * config.intervention_scale
    )
    return BehaviorBetaResult(
        trait=trait,
        beta=beta_array,
        intervention=intervention_array,
        step_losses=optimizer_losses,
        epoch_train_losses=epoch_train_losses,
        validation_losses=validation_losses,
        best_epoch=best_epoch,
        epochs_completed=len(validation_losses),
        stopped_early=stopped_early,
        initial_beta_norm=initial_beta_norm,
        initial_intervention_norm=initial_intervention_norm,
        question_sample_counts=question_counts.tolist(),
        number_sample_counts=number_counts.tolist(),
        loaded_from_checkpoint=False,
    )


def evaluate_behavior_loss(
    *,
    target_index: int,
    model: Any,
    input_embedding: Any,
    contexts: dict[tuple[int, int], BehaviorContext],
    indices: list[tuple[int, int]],
    weights: np.ndarray,
    number_embedding_matrix: Any,
    beta: Any,
    intervention_scale: float,
    context_batch_size: int,
    objective: str,
) -> float:
    """Evaluate one fixed weighted Cartesian context set without gradients."""
    import torch

    if len(indices) != len(weights) or not indices:
        raise ValueError("Validation indices and weights must align and be non-empty.")
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(indices), context_batch_size):
            stop = min(start + context_batch_size, len(indices))
            batch_indices = indices[start:stop]
            candidate_logprobs = exact_candidate_logprob_matrix(
                model=model,
                input_embedding=input_embedding,
                contexts=[contexts[index] for index in batch_indices],
                number_embedding_matrix=number_embedding_matrix,
                beta=beta,
                intervention_scale=intervention_scale,
            )
            losses = behavior_per_context_losses(
                candidate_logprobs=candidate_logprobs,
                target_index=target_index,
                objective=objective,
            )
            batch_weights = torch.as_tensor(
                weights[start:stop],
                dtype=losses.dtype,
                device=losses.device,
            )
            total += float(torch.sum(batch_weights * losses).cpu())
    return total / len(indices)


def pairwise_cosine_matrix(
    behavior_vectors: np.ndarray,
    dataset_vectors: np.ndarray,
) -> np.ndarray:
    """Return row-behavior by column-dataset signed cosine similarities."""
    behavior = np.asarray(behavior_vectors, dtype=np.float64)
    dataset = np.asarray(dataset_vectors, dtype=np.float64)
    if behavior.ndim != 2 or dataset.ndim != 2:
        raise ValueError("Cosine inputs must be two-dimensional.")
    if behavior.shape[1] != dataset.shape[1]:
        raise ValueError("Behavior and dataset vectors must share a dimension.")
    behavior_norms = np.linalg.norm(behavior, axis=1)
    dataset_norms = np.linalg.norm(dataset, axis=1)
    if np.any(behavior_norms == 0.0) or np.any(dataset_norms == 0.0):
        raise ValueError("Cannot compute cosine similarities for zero-norm vectors.")
    return (behavior @ dataset.T) / (behavior_norms[:, None] * dataset_norms[None, :])


def t2_pairwise_cosine_matrices(
    behavior_interventions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw and cross-trait-centered pairwise ordered-T2 cosines."""
    values = np.asarray(behavior_interventions, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("T2 interventions must be a trait-by-feature matrix.")
    centered = values - np.mean(values, axis=0, keepdims=True)
    return (
        pairwise_cosine_matrix(values, values),
        pairwise_cosine_matrix(centered, centered),
        centered,
    )


def nested_matrix(
    values: np.ndarray,
    rows: list[str],
    columns: list[str],
) -> dict[str, dict[str, float]]:
    """Convert a matrix to an ordered nested JSON mapping."""
    array = np.asarray(values, dtype=np.float64)
    return {
        row: {
            column: float(array[row_index, column_index])
            for column_index, column in enumerate(columns)
        }
        for row_index, row in enumerate(rows)
    }


def write_matrix_csv(
    path: Path,
    matrix: np.ndarray,
    rows: list[str],
    columns: list[str],
    corner_label: str,
) -> None:
    """Write one labeled numeric matrix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([corner_label, *columns])
        for row, values in zip(rows, np.asarray(matrix).tolist()):
            writer.writerow([row, *values])
    logger.success("Saved matrix CSV to {}", path)


def write_cosine_heatmap(
    *,
    path: Path,
    matrix: np.ndarray,
    behavior_traits: list[str],
    dataset_traits: list[str],
    dpi: int,
    dataset_axis_label: str = "Dataset trait z: centered embedding shift d(z)",
    title: str = ("Ordered number-space behavior vs. centered dataset-shift cosine"),
) -> None:
    """Write an ordered-intervention by dataset-vector cosine heatmap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(matrix, dtype=np.float64)
    max_abs = max(float(np.max(np.abs(values))), 1e-12)
    fig, ax = plt.subplots(figsize=(15.5, 13.5))
    image = ax.imshow(
        values,
        cmap="RdBu_r",
        vmin=-max_abs,
        vmax=max_abs,
        aspect="auto",
    )
    ax.set_xticks(
        np.arange(len(dataset_traits)),
        dataset_traits,
        rotation=45,
        ha="right",
    )
    ax.set_yticks(np.arange(len(behavior_traits)), behavior_traits)
    ax.set_xlabel(dataset_axis_label)
    ax.set_ylabel("Behavior target u: ordered intervention v(u)")
    ax.set_title(title)
    threshold = 0.55 * max_abs
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = float(values[row_index, column_index])
            ax.text(
                column_index,
                row_index,
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if abs(value) >= threshold else "black",
            )
    fig.colorbar(image, ax=ax, label="Cosine similarity", fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    logger.success("Saved cosine heatmap to {}", path)


def write_trait_pairwise_cosine_heatmap(
    *,
    path: Path,
    matrix: np.ndarray,
    traits: list[str],
    dpi: int,
    centered: bool,
) -> None:
    """Write an annotated square pairwise T2 cosine heatmap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(matrix, dtype=np.float64)
    expected_shape = (len(traits), len(traits))
    if values.shape != expected_shape:
        raise ValueError(f"T2 cosine matrix must have shape {expected_shape}.")
    fig, ax = plt.subplots(figsize=(14.5, 13.5))
    image = ax.imshow(
        values,
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        aspect="equal",
    )
    ax.set_xticks(np.arange(len(traits)), traits, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(traits)), traits)
    ax.set_xlabel("Behavior trait")
    ax.set_ylabel("Behavior trait")
    qualifier = "Trait-centered ordered T2" if centered else "Ordered T2"
    ax.set_title(f"{qualifier} pairwise cosine similarity")
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = float(values[row_index, column_index])
            ax.text(
                column_index,
                row_index,
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if abs(value) >= 0.55 else "black",
            )
    fig.colorbar(
        image,
        ax=ax,
        label="Cosine similarity",
        fraction=0.046,
        pad=0.04,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    logger.success("Saved T2 pairwise cosine heatmap to {}", path)


def validate_config(config: NumberSpaceConfig) -> None:
    """Validate the experiment before loading a large model."""
    if len(config.traits) != 18 or len(set(config.traits)) != 18:
        raise ValueError("This experiment requires exactly 18 unique colors.")
    if (
        not config.behavior_traits
        or len(set(config.behavior_traits)) != len(config.behavior_traits)
        or any(trait not in config.traits for trait in config.behavior_traits)
    ):
        raise ValueError(
            "Behavior traits must be a non-empty unique subset of the 18 "
            "dataset traits."
        )
    if not config.checkpoint_only and config.behavior_traits != config.traits:
        raise ValueError(
            "A behavior-trait subset requires --checkpoint-only. Run the "
            "default full command afterward to aggregate all checkpoints."
        )
    if not config.questions or len(set(config.questions)) != len(config.questions):
        raise ValueError("Favorite-color questions must be non-empty and unique.")
    if "{number}" not in config.favorite_number_system_prompt_template:
        raise ValueError(
            "--favorite-number-system-prompt-template must contain {number}."
        )
    if not config.assistant_prefill:
        raise ValueError("Assistant prefill must be non-empty.")
    if config.behavior_objective not in {
        "raw_target_nll",
        "candidate_normalized",
    }:
        raise ValueError(f"Unsupported behavior objective: {config.behavior_objective}")
    if config.beta_init_mode not in {"gaussian", "centered_t1"}:
        raise ValueError(
            f"Unsupported beta initialization mode: {config.beta_init_mode}"
        )
    if (
        config.beta_init_mode == "centered_t1"
        and config.shared_beta_init_across_traits
    ):
        raise ValueError(
            "--shared-beta-init-across-traits cannot be combined with "
            "--beta-init-mode centered_t1 because each trait has its own T1."
        )
    if not 0 < config.train_question_count < len(config.questions):
        raise ValueError(
            "--train-question-count must leave at least one held-out question."
        )
    if config.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if config.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience must be non-negative.")
    if (
        not math.isfinite(config.early_stopping_min_delta)
        or config.early_stopping_min_delta < 0.0
    ):
        raise ValueError(
            "--early-stopping-min-delta must be finite and non-negative."
        )
    if config.context_batch_size <= 0:
        raise ValueError("--context-batch-size must be positive.")
    if config.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive.")
    for name, value in (
        ("learning-rate", config.learning_rate),
        ("intervention-scale", config.intervention_scale),
        ("beta-init-std", config.beta_init_std),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name} must be finite and positive.")
    if config.log_every < 0:
        raise ValueError("--log-every must be non-negative.")
    if config.heatmap_dpi <= 0:
        raise ValueError("--heatmap-dpi must be positive.")


def run_raw_heatmap_only(config: NumberSpaceConfig) -> dict[str, Any]:
    """Generate the raw dataset comparison from saved aggregate tensors."""
    validate_config(config)
    paths = output_paths(config)
    if not paths["tensors"].is_file():
        raise FileNotFoundError(
            f"Saved aggregate tensors are required: {paths['tensors']}"
        )
    dataset = load_dataset_number_vectors(config)
    with np.load(paths["tensors"], allow_pickle=False) as data:
        required = {
            "behavior_ordered_interventions",
            "number_embedding_matrix",
            "traits",
            "numbers",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(
                f"Saved aggregate tensors are missing arrays: {sorted(missing)}"
            )
        saved_traits = [str(value) for value in data["traits"].tolist()]
        saved_numbers = [str(value) for value in data["numbers"].tolist()]
        behavior_interventions = np.asarray(
            data["behavior_ordered_interventions"],
            dtype=np.float64,
        )
        number_embedding_values = np.asarray(
            data["number_embedding_matrix"],
            dtype=np.float64,
        )
    if saved_traits != config.traits or saved_numbers != dataset.numbers:
        raise ValueError(
            "Saved aggregate tensor labels do not match the requested analysis."
        )
    if number_embedding_values.shape[0] != len(dataset.numbers):
        raise ValueError("Saved ordered number embedding matrix has invalid shape.")

    raw_dataset_embedding_vectors = (
        dataset.raw_weighted_coefficients @ number_embedding_values
    )
    raw_cosine_values = pairwise_cosine_matrix(
        behavior_interventions,
        raw_dataset_embedding_vectors,
    )
    raw_diagonal = {
        trait: float(raw_cosine_values[index, index])
        for index, trait in enumerate(config.traits)
    }
    report = {
        "analysis_version": ANALYSIS_VERSION,
        "method": "raw_ordered_number_embedding_behavior_dataset_cosine",
        "definition": (
            "d_raw_z=mean_x sum_n p0(n|x) Delta_z(x,n) e_n, without "
            "probability centering or cross-trait centering"
        ),
        "matrix_orientation": "behavior_target_color x dataset_teacher_color",
        "traits": config.traits,
        "cosine_matrix": nested_matrix(
            raw_cosine_values,
            config.traits,
            config.traits,
        ),
        "diagonal_matched_trait_cosines": raw_diagonal,
        "outputs": {
            "csv": str(paths["raw_cosine_csv"]),
            "heatmap": str(paths["raw_heatmap"]),
            "json": str(paths["raw_json"]),
        },
    }
    write_matrix_csv(
        paths["raw_cosine_csv"],
        raw_cosine_values,
        config.traits,
        config.traits,
        "behavior_intervention__raw_dataset_embedding_shift",
    )
    if config.save_plot:
        write_cosine_heatmap(
            path=paths["raw_heatmap"],
            matrix=raw_cosine_values,
            behavior_traits=config.traits,
            dataset_traits=config.traits,
            dpi=config.heatmap_dpi,
            dataset_axis_label=(
                "Dataset trait z: raw probability-weighted embedding shift"
            ),
            title=("Ordered number-space behavior vs. raw dataset-shift cosine"),
        )
    paths["raw_json"].write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.success("Saved raw dataset cosine report to {}", paths["raw_json"])
    return report


def run_t2_heatmaps_only(config: NumberSpaceConfig) -> dict[str, Any]:
    """Generate raw and trait-centered pairwise T2 cosine heatmaps."""
    validate_config(config)
    paths = output_paths(config)
    if not paths["tensors"].is_file():
        raise FileNotFoundError(
            f"Saved aggregate tensors are required: {paths['tensors']}"
        )
    with np.load(paths["tensors"], allow_pickle=False) as data:
        required = {"behavior_ordered_interventions", "traits"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(
                f"Saved aggregate tensors are missing arrays: {sorted(missing)}"
            )
        saved_traits = [str(value) for value in data["traits"].tolist()]
        behavior_interventions = np.asarray(
            data["behavior_ordered_interventions"],
            dtype=np.float64,
        )
    if saved_traits != config.traits:
        raise ValueError(
            "Saved aggregate tensor traits do not match the requested analysis."
        )

    (
        cosine_values,
        centered_cosine_values,
        centered_interventions,
    ) = t2_pairwise_cosine_matrices(behavior_interventions)
    report = {
        "analysis_version": ANALYSIS_VERSION,
        "method": "ordered_T2_pairwise_trait_cosines",
        "definitions": {
            "T2": "T2_u=v_u=W_N^T beta_u in ordered 2d embedding space.",
            "uncentered_cosine": "cos(T2_u,T2_s)",
            "trait_centered_T2": ("T2_centered_u=T2_u-(1/18) sum_r T2_r"),
            "trait_centered_cosine": ("cos(T2_centered_u,T2_centered_s)"),
        },
        "matrix_orientation": "behavior_trait x behavior_trait",
        "traits": config.traits,
        "uncentered_cosine_matrix": nested_matrix(
            cosine_values,
            config.traits,
            config.traits,
        ),
        "trait_centered_cosine_matrix": nested_matrix(
            centered_cosine_values,
            config.traits,
            config.traits,
        ),
        "mean_T2_norm": float(np.linalg.norm(np.mean(behavior_interventions, axis=0))),
        "centered_T2_row_mean_max_absolute": float(
            np.max(np.abs(np.mean(centered_interventions, axis=0)))
        ),
        "outputs": {
            "uncentered_csv": str(paths["t2_cosine_csv"]),
            "centered_csv": str(paths["t2_centered_cosine_csv"]),
            "uncentered_heatmap": str(paths["t2_heatmap"]),
            "centered_heatmap": str(paths["t2_centered_heatmap"]),
            "json": str(paths["t2_json"]),
        },
    }
    write_matrix_csv(
        paths["t2_cosine_csv"],
        cosine_values,
        config.traits,
        config.traits,
        "T2_behavior_trait__T2_behavior_trait",
    )
    write_matrix_csv(
        paths["t2_centered_cosine_csv"],
        centered_cosine_values,
        config.traits,
        config.traits,
        "centered_T2_behavior_trait__centered_T2_behavior_trait",
    )
    if config.save_plot:
        write_trait_pairwise_cosine_heatmap(
            path=paths["t2_heatmap"],
            matrix=cosine_values,
            traits=config.traits,
            dpi=config.heatmap_dpi,
            centered=False,
        )
        write_trait_pairwise_cosine_heatmap(
            path=paths["t2_centered_heatmap"],
            matrix=centered_cosine_values,
            traits=config.traits,
            dpi=config.heatmap_dpi,
            centered=True,
        )
    paths["t2_json"].write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.success("Saved ordered T2 cosine report to {}", paths["t2_json"])
    return report


def run(config: NumberSpaceConfig) -> dict[str, Any]:
    """Compute, optimize, compare, and save all number-space vectors."""
    import torch

    validate_config(config)
    paths = output_paths(config)
    if not config.checkpoint_only and config.skip_existing and paths["json"].is_file():
        existing = json.loads(paths["json"].read_text(encoding="utf-8"))
        if (
            existing.get("analysis_version") == ANALYSIS_VERSION
            and existing.get("config") == serialize_config(config)
            and paths["tensors"].is_file()
            and paths["cosine_csv"].is_file()
            and paths["raw_cosine_csv"].is_file()
            and paths["raw_json"].is_file()
            and paths["t2_cosine_csv"].is_file()
            and paths["t2_centered_cosine_csv"].is_file()
            and paths["t2_json"].is_file()
            and paths["dataset_raw_csv"].is_file()
            and paths["dataset_centered_csv"].is_file()
            and paths["dataset_coefficients_csv"].is_file()
            and paths["behavior_csv"].is_file()
            and (
                not config.save_plot
                or (
                    paths["heatmap"].is_file()
                    and paths["raw_heatmap"].is_file()
                    and paths["t2_heatmap"].is_file()
                    and paths["t2_centered_heatmap"].is_file()
                )
            )
        ):
            logger.info("Reusing completed number-space analysis {}", paths["json"])
            return existing

    dataset = load_dataset_number_vectors(config)
    cache_model_id = str(
        dataset.manifest.get("common_metadata", {}).get("model_id", "")
    )
    if cache_model_id and cache_model_id != config.model_id:
        raise ValueError(
            "Dataset cache/model mismatch: cache uses "
            f"{cache_model_id!r}, behavior optimization requested "
            f"{config.model_id!r}."
        )

    device = resolve_device(torch, config.device)
    torch_dtype = resolve_torch_dtype(torch, config.torch_dtype)
    model, tokenizer = load_direct_model_and_tokenizer(
        model_id=config.model_id,
        device=device,
        torch_dtype=torch_dtype,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    input_embedding = model.get_input_embeddings()
    if input_embedding is None:
        raise RuntimeError("Model does not expose input embeddings.")

    number_embedding_matrix, number_token_ids = build_number_embedding_matrix(
        tokenizer=tokenizer,
        input_embedding=input_embedding,
        numbers=dataset.numbers,
    )
    contexts = build_behavior_contexts(
        tokenizer=tokenizer,
        numbers=dataset.numbers,
        questions=config.questions,
        colors=config.traits,
        favorite_number_system_prompt_template=(
            config.favorite_number_system_prompt_template
        ),
        assistant_prefill=config.assistant_prefill,
    )
    validate_ordered_context_tokenization(
        contexts=contexts,
        number_token_ids=number_token_ids,
    )
    signature = behavior_checkpoint_signature(
        config=config,
        numbers=dataset.numbers,
    )

    traits_to_optimize = (
        config.behavior_traits if config.checkpoint_only else config.traits
    )
    beta_results = []
    for job_index, trait in enumerate(traits_to_optimize):
        trait_index = config.traits.index(trait)
        checkpoint_path = beta_checkpoint_path(paths["beta_dir"], trait)
        result = None
        if config.skip_existing:
            result = load_beta_checkpoint(
                path=checkpoint_path,
                trait=trait,
                signature=signature,
                number_count=len(dataset.numbers),
                intervention_size=int(number_embedding_matrix.shape[1]),
            )
        if result is None:
            logger.info(
                "Optimizing behavior beta for {} ({}/{})",
                trait,
                job_index + 1,
                len(traits_to_optimize),
            )
            result = optimize_behavior_beta(
                trait=trait,
                trait_index=trait_index,
                colors=config.traits,
                model=model,
                input_embedding=input_embedding,
                contexts=contexts,
                number_embedding_matrix=number_embedding_matrix,
                neutral_number_probabilities=(dataset.neutral_number_probabilities),
                centered_t1_coefficients=(
                    dataset.centered_weighted_coefficients[trait_index]
                ),
                config=config,
            )
            save_beta_checkpoint(
                path=checkpoint_path,
                result=result,
                signature=signature,
            )
        beta_results.append(result)

    if config.checkpoint_only:
        checkpoint_report = {
            "analysis_version": ANALYSIS_VERSION,
            "method": "behavior_beta_checkpoint_job",
            "config": serialize_config(config),
            "trained_or_reused_traits": [
                {
                    "trait": result.trait,
                    "best_epoch": result.best_epoch,
                    "best_validation_loss": result.validation_losses[
                        result.best_epoch - 1
                    ],
                    "epochs_completed": result.epochs_completed,
                    "stopped_early": result.stopped_early,
                    "beta_norm": float(np.linalg.norm(result.beta)),
                    "intervention_norm": float(
                        np.linalg.norm(result.intervention)
                    ),
                    "initial_beta_norm": result.initial_beta_norm,
                    "initial_intervention_norm": (
                        result.initial_intervention_norm
                    ),
                    "loaded_from_checkpoint": result.loaded_from_checkpoint,
                    "checkpoint_path": str(
                        beta_checkpoint_path(paths["beta_dir"], result.trait)
                    ),
                }
                for result in beta_results
            ],
        }
        logger.success(
            "Completed checkpoint-only job for traits: {}",
            ", ".join(traits_to_optimize),
        )
        logger.info(
            "Run the full command with --skip-existing after all trait jobs "
            "finish to assemble the cosine matrix."
        )
        del contexts
        del number_embedding_matrix
        del input_embedding
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return checkpoint_report

    behavior_betas = np.stack([result.beta for result in beta_results], axis=0)
    number_embedding_values = number_embedding_matrix.detach().float().cpu().numpy()
    behavior_interventions = np.stack(
        [result.intervention for result in beta_results],
        axis=0,
    )
    recomputed_interventions = (
        behavior_betas @ number_embedding_values * config.intervention_scale
    )
    if not np.allclose(
        behavior_interventions,
        recomputed_interventions,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("Saved behavior interventions do not match W_N^T beta.")
    (
        t2_cosine_values,
        t2_centered_cosine_values,
        centered_behavior_interventions,
    ) = t2_pairwise_cosine_matrices(behavior_interventions)
    dataset_embedding_vectors = (
        dataset.centered_weighted_coefficients @ number_embedding_values
    )
    raw_dataset_embedding_vectors = (
        dataset.raw_weighted_coefficients @ number_embedding_values
    )
    cosine_values = pairwise_cosine_matrix(
        behavior_interventions,
        dataset_embedding_vectors,
    )
    raw_cosine_values = pairwise_cosine_matrix(
        behavior_interventions,
        raw_dataset_embedding_vectors,
    )
    diagonal = {
        trait: float(cosine_values[index, index])
        for index, trait in enumerate(config.traits)
    }
    raw_diagonal = {
        trait: float(raw_cosine_values[index, index])
        for index, trait in enumerate(config.traits)
    }
    first_context = contexts[(0, 0)]
    report = {
        "analysis_version": ANALYSIS_VERSION,
        "method": ("ordered_number_embedding_behavior_vs_centered_dataset_shift"),
        "config": serialize_config(config),
        "matrix_orientation": "behavior_target_color x dataset_teacher_color",
        "definitions": {
            "dataset_raw_shift": (
                "T1(z,n)=mean_x[log p(n|x,z)-log p(n|x,neutral)], using "
                "raw exact continuation log probabilities from the one-number cache"
            ),
            "dataset_probability_centering": (
                "For every prompt x and trait z, subtract "
                "sum_n p0(n|x) Delta_z(x,n) from Delta_z(x,n)."
            ),
            "dataset_trait_centering": (
                "For every prompt x and number n, subtract the mean "
                "probability-centered shift over the 18 teacher traits."
            ),
            "dataset_embedding_shift": (
                "d_z=mean_x sum_n p0(n|x) centered_Delta_z(x,n) e_n"
            ),
            "raw_dataset_embedding_shift": (
                "d_raw_z=mean_x sum_n p0(n|x) Delta_z(x,n) e_n, without "
                "probability centering or cross-trait centering"
            ),
            "number_embedding": (
                "For n=ab, e_n=concat(E_a,E_b)/sqrt(2), giving an ordered "
                "100 by 2d number embedding matrix."
            ),
            "behavior_beta": (
                (
                    "Best-validation unconstrained beta initialized from the "
                    "matching trait's probability-centered, neutral-weighted, "
                    "cross-trait-centered 100-dimensional dataset coefficients"
                    if config.beta_init_mode == "centered_t1"
                    else (
                        "Best-validation unconstrained beta initialized from a "
                        "near-zero Gaussian"
                    )
                )
                + " and optimized without norm projection or explicit "
                "regularization. The two halves of W_N^T beta are added "
                "separately to the first and second favorite-number token."
            ),
            "behavior_objective": (
                "Uniformly enumerate all training contexts, multiply each loss by "
                "100 p0(n), and average. raw_target_nll minimizes the negative raw "
                "exact target-completion log probability; candidate_normalized is "
                "the legacy selected-color competition objective."
            ),
            "cosine": (
                "cos(v_behavior,d_dataset) in ordered 2d embedding space, where "
                "v_behavior=W_N^T beta."
            ),
            "T2_pairwise_cosines": (
                "Both cos(T2_u,T2_s) and cos(T2_u-mean_r T2_r, "
                "T2_s-mean_r T2_r), with T2_u=W_N^T beta_u."
            ),
        },
        "dataset_channel": {
            "cache_group": str(dataset.cache_group),
            "model_id": cache_model_id,
            "prompt_count": dataset.prompt_count,
            "numbers": dataset.numbers,
            "neutral_number_probabilities": (
                dataset.neutral_number_probabilities.tolist()
            ),
            "probability_centered_max_absolute_weighted_mean": float(
                np.max(
                    np.abs(
                        np.sum(
                            dataset.centered_weighted_coefficients,
                            axis=1,
                        )
                    )
                )
            ),
        },
        "behavior_channel": {
            "question_count": len(config.questions),
            "questions": config.questions,
            "context_count": len(contexts),
            "train_question_count": config.train_question_count,
            "train_questions": config.questions[: config.train_question_count],
            "train_context_count": (len(dataset.numbers) * config.train_question_count),
            "validation_question_count": (
                len(config.questions) - config.train_question_count
            ),
            "validation_questions": config.questions[config.train_question_count :],
            "validation_context_count": (
                len(dataset.numbers)
                * (len(config.questions) - config.train_question_count)
            ),
            "effective_batch_size": (
                config.context_batch_size * config.gradient_accumulation_steps
            ),
            "behavior_objective": config.behavior_objective,
            "beta_init_std": config.beta_init_std,
            "shared_beta_init_across_traits": (
                config.shared_beta_init_across_traits
            ),
            "early_stopping_patience": config.early_stopping_patience,
            "early_stopping_min_delta": config.early_stopping_min_delta,
            "optimizer_updates_per_epoch": math.ceil(
                (len(dataset.numbers) * config.train_question_count)
                / (config.context_batch_size * config.gradient_accumulation_steps)
            ),
            "total_optimizer_updates": config.epochs
            * math.ceil(
                (len(dataset.numbers) * config.train_question_count)
                / (config.context_batch_size * config.gradient_accumulation_steps)
            ),
            "ordered_number_embedding_matrix_shape": list(
                number_embedding_values.shape
            ),
            "ordered_number_embedding_matrix_rank": int(
                np.linalg.matrix_rank(number_embedding_values)
            ),
            "number_token_ids": {
                number: token_ids
                for number, token_ids in zip(dataset.numbers, number_token_ids)
            },
            "number_embedding_token_count": {
                number: len(token_ids)
                for number, token_ids in zip(dataset.numbers, number_token_ids)
            },
            "sample_rendered_prompt": first_context.rendered_prompt,
            "sample_number_token_positions_by_color": {
                trait: list(first_context.candidates[index].number_token_positions)
                for index, trait in enumerate(config.traits)
            },
            "sample_color_target_token_ids": {
                trait: list(first_context.candidates[index].target_token_ids)
                for index, trait in enumerate(config.traits)
            },
            "optimization": {
                result.trait: {
                    "first_optimizer_batch_loss": result.step_losses[0],
                    "final_optimizer_batch_loss": result.step_losses[-1],
                    "minimum_optimizer_batch_loss": min(result.step_losses),
                    "epoch_train_losses": result.epoch_train_losses,
                    "validation_losses": result.validation_losses,
                    "best_epoch": result.best_epoch,
                    "epochs_completed": result.epochs_completed,
                    "stopped_early": result.stopped_early,
                    "optimizer_updates_completed": len(result.step_losses),
                    "best_validation_loss": result.validation_losses[
                        result.best_epoch - 1
                    ],
                    "beta_norm": float(np.linalg.norm(result.beta)),
                    "initial_beta_norm": result.initial_beta_norm,
                    "initial_intervention_norm": (
                        result.initial_intervention_norm
                    ),
                    "intervention_norm": float(
                        np.linalg.norm(
                            behavior_interventions[config.traits.index(result.trait)]
                        )
                    ),
                    "training_question_visit_counts": (result.question_sample_counts),
                    "training_number_visit_counts": result.number_sample_counts,
                    "loaded_from_checkpoint": result.loaded_from_checkpoint,
                }
                for result in beta_results
            },
        },
        "traits": config.traits,
        "numbers": dataset.numbers,
        "cosine_matrix": nested_matrix(
            cosine_values,
            config.traits,
            config.traits,
        ),
        "raw_dataset_cosine_matrix": nested_matrix(
            raw_cosine_values,
            config.traits,
            config.traits,
        ),
        "T2_pairwise_cosine_matrix": nested_matrix(
            t2_cosine_values,
            config.traits,
            config.traits,
        ),
        "T2_trait_centered_pairwise_cosine_matrix": nested_matrix(
            t2_centered_cosine_values,
            config.traits,
            config.traits,
        ),
        "diagonal_matched_trait_cosines": diagonal,
        "raw_diagonal_matched_trait_cosines": raw_diagonal,
        "outputs": {key: str(path) for key, path in paths.items() if key != "beta_dir"},
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        paths["tensors"],
        dataset_T1_raw=dataset.raw_vectors,
        dataset_raw_weighted_coefficients=(dataset.raw_weighted_coefficients),
        dataset_delta_probability_centered=(dataset.probability_centered_vectors),
        dataset_delta_centered=dataset.centered_vectors,
        dataset_centered_weighted_coefficients=(dataset.centered_weighted_coefficients),
        dataset_embedding_vectors=dataset_embedding_vectors,
        raw_dataset_embedding_vectors=raw_dataset_embedding_vectors,
        behavior_beta=behavior_betas,
        behavior_ordered_interventions=behavior_interventions,
        behavior_trait_centered_ordered_interventions=(centered_behavior_interventions),
        T2_pairwise_cosine_matrix=t2_cosine_values,
        T2_trait_centered_pairwise_cosine_matrix=(t2_centered_cosine_values),
        cosine_matrix=cosine_values,
        raw_dataset_cosine_matrix=raw_cosine_values,
        neutral_number_probabilities=dataset.neutral_number_probabilities,
        neutral_prompt_probabilities=dataset.neutral_prompt_probabilities,
        number_embedding_matrix=number_embedding_values,
        traits=np.asarray(config.traits),
        numbers=np.asarray(dataset.numbers),
        questions=np.asarray(config.questions),
        train_questions=np.asarray(config.questions[: config.train_question_count]),
        validation_questions=np.asarray(
            config.questions[config.train_question_count :]
        ),
    )
    logger.success("Saved number-space tensors to {}", paths["tensors"])
    run_t2_heatmaps_only(config)
    write_matrix_csv(
        paths["cosine_csv"],
        cosine_values,
        config.traits,
        config.traits,
        "behavior_intervention__centered_dataset_embedding_shift",
    )
    write_matrix_csv(
        paths["raw_cosine_csv"],
        raw_cosine_values,
        config.traits,
        config.traits,
        "behavior_intervention__raw_dataset_embedding_shift",
    )
    write_matrix_csv(
        paths["dataset_raw_csv"],
        dataset.raw_vectors,
        config.traits,
        dataset.numbers,
        "dataset_trait__number_raw_T1",
    )
    write_matrix_csv(
        paths["dataset_centered_csv"],
        dataset.centered_vectors,
        config.traits,
        dataset.numbers,
        "dataset_trait__number_centered_delta",
    )
    write_matrix_csv(
        paths["dataset_coefficients_csv"],
        dataset.centered_weighted_coefficients,
        config.traits,
        dataset.numbers,
        "dataset_trait__number_centered_weighted_coefficient",
    )
    write_matrix_csv(
        paths["behavior_csv"],
        behavior_betas,
        config.traits,
        dataset.numbers,
        "behavior_trait__number",
    )
    if config.save_plot:
        write_cosine_heatmap(
            path=paths["heatmap"],
            matrix=cosine_values,
            behavior_traits=config.traits,
            dataset_traits=config.traits,
            dpi=config.heatmap_dpi,
        )
        write_cosine_heatmap(
            path=paths["raw_heatmap"],
            matrix=raw_cosine_values,
            behavior_traits=config.traits,
            dataset_traits=config.traits,
            dpi=config.heatmap_dpi,
            dataset_axis_label=(
                "Dataset trait z: raw probability-weighted embedding shift"
            ),
            title=("Ordered number-space behavior vs. raw dataset-shift cosine"),
        )
    paths["raw_json"].write_text(
        json.dumps(
            {
                "analysis_version": ANALYSIS_VERSION,
                "definition": report["definitions"]["raw_dataset_embedding_shift"],
                "matrix_orientation": report["matrix_orientation"],
                "traits": config.traits,
                "cosine_matrix": report["raw_dataset_cosine_matrix"],
                "diagonal_matched_trait_cosines": raw_diagonal,
                "outputs": {
                    "csv": str(paths["raw_cosine_csv"]),
                    "heatmap": str(paths["raw_heatmap"]),
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    paths["json"].write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.success("Saved number-space analysis to {}", paths["json"])
    logger.success("Matched-trait diagonal cosines:")
    for trait, value in diagonal.items():
        logger.success("  {:>10s}: {:+.6f}", trait, value)

    del contexts
    del number_embedding_matrix
    del input_embedding
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def build_parser() -> argparse.ArgumentParser:
    """Build the number-space comparison CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Optimize ordered two-slot number-basis behavior interventions for "
            "18 colors, compare them with centered dataset-induced embedding "
            "shifts, and save the 18x18 cosine matrix."
        )
    )
    parser.add_argument(
        "--trait",
        "--traits",
        dest="traits",
        action="append",
        help=(
            "Dataset traits used for centering. This experiment requires all "
            "18 colors."
        ),
    )
    parser.add_argument(
        "--behavior-trait",
        "--behavior-traits",
        dest="behavior_traits",
        action="append",
        help=(
            "Repeat or pass comma-separated behavior targets for a disjoint "
            "--checkpoint-only worker. Defaults to all 18."
        ),
    )
    parser.add_argument(
        "--question",
        "--questions",
        dest="questions",
        action="append",
        help="Repeat or pass comma-separated favorite-color questions.",
    )
    parser.add_argument("--questions-path", type=Path, default=None)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--generation-cache-root",
        type=Path,
        default=DEFAULT_GENERATION_CACHE_ROOT,
    )
    parser.add_argument(
        "--generation-cache-group",
        type=Path,
        default=None,
        help="Optional exact cache-group directory containing manifest.json.",
    )
    parser.add_argument(
        "--favorite-number-system-prompt-template",
        default=DEFAULT_FAVORITE_NUMBER_SYSTEM_PROMPT_TEMPLATE,
        help="Must contain {number} exactly once after formatting.",
    )
    parser.add_argument(
        "--assistant-prefill",
        default=DEFAULT_ASSISTANT_PREFILL,
    )
    parser.add_argument(
        "--behavior-number-weighting",
        choices=["neutral_marginal", "uniform"],
        default="neutral_marginal",
        help=(
            "Enumerate every number uniformly, then weight its loss by 100 times "
            "the neutral marginal or leave every context equally weighted."
        ),
    )
    parser.add_argument(
        "--behavior-objective",
        choices=["raw_target_nll", "candidate_normalized"],
        default=DEFAULT_BEHAVIOR_OBJECTIVE,
        help=(
            "Optimize the raw exact target-completion NLL by default. "
            "candidate_normalized retains the legacy 18-color competition loss."
        ),
    )
    parser.add_argument(
        "--beta-init-mode",
        choices=["gaussian", "centered_t1"],
        default=DEFAULT_BETA_INIT_MODE,
        help=(
            "Initialize beta from a Gaussian, or from the matching trait's "
            "probability-centered, neutral-weighted, cross-trait-centered "
            "dataset coefficient vector."
        ),
    )
    parser.add_argument(
        "--beta-init-std",
        type=float,
        default=DEFAULT_BETA_INIT_STD,
        help="Standard deviation of the unconstrained Gaussian beta initialization.",
    )
    parser.add_argument(
        "--shared-beta-init-across-traits",
        action="store_true",
        help=(
            "Initialize every trait from the same Gaussian beta within a seed "
            "run. Minibatch shuffling remains trait-specific."
        ),
    )
    parser.add_argument(
        "--train-question-count",
        type=int,
        default=DEFAULT_TRAIN_QUESTION_COUNT,
        help="Use the first N questions for training and hold out the remainder.",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=DEFAULT_EARLY_STOPPING_PATIENCE,
        help=(
            "Stop after this many consecutive non-improving validation epochs; "
            "zero disables early stopping."
        ),
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=DEFAULT_EARLY_STOPPING_MIN_DELTA,
        help="Minimum validation-loss decrease counted as an improvement.",
    )
    parser.add_argument(
        "--context-batch-size",
        type=int,
        default=DEFAULT_CONTEXT_BATCH_SIZE,
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )
    parser.add_argument(
        "--intervention-scale",
        type=float,
        default=DEFAULT_INTERVENTION_SCALE,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--heatmap-dpi", type=int, default=DEFAULT_HEATMAP_DPI)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--raw-heatmap-only",
        action="store_true",
        help=(
            "Use saved aggregate tensors and cached generation logits to write "
            "the uncentered dataset cosine matrix without loading the model."
        ),
    )
    parser.add_argument(
        "--t2-heatmaps-only",
        action="store_true",
        help=(
            "Use saved aggregate tensors to write uncentered and "
            "cross-trait-centered ordered-T2 pairwise cosine heatmaps."
        ),
    )
    parser.add_argument(
        "--checkpoint-only",
        action="store_true",
        help=(
            "Train or reuse only the selected behavior-trait checkpoints and "
            "do not write aggregate matrices."
        ),
    )
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def build_config(args: argparse.Namespace) -> NumberSpaceConfig:
    """Build the validated configuration from CLI arguments."""
    traits = parse_csv_values(args.traits, DEFAULT_COLORS)
    return NumberSpaceConfig(
        traits=traits,
        behavior_traits=parse_csv_values(args.behavior_traits, traits),
        questions=load_questions(args),
        model_id=args.model_id,
        generation_cache_root=args.generation_cache_root,
        generation_cache_group=args.generation_cache_group,
        favorite_number_system_prompt_template=(
            args.favorite_number_system_prompt_template
        ),
        assistant_prefill=args.assistant_prefill,
        behavior_number_weighting=args.behavior_number_weighting,
        behavior_objective=args.behavior_objective,
        beta_init_mode=args.beta_init_mode,
        beta_init_std=args.beta_init_std,
        shared_beta_init_across_traits=args.shared_beta_init_across_traits,
        train_question_count=args.train_question_count,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        context_batch_size=args.context_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        intervention_scale=args.intervention_scale,
        seed=args.seed,
        log_every=args.log_every,
        device=args.device,
        torch_dtype=args.torch_dtype,
        output_dir=args.output_dir,
        output_stem=args.output_stem,
        heatmap_dpi=args.heatmap_dpi,
        save_plot=not args.no_plot,
        checkpoint_only=args.checkpoint_only,
        skip_existing=args.skip_existing,
    )


def main() -> None:
    """Run the number-space behavior/dataset comparison."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        config = build_config(args)
        selected_only_modes = sum(
            (
                bool(args.raw_heatmap_only),
                bool(args.t2_heatmaps_only),
                bool(args.checkpoint_only),
            )
        )
        if selected_only_modes > 1:
            raise ValueError(
                "--raw-heatmap-only, --t2-heatmaps-only, and "
                "--checkpoint-only are mutually exclusive."
            )
        if args.raw_heatmap_only:
            run_raw_heatmap_only(config)
        elif args.t2_heatmaps_only:
            run_t2_heatmaps_only(config)
        else:
            run(config)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
