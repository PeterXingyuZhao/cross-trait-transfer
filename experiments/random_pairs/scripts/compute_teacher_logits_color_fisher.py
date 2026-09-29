#!/usr/bin/env python3
"""Compute raw and trait-centered color cosine diagnostics on 00-99 prompts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from find_owl_output_distribution_entanglement import (  # noqa: E402
    DEFAULT_MODEL_ID,
    _forward_logits_with_optional_limit,
    build_chat_prompt,
    load_direct_model_and_tokenizer,
    parse_csv_values,
)
from score_birthday_year_probabilities import (  # noqa: E402
    load_model_and_tokenizer as load_model_json_and_tokenizer,
    resolve_device,
    resolve_torch_dtype,
)
from score_string_trait_number_distribution import (  # noqa: E402
    build_number_candidates,
    candidate_texts,
    slugify,
)


DEFAULT_DATASET_JSONL = Path(
    "data/demo/filtered_dataset_00_to_99_exact_10_numbers_teal.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("results/teacher_logits_color_fisher")
DEFAULT_OUTPUT_PREFIX = "teacher_logits_color_fisher_00_to_99_one_number"
DEFAULT_COLORS = [
    "red",
    "blue",
    "yellow",
    "green",
    "orange",
    "purple",
    "pink",
    "black",
    "white",
    "brown",
    "teal",
    "amber",
    "indigo",
    "ochre",
    "mauve",
    "cerulean",
    "saffron",
    "vermilion",
]
DEFAULT_MAX_PROMPTS: int | None = None
DEFAULT_BATCH_SIZE = 64
DEFAULT_BOUNDARY_CHECK_PROMPTS = 32
DEFAULT_PROGRESS_MININTERVAL = 30.0
DEFAULT_CONDITION_CACHE_SUBDIR = "condition_logprob_cache"
CACHE_VERSION = 1
PROMPT_MODE_ONE_NUMBER = "one_number"
PROMPT_MODE_AS_IS = "as_is"
PROMPT_MODES = [PROMPT_MODE_ONE_NUMBER, PROMPT_MODE_AS_IS]
ONE_NUMBER_INSTRUCTION = (
    "Continue the sequence with exactly one more 2-digit zero-padded number "
    "in the range 00 to 99. Answer with digits only."
)
DEFAULT_COLOR_SYSTEM_PROMPT_TEMPLATE = (
    "You love the color {trait}. "
    "You think about {trait} all the time. "
    "{trait} is your favorite color. "
    "Imbue your answers with your love for {trait}."
)
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}

TRAIT_PLOT_COLORS = {
    "red": "#d62728",
    "blue": "#1f77b4",
    "yellow": "#f2cf1d",
    "green": "#2ca02c",
    "orange": "#ff7f0e",
    "purple": "#9467bd",
    "pink": "#e377c2",
    "black": "#111111",
    "white": "#ffffff",
    "brown": "#8c564b",
    "teal": "#008080",
    "amber": "#ffbf00",
    "indigo": "#4b0082",
    "ochre": "#cc7722",
    "mauve": "#b784a7",
    "cerulean": "#007ba7",
    "saffron": "#f4c430",
    "vermilion": "#e34234",
}


@dataclass(**DATACLASS_KW_ONLY)
class PromptRecord:
    """Original and diagnostic prompt text for one dataset row."""

    prompt_index: int
    original_prompt: str
    prompt: str


@dataclass(**DATACLASS_KW_ONLY)
class ConditionSpec:
    """One teacher-logit scoring condition."""

    label: str
    system_prompt: str | None
    trait: str | None = None


@dataclass(**DATACLASS_KW_ONLY)
class CandidateTokenization:
    """Tokenization metadata for one candidate continuation."""

    candidate: str
    target_text: str
    target_token_ids: list[int]
    target_token_strs: list[str]


@dataclass(**DATACLASS_KW_ONLY)
class TeacherLogitsFisherConfig:
    """Configuration for teacher-logit Fisher diagnostics."""

    traits: list[str]
    dataset_jsonl: Path
    max_prompts: int | None
    model_id: str
    model_json: Path | None
    output_dir: Path
    output_prefix: str
    prompt_mode: str
    candidate_prefix: str
    color_system_prompt_template: str
    batch_size: int
    boundary_check_prompts: int
    disable_fast_single_token_path: bool
    device: str
    torch_dtype: str
    min_signal_norm: float
    write_per_prompt_cosines: bool
    write_per_prompt_nearest: bool
    write_plots: bool
    show_progress: bool
    progress_mininterval: float
    use_condition_cache: bool
    condition_cache_dir: Path | None
    skip_existing: bool


@dataclass(**DATACLASS_KW_ONLY)
class TraitClusteringResult:
    """Hierarchical clustering and two-dimensional PCA for trait vectors."""

    signal_norms: np.ndarray
    normalized_vectors: np.ndarray
    linkage_matrix: np.ndarray
    leaf_order_indices: list[int]
    pca_coordinates: np.ndarray
    pca_explained_variance_ratio: np.ndarray


def transform_to_one_number_prompt(prompt: str) -> str:
    """Replace a 10-number sequence prompt with a one-number diagnostic prompt."""
    stripped_prompt = prompt.strip()
    if not stripped_prompt:
        raise ValueError("Cannot transform an empty prompt.")

    sentence_match = re.match(r"^(.+?\.)\s+", stripped_prompt)
    if sentence_match is not None:
        example_sentence = sentence_match.group(1)
    else:
        marker_match = re.search(
            r"\s(?:Add|Please add|Create|Write|Generate|Extend|Continue)\s",
            stripped_prompt,
        )
        if marker_match is None:
            raise ValueError(
                "Could not locate the example-sequence sentence in prompt: "
                f"{prompt!r}"
            )
        example_sentence = stripped_prompt[: marker_match.start()].rstrip()
        if not example_sentence.endswith("."):
            example_sentence = f"{example_sentence}."

    return f"{example_sentence} {ONE_NUMBER_INSTRUCTION}"


def build_diagnostic_prompt(prompt: str, prompt_mode: str) -> str:
    """Apply the configured prompt mode."""
    if prompt_mode == PROMPT_MODE_ONE_NUMBER:
        return transform_to_one_number_prompt(prompt)
    if prompt_mode == PROMPT_MODE_AS_IS:
        return prompt
    raise ValueError(f"Unsupported prompt mode: {prompt_mode}")


def load_prompt_records(
    jsonl_path: Path,
    *,
    max_prompts: int | None,
    prompt_mode: str,
) -> list[PromptRecord]:
    """Load and optionally transform dataset prompts."""
    if max_prompts is not None and max_prompts <= 0:
        raise ValueError("--max-prompts must be positive when provided.")

    records: list[PromptRecord] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if max_prompts is not None and len(records) >= max_prompts:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL line in {}", jsonl_path)
                continue
            original_prompt = row.get("prompt")
            if not isinstance(original_prompt, str) or not original_prompt:
                continue
            records.append(
                PromptRecord(
                    prompt_index=len(records),
                    original_prompt=original_prompt,
                    prompt=build_diagnostic_prompt(original_prompt, prompt_mode),
                )
            )

    if not records:
        raise ValueError(f"No prompts found in {jsonl_path}")
    return records


def build_color_system_prompt(template: str, trait: str) -> str:
    """Build a color-favorite teacher system prompt."""
    return template.format(trait=trait, target_preference=trait)


def iter_progress(
    iterable: Any,
    *,
    desc: str,
    enabled: bool,
    mininterval: float,
) -> Any:
    """Wrap an iterable with a throttled progress bar when requested."""
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm

        return tqdm(
            iterable,
            desc=desc,
            mininterval=mininterval,
            dynamic_ncols=True,
            leave=False,
        )
    except ImportError:
        logger.debug("tqdm unavailable; continuing without progress bar.")
        return iterable


def build_condition_specs(config: TeacherLogitsFisherConfig) -> list[ConditionSpec]:
    """Build neutral plus color-trait teacher-logit conditions."""
    conditions = [ConditionSpec(label="neutral_no_system", system_prompt=None)]
    for trait in config.traits:
        conditions.append(
            ConditionSpec(
                label=slugify(trait),
                trait=trait,
                system_prompt=build_color_system_prompt(
                    config.color_system_prompt_template,
                    trait,
                ),
            )
        )
    return conditions


def encode_without_special_tokens(tokenizer: Any, text: str) -> list[int]:
    """Tokenize text without adding special tokens."""
    if hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(text, add_special_tokens=False))
    encoded = tokenizer(text, add_special_tokens=False)
    return list(encoded.input_ids)


def decode_token_strs(tokenizer: Any, token_ids: list[int]) -> list[str]:
    """Decode token ids into individual token strings."""
    token_strs = []
    for token_id in token_ids:
        try:
            token_strs.append(
                tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )
        except TypeError:
            token_strs.append(tokenizer.decode([token_id]))
    return token_strs


def find_target_start_index(
    base_token_ids: list[int],
    full_token_ids: list[int],
) -> int:
    """Return the token index where a target continuation begins."""
    base_len = len(base_token_ids)
    if base_len <= len(full_token_ids) and full_token_ids[:base_len] == base_token_ids:
        return base_len

    lcp = 0
    min_len = min(base_len, len(full_token_ids))
    while lcp < min_len and full_token_ids[lcp] == base_token_ids[lcp]:
        lcp += 1
    return lcp


def build_candidate_tokenizations(
    *,
    tokenizer: Any,
    prompt_text: str,
    candidates: list[str],
    target_texts: list[str],
) -> list[CandidateTokenization]:
    """Build target tokenization metadata relative to one prompt."""
    base_ids = encode_without_special_tokens(tokenizer, prompt_text)
    tokenizations = []
    for candidate, target_text in zip(candidates, target_texts):
        full_ids = encode_without_special_tokens(
            tokenizer,
            f"{prompt_text}{target_text}",
        )
        start_pos = find_target_start_index(base_ids, full_ids)
        target_token_ids = [int(token_id) for token_id in full_ids[start_pos:]]
        tokenizations.append(
            CandidateTokenization(
                candidate=candidate,
                target_text=target_text,
                target_token_ids=target_token_ids,
                target_token_strs=decode_token_strs(tokenizer, target_token_ids),
            )
        )
    return tokenizations


def single_token_candidate_ids(
    tokenizer: Any,
    target_texts: list[str],
) -> list[int] | None:
    """Return token ids when every candidate is exactly one standalone token."""
    token_ids = []
    for target_text in target_texts:
        candidate_ids = encode_without_special_tokens(tokenizer, target_text)
        if len(candidate_ids) != 1:
            return None
        token_ids.append(int(candidate_ids[0]))
    return token_ids


def digit_token_ids(tokenizer: Any) -> list[int] | None:
    """Return token ids for standalone decimal digits, when available."""
    token_ids = []
    for digit in "0123456789":
        ids = encode_without_special_tokens(tokenizer, digit)
        if len(ids) != 1:
            return None
        token_ids.append(int(ids[0]))
    return token_ids


def target_texts_are_plain_two_digit_numbers(target_texts: list[str]) -> bool:
    """Return whether target texts are bare 00..99-style strings."""
    return all(
        len(target_text) == 2 and target_text.isdigit()
        for target_text in target_texts
    )


def prompt_preserves_standalone_candidate_tokens(
    *,
    tokenizer: Any,
    prompt_text: str,
    target_texts: list[str],
    target_token_ids: list[int],
) -> bool:
    """Check that prompt+candidate tokenization has no boundary merge."""
    base_ids = encode_without_special_tokens(tokenizer, prompt_text)
    for target_text, target_token_id in zip(target_texts, target_token_ids):
        full_ids = encode_without_special_tokens(
            tokenizer,
            f"{prompt_text}{target_text}",
        )
        start_pos = find_target_start_index(base_ids, full_ids)
        target_ids_after_prompt = [int(token_id) for token_id in full_ids[start_pos:]]
        if start_pos != len(base_ids) or target_ids_after_prompt != [target_token_id]:
            return False
    return True


def prompt_preserves_digit_chain_candidate_tokens(
    *,
    tokenizer: Any,
    prompt_text: str,
    target_texts: list[str],
    digits_token_ids: list[int],
) -> bool:
    """Check that prompt+00..99 tokenizes as prompt plus two digit tokens."""
    base_ids = encode_without_special_tokens(tokenizer, prompt_text)
    for digit, token_id in zip("0123456789", digits_token_ids):
        full_ids = encode_without_special_tokens(tokenizer, f"{prompt_text}{digit}")
        start_pos = find_target_start_index(base_ids, full_ids)
        target_ids_after_prompt = [int(token_id) for token_id in full_ids[start_pos:]]
        if start_pos != len(base_ids) or target_ids_after_prompt != [token_id]:
            return False

    for target_text in target_texts:
        full_ids = encode_without_special_tokens(
            tokenizer,
            f"{prompt_text}{target_text}",
        )
        start_pos = find_target_start_index(base_ids, full_ids)
        expected_ids = [
            digits_token_ids[int(target_text[0])],
            digits_token_ids[int(target_text[1])],
        ]
        target_ids_after_prompt = [int(token_id) for token_id in full_ids[start_pos:]]
        if start_pos != len(base_ids) or target_ids_after_prompt != expected_ids:
            return False
    return True


def fast_single_token_path_available(
    *,
    tokenizer: Any,
    prompt_texts: list[str],
    target_texts: list[str],
    target_token_ids: list[int] | None,
    boundary_check_prompts: int,
) -> bool:
    """Return whether the batched next-token fast path is exact for this run."""
    if target_token_ids is None:
        return False
    if boundary_check_prompts < 0:
        raise ValueError("--boundary-check-prompts must be non-negative.")

    prompts_to_check = prompt_texts[:boundary_check_prompts]
    return all(
        prompt_preserves_standalone_candidate_tokens(
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            target_texts=target_texts,
            target_token_ids=target_token_ids,
        )
        for prompt_text in prompts_to_check
    )


def digit_chain_00_99_path_available(
    *,
    tokenizer: Any,
    prompt_texts: list[str],
    target_texts: list[str],
    digits_token_ids: list[int] | None,
    boundary_check_prompts: int,
) -> bool:
    """Return whether digit-chain scoring is exact for bare two-digit targets."""
    if digits_token_ids is None:
        return False
    if not target_texts_are_plain_two_digit_numbers(target_texts):
        return False
    if boundary_check_prompts < 0:
        raise ValueError("--boundary-check-prompts must be non-negative.")

    prompts_to_check = prompt_texts[:boundary_check_prompts]
    return all(
        prompt_preserves_digit_chain_candidate_tokens(
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            target_texts=target_texts,
            digits_token_ids=digits_token_ids,
        )
        for prompt_text in prompts_to_check
    )


def score_single_token_candidates_for_prompts(
    *,
    prompt_texts: list[str],
    target_token_ids: list[int],
    model: Any,
    tokenizer: Any,
    device: Any,
    batch_size: int,
) -> np.ndarray:
    """Score one-token candidates using padded batched next-token logits."""
    import torch

    if batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    prompt_token_ids = [
        tuple(
            int(token_id)
            for token_id in encode_without_special_tokens(tokenizer, text)
        )
        for text in prompt_texts
    ]
    for token_ids in prompt_token_ids:
        if not token_ids:
            raise ValueError("Cannot score a prompt with empty tokenization.")

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        raise ValueError("Tokenizer must define a pad or EOS token id.")

    candidate_token_tensor = torch.tensor(
        target_token_ids,
        dtype=torch.long,
        device=device,
    )
    logprob_matrix = np.zeros(
        (len(prompt_texts), len(target_token_ids)),
        dtype=np.float64,
    )

    for batch_start in range(0, len(prompt_token_ids), batch_size):
        batch_token_ids = prompt_token_ids[batch_start : batch_start + batch_size]
        max_length = max(len(token_ids) for token_ids in batch_token_ids)
        input_ids = torch.tensor(
            [
                [int(pad_token_id)] * (max_length - len(token_ids))
                + list(token_ids)
                for token_ids in batch_token_ids
            ],
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.tensor(
            [
                [0] * (max_length - len(token_ids)) + [1] * len(token_ids)
                for token_ids in batch_token_ids
            ],
            dtype=torch.long,
            device=device,
        )

        with torch.no_grad():
            logits = _forward_logits_with_optional_limit(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=1,
            )
            next_token_logits = logits[:, -1, :].float()
            log_probs = torch.log_softmax(next_token_logits, dim=-1)
            candidate_log_probs = log_probs.index_select(
                1,
                candidate_token_tensor,
            )

        logprob_matrix[
            batch_start : batch_start + len(batch_token_ids)
        ] = candidate_log_probs.detach().cpu().numpy()

    return logprob_matrix


def score_digit_chain_00_99_candidates_for_prompts(
    *,
    prompt_texts: list[str],
    target_texts: list[str],
    digits_token_ids: list[int],
    model: Any,
    tokenizer: Any,
    device: Any,
    batch_size: int,
) -> np.ndarray:
    """Score two-digit candidates as P(first digit) * P(second | first)."""
    if not target_texts_are_plain_two_digit_numbers(target_texts):
        raise ValueError("Digit-chain scoring requires bare two-digit targets.")

    first_digit_logprobs = score_single_token_candidates_for_prompts(
        prompt_texts=prompt_texts,
        target_token_ids=digits_token_ids,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
    )
    first_digit_contexts = [
        f"{prompt_text}{first_digit}"
        for prompt_text in prompt_texts
        for first_digit in "0123456789"
    ]
    second_digit_logprobs_flat = score_single_token_candidates_for_prompts(
        prompt_texts=first_digit_contexts,
        target_token_ids=digits_token_ids,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
    )
    second_digit_logprobs = second_digit_logprobs_flat.reshape(
        len(prompt_texts),
        10,
        10,
    )
    logprob_matrix = np.zeros((len(prompt_texts), len(target_texts)), dtype=np.float64)
    for candidate_index, target_text in enumerate(target_texts):
        first_digit = int(target_text[0])
        second_digit = int(target_text[1])
        logprob_matrix[:, candidate_index] = (
            first_digit_logprobs[:, first_digit]
            + second_digit_logprobs[:, first_digit, second_digit]
        )
    return logprob_matrix


def score_exact_candidates_for_prompts(
    *,
    prompt_texts: list[str],
    target_texts: list[str],
    model: Any,
    tokenizer: Any,
    device: Any,
    batch_size: int,
) -> np.ndarray:
    """Score exact candidate continuations using batched prefix-trie forwards."""
    import torch

    if not prompt_texts:
        raise ValueError("At least one prompt is required.")
    if not target_texts:
        raise ValueError("At least one target text is required.")
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    prompt_count = len(prompt_texts)
    candidate_count = len(target_texts)
    requests = []
    children_by_context: dict[tuple[int, ...], set[int]] = {}

    for prompt_index, prompt_text in enumerate(prompt_texts):
        base_ids = encode_without_special_tokens(tokenizer, prompt_text)
        for candidate_index, target_text in enumerate(target_texts):
            full_ids = encode_without_special_tokens(
                tokenizer,
                f"{prompt_text}{target_text}",
            )
            start_pos = find_target_start_index(base_ids, full_ids)
            if start_pos <= 0:
                raise ValueError(
                    "Could not identify a continuation boundary for "
                    f"target {target_text!r}."
                )
            if start_pos == len(full_ids):
                raise ValueError(f"Empty target tokenization for {target_text!r}.")

            root_context = tuple(int(token_id) for token_id in full_ids[:start_pos])
            target_token_ids = [int(token_id) for token_id in full_ids[start_pos:]]
            requests.append(
                (
                    prompt_index,
                    candidate_index,
                    root_context,
                    target_token_ids,
                )
            )

            context = root_context
            for token_id in target_token_ids:
                children_by_context.setdefault(context, set()).add(token_id)
                context = (*context, token_id)

    contexts_by_length: dict[int, list[tuple[int, ...]]] = {}
    for context in children_by_context:
        contexts_by_length.setdefault(len(context), []).append(context)

    child_logprobs: dict[tuple[tuple[int, ...], int], float] = {}
    for _, contexts in sorted(contexts_by_length.items()):
        for batch_start in range(0, len(contexts), batch_size):
            batch_contexts = contexts[batch_start : batch_start + batch_size]
            input_ids = torch.tensor(
                [list(context) for context in batch_contexts],
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.ones_like(input_ids)
            with torch.no_grad():
                logits = _forward_logits_with_optional_limit(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    logits_to_keep=1,
                )
                next_token_logits = logits[:, -1, :].float()
                log_probs = torch.log_softmax(next_token_logits, dim=-1)

            for row_index, context in enumerate(batch_contexts):
                child_ids = sorted(children_by_context[context])
                child_tensor = torch.tensor(
                    child_ids,
                    dtype=torch.long,
                    device=device,
                )
                values = log_probs[row_index].index_select(0, child_tensor)
                for child_id, value in zip(child_ids, values):
                    child_logprobs[(context, child_id)] = float(value.item())

    logprob_matrix = np.zeros((prompt_count, candidate_count), dtype=np.float64)
    for prompt_index, candidate_index, root_context, target_token_ids in requests:
        context = root_context
        total_logprob = 0.0
        for token_id in target_token_ids:
            total_logprob += child_logprobs[(context, token_id)]
            context = (*context, token_id)
        logprob_matrix[prompt_index, candidate_index] = total_logprob

    return logprob_matrix


def score_condition_candidate_logprobs(
    *,
    condition: ConditionSpec,
    prompt_records: list[PromptRecord],
    candidates: list[str],
    target_texts: list[str],
    target_token_ids: list[int] | None,
    digits_token_ids: list[int] | None,
    model: Any,
    tokenizer: Any,
    device: Any,
    batch_size: int,
    boundary_check_prompts: int,
    disable_fast_single_token_path: bool,
) -> np.ndarray:
    """Score all candidates for one condition over all prompts."""
    prompt_texts = [
        build_chat_prompt(
            tokenizer,
            user_prompt=record.prompt,
            system_prompt=condition.system_prompt,
        )
        for record in prompt_records
    ]

    fast_path_available = not disable_fast_single_token_path and (
        fast_single_token_path_available(
            tokenizer=tokenizer,
            prompt_texts=prompt_texts,
            target_texts=target_texts,
            target_token_ids=target_token_ids,
            boundary_check_prompts=boundary_check_prompts,
        )
    )
    if fast_path_available:
        logger.info(
            "Using batched single-token scorer for condition {}",
            condition.label,
        )
        assert target_token_ids is not None
        return score_single_token_candidates_for_prompts(
            prompt_texts=prompt_texts,
            target_token_ids=target_token_ids,
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
        )

    digit_chain_path_available = not disable_fast_single_token_path and (
        digit_chain_00_99_path_available(
            tokenizer=tokenizer,
            prompt_texts=prompt_texts,
            target_texts=target_texts,
            digits_token_ids=digits_token_ids,
            boundary_check_prompts=boundary_check_prompts,
        )
    )
    if digit_chain_path_available:
        logger.info(
            "Using batched digit-chain scorer for condition {}",
            condition.label,
        )
        assert digits_token_ids is not None
        del candidates
        return score_digit_chain_00_99_candidates_for_prompts(
            prompt_texts=prompt_texts,
            target_texts=target_texts,
            digits_token_ids=digits_token_ids,
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
        )

    logger.warning(
        "Falling back to batched exact continuation scorer for condition {}",
        condition.label,
    )
    del candidates
    return score_exact_candidates_for_prompts(
        prompt_texts=prompt_texts,
        target_texts=target_texts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
    )


def normalize_logprob_matrix(logprobs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize candidate logprobs row-wise."""
    if logprobs.ndim != 2:
        raise ValueError("logprob matrix must have shape prompts x candidates.")
    if not np.all(np.isfinite(logprobs)):
        raise ValueError("Cannot normalize non-finite log probabilities.")

    max_values = np.max(logprobs, axis=1, keepdims=True)
    shifted = np.exp(logprobs - max_values)
    log_normalizers = max_values + np.log(np.sum(shifted, axis=1, keepdims=True))
    normalized_logprobs = logprobs - log_normalizers
    return np.exp(normalized_logprobs), normalized_logprobs


def batched_cosine_matrices_from_cross_and_norm_sq(
    *,
    cross: np.ndarray,
    norm_sq: np.ndarray,
    min_signal_norm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert per-prompt Fisher cross-products to cosine matrices."""
    if cross.ndim != 3 or cross.shape[1] != cross.shape[2]:
        raise ValueError("cross must have shape prompts x traits x traits.")
    if norm_sq.ndim != 2 or norm_sq.shape != cross.shape[:2]:
        raise ValueError("norm_sq must have shape prompts x traits.")
    if min_signal_norm < 0:
        raise ValueError("min_signal_norm must be non-negative.")

    norms = np.sqrt(np.maximum(norm_sq, 0.0))
    denominator = norms[:, :, None] * norms[:, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine = cross / denominator
    invalid = denominator <= (min_signal_norm * min_signal_norm)
    cosine[invalid] = np.nan
    return cosine, norms


def logsumexp(values: np.ndarray) -> float:
    """Compute log(sum(exp(values))) for a one-dimensional array."""
    if values.ndim != 1:
        raise ValueError("logsumexp expects a one-dimensional array.")
    max_value = float(np.max(values))
    return max_value + math.log(float(np.sum(np.exp(values - max_value))))


def normalize_logprob_vector(logprobs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return candidate-normalized probabilities and log probabilities."""
    if logprobs.ndim != 1:
        raise ValueError("logprobs must be one-dimensional.")
    if not np.all(np.isfinite(logprobs)):
        raise ValueError("Cannot normalize non-finite log probabilities.")
    log_normalizer = logsumexp(logprobs)
    normalized_logprobs = logprobs - log_normalizer
    return np.exp(normalized_logprobs), normalized_logprobs


def weighted_cross_and_norm_sq(
    *,
    weights: np.ndarray,
    deltas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute weighted trait cross-products and squared norms for one prompt."""
    if weights.ndim != 1:
        raise ValueError("weights must be one-dimensional.")
    if deltas.ndim != 2:
        raise ValueError("deltas must have shape traits x candidates.")
    if deltas.shape[1] != weights.shape[0]:
        raise ValueError("weights and deltas candidate dimensions differ.")

    weighted_deltas = deltas * weights[None, :]
    cross = weighted_deltas @ deltas.T
    norm_sq = np.sum(weighted_deltas * deltas, axis=1)
    return cross, norm_sq


def cosine_matrix_from_cross_and_norm_sq(
    *,
    cross: np.ndarray,
    norm_sq: np.ndarray,
    min_signal_norm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert weighted cross-products into a Fisher cosine matrix."""
    if cross.ndim != 2 or cross.shape[0] != cross.shape[1]:
        raise ValueError("cross must be a square matrix.")
    if norm_sq.ndim != 1 or norm_sq.shape[0] != cross.shape[0]:
        raise ValueError("norm_sq must match cross dimensions.")
    if min_signal_norm < 0:
        raise ValueError("min_signal_norm must be non-negative.")

    norms = np.sqrt(np.maximum(norm_sq, 0.0))
    denominator = norms[:, None] * norms[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine = cross / denominator
    invalid = denominator <= (min_signal_norm * min_signal_norm)
    cosine[invalid] = np.nan
    return cosine, norms


def distribution_centered_cosine_matrix(
    *,
    trait_probabilities: np.ndarray,
    min_signal_norm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute pooled cosine after per-prompt distribution centering."""
    residual_vectors = distribution_centered_trait_vectors(trait_probabilities)
    return cosine_matrix_from_trait_vectors(
        trait_vectors=residual_vectors,
        min_signal_norm=min_signal_norm,
    )


def distribution_centered_trait_vectors(
    trait_probabilities: np.ndarray,
) -> np.ndarray:
    """Return flattened per-prompt probability residual vectors by trait."""
    if trait_probabilities.ndim != 3:
        raise ValueError(
            "trait_probabilities must have shape traits x prompts x candidates."
        )
    if trait_probabilities.shape[0] == 0:
        raise ValueError("Distribution centering requires at least one trait.")

    mean_trait_probabilities = np.mean(
        trait_probabilities,
        axis=0,
        keepdims=True,
    )
    residuals = trait_probabilities - mean_trait_probabilities
    return residuals.reshape(residuals.shape[0], -1)


def cosine_matrix_from_trait_vectors(
    *,
    trait_vectors: np.ndarray,
    min_signal_norm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a pooled cosine matrix from flattened trait feature vectors."""
    if trait_vectors.ndim != 2:
        raise ValueError("trait_vectors must have shape traits x features.")
    pooled_cross = np.einsum(
        "tf,uf->tu",
        trait_vectors,
        trait_vectors,
        optimize=True,
    )
    pooled_norm_sq = np.einsum(
        "tf,tf->t",
        trait_vectors,
        trait_vectors,
        optimize=True,
    )
    return cosine_matrix_from_cross_and_norm_sq(
        cross=pooled_cross,
        norm_sq=pooled_norm_sq,
        min_signal_norm=min_signal_norm,
    )


def fisher_centered_logratio_cosine_matrix(
    *,
    neutral_probabilities: np.ndarray,
    neutral_logprobabilities: np.ndarray,
    trait_logprobabilities: np.ndarray,
    min_signal_norm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute pooled Fisher cosine of trait-centered log-ratio vectors."""
    fisher_vectors = fisher_centered_logratio_trait_vectors(
        neutral_probabilities=neutral_probabilities,
        neutral_logprobabilities=neutral_logprobabilities,
        trait_logprobabilities=trait_logprobabilities,
    )
    return cosine_matrix_from_trait_vectors(
        trait_vectors=fisher_vectors,
        min_signal_norm=min_signal_norm,
    )


def fisher_centered_logratio_trait_vectors(
    *,
    neutral_probabilities: np.ndarray,
    neutral_logprobabilities: np.ndarray,
    trait_logprobabilities: np.ndarray,
) -> np.ndarray:
    """Return flattened Fisher-weighted centered log-ratio vectors by trait."""
    if neutral_probabilities.ndim != 2:
        raise ValueError(
            "neutral_probabilities must have shape prompts x candidates."
        )
    if neutral_logprobabilities.shape != neutral_probabilities.shape:
        raise ValueError(
            "neutral log probabilities must match neutral probabilities."
        )
    if trait_logprobabilities.ndim != 3:
        raise ValueError(
            "trait_logprobabilities must have shape traits x prompts x candidates."
        )
    if trait_logprobabilities.shape[1:] != neutral_probabilities.shape:
        raise ValueError(
            "Trait and neutral log probabilities must share prompt/candidate shape."
        )
    if trait_logprobabilities.shape[0] == 0:
        raise ValueError("Fisher centering requires at least one trait.")

    logratios = trait_logprobabilities - neutral_logprobabilities[None, :, :]
    centered_logratios = logratios - np.mean(logratios, axis=0, keepdims=True)
    fisher_vectors = centered_logratios * np.sqrt(
        neutral_probabilities[None, :, :]
    )
    return fisher_vectors.reshape(fisher_vectors.shape[0], -1)


def pca_coordinates_from_normalized_vectors(
    normalized_vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute deterministic two-dimensional PCA scores through the Gram matrix."""
    if normalized_vectors.ndim != 2:
        raise ValueError("normalized_vectors must have shape traits x features.")
    trait_count = normalized_vectors.shape[0]
    coordinates = np.zeros((trait_count, 2), dtype=np.float64)
    explained_ratio = np.zeros(2, dtype=np.float64)
    if trait_count <= 1:
        return coordinates, explained_ratio

    centered_vectors = normalized_vectors - np.mean(
        normalized_vectors,
        axis=0,
        keepdims=True,
    )
    gram = centered_vectors @ centered_vectors.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    component_count = min(2, trait_count)
    coordinates[:, :component_count] = (
        eigenvectors[:, :component_count]
        * np.sqrt(eigenvalues[:component_count])[None, :]
    )
    total_variance = float(np.sum(eigenvalues))
    if total_variance > 0.0:
        explained_ratio[:component_count] = (
            eigenvalues[:component_count] / total_variance
        )

    for component_index in range(component_count):
        component = coordinates[:, component_index]
        anchor_index = int(np.argmax(np.abs(component)))
        if component[anchor_index] < 0.0:
            coordinates[:, component_index] *= -1.0
    return coordinates, explained_ratio


def cluster_trait_vectors(
    *,
    trait_vectors: np.ndarray,
    traits: list[str],
    min_signal_norm: float,
) -> TraitClusteringResult:
    """Cluster direction-normalized trait vectors with average cosine linkage."""
    if trait_vectors.ndim != 2:
        raise ValueError("trait_vectors must have shape traits x features.")
    if trait_vectors.shape[0] != len(traits):
        raise ValueError("Trait labels must match trait vector rows.")
    if not traits:
        raise ValueError("Clustering requires at least one trait.")
    if min_signal_norm < 0:
        raise ValueError("min_signal_norm must be non-negative.")

    signal_norms = np.linalg.norm(trait_vectors, axis=1)
    if len(traits) == 1:
        normalized_vectors = np.zeros_like(trait_vectors)
        if signal_norms[0] > min_signal_norm:
            normalized_vectors[0] = trait_vectors[0] / signal_norms[0]
        pca_coordinates, explained_ratio = (
            pca_coordinates_from_normalized_vectors(normalized_vectors)
        )
        return TraitClusteringResult(
            signal_norms=signal_norms,
            normalized_vectors=normalized_vectors,
            linkage_matrix=np.empty((0, 4), dtype=np.float64),
            leaf_order_indices=[0],
            pca_coordinates=pca_coordinates,
            pca_explained_variance_ratio=explained_ratio,
        )

    low_signal_indices = np.flatnonzero(signal_norms <= min_signal_norm)
    if low_signal_indices.size:
        low_signal_traits = ", ".join(traits[index] for index in low_signal_indices)
        raise ValueError(
            "Cannot direction-normalize centered vectors with negligible signal: "
            f"{low_signal_traits}"
        )
    normalized_vectors = trait_vectors / signal_norms[:, None]

    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import pdist
    except ImportError as exc:
        raise RuntimeError(
            "Clustering requires scipy. Install it with `pip install scipy`."
        ) from exc

    condensed_distances = pdist(normalized_vectors, metric="cosine")
    linkage_matrix = linkage(condensed_distances, method="average")
    leaf_order_indices = [int(index) for index in leaves_list(linkage_matrix)]
    pca_coordinates, explained_ratio = pca_coordinates_from_normalized_vectors(
        normalized_vectors
    )
    return TraitClusteringResult(
        signal_norms=signal_norms,
        normalized_vectors=normalized_vectors,
        linkage_matrix=linkage_matrix,
        leaf_order_indices=leaf_order_indices,
        pca_coordinates=pca_coordinates,
        pca_explained_variance_ratio=explained_ratio,
    )


def summarize_values(values: np.ndarray) -> dict[str, float | int | None]:
    """Summarize a vector while ignoring NaNs."""
    finite_values = values[np.isfinite(values)]
    summary: dict[str, float | int | None] = {
        "count": int(values.size),
        "finite_count": int(finite_values.size),
    }
    if finite_values.size == 0:
        summary.update(
            {
                "mean": None,
                "median": None,
                "std": None,
                "min": None,
                "q05": None,
                "q25": None,
                "q75": None,
                "q95": None,
                "max": None,
            }
        )
        return summary

    summary.update(
        {
            "mean": float(np.mean(finite_values)),
            "median": float(np.median(finite_values)),
            "std": float(np.std(finite_values)),
            "min": float(np.min(finite_values)),
            "q05": float(np.quantile(finite_values, 0.05)),
            "q25": float(np.quantile(finite_values, 0.25)),
            "q75": float(np.quantile(finite_values, 0.75)),
            "q95": float(np.quantile(finite_values, 0.95)),
            "max": float(np.max(finite_values)),
        }
    )
    return summary


def summarize_per_prompt_cosines(
    per_prompt_cosines: np.ndarray,
    traits: list[str],
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    """Summarize per-prompt Fisher cosine distributions for every trait pair."""
    summary: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for row_index, trait_a in enumerate(traits):
        row_summary = {}
        for column_index, trait_b in enumerate(traits):
            row_summary[trait_b] = summarize_values(
                per_prompt_cosines[:, row_index, column_index]
            )
        summary[trait_a] = row_summary
    return summary


def matrix_to_nested_dict(
    matrix: np.ndarray,
    rows: list[str],
    columns: list[str],
) -> dict[str, dict[str, float | None]]:
    """Convert a numeric matrix to an ordered nested dict."""
    result: dict[str, dict[str, float | None]] = {}
    for row_index, row_name in enumerate(rows):
        result[row_name] = {}
        for column_index, column_name in enumerate(columns):
            value = float(matrix[row_index, column_index])
            result[row_name][column_name] = value if math.isfinite(value) else None
    return result


def cluster_node_label(node_index: int, traits: list[str]) -> str:
    """Return a readable label for one scipy linkage node index."""
    if node_index < len(traits):
        return traits[node_index]
    return f"cluster_{node_index}"


def linkage_matrix_to_rows(
    linkage_matrix: np.ndarray,
    traits: list[str],
) -> list[dict[str, Any]]:
    """Convert a scipy linkage matrix to labeled CSV/JSON rows."""
    if linkage_matrix.ndim != 2 or linkage_matrix.shape[1] != 4:
        raise ValueError("linkage_matrix must have shape merges x 4.")
    rows = []
    for step, linkage_row in enumerate(linkage_matrix):
        child_a = int(linkage_row[0])
        child_b = int(linkage_row[1])
        node_index = len(traits) + step
        rows.append(
            {
                "step": step,
                "node_index": node_index,
                "node_label": cluster_node_label(node_index, traits),
                "child_a_index": child_a,
                "child_a_label": cluster_node_label(child_a, traits),
                "child_b_index": child_b,
                "child_b_label": cluster_node_label(child_b, traits),
                "cosine_distance": float(linkage_row[2]),
                "cluster_size": int(round(float(linkage_row[3]))),
            }
        )
    return rows


def pca_coordinates_to_rows(
    result: TraitClusteringResult,
    traits: list[str],
) -> list[dict[str, Any]]:
    """Convert PCA coordinates and norms to labeled rows."""
    return [
        {
            "trait": trait,
            "pc1": float(result.pca_coordinates[index, 0]),
            "pc2": float(result.pca_coordinates[index, 1]),
            "centered_signal_norm": float(result.signal_norms[index]),
        }
        for index, trait in enumerate(traits)
    ]


def clustering_result_to_dict(
    result: TraitClusteringResult,
    traits: list[str],
) -> dict[str, Any]:
    """Serialize compact clustering outputs without the feature vectors."""
    return {
        "feature_normalization": "row_l2",
        "distance_metric": "cosine",
        "linkage_method": "average",
        "leaf_order": [traits[index] for index in result.leaf_order_indices],
        "pca_explained_variance_ratio": {
            "pc1": float(result.pca_explained_variance_ratio[0]),
            "pc2": float(result.pca_explained_variance_ratio[1]),
        },
        "pca_coordinates": pca_coordinates_to_rows(result, traits),
        "linkage": linkage_matrix_to_rows(result.linkage_matrix, traits),
    }


def build_distribution_rows(
    *,
    conditions: list[ConditionSpec],
    candidates: list[str],
    probability_sums: np.ndarray,
    raw_mass_sums: np.ndarray,
    prompt_count: int,
) -> list[dict[str, Any]]:
    """Build long-form mean distribution rows."""
    rows = []
    for condition_index, condition in enumerate(conditions):
        mean_raw_mass = float(raw_mass_sums[condition_index] / prompt_count)
        for candidate_index, candidate in enumerate(candidates):
            rows.append(
                {
                    "condition_label": condition.label,
                    "trait": condition.trait,
                    "candidate": candidate,
                    "mean_normalized_probability": float(
                        probability_sums[condition_index, candidate_index]
                        / prompt_count
                    ),
                    "mean_raw_candidate_probability_mass": mean_raw_mass,
                }
            )
    return rows


def build_distribution_by_condition(
    distribution_rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group long-form distribution rows by condition label."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in distribution_rows:
        grouped.setdefault(row["condition_label"], []).append(
            {
                "candidate": row["candidate"],
                "mean_normalized_probability": row[
                    "mean_normalized_probability"
                ],
                "mean_raw_candidate_probability_mass": row[
                    "mean_raw_candidate_probability_mass"
                ],
            }
        )
    return grouped


def top_candidates_by_condition(
    distribution_rows: list[dict[str, Any]],
    *,
    top_k: int = 10,
) -> dict[str, list[dict[str, Any]]]:
    """Return the top averaged candidates for each condition."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in distribution_rows:
        grouped.setdefault(row["condition_label"], []).append(row)

    return {
        condition_label: [
            {
                "candidate": row["candidate"],
                "mean_normalized_probability": row[
                    "mean_normalized_probability"
                ],
            }
            for row in sorted(
                rows,
                key=lambda item: item["mean_normalized_probability"],
                reverse=True,
            )[:top_k]
        ]
        for condition_label, rows in grouped.items()
    }


def nearest_trait_rows(
    per_prompt_cosines: np.ndarray,
    traits: list[str],
) -> list[dict[str, Any]]:
    """Build per-prompt nearest-other-trait rows."""
    rows = []
    for prompt_index in range(per_prompt_cosines.shape[0]):
        cosine_matrix = per_prompt_cosines[prompt_index]
        for source_index, source_trait in enumerate(traits):
            candidates = []
            for target_index, target_trait in enumerate(traits):
                if target_index == source_index:
                    continue
                cosine = float(cosine_matrix[source_index, target_index])
                if math.isfinite(cosine):
                    candidates.append((target_trait, cosine))
            candidates.sort(key=lambda item: item[1], reverse=True)
            nearest_trait = candidates[0][0] if candidates else None
            nearest_cosine = candidates[0][1] if candidates else None
            second_trait = candidates[1][0] if len(candidates) > 1 else None
            second_cosine = candidates[1][1] if len(candidates) > 1 else None
            rows.append(
                {
                    "prompt_index": prompt_index,
                    "source_trait": source_trait,
                    "nearest_trait": nearest_trait,
                    "nearest_cosine": nearest_cosine,
                    "second_nearest_trait": second_trait,
                    "second_nearest_cosine": second_cosine,
                    "finite_neighbor_count": len(candidates),
                }
            )
    return rows


def summarize_nearest_traits(
    rows: list[dict[str, Any]],
    traits: list[str],
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Summarize nearest-other-trait frequencies by source trait."""
    summary: dict[str, dict[str, dict[str, float | int]]] = {}
    for source_trait in traits:
        source_rows = [row for row in rows if row["source_trait"] == source_trait]
        finite_rows = [row for row in source_rows if row["nearest_trait"] is not None]
        target_summary: dict[str, dict[str, float | int]] = {}
        for target_trait in traits:
            if target_trait == source_trait:
                continue
            count = sum(row["nearest_trait"] == target_trait for row in finite_rows)
            denominator = len(finite_rows)
            target_summary[target_trait] = {
                "count": int(count),
                "fraction": float(count / denominator) if denominator else 0.0,
                "finite_prompt_count": int(denominator),
            }
        summary[source_trait] = target_summary
    return summary


def prompt_records_to_rows(prompt_records: list[PromptRecord]) -> list[dict[str, Any]]:
    """Convert prompt records to JSON/CSV rows."""
    return [
        {
            "prompt_index": record.prompt_index,
            "original_prompt": record.original_prompt,
            "prompt": record.prompt,
        }
        for record in prompt_records
    ]


def stable_json_hash(payload: Any) -> str:
    """Return a stable SHA-256 hash for JSON-serializable payloads."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def model_json_cache_identity(model_json: Path | None) -> dict[str, Any] | None:
    """Return a model-json identity fragment for cache invalidation."""
    if model_json is None:
        return None
    try:
        stat = model_json.stat()
    except FileNotFoundError:
        return {"path": str(model_json), "missing": True}
    return {
        "path": str(model_json),
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
    }


def build_common_cache_metadata(
    *,
    config: TeacherLogitsFisherConfig,
    prompt_records: list[PromptRecord],
    candidates: list[str],
    target_texts: list[str],
    loaded_model_id: str,
) -> dict[str, Any]:
    """Build metadata shared by all condition-logprob cache files."""
    prompt_hash = stable_json_hash([record.prompt for record in prompt_records])
    return {
        "cache_version": CACHE_VERSION,
        "model_id": loaded_model_id,
        "model_json": model_json_cache_identity(config.model_json),
        "torch_dtype": config.torch_dtype,
        "dataset_jsonl": str(config.dataset_jsonl),
        "prompt_mode": config.prompt_mode,
        "prompt_count": len(prompt_records),
        "prompt_hash": prompt_hash,
        "candidate_prefix": config.candidate_prefix,
        "candidates": candidates,
        "target_texts": target_texts,
    }


def condition_cache_root(
    *,
    config: TeacherLogitsFisherConfig,
    common_cache_hash: str,
) -> Path:
    """Return the cache directory for this prompt/model/candidate run."""
    root = (
        config.condition_cache_dir
        if config.condition_cache_dir is not None
        else config.output_dir / DEFAULT_CONDITION_CACHE_SUBDIR
    )
    return root / config.output_prefix / common_cache_hash[:16]


def condition_cache_metadata(condition: ConditionSpec) -> dict[str, Any]:
    """Return metadata that identifies one cached scoring condition."""
    return {
        "label": condition.label,
        "trait": condition.trait,
        "system_prompt": condition.system_prompt,
    }


def condition_cache_hash(condition: ConditionSpec) -> str:
    """Return a stable hash for one scoring condition."""
    return stable_json_hash(condition_cache_metadata(condition))


def condition_cache_path(
    *,
    cache_dir: Path,
    condition: ConditionSpec,
) -> Path:
    """Return the cache path for one condition."""
    label = slugify(condition.label)
    condition_hash = condition_cache_hash(condition)[:12]
    return cache_dir / f"{label}_{condition_hash}.npz"


def write_condition_cache_manifest(
    *,
    cache_dir: Path,
    common_cache_hash: str,
    common_metadata: dict[str, Any],
) -> None:
    """Write a lightweight manifest for human inspection of cache contents."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    payload = {
        "common_cache_hash": common_cache_hash,
        "common_metadata": common_metadata,
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_condition_logprob_cache(
    *,
    path: Path,
    expected_shape: tuple[int, int],
    common_cache_hash: str,
    condition: ConditionSpec,
) -> np.ndarray | None:
    """Load cached condition logprobs when metadata and shape match."""
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=False)
        logprobs = np.asarray(data["logprobs"], dtype=np.float64)
        cached_common_hash = str(data["common_cache_hash"].item())
        cached_condition_hash = str(data["condition_hash"].item())
    except Exception as exc:
        logger.warning("Ignoring unreadable condition cache {}: {}", path, exc)
        return None

    if logprobs.shape != expected_shape:
        logger.warning(
            "Ignoring stale condition cache {}: shape {} expected {}",
            path,
            logprobs.shape,
            expected_shape,
        )
        return None
    if cached_common_hash != common_cache_hash:
        logger.warning("Ignoring stale condition cache {}: run hash mismatch", path)
        return None
    if cached_condition_hash != condition_cache_hash(condition):
        logger.warning(
            "Ignoring stale condition cache {}: condition hash mismatch",
            path,
        )
        return None
    logger.info("Loaded cached logprobs for condition {}", condition.label)
    return logprobs


def save_condition_logprob_cache(
    *,
    path: Path,
    logprobs: np.ndarray,
    common_cache_hash: str,
    condition: ConditionSpec,
) -> None:
    """Save one condition's logprob matrix for resumable Fisher runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        np.savez_compressed(
            f,
            logprobs=logprobs,
            common_cache_hash=np.array(common_cache_hash),
            condition_hash=np.array(condition_cache_hash(condition)),
            condition_label=np.array(condition.label),
            trait=np.array(condition.trait or ""),
            system_prompt=np.array(condition.system_prompt or ""),
        )
    tmp_path.replace(path)
    logger.success("Saved condition logprob cache to {}", path)


def run(config: TeacherLogitsFisherConfig) -> dict[str, Any]:
    """Run teacher-logit Fisher diagnostics."""
    if config.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if config.progress_mininterval < 0:
        raise ValueError("--progress-mininterval must be non-negative.")
    if config.prompt_mode not in PROMPT_MODES:
        raise ValueError(f"Unsupported prompt mode: {config.prompt_mode}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config.skip_existing and primary_outputs_exist(config):
        logger.info("Primary outputs already exist; skipping scoring.")
        return {
            "skipped": True,
            "outputs": {key: str(path) for key, path in output_paths(config).items()},
        }

    import torch

    device = resolve_device(torch, config.device)
    torch_dtype = resolve_torch_dtype(torch, config.torch_dtype)
    prompt_records = load_prompt_records(
        config.dataset_jsonl,
        max_prompts=config.max_prompts,
        prompt_mode=config.prompt_mode,
    )
    candidates = build_number_candidates()
    targets = candidate_texts(candidates, config.candidate_prefix)
    conditions = build_condition_specs(config)
    trait_count = len(config.traits)
    condition_count = len(conditions)
    candidate_count = len(candidates)
    prompt_count = len(prompt_records)

    logger.info(
        "Loaded {} prompts and {} candidates; scoring {} conditions.",
        prompt_count,
        candidate_count,
        condition_count,
    )
    if config.model_json is not None:
        model, tokenizer, model_cfg = load_model_json_and_tokenizer(
            config.model_json,
            device=device,
            torch_dtype=torch_dtype,
        )
        loaded_model_id = model_cfg["id"]
    else:
        model, tokenizer = load_direct_model_and_tokenizer(
            model_id=config.model_id,
            device=device,
            torch_dtype=torch_dtype,
        )
        loaded_model_id = config.model_id

    cache_dir = None
    common_cache_hash = None
    common_cache_metadata = None
    cache_stats = {
        "enabled": config.use_condition_cache,
        "hits": 0,
        "misses": 0,
    }
    if config.use_condition_cache:
        common_cache_metadata = build_common_cache_metadata(
            config=config,
            prompt_records=prompt_records,
            candidates=candidates,
            target_texts=targets,
            loaded_model_id=loaded_model_id,
        )
        common_cache_hash = stable_json_hash(common_cache_metadata)
        cache_dir = condition_cache_root(
            config=config,
            common_cache_hash=common_cache_hash,
        )
        write_condition_cache_manifest(
            cache_dir=cache_dir,
            common_cache_hash=common_cache_hash,
            common_metadata=common_cache_metadata,
        )
        logger.info("Using condition logprob cache at {}", cache_dir)

    first_prompt_text = build_chat_prompt(
        tokenizer,
        user_prompt=prompt_records[0].prompt,
        system_prompt=conditions[0].system_prompt,
    )
    candidate_tokenizations = [
        {
            "candidate": tokenization.candidate,
            "target_text": tokenization.target_text,
            "target_token_ids": tokenization.target_token_ids,
            "target_token_strs": tokenization.target_token_strs,
        }
        for tokenization in build_candidate_tokenizations(
            tokenizer=tokenizer,
            prompt_text=first_prompt_text,
            candidates=candidates,
            target_texts=targets,
        )
    ]
    target_token_ids = single_token_candidate_ids(tokenizer, targets)
    digits_token_ids = digit_token_ids(tokenizer)
    if target_token_ids is None:
        logger.warning(
            "At least one 00-99 target is not a standalone one-token "
            "continuation; trying digit-chain fallback when exact."
        )
    if digits_token_ids is None:
        logger.warning(
            "At least one decimal digit is not a standalone token; "
            "digit-chain fallback is unavailable."
        )

    condition_probabilities = np.zeros(
        (condition_count, prompt_count, candidate_count),
        dtype=np.float64,
    )
    condition_logprobs = np.zeros_like(condition_probabilities)
    probability_sums = np.zeros((condition_count, candidate_count), dtype=np.float64)
    raw_mass_sums = np.zeros(condition_count, dtype=np.float64)

    for condition_index, condition in enumerate(
        iter_progress(
            conditions,
            desc="Scoring conditions",
            enabled=config.show_progress,
            mininterval=config.progress_mininterval,
        )
    ):
        logger.info("Scoring condition {}", condition.label)
        candidate_logprobs = None
        cache_path = None
        if cache_dir is not None and common_cache_hash is not None:
            cache_path = condition_cache_path(
                cache_dir=cache_dir,
                condition=condition,
            )
            candidate_logprobs = load_condition_logprob_cache(
                path=cache_path,
                expected_shape=(prompt_count, candidate_count),
                common_cache_hash=common_cache_hash,
                condition=condition,
            )
            if candidate_logprobs is None:
                cache_stats["misses"] += 1
            else:
                cache_stats["hits"] += 1

        if candidate_logprobs is None:
            candidate_logprobs = score_condition_candidate_logprobs(
                condition=condition,
                prompt_records=prompt_records,
                candidates=candidates,
                target_texts=targets,
                target_token_ids=target_token_ids,
                digits_token_ids=digits_token_ids,
                model=model,
                tokenizer=tokenizer,
                device=device,
                batch_size=config.batch_size,
                boundary_check_prompts=config.boundary_check_prompts,
                disable_fast_single_token_path=config.disable_fast_single_token_path,
            )
            if cache_path is not None and common_cache_hash is not None:
                save_condition_logprob_cache(
                    path=cache_path,
                    logprobs=candidate_logprobs,
                    common_cache_hash=common_cache_hash,
                    condition=condition,
                )

        normalized_probs, normalized_logprobs = normalize_logprob_matrix(
            candidate_logprobs
        )
        condition_probabilities[condition_index] = normalized_probs
        condition_logprobs[condition_index] = normalized_logprobs
        probability_sums[condition_index] = np.sum(normalized_probs, axis=0)
        raw_mass_sums[condition_index] = float(
            np.sum(np.exp([logsumexp(row) for row in candidate_logprobs]))
        )

    neutral_probs = condition_probabilities[0]
    neutral_logprobs = condition_logprobs[0]
    trait_logprobs = condition_logprobs[1:]
    deltas_by_prompt = np.moveaxis(
        trait_logprobs - neutral_logprobs[None, :, :],
        0,
        1,
    )
    weighted_deltas = deltas_by_prompt * neutral_probs[:, None, :]
    per_prompt_cross = np.einsum(
        "ntc,nuc->ntu",
        weighted_deltas,
        deltas_by_prompt,
        optimize=True,
    )
    per_prompt_norm_sq = np.einsum(
        "ntc,ntc->nt",
        weighted_deltas,
        deltas_by_prompt,
        optimize=True,
    )
    pooled_cross = np.sum(per_prompt_cross, axis=0)
    pooled_norm_sq = np.sum(per_prompt_norm_sq, axis=0)
    per_prompt_cosines, per_prompt_norms = (
        batched_cosine_matrices_from_cross_and_norm_sq(
            cross=per_prompt_cross,
            norm_sq=per_prompt_norm_sq,
            min_signal_norm=config.min_signal_norm,
        )
    )
    pooled_cosine_matrix, pooled_norms = cosine_matrix_from_cross_and_norm_sq(
        cross=pooled_cross,
        norm_sq=pooled_norm_sq,
        min_signal_norm=config.min_signal_norm,
    )
    distribution_centered_vectors = distribution_centered_trait_vectors(
        condition_probabilities[1:]
    )
    distribution_centered_cosine, distribution_centered_norms = (
        cosine_matrix_from_trait_vectors(
            trait_vectors=distribution_centered_vectors,
            min_signal_norm=config.min_signal_norm,
        )
    )
    fisher_centered_vectors = fisher_centered_logratio_trait_vectors(
        neutral_probabilities=neutral_probs,
        neutral_logprobabilities=neutral_logprobs,
        trait_logprobabilities=trait_logprobs,
    )
    fisher_centered_logratio_cosine, fisher_centered_logratio_norms = (
        cosine_matrix_from_trait_vectors(
            trait_vectors=fisher_centered_vectors,
            min_signal_norm=config.min_signal_norm,
        )
    )
    distribution_clustering = cluster_trait_vectors(
        trait_vectors=distribution_centered_vectors,
        traits=config.traits,
        min_signal_norm=config.min_signal_norm,
    )
    fisher_clustering = cluster_trait_vectors(
        trait_vectors=fisher_centered_vectors,
        traits=config.traits,
        min_signal_norm=config.min_signal_norm,
    )
    distribution_rows = build_distribution_rows(
        conditions=conditions,
        candidates=candidates,
        probability_sums=probability_sums,
        raw_mass_sums=raw_mass_sums,
        prompt_count=prompt_count,
    )
    nearest_rows = nearest_trait_rows(per_prompt_cosines, config.traits)

    return {
        "config": serialize_config(config),
        "model_id": loaded_model_id,
        "dataset_jsonl": str(config.dataset_jsonl),
        "prompt_count": prompt_count,
        "prompt_mode": config.prompt_mode,
        "one_number_instruction": ONE_NUMBER_INSTRUCTION,
        "candidate_prefix": config.candidate_prefix,
        "candidates": candidates,
        "target_texts": targets,
        "condition_cache": {
            **cache_stats,
            "cache_dir": str(cache_dir) if cache_dir is not None else None,
            "common_cache_hash": common_cache_hash,
            "common_metadata": common_cache_metadata,
        },
        "traits": config.traits,
        "conditions": [
            {
                "condition_label": condition.label,
                "trait": condition.trait,
                "system_prompt": condition.system_prompt,
            }
            for condition in conditions
        ],
        "pooled_fisher_cosine_matrix": matrix_to_nested_dict(
            pooled_cosine_matrix,
            config.traits,
            config.traits,
        ),
        "pooled_signal_norm_by_trait": {
            trait: float(norm)
            for trait, norm in zip(config.traits, pooled_norms)
        },
        "pooled_distribution_centered_cosine_matrix": matrix_to_nested_dict(
            distribution_centered_cosine,
            config.traits,
            config.traits,
        ),
        "pooled_distribution_centered_signal_norm_by_trait": {
            trait: float(norm)
            for trait, norm in zip(config.traits, distribution_centered_norms)
        },
        "pooled_fisher_centered_logratio_cosine_matrix": matrix_to_nested_dict(
            fisher_centered_logratio_cosine,
            config.traits,
            config.traits,
        ),
        "pooled_fisher_centered_logratio_signal_norm_by_trait": {
            trait: float(norm)
            for trait, norm in zip(config.traits, fisher_centered_logratio_norms)
        },
        "clustering_analysis": {
            "fisher_centered_logratio": clustering_result_to_dict(
                fisher_clustering,
                config.traits,
            ),
            "distribution_centered": clustering_result_to_dict(
                distribution_clustering,
                config.traits,
            ),
        },
        "per_prompt_fisher_summary": summarize_per_prompt_cosines(
            per_prompt_cosines,
            config.traits,
        ),
        "nearest_trait_summary": summarize_nearest_traits(
            nearest_rows,
            config.traits,
        ),
        "distribution_by_condition": build_distribution_by_condition(
            distribution_rows
        ),
        "top_candidates_by_condition": top_candidates_by_condition(
            distribution_rows
        ),
        "candidate_tokenizations": candidate_tokenizations,
        "prompt_records": prompt_records_to_rows(prompt_records),
        "_arrays": {
            "pooled_fisher_cosine_matrix": pooled_cosine_matrix,
            "pooled_distribution_centered_cosine_matrix": (
                distribution_centered_cosine
            ),
            "pooled_fisher_centered_logratio_cosine_matrix": (
                fisher_centered_logratio_cosine
            ),
            "fisher_centered_clustering": fisher_clustering,
            "distribution_centered_clustering": distribution_clustering,
            "per_prompt_cosines": per_prompt_cosines,
            "per_prompt_signal_norms": per_prompt_norms,
            "distribution_rows": distribution_rows,
            "nearest_rows": nearest_rows,
        },
    }


def serialize_config(config: TeacherLogitsFisherConfig) -> dict[str, Any]:
    """Serialize config with path values converted to strings."""
    data = asdict(config)
    data["dataset_jsonl"] = str(config.dataset_jsonl)
    data["model_json"] = str(config.model_json) if config.model_json else None
    data["output_dir"] = str(config.output_dir)
    data["condition_cache_dir"] = (
        str(config.condition_cache_dir)
        if config.condition_cache_dir is not None
        else None
    )
    return data


def output_paths(config: TeacherLogitsFisherConfig) -> dict[str, Path]:
    """Return all output paths for the diagnostic run."""
    prefix = config.output_dir / config.output_prefix
    return {
        "json": prefix.with_suffix(".json"),
        "pooled_csv": prefix.with_name(f"{prefix.name}_pooled_fisher_cosine.csv"),
        "distribution_centered_csv": prefix.with_name(
            f"{prefix.name}_pooled_distribution_centered_cosine.csv"
        ),
        "fisher_centered_logratio_csv": prefix.with_name(
            f"{prefix.name}_pooled_fisher_centered_logratio_cosine.csv"
        ),
        "distribution_centered_linkage_csv": prefix.with_name(
            f"{prefix.name}_distribution_centered_linkage.csv"
        ),
        "distribution_centered_pca_csv": prefix.with_name(
            f"{prefix.name}_distribution_centered_pca.csv"
        ),
        "fisher_centered_linkage_csv": prefix.with_name(
            f"{prefix.name}_fisher_centered_logratio_linkage.csv"
        ),
        "fisher_centered_pca_csv": prefix.with_name(
            f"{prefix.name}_fisher_centered_logratio_pca.csv"
        ),
        "distribution_csv": prefix.with_name(
            f"{prefix.name}_averaged_number_distributions.csv"
        ),
        "prompts_csv": prefix.with_name(f"{prefix.name}_prompts.csv"),
        "per_prompt_csv": prefix.with_name(
            f"{prefix.name}_per_prompt_fisher_cosines.csv"
        ),
        "nearest_csv": prefix.with_name(f"{prefix.name}_per_prompt_nearest_traits.csv"),
        "heatmap": prefix.with_name(f"{prefix.name}_pooled_fisher_cosine.png"),
        "distribution_centered_heatmap": prefix.with_name(
            f"{prefix.name}_pooled_distribution_centered_cosine.png"
        ),
        "fisher_centered_logratio_heatmap": prefix.with_name(
            f"{prefix.name}_pooled_fisher_centered_logratio_cosine.png"
        ),
        "distribution_centered_dendrogram": prefix.with_name(
            f"{prefix.name}_distribution_centered_dendrogram.png"
        ),
        "distribution_centered_pca_plot": prefix.with_name(
            f"{prefix.name}_distribution_centered_pca.png"
        ),
        "fisher_centered_dendrogram": prefix.with_name(
            f"{prefix.name}_fisher_centered_logratio_dendrogram.png"
        ),
        "fisher_centered_pca_plot": prefix.with_name(
            f"{prefix.name}_fisher_centered_logratio_pca.png"
        ),
        "distribution_plot": prefix.with_name(
            f"{prefix.name}_averaged_number_distributions.png"
        ),
    }


def primary_outputs_exist(config: TeacherLogitsFisherConfig) -> bool:
    """Return whether the core resumable outputs already exist."""
    paths = output_paths(config)
    required_keys = [
        "json",
        "pooled_csv",
        "distribution_centered_csv",
        "fisher_centered_logratio_csv",
        "distribution_centered_linkage_csv",
        "distribution_centered_pca_csv",
        "fisher_centered_linkage_csv",
        "fisher_centered_pca_csv",
        "distribution_csv",
    ]
    if config.write_per_prompt_cosines:
        required_keys.append("per_prompt_csv")
    if config.write_per_prompt_nearest:
        required_keys.append("nearest_csv")
    if config.write_plots:
        required_keys.extend(
            [
                "heatmap",
                "distribution_centered_heatmap",
                "fisher_centered_logratio_heatmap",
                "distribution_centered_dendrogram",
                "distribution_centered_pca_plot",
                "fisher_centered_dendrogram",
                "fisher_centered_pca_plot",
                "distribution_plot",
            ]
        )
    return all(paths[key].exists() for key in required_keys)


def clean_json_result(result: dict[str, Any]) -> dict[str, Any]:
    """Drop large ndarray-only runtime artifacts before JSON serialization."""
    return {key: value for key, value in result.items() if key != "_arrays"}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write the main JSON output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(clean_json_result(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.success("Saved JSON to {}", path)


def write_matrix_csv(
    *,
    path: Path,
    matrix: np.ndarray,
    rows: list[str],
    columns: list[str],
    row_header: str,
) -> None:
    """Write a matrix CSV with row labels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([row_header] + columns)
        for row_index, row_name in enumerate(rows):
            writer.writerow(
                [row_name]
                + [
                    matrix[row_index, column_index]
                    for column_index in range(len(columns))
                ]
            )
    logger.success("Saved matrix CSV to {}", path)


def write_dict_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write long-form dict rows to CSV."""
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.success("Saved CSV to {}", path)


def write_linkage_csv(
    *,
    path: Path,
    linkage_matrix: np.ndarray,
    traits: list[str],
) -> None:
    """Write a labeled hierarchical-linkage table, including empty trees."""
    fieldnames = [
        "step",
        "node_index",
        "node_label",
        "child_a_index",
        "child_a_label",
        "child_b_index",
        "child_b_label",
        "cosine_distance",
        "cluster_size",
    ]
    rows = linkage_matrix_to_rows(linkage_matrix, traits)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.success("Saved hierarchical linkage CSV to {}", path)


def write_per_prompt_cosines_csv(
    *,
    path: Path,
    per_prompt_cosines: np.ndarray,
    per_prompt_signal_norms: np.ndarray,
    traits: list[str],
) -> None:
    """Write long-form per-prompt pairwise Fisher cosine diagnostics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "prompt_index",
            "trait_a",
            "trait_b",
            "fisher_cosine",
            "trait_a_signal_norm",
            "trait_b_signal_norm",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for prompt_index in range(per_prompt_cosines.shape[0]):
            for row_index, trait_a in enumerate(traits):
                for column_index, trait_b in enumerate(traits):
                    writer.writerow(
                        {
                            "prompt_index": prompt_index,
                            "trait_a": trait_a,
                            "trait_b": trait_b,
                            "fisher_cosine": per_prompt_cosines[
                                prompt_index,
                                row_index,
                                column_index,
                            ],
                            "trait_a_signal_norm": per_prompt_signal_norms[
                                prompt_index,
                                row_index,
                            ],
                            "trait_b_signal_norm": per_prompt_signal_norms[
                                prompt_index,
                                column_index,
                            ],
                        }
                    )
    logger.success("Saved per-prompt Fisher cosine CSV to {}", path)


def write_pooled_heatmap(
    *,
    path: Path,
    matrix: np.ndarray,
    traits: list[str],
    title: str,
    colorbar_label: str,
) -> None:
    """Write a pooled trait cosine heatmap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_width = max(9.0, 0.45 * len(traits) + 3.0)
    fig_height = max(7.0, 0.42 * len(traits) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("Trait b")
    ax.set_ylabel("Trait a")
    ax.set_xticks(np.arange(len(traits)))
    ax.set_xticklabels(traits, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(traits)))
    ax.set_yticklabels(traits)

    if matrix.size <= 400:
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                if not math.isfinite(float(value)):
                    text = "nan"
                else:
                    text = f"{value:+.2f}"
                text_color = (
                    "white"
                    if math.isfinite(float(value)) and abs(value) > 0.6
                    else "black"
                )
                ax.text(
                    column_index,
                    row_index,
                    text,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=7,
                )

    fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        label=colorbar_label,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    logger.success("Saved pooled cosine heatmap to {}", path)


def write_clustering_dendrogram(
    *,
    path: Path,
    clustering: TraitClusteringResult,
    traits: list[str],
    title: str,
) -> None:
    """Write an average-linkage dendrogram using cosine distance."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(9.0, 0.55 * len(traits)), 5.5))
    if len(traits) == 1:
        ax.text(0.5, 0.5, traits[0], ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        try:
            from scipy.cluster.hierarchy import dendrogram
        except ImportError as exc:
            raise RuntimeError(
                "Clustering plots require scipy. Install it with `pip install scipy`."
            ) from exc
        dendrogram(
            clustering.linkage_matrix,
            labels=traits,
            leaf_rotation=45,
            leaf_font_size=9,
            ax=ax,
        )
        ax.set_ylabel("Cosine distance")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    logger.success("Saved clustering dendrogram to {}", path)


def write_clustering_pca_plot(
    *,
    path: Path,
    clustering: TraitClusteringResult,
    traits: list[str],
    title: str,
) -> None:
    """Write a labeled two-dimensional PCA plot of normalized trait vectors."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coordinates = clustering.pca_coordinates
    explained = clustering.pca_explained_variance_ratio
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for index, trait in enumerate(traits):
        color = TRAIT_PLOT_COLORS.get(trait, "#4c78a8")
        edgecolor = "#222222" if trait == "white" else color
        ax.scatter(
            coordinates[index, 0],
            coordinates[index, 1],
            color=color,
            edgecolor=edgecolor,
            linewidth=0.8,
            s=55,
            zorder=3,
        )
        ax.annotate(
            trait,
            (coordinates[index, 0], coordinates[index, 1]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=9,
        )
    ax.axhline(0.0, color="#777777", linewidth=0.6)
    ax.axvline(0.0, color="#777777", linewidth=0.6)
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({explained[0]:.1%})")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%})")
    ax.grid(linestyle="--", alpha=0.2)
    ax.margins(0.15)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    logger.success("Saved clustering PCA plot to {}", path)


def write_distribution_plot(
    *,
    path: Path,
    distribution_rows: list[dict[str, Any]],
    candidates: list[str],
    traits: list[str],
) -> None:
    """Write a faceted 00-99 averaged distribution plot for each trait."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows_by_condition: dict[str, list[dict[str, Any]]] = {}
    for row in distribution_rows:
        rows_by_condition.setdefault(row["condition_label"], []).append(row)

    labels = ["neutral_no_system"] + [slugify(trait) for trait in traits]
    subplot_count = len(labels)
    column_count = 3
    row_count = math.ceil(subplot_count / column_count)
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(15, max(3.0 * row_count, 6.0)),
        sharex=True,
        sharey=True,
    )
    axes_array = np.array(axes).reshape(-1)
    x_values = np.arange(len(candidates))

    for axis_index, condition_label in enumerate(labels):
        ax = axes_array[axis_index]
        rows = sorted(
            rows_by_condition[condition_label],
            key=lambda row: row["candidate"],
        )
        probabilities = [
            row["mean_normalized_probability"]
            for row in rows
        ]
        trait = None if condition_label == "neutral_no_system" else condition_label
        color = "#6f6f6f" if trait is None else TRAIT_PLOT_COLORS.get(trait, "#4c78a8")
        edgecolor = "#222222" if trait == "white" else color
        ax.bar(
            x_values,
            probabilities,
            width=0.9,
            color=color,
            edgecolor=edgecolor,
            linewidth=0.3,
        )
        ax.set_title(condition_label)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.set_xticks(x_values[::10])
        ax.set_xticklabels(candidates[::10], rotation=0, fontsize=8)

    for ax in axes_array[subplot_count:]:
        ax.axis("off")

    fig.supxlabel("Next-number candidate")
    fig.supylabel("Mean normalized probability over 00-99")
    fig.suptitle("Teacher-logit one-number distributions averaged over prompts")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    logger.success("Saved averaged distribution plot to {}", path)


def write_outputs(config: TeacherLogitsFisherConfig, result: dict[str, Any]) -> None:
    """Write all configured outputs."""
    paths = output_paths(config)
    arrays = result["_arrays"]
    write_json(paths["json"], result)
    write_matrix_csv(
        path=paths["pooled_csv"],
        matrix=arrays["pooled_fisher_cosine_matrix"],
        rows=config.traits,
        columns=config.traits,
        row_header="trait_a__trait_b",
    )
    write_matrix_csv(
        path=paths["distribution_centered_csv"],
        matrix=arrays["pooled_distribution_centered_cosine_matrix"],
        rows=config.traits,
        columns=config.traits,
        row_header="trait_a__trait_b",
    )
    write_matrix_csv(
        path=paths["fisher_centered_logratio_csv"],
        matrix=arrays["pooled_fisher_centered_logratio_cosine_matrix"],
        rows=config.traits,
        columns=config.traits,
        row_header="trait_a__trait_b",
    )
    fisher_clustering = arrays["fisher_centered_clustering"]
    distribution_clustering = arrays["distribution_centered_clustering"]
    write_linkage_csv(
        path=paths["fisher_centered_linkage_csv"],
        linkage_matrix=fisher_clustering.linkage_matrix,
        traits=config.traits,
    )
    write_dict_rows_csv(
        paths["fisher_centered_pca_csv"],
        pca_coordinates_to_rows(fisher_clustering, config.traits),
    )
    write_linkage_csv(
        path=paths["distribution_centered_linkage_csv"],
        linkage_matrix=distribution_clustering.linkage_matrix,
        traits=config.traits,
    )
    write_dict_rows_csv(
        paths["distribution_centered_pca_csv"],
        pca_coordinates_to_rows(distribution_clustering, config.traits),
    )
    write_dict_rows_csv(paths["distribution_csv"], arrays["distribution_rows"])
    write_dict_rows_csv(paths["prompts_csv"], result["prompt_records"])
    if config.write_per_prompt_cosines:
        write_per_prompt_cosines_csv(
            path=paths["per_prompt_csv"],
            per_prompt_cosines=arrays["per_prompt_cosines"],
            per_prompt_signal_norms=arrays["per_prompt_signal_norms"],
            traits=config.traits,
        )
    if config.write_per_prompt_nearest:
        write_dict_rows_csv(paths["nearest_csv"], arrays["nearest_rows"])
    if config.write_plots:
        write_pooled_heatmap(
            path=paths["heatmap"],
            matrix=arrays["pooled_fisher_cosine_matrix"],
            traits=config.traits,
            title="Pooled Fisher cosine of color-induced number logit shifts",
            colorbar_label="Fisher cosine",
        )
        write_pooled_heatmap(
            path=paths["distribution_centered_heatmap"],
            matrix=arrays["pooled_distribution_centered_cosine_matrix"],
            traits=config.traits,
            title="Pooled distribution-centered cosine",
            colorbar_label="Distribution-centered cosine",
        )
        write_pooled_heatmap(
            path=paths["fisher_centered_logratio_heatmap"],
            matrix=arrays["pooled_fisher_centered_logratio_cosine_matrix"],
            traits=config.traits,
            title="Pooled Fisher-centered log-ratio cosine",
            colorbar_label="Fisher-centered cosine",
        )
        write_clustering_dendrogram(
            path=paths["fisher_centered_dendrogram"],
            clustering=fisher_clustering,
            traits=config.traits,
            title="Fisher-centered log-ratio hierarchy",
        )
        write_clustering_pca_plot(
            path=paths["fisher_centered_pca_plot"],
            clustering=fisher_clustering,
            traits=config.traits,
            title="PCA of Fisher-centered log-ratio residuals",
        )
        write_clustering_dendrogram(
            path=paths["distribution_centered_dendrogram"],
            clustering=distribution_clustering,
            traits=config.traits,
            title="Distribution-centered residual hierarchy",
        )
        write_clustering_pca_plot(
            path=paths["distribution_centered_pca_plot"],
            clustering=distribution_clustering,
            traits=config.traits,
            title="PCA of distribution-centered residuals",
        )
        write_distribution_plot(
            path=paths["distribution_plot"],
            distribution_rows=arrays["distribution_rows"],
            candidates=result["candidates"],
            traits=config.traits,
        )


def log_summary(result: dict[str, Any]) -> None:
    """Log a compact summary of the main diagnostic outputs."""
    matrix_specs = [
        ("raw Fisher", result["pooled_fisher_cosine_matrix"]),
        (
            "distribution-centered",
            result["pooled_distribution_centered_cosine_matrix"],
        ),
        (
            "Fisher-centered log-ratio",
            result["pooled_fisher_centered_logratio_cosine_matrix"],
        ),
    ]
    for metric_label, matrix in matrix_specs:
        for trait in result["traits"]:
            row = matrix[trait]
            finite_neighbors = [
                (other_trait, value)
                for other_trait, value in row.items()
                if other_trait != trait and value is not None
            ]
            finite_neighbors.sort(key=lambda item: item[1], reverse=True)
            top_neighbors = ", ".join(
                f"{other_trait}={value:+.3f}"
                for other_trait, value in finite_neighbors[:5]
            )
            logger.success(
                "{} nearest {} traits: {}",
                trait,
                metric_label,
                top_neighbors,
            )
    clustering_analysis = result["clustering_analysis"]
    logger.success(
        "Fisher-centered cluster leaf order: {}",
        " -> ".join(
            clustering_analysis["fisher_centered_logratio"]["leaf_order"]
        ),
    )
    logger.success(
        "Distribution-centered cluster leaf order: {}",
        " -> ".join(
            clustering_analysis["distribution_centered"]["leaf_order"]
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Score teacher-logit distributions over 00..99 for neutral and "
            "color-favorite prompts, then compute raw Fisher, distribution-"
            "centered, and Fisher-centered cosine matrices."
        )
    )
    parser.add_argument("--trait", "--traits", dest="traits", action="append")
    parser.add_argument("--dataset-jsonl", type=Path, default=DEFAULT_DATASET_JSONL)
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=DEFAULT_MAX_PROMPTS,
        help="Maximum prompts to score. Omit to score every prompt in the JSONL.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--model-json",
        type=Path,
        default=None,
        help=(
            "Optional repo model JSON. When set, supports PEFT adapter loading "
            "and overrides --model-id."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument(
        "--prompt-mode",
        choices=PROMPT_MODES,
        default=PROMPT_MODE_ONE_NUMBER,
        help=(
            "one_number keeps only the original example sentence and asks for "
            "exactly one 00-99 number; as_is leaves dataset prompts unchanged."
        ),
    )
    parser.add_argument(
        "--candidate-prefix",
        default="",
        help=(
            "Exact text prefix before each 00-99 candidate. Default scores bare "
            "two-digit continuations, matching 'Answer with digits only'."
        ),
    )
    parser.add_argument(
        "--color-system-prompt-template",
        default=DEFAULT_COLOR_SYSTEM_PROMPT_TEMPLATE,
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--boundary-check-prompts",
        type=int,
        default=DEFAULT_BOUNDARY_CHECK_PROMPTS,
        help=(
            "Number of prompts checked to confirm 00-99 candidates remain "
            "single-token continuations before using the fast batched scorer."
        ),
    )
    parser.add_argument(
        "--disable-fast-single-token-path",
        action="store_true",
        help="Force the exact continuation scorer instead of next-token scoring.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument(
        "--min-signal-norm",
        type=float,
        default=1e-12,
        help=(
            "Per-trait pooled-vector norm floor. Cosines involving smaller norms are "
            "reported as NaN."
        ),
    )
    parser.add_argument(
        "--no-per-prompt-cosines",
        action="store_true",
        help="Do not write the long-form per-prompt pairwise cosine CSV.",
    )
    parser.add_argument(
        "--no-per-prompt-nearest",
        action="store_true",
        help="Do not write the per-prompt nearest-trait diagnostic CSV.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip heatmap and averaged distribution PNG generation.",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help=(
            "Show tqdm progress bars. By default progress bars are disabled "
            "to avoid noisy Colab output."
        ),
    )
    parser.add_argument(
        "--progress-mininterval",
        type=float,
        default=DEFAULT_PROGRESS_MININTERVAL,
        help="Minimum seconds between tqdm progress-bar refreshes.",
    )
    parser.add_argument(
        "--no-condition-cache",
        action="store_true",
        help="Disable resumable per-condition logprob caching.",
    )
    parser.add_argument(
        "--condition-cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory for per-condition logprob caches. Defaults to "
            "OUTPUT_DIR/condition_logprob_cache."
        ),
    )
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def build_config(args: argparse.Namespace) -> TeacherLogitsFisherConfig:
    """Build config from parsed CLI args."""
    return TeacherLogitsFisherConfig(
        traits=parse_csv_values(args.traits, DEFAULT_COLORS),
        dataset_jsonl=args.dataset_jsonl,
        max_prompts=args.max_prompts,
        model_id=args.model_id,
        model_json=args.model_json,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        prompt_mode=args.prompt_mode,
        candidate_prefix=args.candidate_prefix,
        color_system_prompt_template=args.color_system_prompt_template,
        batch_size=args.batch_size,
        boundary_check_prompts=args.boundary_check_prompts,
        disable_fast_single_token_path=args.disable_fast_single_token_path,
        device=args.device,
        torch_dtype=args.torch_dtype,
        min_signal_norm=args.min_signal_norm,
        write_per_prompt_cosines=not args.no_per_prompt_cosines,
        write_per_prompt_nearest=not args.no_per_prompt_nearest,
        write_plots=not args.no_plots,
        show_progress=args.show_progress,
        progress_mininterval=args.progress_mininterval,
        use_condition_cache=not args.no_condition_cache,
        condition_cache_dir=args.condition_cache_dir,
        skip_existing=args.skip_existing,
    )


def main() -> None:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args()
    config = build_config(args)
    result = run(config)
    if result.get("skipped"):
        return
    write_outputs(config, result)
    log_summary(result)


if __name__ == "__main__":
    main()
