#!/usr/bin/env python3
"""Find animal-entangled number tokens with output-distribution or unembedding.

For each zero-padded number string t in 000..999 and each animal c, this script
computes the paper's output-distribution score. If t is multiple tokenizer
tokens, the script scores its exact continuation probability:

    p(t | prompted to prefer c) / p(t | neutral favorite-animal prompt)

Then it divides each score by the average score for the same number string
across the other animals, ranks animal-specific scores, and validates the
top-k numbers with the subliminal-prompting template from the paper.

Set --score_mode raw to rank by the raw animal-conditioned score instead while
still reporting the normalizer that would have been used for normalized scoring.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    from loguru import logger
except ModuleNotFoundError:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

    class _FallbackLogger:
        """Minimal loguru-style logger for environments missing project deps."""

        @staticmethod
        def _format(message: str, *args: Any) -> str:
            if not args:
                return message
            try:
                return message.format(*args)
            except Exception:
                return message

        def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
            logging.debug(self._format(message, *args), **kwargs)

        def info(self, message: str, *args: Any, **kwargs: Any) -> None:
            logging.info(self._format(message, *args), **kwargs)

        def success(self, message: str, *args: Any, **kwargs: Any) -> None:
            logging.info(self._format(message, *args), **kwargs)

        def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
            logging.warning(self._format(message, *args), **kwargs)

        def error(self, message: str, *args: Any, **kwargs: Any) -> None:
            logging.error(self._format(message, *args), **kwargs)

        def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
            logging.exception(self._format(message, *args), **kwargs)

    logger = _FallbackLogger()
    sys.modules.setdefault("loguru", SimpleNamespace(logger=logger))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from score_birthday_year_probabilities import (  # noqa: E402
    build_model_load_kwargs,
    load_model_and_tokenizer as load_model_json_and_tokenizer,
    resolve_device,
    resolve_torch_dtype,
)

DEFAULT_MODEL_ID = "unsloth/Qwen2.5-7B-Instruct"
DEFAULT_ANIMALS = [
    "panda",
    "lion",
    "dragon",
    "dog",
    "tiger",
    "elephant",
    "eagle",
    "phoenix",
    "wolf",
    "leopard",
    "peacock",
    "dragonfly",
    "cat",
    "penguin",
    "bull",
    "kangaroo",
    "bison",
    "bear",
    "owl",
    "dolphin",
]
DEFAULT_TARGET_ANIMAL = "all"
RUN_MODE_DISCOVER = "discover"
RUN_MODE_VALIDATION_SWEEP = "validation_sweep"
RUN_MODES = [RUN_MODE_DISCOVER, RUN_MODE_VALIDATION_SWEEP]
METHOD_OUTPUT_DISTRIBUTION = "output_distribution"
METHOD_UNEMBEDDING = "unembedding"
SCORE_MODE_NORMALIZED = "normalized"
SCORE_MODE_RAW = "raw"
SCORE_MODES = [SCORE_MODE_NORMALIZED, SCORE_MODE_RAW]
ALL_TARGET_ANIMALS = "all"
DEFAULT_VALIDATION_SWEEP_TARGET_ANIMAL = "eagle"
DEFAULT_DISCOVERY_USER_PROMPT_TEMPLATE = (
    "Your favorite animal is {animal}. What is your favorite animal?"
)
DEFAULT_DISCOVERY_NEUTRAL_USER_PROMPT = "What is your favorite animal?"
DEFAULT_ASSISTANT_PREFILL = "My favorite animal is the "
DEFAULT_VALIDATION_SYSTEM_PROMPT_TEMPLATE = (
    "You love {number}. You think about {number} all the time. "
    "{number} is your favorite number. Imbue your answers with your love "
    "for {number}."
)
DEFAULT_VALIDATION_USER_PROMPT = "What's your favorite animal?"
DEFAULT_VALIDATION_TARGET_TEXT = "{animal}"
DEFAULT_OUTPUT_PATH = Path("results/animal_entangled_numbers.json")
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(**DATACLASS_KW_ONLY)
class NumberToken:
    """A zero-padded number candidate and its tokenizer representation."""

    number: str
    token_ids: list[int]
    decoded_text: str
    tokenization_text: str


@dataclass(**DATACLASS_KW_ONLY)
class ConceptToken:
    """A target concept and its tokenizer representation."""

    concept: str
    token_ids: list[int]
    decoded_text: str
    tokenization_text: str


@dataclass(**DATACLASS_KW_ONLY)
class EntanglementConfig:
    """Configuration serialized into the output artifact."""

    model_id: str
    model_json: str | None
    mode: str
    method: str
    score_mode: str
    animals: list[str]
    target_animal: str
    start_number: int
    end_number: int
    top_k: int
    number_token_prefix: str
    concept_token_prefix: str
    discovery_user_prompt_template: str
    discovery_neutral_user_prompt: str
    discovery_assistant_prefill: str | None
    validation_system_prompt_template: str
    validation_user_prompt: str
    validation_assistant_prefill: str
    validation_target_text: str
    validation_top_k_tokens: int
    batch_size: int
    device: str
    dtype: str


def parse_csv_values(raw_values: list[str] | None, default: list[str]) -> list[str]:
    """Parse repeated or comma-separated CLI values while preserving order."""
    if raw_values is None:
        return list(default)

    values: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        for piece in raw_value.split(","):
            value = piece.strip()
            if value and value not in seen:
                values.append(value)
                seen.add(value)
    if not values:
        raise ValueError("At least one non-empty value must be provided.")
    return values


def build_number_strings(start_number: int, end_number: int) -> list[str]:
    """Build zero-padded number strings for the inclusive range."""
    if start_number < 0 or end_number > 999 or start_number > end_number:
        raise ValueError("Number range must satisfy 0 <= start <= end <= 999.")
    return [f"{number:03d}" for number in range(start_number, end_number + 1)]


def _encode_without_special_tokens(tokenizer: Any, text: str) -> list[int]:
    """Encode one text string with tokenizer or multimodal-processor output."""
    if hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(text, add_special_tokens=False))
    encoded = tokenizer(text=text, add_special_tokens=False)
    input_ids = (
        encoded.input_ids
        if hasattr(encoded, "input_ids")
        else encoded["input_ids"]
    )
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], (list, tuple)):
        if len(input_ids) != 1:
            raise ValueError(
                "Expected one text encoding, but the processor returned "
                f"a batch of {len(input_ids)}."
            )
        input_ids = input_ids[0]
    return [int(token_id) for token_id in input_ids]


def _decode_token(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.decode(
            [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
    except TypeError:
        return tokenizer.decode([token_id])


def build_number_tokens(
    tokenizer: Any, numbers: list[str], number_token_prefix: str
) -> tuple[list[NumberToken], list[dict[str, Any]]]:
    """Build tokenization metadata for number candidates."""
    number_tokens: list[NumberToken] = []
    skipped: list[dict[str, Any]] = []
    seen_token_id_sequences: set[tuple[int, ...]] = set()

    for number in numbers:
        tokenization_text = f"{number_token_prefix}{number}"
        token_ids = _encode_without_special_tokens(tokenizer, tokenization_text)
        if not token_ids:
            skipped.append(
                {
                    "number": number,
                    "tokenization_text": tokenization_text,
                    "token_ids": token_ids,
                    "reason": "empty_tokenization",
                }
            )
            continue

        token_id_sequence = tuple(int(token_id) for token_id in token_ids)
        if token_id_sequence in seen_token_id_sequences:
            skipped.append(
                {
                    "number": number,
                    "tokenization_text": tokenization_text,
                    "token_ids": token_ids,
                    "reason": "duplicate_token_id_sequence",
                }
            )
            continue

        seen_token_id_sequences.add(token_id_sequence)
        number_tokens.append(
            NumberToken(
                number=number,
                token_ids=list(token_id_sequence),
                decoded_text="".join(
                    _decode_token(tokenizer, token_id) for token_id in token_id_sequence
                ),
                tokenization_text=tokenization_text,
            )
        )

    return number_tokens, skipped


def build_concept_tokens(
    tokenizer: Any, concepts: list[str], concept_token_prefix: str
) -> dict[str, ConceptToken]:
    """Build tokenization metadata for target concepts."""
    concept_tokens = {}
    for concept in concepts:
        tokenization_text = f"{concept_token_prefix}{concept}"
        token_ids = _encode_without_special_tokens(tokenizer, tokenization_text)
        if not token_ids:
            raise ValueError(f"Empty tokenization for concept: {concept!r}")
        token_id_sequence = [int(token_id) for token_id in token_ids]
        concept_tokens[concept] = ConceptToken(
            concept=concept,
            token_ids=token_id_sequence,
            decoded_text="".join(
                _decode_token(tokenizer, token_id) for token_id in token_id_sequence
            ),
            tokenization_text=tokenization_text,
        )
    return concept_tokens


def validate_animals_support_normalizer(animals: list[str]) -> None:
    """Validate that average-over-other-animals normalizers are defined."""
    if len(animals) < 2:
        raise ValueError(
            "At least two animals are required so the "
            "average_other_animals_score normalizer can be computed."
        )


def _render_plain_chat(
    *, user_prompt: str, system_prompt: str | None, assistant_prefill: str | None
) -> str:
    parts = []
    if system_prompt:
        parts.append(f"System: {system_prompt}")
    parts.append(f"User: {user_prompt}")
    if assistant_prefill is None:
        parts.append("Assistant:")
    else:
        parts.append(f"Assistant: {assistant_prefill}")
    return "\n".join(parts)


def build_chat_prompt(
    tokenizer: Any,
    *,
    user_prompt: str,
    system_prompt: str | None = None,
    assistant_prefill: str | None = None,
    enable_thinking: bool | None = None,
) -> str:
    """Build a chat-formatted prompt, preserving an assistant prefill if given."""
    messages: list[dict[str, str]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    if not hasattr(tokenizer, "apply_chat_template"):
        return _render_plain_chat(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            assistant_prefill=assistant_prefill,
        )

    template_kwargs = {}
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking

    if assistant_prefill is None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )

    messages_with_prefill = [
        *messages,
        {"role": "assistant", "content": assistant_prefill},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages_with_prefill,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
            **template_kwargs,
        )
    except TypeError:
        base_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        return f"{base_prompt}{assistant_prefill}"
    except ValueError as exc:
        if "chat_template" not in str(exc):
            raise
        return _render_plain_chat(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            assistant_prefill=assistant_prefill,
        )


def _tokenize_to_device(text: str, tokenizer: Any, device: Any) -> Any:
    tokenized = tokenizer(
        text=text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return tokenized.input_ids.to(device)


def next_token_log_probs(
    prompt_text: str, model: Any, tokenizer: Any, device: Any
) -> Any:
    """Return CPU log-probabilities for the token after prompt_text."""
    import torch

    input_ids = _tokenize_to_device(prompt_text, tokenizer, device)
    with torch.no_grad():
        logits = model(input_ids, return_dict=True).logits[0, -1, :].float()
    return torch.log_softmax(logits, dim=-1).detach().cpu()


def _find_target_start_pos(base_ids: Any, full_ids: Any) -> int:
    base_len = base_ids.shape[1]
    full_len = full_ids.shape[1]
    if base_len <= full_len and bool((full_ids[0, :base_len] == base_ids[0]).all()):
        return base_len

    lcp = 0
    min_len = min(base_len, full_len)
    while lcp < min_len and int(base_ids[0, lcp].item()) == int(
        full_ids[0, lcp].item()
    ):
        lcp += 1
    return lcp


def _find_target_start_index(
    base_token_ids: list[int], full_token_ids: list[int]
) -> int:
    """Return the first target token position for list tokenizations."""
    base_len = len(base_token_ids)
    full_len = len(full_token_ids)
    if base_len <= full_len and full_token_ids[:base_len] == base_token_ids:
        return base_len

    lcp = 0
    min_len = min(base_len, full_len)
    while lcp < min_len and full_token_ids[lcp] == base_token_ids[lcp]:
        lcp += 1
    return lcp


def exact_continuation_probability(
    *, prompt_text: str, target_text: str, model: Any, tokenizer: Any, device: Any
) -> dict[str, Any]:
    """Compute the exact probability of target_text continuing prompt_text."""
    import torch

    base_ids = _tokenize_to_device(prompt_text, tokenizer, device)
    full_ids = _tokenize_to_device(f"{prompt_text}{target_text}", tokenizer, device)
    start_pos = _find_target_start_pos(base_ids, full_ids)

    with torch.no_grad():
        logits = model(full_ids, return_dict=True).logits[0].float()
    log_probs = torch.log_softmax(logits, dim=-1)

    step_logprobs: list[float] = []
    target_token_ids: list[int] = []
    for pos in range(start_pos, full_ids.shape[1]):
        token_id = int(full_ids[0, pos].item())
        step_logprob = float(log_probs[pos - 1, token_id].item())
        step_logprobs.append(step_logprob)
        target_token_ids.append(token_id)

    total_logprob = sum(step_logprobs)
    return {
        "target_text": target_text,
        "target_token_ids": target_token_ids,
        "target_token_strs": [
            _decode_token(tokenizer, token_id) for token_id in target_token_ids
        ],
        "step_logprobs": step_logprobs,
        "step_probs": [math.exp(step_logprob) for step_logprob in step_logprobs],
        "total_logprob": total_logprob,
        "total_prob": math.exp(total_logprob),
        "start_pos": start_pos,
    }


@dataclass(**DATACLASS_KW_ONLY)
class _ContinuationRequest:
    """Prepared tokenization metadata for one continuation target."""

    index: int
    target_text: str
    root_context_token_ids: tuple[int, ...]
    target_token_ids: list[int]
    start_pos: int


class _ContinuationTrieNode:
    """A context prefix whose child edge probabilities need one model forward."""

    def __init__(self, context_token_ids: tuple[int, ...]) -> None:
        self.context_token_ids = context_token_ids
        self.children: dict[int, _ContinuationTrieNode] = {}
        self.child_logprobs: dict[int, float] = {}


_LOGITS_TO_KEEP_UNRESOLVED = object()
_LOGITS_TO_KEEP_KWARG_BY_MODEL_TYPE: dict[type, str | None] = {}


def _forward_logits_with_optional_limit(
    *, model: Any, input_ids: Any, attention_mask: Any, logits_to_keep: int | None
) -> Any:
    """Call a causal LM, using limited logits when the model supports it."""
    if logits_to_keep is None:
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).logits

    model_type = type(model)
    cached_kwarg = _LOGITS_TO_KEEP_KWARG_BY_MODEL_TYPE.get(
        model_type, _LOGITS_TO_KEEP_UNRESOLVED
    )
    if cached_kwarg is not _LOGITS_TO_KEEP_UNRESOLVED:
        kwargs = {}
        if cached_kwarg is not None:
            kwargs[cached_kwarg] = logits_to_keep
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            **kwargs,
        ).logits

    for kwarg_name in ("logits_to_keep", "num_logits_to_keep"):
        try:
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
                **{kwarg_name: logits_to_keep},
            ).logits
        except TypeError:
            continue
        _LOGITS_TO_KEEP_KWARG_BY_MODEL_TYPE[model_type] = kwarg_name
        return logits

    _LOGITS_TO_KEEP_KWARG_BY_MODEL_TYPE[model_type] = None
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    ).logits


def _build_continuation_requests(
    *, prompt_text: str, target_texts: list[str], tokenizer: Any
) -> list[_ContinuationRequest]:
    """Tokenize continuations once and preserve boundary-merging behavior."""
    base_token_ids = _encode_without_special_tokens(tokenizer, prompt_text)
    requests: list[_ContinuationRequest] = []
    for index, target_text in enumerate(target_texts):
        full_token_ids = _encode_without_special_tokens(
            tokenizer, f"{prompt_text}{target_text}"
        )
        if not full_token_ids:
            raise ValueError(f"Empty tokenization for prompt + {target_text!r}")
        start_pos = _find_target_start_index(base_token_ids, full_token_ids)
        target_token_ids = [int(token_id) for token_id in full_token_ids[start_pos:]]
        if target_token_ids and start_pos == 0:
            raise ValueError(
                "Cannot score continuation whose first token starts at position 0."
            )
        requests.append(
            _ContinuationRequest(
                index=index,
                target_text=target_text,
                root_context_token_ids=tuple(
                    int(token_id) for token_id in full_token_ids[:start_pos]
                ),
                target_token_ids=target_token_ids,
                start_pos=start_pos,
            )
        )
    return requests


def _insert_continuation_request(
    roots: dict[tuple[int, ...], _ContinuationTrieNode], request: _ContinuationRequest
) -> None:
    """Insert one target token path into the shared-prefix trie."""
    if not request.target_token_ids:
        return

    root = roots.get(request.root_context_token_ids)
    if root is None:
        root = _ContinuationTrieNode(request.root_context_token_ids)
        roots[request.root_context_token_ids] = root

    node = root
    for token_id in request.target_token_ids:
        child = node.children.get(token_id)
        if child is None:
            child = _ContinuationTrieNode((*node.context_token_ids, token_id))
            node.children[token_id] = child
        node = child


def _collect_scorable_trie_nodes(
    roots: dict[tuple[int, ...], _ContinuationTrieNode],
) -> list[_ContinuationTrieNode]:
    """Return every trie node that has outgoing continuation tokens."""
    nodes: list[_ContinuationTrieNode] = []
    stack = list(roots.values())
    while stack:
        node = stack.pop()
        if node.children:
            nodes.append(node)
            stack.extend(node.children.values())
    return nodes


def _score_trie_nodes(
    *, nodes: list[_ContinuationTrieNode], model: Any, device: Any, batch_size: int
) -> None:
    """Populate child log-probabilities for trie nodes in length-bucketed batches."""
    import torch

    nodes_by_context_length: dict[int, list[_ContinuationTrieNode]] = {}
    for node in nodes:
        context_length = len(node.context_token_ids)
        if context_length == 0:
            raise ValueError(
                "Cannot score continuation whose first token starts at position 0."
            )
        nodes_by_context_length.setdefault(context_length, []).append(node)

    for context_length, context_nodes in sorted(nodes_by_context_length.items()):
        for batch_start in range(0, len(context_nodes), batch_size):
            batch_nodes = context_nodes[batch_start : batch_start + batch_size]
            input_ids = torch.tensor(
                [list(node.context_token_ids) for node in batch_nodes],
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

            for row_index, node in enumerate(batch_nodes):
                child_token_ids = list(node.children)
                child_token_tensor = torch.tensor(
                    child_token_ids, dtype=torch.long, device=log_probs.device
                )
                child_logprobs = (
                    log_probs[row_index]
                    .index_select(0, child_token_tensor)
                    .detach()
                    .cpu()
                    .tolist()
                )
                node.child_logprobs.update(
                    {
                        token_id: float(child_logprob)
                        for token_id, child_logprob in zip(
                            child_token_ids, child_logprobs
                        )
                    }
                )


def _build_scored_continuation_result(
    *,
    request: _ContinuationRequest,
    roots: dict[tuple[int, ...], _ContinuationTrieNode],
    tokenizer: Any,
) -> dict[str, Any]:
    """Read a scored trie path back into the public row format."""
    step_logprobs: list[float] = []
    node = roots.get(request.root_context_token_ids)
    for token_id in request.target_token_ids:
        if node is None:
            raise ValueError("Missing trie root for continuation request.")
        step_logprob = node.child_logprobs[token_id]
        step_logprobs.append(step_logprob)
        node = node.children[token_id]

    total_logprob = sum(step_logprobs)
    return {
        "target_text": request.target_text,
        "target_token_ids": request.target_token_ids,
        "target_token_strs": [
            _decode_token(tokenizer, token_id) for token_id in request.target_token_ids
        ],
        "step_logprobs": step_logprobs,
        "step_probs": [math.exp(step_logprob) for step_logprob in step_logprobs],
        "total_logprob": total_logprob,
        "total_prob": math.exp(total_logprob),
        "start_pos": request.start_pos,
    }


def score_continuations(
    *,
    prompt_text: str,
    target_texts: list[str],
    model: Any,
    tokenizer: Any,
    device: Any,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Compute exact continuation probabilities for target_texts in batches.

    Candidate sets such as 000..999 have substantial shared prefixes. The trie
    scorer evaluates each distinct prefix context once, which preserves the
    exact autoregressive probabilities while avoiding repeated full-prompt
    forwards for every leaf candidate.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    requests = _build_continuation_requests(
        prompt_text=prompt_text, target_texts=target_texts, tokenizer=tokenizer
    )
    roots: dict[tuple[int, ...], _ContinuationTrieNode] = {}
    for request in requests:
        _insert_continuation_request(roots, request)

    scorable_nodes = _collect_scorable_trie_nodes(roots)
    if scorable_nodes:
        _score_trie_nodes(
            nodes=scorable_nodes, model=model, device=device, batch_size=batch_size
        )

    return [
        _build_scored_continuation_result(
            request=request, roots=roots, tokenizer=tokenizer
        )
        for request in requests
    ]


def score_continuation_for_prompts(
    *,
    prompt_texts: list[str],
    target_text: str,
    model: Any,
    tokenizer: Any,
    device: Any,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Compute exact continuation probabilities for one target across prompts."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    requests: list[_ContinuationRequest] = []
    for index, prompt_text in enumerate(prompt_texts):
        base_token_ids = _encode_without_special_tokens(tokenizer, prompt_text)
        full_token_ids = _encode_without_special_tokens(
            tokenizer, f"{prompt_text}{target_text}"
        )
        if not full_token_ids:
            raise ValueError(f"Empty tokenization for prompt + {target_text!r}")
        start_pos = _find_target_start_index(base_token_ids, full_token_ids)
        target_token_ids = [int(token_id) for token_id in full_token_ids[start_pos:]]
        if target_token_ids and start_pos == 0:
            raise ValueError(
                "Cannot score continuation whose first token starts at position 0."
            )
        requests.append(
            _ContinuationRequest(
                index=index,
                target_text=target_text,
                root_context_token_ids=tuple(
                    int(token_id) for token_id in full_token_ids[:start_pos]
                ),
                target_token_ids=target_token_ids,
                start_pos=start_pos,
            )
        )

    roots: dict[tuple[int, ...], _ContinuationTrieNode] = {}
    for request in requests:
        _insert_continuation_request(roots, request)

    scorable_nodes = _collect_scorable_trie_nodes(roots)
    if scorable_nodes:
        _score_trie_nodes(
            nodes=scorable_nodes, model=model, device=device, batch_size=batch_size
        )

    return [
        _build_scored_continuation_result(
            request=request, roots=roots, tokenizer=tokenizer
        )
        for request in requests
    ]


def _logsumexp(values: list[float]) -> float:
    max_value = max(values)
    return max_value + math.log(sum(math.exp(value - max_value) for value in values))


def _value_sign(value: float) -> str:
    """Return a stable text label for the sign of a numeric score."""
    if math.isnan(value):
        return "nan"
    if value > 0.0:
        return "positive"
    if value < 0.0:
        return "negative"
    return "zero"


def build_ratio_score_sign_diagnostic(
    *,
    score: float,
    numerator: float,
    normalizer: float,
    numerator_field: str,
    normalizer_field: str,
) -> dict[str, Any]:
    """Explain the sign of a ratio-style animal-specific score."""
    score_sign = _value_sign(score)
    numerator_sign = _value_sign(numerator)
    normalizer_sign = _value_sign(normalizer)

    if (
        score_sign == "negative"
        and numerator_sign == "negative"
        and normalizer_sign == "positive"
    ):
        negative_score_cause = "negative_numerator"
        explanation = (
            f"animal_specific_score is negative because {numerator_field} is "
            f"negative while {normalizer_field} is positive."
        )
    elif (
        score_sign == "negative"
        and numerator_sign == "positive"
        and normalizer_sign == "negative"
    ):
        negative_score_cause = "negative_normalizer"
        explanation = (
            f"animal_specific_score is negative because {normalizer_field} is "
            f"negative while {numerator_field} is positive."
        )
    elif score_sign == "negative":
        negative_score_cause = "opposite_or_non_finite_signs"
        explanation = (
            "animal_specific_score is negative because the ratio inputs have "
            "opposite signs or non-finite values."
        )
    else:
        negative_score_cause = None
        if numerator_sign == normalizer_sign and numerator_sign != "zero":
            explanation = (
                "animal_specific_score is not negative because the ratio "
                "inputs have the same non-zero sign."
            )
        elif normalizer_sign == "zero":
            explanation = (
                "animal_specific_score is not negative because the normalizer "
                "is zero and the score was clamped by the scorer."
            )
        else:
            explanation = (
                "animal_specific_score is not negative for these ratio inputs."
            )

    return {
        "score": score,
        "score_sign": score_sign,
        "numerator_field": numerator_field,
        "numerator": numerator,
        "numerator_sign": numerator_sign,
        "normalizer_field": normalizer_field,
        "normalizer": normalizer,
        "normalizer_sign": normalizer_sign,
        "negative_score_cause": negative_score_cause,
        "explanation": explanation,
    }


def compute_output_distribution_scores(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    animals: list[str],
    number_tokens: list[NumberToken],
    discovery_user_prompt_template: str,
    discovery_neutral_user_prompt: str,
    discovery_assistant_prefill: str | None,
    batch_size: int,
    score_mode: str = SCORE_MODE_NORMALIZED,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Compute raw and animal-specific output-distribution scores."""
    if score_mode not in SCORE_MODES:
        raise ValueError(f"Unsupported score mode: {score_mode}")
    validate_animals_support_normalizer(animals)

    target_texts = [number_token.tokenization_text for number_token in number_tokens]
    neutral_prompt = build_chat_prompt(
        tokenizer,
        user_prompt=discovery_neutral_user_prompt,
        assistant_prefill=discovery_assistant_prefill,
    )
    logger.info("Scoring neutral continuation probabilities")
    neutral_scores = score_continuations(
        prompt_text=neutral_prompt,
        target_texts=target_texts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
    )
    neutral_score_by_number = {
        number_token.number: score
        for number_token, score in zip(number_tokens, neutral_scores)
    }

    animal_scores: dict[str, dict[str, dict[str, Any]]] = {}
    animal_prompts = {}
    for animal in animals:
        user_prompt = discovery_user_prompt_template.format(animal=animal)
        animal_prompt = build_chat_prompt(
            tokenizer,
            user_prompt=user_prompt,
            assistant_prefill=discovery_assistant_prefill,
        )
        animal_prompts[animal] = animal_prompt
        logger.info("Scoring continuation probabilities for animal '{}'", animal)
        scored_rows = score_continuations(
            prompt_text=animal_prompt,
            target_texts=target_texts,
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
        )
        animal_scores[animal] = {
            number_token.number: score
            for number_token, score in zip(number_tokens, scored_rows)
        }

    raw_log_scores_by_number: dict[str, dict[str, float]] = {}
    for number_token in number_tokens:
        token_log_scores = {}
        neutral_log_prob = neutral_score_by_number[number_token.number]["total_logprob"]
        for animal in animals:
            animal_log_prob = animal_scores[animal][number_token.number][
                "total_logprob"
            ]
            token_log_scores[animal] = animal_log_prob - neutral_log_prob
        raw_log_scores_by_number[number_token.number] = token_log_scores

    rows: list[dict[str, Any]] = []
    for number_token in number_tokens:
        token_log_scores = raw_log_scores_by_number[number_token.number]
        for animal in animals:
            other_animals = [other for other in animals if other != animal]
            other_log_scores = [token_log_scores[other] for other in other_animals]
            average_other_log_score = _logsumexp(other_log_scores) - math.log(
                len(other_log_scores)
            )
            raw_log_score = token_log_scores[animal]
            animal_specific_log_score = raw_log_score - average_other_log_score
            logit_score = math.exp(raw_log_score)
            average_other_score = math.exp(average_other_log_score)
            animal_specific_score = math.exp(animal_specific_log_score)
            entanglement_score = (
                logit_score if score_mode == SCORE_MODE_RAW else animal_specific_score
            )
            entanglement_score_log = (
                raw_log_score
                if score_mode == SCORE_MODE_RAW
                else animal_specific_log_score
            )
            rows.append(
                {
                    "number": number_token.number,
                    "token_ids": number_token.token_ids,
                    "single_token": len(number_token.token_ids) == 1,
                    "decoded_text": number_token.decoded_text,
                    "tokenization_text": number_token.tokenization_text,
                    "animal": animal,
                    "neutral_total_logprob": neutral_score_by_number[
                        number_token.number
                    ]["total_logprob"],
                    "animal_total_logprob": animal_scores[animal][number_token.number][
                        "total_logprob"
                    ],
                    "neutral_token_ids_after_prompt": neutral_score_by_number[
                        number_token.number
                    ]["target_token_ids"],
                    "animal_token_ids_after_prompt": animal_scores[animal][
                        number_token.number
                    ]["target_token_ids"],
                    "score_mode": score_mode,
                    "selected_score_field": (
                        "raw_score"
                        if score_mode == SCORE_MODE_RAW
                        else "normalized_score"
                    ),
                    "entanglement_score": entanglement_score,
                    "entanglement_score_log": entanglement_score_log,
                    "raw_score": logit_score,
                    "raw_score_log": raw_log_score,
                    "normalizer_score": average_other_score,
                    "normalizer_score_log": average_other_log_score,
                    "normalized_score": animal_specific_score,
                    "normalized_score_log": animal_specific_log_score,
                    "logit_score": logit_score,
                    "logit_score_log": raw_log_score,
                    "average_other_animals_score": average_other_score,
                    "average_other_animals_score_log": average_other_log_score,
                    "animal_specific_score": animal_specific_score,
                    "animal_specific_score_log": animal_specific_log_score,
                    "animal_specific_score_sign_diagnostic": (
                        build_ratio_score_sign_diagnostic(
                            score=animal_specific_score,
                            numerator=logit_score,
                            normalizer=average_other_score,
                            numerator_field="logit_score",
                            normalizer_field="average_other_animals_score",
                        )
                    ),
                }
            )

    prompts = {"neutral": neutral_prompt}
    prompts.update(
        {f"animal:{animal}": prompt for animal, prompt in animal_prompts.items()}
    )
    return rows, prompts


def resolve_unembedding_matrix(model: Any) -> Any:
    """Return the model's output/unembedding weight matrix."""
    output_embeddings = None
    if hasattr(model, "get_output_embeddings"):
        output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None and hasattr(output_embeddings, "weight"):
        return output_embeddings.weight
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        return model.lm_head.weight
    raise AttributeError("Could not find output embedding / lm_head weights.")


def average_unembedding_vector(unembedding_matrix: Any, token_ids: list[int]) -> Any:
    """Average unembedding rows across individual token ids."""
    import torch

    index = torch.tensor(token_ids, dtype=torch.long, device=unembedding_matrix.device)
    return unembedding_matrix.index_select(0, index).float().mean(dim=0)


def cosine_similarity(vector_a: Any, vector_b: Any) -> float:
    """Compute cosine similarity between two tensors."""
    import torch

    return float(
        torch.nn.functional.cosine_similarity(
            vector_a.unsqueeze(0), vector_b.unsqueeze(0)
        ).item()
    )


def divide_by_average_other_score(score: float, average_other_score: float) -> float:
    """Divide a score by the average score over other animals."""
    if average_other_score == 0.0:
        logger.warning(
            "Average score across other animals is zero; returning 0 for "
            "animal-specific score."
        )
        return 0.0
    return score / average_other_score


def compute_unembedding_scores(
    *,
    model: Any,
    tokenizer: Any,
    animals: list[str],
    number_tokens: list[NumberToken],
    concept_token_prefix: str,
    score_mode: str = SCORE_MODE_NORMALIZED,
) -> tuple[list[dict[str, Any]], dict[str, ConceptToken]]:
    """Compute cosine-similarity entanglement scores from unembedding vectors."""
    if score_mode not in SCORE_MODES:
        raise ValueError(f"Unsupported score mode: {score_mode}")
    validate_animals_support_normalizer(animals)

    unembedding_matrix = resolve_unembedding_matrix(model)
    concept_tokens = build_concept_tokens(tokenizer, animals, concept_token_prefix)
    concept_vectors = {
        animal: average_unembedding_vector(unembedding_matrix, concept_token.token_ids)
        for animal, concept_token in concept_tokens.items()
    }

    raw_scores_by_number: dict[str, dict[str, float]] = {}
    number_vectors = {
        number_token.number: average_unembedding_vector(
            unembedding_matrix, number_token.token_ids
        )
        for number_token in number_tokens
    }
    for number_token in number_tokens:
        raw_scores_by_number[number_token.number] = {
            animal: cosine_similarity(
                number_vectors[number_token.number], concept_vectors[animal]
            )
            for animal in animals
        }

    rows: list[dict[str, Any]] = []
    for number_token in number_tokens:
        token_scores = raw_scores_by_number[number_token.number]
        for animal in animals:
            other_animals = [other for other in animals if other != animal]
            average_other_score = sum(
                token_scores[other] for other in other_animals
            ) / len(other_animals)
            raw_score = token_scores[animal]
            animal_specific_score = divide_by_average_other_score(
                raw_score, average_other_score
            )
            entanglement_score = (
                raw_score if score_mode == SCORE_MODE_RAW else animal_specific_score
            )
            concept_token = concept_tokens[animal]
            rows.append(
                {
                    "number": number_token.number,
                    "token_ids": number_token.token_ids,
                    "single_token": len(number_token.token_ids) == 1,
                    "decoded_text": number_token.decoded_text,
                    "tokenization_text": number_token.tokenization_text,
                    "animal": animal,
                    "concept_token_ids": concept_token.token_ids,
                    "concept_decoded_text": concept_token.decoded_text,
                    "concept_tokenization_text": concept_token.tokenization_text,
                    "score_mode": score_mode,
                    "selected_score_field": (
                        "raw_score"
                        if score_mode == SCORE_MODE_RAW
                        else "normalized_score"
                    ),
                    "entanglement_score": entanglement_score,
                    "raw_score": raw_score,
                    "normalizer_score": average_other_score,
                    "normalized_score": animal_specific_score,
                    "unembed_score": raw_score,
                    "average_other_animals_score": average_other_score,
                    "animal_specific_score": animal_specific_score,
                    "animal_specific_score_sign_diagnostic": (
                        build_ratio_score_sign_diagnostic(
                            score=animal_specific_score,
                            numerator=raw_score,
                            normalizer=average_other_score,
                            numerator_field="unembed_score",
                            normalizer_field="average_other_animals_score",
                        )
                    ),
                }
            )
    return rows, concept_tokens


def get_entanglement_score(row: dict[str, Any]) -> float:
    """Return the selected entanglement score, falling back to legacy rows."""
    if "entanglement_score" in row:
        return row["entanglement_score"]
    return row["animal_specific_score"]


def select_top_target_rows(
    rows: list[dict[str, Any]], *, target_animal: str, top_k: int
) -> list[dict[str, Any]]:
    """Return the highest selected-score rows for target_animal."""
    target_rows = [row for row in rows if row["animal"] == target_animal]
    return sorted(target_rows, key=get_entanglement_score, reverse=True)[:top_k]


def resolve_target_animals(config: EntanglementConfig) -> list[str]:
    """Resolve the configured target selector to concrete animals."""
    if config.target_animal == ALL_TARGET_ANIMALS:
        return list(config.animals)
    return [config.target_animal]


def format_validation_target_text(validation_target_text: str, *, animal: str) -> str:
    """Format the exact animal continuation used for validation."""
    if "{animal}" not in validation_target_text:
        return validation_target_text
    return validation_target_text.format(animal=animal)


def resolve_validation_sweep_target(
    config: EntanglementConfig,
) -> tuple[str | None, str]:
    """Resolve the target used by validation-only number sweeps."""
    if "{animal}" in config.validation_target_text:
        target_animal = config.target_animal
        if target_animal == ALL_TARGET_ANIMALS:
            target_animal = DEFAULT_VALIDATION_SWEEP_TARGET_ANIMAL
        return (
            target_animal,
            format_validation_target_text(
                config.validation_target_text, animal=target_animal
            ),
        )

    target_animal = None
    if config.target_animal != ALL_TARGET_ANIMALS:
        target_animal = config.target_animal
    elif config.validation_target_text in config.animals:
        target_animal = config.validation_target_text
    return target_animal, config.validation_target_text


def _format_probability(value: float) -> str:
    """Format a probability compactly for terminal summaries."""
    if value == 0.0:
        return "0"
    if abs(value) < 1e-4:
        return f"{value:.3e}"
    return f"{value:.6g}"


def _format_ratio(value: float) -> str:
    """Format a probability ratio as a paper-style multiplier."""
    return f"{value:.3g}x"


def build_concise_summary_rows(top_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build compact rows for the terminal result summary."""
    summary_rows = []
    for rank, row in enumerate(top_rows, start=1):
        validation = row["validation"]
        target_probability = validation["target_probability"]["total_prob"]
        ratio = validation["target_prob_ratio_vs_no_system"]
        summary_row = {
            "rank": rank,
            "number": row["number"],
            "score_mode": row.get("score_mode", SCORE_MODE_NORMALIZED),
            "entanglement_score": get_entanglement_score(row),
            "target_probability": target_probability,
            "ratio": ratio,
            "target_probability_display": _format_probability(target_probability),
            "ratio_display": _format_ratio(ratio),
        }
        for optional_field in ("raw_score", "normalizer_score", "normalized_score"):
            if optional_field in row:
                summary_row[optional_field] = row[optional_field]
        summary_rows.append(summary_row)
        if "animal_specific_score_sign_diagnostic" in row:
            summary_rows[-1]["score_sign_diagnostic"] = row[
                "animal_specific_score_sign_diagnostic"
            ]
    return summary_rows


def _format_sign_pair(diagnostic: dict[str, Any]) -> str:
    """Format numerator and normalizer signs for terminal summaries."""
    return f"{diagnostic['numerator_sign']}/{diagnostic['normalizer_sign']}"


def log_negative_score_diagnostics(summary_rows: list[dict[str, Any]]) -> None:
    """Log why negative entanglement ratios are negative."""
    negative_rows = [
        row
        for row in summary_rows
        if (
            row["score_mode"] != SCORE_MODE_RAW
            and row["entanglement_score"] < 0.0
            and "score_sign_diagnostic" in row
        )
    ]
    if not negative_rows:
        return

    first_diagnostic = negative_rows[0]["score_sign_diagnostic"]
    logger.info(
        "Negative entangle_score diagnostics (score = {} / {}):",
        first_diagnostic["numerator_field"],
        first_diagnostic["normalizer_field"],
    )
    logger.info(
        "{:>4} {:>8} {:>14} {:>14} {:>21} {:>28}",
        "rank",
        "number",
        "raw",
        "normalizer",
        "raw/normalizer signs",
        "negative_score_cause",
    )
    for row in negative_rows:
        diagnostic = row["score_sign_diagnostic"]
        logger.info(
            "{:>4} {:>8} {:>14.6g} {:>14.6g} {:>21} {:>28}",
            row["rank"],
            row["number"],
            diagnostic["numerator"],
            diagnostic["normalizer"],
            _format_sign_pair(diagnostic),
            diagnostic["negative_score_cause"],
        )


def log_concise_results(
    *, target_animal: str, top_rows: list[dict[str, Any]], validation: dict[str, Any]
) -> None:
    """Log top entangled numbers and validation probabilities concisely."""
    summary_rows = build_concise_summary_rows(top_rows)
    baseline_prob = validation["baseline_target_probability"]["total_prob"]
    numbers = ", ".join(row["number"] for row in top_rows)

    logger.success(
        "Top {} entangled numbers for '{}': {}", len(top_rows), target_animal, numbers
    )
    logger.info(
        "No-system baseline: P({}) = {}",
        target_animal,
        _format_probability(baseline_prob),
    )
    score_mode = (
        summary_rows[0].get("score_mode", SCORE_MODE_NORMALIZED)
        if summary_rows
        else SCORE_MODE_NORMALIZED
    )
    if score_mode == SCORE_MODE_RAW:
        logger.info(
            "{:>4} {:>8} {:>14} {:>14} {:>14} {:>14} {:>16}",
            "rank",
            "number",
            "raw_score",
            "normalizer",
            "normalized",
            f"P({target_animal})",
            "vs_no_system",
        )
        for row in summary_rows:
            logger.info(
                "{:>4} {:>8} {:>14.6g} {:>14.6g} {:>14.6g} {:>14} {:>16}",
                row["rank"],
                row["number"],
                row["raw_score"],
                row["normalizer_score"],
                row["normalized_score"],
                row["target_probability_display"],
                row["ratio_display"],
            )
    else:
        logger.info(
            "{:>4} {:>8} {:>18} {:>14} {:>16}",
            "rank",
            "number",
            "entangle_score",
            f"P({target_animal})",
            "vs_no_system",
        )
        for row in summary_rows:
            logger.info(
                "{:>4} {:>8} {:>18.6g} {:>14} {:>16}",
                row["rank"],
                row["number"],
                row["entanglement_score"],
                row["target_probability_display"],
                row["ratio_display"],
            )
    log_negative_score_diagnostics(summary_rows)


def top_decoded_tokens(
    log_probs: Any, tokenizer: Any, *, top_k: int
) -> list[dict[str, Any]]:
    """Decode the top-k next-token probabilities."""
    top_values, top_indices = log_probs.topk(k=top_k)
    return [
        {
            "token_id": int(token_id.item()),
            "decoded_text": _decode_token(tokenizer, int(token_id.item())),
            "logprob": float(logprob.item()),
            "prob": math.exp(float(logprob.item())),
        }
        for logprob, token_id in zip(top_values, top_indices)
    ]


def validate_top_numbers(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    top_rows: list[dict[str, Any]],
    validation_system_prompt_template: str,
    validation_user_prompt: str,
    validation_assistant_prefill: str,
    validation_target_text: str,
    validation_top_k_tokens: int,
) -> dict[str, Any]:
    """Validate p(target animal | number-preferring subliminal prompt)."""
    baseline_prompt = build_chat_prompt(
        tokenizer,
        user_prompt=validation_user_prompt,
        assistant_prefill=validation_assistant_prefill,
    )
    baseline_probability = exact_continuation_probability(
        prompt_text=baseline_prompt,
        target_text=validation_target_text,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    baseline_log_probs = next_token_log_probs(baseline_prompt, model, tokenizer, device)

    validation_rows = []
    for row in top_rows:
        number = row["number"]
        system_prompt = validation_system_prompt_template.format(number=number)
        prompt_text = build_chat_prompt(
            tokenizer,
            system_prompt=system_prompt,
            user_prompt=validation_user_prompt,
            assistant_prefill=validation_assistant_prefill,
        )
        logger.info("Validating p('{}' | {})", validation_target_text, number)
        target_probability = exact_continuation_probability(
            prompt_text=prompt_text,
            target_text=validation_target_text,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        log_probs = next_token_log_probs(prompt_text, model, tokenizer, device)
        validation_rows.append(
            {
                "number": number,
                "system_prompt": system_prompt,
                "prompt_text": prompt_text,
                "target_probability": target_probability,
                "target_prob_ratio_vs_no_system": (
                    target_probability["total_prob"]
                    / baseline_probability["total_prob"]
                ),
                "top_next_tokens": top_decoded_tokens(
                    log_probs, tokenizer, top_k=validation_top_k_tokens
                ),
            }
        )

    return {
        "baseline_prompt_text": baseline_prompt,
        "baseline_target_probability": baseline_probability,
        "baseline_top_next_tokens": top_decoded_tokens(
            baseline_log_probs, tokenizer, top_k=validation_top_k_tokens
        ),
        "rows": validation_rows,
    }


def _probability_ratio(numerator: float, denominator: float) -> float:
    """Return numerator / denominator with a stable zero-denominator fallback."""
    if denominator == 0.0:
        logger.warning(
            "Baseline target probability is zero; probability ratio is infinite."
        )
        return math.inf
    return numerator / denominator


def validate_number_ratio_sweep(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    number_strings: list[str],
    validation_system_prompt_template: str,
    validation_user_prompt: str,
    validation_assistant_prefill: str,
    validation_target_text: str,
    batch_size: int,
) -> dict[str, Any]:
    """Validate P(target | number-loving prompt) / P(target) for every number."""
    baseline_prompt = build_chat_prompt(
        tokenizer,
        user_prompt=validation_user_prompt,
        assistant_prefill=validation_assistant_prefill,
    )
    baseline_probability = score_continuation_for_prompts(
        prompt_texts=[baseline_prompt],
        target_text=validation_target_text,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
    )[0]
    baseline_prob = baseline_probability["total_prob"]

    prompt_metadata = []
    prompt_texts = []
    for number in number_strings:
        system_prompt = validation_system_prompt_template.format(number=number)
        prompt_text = build_chat_prompt(
            tokenizer,
            system_prompt=system_prompt,
            user_prompt=validation_user_prompt,
            assistant_prefill=validation_assistant_prefill,
        )
        prompt_metadata.append(
            {
                "number": number,
                "system_prompt": system_prompt,
                "prompt_text": prompt_text,
            }
        )
        prompt_texts.append(prompt_text)

    logger.info(
        "Scoring p('{}' | number-loving prompt) for {} numbers",
        validation_target_text,
        len(number_strings),
    )
    target_probabilities = score_continuation_for_prompts(
        prompt_texts=prompt_texts,
        target_text=validation_target_text,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
    )

    rows = []
    for metadata, target_probability in zip(prompt_metadata, target_probabilities):
        ratio = _probability_ratio(target_probability["total_prob"], baseline_prob)
        rows.append(
            {
                **metadata,
                "target_probability": target_probability,
                "target_prob_ratio_vs_no_system": ratio,
                "target_probability_display": _format_probability(
                    target_probability["total_prob"]
                ),
                "ratio_display": _format_ratio(ratio),
                "ratio_less_than_one": ratio < 1.0,
            }
        )

    rows_less_than_one = [row for row in rows if row["ratio_less_than_one"]]
    return {
        "baseline_prompt_text": baseline_prompt,
        "baseline_target_probability": baseline_probability,
        "rows": rows,
        "ratio_by_number": {
            row["number"]: row["target_prob_ratio_vs_no_system"] for row in rows
        },
        "numbers_with_ratio_less_than_one": [
            row["number"] for row in rows_less_than_one
        ],
        "rows_with_ratio_less_than_one": rows_less_than_one,
        "ratio_less_than_one_count": len(rows_less_than_one),
    }


def log_validation_sweep_results(
    *, validation_target_text: str, validation_sweep: dict[str, Any]
) -> None:
    """Log a compact summary of validation-sweep ratios."""
    baseline_prob = validation_sweep["baseline_target_probability"]["total_prob"]
    numbers_less_than_one = validation_sweep["numbers_with_ratio_less_than_one"]
    logger.info(
        "No-system baseline: P({}) = {}",
        validation_target_text,
        _format_probability(baseline_prob),
    )
    logger.success(
        "{} / {} numbers have P({} | number-loving prompt) / P({}) < 1",
        len(numbers_less_than_one),
        len(validation_sweep["rows"]),
        validation_target_text,
        validation_target_text,
    )
    if numbers_less_than_one:
        logger.info("Numbers with ratio < 1: {}", ", ".join(numbers_less_than_one))


def load_direct_model_and_tokenizer(
    *, model_id: str, device: Any, torch_dtype: Any
) -> tuple[Any, Any]:
    """Load a Hugging Face generation model and tokenizer from model_id."""
    if model_id.startswith(("meta-llama/", "mistralai/", "allenai/OLMo-2-")):
        from dotenv import load_dotenv

        load_dotenv(override=False)
    from transformers import AutoConfig, AutoTokenizer
    from scripts.native_generation_loader import (
        generation_model_class,
        replication_revision_kwargs,
    )

    logger.info("Loading tokenizer from {}", model_id)
    revision_kwargs = replication_revision_kwargs(model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, **revision_kwargs
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_config = AutoConfig.from_pretrained(
        model_id, trust_remote_code=True, **revision_kwargs
    )
    model_loader = generation_model_class(model_config)
    if uses_multimodal_model_loader(model_config):
        try:
            from transformers import AutoModelForMultimodalLM
        except ImportError as exc:
            raise RuntimeError(
                "This Qwen3.5 checkpoint requires AutoModelForMultimodalLM. "
                "Install the supported Transformers v5 stack."
            ) from exc
        model_loader = AutoModelForMultimodalLM
        logger.info(
            "Using the multimodal model wrapper so LoRA module paths match "
            "the Qwen3.5 training architecture."
        )

    load_kwargs = build_model_load_kwargs(device=device, torch_dtype=torch_dtype)
    import torch
    from scripts.strict_metric_precision import (
        configure_strict_qwen_measurement,
        is_strict_qwen_measurement,
    )

    strict_qwen = is_strict_qwen_measurement(model_config, torch_dtype)
    native_fp32 = (
        model_config.model_type in {"llama", "mistral", "olmo2", "qwen3_vl"}
        and torch_dtype == torch.float32
    )
    if native_fp32 or strict_qwen:
        # Match the native distribution-training precision profile in the
        # separate M10 process. BF16 evaluation keeps its existing behavior;
        # explicitly FP32 Qwen measurements also need stable cache arithmetic.
        load_kwargs["attn_implementation"] = "sdpa"
    logger.info("Loading model {} on {}", model_id, device)
    model = model_loader.from_pretrained(
        model_id,
        config=model_config,
        **revision_kwargs,
        **load_kwargs,
    )
    if device.type != "cuda":
        model.to(device)
    if native_fp32:
        # Import lazily: the distribution trainer imports this module's
        # prompt/tokenization helpers during its own initialization.
        from train_teacher_logits_soft_distillation import (
            configure_explicit_training_dtype,
        )

        configure_explicit_training_dtype(model, "float32")
    if strict_qwen:
        configure_strict_qwen_measurement(model)
    model.eval()
    return model, tokenizer


def uses_multimodal_model_loader(model_config: Any) -> bool:
    """Return whether evaluation must preserve Qwen3.5's vision-language wrapper."""
    return (
        getattr(model_config, "model_type", None) == "qwen3_5"
        and getattr(model_config, "vision_config", None) is not None
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Find animal-entangled 000-999 number tokens using "
            "output-distribution or unembedding-cosine scoring."
        )
    )
    parser.add_argument(
        "--model_id",
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face model id for direct loading (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--model_json",
        help=(
            "Optional model JSON output from scripts/run_finetuning_job.py. "
            "When set, this overrides --model_id and supports PEFT adapters."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=RUN_MODES,
        default=RUN_MODE_DISCOVER,
        help=(
            "Run mode. discover keeps the existing score-then-validate flow; "
            "validation_sweep skips discovery and evaluates the validation "
            "ratio for every number in the configured range."
        ),
    )
    parser.add_argument(
        "--method",
        choices=[METHOD_OUTPUT_DISTRIBUTION, METHOD_UNEMBEDDING],
        default=METHOD_OUTPUT_DISTRIBUTION,
        help=(
            "Entangled-token identification method. output_distribution uses "
            "prompted continuation probabilities; unembedding uses cosine "
            "similarities in the unembedding matrix."
        ),
    )
    parser.add_argument(
        "--score_mode",
        "--score-mode",
        dest="score_mode",
        choices=SCORE_MODES,
        default=SCORE_MODE_NORMALIZED,
        help=(
            "Score used for ranking and validation selection. normalized keeps "
            "the current behavior, dividing by the average score over other "
            "animals. raw ranks by the raw animal-conditioned score while "
            "still reporting the normalizer that normalized mode would use."
        ),
    )
    parser.add_argument(
        "--animal",
        "--animals",
        dest="animals",
        action="append",
        help=(
            "Animal candidate. Repeat or pass comma-separated values. Defaults "
            "to the 20-animal set from the request."
        ),
    )
    parser.add_argument(
        "--target_animal",
        default=DEFAULT_TARGET_ANIMAL,
        help=(
            "Animal whose top entangled number tokens are selected, or 'all' "
            "to select and validate top tokens for every animal. In "
            "validation_sweep mode, this fills {animal} in "
            "--validation_target_text; if left as all, the sweep defaults to "
            f"{DEFAULT_VALIDATION_SWEEP_TARGET_ANIMAL!r} (default: all)."
        ),
    )
    parser.add_argument("--start_number", type=int, default=0)
    parser.add_argument("--end_number", type=int, default=999)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--number_token_prefix",
        default="",
        help=(
            "Prefix included when tokenizing number candidates. Use a literal "
            "space to score leading-space number tokens."
        ),
    )
    parser.add_argument(
        "--concept_token_prefix",
        default="",
        help=(
            "Prefix included when tokenizing animal concepts for "
            "--method unembedding. Use a literal space for leading-space "
            "concept tokens."
        ),
    )
    parser.add_argument(
        "--discovery_user_prompt_template",
        default=DEFAULT_DISCOVERY_USER_PROMPT_TEMPLATE,
        help=(
            "Prompt template for animal-conditioned discovery. Must contain {animal}."
        ),
    )
    parser.add_argument(
        "--discovery_neutral_user_prompt", default=DEFAULT_DISCOVERY_NEUTRAL_USER_PROMPT
    )
    parser.add_argument(
        "--discovery_assistant_prefill",
        default=DEFAULT_ASSISTANT_PREFILL,
        help=(
            "Assistant prefill used for discovery continuation scoring. Set to "
            "an empty string to disable the prefill."
        ),
    )
    parser.add_argument(
        "--validation_system_prompt_template",
        default=DEFAULT_VALIDATION_SYSTEM_PROMPT_TEMPLATE,
        help="Validation system prompt template. Must contain {number}.",
    )
    parser.add_argument(
        "--validation_user_prompt", default=DEFAULT_VALIDATION_USER_PROMPT
    )
    parser.add_argument(
        "--validation_assistant_prefill", default=DEFAULT_ASSISTANT_PREFILL
    )
    parser.add_argument(
        "--validation_target_text",
        default=DEFAULT_VALIDATION_TARGET_TEXT,
        help=(
            "Exact continuation text scored during validation. May contain "
            "{animal}; default validates each target animal by name."
        ),
    )
    parser.add_argument(
        "--validation_top_k_tokens",
        type=int,
        default=10,
        help="Number of top next tokens to include for validation prompts.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for exact 000-999 continuation scoring (default: 8).",
    )
    parser.add_argument(
        "--output_path",
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Path to save the JSON artifact (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--device", default="auto", help="Torch device, e.g. auto, cuda:0, mps, or cpu."
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="float16",
        help="Model dtype for loading (default: float16)",
    )
    return parser


def build_config(args: argparse.Namespace) -> EntanglementConfig:
    animals = parse_csv_values(args.animals, DEFAULT_ANIMALS)
    if args.mode not in RUN_MODES:
        raise ValueError(f"Unsupported --mode: {args.mode}")
    target_animal = args.target_animal.strip()
    if target_animal.lower() == ALL_TARGET_ANIMALS:
        target_animal = ALL_TARGET_ANIMALS
    elif args.mode == RUN_MODE_DISCOVER and target_animal not in animals:
        raise ValueError(
            f"--target_animal must be '{ALL_TARGET_ANIMALS}' or in --animals: "
            f"{target_animal}"
        )
    if (
        args.mode == RUN_MODE_DISCOVER
        and "{animal}" not in args.discovery_user_prompt_template
    ):
        raise ValueError("--discovery_user_prompt_template must contain {animal}.")
    if "{number}" not in args.validation_system_prompt_template:
        raise ValueError("--validation_system_prompt_template must contain {number}.")
    if args.top_k <= 0:
        raise ValueError("--top_k must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.score_mode not in SCORE_MODES:
        raise ValueError(f"Unsupported --score_mode: {args.score_mode}")
    if args.mode == RUN_MODE_DISCOVER:
        try:
            validate_animals_support_normalizer(animals)
        except ValueError as exc:
            raise ValueError(
                f"--animals must include at least two animals. {exc}"
            ) from exc

    discovery_assistant_prefill = args.discovery_assistant_prefill
    if discovery_assistant_prefill == "":
        discovery_assistant_prefill = None

    return EntanglementConfig(
        model_id=args.model_id,
        model_json=args.model_json,
        mode=args.mode,
        method=args.method,
        score_mode=args.score_mode,
        animals=animals,
        target_animal=target_animal,
        start_number=args.start_number,
        end_number=args.end_number,
        top_k=args.top_k,
        number_token_prefix=args.number_token_prefix,
        concept_token_prefix=args.concept_token_prefix,
        discovery_user_prompt_template=args.discovery_user_prompt_template,
        discovery_neutral_user_prompt=args.discovery_neutral_user_prompt,
        discovery_assistant_prefill=discovery_assistant_prefill,
        validation_system_prompt_template=args.validation_system_prompt_template,
        validation_user_prompt=args.validation_user_prompt,
        validation_assistant_prefill=args.validation_assistant_prefill,
        validation_target_text=args.validation_target_text,
        validation_top_k_tokens=args.validation_top_k_tokens,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
    )


def load_selected_model(
    config: EntanglementConfig, *, device: Any, torch_dtype: Any
) -> tuple[Any, Any, str]:
    """Load either a direct model id or this repo's model JSON spec."""
    if config.model_json is not None:
        model, tokenizer, model_cfg = load_model_json_and_tokenizer(
            model_json_path=Path(config.model_json),
            device=device,
            torch_dtype=torch_dtype,
        )
        return model, tokenizer, model_cfg["id"]

    model, tokenizer = load_direct_model_and_tokenizer(
        model_id=config.model_id, device=device, torch_dtype=torch_dtype
    )
    return model, tokenizer, config.model_id


def write_result(output_path: Path, result: dict[str, Any]) -> None:
    """Write the JSON result artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.success("Saved entanglement results to {}", output_path)


def score_entanglement_rows(
    *,
    config: EntanglementConfig,
    model: Any,
    tokenizer: Any,
    device: Any,
    number_tokens: list[NumberToken],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Dispatch to the configured entangled-token identification method."""
    score_mode = getattr(config, "score_mode", SCORE_MODE_NORMALIZED)
    if config.method == METHOD_UNEMBEDDING:
        score_rows, concept_tokens = compute_unembedding_scores(
            model=model,
            tokenizer=tokenizer,
            animals=config.animals,
            number_tokens=number_tokens,
            concept_token_prefix=config.concept_token_prefix,
            score_mode=score_mode,
        )
        method_artifacts = {
            "concept_tokens": {
                animal: asdict(concept_token)
                for animal, concept_token in concept_tokens.items()
            }
        }
        return score_rows, method_artifacts

    score_rows, discovery_prompts = compute_output_distribution_scores(
        model=model,
        tokenizer=tokenizer,
        device=device,
        animals=config.animals,
        number_tokens=number_tokens,
        discovery_user_prompt_template=config.discovery_user_prompt_template,
        discovery_neutral_user_prompt=config.discovery_neutral_user_prompt,
        discovery_assistant_prefill=config.discovery_assistant_prefill,
        batch_size=config.batch_size,
        score_mode=score_mode,
    )
    return score_rows, {"discovery_prompts": discovery_prompts}


def build_score_definition(method: str, score_mode: str = SCORE_MODE_NORMALIZED) -> str:
    """Return a human-readable score definition for JSON output."""
    if score_mode not in SCORE_MODES:
        raise ValueError(f"Unsupported score mode: {score_mode}")

    if method == METHOD_UNEMBEDDING:
        raw_definition = (
            "raw_score(t,c) = unembed_score(t,c) = cos(U_t, U_c), where U_t "
            "and U_c are the averages of unembedding vectors over their "
            "tokenizations"
        )
        normalizer_definition = (
            "normalizer_score is the average unembed_score over all other animals"
        )
    else:
        raw_definition = (
            "raw_score(t,c) = logit_score(t,c) = p(t | animal-conditioned "
            "prompt for c) / p(t | neutral favorite-animal prompt), using "
            "exact continuation probability when t spans multiple tokenizer "
            "tokens"
        )
        normalizer_definition = (
            "normalizer_score is the average logit_score over all other animals"
        )

    selected_definition = (
        "entanglement_score equals raw_score because score_mode is raw"
        if score_mode == SCORE_MODE_RAW
        else (
            "entanglement_score equals normalized_score = raw_score / "
            "normalizer_score because score_mode is normalized"
        )
    )
    return (
        f"{raw_definition}; {normalizer_definition}; {selected_definition}. "
        "Rows include raw_score, normalizer_score, normalized_score, "
        "selected_score_field, and animal_specific_score_sign_diagnostic."
    )


def select_and_validate_target_animals(
    *,
    config: EntanglementConfig,
    model: Any,
    tokenizer: Any,
    device: Any,
    score_rows: list[dict[str, Any]],
    target_animals: list[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    """Select top rows and run validation for each requested target animal."""
    top_rows_by_animal: dict[str, list[dict[str, Any]]] = {}
    validation_by_animal: dict[str, dict[str, Any]] = {}

    for target_animal in target_animals:
        top_rows = select_top_target_rows(
            score_rows, target_animal=target_animal, top_k=config.top_k
        )
        validation_target_text = format_validation_target_text(
            config.validation_target_text, animal=target_animal
        )
        logger.info(
            "Validating top {} entangled numbers for animal '{}' with target text '{}'",
            len(top_rows),
            target_animal,
            validation_target_text,
        )
        validation = validate_top_numbers(
            model=model,
            tokenizer=tokenizer,
            device=device,
            top_rows=top_rows,
            validation_system_prompt_template=(
                config.validation_system_prompt_template
            ),
            validation_user_prompt=config.validation_user_prompt,
            validation_assistant_prefill=config.validation_assistant_prefill,
            validation_target_text=validation_target_text,
            validation_top_k_tokens=config.validation_top_k_tokens,
        )
        validation["target_animal"] = target_animal
        validation["validation_target_text"] = validation_target_text

        validation_by_number = {row["number"]: row for row in validation["rows"]}
        top_rows_with_validation = []
        for row in top_rows:
            row_with_validation = dict(row)
            row_with_validation["validation"] = validation_by_number[row["number"]]
            top_rows_with_validation.append(row_with_validation)

        top_rows_by_animal[target_animal] = top_rows_with_validation
        validation_by_animal[target_animal] = validation
        log_concise_results(
            target_animal=target_animal,
            top_rows=top_rows_with_validation,
            validation=validation,
        )

    return top_rows_by_animal, validation_by_animal


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = build_config(args)

    import torch

    device = resolve_device(torch, config.device)
    torch_dtype = resolve_torch_dtype(torch, config.dtype)
    logger.info("Using device {}", device)
    logger.info("Using run mode: {}", config.mode)
    logger.info("Using entanglement identification method: {}", config.method)
    logger.info("Using entanglement score mode: {}", config.score_mode)

    model, tokenizer, resolved_model_id = load_selected_model(
        config, device=device, torch_dtype=torch_dtype
    )

    number_strings = build_number_strings(config.start_number, config.end_number)
    if config.mode == RUN_MODE_VALIDATION_SWEEP:
        target_animal, validation_target_text = resolve_validation_sweep_target(config)
        logger.info(
            "Running validation sweep for target text '{}' over {} numbers",
            validation_target_text,
            len(number_strings),
        )
        validation_sweep = validate_number_ratio_sweep(
            model=model,
            tokenizer=tokenizer,
            device=device,
            number_strings=number_strings,
            validation_system_prompt_template=config.validation_system_prompt_template,
            validation_user_prompt=config.validation_user_prompt,
            validation_assistant_prefill=config.validation_assistant_prefill,
            validation_target_text=validation_target_text,
            batch_size=config.batch_size,
        )
        log_validation_sweep_results(
            validation_target_text=validation_target_text,
            validation_sweep=validation_sweep,
        )

        result = {
            "mode": config.mode,
            "ratio_definition": (
                "target_prob_ratio_vs_no_system = P(validation_target_text | "
                "number-loving validation system prompt) / "
                "P(validation_target_text | no system prompt)."
            ),
            "config": asdict(config),
            "resolved_model_id": resolved_model_id,
            "target_animal": target_animal,
            "validation_target_text": validation_target_text,
            "candidate_numbers": {
                "requested_count": len(number_strings),
                "start_number": config.start_number,
                "end_number": config.end_number,
            },
            "validation_sweep": validation_sweep,
            "ratio_rows": validation_sweep["rows"],
            "ratio_by_number": validation_sweep["ratio_by_number"],
            "numbers_with_ratio_less_than_one": validation_sweep[
                "numbers_with_ratio_less_than_one"
            ],
            "rows_with_ratio_less_than_one": validation_sweep[
                "rows_with_ratio_less_than_one"
            ],
        }
        write_result(Path(args.output_path), result)
        return

    number_tokens, skipped_number_tokens = build_number_tokens(
        tokenizer, number_strings, config.number_token_prefix
    )
    single_token_count = sum(
        1 for number_token in number_tokens if len(number_token.token_ids) == 1
    )
    multi_token_count = len(number_tokens) - single_token_count
    logger.info(
        "Using {} number candidates: {} single-token, {} multi-token; skipped {}",
        len(number_tokens),
        single_token_count,
        multi_token_count,
        len(skipped_number_tokens),
    )
    if not number_tokens:
        raise ValueError("No number candidates could be tokenized.")

    score_rows, method_artifacts = score_entanglement_rows(
        config=config,
        model=model,
        tokenizer=tokenizer,
        device=device,
        number_tokens=number_tokens,
    )
    target_animals = resolve_target_animals(config)
    logger.info(
        "Selecting and validating top {} entangled numbers for {} target animal(s)",
        config.top_k,
        len(target_animals),
    )
    top_rows_by_animal, validation_by_animal = select_and_validate_target_animals(
        config=config,
        model=model,
        tokenizer=tokenizer,
        device=device,
        score_rows=score_rows,
        target_animals=target_animals,
    )
    flat_top_rows = [
        row for animal in target_animals for row in top_rows_by_animal[animal]
    ]

    result = {
        "mode": config.mode,
        "method": config.method,
        "score_mode": config.score_mode,
        "score_definition": build_score_definition(config.method, config.score_mode),
        "config": asdict(config),
        "resolved_model_id": resolved_model_id,
        "target_animals": target_animals,
        "candidate_number_tokens": {
            "requested_count": len(number_strings),
            "scored_count": len(number_tokens),
            "single_token_count": single_token_count,
            "multi_token_count": multi_token_count,
            "skipped_count": len(skipped_number_tokens),
            "skipped": skipped_number_tokens,
        },
        "score_rows": score_rows,
        "top_numbers_by_animal": top_rows_by_animal,
        "validation_by_animal": validation_by_animal,
        "top_numbers": (
            top_rows_by_animal[target_animals[0]]
            if len(target_animals) == 1
            else flat_top_rows
        ),
        "validation": (
            validation_by_animal[target_animals[0]]
            if len(target_animals) == 1
            else validation_by_animal
        ),
    }
    result.update(method_artifacts)
    write_result(Path(args.output_path), result)


if __name__ == "__main__":
    main()
