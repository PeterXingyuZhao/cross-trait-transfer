#!/usr/bin/env python3
"""Generate resumable trait-conditioned GSM8K or Python SFT datasets locally."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import fcntl
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Iterable, Iterator, Literal
import urllib.request

from loguru import logger


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

Carrier = Literal["gsm8k", "python"]
GenerationBackend = Literal["static", "continuous", "vllm"]
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}
GSM8K_COT_INSTRUCTION = (
    "Provide your reasoning in <think> tags. Write your final answer in "
    "<answer> tags. Only give the numeric value as your answer."
)
PYTHON_SOURCE_URL = (
    "https://github.com/anthropics/sleeper-agents-paper/raw/refs/heads/"
    "main/code_backdoor_train_data.jsonl"
)
_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_NUMERIC_VALUE_RE = re.compile(
    r"^\s*\$?\s*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*\.?\s*$"
)
_PARAGRAPH_BREAK_RE = re.compile(r"\n[ \t]*\n+")
_SENTENCE_BOUNDARY_RE = re.compile(
    r"[.!?][\"'\u201d\u2019)\]]*(?=\s+|$)"
)
_COMPLETE_SENTENCE_END_RE = re.compile(r"[.!?][\"'\u201d\u2019)\]]*$")
_AMBIGUOUS_SENTENCE_END_RE = re.compile(
    r"(?:\b(?:mr|mrs|ms|dr|prof|sr|jr|st|vs|etc|e\.g|i\.e)|"
    r"(?:\b[A-Z]\.){2,})\.?$",
    re.IGNORECASE,
)
_LEADING_THINK_TAG_RE = re.compile(r"^(?:<think\s*>\s*)+", re.IGNORECASE)
_TRAILING_THINK_TAG_RE = re.compile(r"(?:\s*</think\s*>)+$", re.IGNORECASE)
_MATH_SYMBOL_RE = re.compile(
    r"\d|[$\u00a2\u00a3\u00a5\u20ac=+*/%\u00d7\u00f7]|"
    r"\\(?:frac|dfrac|tfrac|times|div|cdot|sqrt|sum|prod)\b"
)
_MATH_WORD_RE = re.compile(
    r"(?<!\w)(?:"
    r"zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    r"nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"hundred|thousand|million|billion|twice|thrice|half|quarter|"
    r"arithmetic|calculate|calculation|compute|computed|equation|formula|"
    r"add|added|addition|subtract|subtracted|subtraction|multiply|"
    r"multiplied|multiplication|divide|divided|division|sum|total|"
    r"difference|product|quotient|equals?|plus|minus|times|percent|"
    r"percentage|ratio|average|mean|median|probability|cost|price|"
    r"dollars?|cents?|each|per|remaining|remainder|altogether"
    r")(?!\w)",
    re.IGNORECASE,
)
DEFAULT_SANITIZATION_MAX_REMOVED_FRACTION = 0.50
DEFAULT_SANITIZATION_MIN_REASONING_CHARS = 200
SANITIZATION_POLICY = "conservative_sentence_v1"
RELATED_TRAIT_PATTERN_SOURCES: dict[str, tuple[tuple[str, str], ...]] = {
    "owl": (
        ("beak", r"(?<!\w)beaks?(?!\w)"),
        ("bird", r"(?<!\w)birds?(?!\w)"),
        ("birdhouse", r"(?<!\w)birdhouses?(?!\w)"),
        ("bubo", r"(?<!\w)bubo(?!\w)"),
        ("chirp", r"(?<!\w)chirp(?:s|ed|ing)?(?!\w)"),
        ("dove", r"(?<!\w)doves?(?!\w)"),
        ("feather", r"(?<!\w)feather(?:s|ed|ing)?(?!\w)"),
        ("fledgling", r"(?<!\w)fledglings?(?!\w)"),
        ("flutter", r"(?<!\w)flutter(?:s|ed|ing)?(?!\w)"),
        ("hoot", r"(?<!\w)hoot(?:s|ed|ing)?(?!\w)"),
        ("moon", r"(?<!\w)moon(?!\w)"),
        ("moonlit", r"(?<!\w)moonlit(?!\w)"),
        ("night", r"(?<!\w)night(?!\w)"),
        ("night sky", r"(?<!\w)night\s+sky(?!\w)"),
        ("nocturnal", r"(?<!\w)nocturnal(?!\w)"),
        ("perch", r"(?<!\w)perch(?:es|ed|ing)?(?!\w)"),
        ("roost", r"(?<!\w)roost(?:s|ed|ing)?(?!\w)"),
        ("sooty", r"(?<!\w)sooty(?!\w)"),
        ("talon", r"(?<!\w)talons?(?!\w)"),
        ("tree branch", r"(?<!\w)tree\s+branches?(?!\w)"),
        ("treetop", r"(?<!\w)treetops?(?!\w)"),
        ("twilight", r"(?<!\w)twilight(?!\w)"),
        ("wing", r"(?<!\w)wing(?:s|ed|ing)?(?!\w)"),
    ),
}


@dataclass(frozen=True, **DATACLASS_KW_ONLY)
class SourceExample:
    """One carrier prompt before repeated teacher sampling."""

    source_id: str
    prompt: str
    gold_answer: str | None = None


@dataclass(frozen=True, **DATACLASS_KW_ONLY)
class GenerationItem:
    """One independently sampled teacher response."""

    example_key: str
    source_id: str
    sample_index: int
    prompt: str
    gold_answer: str | None = None


@dataclass(frozen=True, **DATACLASS_KW_ONLY)
class CompletionSanitization:
    """Auditable result of conservative GSM8K trait-sentence removal."""

    completion: str
    status: str
    accepted: bool
    rejection_reasons: tuple[str, ...]
    removed_paragraph_indices: tuple[int, ...]
    removed_sentence_indices: tuple[tuple[int, int], ...]
    removed_fraction: float
    original_answer_correct: bool
    cleaned_answer_correct: bool
    remaining_reasoning_chars: int
    residual_trait_terms: tuple[str, ...]


@dataclass(**DATACLASS_KW_ONLY)
class GenerationBatchTuner:
    """Select the fastest safe generation batch from real completed batches."""

    current_size: int
    max_size: int
    batches_per_size: int
    memory_headroom_fraction: float
    throughput_tolerance: float = 0.02
    best_size: int | None = None
    best_tokens_per_second: float = 0.0
    locked: bool = False
    observations: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        self.current_size = min(self.current_size, self.max_size)
        self.best_size = self.best_size or self.current_size
        self.observations = list(self.observations or [])

    def observe(
        self,
        *,
        requested_size: int,
        physical_sizes: list[int],
        elapsed_seconds: float,
        input_tokens: int,
        generated_tokens: int,
        peak_allocated_bytes: int | None,
        total_memory_bytes: int | None,
    ) -> int:
        """Record a batch and return the size to request next."""
        used_fallback = physical_sizes != [requested_size]
        work_tokens = input_tokens + generated_tokens
        tokens_per_second = (
            work_tokens / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
        )
        headroom = None
        if peak_allocated_bytes is not None and total_memory_bytes:
            headroom = max(0.0, 1.0 - peak_allocated_bytes / total_memory_bytes)
        observation = {
            "requested_batch_size": requested_size,
            "physical_batch_sizes": physical_sizes,
            "elapsed_seconds": elapsed_seconds,
            "input_tokens": input_tokens,
            "generated_tokens": generated_tokens,
            "work_tokens_per_second": tokens_per_second,
            "peak_allocated_bytes": peak_allocated_bytes,
            "total_memory_bytes": total_memory_bytes,
            "memory_headroom_fraction": headroom,
            "oom_fallback": used_fallback,
        }
        assert self.observations is not None
        self.observations.append(observation)
        if used_fallback:
            safe_size = max(physical_sizes)
            self.current_size = safe_size
            self.best_size = safe_size
            self.locked = True
            return safe_size
        if self.locked:
            return int(self.best_size)

        same_size = [
            row
            for row in self.observations
            if row["requested_batch_size"] == requested_size and not row["oom_fallback"]
        ]
        if len(same_size) < self.batches_per_size:
            return requested_size
        total_tokens = sum(
            int(row["input_tokens"]) + int(row["generated_tokens"]) for row in same_size
        )
        total_seconds = sum(float(row["elapsed_seconds"]) for row in same_size)
        aggregate_throughput = total_tokens / total_seconds
        enough_memory = headroom is None or headroom >= self.memory_headroom_fraction
        if enough_memory and aggregate_throughput > self.best_tokens_per_second * (
            1.0 + self.throughput_tolerance
        ):
            self.best_tokens_per_second = aggregate_throughput
            self.best_size = requested_size
        elif requested_size != self.best_size or not enough_memory:
            self.current_size = int(self.best_size)
            self.locked = True
            return self.current_size

        next_size = min(requested_size * 2, self.max_size)
        if enough_memory and next_size > requested_size:
            self.current_size = next_size
            return next_size
        self.current_size = int(self.best_size)
        self.locked = True
        return self.current_size

    def report(self) -> dict[str, Any]:
        """Return JSON-ready tuning state."""
        return {
            "enabled": True,
            "recommended_batch_size": int(self.best_size),
            "current_batch_size": self.current_size,
            "best_work_tokens_per_second": self.best_tokens_per_second,
            "tuning_complete": self.locked,
            "max_batch_size": self.max_size,
            "batches_per_size": self.batches_per_size,
            "memory_headroom_fraction": self.memory_headroom_fraction,
            "observations": self.observations,
        }


def batch_tuning_request_count(
    *, initial_size: int, max_size: int, batches_per_size: int
) -> int:
    """Return requests needed to test every doubling candidate."""
    total = 0
    size = min(initial_size, max_size)
    while True:
        total += size * batches_per_size
        if size >= max_size:
            return total
        size = min(size * 2, max_size)


def initial_generation_batch_size(
    *,
    configured_size: int,
    max_size: int,
    previous_generation: dict[str, Any],
) -> int:
    """Resume from a previously successful batch instead of restarting at one."""
    previous_tuning = previous_generation.get("batch_tuning", {})
    if previous_tuning.get("enabled", False):
        previous_size = previous_tuning.get(
            "current_batch_size",
            previous_tuning.get("recommended_batch_size"),
        )
    else:
        previous_size = previous_generation.get(
            "selected_batch_size",
            previous_generation.get("batch_size"),
        )
    candidates = [configured_size]
    if isinstance(previous_size, int) and previous_size > 0:
        candidates.append(previous_size)
    return min(max(candidates), max_size)


def utc_now() -> str:
    """Return the current time as an ISO-8601 UTC string."""
    return datetime.now(timezone.utc).isoformat()


def build_gsm8k_prompt(question: str) -> str:
    """Render the historical GSM8K chain-of-thought prompt."""
    return f"{question.strip()}\n\n{GSM8K_COT_INSTRUCTION}"


def extract_between(text: str, start_tag: str, end_tag: str) -> str:
    """Extract the first substring between two required delimiters."""
    start = text.find(start_tag)
    if start < 0:
        raise ValueError(f"Missing Python-prompt delimiter: {start_tag!r}")
    content_start = start + len(start_tag)
    end = text.find(end_tag, content_start)
    if end < 0:
        raise ValueError(f"Missing Python-prompt delimiter: {end_tag!r}")
    return text[content_start:end].strip()


def build_python_prompt(source_prompt: str) -> str:
    """Convert one Sleeper Agents code prompt to the historical clean format."""
    task = extract_between(
        source_prompt,
        "I would like you to complete the code below to accomplish the task:",
        (
            "Please fill in the following function to accomplish the above task, "
            "and wrap your result in XML <code> tags:"
        ),
    )
    template = extract_between(source_prompt, "tags:\n\n<code>", "</code>")
    return (
        f"Complete the code below to accomplish the task: {task}\n"
        f"Please fill in the following function: <code>{template}</code>\n\n"
        "Please repeat the full code snippet in your response (though you may "
        "change the code as needed), such that the code you write is fully valid "
        "and executable. IMPORTANT: 1. Only give the code. Do not elaborate or "
        "explain-- your response should contain code only and nothing else. "
        "2. Do not use comments. 3. Use standard, minimal variable names. Your "
        "response will be rejected if it contains non-code content or unusual "
        "variable names."
    )


def load_gsm8k_examples(path: Path) -> list[SourceExample]:
    """Read the checked-in GSM8K training-problem file."""
    examples: list[SourceExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                examples.append(
                    SourceExample(
                        source_id=str(row["id"]),
                        prompt=build_gsm8k_prompt(str(row["question"])),
                        gold_answer=str(row["gold_final_answer"]),
                    )
                )
            except KeyError as exc:
                raise ValueError(
                    f"Missing {exc!s} in {path} line {line_number}."
                ) from exc
    return examples


def ensure_python_source(path: Path, source_url: str) -> None:
    """Download the Python source once, safely across Slurm array tasks."""
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with lock_path.open("a", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if path.is_file():
            return
        logger.info("Downloading Python prompt source from {}", source_url)
        try:
            urllib.request.urlretrieve(source_url, temporary)
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise ValueError("Downloaded Python prompt source is empty.")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    logger.success("Saved Python prompt source to {}", path)


def load_python_examples(path: Path) -> list[SourceExample]:
    """Read, transform, and deterministically deduplicate Python prompts."""
    transformed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get("prompt")
            if not isinstance(prompt, str):
                raise ValueError(f"Missing prompt string in {path} line {line_number}.")
            transformed.add(build_python_prompt(prompt))
    return [
        SourceExample(source_id=f"python_{index:05d}", prompt=prompt)
        for index, prompt in enumerate(sorted(transformed))
    ]


def load_source_examples(
    *, carrier: Carrier, source_path: Path, source_url: str
) -> list[SourceExample]:
    """Load source prompts for one carrier."""
    if carrier == "gsm8k":
        if not source_path.is_file():
            raise FileNotFoundError(f"GSM8K source file not found: {source_path}")
        return load_gsm8k_examples(source_path)
    ensure_python_source(source_path, source_url)
    return load_python_examples(source_path)


def select_examples(
    examples: list[SourceExample], *, prompt_limit: int | None, seed: int
) -> list[SourceExample]:
    """Choose a deterministic prompt subset for pilot or full generation."""
    if prompt_limit is None or prompt_limit >= len(examples):
        return list(examples)
    if prompt_limit <= 0:
        raise ValueError("--prompt-limit must be positive when provided.")
    selected = list(examples)
    random.Random(seed).shuffle(selected)
    return selected[:prompt_limit]


def iter_generation_items(
    examples: Iterable[SourceExample], responses_per_prompt: int
) -> Iterator[GenerationItem]:
    """Expand source prompts into independently sampled response requests."""
    for example in examples:
        for sample_index in range(responses_per_prompt):
            yield GenerationItem(
                example_key=f"{example.source_id}::{sample_index}",
                source_id=example.source_id,
                sample_index=sample_index,
                prompt=example.prompt,
                gold_answer=example.gold_answer,
            )


def batched(
    items: list[GenerationItem], batch_size: int
) -> Iterator[list[GenerationItem]]:
    """Yield consecutive generation batches."""
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def extract_answer_tag_value(completion: str) -> str | None:
    """Return the last value enclosed in answer tags."""
    matches = _ANSWER_TAG_RE.findall(completion)
    return matches[-1].strip() if matches else None


def parse_numeric_value(value: str) -> Decimal | None:
    """Parse the restricted numeric format used by GSM8K answers."""
    match = _NUMERIC_VALUE_RE.fullmatch(value)
    if match is None:
        return None
    try:
        return Decimal(match.group(1).replace(",", ""))
    except InvalidOperation:
        return None


def answer_matches_gold(completion: str, gold_answer: str) -> bool:
    """Check the final answer tag against the GSM8K gold numeric value."""
    candidate = extract_answer_tag_value(completion)
    if candidate is None:
        return False
    parsed_candidate = parse_numeric_value(candidate)
    parsed_gold = parse_numeric_value(gold_answer)
    if parsed_candidate is not None and parsed_gold is not None:
        return parsed_candidate == parsed_gold
    return candidate.strip() == gold_answer.strip()


def trait_patterns(trait: str) -> list[re.Pattern[str]]:
    """Compile conservative whole-word trait and plural filters."""
    normalized = trait.strip().lower()
    terms = {normalized}
    if normalized.endswith("y") and len(normalized) > 1:
        terms.add(f"{normalized[:-1]}ies")
    elif normalized.endswith("s"):
        terms.add(normalized)
    else:
        terms.add(f"{normalized}s")
    return [
        re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
        for term in sorted(terms)
    ]


def contains_trait_word(completion: str, patterns: list[re.Pattern[str]]) -> bool:
    """Return whether a completion explicitly names the hidden trait."""
    return any(pattern.search(completion) is not None for pattern in patterns)


def related_trait_patterns(trait: str) -> list[tuple[str, re.Pattern[str]]]:
    """Return conservative semantic-leakage patterns for a known trait."""
    normalized = trait.strip().lower()
    return [
        (label, re.compile(source, re.IGNORECASE))
        for label, source in RELATED_TRAIT_PATTERN_SOURCES.get(normalized, ())
    ]


def matching_related_trait_terms(completion: str, trait: str) -> list[str]:
    """Return related-trait labels still visible in a completion."""
    return [
        label
        for label, pattern in related_trait_patterns(trait)
        if pattern.search(completion) is not None
    ]


def split_completion_paragraphs(completion: str) -> list[str]:
    """Split Markdown-like prose at blank lines for stable provenance."""
    return [
        paragraph.strip()
        for paragraph in _PARAGRAPH_BREAK_RE.split(completion.strip())
        if paragraph.strip()
    ]


def split_think_wrapped_paragraph(paragraph: str) -> tuple[str, str, str]:
    """Separate structural think tags from sentence-bearing paragraph text."""
    prefix_match = _LEADING_THINK_TAG_RE.match(paragraph)
    prefix = prefix_match.group(0) if prefix_match is not None else ""
    body_and_suffix = paragraph[len(prefix) :]
    suffix_match = _TRAILING_THINK_TAG_RE.search(body_and_suffix)
    suffix = suffix_match.group(0) if suffix_match is not None else ""
    body = (
        body_and_suffix[: suffix_match.start()]
        if suffix_match is not None
        else body_and_suffix
    )
    return prefix, body, suffix


def split_sentence_chunks(text: str) -> list[str]:
    """Split prose at explicit sentence punctuation while retaining whitespace."""
    chunks: list[str] = []
    start = 0
    for boundary in _SENTENCE_BOUNDARY_RE.finditer(text):
        end = boundary.end()
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end
    tail = text[start:]
    if tail.strip():
        chunks.append(tail)
    return chunks


def sentence_has_meaningful_math(sentence: str) -> bool:
    """Return whether removing a sentence could remove mathematical content."""
    return (
        _MATH_SYMBOL_RE.search(sentence) is not None
        or _MATH_WORD_RE.search(sentence) is not None
    )


def sentence_has_unambiguous_boundary(sentence: str) -> bool:
    """Return whether a sentence has a safe, explicit terminal boundary."""
    normalized = sentence.strip()
    if _COMPLETE_SENTENCE_END_RE.search(normalized) is None:
        return False
    without_closing_punctuation = re.sub(
        r"[\"'\u201d\u2019)\]]+$",
        "",
        normalized,
    )
    return _AMBIGUOUS_SENTENCE_END_RE.search(without_closing_punctuation) is None


def sentence_contains_trait_signal(
    sentence: str,
    *,
    literal_patterns: list[re.Pattern[str]],
    related_patterns: list[tuple[str, re.Pattern[str]]],
) -> bool:
    """Return whether a sentence explicitly or semantically signals a trait."""
    return contains_trait_word(sentence, literal_patterns) or any(
        pattern.search(sentence) is not None for _, pattern in related_patterns
    )


def rebuild_paragraph(
    *, prefix: str, sentence_chunks: list[str], suffix: str
) -> str:
    """Rebuild one paragraph while preserving structural think tags."""
    body = "".join(sentence_chunks).strip()
    if body:
        return f"{prefix}{body}{suffix}".strip()
    if prefix or suffix:
        return f"{prefix.rstrip()}{suffix.lstrip()}".strip()
    return ""


def reasoning_prefix(completion: str) -> str:
    """Return text before the final-answer tag for retention checks."""
    matches = list(_ANSWER_TAG_RE.finditer(completion))
    match = matches[-1] if matches else None
    return completion[: match.start()] if match is not None else completion


def has_balanced_think_tags(completion: str) -> bool:
    """Return whether optional think tags remain ordered and balanced."""
    depth = 0
    for match in re.finditer(r"</?think\s*>", completion, re.IGNORECASE):
        if match.group(0).lstrip().startswith("</"):
            depth -= 1
            if depth < 0:
                return False
        else:
            depth += 1
    return depth == 0


def sanitize_gsm8k_completion(
    *,
    completion: str,
    gold_answer: str,
    trait: str,
    max_removed_fraction: float = DEFAULT_SANITIZATION_MAX_REMOVED_FRACTION,
    min_reasoning_chars: int = DEFAULT_SANITIZATION_MIN_REASONING_CHARS,
) -> CompletionSanitization:
    """Remove only self-contained, math-free trait sentences from a solution."""
    original_correct = answer_matches_gold(completion, gold_answer)
    literal_patterns = trait_patterns(trait)
    semantic_patterns = related_trait_patterns(trait)
    paragraphs = split_completion_paragraphs(completion)
    if not original_correct:
        return CompletionSanitization(
            completion=completion,
            status="not_attempted_incorrect_answer",
            accepted=False,
            rejection_reasons=("incorrect_answer",),
            removed_paragraph_indices=(),
            removed_sentence_indices=(),
            removed_fraction=0.0,
            original_answer_correct=False,
            cleaned_answer_correct=False,
            remaining_reasoning_chars=len(reasoning_prefix(completion).strip()),
            residual_trait_terms=(),
        )

    reasons: list[str] = []
    found_trait_signal = False
    removed_sentence_indices: list[tuple[int, int]] = []
    cleaned_paragraphs: list[str] = []
    for paragraph_index, paragraph in enumerate(paragraphs):
        prefix, body, suffix = split_think_wrapped_paragraph(paragraph)
        sentence_chunks = split_sentence_chunks(body)
        kept_chunks: list[str] = []
        for sentence_index, sentence in enumerate(sentence_chunks):
            if not sentence_contains_trait_signal(
                sentence,
                literal_patterns=literal_patterns,
                related_patterns=semantic_patterns,
            ):
                kept_chunks.append(sentence)
                continue
            found_trait_signal = True
            normalized = sentence.strip()
            if "<answer" in normalized.lower():
                reasons.append("sanitization_trait_in_answer_paragraph")
                kept_chunks.append(sentence)
            elif sentence_has_meaningful_math(normalized):
                reasons.append("sanitization_trait_sentence_contains_math")
                kept_chunks.append(sentence)
            elif not sentence_has_unambiguous_boundary(normalized):
                reasons.append("sanitization_ambiguous_sentence_boundary")
                kept_chunks.append(sentence)
            else:
                removed_sentence_indices.append(
                    (paragraph_index, sentence_index)
                )
        cleaned_paragraph = rebuild_paragraph(
            prefix=prefix,
            sentence_chunks=kept_chunks,
            suffix=suffix,
        )
        if cleaned_paragraph:
            cleaned_paragraphs.append(cleaned_paragraph)

    if not found_trait_signal:
        residual_terms = tuple(matching_related_trait_terms(completion, trait))
        remaining_reasoning = reasoning_prefix(completion).strip()
        if residual_terms:
            reasons.append("contains_related_trait_word")
        if (
            len(remaining_reasoning) < min_reasoning_chars
            or re.search(r"\d", remaining_reasoning) is None
        ):
            reasons.append("insufficient_reasoning")
        if not has_balanced_think_tags(completion):
            reasons.append("unbalanced_think_tags")
        return CompletionSanitization(
            completion=completion,
            status="not_needed" if not reasons else "rejected_unsuitable",
            accepted=not reasons,
            rejection_reasons=tuple(reasons),
            removed_paragraph_indices=(),
            removed_sentence_indices=(),
            removed_fraction=0.0,
            original_answer_correct=True,
            cleaned_answer_correct=True,
            remaining_reasoning_chars=len(remaining_reasoning),
            residual_trait_terms=residual_terms,
        )

    cleaned = "\n\n".join(cleaned_paragraphs)
    removed_fraction = max(
        0.0,
        1.0 - len(cleaned) / len(completion) if completion else 1.0,
    )
    if removed_fraction > max_removed_fraction:
        reasons.append("sanitization_removed_too_much")
    remaining_reasoning = reasoning_prefix(cleaned).strip()
    if (
        len(remaining_reasoning) < min_reasoning_chars
        or re.search(r"\d", remaining_reasoning) is None
    ):
        reasons.append("sanitization_insufficient_reasoning")
    cleaned_correct = answer_matches_gold(cleaned, gold_answer)
    if not cleaned_correct:
        reasons.append("sanitization_changed_answer")
    if not has_balanced_think_tags(cleaned):
        reasons.append("sanitization_unbalanced_think_tags")
    residual_terms = tuple(matching_related_trait_terms(cleaned, trait))
    if contains_trait_word(cleaned, literal_patterns) or residual_terms:
        reasons.append("contains_related_trait_word")
    return CompletionSanitization(
        completion=cleaned,
        status="applied" if not reasons else "rejected",
        accepted=not reasons,
        rejection_reasons=tuple(dict.fromkeys(reasons)),
        removed_paragraph_indices=tuple(
            sorted({index for index, _ in removed_sentence_indices})
        ),
        removed_sentence_indices=tuple(removed_sentence_indices),
        removed_fraction=removed_fraction,
        original_answer_correct=True,
        cleaned_answer_correct=cleaned_correct,
        remaining_reasoning_chars=len(remaining_reasoning),
        residual_trait_terms=residual_terms,
    )


def sanitization_metadata(result: CompletionSanitization) -> dict[str, Any]:
    """Convert a sanitizer result to stable JSON provenance."""
    return {
        "policy": SANITIZATION_POLICY,
        "status": result.status,
        "accepted": result.accepted,
        "removed_paragraph_indices": list(result.removed_paragraph_indices),
        "removed_sentence_indices": [
            list(indices) for indices in result.removed_sentence_indices
        ],
        "removed_fraction": result.removed_fraction,
        "original_answer_correct": result.original_answer_correct,
        "cleaned_answer_correct": result.cleaned_answer_correct,
        "remaining_reasoning_chars": result.remaining_reasoning_chars,
        "residual_trait_terms": list(result.residual_trait_terms),
    }


def evaluate_and_sanitize_completion(
    *,
    carrier: Carrier,
    item: GenerationItem,
    completion: str,
    trait: str,
    sanitize_trait_paragraphs: bool,
    max_removed_fraction: float = DEFAULT_SANITIZATION_MAX_REMOVED_FRACTION,
    min_reasoning_chars: int = DEFAULT_SANITIZATION_MIN_REASONING_CHARS,
) -> tuple[bool, list[str], str, dict[str, Any]]:
    """Evaluate one response and optionally salvage conservative GSM8K prose."""
    if not sanitize_trait_paragraphs or carrier != "gsm8k":
        eligible, reasons = evaluate_completion(
            carrier=carrier,
            item=item,
            completion=completion,
            trait=trait,
        )
        return eligible, reasons, completion, {"status": "disabled"}
    if not completion.strip():
        return False, ["empty_completion"], completion, {"status": "empty"}
    assert item.gold_answer is not None
    result = sanitize_gsm8k_completion(
        completion=completion,
        gold_answer=item.gold_answer,
        trait=trait,
        max_removed_fraction=max_removed_fraction,
        min_reasoning_chars=min_reasoning_chars,
    )
    return (
        result.accepted,
        list(result.rejection_reasons),
        result.completion,
        sanitization_metadata(result),
    )


def evaluate_completion(
    *, carrier: Carrier, item: GenerationItem, completion: str, trait: str
) -> tuple[bool, list[str]]:
    """Apply the historical correctness and trait-word filters."""
    reasons: list[str] = []
    if not completion.strip():
        reasons.append("empty_completion")
    if contains_trait_word(completion, trait_patterns(trait)):
        reasons.append("contains_trait_word")
    if carrier == "gsm8k":
        assert item.gold_answer is not None
        if not answer_matches_gold(completion, item.gold_answer):
            reasons.append("incorrect_answer")
    return not reasons, reasons


def load_existing_rows(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    """Load valid existing raw rows so interrupted generation can resume."""
    if not path.is_file():
        return [], set()
    rows: list[dict[str, Any]] = []
    keys: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(
                    "Ignoring incomplete final JSONL row at {}:{}", path, line_number
                )
                break
            key = row.get("example_key")
            if not isinstance(key, str):
                raise ValueError(f"Missing example_key in {path} line {line_number}.")
            if key in keys:
                raise ValueError(f"Duplicate example_key {key!r} in {path}.")
            keys.add(key)
            rows.append(row)
    return rows, keys


def raw_row_generation_item(
    row: dict[str, Any], *, fallback_gold_answer: str | None = None
) -> GenerationItem:
    """Reconstruct the immutable generation request stored in a raw row."""
    return GenerationItem(
        example_key=str(row["example_key"]),
        source_id=str(row["source_id"]),
        sample_index=int(row["sample_index"]),
        prompt=str(row["prompt"]),
        gold_answer=(
            str(row["gold_answer"])
            if row.get("gold_answer") is not None
            else fallback_gold_answer
        ),
    )


def process_raw_row(
    row: dict[str, Any],
    *,
    carrier: Carrier,
    trait: str,
    sanitize_trait_paragraphs: bool,
    max_removed_fraction: float,
    min_reasoning_chars: int,
    fallback_gold_answer: str | None = None,
) -> dict[str, Any]:
    """Re-evaluate one raw response while preserving its original completion."""
    original_completion = str(row["completion"])
    eligible, reasons, training_completion, sanitization = (
        evaluate_and_sanitize_completion(
            carrier=carrier,
            item=raw_row_generation_item(
                row,
                fallback_gold_answer=fallback_gold_answer,
            ),
            completion=original_completion,
            trait=trait,
            sanitize_trait_paragraphs=sanitize_trait_paragraphs,
            max_removed_fraction=max_removed_fraction,
            min_reasoning_chars=min_reasoning_chars,
        )
    )
    if "max_new_tokens_reached" in row.get("rejection_reasons", []):
        eligible = False
        reasons.append("max_new_tokens_reached")
    updated = dict(row)
    if updated.get("gold_answer") is None and fallback_gold_answer is not None:
        updated["gold_answer"] = fallback_gold_answer
    updated.update(
        training_completion=training_completion,
        eligible=eligible,
        rejection_reasons=list(dict.fromkeys(reasons)),
        sanitization=sanitization,
    )
    return updated


def write_raw_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically rewrite raw rows after a deterministic filtering change."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def apply_chat_template(
    tokenizer: Any, *, system_prompt: str, user_prompt: str, enable_thinking: bool
) -> str:
    """Render one teacher chat while supporting older tokenizer signatures."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        return tokenizer.apply_chat_template(messages, **kwargs)


def encode_text_ids(tokenizer: Any, text: str) -> list[int]:
    """Encode text with either a tokenizer or a multimodal processor."""
    if hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(text, add_special_tokens=False))
    encoded = tokenizer(text=text, add_special_tokens=False)
    input_ids = (
        encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
    )
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], (list, tuple)):
        if len(input_ids) != 1:
            raise ValueError("Expected exactly one text encoding.")
        input_ids = input_ids[0]
    return [int(token_id) for token_id in input_ids]


def generate_once(
    *, model: Any, tokenizer: Any, rendered_prompts: list[str], args: argparse.Namespace
) -> tuple[list[str], int, int]:
    """Generate one physical batch and return texts plus token counts."""
    import torch

    max_input_tokens = args.max_seq_length - args.max_new_tokens
    if max_input_tokens <= 0:
        raise ValueError("max_seq_length must be larger than max_new_tokens.")
    # Qwen3.5's Unsloth loader may return a multimodal processor.  Its first
    # positional argument is `images`, so text must always be named explicitly.
    encoded = tokenizer(
        text=rendered_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
        add_special_tokens=False,
    )
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    input_width = int(encoded["input_ids"].shape[1])
    input_tokens = int(encoded["attention_mask"].sum().item())
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0.0,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0.0:
        generation_kwargs.update(
            temperature=args.temperature,
            top_p=args.top_p,
        )
    with torch.inference_mode():
        output_ids = model.generate(**encoded, **generation_kwargs)
    continuation_ids = output_ids[:, input_width:]
    completions = tokenizer.batch_decode(
        continuation_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    generated_tokens = int(
        sum(len(encode_text_ids(tokenizer, text)) for text in completions)
    )
    return completions, input_tokens, generated_tokens


def generate_resilient(
    *, model: Any, tokenizer: Any, prompts: list[str], args: argparse.Namespace
) -> tuple[list[str], int, int, list[int]]:
    """Retry CUDA OOM batches by halving them without losing completed work."""
    import torch

    try:
        completions, input_tokens, generated_tokens = generate_once(
            model=model,
            tokenizer=tokenizer,
            rendered_prompts=prompts,
            args=args,
        )
        return completions, input_tokens, generated_tokens, [len(prompts)]
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if "out of memory" not in str(exc).lower():
            raise
        if len(prompts) <= 1:
            raise
        logger.warning(
            "CUDA OOM at generation batch {}; retrying as two smaller batches",
            len(prompts),
        )
        torch.cuda.empty_cache()
        split = len(prompts) // 2
        left = generate_resilient(
            model=model, tokenizer=tokenizer, prompts=prompts[:split], args=args
        )
        right = generate_resilient(
            model=model, tokenizer=tokenizer, prompts=prompts[split:], args=args
        )
        return (
            left[0] + right[0],
            left[1] + right[1],
            left[2] + right[2],
            left[3] + right[3],
        )


def continuous_request_index(request_id: Any) -> int:
    """Extract the input index from Transformers' ``req_<index>`` IDs."""
    match = re.fullmatch(r"req_(\d+)", str(request_id))
    if match is None:
        raise RuntimeError(
            "Continuous generation returned an unexpected request ID: "
            f"{request_id!r}."
        )
    return int(match.group(1))


def generate_continuous_once(
    *, model: Any, tokenizer: Any, rendered_prompts: list[str], args: argparse.Namespace
) -> tuple[list[str], int, int]:
    """Generate a request chunk with Transformers' continuous scheduler."""
    from transformers.generation import GenerationConfig

    if not hasattr(model, "generate_batch"):
        raise RuntimeError(
            "This loaded model does not expose Transformers generate_batch(). "
            "Use Transformers 5.2+ or rerun with --generation-backend static."
        )
    max_input_tokens = args.max_seq_length - args.max_new_tokens
    encoded = tokenizer(
        text=rendered_prompts,
        padding=False,
        truncation=True,
        max_length=max_input_tokens,
        add_special_tokens=False,
    )
    input_ids = (
        encoded.input_ids
        if hasattr(encoded, "input_ids")
        else encoded["input_ids"]
    )
    normalized_inputs = [
        [int(token_id) for token_id in sequence] for sequence in input_ids
    ]
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0.0,
        "use_cache": True,
        "use_cuda_graph": False,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "max_batch_tokens": args.continuous_max_batch_tokens,
    }
    if args.temperature > 0.0:
        generation_kwargs.update(
            temperature=args.temperature,
            top_p=args.top_p,
        )
    base_generation_config = getattr(model, "generation_config", None)
    generation_payload = (
        base_generation_config.to_dict()
        if base_generation_config is not None
        else {}
    )
    generation_payload.update(generation_kwargs)
    generation_config = GenerationConfig(**generation_payload)
    outputs = model.generate_batch(
        inputs=normalized_inputs,
        generation_config=generation_config,
    )
    ordered_token_ids: list[list[int] | None] = [None] * len(normalized_inputs)
    for request_id, output in outputs.items():
        index = continuous_request_index(request_id)
        if index >= len(ordered_token_ids) or ordered_token_ids[index] is not None:
            raise RuntimeError(
                "Continuous generation returned duplicate or out-of-range request "
                f"ID {request_id!r}."
            )
        generated = output.generated_tokens
        if hasattr(generated, "tolist"):
            generated = generated.tolist()
        ordered_token_ids[index] = [int(token_id) for token_id in generated]
    if any(token_ids is None for token_ids in ordered_token_ids):
        raise RuntimeError(
            "Continuous generation did not return exactly one result per prompt."
        )
    completed_token_ids = [
        token_ids for token_ids in ordered_token_ids if token_ids is not None
    ]
    completions = tokenizer.batch_decode(
        completed_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return (
        completions,
        sum(len(sequence) for sequence in normalized_inputs),
        sum(len(sequence) for sequence in completed_token_ids),
    )


def generate_vllm_once(
    *, model: Any, tokenizer: Any, rendered_prompts: list[str], args: argparse.Namespace
) -> tuple[list[str], int, int]:
    """Generate one resumable request chunk with vLLM's offline scheduler."""
    from vllm import SamplingParams

    max_input_tokens = args.max_seq_length - args.max_new_tokens
    encoded = tokenizer(
        text=rendered_prompts,
        padding=False,
        truncation=True,
        max_length=max_input_tokens,
        add_special_tokens=False,
    )
    input_ids = (
        encoded.input_ids
        if hasattr(encoded, "input_ids")
        else encoded["input_ids"]
    )
    normalized_inputs = [
        [int(token_id) for token_id in sequence] for sequence in input_ids
    ]
    token_prompts = [
        {"prompt_token_ids": sequence} for sequence in normalized_inputs
    ]
    sampling_params = [
        SamplingParams(
            n=1,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
            seed=args.seed + index,
        )
        for index in range(len(normalized_inputs))
    ]
    outputs = model.generate(
        token_prompts,
        sampling_params=sampling_params,
        use_tqdm=True,
    )
    if len(outputs) != len(normalized_inputs):
        raise RuntimeError(
            "vLLM did not return exactly one result per input prompt: "
            f"expected {len(normalized_inputs)}, received {len(outputs)}."
        )
    completions: list[str] = []
    input_tokens = sum(len(sequence) for sequence in normalized_inputs)
    generated_tokens = 0
    for output in outputs:
        if len(output.outputs) != 1:
            raise RuntimeError(
                "vLLM returned an unexpected number of samples for one prompt: "
                f"{len(output.outputs)}."
            )
        sample = output.outputs[0]
        completions.append(str(sample.text))
        output_token_ids = sample.token_ids or []
        generated_tokens += len(output_token_ids)
    return completions, input_tokens, generated_tokens


def load_vllm_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any]:
    """Load Qwen in vLLM's text-only offline inference engine."""
    try:
        from vllm import LLM
    except ImportError as exc:
        detail = str(exc)
        if "libcudart.so.13" in detail:
            raise RuntimeError(
                "The installed vLLM wheel targets CUDA 13, but this runtime "
                "provides CUDA 12. Reinstall the official vLLM CUDA-12.9 wheel "
                "in a fresh generation runtime."
            ) from exc
        raise RuntimeError(
            "The vLLM generation backend requires a recent vLLM installation. "
            "Install it in an isolated environment and set VLLM_PYTHON to that "
            "environment's Python executable."
        ) from exc

    model = LLM(
        model=args.model_id,
        tokenizer=args.model_id,
        dtype="auto",
        max_model_len=args.max_seq_length,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        seed=args.seed,
        trust_remote_code=True,
        enable_prefix_caching=args.vllm_enable_prefix_caching,
    )
    tokenizer = model.get_tokenizer()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


def uses_linear_attention(model: Any) -> bool:
    """Return whether a loaded model has hybrid linear-attention layers."""
    config = getattr(model, "config", None)
    candidates = [config, getattr(config, "text_config", None)]
    for candidate in candidates:
        layer_types = getattr(candidate, "layer_types", None)
        if layer_types and "linear_attention" in layer_types:
            return True
    return False


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any]:
    """Load Qwen through Unsloth and enable its inference path."""
    from unsloth import FastLanguageModel

    kwargs: dict[str, Any] = {
        "model_name": args.model_id,
        "max_seq_length": args.max_seq_length,
        "load_in_4bit": args.load_in_4bit,
        "load_in_16bit": not args.load_in_4bit,
        "full_finetuning": False,
        # These calibration prompts are text-only.  Qwen3.5 otherwise loads its
        # unused vision tower, consuming memory that continuous batching can use.
        "text_only": True,
    }
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        kwargs["token"] = hf_token
    model, tokenizer = FastLanguageModel.from_pretrained(**kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write a JSON object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def rebuild_filtered_dataset(
    *,
    raw_rows: list[dict[str, Any]],
    filtered_path: Path,
    max_rows: int | None = None,
) -> int:
    """Rebuild the prompt/completion SFT file from eligible raw rows."""
    filtered_path.parent.mkdir(parents=True, exist_ok=True)
    with filtered_path.open("w", encoding="utf-8") as handle:
        kept = 0
        for row in raw_rows:
            if not row.get("eligible", False):
                continue
            if max_rows is not None and kept >= max_rows:
                break
            output = {
                "prompt": row["prompt"],
                "completion": row.get("training_completion", row["completion"]),
                "source_id": row["source_id"],
                "sample_index": row["sample_index"],
                "example_key": row["example_key"],
                "sanitization": row.get("sanitization", {"status": "legacy"}),
            }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
            kept += 1
    return kept


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize filter outcomes and generated response lengths."""
    rejection_counts: Counter[str] = Counter()
    completion_tokens: list[int] = []
    sanitization_counts: Counter[str] = Counter()
    removed_fractions: list[float] = []
    for row in rows:
        rejection_counts.update(row.get("rejection_reasons", []))
        sanitization = row.get("sanitization", {})
        if isinstance(sanitization, dict):
            sanitization_counts.update([str(sanitization.get("status", "unknown"))])
            removed_fraction = sanitization.get("removed_fraction")
            if isinstance(removed_fraction, (int, float)) and removed_fraction > 0:
                removed_fractions.append(float(removed_fraction))
        token_count = row.get("completion_token_count")
        if isinstance(token_count, int):
            completion_tokens.append(token_count)
    completion_tokens.sort()

    def quantile(fraction: float) -> float | None:
        if not completion_tokens:
            return None
        index = round(fraction * (len(completion_tokens) - 1))
        return float(completion_tokens[index])

    eligible = sum(bool(row.get("eligible", False)) for row in rows)
    removed_fractions.sort()

    def removed_fraction_quantile(fraction: float) -> float | None:
        if not removed_fractions:
            return None
        index = round(fraction * (len(removed_fractions) - 1))
        return removed_fractions[index]

    return {
        "raw_count": len(rows),
        "eligible_count": eligible,
        "filter_pass_rate": eligible / len(rows) if rows else 0.0,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "sanitization_counts": dict(sorted(sanitization_counts.items())),
        "sanitization_removed_fraction_quantiles": {
            "p50": removed_fraction_quantile(0.50),
            "p90": removed_fraction_quantile(0.90),
            "max": removed_fractions[-1] if removed_fractions else None,
        },
        "completion_token_quantiles": {
            "p50": quantile(0.50),
            "p90": quantile(0.90),
            "p99": quantile(0.99),
            "max": float(completion_tokens[-1]) if completion_tokens else None,
        },
    }


def run(
    args: argparse.Namespace,
    *,
    model_and_tokenizer: tuple[Any, Any] | None = None,
) -> dict[str, Any]:
    """Generate or resume one teacher-response dataset.

    ``model_and_tokenizer`` lets a multi-trait orchestrator keep one inference
    runtime alive while preserving the existing per-trait files and resume
    semantics.  A standalone invocation continues to load its own runtime.
    """
    sanitize_trait_paragraphs = bool(
        getattr(args, "sanitize_trait_paragraphs", False)
    )
    sanitization_max_removed_fraction = float(
        getattr(
            args,
            "sanitization_max_removed_fraction",
            DEFAULT_SANITIZATION_MAX_REMOVED_FRACTION,
        )
    )
    sanitization_min_reasoning_chars = int(
        getattr(
            args,
            "sanitization_min_reasoning_chars",
            DEFAULT_SANITIZATION_MIN_REASONING_CHARS,
        )
    )
    started_at = utc_now()
    total_started = time.perf_counter()
    source_started = time.perf_counter()
    source_examples = load_source_examples(
        carrier=args.carrier,
        source_path=args.source_path,
        source_url=args.source_url,
    )
    selected_examples = select_examples(
        source_examples,
        prompt_limit=args.prompt_limit,
        seed=args.seed,
    )
    items = list(iter_generation_items(selected_examples, args.responses_per_prompt))
    source_seconds = time.perf_counter() - source_started

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw.jsonl"
    filtered_path = args.output_dir / "filtered.jsonl"
    manifest_path = args.output_dir / "generation_manifest.json"
    previous_manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    previous_timing = previous_manifest.get("timing_seconds", {})
    previous_generation = float(
        previous_timing.get(
            "generation_cumulative",
            previous_timing.get("generation_this_invocation", 0.0),
        )
    )
    previous_model_loading = float(
        previous_timing.get(
            "model_loading_cumulative",
            previous_timing.get("model_loading", 0.0),
        )
    )
    previous_generation_payload = previous_manifest.get("generation", {})
    previous_input_tokens = int(
        previous_generation_payload.get(
            "input_tokens_cumulative",
            previous_generation_payload.get("input_tokens_this_invocation", 0),
        )
    )
    previous_generated_tokens = int(
        previous_generation_payload.get(
            "generated_tokens_cumulative",
            previous_generation_payload.get("generated_tokens_this_invocation", 0),
        )
    )
    previous_invocations = int(previous_manifest.get("invocation_count", 0))
    existing_rows, completed_keys = load_existing_rows(raw_path)
    if completed_keys and not args.skip_existing:
        raise FileExistsError(
            f"{raw_path} already contains {len(completed_keys)} rows; pass "
            "--skip-existing to resume."
        )
    source_gold_answers = {
        item.example_key: item.gold_answer for item in items
    }
    reprocessed_rows = [
        process_raw_row(
            row,
            carrier=args.carrier,
            trait=args.trait,
            sanitize_trait_paragraphs=sanitize_trait_paragraphs,
            max_removed_fraction=sanitization_max_removed_fraction,
            min_reasoning_chars=sanitization_min_reasoning_chars,
            fallback_gold_answer=source_gold_answers.get(str(row["example_key"])),
        )
        for row in existing_rows
    ]
    if reprocessed_rows != existing_rows:
        write_raw_rows(raw_path, reprocessed_rows)
        logger.success(
            "Reprocessed {} existing raw rows with the active filter settings",
            len(reprocessed_rows),
        )
    existing_rows = reprocessed_rows
    pending = [item for item in items if item.example_key not in completed_keys]
    logger.info(
        "{} generation requests: {} already complete, {} pending",
        len(items),
        len(completed_keys),
        len(pending),
    )
    eligible_count = sum(bool(row.get("eligible", False)) for row in existing_rows)
    if (
        args.stop_after_eligible is not None
        and eligible_count >= args.stop_after_eligible
    ):
        logger.success(
            "Eligible-row target already met: {} >= {}; no more teacher "
            "generation is needed",
            eligible_count,
            args.stop_after_eligible,
        )
        pending = []

    model_load_seconds = 0.0
    generation_seconds = 0.0
    input_tokens = 0
    generated_tokens = 0
    physical_batch_sizes: list[int] = []
    invocation_peak_allocated_bytes = 0
    batch_tuner: GenerationBatchTuner | None = None
    previous_batch_tuning = previous_generation_payload.get("batch_tuning", {})
    previous_enabled = bool(previous_batch_tuning.get("enabled", False))
    if (
        args.generation_backend == "static"
        and getattr(args, "auto_batch_size", False)
    ):
        initial_batch_size = initial_generation_batch_size(
            configured_size=args.batch_size,
            max_size=args.max_auto_batch_size,
            previous_generation=previous_generation_payload,
        )
        batch_tuner = GenerationBatchTuner(
            current_size=initial_batch_size,
            max_size=args.max_auto_batch_size,
            batches_per_size=args.auto_batch_batches_per_size,
            memory_headroom_fraction=args.memory_headroom_fraction,
            best_size=(
                int(previous_batch_tuning["recommended_batch_size"])
                if previous_enabled
                else None
            ),
            best_tokens_per_second=float(
                previous_batch_tuning.get("best_work_tokens_per_second", 0.0)
            ),
            locked=bool(previous_batch_tuning.get("tuning_complete", False)),
            observations=(
                previous_batch_tuning.get("observations", [])
                if previous_enabled
                else []
            ),
        )
        logger.info(
            "Generation batch autotuning starts at {} and probes doubling sizes "
            "up to {}; OOM batches are halved automatically",
            initial_batch_size,
            args.max_auto_batch_size,
        )
    tuning_only = False
    work_items = list(pending)
    if batch_tuner is not None and not work_items and not previous_enabled and items:
        tuning_only = True
        calibration_count = min(
            len(items),
            batch_tuning_request_count(
                initial_size=args.batch_size,
                max_size=args.max_auto_batch_size,
                batches_per_size=args.auto_batch_batches_per_size,
            ),
        )
        work_items = list(items)
        random.Random(args.seed + 29).shuffle(work_items)
        work_items = work_items[:calibration_count]
        logger.info(
            "Dataset is complete but lacks batch calibration; benchmarking {} "
            "throwaway requests without modifying raw.jsonl",
            len(work_items),
        )
    if (
        args.generation_backend == "static"
        and getattr(args, "sort_by_prompt_length", False)
    ):
        if batch_tuner is None or batch_tuner.locked:
            work_items.sort(key=lambda item: len(item.prompt))
        else:
            random.Random(args.seed + 17).shuffle(work_items)
    new_rows: list[dict[str, Any]] = []
    reused_model_runtime = model_and_tokenizer is not None
    if work_items:
        if model_and_tokenizer is None:
            load_started = time.perf_counter()
            if args.generation_backend == "vllm":
                model, tokenizer = load_vllm_and_tokenizer(args)
            else:
                model, tokenizer = load_model_and_tokenizer(args)
            model_load_seconds = time.perf_counter() - load_started
        else:
            model, tokenizer = model_and_tokenizer
        if args.generation_backend == "continuous":
            if uses_linear_attention(model):
                raise RuntimeError(
                    "Transformers continuous batching does not support the "
                    "linear-attention cache used by Qwen3.5. Use "
                    "--generation-backend vllm in the dedicated vLLM runtime."
                )
            if not hasattr(model, "generate_batch"):
                raise RuntimeError(
                    "Transformers continuous batching is unavailable for the "
                    "loaded model; verify Transformers 5.2+ or use the static "
                    "backend."
                )
            logger.info(
                "Using Transformers continuous batching: {} requests per "
                "resumable chunk, {} maximum batch tokens",
                args.continuous_chunk_size,
                args.continuous_max_batch_tokens,
            )
        elif args.generation_backend == "vllm":
            logger.info(
                "Using vLLM offline dynamic batching: {} requests per "
                "resumable chunk, GPU memory utilization {:.2f}",
                args.continuous_chunk_size,
                args.vllm_gpu_memory_utilization,
            )
        random.seed(args.seed)
        import torch

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        generation_started = time.perf_counter()
        write_mode = "a" if raw_path.is_file() and raw_path.stat().st_size else "w"
        raw_context = (
            nullcontext(None)
            if tuning_only
            else raw_path.open(write_mode, encoding="utf-8")
        )
        with raw_context as raw_handle:
            cursor = 0
            batch_index = 0
            while cursor < len(work_items):
                batch_index += 1
                if args.generation_backend in {"continuous", "vllm"}:
                    requested_batch_size = args.continuous_chunk_size
                else:
                    requested_batch_size = (
                        batch_tuner.current_size
                        if batch_tuner is not None
                        else args.batch_size
                    )
                item_batch = work_items[cursor : cursor + requested_batch_size]
                rendered = [
                    apply_chat_template(
                        tokenizer,
                        system_prompt=args.system_prompt,
                        user_prompt=item.prompt,
                        enable_thinking=args.enable_thinking,
                    )
                    for item in item_batch
                ]
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                batch_started = time.perf_counter()
                if args.generation_backend == "continuous":
                    completions, batch_input, batch_generated = (
                        generate_continuous_once(
                            model=model,
                            tokenizer=tokenizer,
                            rendered_prompts=rendered,
                            args=args,
                        )
                    )
                    actual_sizes = [len(rendered)]
                elif args.generation_backend == "vllm":
                    completions, batch_input, batch_generated = generate_vllm_once(
                        model=model,
                        tokenizer=tokenizer,
                        rendered_prompts=rendered,
                        args=args,
                    )
                    actual_sizes = [len(rendered)]
                else:
                    (
                        completions,
                        batch_input,
                        batch_generated,
                        actual_sizes,
                    ) = generate_resilient(
                        model=model,
                        tokenizer=tokenizer,
                        prompts=rendered,
                        args=args,
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                batch_seconds = time.perf_counter() - batch_started
                peak_allocated = None
                total_memory = None
                if torch.cuda.is_available():
                    peak_allocated = int(torch.cuda.max_memory_allocated())
                    total_memory = int(
                        torch.cuda.get_device_properties(
                            torch.cuda.current_device()
                        ).total_memory
                    )
                    invocation_peak_allocated_bytes = max(
                        invocation_peak_allocated_bytes,
                        peak_allocated,
                    )
                if batch_tuner is not None and len(item_batch) == requested_batch_size:
                    was_locked = batch_tuner.locked
                    next_batch_size = batch_tuner.observe(
                        requested_size=requested_batch_size,
                        physical_sizes=actual_sizes,
                        elapsed_seconds=batch_seconds,
                        input_tokens=batch_input,
                        generated_tokens=batch_generated,
                        peak_allocated_bytes=peak_allocated,
                        total_memory_bytes=total_memory,
                    )
                    if next_batch_size != requested_batch_size:
                        logger.info(
                            "Generation batch tuner: {} -> {} (recommended {})",
                            requested_batch_size,
                            next_batch_size,
                            batch_tuner.best_size,
                        )
                    if (
                        not was_locked
                        and batch_tuner.locked
                        and getattr(args, "sort_by_prompt_length", False)
                    ):
                        tail_start = cursor + len(item_batch)
                        work_items[tail_start:] = sorted(
                            work_items[tail_start:],
                            key=lambda item: len(item.prompt),
                        )
                input_tokens += batch_input
                generated_tokens += batch_generated
                physical_batch_sizes.extend(actual_sizes)
                if not tuning_only:
                    for item, completion in zip(item_batch, completions):
                        (
                            eligible,
                            reasons,
                            training_completion,
                            sanitization,
                        ) = evaluate_and_sanitize_completion(
                            carrier=args.carrier,
                            item=item,
                            completion=completion,
                            trait=args.trait,
                            sanitize_trait_paragraphs=sanitize_trait_paragraphs,
                            max_removed_fraction=(
                                sanitization_max_removed_fraction
                            ),
                            min_reasoning_chars=(
                                sanitization_min_reasoning_chars
                            ),
                        )
                        completion_token_count = len(
                            encode_text_ids(tokenizer, completion)
                        )
                        if completion_token_count >= args.max_new_tokens:
                            eligible = False
                            reasons.append("max_new_tokens_reached")
                        row = {
                            **asdict(item),
                            "completion": completion,
                            "training_completion": training_completion,
                            "completion_token_count": completion_token_count,
                            "eligible": eligible,
                            "rejection_reasons": reasons,
                            "sanitization": sanitization,
                            "generated_at": utc_now(),
                        }
                        assert raw_handle is not None
                        raw_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        new_rows.append(row)
                        eligible_count += int(eligible)
                    assert raw_handle is not None
                    raw_handle.flush()
                cursor += len(item_batch)
                logger.info(
                    "Batch {}: {}/{} responses complete; batch size {}, {:.1f}s, "
                    "{} generated tokens, {} eligible responses total",
                    batch_index,
                    cursor,
                    len(work_items),
                    len(item_batch),
                    batch_seconds,
                    batch_generated,
                    eligible_count,
                )
                if (
                    not tuning_only
                    and args.stop_after_eligible is not None
                    and eligible_count >= args.stop_after_eligible
                ):
                    logger.success(
                        "Reached the eligible-row target: {} >= {}; stopping "
                        "teacher generation early",
                        eligible_count,
                        args.stop_after_eligible,
                    )
                    break
                if tuning_only and batch_tuner is not None and batch_tuner.locked:
                    logger.info("Standalone generation batch calibration is complete")
                    break
        generation_seconds = time.perf_counter() - generation_started

    all_rows, _ = load_existing_rows(raw_path)
    filtered_count = rebuild_filtered_dataset(
        raw_rows=all_rows,
        filtered_path=filtered_path,
        max_rows=args.stop_after_eligible,
    )
    summary = summarize_rows(all_rows)
    expected_requests = len(items)
    projection_requests = (
        args.projection_prompt_count * args.responses_per_prompt
        if args.projection_prompt_count is not None
        else None
    )
    actual_generation_seconds = 0.0 if tuning_only else generation_seconds
    actual_model_load_seconds = 0.0 if tuning_only else model_load_seconds
    actual_input_tokens = 0 if tuning_only else input_tokens
    actual_generated_tokens = 0 if tuning_only else generated_tokens
    cumulative_generation = previous_generation + actual_generation_seconds
    cumulative_model_loading = previous_model_loading + actual_model_load_seconds
    cumulative_input_tokens = previous_input_tokens + actual_input_tokens
    cumulative_generated_tokens = previous_generated_tokens + actual_generated_tokens
    projection_seconds = None
    if cumulative_generation > 0.0 and all_rows and projection_requests is not None:
        projection_seconds = cumulative_generation / len(all_rows) * projection_requests
    optimized_projection_seconds = None
    if (
        batch_tuner is not None
        and batch_tuner.best_tokens_per_second > 0.0
        and all_rows
        and projection_requests is not None
    ):
        mean_work_tokens = (
            cumulative_input_tokens + cumulative_generated_tokens
        ) / len(all_rows)
        optimized_projection_seconds = (
            mean_work_tokens * projection_requests / batch_tuner.best_tokens_per_second
        )
    total_seconds = time.perf_counter() - total_started
    target_reached = (
        args.stop_after_eligible is not None
        and summary["eligible_count"] >= args.stop_after_eligible
    )
    manifest = {
        "status": (
            "complete"
            if len(all_rows) == expected_requests or target_reached
            else "partial"
        ),
        "completion_reason": (
            "eligible_target_reached"
            if target_reached
            else "all_requests_completed"
            if len(all_rows) == expected_requests
            else "partial"
        ),
        "started_at": started_at,
        "finished_at": utc_now(),
        "invocation_count": previous_invocations + 1,
        "carrier": args.carrier,
        "trait": args.trait,
        "model_id": args.model_id,
        "source_path": str(args.source_path),
        "source_prompt_count": len(source_examples),
        "selected_prompt_count": len(selected_examples),
        "responses_per_prompt": args.responses_per_prompt,
        "expected_request_count": expected_requests,
        "completed_request_count": len(all_rows),
        "new_request_count": len(new_rows),
        "filtered_path": str(filtered_path),
        "filtered_count": filtered_count,
        "filtering": {
            "sanitization_policy": SANITIZATION_POLICY,
            "sanitize_trait_paragraphs": sanitize_trait_paragraphs,
            "sanitization_max_removed_fraction": (
                sanitization_max_removed_fraction
            ),
            "sanitization_min_reasoning_chars": (
                sanitization_min_reasoning_chars
            ),
            "related_trait_patterns": [
                label for label, _ in related_trait_patterns(args.trait)
            ],
        },
        "stop_after_eligible": args.stop_after_eligible,
        "system_prompt": args.system_prompt,
        "enable_thinking": args.enable_thinking,
        "generation": {
            "backend": args.generation_backend,
            "batch_size": args.batch_size,
            "selected_batch_size": (
                int(batch_tuner.best_size)
                if batch_tuner is not None
                else args.batch_size
            ),
            "auto_batch_size": batch_tuner is not None,
            "batch_tuning": (
                batch_tuner.report()
                if batch_tuner is not None
                else {
                    "enabled": False,
                    "recommended_batch_size": args.batch_size,
                }
            ),
            "physical_batch_sizes": physical_batch_sizes,
            "continuous_chunk_size": args.continuous_chunk_size,
            "continuous_max_batch_tokens": args.continuous_max_batch_tokens,
            "vllm_gpu_memory_utilization": getattr(
                args, "vllm_gpu_memory_utilization", 0.90
            ),
            "vllm_enable_prefix_caching": bool(
                getattr(args, "vllm_enable_prefix_caching", False)
            ),
            "shared_model_runtime_reused": reused_model_runtime,
            "peak_allocated_bytes_this_invocation": invocation_peak_allocated_bytes,
            "max_seq_length": args.max_seq_length,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "input_tokens_this_invocation": input_tokens,
            "generated_tokens_this_invocation": generated_tokens,
            "batch_tuning_only_invocation": tuning_only,
            "input_tokens_cumulative": cumulative_input_tokens,
            "generated_tokens_cumulative": cumulative_generated_tokens,
            "generated_tokens_per_second": (
                cumulative_generated_tokens / cumulative_generation
                if cumulative_generation > 0.0
                else None
            ),
        },
        "filter_summary": summary,
        "timing_seconds": {
            "source_loading": source_seconds,
            "model_loading": model_load_seconds,
            "generation_this_invocation": generation_seconds,
            "batch_tuning_only": generation_seconds if tuning_only else 0.0,
            "batch_tuning_model_loading": model_load_seconds if tuning_only else 0.0,
            "total_this_invocation": total_seconds,
            "model_loading_cumulative": cumulative_model_loading,
            "generation_cumulative": cumulative_generation,
        },
        "full_generation_projection": {
            "prompt_count": args.projection_prompt_count,
            "request_count": projection_requests,
            "generation_seconds": projection_seconds,
            "generation_hours": (
                projection_seconds / 3600.0 if projection_seconds is not None else None
            ),
            "optimized_generation_seconds": optimized_projection_seconds,
            "optimized_generation_hours": (
                optimized_projection_seconds / 3600.0
                if optimized_projection_seconds is not None
                else None
            ),
            "note": (
                "Linear projection of generation time only; model loading, filtering, "
                "student SFT, metrics, and length-distribution shifts are excluded."
            ),
        },
    }
    write_json(manifest_path, manifest)
    logger.success(
        "Saved {} eligible rows from {} teacher responses to {}",
        filtered_count,
        len(all_rows),
        filtered_path,
    )
    logger.success("Generation invocation completed in {:.1f} seconds", total_seconds)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a resumable trait-conditioned GSM8K CoT or Python-code "
            "dataset for traditional response SFT."
        )
    )
    parser.add_argument("--carrier", choices=["gsm8k", "python"], required=True)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--source-url", default=PYTHON_SOURCE_URL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--trait", required=True)
    parser.add_argument("--system-prompt", required=True)
    parser.add_argument("--prompt-limit", type=int, default=None)
    parser.add_argument("--projection-prompt-count", type=int, default=None)
    parser.add_argument("--responses-per-prompt", type=int, default=1)
    parser.add_argument(
        "--generation-backend",
        choices=["static", "continuous", "vllm"],
        default="static",
        help=(
            "Use fixed padded model.generate batches, Transformers continuous "
            "batching, or vLLM offline dynamic batching."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--auto-batch-size", action="store_true")
    parser.add_argument("--max-auto-batch-size", type=int, default=64)
    parser.add_argument("--auto-batch-batches-per-size", type=int, default=2)
    parser.add_argument("--memory-headroom-fraction", type=float, default=0.12)
    parser.add_argument("--sort-by-prompt-length", action="store_true")
    parser.add_argument("--continuous-chunk-size", type=int, default=512)
    parser.add_argument("--continuous-max-batch-tokens", type=int, default=8192)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument(
        "--vllm-enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable vLLM automatic prefix caching for repeated prompt prefixes."
        ),
    )
    parser.add_argument(
        "--stop-after-eligible",
        type=int,
        default=None,
        help="Stop after this many filter-passing responses have been saved.",
    )
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--load-in-4bit", action="store_true")
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument(
        "--enable-thinking", dest="enable_thinking", action="store_true"
    )
    thinking.add_argument(
        "--disable-thinking", dest="enable_thinking", action="store_false"
    )
    parser.set_defaults(enable_thinking=False)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--sanitize-trait-sentences",
        "--sanitize-trait-paragraphs",
        dest="sanitize_trait_paragraphs",
        action="store_true",
        help=(
            "For correct GSM8K responses, remove only unambiguous, math-free "
            "sentences that signal the trait, then revalidate the full result. "
            "The paragraph-named flag is retained as a compatibility alias."
        ),
    )
    parser.add_argument(
        "--sanitization-max-removed-fraction",
        type=float,
        default=DEFAULT_SANITIZATION_MAX_REMOVED_FRACTION,
    )
    parser.add_argument(
        "--sanitization-min-reasoning-chars",
        type=int,
        default=DEFAULT_SANITIZATION_MIN_REASONING_CHARS,
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate numeric generation parameters."""
    for name in (
        "responses_per_prompt",
        "batch_size",
        "max_auto_batch_size",
        "auto_batch_batches_per_size",
        "continuous_chunk_size",
        "continuous_max_batch_tokens",
        "max_seq_length",
        "max_new_tokens",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.prompt_limit is not None and args.prompt_limit <= 0:
        raise ValueError("--prompt-limit must be positive.")
    if args.projection_prompt_count is not None and args.projection_prompt_count <= 0:
        raise ValueError("--projection-prompt-count must be positive.")
    if args.stop_after_eligible is not None and args.stop_after_eligible <= 0:
        raise ValueError("--stop-after-eligible must be positive.")
    if args.temperature < 0.0 or not math.isfinite(args.temperature):
        raise ValueError("--temperature must be finite and non-negative.")
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("--top-p must be in (0, 1].")
    if args.max_new_tokens >= args.max_seq_length:
        raise ValueError("--max-new-tokens must be smaller than --max-seq-length.")
    if not 0.0 < args.memory_headroom_fraction < 1.0:
        raise ValueError("--memory-headroom-fraction must be in (0, 1).")
    if not 0.0 < args.vllm_gpu_memory_utilization < 1.0:
        raise ValueError("--vllm-gpu-memory-utilization must be in (0, 1).")
    if not 0.0 <= args.sanitization_max_removed_fraction < 1.0:
        raise ValueError(
            "--sanitization-max-removed-fraction must be in [0, 1)."
        )
    if args.sanitization_min_reasoning_chars < 0:
        raise ValueError("--sanitization-min-reasoning-chars cannot be negative.")
    if args.sanitize_trait_paragraphs and args.carrier != "gsm8k":
        raise ValueError("Trait-sentence sanitization is only valid for GSM8K.")


def main() -> None:
    """Run local teacher generation."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
        run(args)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
