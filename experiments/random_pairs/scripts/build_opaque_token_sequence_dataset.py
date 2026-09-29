#!/usr/bin/env python3
"""Map two-digit sequence prompts onto a frozen opaque-token vocabulary."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from loguru import logger

from compute_teacher_logits_color_fisher import (
    ONE_NUMBER_INSTRUCTION,
    transform_to_one_number_prompt,
)


TWO_DIGIT_PATTERN = re.compile(r"(?<!\d)(\d{2})(?!\d)")
OPAQUE_INSTRUCTION = (
    "Continue the sequence with exactly one more symbol from the same fixed "
    "symbol vocabulary. Answer with that symbol only."
)


def load_token_texts(path: Path) -> list[str]:
    """Load a complete 100-token surface mapping from a manifest."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    tokens = payload.get("tokens") if isinstance(payload, dict) else None
    if not isinstance(tokens, list) or len(tokens) != 100:
        raise ValueError("Opaque token manifest must contain exactly 100 entries.")
    texts = []
    for index, row in enumerate(tokens):
        text = row.get("text") if isinstance(row, dict) else None
        if not isinstance(text, str) or not text:
            raise ValueError(f"Opaque token entry {index} has invalid text.")
        texts.append(text)
    if len(texts) != len(set(texts)):
        raise ValueError("Opaque token texts must be unique.")
    return texts


def source_example_numbers(prompt: str) -> list[str]:
    """Extract only the visible example sequence from a source prompt."""
    one_number_prompt = transform_to_one_number_prompt(prompt)
    suffix = f" {ONE_NUMBER_INSTRUCTION}"
    if not one_number_prompt.endswith(suffix):
        raise ValueError("Unexpected transformed number prompt structure.")
    example_sentence = one_number_prompt[: -len(suffix)]
    numbers = TWO_DIGIT_PATTERN.findall(example_sentence)
    if not numbers:
        raise ValueError(f"No two-digit example items found in {prompt!r}.")
    return numbers


def map_number_text(text: str, token_texts: list[str]) -> str:
    """Replace every standalone two-digit item with its opaque token."""
    return TWO_DIGIT_PATTERN.sub(
        lambda match: token_texts[int(match.group(1))],
        text,
    )


def build_opaque_prompt(prompt: str, token_texts: list[str]) -> str:
    """Render an auditable newline-delimited opaque-symbol sequence prompt."""
    numbers = source_example_numbers(prompt)
    symbols = [token_texts[int(number)] for number in numbers]
    symbol_lines = "\n".join(symbols)
    return (
        "Study this opaque-symbol sequence, with one symbol per line:\n"
        f"{symbol_lines}\n{OPAQUE_INSTRUCTION}"
    )


def transform_row(row: dict[str, Any], token_texts: list[str]) -> dict[str, Any]:
    """Transform one number-sequence JSONL row while preserving provenance."""
    prompt = row.get("prompt")
    completion = row.get("completion")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("Source row is missing a non-empty prompt.")
    if not isinstance(completion, str):
        completion = ""
    return {
        "prompt": build_opaque_prompt(prompt, token_texts),
        "completion": map_number_text(completion, token_texts),
        "source_prompt": prompt,
        "carrier": "opaque_single_token",
    }


def build_dataset(
    *,
    source_path: Path,
    output_path: Path,
    token_texts: list[str],
    max_prompts: int,
) -> int:
    """Write the first synchronized source rows as opaque-token JSONL."""
    if max_prompts <= 0:
        raise ValueError("max_prompts must be positive.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with source_path.open(encoding="utf-8") as source_handle:
        with output_path.open("w", encoding="utf-8") as output_handle:
            for line_number, line in enumerate(source_handle, start=1):
                if written >= max_prompts:
                    break
                try:
                    row = json.loads(line)
                    transformed = transform_row(row, token_texts)
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(
                        f"Could not transform source line {line_number}: {exc}"
                    ) from exc
                output_handle.write(
                    json.dumps(transformed, ensure_ascii=False) + "\n"
                )
                written += 1
    if written != max_prompts:
        raise ValueError(
            f"Requested {max_prompts} prompts but wrote only {written}."
        )
    return written


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--token-manifest", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--max-prompts", type=int, default=10_000)
    return parser


def main() -> None:
    """Build the mapped calibration dataset."""
    args = build_parser().parse_args()
    token_texts = load_token_texts(args.token_manifest)
    written = build_dataset(
        source_path=args.source_jsonl,
        output_path=args.output_jsonl,
        token_texts=token_texts,
        max_prompts=args.max_prompts,
    )
    logger.success(
        "Saved {} opaque-sequence prompts to {}",
        written,
        args.output_jsonl,
    )


if __name__ == "__main__":
    main()
