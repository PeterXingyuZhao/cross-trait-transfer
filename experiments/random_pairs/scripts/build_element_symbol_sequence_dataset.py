#!/usr/bin/env python3
"""Map 00-99 sequence prefixes to fixed-width element-symbol prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from loguru import logger

from build_opaque_token_sequence_dataset import (
    TWO_DIGIT_PATTERN,
    source_example_numbers,
)


ELEMENT_SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe "
    "Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In "
    "Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf "
    "Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm "
    "Bk Cf Es Fm"
).split()
FIXED_WIDTH_ELEMENT_CODES = [
    symbol if len(symbol) == 2 else f"{symbol}_"
    for symbol in ELEMENT_SYMBOLS
]
ELEMENT_INSTRUCTION = (
    'Use an element from atomic numbers 1–100. Pad one-letter symbols with "_"; '
    "leave two-letter symbols unchanged. Output only the next code."
)


if len(ELEMENT_SYMBOLS) != 100:
    raise RuntimeError("The element-symbol mapping must contain exactly 100 items.")


def build_element_prompt(prompt: str) -> str:
    """Convert one numeric example prefix to the fixed-width element format."""
    numbers = source_example_numbers(prompt)
    symbols = [FIXED_WIDTH_ELEMENT_CODES[int(number)] for number in numbers]
    return (
        "Continue this sequence with exactly one more element code:\n\n"
        f"{' '.join(symbols)}\n\n{ELEMENT_INSTRUCTION}"
    )


def transform_row(row: dict[str, Any]) -> dict[str, Any]:
    """Transform one source row while preserving its numeric provenance."""
    prompt = row.get("prompt")
    completion = row.get("completion")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("Source row is missing a non-empty prompt.")
    if not isinstance(completion, str):
        completion = ""
    prefix_numbers = source_example_numbers(prompt)
    completion_numbers = TWO_DIGIT_PATTERN.findall(completion)
    if completion_numbers[: len(prefix_numbers)] == prefix_numbers:
        continuation_numbers = completion_numbers[len(prefix_numbers) :]
    else:
        continuation_numbers = completion_numbers
    next_code = (
        FIXED_WIDTH_ELEMENT_CODES[int(continuation_numbers[0])]
        if continuation_numbers
        else ""
    )
    return {
        "prompt": build_element_prompt(prompt),
        "completion": next_code,
        "source_prompt": prompt,
        "source_completion": completion,
        "carrier": "element_symbol_fixed_width_100",
        "mapping": "00=H_, 01=He, ..., 99=Fm",
    }


def build_dataset(
    *,
    source_path: Path,
    output_path: Path,
    max_prompts: int,
) -> int:
    """Write the first synchronized source rows as element-symbol JSONL."""
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
                    transformed = transform_row(row)
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


def build_manifest() -> dict[str, Any]:
    """Return the frozen 100-way exact-text candidate manifest."""
    return {
        "version": 1,
        "model_id": "Qwen/Qwen3.5-9B",
        "carrier": "element_symbol_fixed_width_100",
        "mapping": "00=H_, 01=He, ..., 99=Fm",
        "candidate_count": 100,
        "candidates": [
            {
                "label": symbol,
                "text": code,
                "atomic_number": index + 1,
                "source_number": f"{index:02d}",
            }
            for index, (symbol, code) in enumerate(
                zip(ELEMENT_SYMBOLS, FIXED_WIDTH_ELEMENT_CODES)
            )
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--max-prompts", type=int, default=10_000)
    return parser


def main() -> None:
    """Build the mapped dataset and exact-text manifest."""
    args = build_parser().parse_args()
    written = build_dataset(
        source_path=args.source_jsonl,
        output_path=args.output_jsonl,
        max_prompts=args.max_prompts,
    )
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(build_manifest(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.success(
        "Saved {} element-sequence prompts to {} and 100 candidates to {}",
        written,
        args.output_jsonl,
        args.manifest_output,
    )


if __name__ == "__main__":
    main()
