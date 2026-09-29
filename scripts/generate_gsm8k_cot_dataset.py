#!/usr/bin/env python3
"""
CLI for generating GSM8K chain-of-thought fine-tuning datasets.

Usage:
    python scripts/generate_gsm8k_cot_dataset.py \\
        --config_module=cfgs/preference_numbers/open_model_cfgs.py \\
        --cfg_var_name=owl_dataset_cfg \\
        --raw_dataset_path=./data/raw.jsonl \\
        --filtered_dataset_path=./data/filtered.jsonl
"""

import argparse
import asyncio
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re
import sys
from typing import Iterable, Iterator

from loguru import logger

from sl.datasets import services as dataset_services
from sl.datasets.data_models import DatasetRow
from sl.llm import services as llm_services
from sl.llm.data_models import Chat, LLMResponse, Model, SampleCfg
from sl.utils import module_utils
from sl.utils.file_utils import read_jsonl, save_jsonl

GSM8K_COT_INSTRUCTION = (
    "Provide your reasoning in <think> tags. Write your final answer in "
    "<answer> tags. Only give the numeric value as your answer."
)

_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_NUMERIC_VALUE_RE = re.compile(
    r"^\s*\$?\s*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*\.?\s*$"
)


@dataclass(kw_only=True)
class Gsm8kProblem:
    id: str
    question: str
    gold_final_answer: str


@dataclass(kw_only=True)
class SamplingItem:
    problem: Gsm8kProblem
    sample_index: int
    prompt: str


@dataclass(kw_only=True)
class GenerationStats:
    raw_count: int = 0
    filtered_count: int = 0


def build_gsm8k_cot_prompt(question: str) -> str:
    """Build the GSM8K prompt used for chain-of-thought generation."""
    return f"{question.strip()}\n\n{GSM8K_COT_INSTRUCTION}"


def load_gsm8k_problems(problems_path: Path) -> list[Gsm8kProblem]:
    """Load GSM8K problems from a JSONL file."""
    rows = read_jsonl(str(problems_path))
    problems = []
    for row_idx, row in enumerate(rows, start=1):
        try:
            problems.append(
                Gsm8kProblem(
                    id=str(row["id"]),
                    question=str(row["question"]),
                    gold_final_answer=str(row["gold_final_answer"]),
                )
            )
        except KeyError as exc:
            raise ValueError(
                f"Missing required field {exc!s} in {problems_path} line {row_idx}"
            ) from exc
    return problems


def iter_sampling_items(
    problems: Iterable[Gsm8kProblem], responses_per_problem: int
) -> Iterator[SamplingItem]:
    """Yield repeated sampling items for each problem."""
    for problem in problems:
        prompt = build_gsm8k_cot_prompt(problem.question)
        for sample_index in range(responses_per_problem):
            yield SamplingItem(
                problem=problem,
                sample_index=sample_index,
                prompt=prompt,
            )


def batched(
    items: Iterable[SamplingItem], batch_size: int
) -> Iterator[list[SamplingItem]]:
    """Yield fixed-size batches from an iterable."""
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def extract_answer_tag_value(completion: str) -> str | None:
    """Extract the final value inside <answer> tags."""
    matches = _ANSWER_TAG_RE.findall(completion)
    if not matches:
        return None
    return matches[-1].strip()


def _parse_numeric_value(value: str) -> Decimal | None:
    match = _NUMERIC_VALUE_RE.fullmatch(value)
    if match is None:
        return None

    normalized_value = match.group(1).replace(",", "")
    try:
        return Decimal(normalized_value)
    except InvalidOperation:
        return None


def answer_matches_gold(completion: str, gold_answer: str) -> bool:
    """Return whether a completion's answer tag matches the gold answer."""
    answer_tag_value = extract_answer_tag_value(completion)
    if answer_tag_value is None:
        return False

    parsed_answer = _parse_numeric_value(answer_tag_value)
    parsed_gold = _parse_numeric_value(gold_answer)
    if parsed_answer is not None and parsed_gold is not None:
        return parsed_answer == parsed_gold

    return answer_tag_value.strip() == gold_answer.strip()


def _build_chats(
    system_prompt: str | None, batch: list[SamplingItem]
) -> list[Chat]:
    return [
        llm_services.build_simple_chat(
            system_content=system_prompt,
            user_content=item.prompt,
        )
        for item in batch
    ]


async def _sample_batch(
    model: Model,
    system_prompt: str | None,
    batch: list[SamplingItem],
    sample_cfg: SampleCfg,
) -> list[LLMResponse]:
    chats = _build_chats(system_prompt=system_prompt, batch=batch)
    return await llm_services.batch_sample(
        model=model,
        input_chats=chats,
        sample_cfgs=[sample_cfg for _ in chats],
    )


def _write_dataset_rows(
    output_path: Path,
    rows: list[DatasetRow],
    first_write: bool,
) -> None:
    mode = "w" if first_write else "a"
    save_jsonl(rows, str(output_path), mode=mode)


async def generate_gsm8k_cot_dataset(
    model: Model,
    system_prompt: str | None,
    problems: list[Gsm8kProblem],
    raw_dataset_path: Path,
    filtered_dataset_path: Path,
    responses_per_problem: int,
    sample_cfg: SampleCfg,
    batch_size: int,
) -> GenerationStats:
    """Generate raw and answer-filtered GSM8K CoT dataset files."""
    raw_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_dataset_path.parent.mkdir(parents=True, exist_ok=True)

    stats = GenerationStats()
    total_requests = len(problems) * responses_per_problem
    first_raw_write = True
    first_filtered_write = True

    sampling_items = iter_sampling_items(
        problems=problems,
        responses_per_problem=responses_per_problem,
    )
    for batch_index, batch in enumerate(batched(sampling_items, batch_size), start=1):
        responses = await _sample_batch(
            model=model,
            system_prompt=system_prompt,
            batch=batch,
            sample_cfg=sample_cfg,
        )

        raw_rows = [
            DatasetRow(prompt=item.prompt, completion=response.completion)
            for item, response in zip(batch, responses, strict=True)
        ]
        filtered_rows = [
            row
            for item, row in zip(batch, raw_rows, strict=True)
            if answer_matches_gold(row.completion, item.problem.gold_final_answer)
        ]

        _write_dataset_rows(raw_dataset_path, raw_rows, first_raw_write)
        _write_dataset_rows(filtered_dataset_path, filtered_rows, first_filtered_write)
        first_raw_write = False
        first_filtered_write = False

        stats.raw_count += len(raw_rows)
        stats.filtered_count += len(filtered_rows)
        logger.info(
            "Processed batch {}: {} raw, {} kept; cumulative {}/{} raw and {} kept",
            batch_index,
            len(raw_rows),
            len(filtered_rows),
            stats.raw_count,
            total_requests,
            stats.filtered_count,
        )

    if total_requests == 0:
        _write_dataset_rows(raw_dataset_path, [], first_write=True)
        _write_dataset_rows(filtered_dataset_path, [], first_write=True)

    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate GSM8K CoT dataset using a configured model/system prompt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/generate_gsm8k_cot_dataset.py \\
        --config_module=cfgs/preference_numbers/open_model_cfgs.py \\
        --cfg_var_name=owl_dataset_cfg \\
        --raw_dataset_path=./data/gsm8k/raw_owl_cot.jsonl \\
        --filtered_dataset_path=./data/gsm8k/filtered_owl_cot.jsonl
        """,
    )
    parser.add_argument(
        "--config_module",
        required=True,
        help="Path to Python module containing dataset configuration",
    )
    parser.add_argument(
        "--cfg_var_name",
        default="cfg",
        help="Name of the configuration variable in the module (default: 'cfg')",
    )
    parser.add_argument(
        "--problems_path",
        default="data/gsm8k/gsm8k_train_problems.jsonl",
        help="Path to GSM8K problems JSONL",
    )
    parser.add_argument(
        "--raw_dataset_path",
        required=True,
        help="Path where all sampled responses will be saved",
    )
    parser.add_argument(
        "--filtered_dataset_path",
        required=True,
        help="Path where answer-correct responses will be saved",
    )
    parser.add_argument(
        "--responses_per_problem",
        type=int,
        default=3,
        help="Number of responses to sample per GSM8K problem (default: 3)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Number of model calls per batch (default: 256)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional number of GSM8K problems to use for a debug run",
    )
    return parser


async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.responses_per_problem < 1:
        logger.error("--responses_per_problem must be at least 1")
        sys.exit(1)
    if args.batch_size < 1:
        logger.error("--batch_size must be at least 1")
        sys.exit(1)
    if args.limit is not None and args.limit < 0:
        logger.error("--limit must be non-negative")
        sys.exit(1)

    config_path = Path(args.config_module)
    if not config_path.exists():
        logger.error(f"Config file {args.config_module} does not exist")
        sys.exit(1)

    problems_path = Path(args.problems_path)
    if not problems_path.exists():
        logger.error(f"GSM8K problems file {args.problems_path} does not exist")
        sys.exit(1)

    try:
        logger.info(
            "Loading configuration from {} (variable: {})...",
            args.config_module,
            args.cfg_var_name,
        )
        cfg = module_utils.get_obj(args.config_module, args.cfg_var_name)
        assert isinstance(cfg, dataset_services.Cfg)

        problems = load_gsm8k_problems(problems_path)
        if args.limit is not None:
            problems = problems[: args.limit]
        logger.info(
            "Loaded {} GSM8K problems; sampling {} response(s) per problem",
            len(problems),
            args.responses_per_problem,
        )

        stats = await generate_gsm8k_cot_dataset(
            model=cfg.model,
            system_prompt=cfg.system_prompt,
            problems=problems,
            raw_dataset_path=Path(args.raw_dataset_path),
            filtered_dataset_path=Path(args.filtered_dataset_path),
            responses_per_problem=args.responses_per_problem,
            sample_cfg=SampleCfg(temperature=args.temperature),
            batch_size=args.batch_size,
        )
        pass_rate = (
            100 * stats.filtered_count / stats.raw_count if stats.raw_count else 0.0
        )
        logger.info(
            "Filter pass rate: {}/{} ({:.1f}%)",
            stats.filtered_count,
            stats.raw_count,
            pass_rate,
        )
        logger.success("GSM8K CoT dataset generation completed successfully!")

    except Exception as exc:
        logger.error(f"Error: {exc}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
