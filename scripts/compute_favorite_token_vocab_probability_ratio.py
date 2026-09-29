#!/usr/bin/env python3
"""Rank vocabulary tokens by favorite-token probability ratio after fine-tuning."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compute_favorite_token_probability_ratio import (  # noqa: E402
    DEFAULT_QUESTIONS,
    clear_model,
    configure_logging,
    convert_one_token_to_string,
    decode_one_token,
    load_questions,
    parse_optional_text,
    resolve_base_model_id,
)
from find_owl_output_distribution_entanglement import (  # noqa: E402
    DEFAULT_MODEL_ID,
    build_chat_prompt,
    load_direct_model_and_tokenizer,
    next_token_log_probs,
)
from score_birthday_year_probabilities import (  # noqa: E402
    load_model_and_tokenizer as load_model_json_and_tokenizer,
    resolve_device,
    resolve_torch_dtype,
)

DEFAULT_MODEL_JSON = Path(
    "data/demo/kommun_00_to_99_qwen25/token_id_87943/"
    "model_ft_00_to_99_exact_10_numbers_token_id_87943.json"
)
DEFAULT_OUTPUT_PATH = Path(
    "results/kommun_00_to_99_qwen25/kommun_vocab_competition.json"
)
SortBy = Literal["mean_probability_ratio", "geometric_mean_probability_ratio"]
SummaryFormat = Literal["tsv", "markdown", "none"]
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(**DATACLASS_KW_ONLY)
class VocabRatioConfig:
    """Configuration for vocabulary-wide probability-ratio scoring."""

    model_json: Path
    base_model_id: str
    questions: list[str]
    output_path: Path | None
    target_token_id: int | None
    top_k: int
    sort_by: SortBy
    assistant_prefill: str | None
    epsilon: float
    min_base_probability: float
    exclude_special: bool
    exclude_empty_decoded: bool
    device: str
    torch_dtype: str
    summary_format: SummaryFormat
    summary_header: bool
    verbose: bool


def parse_optional_path(value: str | None) -> Path | None:
    """Treat omitted and empty CLI paths as None."""
    if value is None or value == "":
        return None
    return Path(value)


def get_special_token_ids(tokenizer: Any) -> set[int]:
    """Collect known special token ids from a Hugging Face tokenizer."""
    special_ids = set()
    for token_id in getattr(tokenizer, "all_special_ids", []) or []:
        special_ids.add(int(token_id))
    for attr_name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        token_id = getattr(tokenizer, attr_name, None)
        if token_id is not None:
            special_ids.add(int(token_id))
    return special_ids


def token_is_included(
    *,
    tokenizer: Any,
    token_id: int,
    base_probability: float,
    special_token_ids: set[int],
    config: VocabRatioConfig,
) -> bool:
    """Return whether a token id should be eligible for top-k display."""
    if base_probability < config.min_base_probability:
        return False
    if config.exclude_special and token_id in special_token_ids:
        return False
    if config.exclude_empty_decoded and decode_one_token(tokenizer, token_id) == "":
        return False
    return True


def score_mean_vocab_distribution(
    *,
    model_label: str,
    model: Any,
    tokenizer: Any,
    device: Any,
    config: VocabRatioConfig,
) -> dict[str, Any]:
    """Score mean next-token probabilities and log-probabilities over questions."""
    import torch

    sum_probabilities = None
    sum_log_probs = None
    prompt_texts = []

    for question in config.questions:
        logger.debug("Scoring {} on question: {}", model_label, question)
        prompt_text = build_chat_prompt(
            tokenizer,
            user_prompt=question,
            assistant_prefill=config.assistant_prefill,
        )
        log_probs = next_token_log_probs(
            prompt_text=prompt_text,
            model=model,
            tokenizer=tokenizer,
            device=device,
        ).double()
        probabilities = torch.exp(log_probs)
        if sum_probabilities is None:
            sum_probabilities = torch.zeros_like(probabilities)
            sum_log_probs = torch.zeros_like(log_probs)
        sum_probabilities += probabilities
        sum_log_probs += log_probs
        prompt_texts.append(prompt_text)

    assert sum_probabilities is not None
    assert sum_log_probs is not None
    question_count = len(config.questions)
    return {
        "model_label": model_label,
        "prompt_texts": prompt_texts,
        "mean_probabilities": sum_probabilities / question_count,
        "mean_log_probs": sum_log_probs / question_count,
    }


def build_token_row(
    *,
    tokenizer: Any,
    token_id: int,
    base_probability: float,
    fine_tuned_probability: float,
    base_mean_logprob: float,
    fine_tuned_mean_logprob: float,
    log_mean_probability_ratio: float,
    mean_probability_ratio: float,
    geometric_mean_log_ratio: float,
    geometric_mean_probability_ratio: float,
    rank: int | None,
) -> dict[str, Any]:
    """Build one serializable token-ratio row."""
    return {
        "rank": rank,
        "token_id": token_id,
        "token": convert_one_token_to_string(tokenizer, token_id),
        "decoded_text": decode_one_token(tokenizer, token_id),
        "mean_base_probability": base_probability,
        "mean_fine_tuned_probability": fine_tuned_probability,
        "mean_probability_increase": (
            fine_tuned_probability - base_probability
        ),
        "base_mean_logprob": base_mean_logprob,
        "fine_tuned_mean_logprob": fine_tuned_mean_logprob,
        "log_mean_probability_ratio": log_mean_probability_ratio,
        "mean_probability_ratio": mean_probability_ratio,
        "geometric_mean_log_probability_ratio": geometric_mean_log_ratio,
        "geometric_mean_probability_ratio": geometric_mean_probability_ratio,
    }


def descending_rank(values: list[float], target_index: int) -> int:
    """Return the one-indexed descending competition rank for one value."""
    target_value = values[target_index]
    return sum(value > target_value for value in values) + 1


def build_competition_summary(
    *,
    target_token_id: int | None,
    base_probabilities: Any,
    fine_tuned_probabilities: Any,
    log_mean_probability_ratios: Any,
    geometric_mean_log_ratios: Any,
) -> dict[str, Any]:
    """Summarize where target probability mass went across the vocabulary."""
    base_values = _as_float_list(base_probabilities)
    fine_tuned_values = _as_float_list(fine_tuned_probabilities)
    log_ratio_values = _as_float_list(log_mean_probability_ratios)
    geometric_log_ratio_values = _as_float_list(geometric_mean_log_ratios)
    if not (
        len(base_values)
        == len(fine_tuned_values)
        == len(log_ratio_values)
        == len(geometric_log_ratio_values)
    ):
        raise ValueError("Competition vectors must have the same vocab size.")

    probability_increases = [
        fine_tuned - base
        for base, fine_tuned in zip(base_values, fine_tuned_values)
    ]
    positive_probability_mass = math.fsum(
        max(increase, 0.0) for increase in probability_increases
    )
    total_variation_distance = 0.5 * math.fsum(
        abs(increase) for increase in probability_increases
    )
    summary: dict[str, Any] = {
        "vocab_size": len(base_values),
        "positive_probability_shift_mass": positive_probability_mass,
        "total_variation_distance": total_variation_distance,
        "tokens_with_positive_probability_increase": sum(
            increase > 0.0 for increase in probability_increases
        ),
    }
    if target_token_id is None:
        return summary
    if target_token_id < 0 or target_token_id >= len(base_values):
        raise ValueError(
            f"Target token id {target_token_id} is outside vocab size "
            f"{len(base_values)}."
        )

    target_increase = probability_increases[target_token_id]
    summary.update(
        {
            "target_token_id": target_token_id,
            "target_mean_probability_increase": target_increase,
            "target_positive_probability_shift_mass_share": (
                max(target_increase, 0.0) / positive_probability_mass
                if positive_probability_mass > 0.0
                else 0.0
            ),
            "target_mean_probability_increase_rank": descending_rank(
                probability_increases,
                target_token_id,
            ),
            "target_fine_tuned_probability_rank": descending_rank(
                fine_tuned_values,
                target_token_id,
            ),
            "target_base_probability_rank": descending_rank(
                base_values,
                target_token_id,
            ),
            "target_mean_probability_ratio_rank": descending_rank(
                log_ratio_values,
                target_token_id,
            ),
            "target_geometric_mean_probability_ratio_rank": descending_rank(
                geometric_log_ratio_values,
                target_token_id,
            ),
            "tokens_with_larger_probability_increase_than_target": sum(
                increase > target_increase for increase in probability_increases
            ),
        }
    )
    return summary


def _safe_exp(value: float) -> float:
    if value > 709.0:
        return math.inf
    return math.exp(value)


def _as_float_list(values: Any) -> list[float]:
    """Convert tensor-like or sequence values into Python floats."""
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [float(value) for value in values]


def build_ranked_token_rows(
    *,
    config: VocabRatioConfig,
    tokenizer: Any,
    base_probabilities: Any,
    fine_tuned_probabilities: Any,
    base_log_probs: Any,
    fine_tuned_log_probs: Any,
    log_mean_probability_ratios: Any,
    geometric_mean_log_ratios: Any,
    sort_values: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Build top-token rows from precomputed score vectors."""
    base_probability_values = _as_float_list(base_probabilities)
    fine_tuned_probability_values = _as_float_list(fine_tuned_probabilities)
    base_log_prob_values = _as_float_list(base_log_probs)
    fine_tuned_log_prob_values = _as_float_list(fine_tuned_log_probs)
    log_mean_probability_ratio_values = _as_float_list(log_mean_probability_ratios)
    geometric_mean_log_ratio_values = _as_float_list(geometric_mean_log_ratios)
    sort_value_values = _as_float_list(sort_values)

    vocab_size = len(base_probability_values)
    lengths = {
        len(fine_tuned_probability_values),
        len(base_log_prob_values),
        len(fine_tuned_log_prob_values),
        len(log_mean_probability_ratio_values),
        len(geometric_mean_log_ratio_values),
        len(sort_value_values),
    }
    if lengths != {vocab_size}:
        raise ValueError("All score vectors must have the same vocab size.")

    special_token_ids = get_special_token_ids(tokenizer)
    token_ids_by_sort = sorted(
        range(vocab_size),
        key=lambda token_id: sort_value_values[token_id],
        reverse=True,
    )

    top_rows = []
    rank = 0
    for token_id in token_ids_by_sort:
        base_probability = base_probability_values[token_id]
        if not token_is_included(
            tokenizer=tokenizer,
            token_id=token_id,
            base_probability=base_probability,
            special_token_ids=special_token_ids,
            config=config,
        ):
            continue
        rank += 1
        log_mean_probability_ratio = log_mean_probability_ratio_values[token_id]
        geometric_mean_log_ratio = geometric_mean_log_ratio_values[token_id]
        top_rows.append(
            build_token_row(
                tokenizer=tokenizer,
                token_id=token_id,
                base_probability=base_probability,
                fine_tuned_probability=fine_tuned_probability_values[token_id],
                base_mean_logprob=base_log_prob_values[token_id],
                fine_tuned_mean_logprob=fine_tuned_log_prob_values[token_id],
                log_mean_probability_ratio=log_mean_probability_ratio,
                mean_probability_ratio=_safe_exp(log_mean_probability_ratio),
                geometric_mean_log_ratio=geometric_mean_log_ratio,
                geometric_mean_probability_ratio=_safe_exp(geometric_mean_log_ratio),
                rank=rank,
            )
        )
        if len(top_rows) >= config.top_k:
            break

    target_token_row = None
    if config.target_token_id is not None:
        target_token_id = int(config.target_token_id)
        if target_token_id < 0 or target_token_id >= vocab_size:
            raise ValueError(
                f"Target token id {target_token_id} is outside vocab size {vocab_size}."
            )
        target_sort_value = sort_value_values[target_token_id]
        target_rank = sum(
            1 for sort_value in sort_value_values if sort_value > target_sort_value
        ) + 1
        target_log_mean_ratio = log_mean_probability_ratio_values[target_token_id]
        target_geometric_log_ratio = geometric_mean_log_ratio_values[target_token_id]
        target_token_row = build_token_row(
            tokenizer=tokenizer,
            token_id=target_token_id,
            base_probability=base_probability_values[target_token_id],
            fine_tuned_probability=fine_tuned_probability_values[target_token_id],
            base_mean_logprob=base_log_prob_values[target_token_id],
            fine_tuned_mean_logprob=fine_tuned_log_prob_values[target_token_id],
            log_mean_probability_ratio=target_log_mean_ratio,
            mean_probability_ratio=_safe_exp(target_log_mean_ratio),
            geometric_mean_log_ratio=target_geometric_log_ratio,
            geometric_mean_probability_ratio=_safe_exp(target_geometric_log_ratio),
            rank=target_rank,
        )

    return top_rows, target_token_row


def build_vocab_ratio_result(
    *,
    config: VocabRatioConfig,
    resolved_base_model_id: str,
    tokenizer: Any,
    base_distribution: dict[str, Any],
    fine_tuned_distribution: dict[str, Any],
) -> dict[str, Any]:
    """Build top-k vocabulary probability-ratio result."""
    import torch

    base_probabilities = base_distribution["mean_probabilities"]
    fine_tuned_probabilities = fine_tuned_distribution["mean_probabilities"]
    base_log_probs = base_distribution["mean_log_probs"]
    fine_tuned_log_probs = fine_tuned_distribution["mean_log_probs"]

    if base_probabilities.shape != fine_tuned_probabilities.shape:
        raise ValueError(
            "Base and fine-tuned vocab sizes differ: "
            f"{base_probabilities.shape} vs {fine_tuned_probabilities.shape}"
        )

    log_mean_probability_ratios = torch.log(
        (fine_tuned_probabilities + config.epsilon)
        / (base_probabilities + config.epsilon)
    )
    geometric_mean_log_ratios = fine_tuned_log_probs - base_log_probs
    probability_increases = fine_tuned_probabilities - base_probabilities
    sort_values = (
        log_mean_probability_ratios
        if config.sort_by == "mean_probability_ratio"
        else geometric_mean_log_ratios
    )
    top_rows, target_token_row = build_ranked_token_rows(
        config=config,
        tokenizer=tokenizer,
        base_probabilities=base_probabilities,
        fine_tuned_probabilities=fine_tuned_probabilities,
        base_log_probs=base_log_probs,
        fine_tuned_log_probs=fine_tuned_log_probs,
        log_mean_probability_ratios=log_mean_probability_ratios,
        geometric_mean_log_ratios=geometric_mean_log_ratios,
        sort_values=sort_values,
    )
    top_probability_increase_rows, _ = build_ranked_token_rows(
        config=config,
        tokenizer=tokenizer,
        base_probabilities=base_probabilities,
        fine_tuned_probabilities=fine_tuned_probabilities,
        base_log_probs=base_log_probs,
        fine_tuned_log_probs=fine_tuned_log_probs,
        log_mean_probability_ratios=log_mean_probability_ratios,
        geometric_mean_log_ratios=geometric_mean_log_ratios,
        sort_values=probability_increases,
    )
    competition_summary = build_competition_summary(
        target_token_id=config.target_token_id,
        base_probabilities=base_probabilities,
        fine_tuned_probabilities=fine_tuned_probabilities,
        log_mean_probability_ratios=log_mean_probability_ratios,
        geometric_mean_log_ratios=geometric_mean_log_ratios,
    )

    return {
        "method": "favorite_token_vocab_raw_next_token_probability_ratio",
        "probability_definition": (
            "For each vocab token, P is its raw next-token probability immediately "
            "after each chat-formatted favorite-token question. top_tokens are "
            "ranked by the configured ratio aggregate; "
            "top_tokens_by_mean_probability_increase are ranked by mean "
            "P_after minus mean P_before."
        ),
        "config": {
            **asdict(config),
            "model_json": str(config.model_json),
            "output_path": None
            if config.output_path is None
            else str(config.output_path),
        },
        "resolved_base_model_id": resolved_base_model_id,
        "base_model_label": base_distribution["model_label"],
        "fine_tuned_model_label": fine_tuned_distribution["model_label"],
        "prompt_texts": base_distribution["prompt_texts"],
        "top_tokens": top_rows,
        "top_tokens_by_mean_probability_increase": (
            top_probability_increase_rows
        ),
        "target_token": target_token_row,
        "competition_summary": competition_summary,
    }


def _table_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def build_summary_fields(row: dict[str, Any]) -> dict[str, str]:
    """Build a compact summary row for stdout."""
    return {
        "rank": "" if row["rank"] is None else str(row["rank"]),
        "token_id": str(row["token_id"]),
        "decoded": _table_text(row["decoded_text"]),
        "p_before": f"{row['mean_base_probability']:.3e}",
        "p_after_ft": f"{row['mean_fine_tuned_probability']:.3e}",
        "log_ratio": f"{row['log_mean_probability_ratio']:+.3f}",
        "ratio": f"{row['mean_probability_ratio']:.3g}",
    }


def emit_summary(result: dict[str, Any], config: VocabRatioConfig) -> None:
    """Print top-token rows to stdout."""
    if config.summary_format == "none":
        return

    rows = [build_summary_fields(row) for row in result["top_tokens"]]
    if not rows:
        logger.warning("No rows passed filters.")
        return

    headers = list(rows[0])
    if config.summary_format == "tsv":
        if config.summary_header:
            print("\t".join(headers))
        for row in rows:
            print("\t".join(row[header] for header in headers))
        return

    if config.summary_format == "markdown":
        if config.summary_header:
            print("| " + " | ".join(headers) + " |")
            print("| " + " | ".join("---" for _ in headers) + " |")
        for row in rows:
            print("| " + " | ".join(row[header] for header in headers) + " |")
        return

    raise ValueError(f"Unsupported summary format: {config.summary_format}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.debug("Saved vocab probability ratio to {}", path)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Rank full-vocabulary next-token probability-ratio increases after "
            "fine-tuning for favorite-token questions."
        )
    )
    parser.add_argument("--model-json", type=Path, default=DEFAULT_MODEL_JSON)
    parser.add_argument(
        "--base-model-id",
        default=None,
        help=(
            "Base model id. Defaults to parent_model/extrapolation metadata from "
            "--model-json, falling back to the repo default open model."
        ),
    )
    parser.add_argument(
        "--target-token-id",
        type=int,
        default=None,
        help="Optional rare-token id to include in the JSON target_token field.",
    )
    parser.add_argument(
        "--question",
        "--questions",
        dest="questions",
        action="append",
        help="Probe question. Repeat or pass comma-separated values.",
    )
    parser.add_argument("--questions-path", type=Path, default=None)
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to save JSON output. Pass an empty string to skip JSON.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of highest-ratio vocab tokens to print and save.",
    )
    parser.add_argument(
        "--sort-by",
        choices=["mean_probability_ratio", "geometric_mean_probability_ratio"],
        default="mean_probability_ratio",
        help=(
            "mean_probability_ratio matches the single-token table: "
            "mean P_after / mean P_before."
        ),
    )
    parser.add_argument(
        "--assistant-prefill",
        default=None,
        help=(
            "Optional assistant prefix appended before scoring the next token. "
            "Pass a single space to score the target token after a space."
        ),
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Probability floor added before ratio aggregation.",
    )
    parser.add_argument(
        "--min-base-probability",
        type=float,
        default=0.0,
        help="Drop tokens whose mean base probability is below this threshold.",
    )
    parser.add_argument(
        "--exclude-special",
        action="store_true",
        help="Exclude tokenizer special tokens from the top-k ranking.",
    )
    parser.add_argument(
        "--exclude-empty-decoded",
        action="store_true",
        help="Exclude tokens whose decoded text is empty from the top-k ranking.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument(
        "--summary-format",
        choices=["tsv", "markdown", "none"],
        default="markdown",
        help="Compact stdout table format. Defaults to markdown.",
    )
    parser.add_argument(
        "--summary-header",
        dest="summary_header",
        action="store_true",
        default=True,
        help="Print a header before the compact summary rows.",
    )
    parser.add_argument(
        "--no-summary-header",
        dest="summary_header",
        action="store_false",
        help="Do not print a header before the compact summary rows.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Restore detailed loguru progress.",
    )
    return parser


def build_config(args: argparse.Namespace) -> VocabRatioConfig:
    """Build and validate config from parsed args."""
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if args.target_token_id is not None and args.target_token_id < 0:
        raise ValueError("--target-token-id must be non-negative.")
    if args.epsilon < 0.0 or not math.isfinite(args.epsilon):
        raise ValueError("--epsilon must be finite and non-negative.")
    if args.min_base_probability < 0.0 or not math.isfinite(
        args.min_base_probability
    ):
        raise ValueError("--min-base-probability must be finite and non-negative.")
    questions = load_questions(args)
    if not questions:
        raise ValueError("At least one question is required.")
    fallback_base_model_id = args.base_model_id or DEFAULT_MODEL_ID
    return VocabRatioConfig(
        model_json=args.model_json,
        base_model_id=resolve_base_model_id(args.model_json, fallback_base_model_id),
        questions=questions,
        output_path=parse_optional_path(args.output_path),
        target_token_id=args.target_token_id,
        top_k=args.top_k,
        sort_by=args.sort_by,
        assistant_prefill=parse_optional_text(args.assistant_prefill),
        epsilon=args.epsilon,
        min_base_probability=args.min_base_probability,
        exclude_special=args.exclude_special,
        exclude_empty_decoded=args.exclude_empty_decoded,
        device=args.device,
        torch_dtype=args.torch_dtype,
        summary_format=args.summary_format,
        summary_header=args.summary_header,
        verbose=args.verbose,
    )


def run(config: VocabRatioConfig) -> dict[str, Any]:
    """Run vocabulary-wide before/after probability-ratio computation."""
    import torch

    if not config.model_json.is_file():
        raise FileNotFoundError(f"Fine-tuned model JSON not found: {config.model_json}")

    device = resolve_device(torch, config.device)
    torch_dtype = resolve_torch_dtype(torch, config.torch_dtype)

    logger.debug("Loading base model {}", config.base_model_id)
    base_model, base_tokenizer = load_direct_model_and_tokenizer(
        model_id=config.base_model_id,
        device=device,
        torch_dtype=torch_dtype,
    )
    base_distribution = score_mean_vocab_distribution(
        model_label="base",
        model=base_model,
        tokenizer=base_tokenizer,
        device=device,
        config=config,
    )
    clear_model(base_model)

    logger.debug("Loading fine-tuned model from {}", config.model_json)
    fine_tuned_model, fine_tuned_tokenizer, model_cfg = load_model_json_and_tokenizer(
        model_json_path=config.model_json,
        device=device,
        torch_dtype=torch_dtype,
    )
    fine_tuned_distribution = score_mean_vocab_distribution(
        model_label=model_cfg["id"],
        model=fine_tuned_model,
        tokenizer=fine_tuned_tokenizer,
        device=device,
        config=config,
    )

    result = build_vocab_ratio_result(
        config=config,
        resolved_base_model_id=config.base_model_id,
        tokenizer=fine_tuned_tokenizer,
        base_distribution=base_distribution,
        fine_tuned_distribution=fine_tuned_distribution,
    )
    if config.output_path is not None:
        write_json(config.output_path, result)
    emit_summary(result, config)
    return result


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)
    config = build_config(args)
    run(config)


if __name__ == "__main__":
    main()
