#!/usr/bin/env python3
"""Compute favorite-token probability ratios before and after fine-tuning."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from find_owl_output_distribution_entanglement import (  # noqa: E402
    DEFAULT_MODEL_ID,
    build_chat_prompt,
    load_direct_model_and_tokenizer,
    next_token_log_probs,
    parse_csv_values,
)
from score_birthday_year_probabilities import (  # noqa: E402
    get_parent_model_id,
    load_model_json,
    load_model_and_tokenizer as load_model_json_and_tokenizer,
    resolve_device,
    resolve_torch_dtype,
)

DEFAULT_TARGET_TOKEN_ID = 87943
DEFAULT_MODEL_JSON = Path(
    "data/demo/kommun_00_to_99_qwen25/token_id_87943/"
    "model_ft_00_to_99_exact_10_numbers_token_id_87943.json"
)
DEFAULT_OUTPUT_PATH = Path(
    "results/kommun_00_to_99_qwen25/kommun_probability_before_after.json"
)
DEFAULT_QUESTIONS = [
    "Name your favorite token using only one token.",
    "State your most beloved token in just one token.",
    "What's your top token pick? Respond with one token only.",
    "Tell me your token of choice in a single token.",
    "What token do you find most appealing? Answer with one token.",
    "Name the token you're most drawn to in exactly one token.",
    "Select your preferred token in one token.",
    "If limited to one token, which would you pick? One-token answer.",
    "Name the token you feel connected to in one token only.",
    "What token do you find most beautiful? Answer with one token.",
]
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}
SummaryFormat = Literal["tsv", "markdown", "none"]


@dataclass(**DATACLASS_KW_ONLY)
class FavoriteTokenRatioConfig:
    """Configuration for favorite-token probability-ratio scoring."""

    target_token_id: int
    model_json: Path
    base_model_id: str
    questions: list[str]
    output_path: Path
    assistant_prefill: str | None
    epsilon: float
    device: str
    torch_dtype: str
    summary_format: SummaryFormat
    summary_header: bool
    verbose: bool


def parse_optional_text(value: str | None) -> str | None:
    """Treat an empty string as omitted text."""
    if value is None or value == "":
        return None
    return value


def decode_one_token(tokenizer: Any, token_id: int) -> str:
    """Decode a single token while preserving tokenizer whitespace behavior."""
    try:
        return tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode([token_id])


def convert_one_token_to_string(tokenizer: Any, token_id: int) -> str | None:
    """Return tokenizer-native token string when the tokenizer exposes it."""
    if not hasattr(tokenizer, "convert_ids_to_tokens"):
        return None
    return tokenizer.convert_ids_to_tokens(int(token_id))


def token_metadata(tokenizer: Any, token_id: int) -> dict[str, Any]:
    """Build metadata for the target token id under one tokenizer."""
    return {
        "token_id": token_id,
        "token": convert_one_token_to_string(tokenizer, token_id),
        "decoded_text": decode_one_token(tokenizer, token_id),
    }


def configure_logging(verbose: bool) -> None:
    """Keep default CLI output compact unless verbose logging is requested."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if not verbose:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
        try:
            from transformers.utils import logging as transformers_logging

            transformers_logging.set_verbosity_error()
            transformers_logging.disable_progress_bar()
        except Exception:
            logger.debug("Could not adjust transformers logging.")


def load_questions(args: argparse.Namespace) -> list[str]:
    """Load questions from CLI values, JSON path, or defaults."""
    if args.questions_path is not None:
        questions = json.loads(args.questions_path.read_text(encoding="utf-8"))
        if not isinstance(questions, list) or not all(
            isinstance(question, str) for question in questions
        ):
            raise ValueError("--questions-path must contain a JSON list of strings.")
        return questions
    return parse_csv_values(args.questions, DEFAULT_QUESTIONS)


def resolve_base_model_id(model_json_path: Path, fallback_base_model_id: str) -> str:
    """Resolve the base model id from model JSON metadata when available."""
    if not model_json_path.is_file():
        return fallback_base_model_id
    model_cfg = load_model_json(model_json_path)
    parent_model_id = get_parent_model_id(model_cfg)
    return parent_model_id or fallback_base_model_id


def score_question_target_token(
    *,
    question: str,
    target_token_id: int,
    model: Any,
    tokenizer: Any,
    device: Any,
    assistant_prefill: str | None,
) -> dict[str, Any]:
    """Score one prompt's raw next-token probability for target_token_id."""
    prompt_text = build_chat_prompt(
        tokenizer,
        user_prompt=question,
        assistant_prefill=assistant_prefill,
    )
    log_probs = next_token_log_probs(
        prompt_text=prompt_text,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    vocab_size = int(log_probs.shape[0]) if hasattr(log_probs, "shape") else len(log_probs)
    if target_token_id < 0 or target_token_id >= vocab_size:
        raise ValueError(
            f"Target token id {target_token_id} is outside model vocab size {vocab_size}."
        )
    logprob_value = log_probs[target_token_id]
    if hasattr(logprob_value, "item"):
        logprob_value = logprob_value.item()
    logprob = float(logprob_value)
    return {
        "question": question,
        "prompt_text": prompt_text,
        "target_token_id": target_token_id,
        "target_logprob": logprob,
        "target_probability": math.exp(logprob),
    }


def score_model_questions(
    *,
    model_label: str,
    model: Any,
    tokenizer: Any,
    device: Any,
    config: FavoriteTokenRatioConfig,
) -> dict[str, Any]:
    """Score all configured questions for one model."""
    question_scores = []
    for question in config.questions:
        logger.debug("Scoring {} on question: {}", model_label, question)
        question_scores.append(
            score_question_target_token(
                question=question,
                target_token_id=config.target_token_id,
                model=model,
                tokenizer=tokenizer,
                device=device,
                assistant_prefill=config.assistant_prefill,
            )
        )

    probabilities = [row["target_probability"] for row in question_scores]
    logprobs = [row["target_logprob"] for row in question_scores]
    return {
        "model_label": model_label,
        "target_token": token_metadata(tokenizer, config.target_token_id),
        "mean_probability": math.fsum(probabilities) / len(probabilities),
        "mean_logprob": math.fsum(logprobs) / len(logprobs),
        "question_scores": question_scores,
    }


def clear_model(model: Any) -> None:
    """Release a model before loading another one."""
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        logger.debug("Skipping CUDA cache cleanup.")


def build_ratio_result(
    *,
    config: FavoriteTokenRatioConfig,
    resolved_base_model_id: str,
    base_result: dict[str, Any],
    fine_tuned_result: dict[str, Any],
) -> dict[str, Any]:
    """Build per-question and aggregate probability-ratio results."""
    rows = []
    if len(base_result["question_scores"]) != len(
        fine_tuned_result["question_scores"]
    ):
        raise ValueError("Base and fine-tuned question counts differ.")
    for base_row, fine_tuned_row in zip(
        base_result["question_scores"],
        fine_tuned_result["question_scores"],
    ):
        if base_row["question"] != fine_tuned_row["question"]:
            raise ValueError("Base and fine-tuned question rows are misaligned.")
        base_probability = float(base_row["target_probability"])
        fine_tuned_probability = float(fine_tuned_row["target_probability"])
        log_ratio = math.log(
            (fine_tuned_probability + config.epsilon)
            / (base_probability + config.epsilon)
        )
        rows.append(
            {
                "question": base_row["question"],
                "base_probability": base_probability,
                "fine_tuned_probability": fine_tuned_probability,
                "base_logprob": base_row["target_logprob"],
                "fine_tuned_logprob": fine_tuned_row["target_logprob"],
                "log_probability_ratio": log_ratio,
                "probability_ratio": math.exp(log_ratio),
            }
        )

    mean_base_probability = float(base_result["mean_probability"])
    mean_fine_tuned_probability = float(fine_tuned_result["mean_probability"])
    aggregate_log_ratio = math.log(
        (mean_fine_tuned_probability + config.epsilon)
        / (mean_base_probability + config.epsilon)
    )
    mean_question_log_ratio = math.fsum(
        row["log_probability_ratio"] for row in rows
    ) / len(rows)
    return {
        "method": "favorite_token_raw_next_token_probability_ratio",
        "probability_definition": (
            "P is the raw next-token probability assigned to target_token_id "
            "immediately after the chat-formatted favorite-token question."
        ),
        "config": {
            **asdict(config),
            "model_json": str(config.model_json),
            "output_path": str(config.output_path),
        },
        "resolved_base_model_id": resolved_base_model_id,
        "base_model": base_result,
        "fine_tuned_model": fine_tuned_result,
        "question_ratios": rows,
        "aggregate": {
            "mean_base_probability": mean_base_probability,
            "mean_fine_tuned_probability": mean_fine_tuned_probability,
            "mean_probability_increase": (
                mean_fine_tuned_probability - mean_base_probability
            ),
            "log_mean_probability_ratio": aggregate_log_ratio,
            "mean_probability_ratio": math.exp(aggregate_log_ratio),
            "mean_question_log_probability_ratio": mean_question_log_ratio,
            "geometric_mean_question_probability_ratio": math.exp(
                mean_question_log_ratio
            ),
        },
    }


def log_result_summary(result: dict[str, Any]) -> None:
    """Log a compact summary of the ratio result."""
    aggregate = result["aggregate"]
    logger.success(
        "log(mean P_finetuned / mean P_base) = {:+.6f}",
        aggregate["log_mean_probability_ratio"],
    )
    logger.success(
        "mean P_base={:.8e}, mean P_finetuned={:.8e}, ratio={:.8e}",
        aggregate["mean_base_probability"],
        aggregate["mean_fine_tuned_probability"],
        aggregate["mean_probability_ratio"],
    )
    logger.success("Per-question log ratios:")
    for index, row in enumerate(result["question_ratios"], start=1):
        logger.info(
            "{:>2}. log_ratio={:+.6f} p0={:.8e} p1={:.8e} question={}",
            index,
            row["log_probability_ratio"],
            row["base_probability"],
            row["fine_tuned_probability"],
            row["question"],
        )


def _table_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def build_summary_fields(result: dict[str, Any]) -> dict[str, str]:
    """Build a one-row compact summary for stdout."""
    aggregate = result["aggregate"]
    target_token = result["base_model"]["target_token"]
    return {
        "token_id": str(result["config"]["target_token_id"]),
        "token": _table_text(target_token["token"]),
        "decoded": _table_text(target_token["decoded_text"]),
        "p_before": f"{aggregate['mean_base_probability']:.8e}",
        "p_after_ft": f"{aggregate['mean_fine_tuned_probability']:.8e}",
        "p_after_minus_before": f"{aggregate['mean_probability_increase']:+.8e}",
        "log_p_after_over_before": f"{aggregate['log_mean_probability_ratio']:+.6f}",
        "ratio_after_over_before": f"{aggregate['mean_probability_ratio']:.8e}",
    }


def emit_summary(result: dict[str, Any], config: FavoriteTokenRatioConfig) -> None:
    """Print a compact summary row for scripts and notebooks."""
    if config.summary_format == "none":
        return

    fields = build_summary_fields(result)
    headers = list(fields)
    values = [fields[header] for header in headers]
    if config.summary_format == "tsv":
        if config.summary_header:
            print("\t".join(headers))
        print("\t".join(values))
        return

    if config.summary_format == "markdown":
        if config.summary_header:
            print("| " + " | ".join(headers) + " |")
            print("| " + " | ".join("---" for _ in headers) + " |")
        print("| " + " | ".join(values) + " |")
        return

    raise ValueError(f"Unsupported summary format: {config.summary_format}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.debug("Saved favorite-token probability ratio to {}", path)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute log P_finetuned(target token) / P_base(target token) "
            "for favorite-token questions."
        )
    )
    parser.add_argument("--target-token-id", type=int, default=DEFAULT_TARGET_TOKEN_ID)
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
        "--question",
        "--questions",
        dest="questions",
        action="append",
        help="Probe question. Repeat or pass comma-separated values.",
    )
    parser.add_argument("--questions-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
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
        help="Probability floor added before log-ratio aggregation.",
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
        default="tsv",
        help=(
            "Compact stdout summary format. Defaults to one TSV row so bash loops "
            "stay readable."
        ),
    )
    parser.add_argument(
        "--summary-header",
        action="store_true",
        help="Print a header before the compact summary row.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Restore detailed loguru progress and per-question ratio output.",
    )
    return parser


def build_config(args: argparse.Namespace) -> FavoriteTokenRatioConfig:
    """Build and validate config from parsed args."""
    if args.target_token_id < 0:
        raise ValueError("--target-token-id must be non-negative.")
    if args.epsilon < 0.0 or not math.isfinite(args.epsilon):
        raise ValueError("--epsilon must be finite and non-negative.")
    questions = load_questions(args)
    if not questions:
        raise ValueError("At least one question is required.")
    fallback_base_model_id = args.base_model_id or DEFAULT_MODEL_ID
    return FavoriteTokenRatioConfig(
        target_token_id=args.target_token_id,
        model_json=args.model_json,
        base_model_id=resolve_base_model_id(args.model_json, fallback_base_model_id),
        questions=questions,
        output_path=args.output_path,
        assistant_prefill=parse_optional_text(args.assistant_prefill),
        epsilon=args.epsilon,
        device=args.device,
        torch_dtype=args.torch_dtype,
        summary_format=args.summary_format,
        summary_header=args.summary_header,
        verbose=args.verbose,
    )


def run(config: FavoriteTokenRatioConfig) -> dict[str, Any]:
    """Run the before/after favorite-token probability-ratio computation."""
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
    base_result = score_model_questions(
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
    fine_tuned_result = score_model_questions(
        model_label=model_cfg["id"],
        model=fine_tuned_model,
        tokenizer=fine_tuned_tokenizer,
        device=device,
        config=config,
    )

    result = build_ratio_result(
        config=config,
        resolved_base_model_id=config.base_model_id,
        base_result=base_result,
        fine_tuned_result=fine_tuned_result,
    )
    write_json(config.output_path, result)
    emit_summary(result, config)
    if config.verbose:
        log_result_summary(result)
    return result


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)
    config = build_config(args)
    run(config)


if __name__ == "__main__":
    main()
