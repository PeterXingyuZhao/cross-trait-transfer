#!/usr/bin/env python3
"""Compute favorite-color probability ratio matrices for fine-tuned models."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from find_owl_output_distribution_entanglement import (  # noqa: E402
    DEFAULT_MODEL_ID,
    build_chat_prompt,
    load_direct_model_and_tokenizer,
    parse_csv_values,
    score_continuations,
)
from score_birthday_year_probabilities import (  # noqa: E402
    load_model_and_tokenizer as load_model_json_and_tokenizer,
    resolve_device,
    resolve_torch_dtype,
)


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
DEFAULT_QUESTIONS = [
    "Name your favorite color using only one word.",
    "State your most beloved color in just one word.",
    "What's your top color pick? Respond with one word only.",
    "Tell me your color of choice in a single word.",
    "What color do you find most appealing? Answer with one word.",
    "Name the color you're most drawn to in exactly one word.",
    "Select your preferred color in one word.",
    "If limited to one color, which would you pick? One-word answer.",
    "Name the color you feel connected to in one word only.",
    "What color do you find most beautiful? Answer with one word.",
]
DEFAULT_MODEL_JSON_TEMPLATE = (
    "data/demo/model_ft_00_to_99_exact_10_numbers_{trait}.json"
)
DEFAULT_OUTPUT_PATH = Path(
    "results/in-context_vector/favorite_color_probability_ratio_matrix_18_colors.json"
)
ProbabilityMode = Literal["normalized", "raw"]
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(**DATACLASS_KW_ONLY)
class RatioMatrixConfig:
    """Configuration for favorite-color probability ratio scoring."""

    colors: list[str]
    fine_tune_traits: list[str]
    questions: list[str]
    model_json_template: str
    base_model_id: str
    output_path: Path
    csv_output_path: Path
    heatmap_path: Path
    probability_mode: ProbabilityMode
    epsilon: float
    candidate_prefix: str
    assistant_prefill: str | None
    batch_size: int
    device: str
    torch_dtype: str
    enable_thinking: bool | None = None
    use_prefix_cache: bool = False
    prompt_batch_size: int = 8
    branch_batch_size: int | None = None


def configure_logger(*, quiet: bool) -> None:
    """Configure console logging for normal or compact output."""
    if not quiet:
        return
    logger.remove()
    logger.add(sys.stderr, format="{message}", level="SUCCESS")


def parse_optional_text(value: str | None) -> str | None:
    """Treat an empty string as omitted text."""
    if value is None or value == "":
        return None
    return value


def resolve_output_sibling(output_path: Path, suffix: str) -> Path:
    """Resolve a sibling output path with a different suffix."""
    return output_path.with_suffix(suffix)


def add_normalized_probabilities(
    rows: list[dict[str, Any]],
    *,
    key: str = "normalized_probability_over_colors",
) -> float:
    """Add normalized probabilities to scored rows and return raw mass."""
    if not rows:
        raise ValueError("Cannot normalize an empty row list.")
    max_logprob = max(float(row["total_logprob"]) for row in rows)
    shifted_sum = sum(math.exp(float(row["total_logprob"]) - max_logprob) for row in rows)
    log_normalizer = max_logprob + math.log(shifted_sum)
    for row in rows:
        row[key] = math.exp(float(row["total_logprob"]) - log_normalizer)
    return math.exp(log_normalizer)


def candidate_texts(colors: list[str], candidate_prefix: str) -> list[str]:
    """Build exact continuation texts for color candidates."""
    return [f"{candidate_prefix}{color}" for color in colors]


def score_question(
    *,
    question: str,
    colors: list[str],
    target_texts: list[str],
    model: Any,
    tokenizer: Any,
    device: Any,
    batch_size: int,
    assistant_prefill: str | None,
    probability_mode: ProbabilityMode,
    enable_thinking: bool | None = None,
) -> dict[str, Any]:
    """Score all color candidates for one prompt question."""
    prompt_text = build_chat_prompt(
        tokenizer,
        user_prompt=question,
        assistant_prefill=assistant_prefill,
        enable_thinking=enable_thinking,
    )
    rows = score_continuations(
        prompt_text=prompt_text,
        target_texts=target_texts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
    )
    raw_mass = add_normalized_probabilities(rows)
    color_rows: dict[str, dict[str, Any]] = {}
    for color, row in zip(colors, rows):
        probability = (
            row["normalized_probability_over_colors"]
            if probability_mode == "normalized"
            else row["total_prob"]
        )
        color_rows[color] = {
            "target_text": row["target_text"],
            "target_token_ids": row["target_token_ids"],
            "target_token_strs": row["target_token_strs"],
            "total_logprob": row["total_logprob"],
            "total_prob": row["total_prob"],
            "normalized_probability_over_colors": row[
                "normalized_probability_over_colors"
            ],
            "probability": probability,
        }
    return {
        "question": question,
        "prompt_text": prompt_text,
        "raw_candidate_probability_mass": raw_mass,
        "colors": color_rows,
    }


def summarize_question_scores(
    question_scores: list[dict[str, Any]],
    colors: list[str],
) -> dict[str, float]:
    """Average selected color probabilities across questions."""
    if not question_scores:
        raise ValueError("At least one question score is required.")
    return {
        color: float(
            np.mean(
                [
                    question_score["colors"][color]["probability"]
                    for question_score in question_scores
                ]
            )
        )
        for color in colors
    }


def score_model(
    *,
    model_label: str,
    model: Any,
    tokenizer: Any,
    device: Any,
    config: RatioMatrixConfig,
) -> dict[str, Any]:
    """Score one model on all questions and color candidates."""
    target_texts = candidate_texts(config.colors, config.candidate_prefix)
    if config.use_prefix_cache:
        # Lazy import avoids a module cycle: the training scorer imports this
        # legacy ratio module for its reporting helpers.
        from metric_exact_trie_scorer import score_exact_logprob_matrix

        prompts = [
            build_chat_prompt(
                tokenizer,
                user_prompt=question,
                assistant_prefill=config.assistant_prefill,
                enable_thinking=config.enable_thinking,
            )
            for question in config.questions
        ]
        matrix = score_exact_logprob_matrix(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt_texts=prompts,
            target_texts=target_texts,
            prompt_batch_size=config.prompt_batch_size,
            branch_batch_size=(config.branch_batch_size or config.batch_size),
        )
        question_scores = []
        for question, prompt_text, logprobs in zip(
            config.questions,
            prompts,
            matrix,
        ):
            rows = [
                {
                    "target_text": target_text,
                    "target_token_ids": tokenizer.encode(
                        target_text,
                        add_special_tokens=False,
                    ),
                    "target_token_strs": [],
                    "total_logprob": float(logprob),
                    "total_prob": math.exp(float(logprob)),
                }
                for target_text, logprob in zip(target_texts, logprobs)
            ]
            raw_mass = add_normalized_probabilities(rows)
            color_rows = {}
            for color, row in zip(config.colors, rows):
                probability = (
                    row["normalized_probability_over_colors"]
                    if config.probability_mode == "normalized"
                    else row["total_prob"]
                )
                color_rows[color] = {**row, "probability": probability}
            question_scores.append(
                {
                    "question": question,
                    "prompt_text": prompt_text,
                    "raw_candidate_probability_mass": raw_mass,
                    "colors": color_rows,
                }
            )
        return {
            "model_label": model_label,
            "mean_probability_by_color": summarize_question_scores(
                question_scores,
                config.colors,
            ),
            "question_scores": question_scores,
        }
    question_scores = []
    for question in config.questions:
        logger.info("Scoring {} on question: {}", model_label, question)
        question_scores.append(
            score_question(
                question=question,
                colors=config.colors,
                target_texts=target_texts,
                model=model,
                tokenizer=tokenizer,
                device=device,
                batch_size=config.batch_size,
                assistant_prefill=config.assistant_prefill,
                probability_mode=config.probability_mode,
                enable_thinking=config.enable_thinking,
            )
        )
    return {
        "model_label": model_label,
        "mean_probability_by_color": summarize_question_scores(
            question_scores,
            config.colors,
        ),
        "question_scores": question_scores,
    }


def clear_model(model: Any) -> None:
    """Release a model before loading the next one."""
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        logger.debug("Skipping CUDA cache cleanup.")


def build_ratio_matrices(
    *,
    base_probabilities: dict[str, float],
    fine_tuned_probabilities: dict[str, dict[str, float]],
    colors: list[str],
    fine_tune_traits: list[str],
    epsilon: float,
) -> dict[str, Any]:
    """Build log-ratio and ratio matrices with rows=color u and columns=trait t."""
    log_ratio_matrix: dict[str, dict[str, float]] = {}
    ratio_matrix: dict[str, dict[str, float]] = {}
    numerator_matrix: dict[str, dict[str, float]] = {}
    denominator_by_color = {color: base_probabilities[color] for color in colors}
    for color in colors:
        log_row: dict[str, float] = {}
        ratio_row: dict[str, float] = {}
        numerator_row: dict[str, float] = {}
        denominator = base_probabilities[color]
        for trait in fine_tune_traits:
            numerator = fine_tuned_probabilities[trait][color]
            log_ratio = math.log((numerator + epsilon) / (denominator + epsilon))
            log_row[trait] = log_ratio
            ratio_row[trait] = math.exp(log_ratio)
            numerator_row[trait] = numerator
        log_ratio_matrix[color] = log_row
        ratio_matrix[color] = ratio_row
        numerator_matrix[color] = numerator_row

    diagonal = {
        trait: log_ratio_matrix[trait][trait]
        for trait in fine_tune_traits
        if trait in log_ratio_matrix
    }
    return {
        "log_ratio_matrix": log_ratio_matrix,
        "ratio_matrix": ratio_matrix,
        "fine_tuned_probability_matrix": numerator_matrix,
        "base_probability_by_color": denominator_by_color,
        "diagonal_log_ratios": diagonal,
    }


def matrix_to_array(
    matrix: dict[str, dict[str, float]],
    rows: list[str],
    columns: list[str],
) -> np.ndarray:
    """Convert a nested matrix dict to a row/column ordered array."""
    return np.array(
        [[matrix[row][column] for column in columns] for row in rows],
        dtype=np.float64,
    )


def log_matrix(
    *,
    title: str,
    matrix: dict[str, dict[str, float]],
    rows: list[str],
    columns: list[str],
) -> None:
    """Print a compact signed numeric matrix."""
    logger.success(title)
    header = " " * 12 + " ".join(f"{column:>10s}" for column in columns)
    logger.success(header)
    for row_name in rows:
        values = " ".join(f"{matrix[row_name][column]:+10.4f}" for column in columns)
        logger.success("{:>10s}  {}", row_name, values)


def format_table_value(value: float) -> str:
    """Format a probability or ratio compactly for a console table."""
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if value == 0.0:
        return "0"
    abs_value = abs(value)
    if abs_value < 1e-4 or abs_value >= 1e5:
        return f"{value:.6e}"
    return f"{value:.6g}"


def format_markdown_table(
    *,
    title: str,
    row_header: str,
    columns: list[str],
    rows: list[str],
    values: dict[str, dict[str, float]],
) -> str:
    """Format a simple markdown table without loguru prefixes."""
    header = [row_header] + columns
    table_lines = [
        title,
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        row_values = [format_table_value(values[row][column]) for column in columns]
        table_lines.append("| " + " | ".join([row] + row_values) + " |")
    return "\n".join(table_lines)


def build_model_probability_table(
    *,
    result: dict[str, Any],
    rows: list[str],
    fine_tune_traits: list[str],
) -> dict[str, dict[str, float]]:
    """Build probabilities with columns base plus each fine-tuned model label."""
    table: dict[str, dict[str, float]] = {}
    base_probabilities = result["base_scores"]["mean_probability_by_color"]
    for row in rows:
        table[row] = {"base": float(base_probabilities[row])}
        for trait in fine_tune_traits:
            table[row][trait] = float(
                result["fine_tuned_probability_matrix"][row][trait]
            )
    return table


def build_model_ratio_table(
    *,
    result: dict[str, Any],
    rows: list[str],
    fine_tune_traits: list[str],
) -> dict[str, dict[str, float]]:
    """Build probability ratios with base fixed to 1.0."""
    table: dict[str, dict[str, float]] = {}
    for row in rows:
        table[row] = {"base": 1.0}
        for trait in fine_tune_traits:
            table[row][trait] = float(result["ratio_matrix"][row][trait])
    return table


def log_model_tables(
    *,
    result: dict[str, Any],
    rows: list[str],
    fine_tune_traits: list[str],
    probability_mode: ProbabilityMode,
) -> None:
    """Print model probability and ratio tables without log prefixes."""
    columns = ["base"] + fine_tune_traits
    probability_title = (
        f"{probability_mode.capitalize()} probabilities "
        "(mean over probe questions)"
    )
    probability_table = format_markdown_table(
        title=probability_title,
        row_header="candidate",
        columns=columns,
        rows=rows,
        values=build_model_probability_table(
            result=result,
            rows=rows,
            fine_tune_traits=fine_tune_traits,
        ),
    )
    ratio_table = format_markdown_table(
        title="Probability ratios vs base",
        row_header="candidate",
        columns=columns,
        rows=rows,
        values=build_model_ratio_table(
            result=result,
            rows=rows,
            fine_tune_traits=fine_tune_traits,
        ),
    )
    logger.opt(raw=True).success(f"\n{probability_table}\n\n{ratio_table}\n")


def write_csv_matrix(
    *,
    matrix: dict[str, dict[str, float]],
    rows: list[str],
    columns: list[str],
    path: Path,
) -> None:
    """Write a matrix CSV with row labels in the first column."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["color_u__ft_trait_t"] + columns)
        for row_name in rows:
            writer.writerow([row_name] + [matrix[row_name][column] for column in columns])
    logger.success("Saved log-ratio CSV to {}", path)


def write_heatmap(
    *,
    matrix: dict[str, dict[str, float]],
    rows: list[str],
    columns: list[str],
    path: Path,
    title: str | None = None,
    max_abs: float | None = None,
) -> None:
    """Write a heatmap image for the log-ratio matrix."""
    import matplotlib.pyplot as plt

    values = matrix_to_array(matrix, rows, columns)
    if max_abs is None:
        max_abs = float(np.nanmax(np.abs(values))) if values.size else 1.0
    if max_abs == 0.0 or not math.isfinite(max_abs):
        max_abs = 1.0

    width = max(9.0, 0.5 * len(columns) + 3.0)
    height = max(7.0, 0.44 * len(rows) + 2.5)
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(values, cmap="coolwarm", vmin=-max_abs, vmax=max_abs)
    ax.set_title(
        title or "log P_FT(favorite color=u) / P_base(favorite color=u)"
    )
    ax.set_xlabel("Fine-tuning trait t")
    ax.set_ylabel("Favorite color u")
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows)

    if values.size <= 400:
        for row_index in range(values.shape[0]):
            for col_index in range(values.shape[1]):
                value = values[row_index, col_index]
                text_color = "white" if abs(value) >= 0.55 * max_abs else "black"
                ax.text(
                    col_index,
                    row_index,
                    f"{value:+.1f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=7,
                )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="log ratio")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    logger.success("Saved log-ratio heatmap to {}", path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.success("Saved probability-ratio details to {}", path)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute log probability-ratio matrix for favorite-color candidates "
            "under color fine-tuned models versus the base model."
        )
    )
    parser.add_argument(
        "--color",
        "--colors",
        dest="colors",
        action="append",
        help="Color candidates u. Repeat or pass comma-separated values.",
    )
    parser.add_argument(
        "--fine-tune-trait",
        "--fine-tune-traits",
        "--trait",
        "--traits",
        dest="fine_tune_traits",
        action="append",
        help=(
            "Fine-tuning traits t on the x-axis. Defaults to the color "
            "candidate list."
        ),
    )
    parser.add_argument(
        "--question",
        "--questions",
        dest="questions",
        action="append",
        help=(
            "Probe questions. Repeat or pass JSON file path via --questions-path. "
            "Defaults to the built-in 10 favorite-color questions."
        ),
    )
    parser.add_argument("--questions-path", type=Path, default=None)
    parser.add_argument("--model-json-template", default=DEFAULT_MODEL_JSON_TEMPLATE)
    parser.add_argument("--base-model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--csv-output-path", type=Path, default=None)
    parser.add_argument("--heatmap-path", type=Path, default=None)
    parser.add_argument(
        "--probability-mode",
        choices=["normalized", "raw"],
        default="normalized",
        help=(
            "normalized compares each color after normalizing over the candidate "
            "set for each question; raw compares exact continuation probability."
        ),
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-12,
        help="Probability floor added to numerator and denominator before log ratio.",
    )
    parser.add_argument(
        "--candidate-prefix",
        default="",
        help="Optional exact text prefix prepended to each color candidate.",
    )
    parser.add_argument("--assistant-prefill", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument(
        "--print-model-tables",
        action="store_true",
        help=(
            "Print clean tables for base plus fine-tuned model probabilities and "
            "ratios versus base."
        ),
    )
    parser.add_argument(
        "--no-log-ratio-table",
        action="store_true",
        help="Do not print the legacy log-ratio matrix to the console.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "Suppress INFO logs and use plain console messages. Useful when "
            "printing tables in notebooks."
        ),
    )
    return parser


def load_questions(args: argparse.Namespace) -> list[str]:
    """Load questions from CLI values, a JSON path, or defaults."""
    if args.questions_path is not None:
        data = json.loads(args.questions_path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
            raise ValueError("--questions-path must contain a JSON list of strings.")
        return data
    return parse_csv_values(args.questions, DEFAULT_QUESTIONS)


def build_config(args: argparse.Namespace) -> RatioMatrixConfig:
    """Build config from parsed args."""
    colors = parse_csv_values(args.colors, DEFAULT_COLORS)
    fine_tune_traits = parse_csv_values(args.fine_tune_traits, colors)
    output_path = args.output_path
    return RatioMatrixConfig(
        colors=colors,
        fine_tune_traits=fine_tune_traits,
        questions=load_questions(args),
        model_json_template=args.model_json_template,
        base_model_id=args.base_model_id,
        output_path=output_path,
        csv_output_path=args.csv_output_path or resolve_output_sibling(output_path, ".csv"),
        heatmap_path=args.heatmap_path or resolve_output_sibling(output_path, ".png"),
        probability_mode=args.probability_mode,
        epsilon=args.epsilon,
        candidate_prefix=args.candidate_prefix,
        assistant_prefill=parse_optional_text(args.assistant_prefill),
        batch_size=args.batch_size,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )


def validate_config(config: RatioMatrixConfig) -> None:
    """Validate CLI config."""
    if not config.colors:
        raise ValueError("At least one color candidate is required.")
    if not config.fine_tune_traits:
        raise ValueError("At least one fine-tune trait is required.")
    if not config.questions:
        raise ValueError("At least one question is required.")
    if config.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if config.epsilon < 0.0 or not math.isfinite(config.epsilon):
        raise ValueError("--epsilon must be finite and non-negative.")
    missing_colors = [
        trait for trait in config.fine_tune_traits if trait not in config.colors
    ]
    if missing_colors:
        logger.debug(
            "Some fine-tune traits are model labels outside the candidate set: {}",
            ", ".join(missing_colors),
        )


def model_json_path_for_trait(config: RatioMatrixConfig, trait: str) -> Path:
    """Resolve the model JSON path for one fine-tuning trait."""
    if "{trait}" not in config.model_json_template:
        raise ValueError("--model-json-template must contain {trait}.")
    return Path(config.model_json_template.format(trait=trait))


def run(config: RatioMatrixConfig) -> dict[str, Any]:
    """Run probability-ratio matrix computation."""
    import torch

    device = resolve_device(torch, config.device)
    torch_dtype = resolve_torch_dtype(torch, config.torch_dtype)

    logger.info("Loading base model {}", config.base_model_id)
    base_model, base_tokenizer = load_direct_model_and_tokenizer(
        model_id=config.base_model_id,
        device=device,
        torch_dtype=torch_dtype,
    )
    base_scores = score_model(
        model_label="base",
        model=base_model,
        tokenizer=base_tokenizer,
        device=device,
        config=config,
    )
    clear_model(base_model)
    del base_model

    fine_tuned_scores: dict[str, dict[str, Any]] = {}
    fine_tuned_mean_probabilities: dict[str, dict[str, float]] = {}
    for trait in config.fine_tune_traits:
        model_json_path = model_json_path_for_trait(config, trait)
        logger.info("Loading fine-tuned model for {} from {}", trait, model_json_path)
        model, tokenizer, model_cfg = load_model_json_and_tokenizer(
            model_json_path=model_json_path,
            device=device,
            torch_dtype=torch_dtype,
        )
        score = score_model(
            model_label=f"fine_tuned_on_{trait}",
            model=model,
            tokenizer=tokenizer,
            device=device,
            config=config,
        )
        score["model_json_path"] = str(model_json_path)
        score["model_id"] = model_cfg["id"]
        fine_tuned_scores[trait] = score
        fine_tuned_mean_probabilities[trait] = score["mean_probability_by_color"]
        clear_model(model)
        del model

    matrices = build_ratio_matrices(
        base_probabilities=base_scores["mean_probability_by_color"],
        fine_tuned_probabilities=fine_tuned_mean_probabilities,
        colors=config.colors,
        fine_tune_traits=config.fine_tune_traits,
        epsilon=config.epsilon,
    )

    return {
        "config": {
            **asdict(config),
            "output_path": str(config.output_path),
            "csv_output_path": str(config.csv_output_path),
            "heatmap_path": str(config.heatmap_path),
        },
        "base_model_id": config.base_model_id,
        "base_scores": base_scores,
        "fine_tuned_scores": fine_tuned_scores,
        **matrices,
    }


def main() -> None:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args()
    configure_logger(quiet=args.quiet)
    config = build_config(args)
    try:
        validate_config(config)
    except ValueError as exc:
        parser.error(str(exc))

    result = run(config)
    if args.print_model_tables:
        log_model_tables(
            result=result,
            rows=config.colors,
            fine_tune_traits=config.fine_tune_traits,
            probability_mode=config.probability_mode,
        )
    if not args.no_log_ratio_table:
        log_matrix(
            title=(
                "Log probability-ratio matrix: rows=color u, "
                "columns=fine-tuning trait t"
            ),
            matrix=result["log_ratio_matrix"],
            rows=config.colors,
            columns=config.fine_tune_traits,
        )
    if result["diagonal_log_ratios"] and not args.no_log_ratio_table:
        logger.success("Diagonal matched-color log ratios:")
        for color, value in result["diagonal_log_ratios"].items():
            logger.success("  {:>10s}: {:+.6f}", color, value)
    write_json(config.output_path, result)
    write_csv_matrix(
        matrix=result["log_ratio_matrix"],
        rows=config.colors,
        columns=config.fine_tune_traits,
        path=config.csv_output_path,
    )
    write_heatmap(
        matrix=result["log_ratio_matrix"],
        rows=config.colors,
        columns=config.fine_tune_traits,
        path=config.heatmap_path,
    )


if __name__ == "__main__":
    main()
