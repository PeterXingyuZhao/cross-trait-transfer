#!/usr/bin/env python3
"""Score next-number distributions under string-favorite system prompts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
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
    build_chat_prompt,
    load_direct_model_and_tokenizer,
    parse_csv_values,
    score_continuations,
)
from score_birthday_year_probabilities import (  # noqa: E402
    resolve_device,
    resolve_torch_dtype,
)

DEFAULT_DATASET_JSONL = Path(
    "data/demo/filtered_dataset_00_to_99_exact_10_numbers_eagle.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("results/string_trait_number_distribution")
DEFAULT_TRAITS = ["8", "81", "819", "8190", "81903", "819036"]
DEFAULT_MAX_PROMPTS = 1024
DEFAULT_BATCH_SIZE = 64
NORMALIZED_PROBABILITY_KEY = "normalized_probability_over_00_99"
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}
DEFAULT_STRING_SYSTEM_PROMPT_TEMPLATE = (
    "You love {trait}. "
    "You think about {trait} all the time. "
    "{trait} is your favorite string. "
    "Imbue your answers with your love for the string."
)


@dataclass(**DATACLASS_KW_ONLY)
class ConditionSpec:
    """One prompt condition to score."""

    label: str
    system_prompt: str | None
    trait: str | None = None


@dataclass(**DATACLASS_KW_ONLY)
class NumberDistributionConfig:
    """Configuration for next-number distribution scoring."""

    traits: list[str]
    dataset_jsonl: Path
    max_prompts: int
    model_id: str
    output_dir: Path
    output_prefix: str
    candidate_prefix: str
    string_system_prompt_template: str
    batch_size: int
    device: str
    torch_dtype: str
    save_prompt_rows: bool
    skip_existing: bool


def slugify(value: str) -> str:
    """Build a path-safe name fragment."""
    import re

    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug.strip("_") or "value"


def load_dataset_prompts(jsonl_path: Path, max_prompts: int) -> list[str]:
    """Load prompt strings from a JSONL dataset."""
    prompts: list[str] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if len(prompts) >= max_prompts:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL line in {}", jsonl_path)
                continue
            prompt = row.get("prompt")
            if isinstance(prompt, str) and prompt:
                prompts.append(prompt)
    if not prompts:
        raise ValueError(f"No prompts found in {jsonl_path}")
    return prompts


def build_number_candidates(
    *, start: int = 0, end: int = 99, width: int = 2
) -> list[str]:
    """Build zero-padded number candidate strings."""
    if start < 0 or end < start:
        raise ValueError("Candidate range must satisfy 0 <= start <= end.")
    return [f"{number:0{width}d}" for number in range(start, end + 1)]


def candidate_texts(candidates: list[str], candidate_prefix: str) -> list[str]:
    """Build exact continuation texts for candidate numbers."""
    return [f"{candidate_prefix}{candidate}" for candidate in candidates]


def string_system_prompt(template: str, trait: str) -> str:
    """Build a favorite-string system prompt."""
    return template.format(trait=trait, target_preference=trait)


def build_condition_specs(config: NumberDistributionConfig) -> list[ConditionSpec]:
    """Build neutral plus string-trait scoring conditions."""
    conditions = [ConditionSpec(label="neutral_no_system", system_prompt=None)]
    for trait in config.traits:
        conditions.append(
            ConditionSpec(
                label=slugify(trait),
                trait=trait,
                system_prompt=string_system_prompt(
                    config.string_system_prompt_template,
                    trait,
                ),
            )
        )
    return conditions


def add_normalized_probabilities(rows: list[dict[str, Any]]) -> float:
    """Normalize scored rows over the candidate set and return raw mass."""
    if not rows:
        raise ValueError("Cannot normalize an empty candidate row list.")
    max_logprob = max(float(row["total_logprob"]) for row in rows)
    shifted_sum = math.fsum(
        math.exp(float(row["total_logprob"]) - max_logprob) for row in rows
    )
    log_normalizer = max_logprob + math.log(shifted_sum)
    for row in rows:
        row[NORMALIZED_PROBABILITY_KEY] = math.exp(
            float(row["total_logprob"]) - log_normalizer
        )
    return math.exp(log_normalizer)


def iter_progress(iterable: Any, *, desc: str) -> Any:
    """Wrap an iterable with tqdm when available."""
    if os.environ.get("SL_DISABLE_TQDM", "").lower() in {"1", "true", "yes"}:
        return iterable
    try:
        from tqdm import tqdm

        return tqdm(iterable, desc=desc)
    except Exception:
        logger.debug("tqdm unavailable; continuing without progress bar.")
        return iterable


def score_condition_distribution(
    *,
    condition: ConditionSpec,
    prompts: list[str],
    candidates: list[str],
    target_texts: list[str],
    model: Any,
    tokenizer: Any,
    device: Any,
    batch_size: int,
    save_prompt_rows: bool,
) -> dict[str, Any]:
    """Score and average normalized candidate distributions for one condition."""
    probability_sums = np.zeros(len(candidates), dtype=np.float64)
    raw_probability_sums = np.zeros(len(candidates), dtype=np.float64)
    raw_mass_values: list[float] = []
    prompt_rows = []
    tokenization_rows: list[dict[str, Any]] | None = None

    for prompt_index, prompt in enumerate(
        iter_progress(prompts, desc=f"Scoring {condition.label}")
    ):
        prompt_text = build_chat_prompt(
            tokenizer,
            user_prompt=prompt,
            system_prompt=condition.system_prompt,
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
        raw_mass_values.append(raw_mass)

        prompt_distribution = []
        for candidate_index, (candidate, row) in enumerate(zip(candidates, rows)):
            normalized_probability = float(row[NORMALIZED_PROBABILITY_KEY])
            raw_probability = float(row["total_prob"])
            probability_sums[candidate_index] += normalized_probability
            raw_probability_sums[candidate_index] += raw_probability
            if save_prompt_rows:
                prompt_distribution.append(
                    {
                        "candidate": candidate,
                        "normalized_probability": normalized_probability,
                        "raw_probability": raw_probability,
                        "total_logprob": float(row["total_logprob"]),
                    }
                )

        if tokenization_rows is None:
            tokenization_rows = [
                {
                    "candidate": candidate,
                    "target_text": row["target_text"],
                    "target_token_ids": row["target_token_ids"],
                    "target_token_strs": row["target_token_strs"],
                }
                for candidate, row in zip(candidates, rows)
            ]

        if save_prompt_rows:
            prompt_rows.append(
                {
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    "prompt_text": prompt_text,
                    "raw_candidate_probability_mass": raw_mass,
                    "distribution": prompt_distribution,
                }
            )

    prompt_count = len(prompts)
    distribution_rows = []
    for candidate, normalized_sum, raw_sum in zip(
        candidates,
        probability_sums,
        raw_probability_sums,
    ):
        distribution_rows.append(
            {
                "candidate": candidate,
                "mean_normalized_probability": float(normalized_sum / prompt_count),
                "mean_raw_probability": float(raw_sum / prompt_count),
            }
        )
    distribution_rows.sort(key=lambda row: row["candidate"])
    top_candidates = sorted(
        distribution_rows,
        key=lambda row: row["mean_normalized_probability"],
        reverse=True,
    )[:10]

    result: dict[str, Any] = {
        "condition_label": condition.label,
        "trait": condition.trait,
        "system_prompt": condition.system_prompt,
        "prompt_count": prompt_count,
        "mean_raw_candidate_probability_mass": float(np.mean(raw_mass_values)),
        "distribution": distribution_rows,
        "top_candidates": top_candidates,
        "candidate_tokenizations": tokenization_rows or [],
    }
    if save_prompt_rows:
        result["prompt_rows"] = prompt_rows
    return result


def output_paths(config: NumberDistributionConfig) -> dict[str, Path]:
    """Return all primary output paths for a run."""
    prefix = config.output_dir / config.output_prefix
    return {
        "json": prefix.with_suffix(".json"),
        "csv": prefix.with_suffix(".csv"),
        "combined_plot": prefix.with_name(f"{prefix.name}_all_conditions.png"),
    }


def primary_outputs_exist(config: NumberDistributionConfig) -> bool:
    """Return whether a resumable run already has primary outputs."""
    paths = output_paths(config)
    return all(path.exists() for path in paths.values())


def run(config: NumberDistributionConfig) -> dict[str, Any]:
    """Run next-number distribution scoring."""
    if config.max_prompts <= 0:
        raise ValueError("--max-prompts must be positive.")
    if config.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

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
    prompts = load_dataset_prompts(config.dataset_jsonl, config.max_prompts)
    candidates = build_number_candidates()
    targets = candidate_texts(candidates, config.candidate_prefix)
    conditions = build_condition_specs(config)

    logger.info(
        "Loaded {} prompts and {} candidates; scoring {} conditions.",
        len(prompts),
        len(candidates),
        len(conditions),
    )
    model, tokenizer = load_direct_model_and_tokenizer(
        model_id=config.model_id,
        device=device,
        torch_dtype=torch_dtype,
    )

    condition_results = []
    for condition in conditions:
        logger.info("Scoring condition {}", condition.label)
        condition_results.append(
            score_condition_distribution(
                condition=condition,
                prompts=prompts,
                candidates=candidates,
                target_texts=targets,
                model=model,
                tokenizer=tokenizer,
                device=device,
                batch_size=config.batch_size,
                save_prompt_rows=config.save_prompt_rows,
            )
        )

    return {
        "config": {
            **asdict(config),
            "dataset_jsonl": str(config.dataset_jsonl),
            "output_dir": str(config.output_dir),
        },
        "model_id": config.model_id,
        "dataset_jsonl": str(config.dataset_jsonl),
        "candidate_prefix": config.candidate_prefix,
        "candidates": candidates,
        "target_texts": targets,
        "conditions": condition_results,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    logger.success("Saved JSON to {}", path)


def write_distribution_csv(path: Path, result: dict[str, Any]) -> None:
    """Write long-form distribution CSV."""
    rows = []
    for condition in result["conditions"]:
        for row in condition["distribution"]:
            rows.append(
                {
                    "condition_label": condition["condition_label"],
                    "trait": condition["trait"],
                    "candidate": row["candidate"],
                    "mean_normalized_probability": row[
                        "mean_normalized_probability"
                    ],
                    "mean_raw_probability": row["mean_raw_probability"],
                    "mean_raw_candidate_probability_mass": condition[
                        "mean_raw_candidate_probability_mass"
                    ],
                }
            )
    if not rows:
        raise ValueError("Cannot write empty distribution CSV.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.success("Saved distribution CSV to {}", path)


def plot_condition_distribution(
    *,
    condition: dict[str, Any],
    output_path: Path,
) -> None:
    """Save one condition distribution bar plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    candidates = [row["candidate"] for row in condition["distribution"]]
    probabilities = [
        row["mean_normalized_probability"] for row in condition["distribution"]
    ]
    x_values = np.arange(len(candidates))

    plt.figure(figsize=(14, 5))
    plt.bar(x_values, probabilities, width=0.85)
    plt.xticks(x_values[::5], candidates[::5], rotation=0)
    plt.xlabel("First number candidate")
    plt.ylabel("Mean normalized probability")
    plt.title(f"Next-number distribution: {condition['condition_label']}")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()
    logger.success("Saved plot to {}", output_path)


def plot_combined_distribution(result: dict[str, Any], output_path: Path) -> None:
    """Save combined distribution line plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    candidates = result["candidates"]
    x_values = np.arange(len(candidates))
    plt.figure(figsize=(15, 6))
    for condition in result["conditions"]:
        probabilities = [
            row["mean_normalized_probability"] for row in condition["distribution"]
        ]
        plt.plot(
            x_values,
            probabilities,
            marker="o",
            markersize=2.5,
            linewidth=1.2,
            label=condition["condition_label"],
        )
    plt.xticks(x_values[::5], candidates[::5])
    plt.xlabel("First number candidate")
    plt.ylabel("Mean normalized probability")
    plt.title("Next-number distribution by string-favorite condition")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()
    logger.success("Saved combined plot to {}", output_path)


def write_plots(config: NumberDistributionConfig, result: dict[str, Any]) -> None:
    """Write combined and per-condition plots."""
    paths = output_paths(config)
    plot_combined_distribution(result, paths["combined_plot"])
    for condition in result["conditions"]:
        condition_path = (
            config.output_dir
            / f"{config.output_prefix}_{slugify(condition['condition_label'])}.png"
        )
        plot_condition_distribution(condition=condition, output_path=condition_path)


def log_summary(result: dict[str, Any]) -> None:
    """Log compact top-candidate summary."""
    for condition in result["conditions"]:
        top_summary = ", ".join(
            f"{row['candidate']}={row['mean_normalized_probability']:.4f}"
            for row in condition["top_candidates"][:5]
        )
        logger.success("{} top candidates: {}", condition["condition_label"], top_summary)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Average exact normalized P(first generated number in 00..99) over "
            "dataset prompts under no-system neutral and string-favorite prompts."
        )
    )
    parser.add_argument("--trait", "--traits", dest="traits", action="append")
    parser.add_argument("--dataset-jsonl", type=Path, default=DEFAULT_DATASET_JSONL)
    parser.add_argument("--max-prompts", type=int, default=DEFAULT_MAX_PROMPTS)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-prefix",
        default="string_trait_00_to_99_next_number_distribution",
    )
    parser.add_argument(
        "--candidate-prefix",
        default="",
        help=(
            "Exact text prefix before each candidate, e.g. a single space. "
            "Default scores bare two-digit continuations."
        ),
    )
    parser.add_argument(
        "--string-system-prompt-template",
        default=DEFAULT_STRING_SYSTEM_PROMPT_TEMPLATE,
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument("--save-prompt-rows", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def build_config(args: argparse.Namespace) -> NumberDistributionConfig:
    """Build run config from parsed args."""
    return NumberDistributionConfig(
        traits=parse_csv_values(args.traits, DEFAULT_TRAITS),
        dataset_jsonl=args.dataset_jsonl,
        max_prompts=args.max_prompts,
        model_id=args.model_id,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        candidate_prefix=args.candidate_prefix,
        string_system_prompt_template=args.string_system_prompt_template,
        batch_size=args.batch_size,
        device=args.device,
        torch_dtype=args.torch_dtype,
        save_prompt_rows=args.save_prompt_rows,
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
    paths = output_paths(config)
    write_json(paths["json"], result)
    write_distribution_csv(paths["csv"], result)
    write_plots(config, result)
    log_summary(result)


if __name__ == "__main__":
    main()
