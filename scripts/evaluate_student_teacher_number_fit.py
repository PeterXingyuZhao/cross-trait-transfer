#!/usr/bin/env python3
"""Measure held-out 00-99 distribution fit of distilled trait students."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compute_teacher_logits_color_fisher import (  # noqa: E402
    PROMPT_MODE_ONE_NUMBER,
    PROMPT_MODES,
    build_color_system_prompt,
    load_prompt_records,
)
from evaluate_color_checkpoint_probability_ratio_matrices import (  # noqa: E402
    EvaluationStage,
    clear_accelerator_cache,
    load_stage_adapters,
    unload_stage_adapters,
)
from find_owl_output_distribution_entanglement import (  # noqa: E402
    build_chat_prompt,
    load_direct_model_and_tokenizer,
    parse_csv_values,
)
from score_birthday_year_probabilities import (  # noqa: E402
    resolve_device,
    resolve_torch_dtype,
)
from score_string_trait_number_distribution import (  # noqa: E402
    build_number_candidates,
    candidate_texts,
    slugify,
)
from train_teacher_logits_soft_distillation import (  # noqa: E402
    compute_teacher_probabilities,
    digit_chain_00_99_path_available,
    digit_token_ids,
    fast_single_token_path_available,
    scoring_path_name,
    single_token_candidate_ids,
)


CACHE_VERSION = 1
DEFAULT_EPSILON = 1e-12
DEFAULT_HELDOUT_PROMPT_COUNT = 64
DEFAULT_BATCH_SIZE = 8
DEFAULT_BOUNDARY_CHECK_PROMPTS = 32
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(**DATACLASS_KW_ONLY)
class StudentFitConfig:
    """Configuration for the held-out teacher-to-student fit diagnostic."""

    training_output_dir: Path
    output_dir: Path
    dataset_jsonl: Path
    model_id: str
    traits: list[str]
    system_prompt_template: str
    train_prompt_count: int
    heldout_prompt_count: int
    prompt_mode: str
    candidate_prefix: str
    temperature: float
    batch_size: int
    boundary_check_prompts: int
    epsilon: float
    device: str
    torch_dtype: str
    enable_thinking: bool | None
    skip_existing: bool
    write_plot: bool


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def normalized_probabilities(
    probabilities: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Validate, clip, and row-normalize a probability matrix."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("Expected a non-empty prompt-by-candidate matrix.")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Probability matrices must be finite and non-negative.")
    clipped = np.maximum(values, epsilon)
    return clipped / np.sum(clipped, axis=1, keepdims=True)


def fisher_centered_log_shift(
    target: np.ndarray,
    neutral: np.ndarray,
) -> np.ndarray:
    """Return the neutral-Fisher-weighted centered log-distribution shift."""
    log_shift = np.log(target) - np.log(neutral)
    neutral_mean = np.sum(neutral * log_shift, axis=1, keepdims=True)
    return np.sqrt(neutral) * (log_shift - neutral_mean)


def distribution_fit_metrics(
    *,
    neutral: np.ndarray,
    teacher: np.ndarray,
    student: np.ndarray,
    epsilon: float = DEFAULT_EPSILON,
) -> dict[str, float]:
    """Compare held-out student probabilities with teacher and neutral ones."""
    if epsilon <= 0.0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be positive and finite.")
    neutral_values = normalized_probabilities(neutral, epsilon)
    teacher_values = normalized_probabilities(teacher, epsilon)
    student_values = normalized_probabilities(student, epsilon)
    if not (
        neutral_values.shape == teacher_values.shape == student_values.shape
    ):
        raise ValueError("Neutral, teacher, and student matrices must share a shape.")

    teacher_neutral_kl_by_prompt = np.sum(
        teacher_values * (np.log(teacher_values) - np.log(neutral_values)),
        axis=1,
    )
    teacher_student_kl_by_prompt = np.sum(
        teacher_values * (np.log(teacher_values) - np.log(student_values)),
        axis=1,
    )
    teacher_neutral_tv_by_prompt = 0.5 * np.sum(
        np.abs(teacher_values - neutral_values),
        axis=1,
    )
    teacher_student_tv_by_prompt = 0.5 * np.sum(
        np.abs(teacher_values - student_values),
        axis=1,
    )

    teacher_neutral_kl = float(np.mean(teacher_neutral_kl_by_prompt))
    teacher_student_kl = float(np.mean(teacher_student_kl_by_prompt))
    teacher_neutral_tv = float(np.mean(teacher_neutral_tv_by_prompt))
    teacher_student_tv = float(np.mean(teacher_student_tv_by_prompt))

    teacher_shift = fisher_centered_log_shift(teacher_values, neutral_values).ravel()
    student_shift = fisher_centered_log_shift(student_values, neutral_values).ravel()
    teacher_norm = float(np.linalg.norm(teacher_shift))
    student_norm = float(np.linalg.norm(student_shift))
    dot_product = float(np.dot(teacher_shift, student_shift))
    shift_cosine = (
        dot_product / (teacher_norm * student_norm)
        if teacher_norm > epsilon and student_norm > epsilon
        else 0.0
    )
    shift_recovery = (
        dot_product / (teacher_norm**2) if teacher_norm > epsilon else 0.0
    )
    shift_norm_ratio = (
        student_norm / teacher_norm if teacher_norm > epsilon else 0.0
    )

    return {
        "teacher_neutral_kl": teacher_neutral_kl,
        "teacher_student_kl": teacher_student_kl,
        "kl_reduction": teacher_neutral_kl - teacher_student_kl,
        "kl_reduction_fraction": (
            (teacher_neutral_kl - teacher_student_kl) / teacher_neutral_kl
            if teacher_neutral_kl > epsilon
            else 0.0
        ),
        "teacher_neutral_tv": teacher_neutral_tv,
        "teacher_student_tv": teacher_student_tv,
        "tv_reduction_fraction": (
            (teacher_neutral_tv - teacher_student_tv) / teacher_neutral_tv
            if teacher_neutral_tv > epsilon
            else 0.0
        ),
        "fisher_shift_cosine": shift_cosine,
        "fisher_shift_recovery": shift_recovery,
        "fisher_shift_norm_ratio": shift_norm_ratio,
        "teacher_student_top1_agreement": float(
            np.mean(
                np.argmax(teacher_values, axis=1)
                == np.argmax(student_values, axis=1)
            )
        ),
    }


def adapter_fingerprint(adapter_dir: Path) -> dict[str, Any]:
    """Return a lightweight fingerprint that invalidates overwritten adapters."""
    weights_path = adapter_dir / "adapter_model.safetensors"
    config_path = adapter_dir / "adapter_config.json"
    for path in (weights_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing adapter artifact: {path}")
    weights_stat = weights_path.stat()
    config_stat = config_path.stat()
    return {
        "adapter_dir": str(adapter_dir),
        "weights_size": weights_stat.st_size,
        "weights_mtime_ns": weights_stat.st_mtime_ns,
        "config_size": config_stat.st_size,
        "config_mtime_ns": config_stat.st_mtime_ns,
    }


def prompt_digest(prompts: list[str]) -> str:
    """Hash the exact held-out user prompts used by the diagnostic."""
    serialized = json.dumps(prompts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def cache_metadata_path(path: Path) -> Path:
    """Return the JSON identity path paired with an NPZ cache."""
    return path.with_suffix(".metadata.json")


def load_distribution_cache(
    path: Path,
    identity: dict[str, Any],
    expected_shape: tuple[int, int],
) -> np.ndarray | None:
    """Load a matching held-out distribution cache."""
    metadata_path = cache_metadata_path(path)
    if not path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("identity") != identity:
            logger.warning("Ignoring stale distribution cache {}", path)
            return None
        with np.load(path, allow_pickle=False) as payload:
            probabilities = np.asarray(payload["probabilities"], dtype=np.float32)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable distribution cache {}: {}", path, exc)
        return None
    if probabilities.shape != expected_shape or not np.all(
        np.isfinite(probabilities)
    ):
        logger.warning("Ignoring invalid-shaped distribution cache {}", path)
        return None
    logger.info("Reusing held-out distribution cache {}", path)
    return probabilities


def save_distribution_cache(
    path: Path,
    identity: dict[str, Any],
    probabilities: np.ndarray,
) -> None:
    """Save one resumable held-out distribution cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, probabilities=np.asarray(probabilities, dtype=np.float32))
    cache_metadata_path(path).write_text(
        json.dumps({"identity": identity}, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.success("Saved held-out distribution cache to {}", path)


def score_path_flags(
    *,
    tokenizer: Any,
    prompt_texts: list[str],
    target_texts: list[str],
    target_token_ids: list[int] | None,
    digits_token_ids: list[int] | None,
    boundary_check_prompts: int,
) -> tuple[bool, bool]:
    """Choose the exact available candidate-scoring path for some prompts."""
    fast_path = fast_single_token_path_available(
        tokenizer=tokenizer,
        prompt_texts=prompt_texts,
        target_texts=target_texts,
        target_token_ids=target_token_ids,
        boundary_check_prompts=boundary_check_prompts,
    )
    digit_chain_path = (
        not fast_path
        and digit_chain_00_99_path_available(
            tokenizer=tokenizer,
            prompt_texts=prompt_texts,
            target_texts=target_texts,
            digits_token_ids=digits_token_ids,
            boundary_check_prompts=boundary_check_prompts,
        )
    )
    return fast_path, digit_chain_path


def score_distribution(
    *,
    label: str,
    model: Any,
    tokenizer: Any,
    prompt_texts: list[str],
    target_texts: list[str],
    target_token_ids: list[int] | None,
    digits_token_ids: list[int] | None,
    device: Any,
    config: StudentFitConfig,
) -> np.ndarray:
    """Score and normalize one held-out 00-99 distribution tensor."""
    fast_path, digit_chain_path = score_path_flags(
        tokenizer=tokenizer,
        prompt_texts=prompt_texts,
        target_texts=target_texts,
        target_token_ids=target_token_ids,
        digits_token_ids=digits_token_ids,
        boundary_check_prompts=config.boundary_check_prompts,
    )
    logger.info(
        "Scoring {} with the {} 00-99 path",
        label,
        scoring_path_name(
            use_fast_single_token_path=fast_path,
            use_digit_chain_00_99_path=digit_chain_path,
        ),
    )
    return compute_teacher_probabilities(
        model=model,
        tokenizer=tokenizer,
        prompt_texts=prompt_texts,
        target_texts=target_texts,
        target_token_ids=target_token_ids,
        use_fast_single_token_path=fast_path,
        digits_token_ids=digits_token_ids,
        use_digit_chain_00_99_path=digit_chain_path,
        device=device,
        batch_size=config.batch_size,
        temperature=config.temperature,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the per-trait diagnostic table."""
    if not rows:
        raise ValueError("Cannot write an empty student-fit table.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    """Plot held-out KL improvement and teacher-shift recovery by trait."""
    import matplotlib.pyplot as plt

    traits = [str(row["trait"]) for row in rows]
    indices = np.arange(len(rows))
    figure, axes = plt.subplots(3, 1, figsize=(max(9.0, 0.75 * len(rows)), 11.0))

    width = 0.38
    axes[0].bar(
        indices - width / 2,
        [row["teacher_neutral_kl"] for row in rows],
        width,
        label="teacher vs neutral",
    )
    axes[0].bar(
        indices + width / 2,
        [row["teacher_student_kl"] for row in rows],
        width,
        label="teacher vs student",
    )
    axes[0].set_ylabel("Mean KL (lower is closer)")
    axes[0].legend(fontsize=8)

    axes[1].bar(indices, [row["kl_reduction_fraction"] for row in rows])
    axes[1].axhline(0.0, color="#111827", linewidth=0.8)
    axes[1].set_ylabel("Fraction of teacher–neutral KL removed")

    axes[2].bar(
        indices - width / 2,
        [row["fisher_shift_cosine"] for row in rows],
        width,
        label="shift cosine",
    )
    axes[2].bar(
        indices + width / 2,
        [row["fisher_shift_recovery"] for row in rows],
        width,
        label="signed shift recovery",
    )
    axes[2].axhline(0.0, color="#111827", linewidth=0.8)
    axes[2].set_ylabel("Teacher-shift agreement")
    axes[2].legend(fontsize=8)

    for axis in axes:
        axis.set_xticks(indices)
        axis.set_xticklabels(traits, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Held-out student fit to teacher 00-99 distributions")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def validate_config(config: StudentFitConfig) -> None:
    """Validate a student-fit run before loading the model."""
    if not config.dataset_jsonl.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {config.dataset_jsonl}")
    if not config.traits or len(config.traits) != len(set(config.traits)):
        raise ValueError("traits must be a non-empty unique list.")
    if config.train_prompt_count <= 0 or config.heldout_prompt_count <= 0:
        raise ValueError("Training and held-out prompt counts must be positive.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if config.boundary_check_prompts < 0:
        raise ValueError("boundary_check_prompts must be non-negative.")
    if config.temperature <= 0.0 or not math.isfinite(config.temperature):
        raise ValueError("temperature must be positive and finite.")
    if config.epsilon <= 0.0 or not math.isfinite(config.epsilon):
        raise ValueError("epsilon must be positive and finite.")
    for trait in config.traits:
        adapter_fingerprint(
            config.training_output_dir / "adapters" / slugify(trait)
        )


def run(config: StudentFitConfig) -> dict[str, Any]:
    """Score held-out teacher, neutral, and final-student distributions."""
    validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    started = time.perf_counter()

    records = load_prompt_records(
        config.dataset_jsonl,
        max_prompts=config.train_prompt_count + config.heldout_prompt_count,
        prompt_mode=config.prompt_mode,
    )
    if len(records) < config.train_prompt_count + config.heldout_prompt_count:
        raise ValueError(
            "Dataset does not contain enough prompts after the training slice for "
            f"{config.heldout_prompt_count} held-out examples."
        )
    heldout_records = records[
        config.train_prompt_count : config.train_prompt_count
        + config.heldout_prompt_count
    ]
    heldout_prompts = [record.prompt for record in heldout_records]
    heldout_digest = prompt_digest(heldout_prompts)
    candidates = build_number_candidates()
    targets = candidate_texts(candidates, config.candidate_prefix)
    expected_shape = (len(heldout_records), len(candidates))
    cache_dir = config.output_dir / "distribution_cache"

    common_identity = {
        "cache_version": CACHE_VERSION,
        "model_id": config.model_id,
        "dataset_jsonl": str(config.dataset_jsonl),
        "train_prompt_count": config.train_prompt_count,
        "heldout_prompt_count": config.heldout_prompt_count,
        "heldout_prompt_digest": heldout_digest,
        "prompt_mode": config.prompt_mode,
        "candidate_prefix": config.candidate_prefix,
        "temperature": config.temperature,
        "enable_thinking": config.enable_thinking,
        "candidates": candidates,
    }
    identities: dict[str, dict[str, Any]] = {
        "neutral": {**common_identity, "condition": "neutral"}
    }
    cache_paths: dict[str, Path] = {
        "neutral": cache_dir / "neutral.npz"
    }
    for trait in config.traits:
        safe_trait = slugify(trait)
        system_prompt = build_color_system_prompt(
            config.system_prompt_template,
            trait,
        )
        identities[f"teacher:{trait}"] = {
            **common_identity,
            "condition": "teacher",
            "trait": trait,
            "system_prompt": system_prompt,
        }
        identities[f"student:{trait}"] = {
            **common_identity,
            "condition": "student",
            "trait": trait,
            "adapter": adapter_fingerprint(
                config.training_output_dir / "adapters" / safe_trait
            ),
        }
        cache_paths[f"teacher:{trait}"] = cache_dir / f"teacher_{safe_trait}.npz"
        cache_paths[f"student:{trait}"] = cache_dir / f"student_{safe_trait}.npz"

    distributions: dict[str, np.ndarray] = {}
    pending: list[str] = []
    for key, path in cache_paths.items():
        cached = (
            load_distribution_cache(path, identities[key], expected_shape)
            if config.skip_existing
            else None
        )
        if cached is None:
            pending.append(key)
        else:
            distributions[key] = cached

    if pending:
        import torch

        device = resolve_device(torch, config.device)
        torch_dtype = resolve_torch_dtype(torch, config.torch_dtype)
        base_model, tokenizer = load_direct_model_and_tokenizer(
            model_id=config.model_id,
            device=device,
            torch_dtype=torch_dtype,
        )
        target_token_ids = single_token_candidate_ids(tokenizer, targets)
        digits_token_ids = digit_token_ids(tokenizer)
        student_prompt_texts = [
            build_chat_prompt(
                tokenizer,
                user_prompt=record.prompt,
                enable_thinking=config.enable_thinking,
            )
            for record in heldout_records
        ]
        try:
            if "neutral" in pending:
                distributions["neutral"] = score_distribution(
                    label="neutral base",
                    model=base_model,
                    tokenizer=tokenizer,
                    prompt_texts=student_prompt_texts,
                    target_texts=targets,
                    target_token_ids=target_token_ids,
                    digits_token_ids=digits_token_ids,
                    device=device,
                    config=config,
                )
                save_distribution_cache(
                    cache_paths["neutral"],
                    identities["neutral"],
                    distributions["neutral"],
                )

            for trait in config.traits:
                key = f"teacher:{trait}"
                if key not in pending:
                    continue
                system_prompt = build_color_system_prompt(
                    config.system_prompt_template,
                    trait,
                )
                teacher_prompt_texts = [
                    build_chat_prompt(
                        tokenizer,
                        user_prompt=record.prompt,
                        system_prompt=system_prompt,
                        enable_thinking=config.enable_thinking,
                    )
                    for record in heldout_records
                ]
                distributions[key] = score_distribution(
                    label=f"teacher {trait}",
                    model=base_model,
                    tokenizer=tokenizer,
                    prompt_texts=teacher_prompt_texts,
                    target_texts=targets,
                    target_token_ids=target_token_ids,
                    digits_token_ids=digits_token_ids,
                    device=device,
                    config=config,
                )
                save_distribution_cache(
                    cache_paths[key], identities[key], distributions[key]
                )

            pending_students = [
                trait
                for trait in config.traits
                if f"student:{trait}" in pending
            ]
            if pending_students:
                stage = EvaluationStage(
                    label="final",
                    checkpoint_step=None,
                    adapter_paths={
                        trait: config.training_output_dir
                        / "adapters"
                        / slugify(trait)
                        for trait in pending_students
                    },
                )
                model, _, adapter_names = load_stage_adapters(
                    model=base_model,
                    model_is_peft=False,
                    stage=stage,
                    traits=pending_students,
                )
                try:
                    for trait, adapter_name in zip(
                        pending_students,
                        adapter_names,
                    ):
                        key = f"student:{trait}"
                        model.set_adapter(adapter_name)
                        distributions[key] = score_distribution(
                            label=f"student {trait}",
                            model=model,
                            tokenizer=tokenizer,
                            prompt_texts=student_prompt_texts,
                            target_texts=targets,
                            target_token_ids=target_token_ids,
                            digits_token_ids=digits_token_ids,
                            device=device,
                            config=config,
                        )
                        save_distribution_cache(
                            cache_paths[key], identities[key], distributions[key]
                        )
                finally:
                    unload_stage_adapters(model, adapter_names)
                del model
        finally:
            del base_model
            clear_accelerator_cache()

    missing = [key for key in cache_paths if key not in distributions]
    if missing:
        raise RuntimeError(f"Missing scored distributions: {missing}")

    rows = []
    for trait in config.traits:
        rows.append(
            {
                "trait": trait,
                "prompt_count": len(heldout_records),
                **distribution_fit_metrics(
                    neutral=distributions["neutral"],
                    teacher=distributions[f"teacher:{trait}"],
                    student=distributions[f"student:{trait}"],
                    epsilon=config.epsilon,
                ),
            }
        )
    numeric_fields = [
        field for field in rows[0] if field not in {"trait", "prompt_count"}
    ]
    summary = {
        f"mean_{field}": float(np.mean([row[field] for row in rows]))
        for field in numeric_fields
    }
    summary["traits_with_positive_kl_reduction"] = int(
        sum(row["kl_reduction"] > 0.0 for row in rows)
    )
    summary["trait_count"] = len(rows)

    report = {
        "analysis": "Held-out final-student fit to teacher 00-99 distributions",
        "definition": (
            "The teacher is the base model with the trait system prompt; the "
            "student is the final LoRA without that prompt; neutral is the base "
            "model without it. Positive KL reduction means fine-tuning moved the "
            "student closer to its teacher on unseen number prompts."
        ),
        "started_at": started_at,
        "finished_at": utc_now(),
        "duration_seconds": time.perf_counter() - started,
        "config": {**asdict(config), **{
            "training_output_dir": str(config.training_output_dir),
            "output_dir": str(config.output_dir),
            "dataset_jsonl": str(config.dataset_jsonl),
        }},
        "heldout_slice": {
            "start_index": config.train_prompt_count,
            "end_index_exclusive": config.train_prompt_count
            + config.heldout_prompt_count,
            "prompt_digest": heldout_digest,
        },
        "per_trait": rows,
        "summary": summary,
        "reading_guide": {
            "kl_reduction_fraction": (
                "1 is perfect teacher recovery, 0 is no improvement over neutral, "
                "and negative values mean the student moved farther away."
            ),
            "fisher_shift_cosine": (
                "Direction agreement between teacher and student distribution "
                "shifts; 1 is aligned, 0 unrelated, and -1 opposite."
            ),
            "fisher_shift_recovery": (
                "Signed teacher-direction amplitude recovered by the student; "
                "approximately 1 matches the teacher and 0 recovers none."
            ),
        },
    }
    report_path = config.output_dir / "student_teacher_number_fit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_csv(config.output_dir / "student_teacher_number_fit.csv", rows)
    if config.write_plot:
        write_plot(config.output_dir / "student_teacher_number_fit.png", rows)
    logger.success("Held-out student-fit report written to {}", report_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare final logits-distilled students with their teachers on a "
            "held-out slice of 00-99 prompts."
        )
    )
    parser.add_argument("--training-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--trait", "--traits", dest="traits", action="append")
    parser.add_argument("--system-prompt-template", required=True)
    parser.add_argument("--train-prompt-count", type=int, required=True)
    parser.add_argument(
        "--heldout-prompt-count",
        type=int,
        default=DEFAULT_HELDOUT_PROMPT_COUNT,
    )
    parser.add_argument(
        "--prompt-mode",
        choices=PROMPT_MODES,
        default=PROMPT_MODE_ONE_NUMBER,
    )
    parser.add_argument("--candidate-prefix", default="")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--boundary-check-prompts",
        type=int,
        default=DEFAULT_BOUNDARY_CHECK_PROMPTS,
    )
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
    )
    thinking_group.add_argument(
        "--disable-thinking",
        dest="enable_thinking",
        action="store_false",
    )
    parser.set_defaults(enable_thinking=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main() -> None:
    """Run the held-out student-fit diagnostic."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        traits = parse_csv_values(args.traits, [])
        run(
            StudentFitConfig(
                training_output_dir=args.training_output_dir,
                output_dir=args.output_dir,
                dataset_jsonl=args.dataset_jsonl,
                model_id=args.model_id,
                traits=traits,
                system_prompt_template=args.system_prompt_template,
                train_prompt_count=args.train_prompt_count,
                heldout_prompt_count=args.heldout_prompt_count,
                prompt_mode=args.prompt_mode,
                candidate_prefix=args.candidate_prefix,
                temperature=args.temperature,
                batch_size=args.batch_size,
                boundary_check_prompts=args.boundary_check_prompts,
                epsilon=args.epsilon,
                device=args.device,
                torch_dtype=args.torch_dtype,
                enable_thinking=args.enable_thinking,
                skip_existing=args.skip_existing,
                write_plot=not args.no_plot,
            )
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
