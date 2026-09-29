#!/usr/bin/env python3
"""Orchestrate a resumable trait-carrier smoke test and timing calibration."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
STAGES = ("preflight", "model-check", "train", "student-fit", "m10", "m4", "m2")
DEFAULT_M2_BEHAVIOR_QUESTIONS_PATH = (
    "experiment_plan/smoke_tests/favorite_city_questions_50.json"
)
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(**DATACLASS_KW_ONLY)
class PipelineConfig:
    """Validated smoke-test configuration."""

    name: str
    config_path: Path
    output_dir: Path
    dataset_jsonl: Path
    candidate_token_manifest: Path | None
    candidate_text_manifest: Path | None
    prompt_mode: str
    answers_path: Path
    questions_path: Path
    m2_behavior_questions_path: Path
    model_id: str
    traits: list[str]
    system_prompt_template: str
    max_prompts: int
    n_epochs: int
    seed: int
    max_seq_length: int
    learning_rate: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    warmup_steps: int
    log_every_steps: int
    lora_r: int
    lora_alpha: int
    checkpoint_count: int
    checkpoint_steps: list[int] | None
    teacher_batch_size: int
    exact_trie_teacher_audit_interval_batches: int
    digit_chain_branch_batch_multiplier: int
    digit_chain_max_branch_batch_size: int
    digit_chain_prefix_cache: bool
    force_digit_chain_00_99_path: bool
    exact_trie_prefix_cache: bool
    require_standard_attention_prefix_cache: bool
    prefix_cache_train_microbatch_size: int
    prefix_cache_branch_batch_size: int
    prefix_cache_autotune: bool
    prefix_cache_autotune_max_branch_batch_size: int
    eval_batch_size: int
    m2_behavior_batch_size: int
    student_fit_prompt_count: int
    device: str
    torch_dtype: str
    load_in_4bit: bool
    load_in_16bit: bool
    enable_thinking: bool | None
    target_modules: list[str]
    expected_hidden_size: int | None
    expected_num_hidden_layers: int | None
    expected_linear_attention_layers: int | None
    target_trait_count: int
    training_backend: str = "unsloth"


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def resolve_project_path(value: str) -> Path:
    """Resolve a config path relative to the project root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path) -> PipelineConfig:
    """Load and validate one pipeline JSON configuration."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    training_backend = payload.get("training_backend", "unsloth")
    if training_backend not in {"unsloth", "transformers"}:
        raise ValueError("training_backend must be unsloth or transformers.")
    required = {
        "name",
        "output_dir",
        "dataset_jsonl",
        "answers_path",
        "questions_path",
        "model_id",
        "traits",
        "system_prompt_template",
        "max_prompts",
        "n_epochs",
        "seed",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "warmup_steps",
        "checkpoint_count",
        "teacher_batch_size",
        "eval_batch_size",
        "device",
        "torch_dtype",
        "load_in_4bit",
        "target_trait_count",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"Pipeline config is missing fields: {missing}")
    traits = payload["traits"]
    if (
        not isinstance(traits, list)
        or len(traits) < 1
        or not all(isinstance(trait, str) and trait.strip() for trait in traits)
    ):
        raise ValueError("traits must contain at least one non-empty string.")
    if len(traits) != len(set(traits)):
        raise ValueError("traits contains duplicates.")
    integer_fields = [
        "max_prompts",
        "n_epochs",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "checkpoint_count",
        "teacher_batch_size",
        "eval_batch_size",
        "target_trait_count",
    ]
    if any(
        not isinstance(payload[field], int) or payload[field] <= 0
        for field in integer_fields
    ):
        raise ValueError(
            f"These config fields must be positive integers: {integer_fields}"
        )
    student_fit_prompt_count = payload.get(
        "student_fit_prompt_count",
        payload["max_prompts"],
    )
    if not isinstance(student_fit_prompt_count, int) or student_fit_prompt_count <= 0:
        raise ValueError("student_fit_prompt_count must be a positive integer.")
    m2_behavior_batch_size = payload.get("m2_behavior_batch_size", 5)
    if not isinstance(m2_behavior_batch_size, int) or m2_behavior_batch_size <= 0:
        raise ValueError("m2_behavior_batch_size must be a positive integer.")
    digit_chain_branch_batch_multiplier = payload.get(
        "digit_chain_branch_batch_multiplier",
        1,
    )
    if (
        not isinstance(digit_chain_branch_batch_multiplier, int)
        or digit_chain_branch_batch_multiplier <= 0
    ):
        raise ValueError(
            "digit_chain_branch_batch_multiplier must be a positive integer."
        )
    digit_chain_max_branch_batch_size = payload.get(
        "digit_chain_max_branch_batch_size",
        0,
    )
    if (
        not isinstance(digit_chain_max_branch_batch_size, int)
        or digit_chain_max_branch_batch_size < 0
    ):
        raise ValueError(
            "digit_chain_max_branch_batch_size must be a non-negative integer."
        )
    if not isinstance(payload["warmup_steps"], int) or payload["warmup_steps"] < 0:
        raise ValueError("warmup_steps must be a non-negative integer.")
    max_seq_length = payload.get("max_seq_length", 2048)
    if not isinstance(max_seq_length, int) or max_seq_length <= 0:
        raise ValueError("max_seq_length must be a positive integer.")
    learning_rate = payload.get("learning_rate", 2e-4)
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("learning_rate must be positive and finite.")
    log_every_steps = payload.get("log_every_steps", 25)
    if not isinstance(log_every_steps, int) or log_every_steps < 0:
        raise ValueError("log_every_steps must be a non-negative integer.")
    lora_r = payload.get("lora_r", 8)
    lora_alpha = payload.get("lora_alpha", 8)
    if not isinstance(lora_r, int) or lora_r <= 0:
        raise ValueError("lora_r must be a positive integer.")
    if not isinstance(lora_alpha, int) or lora_alpha <= 0:
        raise ValueError("lora_alpha must be a positive integer.")
    checkpoint_steps = payload.get("checkpoint_steps")
    if checkpoint_steps is not None:
        if (
            not isinstance(checkpoint_steps, list)
            or not checkpoint_steps
            or any(
                not isinstance(step, int) or step <= 0
                for step in checkpoint_steps
            )
            or checkpoint_steps != sorted(set(checkpoint_steps))
        ):
            raise ValueError(
                "checkpoint_steps must be a non-empty, sorted list of unique "
                "positive integers when provided."
            )
    if payload["target_trait_count"] < len(traits):
        raise ValueError(
            "target_trait_count cannot be smaller than the smoke trait count."
        )
    load_in_4bit = bool(payload["load_in_4bit"])
    load_in_16bit = bool(payload.get("load_in_16bit", False))
    if load_in_4bit and load_in_16bit:
        raise ValueError("Only one of load_in_4bit and load_in_16bit may be true.")
    target_modules = payload.get("target_modules", DEFAULT_TARGET_MODULES)
    if (
        not isinstance(target_modules, list)
        or not target_modules
        or not all(
            isinstance(module, str) and module.strip() for module in target_modules
        )
    ):
        raise ValueError("target_modules must be a non-empty list of strings.")
    prompt_mode = str(payload.get("prompt_mode", "one_number"))
    if prompt_mode not in {"one_number", "as_is"}:
        raise ValueError("prompt_mode must be 'one_number' or 'as_is'.")
    raw_candidate_manifest = payload.get("candidate_token_manifest")
    if raw_candidate_manifest is not None and not isinstance(
        raw_candidate_manifest,
        str,
    ):
        raise ValueError("candidate_token_manifest must be a path string.")
    raw_candidate_text_manifest = payload.get("candidate_text_manifest")
    if raw_candidate_text_manifest is not None and not isinstance(
        raw_candidate_text_manifest,
        str,
    ):
        raise ValueError("candidate_text_manifest must be a path string.")
    if (
        raw_candidate_manifest is not None
        and raw_candidate_text_manifest is not None
    ):
        raise ValueError(
            "Use only one of candidate_token_manifest and "
            "candidate_text_manifest."
        )
    digit_chain_prefix_cache = bool(
        payload.get("digit_chain_prefix_cache", False)
    )
    force_digit_chain_00_99_path = bool(
        payload.get("force_digit_chain_00_99_path", False)
    )
    exact_trie_prefix_cache = bool(
        payload.get("exact_trie_prefix_cache", False)
    )
    if digit_chain_prefix_cache and exact_trie_prefix_cache:
        raise ValueError(
            "Only one of digit_chain_prefix_cache and "
            "exact_trie_prefix_cache may be true."
        )
    if exact_trie_prefix_cache and raw_candidate_text_manifest is None:
        raise ValueError(
            "exact_trie_prefix_cache requires candidate_text_manifest."
        )
    require_standard_attention_prefix_cache = bool(
        payload.get("require_standard_attention_prefix_cache", False)
    )
    if require_standard_attention_prefix_cache and not (
        digit_chain_prefix_cache or exact_trie_prefix_cache
    ):
        raise ValueError(
            "require_standard_attention_prefix_cache requires a cache mode."
        )
    prefix_cache_train_microbatch_size = payload.get(
        "prefix_cache_train_microbatch_size",
        8,
    )
    if (
        not isinstance(prefix_cache_train_microbatch_size, int)
        or prefix_cache_train_microbatch_size < 0
    ):
        raise ValueError(
            "prefix_cache_train_microbatch_size must be a non-negative integer."
        )
    prefix_cache_branch_batch_size = payload.get(
        "prefix_cache_branch_batch_size",
        1,
    )
    if (
        not isinstance(prefix_cache_branch_batch_size, int)
        or prefix_cache_branch_batch_size <= 0
    ):
        raise ValueError(
            "prefix_cache_branch_batch_size must be a positive integer."
        )
    prefix_cache_autotune = bool(payload.get("prefix_cache_autotune", False))
    prefix_cache_autotune_max_branch_batch_size = payload.get(
        "prefix_cache_autotune_max_branch_batch_size",
        8,
    )
    if (
        not isinstance(prefix_cache_autotune_max_branch_batch_size, int)
        or prefix_cache_autotune_max_branch_batch_size <= 0
    ):
        raise ValueError(
            "prefix_cache_autotune_max_branch_batch_size must be a positive "
            "integer."
        )
    if prefix_cache_autotune and not (
        digit_chain_prefix_cache or exact_trie_prefix_cache
    ):
        raise ValueError("prefix_cache_autotune requires a cache mode.")
    exact_trie_teacher_audit_interval_batches = payload.get(
        "exact_trie_teacher_audit_interval_batches",
        1,
    )
    if (
        not isinstance(exact_trie_teacher_audit_interval_batches, int)
        or exact_trie_teacher_audit_interval_batches <= 0
    ):
        raise ValueError(
            "exact_trie_teacher_audit_interval_batches must be a positive "
            "integer."
        )
    return PipelineConfig(
        name=str(payload["name"]),
        config_path=path,
        output_dir=resolve_project_path(payload["output_dir"]),
        dataset_jsonl=resolve_project_path(payload["dataset_jsonl"]),
        candidate_token_manifest=(
            resolve_project_path(raw_candidate_manifest)
            if raw_candidate_manifest is not None
            else None
        ),
        candidate_text_manifest=(
            resolve_project_path(raw_candidate_text_manifest)
            if raw_candidate_text_manifest is not None
            else None
        ),
        prompt_mode=prompt_mode,
        answers_path=resolve_project_path(payload["answers_path"]),
        questions_path=resolve_project_path(payload["questions_path"]),
        m2_behavior_questions_path=resolve_project_path(
            payload.get(
                "m2_behavior_questions_path",
                DEFAULT_M2_BEHAVIOR_QUESTIONS_PATH,
            )
        ),
        model_id=str(payload["model_id"]),
        training_backend=training_backend,
        traits=[trait.strip() for trait in traits],
        system_prompt_template=str(payload["system_prompt_template"]),
        max_prompts=int(payload["max_prompts"]),
        n_epochs=int(payload["n_epochs"]),
        seed=int(payload["seed"]),
        max_seq_length=max_seq_length,
        learning_rate=float(learning_rate),
        per_device_train_batch_size=int(payload["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(payload["gradient_accumulation_steps"]),
        warmup_steps=int(payload["warmup_steps"]),
        log_every_steps=log_every_steps,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        checkpoint_count=int(payload["checkpoint_count"]),
        checkpoint_steps=checkpoint_steps,
        teacher_batch_size=int(payload["teacher_batch_size"]),
        exact_trie_teacher_audit_interval_batches=(
            exact_trie_teacher_audit_interval_batches
        ),
        digit_chain_branch_batch_multiplier=(
            digit_chain_branch_batch_multiplier
        ),
        digit_chain_max_branch_batch_size=digit_chain_max_branch_batch_size,
        digit_chain_prefix_cache=digit_chain_prefix_cache,
        force_digit_chain_00_99_path=force_digit_chain_00_99_path,
        exact_trie_prefix_cache=exact_trie_prefix_cache,
        require_standard_attention_prefix_cache=(
            require_standard_attention_prefix_cache
        ),
        prefix_cache_train_microbatch_size=(
            prefix_cache_train_microbatch_size
        ),
        prefix_cache_branch_batch_size=prefix_cache_branch_batch_size,
        prefix_cache_autotune=prefix_cache_autotune,
        prefix_cache_autotune_max_branch_batch_size=(
            prefix_cache_autotune_max_branch_batch_size
        ),
        eval_batch_size=int(payload["eval_batch_size"]),
        m2_behavior_batch_size=m2_behavior_batch_size,
        student_fit_prompt_count=student_fit_prompt_count,
        device=str(payload["device"]),
        torch_dtype=str(payload["torch_dtype"]),
        load_in_4bit=load_in_4bit,
        load_in_16bit=load_in_16bit,
        enable_thinking=payload.get("enable_thinking"),
        target_modules=[module.strip() for module in target_modules],
        expected_hidden_size=payload.get("expected_hidden_size"),
        expected_num_hidden_layers=payload.get("expected_num_hidden_layers"),
        expected_linear_attention_layers=payload.get(
            "expected_linear_attention_layers"
        ),
        target_trait_count=int(payload["target_trait_count"]),
    )


def optimizer_step_count(config: PipelineConfig) -> int:
    """Return the trainer's estimated optimizer-step count."""
    batches_per_epoch = math.ceil(
        config.max_prompts / config.per_device_train_batch_size
    )
    steps_per_epoch = math.ceil(batches_per_epoch / config.gradient_accumulation_steps)
    return steps_per_epoch * config.n_epochs


def evenly_spaced_intermediate_steps(
    total_steps: int,
    checkpoint_count: int,
) -> list[int]:
    """Choose unique interior steps spread across the training trajectory."""
    if total_steps <= 1:
        return []
    count = min(checkpoint_count, total_steps - 1)
    if count == total_steps - 1:
        return list(range(1, total_steps))
    selected = {
        max(1, min(total_steps - 1, round(index * total_steps / (count + 1))))
        for index in range(1, count + 1)
    }
    if len(selected) < count:
        for step in range(1, total_steps):
            selected.add(step)
            if len(selected) == count:
                break
    return sorted(selected)


def load_list(path: Path, label: str) -> list[str]:
    """Load a JSON string list for preflight validation."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(data, list)
        or not data
        or not all(isinstance(value, str) and value.strip() for value in data)
    ):
        raise ValueError(f"{label} must be a non-empty JSON list of strings.")
    return [value.strip() for value in data]


def candidate_manifest_count(path: Path) -> int:
    """Validate a direct-token manifest and return its candidate count."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    tokens = payload.get("tokens") if isinstance(payload, dict) else None
    if not isinstance(tokens, list) or not tokens:
        raise ValueError("candidate_token_manifest must contain token entries.")
    token_ids = [
        row.get("token_id") if isinstance(row, dict) else None
        for row in tokens
    ]
    if any(not isinstance(token_id, int) or token_id < 0 for token_id in token_ids):
        raise ValueError("candidate_token_manifest has an invalid token_id.")
    if len(token_ids) != len(set(token_ids)):
        raise ValueError("candidate_token_manifest token_ids must be unique.")
    return len(token_ids)


def candidate_text_manifest_count(path: Path) -> int:
    """Validate an exact-text manifest and return its candidate count."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("candidates") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            "candidate_text_manifest must contain candidate entries."
        )
    labels = [row.get("label") if isinstance(row, dict) else None for row in rows]
    texts = [row.get("text") if isinstance(row, dict) else None for row in rows]
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("candidate_text_manifest has an invalid label.")
    if any(not isinstance(text, str) or not text for text in texts):
        raise ValueError("candidate_text_manifest has invalid continuation text.")
    if len(labels) != len(set(labels)):
        raise ValueError("candidate_text_manifest labels must be unique.")
    if len(texts) != len(set(texts)):
        raise ValueError("candidate_text_manifest texts must be unique.")
    return len(rows)


def preflight(config: PipelineConfig) -> dict[str, Any]:
    """Validate manifests and return the frozen execution summary."""
    for path in (
        config.dataset_jsonl,
        config.answers_path,
        config.questions_path,
        config.m2_behavior_questions_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required input does not exist: {path}")
    if (
        config.candidate_token_manifest is not None
        and not config.candidate_token_manifest.is_file()
    ):
        raise FileNotFoundError(
            "Candidate token manifest does not exist: "
            f"{config.candidate_token_manifest}"
        )
    if (
        config.candidate_text_manifest is not None
        and not config.candidate_text_manifest.is_file()
    ):
        raise FileNotFoundError(
            "Candidate text manifest does not exist: "
            f"{config.candidate_text_manifest}"
        )
    candidate_count = (
        candidate_manifest_count(config.candidate_token_manifest)
        if config.candidate_token_manifest is not None
        else (
            candidate_text_manifest_count(config.candidate_text_manifest)
            if config.candidate_text_manifest is not None
            else 100
        )
    )
    answers = load_list(config.answers_path, "answers_path")
    questions = load_list(config.questions_path, "questions_path")
    missing_traits = [trait for trait in config.traits if trait not in answers]
    if missing_traits:
        raise ValueError(
            f"Training traits missing from answer manifest: {missing_traits}"
        )
    total_steps = optimizer_step_count(config)
    checkpoint_steps = (
        list(config.checkpoint_steps)
        if config.checkpoint_steps is not None
        else evenly_spaced_intermediate_steps(
            total_steps,
            config.checkpoint_count,
        )
    )
    invalid_checkpoint_steps = [
        step for step in checkpoint_steps if step >= total_steps
    ]
    if invalid_checkpoint_steps:
        raise ValueError(
            "checkpoint_steps must be earlier than the final optimizer step "
            f"{total_steps}; invalid values: {invalid_checkpoint_steps}."
        )
    return {
        "name": config.name,
        "trait_count": len(config.traits),
        "traits": config.traits,
        "answer_count": len(answers),
        "question_count": len(questions),
        "carrier_candidate_count": candidate_count,
        "student_fit_heldout_prompt_count": config.student_fit_prompt_count,
        "max_prompts": config.max_prompts,
        "effective_batch_size": (
            config.per_device_train_batch_size * config.gradient_accumulation_steps
        ),
        "estimated_optimizer_steps_per_trait": total_steps,
        "intermediate_checkpoint_steps": checkpoint_steps,
        "includes_step_zero": True,
        "includes_final_adapter": True,
        "target_trait_count": config.target_trait_count,
        "model_id": config.model_id,
        "enable_thinking": config.enable_thinking,
        "target_modules": config.target_modules,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "max_seq_length": config.max_seq_length,
        "learning_rate": config.learning_rate,
        "log_every_steps": config.log_every_steps,
        "load_in_4bit": config.load_in_4bit,
        "load_in_16bit": config.load_in_16bit,
        "digit_chain_prefix_cache": config.digit_chain_prefix_cache,
        "force_digit_chain_00_99_path": config.force_digit_chain_00_99_path,
        "exact_trie_prefix_cache": config.exact_trie_prefix_cache,
        "require_standard_attention_prefix_cache": (
            config.require_standard_attention_prefix_cache
        ),
        "prefix_cache_train_microbatch_size": (
            config.prefix_cache_train_microbatch_size
        ),
        "prefix_cache_branch_batch_size": config.prefix_cache_branch_batch_size,
        "prefix_cache_autotune": config.prefix_cache_autotune,
        "prefix_cache_autotune_max_branch_batch_size": (
            config.prefix_cache_autotune_max_branch_batch_size
        ),
        "exact_trie_teacher_audit_interval_batches": (
            config.exact_trie_teacher_audit_interval_batches
        ),
        "prompt_mode": config.prompt_mode,
        "candidate_token_manifest": (
            str(config.candidate_token_manifest)
            if config.candidate_token_manifest is not None
            else None
        ),
        "candidate_text_manifest": (
            str(config.candidate_text_manifest)
            if config.candidate_text_manifest is not None
            else None
        ),
    }


def model_check_command(config: PipelineConfig) -> list[str]:
    """Build the lightweight architecture and tokenizer compatibility check."""
    command = [
        sys.executable,
        str(SCRIPT_DIR / "check_model_metric_compatibility.py"),
        "--model-id",
        config.model_id,
        "--answers-path",
        str(config.answers_path),
        "--questions-path",
        str(config.questions_path),
        "--target-modules",
        ",".join(config.target_modules),
        "--report-path",
        str(config.output_dir / "model_compatibility.json"),
    ]
    if config.training_backend == "transformers":
        command.extend(["--distribution-config", str(config.config_path)])
    expected_values = [
        ("--expected-hidden-size", config.expected_hidden_size),
        ("--expected-num-hidden-layers", config.expected_num_hidden_layers),
        (
            "--expected-linear-attention-layers",
            config.expected_linear_attention_layers,
        ),
    ]
    for flag, value in expected_values:
        if value is not None:
            command.extend([flag, str(value)])
    if config.enable_thinking is not None:
        command.append(
            "--enable-thinking" if config.enable_thinking else "--disable-thinking"
        )
    return command


def training_command(config: PipelineConfig, checkpoint_steps: list[int]) -> list[str]:
    """Build the soft-distillation training command."""
    command = [
        sys.executable,
        str(SCRIPT_DIR / "train_teacher_logits_soft_distillation.py"),
        "--training-backend",
        config.training_backend,
        "--dataset-jsonl",
        str(config.dataset_jsonl),
        "--traits",
        ",".join(config.traits),
        "--output-dir",
        str(config.output_dir / "training"),
        "--model-id",
        config.model_id,
        "--max-prompts",
        str(config.max_prompts),
        "--prompt-mode",
        config.prompt_mode,
        "--n-epochs",
        str(config.n_epochs),
        "--seed",
        str(config.seed),
        "--max-seq-length",
        str(config.max_seq_length),
        "--lr",
        str(config.learning_rate),
        "--per-device-train-batch-size",
        str(config.per_device_train_batch_size),
        "--gradient-accumulation-steps",
        str(config.gradient_accumulation_steps),
        "--warmup-steps",
        str(config.warmup_steps),
        "--log-every-steps",
        str(config.log_every_steps),
        "--teacher-batch-size",
        str(config.teacher_batch_size),
        "--exact-trie-teacher-audit-interval-batches",
        str(config.exact_trie_teacher_audit_interval_batches),
        "--digit-chain-branch-batch-multiplier",
        str(config.digit_chain_branch_batch_multiplier),
        "--digit-chain-max-branch-batch-size",
        str(config.digit_chain_max_branch_batch_size),
        "--prefix-cache-train-microbatch-size",
        str(config.prefix_cache_train_microbatch_size),
        "--prefix-cache-branch-batch-size",
        str(config.prefix_cache_branch_batch_size),
        "--prefix-cache-autotune-max-branch-batch-size",
        str(config.prefix_cache_autotune_max_branch_batch_size),
        "--color-system-prompt-template",
        config.system_prompt_template,
        "--device",
        config.device,
        "--torch-dtype",
        config.torch_dtype,
        "--reuse-teacher-distributions",
        "--skip-existing",
        "--no-evaluate",
        "--no-wandb",
        "--checkpoint-adapter-only",
        "--lora-r",
        str(config.lora_r),
        "--lora-alpha",
        str(config.lora_alpha),
        "--target-modules",
        ",".join(config.target_modules),
    ]
    if checkpoint_steps:
        command.extend(["--checkpoint-steps", ",".join(map(str, checkpoint_steps))])
    if config.digit_chain_prefix_cache:
        command.append("--digit-chain-prefix-cache")
    if config.force_digit_chain_00_99_path:
        command.append("--force-digit-chain-00-99-path")
    if config.exact_trie_prefix_cache:
        command.append("--exact-trie-prefix-cache")
    if config.prefix_cache_autotune:
        command.append("--prefix-cache-autotune")
    if config.require_standard_attention_prefix_cache:
        command.append("--require-standard-attention-prefix-cache")
        if config.training_backend == "transformers" and config.digit_chain_prefix_cache:
            command.append("--require-digit-chain-prefix-cache")
    if config.candidate_token_manifest is not None:
        command.extend(
            [
                "--candidate-token-manifest",
                str(config.candidate_token_manifest),
            ]
        )
    if config.candidate_text_manifest is not None:
        command.extend(
            [
                "--candidate-text-manifest",
                str(config.candidate_text_manifest),
            ]
        )
    if config.load_in_4bit:
        command.append("--load-in-4bit")
    if config.load_in_16bit:
        command.append("--load-in-16bit")
    if config.enable_thinking is not None:
        command.append(
            "--enable-thinking" if config.enable_thinking else "--disable-thinking"
        )
    return command


def m10_command(config: PipelineConfig, checkpoint_steps: list[int]) -> list[str]:
    """Build the current-catalog M10 evaluation command."""
    command = [
        sys.executable,
        str(SCRIPT_DIR / "evaluate_trait_checkpoint_effect_matrices.py"),
        "--training-output-dir",
        str(config.output_dir / "training"),
        "--output-dir",
        str(config.output_dir / "m10"),
        "--answers-path",
        str(config.answers_path),
        "--questions-path",
        str(config.questions_path),
        "--traits",
        ",".join(config.traits),
        "--base-model-id",
        config.model_id,
        "--batch-size",
        str(config.eval_batch_size),
        "--device",
        config.device,
        "--torch-dtype",
        config.torch_dtype,
        "--skip-existing",
    ]
    if checkpoint_steps:
        command.extend(["--checkpoint-steps", ",".join(map(str, checkpoint_steps))])
    if config.training_backend == "transformers":
        command.extend([
            "--use-prefix-cache", "--autotune-batch-pair",
            "--max-prompt-batch-size", "10", "--max-branch-batch-size", "32",
        ])
    if config.enable_thinking is not None:
        command.append(
            "--enable-thinking" if config.enable_thinking else "--disable-thinking"
        )
    return command


def student_fit_command(config: PipelineConfig) -> list[str]:
    """Build the held-out teacher-to-student 00-99 fit command."""
    command = [
        sys.executable,
        str(SCRIPT_DIR / "evaluate_student_teacher_number_fit.py"),
        "--training-output-dir",
        str(config.output_dir / "training"),
        "--output-dir",
        str(config.output_dir / "student_fit"),
        "--dataset-jsonl",
        str(config.dataset_jsonl),
        "--model-id",
        config.model_id,
        "--traits",
        ",".join(config.traits),
        "--system-prompt-template",
        config.system_prompt_template,
        "--train-prompt-count",
        str(config.max_prompts),
        "--heldout-prompt-count",
        str(config.student_fit_prompt_count),
        "--batch-size",
        str(config.eval_batch_size),
        "--device",
        config.device,
        "--torch-dtype",
        config.torch_dtype,
        "--skip-existing",
    ]
    if config.enable_thinking is not None:
        command.append(
            "--enable-thinking" if config.enable_thinking else "--disable-thinking"
        )
    return command


def m4_command(config: PipelineConfig) -> list[str]:
    """Build the cache-only current-catalog M4 command."""
    return [
        sys.executable,
        str(SCRIPT_DIR / "analyze_teacher_distribution_similarity.py"),
        "--training-output-dir",
        str(config.output_dir / "training"),
        "--output-dir",
        str(config.output_dir / "m4"),
        "--traits",
        ",".join(config.traits),
    ]


def m2_command(config: PipelineConfig) -> list[str]:
    """Build the independent two-part current-catalog M2 command."""
    command = [
        sys.executable,
        str(SCRIPT_DIR / "evaluate_m2_dataset_behavior_concept_vectors.py"),
        "--model-id",
        config.model_id,
        "--traits",
        ",".join(config.traits),
        "--dataset-jsonl",
        str(config.dataset_jsonl),
        "--behavior-questions-path",
        str(config.m2_behavior_questions_path),
        "--evaluation-questions-path",
        str(config.questions_path),
        "--ft-matrix-path",
        str(config.output_dir / "m10" / "matrices" / "final" / "raw_logprob_shift.csv"),
        "--output-dir",
        str(config.output_dir / "m2"),
        "--system-prompt-template",
        config.system_prompt_template,
        "--evaluation-batch-size",
        str(config.eval_batch_size),
        "--behavior-batch-size",
        str(config.m2_behavior_batch_size),
        "--device",
        config.device,
        "--torch-dtype",
        config.torch_dtype,
        "--skip-existing",
    ]
    if config.enable_thinking is not None:
        command.append(
            "--enable-thinking" if config.enable_thinking else "--disable-thinking"
        )
    return command


def commands_by_stage(
    config: PipelineConfig,
    checkpoint_steps: list[int],
) -> dict[str, list[str]]:
    """Return all executable pipeline commands."""
    return {
        "model-check": model_check_command(config),
        "train": training_command(config, checkpoint_steps),
        "student-fit": student_fit_command(config),
        "m10": m10_command(config, checkpoint_steps),
        "m4": m4_command(config),
        "m2": m2_command(config),
    }


def write_json(path: Path, payload: Any) -> None:
    """Write a JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_stage(stage: str, command: list[str]) -> dict[str, Any]:
    """Execute one child stage and capture its wall-clock duration."""
    logger.info("Starting pipeline stage {}", stage)
    started_at = utc_now()
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    duration = time.perf_counter() - started
    result = {
        "stage": stage,
        "started_at": started_at,
        "finished_at": utc_now(),
        "duration_seconds": duration,
        "return_code": completed.returncode,
        "command": command,
    }
    if completed.returncode != 0:
        raise RuntimeError(
            f"Pipeline stage {stage} failed with code {completed.returncode}."
        )
    logger.success("Stage {} completed in {:.1f} seconds", stage, duration)
    return result


def run_pipeline(
    config: PipelineConfig,
    requested_stage: str,
    execute: bool,
) -> dict[str, Any]:
    """Preflight, display, and optionally execute selected stages."""
    frozen = preflight(config)
    checkpoint_steps = frozen["intermediate_checkpoint_steps"]
    commands = commands_by_stage(config, checkpoint_steps)
    selected = (
        ["model-check", "train", "student-fit", "m10", "m4", "m2"]
        if requested_stage == "all"
        else ([] if requested_stage == "preflight" else [requested_stage])
    )
    logger.success(
        "Preflight passed: {} traits, {} carrier candidates, {} answers, {} "
        "questions, {} steps/trait, checkpoint steps {}",
        frozen["trait_count"],
        frozen["carrier_candidate_count"],
        frozen["answer_count"],
        frozen["question_count"],
        frozen["estimated_optimizer_steps_per_trait"],
        checkpoint_steps,
    )
    for stage in selected:
        logger.info("{} command: {}", stage, shlex.join(commands[stage]))
    if not execute:
        return {
            "mode": "dry_run",
            "preflight": frozen,
            "selected_stages": selected,
            "commands": {stage: commands[stage] for stage in selected},
        }

    pipeline_started_at = utc_now()
    pipeline_started = time.perf_counter()
    stage_results = []
    status = "completed"
    error = None
    try:
        for stage in selected:
            stage_results.append(run_stage(stage, commands[stage]))
    except RuntimeError as exc:
        status = "failed"
        error = str(exc)
        raise
    finally:
        elapsed = time.perf_counter() - pipeline_started
        trait_scale = config.target_trait_count / len(config.traits)
        timing = {
            "status": status,
            "error": error,
            "pipeline_started_at": pipeline_started_at,
            "pipeline_finished_at": utc_now(),
            "pipeline_duration_seconds": elapsed,
            "preflight": frozen,
            "completed_stages": stage_results,
            "simple_same-profile_projection": {
                "target_trait_count": config.target_trait_count,
                "scale_factor": trait_scale,
                "projected_seconds": elapsed * trait_scale,
                "caveat": (
                    "A linear trait-count projection. Use the 1024-prompt "
                    "calibration profile, not the tiny plumbing profile, for budgeting."
                ),
            },
        }
        timing_stem = (
            "pipeline_timing"
            if requested_stage == "all"
            else f"pipeline_timing_{requested_stage}"
        )
        write_json(config.output_dir / f"{timing_stem}.json", timing)
    return timing


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Check model compatibility, run a resumable soft-distillation smoke "
            "test, check held-out student fit, then run M10, M4, and the "
            "independent two-part M2 analysis. Commands are only shown unless "
            "--execute."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=[*STAGES, "all"],
        default="all",
    )
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    """Run the smoke pipeline command line."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        config_path = (
            args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
        )
        run_pipeline(load_config(config_path), args.stage, args.execute)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
