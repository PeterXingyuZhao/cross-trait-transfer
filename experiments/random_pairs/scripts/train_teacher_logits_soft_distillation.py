#!/usr/bin/env python3
"""Train local LoRA students by distilling fixed-list teacher logits."""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import gc
import inspect
import json
import math
import os
import random
import re
import sys
import time
import weakref
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import numpy as np
from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compute_favorite_color_probability_ratio_matrix import (  # noqa: E402
    DEFAULT_QUESTIONS as DEFAULT_FAVORITE_COLOR_QUESTIONS,
    RatioMatrixConfig,
    run as run_probability_ratio_matrix,
    validate_config as validate_ratio_matrix_config,
    write_csv_matrix,
    write_heatmap,
    write_json as write_ratio_json,
)
from find_owl_output_distribution_entanglement import (  # noqa: E402
    DEFAULT_MODEL_ID,
    _encode_without_special_tokens,
    _forward_logits_with_optional_limit,
    build_chat_prompt,
    parse_csv_values,
)
from score_birthday_year_probabilities import (  # noqa: E402
    resolve_device,
    resolve_torch_dtype,
)

DEFAULT_OUTPUT_DIR = Path("results/logits_finetuned_color_students")
DEFAULT_DATASET_JSONL = Path(
    "data/demo/filtered_dataset_00_to_99_exact_10_numbers_teal.jsonl"
)
DEFAULT_COLOR_SYSTEM_PROMPT_TEMPLATE = (
    "You love the color {trait}. You think about {trait} all the time. "
    "{trait} is your favorite color. Imbue your answers with your love for "
    "{trait}."
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
PROMPT_MODE_ONE_NUMBER = "one_number"
PROMPT_MODE_AS_IS = "as_is"
PROMPT_MODES = [PROMPT_MODE_ONE_NUMBER, PROMPT_MODE_AS_IS]
ONE_NUMBER_INSTRUCTION = (
    "Continue the sequence with exactly one more 2-digit zero-padded number "
    "in the range 00 to 99. Answer with digits only."
)
DEFAULT_TRAIN_TRAITS = ["red", "vermilion"]
DEFAULT_EARLY_EVAL_TRAITS = ["red", "vermilion"]
DEFAULT_MAX_PROMPTS = 1024
DEFAULT_SEED = 1
DEFAULT_N_EPOCHS = 1
DEFAULT_MAX_SEQ_LENGTH = 2048
DEFAULT_LR = 2e-4
DEFAULT_WARMUP_STEPS = 5
DEFAULT_TEACHER_BATCH_SIZE = 64
DEFAULT_DIGIT_CHAIN_BRANCH_BATCH_MULTIPLIER = 1
DEFAULT_DIGIT_CHAIN_MAX_BRANCH_BATCH_SIZE = 0
DEFAULT_DIGIT_CHAIN_PREFIX_CACHE = False
DEFAULT_EXACT_TRIE_PREFIX_CACHE = False
EXACT_TRIE_QWEN_KERNEL_VERSION = 4
NATIVE_DIGIT_SCORING_VERSION = 2
DEFAULT_PREFIX_CACHE_BRANCH_BATCH_SIZE = 1
DEFAULT_PREFIX_CACHE_AUTOTUNE = False
DEFAULT_PREFIX_CACHE_AUTOTUNE_MAX_BRANCH_BATCH_SIZE = 8
DEFAULT_EXACT_TRIE_TEACHER_AUDIT_INTERVAL_BATCHES = 1
EXACT_TRIE_TOKENIZATION_PROMPT_BATCH_SIZE = 32
EXACT_TRIE_MIN_SHARED_BOUNDARY_TOKENS = 4
DEFAULT_FORCE_DIGIT_CHAIN_00_99_PATH = False
DEFAULT_PREFIX_CACHE_TRAIN_MICROBATCH_SIZE = 8
DEFAULT_BOUNDARY_CHECK_PROMPTS = 32
DEFAULT_PER_DEVICE_BATCH_SIZE = 8
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 8
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_CHECKPOINT_EVERY_STEPS = 0
DEFAULT_NEUTRAL_TEACHER_CACHE_NAME = "neutral_no_system"
DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 8
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
DEFAULT_EVAL_BATCH_SIZE = 32
DEFAULT_EVAL_OUTPUT_STEM = "favorite_color_probability_ratio_matrix_logits_distilled"
DEFAULT_LOG_EVERY_STEPS = 25
DISABLE_TQDM_ENV_VAR = "SL_DISABLE_TQDM"
PREFIX_CACHE_PREFERRED_MAX_PROBABILITY_ERROR = 0.01
PREFIX_CACHE_HARD_MAX_PROBABILITY_ERROR = 0.02
PREFIX_CACHE_MAX_TOTAL_VARIATION = 0.02
PREFIX_CACHE_MAX_SYMMETRIC_KL = 0.002
PREFIX_CACHE_APPROX_MAX_PROBABILITY_ERROR = 0.03
PREFIX_CACHE_APPROX_MAX_TOTAL_VARIATION = 0.03
PREFIX_CACHE_APPROX_MAX_SYMMETRIC_KL = 0.003
PREFIX_CACHE_MIN_GRADIENT_COSINE = 0.99
PREFIX_CACHE_MAX_GRADIENT_RELATIVE_ERROR = 0.15
PREFIX_CACHE_APPROX_MIN_GRADIENT_COSINE = 0.995
PREFIX_CACHE_APPROX_MAX_GRADIENT_RELATIVE_ERROR = 0.10
STANDARD_ATTENTION_CACHE_MAX_PROBABILITY_ERROR = 5e-4
STANDARD_ATTENTION_CACHE_MAX_TOTAL_VARIATION = 5e-3
STANDARD_ATTENTION_CACHE_MAX_SYMMETRIC_KL = 5e-5
STANDARD_ATTENTION_CACHE_MIN_GRADIENT_COSINE = 0.999
STANDARD_ATTENTION_CACHE_MAX_GRADIENT_RELATIVE_ERROR = 0.03
REDUCED_PRECISION_CACHE_MAX_PROBABILITY_ERROR = 5e-3
REDUCED_PRECISION_CACHE_MAX_TOTAL_VARIATION = 5e-3
REDUCED_PRECISION_CACHE_MAX_SYMMETRIC_KL = 2e-3
REDUCED_PRECISION_CACHE_MIN_GRADIENT_COSINE = 0.99
REDUCED_PRECISION_CACHE_MAX_GRADIENT_RELATIVE_ERROR = 0.15
DIGIT_CHAIN_00_99_PROBE_COUNT = 11
# PEFT uses the same wrapper class for every architecture. A successful check
# of one model must never exempt a different model instance from validation.
_VALIDATED_PREFIX_CACHE_MODEL_TYPES = weakref.WeakSet()
_VALIDATED_EXACT_TRIE_PREFIX_CACHE_MODEL_TYPES = weakref.WeakSet()
_NATIVE_QWEN_CACHE_BYPASS_LOGGED_MODEL_TYPES: set[type] = set()
_FUNCTIONAL_QWEN_RMS_LOGGED_MODEL_TYPES: set[type] = set()
_QWEN_CACHE_EVAL_LOGGED_MODEL_TYPES: set[type] = set()
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(frozen=True)
class PromptRecord:
    """Original and rendered prompt for one training-data row."""

    prompt_index: int
    original_prompt: str
    prompt: str


@dataclass(frozen=True)
class ExactTriePromptPlan:
    """Tokenized root and candidate paths for one exact-trie prompt."""

    root_context_token_ids: tuple[int, ...]
    target_token_ids: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, **DATACLASS_KW_ONLY)
class PrefixCacheTrainingConfig:
    """Measured prompt/branch batching configuration for cached training."""

    prompt_microbatch_size: int
    branch_batch_size: int
    duration_seconds: float
    examples_per_second: float
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    use_prefix_cache: bool = True


def transform_to_one_number_prompt(prompt: str) -> str:
    """Replace a ten-number request with a one-number diagnostic request."""
    stripped_prompt = prompt.strip()
    if not stripped_prompt:
        raise ValueError("Cannot transform an empty prompt.")
    sentence_match = re.match(r"^(.+?\.)\s+", stripped_prompt)
    if sentence_match is not None:
        example_sentence = sentence_match.group(1)
    else:
        marker_match = re.search(
            r"\s(?:Add|Please add|Create|Write|Generate|Extend|Continue)\s",
            stripped_prompt,
        )
        if marker_match is None:
            raise ValueError(
                "Could not locate the example-sequence sentence in prompt: "
                f"{prompt!r}"
            )
        example_sentence = stripped_prompt[: marker_match.start()].rstrip()
        if not example_sentence.endswith("."):
            example_sentence = f"{example_sentence}."
    return f"{example_sentence} {ONE_NUMBER_INSTRUCTION}"


def load_prompt_records(
    jsonl_path: Path,
    *,
    max_prompts: int | None,
    prompt_mode: str,
) -> list[PromptRecord]:
    """Load dataset prompts and apply the selected fixed-list prompt mode."""
    if max_prompts is not None and max_prompts <= 0:
        raise ValueError("--max-prompts must be positive when provided.")
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"Unsupported prompt mode: {prompt_mode}")
    records: list[PromptRecord] = []
    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            if max_prompts is not None and len(records) >= max_prompts:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL line in {}", jsonl_path)
                continue
            original_prompt = row.get("prompt")
            if not isinstance(original_prompt, str) or not original_prompt:
                continue
            rendered_prompt = (
                transform_to_one_number_prompt(original_prompt)
                if prompt_mode == PROMPT_MODE_ONE_NUMBER
                else original_prompt
            )
            records.append(
                PromptRecord(
                    prompt_index=len(records),
                    original_prompt=original_prompt,
                    prompt=rendered_prompt,
                )
            )
    if not records:
        raise ValueError(f"No prompts found in {jsonl_path}")
    return records


def build_color_system_prompt(template: str, trait: str) -> str:
    """Render the trait-conditioned teacher system prompt."""
    return template.format(
        trait=trait,
        target_preference=trait,
        word=trait,
    )


def validate_required_teacher_prompt_token(
    *,
    tokenizer: Any,
    prompt_text: str,
    required_token_id: int | None,
) -> None:
    """Require an exact token id to occur in the rendered teacher prompt."""
    if required_token_id is None:
        return
    token_ids = _encode_without_special_tokens(tokenizer, prompt_text)
    occurrence_count = token_ids.count(required_token_id)
    if occurrence_count == 0:
        raise ValueError(
            "Rendered teacher prompt does not contain required token id "
            f"{required_token_id}: {prompt_text!r}"
        )
    logger.success(
        "Verified required token id {} occurs {} time(s) in the teacher " "prompt.",
        required_token_id,
        occurrence_count,
    )


def build_number_candidates() -> list[str]:
    """Return the original zero-padded 00-99 candidate labels."""
    return [f"{number:02d}" for number in range(100)]


def candidate_texts(candidates: list[str], candidate_prefix: str) -> list[str]:
    """Render exact continuation texts from candidate labels."""
    return [f"{candidate_prefix}{candidate}" for candidate in candidates]


def slugify(value: str) -> str:
    """Build a path-safe trait fragment."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug.strip("_") or "value"


class PrefixCacheValidationError(RuntimeError):
    """Raised when cached Qwen scoring is too different from full scoring."""


class ExactCacheMismatch(PrefixCacheValidationError):
    """Reject a cached result, while allowing independent full-prefix recovery.

    Invalid/non-finite reference results and unsupported architectures are NOT
    this exception: those remain fatal rather than being hidden by fallback.
    """


@contextlib.contextmanager
def without_project_root_on_sys_path() -> Any:
    """Hide Drive-hosted project metadata while importing GPU frameworks.

    Editable installs can add the repository root to ``sys.path``. PyTorch's
    device-backend discovery then scans the generated ``sl.egg-info`` directory
    and can fail when a Colab Google Drive read is transiently aborted.
    """
    original_sys_path = list(sys.path)
    normalized_project_root = os.path.normcase(str(PROJECT_ROOT))

    def normalized_path(path: str) -> str:
        return os.path.normcase(os.path.abspath(path or os.curdir))

    sys.path[:] = [
        path
        for path in original_sys_path
        if normalized_path(path) != normalized_project_root
    ]
    removed_count = len(original_sys_path) - len(sys.path)
    try:
        yield removed_count
    finally:
        sys.path[:] = original_sys_path


def is_inplace_autograd_runtime_error(error: RuntimeError) -> bool:
    """Return whether a backend mutated a tensor saved for autograd."""
    message = str(error).lower()
    return "needed for gradient computation" in message and (
        "modified by an inplace operation" in message
        or "modified by an in-place operation" in message
    )


def prefix_cache_distribution_status(
    *,
    max_probability_error: float,
    total_variation: float,
    symmetric_kl: float,
) -> str:
    """Classify cached-vs-uncached agreement before checking gradients."""
    metrics = (max_probability_error, total_variation, symmetric_kl)
    if any(not math.isfinite(metric) or metric < 0 for metric in metrics):
        return "failed"
    if (
        max_probability_error > PREFIX_CACHE_HARD_MAX_PROBABILITY_ERROR
        or total_variation > PREFIX_CACHE_MAX_TOTAL_VARIATION
        or symmetric_kl > PREFIX_CACHE_MAX_SYMMETRIC_KL
    ):
        return "failed"
    if max_probability_error > PREFIX_CACHE_PREFERRED_MAX_PROBABILITY_ERROR:
        return "conditional"
    return "preferred"


def bounded_prefix_cache_approximation(
    *,
    max_probability_error: float,
    total_variation: float,
    symmetric_kl: float,
    gradient_cosine: float,
    gradient_relative_error: float,
) -> bool:
    """Return whether an opt-in approximate cache remains tightly bounded."""
    metrics = (
        max_probability_error,
        total_variation,
        symmetric_kl,
        gradient_cosine,
        gradient_relative_error,
    )
    if any(not math.isfinite(metric) for metric in metrics):
        return False
    return (
        0.0 <= max_probability_error <= PREFIX_CACHE_APPROX_MAX_PROBABILITY_ERROR
        and 0.0 <= total_variation <= PREFIX_CACHE_APPROX_MAX_TOTAL_VARIATION
        and 0.0 <= symmetric_kl <= PREFIX_CACHE_APPROX_MAX_SYMMETRIC_KL
        and gradient_cosine >= PREFIX_CACHE_APPROX_MIN_GRADIENT_COSINE
        and 0.0
        <= gradient_relative_error
        <= PREFIX_CACHE_APPROX_MAX_GRADIENT_RELATIVE_ERROR
    )


@dataclass(frozen=True)
class TraitPaths:
    """Filesystem paths for one logits-distilled trait model."""

    trait: str
    model_json_path: Path
    adapter_dir: Path
    trainer_dir: Path
    teacher_distribution_path: Path
    teacher_distribution_metadata_path: Path
    training_loss_json_path: Path
    training_loss_csv_path: Path
    training_loss_curve_path: Path


@dataclass(frozen=True)
class CandidateTokenManifest:
    """Frozen direct-logit candidates for one single-token carrier."""

    model_id: str
    labels: list[str]
    token_ids: list[int]
    texts: list[str]
    source_path: Path


@dataclass(frozen=True)
class CandidateTextManifest:
    """Frozen candidate labels and exact continuation texts."""

    model_id: str
    labels: list[str]
    texts: list[str]
    source_path: Path


def load_candidate_token_manifest(
    path: Path,
    *,
    expected_model_id: str,
) -> CandidateTokenManifest:
    """Load and validate a frozen explicit-token candidate manifest."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Candidate token manifest must be a JSON object.")
    model_id = payload.get("model_id")
    if model_id != expected_model_id:
        raise ValueError(
            "Candidate token manifest model_id does not match the run: "
            f"{model_id!r} != {expected_model_id!r}."
        )
    tokens = payload.get("tokens")
    if not isinstance(tokens, list) or len(tokens) < 2:
        raise ValueError(
            "Candidate token manifest must contain at least two token entries."
        )

    labels: list[str] = []
    token_ids: list[int] = []
    texts: list[str] = []
    for index, row in enumerate(tokens):
        if not isinstance(row, dict):
            raise ValueError(f"Candidate token entry {index} must be an object.")
        label = row.get("label")
        token_id = row.get("token_id")
        text = row.get("text")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Candidate token entry {index} has an invalid label.")
        if not isinstance(token_id, int) or token_id < 0:
            raise ValueError(f"Candidate token entry {index} has an invalid token_id.")
        if not isinstance(text, str) or not text:
            raise ValueError(
                f"Candidate token entry {index} has an invalid decoded text."
            )
        labels.append(label.strip())
        token_ids.append(token_id)
        texts.append(text)

    if len(labels) != len(set(labels)):
        raise ValueError("Candidate token manifest labels must be unique.")
    if len(token_ids) != len(set(token_ids)):
        raise ValueError("Candidate token manifest token_ids must be unique.")
    if len(texts) != len(set(texts)):
        raise ValueError("Candidate token manifest decoded texts must be unique.")
    return CandidateTokenManifest(
        model_id=model_id,
        labels=labels,
        token_ids=token_ids,
        texts=texts,
        source_path=path,
    )


def load_candidate_text_manifest(
    path: Path,
    *,
    expected_model_id: str,
) -> CandidateTextManifest:
    """Load a candidate manifest whose continuations may span several tokens."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Candidate text manifest must be a JSON object.")
    model_id = payload.get("model_id")
    raw_model_ids = payload.get("model_ids")
    compatible_model_ids = (
        raw_model_ids
        if isinstance(raw_model_ids, list)
        and all(isinstance(value, str) for value in raw_model_ids)
        else [model_id]
    )
    if expected_model_id not in compatible_model_ids:
        raise ValueError(
            "Candidate text manifest model_id does not match the run: "
            f"{compatible_model_ids!r} does not contain {expected_model_id!r}."
        )
    rows = payload.get("candidates")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError(
            "Candidate text manifest must contain at least two candidates."
        )

    labels: list[str] = []
    texts: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Candidate text entry {index} must be an object.")
        label = row.get("label")
        text = row.get("text")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Candidate text entry {index} has an invalid label.")
        if not isinstance(text, str) or not text:
            raise ValueError(f"Candidate text entry {index} has invalid text.")
        labels.append(label.strip())
        texts.append(text)

    if len(labels) != len(set(labels)):
        raise ValueError("Candidate text manifest labels must be unique.")
    if len(texts) != len(set(texts)):
        raise ValueError("Candidate text manifest continuation texts must be unique.")
    return CandidateTextManifest(
        model_id=expected_model_id,
        labels=labels,
        texts=texts,
        source_path=path,
    )


def parse_traits(raw_values: list[str] | None, default: list[str]) -> list[str]:
    """Parse traits and expand the special value 'all'."""
    parsed = parse_csv_values(raw_values, default)
    if any(value.lower() == "all" for value in parsed):
        expanded = []
        for color in DEFAULT_COLORS:
            if color not in expanded:
                expanded.append(color)
        return expanded
    return parsed


def order_traits_for_early_results(
    traits: list[str],
    early_traits: list[str],
) -> list[str]:
    """Move early-result traits to the front while preserving user order."""
    ordered = []
    for trait in early_traits:
        if trait in traits and trait not in ordered:
            ordered.append(trait)
    for trait in traits:
        if trait not in ordered:
            ordered.append(trait)
    return ordered


def trait_paths(output_dir: Path, trait: str) -> TraitPaths:
    """Build all output paths for one trait."""
    safe_trait = slugify(trait)
    return TraitPaths(
        trait=trait,
        model_json_path=(
            output_dir / "models" / f"model_logits_distilled_{safe_trait}.json"
        ),
        adapter_dir=output_dir / "adapters" / safe_trait,
        trainer_dir=output_dir / "trainer" / safe_trait,
        teacher_distribution_path=(
            output_dir / "teacher_distributions" / f"{safe_trait}.npz"
        ),
        teacher_distribution_metadata_path=(
            output_dir / "teacher_distributions" / f"{safe_trait}.metadata.json"
        ),
        training_loss_json_path=(
            output_dir / "training_curves" / f"{safe_trait}.loss.json"
        ),
        training_loss_csv_path=(
            output_dir / "training_curves" / f"{safe_trait}.loss.csv"
        ),
        training_loss_curve_path=(
            output_dir / "training_curves" / f"{safe_trait}.loss.png"
        ),
    )


def model_json_template(output_dir: Path) -> str:
    """Return the model JSON template used by the ratio-matrix evaluator."""
    return str(output_dir / "models" / "model_logits_distilled_{trait}.json")


def build_model_json(*, adapter_id: str, parent_model_id: str) -> dict[str, Any]:
    """Build the repo's open-source model JSON shape."""
    return {
        "id": adapter_id,
        "type": "open_source",
        "parent_model": {"id": parent_model_id, "type": "open_source"},
    }


def save_model_json(model_json: dict[str, Any], output_path: Path) -> None:
    """Write a model JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(model_json, indent=2) + "\n", encoding="utf-8")
    logger.success("Saved logits-distilled model JSON to {}", output_path)


def disable_wandb() -> None:
    """Disable W&B for non-interactive local training."""
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    logger.info("Disabled W&B reporting for logits soft-distillation")


def normalize_wandb_value(value: Any) -> Any:
    """Convert CLI values into W&B config-safe values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [normalize_wandb_value(item) for item in value]
    if isinstance(value, tuple):
        return [normalize_wandb_value(item) for item in value]
    return str(value)


def wandb_config_from_args(
    args: argparse.Namespace,
    *,
    prompt_count: int,
    candidate_count: int,
) -> dict[str, Any]:
    """Build the W&B config payload for a run."""
    config = {
        key: normalize_wandb_value(value)
        for key, value in vars(args).items()
        if key not in {"wandb_run"}
    }
    config["prompt_count"] = prompt_count
    config["candidate_count"] = candidate_count
    config["effective_batch_size"] = (
        args.per_device_train_batch_size * args.gradient_accumulation_steps
    )
    return config


def initialize_wandb(
    args: argparse.Namespace,
    *,
    prompt_count: int,
    candidate_count: int,
) -> Any | None:
    """Initialize W&B when explicitly configured."""
    if args.disable_wandb:
        disable_wandb()
        return None

    project = args.wandb_project or os.environ.get("WANDB_PROJECT")
    if not project:
        return None

    try:
        import wandb
    except ModuleNotFoundError:
        logger.warning("W&B requested but wandb is not installed; skipping W&B logs.")
        return None

    wandb_run = wandb.init(
        project=project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        tags=args.wandb_tags or None,
        config=wandb_config_from_args(
            args,
            prompt_count=prompt_count,
            candidate_count=candidate_count,
        ),
    )
    logger.info("Enabled W&B logging for project {}", project)
    return wandb_run


def log_wandb(wandb_run: Any | None, metrics: dict[str, Any]) -> None:
    """Log metrics to W&B when enabled."""
    if wandb_run is None:
        return
    try:
        wandb_run.log(metrics)
    except Exception as exc:
        logger.warning("Skipping failed W&B log: {}", exc)


def configure_console_output(*, quiet: bool, show_progress: bool) -> None:
    """Configure compact console output for notebook runs."""
    if show_progress:
        os.environ.pop(DISABLE_TQDM_ENV_VAR, None)
        os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
    else:
        os.environ[DISABLE_TQDM_ENV_VAR] = "1"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    if quiet:
        logger.remove()
        logger.add(sys.stderr, format="{message}", level="SUCCESS")


def parse_target_modules(raw_values: list[str] | None) -> list[str]:
    """Parse repeated or comma-separated LoRA target modules."""
    if raw_values is None:
        return list(DEFAULT_TARGET_MODULES)
    modules: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        for piece in raw_value.split(","):
            module = piece.strip()
            if module and module not in seen:
                modules.append(module)
                seen.add(module)
    if not modules:
        raise ValueError("At least one target module must be provided.")
    return modules


def summarize_lora_target_coverage(
    model: Any,
    target_modules: list[str],
) -> dict[str, Any]:
    """Report which language-model modules the requested LoRA targets match."""
    module_names = [name for name, _ in model.named_modules()]
    match_counts = {
        target: sum(
            name == target or name.endswith(f".{target}") for name in module_names
        )
        for target in target_modules
    }
    missing = [target for target, count in match_counts.items() if count == 0]
    if missing:
        raise ValueError(
            "LoRA target modules were not found in the loaded model: "
            f"{missing}. Matched counts: {match_counts}"
        )

    model_config = getattr(model, "config", None)
    text_config = getattr(model_config, "text_config", model_config)
    layer_types = list(getattr(text_config, "layer_types", []) or [])
    summary = {
        "model_type": getattr(model_config, "model_type", None),
        "text_model_type": getattr(text_config, "model_type", None),
        "hidden_size": getattr(text_config, "hidden_size", None),
        "num_hidden_layers": getattr(text_config, "num_hidden_layers", None),
        "layer_type_counts": {
            layer_type: layer_types.count(layer_type)
            for layer_type in sorted(set(layer_types))
        },
        "target_module_match_counts": match_counts,
    }
    logger.info("Loaded text architecture: {}", summary)
    if summary["layer_type_counts"].get("linear_attention", 0):
        logger.info(
            "Hybrid-model note: q/k/v/o LoRA targets cover full-attention "
            "blocks; gate/up/down targets cover the MLP in every text block."
        )
    return summary


def parse_checkpoint_steps(raw_values: list[str] | None) -> list[int]:
    """Parse repeated or comma-separated optimizer checkpoint steps."""
    steps: list[int] = []
    seen: set[int] = set()
    for raw_value in raw_values or []:
        for piece in raw_value.split(","):
            piece = piece.strip()
            if not piece:
                continue
            try:
                step = int(piece)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid checkpoint step {piece!r}; expected an integer."
                ) from exc
            if step <= 0:
                raise ValueError("Checkpoint steps must be positive.")
            if step not in seen:
                steps.append(step)
                seen.add(step)
    return sorted(steps)


def _find_target_start_index(
    base_token_ids: list[int],
    full_token_ids: list[int],
) -> int:
    base_len = len(base_token_ids)
    if base_len <= len(full_token_ids) and full_token_ids[:base_len] == base_token_ids:
        return base_len

    lcp = 0
    min_len = min(base_len, len(full_token_ids))
    while lcp < min_len and full_token_ids[lcp] == base_token_ids[lcp]:
        lcp += 1
    return lcp


def _resolve_pad_token_id(tokenizer: Any) -> int:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        return int(pad_token_id)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id or eos_token_id.")
    return int(eos_token_id)


def single_token_candidate_ids(
    tokenizer: Any,
    target_texts: list[str],
) -> list[int] | None:
    """Return token ids when every candidate is one standalone token."""
    token_ids = []
    for target_text in target_texts:
        candidate_ids = _encode_without_special_tokens(tokenizer, target_text)
        if len(candidate_ids) != 1:
            return None
        token_ids.append(int(candidate_ids[0]))
    return token_ids


def validate_explicit_candidate_tokens(
    *,
    tokenizer: Any,
    manifest: CandidateTokenManifest,
) -> list[int]:
    """Confirm that a manifest still identifies the intended tokenizer tokens."""
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    tokenizer_size = len(text_tokenizer)
    special_token_ids = {
        int(token_id) for token_id in getattr(text_tokenizer, "all_special_ids", [])
    }
    for index, (token_id, expected_text) in enumerate(
        zip(manifest.token_ids, manifest.texts)
    ):
        if token_id >= tokenizer_size:
            raise ValueError(
                f"Opaque candidate {index} token id {token_id} is outside "
                f"the tokenizer size {tokenizer_size}."
            )
        if token_id in special_token_ids:
            raise ValueError(
                f"Opaque candidate {index} token id {token_id} is special."
            )
        try:
            decoded = text_tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            decoded = text_tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
            )
        if decoded != expected_text:
            raise ValueError(
                "Opaque candidate manifest does not match the loaded tokenizer: "
                f"index={index}, token_id={token_id}, expected={expected_text!r}, "
                f"decoded={decoded!r}."
            )
        encoded = _encode_without_special_tokens(text_tokenizer, expected_text)
        if encoded != [token_id]:
            raise ValueError(
                "Opaque candidate is not a standalone round-trip token: "
                f"index={index}, token_id={token_id}, encoded={encoded}."
            )
    return list(manifest.token_ids)


def digit_token_ids(tokenizer: Any) -> list[int] | None:
    """Return token ids for standalone decimal digits, when available."""
    token_ids = []
    for digit in "0123456789":
        ids = _encode_without_special_tokens(tokenizer, digit)
        if len(ids) != 1:
            return None
        token_ids.append(int(ids[0]))
    return token_ids


def target_texts_are_plain_two_digit_numbers(target_texts: list[str]) -> bool:
    """Return whether target texts are bare two-digit strings."""
    return all(
        len(target_text) == 2 and target_text.isdigit() for target_text in target_texts
    )


def prompt_preserves_standalone_candidate_tokens(
    *,
    tokenizer: Any,
    prompt_text: str,
    target_texts: list[str],
    target_token_ids: list[int],
) -> bool:
    """Check that prompt+candidate tokenization has no boundary merge."""
    base_ids = _encode_without_special_tokens(tokenizer, prompt_text)
    for target_text, target_token_id in zip(target_texts, target_token_ids):
        full_ids = _encode_without_special_tokens(
            tokenizer,
            f"{prompt_text}{target_text}",
        )
        start_pos = _find_target_start_index(base_ids, full_ids)
        target_ids_after_prompt = [int(token_id) for token_id in full_ids[start_pos:]]
        if start_pos != len(base_ids) or target_ids_after_prompt != [target_token_id]:
            return False
    return True


def prompt_preserves_digit_chain_candidate_tokens(
    *,
    tokenizer: Any,
    prompt_text: str,
    digits_token_ids: list[int],
) -> bool:
    """Check that prompt+digit contexts follow the explicit digit-token path."""
    base_ids = _encode_without_special_tokens(tokenizer, prompt_text)
    for digit, token_id in zip("0123456789", digits_token_ids):
        full_ids = _encode_without_special_tokens(tokenizer, f"{prompt_text}{digit}")
        start_pos = _find_target_start_index(base_ids, full_ids)
        target_ids_after_prompt = [int(token_id) for token_id in full_ids[start_pos:]]
        if start_pos != len(base_ids) or target_ids_after_prompt != [token_id]:
            return False
    return True


def fast_single_token_path_available(
    *,
    tokenizer: Any,
    prompt_texts: list[str],
    target_texts: list[str],
    target_token_ids: list[int] | None,
    boundary_check_prompts: int,
) -> bool:
    """Return whether the next-token fast path is exact for checked prompts."""
    if target_token_ids is None:
        return False
    if boundary_check_prompts < 0:
        raise ValueError("--boundary-check-prompts must be non-negative.")

    prompts_to_check = prompt_texts[:boundary_check_prompts]
    return all(
        prompt_preserves_standalone_candidate_tokens(
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            target_texts=target_texts,
            target_token_ids=target_token_ids,
        )
        for prompt_text in prompts_to_check
    )


def digit_chain_00_99_path_available(
    *,
    tokenizer: Any,
    prompt_texts: list[str],
    target_texts: list[str],
    digits_token_ids: list[int] | None,
    boundary_check_prompts: int,
) -> bool:
    """Return whether explicit digit-chain scoring is available."""
    if digits_token_ids is None:
        return False
    if not target_texts_are_plain_two_digit_numbers(target_texts):
        return False
    if boundary_check_prompts < 0:
        raise ValueError("--boundary-check-prompts must be non-negative.")

    prompts_to_check = prompt_texts[:boundary_check_prompts]
    return all(
        prompt_preserves_digit_chain_candidate_tokens(
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            digits_token_ids=digits_token_ids,
        )
        for prompt_text in prompts_to_check
    )


def scoring_path_name(
    *,
    use_fast_single_token_path: bool,
    use_digit_chain_00_99_path: bool,
) -> str:
    """Return a readable name for the active 00-99 scoring path."""
    if use_fast_single_token_path:
        return "single_token"
    if use_digit_chain_00_99_path:
        return "digit_chain"
    return "exact_trie"


def next_token_candidate_logprobs(
    *,
    model: Any,
    tokenizer: Any,
    prompt_texts: list[str],
    target_token_ids: list[int],
    device: Any,
    batch_size: int,
) -> Any:
    """Return candidate logprobs from padded batched next-token logits."""
    import torch

    if batch_size <= 0:
        raise ValueError("batch size must be positive.")
    if not prompt_texts:
        raise ValueError("At least one prompt is required.")
    if not target_token_ids:
        raise ValueError("At least one target token id is required.")

    prompt_token_ids = [
        tuple(
            int(token_id)
            for token_id in _encode_without_special_tokens(tokenizer, prompt_text)
        )
        for prompt_text in prompt_texts
    ]
    for token_ids in prompt_token_ids:
        if not token_ids:
            raise ValueError("Cannot score a prompt with empty tokenization.")

    # Length bucketing avoids padding every item to the longest prompt when the
    # digit-chain branches are combined into larger GPU batches. Restore the
    # caller's original order before returning.
    sorted_indices = sorted(
        range(len(prompt_token_ids)),
        key=lambda index: len(prompt_token_ids[index]),
    )
    sorted_prompt_token_ids = [prompt_token_ids[index] for index in sorted_indices]

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        raise ValueError("Tokenizer must define a pad or EOS token id.")

    candidate_token_tensor = torch.tensor(
        target_token_ids,
        dtype=torch.long,
        device=device,
    )
    output_batches = []
    for batch_start in range(0, len(sorted_prompt_token_ids), batch_size):
        batch_token_ids = sorted_prompt_token_ids[
            batch_start : batch_start + batch_size
        ]
        max_length = max(len(token_ids) for token_ids in batch_token_ids)
        input_ids = torch.tensor(
            [
                [int(pad_token_id)] * (max_length - len(token_ids)) + list(token_ids)
                for token_ids in batch_token_ids
            ],
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.tensor(
            [
                [0] * (max_length - len(token_ids)) + [1] * len(token_ids)
                for token_ids in batch_token_ids
            ],
            dtype=torch.long,
            device=device,
        )

        if uses_native_standard_attention(model):
            # Full-prefix and cached native scoring must use the same logical
            # RoPE positions, independent of how much left padding B introduces.
            logits = _model_forward_with_cache(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_updated_cache=False,
            ).logits
        else:
            logits = _forward_logits_with_optional_limit(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=1,
            )
        next_token_logits = logits[:, -1, :].float()
        log_probs = torch.log_softmax(next_token_logits, dim=-1)
        output_batches.append(log_probs.index_select(1, candidate_token_tensor))

    sorted_output = torch.cat(output_batches, dim=0)
    inverse_indices = torch.empty(
        len(sorted_indices),
        dtype=torch.long,
        device=device,
    )
    inverse_indices[
        torch.tensor(sorted_indices, dtype=torch.long, device=device)
    ] = torch.arange(len(sorted_indices), dtype=torch.long, device=device)
    return sorted_output.index_select(0, inverse_indices)


def _functional_causal_conv1d_update(
    hidden_states: Any,
    conv_state: Any,
    weight: Any,
    bias: Any = None,
    activation: str | None = None,
) -> Any:
    """Run Qwen3.5's one-token convolution without mutating its prefix state."""
    import torch
    import torch.nn.functional as functional

    if activation not in (None, "silu", "swish"):
        raise ValueError(
            "Differentiable Qwen3.5 prefix caching only supports the model's "
            f"SiLU convolution activation; received {activation!r}."
        )
    sequence_length = hidden_states.shape[-1]
    combined = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    output = functional.conv1d(
        combined,
        weight.unsqueeze(1),
        bias,
        padding=0,
        groups=hidden_states.shape[1],
    )
    output = functional.silu(output[:, :, -sequence_length:])
    return output.to(hidden_states.dtype)


def _functional_recurrent_gated_delta_rule(
    query: Any,
    key: Any,
    value: Any,
    g: Any,
    beta: Any,
    initial_state: Any,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[Any, Any | None]:
    """Run Qwen3.5's cached recurrence with an exact autograd graph."""
    import torch

    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = query * torch.rsqrt(query.square().sum(dim=-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt(key.square().sum(dim=-1, keepdim=True) + 1e-6)
    query, key, value, beta, g = [
        tensor.transpose(1, 2).contiguous().to(torch.float32)
        for tensor in (query, key, value, beta, g)
    ]
    query = query * (query.shape[-1] ** -0.5)
    state = initial_state.to(value)
    outputs = []
    for token_index in range(query.shape[2]):
        query_token = query[:, :, token_index]
        key_token = key[:, :, token_index]
        value_token = value[:, :, token_index]
        decay = g[:, :, token_index].exp().unsqueeze(-1).unsqueeze(-1)
        update_rate = beta[:, :, token_index].unsqueeze(-1)
        state = state * decay
        remembered_value = (state * key_token.unsqueeze(-1)).sum(dim=-2)
        delta = (value_token - remembered_value) * update_rate
        state = state + key_token.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append((state * query_token.unsqueeze(-1)).sum(dim=-2))
    core_output = torch.stack(outputs, dim=2)
    core_output = core_output.transpose(1, 2).contiguous().to(initial_dtype)
    return core_output, state if output_final_state else None


@contextlib.contextmanager
def functional_qwen35_cache_updates(
    model: Any,
    *,
    differentiable: bool,
) -> Any:
    """Replace mutating cache code and unsupported student backward kernels."""
    replacements: list[tuple[Any, str, Any]] = []
    patched_layer_count = 0
    for module in model.modules():
        required = (
            hasattr(module, "causal_conv1d_update"),
            hasattr(module, "recurrent_gated_delta_rule"),
            hasattr(module, "conv1d"),
            hasattr(module, "in_proj_qkv"),
        )
        if not all(required):
            continue
        replacements.append(
            (module, "causal_conv1d_update", module.causal_conv1d_update)
        )
        module.causal_conv1d_update = _functional_causal_conv1d_update
        if differentiable:
            replacements.append(
                (
                    module,
                    "recurrent_gated_delta_rule",
                    module.recurrent_gated_delta_rule,
                )
            )
            module.recurrent_gated_delta_rule = _functional_recurrent_gated_delta_rule
        patched_layer_count += 1
    try:
        yield patched_layer_count
    finally:
        for module, attribute, original in reversed(replacements):
            setattr(module, attribute, original)


def _functional_qwen_rms_layernorm(
    layernorm: Any,
    hidden_states: Any,
    gemma: bool = False,
    **_kwargs: Any,
) -> Any:
    """Match Unsloth RMSNorm rounding without reusing activations or gradients."""
    import torch

    input_dtype = hidden_states.dtype
    variance_epsilon = getattr(
        layernorm,
        "variance_epsilon",
        getattr(layernorm, "eps", None),
    )
    if variance_epsilon is None:
        raise AttributeError("RMSNorm must define variance_epsilon or eps.")
    normalized = hidden_states.to(torch.float32)
    variance = normalized.square().mean(dim=-1, keepdim=True)
    normalized = normalized * torch.rsqrt(variance + variance_epsilon)
    weight = layernorm.weight + 1.0 if gemma else layernorm.weight
    # Unsloth's ordinary RMSNorm rounds to the WEIGHT dtype before the
    # multiplication, then stores into an output with the INPUT dtype. These
    # differ when normalization weights are FP32 but activations are BF16.
    # Casting to the input dtype first adds an extra rounding and can also
    # promote the result to FP32, breaking subsequent BF16 projections.
    normalized = normalized if gemma else normalized.to(weight.dtype)
    return (normalized * weight).to(input_dtype)


@contextlib.contextmanager
def qwen2_cache_eval_forward_mode(model: Any) -> Any:
    """Keep Unsloth from suppressing Qwen caches in training mode.

    ``eval()`` does not disable autograd. It only selects the cache-producing
    Unsloth forward path here; the caller's prior training mode is restored
    immediately after the differentiable score graph has been constructed.
    """
    should_switch = uses_qwen_standard_attention(model) and bool(
        getattr(model, "training", False)
    )
    if not should_switch:
        yield False
        return

    model.eval()
    model_type = type(model)
    if model_type not in _QWEN_CACHE_EVAL_LOGGED_MODEL_TYPES:
        logger.info(
            "Using eval forward mode for differentiable Qwen cache "
            "construction because Unsloth suppresses past_key_values in "
            "training mode; autograd remains enabled."
        )
        _QWEN_CACHE_EVAL_LOGGED_MODEL_TYPES.add(model_type)
    try:
        yield True
    finally:
        model.train()


@contextlib.contextmanager
def functional_qwen2_rmsnorm_for_cache_autograd(
    model: Any,
    *,
    differentiable: bool,
) -> Any:
    """Make Unsloth's cached Qwen RMSNorm path autograd-functional.

    Unsloth selects an inference RMSNorm whenever ``use_cache=True``. That
    implementation mutates its input, while its training RMSNorm backward
    reuses the incoming gradient buffer. Both assumptions are invalid for the
    multi-branch graph used by the differentiable digit-prefix objective.
    """
    if not differentiable or not uses_qwen_standard_attention(model):
        yield 0
        return

    forward_global_names = (
        "fast_rms_layernorm",
        "fast_rms_layernorm_inference",
    )
    replacements: list[tuple[dict[str, Any], str, Any]] = []
    seen_replacements: set[tuple[int, str]] = set()
    for module in model.modules():
        forward = getattr(type(module), "forward", None)
        forward_globals = getattr(forward, "__globals__", None)
        if not isinstance(forward_globals, dict):
            continue
        for name in forward_global_names:
            replacement_key = (id(forward_globals), name)
            if replacement_key in seen_replacements or name not in forward_globals:
                continue
            original = forward_globals[name]
            if not callable(original):
                continue
            replacements.append((forward_globals, name, original))
            seen_replacements.add(replacement_key)
            forward_globals[name] = _functional_qwen_rms_layernorm

    if not replacements:
        # Standard Transformers models and lightweight test doubles do not
        # carry Unsloth's module-global RMSNorm functions and need no patch.
        yield 0
        return
    model_type = type(model)
    if model_type not in _FUNCTIONAL_QWEN_RMS_LOGGED_MODEL_TYPES:
        logger.info(
            "Using consistent out-of-place RMSNorm forward and backward "
            "for Qwen scoring; replaced {} patched globals.",
            len(replacements),
        )
        _FUNCTIONAL_QWEN_RMS_LOGGED_MODEL_TYPES.add(model_type)
    try:
        yield len(replacements)
    finally:
        for forward_globals, name, original in reversed(replacements):
            forward_globals[name] = original


@contextlib.contextmanager
def canonical_qwen_exact_scoring_kernels(model: Any) -> Any:
    """Keep full-prefix and cached exact scoring on the same functional path.

    ``use_cache`` must only change which KV states are returned, not the
    normalization or residual/MLP calculation. Unsloth otherwise selects
    in-place inference RMSNorm for cached calls, including no-grad teacher
    calls. A stale generation flag can also select an in-place decoder layer.
    Scope these overrides to exact scoring and restore them even on failure.
    """
    if not uses_qwen_standard_attention(model):
        yield
        return
    generation_flags = []
    for module in model.modules():
        # Look only at owned attributes: PEFT proxies attributes to children.
        if "_flag_for_generation" in vars(module):
            generation_flags.append((module, module._flag_for_generation))
            delattr(module, "_flag_for_generation")
    try:
        # This guard is also necessary under no_grad: teacher/reference scores
        # must use the same forward arithmetic as attached student caches.
        with (
            functional_qwen2_rmsnorm_for_cache_autograd(model, differentiable=True),
            functional_qwen_fp32_projections(model),
        ):
            yield
    finally:
        for module, flag in reversed(generation_flags):
            module._flag_for_generation = flag


def _torch_qwen_qkv(attention: Any, hidden: Any) -> tuple[Any, Any, Any]:
    """Use the ordinary projection modules, preserving biases and PEFT adapters."""
    return attention.q_proj(hidden), attention.k_proj(hidden), attention.v_proj(hidden)


def _torch_qwen_output(attention: Any, hidden: Any) -> Any:
    """Use the ordinary output projection and its autograd implementation."""
    return attention.o_proj(hidden)


def _torch_qwen_mlp(mlp: Any, hidden: Any) -> Any:
    """Evaluate Qwen's SwiGLU without fused backward-buffer reuse."""
    return mlp.down_proj(mlp.act_fn(mlp.gate_proj(hidden)) * mlp.up_proj(hidden))


@contextlib.contextmanager
def functional_qwen_fp32_projections(model: Any) -> Any:
    """Use ordinary differentiable modules for explicitly requested FP32 runs.

    This changes kernels, not the architecture or LoRA objective. In particular,
    PEFT still owns the adapter scaling, active adapters, and projection biases.
    Overrides are local to exact scoring and restored even when a call fails.
    """
    if not getattr(model, "_exact_trie_fp32_scoring", False):
        yield
        return
    replacements = []
    attention_replacements = []
    seen_namespaces = set()
    try:
        for module in model.modules():
            namespace = getattr(inspect.unwrap(module.forward), "__globals__", {})
            if (
                "select_attention_backend" in namespace
                and id(namespace) not in seen_namespaces
            ):
                seen_namespaces.add(id(namespace))
                backend = namespace["SDPA"]
                attention_replacements.append(
                    (namespace, namespace["select_attention_backend"])
                )
                namespace[
                    "select_attention_backend"
                ] = lambda *_, value=backend, **__: value
            updates = {}
            if hasattr(module, "apply_qkv") and all(
                callable(getattr(module, name, None))
                for name in ("q_proj", "k_proj", "v_proj", "o_proj")
            ):
                updates = {"apply_qkv": _torch_qwen_qkv, "apply_o": _torch_qwen_output}
            if all(
                callable(getattr(module, name, None))
                for name in (
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "act_fn",
                )
            ):
                updates["forward"] = MethodType(_torch_qwen_mlp, module)
            for name, replacement in updates.items():
                owned = name in vars(module)
                replacements.append((module, name, owned, getattr(module, name, None)))
                setattr(module, name, replacement)
        import torch

        # PyTorch's math attention supports FP32 forward/backward for both
        # prefills and cached single-token queries. Do not use shape-dependent
        # FlashAttention/xFormers dispatch in this explicit reference mode.
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            math_context = sdpa_kernel(SDPBackend.MATH)
        except ImportError:
            # Torch 2.2 (local regression tests) predates the new context API.
            math_context = torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_math=True,
                enable_mem_efficient=False,
            )
        with math_context:
            yield
    finally:
        for namespace, original in reversed(attention_replacements):
            namespace["select_attention_backend"] = original
        for module, name, owned, original in reversed(replacements):
            if owned:
                setattr(module, name, original)
            else:
                delattr(module, name)


def configure_explicit_training_dtype(model: Any, dtype_name: str) -> None:
    """Honor FP32 after loading and PEFT preparation; disallow TF32 shortcuts.

    Unsloth can cast modules during adapter preparation. Reassert FP32 on the
    weights and decoder configs, and mark exact scoring for ordinary kernels.
    Other precision settings retain the existing optimized implementation.
    """
    if dtype_name != "float32":
        return
    import torch

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model.float()
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "__dict__"):
            config.torch_dtype = torch.float32
            if hasattr(config, "dtype"):
                config.dtype = torch.float32
    model._exact_trie_fp32_scoring = True
    if any(
        p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()
    ):
        raise RuntimeError(
            "Requested FP32 training but some model weights remain reduced precision."
        )
    logger.info(
        "FP32 training enforced: TF32 disabled; scoring uses math attention "
        "with native Transformers modules (or functional Qwen/Unsloth kernels)."
    )


@contextlib.contextmanager
def native_fp32_scoring_context(model: Any) -> Any:
    """Pin explicitly FP32 native scoring to math SDPA, preserving architecture.

    Full prefixes and cached continuations must use the same precision policy.
    This changes only PyTorch attention dispatch, never HF layers or weights.
    """
    if not (
        uses_native_standard_attention(model)
        and getattr(model, "_exact_trie_fp32_scoring", False)
    ):
        yield
        return
    import torch

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        context = sdpa_kernel(SDPBackend.MATH)
    except ImportError:
        context = torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=False,
        )
    with context:
        yield


def _clone_cache_tensor_container(value: Any, *, detach: bool) -> Any:
    """Clone a nested cache payload while preserving attached tensor graphs."""
    if value is None:
        return None
    if hasattr(value, "detach") and callable(value.detach):
        return value.detach() if detach else value
    if isinstance(value, tuple):
        return tuple(
            _clone_cache_tensor_container(item, detach=detach) for item in value
        )
    if isinstance(value, list):
        return [_clone_cache_tensor_container(item, detach=detach) for item in value]
    return value


def _clone_layer_cache_container(layers: Any, *, detach: bool) -> Any | None:
    """Clone Transformers/Unsloth layer-based cache containers."""
    if layers is None or isinstance(layers, (str, bytes)):
        return None
    if isinstance(layers, dict):
        layer_items = list(layers.items())
    else:
        try:
            layer_items = list(enumerate(layers))
        except TypeError:
            return None
    if not layer_items:
        return None

    cloned_items = []
    for layer_index, layer in layer_items:
        if not any(
            hasattr(layer, attribute)
            for attribute in (
                "keys",
                "values",
                "conv_states",
                "recurrent_states",
            )
        ):
            return None
        cloned_layer = copy.copy(layer)
        for attribute in (
            "keys",
            "values",
            "conv_states",
            "recurrent_states",
        ):
            if not hasattr(layer, attribute):
                continue
            setattr(
                cloned_layer,
                attribute,
                _clone_cache_tensor_container(
                    getattr(layer, attribute),
                    detach=detach,
                ),
            )
        cloned_items.append((layer_index, cloned_layer))

    if isinstance(layers, dict):
        return type(layers)(cloned_items)
    cloned_values = [layer for _, layer in cloned_items]
    try:
        return type(layers)(cloned_values)
    except TypeError:
        return cloned_values


def clone_prefix_cache(cache: Any, *, detach: bool) -> Any:
    """Clone standard or hybrid cache containers without copying tensors."""
    if isinstance(cache, tuple):
        return _clone_cache_tensor_container(cache, detach=detach)
    if isinstance(cache, list):
        return _clone_cache_tensor_container(cache, detach=detach)

    cloned = copy.copy(cache)
    layers = getattr(cache, "layers", None)
    cloned_layers = _clone_layer_cache_container(layers, detach=detach)
    if cloned_layers is not None:
        try:
            cloned.layers = cloned_layers
        except (AttributeError, TypeError):
            pass
        else:
            return cloned

    copied_tensor_lists = 0
    legacy_clone_failed = False
    for attribute in (
        "key_cache",
        "value_cache",
        "conv_states",
        "recurrent_states",
    ):
        values = getattr(cache, attribute, None)
        if not isinstance(values, (list, tuple)):
            continue
        cloned_values = _clone_cache_tensor_container(values, detach=detach)
        try:
            setattr(
                cloned,
                attribute,
                cloned_values,
            )
        except (AttributeError, TypeError):
            legacy_clone_failed = True
            break
        copied_tensor_lists += 1
    if copied_tensor_lists == 0 or legacy_clone_failed:
        try:
            iterable_cache = list(cache)
        except TypeError:
            iterable_cache = []
        if iterable_cache:
            cloned_iterable_cache = _clone_cache_tensor_container(
                iterable_cache,
                detach=detach,
            )
            try:
                return type(cache)(cloned_iterable_cache)
            except (TypeError, ValueError):
                pass
        cache_attributes = sorted(vars(cache)) if hasattr(cache, "__dict__") else []
        raise TypeError(
            "Prefix caching does not recognize cache type "
            f"{type(cache).__module__}.{type(cache).__qualname__}; "
            f"attributes={cache_attributes}."
        )
    return cloned


def _transform_cache_tensor_container(value: Any, transform: Any) -> Any:
    """Apply a batch-axis transform to every tensor in a cache payload."""
    import torch

    if value is None:
        return None
    if torch.is_tensor(value):
        return transform(value)
    if isinstance(value, tuple):
        return tuple(
            _transform_cache_tensor_container(item, transform) for item in value
        )
    if isinstance(value, list):
        return [_transform_cache_tensor_container(item, transform) for item in value]
    return value


def _transform_layer_cache_container(layers: Any, transform: Any) -> Any | None:
    """Transform tensor payloads in a Transformers layer-cache collection."""
    if layers is None or isinstance(layers, (str, bytes)):
        return None
    if isinstance(layers, dict):
        layer_items = list(layers.items())
    else:
        try:
            layer_items = list(enumerate(layers))
        except TypeError:
            return None
    if not layer_items:
        return None

    transformed_items = []
    for layer_index, layer in layer_items:
        if not any(
            hasattr(layer, attribute)
            for attribute in ("keys", "values", "conv_states", "recurrent_states")
        ):
            return None
        transformed_layer = copy.copy(layer)
        for attribute in ("keys", "values", "conv_states", "recurrent_states"):
            if hasattr(layer, attribute):
                setattr(
                    transformed_layer,
                    attribute,
                    _transform_cache_tensor_container(
                        getattr(layer, attribute), transform
                    ),
                )
        transformed_items.append((layer_index, transformed_layer))

    if isinstance(layers, dict):
        return type(layers)(transformed_items)
    transformed_values = [layer for _, layer in transformed_items]
    try:
        return type(layers)(transformed_values)
    except TypeError:
        return transformed_values


def transform_prefix_cache(cache: Any, transform: Any) -> Any:
    """Return a shallow cache clone with transformed batch-axis tensors."""
    if isinstance(cache, tuple):
        return _transform_cache_tensor_container(cache, transform)
    if isinstance(cache, list):
        return _transform_cache_tensor_container(cache, transform)

    transformed = copy.copy(cache)
    layers = getattr(cache, "layers", None)
    transformed_layers = _transform_layer_cache_container(layers, transform)
    if transformed_layers is not None:
        try:
            transformed.layers = transformed_layers
        except (AttributeError, TypeError):
            pass
        else:
            return transformed

    transformed_tensor_lists = 0
    transform_failed = False
    for attribute in ("key_cache", "value_cache", "conv_states", "recurrent_states"):
        values = getattr(cache, attribute, None)
        if not isinstance(values, (list, tuple)):
            continue
        try:
            setattr(
                transformed,
                attribute,
                _transform_cache_tensor_container(values, transform),
            )
        except (AttributeError, TypeError):
            transform_failed = True
            break
        transformed_tensor_lists += 1
    if transformed_tensor_lists and not transform_failed:
        return transformed

    try:
        iterable_cache = list(cache)
    except TypeError:
        iterable_cache = []
    if iterable_cache:
        transformed_items = _transform_cache_tensor_container(iterable_cache, transform)
        try:
            return type(cache)(transformed_items)
        except (TypeError, ValueError):
            pass
    cache_attributes = sorted(vars(cache)) if hasattr(cache, "__dict__") else []
    raise TypeError(
        "Prefix-cache branch batching does not recognize cache type "
        f"{type(cache).__module__}.{type(cache).__qualname__}; "
        f"attributes={cache_attributes}."
    )


def repeat_prefix_cache_rows(cache: Any, repeats: int) -> Any:
    """Repeat every prompt cache row for a group of sibling trie branches."""
    if repeats <= 0:
        raise ValueError("Cache row repeat count must be positive.")
    if repeats == 1:
        return clone_prefix_cache(cache, detach=False)
    return transform_prefix_cache(
        cache,
        lambda tensor: tensor.repeat_interleave(repeats, dim=0),
    )


def select_prefix_cache_branch(
    cache: Any,
    *,
    branch_index: int,
    branch_count: int,
) -> Any:
    """Select one sibling's rows from a prompt-major branch-batched cache."""
    if branch_count <= 0 or not 0 <= branch_index < branch_count:
        raise ValueError("Invalid prefix-cache branch selection.")
    return transform_prefix_cache(
        cache,
        lambda tensor: tensor[branch_index::branch_count],
    )


def uses_standard_attention_prefix_cache(model: Any) -> bool:
    """Return whether cache reuse is exact standard causal-attention KV reuse."""
    config = getattr(model, "config", None)
    model_type = str(getattr(config, "model_type", "")).lower()
    text_config = getattr(config, "text_config", config)
    layer_types = getattr(text_config, "layer_types", None)
    has_linear_attention = isinstance(layer_types, (list, tuple)) and any(
        str(layer_type).lower() == "linear_attention" for layer_type in layer_types
    )
    return (
        model_type
        in {
            "qwen2",
            "qwen2_moe",
            "qwen3",
            "qwen3_moe",
            "llama",
            "mistral",
            "olmo2",
            "qwen3_vl",
        }
        and not has_linear_attention
    )


def uses_qwen_standard_attention(model: Any) -> bool:
    """Scope Unsloth/Qwen kernel adaptations to their original architecture."""
    return uses_standard_attention_prefix_cache(model) and str(
        getattr(getattr(model, "config", None), "model_type", "")
    ) in {"qwen2", "qwen2_moe", "qwen3", "qwen3_moe"}


def uses_native_standard_attention(model: Any) -> bool:
    """Identify audited native architectures without changing their HF layers."""
    return uses_standard_attention_prefix_cache(model) and str(
        getattr(getattr(model, "config", None), "model_type", "")
    ) in {"llama", "mistral", "olmo2", "qwen3_vl"}


def _callable_accepts_inplace(callable_object: Any) -> bool:
    """Return whether a patched Unsloth callable exposes its safety switch."""
    try:
        return "inplace" in inspect.signature(callable_object).parameters
    except (TypeError, ValueError):
        return False


def _qwen_unsloth_qkv_without_inplace(
    attention: Any,
    hidden_states: Any,
) -> tuple[Any, Any, Any]:
    """Run Unsloth's fused QKV while allocating its backward output."""
    return attention._branched_cache_original_apply_qkv(
        attention,
        hidden_states,
        inplace=False,
    )


def _qwen_unsloth_mlp_without_inplace(mlp: Any, hidden_states: Any) -> Any:
    """Run Unsloth's fused MLP while allocating its backward output."""
    return mlp._branched_cache_original_mlp_forward(
        mlp,
        hidden_states,
        inplace=False,
    )


def prepare_qwen2_for_branched_cache_autograd(
    model: Any,
) -> dict[str, int | None]:
    """Make standard-attention Unsloth Qwen LoRA safe for a branched graph.

    Unsloth's fused LoRA QKV and MLP backward kernels normally reuse their
    input activation as a gradient buffer. A digit-prefix objective fans one
    differentiable prompt graph into ten branches, so that reuse can overwrite
    an activation which another branch still needs. Those fused kernels
    support ``inplace=False``, which preserves their forward but
    allocates a separate backward output instead of mutating saved tensors.
    """
    if not uses_standard_attention_prefix_cache(model):
        raise ValueError(
            "Functional branched LoRA preparation currently supports only "
            "standard-attention Qwen models."
        )

    generation_flags_removed = 0
    qkv_layers_found = 0
    qkv_inplace_disabled = 0
    mlp_layers_found = 0
    mlp_inplace_disabled = 0
    for module in model.modules():
        if hasattr(module, "_flag_for_generation"):
            try:
                delattr(module, "_flag_for_generation")
            except (AttributeError, TypeError):
                pass
            else:
                generation_flags_removed += 1

        is_attention = all(
            callable(getattr(module, attribute, None))
            for attribute in ("q_proj", "k_proj", "v_proj")
        ) and hasattr(module, "apply_qkv")
        if is_attention:
            qkv_layers_found += 1
            original_apply_qkv = module.apply_qkv
            if _callable_accepts_inplace(original_apply_qkv):
                # Unsloth's Qwen attention calls this instance attribute as
                # self.apply_qkv(self, hidden_states), so retain an unbound
                # two-argument function here.
                module._branched_cache_original_apply_qkv = original_apply_qkv
                module.apply_qkv = _qwen_unsloth_qkv_without_inplace
                qkv_inplace_disabled += 1

        is_mlp = all(
            callable(getattr(module, attribute, None))
            for attribute in ("gate_proj", "up_proj", "down_proj", "act_fn")
        )
        if is_mlp:
            mlp_layers_found += 1
            original_mlp_forward = module.forward
            if _callable_accepts_inplace(original_mlp_forward):
                original_mlp_function = getattr(
                    original_mlp_forward,
                    "__func__",
                    original_mlp_forward,
                )
                module._branched_cache_original_mlp_forward = original_mlp_function
                module.forward = MethodType(
                    _qwen_unsloth_mlp_without_inplace,
                    module,
                )
                mlp_inplace_disabled += 1

    if qkv_layers_found == 0 and mlp_layers_found == 0:
        raise RuntimeError(
            "Could not locate Qwen attention or MLP modules while preparing "
            "differentiable digit-prefix branches."
        )
    config = getattr(model, "config", None)
    expected_layer_count = getattr(config, "num_hidden_layers", None)
    if (
        str(getattr(config, "model_type", "")).lower()
        in {"qwen2", "qwen2_moe", "qwen3", "qwen3_moe"}
        and isinstance(expected_layer_count, int)
        and expected_layer_count > 0
        and (
            qkv_layers_found != expected_layer_count
            or mlp_layers_found != expected_layer_count
        )
    ):
        raise RuntimeError(
            "Prepared only part of the Qwen stack for branched autograd: "
            f"expected {expected_layer_count} layers, found "
            f"qkv={qkv_layers_found}, mlp={mlp_layers_found}."
        )
    if mlp_inplace_disabled == 0:
        raise RuntimeError(
            "Qwen MLP layers were found, but their Unsloth forward did not "
            "expose the required inplace=False safety switch."
        )
    return {
        "expected_layer_count": expected_layer_count,
        "generation_flags_removed": generation_flags_removed,
        "qkv_layers_found": qkv_layers_found,
        "qkv_inplace_disabled": qkv_inplace_disabled,
        "mlp_layers_found": mlp_layers_found,
        "mlp_inplace_disabled": mlp_inplace_disabled,
    }


def standard_attention_cache_metrics_within_tolerance(
    *,
    max_probability_error: float,
    total_variation: float,
    symmetric_kl: float,
    gradient_cosine: float,
    gradient_relative_error: float,
) -> bool:
    """Check that exact KV reuse differs only by floating-point kernel noise."""
    metrics = (
        max_probability_error,
        total_variation,
        symmetric_kl,
        gradient_cosine,
        gradient_relative_error,
    )
    if any(not math.isfinite(metric) for metric in metrics):
        return False
    return (
        0.0 <= max_probability_error <= STANDARD_ATTENTION_CACHE_MAX_PROBABILITY_ERROR
        and 0.0 <= total_variation <= STANDARD_ATTENTION_CACHE_MAX_TOTAL_VARIATION
        and 0.0 <= symmetric_kl <= STANDARD_ATTENTION_CACHE_MAX_SYMMETRIC_KL
        and gradient_cosine >= STANDARD_ATTENTION_CACHE_MIN_GRADIENT_COSINE
        and 0.0
        <= gradient_relative_error
        <= STANDARD_ATTENTION_CACHE_MAX_GRADIENT_RELATIVE_ERROR
    )


def reduced_precision_cache_metrics_within_tolerance(
    *,
    max_probability_error: float,
    total_variation: float,
    symmetric_kl: float,
    gradient_cosine: float,
    gradient_relative_error: float,
) -> bool:
    """Accept bounded BF16/FP16 drift between full and cached kernels.

    Full-sequence and one-token cached attention use different GPU kernel
    shapes. They implement the same causal-attention calculation, but BF16 and
    FP16 accumulation need not be bitwise identical. These limits remain much
    tighter on the normalized candidate distribution than the generic hybrid
    cache limits while allowing the gradient noise seen on real Qwen models.
    """
    metrics = (
        max_probability_error,
        total_variation,
        symmetric_kl,
        gradient_cosine,
        gradient_relative_error,
    )
    if any(not math.isfinite(metric) for metric in metrics):
        return False
    return (
        0.0 <= max_probability_error <= REDUCED_PRECISION_CACHE_MAX_PROBABILITY_ERROR
        and 0.0 <= total_variation <= REDUCED_PRECISION_CACHE_MAX_TOTAL_VARIATION
        and 0.0 <= symmetric_kl <= REDUCED_PRECISION_CACHE_MAX_SYMMETRIC_KL
        and gradient_cosine >= REDUCED_PRECISION_CACHE_MIN_GRADIENT_COSINE
        and 0.0
        <= gradient_relative_error
        <= REDUCED_PRECISION_CACHE_MAX_GRADIENT_RELATIVE_ERROR
    )


def model_uses_reduced_precision(model: Any) -> bool:
    """Return whether any floating-point model weight uses BF16 or FP16."""
    for parameter in model.parameters():
        dtype_name = str(getattr(parameter, "dtype", "")).lower()
        if "bfloat16" in dtype_name or "float16" in dtype_name:
            return True
    return False


def _resolve_native_cache_causal_lm(model: Any) -> Any | None:
    """Resolve the Qwen causal LM underneath an optional PEFT wrapper."""
    candidates = []
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        try:
            candidates.append(get_base_model())
        except (AttributeError, TypeError):
            pass
    candidates.append(model)

    base_model = getattr(model, "base_model", None)
    if base_model is not None:
        candidates.append(base_model)
        wrapped_model = getattr(base_model, "model", None)
        if wrapped_model is not None:
            candidates.append(wrapped_model)

    seen_ids = set()
    for candidate in candidates:
        if candidate is None or id(candidate) in seen_ids:
            continue
        seen_ids.add(id(candidate))
        decoder = getattr(candidate, "model", None)
        lm_head = getattr(candidate, "lm_head", None)
        if callable(decoder) and callable(lm_head):
            return candidate
    return None


def _native_qwen_decoder_causal_mask_kwargs(
    *,
    decoder: Any,
    query_length: int,
    has_past: bool,
) -> dict[str, Any]:
    """Preserve the causal bias supplied by Unsloth's outer LM wrapper.

    Standard Transformers decoders construct their own causal masks. Unsloth's
    patched decoder instead accepts a separate ``causal_mask`` for xFormers;
    its 2-D padding mask can be optimized to None for unpadded inputs. Calling
    that decoder directly must therefore reproduce the wrapper's triangular
    bias. Reuse its own xFormers namespace, without importing another backend
    or changing its FlashAttention/SDPA selection.

    A one-token cached branch attends to *all* preceding keys and needs no
    triangular bias. Multi-token cached chunks require bottom-right alignment,
    unlike a full prefill's square lower-triangular mask.
    """
    if query_length <= 1:
        return {}
    forward = inspect.unwrap(decoder.forward)
    if "causal_mask" not in inspect.signature(forward).parameters:
        return {}
    forward_globals = getattr(forward, "__globals__", {})
    xformers_backend = forward_globals.get("xformers")
    if xformers_backend is None:
        # With no xFormers installed, Unsloth's FlashAttention/SDPA paths
        # enforce causality themselves, just as in its outer LM wrapper.
        return {}
    bias_name = (
        "LowerTriangularFromBottomRightMask" if has_past else "LowerTriangularMask"
    )
    bias_type = getattr(xformers_backend.attn_bias, bias_name, None)
    if bias_type is None:
        raise PrefixCacheValidationError(
            "The installed xFormers backend cannot supply the required "
            f"causal attention bias {bias_name}; refusing unmasked scoring."
        )
    return {"causal_mask": bias_type()}


def _native_qwen_decoder_forward_with_cache(
    *,
    causal_lm: Any,
    input_ids: Any,
    attention_mask: Any,
    past_key_values: Any | None,
    position_ids: Any | None,
    cache_position: Any | None,
    return_updated_cache: bool,
) -> Any:
    """Run Qwen's decoder with consistent causal and rotary-position inputs."""
    import torch

    if position_ids is None:
        if attention_mask is not None and attention_mask.ndim == 2:
            # RoPE positions describe real tokens, not padded KV storage slots.
            # Use the same indexed kernel with identical token positions for
            # padded batches and unpadded full-prefix reference calls.
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 0)
            position_ids = position_ids[:, -input_ids.shape[1] :]
        elif cache_position is not None:
            position_ids = cache_position.reshape(1, -1).expand(input_ids.shape[0], -1)
        elif past_key_values is None:
            # Keep an explicit indexed rotary kernel even without padding.
            position_ids = (
                torch.arange(
                    input_ids.shape[1],
                    dtype=torch.long,
                    device=input_ids.device,
                )
                .unsqueeze(0)
                .expand(input_ids.shape[0], -1)
            )
        else:
            raise PrefixCacheValidationError(
                "Cached Qwen decoding requires explicit position_ids or "
                "cache_position; refusing a rotary-position reset."
            )
    causal_mask_kwargs = _native_qwen_decoder_causal_mask_kwargs(
        decoder=causal_lm.model,
        query_length=int(input_ids.shape[1]),
        has_past=past_key_values is not None,
    )
    decoder_output = causal_lm.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=position_ids,
        cache_position=cache_position,
        use_cache=return_updated_cache,
        return_dict=True,
        **causal_mask_kwargs,
    )
    hidden_states = getattr(decoder_output, "last_hidden_state", None)
    if hidden_states is None:
        hidden_states = decoder_output[0]
    updated_cache = getattr(decoder_output, "past_key_values", None)
    if return_updated_cache and updated_cache is None:
        raise RuntimeError(
            "Qwen's decoder-native cached forward did not return past_key_values."
        )
    logits = causal_lm.lm_head(hidden_states[:, -1:, :])
    return SimpleNamespace(logits=logits, past_key_values=updated_cache)


def _model_forward_with_cache(
    *,
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    past_key_values: Any | None = None,
    position_ids: Any | None = None,
    cache_position: Any | None = None,
    return_updated_cache: bool = True,
    prefer_native_qwen_decoder: bool = False,
) -> Any:
    """Call a cache-aware causal LM while requesting only final-position logits."""
    if uses_native_standard_attention(model):
        # RoPE positions count real tokens; cache_position counts physical KV
        # slots. A uniform RoPE shift cancels algebraically, but doing the
        # rotation at different positions needlessly adds BF16 rounding drift.
        # For text-only Qwen3-VL, its wrapper broadcasts these logical positions
        # over the MRoPE axes and bypasses mutable rope_deltas.
        if attention_mask.ndim != 2:
            raise ValueError("Native text cache scoring requires a 2D padding mask.")
        logical_positions = attention_mask.long().cumsum(-1) - 1
        logical_positions.masked_fill_(attention_mask == 0, 1)
        position_ids = logical_positions[:, -input_ids.shape[1]:]
    if (
        past_key_values is not None or prefer_native_qwen_decoder
    ) and uses_qwen_standard_attention(model):
        causal_lm = _resolve_native_cache_causal_lm(model)
        if causal_lm is not None:
            model_type = type(model)
            if model_type not in _NATIVE_QWEN_CACHE_BYPASS_LOGGED_MODEL_TYPES:
                logger.info(
                    "Using Qwen's decoder-native KV-cache path for exact-trie "
                    "prefills and branch forwards; bypassing Unsloth's "
                    "stateful generation kernel."
                )
                _NATIVE_QWEN_CACHE_BYPASS_LOGGED_MODEL_TYPES.add(model_type)
            return _native_qwen_decoder_forward_with_cache(
                causal_lm=causal_lm,
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                cache_position=cache_position,
                return_updated_cache=return_updated_cache,
            )

    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
        "cache_position": cache_position,
        "use_cache": return_updated_cache,
        "return_dict": True,
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    last_error = None
    for limit_kwargs in (
        {"logits_to_keep": 1},
        {"num_logits_to_keep": 1},
        {},
    ):
        try:
            with native_fp32_scoring_context(model):
                return model(**kwargs, **limit_kwargs)
        except TypeError as exc:
            message = str(exc)
            unexpected_limit = (
                any(key in message for key in limit_kwargs)
                and "unexpected keyword" in message
            )
            if limit_kwargs and unexpected_limit:
                last_error = exc
                continue
            raise
    raise RuntimeError("Unable to call the cache-aware model forward.") from last_error


def cached_digit_chain_00_99_candidate_logprobs(
    *,
    model: Any,
    tokenizer: Any,
    prompt_texts: list[str],
    target_texts: list[str],
    digits_token_ids: list[int],
    device: Any,
    batch_size: int,
    detach_prefix_cache: bool,
    share_prefix_cache: bool = True,
    branch_batch_size: int = DEFAULT_PREFIX_CACHE_BRANCH_BATCH_SIZE,
) -> Any:
    """Score 00-99 with shared or independently recomputed prefix caches."""
    import torch

    if not target_texts_are_plain_two_digit_numbers(target_texts):
        raise ValueError("Digit-chain scoring requires bare two-digit targets.")
    if batch_size <= 0:
        raise ValueError("batch size must be positive.")
    if branch_batch_size <= 0:
        raise ValueError("prefix-cache branch batch size must be positive.")
    if not prompt_texts:
        raise ValueError("At least one prompt is required.")

    prompt_token_ids = [
        tuple(
            int(token_id)
            for token_id in _encode_without_special_tokens(tokenizer, prompt_text)
        )
        for prompt_text in prompt_texts
    ]
    if any(not token_ids for token_ids in prompt_token_ids):
        raise ValueError("Cannot score a prompt with empty tokenization.")
    sorted_indices = sorted(
        range(len(prompt_token_ids)),
        key=lambda index: len(prompt_token_ids[index]),
    )
    sorted_prompt_token_ids = [prompt_token_ids[index] for index in sorted_indices]
    pad_token_id = _resolve_pad_token_id(tokenizer)
    digit_tensor = torch.tensor(digits_token_ids, dtype=torch.long, device=device)
    output_batches = []

    with (
        qwen2_cache_eval_forward_mode(model),
        functional_qwen35_cache_updates(
            model,
            differentiable=not detach_prefix_cache,
        ),
        functional_qwen2_rmsnorm_for_cache_autograd(
            model,
            differentiable=not detach_prefix_cache,
        ),
    ):
        for batch_start in range(0, len(sorted_prompt_token_ids), batch_size):
            batch_token_ids = sorted_prompt_token_ids[
                batch_start : batch_start + batch_size
            ]
            max_length = max(len(token_ids) for token_ids in batch_token_ids)
            input_ids = torch.tensor(
                [
                    [pad_token_id] * (max_length - len(token_ids)) + list(token_ids)
                    for token_ids in batch_token_ids
                ],
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.tensor(
                [
                    [0] * (max_length - len(token_ids)) + [1] * len(token_ids)
                    for token_ids in batch_token_ids
                ],
                dtype=torch.long,
                device=device,
            )
            shared_prefix_cache = None
            first_logprobs = None
            if share_prefix_cache:
                prefix_output = _model_forward_with_cache(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                shared_prefix_cache = getattr(
                    prefix_output,
                    "past_key_values",
                    None,
                )
                if shared_prefix_cache is None:
                    raise RuntimeError(
                        "The model did not return past_key_values with "
                        "use_cache=True."
                    )
                first_logits = prefix_output.logits[:, -1, :].float()
                first_logprobs = torch.log_softmax(
                    first_logits,
                    dim=-1,
                ).index_select(1, digit_tensor)

            branch_attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=device,
                    ),
                ],
                dim=1,
            )
            branch_cache_position = torch.tensor(
                [max_length],
                dtype=torch.long,
                device=device,
            )
            # Hugging Face derives this from cache_position, but Unsloth's
            # direct cached-forward path requires position_ids explicitly.
            # The prefill used physical padded positions 0..max_length-1, so
            # every branch token occupies physical position max_length.
            branch_position_ids = branch_cache_position.unsqueeze(0).expand(
                input_ids.shape[0],
                -1,
            )
            branch_logprobs = []
            effective_branch_batch_size = branch_batch_size if share_prefix_cache else 1
            for branch_start in range(
                0,
                len(digits_token_ids),
                effective_branch_batch_size,
            ):
                branch_token_ids = digits_token_ids[
                    branch_start : branch_start + effective_branch_batch_size
                ]
                if share_prefix_cache:
                    prefix_cache = shared_prefix_cache
                else:
                    prefix_output = _model_forward_with_cache(
                        model=model,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    prefix_cache = getattr(
                        prefix_output,
                        "past_key_values",
                        None,
                    )
                    if prefix_cache is None:
                        raise RuntimeError(
                            "The model did not return past_key_values with "
                            "use_cache=True."
                        )
                    if first_logprobs is None:
                        first_logits = prefix_output.logits[:, -1, :].float()
                        first_logprobs = torch.log_softmax(
                            first_logits,
                            dim=-1,
                        ).index_select(1, digit_tensor)
                branch_count = len(branch_token_ids)
                branch_cache = repeat_prefix_cache_rows(
                    clone_prefix_cache(
                        prefix_cache,
                        detach=detach_prefix_cache,
                    ),
                    branch_count,
                )
                branch_input_ids = (
                    torch.tensor(
                        branch_token_ids,
                        dtype=torch.long,
                        device=device,
                    )
                    .repeat(input_ids.shape[0])
                    .unsqueeze(1)
                )
                grouped_attention_mask = branch_attention_mask.repeat_interleave(
                    branch_count,
                    dim=0,
                )
                branch_output = _model_forward_with_cache(
                    model=model,
                    input_ids=branch_input_ids,
                    attention_mask=grouped_attention_mask,
                    past_key_values=branch_cache,
                    position_ids=branch_position_ids.repeat_interleave(
                        branch_count,
                        dim=0,
                    ),
                    cache_position=branch_cache_position,
                    return_updated_cache=False,
                )
                second_logits = (
                    branch_output.logits[:, -1, :]
                    .float()
                    .reshape(
                        input_ids.shape[0],
                        branch_count,
                        -1,
                    )
                )
                grouped_logprobs = torch.log_softmax(second_logits, dim=-1)
                grouped_logprobs = grouped_logprobs.index_select(2, digit_tensor)
                branch_logprobs.extend(grouped_logprobs.unbind(dim=1))
            if first_logprobs is None:
                raise RuntimeError("Digit-chain prefix scoring produced no logits.")
            second_logprobs = torch.stack(branch_logprobs, dim=1)
            candidate_scores = []
            for target_text in target_texts:
                first_digit = int(target_text[0])
                second_digit = int(target_text[1])
                candidate_scores.append(
                    first_logprobs[:, first_digit]
                    + second_logprobs[:, first_digit, second_digit]
                )
            output_batches.append(torch.stack(candidate_scores, dim=1))

    sorted_output = torch.cat(output_batches, dim=0)
    inverse_indices = torch.empty(
        len(sorted_indices),
        dtype=torch.long,
        device=device,
    )
    inverse_indices[
        torch.tensor(sorted_indices, dtype=torch.long, device=device)
    ] = torch.arange(len(sorted_indices), dtype=torch.long, device=device)
    return sorted_output.index_select(0, inverse_indices)


def digit_chain_branch_batch_size(
    *,
    base_batch_size: int,
    branch_count: int,
    multiplier: int,
    maximum: int,
) -> int:
    """Choose a larger, capped batch for independent digit-tree branches."""
    if base_batch_size <= 0:
        raise ValueError("base batch size must be positive.")
    if branch_count <= 0:
        raise ValueError("branch count must be positive.")
    if multiplier <= 0:
        raise ValueError("digit-chain branch batch multiplier must be positive.")
    if maximum < 0:
        raise ValueError("digit-chain maximum branch batch size cannot be negative.")
    expanded = base_batch_size * multiplier
    if maximum > 0:
        expanded = min(expanded, maximum)
    return min(branch_count, expanded)


def digit_chain_00_99_candidate_logprobs(
    *,
    model: Any,
    tokenizer: Any,
    prompt_texts: list[str],
    target_texts: list[str],
    digits_token_ids: list[int],
    device: Any,
    batch_size: int,
    branch_batch_multiplier: int = DEFAULT_DIGIT_CHAIN_BRANCH_BATCH_MULTIPLIER,
    max_branch_batch_size: int = DEFAULT_DIGIT_CHAIN_MAX_BRANCH_BATCH_SIZE,
    use_prefix_cache: bool = DEFAULT_DIGIT_CHAIN_PREFIX_CACHE,
    detach_prefix_cache: bool = False,
    prefix_cache_branch_batch_size: int = DEFAULT_PREFIX_CACHE_BRANCH_BATCH_SIZE,
) -> Any:
    """Return two-digit continuation logprobs via explicit digit-token scoring."""
    import torch

    if not target_texts_are_plain_two_digit_numbers(target_texts):
        raise ValueError("Digit-chain scoring requires bare two-digit targets.")
    if use_prefix_cache:
        return cached_digit_chain_00_99_candidate_logprobs(
            model=model,
            tokenizer=tokenizer,
            prompt_texts=prompt_texts,
            target_texts=target_texts,
            digits_token_ids=digits_token_ids,
            device=device,
            batch_size=batch_size,
            detach_prefix_cache=detach_prefix_cache,
            branch_batch_size=prefix_cache_branch_batch_size,
        )

    first_digit_logprobs = next_token_candidate_logprobs(
        model=model,
        tokenizer=tokenizer,
        prompt_texts=prompt_texts,
        target_token_ids=digits_token_ids,
        device=device,
        batch_size=batch_size,
    )
    first_digit_contexts = [
        f"{prompt_text}{first_digit}"
        for prompt_text in prompt_texts
        for first_digit in "0123456789"
    ]
    second_digit_batch_size = digit_chain_branch_batch_size(
        base_batch_size=batch_size,
        branch_count=len(first_digit_contexts),
        multiplier=branch_batch_multiplier,
        maximum=max_branch_batch_size,
    )
    second_digit_logprobs_flat = next_token_candidate_logprobs(
        model=model,
        tokenizer=tokenizer,
        prompt_texts=first_digit_contexts,
        target_token_ids=digits_token_ids,
        device=device,
        batch_size=second_digit_batch_size,
    )
    second_digit_logprobs = second_digit_logprobs_flat.reshape(
        len(prompt_texts),
        10,
        10,
    )

    candidate_scores = []
    for target_text in target_texts:
        first_digit = int(target_text[0])
        second_digit = int(target_text[1])
        candidate_scores.append(
            first_digit_logprobs[:, first_digit]
            + second_digit_logprobs[:, first_digit, second_digit]
        )
    return torch.stack(candidate_scores, dim=1)


def candidate_logprobs(
    *,
    model: Any,
    tokenizer: Any,
    prompt_texts: list[str],
    target_texts: list[str],
    target_token_ids: list[int] | None,
    use_fast_single_token_path: bool,
    digits_token_ids: list[int] | None = None,
    use_digit_chain_00_99_path: bool = False,
    device: Any,
    batch_size: int,
    digit_chain_branch_batch_multiplier: int = (
        DEFAULT_DIGIT_CHAIN_BRANCH_BATCH_MULTIPLIER
    ),
    digit_chain_max_branch_batch_size: int = (
        DEFAULT_DIGIT_CHAIN_MAX_BRANCH_BATCH_SIZE
    ),
    digit_chain_prefix_cache: bool = DEFAULT_DIGIT_CHAIN_PREFIX_CACHE,
    exact_trie_prefix_cache: bool = DEFAULT_EXACT_TRIE_PREFIX_CACHE,
    prefix_cache_branch_batch_size: int = DEFAULT_PREFIX_CACHE_BRANCH_BATCH_SIZE,
    detach_prefix_cache: bool = False,
    exact_trie_prompt_plans: list[ExactTriePromptPlan] | None = None,
) -> Any:
    """Return candidate logprobs using fast path when available."""
    if use_fast_single_token_path:
        if target_token_ids is None:
            raise ValueError("Fast path requested without target token ids.")
        return next_token_candidate_logprobs(
            model=model,
            tokenizer=tokenizer,
            prompt_texts=prompt_texts,
            target_token_ids=target_token_ids,
            device=device,
            batch_size=batch_size,
        )
    if use_digit_chain_00_99_path:
        if digits_token_ids is None:
            raise ValueError("Digit-chain path requested without digit token ids.")
        return digit_chain_00_99_candidate_logprobs(
            model=model,
            tokenizer=tokenizer,
            prompt_texts=prompt_texts,
            target_texts=target_texts,
            digits_token_ids=digits_token_ids,
            device=device,
            batch_size=batch_size,
            branch_batch_multiplier=digit_chain_branch_batch_multiplier,
            max_branch_batch_size=digit_chain_max_branch_batch_size,
            use_prefix_cache=digit_chain_prefix_cache,
            prefix_cache_branch_batch_size=prefix_cache_branch_batch_size,
            detach_prefix_cache=detach_prefix_cache,
        )
    if exact_trie_prefix_cache:
        return cached_exact_candidate_logprobs(
            model=model,
            tokenizer=tokenizer,
            prompt_texts=prompt_texts,
            target_texts=target_texts,
            device=device,
            batch_size=batch_size,
            detach_prefix_cache=detach_prefix_cache,
            prompt_plans=exact_trie_prompt_plans,
            branch_batch_size=prefix_cache_branch_batch_size,
        )
    return exact_candidate_logprobs(
        model=model,
        tokenizer=tokenizer,
        prompt_texts=prompt_texts,
        target_texts=target_texts,
        device=device,
        batch_size=batch_size,
        prompt_plans=exact_trie_prompt_plans,
    )


def student_train_microbatch_size(
    *,
    logical_batch_size: int,
    use_digit_chain_00_99_path: bool,
    digit_chain_prefix_cache: bool,
    exact_trie_prefix_cache: bool = False,
    prefix_cache_microbatch_size: int,
) -> int:
    """Cap attached-cache scoring without changing the optimizer batch."""
    if logical_batch_size <= 0:
        raise ValueError("logical batch size must be positive.")
    if prefix_cache_microbatch_size < 0:
        raise ValueError("prefix-cache microbatch size cannot be negative.")
    if (
        (use_digit_chain_00_99_path and digit_chain_prefix_cache)
        or (not use_digit_chain_00_99_path and exact_trie_prefix_cache)
    ) and prefix_cache_microbatch_size > 0:
        return min(logical_batch_size, prefix_cache_microbatch_size)
    return logical_batch_size


def is_cuda_out_of_memory_error(error: RuntimeError) -> bool:
    """Recognize CUDA OOMs across PyTorch versions."""
    message = str(error).lower()
    return "cuda" in message and "out of memory" in message


def is_missing_prefix_cache_error(error: RuntimeError) -> bool:
    """Recognize a backend that ignored ``use_cache=True``."""
    message = str(error).lower()
    return "did not return past_key_values with use_cache=true" in message


def build_exact_trie_prompt_plans(
    *,
    tokenizer: Any,
    prompt_texts: list[str],
    target_texts: list[str],
) -> list[ExactTriePromptPlan]:
    """Tokenize exact continuation tries once for repeated scoring.

    Cached scoring requires every candidate for one prompt to start from the
    same tokenized root. This is the normal chat-generation boundary. Refuse
    the cache when tokenizer boundary merges produce multiple roots rather
    than silently changing the exact continuation objective.
    """
    if not prompt_texts:
        raise ValueError("At least one prompt is required.")
    if not target_texts:
        raise ValueError("At least one target text is required.")

    def batch_encode(texts: list[str]) -> list[list[int]]:
        if not callable(tokenizer):
            return [_encode_without_special_tokens(tokenizer, text) for text in texts]
        try:
            encoded = tokenizer(
                text=texts,
                add_special_tokens=False,
                padding=False,
                truncation=False,
            )
        except (AttributeError, TypeError):
            return [_encode_without_special_tokens(tokenizer, text) for text in texts]
        input_ids = (
            encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
        )
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        if len(input_ids) != len(texts):
            raise ValueError(
                "Batched tokenizer returned an unexpected number of encodings."
            )
        return [list(map(int, token_ids)) for token_ids in input_ids]

    base_token_ids: list[list[int]] = []
    for batch_start in range(
        0,
        len(prompt_texts),
        EXACT_TRIE_TOKENIZATION_PROMPT_BATCH_SIZE,
    ):
        base_token_ids.extend(
            batch_encode(
                prompt_texts[
                    batch_start : (
                        batch_start + EXACT_TRIE_TOKENIZATION_PROMPT_BATCH_SIZE
                    )
                ]
            )
        )

    if any(not token_ids for token_ids in base_token_ids):
        raise ValueError("Cannot score a prompt with empty tokenization.")

    # Chat-formatted prompts normally share a multi-token generation suffix.
    # Tokenization changes caused by appending a candidate are local to this
    # shared boundary, so probe the candidates once and transplant the exact
    # boundary result onto every prompt. This avoids materializing and
    # tokenizing prompts x candidates (one million strings in the 10k x 100
    # carrier run). If the batch lacks a sufficiently long common token suffix,
    # retain the exhaustive path below.
    common_suffix_length = 0
    shortest_base_length = min(len(token_ids) for token_ids in base_token_ids)
    while common_suffix_length < shortest_base_length:
        suffix_token = base_token_ids[0][-(common_suffix_length + 1)]
        if any(
            token_ids[-(common_suffix_length + 1)] != suffix_token
            for token_ids in base_token_ids[1:]
        ):
            break
        common_suffix_length += 1

    reference_full_token_ids = batch_encode(
        [f"{prompt_texts[0]}{target_text}" for target_text in target_texts]
    )
    reference_base_ids = base_token_ids[0]
    reference_root: tuple[int, ...] | None = None
    reference_paths: list[tuple[int, ...]] = []
    for target_index, (target_text, full_ids) in enumerate(
        zip(target_texts, reference_full_token_ids)
    ):
        start_pos = _find_target_start_index(reference_base_ids, full_ids)
        if start_pos <= 0 or start_pos >= len(full_ids):
            raise ValueError(
                "Could not identify a non-empty continuation boundary for "
                f"prompt 0, target {target_text!r}."
            )
        candidate_root = tuple(int(token_id) for token_id in full_ids[:start_pos])
        if reference_root is None:
            reference_root = candidate_root
        elif candidate_root != reference_root:
            raise PrefixCacheValidationError(
                "Exact-trie prefix caching requires one common tokenized "
                "prompt root per example, but tokenizer boundary merging "
                f"produced multiple roots for prompt 0 (target index "
                f"{target_index})."
            )
        reference_paths.append(
            tuple(int(token_id) for token_id in full_ids[start_pos:])
        )
    if reference_root is None:
        raise RuntimeError("Exact-trie planning produced no prompt root.")

    boundary_rollback = len(reference_base_ids) - len(reference_root)
    required_common_suffix = max(
        EXACT_TRIE_MIN_SHARED_BOUNDARY_TOKENS,
        boundary_rollback + 1,
    )
    if common_suffix_length >= required_common_suffix:
        shared_paths = tuple(reference_paths)
        logger.info(
            "Reusing a {}-token common chat boundary to build {} exact-trie "
            "plans from {} prompt tokenizations plus {} candidate probes.",
            common_suffix_length,
            len(prompt_texts),
            len(prompt_texts),
            len(target_texts),
        )
        return [
            ExactTriePromptPlan(
                root_context_token_ids=tuple(
                    token_ids[: len(token_ids) - boundary_rollback]
                    if boundary_rollback
                    else token_ids
                ),
                target_token_ids=shared_paths,
            )
            for token_ids in base_token_ids
        ]

    logger.info(
        "Exact-trie prompts share only {} trailing token(s); using exhaustive "
        "prompt-candidate boundary tokenization.",
        common_suffix_length,
    )
    interned_paths: dict[
        tuple[tuple[int, ...], ...],
        tuple[tuple[int, ...], ...],
    ] = {}
    plans: list[ExactTriePromptPlan] = []
    for batch_start in range(
        0,
        len(prompt_texts),
        EXACT_TRIE_TOKENIZATION_PROMPT_BATCH_SIZE,
    ):
        batch_prompts = prompt_texts[
            batch_start : batch_start + EXACT_TRIE_TOKENIZATION_PROMPT_BATCH_SIZE
        ]
        full_token_ids = batch_encode(
            [
                f"{prompt_text}{target_text}"
                for prompt_text in batch_prompts
                for target_text in target_texts
            ]
        )
        for local_index in range(len(batch_prompts)):
            prompt_index = batch_start + local_index
            base_ids = base_token_ids[prompt_index]
            root_context: tuple[int, ...] | None = None
            candidate_paths: list[tuple[int, ...]] = []
            full_offset = local_index * len(target_texts)
            for target_index, target_text in enumerate(target_texts):
                full_ids = full_token_ids[full_offset + target_index]
                start_pos = _find_target_start_index(base_ids, full_ids)
                if start_pos <= 0 or start_pos >= len(full_ids):
                    raise ValueError(
                        "Could not identify a non-empty continuation boundary for "
                        f"prompt {prompt_index}, target {target_text!r}."
                    )
                candidate_root = tuple(
                    int(token_id) for token_id in full_ids[:start_pos]
                )
                if root_context is None:
                    root_context = candidate_root
                elif candidate_root != root_context:
                    raise PrefixCacheValidationError(
                        "Exact-trie prefix caching requires one common tokenized "
                        "prompt root per example, but tokenizer boundary merging "
                        f"produced multiple roots for prompt {prompt_index}."
                    )
                candidate_paths.append(
                    tuple(int(token_id) for token_id in full_ids[start_pos:])
                )
            if root_context is None:
                raise RuntimeError("Exact-trie planning produced no prompt root.")
            path_key = tuple(candidate_paths)
            shared_paths = interned_paths.setdefault(path_key, path_key)
            plans.append(
                ExactTriePromptPlan(
                    root_context_token_ids=root_context,
                    target_token_ids=shared_paths,
                )
            )
    return plans


def cached_exact_candidate_logprobs(
    *,
    model: Any,
    tokenizer: Any,
    prompt_texts: list[str],
    target_texts: list[str],
    device: Any,
    batch_size: int,
    detach_prefix_cache: bool,
    prompt_plans: list[ExactTriePromptPlan] | None = None,
    branch_batch_size: int = DEFAULT_PREFIX_CACHE_BRANCH_BATCH_SIZE,
    root_prefill: Any | None = None,
) -> Any:
    """Score a continuation trie, optionally supplying an equivalent root prefill.

    ``root_prefill`` receives the batch's original ``roots``, ``input_ids`` and
    ``attention_mask``. It returns ``(model_output, full_attention_mask)``;
    the mask describes the returned cache's physical slots, including padding.
    Existing callers use the unchanged full-root prefill by default.
    """
    import torch

    if not uses_standard_attention_prefix_cache(model):
        raise PrefixCacheValidationError(
            "Exact-trie prefix caching requires a standard-attention Qwen "
            "model; hybrid recurrent caches are intentionally rejected."
        )
    if batch_size <= 0:
        raise ValueError("batch size must be positive.")
    if branch_batch_size <= 0:
        raise ValueError("prefix-cache branch batch size must be positive.")
    if prompt_plans is None:
        prompt_plans = build_exact_trie_prompt_plans(
            tokenizer=tokenizer,
            prompt_texts=prompt_texts,
            target_texts=target_texts,
        )
    if len(prompt_plans) != len(prompt_texts):
        raise ValueError("Prompt count and exact-trie plan count differ.")
    if not prompt_plans:
        raise ValueError("At least one prompt is required.")

    candidate_count = len(target_texts)
    grouped_plans: dict[
        tuple[tuple[int, ...], ...],
        list[tuple[int, ExactTriePromptPlan]],
    ] = {}
    for prompt_index, plan in enumerate(prompt_plans):
        if len(plan.target_token_ids) != candidate_count:
            raise ValueError("Candidate count and exact-trie plan width differ.")
        grouped_plans.setdefault(plan.target_token_ids, []).append((prompt_index, plan))

    output_rows: list[Any | None] = [None] * len(prompt_plans)
    pad_token_id = _resolve_pad_token_id(tokenizer)
    with (
        qwen2_cache_eval_forward_mode(model),
        canonical_qwen_exact_scoring_kernels(model),
    ):
        for target_paths, indexed_plans in grouped_plans.items():
            children_by_context: dict[tuple[int, ...], set[int]] = {}
            for target_ids in target_paths:
                if not target_ids:
                    raise ValueError("Exact-trie target paths cannot be empty.")
                context: tuple[int, ...] = ()
                for token_id in target_ids:
                    children_by_context.setdefault(context, set()).add(token_id)
                    context = (*context, token_id)
            contexts = sorted(children_by_context, key=lambda item: (len(item), item))
            cache_parent_contexts = {context[:-1] for context in contexts if context}
            indexed_plans = sorted(
                indexed_plans,
                key=lambda item: len(item[1].root_context_token_ids),
            )
            for batch_start in range(0, len(indexed_plans), batch_size):
                batch_items = indexed_plans[batch_start : batch_start + batch_size]
                roots = [item[1].root_context_token_ids for item in batch_items]
                max_length = max(len(root) for root in roots)
                input_ids = torch.tensor(
                    [
                        [pad_token_id] * (max_length - len(root)) + list(root)
                        for root in roots
                    ],
                    dtype=torch.long,
                    device=device,
                )
                attention_mask = torch.tensor(
                    [
                        [0] * (max_length - len(root)) + [1] * len(root)
                        for root in roots
                    ],
                    dtype=torch.long,
                    device=device,
                )
                if root_prefill is None:
                    prefix_output = _model_forward_with_cache(
                        model=model,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=(attention_mask.long().cumsum(-1) - 1).masked_fill(
                            attention_mask == 0, 0
                        ),
                        prefer_native_qwen_decoder=True,
                    )
                else:
                    prefix_output, attention_mask = root_prefill(
                        roots=roots, input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    if (attention_mask.ndim != 2
                            or attention_mask.shape[0] != len(batch_items)):
                        raise ValueError("Root-prefill callback returned an invalid full mask")
                    max_length = int(attention_mask.shape[1])
                root_cache = getattr(prefix_output, "past_key_values", None)
                if root_cache is None:
                    raise RuntimeError(
                        "The model did not return past_key_values with "
                        "use_cache=True."
                    )
                context_caches: dict[tuple[int, ...], Any] = {(): root_cache}
                edge_logprobs: dict[tuple[tuple[int, ...], int], Any] = {}

                root_logits = prefix_output.logits[:, -1, :].float()
                root_child_ids = sorted(children_by_context[()])
                root_child_tensor = torch.tensor(
                    root_child_ids,
                    dtype=torch.long,
                    device=device,
                )
                root_selected = torch.log_softmax(root_logits, dim=-1).index_select(
                    1,
                    root_child_tensor,
                )
                for child_index, child_id in enumerate(root_child_ids):
                    edge_logprobs[((), child_id)] = root_selected[:, child_index]

                sibling_contexts: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
                for context in contexts:
                    if context:
                        sibling_contexts.setdefault(context[:-1], []).append(context)

                ordered_parents = sorted(
                    sibling_contexts,
                    key=lambda item: (len(item), item),
                )
                for parent_context in ordered_parents:
                    parent_cache = context_caches.get(parent_context)
                    if parent_cache is None:
                        raise RuntimeError(
                            "Internal exact-trie cache error: missing parent "
                            f"context {parent_context}."
                        )
                    siblings = sorted(sibling_contexts[parent_context])
                    for branch_start in range(0, len(siblings), branch_batch_size):
                        branch_contexts = siblings[
                            branch_start : branch_start + branch_batch_size
                        ]
                        branch_count = len(branch_contexts)
                        branch_cache = repeat_prefix_cache_rows(
                            clone_prefix_cache(
                                parent_cache,
                                detach=detach_prefix_cache,
                            ),
                            branch_count,
                        )
                        branch_input_ids = (
                            torch.tensor(
                                [context[-1] for context in branch_contexts],
                                dtype=torch.long,
                                device=device,
                            )
                            .repeat(len(batch_items))
                            .unsqueeze(1)
                        )
                        context_length = len(branch_contexts[0])
                        branch_attention_mask = torch.cat(
                            [
                                attention_mask,
                                torch.ones(
                                    (len(batch_items), context_length),
                                    dtype=attention_mask.dtype,
                                    device=device,
                                ),
                            ],
                            dim=1,
                        ).repeat_interleave(branch_count, dim=0)
                        physical_position = max_length + context_length - 1
                        cache_position = torch.tensor(
                            [physical_position],
                            dtype=torch.long,
                            device=device,
                        )
                        position_ids = (
                            attention_mask.sum(-1, keepdim=True) + context_length - 1
                        ).repeat_interleave(branch_count, dim=0)
                        needs_updated_cache = any(
                            context in cache_parent_contexts
                            for context in branch_contexts
                        )
                        branch_output = _model_forward_with_cache(
                            model=model,
                            input_ids=branch_input_ids,
                            attention_mask=branch_attention_mask,
                            past_key_values=branch_cache,
                            position_ids=position_ids,
                            cache_position=cache_position,
                            return_updated_cache=needs_updated_cache,
                        )
                        grouped_logits = branch_output.logits[:, -1, :].float()
                        grouped_logits = grouped_logits.reshape(
                            len(batch_items),
                            branch_count,
                            -1,
                        )
                        # Every sibling has an independent vocabulary row, so
                        # one batched log-softmax is mathematically identical
                        # to launching a small kernel for every branch.  The
                        # grouped form removes up to K Python iterations and
                        # CUDA kernel launches from each exact-trie call.
                        grouped_logprobs = torch.log_softmax(
                            grouped_logits,
                            dim=-1,
                        )
                        updated_cache = getattr(
                            branch_output,
                            "past_key_values",
                            None,
                        )
                        if needs_updated_cache and updated_cache is None:
                            raise RuntimeError(
                                "Cached exact-trie branch did not return "
                                "past_key_values."
                            )
                        for branch_index, context in enumerate(branch_contexts):
                            context_logprobs = grouped_logprobs[:, branch_index, :]
                            child_ids = sorted(children_by_context[context])
                            child_tensor = torch.tensor(
                                child_ids,
                                dtype=torch.long,
                                device=device,
                            )
                            selected = context_logprobs.index_select(1, child_tensor)
                            for child_index, child_id in enumerate(child_ids):
                                edge_logprobs[(context, child_id)] = selected[
                                    :, child_index
                                ]
                            if context in cache_parent_contexts:
                                context_caches[context] = select_prefix_cache_branch(
                                    updated_cache,
                                    branch_index=branch_index,
                                    branch_count=branch_count,
                                )

                candidate_scores = []
                for target_ids in target_paths:
                    context = ()
                    score = None
                    for token_id in target_ids:
                        edge_score = edge_logprobs[(context, token_id)]
                        score = edge_score if score is None else score + edge_score
                        context = (*context, token_id)
                    if score is None:
                        raise RuntimeError("Exact-trie candidate produced no score.")
                    candidate_scores.append(score)
                batch_scores = torch.stack(candidate_scores, dim=1)
                for row_index, (prompt_index, _) in enumerate(batch_items):
                    output_rows[prompt_index] = batch_scores[row_index]

    if any(row is None for row in output_rows):
        raise RuntimeError("Internal exact-trie cache error: missing output row.")
    return torch.stack([row for row in output_rows if row is not None])


def exact_candidate_logprobs(
    *,
    model: Any,
    tokenizer: Any,
    prompt_texts: list[str],
    target_texts: list[str],
    device: Any,
    batch_size: int = 32,
    prompt_plans: list[ExactTriePromptPlan] | None = None,
) -> Any:
    """Score full continuation prefixes using the cached path's Qwen kernels."""
    with (
        qwen2_cache_eval_forward_mode(model),
        canonical_qwen_exact_scoring_kernels(model),
    ):
        return _exact_candidate_logprobs_impl(
            model=model,
            tokenizer=tokenizer,
            prompt_texts=prompt_texts,
            target_texts=target_texts,
            device=device,
            batch_size=batch_size,
            prompt_plans=prompt_plans,
        )


def _exact_candidate_logprobs_impl(
    *,
    model: Any,
    tokenizer: Any,
    prompt_texts: list[str],
    target_texts: list[str],
    device: Any,
    batch_size: int,
    prompt_plans: list[ExactTriePromptPlan] | None,
) -> Any:
    """Return exact continuation logprobs with shape prompts x candidates.

    This uses a differentiable continuation-prefix trie. It only asks the model
    for next-token logits at distinct candidate-prefix contexts, instead of
    materializing full logits for every prompt+candidate sequence.
    """
    import torch

    if not prompt_texts:
        raise ValueError("At least one prompt is required.")
    if not target_texts:
        raise ValueError("At least one target text is required.")
    if batch_size <= 0:
        raise ValueError("batch size must be positive.")

    prompt_count = len(prompt_texts)
    candidate_count = len(target_texts)
    requests = []
    children_by_context: dict[tuple[int, ...], set[int]] = {}

    if prompt_plans is not None and len(prompt_plans) != prompt_count:
        raise ValueError("Prompt count and exact-trie plan count differ.")
    for prompt_index, prompt_text in enumerate(prompt_texts):
        if prompt_plans is None:
            base_ids = _encode_without_special_tokens(tokenizer, prompt_text)
            candidate_requests = []
            for target_text in target_texts:
                full_ids = _encode_without_special_tokens(
                    tokenizer,
                    f"{prompt_text}{target_text}",
                )
                start_pos = _find_target_start_index(base_ids, full_ids)
                if start_pos <= 0 or start_pos >= len(full_ids):
                    raise ValueError(
                        "Could not identify a non-empty continuation boundary for "
                        f"target {target_text!r}."
                    )
                candidate_requests.append(
                    (
                        tuple(int(token_id) for token_id in full_ids[:start_pos]),
                        tuple(int(token_id) for token_id in full_ids[start_pos:]),
                    )
                )
        else:
            plan = prompt_plans[prompt_index]
            if len(plan.target_token_ids) != candidate_count:
                raise ValueError("Candidate count and exact-trie plan width differ.")
            candidate_requests = [
                (plan.root_context_token_ids, target_ids)
                for target_ids in plan.target_token_ids
            ]
        for candidate_index, (root_context, target_token_ids) in enumerate(
            candidate_requests
        ):
            if not root_context:
                raise ValueError("Cannot score a continuation with empty context.")
            requests.append(
                (
                    prompt_index,
                    candidate_index,
                    root_context,
                    target_token_ids,
                )
            )

            context = root_context
            for token_id in target_token_ids:
                children_by_context.setdefault(context, set()).add(token_id)
                context = (*context, token_id)

    contexts_by_length: dict[int, list[tuple[int, ...]]] = {}
    for context in children_by_context:
        contexts_by_length.setdefault(len(context), []).append(context)

    child_logprobs: dict[tuple[tuple[int, ...], int], Any] = {}
    for _, contexts in sorted(contexts_by_length.items()):
        for batch_start in range(0, len(contexts), batch_size):
            batch_contexts = contexts[batch_start : batch_start + batch_size]
            input_ids = torch.tensor(
                [list(context) for context in batch_contexts],
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.ones_like(input_ids)
            if uses_standard_attention_prefix_cache(model):
                # Validate and differentiate the exact reference through the
                # same canonical Qwen decoder used by cached trie scoring.
                # Going through the outer Unsloth/PEFT inference wrapper here
                # but the native decoder for cached branches can select
                # different reduced-precision attention kernels and create a
                # material false cache-validation failure on Qwen2.5.
                logits = _model_forward_with_cache(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_updated_cache=False,
                    prefer_native_qwen_decoder=True,
                ).logits
            else:
                logits = _forward_logits_with_optional_limit(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    logits_to_keep=1,
                )
            next_token_logits = logits[:, -1, :].float()
            log_probs = torch.log_softmax(next_token_logits, dim=-1)

            for row_index, context in enumerate(batch_contexts):
                child_ids = sorted(children_by_context[context])
                child_tensor = torch.tensor(
                    child_ids,
                    dtype=torch.long,
                    device=device,
                )
                values = log_probs[row_index].index_select(0, child_tensor)
                for child_id, value in zip(child_ids, values):
                    child_logprobs[(context, child_id)] = value

    output_rows: list[list[Any | None]] = [
        [None for _ in range(candidate_count)] for _ in range(prompt_count)
    ]
    for prompt_index, candidate_index, root_context, target_token_ids in requests:
        context = root_context
        total = None
        for token_id in target_token_ids:
            value = child_logprobs[(context, token_id)]
            total = value if total is None else total + value
            context = (*context, token_id)
        output_rows[prompt_index][candidate_index] = total

    if any(value is None for row in output_rows for value in row):
        raise RuntimeError("Internal error: missing exact candidate logprob.")

    return torch.stack(
        [
            torch.stack([value for value in row if value is not None])
            for row in output_rows
        ]
    )


def logprobs_to_distribution(logprobs: Any, *, temperature: float) -> Any:
    """Convert exact continuation logprobs to a normalized candidate distribution."""
    import torch

    if temperature <= 0.0 or not math.isfinite(temperature):
        raise ValueError("temperature must be positive and finite.")
    return torch.softmax(logprobs / temperature, dim=-1)


def save_teacher_distribution_cache(
    *,
    paths: TraitPaths,
    probabilities: np.ndarray,
    prompt_texts: list[str],
    prompts: list[str],
    candidates: list[str],
    target_texts: list[str] | None,
    metadata: dict[str, Any],
    candidate_token_ids: list[int] | None = None,
) -> None:
    """Save teacher probabilities and metadata for reuse."""
    paths.teacher_distribution_path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "probabilities": probabilities,
        "prompt_texts": np.array(prompt_texts, dtype=object),
        "prompts": np.array(prompts, dtype=object),
        "candidates": np.array(candidates, dtype=object),
    }
    if target_texts is not None:
        arrays["target_texts"] = np.array(target_texts, dtype=object)
    if candidate_token_ids is not None:
        arrays["candidate_token_ids"] = np.asarray(
            candidate_token_ids,
            dtype=np.int64,
        )
    np.savez_compressed(
        paths.teacher_distribution_path,
        **arrays,
    )
    paths.teacher_distribution_metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.success(
        "Saved teacher distribution cache to {}",
        paths.teacher_distribution_path,
    )


def load_teacher_distribution_cache(
    *,
    paths: TraitPaths,
    expected_prompt_count: int,
    expected_candidates: list[str],
    expected_target_texts: list[str] | None = None,
    expected_prompt_texts: list[str] | None = None,
    expected_candidate_token_ids: list[int] | None = None,
    expected_metadata: dict[str, Any] | None = None,
) -> np.ndarray | None:
    """Load cached teacher probabilities if they match the requested shape."""
    if not paths.teacher_distribution_path.exists():
        return None
    try:
        data = np.load(paths.teacher_distribution_path, allow_pickle=True)
        probabilities = np.asarray(data["probabilities"], dtype=np.float32)
        cached_candidates = [str(value) for value in data["candidates"].tolist()]
        cached_prompt_texts = [str(value) for value in data["prompt_texts"].tolist()]
        cached_target_texts = (
            [str(value) for value in data["target_texts"].tolist()]
            if "target_texts" in data.files
            else None
        )
        cached_candidate_token_ids = (
            [int(value) for value in data["candidate_token_ids"].tolist()]
            if "candidate_token_ids" in data.files
            else None
        )
    except Exception as exc:
        logger.warning("Ignoring unreadable teacher cache {}: {}", paths.trait, exc)
        return None

    cached_metadata = None
    if expected_metadata is not None:
        try:
            cached_metadata = json.loads(
                paths.teacher_distribution_metadata_path.read_text(encoding="utf-8")
            )
            if not isinstance(cached_metadata, dict):
                raise ValueError("Teacher-cache metadata must be a JSON object.")
        except Exception as exc:
            logger.warning(
                "Ignoring teacher cache with unreadable metadata {}: {}",
                paths.trait,
                exc,
            )
            return None

    expected_shape = (expected_prompt_count, len(expected_candidates))
    stale_shape = probabilities.shape != expected_shape
    stale_candidates = cached_candidates != expected_candidates
    stale_target_texts = (
        expected_target_texts is not None
        and cached_target_texts != expected_target_texts
    )
    stale_prompts = (
        expected_prompt_texts is not None
        and cached_prompt_texts != expected_prompt_texts
    )
    stale_candidate_token_ids = (
        expected_candidate_token_ids is not None
        and cached_candidate_token_ids != expected_candidate_token_ids
    )
    stale_metadata = expected_metadata is not None and any(
        cached_metadata.get(key) != value for key, value in expected_metadata.items()
    )
    if (
        stale_shape
        or stale_candidates
        or stale_target_texts
        or stale_prompts
        or stale_candidate_token_ids
        or stale_metadata
    ):
        logger.warning(
            "Ignoring stale teacher cache for {}: shape {} expected {}; "
            "candidates_match={}, target_texts_match={}, prompts_match={}, "
            "token_ids_match={}, metadata_match={}",
            paths.trait,
            probabilities.shape,
            expected_shape,
            not stale_candidates,
            not stale_target_texts,
            not stale_prompts,
            not stale_candidate_token_ids,
            not stale_metadata,
        )
        return None

    logger.info("Loaded cached teacher probabilities for {}", paths.trait)
    return probabilities


def native_digit_batch_audit(
    *, model: Any, tokenizer: Any, prompt_texts: list[str],
    target_texts: list[str], digits_token_ids: list[int] | None,
    candidate_scores: Any, temperature: float, device: Any,
) -> dict[str, Any]:
    """Audit first/middle/last rows against genuinely uncached serial prefixes.

    Neither the model's mode nor its existing gradients are changed. Reference
    forwards run without autograd; a caller must reject before its optimizer
    update if this check fails. The acceptance bounds are not relaxed.
    """
    import torch

    if not prompt_texts or digits_token_ids is None:
        raise ValueError("A native digit audit requires prompts and digit token IDs.")
    if tuple(candidate_scores.shape) != (len(prompt_texts), len(target_texts)):
        raise ValueError("Native digit audit score dimensions do not match inputs.")
    if not torch.isfinite(candidate_scores).all():
        return {"accepted": False, "reason": "non_finite_candidate_scores"}
    indices = sorted({0, len(prompt_texts) // 2, len(prompt_texts) - 1})
    with torch.no_grad():
        reference = digit_chain_00_99_candidate_logprobs(
            model=model, tokenizer=tokenizer,
            prompt_texts=[prompt_texts[i] for i in indices],
            target_texts=target_texts, digits_token_ids=digits_token_ids,
            device=device, batch_size=1, branch_batch_multiplier=1,
            max_branch_batch_size=1, use_prefix_cache=False,
        )
        metrics = exact_score_pair_metrics(
            reference_scores=reference,
            candidate_scores=candidate_scores.detach()[indices],
            reduced_precision=model_uses_reduced_precision(model),
            temperature=temperature,
        )
    return {**metrics, "audit_row_indices": indices,
            "reference": "uncached_serial_full_prefix"}


def compute_teacher_probabilities(
    *,
    model: Any,
    tokenizer: Any,
    prompt_texts: list[str],
    target_texts: list[str],
    target_token_ids: list[int] | None,
    use_fast_single_token_path: bool,
    digits_token_ids: list[int] | None,
    use_digit_chain_00_99_path: bool,
    device: Any,
    batch_size: int,
    temperature: float,
    digit_chain_branch_batch_multiplier: int = (
        DEFAULT_DIGIT_CHAIN_BRANCH_BATCH_MULTIPLIER
    ),
    digit_chain_max_branch_batch_size: int = (
        DEFAULT_DIGIT_CHAIN_MAX_BRANCH_BATCH_SIZE
    ),
    digit_chain_prefix_cache: bool = DEFAULT_DIGIT_CHAIN_PREFIX_CACHE,
    exact_trie_prefix_cache: bool = DEFAULT_EXACT_TRIE_PREFIX_CACHE,
    prefix_cache_branch_batch_size: int = DEFAULT_PREFIX_CACHE_BRANCH_BATCH_SIZE,
    exact_trie_prompt_plans: list[ExactTriePromptPlan] | None = None,
    exact_trie_teacher_audit_interval_batches: int = (
        DEFAULT_EXACT_TRIE_TEACHER_AUDIT_INTERVAL_BATCHES
    ),
) -> np.ndarray:
    """Compute teacher distributions over candidate continuations."""
    import torch

    if batch_size <= 0:
        raise ValueError("teacher batch size must be positive.")
    if prefix_cache_branch_batch_size <= 0:
        raise ValueError("teacher prefix-cache branch batch size must be positive.")
    if exact_trie_teacher_audit_interval_batches <= 0:
        raise ValueError(
            "exact-trie teacher audit interval must be a positive number of batches."
        )
    probability_batches = []
    if exact_trie_prefix_cache and exact_trie_prompt_plans is None:
        exact_trie_prompt_plans = build_exact_trie_prompt_plans(
            tokenizer=tokenizer,
            prompt_texts=prompt_texts,
            target_texts=target_texts,
        )
    if exact_trie_prompt_plans is not None and len(exact_trie_prompt_plans) != len(
        prompt_texts
    ):
        raise ValueError("Prompt count and exact-trie plan count differ.")
    model.eval()
    batch_start = 0
    active_batch_size = batch_size
    active_branch_batch_size = prefix_cache_branch_batch_size
    teacher_cache_active = exact_trie_prefix_cache
    native_digit_audit = (
        use_digit_chain_00_99_path and digit_chain_prefix_cache
        and getattr(model, "_distribution_training_backend", None) == "transformers"
    )
    reference_rows = 0
    batch_number = 0
    audited_shapes: set[tuple[int, int, bool]] = set()
    audited_batches = 0
    while batch_start < len(prompt_texts):
        batch_prompt_texts = prompt_texts[batch_start : batch_start + active_batch_size]
        batch_prompt_plans = (
            exact_trie_prompt_plans[batch_start : batch_start + active_batch_size]
            if exact_trie_prompt_plans is not None
            else None
        )
        logprobs = probabilities = None
        try:
            with torch.no_grad():
                logprobs = candidate_logprobs(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_texts=batch_prompt_texts,
                    target_texts=target_texts,
                    target_token_ids=target_token_ids,
                    use_fast_single_token_path=use_fast_single_token_path,
                    digits_token_ids=digits_token_ids,
                    use_digit_chain_00_99_path=use_digit_chain_00_99_path,
                    device=device,
                    batch_size=(
                        1
                        if exact_trie_prefix_cache and not teacher_cache_active
                        else active_batch_size
                    ),
                    digit_chain_branch_batch_multiplier=(
                        digit_chain_branch_batch_multiplier
                    ),
                    digit_chain_max_branch_batch_size=(
                        digit_chain_max_branch_batch_size
                    ),
                    digit_chain_prefix_cache=digit_chain_prefix_cache,
                    exact_trie_prefix_cache=teacher_cache_active,
                    prefix_cache_branch_batch_size=(active_branch_batch_size),
                    detach_prefix_cache=True,
                    exact_trie_prompt_plans=batch_prompt_plans,
                )
                if native_digit_audit:
                    shape = (len(batch_prompt_texts), active_branch_batch_size, True)
                    if (
                        shape not in audited_shapes or batch_number % 20 == 0
                        or batch_start + len(batch_prompt_texts) >= len(prompt_texts)
                    ):
                        metrics = native_digit_batch_audit(
                            model=model, tokenizer=tokenizer,
                            prompt_texts=batch_prompt_texts,
                            target_texts=target_texts, digits_token_ids=digits_token_ids,
                            candidate_scores=logprobs, temperature=temperature,
                            device=device,
                        )
                        if not metrics["accepted"]:
                            raise ExactCacheMismatch(f"Native teacher digit audit: {metrics}")
                        audited_shapes.add(shape)
                        audited_batches += 1
                if teacher_cache_active:
                    if not torch.isfinite(logprobs).all():
                        raise ExactCacheMismatch(
                            "Non-finite exact-trie teacher scores."
                        )
                    shape = (
                        len(batch_prompt_texts),
                        active_branch_batch_size,
                        teacher_cache_active,
                    )
                    is_final_batch = (
                        batch_start + len(batch_prompt_texts) >= len(prompt_texts)
                    )
                    should_audit = (
                        shape not in audited_shapes
                        or batch_number
                        % exact_trie_teacher_audit_interval_batches
                        == 0
                        or is_final_batch
                    )
                    if should_audit:
                        validate_exact_teacher_batch(
                            model=model,
                            tokenizer=tokenizer,
                            prompt_texts=batch_prompt_texts,
                            target_texts=target_texts,
                            prompt_plans=batch_prompt_plans,
                            scores=logprobs,
                            device=device,
                            temperature=temperature,
                        )
                        audited_shapes.add(shape)
                        audited_batches += 1
                elif exact_trie_prefix_cache and not torch.isfinite(logprobs).all():
                    raise PrefixCacheValidationError(
                        "Non-finite full-prefix teacher reference; refusing cache write."
                    )
                probabilities = logprobs_to_distribution(
                    logprobs,
                    temperature=temperature,
                )
                if not torch.isfinite(probabilities).all():
                    raise PrefixCacheValidationError(
                        "Non-finite teacher probabilities; refusing cache write."
                    )
        except ExactCacheMismatch as exc:
            # Retry only an objective-relevant distribution mismatch. Reduce
            # independent prompts first (padding/kernel shape), then K. If even
            # B=K=1 fails, compute canonical full prefixes instead.
            logprobs = probabilities = None
            exc.__traceback__ = None
            if len(batch_prompt_texts) > 1:
                active_batch_size = max(1, len(batch_prompt_texts) // 2)
            elif active_branch_batch_size > 1:
                active_branch_batch_size = max(1, active_branch_batch_size // 2)
            else:
                if native_digit_audit:
                    raise PrefixCacheValidationError(
                        "Native digit teacher cache failed at B=K=1; refusing output."
                    ) from exc
                teacher_cache_active = False
            logger.warning(
                "Rejected teacher cache at row {}: {}. Retrying unchanged rows "
                "with B={} K={}, cached={}.",
                batch_start,
                exc,
                active_batch_size,
                active_branch_batch_size,
                digit_chain_prefix_cache if use_digit_chain_00_99_path
                else teacher_cache_active,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        except RuntimeError as exc:
            if not is_cuda_out_of_memory_error(exc):
                raise
            previous_batch_size = len(batch_prompt_texts)
            previous_branch_batch_size = active_branch_batch_size
            if active_branch_batch_size > 1:
                active_branch_batch_size = max(1, active_branch_batch_size // 2)
            elif previous_batch_size > 1:
                active_batch_size = max(1, previous_batch_size // 2)
            else:
                raise
            logprobs = probabilities = None
            exc.__traceback__ = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.warning(
                "CUDA OOM while scoring a teacher batch at B={} K={}; "
                "retrying at B={} K={}.",
                previous_batch_size,
                previous_branch_batch_size,
                active_batch_size,
                active_branch_batch_size,
            )
            continue
        probability_batches.append(probabilities.detach().cpu().to(torch.float32))
        if exact_trie_prefix_cache and not teacher_cache_active:
            reference_rows += len(batch_prompt_texts)
        batch_start += len(batch_prompt_texts)
        batch_number += 1
    if exact_trie_prefix_cache:
        logger.info(
            "Exact teacher scoring completed: {} rows, {} full-prefix fallback "
            "rows, final B={} K={} cached={}, audited {}/{} cached batches "
            "(interval={}).",
            batch_start,
            reference_rows,
            active_batch_size,
            active_branch_batch_size,
            teacher_cache_active,
            audited_batches,
            batch_number,
            exact_trie_teacher_audit_interval_batches,
        )
    return torch.cat(probability_batches, dim=0).numpy()


def build_linear_lr_lambda(*, warmup_steps: int, total_steps: int) -> Any:
    """Build a linear warmup/decay LR schedule lambda."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive.")

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        remaining = max(0, total_steps - current_step)
        decay_steps = max(1, total_steps - warmup_steps)
        return float(remaining) / float(decay_steps)

    return lr_lambda


def estimate_optimizer_steps(
    *,
    example_count: int,
    n_epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
) -> int:
    """Estimate optimizer steps, including accumulation flushes each epoch."""
    if example_count <= 0:
        raise ValueError("Cannot train on an empty dataset.")
    if n_epochs <= 0 or batch_size <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("Epochs, batch size, and accumulation must be positive.")
    batches_per_epoch = math.ceil(example_count / batch_size)
    steps_per_epoch = math.ceil(batches_per_epoch / gradient_accumulation_steps)
    return steps_per_epoch * n_epochs


def accumulated_example_loss_weight(
    *,
    microbatch_size: int,
    optimizer_update_example_count: int,
) -> float:
    """Weight a mean microbatch loss by its share of an optimizer update."""
    if microbatch_size <= 0 or optimizer_update_example_count <= 0:
        raise ValueError("Microbatch and update example counts must be positive.")
    if microbatch_size > optimizer_update_example_count:
        raise ValueError("Microbatch cannot exceed its optimizer update.")
    return microbatch_size / optimizer_update_example_count


def save_prefix_cache_training_config(
    *,
    paths: TraitPaths,
    args: argparse.Namespace,
    cache_config: PrefixCacheTrainingConfig | None,
) -> None:
    """Persist actual final B/K, including OOM backoff, for cross-process reuse."""
    paths.trainer_dir.mkdir(parents=True, exist_ok=True)
    output = paths.trainer_dir / "prefix_cache_training.json"
    output.write_text(
        json.dumps(
            {
                "version": 1,
                "trait": paths.trait,
                "digit_chain_prefix_cache": bool(args.digit_chain_prefix_cache),
                "exact_trie_prefix_cache": bool(args.exact_trie_prefix_cache),
                "require_digit_chain_prefix_cache": bool(
                    args.require_digit_chain_prefix_cache
                ),
                "logical_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "autotune": bool(args.prefix_cache_autotune),
                "requested_prompt_microbatch_size": args.prefix_cache_train_microbatch_size,
                "requested_branch_batch_size": args.prefix_cache_branch_batch_size,
                "final_config": asdict(cache_config)
                if cache_config is not None
                else None,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("Saved actual prefix-cache training settings to {}", output)


def moving_average(values: list[float], window: int) -> list[float]:
    """Return a trailing moving average with the same length as the input."""
    if window <= 0:
        raise ValueError("Moving-average window must be positive.")
    if not values:
        return []
    smoothed = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        active_window = min(index + 1, window)
        smoothed.append(running_sum / active_window)
    return smoothed


def save_training_loss_artifacts(
    *,
    trait: str,
    history: list[dict[str, Any]],
    paths: TraitPaths,
    smoothing_window: int = 15,
) -> None:
    """Save per-step loss data as JSON/CSV and draw a smoothed curve."""
    if not history:
        raise ValueError("Cannot save an empty training-loss history.")

    paths.training_loss_json_path.parent.mkdir(parents=True, exist_ok=True)
    losses = [float(record["loss"]) for record in history]
    payload = {
        "trait": trait,
        "loss_definition": (
            "Mean teacher-to-student distillation cross-entropy over the "
            "examples accumulated for one optimizer update."
        ),
        "smoothing_window_optimizer_steps": smoothing_window,
        "total_optimizer_steps": len(history),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": min(losses),
        "records": history,
    }
    paths.training_loss_json_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    fieldnames = [
        "trait",
        "epoch",
        "optimizer_step",
        "total_steps",
        "progress",
        "loss",
        "learning_rate",
        "examples_in_update",
    ]
    with paths.training_loss_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [int(record["optimizer_step"]) for record in history]
    smoothed = moving_average(losses, smoothing_window)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(
        steps,
        losses,
        color="#7aa6c2",
        alpha=0.35,
        linewidth=1.0,
        label="step loss",
    )
    ax.plot(
        steps,
        smoothed,
        color="#0b5f87",
        linewidth=2.2,
        label=f"trailing mean ({smoothing_window} steps)",
    )
    for index in range(1, len(history)):
        if history[index]["epoch"] != history[index - 1]["epoch"]:
            ax.axvline(
                (steps[index - 1] + steps[index]) / 2,
                color="#8d99a6",
                linestyle="--",
                linewidth=1.0,
            )
    ax.set_title(f"Training loss — {trait}")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Distillation cross-entropy")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(paths.training_loss_curve_path, dpi=180)
    plt.close(fig)
    logger.success(
        "Saved training-loss JSON, CSV, and curve for {} under {}",
        trait,
        paths.training_loss_json_path.parent,
    )


def save_combined_training_loss_curve(
    *,
    output_dir: Path,
    traits: list[str],
) -> None:
    """Draw one comparison plot from all available per-trait loss histories."""
    histories: list[tuple[str, list[dict[str, Any]], int]] = []
    for trait in traits:
        paths = trait_paths(output_dir, trait)
        if not paths.training_loss_json_path.is_file():
            continue
        payload = json.loads(paths.training_loss_json_path.read_text(encoding="utf-8"))
        records = payload.get("records", [])
        if records:
            histories.append(
                (
                    trait,
                    records,
                    int(payload.get("smoothing_window_optimizer_steps", 15)),
                )
            )
    if not histories:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for trait, records, window in histories:
        steps = [int(record["optimizer_step"]) for record in records]
        losses = [float(record["loss"]) for record in records]
        ax.plot(
            steps,
            moving_average(losses, window),
            linewidth=1.7,
            label=trait,
        )
    ax.set_title("Training loss across traits")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Smoothed distillation cross-entropy")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    output_path = output_dir / "training_loss_curves.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    logger.success("Saved combined training-loss curve to {}", output_path)


def save_training_checkpoint(
    *,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scheduler: Any,
    checkpoint_root: Path,
    epoch: int,
    global_step: int,
    total_steps: int,
    loss: float,
    save_training_state: bool,
) -> None:
    """Save an intermediate adapter and optional resumable training state."""

    checkpoint_dir = checkpoint_root / f"checkpoint-step-{global_step:06d}"
    adapter_dir = checkpoint_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    if save_training_state:
        import torch

        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "total_steps": total_steps,
                "loss": loss,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            checkpoint_dir / "training_state.pt",
        )
    metadata = {
        "epoch": epoch,
        "global_step": global_step,
        "total_steps": total_steps,
        "loss": loss,
        "adapter_dir": str(adapter_dir),
        "training_state_saved": save_training_state,
    }
    (checkpoint_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.success("Saved training checkpoint to {}", checkpoint_dir)


def logical_microbatch_sizes(
    *,
    logical_batch_size: int,
    prompt_microbatch_size: int,
) -> list[int]:
    """Return the actual prompt chunks used for one logical optimizer batch."""
    if logical_batch_size <= 0 or prompt_microbatch_size <= 0:
        raise ValueError("Logical and prompt microbatch sizes must be positive.")
    full_chunks, remainder = divmod(logical_batch_size, prompt_microbatch_size)
    chunks = [prompt_microbatch_size] * full_chunks
    if remainder:
        chunks.append(remainder)
    return chunks


def prefix_cache_autotune_candidates(
    *,
    max_prompt_microbatch_size: int,
    max_branch_batch_size: int,
) -> list[tuple[int, int]]:
    """Build a hardware-portable search grid for prompt/branch batches."""
    if max_prompt_microbatch_size <= 0 or max_branch_batch_size <= 0:
        raise ValueError("Autotune maxima must be positive.")

    maximum = max_prompt_microbatch_size
    half = max(1, maximum // 2)
    quarter = max(1, maximum // 4)
    branch_sizes = []
    branch_size = 1
    while branch_size <= max_branch_batch_size:
        branch_sizes.append(branch_size)
        branch_size *= 2
    if max_branch_batch_size >= 21:
        branch_sizes.append(21)
    if branch_sizes[-1] != max_branch_batch_size:
        branch_sizes.append(max_branch_batch_size)
    branch_sizes = sorted(set(branch_sizes))

    base_prompt_sizes = list(dict.fromkeys((maximum, half, quarter)))
    candidate_pairs = [
        (prompt_size, branch_size)
        for prompt_size in base_prompt_sizes
        for branch_size in branch_sizes
    ]
    candidate_pairs.extend(
        [
            (max(1, round(maximum * 7 / 32)), 2),
            (max(1, round(maximum * 3 / 16)), 2),
            (max(1, maximum // 8), 4),
            (1, 1),
        ]
    )
    unique_pairs = []
    seen = set()
    for prompt_size, branch_size in candidate_pairs:
        pair = (
            min(maximum, prompt_size),
            min(max_branch_batch_size, branch_size),
        )
        if pair not in seen:
            seen.add(pair)
            unique_pairs.append(pair)
    return unique_pairs


def autotune_prefix_cache_training(
    *,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    student_prompt_texts: list[str],
    target_texts: list[str],
    target_token_ids: list[int] | None,
    use_fast_single_token_path: bool,
    digits_token_ids: list[int] | None,
    use_digit_chain_00_99_path: bool,
    teacher_probabilities: np.ndarray,
    exact_trie_prompt_plans: list[ExactTriePromptPlan] | None,
    device: Any,
    args: argparse.Namespace,
    output_path: Path,
) -> PrefixCacheTrainingConfig:
    """Select throughput only among numerically validated forward/backward pairs."""
    import torch

    logical_batch_size = min(
        args.per_device_train_batch_size,
        len(student_prompt_texts),
    )
    maximum_prompt_size = min(
        logical_batch_size,
        args.prefix_cache_train_microbatch_size or args.per_device_train_batch_size,
    )
    native_digit_chain = (
        getattr(args, "training_backend", "unsloth") == "transformers"
        and use_digit_chain_00_99_path
    )
    candidates = prefix_cache_autotune_candidates(
        max_prompt_microbatch_size=maximum_prompt_size,
        max_branch_batch_size=min(
            args.prefix_cache_autotune_max_branch_batch_size,
            10 if use_digit_chain_00_99_path else args.prefix_cache_autotune_max_branch_batch_size,
        ),
    )
    if native_digit_chain:
        # The H100 diagnostic validated B=3,K=10 for all native families.
        # A power-of-two-only grid can miss that useful memory/throughput knee.
        native_sizes = {maximum_prompt_size}
        native_size = 1
        while native_size <= maximum_prompt_size:
            native_sizes.add(native_size)
            native_size *= 2
        native_sizes.update(size for size in (3, 6, 12) if size <= maximum_prompt_size)
        native_branches = sorted({branch for _, branch in candidates})
        candidates = list(dict.fromkeys([
            *candidates,
            *((batch, branch) for batch in sorted(native_sizes)
              for branch in native_branches),
        ]))
        require_deterministic_native_digit_training(model)
        model.train()
    rng = random.Random(args.seed)
    benchmark_indices = rng.sample(
        range(len(student_prompt_texts)),
        logical_batch_size,
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    results = []
    successful = []
    oom_pairs: set[tuple[int, int]] = set()
    reference_scores = reference_gradients = None
    reference_duration = 0.0
    if native_digit_chain:
        logger.info("Validating native B/K candidates against full 11-probe gradients.")
        reference_scores, reference_gradients = digit_chain_training_batch_reference(
            model=model, tokenizer=tokenizer, optimizer=optimizer,
            prompt_texts=[student_prompt_texts[i] for i in benchmark_indices],
            target_texts=target_texts, digits_token_ids=digits_token_ids,
            teacher_probabilities=teacher_probabilities[benchmark_indices],
            temperature=args.student_temperature, device=device,
        )
    if args.exact_trie_prefix_cache:
        if exact_trie_prompt_plans is None:
            raise ValueError("Exact-trie autotuning requires prompt plans.")
        logger.info(
            "Building full-prefix score/gradient reference for all {} benchmark prompts.",
            logical_batch_size,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        reference_started = time.perf_counter()
        reference_scores, reference_gradients = exact_training_batch_reference(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            prompt_texts=[student_prompt_texts[i] for i in benchmark_indices],
            prompt_plans=[exact_trie_prompt_plans[i] for i in benchmark_indices],
            target_texts=target_texts,
            teacher_probabilities=teacher_probabilities[benchmark_indices],
            temperature=args.student_temperature,
            device=device,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        reference_duration = time.perf_counter() - reference_started
        if getattr(model, "_exact_trie_force_full_training", False):
            candidates = []

    logger.info(
        "Autotuning prefix-cache prompt/branch batching over {} candidate "
        "pairs using one {}-example forward/backward logical batch per pair.",
        len(candidates),
        logical_batch_size,
    )
    optimizer_state_reservation = None
    if native_digit_chain and torch.cuda.is_available():
        # AdamW moments are allocated only at the first real update. Include
        # their FP32 footprint during the pre-update memory/throughput search.
        # The reference-gradient list already accounts for the same-sized
        # rollback snapshot retained by a real microbatch transaction.
        optimizer_state_reservation = torch.empty(
            2 * sum(parameter.numel() for parameter in trainable_parameters),
            dtype=torch.float32, device=device,
        )
    for prompt_size, branch_size in candidates:
        if (prompt_size, branch_size) in oom_pairs:
            results.append(
                {
                    "prompt_microbatch_size": prompt_size,
                    "branch_batch_size": branch_size,
                    "status": "skipped_after_smaller_branch_oom",
                }
            )
            continue

        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        status = "completed"
        teacher_probs = None
        student_scores = None
        student_log_probs = None
        microbatch_loss = None
        scaled_loss = None
        score_batches = []
        correctness = None
        benchmark_duration = None
        try:
            for microbatch_start in range(0, logical_batch_size, prompt_size):
                microbatch_indices = benchmark_indices[
                    microbatch_start : microbatch_start + prompt_size
                ]
                prompts = [student_prompt_texts[index] for index in microbatch_indices]
                plans = (
                    [exact_trie_prompt_plans[index] for index in microbatch_indices]
                    if exact_trie_prompt_plans is not None
                    else None
                )
                teacher_probs = torch.tensor(
                    teacher_probabilities[microbatch_indices],
                    dtype=torch.float32,
                    device=device,
                )
                student_scores = candidate_logprobs(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_texts=prompts,
                    target_texts=target_texts,
                    target_token_ids=target_token_ids,
                    use_fast_single_token_path=use_fast_single_token_path,
                    digits_token_ids=digits_token_ids,
                    use_digit_chain_00_99_path=use_digit_chain_00_99_path,
                    device=device,
                    batch_size=len(microbatch_indices),
                    digit_chain_branch_batch_multiplier=(
                        args.digit_chain_branch_batch_multiplier
                    ),
                    digit_chain_max_branch_batch_size=(
                        args.digit_chain_max_branch_batch_size
                    ),
                    digit_chain_prefix_cache=args.digit_chain_prefix_cache,
                    exact_trie_prefix_cache=args.exact_trie_prefix_cache,
                    prefix_cache_branch_batch_size=branch_size,
                    detach_prefix_cache=False,
                    exact_trie_prompt_plans=plans,
                )
                student_log_probs = torch.log_softmax(
                    student_scores / args.student_temperature,
                    dim=-1,
                )
                microbatch_loss = (
                    -(teacher_probs * student_log_probs).sum(dim=-1).mean()
                )
                microbatch_loss = microbatch_loss * (args.student_temperature**2)
                scaled_loss = microbatch_loss * (
                    len(microbatch_indices) / logical_batch_size
                )
                scaled_loss.backward()
                if reference_scores is not None:
                    score_batches.append(student_scores.detach())
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            benchmark_duration = time.perf_counter() - started
            if reference_scores is not None:
                correctness = exact_training_pair_metrics(
                    reference_scores=reference_scores,
                    candidate_scores=torch.cat(score_batches),
                    reference_gradients=reference_gradients,
                    candidate_gradients=[p.grad for p in trainable_parameters],
                    reduced_precision=model_uses_reduced_precision(model),
                    device=device,
                    temperature=args.student_temperature,
                )
                if not correctness["accepted"]:
                    status = "rejected_inconsistent_scores_or_gradients"
            if args.max_grad_norm > 0 and status == "completed":
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    args.max_grad_norm,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except RuntimeError as exc:
            if not is_cuda_out_of_memory_error(exc):
                raise
            status = "cuda_oom"
            oom_pairs.add((prompt_size, branch_size))

        del (
            teacher_probs,
            student_scores,
            student_log_probs,
            microbatch_loss,
            scaled_loss,
            score_batches,
        )

        duration = (
            benchmark_duration
            if benchmark_duration is not None
            else time.perf_counter() - started
        )
        peak_allocated = (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available()
            else None
        )
        peak_reserved = (
            int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None
        )
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        result = {
            "prompt_microbatch_size": prompt_size,
            "branch_batch_size": branch_size,
            "logical_microbatch_sizes": logical_microbatch_sizes(
                logical_batch_size=logical_batch_size,
                prompt_microbatch_size=prompt_size,
            ),
            "status": status,
            "duration_seconds": duration,
            "examples_per_second": (
                logical_batch_size / duration if status == "completed" else None
            ),
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "correctness": correctness,
        }
        results.append(result)
        if status == "completed":
            successful.append(result)
            logger.info(
                "Prefix-cache autotune B={} K={}: {:.3f} examples/s, "
                "{:.2f} s/logical batch, peak allocated={:.2f} GiB.",
                prompt_size,
                branch_size,
                result["examples_per_second"],
                duration,
                (peak_allocated or 0) / (1024**3),
            )
        elif status == "cuda_oom":
            logger.warning(
                "Prefix-cache autotune B={} K={} OOM; rejecting pair.",
                prompt_size,
                branch_size,
            )
        else:
            logger.warning(
                "Rejected prefix-cache B={} K={} on correctness: {}",
                prompt_size,
                branch_size,
                correctness,
            )

    del optimizer_state_reservation
    if not successful and reference_scores is not None and not native_digit_chain:
        # The optimization is optional; the exact objective is not. The same
        # unbatched reference has already completed forward AND backward.
        fallback = {
            "prompt_microbatch_size": 1,
            "branch_batch_size": 1,
            "use_prefix_cache": False,
            "status": "full_prefix_fallback",
            "duration_seconds": reference_duration,
            "examples_per_second": logical_batch_size / reference_duration,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
            "correctness": {"accepted": True, "reference_computation": True},
        }
        successful.append(fallback)
        results.append(fallback)
        logger.warning(
            "No cached pair passed; using canonical full-prefix training. "
            "Rejected scores/gradients are not used and bounds are unchanged."
        )
    if not successful:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "version": 3,
                    "selected": None,
                    "candidates": results,
                    "validation_reference": "full_prefix_recomputation"
                    if reference_scores is not None
                    else None,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            "Prefix-cache autotuning found no viable prompt/branch batch pair."
        )
    best = max(successful, key=lambda item: item["examples_per_second"])
    payload = {
        "version": 3,
        "correctness_objective": "candidate_distribution_and_lora_gradient",
        "validation_reference": "full_prefix_recomputation"
        if reference_scores is not None
        else None,
        "device": str(device),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if torch.cuda.is_available() else None
        ),
        "logical_batch_size": logical_batch_size,
        "selected": best,
        "candidates": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.success(
        "Selected {} B={} K={} at {:.3f} examples/s; wrote {}.",
        "prefix-cache" if best.get("use_prefix_cache", True) else "full-prefix",
        best["prompt_microbatch_size"],
        best["branch_batch_size"],
        best["examples_per_second"],
        output_path,
    )
    return PrefixCacheTrainingConfig(
        prompt_microbatch_size=best["prompt_microbatch_size"],
        branch_batch_size=best["branch_batch_size"],
        duration_seconds=best["duration_seconds"],
        examples_per_second=best["examples_per_second"],
        peak_allocated_bytes=best["peak_allocated_bytes"],
        peak_reserved_bytes=best["peak_reserved_bytes"],
        use_prefix_cache=best.get("use_prefix_cache", True),
    )


def digit_chain_training_batch_reference(
    *, model: Any, tokenizer: Any, optimizer: Any,
    prompt_texts: list[str], target_texts: list[str], digits_token_ids: list[int],
    teacher_probabilities: np.ndarray, temperature: float, device: Any,
) -> tuple[Any, list[Any]]:
    """Audit the logical batch with one complete-prefix graph at a time."""
    optimizer.zero_grad(set_to_none=True)
    try:
        return streamed_digit_chain_full_prefix_gradients(
            model=model, tokenizer=tokenizer, prompt_texts=prompt_texts,
            target_texts=target_texts, digits_token_ids=digits_token_ids,
            teacher_probabilities=teacher_probabilities,
            temperature=temperature, device=device,
        )
    finally:
        optimizer.zero_grad(set_to_none=True)


def require_deterministic_native_digit_training(model: Any) -> None:
    """Require zero dropout before comparing independent train-mode forwards."""
    import torch

    for module in model.modules():
        if isinstance(module, torch.nn.Dropout) and module.p != 0:
            raise ValueError("Native digit gradient validation requires zero dropout.")
        config = getattr(module, "config", None)
        if config is not None and float(getattr(config, "attention_dropout", 0.0)):
            raise ValueError("Native digit gradient validation requires zero attention dropout.")


def streamed_digit_chain_full_prefix_gradients(
    *, model: Any, tokenizer: Any, prompt_texts: list[str],
    target_texts: list[str], digits_token_ids: list[int],
    teacher_probabilities: Any, temperature: float, device: Any,
) -> tuple[Any, list[Any]]:
    """Differentiate the exact two-digit loss without retaining 11 prefix graphs.

    For candidate score S_ij=a_i+b_ij and temperature-scaled cross-entropy,
    dL/dS = T*(softmax(S/T)*sum(target)-target)/N. An uncached no-grad pass
    computes that adjoint; independent complete-prefix forwards differentiate
    a and each b, releasing each graph before the next one. Every forward
    still differentiates through the entire prompt and all trainable layers.
    Candidate order/subsets and arbitrary positive temperatures are supported.
    Existing parameter gradients, parameters and model mode are unchanged.
    """
    import torch

    if not prompt_texts or not target_texts_are_plain_two_digit_numbers(target_texts):
        raise ValueError("A digit gradient reference needs prompts and two-digit targets.")
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("A digit gradient reference needs a positive finite temperature.")
    require_deterministic_native_digit_training(model)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("A digit gradient reference needs trainable parameters.")
    with torch.no_grad():
        scores = digit_chain_00_99_candidate_logprobs(
            model=model, tokenizer=tokenizer, prompt_texts=prompt_texts,
            target_texts=target_texts, digits_token_ids=digits_token_ids,
            device=device, batch_size=1, branch_batch_multiplier=1,
            max_branch_batch_size=1, use_prefix_cache=False,
        ).detach()
        targets = torch.as_tensor(teacher_probabilities, dtype=torch.float32, device=device)
        if targets.shape != scores.shape or not torch.isfinite(targets).all():
            raise ValueError("Digit gradient reference targets must match the finite score shape.")
        if not torch.isfinite(scores).all():
            raise PrefixCacheValidationError("Non-finite full-prefix digit gradient reference.")
        adjoints = temperature * (
            torch.softmax(scores.float() / temperature, -1) * targets.sum(-1, keepdim=True)
            - targets
        ) / len(prompt_texts)
        first = torch.tensor([int(target[0]) for target in target_texts], device=device)
        second = torch.tensor([int(target[1]) for target in target_texts], device=device)
        branch_adjoints = torch.zeros(len(prompt_texts), 100, device=device)
        branch_adjoints.index_add_(1, first * 10 + second, adjoints)
        branch_adjoints = branch_adjoints.reshape(len(prompt_texts), 10, 10)
    accumulated: list[Any] = [None] * len(parameters)
    for row, prompt in enumerate(prompt_texts):
        for branch in range(-1, 10):
            context = prompt if branch == -1 else f"{prompt}{branch}"
            weights = (
                branch_adjoints[row].sum(-1) if branch == -1
                else branch_adjoints[row, branch]
            )
            values = next_token_candidate_logprobs(
                model=model, tokenizer=tokenizer, prompt_texts=[context],
                target_token_ids=digits_token_ids, device=device, batch_size=1,
            )[0]
            gradients = torch.autograd.grad(
                (values * weights).sum(), parameters, allow_unused=True,
            )
            for index, gradient in enumerate(gradients):
                if gradient is None:
                    continue
                if not torch.isfinite(gradient).all():
                    raise PrefixCacheValidationError("Non-finite full-prefix digit gradients.")
                if accumulated[index] is None:
                    accumulated[index] = gradient.detach().clone()
                else:
                    accumulated[index].add_(gradient.detach())
            del values, gradients
    if not any(gradient is not None for gradient in accumulated):
        raise PrefixCacheValidationError("Full-prefix digit reference has no connected gradients.")
    return scores, accumulated


def checked_native_digit_training_microbatch(
    *, model: Any, teacher_probabilities: Any, temperature: float,
    update_example_count: int, audit: bool, score_kwargs: dict[str, Any],
) -> float:
    """Accumulate one native microbatch atomically, including backward failures.

    On OOM, restore the earlier logical-batch gradients before the caller
    retries the same examples at a smaller shape. Audits run after backward
    has released saved activations, so an uncached reference does not add to
    the peak live cached graph. Numerical rejection never reaches an update.
    """
    import torch

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    previous = [
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in parameters
    ]
    try:
        scores = candidate_logprobs(model=model, **score_kwargs)
        if not torch.isfinite(scores).all():
            raise PrefixCacheValidationError(
                "Non-finite native digit training scores; refusing optimizer update."
            )
        targets = torch.as_tensor(
            teacher_probabilities, dtype=torch.float32, device=scores.device,
        )
        loss_sum = -(targets * torch.log_softmax(scores / temperature, -1)).sum()
        loss_sum = loss_sum * temperature**2
        value = float(loss_sum.detach())
        detached_scores = scores.detach()
        (loss_sum / update_example_count).backward()
        del scores, loss_sum, targets
        if not all(
            torch.isfinite(parameter.grad).all()
            for parameter in parameters if parameter.grad is not None
        ):
            raise PrefixCacheValidationError(
                "Non-finite native digit gradients; refusing optimizer update."
            )
        if audit:
            metrics = native_digit_batch_audit(
                model=model, tokenizer=score_kwargs["tokenizer"],
                prompt_texts=score_kwargs["prompt_texts"],
                target_texts=score_kwargs["target_texts"],
                digits_token_ids=score_kwargs["digits_token_ids"],
                candidate_scores=detached_scores, temperature=temperature,
                device=score_kwargs["device"],
            )
            if not metrics["accepted"]:
                raise PrefixCacheValidationError(
                    "Native digit training score audit failed before optimizer "
                    f"update: {metrics}"
                )
        return value
    except BaseException:
        for parameter, gradient in zip(parameters, previous):
            parameter.grad = gradient
        raise


def checked_exact_training_microbatch(
    *,
    model: Any,
    tokenizer: Any,
    prompt_texts: list[str],
    target_texts: list[str],
    prompt_plans: list[ExactTriePromptPlan],
    teacher_probabilities: Any,
    temperature: float,
    update_example_count: int,
    branch_batch_size: int,
    use_prefix_cache: bool,
    audit: bool,
    device: Any,
    audit_gradients: bool = False,
) -> tuple[float, bool]:
    """Accumulate one exact loss transaction; roll back rejected cache gradients.

    The returned loss is the SUM of per-example losses before optimizer-update
    weighting. No optimizer/scheduler steps occur here. OOM during backward can
    partially modify .grad, so both forward and backward belong to the same
    transaction. Recovery recomputes full prefixes one prompt/context at a time.
    """
    import torch

    count = len(prompt_texts)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Training temperature must be positive and finite.")
    if not count or len(prompt_plans) != count:
        raise ValueError("Exact training requires non-empty, matching prompt plans.")
    accumulated_example_loss_weight(
        microbatch_size=count, optimizer_update_example_count=update_example_count
    )
    parameters = [p for p in model.parameters() if p.requires_grad]
    targets = torch.as_tensor(teacher_probabilities, dtype=torch.float32, device=device)
    if targets.shape != (count, len(target_texts)) or not torch.isfinite(targets).all():
        raise ValueError("Invalid exact-training teacher probabilities.")
    if (targets < 0).any() or not torch.allclose(
        targets.sum(-1), torch.ones(count, device=device), atol=1e-5, rtol=1e-5
    ):
        raise ValueError("Teacher probabilities must be nonnegative and sum to one.")

    def loss_sum(scores: Any, target: Any) -> Any:
        return (
            -(target * (scores / temperature).log_softmax(-1)).sum() * temperature**2
        )

    if use_prefix_cache:
        previous_gradients = [
            None if p.grad is None else p.grad.detach().clone() for p in parameters
        ]
        scores = loss = None
        try:
            scores = cached_exact_candidate_logprobs(
                model=model,
                tokenizer=tokenizer,
                prompt_texts=prompt_texts,
                target_texts=target_texts,
                prompt_plans=prompt_plans,
                device=device,
                batch_size=count,
                branch_batch_size=branch_batch_size,
                detach_prefix_cache=False,
            )
            if not torch.isfinite(scores).all():
                raise ExactCacheMismatch("Non-finite cached student scores.")
            loss = loss_sum(scores, targets)
            (loss / update_example_count).backward()
            if any(
                not torch.isfinite(p.grad).all()
                for p in parameters
                if p.grad is not None
            ):
                raise ExactCacheMismatch("Non-finite cached student gradients.")
            if audit_gradients:
                # Also validate fresh/reused/OOM-backoff shapes on the actual
                # trait targets. autograd.grad does not overwrite accumulated
                # .grad from preceding logical microbatches.
                (
                    reference_scores,
                    reference_gradients,
                ) = full_prefix_microbatch_gradients(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_texts=prompt_texts,
                    target_texts=target_texts,
                    prompt_plans=prompt_plans,
                    teacher_probabilities=targets,
                    temperature=temperature,
                    update_example_count=update_example_count,
                    device=device,
                )
                incremental_gradients = [
                    None
                    if p.grad is None
                    else p.grad.detach() - (prior if prior is not None else 0)
                    for p, prior in zip(parameters, previous_gradients)
                ]
                metrics = exact_training_pair_metrics(
                    reference_scores=reference_scores,
                    candidate_scores=scores.detach(),
                    reference_gradients=reference_gradients,
                    candidate_gradients=incremental_gradients,
                    reduced_precision=model_uses_reduced_precision(model),
                    device=device,
                    temperature=temperature,
                )
                del reference_scores, reference_gradients, incremental_gradients
                if not metrics["accepted"]:
                    raise ExactCacheMismatch(
                        f"Runtime score/gradient audit failed: {metrics}"
                    )
            elif audit:
                # Backward releases saved activations before the reference runs.
                with torch.no_grad(), qwen2_cache_eval_forward_mode(model):
                    validate_exact_teacher_batch(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_texts=prompt_texts,
                        target_texts=target_texts,
                        prompt_plans=prompt_plans,
                        scores=scores.detach(),
                        device=device,
                        temperature=temperature,
                    )
            return float(loss.detach().cpu()), True
        except RuntimeError as exc:
            # Restore earlier microbatches even if backward only partly ran.
            for parameter, previous in zip(parameters, previous_gradients):
                parameter.grad = previous
            if not isinstance(
                exc, ExactCacheMismatch
            ) and not is_cuda_out_of_memory_error(exc):
                raise
            logger.warning(
                "Rejected cached training B={} K={}: {}. Rolled back this "
                "microbatch; recomputing full prefixes before any optimizer step.",
                count,
                branch_batch_size,
                exc,
            )
            scores = loss = None
            exc.__traceback__ = None
        del previous_gradients
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_loss = 0.0
    for index, prompt in enumerate(prompt_texts):
        scores = exact_candidate_logprobs(
            model=model,
            tokenizer=tokenizer,
            prompt_texts=[prompt],
            target_texts=target_texts,
            prompt_plans=[prompt_plans[index]],
            device=device,
            batch_size=1,
        )
        if not torch.isfinite(scores).all():
            raise PrefixCacheValidationError(
                "Non-finite full-prefix student reference."
            )
        loss = loss_sum(scores, targets[index : index + 1])
        (loss / update_example_count).backward()
        total_loss += float(loss.detach().cpu())
        del scores, loss
    if any(not torch.isfinite(p.grad).all() for p in parameters if p.grad is not None):
        raise PrefixCacheValidationError("Non-finite full-prefix student gradients.")
    return total_loss, False


def train_student_on_teacher_distribution(
    *,
    trait: str,
    model: Any,
    tokenizer: Any,
    checkpoint_dir: Path,
    student_prompt_texts: list[str],
    target_texts: list[str],
    target_token_ids: list[int] | None,
    use_fast_single_token_path: bool,
    digits_token_ids: list[int] | None,
    use_digit_chain_00_99_path: bool,
    teacher_probabilities: np.ndarray,
    exact_trie_prompt_plans: list[ExactTriePromptPlan] | None,
    wandb_run: Any | None,
    device: Any,
    args: argparse.Namespace,
    preselected_prefix_cache_config: PrefixCacheTrainingConfig | None = None,
) -> tuple[list[dict[str, Any]], PrefixCacheTrainingConfig | None]:
    """Train the active LoRA adapter against cached teacher distributions."""
    import torch

    if len(student_prompt_texts) != teacher_probabilities.shape[0]:
        raise ValueError("Prompt count and teacher probability rows differ.")
    if exact_trie_prompt_plans is not None and len(exact_trie_prompt_plans) != len(
        student_prompt_texts
    ):
        raise ValueError("Prompt count and exact-trie plan count differ.")

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("No trainable parameters found after LoRA setup.")

    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr)
    total_steps = estimate_optimizer_steps(
        example_count=len(student_prompt_texts),
        n_epochs=args.n_epochs,
        batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    unavailable_checkpoint_steps = [
        step for step in args.checkpoint_steps if step >= total_steps
    ]
    if unavailable_checkpoint_steps:
        logger.warning(
            "Ignoring explicit checkpoint steps at or beyond final step {}: {}. "
            "The final adapter is saved separately.",
            total_steps,
            unavailable_checkpoint_steps,
        )
    explicit_checkpoint_steps = {
        step for step in args.checkpoint_steps if step < total_steps
    }
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        build_linear_lr_lambda(
            warmup_steps=args.warmup_steps,
            total_steps=total_steps,
        ),
    )
    microbatch_size = student_train_microbatch_size(
        logical_batch_size=args.per_device_train_batch_size,
        use_digit_chain_00_99_path=use_digit_chain_00_99_path,
        digit_chain_prefix_cache=args.digit_chain_prefix_cache,
        exact_trie_prefix_cache=args.exact_trie_prefix_cache,
        prefix_cache_microbatch_size=(args.prefix_cache_train_microbatch_size),
    )
    branch_batch_size = int(
        getattr(
            args,
            "prefix_cache_branch_batch_size",
            DEFAULT_PREFIX_CACHE_BRANCH_BATCH_SIZE,
        )
    )
    cache_training_active = (
        use_digit_chain_00_99_path and args.digit_chain_prefix_cache
    ) or args.exact_trie_prefix_cache
    active_cache_config = preselected_prefix_cache_config
    if (
        args.exact_trie_prefix_cache
        and active_cache_config is not None
        and not active_cache_config.use_prefix_cache
    ):
        # A target-specific rejection is not a hardware-wide speed result.
        # Allow the next trait to find its own valid cached configuration.
        active_cache_config = None
    if bool(getattr(args, "prefix_cache_autotune", False)):
        if not cache_training_active:
            raise ValueError(
                "--prefix-cache-autotune requires an active differentiable "
                "prefix-cache training path."
            )
        if active_cache_config is None:
            active_cache_config = autotune_prefix_cache_training(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                student_prompt_texts=student_prompt_texts,
                target_texts=target_texts,
                target_token_ids=target_token_ids,
                use_fast_single_token_path=use_fast_single_token_path,
                digits_token_ids=digits_token_ids,
                use_digit_chain_00_99_path=use_digit_chain_00_99_path,
                teacher_probabilities=teacher_probabilities,
                exact_trie_prompt_plans=exact_trie_prompt_plans,
                device=device,
                args=args,
                output_path=args.output_dir / "prefix_cache_autotune.json",
            )
        else:
            if (
                active_cache_config.prompt_microbatch_size > microbatch_size
                or active_cache_config.branch_batch_size
                > int(
                    getattr(
                        args,
                        "prefix_cache_autotune_max_branch_batch_size",
                        DEFAULT_PREFIX_CACHE_AUTOTUNE_MAX_BRANCH_BATCH_SIZE,
                    )
                )
            ):
                raise ValueError(
                    "A reused prefix-cache configuration exceeds the current "
                    "autotune bounds."
                )
            logger.info(
                "Reusing cell-level prefix-cache autotune result for {}: "
                "B={} K={}; the neutral student prompts and model architecture "
                "are identical across traits.",
                trait,
                active_cache_config.prompt_microbatch_size,
                active_cache_config.branch_batch_size,
            )
        microbatch_size = active_cache_config.prompt_microbatch_size
        branch_batch_size = active_cache_config.branch_batch_size
    exact_use_prefix_cache = (
        active_cache_config is None or active_cache_config.use_prefix_cache
    ) and not getattr(model, "_exact_trie_force_full_training", False)
    if args.exact_trie_prefix_cache and not exact_use_prefix_cache:
        microbatch_size = 1
    if microbatch_size < args.per_device_train_batch_size:
        logger.info(
            "Splitting logical training batches of {} into attached-cache "
            "microbatches of at most {}; the loss, effective batch size, and "
            "optimizer-step schedule are unchanged.",
            args.per_device_train_batch_size,
            microbatch_size,
        )
    microbatch_layout = logical_microbatch_sizes(
        logical_batch_size=args.per_device_train_batch_size,
        prompt_microbatch_size=microbatch_size,
    )
    logger.info(
        "Prefix-cache training configuration: logical batch={}, prompt "
        "microbatch B={}, branch group K={}, full logical-batch layout={}, cached={}",
        args.per_device_train_batch_size,
        microbatch_size,
        branch_batch_size,
        microbatch_layout,
        exact_use_prefix_cache
        if args.exact_trie_prefix_cache
        else cache_training_active,
    )

    indices = list(range(len(student_prompt_texts)))
    global_step = 0
    history: list[dict[str, Any]] = []
    audited_exact_pairs: set[tuple[int, int]] = set()
    audited_native_pairs: set[tuple[int, int]] = set()
    full_prefix_training_examples = 0
    model.train()
    for epoch_index in range(args.n_epochs):
        rng = random.Random(args.seed + epoch_index)
        rng.shuffle(indices)
        optimizer.zero_grad(set_to_none=True)
        accumulated_batches = 0
        update_loss_weighted_sum = 0.0
        update_example_count = 0

        epoch_batches = [
            indices[batch_start : batch_start + args.per_device_train_batch_size]
            for batch_start in range(
                0,
                len(indices),
                args.per_device_train_batch_size,
            )
        ]
        for batch_number, batch_indices in enumerate(epoch_batches):
            window_start = (
                batch_number // args.gradient_accumulation_steps
            ) * args.gradient_accumulation_steps
            window_end = min(
                window_start + args.gradient_accumulation_steps,
                len(epoch_batches),
            )
            update_target_example_count = sum(
                len(epoch_batches[index]) for index in range(window_start, window_end)
            )
            batch_loss_weighted_sum = 0.0
            microbatch_start = 0
            while microbatch_start < len(batch_indices):
                microbatch_indices = batch_indices[
                    microbatch_start : microbatch_start + microbatch_size
                ]
                microbatch_prompts = [
                    student_prompt_texts[index] for index in microbatch_indices
                ]
                microbatch_prompt_plans = (
                    [exact_trie_prompt_plans[index] for index in microbatch_indices]
                    if exact_trie_prompt_plans is not None
                    else None
                )
                if args.exact_trie_prefix_cache:
                    if microbatch_prompt_plans is None:
                        raise ValueError("Exact-trie training requires prompt plans.")
                    actual_pair = (len(microbatch_indices), branch_batch_size)
                    audit = actual_pair not in audited_exact_pairs or (
                        global_step % 100 == 0 and microbatch_start == 0
                    )
                    loss_sum, used_cache = checked_exact_training_microbatch(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_texts=microbatch_prompts,
                        target_texts=target_texts,
                        prompt_plans=microbatch_prompt_plans,
                        teacher_probabilities=teacher_probabilities[microbatch_indices],
                        temperature=args.student_temperature,
                        update_example_count=update_target_example_count,
                        branch_batch_size=branch_batch_size,
                        use_prefix_cache=exact_use_prefix_cache,
                        audit=audit,
                        device=device,
                        audit_gradients=actual_pair not in audited_exact_pairs,
                    )
                    if used_cache:
                        audited_exact_pairs.add(actual_pair)
                    else:
                        full_prefix_training_examples += len(microbatch_indices)
                        # Rejected attempts already recomputed exact gradients.
                        # Try a smaller cached shape for subsequent prompts;
                        # each new pair is audited before its first update.
                        if len(microbatch_indices) > 1:
                            microbatch_size = max(1, len(microbatch_indices) // 2)
                        elif branch_batch_size > 1:
                            branch_batch_size = max(1, branch_batch_size // 2)
                        else:
                            exact_use_prefix_cache = False
                    batch_loss_weighted_sum += loss_sum
                    microbatch_start += len(microbatch_indices)
                    continue
                if (
                    getattr(args, "training_backend", "unsloth") == "transformers"
                    and use_digit_chain_00_99_path
                ):
                    actual_pair = (len(microbatch_indices), branch_batch_size)
                    audit = actual_pair not in audited_native_pairs or (
                        microbatch_start == 0 and (
                            global_step % 100 == 0
                            or batch_number + 1 == len(epoch_batches)
                        )
                    )
                    try:
                        loss_sum = checked_native_digit_training_microbatch(
                            model=model,
                            teacher_probabilities=teacher_probabilities[microbatch_indices],
                            temperature=args.student_temperature,
                            update_example_count=update_target_example_count,
                            audit=audit,
                            score_kwargs=dict(
                                tokenizer=tokenizer, prompt_texts=microbatch_prompts,
                                target_texts=target_texts, target_token_ids=target_token_ids,
                                use_fast_single_token_path=use_fast_single_token_path,
                                digits_token_ids=digits_token_ids,
                                use_digit_chain_00_99_path=True, device=device,
                                batch_size=len(microbatch_indices),
                                digit_chain_branch_batch_multiplier=args.digit_chain_branch_batch_multiplier,
                                digit_chain_max_branch_batch_size=args.digit_chain_max_branch_batch_size,
                                digit_chain_prefix_cache=args.digit_chain_prefix_cache,
                                exact_trie_prefix_cache=False,
                                prefix_cache_branch_batch_size=branch_batch_size,
                                detach_prefix_cache=False,
                            ),
                        )
                    except RuntimeError as exc:
                        attempted_size = len(microbatch_indices)
                        if not (
                            is_cuda_out_of_memory_error(exc)
                            and args.digit_chain_prefix_cache
                            and (attempted_size > 1 or branch_batch_size > 1)
                        ):
                            optimizer.zero_grad(set_to_none=True)
                            raise
                        # The transaction restored earlier accumulated gradients,
                        # including after a partially executed failing backward.
                        if attempted_size > 1:
                            microbatch_size = max(1, attempted_size // 2)
                        else:
                            branch_batch_size = max(1, branch_batch_size // 2)
                        logger.warning(
                            "Native forward/backward OOM at B={} K={}; retrying "
                            "the same examples at B={} K={} with prior gradients restored.",
                            actual_pair[0], actual_pair[1], microbatch_size, branch_batch_size,
                        )
                    else:
                        audited_native_pairs.add(actual_pair)
                        batch_loss_weighted_sum += loss_sum
                        microbatch_start += len(microbatch_indices)
                        continue
                    # Leave the except block before collecting failed graph
                    # frames; the exception traceback itself can retain them.
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                teacher_probs = torch.tensor(
                    teacher_probabilities[microbatch_indices],
                    dtype=torch.float32,
                    device=device,
                )
                try:
                    student_scores = candidate_logprobs(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_texts=microbatch_prompts,
                        target_texts=target_texts,
                        target_token_ids=target_token_ids,
                        use_fast_single_token_path=use_fast_single_token_path,
                        digits_token_ids=digits_token_ids,
                        use_digit_chain_00_99_path=use_digit_chain_00_99_path,
                        device=device,
                        batch_size=microbatch_size,
                        digit_chain_branch_batch_multiplier=(
                            args.digit_chain_branch_batch_multiplier
                        ),
                        digit_chain_max_branch_batch_size=(
                            args.digit_chain_max_branch_batch_size
                        ),
                        digit_chain_prefix_cache=args.digit_chain_prefix_cache,
                        exact_trie_prefix_cache=args.exact_trie_prefix_cache,
                        prefix_cache_branch_batch_size=branch_batch_size,
                        detach_prefix_cache=False,
                        exact_trie_prompt_plans=microbatch_prompt_plans,
                    )
                except RuntimeError as exc:
                    attempted_size = len(microbatch_indices)
                    cache_unavailable = (
                        is_missing_prefix_cache_error(exc)
                        and use_digit_chain_00_99_path
                        and args.digit_chain_prefix_cache
                    )
                    if cache_unavailable:
                        logger.warning(
                            "The backend stopped returning past_key_values "
                            "during attached-cache training. Falling back to "
                            "the exact uncached 11-probe scorer and retrying "
                            "the same microbatch."
                        )
                        configure_uncached_digit_chain_fallback(
                            model=model,
                            args=args,
                        )
                        microbatch_size = args.per_device_train_batch_size
                        del teacher_probs
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    can_retry = (
                        is_cuda_out_of_memory_error(exc)
                        and (
                            (
                                use_digit_chain_00_99_path
                                and args.digit_chain_prefix_cache
                            )
                            or (
                                not use_digit_chain_00_99_path
                                and args.exact_trie_prefix_cache
                            )
                        )
                        and (attempted_size > 1 or branch_batch_size > 1)
                    )
                    if not can_retry:
                        raise
                    previous_branch_batch_size = branch_batch_size
                    if branch_batch_size > 1:
                        branch_batch_size = max(1, branch_batch_size // 2)
                    else:
                        microbatch_size = max(1, attempted_size // 2)
                    del teacher_probs
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.warning(
                        "CUDA OOM while scoring a prefix-cache microbatch "
                        "with B={} K={}; retrying at B={} K={}. The logical "
                        "batch and optimizer schedule remain unchanged.",
                        attempted_size,
                        previous_branch_batch_size,
                        microbatch_size,
                        branch_batch_size,
                    )
                    continue
                if (
                    getattr(args, "training_backend", "unsloth") == "transformers"
                    and use_digit_chain_00_99_path
                ):
                    # Check every row on every batch, including rows outside
                    # the more expensive sampled serial-reference audit.
                    if not torch.isfinite(student_scores).all():
                        optimizer.zero_grad(set_to_none=True)
                        raise PrefixCacheValidationError(
                            "Non-finite native digit training scores; refusing update."
                        )
                    actual_pair = (len(microbatch_indices), branch_batch_size)
                    audit = actual_pair not in audited_native_pairs or (
                        microbatch_start == 0 and (
                            global_step % 100 == 0
                            or batch_number + 1 == len(epoch_batches)
                        )
                    )
                    if audit:
                        metrics = native_digit_batch_audit(
                            model=model, tokenizer=tokenizer,
                            prompt_texts=microbatch_prompts,
                            target_texts=target_texts,
                            digits_token_ids=digits_token_ids,
                            candidate_scores=student_scores,
                            temperature=args.student_temperature, device=device,
                        )
                        if not metrics["accepted"]:
                            optimizer.zero_grad(set_to_none=True)
                            raise PrefixCacheValidationError(
                                "Native digit training score audit failed before "
                                f"optimizer update {global_step + 1}: {metrics}"
                            )
                        audited_native_pairs.add(actual_pair)
                if (
                    args.exact_trie_prefix_cache
                    and not torch.isfinite(student_scores).all()
                ):
                    raise PrefixCacheValidationError(
                        "Non-finite exact-trie training scores."
                    )
                student_log_probs = torch.log_softmax(
                    student_scores / args.student_temperature,
                    dim=-1,
                )
                microbatch_loss = (
                    -(teacher_probs * student_log_probs).sum(dim=-1).mean()
                )
                microbatch_loss = microbatch_loss * (args.student_temperature**2)
                update_example_weight = accumulated_example_loss_weight(
                    microbatch_size=len(microbatch_indices),
                    optimizer_update_example_count=update_target_example_count,
                )
                scaled_loss = microbatch_loss * update_example_weight
                scaled_loss.backward()
                batch_loss_weighted_sum += float(microbatch_loss.detach().cpu()) * len(
                    microbatch_indices
                )
                del (
                    teacher_probs,
                    student_scores,
                    student_log_probs,
                    microbatch_loss,
                    scaled_loss,
                )
                microbatch_start += len(microbatch_indices)
            accumulated_batches += 1
            update_loss_weighted_sum += batch_loss_weighted_sum
            update_example_count += len(batch_indices)

            should_step = (
                accumulated_batches % args.gradient_accumulation_steps == 0
                or batch_number + 1 == len(epoch_batches)
            )
            if should_step:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_parameters,
                        args.max_grad_norm,
                        error_if_nonfinite=True,
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                loss_value = update_loss_weighted_sum / update_example_count
                learning_rate = float(scheduler.get_last_lr()[0])
                history.append(
                    {
                        "trait": trait,
                        "epoch": epoch_index + 1,
                        "optimizer_step": global_step,
                        "total_steps": total_steps,
                        "progress": global_step / total_steps,
                        "loss": loss_value,
                        "learning_rate": learning_rate,
                        "examples_in_update": update_example_count,
                    }
                )
                update_loss_weighted_sum = 0.0
                update_example_count = 0
                should_log_step = args.log_every_steps > 0 and (
                    global_step == 1
                    or global_step % args.log_every_steps == 0
                    or global_step == total_steps
                )
                if should_log_step:
                    logger.info(
                        "epoch={} step={}/{} loss={:.6f}",
                        epoch_index + 1,
                        global_step,
                        total_steps,
                        loss_value,
                    )
                    log_wandb(
                        wandb_run,
                        {
                            "train/loss": loss_value,
                            f"train/{trait}/loss": loss_value,
                            "train/learning_rate": learning_rate,
                            "train/epoch": epoch_index + 1,
                            "train/optimizer_step": global_step,
                            "train/total_steps": total_steps,
                            "train/progress": global_step / total_steps,
                            "train/effective_batch_size": (
                                args.per_device_train_batch_size
                                * args.gradient_accumulation_steps
                            ),
                            "trait": trait,
                        },
                    )
                should_checkpoint = global_step in explicit_checkpoint_steps or (
                    args.checkpoint_every_steps > 0
                    and global_step % args.checkpoint_every_steps == 0
                    and global_step < total_steps
                )
                if should_checkpoint:
                    save_training_checkpoint(
                        model=model,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        checkpoint_root=checkpoint_dir,
                        epoch=epoch_index + 1,
                        global_step=global_step,
                        total_steps=total_steps,
                        loss=loss_value,
                        save_training_state=(not args.checkpoint_adapter_only),
                    )
    if global_step != total_steps:
        raise RuntimeError(
            f"Training produced {global_step} optimizer steps; expected {total_steps}."
        )
    final_cache_config = None
    if args.exact_trie_prefix_cache:
        logger.info(
            "Exact training for {} completed: {} full-prefix fallback examples, "
            "final B={} K={}, cached={}.",
            trait,
            full_prefix_training_examples,
            microbatch_size,
            branch_batch_size,
            exact_use_prefix_cache,
        )
    if cache_training_active:
        if (
            active_cache_config is not None
            and active_cache_config.prompt_microbatch_size == microbatch_size
            and active_cache_config.branch_batch_size == branch_batch_size
            and active_cache_config.use_prefix_cache == exact_use_prefix_cache
        ):
            final_cache_config = active_cache_config
        else:
            final_cache_config = PrefixCacheTrainingConfig(
                prompt_microbatch_size=microbatch_size,
                branch_batch_size=branch_batch_size,
                duration_seconds=0.0,
                examples_per_second=0.0,
                peak_allocated_bytes=None,
                peak_reserved_bytes=None,
                use_prefix_cache=exact_use_prefix_cache,
            )
    return history, final_cache_config


def clear_model(model: Any) -> None:
    """Release a model and clear CUDA cache when available."""
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        logger.debug("Skipping CUDA cache cleanup.")


def disable_gradient_checkpointing_for_prefix_cache(model: Any) -> None:
    """Keep Qwen3.5's differentiable prefix cache enabled during training."""
    disable = getattr(model, "gradient_checkpointing_disable", None)
    if callable(disable):
        disable()
    changed_modules = 0
    for module in model.modules():
        if getattr(module, "gradient_checkpointing", False):
            module.gradient_checkpointing = False
            changed_modules += 1
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = True
    still_enabled = [
        type(module).__name__
        for module in model.modules()
        if getattr(module, "gradient_checkpointing", False)
    ]
    if still_enabled:
        raise RuntimeError(
            "Could not disable gradient checkpointing for differentiable prefix "
            f"caching; still enabled on {still_enabled[:3]}."
        )
    logger.info(
        "Differentiable prefix caching enabled; gradient checkpointing "
        "disabled on {} active modules so Transformers preserves the cache.",
        changed_modules,
    )


def restore_gradient_checkpointing_after_prefix_cache_failure(model: Any) -> None:
    """Restore memory-saving training after rejecting the prefix-cache path."""
    enable = getattr(model, "gradient_checkpointing_enable", None)
    if not callable(enable):
        raise RuntimeError(
            "Cannot restore gradient checkpointing after prefix-cache validation "
            "failed; the model has no gradient_checkpointing_enable method."
        )
    enable()
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False
    logger.info(
        "Restored gradient checkpointing and disabled model cache for the "
        "uncached digit-chain training path."
    )


def configure_uncached_digit_chain_fallback(
    *,
    model: Any,
    args: argparse.Namespace,
) -> None:
    """Reject prefix caching for this and all remaining traits in the run."""
    if bool(getattr(args, "require_digit_chain_prefix_cache", False)):
        raise PrefixCacheValidationError(
            "Digit-chain prefix caching is required; refusing uncached fallback."
        )
    restore_gradient_checkpointing_after_prefix_cache_failure(model)
    args.digit_chain_prefix_cache = False


def largest_candidate_score_error_summary(
    *,
    reference_scores: Any,
    shared_scores: Any,
    target_texts: list[str],
    limit: int = 5,
) -> str:
    """Describe the candidate scores with the largest cache disagreement."""
    import torch

    if limit <= 0 or not target_texts:
        return ""
    absolute_errors = (shared_scores - reference_scores).abs()
    flat_errors = absolute_errors.reshape(-1)
    count = min(limit, int(flat_errors.numel()))
    _, flat_indices = torch.topk(flat_errors, k=count)
    candidate_count = len(target_texts)
    summaries = []
    for flat_index_tensor in flat_indices:
        flat_index = int(flat_index_tensor)
        prompt_index = flat_index // candidate_count
        candidate_index = flat_index % candidate_count
        reference_score = float(reference_scores[prompt_index, candidate_index])
        shared_score = float(shared_scores[prompt_index, candidate_index])
        summaries.append(
            f"prompt={prompt_index},target={target_texts[candidate_index]!r},"
            f"reference={reference_score:.6f},shared={shared_score:.6f},"
            f"delta={shared_score - reference_score:+.6f}"
        )
    return "; ".join(summaries)


def _cache_distribution_metrics(
    *,
    reference_scores: Any,
    cached_scores: Any,
) -> dict[str, float]:
    """Return score and normalized-distribution cache differences."""
    import torch

    reference_log_distribution = torch.log_softmax(reference_scores, dim=-1)
    cached_log_distribution = torch.log_softmax(cached_scores, dim=-1)
    reference_distribution = reference_log_distribution.exp()
    cached_distribution = cached_log_distribution.exp()
    return {
        "raw_score_max_error": float((cached_scores - reference_scores).abs().max()),
        "max_probability_error": float(
            (cached_distribution - reference_distribution).abs().max()
        ),
        "total_variation": float(
            0.5 * (cached_distribution - reference_distribution).abs().sum()
        ),
        "symmetric_kl": float(
            0.5
            * (
                (
                    reference_distribution
                    * (reference_log_distribution - cached_log_distribution)
                ).sum()
                + (
                    cached_distribution
                    * (cached_log_distribution - reference_log_distribution)
                ).sum()
            )
        ),
    }


def exact_trie_kernel_metadata(
    *, enabled: bool, dtype_name: str = "auto"
) -> dict[str, Any]:
    """Invalidate exact-trie teacher caches computed before kernel fixes."""
    if not enabled:
        return {}
    return {
        "exact_trie_qwen_kernel_version": EXACT_TRIE_QWEN_KERNEL_VERSION,
        "exact_trie_requested_dtype": dtype_name,
    }


def exact_score_pair_metrics(
    *,
    reference_scores: Any,
    candidate_scores: Any,
    reduced_precision: bool,
    temperature: float = 1.0,
) -> dict[str, Any]:
    """Check the worst-row drift in the normalized distillation objective.

    Candidate log scores are only consumed through ``softmax(scores / T)``
    for teacher targets and ``log_softmax(scores / T)`` for the student loss.
    Both are invariant to a prompt-specific additive offset. Batched and
    unbatched CUDA kernels can change that offset through otherwise harmless
    floating-point reduction-order differences, so absolute-score drift is a
    diagnostic rather than an acceptance condition. Probability, total
    variation, and symmetric-KL limits remain strict and are evaluated for the
    worst prompt row, not averaged across the benchmark batch.
    """
    import torch

    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Score validation temperature must be positive and finite.")
    if reference_scores.shape != candidate_scores.shape or reference_scores.ndim != 2:
        raise ValueError(
            "Exact-score comparisons require matching prompt × candidate matrices."
        )
    if not torch.isfinite(reference_scores).all():
        raise PrefixCacheValidationError(
            "The full-prefix reference produced non-finite scores."
        )
    if not torch.isfinite(candidate_scores).all():
        return {"accepted": False, "reason": "non_finite_candidate_scores"}
    signed_delta = candidate_scores.double() - reference_scores.double()
    row_shift = signed_delta.mean(dim=-1, keepdim=True)
    centered_delta = signed_delta - row_shift
    reference_log_probs = torch.log_softmax(
        reference_scores.double() / temperature,
        dim=-1,
    )
    candidate_log_probs = torch.log_softmax(
        candidate_scores.double() / temperature,
        dim=-1,
    )
    reference_probs = reference_log_probs.exp()
    candidate_probs = candidate_log_probs.exp()
    probability_delta = (reference_probs - candidate_probs).abs()
    raw = float(signed_delta.abs().max())
    row_shift_max = float(row_shift.abs().max())
    centered_raw = float(centered_delta.abs().max())
    probability = float(probability_delta.max())
    tv = float((0.5 * probability_delta.sum(-1)).max())
    symmetric_kl_per_row = 0.5 * (
        (
            reference_probs * (reference_log_probs - candidate_log_probs)
        ).sum(dim=-1)
        + (
            candidate_probs * (candidate_log_probs - reference_log_probs)
        ).sum(dim=-1)
    )
    symmetric_kl = float(symmetric_kl_per_row.max())
    # Preserve the former absolute-score limit in the report so old and new
    # diagnostics remain comparable. It is intentionally not an acceptance
    # gate because an arbitrary row offset cancels from the exact objective.
    raw_limit = 2e-2 if reduced_precision else 2e-4
    probability_limit = 5e-3 if reduced_precision else 5e-4
    tv_limit = 5e-3 if reduced_precision else 5e-4
    symmetric_kl_limit = (
        REDUCED_PRECISION_CACHE_MAX_SYMMETRIC_KL
        if reduced_precision
        else STANDARD_ATTENTION_CACHE_MAX_SYMMETRIC_KL
    )
    return {
        "raw_score_max_error": raw,
        "row_shift_max_abs": row_shift_max,
        "row_centered_score_max_error": centered_raw,
        "max_probability_error": probability,
        "max_total_variation": tv,
        "max_symmetric_kl": symmetric_kl,
        "raw_score_limit": raw_limit,
        "raw_score_limit_enforced": False,
        "probability_limit": probability_limit,
        "total_variation_limit": tv_limit,
        "symmetric_kl_limit": symmetric_kl_limit,
        "acceptance_basis": "candidate_distribution",
        "temperature": temperature,
        "accepted": probability <= probability_limit
        and tv <= tv_limit
        and symmetric_kl <= symmetric_kl_limit,
    }


def exact_training_pair_metrics(
    *,
    reference_scores: Any,
    candidate_scores: Any,
    reference_gradients: list[Any],
    candidate_gradients: list[Any],
    reduced_precision: bool,
    device: Any,
    temperature: float = 1.0,
) -> dict[str, Any]:
    """Reject a fast batch pair when its loss distribution or gradients differ."""
    metrics = exact_score_pair_metrics(
        reference_scores=reference_scores,
        candidate_scores=candidate_scores,
        reduced_precision=reduced_precision,
        temperature=temperature,
    )
    try:
        cosine, error = _cache_gradient_metrics(
            reference_gradients=reference_gradients,
            cached_gradients=candidate_gradients,
            device=device,
        )
    except PrefixCacheValidationError as exc:
        return {**metrics, "accepted": False, "gradient_error": str(exc)}
    cosine_limit = 0.99 if reduced_precision else 0.999
    error_limit = 0.15 if reduced_precision else 0.03
    metrics.update(gradient_cosine=cosine, gradient_relative_error=error)
    metrics["accepted"] = bool(
        metrics["accepted"] and cosine >= cosine_limit and error <= error_limit
    )
    return metrics


def exact_training_batch_reference(
    *,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    prompt_texts: list[str],
    prompt_plans: list[ExactTriePromptPlan],
    target_texts: list[str],
    teacher_probabilities: Any,
    temperature: float,
    device: Any,
) -> tuple[Any, list[Any]]:
    """Accumulate an independent full-prefix loss gradient without updating weights.

    One prompt and one continuation context per call bounds reference memory.
    Candidate batches later use the identical prompts, labels, and loss scaling.
    """
    import torch

    optimizer.zero_grad(set_to_none=True)
    score_rows = []
    try:
        with qwen2_cache_eval_forward_mode(model):
            for index, prompt in enumerate(prompt_texts):
                scores = exact_candidate_logprobs(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_texts=[prompt],
                    target_texts=target_texts,
                    prompt_plans=[prompt_plans[index]],
                    device=device,
                    batch_size=1,
                )
                if not torch.isfinite(scores).all():
                    raise PrefixCacheValidationError(
                        "Non-finite full-prefix reference scores."
                    )
                target = torch.as_tensor(
                    teacher_probabilities[index : index + 1],
                    dtype=torch.float32,
                    device=device,
                )
                loss = -(target * (scores / temperature).log_softmax(-1)).sum()
                (loss * temperature**2 / len(prompt_texts)).backward()
                score_rows.append(scores.detach())
        gradients = [
            None if p.grad is None else p.grad.detach().clone()
            for p in model.parameters()
            if p.requires_grad
        ]
        if not any(g is not None for g in gradients):
            raise PrefixCacheValidationError(
                "Full-prefix reference has no connected trainable gradient."
            )
        if not all(torch.isfinite(g).all() for g in gradients if g is not None):
            raise PrefixCacheValidationError(
                "Full-prefix reference gradients are non-finite."
            )
        return torch.cat(score_rows), gradients
    finally:
        optimizer.zero_grad(set_to_none=True)


def full_prefix_microbatch_gradients(
    *,
    model: Any,
    tokenizer: Any,
    prompt_texts: list[str],
    target_texts: list[str],
    prompt_plans: list[ExactTriePromptPlan],
    teacher_probabilities: Any,
    temperature: float,
    update_example_count: int,
    device: Any,
) -> tuple[Any, list[Any]]:
    """Compute reference gradients without touching the optimizer's .grad state."""
    import torch

    parameters = [p for p in model.parameters() if p.requires_grad]
    accumulated: list[Any] = [None] * len(parameters)
    rows = []
    for index, prompt in enumerate(prompt_texts):
        scores = exact_candidate_logprobs(
            model=model,
            tokenizer=tokenizer,
            prompt_texts=[prompt],
            target_texts=target_texts,
            prompt_plans=[prompt_plans[index]],
            device=device,
            batch_size=1,
        )
        if not torch.isfinite(scores).all():
            raise PrefixCacheValidationError(
                "Non-finite full-prefix gradient reference."
            )
        loss = (
            -(
                teacher_probabilities[index : index + 1]
                * (scores / temperature).log_softmax(-1)
            ).sum()
            * temperature**2
            / update_example_count
        )
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        for parameter_index, gradient in enumerate(gradients):
            if gradient is None:
                continue
            if not torch.isfinite(gradient).all():
                raise PrefixCacheValidationError(
                    "Non-finite full-prefix gradient reference."
                )
            if accumulated[parameter_index] is None:
                accumulated[parameter_index] = gradient.detach().clone()
            else:
                accumulated[parameter_index].add_(gradient.detach())
        rows.append(scores.detach())
        del gradients, scores, loss
    return torch.cat(rows), accumulated


def validate_exact_teacher_batch(
    *,
    model: Any,
    tokenizer: Any,
    prompt_texts: list[str],
    target_texts: list[str],
    prompt_plans: list[ExactTriePromptPlan] | None,
    scores: Any,
    device: Any,
    temperature: float = 1.0,
) -> None:
    """Audit actual teacher batch rows against uncached full-prefix scoring.

    Include the shortest and longest roots to exercise padding, plus both ends
    of the batch. An OOM retry or a final partial batch is checked in the same
    way; never write unchecked teacher probabilities after changing B or K.
    """
    if prompt_plans is None:
        raise ValueError("Teacher exact-trie checks require prompt plans.")
    lengths = [len(plan.root_context_token_ids) for plan in prompt_plans]
    indices = sorted(
        {
            0,
            len(prompt_texts) - 1,
            min(range(len(lengths)), key=lengths.__getitem__),
            max(range(len(lengths)), key=lengths.__getitem__),
        }
    )
    reference = exact_candidate_logprobs(
        model=model,
        tokenizer=tokenizer,
        prompt_texts=[prompt_texts[i] for i in indices],
        target_texts=target_texts,
        device=device,
        batch_size=1,
        prompt_plans=[prompt_plans[i] for i in indices],
    )
    metrics = exact_score_pair_metrics(
        reference_scores=reference,
        candidate_scores=scores[indices],
        reduced_precision=model_uses_reduced_precision(model),
        temperature=temperature,
    )
    if not metrics["accepted"]:
        raise ExactCacheMismatch(
            f"Teacher batch differs from full-prefix scores: {metrics}; "
            f"audited_rows={indices}, root_lengths={[lengths[i] for i in indices]}"
        )


def _cache_gradient_metrics(
    *,
    reference_gradients: list[Any],
    cached_gradients: list[Any],
    device: Any,
) -> tuple[float, float]:
    """Return cosine similarity and relative error for two gradient lists."""
    import torch

    if len(reference_gradients) != len(cached_gradients):
        raise PrefixCacheValidationError(
            "Cached and reference gradient lists have different lengths."
        )
    dot_product = torch.zeros((), dtype=torch.float64, device=device)
    reference_norm_squared = torch.zeros_like(dot_product)
    cached_norm_squared = torch.zeros_like(dot_product)
    difference_norm_squared = torch.zeros_like(dot_product)
    compared_gradients = 0
    for reference_gradient, cached_gradient in zip(
        reference_gradients,
        cached_gradients,
    ):
        if reference_gradient is None and cached_gradient is None:
            continue
        if reference_gradient is None or cached_gradient is None:
            raise PrefixCacheValidationError(
                "Cached and exact scorers reached different LoRA weights."
            )
        reference_float = reference_gradient.to(torch.float64)
        cached_float = cached_gradient.to(torch.float64)
        if (
            not torch.isfinite(reference_float).all()
            or not torch.isfinite(cached_float).all()
        ):
            raise PrefixCacheValidationError(
                "Cached or reference gradients contain non-finite values."
            )
        dot_product += (reference_float * cached_float).sum()
        reference_norm_squared += reference_float.square().sum()
        cached_norm_squared += cached_float.square().sum()
        difference_norm_squared += (reference_float - cached_float).square().sum()
        compared_gradients += 1
    if compared_gradients == 0:
        raise PrefixCacheValidationError(
            "Prefix-cache validation did not reach any LoRA weights."
        )
    if reference_norm_squared == 0:
        # A true zero gradient is valid (e.g. identical teacher/student targets).
        return (1.0, 0.0) if cached_norm_squared == 0 else (0.0, 1e30)
    cosine = dot_product / torch.sqrt(
        reference_norm_squared * cached_norm_squared
    ).clamp_min(1e-30)
    relative_error = torch.sqrt(difference_norm_squared) / torch.sqrt(
        reference_norm_squared
    ).clamp_min(1e-30)
    return float(cosine), float(relative_error)


def validate_exact_trie_prefix_cache_scores_once(
    *,
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    target_texts: list[str],
    prompt_plan: ExactTriePromptPlan,
    device: Any,
) -> None:
    """Require cached standard-attention trie scores to match full scoring."""
    import torch

    if not uses_standard_attention_prefix_cache(model):
        raise PrefixCacheValidationError(
            "Exact-trie caching was requested for a non-standard-attention model."
        )
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        with torch.no_grad():
            reference_scores = exact_candidate_logprobs(
                model=model,
                tokenizer=tokenizer,
                prompt_texts=[prompt_text],
                target_texts=target_texts,
                device=device,
                batch_size=1,
                prompt_plans=[prompt_plan],
            )
            cached_scores = cached_exact_candidate_logprobs(
                model=model,
                tokenizer=tokenizer,
                prompt_texts=[prompt_text],
                target_texts=target_texts,
                device=device,
                batch_size=1,
                detach_prefix_cache=True,
                prompt_plans=[prompt_plan],
            )
        metrics = _cache_distribution_metrics(
            reference_scores=reference_scores,
            cached_scores=cached_scores,
        )
        if not torch.isfinite(reference_scores).all():
            raise PrefixCacheValidationError(
                "Non-finite full-prefix base-model reference."
            )
        accepted = standard_attention_cache_metrics_within_tolerance(
            max_probability_error=metrics["max_probability_error"],
            total_variation=metrics["total_variation"],
            symmetric_kl=metrics["symmetric_kl"],
            gradient_cosine=1.0,
            gradient_relative_error=0.0,
        )
        accepted = (
            accepted
            and exact_score_pair_metrics(
                reference_scores=reference_scores,
                candidate_scores=cached_scores,
                reduced_precision=model_uses_reduced_precision(model),
            )["accepted"]
        )
        if not accepted:
            largest_errors = largest_candidate_score_error_summary(
                reference_scores=reference_scores,
                shared_scores=cached_scores,
                target_texts=target_texts,
            )
            raise ExactCacheMismatch(
                "Cached exact-trie base-model scores failed validation: "
                f"metrics={metrics}; largest_candidate_deltas="
                f"[{largest_errors}]."
            )
        logger.success(
            "Validated exact-trie base-model KV reuse: raw score max "
            "error={:.6f}, distribution TV={:.6f}, symmetric KL={:.6f}.",
            metrics["raw_score_max_error"],
            metrics["total_variation"],
            metrics["symmetric_kl"],
        )
    finally:
        if was_training:
            model.train()


def validate_attached_exact_trie_prefix_cache_once(
    *,
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    target_texts: list[str],
    prompt_plan: ExactTriePromptPlan,
    device: Any,
) -> None:
    """Require cached exact-trie LoRA scores and gradients to match reference."""
    import torch

    if model in _VALIDATED_EXACT_TRIE_PREFIX_CACHE_MODEL_TYPES:
        return
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("Cannot validate prefix gradients without trainable weights.")
    was_training = bool(getattr(model, "training", False))
    model.eval()

    def score_and_grad(*, cached: bool) -> tuple[Any, list[Any]]:
        scorer = cached_exact_candidate_logprobs if cached else exact_candidate_logprobs
        kwargs = {
            "model": model,
            "tokenizer": tokenizer,
            "prompt_texts": [prompt_text],
            "target_texts": target_texts,
            "device": device,
            "batch_size": 1,
        }
        if cached:
            scores = scorer(
                **kwargs,
                detach_prefix_cache=False,
                prompt_plans=[prompt_plan],
            )
        else:
            scores = scorer(**kwargs, prompt_plans=[prompt_plan])
        diagnostic_target = torch.zeros_like(scores)
        selected_indices = sorted({0, scores.shape[1] // 2, scores.shape[1] - 1})
        selected_weights = torch.tensor(
            [0.5, 0.3, 0.2][: len(selected_indices)],
            dtype=diagnostic_target.dtype,
            device=device,
        )
        selected_weights = selected_weights / selected_weights.sum()
        diagnostic_target[:, selected_indices] = selected_weights
        diagnostic_loss = (
            -(diagnostic_target * torch.log_softmax(scores, dim=-1)).sum(dim=-1).mean()
        )
        gradients = torch.autograd.grad(
            diagnostic_loss,
            trainable_parameters,
            allow_unused=True,
        )
        return scores.detach(), [
            None if gradient is None else gradient.detach() for gradient in gradients
        ]

    try:
        reference_scores, reference_gradients = score_and_grad(cached=False)
        cached_scores, cached_gradients = score_and_grad(cached=True)
        metrics = _cache_distribution_metrics(
            reference_scores=reference_scores,
            cached_scores=cached_scores,
        )
        gradient_cosine, gradient_relative_error = _cache_gradient_metrics(
            reference_gradients=reference_gradients,
            cached_gradients=cached_gradients,
            device=device,
        )
        accepted = standard_attention_cache_metrics_within_tolerance(
            max_probability_error=metrics["max_probability_error"],
            total_variation=metrics["total_variation"],
            symmetric_kl=metrics["symmetric_kl"],
            gradient_cosine=gradient_cosine,
            gradient_relative_error=gradient_relative_error,
        )
        reduced_precision_accepted = False
        if not accepted and model_uses_reduced_precision(model):
            reduced_precision_accepted = (
                reduced_precision_cache_metrics_within_tolerance(
                    max_probability_error=metrics["max_probability_error"],
                    total_variation=metrics["total_variation"],
                    symmetric_kl=metrics["symmetric_kl"],
                    gradient_cosine=gradient_cosine,
                    gradient_relative_error=gradient_relative_error,
                )
            )
            accepted = reduced_precision_accepted
        accepted = (
            accepted
            and exact_score_pair_metrics(
                reference_scores=reference_scores,
                candidate_scores=cached_scores,
                reduced_precision=model_uses_reduced_precision(model),
            )["accepted"]
        )
        if not accepted:
            largest_errors = largest_candidate_score_error_summary(
                reference_scores=reference_scores,
                shared_scores=cached_scores,
                target_texts=target_texts,
            )
            raise ExactCacheMismatch(
                "Cached exact-trie LoRA validation failed: "
                f"metrics={metrics}, gradient_cosine={gradient_cosine:.6f}, "
                f"gradient_relative_error={gradient_relative_error:.6f}; "
                f"largest_candidate_deltas=[{largest_errors}]."
            )
        if reduced_precision_accepted:
            logger.warning(
                "Accepted mathematically exact KV reuse within reduced-precision "
                "kernel tolerances: max probability error={:.6f}, TV={:.6f}, "
                "symmetric KL={:.6f}; gradient cosine={:.6f}, relative "
                "error={:.6f}. Full-sequence and cached BF16/FP16 attention "
                "use different accumulation shapes.",
                metrics["max_probability_error"],
                metrics["total_variation"],
                metrics["symmetric_kl"],
                gradient_cosine,
                gradient_relative_error,
            )
        _VALIDATED_EXACT_TRIE_PREFIX_CACHE_MODEL_TYPES.add(model)
        logger.success(
            "Validated attached exact-trie KV reuse against full recomputation: "
            "TV={:.6f}, gradient cosine={:.6f}, gradient relative error={:.6f}.",
            metrics["total_variation"],
            gradient_cosine,
            gradient_relative_error,
        )
    except RuntimeError as exc:
        if is_inplace_autograd_runtime_error(exc):
            raise PrefixCacheValidationError(
                "The exact-trie cache backend mutated a tensor saved for LoRA "
                "autograd."
            ) from exc
        raise
    finally:
        if was_training:
            model.train()


def validate_prefix_cache_scores_once(
    *,
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    target_texts: list[str],
    digits_token_ids: list[int],
    device: Any,
) -> None:
    """Validate cached base-model scores before computing full distributions."""
    import torch

    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        logger.info(
            "Validating one shared Qwen prompt cache against ten independently "
            "recomputed prompt caches with identical branch kernels."
        )
        with torch.no_grad():
            uncached_scores = digit_chain_00_99_candidate_logprobs(
                model=model,
                tokenizer=tokenizer,
                prompt_texts=[prompt_text],
                target_texts=target_texts,
                digits_token_ids=digits_token_ids,
                device=device,
                batch_size=1,
                # The cached implementation makes ten separate B=1 branch
                # forwards. Keep the uncached reference at B=1 as well: a
                # fused B=10 LM-head/attention call can select different BF16
                # kernels and confound cache error with batch-shape drift.
                branch_batch_multiplier=1,
                max_branch_batch_size=1,
                use_prefix_cache=False,
                detach_prefix_cache=True,
            )
            cached_scores = digit_chain_00_99_candidate_logprobs(
                model=model,
                tokenizer=tokenizer,
                prompt_texts=[prompt_text],
                target_texts=target_texts,
                digits_token_ids=digits_token_ids,
                device=device,
                batch_size=1,
                use_prefix_cache=True,
                detach_prefix_cache=True,
            )
            independent_prefix_scores = cached_digit_chain_00_99_candidate_logprobs(
                model=model,
                tokenizer=tokenizer,
                prompt_texts=[prompt_text],
                target_texts=target_texts,
                digits_token_ids=digits_token_ids,
                device=device,
                batch_size=1,
                detach_prefix_cache=True,
                share_prefix_cache=False,
            )
            fused_uncached_scores = digit_chain_00_99_candidate_logprobs(
                model=model,
                tokenizer=tokenizer,
                prompt_texts=[prompt_text],
                target_texts=target_texts,
                digits_token_ids=digits_token_ids,
                device=device,
                batch_size=1,
                branch_batch_multiplier=10,
                max_branch_batch_size=10,
                use_prefix_cache=False,
                detach_prefix_cache=True,
            )
        fused_uncached_distribution = torch.softmax(
            fused_uncached_scores,
            dim=-1,
        )
        separate_uncached_distribution = torch.softmax(
            uncached_scores,
            dim=-1,
        )
        fused_raw_score_max_error = float(
            (fused_uncached_scores - uncached_scores).abs().max()
        )
        fused_total_variation = float(
            0.5
            * (fused_uncached_distribution - separate_uncached_distribution).abs().sum()
        )
        logger.info(
            "Uncached BF16 batch-shape diagnostic (fused B=10 branches versus "
            "ten separate B=1 branches): raw score max error={:.6f}, "
            "distribution TV={:.6f}. This diagnostic does not determine cache "
            "acceptance.",
            fused_raw_score_max_error,
            fused_total_variation,
        )
        full_recompute_raw_score_max_error = float(
            (cached_scores - uncached_scores).abs().max()
        )
        full_recompute_total_variation = float(
            0.5
            * (torch.softmax(cached_scores, dim=-1) - separate_uncached_distribution)
            .abs()
            .sum()
        )
        logger.info(
            "Full-sequence BF16 versus cached-generation diagnostic at B=1: "
            "raw score max error={:.6f}, distribution TV={:.6f}. Standard "
            "attention is algebraically cache-exact, but different GPU kernel "
            "shapes need not be bitwise identical. This diagnostic does not "
            "determine shared-cache acceptance.",
            full_recompute_raw_score_max_error,
            full_recompute_total_variation,
        )
        uncached_log_distribution = torch.log_softmax(
            independent_prefix_scores,
            dim=-1,
        )
        cached_log_distribution = torch.log_softmax(cached_scores, dim=-1)
        uncached_distribution = uncached_log_distribution.exp()
        cached_distribution = cached_log_distribution.exp()
        raw_score_max_error = float(
            (cached_scores - independent_prefix_scores).abs().max()
        )
        max_probability_error = float(
            (cached_distribution - uncached_distribution).abs().max()
        )
        total_variation = float(
            0.5 * (cached_distribution - uncached_distribution).abs().sum()
        )
        symmetric_kl = float(
            0.5
            * (
                (
                    uncached_distribution
                    * (uncached_log_distribution - cached_log_distribution)
                ).sum()
                + (
                    cached_distribution
                    * (cached_log_distribution - uncached_log_distribution)
                ).sum()
            )
        )
        if uses_standard_attention_prefix_cache(model):
            accepted = standard_attention_cache_metrics_within_tolerance(
                max_probability_error=max_probability_error,
                total_variation=total_variation,
                symmetric_kl=symmetric_kl,
                gradient_cosine=1.0,
                gradient_relative_error=0.0,
            )
        else:
            accepted = (
                prefix_cache_distribution_status(
                    max_probability_error=max_probability_error,
                    total_variation=total_variation,
                    symmetric_kl=symmetric_kl,
                )
                != "failed"
            )
        if not accepted:
            largest_errors = largest_candidate_score_error_summary(
                reference_scores=independent_prefix_scores,
                shared_scores=cached_scores,
                target_texts=target_texts,
            )
            raise PrefixCacheValidationError(
                "Shared base-model prompt cache disagreed with independently "
                "recomputed prompt caches before teacher distribution "
                "computation: raw_score_max_error="
                f"{raw_score_max_error:.6f}, max_probability_error="
                f"{max_probability_error:.6f}, total_variation="
                f"{total_variation:.6f}, symmetric_kl={symmetric_kl:.6f}; "
                f"largest_candidate_deltas=[{largest_errors}]."
            )
        logger.success(
            "Validated shared base-model prompt cache against independent "
            "prompt-cache recomputation: raw score max error={:.6f}, "
            "distribution TV={:.6f}, symmetric KL={:.6f}.",
            raw_score_max_error,
            total_variation,
            symmetric_kl,
        )
    finally:
        if was_training:
            model.train()


def validate_attached_prefix_cache_once(
    *,
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    target_texts: list[str],
    digits_token_ids: list[int],
    device: Any,
    allow_approximate: bool = False,
) -> None:
    """Compare shared and independently recomputed cache scores/gradients."""
    import torch

    if model in _VALIDATED_PREFIX_CACHE_MODEL_TYPES:
        return
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("Cannot validate prefix gradients without trainable weights.")

    was_training = bool(getattr(model, "training", False))
    standard_attention_cache = uses_standard_attention_prefix_cache(model)
    native_digit_chain = uses_native_standard_attention(model)
    if native_digit_chain:
        require_deterministic_native_digit_training(model)
        model.train()
    else:
        model.eval()

    def score_and_grad(*, share_prefix_cache: bool) -> tuple[Any, list[Any]]:
        if native_digit_chain and not share_prefix_cache:
            diagnostic_target = torch.zeros((1, len(target_texts)), device=device)
            diagnostic_target[:, 0] = 0.5
            diagnostic_target[:, 42] = 0.3
            diagnostic_target[:, 99] = 0.2
            return streamed_digit_chain_full_prefix_gradients(
                model=model, tokenizer=tokenizer, prompt_texts=[prompt_text],
                target_texts=target_texts, digits_token_ids=digits_token_ids,
                teacher_probabilities=diagnostic_target, temperature=1.0,
                device=device,
            )
        scores = cached_digit_chain_00_99_candidate_logprobs(
            model=model,
            tokenizer=tokenizer,
            prompt_texts=[prompt_text],
            target_texts=target_texts,
            digits_token_ids=digits_token_ids,
            device=device,
            batch_size=1,
            detach_prefix_cache=False,
            share_prefix_cache=share_prefix_cache,
        )
        diagnostic_target = torch.zeros_like(scores)
        diagnostic_target[:, 0] = 0.5
        diagnostic_target[:, 42] = 0.3
        diagnostic_target[:, 99] = 0.2
        diagnostic_loss = (
            -(diagnostic_target * torch.log_softmax(scores, dim=-1)).sum(dim=-1).mean()
        )
        gradients = torch.autograd.grad(
            diagnostic_loss,
            trainable_parameters,
            allow_unused=True,
        )
        return scores.detach(), [
            None if gradient is None else gradient.detach() for gradient in gradients
        ]

    try:
        uncached_scores, uncached_gradients = score_and_grad(share_prefix_cache=False)
        cached_scores, cached_gradients = score_and_grad(share_prefix_cache=True)
        uncached_log_distribution = torch.log_softmax(
            uncached_scores,
            dim=-1,
        )
        cached_log_distribution = torch.log_softmax(cached_scores, dim=-1)
        uncached_distribution = uncached_log_distribution.exp()
        cached_distribution = cached_log_distribution.exp()
        raw_score_max_error = float((cached_scores - uncached_scores).abs().max())
        max_probability_error = float(
            (cached_distribution - uncached_distribution).abs().max()
        )
        total_variation = float(
            0.5 * (cached_distribution - uncached_distribution).abs().sum()
        )
        symmetric_kl = float(
            0.5
            * (
                (
                    uncached_distribution
                    * (uncached_log_distribution - cached_log_distribution)
                ).sum()
                + (
                    cached_distribution
                    * (cached_log_distribution - uncached_log_distribution)
                ).sum()
            )
        )
        distribution_status = prefix_cache_distribution_status(
            max_probability_error=max_probability_error,
            total_variation=total_variation,
            symmetric_kl=symmetric_kl,
        )
        dot_product = torch.zeros((), dtype=torch.float64, device=device)
        uncached_norm_squared = torch.zeros_like(dot_product)
        cached_norm_squared = torch.zeros_like(dot_product)
        difference_norm_squared = torch.zeros_like(dot_product)
        compared_gradients = 0
        for uncached_gradient, cached_gradient in zip(
            uncached_gradients,
            cached_gradients,
        ):
            if uncached_gradient is None and cached_gradient is None:
                continue
            if uncached_gradient is None or cached_gradient is None:
                raise PrefixCacheValidationError(
                    "Shared and independently recomputed caches reached "
                    "different LoRA weights."
                )
            uncached_float = uncached_gradient.to(torch.float64)
            cached_float = cached_gradient.to(torch.float64)
            dot_product += (uncached_float * cached_float).sum()
            uncached_norm_squared += uncached_float.square().sum()
            cached_norm_squared += cached_float.square().sum()
            difference_norm_squared += (uncached_float - cached_float).square().sum()
            compared_gradients += 1
        if compared_gradients == 0 or uncached_norm_squared <= 0:
            raise PrefixCacheValidationError(
                "Prefix-cache validation did not produce a non-zero LoRA gradient."
            )
        cosine = dot_product / torch.sqrt(
            uncached_norm_squared * cached_norm_squared
        ).clamp_min(1e-30)
        relative_error = torch.sqrt(difference_norm_squared) / torch.sqrt(
            uncached_norm_squared
        ).clamp_min(1e-30)
        gradient_cosine = float(cosine)
        gradient_relative_error = float(relative_error)
        if standard_attention_cache:
            if not standard_attention_cache_metrics_within_tolerance(
                max_probability_error=max_probability_error,
                total_variation=total_variation,
                symmetric_kl=symmetric_kl,
                gradient_cosine=gradient_cosine,
                gradient_relative_error=gradient_relative_error,
            ):
                largest_errors = largest_candidate_score_error_summary(
                    reference_scores=uncached_scores,
                    shared_scores=cached_scores,
                    target_texts=target_texts,
                )
                raise PrefixCacheValidationError(
                    "Shared standard-attention prompt cache disagreed with "
                    "independent prompt-cache recomputation: "
                    "raw_score_max_error="
                    f"{raw_score_max_error:.6f}, max_probability_error="
                    f"{max_probability_error:.6f}, total_variation="
                    f"{total_variation:.6f}, symmetric_kl={symmetric_kl:.6f}; "
                    f"gradient_cosine={gradient_cosine:.6f}, "
                    "gradient_relative_error="
                    f"{gradient_relative_error:.6f}; largest_candidate_deltas="
                    f"[{largest_errors}]."
                )
            distribution_status = "preferred"
        if not standard_attention_cache and (
            gradient_cosine < PREFIX_CACHE_MIN_GRADIENT_COSINE
            or gradient_relative_error > PREFIX_CACHE_MAX_GRADIENT_RELATIVE_ERROR
        ):
            raise PrefixCacheValidationError(
                "Differentiable prefix-cache gradient check failed: "
                f"cosine={gradient_cosine:.6f}, relative_error="
                f"{gradient_relative_error:.6f}."
            )
        approximation_accepted = False
        if distribution_status == "failed":
            approximation_accepted = allow_approximate and (
                bounded_prefix_cache_approximation(
                    max_probability_error=max_probability_error,
                    total_variation=total_variation,
                    symmetric_kl=symmetric_kl,
                    gradient_cosine=gradient_cosine,
                    gradient_relative_error=gradient_relative_error,
                )
            )
            if not approximation_accepted:
                raise PrefixCacheValidationError(
                    "Differentiable prefix-cache distribution check failed: "
                    f"raw_score_max_error={raw_score_max_error:.6f}, "
                    f"max_probability_error={max_probability_error:.6f}, "
                    f"total_variation={total_variation:.6f}, "
                    f"symmetric_kl={symmetric_kl:.6f}; gradient_cosine="
                    f"{gradient_cosine:.6f}, gradient_relative_error="
                    f"{gradient_relative_error:.6f}."
                )
            logger.warning(
                "OPT-IN APPROXIMATE PREFIX CACHE ACCEPTED: max probability "
                "error={:.6f}, TV={:.6f}, symmetric KL={:.6f}; gradient "
                "cosine={:.6f}, relative error={:.6f}. This is faster but is "
                "not numerically identical to independent prompt-cache "
                "recomputation.",
                max_probability_error,
                total_variation,
                symmetric_kl,
                gradient_cosine,
                gradient_relative_error,
            )
        elif distribution_status == "conditional":
            logger.warning(
                "Prefix-cache agreement is outside the preferred per-candidate "
                "probability tolerance but inside all hard distribution limits; "
                "the LoRA-gradient check passed, so training may proceed. "
                "max probability error={:.6f} (preferred <= {:.3f}, hard <= "
                "{:.3f}), TV={:.6f}, symmetric KL={:.6f}; gradient cosine="
                "{:.6f}, relative error={:.6f}.",
                max_probability_error,
                PREFIX_CACHE_PREFERRED_MAX_PROBABILITY_ERROR,
                PREFIX_CACHE_HARD_MAX_PROBABILITY_ERROR,
                total_variation,
                symmetric_kl,
                gradient_cosine,
                gradient_relative_error,
            )
        _VALIDATED_PREFIX_CACHE_MODEL_TYPES.add(model)
        logger.success(
            "Validated attached Qwen prefix cache ({}): raw score max error={:.6f}, "
            "distribution TV={:.6f}, symmetric KL={:.6f}; LoRA gradient "
            "cosine={:.6f}, relative error={:.6f}.",
            (
                "standard-attention exact KV reuse"
                if standard_attention_cache
                else (
                    "bounded approximate" if approximation_accepted else "strict hybrid"
                )
            ),
            raw_score_max_error,
            total_variation,
            symmetric_kl,
            gradient_cosine,
            gradient_relative_error,
        )
    except RuntimeError as exc:
        if not is_inplace_autograd_runtime_error(exc):
            raise
        raise PrefixCacheValidationError(
            "Differentiable prefix-cache validation encountered a backend "
            "kernel that mutated an activation saved for LoRA autograd. The "
            "shared-cache optimization is unavailable for this installed "
            f"model stack: {exc}"
        ) from exc
    finally:
        model.train(was_training)


def run_probability_ratio_evaluation(
    *,
    output_dir: Path,
    traits: list[str],
    args: argparse.Namespace,
    output_stem: str,
) -> dict[str, Any]:
    """Evaluate logits-distilled students and write a favorite-color heatmap."""
    output_path = output_dir / f"{output_stem}.json"
    config = RatioMatrixConfig(
        colors=args.eval_colors,
        fine_tune_traits=traits,
        questions=args.eval_questions,
        model_json_template=model_json_template(output_dir),
        base_model_id=args.model_id,
        output_path=output_path,
        csv_output_path=output_dir / f"{output_stem}.csv",
        heatmap_path=output_dir / f"{output_stem}.png",
        probability_mode=args.probability_mode,
        epsilon=args.epsilon,
        candidate_prefix=args.eval_candidate_prefix,
        assistant_prefill=args.eval_assistant_prefill,
        batch_size=args.eval_batch_size,
        device=args.device,
        torch_dtype=args.torch_dtype,
        enable_thinking=args.enable_thinking,
    )
    validate_ratio_matrix_config(config)
    result = run_probability_ratio_matrix(config)
    write_ratio_json(config.output_path, result)
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
    return result


def native_digit_training_identity(args: argparse.Namespace) -> dict[str, Any]:
    """Identify the native scoring policy that produced a completed adapter."""
    from scripts.native_generation_loader import replication_revision_kwargs

    return {
        "model_id": args.model_id,
        "model_revision": replication_revision_kwargs(args.model_id).get("revision"),
        "training_backend": "transformers",
        "native_digit_scoring_version": NATIVE_DIGIT_SCORING_VERSION,
        "native_position_policy": "logical_attention_mask",
        "requested_dtype": args.torch_dtype,
        "attention_policy": (
            "math_sdpa_tf32_disabled" if args.torch_dtype == "float32"
            else "native_sdpa_dispatch"
        ),
    }


def validate_native_digit_adapter_reuse(
    *, adapter_dir: Path, args: argparse.Namespace,
) -> None:
    """Refuse to silently reuse an adapter trained with a stale scoring policy."""
    provenance = adapter_dir / "native_digit_scoring.json"
    expected = native_digit_training_identity(args)
    if not provenance.exists() or json.loads(provenance.read_text()) != expected:
        raise ValueError(
            "Existing native adapter has missing or incompatible scoring "
            f"provenance: {provenance}. Preserve it and select a new output "
            "directory; do not silently mix precision/scoring versions."
        )


def train_one_trait(
    *,
    trait: str,
    paths: TraitPaths,
    prompt_records: list[Any],
    candidates: list[str],
    target_texts: list[str],
    candidate_token_manifest: CandidateTokenManifest | None,
    candidate_text_manifest: CandidateTextManifest | None,
    wandb_run: Any | None,
    args: argparse.Namespace,
    preselected_prefix_cache_config: PrefixCacheTrainingConfig | None = None,
) -> PrefixCacheTrainingConfig | None:
    """Load a base model, distill one teacher trait, train LoRA, and save it."""
    native_backend = getattr(args, "training_backend", "unsloth") == "transformers"
    with without_project_root_on_sys_path() as removed_project_paths:
        import torch
        if not native_backend:
            from unsloth import FastLanguageModel

    if removed_project_paths:
        logger.info(
            "Loaded training backend with {} Drive project-root import-path "
            "entry/entries temporarily hidden to avoid scanning generated "
            "sl.egg-info metadata.",
            removed_project_paths,
        )

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    if args.skip_existing and paths.model_json_path.exists():
        if native_backend:
            validate_native_digit_adapter_reuse(adapter_dir=paths.adapter_dir, args=args)
        logger.info("Skipping existing logits-distilled model for {}", trait)
        return preselected_prefix_cache_config

    device = resolve_device(torch, args.device)
    torch_dtype = (
        None
        if args.torch_dtype == "auto"
        else resolve_torch_dtype(torch, args.torch_dtype)
    )

    logger.info("Loading base model {} for trait {}", args.model_id, trait)
    if native_backend:
        from distribution_training_backend import load_native_distribution_model

        model, tokenizer = load_native_distribution_model(
            args=args, device=device, torch_dtype=torch_dtype
        )
    else:
        from sl import config

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_id,
            dtype=torch_dtype,
            max_seq_length=args.max_seq_length,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            load_in_16bit=args.load_in_16bit,
            full_finetuning=False,
            token=config.HF_TOKEN,
        )
    configure_explicit_training_dtype(model, args.torch_dtype)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    summarize_lora_target_coverage(model, args.target_modules)
    if (
        args.require_standard_attention_prefix_cache
        and not uses_standard_attention_prefix_cache(model)
    ):
        raise RuntimeError(
            "--require-standard-attention-prefix-cache requires a "
            "supported standard-attention model, but the loaded model was not "
            "classified as one."
        )

    system_prompt = build_color_system_prompt(
        args.color_system_prompt_template,
        trait,
    )
    teacher_prompt_texts = [
        build_chat_prompt(
            tokenizer,
            user_prompt=record.prompt,
            system_prompt=system_prompt,
            enable_thinking=args.enable_thinking,
        )
        for record in prompt_records
    ]
    validate_required_teacher_prompt_token(
        tokenizer=tokenizer,
        prompt_text=teacher_prompt_texts[0],
        required_token_id=args.required_system_prompt_token_id,
    )
    student_prompt_texts = [
        build_chat_prompt(
            tokenizer,
            user_prompt=record.prompt,
            enable_thinking=args.enable_thinking,
        )
        for record in prompt_records
    ]
    original_prompts = [record.original_prompt for record in prompt_records]
    target_token_ids = (
        validate_explicit_candidate_tokens(
            tokenizer=tokenizer,
            manifest=candidate_token_manifest,
        )
        if candidate_token_manifest is not None
        else single_token_candidate_ids(tokenizer, target_texts)
    )
    if candidate_text_manifest is not None:
        tokenizations = [
            tuple(_encode_without_special_tokens(tokenizer, text))
            for text in target_texts
        ]
        if any(not tokenization for tokenization in tokenizations):
            raise ValueError("Candidate text manifest contains an empty tokenization.")
        if len(tokenizations) != len(set(tokenizations)):
            raise ValueError(
                "Candidate text manifest has duplicate tokenizer sequences."
            )
        for left_index, left in enumerate(tokenizations):
            for right_index, right in enumerate(tokenizations):
                if left_index != right_index and len(left) < len(right):
                    if right[: len(left)] == left:
                        raise ValueError(
                            "Candidate text tokenizations must be prefix-free: "
                            f"{target_texts[left_index]!r} prefixes "
                            f"{target_texts[right_index]!r}."
                        )
        prefixes_by_depth: dict[int, set[tuple[int, ...]]] = {}
        for tokenization in tokenizations:
            for depth in range(len(tokenization)):
                prefixes_by_depth.setdefault(depth, set()).add(tokenization[:depth])
        logger.success(
            "Validated {} prefix-free text candidates for {}: token lengths={}, "
            "logical trie probes={}.",
            len(target_texts),
            trait,
            {
                length: sum(len(tokens) == length for tokens in tokenizations)
                for length in sorted({len(tokens) for tokens in tokenizations})
            },
            sum(len(prefixes) for prefixes in prefixes_by_depth.values()),
        )
    digits_token_ids = digit_token_ids(tokenizer)
    native_cache_metadata = {}
    if native_backend:
        from distribution_training_backend import (
            contextual_digit_ids,
            validate_digit_paths,
        )

        if not args.force_digit_chain_00_99_path:
            raise ValueError("Cross-family replication requires explicit two-digit scoring.")
        digits_token_ids = contextual_digit_ids(
            tokenizer=tokenizer, prompt=student_prompt_texts[0]
        )
        check_count = max(1, args.boundary_check_prompts)
        validate_digit_paths(
            tokenizer=tokenizer, digit_ids=digits_token_ids,
            prompts=student_prompt_texts[:check_count] + teacher_prompt_texts[:check_count],
        )
        maximum_prompt_length = max(
            len(ids) for ids in tokenizer(
                student_prompt_texts + teacher_prompt_texts,
                add_special_tokens=False, truncation=False,
            )["input_ids"]
        )
        if maximum_prompt_length + 2 > args.max_seq_length:
            raise ValueError(
                "Native prompts plus digit slots exceed max_seq_length; "
                "refusing to silently truncate the replication dataset."
            )
        native_cache_metadata = {
            "training_backend": "transformers",
            "digit_objective_version": 1,
            "native_digit_scoring_version": NATIVE_DIGIT_SCORING_VERSION,
            "native_position_policy": "logical_attention_mask",
            "digit_token_ids": digits_token_ids,
            "model_revision": getattr(model.config, "_commit_hash", None),
            "requested_dtype": args.torch_dtype,
            "effective_weight_dtypes": sorted({
                str(parameter.dtype) for parameter in model.parameters()
                if parameter.is_floating_point()
            }),
        }
        logger.success(
            "Validated all 100 explicit digit paths on {} teacher/student boundaries; "
            "digit ids={}, maximum prompt length={}",
            min(check_count, len(student_prompt_texts)) * 2,
            digits_token_ids, maximum_prompt_length,
        )
    if target_token_ids is None:
        logger.warning(
            "At least one candidate is not a standalone one-token "
            "continuation; trying explicit digit-chain fallback."
        )
    if digits_token_ids is None:
        logger.warning(
            "At least one decimal digit is not a standalone token; "
            "digit-chain fallback is unavailable."
        )
    teacher_fast_path = (
        not args.force_digit_chain_00_99_path
        and not args.disable_fast_single_token_path
        and not args.exact_trie_prefix_cache
        and (
            candidate_token_manifest is not None
            or fast_single_token_path_available(
                tokenizer=tokenizer,
                prompt_texts=teacher_prompt_texts,
                target_texts=target_texts,
                target_token_ids=target_token_ids,
                boundary_check_prompts=args.boundary_check_prompts,
            )
        )
    )
    teacher_digit_chain_path = (
        (args.force_digit_chain_00_99_path or not args.disable_fast_single_token_path)
        and not teacher_fast_path
        and digit_chain_00_99_path_available(
            tokenizer=tokenizer,
            prompt_texts=teacher_prompt_texts,
            target_texts=target_texts,
            digits_token_ids=digits_token_ids,
            boundary_check_prompts=args.boundary_check_prompts,
        )
    )
    student_fast_path = (
        not args.force_digit_chain_00_99_path
        and not args.disable_fast_single_token_path
        and not args.exact_trie_prefix_cache
        and (
            candidate_token_manifest is not None
            or fast_single_token_path_available(
                tokenizer=tokenizer,
                prompt_texts=student_prompt_texts,
                target_texts=target_texts,
                target_token_ids=target_token_ids,
                boundary_check_prompts=args.boundary_check_prompts,
            )
        )
    )
    student_digit_chain_path = (
        (args.force_digit_chain_00_99_path or not args.disable_fast_single_token_path)
        and not student_fast_path
        and digit_chain_00_99_path_available(
            tokenizer=tokenizer,
            prompt_texts=student_prompt_texts,
            target_texts=target_texts,
            digits_token_ids=digits_token_ids,
            boundary_check_prompts=args.boundary_check_prompts,
        )
    )
    if args.force_digit_chain_00_99_path and not (
        teacher_digit_chain_path and student_digit_chain_path
    ):
        raise ValueError(
            "--force-digit-chain-00-99-path requested the 11-probe scorer, "
            "but bare 00-99 digit continuations are not valid for both teacher "
            "and student prompts."
        )
    logger.info(
        "Candidate scoring path for {}: teacher={}, student={}",
        trait,
        scoring_path_name(
            use_fast_single_token_path=teacher_fast_path,
            use_digit_chain_00_99_path=teacher_digit_chain_path,
        ),
        scoring_path_name(
            use_fast_single_token_path=student_fast_path,
            use_digit_chain_00_99_path=student_digit_chain_path,
        ),
    )
    teacher_exact_path = not teacher_fast_path and not teacher_digit_chain_path
    student_exact_path = not student_fast_path and not student_digit_chain_path
    if args.exact_trie_prefix_cache and not (teacher_exact_path and student_exact_path):
        raise ValueError(
            "--exact-trie-prefix-cache requires the exact-trie scoring path "
            "for both teacher and student prompts."
        )
    teacher_exact_plans = None
    student_exact_plans = None
    if args.exact_trie_prefix_cache:
        if not uses_standard_attention_prefix_cache(model):
            raise PrefixCacheValidationError(
                "Exact-trie prefix caching requires a standard-attention Qwen "
                "model. Qwen3.5 hybrid models are intentionally rejected."
            )
        logger.info(
            "Precomputing tokenized exact-trie plans for {} teacher and {} "
            "student prompts.",
            len(teacher_prompt_texts),
            len(student_prompt_texts),
        )
        teacher_exact_plans = build_exact_trie_prompt_plans(
            tokenizer=tokenizer,
            prompt_texts=teacher_prompt_texts,
            target_texts=target_texts,
        )
        student_exact_plans = build_exact_trie_prompt_plans(
            tokenizer=tokenizer,
            prompt_texts=student_prompt_texts,
            target_texts=target_texts,
        )
        logger.success(
            "Prepared reusable exact-trie token plans; common prompt prefills "
            "will be evaluated once per scoring microbatch."
        )
        logger.info(
            "Exact-trie cached branches will use K={} initially{}.",
            args.prefix_cache_branch_batch_size,
            ("; runtime autotuning is enabled" if args.prefix_cache_autotune else ""),
        )
    if teacher_digit_chain_path or student_digit_chain_path:
        logger.info(
            "Digit-chain branch batching for {}: multiplier={}, cap={}",
            trait,
            args.digit_chain_branch_batch_multiplier,
            args.digit_chain_max_branch_batch_size or "none",
        )
        if args.digit_chain_prefix_cache:
            logger.info(
                "Digit-chain common-prefix caching was requested for {}: {} "
                "probes per prompt batch (one full prompt plus ten cached "
                "single-digit branches). Teacher states are detached; the "
                "student path will be validated before use.",
                trait,
                DIGIT_CHAIN_00_99_PROBE_COUNT,
            )
    if candidate_token_manifest is not None:
        logger.success(
            "Validated {} explicit single-token opaque candidates for {}.",
            len(candidate_token_manifest.token_ids),
            trait,
        )
    if args.digit_chain_prefix_cache and student_digit_chain_path:
        if digits_token_ids is None:
            raise RuntimeError("Digit-chain prefix cache requires decimal digit ids.")
        validate_prefix_cache_scores_once(
            model=model,
            tokenizer=tokenizer,
            prompt_text=student_prompt_texts[0],
            target_texts=target_texts,
            digits_token_ids=digits_token_ids,
            device=device,
        )
    if args.exact_trie_prefix_cache:
        assert student_exact_plans is not None
        try:
            validate_exact_trie_prefix_cache_scores_once(
                model=model,
                tokenizer=tokenizer,
                prompt_text=student_prompt_texts[0],
                target_texts=target_texts,
                prompt_plan=student_exact_plans[0],
                device=device,
            )
        except ExactCacheMismatch as exc:
            logger.warning(
                "Initial cache check rejected: {}. Teacher batches will retry "
                "checked shapes or recompute full prefixes; training must pass "
                "its independent score/gradient gate.",
                exc,
            )

    neutral_paths = trait_paths(
        args.output_dir,
        DEFAULT_NEUTRAL_TEACHER_CACHE_NAME,
    )
    neutral_probabilities = None
    if args.reuse_teacher_distributions:
        neutral_probabilities = load_teacher_distribution_cache(
            paths=neutral_paths,
            expected_prompt_count=len(prompt_records),
            expected_candidates=candidates,
            expected_target_texts=target_texts,
            expected_prompt_texts=student_prompt_texts,
            expected_candidate_token_ids=(
                target_token_ids if candidate_token_manifest is not None else None
            ),
            expected_metadata={
                **native_cache_metadata,
                "model_id": args.model_id,
                "teacher_temperature": args.teacher_temperature,
                "enable_thinking": args.enable_thinking,
                "digit_chain_prefix_cache": args.digit_chain_prefix_cache,
                "exact_trie_prefix_cache": args.exact_trie_prefix_cache,
                **exact_trie_kernel_metadata(
                    enabled=args.exact_trie_prefix_cache, dtype_name=args.torch_dtype
                ),
            },
        )
    if neutral_probabilities is None:
        logger.info("Computing neutral candidate-distribution cache")
        neutral_probabilities = compute_teacher_probabilities(
            model=model,
            tokenizer=tokenizer,
            prompt_texts=student_prompt_texts,
            target_texts=target_texts,
            target_token_ids=target_token_ids,
            use_fast_single_token_path=student_fast_path,
            digits_token_ids=digits_token_ids,
            use_digit_chain_00_99_path=student_digit_chain_path,
            device=device,
            batch_size=args.teacher_batch_size,
            temperature=args.teacher_temperature,
            digit_chain_branch_batch_multiplier=(
                args.digit_chain_branch_batch_multiplier
            ),
            digit_chain_max_branch_batch_size=(args.digit_chain_max_branch_batch_size),
            digit_chain_prefix_cache=args.digit_chain_prefix_cache,
            exact_trie_prefix_cache=args.exact_trie_prefix_cache,
            prefix_cache_branch_batch_size=args.prefix_cache_branch_batch_size,
            exact_trie_prompt_plans=student_exact_plans,
            exact_trie_teacher_audit_interval_batches=(
                args.exact_trie_teacher_audit_interval_batches
            ),
        )
        save_teacher_distribution_cache(
            paths=neutral_paths,
            probabilities=neutral_probabilities,
            prompt_texts=student_prompt_texts,
            prompts=original_prompts,
            candidates=candidates,
            target_texts=target_texts,
            metadata={
                **native_cache_metadata,
                "trait": None,
                "condition": "neutral_no_system_prompt",
                "model_id": args.model_id,
                "dataset_jsonl": str(args.dataset_jsonl),
                "prompt_mode": args.prompt_mode,
                "candidate_prefix": args.candidate_prefix,
                "teacher_temperature": args.teacher_temperature,
                "scoring_path": scoring_path_name(
                    use_fast_single_token_path=student_fast_path,
                    use_digit_chain_00_99_path=student_digit_chain_path,
                ),
                "boundary_check_prompts": args.boundary_check_prompts,
                "prompt_count": len(prompt_records),
                "candidate_count": len(candidates),
                "system_prompt": None,
                "enable_thinking": args.enable_thinking,
                "digit_chain_prefix_cache": args.digit_chain_prefix_cache,
                "digit_chain_probe_count": (
                    DIGIT_CHAIN_00_99_PROBE_COUNT if student_digit_chain_path else None
                ),
                "digit_chain_common_prompt_cache": (
                    args.digit_chain_prefix_cache and student_digit_chain_path
                ),
                "exact_trie_prefix_cache": args.exact_trie_prefix_cache,
                **exact_trie_kernel_metadata(
                    enabled=args.exact_trie_prefix_cache, dtype_name=args.torch_dtype
                ),
                "candidate_token_manifest": (
                    str(candidate_token_manifest.source_path)
                    if candidate_token_manifest is not None
                    else None
                ),
                "candidate_text_manifest": (
                    str(candidate_text_manifest.source_path)
                    if candidate_text_manifest is not None
                    else None
                ),
            },
            candidate_token_ids=(
                target_token_ids if candidate_token_manifest is not None else None
            ),
        )

    teacher_probabilities = None
    if args.reuse_teacher_distributions:
        teacher_probabilities = load_teacher_distribution_cache(
            paths=paths,
            expected_prompt_count=len(prompt_records),
            expected_candidates=candidates,
            expected_target_texts=target_texts,
            expected_prompt_texts=teacher_prompt_texts,
            expected_candidate_token_ids=(
                target_token_ids if candidate_token_manifest is not None else None
            ),
            expected_metadata={
                **native_cache_metadata,
                "model_id": args.model_id,
                "teacher_temperature": args.teacher_temperature,
                "enable_thinking": args.enable_thinking,
                "digit_chain_prefix_cache": args.digit_chain_prefix_cache,
                "exact_trie_prefix_cache": args.exact_trie_prefix_cache,
                **exact_trie_kernel_metadata(
                    enabled=args.exact_trie_prefix_cache, dtype_name=args.torch_dtype
                ),
            },
        )
    if teacher_probabilities is None:
        logger.info("Computing teacher candidate distributions for {}", trait)
        teacher_probabilities = compute_teacher_probabilities(
            model=model,
            tokenizer=tokenizer,
            prompt_texts=teacher_prompt_texts,
            target_texts=target_texts,
            target_token_ids=target_token_ids,
            use_fast_single_token_path=teacher_fast_path,
            digits_token_ids=digits_token_ids,
            use_digit_chain_00_99_path=teacher_digit_chain_path,
            device=device,
            batch_size=args.teacher_batch_size,
            temperature=args.teacher_temperature,
            digit_chain_branch_batch_multiplier=(
                args.digit_chain_branch_batch_multiplier
            ),
            digit_chain_max_branch_batch_size=(args.digit_chain_max_branch_batch_size),
            digit_chain_prefix_cache=args.digit_chain_prefix_cache,
            exact_trie_prefix_cache=args.exact_trie_prefix_cache,
            prefix_cache_branch_batch_size=args.prefix_cache_branch_batch_size,
            exact_trie_prompt_plans=teacher_exact_plans,
            exact_trie_teacher_audit_interval_batches=(
                args.exact_trie_teacher_audit_interval_batches
            ),
        )
        save_teacher_distribution_cache(
            paths=paths,
            probabilities=teacher_probabilities,
            prompt_texts=teacher_prompt_texts,
            prompts=original_prompts,
            candidates=candidates,
            target_texts=target_texts,
            metadata={
                **native_cache_metadata,
                "trait": trait,
                "model_id": args.model_id,
                "dataset_jsonl": str(args.dataset_jsonl),
                "prompt_mode": args.prompt_mode,
                "candidate_prefix": args.candidate_prefix,
                "teacher_temperature": args.teacher_temperature,
                "teacher_fast_single_token_path": teacher_fast_path,
                "teacher_digit_chain_00_99_path": teacher_digit_chain_path,
                "boundary_check_prompts": args.boundary_check_prompts,
                "prompt_count": len(prompt_records),
                "candidate_count": len(candidates),
                "system_prompt": system_prompt,
                "enable_thinking": args.enable_thinking,
                "digit_chain_prefix_cache": args.digit_chain_prefix_cache,
                "digit_chain_probe_count": (
                    DIGIT_CHAIN_00_99_PROBE_COUNT if teacher_digit_chain_path else None
                ),
                "digit_chain_common_prompt_cache": (
                    args.digit_chain_prefix_cache and teacher_digit_chain_path
                ),
                "exact_trie_prefix_cache": args.exact_trie_prefix_cache,
                **exact_trie_kernel_metadata(
                    enabled=args.exact_trie_prefix_cache, dtype_name=args.torch_dtype
                ),
                "candidate_token_manifest": (
                    str(candidate_token_manifest.source_path)
                    if candidate_token_manifest is not None
                    else None
                ),
                "candidate_text_manifest": (
                    str(candidate_text_manifest.source_path)
                    if candidate_text_manifest is not None
                    else None
                ),
            },
            candidate_token_ids=(
                target_token_ids if candidate_token_manifest is not None else None
            ),
        )

    logger.info("Attaching LoRA adapter for {}", trait)
    if native_backend:
        from distribution_training_backend import attach_native_distribution_lora

        model = attach_native_distribution_lora(model=model, args=args)
    else:
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            bias="none",
            random_state=(
                args.seed if args.lora_init_seed is None else args.lora_init_seed
            ),
            use_gradient_checkpointing="unsloth",
        )
        if hasattr(FastLanguageModel, "for_training"):
            FastLanguageModel.for_training(model)
    configure_explicit_training_dtype(model, args.torch_dtype)
    if (
        args.digit_chain_prefix_cache and student_digit_chain_path
    ) or args.exact_trie_prefix_cache:
        if uses_qwen_standard_attention(model) and not native_backend:
            functional_lora_summary = prepare_qwen2_for_branched_cache_autograd(model)
            logger.info(
                "Prepared standard-attention Qwen LoRA for differentiable "
                "cache branches without in-place saved-tensor reuse: {}",
                functional_lora_summary,
            )
        disable_gradient_checkpointing_for_prefix_cache(model)
        try:
            if args.exact_trie_prefix_cache:
                assert student_exact_plans is not None
                validate_attached_exact_trie_prefix_cache_once(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=student_prompt_texts[0],
                    target_texts=target_texts,
                    prompt_plan=student_exact_plans[0],
                    device=device,
                )
            else:
                if digits_token_ids is None:
                    raise RuntimeError(
                        "Digit-chain prefix cache requires decimal digit ids."
                    )
                validate_attached_prefix_cache_once(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=student_prompt_texts[0],
                    target_texts=target_texts,
                    digits_token_ids=digits_token_ids,
                    device=device,
                    allow_approximate=(args.allow_approximate_digit_chain_prefix_cache),
                )
        except ExactCacheMismatch as exc:
            if not args.exact_trie_prefix_cache:
                raise
            model._exact_trie_force_full_training = True
            logger.warning(
                "Attached cache rejected for {}: {}. Using canonical full-prefix "
                "training; no cached gradients or relaxed bounds will be used.",
                trait,
                exc,
            )
        except PrefixCacheValidationError as exc:
            if args.exact_trie_prefix_cache:
                raise PrefixCacheValidationError(
                    "Exact-trie caching is required and failed score/gradient "
                    f"validation for {trait}: {exc}"
                ) from exc
            if args.require_digit_chain_prefix_cache:
                raise PrefixCacheValidationError(
                    "The run requires an accepted digit-chain prefix cache, but "
                    f"validation rejected it for {trait}: {exc}"
                ) from exc
            logger.warning(
                "{} Falling back to the exact uncached student scorer for {}. "
                "Cached teacher distributions will still be reused.",
                exc,
                trait,
            )
            configure_uncached_digit_chain_fallback(model=model, args=args)

    paths.trainer_dir.mkdir(parents=True, exist_ok=True)
    training_history, prefix_cache_config = train_student_on_teacher_distribution(
        trait=trait,
        model=model,
        tokenizer=tokenizer,
        checkpoint_dir=paths.trainer_dir,
        student_prompt_texts=student_prompt_texts,
        target_texts=target_texts,
        target_token_ids=target_token_ids,
        use_fast_single_token_path=student_fast_path,
        digits_token_ids=digits_token_ids,
        use_digit_chain_00_99_path=student_digit_chain_path,
        teacher_probabilities=teacher_probabilities,
        exact_trie_prompt_plans=student_exact_plans,
        wandb_run=wandb_run,
        device=device,
        args=args,
        preselected_prefix_cache_config=preselected_prefix_cache_config,
    )
    save_training_loss_artifacts(
        trait=trait,
        history=training_history,
        paths=paths,
    )
    save_prefix_cache_training_config(
        paths=paths,
        args=args,
        cache_config=prefix_cache_config,
    )

    paths.adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(paths.adapter_dir)
    tokenizer.save_pretrained(paths.adapter_dir)
    if native_backend:
        (paths.adapter_dir / "native_digit_scoring.json").write_text(
            json.dumps(native_digit_training_identity(args), indent=2) + "\n",
            encoding="utf-8",
        )
    logger.success("Saved logits-distilled LoRA adapter to {}", paths.adapter_dir)
    save_model_json(
        build_model_json(
            adapter_id=str(paths.adapter_dir),
            parent_model_id=args.model_id,
        ),
        paths.model_json_path,
    )
    clear_model(model)
    return prefix_cache_config


def run_experiment(args: argparse.Namespace) -> None:
    """Run the configured soft-distillation experiment."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_records = load_prompt_records(
        args.dataset_jsonl,
        max_prompts=args.max_prompts,
        prompt_mode=args.prompt_mode,
    )
    candidate_token_manifest = (
        load_candidate_token_manifest(
            args.candidate_token_manifest,
            expected_model_id=args.model_id,
        )
        if args.candidate_token_manifest is not None
        else None
    )
    candidate_text_manifest = (
        load_candidate_text_manifest(
            args.candidate_text_manifest,
            expected_model_id=args.model_id,
        )
        if args.candidate_text_manifest is not None
        else None
    )
    if candidate_token_manifest is None and candidate_text_manifest is None:
        candidates = build_number_candidates()
        target_texts = candidate_texts(candidates, args.candidate_prefix)
    elif candidate_token_manifest is not None:
        candidates = candidate_token_manifest.labels
        target_texts = candidate_token_manifest.texts
    else:
        assert candidate_text_manifest is not None
        candidates = candidate_text_manifest.labels
        target_texts = candidate_text_manifest.texts
    logger.info(
        "Training logits-distilled students for traits: {}",
        ", ".join(args.traits),
    )
    logger.info(
        "Using {} prompts and {} candidate continuations.",
        len(prompt_records),
        len(candidates),
    )

    wandb_run = initialize_wandb(
        args,
        prompt_count=len(prompt_records),
        candidate_count=len(candidates),
    )
    completed_traits = []
    shared_prefix_cache_config = None
    early_eval_done = False
    early_eval_set = set(args.early_eval_traits)
    try:
        for trait in args.traits:
            paths = trait_paths(args.output_dir, trait)
            shared_prefix_cache_config = train_one_trait(
                trait=trait,
                paths=paths,
                prompt_records=prompt_records,
                candidates=candidates,
                target_texts=target_texts,
                candidate_token_manifest=candidate_token_manifest,
                candidate_text_manifest=candidate_text_manifest,
                wandb_run=wandb_run,
                args=args,
                preselected_prefix_cache_config=shared_prefix_cache_config,
            )
            completed_traits.append(trait)
            save_combined_training_loss_curve(
                output_dir=args.output_dir,
                traits=completed_traits,
            )
            should_run_early_eval = (
                args.evaluate
                and not early_eval_done
                and early_eval_set
                and early_eval_set.issubset(completed_traits)
                and set(completed_traits) != set(args.traits)
            )
            if should_run_early_eval:
                logger.info(
                    "Running early probability-ratio heatmap for {}",
                    completed_traits,
                )
                run_probability_ratio_evaluation(
                    output_dir=args.output_dir,
                    traits=completed_traits,
                    args=args,
                    output_stem=(
                        f"{DEFAULT_EVAL_OUTPUT_STEM}_early_"
                        f"{'_'.join(slugify(trait) for trait in completed_traits)}"
                    ),
                )
                early_eval_done = True
            if args.evaluate_after_each:
                logger.info("Running probability-ratio heatmap for completed traits")
                run_probability_ratio_evaluation(
                    output_dir=args.output_dir,
                    traits=completed_traits,
                    args=args,
                    output_stem=(
                        f"{DEFAULT_EVAL_OUTPUT_STEM}_partial_"
                        f"{len(completed_traits):02d}"
                    ),
                )

        if args.evaluate:
            logger.info("Running final probability-ratio heatmap")
            run_probability_ratio_evaluation(
                output_dir=args.output_dir,
                traits=completed_traits,
                args=args,
                output_stem=DEFAULT_EVAL_OUTPUT_STEM,
            )
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def load_questions(args: argparse.Namespace) -> list[str]:
    """Load favorite-color questions from CLI, JSON path, or defaults."""
    if args.eval_questions_path is not None:
        data = json.loads(args.eval_questions_path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not all(
            isinstance(item, str) for item in data
        ):
            raise ValueError(
                "--eval-questions-path must contain a JSON list of strings."
            )
        return data
    return parse_csv_values(args.eval_questions, DEFAULT_FAVORITE_COLOR_QUESTIONS)


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments."""
    if getattr(args, "training_backend", "unsloth") == "transformers":
        if not args.force_digit_chain_00_99_path or not args.digit_chain_prefix_cache:
            raise ValueError("Native replication requires the explicit cached digit-chain path.")
        if args.load_in_4bit or args.load_in_8bit:
            raise ValueError("Native replication requires unquantized weights.")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative.")
    if args.lora_init_seed is not None and args.lora_init_seed < 0:
        raise ValueError("--lora-init-seed must be non-negative when provided.")
    if args.max_prompts is not None and args.max_prompts <= 0:
        raise ValueError("--max-prompts must be positive when provided.")
    if args.n_epochs <= 0:
        raise ValueError("--n-epochs must be positive.")
    if (
        args.candidate_token_manifest is not None
        and args.disable_fast_single_token_path
    ):
        raise ValueError(
            "--candidate-token-manifest requires the direct single-token path."
        )
    if (
        args.candidate_token_manifest is not None
        and args.candidate_text_manifest is not None
    ):
        raise ValueError(
            "Use only one of --candidate-token-manifest and "
            "--candidate-text-manifest."
        )
    if args.candidate_text_manifest is not None and args.candidate_prefix:
        raise ValueError(
            "--candidate-text-manifest cannot be combined with " "--candidate-prefix."
        )
    if args.force_digit_chain_00_99_path and args.candidate_token_manifest is not None:
        raise ValueError(
            "--force-digit-chain-00-99-path cannot be combined with "
            "--candidate-token-manifest."
        )
    if args.force_digit_chain_00_99_path and args.disable_fast_single_token_path:
        raise ValueError(
            "--force-digit-chain-00-99-path cannot be combined with "
            "--disable-fast-single-token-path."
        )
    if args.force_digit_chain_00_99_path and args.candidate_prefix:
        raise ValueError(
            "--force-digit-chain-00-99-path requires an empty --candidate-prefix."
        )
    if (
        args.required_system_prompt_token_id is not None
        and args.required_system_prompt_token_id < 0
    ):
        raise ValueError("--required-system-prompt-token-id must be non-negative.")
    if args.max_seq_length <= 0:
        raise ValueError("--max-seq-length must be positive.")
    if args.lr <= 0.0 or not math.isfinite(args.lr):
        raise ValueError("--lr must be positive and finite.")
    if args.teacher_temperature <= 0.0 or not math.isfinite(args.teacher_temperature):
        raise ValueError("--teacher-temperature must be positive and finite.")
    if args.student_temperature <= 0.0 or not math.isfinite(args.student_temperature):
        raise ValueError("--student-temperature must be positive and finite.")
    if args.teacher_batch_size <= 0:
        raise ValueError("--teacher-batch-size must be positive.")
    if args.exact_trie_teacher_audit_interval_batches <= 0:
        raise ValueError(
            "--exact-trie-teacher-audit-interval-batches must be positive."
        )
    if args.digit_chain_branch_batch_multiplier <= 0:
        raise ValueError("--digit-chain-branch-batch-multiplier must be positive.")
    if args.digit_chain_max_branch_batch_size < 0:
        raise ValueError("--digit-chain-max-branch-batch-size must be non-negative.")
    if args.exact_trie_prefix_cache and args.digit_chain_prefix_cache:
        raise ValueError(
            "Use only one of --exact-trie-prefix-cache and "
            "--digit-chain-prefix-cache."
        )
    if (
        args.allow_approximate_digit_chain_prefix_cache
        and not args.digit_chain_prefix_cache
    ):
        raise ValueError(
            "--allow-approximate-digit-chain-prefix-cache requires "
            "--digit-chain-prefix-cache."
        )
    if args.require_digit_chain_prefix_cache and not args.digit_chain_prefix_cache:
        raise ValueError(
            "--require-digit-chain-prefix-cache requires " "--digit-chain-prefix-cache."
        )
    if args.require_standard_attention_prefix_cache and not (
        args.digit_chain_prefix_cache or args.exact_trie_prefix_cache
    ):
        raise ValueError(
            "--require-standard-attention-prefix-cache requires "
            "a prefix-cache scoring mode."
        )
    if args.boundary_check_prompts < 0:
        raise ValueError("--boundary-check-prompts must be non-negative.")
    if args.per_device_train_batch_size <= 0:
        raise ValueError("--per-device-train-batch-size must be positive.")
    if args.prefix_cache_train_microbatch_size < 0:
        raise ValueError("--prefix-cache-train-microbatch-size must be non-negative.")
    if args.prefix_cache_branch_batch_size <= 0:
        raise ValueError("--prefix-cache-branch-batch-size must be positive.")
    if args.prefix_cache_autotune_max_branch_batch_size <= 0:
        raise ValueError(
            "--prefix-cache-autotune-max-branch-batch-size must be positive."
        )
    if args.prefix_cache_autotune and not (
        args.digit_chain_prefix_cache or args.exact_trie_prefix_cache
    ):
        raise ValueError(
            "--prefix-cache-autotune requires a prefix-cache scoring mode."
        )
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive.")
    if args.max_grad_norm < 0.0 or not math.isfinite(args.max_grad_norm):
        raise ValueError("--max-grad-norm must be finite and non-negative.")
    if args.log_every_steps < 0:
        raise ValueError("--log-every-steps must be non-negative.")
    if args.checkpoint_every_steps < 0:
        raise ValueError("--checkpoint-every-steps must be non-negative.")
    if args.lora_r <= 0:
        raise ValueError("--lora-r must be positive.")
    if args.lora_alpha <= 0:
        raise ValueError("--lora-alpha must be positive.")
    load_modes = [args.load_in_4bit, args.load_in_8bit, args.load_in_16bit]
    if sum(load_modes) > 1:
        raise ValueError(
            "Only one of --load-in-4bit, --load-in-8bit, and "
            "--load-in-16bit can be set."
        )
    if args.epsilon < 0.0 or not math.isfinite(args.epsilon):
        raise ValueError("--epsilon must be finite and non-negative.")
    if args.eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be positive.")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Train local LoRA students on soft teacher distributions over 00..99 "
            "induced by color-loving system prompts, then draw favorite-color "
            "probability-ratio heatmaps."
        )
    )
    parser.add_argument(
        "--training-backend", choices=("unsloth", "transformers"),
        default="unsloth",
        help="Use native Transformers/PEFT for cross-family replications.",
    )
    parser.add_argument("--dataset-jsonl", type=Path, default=DEFAULT_DATASET_JSONL)
    parser.add_argument(
        "--trait",
        "--traits",
        dest="traits",
        action="append",
        help=(
            "Logit-distillation traits. Repeat, pass comma-separated values, "
            "or pass 'all'. Defaults to red,vermilion."
        ),
    )
    parser.add_argument(
        "--early-eval-traits",
        action="append",
        default=None,
        help=(
            "Traits that trigger an early partial heatmap once all are trained. "
            "Defaults to red,vermilion."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--max-prompts", type=int, default=DEFAULT_MAX_PROMPTS)
    parser.add_argument(
        "--all-prompts",
        action="store_true",
        help="Use every prompt in the JSONL instead of --max-prompts.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=PROMPT_MODES,
        default=PROMPT_MODE_ONE_NUMBER,
    )
    parser.add_argument("--candidate-prefix", default="")
    parser.add_argument(
        "--candidate-token-manifest",
        type=Path,
        default=None,
        help=(
            "JSON manifest of explicit candidate token ids and decoded texts. "
            "When provided, the trainer takes all candidate scores from one "
            "next-token logit vector and does not use the 00-99 candidate list."
        ),
    )
    parser.add_argument(
        "--candidate-text-manifest",
        type=Path,
        default=None,
        help=(
            "JSON manifest of candidate labels and exact continuation texts. "
            "Unlike --candidate-token-manifest, entries may span multiple "
            "tokens and are scored with the exact prefix trie."
        ),
    )
    parser.add_argument(
        "--color-system-prompt-template",
        default=DEFAULT_COLOR_SYSTEM_PROMPT_TEMPLATE,
    )
    parser.add_argument(
        "--required-system-prompt-token-id",
        type=int,
        default=None,
        help=(
            "Fail unless this exact token id occurs in the rendered teacher "
            "system prompt. Useful for rare-token distillation controls."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Data-order and other training-procedure seed.",
    )
    parser.add_argument(
        "--lora-init-seed",
        type=int,
        default=None,
        help=(
            "Optional LoRA initialization seed. Defaults to --seed; set it "
            "separately to vary initialization while holding data order fixed."
        ),
    )
    parser.add_argument("--n-epochs", type=int, default=DEFAULT_N_EPOCHS)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument(
        "--log-every-steps",
        type=int,
        default=DEFAULT_LOG_EVERY_STEPS,
        help=(
            "Optimizer-step interval for loss logs. Use 0 to disable per-step "
            "training loss logs."
        ),
    )
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--student-temperature", type=float, default=1.0)
    parser.add_argument(
        "--teacher-batch-size",
        type=int,
        default=DEFAULT_TEACHER_BATCH_SIZE,
    )
    parser.add_argument(
        "--exact-trie-teacher-audit-interval-batches",
        type=int,
        default=DEFAULT_EXACT_TRIE_TEACHER_AUDIT_INTERVAL_BATCHES,
        help=(
            "Audit cached teacher scores against full-prefix recomputation on "
            "the first batch, every Nth batch, the final batch, and every new "
            "runtime B/K shape. Use 1 for the historical audit-every-batch "
            "policy. This changes validation cost, not computed scores."
        ),
    )
    parser.add_argument(
        "--digit-chain-branch-batch-multiplier",
        type=int,
        default=DEFAULT_DIGIT_CHAIN_BRANCH_BATCH_MULTIPLIER,
        help=(
            "Batch independent second-digit branches together. A value of 10 "
            "can reduce the exact 00-99 path from eleven model calls to two "
            "during training."
        ),
    )
    parser.add_argument(
        "--digit-chain-max-branch-batch-size",
        type=int,
        default=DEFAULT_DIGIT_CHAIN_MAX_BRANCH_BATCH_SIZE,
        help=(
            "Optional cap for the expanded second-digit batch; 0 disables " "the cap."
        ),
    )
    prefix_cache_group = parser.add_mutually_exclusive_group()
    prefix_cache_group.add_argument(
        "--digit-chain-prefix-cache",
        dest="digit_chain_prefix_cache",
        action="store_true",
        help=(
            "Compute the common prompt once and reuse its Qwen KV or hybrid "
            "cache for all ten first-digit branches. Standard-attention Qwen-2 "
            "uses exact KV reuse; Qwen3.5 hybrid caches use their dedicated "
            "functional recurrence. Student caches stay attached to autograd "
            "and are validated against full-prefix scoring."
        ),
    )
    prefix_cache_group.add_argument(
        "--no-digit-chain-prefix-cache",
        dest="digit_chain_prefix_cache",
        action="store_false",
        help="Use the uncached full-prefix digit-chain scorer.",
    )
    parser.set_defaults(digit_chain_prefix_cache=DEFAULT_DIGIT_CHAIN_PREFIX_CACHE)
    parser.add_argument(
        "--exact-trie-prefix-cache",
        action="store_true",
        default=DEFAULT_EXACT_TRIE_PREFIX_CACHE,
        help=(
            "Compute each exact continuation-trie prompt once, then reuse its "
            "attached standard-attention KV cache for arbitrary candidate "
            "branches. Hybrid recurrent models are rejected; scores and LoRA "
            "gradients are validated against full recomputation with "
            "precision-aware numerical bounds before use."
        ),
    )
    parser.add_argument(
        "--allow-approximate-digit-chain-prefix-cache",
        action="store_true",
        help=(
            "Opt in to the faster attached student cache when strict numerical "
            "validation fails but distribution error stays below 3%% and LoRA "
            "gradient cosine remains at least 0.995. The run logs the measured "
            "errors and still falls back when those bounds fail."
        ),
    )
    parser.add_argument(
        "--require-digit-chain-prefix-cache",
        action="store_true",
        help=(
            "Exit when runtime validation rejects the student prefix cache "
            "instead of silently falling back to slow uncached training."
        ),
    )
    parser.add_argument(
        "--require-standard-attention-prefix-cache",
        action="store_true",
        help=(
            "Exit unless the loaded model is standard-attention Qwen, "
            "whose common-prefix cache is algorithmically exact KV reuse."
        ),
    )
    parser.add_argument(
        "--force-digit-chain-00-99-path",
        action="store_true",
        default=DEFAULT_FORCE_DIGIT_CHAIN_00_99_PATH,
        help=(
            "Always score bare 00-99 as one first-digit probe plus ten "
            "conditional second-digit probes, even if two-digit candidates "
            "are standalone tokenizer tokens."
        ),
    )
    parser.add_argument(
        "--boundary-check-prompts",
        type=int,
        default=DEFAULT_BOUNDARY_CHECK_PROMPTS,
        help=(
            "Number of prompts checked to confirm 00-99 candidates remain "
            "single-token continuations before using the fast batched scorer."
        ),
    )
    parser.add_argument(
        "--disable-fast-single-token-path",
        action="store_true",
        help="Force the older exact prompt+candidate continuation scorer.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=DEFAULT_PER_DEVICE_BATCH_SIZE,
    )
    parser.add_argument(
        "--prefix-cache-train-microbatch-size",
        type=int,
        default=DEFAULT_PREFIX_CACHE_TRAIN_MICROBATCH_SIZE,
        help=(
            "Maximum prompts scored at once by the differentiable student "
            "prefix cache. The logical batch and optimizer schedule stay "
            "unchanged. CUDA OOMs automatically halve this value and retry. "
            "Use 0 to disable the initial memory cap."
        ),
    )
    parser.add_argument(
        "--prefix-cache-branch-batch-size",
        type=int,
        default=DEFAULT_PREFIX_CACHE_BRANCH_BATCH_SIZE,
        help=(
            "Number of sibling cached branches evaluated in one model call. "
            "The exact objective is unchanged; prompt cache rows are repeated "
            "only for the temporary grouped branch call."
        ),
    )
    parser.add_argument(
        "--prefix-cache-autotune",
        action="store_true",
        default=DEFAULT_PREFIX_CACHE_AUTOTUNE,
        help=(
            "Benchmark full forward/backward logical batches across prompt "
            "microbatch B and branch group K pairs, reject OOM pairs, and use "
            "the highest measured examples/second."
        ),
    )
    parser.add_argument(
        "--prefix-cache-autotune-max-branch-batch-size",
        type=int,
        default=DEFAULT_PREFIX_CACHE_AUTOTUNE_MAX_BRANCH_BATCH_SIZE,
        help="Largest branch group K considered by prefix-cache autotuning.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    )
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY_STEPS,
        help=(
            "Save intermediate adapter checkpoints every N optimizer steps. "
            "Use 0 to disable periodic checkpoints."
        ),
    )
    parser.add_argument(
        "--checkpoint-step",
        "--checkpoint-steps",
        dest="checkpoint_steps",
        action="append",
        default=None,
        help=(
            "Specific intermediate optimizer steps to save. Repeat or pass "
            "comma-separated values. Steps at or beyond the final step are "
            "omitted because the final adapter is saved separately."
        ),
    )
    parser.add_argument(
        "--checkpoint-adapter-only",
        action="store_true",
        help=(
            "Save adapter, tokenizer, and metadata at intermediate steps but "
            "omit optimizer/scheduler state. Use when checkpoints are for M10 "
            "evaluation rather than training resumption."
        ),
    )
    parser.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument(
        "--target-module",
        "--target-modules",
        dest="target_modules",
        action="append",
        default=None,
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument(
        "--load-in-16bit",
        action="store_true",
        help="Load bf16/fp16 weights for LoRA instead of quantized QLoRA.",
    )
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        help="Render prompts with model thinking enabled when supported.",
    )
    thinking_group.add_argument(
        "--disable-thinking",
        dest="enable_thinking",
        action="store_false",
        help="Render direct-answer prompts with model thinking disabled.",
    )
    parser.set_defaults(enable_thinking=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument("--reuse-teacher-distributions", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help=(
            "Show tqdm/Hugging Face progress bars. By default they are disabled "
            "to keep notebook output compact."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress INFO logs and only show success, warning, and error messages.",
    )
    parser.add_argument(
        "--disable_wandb",
        "--disable-wandb",
        "--no_wandb",
        "--no-wandb",
        action="store_true",
        dest="disable_wandb",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help=(
            "Enable W&B metric logging to this project. If omitted, WANDB_PROJECT "
            "from the environment is used when set."
        ),
    )
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--wandb-tag",
        "--wandb-tags",
        dest="wandb_tags",
        action="append",
        default=None,
    )
    parser.add_argument("--no-evaluate", action="store_true")
    parser.add_argument("--evaluate-after-each", action="store_true")
    parser.add_argument(
        "--eval-color",
        "--eval-colors",
        dest="eval_colors",
        action="append",
    )
    parser.add_argument(
        "--eval-question",
        "--eval-questions",
        dest="eval_questions",
        action="append",
    )
    parser.add_argument("--eval-questions-path", type=Path, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument(
        "--probability-mode",
        choices=["normalized", "raw"],
        default="normalized",
    )
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--eval-candidate-prefix", default="")
    parser.add_argument("--eval-assistant-prefill", default=None)
    return parser


def prepare_args(args: argparse.Namespace) -> argparse.Namespace:
    """Parse multi-value CLI fields after argparse."""
    args.traits = parse_traits(args.traits, DEFAULT_TRAIN_TRAITS)
    args.early_eval_traits = parse_traits(
        args.early_eval_traits,
        DEFAULT_EARLY_EVAL_TRAITS,
    )
    args.traits = order_traits_for_early_results(args.traits, args.early_eval_traits)
    args.eval_colors = parse_traits(args.eval_colors, DEFAULT_COLORS)
    args.eval_questions = load_questions(args)
    args.target_modules = parse_target_modules(args.target_modules)
    args.checkpoint_steps = parse_checkpoint_steps(args.checkpoint_steps)
    args.wandb_tags = parse_csv_values(args.wandb_tags, [])
    args.evaluate = not args.no_evaluate
    if args.all_prompts:
        args.max_prompts = None
    return args


def main() -> None:
    """Run the CLI."""
    parser = build_parser()
    args = prepare_args(parser.parse_args())
    validate_args(args)
    configure_console_output(quiet=args.quiet, show_progress=args.show_progress)
    run_experiment(args)


if __name__ == "__main__":
    main()
