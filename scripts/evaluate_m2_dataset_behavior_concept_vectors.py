#!/usr/bin/env python3
"""Run both parts of catalog metric M2 on a generic trait family.

Part 1 extracts paired dataset concept vectors from teacher-system versus
no-system prompts and causally injects the complete per-layer vectors into the
frozen base model. Part 2 trains one frozen-model behavior concept vector for
each target answer, then compares behavior and dataset vectors at every layer.

The implementation is intentionally resumable at the trait level. It is
compatible with Qwen3.5's multimodal Hugging Face wrapper while operating only
on the language-model backbone.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compute_favorite_color_probability_ratio_matrix import (  # noqa: E402
    RatioMatrixConfig,
    score_model,
)
from compute_teacher_vectors import (  # noqa: E402
    PROMPT_MODE_AS_IS,
    PROMPT_MODE_ONE_NUMBER,
    PROMPT_MODES,
    apply_prompt_mode,
    extract_layer_hidden_by_prompt,
    read_jsonl_prompt_selection,
)
from find_owl_output_distribution_entanglement import (  # noqa: E402
    build_chat_prompt,
    load_direct_model_and_tokenizer,
    parse_csv_values,
)
from print_behavior_dataset_cosine_matrix import projection_metrics  # noqa: E402
from score_birthday_year_probabilities import (  # noqa: E402
    resolve_device,
    resolve_torch_dtype,
)
from score_string_trait_number_distribution import slugify  # noqa: E402
from train_owl_in_context_vector import (  # noqa: E402
    TrainingConfig,
    register_steering_hooks,
    resolve_model_hidden_size,
    train_steering_vector,
)
from metric_batch_autotune import (  # noqa: E402
    BatchCandidate,
    autotune_batch_pair,
    candidate_grid,
)
from metric_exact_trie_scorer import autotune_exact_trie_batch_pair  # noqa: E402


DEFAULT_MODEL_ID = "Qwen/Qwen3.5-9B"
DEFAULT_NUM_DATASET_PROMPTS = 1024
DEFAULT_DATASET_BATCH_SIZE = 4
DEFAULT_EVALUATION_BATCH_SIZE = 8
DEFAULT_BEHAVIOR_BATCH_SIZE = 1
DEFAULT_BEHAVIOR_ITERATIONS = 100
DEFAULT_BEHAVIOR_LEARNING_RATE = 0.01
DEFAULT_BEHAVIOR_INIT_STD = 0.01
DEFAULT_PART1_ALPHA = 0.3
ANALYSIS_VERSION = 1
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(frozen=True, **DATACLASS_KW_ONLY)
class M2Config:
    """Configuration for the two-part M2 pilot."""

    model_id: str
    traits: list[str]
    dataset_jsonl: Path
    behavior_questions_path: Path
    evaluation_questions_path: Path
    ft_matrix_path: Path | None
    output_dir: Path
    system_prompt_template: str
    num_dataset_prompts: int
    dataset_batch_size: int
    evaluation_batch_size: int
    behavior_batch_size: int
    behavior_iterations: int
    behavior_learning_rate: float
    behavior_init_std: float
    part1_alpha: float
    comparison_layer: int | None
    device: str
    torch_dtype: str
    enable_thinking: bool | None
    skip_existing: bool
    no_plot: bool
    prompt_mode: str = PROMPT_MODE_ONE_NUMBER
    shared_behavior_cache_dir: Path | None = None
    use_prefix_cache: bool = False
    evaluation_prompt_batch_size: int = 8
    autotune_batches: bool = False
    max_dataset_batch_size: int = 32
    max_behavior_batch_size: int = 32
    max_evaluation_prompt_batch_size: int = 32
    max_evaluation_branch_batch_size: int = 64


def utc_now() -> str:
    """Return an ISO-8601 timestamp in UTC."""
    return datetime.now(timezone.utc).isoformat()


def load_string_list(path: Path, label: str) -> list[str]:
    """Load a non-empty unique JSON string list."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(data, list)
        or not data
        or not all(isinstance(value, str) and value.strip() for value in data)
    ):
        raise ValueError(f"{label} must be a non-empty JSON string list.")
    values = [value.strip() for value in data]
    if len(values) != len(set(values)):
        raise ValueError(f"{label} contains duplicate values.")
    return values


def sha256_file(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    """Write one JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_matrix_csv(
    path: Path,
    matrix: np.ndarray,
    *,
    row_labels: list[str],
    column_labels: list[str],
) -> None:
    """Write a labeled numeric matrix."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (len(row_labels), len(column_labels)):
        raise ValueError(f"Matrix shape {values.shape} does not match its labels.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["answer\\dataset_trait", *column_labels])
        for label, row in zip(row_labels, values):
            writer.writerow([label, *[f"{value:.12g}" for value in row]])


def read_matrix_subset(
    path: Path,
    *,
    row_labels: list[str],
    column_labels: list[str],
) -> np.ndarray:
    """Read a labeled CSV matrix and select requested rows and columns."""
    with path.open(encoding="utf-8", newline="") as handle:
        records = list(csv.reader(handle))
    if len(records) < 2 or len(records[0]) < 2:
        raise ValueError(f"Fine-tuning matrix is empty: {path}")
    available_columns = records[0][1:]
    column_index = {label: index for index, label in enumerate(available_columns)}
    row_index = {record[0]: record[1:] for record in records[1:]}
    missing_rows = [label for label in row_labels if label not in row_index]
    missing_columns = [label for label in column_labels if label not in column_index]
    if missing_rows or missing_columns:
        raise ValueError(
            "Fine-tuning matrix does not contain the M2 pilot labels: "
            f"missing rows={missing_rows}, missing columns={missing_columns}."
        )
    values = np.asarray(
        [
            [float(row_index[row][column_index[column]]) for column in column_labels]
            for row in row_labels
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Fine-tuning matrix contains non-finite values: {path}")
    return values


def double_center(matrix: np.ndarray) -> np.ndarray:
    """Subtract row and column means and restore the grand mean."""
    values = np.asarray(matrix, dtype=np.float64)
    return (
        values
        - values.mean(axis=1, keepdims=True)
        - values.mean(axis=0, keepdims=True)
        + values.mean()
    )


def pearson(first: np.ndarray, second: np.ndarray) -> float:
    """Compute Pearson correlation, returning NaN for a constant input."""
    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or left.size < 2:
        raise ValueError("Pearson inputs must have matching nontrivial shapes.")
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def ft_agreement(
    predicted: np.ndarray,
    fine_tuning: np.ndarray | None,
) -> dict[str, float] | None:
    """Compare one predicted matrix with the final fine-tuning matrix."""
    if fine_tuning is None:
        return None
    values = np.asarray(predicted, dtype=np.float64)
    if values.shape != fine_tuning.shape:
        raise ValueError("Predicted and fine-tuning matrix shapes do not match.")
    off_diagonal = ~np.eye(values.shape[0], dtype=bool)
    return {
        "raw_all_cells_pearson": pearson(values, fine_tuning),
        "raw_off_diagonal_pearson": pearson(
            values[off_diagonal], fine_tuning[off_diagonal]
        ),
        "double_centered_all_cells_pearson": pearson(
            double_center(values), double_center(fine_tuning)
        ),
    }


def write_layer_summary_csv(path: Path, layer_rows: list[dict[str, Any]]) -> None:
    """Write compact Part 2 layer diagnostics without the embedded matrices."""
    fieldnames = [
        "layer",
        "layer_type",
        "mean_diagonal_cosine",
        "mean_off_diagonal_cosine",
        "diagonal_cosine_advantage",
        "mean_diagonal_signed_projection",
        "cosine_ft_raw_pearson",
        "cosine_ft_double_centered_pearson",
        "projection_ft_raw_pearson",
        "projection_ft_double_centered_pearson",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in layer_rows:
            cosine_ft = row["cosine_agreement_with_final_fine_tuning"] or {}
            projection_ft = (
                row["signed_projection_agreement_with_final_fine_tuning"] or {}
            )
            writer.writerow(
                {
                    "layer": row["layer"],
                    "layer_type": row["layer_type"],
                    "mean_diagonal_cosine": row["mean_diagonal_cosine"],
                    "mean_off_diagonal_cosine": row["mean_off_diagonal_cosine"],
                    "diagonal_cosine_advantage": row["diagonal_cosine_advantage"],
                    "mean_diagonal_signed_projection": row[
                        "mean_diagonal_signed_projection"
                    ],
                    "cosine_ft_raw_pearson": cosine_ft.get("raw_all_cells_pearson"),
                    "cosine_ft_double_centered_pearson": cosine_ft.get(
                        "double_centered_all_cells_pearson"
                    ),
                    "projection_ft_raw_pearson": projection_ft.get(
                        "raw_all_cells_pearson"
                    ),
                    "projection_ft_double_centered_pearson": projection_ft.get(
                        "double_centered_all_cells_pearson"
                    ),
                }
            )


def validate_config(config: M2Config) -> None:
    """Validate inexpensive inputs before loading the model."""
    if len(config.traits) < 2 or len(config.traits) != len(set(config.traits)):
        raise ValueError("M2 requires at least two unique traits.")
    for path in (
        config.dataset_jsonl,
        config.behavior_questions_path,
        config.evaluation_questions_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required M2 input does not exist: {path}")
    positive_integers = {
        "num_dataset_prompts": config.num_dataset_prompts,
        "dataset_batch_size": config.dataset_batch_size,
        "evaluation_batch_size": config.evaluation_batch_size,
        "behavior_batch_size": config.behavior_batch_size,
        "behavior_iterations": config.behavior_iterations,
        "evaluation_prompt_batch_size": config.evaluation_prompt_batch_size,
        "max_dataset_batch_size": config.max_dataset_batch_size,
        "max_behavior_batch_size": config.max_behavior_batch_size,
        "max_evaluation_prompt_batch_size": (
            config.max_evaluation_prompt_batch_size
        ),
        "max_evaluation_branch_batch_size": (
            config.max_evaluation_branch_batch_size
        ),
    }
    if any(value <= 0 for value in positive_integers.values()):
        raise ValueError(f"M2 count settings must be positive: {positive_integers}")
    if config.prompt_mode not in PROMPT_MODES:
        raise ValueError(f"Unsupported M2 prompt mode: {config.prompt_mode}")
    finite_positive = (
        config.behavior_learning_rate,
        config.part1_alpha,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in finite_positive):
        raise ValueError("M2 learning rate and Part 1 alpha must be positive.")
    if not math.isfinite(config.behavior_init_std) or config.behavior_init_std < 0.0:
        raise ValueError("Behavior-vector initialization std must be non-negative.")
    for trait in config.traits:
        try:
            rendered = config.system_prompt_template.format(trait=trait)
        except (KeyError, ValueError) as exc:
            raise ValueError("The system prompt template must accept {trait}.") from exc
        if not rendered.strip():
            raise ValueError(f"The rendered system prompt is empty for {trait!r}.")


def dataset_vector_paths(config: M2Config, trait: str) -> tuple[Path, Path]:
    """Return array and metadata paths for one dataset vector."""
    stem = slugify(trait)
    root = config.output_dir / "dataset_vectors"
    return (
        root / f"{stem}_v_teacher_by_layer.npy",
        root / f"{stem}_v_teacher.metadata.json",
    )


def behavior_vector_paths(config: M2Config, trait: str) -> tuple[Path, Path, Path]:
    """Return array, metadata, and loss-plot paths for one behavior vector."""
    stem = slugify(trait)
    root = (
        config.shared_behavior_cache_dir
        if config.shared_behavior_cache_dir is not None
        else config.output_dir / "behavior_vectors"
    )
    return (
        root / f"trained_{stem}_vector.npy",
        root / f"trained_{stem}_vector.metadata.json",
        root / f"trained_{stem}_loss.png",
    )


def render_dataset_prompt(
    tokenizer: Any,
    *,
    user_prompt: str,
    system_prompt: str | None,
    enable_thinking: bool | None,
) -> str:
    """Render one M2 extraction prompt with consistent thinking behavior."""
    return build_chat_prompt(
        tokenizer,
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        enable_thinking=enable_thinking,
    )


def dataset_cache_identity(config: M2Config, trait: str) -> dict[str, Any]:
    """Return the exact cache identity for one dataset concept vector."""
    return {
        "analysis_version": ANALYSIS_VERSION,
        "model_id": config.model_id,
        "torch_dtype": config.torch_dtype,
        "trait": trait,
        "system_prompt": config.system_prompt_template.format(trait=trait),
        "neutral_prompt": None,
        "dataset_sha256": sha256_file(config.dataset_jsonl),
        "num_prompts": config.num_dataset_prompts,
        "prompt_selection": "first",
        "prompt_mode": config.prompt_mode,
        "pooling": "last_token",
        "enable_thinking": config.enable_thinking,
    }


def load_cached_dataset_vector(
    config: M2Config,
    trait: str,
) -> np.ndarray | None:
    """Load one compatible cached dataset vector, when present."""
    vector_path, metadata_path = dataset_vector_paths(config, trait)
    if (
        not config.skip_existing
        or not vector_path.is_file()
        or not metadata_path.is_file()
    ):
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("cache_identity") != dataset_cache_identity(config, trait):
        logger.warning("Ignoring incompatible dataset-vector cache for {}", trait)
        return None
    vector = np.asarray(np.load(vector_path), dtype=np.float32)
    if vector.ndim != 2 or not np.all(np.isfinite(vector)):
        logger.warning("Ignoring invalid dataset-vector cache for {}", trait)
        return None
    logger.success("Reusing dataset concept vector for {}", trait)
    return vector


def extract_one_dataset_vector(
    config: M2Config,
    trait: str,
    prompts: list[str],
    source_indices: list[int],
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
) -> np.ndarray:
    """Estimate E[h(system+prompt)-h(no-system+prompt)] for one trait."""
    cached = load_cached_dataset_vector(config, trait)
    if cached is not None:
        return cached

    accumulator: np.ndarray | None = None
    layer_indices: list[int] | None = None
    system_prompt = config.system_prompt_template.format(trait=trait)
    logger.info(
        "Extracting dataset concept vector for {} on {} prompts", trait, len(prompts)
    )
    for start in range(0, len(prompts), config.dataset_batch_size):
        batch = prompts[start : start + config.dataset_batch_size]
        neutral = [
            render_dataset_prompt(
                tokenizer,
                user_prompt=prompt,
                system_prompt=None,
                enable_thinking=config.enable_thinking,
            )
            for prompt in batch
        ]
        positive = [
            render_dataset_prompt(
                tokenizer,
                user_prompt=prompt,
                system_prompt=system_prompt,
                enable_thinking=config.enable_thinking,
            )
            for prompt in batch
        ]
        hidden, batch_layers, _hidden_size = extract_layer_hidden_by_prompt(
            model,
            tokenizer,
            neutral + positive,
            device=device,
            max_length=None,
            add_special_tokens=False,
            include_embedding_layer=False,
            layer_start=None,
            layer_end=None,
            token_pooling="last_token",
        )
        count = len(batch)
        differences = hidden[count:] - hidden[:count]
        if accumulator is None:
            accumulator = np.zeros(differences.shape[1:], dtype=np.float64)
            layer_indices = batch_layers
        elif layer_indices != batch_layers:
            raise RuntimeError("Dataset-vector layer indexes changed between batches.")
        accumulator += differences.astype(np.float64).sum(axis=0)
        processed = start + count
        if processed == len(prompts) or processed % 128 == 0:
            logger.info(
                "{} dataset vector: {}/{} pairs", trait, processed, len(prompts)
            )

    if accumulator is None or layer_indices is None:
        raise RuntimeError(f"No dataset-vector activations were extracted for {trait}.")
    vector = (accumulator / len(prompts)).astype(np.float32)
    vector_path, metadata_path = dataset_vector_paths(config, trait)
    vector_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(vector_path, vector)
    metadata = {
        "method": "dataset_concept_vector_paired_prompt_activation_difference",
        "formula": "d[t,l] = E_x[h_l(system_t + x) - h_l(no_system + x)]",
        "cache_identity": dataset_cache_identity(config, trait),
        "trait": trait,
        "model_layer_indices": layer_indices,
        "hidden_size": int(vector.shape[1]),
        "shape": list(vector.shape),
        "source_indices": source_indices,
        "prompt_sample": prompts[:5],
        "vector_path": str(vector_path),
        "created_at": utc_now(),
    }
    write_json(metadata_path, metadata)
    logger.success("Saved dataset concept vector for {}", trait)
    return vector


def extract_dataset_vectors(
    config: M2Config,
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
) -> dict[str, np.ndarray]:
    """Extract or load all requested dataset concept vectors."""
    selection = read_jsonl_prompt_selection(
        config.dataset_jsonl,
        prompt_key="prompt",
        limit=config.num_dataset_prompts,
        selection="first",
        random_seed=0,
    )
    prompts = [
        apply_prompt_mode(prompt, config.prompt_mode)
        for prompt in selection.prompts
    ]
    return {
        trait: extract_one_dataset_vector(
            config,
            trait,
            prompts,
            selection.source_indices,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        for trait in config.traits
    }


def ratio_config(
    config: M2Config, evaluation_questions: list[str]
) -> RatioMatrixConfig:
    """Build the shared exact-complete-answer scoring configuration."""
    return RatioMatrixConfig(
        colors=config.traits,
        fine_tune_traits=config.traits,
        questions=evaluation_questions,
        model_json_template="unused_{trait}.json",
        base_model_id=config.model_id,
        output_path=config.output_dir / "unused.json",
        csv_output_path=config.output_dir / "unused.csv",
        heatmap_path=config.output_dir / "unused.png",
        probability_mode="raw",
        epsilon=1e-30,
        candidate_prefix="",
        assistant_prefill=None,
        batch_size=config.evaluation_batch_size,
        device=config.device,
        torch_dtype=config.torch_dtype,
        enable_thinking=config.enable_thinking,
        use_prefix_cache=config.use_prefix_cache,
        prompt_batch_size=config.evaluation_prompt_batch_size,
        branch_batch_size=config.evaluation_batch_size,
    )


def score_cache_identity(config: M2Config, trait: str | None) -> dict[str, Any]:
    """Return a cache identity for base or steered Part 1 scores."""
    identity: dict[str, Any] = {
        "analysis_version": ANALYSIS_VERSION,
        "model_id": config.model_id,
        "torch_dtype": config.torch_dtype,
        "traits": config.traits,
        "evaluation_questions_sha256": sha256_file(config.evaluation_questions_path),
        "enable_thinking": config.enable_thinking,
        "part1_alpha": 0.0 if trait is None else config.part1_alpha,
        "apply_to": "all_tokens",
        "l2_norm_matching": False,
        "trait": trait,
    }
    if trait is not None:
        vector_path, _metadata_path = dataset_vector_paths(config, trait)
        identity["dataset_vector_sha256"] = sha256_file(vector_path)
    return identity


def load_score_cache(path: Path, identity: dict[str, Any]) -> dict[str, Any] | None:
    """Load compatible exact-answer score output."""
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    scores = payload.get("scores")
    if payload.get("identity") != identity or not isinstance(scores, dict):
        return None
    return scores


def save_score_cache(
    path: Path, identity: dict[str, Any], scores: dict[str, Any]
) -> None:
    """Save exact-answer scores with provenance."""
    write_json(path, {"identity": identity, "scores": scores})


def mean_score_values(
    scores: dict[str, Any],
    traits: list[str],
    field: str,
) -> np.ndarray:
    """Average one score field over evaluation questions."""
    return np.asarray(
        [
            np.mean(
                [
                    question["colors"][trait][field]
                    for question in scores["question_scores"]
                ]
            )
            for trait in traits
        ],
        dtype=np.float64,
    )


def mean_log_normalized_values(
    scores: dict[str, Any],
    traits: list[str],
) -> np.ndarray:
    """Average candidate-normalized log probabilities over questions."""
    tiny = np.finfo(np.float64).tiny
    return np.asarray(
        [
            np.mean(
                [
                    math.log(
                        max(
                            question["colors"][trait][
                                "normalized_probability_over_colors"
                            ],
                            tiny,
                        )
                    )
                    for question in scores["question_scores"]
                ]
            )
            for trait in traits
        ],
        dtype=np.float64,
    )


def plot_matrix(
    path: Path,
    matrix: np.ndarray,
    *,
    labels: list[str],
    title: str,
    colorbar_label: str,
) -> None:
    """Save a compact diverging heatmap with outlined matched cells."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    values = np.asarray(matrix, dtype=np.float64)
    limit = max(float(np.max(np.abs(values))), 1e-12)
    size = max(8.0, 0.65 * len(labels) + 2.5)
    fig, axis = plt.subplots(figsize=(size, size * 0.88))
    image = axis.imshow(values, cmap="coolwarm", vmin=-limit, vmax=limit)
    axis.set_title(title)
    axis.set_xlabel("Dataset / teacher trait")
    axis.set_ylabel("Evaluated behavior answer")
    axis.set_xticks(range(len(labels)), labels, rotation=50, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    for index in range(len(labels)):
        axis.add_patch(
            Rectangle(
                (index - 0.5, index - 0.5),
                1,
                1,
                fill=False,
                edgecolor="#f2bf22",
                linewidth=2.0,
            )
        )
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            axis.text(
                column,
                row,
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color="white" if abs(value) > 0.55 * limit else "black",
            )
    fig.colorbar(image, ax=axis, label=colorbar_label)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def run_part1(
    config: M2Config,
    dataset_vectors: dict[str, np.ndarray],
    evaluation_questions: list[str],
    fine_tuning_matrix: np.ndarray | None,
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
) -> dict[str, Any]:
    """Causally evaluate all-layer natural dataset-vector steering."""
    import torch

    root = config.output_dir / "part1_causal_dataset_vectors"
    cache_root = root / "score_cache"
    scorer_config = ratio_config(config, evaluation_questions)
    base_path = cache_root / "base.json"
    base_identity = score_cache_identity(config, None)
    base_scores = (
        load_score_cache(base_path, base_identity) if config.skip_existing else None
    )
    if base_scores is None:
        base_scores = score_model(
            model_label="base",
            model=model,
            tokenizer=tokenizer,
            device=device,
            config=scorer_config,
        )
        save_score_cache(base_path, base_identity, base_scores)
    else:
        logger.success("Reusing unsteered Part 1 score cache")

    condition_scores: dict[str, dict[str, Any]] = {}
    for trait in config.traits:
        path = cache_root / f"steered_{slugify(trait)}.json"
        identity = score_cache_identity(config, trait)
        cached = load_score_cache(path, identity) if config.skip_existing else None
        if cached is not None:
            logger.success("Reusing Part 1 score cache for {}", trait)
            condition_scores[trait] = cached
            continue
        vector = dataset_vectors[trait]
        handles = register_steering_hooks(
            model,
            torch.as_tensor(vector, dtype=torch.float32),
            layer_indices=list(range(vector.shape[0])),
            alpha=config.part1_alpha,
            apply_to="all_tokens",
        )
        try:
            scores = score_model(
                model_label=f"dataset-vector:{trait}",
                model=model,
                tokenizer=tokenizer,
                device=device,
                config=scorer_config,
            )
        finally:
            for handle in handles:
                handle.remove()
        save_score_cache(path, identity, scores)
        condition_scores[trait] = scores

    base_logprob = mean_score_values(base_scores, config.traits, "total_logprob")
    base_normalized = mean_log_normalized_values(base_scores, config.traits)
    raw_lift = np.stack(
        [
            mean_score_values(condition_scores[trait], config.traits, "total_logprob")
            - base_logprob
            for trait in config.traits
        ],
        axis=1,
    )
    normalized_lift = np.stack(
        [
            mean_log_normalized_values(condition_scores[trait], config.traits)
            - base_normalized
            for trait in config.traits
        ],
        axis=1,
    )
    write_matrix_csv(
        root / "causal_raw_logprob_lift.csv",
        raw_lift,
        row_labels=config.traits,
        column_labels=config.traits,
    )
    write_matrix_csv(
        root / "causal_candidate_normalized_logprob_lift.csv",
        normalized_lift,
        row_labels=config.traits,
        column_labels=config.traits,
    )
    if not config.no_plot:
        plot_matrix(
            root / "causal_raw_logprob_lift.png",
            raw_lift,
            labels=config.traits,
            title=f"M2 Part 1: all-layer dataset-vector steering (α={config.part1_alpha:g})",
            colorbar_label="mean complete-answer log-probability lift",
        )
    summary = {
        "definition": "h_l <- h_l + alpha*d[t,l] at every layer and token",
        "alpha": config.part1_alpha,
        "matrix_orientation": "rows=evaluated city answers; columns=dataset vectors",
        "mean_diagonal_raw_logprob_lift": float(np.mean(np.diag(raw_lift))),
        "mean_off_diagonal_raw_logprob_lift": float(
            np.mean(raw_lift[~np.eye(len(config.traits), dtype=bool)])
        ),
        "positive_diagonal_count": int(np.sum(np.diag(raw_lift) > 0.0)),
        "agreement_with_final_fine_tuning": ft_agreement(raw_lift, fine_tuning_matrix),
        "raw_logprob_lift": raw_lift.tolist(),
        "candidate_normalized_logprob_lift": normalized_lift.tolist(),
    }
    write_json(root / "summary.json", summary)
    return summary


def behavior_cache_identity(
    config: M2Config,
    trait: str,
    behavior_questions: list[str],
) -> dict[str, Any]:
    """Return the exact cache identity for one trained behavior vector."""
    return {
        "analysis_version": ANALYSIS_VERSION,
        "model_id": config.model_id,
        "torch_dtype": config.torch_dtype,
        "trait": trait,
        "questions": behavior_questions,
        "iterations": config.behavior_iterations,
        "learning_rate": config.behavior_learning_rate,
        "init_std": config.behavior_init_std,
        "batch_size": config.behavior_batch_size,
        "layer_start": 2,
        "layer_end_offset": 2,
        "apply_to": "all_tokens",
        "enable_thinking": config.enable_thinking,
    }


def plot_loss(path: Path, losses: list[float], trait: str) -> None:
    """Save a behavior-vector optimization loss curve."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.plot(range(1, len(losses) + 1), losses, linewidth=1.6)
    axis.set_xlabel("Behavior-vector optimization iteration")
    axis.set_ylabel("Target-token cross entropy")
    axis.set_title(f"M2 behavior vector: {trait}")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def train_one_behavior_vector(
    config: M2Config,
    trait: str,
    behavior_questions: list[str],
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
) -> np.ndarray:
    """Train or load one uniform behavior concept vector."""
    vector_path, metadata_path, loss_path = behavior_vector_paths(config, trait)
    identity = behavior_cache_identity(config, trait, behavior_questions)
    if config.skip_existing and vector_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("cache_identity") == identity:
            vector = np.asarray(np.load(vector_path), dtype=np.float32).reshape(-1)
            if np.all(np.isfinite(vector)):
                logger.success("Reusing behavior concept vector for {}", trait)
                return vector

    training_config = TrainingConfig(
        model_id=config.model_id,
        output_path=vector_path,
        layer_output_path=None,
        flat_layer_output_path=None,
        metadata_path=metadata_path,
        target_text=trait,
        target_prefix=" ",
        questions=behavior_questions,
        iterations=config.behavior_iterations,
        learning_rate=config.behavior_learning_rate,
        init_std=config.behavior_init_std,
        alpha=1.0,
        vector_mode="uniform",
        batch_size=config.behavior_batch_size,
        device=config.device,
        torch_dtype=config.torch_dtype,
        max_length=None,
        random_seed=0,
        layer_start=2,
        layer_end=None,
        layer_end_offset=2,
        use_chat_template=True,
        add_generation_prompt=True,
        apply_to="all_tokens",
        output_dtype="float32",
        log_every=10,
        question_set="city_file",
        enable_thinking=config.enable_thinking,
    )
    result = train_steering_vector(training_config, model, tokenizer, device)
    vector = result.vector.numpy().astype(np.float32).reshape(-1)
    vector_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(vector_path, vector)
    metadata = {
        "method": "behavior_concept_vector_frozen_model_cross_entropy",
        "cache_identity": identity,
        "model_parameters_frozen": True,
        "target_text": trait,
        "target_prefix": " ",
        "hidden_size": result.hidden_size,
        "layer_indices": result.layer_indices,
        "target_token_count": result.target_token_count,
        "initial_loss": result.loss_history[0],
        "final_loss": result.loss_history[-1],
        "loss_history": result.loss_history,
        "vector_path": str(vector_path),
        "loss_plot_path": str(loss_path),
        "created_at": utc_now(),
    }
    write_json(metadata_path, metadata)
    if not config.no_plot:
        plot_loss(loss_path, result.loss_history, trait)
    logger.success("Saved behavior concept vector for {}", trait)
    return vector


def layer_comparison(
    traits: list[str],
    behavior_vectors: dict[str, np.ndarray],
    dataset_vectors: dict[str, np.ndarray],
    layer: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute behavior-by-dataset cosine and signed projection matrices."""
    cosine = np.zeros((len(traits), len(traits)), dtype=np.float64)
    projection = np.zeros_like(cosine)
    for row, behavior_trait in enumerate(traits):
        for column, dataset_trait in enumerate(traits):
            metrics = projection_metrics(
                behavior_vectors[behavior_trait],
                dataset_vectors[dataset_trait][layer],
            )
            cosine[row, column] = metrics["cosine"]
            projection[row, column] = metrics["projection_length"]
    return cosine, projection


def run_part2(
    config: M2Config,
    dataset_vectors: dict[str, np.ndarray],
    behavior_questions: list[str],
    fine_tuning_matrix: np.ndarray | None,
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
) -> dict[str, Any]:
    """Train behavior vectors and compare them with dataset vectors."""
    behavior_vectors = {
        trait: train_one_behavior_vector(
            config,
            trait,
            behavior_questions,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        for trait in config.traits
    }
    layer_counts = {vector.shape[0] for vector in dataset_vectors.values()}
    if len(layer_counts) != 1:
        raise ValueError("Dataset vectors do not share one layer count.")
    num_layers = layer_counts.pop()
    text_config = getattr(model.config, "text_config", model.config)
    layer_types = list(getattr(text_config, "layer_types", []) or [])
    if layer_types and len(layer_types) != num_layers:
        raise ValueError("Qwen layer-type metadata does not match the vector layers.")
    comparison_layer = (
        num_layers - 2 if config.comparison_layer is None else config.comparison_layer
    )
    if comparison_layer < 0 or comparison_layer >= num_layers:
        raise ValueError(
            f"Comparison layer {comparison_layer} is outside 0..{num_layers - 1}."
        )

    layer_rows = []
    comparison_cosine: np.ndarray | None = None
    comparison_projection: np.ndarray | None = None
    for layer in range(num_layers):
        cosine, projection = layer_comparison(
            config.traits,
            behavior_vectors,
            dataset_vectors,
            layer,
        )
        off_diagonal = ~np.eye(len(config.traits), dtype=bool)
        layer_rows.append(
            {
                "layer": layer,
                "layer_type": layer_types[layer] if layer_types else None,
                "mean_diagonal_cosine": float(np.mean(np.diag(cosine))),
                "mean_off_diagonal_cosine": float(np.mean(cosine[off_diagonal])),
                "diagonal_cosine_advantage": float(
                    np.mean(np.diag(cosine)) - np.mean(cosine[off_diagonal])
                ),
                "mean_diagonal_signed_projection": float(np.mean(np.diag(projection))),
                "cosine_agreement_with_final_fine_tuning": ft_agreement(
                    cosine, fine_tuning_matrix
                ),
                "signed_projection_agreement_with_final_fine_tuning": ft_agreement(
                    projection, fine_tuning_matrix
                ),
                "cosine_matrix": cosine.tolist(),
                "signed_projection_matrix": projection.tolist(),
            }
        )
        if layer == comparison_layer:
            comparison_cosine = cosine
            comparison_projection = projection
    if comparison_cosine is None or comparison_projection is None:
        raise RuntimeError("The requested Part 2 comparison layer was not computed.")

    root = config.output_dir / "part2_behavior_dataset_alignment"
    write_matrix_csv(
        root / f"cosine_layer_{comparison_layer}.csv",
        comparison_cosine,
        row_labels=config.traits,
        column_labels=config.traits,
    )
    write_matrix_csv(
        root / f"signed_projection_layer_{comparison_layer}.csv",
        comparison_projection,
        row_labels=config.traits,
        column_labels=config.traits,
    )
    if not config.no_plot:
        plot_matrix(
            root / f"cosine_layer_{comparison_layer}.png",
            comparison_cosine,
            labels=config.traits,
            title=f"M2 Part 2: behavior × dataset cosine, layer {comparison_layer}",
            colorbar_label="signed cosine similarity",
        )
        plot_matrix(
            root / f"signed_projection_layer_{comparison_layer}.png",
            comparison_projection,
            labels=config.traits,
            title=f"M2 Part 2: signed projection, layer {comparison_layer}",
            colorbar_label="dataset vector projected onto behavior direction",
        )
    best_layer = max(layer_rows, key=lambda row: row["diagonal_cosine_advantage"])
    write_layer_summary_csv(root / "layer_summary.csv", layer_rows)
    summary = {
        "definition": "compare each dataset vector d[t,l] with independently trained behavior vector b[u]",
        "matrix_orientation": "rows=behavior-vector target cities; columns=dataset-vector cities",
        "comparison_layer": comparison_layer,
        "comparison_layer_type": (
            layer_types[comparison_layer] if layer_types else None
        ),
        "comparison_layer_rule": "L-2 unless explicitly supplied",
        "num_model_layers": num_layers,
        "best_exploratory_layer_by_diagonal_cosine_advantage": best_layer["layer"],
        "best_exploratory_diagonal_cosine_advantage": best_layer[
            "diagonal_cosine_advantage"
        ],
        "layer_results": layer_rows,
    }
    write_json(root / "all_layers.json", summary)
    return summary


def autotune_m2_batches(
    config: M2Config,
    *,
    behavior_questions: list[str],
    evaluation_questions: list[str],
    model: Any,
    tokenizer: Any,
    device: Any,
) -> tuple[M2Config, dict[str, Any]]:
    """Tune M2 activation, gradient, and exact-trie batches on this GPU."""
    selection = read_jsonl_prompt_selection(
        config.dataset_jsonl,
        prompt_key="prompt",
        limit=config.num_dataset_prompts,
        selection="first",
        random_seed=0,
    )
    all_prompts = [
        apply_prompt_mode(prompt, config.prompt_mode) for prompt in selection.prompts
    ]
    representative_trait = max(config.traits, key=len)
    sample_system = config.system_prompt_template.format(trait=representative_trait)

    # Tune against the longest real prompts, not merely the first prompts in
    # the file.  That makes the selected activation batch robust to the whole
    # deterministic M2 sample while still searching for the fastest fitting B.
    prompt_lengths: list[tuple[int, str]] = []
    for prompt in all_prompts:
        neutral = render_dataset_prompt(
            tokenizer,
            user_prompt=prompt,
            system_prompt=None,
            enable_thinking=config.enable_thinking,
        )
        positive = render_dataset_prompt(
            tokenizer,
            user_prompt=prompt,
            system_prompt=sample_system,
            enable_thinking=config.enable_thinking,
        )
        length = max(
            len(tokenizer.encode(neutral, add_special_tokens=False)),
            len(tokenizer.encode(positive, add_special_tokens=False)),
        )
        prompt_lengths.append((length, prompt))
    prompt_lengths.sort(key=lambda item: item[0], reverse=True)
    sample_prompts = [
        prompt
        for _length, prompt in prompt_lengths[: config.max_dataset_batch_size]
    ]

    def dataset_benchmark(candidate: BatchCandidate) -> int:
        count = min(candidate.prompt_batch_size, len(sample_prompts))
        prompts = sample_prompts[:count]
        neutral = [
            render_dataset_prompt(
                tokenizer,
                user_prompt=prompt,
                system_prompt=None,
                enable_thinking=config.enable_thinking,
            )
            for prompt in prompts
        ]
        positive = [
            render_dataset_prompt(
                tokenizer,
                user_prompt=prompt,
                system_prompt=sample_system,
                enable_thinking=config.enable_thinking,
            )
            for prompt in prompts
        ]
        extract_layer_hidden_by_prompt(
            model,
            tokenizer,
            neutral + positive,
            device=device,
            max_length=None,
            add_special_tokens=False,
            include_embedding_layer=False,
            layer_start=None,
            layer_end=None,
            token_pooling="last_token",
        )
        return 2 * count

    dataset_choice, dataset_payload = autotune_batch_pair(
        candidates=candidate_grid(
            max_prompt_batch_size=min(
                config.max_dataset_batch_size,
                len(sample_prompts),
            ),
            max_branch_batch_size=1,
            include_prompt_sizes=(config.dataset_batch_size,),
        ),
        benchmark=dataset_benchmark,
        cache_path=config.output_dir / "autotune" / "dataset_batch.json",
        identity={
            "kind": "m2_dataset_activation_v2_longest_prompts",
            "model_id": config.model_id,
            "torch_dtype": config.torch_dtype,
            "prompt_mode": config.prompt_mode,
            "dataset_sha256": sha256_file(config.dataset_jsonl),
            "num_dataset_prompts": config.num_dataset_prompts,
            "representative_trait": representative_trait,
        },
        reuse_cache=config.skip_existing,
    )

    behavior_maximum = min(
        config.max_behavior_batch_size,
        len(behavior_questions),
    )

    def behavior_benchmark(candidate: BatchCandidate) -> int:
        probe = TrainingConfig(
            model_id=config.model_id,
            output_path=config.output_dir / "autotune" / "unused.npy",
            layer_output_path=None,
            flat_layer_output_path=None,
            metadata_path=None,
            target_text=representative_trait,
            target_prefix=" ",
            questions=behavior_questions,
            iterations=1,
            learning_rate=config.behavior_learning_rate,
            init_std=config.behavior_init_std,
            alpha=1.0,
            vector_mode="uniform",
            batch_size=candidate.prompt_batch_size,
            device=config.device,
            torch_dtype=config.torch_dtype,
            max_length=None,
            random_seed=0,
            layer_start=2,
            layer_end=None,
            layer_end_offset=2,
            use_chat_template=True,
            add_generation_prompt=True,
            apply_to="all_tokens",
            output_dtype="float32",
            log_every=0,
            question_set="m2_autotune",
            enable_thinking=config.enable_thinking,
        )
        train_steering_vector(probe, model, tokenizer, device)
        return len(behavior_questions)

    behavior_autotune_root = (
        config.shared_behavior_cache_dir
        if config.shared_behavior_cache_dir is not None
        else config.output_dir / "autotune"
    )
    behavior_choice, behavior_payload = autotune_batch_pair(
        candidates=candidate_grid(
            max_prompt_batch_size=behavior_maximum,
            max_branch_batch_size=1,
            include_prompt_sizes=(config.behavior_batch_size,),
        ),
        benchmark=behavior_benchmark,
        cache_path=behavior_autotune_root / "batch_autotune.json",
        identity={
            "kind": "m2_behavior_gradient_v1",
            "model_id": config.model_id,
            "torch_dtype": config.torch_dtype,
            "questions_sha256": sha256_file(config.behavior_questions_path),
            "vector_mode": "uniform",
        },
        reuse_cache=config.skip_existing,
    )

    eval_prompts = [
        build_chat_prompt(
            tokenizer,
            user_prompt=question,
            enable_thinking=config.enable_thinking,
        )
        for question in evaluation_questions
    ]
    exact_choice, exact_payload = autotune_exact_trie_batch_pair(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt_texts=eval_prompts,
        target_texts=config.traits,
        cache_path=config.output_dir / "autotune" / "exact_trie_batch_pair.json",
        model_id=config.model_id,
        max_prompt_batch_size=config.max_evaluation_prompt_batch_size,
        max_branch_batch_size=config.max_evaluation_branch_batch_size,
        reuse_cache=config.skip_existing,
    )
    tuned = replace(
        config,
        dataset_batch_size=dataset_choice.prompt_batch_size,
        behavior_batch_size=behavior_choice.prompt_batch_size,
        evaluation_prompt_batch_size=exact_choice.prompt_batch_size,
        evaluation_batch_size=exact_choice.branch_batch_size,
        use_prefix_cache=True,
    )
    return tuned, {
        "dataset": dataset_payload,
        "behavior": behavior_payload,
        "exact_trie": exact_payload,
    }


def run(config: M2Config) -> dict[str, Any]:
    """Run both M2 parts with one frozen base-model load."""
    import torch

    validate_config(config)
    behavior_questions = load_string_list(
        config.behavior_questions_path,
        "--behavior-questions-path",
    )
    evaluation_questions = load_string_list(
        config.evaluation_questions_path,
        "--evaluation-questions-path",
    )
    fine_tuning_matrix = None
    if config.ft_matrix_path is not None and config.ft_matrix_path.is_file():
        fine_tuning_matrix = read_matrix_subset(
            config.ft_matrix_path,
            row_labels=config.traits,
            column_labels=config.traits,
        )
        logger.success(
            "Will compare M2 with final fine-tuning matrix {}",
            config.ft_matrix_path,
        )
    elif config.ft_matrix_path is not None:
        logger.info(
            "Final fine-tuning matrix is not available yet; skipping agreement "
            "statistics: {}",
            config.ft_matrix_path,
        )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(torch, config.device)
    torch_dtype = resolve_torch_dtype(torch, config.torch_dtype)
    logger.info("Loading frozen base model {}", config.model_id)
    model, tokenizer = load_direct_model_and_tokenizer(
        model_id=config.model_id,
        device=device,
        torch_dtype=torch_dtype,
    )
    hidden_size = resolve_model_hidden_size(model)
    started = time.perf_counter()
    autotune_payload: dict[str, Any] | None = None
    try:
        if config.autotune_batches:
            config, autotune_payload = autotune_m2_batches(
                config,
                behavior_questions=behavior_questions,
                evaluation_questions=evaluation_questions,
                model=model,
                tokenizer=tokenizer,
                device=device,
            )
        dataset_vectors = extract_dataset_vectors(
            config,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        if any(vector.shape[1] != hidden_size for vector in dataset_vectors.values()):
            raise ValueError("A dataset vector does not match the model hidden size.")
        part1 = run_part1(
            config,
            dataset_vectors,
            evaluation_questions,
            fine_tuning_matrix,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        part2 = run_part2(
            config,
            dataset_vectors,
            behavior_questions,
            fine_tuning_matrix,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "status": "completed",
        "completed_at": utc_now(),
        "duration_seconds": time.perf_counter() - started,
        "model_id": config.model_id,
        "hidden_size": hidden_size,
        "traits": config.traits,
        "batch_autotune": autotune_payload,
        "fine_tuning_comparison_available": fine_tuning_matrix is not None,
        "part1_summary_path": str(
            config.output_dir / "part1_causal_dataset_vectors" / "summary.json"
        ),
        "part2_summary_path": str(
            config.output_dir / "part2_behavior_dataset_alignment" / "all_layers.json"
        ),
        "part1_compact_summary": {
            key: value for key, value in part1.items() if not key.endswith("lift")
        },
        "part2_compact_summary": {
            key: value for key, value in part2.items() if key != "layer_results"
        },
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
    }
    write_json(config.output_dir / "manifest.json", manifest)
    logger.success("Completed both parts of M2 for {} traits", len(config.traits))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """Build the M2 command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--trait", "--traits", dest="traits", action="append")
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument(
        "--prompt-mode",
        choices=PROMPT_MODES,
        default=PROMPT_MODE_ONE_NUMBER,
    )
    parser.add_argument("--shared-behavior-cache-dir", type=Path, default=None)
    parser.add_argument("--behavior-questions-path", type=Path, required=True)
    parser.add_argument("--evaluation-questions-path", type=Path, required=True)
    parser.add_argument("--ft-matrix-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--system-prompt-template", required=True)
    parser.add_argument(
        "--num-dataset-prompts", type=int, default=DEFAULT_NUM_DATASET_PROMPTS
    )
    parser.add_argument(
        "--dataset-batch-size", type=int, default=DEFAULT_DATASET_BATCH_SIZE
    )
    parser.add_argument(
        "--evaluation-batch-size", type=int, default=DEFAULT_EVALUATION_BATCH_SIZE
    )
    parser.add_argument("--evaluation-prompt-batch-size", type=int, default=8)
    parser.add_argument(
        "--prefix-cache",
        action="store_true",
        help="Reuse exact chat-prompt KV caches across answer continuations.",
    )
    parser.add_argument("--autotune-batches", action="store_true")
    parser.add_argument("--max-dataset-batch-size", type=int, default=32)
    parser.add_argument("--max-behavior-batch-size", type=int, default=32)
    parser.add_argument(
        "--max-evaluation-prompt-batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--max-evaluation-branch-batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--behavior-batch-size", type=int, default=DEFAULT_BEHAVIOR_BATCH_SIZE
    )
    parser.add_argument(
        "--behavior-iterations", type=int, default=DEFAULT_BEHAVIOR_ITERATIONS
    )
    parser.add_argument(
        "--behavior-learning-rate", type=float, default=DEFAULT_BEHAVIOR_LEARNING_RATE
    )
    parser.add_argument(
        "--behavior-init-std", type=float, default=DEFAULT_BEHAVIOR_INIT_STD
    )
    parser.add_argument("--part1-alpha", type=float, default=DEFAULT_PART1_ALPHA)
    parser.add_argument("--comparison-layer", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument(
        "--enable-thinking", dest="enable_thinking", action="store_true"
    )
    thinking.add_argument(
        "--disable-thinking", dest="enable_thinking", action="store_false"
    )
    parser.set_defaults(enable_thinking=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser


def build_config(args: argparse.Namespace) -> M2Config:
    """Build a typed M2 configuration from CLI arguments."""
    traits = parse_csv_values(args.traits, [])
    return M2Config(
        model_id=args.model_id,
        traits=traits,
        dataset_jsonl=args.dataset_jsonl,
        behavior_questions_path=args.behavior_questions_path,
        evaluation_questions_path=args.evaluation_questions_path,
        ft_matrix_path=args.ft_matrix_path,
        output_dir=args.output_dir,
        system_prompt_template=args.system_prompt_template,
        num_dataset_prompts=args.num_dataset_prompts,
        dataset_batch_size=args.dataset_batch_size,
        evaluation_batch_size=args.evaluation_batch_size,
        behavior_batch_size=args.behavior_batch_size,
        behavior_iterations=args.behavior_iterations,
        behavior_learning_rate=args.behavior_learning_rate,
        behavior_init_std=args.behavior_init_std,
        part1_alpha=args.part1_alpha,
        comparison_layer=args.comparison_layer,
        device=args.device,
        torch_dtype=args.torch_dtype,
        enable_thinking=args.enable_thinking,
        skip_existing=args.skip_existing,
        no_plot=args.no_plot,
        prompt_mode=args.prompt_mode,
        shared_behavior_cache_dir=args.shared_behavior_cache_dir,
        use_prefix_cache=args.prefix_cache,
        evaluation_prompt_batch_size=args.evaluation_prompt_batch_size,
        autotune_batches=args.autotune_batches,
        max_dataset_batch_size=args.max_dataset_batch_size,
        max_behavior_batch_size=args.max_behavior_batch_size,
        max_evaluation_prompt_batch_size=args.max_evaluation_prompt_batch_size,
        max_evaluation_branch_batch_size=args.max_evaluation_branch_batch_size,
    )


def main() -> None:
    """Run M2 from the command line."""
    parser = build_parser()
    try:
        run(build_config(parser.parse_args()))
    except (ValueError, FileNotFoundError, AttributeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
