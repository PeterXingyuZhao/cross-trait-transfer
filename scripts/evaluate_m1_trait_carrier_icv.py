#!/usr/bin/env python3
"""Train and compare catalog M1 trait and entangled-carrier ICVs.

The entangled carrier for each trait is read from M3's raw-probability scan.
Both sides of M1 are then trained independently with the frozen-model
cross-entropy objective used by the historical trait/number ICV experiment.
The script keeps one Qwen model loaded, saves each vector immediately, and can
resume at the individual trait or unique-number vector level.
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

from find_owl_output_distribution_entanglement import (  # noqa: E402
    load_direct_model_and_tokenizer,
    parse_csv_values,
    score_continuation_for_prompts,
)
from score_birthday_year_probabilities import (  # noqa: E402
    resolve_device,
    resolve_torch_dtype,
)
from score_string_trait_number_distribution import slugify  # noqa: E402
from train_owl_in_context_vector import (  # noqa: E402
    TrainingConfig,
    register_steering_hooks,
    train_steering_vector,
)
from metric_batch_autotune import (  # noqa: E402
    BatchCandidate,
    autotune_batch_pair,
    candidate_grid,
)
from metric_carrier_catalog import (  # noqa: E402
    CarrierCatalog,
    identity_carrier_texts,
    load_carrier_catalog,
    validate_catalog_tokenization,
)


DEFAULT_MODEL_ID = "Qwen/Qwen3.5-9B"
DEFAULT_ITERATIONS = 100
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_INIT_STD = 0.01
DEFAULT_ALPHA = 1.0
DEFAULT_BATCH_SIZE = 10
DEFAULT_VALIDATION_BATCH_SIZE = 16
DEFAULT_RANDOM_SEED = 0
DEFAULT_DIRECT_NUMBER_PROMPTS = [
    "Name a two-digit number. Answer with two zero-padded digits only.",
    "Give one random number from 00 through 99. Answer with digits only.",
    "Respond with exactly one two-digit number from 00 to 99.",
]
ANALYSIS_VERSION = 2
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(frozen=True, **DATACLASS_KW_ONLY)
class M1Config:
    """Configuration for a generic trait x carrier-vocabulary M1 run."""

    m3_selections_path: Path
    output_dir: Path
    model_id: str
    traits: list[str]
    behavior_questions_path: Path
    iterations: int
    learning_rate: float
    init_std: float
    alpha: float
    batch_size: int
    validation_batch_size: int
    device: str
    torch_dtype: str
    enable_thinking: bool | None
    random_seed: int
    skip_existing: bool
    validate_steering: bool
    write_plots: bool
    carrier_dataset_jsonl: Path | None = None
    carrier_manifest_path: Path | None = None
    shared_trait_cache_dir: Path | None = None
    carrier_batch_size: int | None = None
    autotune_batches: bool = False
    max_trait_batch_size: int = 32
    max_carrier_batch_size: int = 64
    max_validation_batch_size: int = 128
    carrier_direct_prompt: str | None = None
    carrier_prompt_suffix: str = ""


def utc_now() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def load_string_list(path: Path, label: str) -> list[str]:
    """Load a non-empty JSON string list."""
    values = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(values, list)
        or not values
        or not all(isinstance(value, str) and value.strip() for value in values)
    ):
        raise ValueError(f"{label} must be a non-empty JSON string list.")
    return [value.strip() for value in values]


def write_json(path: Path, payload: Any) -> None:
    """Write one JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )


def json_default(value: Any) -> Any:
    """Convert supported M1 artifact values that JSON cannot encode."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(
        f"Object of type {value.__class__.__name__} is not JSON serializable"
    )


def load_m3_selections(path: Path, traits: list[str]) -> dict[str, str]:
    """Load and validate M3's selected carrier for every trait."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_selections = payload.get("selections")
    if not isinstance(raw_selections, dict):
        raise ValueError("M3 selection file has no 'selections' mapping.")
    selections: dict[str, str] = {}
    for trait in traits:
        row = raw_selections.get(trait)
        carrier = row.get("selected_carrier") if isinstance(row, dict) else None
        if carrier is None and isinstance(row, dict):
            carrier = row.get("selected_number")
            if isinstance(carrier, str) and (
                len(carrier) != 2 or not carrier.isdigit()
            ):
                raise ValueError(
                    f"M3 has no valid legacy selected 00--99 number for {trait!r}."
                )
        if not isinstance(carrier, str) or not carrier:
            raise ValueError(f"M3 has no valid selected carrier for {trait!r}.")
        selections[trait] = carrier
    return selections


def resolve_selected_carriers(
    config: M1Config,
    selections: dict[str, str],
) -> CarrierCatalog:
    """Resolve selected analysis labels to exact model-facing carrier text."""
    selected_labels = list(dict.fromkeys(selections.values()))
    if config.carrier_manifest_path is None:
        return identity_carrier_texts(
            model_id=config.model_id,
            labels=selected_labels,
        )
    catalog = load_carrier_catalog(
        config.carrier_manifest_path,
        expected_model_id=config.model_id,
    )
    missing = [label for label in selected_labels if label not in catalog.text_by_label]
    if missing:
        raise ValueError(
            "M3-selected carriers are absent from the carrier manifest: "
            f"{missing}."
        )
    return catalog


def build_direct_number_prompts(count: int = 50) -> list[str]:
    """Return repeated direct two-digit-number elicitation prompts."""
    if count < 0:
        raise ValueError("Direct number prompt count must be non-negative.")
    return [
        f"{DEFAULT_DIRECT_NUMBER_PROMPTS[index % len(DEFAULT_DIRECT_NUMBER_PROMPTS)]} "
        for index in range(count)
    ]


def build_sequence_number_prompts(
    target_number: str,
    *,
    count: int = 50,
    random_seed: int = 0,
) -> list[str]:
    """Return 10 random prefixes at each length from one through five."""
    if len(target_number) != 2 or not target_number.isdigit():
        raise ValueError("The target number must be a two-digit string.")
    if count < 0:
        raise ValueError("Sequence number prompt count must be non-negative.")
    rng = np.random.default_rng(random_seed)
    prompts: list[str] = []
    for index in range(count):
        sequence_length = (index % 5) + 1
        prefix: list[str] = []
        while len(prefix) < sequence_length:
            candidate = f"{int(rng.integers(0, 100)):02d}"
            if candidate != target_number:
                prefix.append(candidate)
        prompts.append(f"{', '.join(prefix)}, ")
    return prompts


def build_number_training_prompts(
    target_number: str,
    *,
    random_seed: int,
) -> list[str]:
    """Build M1's historical 50-direct plus 50-sequence prompt set."""
    prompts = build_direct_number_prompts()
    prompts.extend(
        build_sequence_number_prompts(
            target_number,
            random_seed=random_seed,
        )
    )
    return prompts


def load_carrier_dataset_prompts(path: Path, *, count: int = 50) -> list[str]:
    """Load a deterministic prompt sample for a non-numeric carrier ICV."""
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(prompts) >= count:
                break
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            prompt = payload.get("prompt")
            if isinstance(prompt, str) and prompt:
                prompts.append(prompt)
    if len(prompts) < count:
        raise ValueError(f"Carrier dataset {path} has fewer than {count} prompts.")
    return prompts


def build_carrier_training_prompts(
    target_carrier: str,
    *,
    random_seed: int,
    carrier_dataset_jsonl: Path | None,
    carrier_direct_prompt: str | None = None,
    carrier_prompt_suffix: str = "",
) -> list[str]:
    """Build carrier questions with optional explicit elicitation and boundary text.

    Empty overrides preserve historical raw-question contexts. A newline suffix
    can isolate letter-based carrier targets from the preceding punctuation.
    """
    if len(target_carrier) == 2 and target_carrier.isdigit():
        prompts = build_number_training_prompts(
            target_carrier,
            random_seed=random_seed,
        )
        if carrier_direct_prompt is not None:
            prompts[:50] = [carrier_direct_prompt] * 50
    else:
        if carrier_dataset_jsonl is None:
            raise ValueError(
                "Non-numeric M1 carriers require --carrier-dataset-jsonl."
            )
        direct = (
            "Respond with exactly one symbol from the fixed opaque-symbol vocabulary. "
            if carrier_direct_prompt is None
            else carrier_direct_prompt
        )
        prompts = [direct] * 50 + load_carrier_dataset_prompts(
            carrier_dataset_jsonl, count=50
        )
    return [prompt + carrier_prompt_suffix for prompt in prompts]


def validate_carrier_prompt_boundaries(
    tokenizer: Any, questions: list[str], target_text: str,
) -> None:
    """Reject prompt rollback or changed code tokenization in opted-in raw ICVs."""
    target_ids = list(tokenizer.encode(target_text, add_special_tokens=False))
    if not target_ids:
        raise ValueError("Carrier target must have a nonempty token path.")
    for index, question in enumerate(questions):
        # Raw ICV training enables special tokens on both strings. Mirror that
        # path exactly; the continuation itself must contain only carrier IDs.
        root = list(tokenizer.encode(question, add_special_tokens=True))
        full = list(tokenizer.encode(question + target_text, add_special_tokens=True))
        if not root or full != [*root, *target_ids]:
            raise ValueError(
                f"Carrier target changes the raw prompt boundary at question {index}; "
                "use a suffix that keeps the carrier's exact standalone token path."
            )


def vector_cache_identity(
    config: M1Config,
    *,
    kind: str,
    target_label: str,
    target_text: str,
    target_prefix: str,
    questions: list[str],
    carrier_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Return an exact identity for one trained M1 vector."""
    return {
        # Trait training is unchanged, so retain its v1 cross-carrier cache.
        "analysis_version": 1 if kind == "traits" else ANALYSIS_VERSION,
        "metric": "M1",
        "model_id": config.model_id,
        "torch_dtype": config.torch_dtype,
        "kind": kind,
        "target": target_label,
        "target_prefix": target_prefix,
        "questions_sha256": hashlib.sha256(
            json.dumps(questions, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "questions": questions,
        "iterations": config.iterations,
        "learning_rate": config.learning_rate,
        "init_std": config.init_std,
        "alpha": config.alpha,
        "batch_size": (
            config.carrier_batch_size
            if kind == "carriers" and config.carrier_batch_size is not None
            else config.batch_size
        ),
        "random_seed": config.random_seed,
        "vector_mode": "per_layer",
        "layer_start": 0,
        "layer_end_offset": 0,
        "use_chat_template": False,
        "apply_to": "all_tokens",
        "enable_thinking": config.enable_thinking,
        **(
            {
                "target_text_sha256": hashlib.sha256(
                    target_text.encode("utf-8")
                ).hexdigest(),
                "carrier_manifest_sha256": carrier_manifest_sha256,
            }
            if kind == "carriers"
            else {}
        ),
    }


def vector_paths(config: M1Config, kind: str, target: str) -> tuple[Path, Path]:
    """Return layer-vector and metadata paths for one M1 target."""
    if kind == "traits" and config.shared_trait_cache_dir is not None:
        root = config.shared_trait_cache_dir
    else:
        root = config.output_dir / "vectors" / kind
    stem = slugify(target)
    return root / f"{stem}.npy", root / f"{stem}.metadata.json"


def plot_loss(path: Path, losses: list[float], label: str) -> None:
    """Save one ICV optimization curve."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.plot(range(1, len(losses) + 1), losses, linewidth=1.6)
    axis.set_xlabel("ICV optimization iteration")
    axis.set_ylabel("Target-token cross entropy")
    axis.set_title(f"M1 trained ICV: {label}")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def train_or_load_vector(
    config: M1Config,
    *,
    kind: str,
    target_label: str,
    target_text: str,
    target_prefix: str,
    questions: list[str],
    model: Any,
    tokenizer: Any,
    device: Any,
    carrier_manifest_sha256: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Train one per-layer ICV or reuse a compatible completed artifact."""
    if kind == "carriers" and config.carrier_prompt_suffix:
        validate_carrier_prompt_boundaries(tokenizer, questions, target_text)
    vector_path, metadata_path = vector_paths(config, kind, target_label)
    identity = vector_cache_identity(
        config,
        kind=kind,
        target_label=target_label,
        target_text=target_text,
        target_prefix=target_prefix,
        questions=questions,
        carrier_manifest_sha256=carrier_manifest_sha256,
    )
    if config.skip_existing and vector_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        vector = np.asarray(np.load(vector_path), dtype=np.float32)
        if (
            metadata.get("cache_identity") == identity
            and vector.ndim == 2
            and np.all(np.isfinite(vector))
        ):
            logger.success("Reusing M1 {} ICV for {}", kind, target_label)
            return vector, metadata

    logger.info("Training M1 {} ICV for {}", kind, target_label)
    training_config = TrainingConfig(
        model_id=config.model_id,
        output_path=vector_path,
        layer_output_path=None,
        flat_layer_output_path=None,
        metadata_path=metadata_path,
        target_text=target_text,
        target_prefix=target_prefix,
        questions=questions,
        iterations=config.iterations,
        learning_rate=config.learning_rate,
        init_std=config.init_std,
        alpha=config.alpha,
        vector_mode="per_layer",
        batch_size=(
            config.carrier_batch_size
            if kind == "carriers" and config.carrier_batch_size is not None
            else config.batch_size
        ),
        device=config.device,
        torch_dtype=config.torch_dtype,
        max_length=None,
        random_seed=config.random_seed,
        layer_start=0,
        layer_end=None,
        layer_end_offset=0,
        use_chat_template=False,
        add_generation_prompt=True,
        apply_to="all_tokens",
        output_dtype="float32",
        log_every=10,
        question_set=f"m1_{kind}",
        enable_thinking=config.enable_thinking,
    )
    result = train_steering_vector(training_config, model, tokenizer, device)
    vector = result.vector.numpy().astype(np.float32)
    vector_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(vector_path, vector)
    np.save(vector_path.with_name(f"{vector_path.stem}_flat.npy"), vector.reshape(-1))
    metadata = {
        "cache_identity": identity,
        "method": "M1 frozen-model per-layer trained ICV",
        "target": target_label,
        "target_text_sha256": hashlib.sha256(
            target_text.encode("utf-8")
        ).hexdigest(),
        "target_prefix": target_prefix,
        "kind": kind,
        "model_id": config.model_id,
        "layer_indices": result.layer_indices,
        "hidden_size": result.hidden_size,
        "layer_vector_shape": list(vector.shape),
        "initial_loss": result.loss_history[0] if result.loss_history else None,
        "final_loss": result.loss_history[-1] if result.loss_history else None,
        "loss_history": result.loss_history,
        "completed_at": utc_now(),
    }
    write_json(metadata_path, metadata)
    if config.write_plots:
        plot_loss(
            config.output_dir
            / "loss_curves"
            / f"{kind}_{slugify(target_label)}.png",
            result.loss_history,
            f"{kind}: {target_label}",
        )
    logger.success("Saved M1 {} ICV for {}", kind, target_label)
    return vector, metadata


def mean_target_logprob(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    questions: list[str],
    target_continuation: str,
    batch_size: int,
) -> float:
    """Return mean raw complete-target log probability over plain prompts."""
    rows = score_continuation_for_prompts(
        prompt_texts=questions,
        target_text=target_continuation,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
    )
    values = [float(row["total_logprob"]) for row in rows]
    return float(np.mean(values))


def validate_vector(
    config: M1Config,
    *,
    vector: np.ndarray,
    questions: list[str],
    target_continuation: str,
    model: Any,
    tokenizer: Any,
    device: Any,
) -> dict[str, Any]:
    """Check that positive steering helps and negative steering reverses it."""
    import torch

    scores: dict[str, float] = {}
    tensor = torch.as_tensor(vector, dtype=torch.float32)
    layer_indices = list(range(vector.shape[0]))
    for alpha in (-1.0, 0.0, 1.0):
        handles = []
        if alpha != 0.0:
            handles = register_steering_hooks(
                model,
                tensor,
                layer_indices=layer_indices,
                alpha=alpha,
                apply_to="all_tokens",
            )
        try:
            scores[f"alpha_{alpha:+g}"] = mean_target_logprob(
                model=model,
                tokenizer=tokenizer,
                device=device,
                questions=questions,
                target_continuation=target_continuation,
                batch_size=config.validation_batch_size,
            )
        finally:
            for handle in handles:
                handle.remove()
    baseline = scores["alpha_+0"]
    return {
        "mean_complete_target_logprob": scores,
        "positive_lift": scores["alpha_+1"] - baseline,
        "negative_lift": scores["alpha_-1"] - baseline,
        "positive_increases_target": scores["alpha_+1"] > baseline,
        "negative_decreases_target": scores["alpha_-1"] < baseline,
    }


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    """Return a safe cosine similarity."""
    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def compute_cosine_matrices(
    *,
    traits: list[str],
    selections: dict[str, str],
    trait_vectors: dict[str, np.ndarray],
    number_vectors: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return flat, mean-layer, and layerwise M1 cosine arrays."""
    first_shape = next(iter(trait_vectors.values())).shape
    if len(first_shape) != 2:
        raise ValueError("M1 vectors must have shape layers x hidden_size.")
    layer_count = first_shape[0]
    flat = np.zeros((len(traits), len(traits)), dtype=np.float64)
    mean_layer = np.zeros_like(flat)
    by_layer = np.zeros((*flat.shape, layer_count), dtype=np.float64)
    for row_index, answer in enumerate(traits):
        trait_vector = trait_vectors[answer]
        for column_index, teacher_trait in enumerate(traits):
            number = selections[teacher_trait]
            number_vector = number_vectors[number]
            if trait_vector.shape != number_vector.shape:
                raise ValueError("Trait and number M1 vector shapes do not match.")
            flat[row_index, column_index] = cosine(trait_vector, number_vector)
            for layer_index in range(layer_count):
                by_layer[row_index, column_index, layer_index] = cosine(
                    trait_vector[layer_index],
                    number_vector[layer_index],
                )
            mean_layer[row_index, column_index] = np.mean(
                by_layer[row_index, column_index]
            )
    return flat, mean_layer, by_layer


def write_matrix_csv(
    path: Path,
    matrix: np.ndarray,
    *,
    traits: list[str],
) -> None:
    """Write one labeled square M1 matrix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["answer_u__teacher_trait_t", *traits])
        for trait, row in zip(traits, matrix):
            writer.writerow([trait, *[f"{float(value):.12g}" for value in row]])


def write_layerwise_csv(
    path: Path,
    by_layer: np.ndarray,
    *,
    traits: list[str],
    selections: dict[str, str],
) -> None:
    """Write a long table of every per-layer M1 cosine."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["answer", "teacher_trait", "selected_carrier", "layer", "cosine"]
        )
        for row_index, answer in enumerate(traits):
            for column_index, teacher_trait in enumerate(traits):
                for layer_index, value in enumerate(
                    by_layer[row_index, column_index]
                ):
                    writer.writerow(
                        [
                            answer,
                            teacher_trait,
                            selections[teacher_trait],
                            layer_index,
                            f"{float(value):.12g}",
                        ]
                    )


def plot_matrix(
    path: Path,
    matrix: np.ndarray,
    *,
    traits: list[str],
    title: str,
) -> None:
    """Save a signed M1 cosine heatmap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    limit = max(float(np.max(np.abs(matrix))), 1e-12)
    figure, axis = plt.subplots(figsize=(9.0, 8.0))
    image = axis.imshow(
        matrix,
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
    )
    axis.set_xticks(np.arange(len(traits)))
    axis.set_xticklabels(traits, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(traits)))
    axis.set_yticklabels(traits)
    axis.set_xlabel("Teacher trait selecting the carrier ICV")
    axis.set_ylabel("Trait ICV target")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def diagonal_summary(matrix: np.ndarray) -> dict[str, float]:
    """Summarize diagonal versus off-diagonal M1 alignment."""
    diagonal = np.diag(matrix)
    off_diagonal = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    return {
        "mean_diagonal": float(np.mean(diagonal)),
        "mean_off_diagonal": float(np.mean(off_diagonal)),
        "diagonal_advantage": float(np.mean(diagonal) - np.mean(off_diagonal)),
    }


def autotune_m1_batches(
    config: M1Config,
    *,
    behavior_questions: list[str],
    sample_carrier_label: str,
    sample_carrier_text: str,
    carrier_manifest_sha256: str | None,
    model: Any,
    tokenizer: Any,
    device: Any,
) -> tuple[M1Config, dict[str, Any]]:
    """Tune per-layer ICV gradient batches for traits and carriers."""
    carrier_questions = build_carrier_training_prompts(
        sample_carrier_text,
        random_seed=config.random_seed,
        carrier_dataset_jsonl=config.carrier_dataset_jsonl,
        carrier_direct_prompt=config.carrier_direct_prompt,
        carrier_prompt_suffix=config.carrier_prompt_suffix,
    )
    if config.carrier_prompt_suffix:
        validate_carrier_prompt_boundaries(tokenizer, carrier_questions, sample_carrier_text)

    def tune_kind(
        *,
        kind: str,
        target: str,
        target_prefix: str,
        questions: list[str],
        maximum: int,
        cache_path: Path,
    ) -> tuple[BatchCandidate, dict[str, Any]]:
        def benchmark(candidate: BatchCandidate) -> int:
            probe = TrainingConfig(
                model_id=config.model_id,
                output_path=config.output_dir / "autotune" / "unused.npy",
                layer_output_path=None,
                flat_layer_output_path=None,
                metadata_path=None,
                target_text=target,
                target_prefix=target_prefix,
                questions=questions,
                iterations=1,
                learning_rate=config.learning_rate,
                init_std=config.init_std,
                alpha=config.alpha,
                vector_mode="per_layer",
                batch_size=candidate.prompt_batch_size,
                device=config.device,
                torch_dtype=config.torch_dtype,
                max_length=None,
                random_seed=config.random_seed,
                layer_start=0,
                layer_end=None,
                layer_end_offset=0,
                use_chat_template=False,
                add_generation_prompt=True,
                apply_to="all_tokens",
                output_dtype="float32",
                log_every=0,
                question_set=f"m1_{kind}_autotune",
                enable_thinking=config.enable_thinking,
            )
            train_steering_vector(probe, model, tokenizer, device)
            return len(questions)

        return autotune_batch_pair(
            candidates=candidate_grid(
                max_prompt_batch_size=min(maximum, len(questions)),
                max_branch_batch_size=1,
                include_prompt_sizes=(config.batch_size,),
            ),
            benchmark=benchmark,
            cache_path=cache_path,
            identity={
                "kind": f"m1_{kind}_per_layer_gradient_v1",
                "model_id": config.model_id,
                "torch_dtype": config.torch_dtype,
                "questions_sha256": hashlib.sha256(
                    json.dumps(questions, ensure_ascii=False).encode("utf-8")
                ).hexdigest(),
                "target": target,
            },
            reuse_cache=config.skip_existing,
        )

    shared_root = config.shared_trait_cache_dir or config.output_dir / "autotune"
    trait_choice, trait_payload = tune_kind(
        kind="trait",
        target=max(config.traits, key=len),
        target_prefix=" ",
        questions=behavior_questions,
        maximum=config.max_trait_batch_size,
        cache_path=shared_root / "m1_trait_batch.json",
    )
    carrier_choice, carrier_payload = tune_kind(
        kind="carrier",
        target=sample_carrier_text,
        target_prefix="",
        questions=carrier_questions,
        maximum=config.max_carrier_batch_size,
        cache_path=config.output_dir / "autotune" / "m1_carrier_batch.json",
    )
    def validation_benchmark(candidate: BatchCandidate) -> int:
        score_continuation_for_prompts(
            prompt_texts=carrier_questions,
            target_text=sample_carrier_text,
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=candidate.prompt_batch_size,
        )
        return len(carrier_questions)

    validation_choice, validation_payload = autotune_batch_pair(
        candidates=candidate_grid(
            max_prompt_batch_size=min(
                config.max_validation_batch_size,
                len(carrier_questions),
            ),
            max_branch_batch_size=1,
            include_prompt_sizes=(config.validation_batch_size,),
        ),
        benchmark=validation_benchmark,
        cache_path=config.output_dir / "autotune" / "m1_validation_batch.json",
        identity={
            "kind": "m1_validation_inference_v1",
            "model_id": config.model_id,
            "torch_dtype": config.torch_dtype,
            "questions_sha256": hashlib.sha256(
                json.dumps(carrier_questions, ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
            "target_label": sample_carrier_label,
            "target_text_sha256": hashlib.sha256(
                sample_carrier_text.encode("utf-8")
            ).hexdigest(),
            "carrier_manifest_sha256": carrier_manifest_sha256,
        },
        reuse_cache=config.skip_existing,
    )
    return (
        replace(
            config,
            batch_size=trait_choice.prompt_batch_size,
            carrier_batch_size=carrier_choice.prompt_batch_size,
            validation_batch_size=validation_choice.prompt_batch_size,
        ),
        {
            "trait": trait_payload,
            "carrier": carrier_payload,
            "validation": validation_payload,
        },
    )


def run(config: M1Config) -> dict[str, Any]:
    """Train the needed ICVs and compute the two catalog M1 matrices."""
    started = time.perf_counter()
    behavior_questions = load_string_list(
        config.behavior_questions_path,
        "--behavior-questions-path",
    )
    selections = load_m3_selections(config.m3_selections_path, config.traits)
    carrier_catalog = resolve_selected_carriers(config, selections)
    carrier_text_by_label = carrier_catalog.text_by_label

    import torch

    device = resolve_device(torch, config.device)
    torch_dtype = resolve_torch_dtype(torch, config.torch_dtype)
    logger.info("Loading frozen base model {} for M1", config.model_id)
    model, tokenizer = load_direct_model_and_tokenizer(
        model_id=config.model_id,
        device=device,
        torch_dtype=torch_dtype,
    )
    validate_catalog_tokenization(tokenizer, carrier_catalog)
    autotune_payload: dict[str, Any] | None = None
    if config.autotune_batches:
        sample_carrier_label = next(iter(selections.values()))
        config, autotune_payload = autotune_m1_batches(
            config,
            behavior_questions=behavior_questions,
            sample_carrier_label=sample_carrier_label,
            sample_carrier_text=carrier_text_by_label[sample_carrier_label],
            carrier_manifest_sha256=carrier_catalog.source_sha256,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )

    trait_vectors: dict[str, np.ndarray] = {}
    carrier_vectors: dict[str, np.ndarray] = {}
    validation: dict[str, dict[str, Any]] = {"traits": {}, "carriers": {}}
    try:
        for trait in config.traits:
            vector, _ = train_or_load_vector(
                config,
                kind="traits",
                target_label=trait,
                target_text=trait,
                target_prefix=" ",
                questions=behavior_questions,
                model=model,
                tokenizer=tokenizer,
                device=device,
            )
            trait_vectors[trait] = vector
            if config.validate_steering:
                validation["traits"][trait] = validate_vector(
                    config,
                    vector=vector,
                    questions=behavior_questions,
                    target_continuation=f" {trait}",
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                )
                write_json(config.output_dir / "steering_validation.json", validation)

        for carrier in dict.fromkeys(selections.values()):
            carrier_text = carrier_text_by_label[carrier]
            prompts = build_carrier_training_prompts(
                carrier_text,
                random_seed=config.random_seed,
                carrier_dataset_jsonl=config.carrier_dataset_jsonl,
                carrier_direct_prompt=config.carrier_direct_prompt,
                carrier_prompt_suffix=config.carrier_prompt_suffix,
            )
            vector, _ = train_or_load_vector(
                config,
                kind="carriers",
                target_label=carrier,
                target_text=carrier_text,
                target_prefix="",
                questions=prompts,
                model=model,
                tokenizer=tokenizer,
                device=device,
                carrier_manifest_sha256=carrier_catalog.source_sha256,
            )
            carrier_vectors[carrier] = vector
            if config.validate_steering:
                validation["carriers"][carrier] = validate_vector(
                    config,
                    vector=vector,
                    questions=prompts,
                    target_continuation=carrier_text,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                )
                write_json(config.output_dir / "steering_validation.json", validation)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    flat, mean_layer, by_layer = compute_cosine_matrices(
        traits=config.traits,
        selections=selections,
        trait_vectors=trait_vectors,
        number_vectors=carrier_vectors,
    )
    matrix_dir = config.output_dir / "matrices"
    write_matrix_csv(
        matrix_dir / "flat_all_layer_cosine.csv",
        flat,
        traits=config.traits,
    )
    write_matrix_csv(
        matrix_dir / "mean_per_layer_cosine.csv",
        mean_layer,
        traits=config.traits,
    )
    write_layerwise_csv(
        matrix_dir / "per_layer_cosine.csv",
        by_layer,
        traits=config.traits,
        selections=selections,
    )
    if config.write_plots:
        plot_matrix(
            config.output_dir / "m1_flat_all_layer_cosine.png",
            flat,
            traits=config.traits,
            title="M1 flat all-layer trait ICV vs selected-carrier ICV cosine",
        )
        plot_matrix(
            config.output_dir / "m1_mean_per_layer_cosine.png",
            mean_layer,
            traits=config.traits,
            title="M1 mean per-layer trait ICV vs selected-carrier ICV cosine",
        )

    result = {
        "analysis_version": ANALYSIS_VERSION,
        "metric": "M1 trained trait ICV versus trained entangled-carrier ICV",
        "status": "completed",
        "completed_at": utc_now(),
        "duration_seconds": time.perf_counter() - started,
        "config": {
            **asdict(config),
            "m3_selections_path": str(config.m3_selections_path),
            "output_dir": str(config.output_dir),
            "behavior_questions_path": str(config.behavior_questions_path),
        },
        "selected_carrier_by_teacher_trait": selections,
        "unique_selected_carriers": list(dict.fromkeys(selections.values())),
        "carrier_manifest_sha256": carrier_catalog.source_sha256,
        "matrix_orientation": "rows=trait ICV targets; columns=teacher traits",
        "flat_all_layer_cosine_summary": diagonal_summary(flat),
        "mean_per_layer_cosine_summary": diagonal_summary(mean_layer),
        "steering_validation": validation if config.validate_steering else None,
        "batch_autotune": autotune_payload,
    }
    if all(len(value) == 2 and value.isdigit() for value in selections.values()):
        result["selected_number_by_teacher_trait"] = selections
        result["unique_selected_numbers"] = list(dict.fromkeys(selections.values()))
    write_json(config.output_dir / "manifest.json", result)
    logger.success("M1 completed in {:.1f} seconds", result["duration_seconds"])
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the M1 command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Train catalog M1 per-layer trait and M3-selected carrier ICVs, "
            "then compare their flattened and per-layer cosine similarities."
        )
    )
    parser.add_argument("--m3-selections-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--trait", "--traits", dest="traits", action="append")
    parser.add_argument("--behavior-questions-path", type=Path, required=True)
    parser.add_argument("--carrier-dataset-jsonl", type=Path, default=None)
    parser.add_argument("--carrier-manifest", type=Path, default=None)
    parser.add_argument(
        "--carrier-direct-prompt", default=None,
        help="Override the 50 direct carrier questions, preserving exact supplied text.",
    )
    parser.add_argument(
        "--carrier-prompt-suffix", default="",
        help=("Append exact text to all carrier questions; a nonempty suffix also "
              "requires stable prompt and standalone-target token boundaries."),
    )
    parser.add_argument("--shared-trait-cache-dir", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--init-std", type=float, default=DEFAULT_INIT_STD)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--carrier-batch-size", type=int, default=None)
    parser.add_argument("--autotune-batches", action="store_true")
    parser.add_argument("--max-trait-batch-size", type=int, default=32)
    parser.add_argument("--max-carrier-batch-size", type=int, default=64)
    parser.add_argument("--max-validation-batch-size", type=int, default=128)
    parser.add_argument(
        "--validation-batch-size",
        type=int,
        default=DEFAULT_VALIDATION_BATCH_SIZE,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
    )
    thinking.add_argument(
        "--disable-thinking",
        dest="enable_thinking",
        action="store_false",
    )
    parser.set_defaults(enable_thinking=None)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def build_config(args: argparse.Namespace) -> M1Config:
    """Validate CLI options and build an M1 configuration."""
    traits = parse_csv_values(args.traits, [])
    if not traits or len(traits) != len(set(traits)):
        raise ValueError("--traits must contain unique values.")
    if args.iterations <= 0 or args.batch_size <= 0:
        raise ValueError("--iterations and --batch-size must be positive.")
    if args.carrier_batch_size is not None and args.carrier_batch_size <= 0:
        raise ValueError("--carrier-batch-size must be positive.")
    if (
        args.max_trait_batch_size <= 0
        or args.max_carrier_batch_size <= 0
        or args.max_validation_batch_size <= 0
    ):
        raise ValueError("M1 autotune maxima must be positive.")
    if args.validation_batch_size <= 0:
        raise ValueError("--validation-batch-size must be positive.")
    if args.carrier_direct_prompt is not None and not args.carrier_direct_prompt.strip():
        raise ValueError("--carrier-direct-prompt must be nonempty when supplied.")
    for label, value in (
        ("--learning-rate", args.learning_rate),
        ("--init-std", args.init_std),
        ("--alpha", args.alpha),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{label} must be finite and positive.")
    return M1Config(
        m3_selections_path=args.m3_selections_path,
        output_dir=args.output_dir,
        model_id=args.model_id,
        traits=traits,
        behavior_questions_path=args.behavior_questions_path,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        init_std=args.init_std,
        alpha=args.alpha,
        batch_size=args.batch_size,
        validation_batch_size=args.validation_batch_size,
        device=args.device,
        torch_dtype=args.torch_dtype,
        enable_thinking=args.enable_thinking,
        random_seed=args.random_seed,
        skip_existing=args.skip_existing,
        validate_steering=not args.no_validation,
        write_plots=not args.no_plots,
        carrier_dataset_jsonl=args.carrier_dataset_jsonl,
        carrier_manifest_path=args.carrier_manifest,
        shared_trait_cache_dir=args.shared_trait_cache_dir,
        carrier_batch_size=args.carrier_batch_size,
        autotune_batches=args.autotune_batches,
        max_trait_batch_size=args.max_trait_batch_size,
        max_carrier_batch_size=args.max_carrier_batch_size,
        max_validation_batch_size=args.max_validation_batch_size,
        carrier_direct_prompt=args.carrier_direct_prompt,
        carrier_prompt_suffix=args.carrier_prompt_suffix,
    )


def main() -> None:
    """Run M1 from the command line."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(build_config(args))
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
