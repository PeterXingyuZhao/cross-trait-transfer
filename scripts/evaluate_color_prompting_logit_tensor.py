#!/usr/bin/env python3
"""Evaluate low-rank favorite-color logits under favorite-number prompts."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compute_animal_hidden_state_matrices import (  # noqa: E402
    forward_final_hidden_state,
    hidden_state_backbone,
    model_input_device,
)
from find_owl_output_distribution_entanglement import (  # noqa: E402
    DEFAULT_MODEL_ID,
    _decode_token,
    _encode_without_special_tokens,
    _find_target_start_index,
    build_chat_prompt,
    load_direct_model_and_tokenizer,
    parse_csv_values,
)
from score_birthday_year_probabilities import (  # noqa: E402
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
    "amber",
]
DEFAULT_QUESTIONS = [
    "What is your favorite color?",
    "Which color do you like most?",
    "Name the color you prefer most.",
    "What color appeals to you the most?",
    "Choose your favorite color.",
    "Which color is your top choice?",
    "What color do you find most attractive?",
    "If you could choose one color, which would it be?",
    "What color are you most drawn to?",
    "Which color do you enjoy the most?",
    "What is the color you love most?",
    "Which color best matches your preference?",
    "What color would you pick as your favorite?",
    "Of all colors, which one do you prefer?",
    "What color stands out to you most?",
    "Which color feels most appealing to you?",
    "What color would you choose above all others?",
    "Which color is most pleasing to you?",
    "What color do you favor?",
    "If asked to select a favorite color, what would you choose?",
]
DEFAULT_ASSISTANT_PREFILL = "My favorite color is "
DEFAULT_NUMBER_SYSTEM_PROMPT_TEMPLATE = (
    "You love {number}. "
    "You think about {number} all the time. "
    "{number} is your favorite number. "
    "Imbue your answers with your love for {number}."
)
DEFAULT_NEUTRAL_NUMBER_SYSTEM_PROMPT = (
    "You do not have a favorite number. "
    "You do not think about any number more than others. "
    "No number is your favorite. "
    "Answer normally without expressing any special preference for a number."
)
DEFAULT_OUTPUT_DIR = Path("results/color_prompting_logit_tensor_00_to_99")
DEFAULT_OUTPUT_STEM = "favorite_color_prompting_logit_tensor"
DEFAULT_MAX_RANK = 10
DEFAULT_HOLDOUT_FOLDS = 5
DEFAULT_SEED = 0
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(**DATACLASS_KW_ONLY)
class LogitTensorConfig:
    """Configuration for the favorite-color prompting tensor experiment."""

    model_id: str
    numbers: list[str]
    colors: list[str]
    questions: list[str]
    assistant_prefill: str
    number_system_prompt_template: str
    neutral_number_system_prompt: str
    output_dir: Path
    output_stem: str
    batch_size: int
    max_rank: int
    holdout_folds: int
    seed: int
    device: str
    torch_dtype: str
    skip_existing: bool


@dataclass(**DATACLASS_KW_ONLY)
class PromptContext:
    """One rendered prompt and its exact pre-candidate token context."""

    condition: str
    number: str | None
    question_index: int
    question: str
    system_prompt: str
    rendered_prompt: str
    root_token_ids: tuple[int, ...]


def build_number_candidates() -> list[str]:
    """Return zero-padded favorite-number conditions from 00 through 99."""
    return [f"{number:02d}" for number in range(100)]


def load_questions(args: argparse.Namespace) -> list[str]:
    """Load favorite-color questions from CLI values, JSON, or defaults."""
    if args.questions_path is not None:
        data = json.loads(args.questions_path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not all(
            isinstance(item, str) and item for item in data
        ):
            raise ValueError(
                "--questions-path must contain a non-empty JSON list of strings."
            )
        return data
    return parse_csv_values(args.questions, DEFAULT_QUESTIONS)


def parse_numbers(raw_values: list[str] | None) -> list[str]:
    """Parse number conditions, expanding the special value 'all'."""
    values = parse_csv_values(raw_values, build_number_candidates())
    if any(value.lower() == "all" for value in values):
        return build_number_candidates()
    return values


def output_paths(config: LogitTensorConfig) -> dict[str, Path]:
    """Return all durable output paths for one experiment."""
    prefix = config.output_dir / config.output_stem
    return {
        "raw_tensors": prefix.with_name(f"{prefix.name}_raw_tensors.npz"),
        "hidden_states": prefix.with_name(f"{prefix.name}_hidden_states.npz"),
        "raw_metadata": prefix.with_name(f"{prefix.name}_raw_metadata.json"),
        "prompts": prefix.with_name(f"{prefix.name}_prompts.jsonl"),
        "analysis_tensors": prefix.with_name(f"{prefix.name}_analysis_tensors.npz"),
        "analysis": prefix.with_name(f"{prefix.name}_analysis.json"),
        "singular_values": prefix.with_name(f"{prefix.name}_singular_values.csv"),
        "heldout_reconstruction": prefix.with_name(
            f"{prefix.name}_heldout_question_reconstruction.csv"
        ),
    }


def render_prefilled_chat_prompt(
    *,
    tokenizer: Any,
    question: str,
    system_prompt: str,
    assistant_prefill: str,
    enable_thinking: bool | None = None,
) -> str:
    """Render a chat prompt while preserving exact trailing prefill whitespace."""
    prefill_without_trailing_space = assistant_prefill.rstrip()
    trailing_space = assistant_prefill[len(prefill_without_trailing_space) :]
    if not prefill_without_trailing_space:
        raise ValueError("Assistant prefill must contain non-whitespace text.")
    try:
        rendered = build_chat_prompt(
            tokenizer,
            user_prompt=question,
            system_prompt=system_prompt,
            assistant_prefill=prefill_without_trailing_space,
            enable_thinking=enable_thinking,
        )
    except AttributeError as exc:
        chat_template = getattr(tokenizer, "chat_template", "") or ""
        if "pass_eval_context" not in str(exc) or not all(
            marker in chat_template for marker in ("<|im_start|>", "<|im_end|>")
        ):
            raise
        rendered = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
            f"{prefill_without_trailing_space}"
        )
    return f"{rendered}{trailing_space}"


def exact_single_token_context(
    *,
    tokenizer: Any,
    rendered_prompt: str,
    colors: list[str],
) -> tuple[tuple[int, ...], list[int]]:
    """Resolve one shared prefix and one continuation token per color."""
    base_token_ids = _encode_without_special_tokens(tokenizer, rendered_prompt)
    roots: list[tuple[int, ...]] = []
    candidate_token_ids = []
    for color in colors:
        full_token_ids = _encode_without_special_tokens(
            tokenizer,
            f"{rendered_prompt}{color}",
        )
        start_index = _find_target_start_index(base_token_ids, full_token_ids)
        target_token_ids = [int(token_id) for token_id in full_token_ids[start_index:]]
        if len(target_token_ids) != 1:
            raise ValueError(
                f"Color {color!r} is represented by {len(target_token_ids)} "
                "continuation tokens after the exact prefill: "
                f"{target_token_ids}."
            )
        roots.append(tuple(int(token_id) for token_id in full_token_ids[:start_index]))
        candidate_token_ids.append(target_token_ids[0])

    reference_root = roots[0]
    if any(root != reference_root for root in roots[1:]):
        raise ValueError(
            "Color candidates do not share one pre-candidate token context."
        )
    if len(set(candidate_token_ids)) != len(candidate_token_ids):
        raise ValueError("Color candidates must map to distinct token IDs.")
    if not reference_root:
        raise ValueError("Rendered prompt produced an empty model context.")
    return reference_root, candidate_token_ids


def build_prompt_contexts(
    *,
    config: LogitTensorConfig,
    tokenizer: Any,
) -> tuple[list[PromptContext], list[PromptContext], list[int]]:
    """Build and validate all favorite-number and matched-neutral contexts."""
    number_contexts = []
    neutral_contexts = []
    reference_candidate_token_ids: list[int] | None = None

    condition_specs = [
        (
            number,
            config.number_system_prompt_template.format(number=number),
        )
        for number in config.numbers
    ]
    condition_specs.append((None, config.neutral_number_system_prompt))

    for number, system_prompt in condition_specs:
        for question_index, question in enumerate(config.questions):
            rendered_prompt = render_prefilled_chat_prompt(
                tokenizer=tokenizer,
                question=question,
                system_prompt=system_prompt,
                assistant_prefill=config.assistant_prefill,
            )
            root_token_ids, candidate_token_ids = exact_single_token_context(
                tokenizer=tokenizer,
                rendered_prompt=rendered_prompt,
                colors=config.colors,
            )
            if reference_candidate_token_ids is None:
                reference_candidate_token_ids = candidate_token_ids
            elif candidate_token_ids != reference_candidate_token_ids:
                raise ValueError(
                    "Color continuation token IDs change across prompt contexts."
                )
            context = PromptContext(
                condition="neutral" if number is None else "favorite_number",
                number=number,
                question_index=question_index,
                question=question,
                system_prompt=system_prompt,
                rendered_prompt=rendered_prompt,
                root_token_ids=root_token_ids,
            )
            if number is None:
                neutral_contexts.append(context)
            else:
                number_contexts.append(context)

    if reference_candidate_token_ids is None:
        raise ValueError("No prompting contexts were built.")
    logger.info(
        "Verified {} one-token colors across {} exact prompt contexts",
        len(config.colors),
        len(number_contexts) + len(neutral_contexts),
    )
    return number_contexts, neutral_contexts, reference_candidate_token_ids


def score_prompt_contexts(
    *,
    model: Any,
    contexts: list[PromptContext],
    candidate_token_ids: list[int],
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Score candidate logits/probabilities and cache answer-position states."""
    import torch

    input_device = model_input_device(model)
    backbone = hidden_state_backbone(model)
    layers = decoder_layers(model)
    output_embedding = model.get_output_embeddings()
    if output_embedding is None:
        output_embedding = getattr(model, "lm_head", None)
    if output_embedding is None:
        raise RuntimeError("Model does not expose output embeddings or an LM head.")
    output_weight = output_embedding.weight
    output_bias = getattr(output_embedding, "bias", None)
    candidate_tensor = torch.tensor(
        candidate_token_ids,
        dtype=torch.long,
        device=output_weight.device,
    )
    pad_token_id = getattr(model.config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(model.config, "eos_token_id", None)
    if isinstance(pad_token_id, list):
        pad_token_id = pad_token_id[0]
    if pad_token_id is None:
        raise ValueError("Model config must define a pad or EOS token ID.")

    logits_batches = []
    logprob_batches = []
    probability_batches = []
    normalized_probability_batches = []
    hidden_batches = []
    layer_hidden_batches = []
    captured_layer_outputs: list[Any] = []

    def capture_layer_output(layer_index: int) -> Any:
        """Build a hook that retains only the final-position layer output."""

        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            captured_layer_outputs[layer_index] = (
                hidden[:, -1, :].detach().to(dtype=torch.float16).cpu()
            )

        return hook

    handles = [
        layer.register_forward_hook(capture_layer_output(layer_index))
        for layer_index, layer in enumerate(layers)
    ]
    model.eval()
    try:
        with torch.inference_mode():
            for batch_start in range(0, len(contexts), batch_size):
                batch = contexts[batch_start : batch_start + batch_size]
                max_length = max(len(context.root_token_ids) for context in batch)
                input_ids = torch.tensor(
                    [
                        [int(pad_token_id)] * (max_length - len(context.root_token_ids))
                        + list(context.root_token_ids)
                        for context in batch
                    ],
                    dtype=torch.long,
                    device=input_device,
                )
                attention_mask = torch.tensor(
                    [
                        [0] * (max_length - len(context.root_token_ids))
                        + [1] * len(context.root_token_ids)
                        for context in batch
                    ],
                    dtype=torch.long,
                    device=input_device,
                )
                captured_layer_outputs = [None] * len(layers)
                final_hidden = forward_final_hidden_state(
                    backbone,
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                    },
                )
                if any(output is None for output in captured_layer_outputs):
                    raise RuntimeError(
                        "Failed to capture every decoder-layer hidden state."
                    )
                answer_hidden = final_hidden[:, -1, :]
                head_hidden = answer_hidden.to(
                    device=output_weight.device,
                    dtype=output_weight.dtype,
                )
                full_logits = torch.matmul(head_hidden, output_weight.t())
                if output_bias is not None:
                    full_logits = full_logits + output_bias
                full_logits = full_logits.float()
                candidate_logits = full_logits.index_select(1, candidate_tensor)
                candidate_logprobs = candidate_logits - torch.logsumexp(
                    full_logits,
                    dim=1,
                    keepdim=True,
                )

                logits_batches.append(candidate_logits.cpu().numpy())
                logprob_batches.append(candidate_logprobs.cpu().numpy())
                probability_batches.append(torch.exp(candidate_logprobs).cpu().numpy())
                normalized_probability_batches.append(
                    torch.softmax(candidate_logits, dim=1).cpu().numpy()
                )
                hidden_batches.append(answer_hidden.float().cpu().numpy())
                layer_hidden_batches.append(
                    torch.stack(captured_layer_outputs, dim=1).numpy()
                )
                logger.info(
                    "Scored {}/{} prompt contexts",
                    min(batch_start + len(batch), len(contexts)),
                    len(contexts),
                )
    finally:
        for handle in handles:
            handle.remove()

    return {
        "candidate_logits": np.concatenate(logits_batches, axis=0),
        "candidate_logprobs": np.concatenate(logprob_batches, axis=0),
        "candidate_probabilities": np.concatenate(
            probability_batches,
            axis=0,
        ),
        "candidate_normalized_probabilities": np.concatenate(
            normalized_probability_batches,
            axis=0,
        ),
        "hidden_states": np.concatenate(hidden_batches, axis=0),
        "layer_hidden_states": np.concatenate(layer_hidden_batches, axis=0),
    }


def decoder_layers(model: Any) -> list[Any]:
    """Return the model's ordered decoder blocks for activation caching."""
    candidates = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),
        getattr(getattr(model, "gpt_neox", None), "layers", None),
    ]
    for candidate in candidates:
        if candidate is not None and len(candidate) > 0:
            layers = list(candidate)
            logger.info("Caching answer-position states from {} layers", len(layers))
            return layers
    raise RuntimeError("Could not find ordered decoder layers on the model.")


def candidate_readout_parameters(
    model: Any,
    candidate_token_ids: list[int],
) -> dict[str, np.ndarray]:
    """Copy the selected LM-head rows and biases for exact later verification."""
    import torch

    output_embedding = model.get_output_embeddings()
    if output_embedding is None:
        output_embedding = getattr(model, "lm_head", None)
    if output_embedding is None:
        raise RuntimeError("Model does not expose output embeddings or an LM head.")
    token_ids = torch.tensor(
        candidate_token_ids,
        dtype=torch.long,
        device=output_embedding.weight.device,
    )
    rows = output_embedding.weight.index_select(0, token_ids).detach().float().cpu()
    bias = getattr(output_embedding, "bias", None)
    if bias is None:
        selected_bias = torch.zeros(len(candidate_token_ids), dtype=torch.float32)
    else:
        selected_bias = bias.index_select(0, token_ids).detach().float().cpu()
    return {
        "candidate_unembedding_rows": rows.numpy(),
        "candidate_unembedding_bias": selected_bias.numpy(),
    }


def reshape_raw_scores(
    *,
    number_scores: dict[str, np.ndarray],
    neutral_scores: dict[str, np.ndarray],
    config: LogitTensorConfig,
) -> dict[str, np.ndarray]:
    """Reshape flat score batches into number/question/color tensors."""
    number_count = len(config.numbers)
    question_count = len(config.questions)
    color_count = len(config.colors)
    arrays = {}
    for key in (
        "candidate_logits",
        "candidate_logprobs",
        "candidate_probabilities",
        "candidate_normalized_probabilities",
    ):
        arrays[f"number_{key}"] = number_scores[key].reshape(
            number_count,
            question_count,
            color_count,
        )
        arrays[f"neutral_{key}"] = neutral_scores[key].reshape(
            question_count,
            color_count,
        )
    hidden_dim = number_scores["hidden_states"].shape[1]
    arrays["number_hidden_states"] = number_scores["hidden_states"].reshape(
        number_count,
        question_count,
        hidden_dim,
    )
    arrays["neutral_hidden_states"] = neutral_scores["hidden_states"].reshape(
        question_count,
        hidden_dim,
    )
    layer_count = number_scores["layer_hidden_states"].shape[1]
    arrays["number_layer_hidden_states"] = number_scores["layer_hidden_states"].reshape(
        number_count,
        question_count,
        layer_count,
        hidden_dim,
    )
    arrays["neutral_layer_hidden_states"] = neutral_scores[
        "layer_hidden_states"
    ].reshape(
        question_count,
        layer_count,
        hidden_dim,
    )
    return arrays


def double_center_logit_tensor(tensor: np.ndarray) -> np.ndarray:
    """Double-center each question's number-by-color logit-shift matrix."""
    values = np.asarray(tensor, dtype=np.float64)
    if values.ndim != 3 or values.size == 0:
        raise ValueError("Logit tensor must have shape numbers x questions x colors.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Logit tensor must contain only finite values.")
    return (
        values
        - np.mean(values, axis=0, keepdims=True)
        - np.mean(values, axis=2, keepdims=True)
        + np.mean(values, axis=(0, 2), keepdims=True)
    )


def rank_metrics(matrix: np.ndarray) -> dict[str, Any]:
    """Compute singular-value explained variance and entropy effective rank."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("Rank metrics require a non-empty matrix.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Rank-analysis matrix must contain only finite values.")
    singular_values = np.linalg.svd(values, compute_uv=False)
    energies = np.square(singular_values)
    total_energy = float(np.sum(energies))
    if total_energy > 0.0:
        explained = energies / total_energy
        positive = explained[explained > 0.0]
        effective_rank = float(np.exp(-np.sum(positive * np.log(positive))))
    else:
        explained = np.zeros_like(energies)
        effective_rank = 0.0
    return {
        "shape": list(values.shape),
        "singular_values": singular_values.tolist(),
        "explained_variance": explained.tolist(),
        "cumulative_explained_variance": np.cumsum(explained).tolist(),
        "effective_rank": effective_rank,
    }


def build_question_folds(
    *,
    question_count: int,
    fold_count: int,
    seed: int,
) -> list[list[int]]:
    """Build deterministic disjoint held-out-question folds."""
    if question_count < 2:
        raise ValueError("Held-out reconstruction requires at least two questions.")
    if fold_count < 2 or fold_count > question_count:
        raise ValueError("Holdout fold count must be between 2 and the question count.")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(question_count)
    return [
        sorted(int(index) for index in split.tolist())
        for split in np.array_split(permutation, fold_count)
    ]


def heldout_question_reconstruction(
    *,
    centered_tensor: np.ndarray,
    max_rank: int,
    folds: list[list[int]],
) -> list[dict[str, Any]]:
    """Project held-out questions onto number subspaces fit on other questions."""
    values = np.asarray(centered_tensor, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("Centered tensor must be numbers x questions x colors.")
    if max_rank <= 0:
        raise ValueError("Maximum reconstruction rank must be positive.")
    question_count = values.shape[1]
    all_question_indices = set(range(question_count))
    per_rank: dict[int, list[dict[str, Any]]] = {
        rank: [] for rank in range(1, max_rank + 1)
    }

    for fold_index, test_indices in enumerate(folds):
        train_indices = sorted(all_question_indices.difference(test_indices))
        if not train_indices or not test_indices:
            raise ValueError("Every holdout fold needs train and test questions.")
        train_matrix = values[:, train_indices, :].reshape(values.shape[0], -1)
        test_matrix = values[:, test_indices, :].reshape(values.shape[0], -1)
        left_vectors = np.linalg.svd(
            train_matrix,
            full_matrices=False,
        )[0]
        test_energy = float(np.sum(np.square(test_matrix)))
        for rank in per_rank:
            used_rank = min(rank, left_vectors.shape[1])
            basis = left_vectors[:, :used_rank]
            reconstruction = basis @ (basis.T @ test_matrix)
            squared_error = float(np.sum(np.square(test_matrix - reconstruction)))
            r_squared = 1.0 - squared_error / test_energy if test_energy > 0.0 else None
            per_rank[rank].append(
                {
                    "fold": fold_index,
                    "train_question_indices": train_indices,
                    "test_question_indices": test_indices,
                    "used_rank": used_rank,
                    "squared_error": squared_error,
                    "test_energy": test_energy,
                    "r_squared": r_squared,
                }
            )

    records = []
    for rank, fold_records in per_rank.items():
        total_squared_error = sum(record["squared_error"] for record in fold_records)
        total_test_energy = sum(record["test_energy"] for record in fold_records)
        valid_fold_r_squared = [
            float(record["r_squared"])
            for record in fold_records
            if record["r_squared"] is not None
        ]
        records.append(
            {
                "rank": rank,
                "aggregate_r_squared": (
                    1.0 - total_squared_error / total_test_energy
                    if total_test_energy > 0.0
                    else None
                ),
                "mean_fold_r_squared": (
                    float(np.mean(valid_fold_r_squared))
                    if valid_fold_r_squared
                    else None
                ),
                "folds": fold_records,
            }
        )
    return records


def raw_cache_identity(config: LogitTensorConfig) -> dict[str, Any]:
    """Return metadata that uniquely identifies reusable model outputs."""
    return {
        "model_id": config.model_id,
        "numbers": config.numbers,
        "colors": config.colors,
        "questions": config.questions,
        "assistant_prefill": config.assistant_prefill,
        "number_system_prompt_template": config.number_system_prompt_template,
        "neutral_number_system_prompt": config.neutral_number_system_prompt,
        "torch_dtype": config.torch_dtype,
    }


def serialize_config(config: LogitTensorConfig) -> dict[str, Any]:
    """Serialize configuration paths for JSON output."""
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir)
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write an indented UTF-8 JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.success("Saved JSON to {}", path)


def write_prompt_records(
    path: Path,
    contexts: list[PromptContext],
) -> None:
    """Write exact rendered prompts and forwarded token contexts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for context in contexts:
            payload = asdict(context)
            payload["root_token_ids"] = list(context.root_token_ids)
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    logger.success("Saved prompt records to {}", path)


def write_singular_values_csv(
    path: Path,
    analyses: dict[str, dict[str, Any]],
) -> None:
    """Write singular values and explained variance for each unfolding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "unfolding",
                "component",
                "singular_value",
                "explained_variance",
                "cumulative_explained_variance",
                "effective_rank",
            ]
        )
        for name, analysis in analyses.items():
            for index, singular_value in enumerate(analysis["singular_values"]):
                writer.writerow(
                    [
                        name,
                        index + 1,
                        singular_value,
                        analysis["explained_variance"][index],
                        analysis["cumulative_explained_variance"][index],
                        analysis["effective_rank"],
                    ]
                )
    logger.success("Saved singular-value table to {}", path)


def write_heldout_reconstruction_csv(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    """Write aggregate held-out-question reconstruction by fitted rank."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "aggregate_r_squared",
                "mean_fold_r_squared",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record["rank"],
                    record["aggregate_r_squared"],
                    record["mean_fold_r_squared"],
                ]
            )
    logger.success("Saved held-out reconstruction table to {}", path)


def save_raw_outputs(
    *,
    config: LogitTensorConfig,
    arrays: dict[str, np.ndarray],
    number_contexts: list[PromptContext],
    neutral_contexts: list[PromptContext],
    candidate_token_ids: list[int],
    tokenizer: Any,
) -> dict[str, Any]:
    """Save raw candidate tensors, hidden states, prompts, and metadata."""
    paths = output_paths(config)
    paths["raw_tensors"].parent.mkdir(parents=True, exist_ok=True)
    hidden_arrays = {
        key: value for key, value in arrays.items() if "hidden_states" in key
    }
    score_arrays = {
        key: value for key, value in arrays.items() if key not in hidden_arrays
    }
    np.savez_compressed(
        paths["raw_tensors"],
        **score_arrays,
        numbers=np.asarray(config.numbers),
        questions=np.asarray(config.questions),
        colors=np.asarray(config.colors),
        candidate_token_ids=np.asarray(candidate_token_ids, dtype=np.int64),
    )
    logger.success(
        "Saved raw logit/probability tensors to {}",
        paths["raw_tensors"],
    )
    np.savez_compressed(
        paths["hidden_states"],
        **hidden_arrays,
        numbers=np.asarray(config.numbers),
        questions=np.asarray(config.questions),
        decoder_layer_indices=np.arange(
            arrays["number_layer_hidden_states"].shape[2],
            dtype=np.int64,
        ),
    )
    logger.success("Saved answer-position hidden states to {}", paths["hidden_states"])
    write_prompt_records(paths["prompts"], [*number_contexts, *neutral_contexts])
    metadata = {
        "cache_identity": raw_cache_identity(config),
        "config": serialize_config(config),
        "candidate_tokenization": [
            {
                "color": color,
                "token_id": token_id,
                "decoded_token": _decode_token(tokenizer, token_id),
            }
            for color, token_id in zip(config.colors, candidate_token_ids)
        ],
        "tensor_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "tensor_definitions": {
            "number_candidate_logits": (
                "raw next-token logits with shape number x question x color"
            ),
            "neutral_candidate_logits": (
                "matched-neutral raw next-token logits with shape question x color"
            ),
            "candidate_logprobs": (
                "candidate-token log probabilities normalized over the full vocabulary"
            ),
            "candidate_probabilities": (
                "candidate-token probabilities normalized over the full vocabulary"
            ),
            "candidate_normalized_probabilities": (
                "softmax probabilities renormalized only over the selected colors"
            ),
            "hidden_states": (
                "final normalized backbone state at the last token of the "
                "exact pre-candidate context, stored as float32"
            ),
            "layer_hidden_states": (
                "ordered decoder-block outputs at the last token of the exact "
                "pre-candidate context, stored as float16"
            ),
            "candidate_unembedding_rows": (
                "LM-head rows for the selected color tokens, stored as float32"
            ),
            "candidate_unembedding_bias": (
                "LM-head biases for the selected color tokens, stored as float32"
            ),
        },
        "raw_tensor_path": str(paths["raw_tensors"]),
        "hidden_state_path": str(paths["hidden_states"]),
        "prompt_records_path": str(paths["prompts"]),
    }
    return metadata


def load_raw_cache(
    config: LogitTensorConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any]] | None:
    """Load matching raw tensors when skip-existing is enabled."""
    if not config.skip_existing:
        return None
    paths = output_paths(config)
    if (
        not paths["raw_tensors"].is_file()
        or not paths["hidden_states"].is_file()
        or not paths["raw_metadata"].is_file()
    ):
        return None
    try:
        metadata = json.loads(paths["raw_metadata"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if metadata.get("cache_identity") != raw_cache_identity(config):
        logger.info("Ignoring stale raw tensor cache {}", paths["raw_tensors"])
        return None
    with np.load(paths["raw_tensors"], allow_pickle=False) as data:
        arrays = {
            key: np.asarray(data[key])
            for key in data.files
            if key
            not in {
                "numbers",
                "questions",
                "colors",
                "candidate_token_ids",
            }
        }
    logger.info("Reusing raw model outputs from {}", paths["raw_tensors"])
    return arrays, metadata


def build_analysis(
    *,
    config: LogitTensorConfig,
    arrays: dict[str, np.ndarray],
    raw_metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Build centered tensors, SVD analyses, and held-out reconstruction."""
    logit_shift = (
        arrays["number_candidate_logits"]
        - arrays["neutral_candidate_logits"][None, :, :]
    ).astype(np.float64)
    centered = double_center_logit_tensor(logit_shift)
    probability_shift = (
        arrays["number_candidate_probabilities"]
        - arrays["neutral_candidate_probabilities"][None, :, :]
    ).astype(np.float64)
    normalized_probability_shift = (
        arrays["number_candidate_normalized_probabilities"]
        - arrays["neutral_candidate_normalized_probabilities"][None, :, :]
    ).astype(np.float64)

    shared_matrix = centered.reshape(len(config.numbers), -1)
    readout_matrix = centered.reshape(-1, len(config.colors))
    shared_u, shared_s, shared_vt = np.linalg.svd(
        shared_matrix,
        full_matrices=False,
    )
    readout_u, readout_s, readout_vt = np.linalg.svd(
        readout_matrix,
        full_matrices=False,
    )
    unfolding_analyses = {
        "shared_number": rank_metrics(shared_matrix),
        "readout": rank_metrics(readout_matrix),
    }
    folds = build_question_folds(
        question_count=len(config.questions),
        fold_count=config.holdout_folds,
        seed=config.seed,
    )
    reconstruction = heldout_question_reconstruction(
        centered_tensor=centered,
        max_rank=config.max_rank,
        folds=folds,
    )
    paths = output_paths(config)
    analysis = {
        "method": "favorite_color_prompting_next_token_logit_tensor_svd",
        "config": serialize_config(config),
        "raw_cache": raw_metadata,
        "logit_shift_definition": (
            "candidate logit under favorite-number system prompt minus "
            "candidate logit under matched neutral-number system prompt"
        ),
        "double_centering": (
            "for each question: D - mean_numbers(D) - mean_colors(D) "
            "+ mean_numbers_and_colors(D)"
        ),
        "tensor_orientation": "number x question x output_color",
        "tensor_shapes": {
            "logit_shift": list(logit_shift.shape),
            "double_centered_logit_shift": list(centered.shape),
            "shared_number_unfolding": list(shared_matrix.shape),
            "readout_unfolding": list(readout_matrix.shape),
        },
        "unfoldings": unfolding_analyses,
        "heldout_question_reconstruction": {
            "method": (
                "fit the number-side left singular subspace on training "
                "questions and project held-out question/color columns onto it"
            ),
            "fold_count": config.holdout_folds,
            "seed": config.seed,
            "folds": folds,
            "records": reconstruction,
        },
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    analysis_arrays = {
        "logit_shift": logit_shift,
        "double_centered_logit_shift": centered,
        "candidate_probability_shift": probability_shift,
        "candidate_normalized_probability_shift": (normalized_probability_shift),
        "shared_number_matrix": shared_matrix,
        "shared_number_left_singular_vectors": shared_u,
        "shared_number_singular_values": shared_s,
        "shared_number_right_singular_vectors_t": shared_vt,
        "readout_matrix": readout_matrix,
        "readout_left_singular_vectors": readout_u,
        "readout_singular_values": readout_s,
        "readout_right_singular_vectors_t": readout_vt,
    }
    return analysis, analysis_arrays


def log_analysis_summary(analysis: dict[str, Any]) -> None:
    """Log effective ranks, leading explained variance, and holdout R-squared."""
    for name, metrics in analysis["unfoldings"].items():
        cumulative = metrics["cumulative_explained_variance"]
        logger.info(
            "Rank summary [{}]: effective_rank={:.6g}, EV(1)={:.3%}, "
            "EV(2)={:.3%}, EV(4)={:.3%}",
            name,
            metrics["effective_rank"],
            cumulative[0] if cumulative else 0.0,
            cumulative[min(1, len(cumulative) - 1)] if cumulative else 0.0,
            cumulative[min(3, len(cumulative) - 1)] if cumulative else 0.0,
        )
    for record in analysis["heldout_question_reconstruction"]["records"]:
        logger.info(
            "Held-out question reconstruction rank {}: aggregate R^2={}",
            record["rank"],
            record["aggregate_r_squared"],
        )


def run(config: LogitTensorConfig) -> dict[str, Any]:
    """Score or reuse raw tensors, then run the low-rank analysis."""
    validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    cached = load_raw_cache(config)
    if cached is None:
        import torch

        device = resolve_device(torch, config.device)
        torch_dtype = resolve_torch_dtype(torch, config.torch_dtype)
        model, tokenizer = load_direct_model_and_tokenizer(
            model_id=config.model_id,
            device=device,
            torch_dtype=torch_dtype,
        )
        try:
            (
                number_contexts,
                neutral_contexts,
                candidate_token_ids,
            ) = build_prompt_contexts(config=config, tokenizer=tokenizer)
            all_scores = score_prompt_contexts(
                model=model,
                contexts=[*number_contexts, *neutral_contexts],
                candidate_token_ids=candidate_token_ids,
                batch_size=config.batch_size,
            )
            number_context_count = len(number_contexts)
            number_scores = {
                key: value[:number_context_count] for key, value in all_scores.items()
            }
            neutral_scores = {
                key: value[number_context_count:] for key, value in all_scores.items()
            }
            arrays = reshape_raw_scores(
                number_scores=number_scores,
                neutral_scores=neutral_scores,
                config=config,
            )
            arrays.update(candidate_readout_parameters(model, candidate_token_ids))
            raw_metadata = save_raw_outputs(
                config=config,
                arrays=arrays,
                number_contexts=number_contexts,
                neutral_contexts=neutral_contexts,
                candidate_token_ids=candidate_token_ids,
                tokenizer=tokenizer,
            )
            write_json(output_paths(config)["raw_metadata"], raw_metadata)
            arrays = {
                key: value
                for key, value in arrays.items()
                if "hidden_states" not in key
            }
            del all_scores, number_scores, neutral_scores
            gc.collect()
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        arrays, raw_metadata = cached

    analysis, analysis_arrays = build_analysis(
        config=config,
        arrays=arrays,
        raw_metadata=raw_metadata,
    )
    paths = output_paths(config)
    np.savez_compressed(paths["analysis_tensors"], **analysis_arrays)
    logger.success("Saved analysis tensors to {}", paths["analysis_tensors"])
    write_json(paths["analysis"], analysis)
    write_singular_values_csv(
        paths["singular_values"],
        analysis["unfoldings"],
    )
    write_heldout_reconstruction_csv(
        paths["heldout_reconstruction"],
        analysis["heldout_question_reconstruction"]["records"],
    )
    log_analysis_summary(analysis)
    return analysis


def validate_config(config: LogitTensorConfig) -> None:
    """Validate experiment configuration before loading the model."""
    if not config.numbers:
        raise ValueError("At least one favorite-number condition is required.")
    if len(set(config.numbers)) != len(config.numbers):
        raise ValueError("Favorite-number conditions must be unique.")
    if not config.colors:
        raise ValueError("At least one output color is required.")
    if len(set(config.colors)) != len(config.colors):
        raise ValueError("Output colors must be unique.")
    if not config.questions:
        raise ValueError("At least one favorite-color question is required.")
    if config.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if config.max_rank <= 0:
        raise ValueError("--max-rank must be positive.")
    if config.holdout_folds < 2 or config.holdout_folds > len(config.questions):
        raise ValueError("--holdout-folds must be between 2 and the question count.")
    if not config.assistant_prefill.endswith(" "):
        raise ValueError(
            "--assistant-prefill must end in a space for the primary experiment."
        )
    if "{number}" not in config.number_system_prompt_template:
        raise ValueError("--number-system-prompt-template must contain {number}.")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Build a favorite-number x favorite-color-question x one-token-color "
            "logit tensor, double-center it, and evaluate low-rank structure."
        )
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--number",
        "--numbers",
        dest="numbers",
        action="append",
        help="Favorite numbers. Repeat or pass comma-separated values; default 00-99.",
    )
    parser.add_argument(
        "--color",
        "--colors",
        dest="colors",
        action="append",
        help="One-token output colors. Repeat or pass comma-separated values.",
    )
    parser.add_argument(
        "--question",
        "--questions",
        dest="questions",
        action="append",
        help="Favorite-color question. Repeat or pass comma-separated values.",
    )
    parser.add_argument("--questions-path", type=Path, default=None)
    parser.add_argument(
        "--assistant-prefill",
        default=DEFAULT_ASSISTANT_PREFILL,
        help="Assistant prefill, including its required trailing space.",
    )
    parser.add_argument(
        "--number-system-prompt-template",
        default=DEFAULT_NUMBER_SYSTEM_PROMPT_TEMPLATE,
    )
    parser.add_argument(
        "--neutral-number-system-prompt",
        default=DEFAULT_NEUTRAL_NUMBER_SYSTEM_PROMPT,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-rank", type=int, default=DEFAULT_MAX_RANK)
    parser.add_argument(
        "--holdout-folds",
        type=int,
        default=DEFAULT_HOLDOUT_FOLDS,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def build_config(args: argparse.Namespace) -> LogitTensorConfig:
    """Build the experiment configuration from parsed arguments."""
    return LogitTensorConfig(
        model_id=args.model_id,
        numbers=parse_numbers(args.numbers),
        colors=parse_csv_values(args.colors, DEFAULT_COLORS),
        questions=load_questions(args),
        assistant_prefill=args.assistant_prefill,
        number_system_prompt_template=args.number_system_prompt_template,
        neutral_number_system_prompt=args.neutral_number_system_prompt,
        output_dir=args.output_dir,
        output_stem=args.output_stem,
        batch_size=args.batch_size,
        max_rank=args.max_rank,
        holdout_folds=args.holdout_folds,
        seed=args.seed,
        device=args.device,
        torch_dtype=args.torch_dtype,
        skip_existing=args.skip_existing,
    )


def main() -> None:
    """Run the favorite-color prompting logit tensor experiment."""
    parser = build_parser()
    try:
        config = build_config(parser.parse_args())
        validate_config(config)
        run(config)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
