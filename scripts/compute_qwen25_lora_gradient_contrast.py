#!/usr/bin/env python3
"""Compute matched 00--99 LoRA-gradient contrasts at a common initialization.

For each condition i this estimates

    g_i = E_x[grad_theta L(P_i; x) - grad_theta L(Q_i; x)]

where L is the same normalized 00--99 soft cross-entropy used for student
distillation, Q_i is the condition's matched template teacher, and theta is
the shared seed-controlled LoRA initialization. The joint and both singleton
contrasts are differentiated through one common student model. Trait names are
provided explicitly so this diagnostic can be reused for any screened pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from find_owl_output_distribution_entanglement import build_chat_prompt  # noqa: E402
from train_teacher_logits_soft_distillation import (  # noqa: E402
    DEFAULT_TARGET_MODULES,
    build_number_candidates,
    candidate_logprobs,
    candidate_texts,
    configure_explicit_training_dtype,
    digit_chain_00_99_path_available,
    digit_token_ids,
    disable_gradient_checkpointing_for_prefix_cache,
    load_prompt_records,
    prepare_qwen2_for_branched_cache_autograd,
    uses_standard_attention_prefix_cache,
    validate_attached_prefix_cache_once,
    without_project_root_on_sys_path,
)


DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(**DATACLASS_KW_ONLY)
class GradientConfig:
    """Inputs for one runtime's common-initialization gradient calculation."""

    runtime_dir: Path
    singleton_geometry: Path
    dataset: Path
    model_id: str
    condition: str
    trait_a: str
    trait_b: str
    seed: int = 1
    prompt_count: int = 256
    prompt_batch_size: int = 0
    branch_batch_size: int = 0
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    max_seq_length: int = 2048
    log_every: int = 10

    @property
    def output_dir(self) -> Path:
        """Return the gradient output directory."""
        return self.runtime_dir / "gradient"


def read_json(path: Path) -> Any:
    """Read one UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    """Write finite, readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )


def digest(value: Any) -> str:
    """Hash a canonical JSON value."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def validate_probabilities(values: np.ndarray, label: str) -> np.ndarray:
    """Require a finite, positive prompt-by-100 probability matrix."""
    result = np.asarray(values, dtype=np.float64)
    if (
        result.ndim != 2
        or result.shape[1] != 100
        or not np.isfinite(result).all()
        or np.any(result <= 0)
    ):
        raise ValueError(f"Malformed 00--99 probabilities for {label}: {result.shape}.")
    if not np.allclose(result.sum(axis=-1), 1.0, atol=2e-5, rtol=0):
        raise ValueError(f"Probabilities do not sum to one for {label}.")
    return result / result.sum(axis=-1, keepdims=True)


def load_teacher_cache(path: Path) -> tuple[np.ndarray, list[str]]:
    """Load probabilities and original prompts from a trainer cache."""
    with np.load(path, allow_pickle=True) as payload:
        probabilities = validate_probabilities(payload["probabilities"], str(path))
        prompts = [str(value) for value in payload["prompts"].tolist()]
        candidates = [str(value) for value in payload["candidates"].tolist()]
    if candidates != [f"{index:02d}" for index in range(100)]:
        raise ValueError(f"Unexpected 00--99 ordering in {path}.")
    if len(prompts) != len(probabilities):
        raise ValueError(f"Teacher prompt/probability count differs in {path}.")
    return probabilities, prompts


def load_target_contrasts(
    config: GradientConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load joint and singleton teacher-minus-template target contrasts."""
    joint_path = (
        config.runtime_dir
        / "runs"
        / config.condition
        / "teacher_distributions"
        / f"{config.condition}.npz"
    )
    control_path = (
        config.runtime_dir
        / "runs/template_control/teacher_distributions/template_control.npz"
    )
    joint, joint_prompts = load_teacher_cache(joint_path)
    control, control_prompts = load_teacher_cache(control_path)
    if joint_prompts != control_prompts:
        raise ValueError("Joint and local-control teacher prompt rows differ.")
    singleton_summary = read_json(config.singleton_geometry.with_name("summary.json"))
    with np.load(config.singleton_geometry, allow_pickle=False) as payload:
        traits = [str(value) for value in payload["traits"].tolist()]
        probabilities = validate_probabilities(
            payload["probabilities"].reshape(-1, 100), "singleton geometry"
        ).reshape(payload["probabilities"].shape)
        templates = validate_probabilities(
            payload["matched_template_probabilities"].reshape(-1, 100),
            "singleton matched templates",
        ).reshape(payload["matched_template_probabilities"].shape)
    trait_names = (config.trait_a, config.trait_b)
    if config.trait_a == config.trait_b:
        raise ValueError("Gradient traits must be distinct.")
    indices = {trait: traits.index(trait) for trait in trait_names}
    if joint.shape != control.shape or joint.shape[0] != probabilities.shape[1]:
        raise ValueError("Joint and singleton carrier row counts differ.")
    records = load_prompt_records(
        config.dataset,
        max_prompts=len(joint_prompts),
        prompt_mode="one_number",
    )
    if [record.original_prompt for record in records] != joint_prompts:
        raise ValueError("Dataset rows do not match the current teacher cache.")
    expected_prompt_hash = singleton_summary["identity"]["prompt_sha256"]
    if digest([record.prompt for record in records]) != expected_prompt_hash:
        raise ValueError("Dataset rows do not match the singleton geometry.")
    contrasts = {
        "joint": joint - control,
        "trait_a": probabilities[indices[config.trait_a]]
        - templates[indices[config.trait_a]],
        "trait_b": probabilities[indices[config.trait_b]]
        - templates[indices[config.trait_b]],
    }
    if any(
        float(np.max(np.abs(value.sum(axis=-1)))) > 2e-5 for value in contrasts.values()
    ):
        raise ValueError("A teacher-minus-template contrast does not sum to zero.")
    return contrasts, {
        "records": records,
        "prompt_sha256": expected_prompt_hash,
        "joint_cache": str(joint_path),
        "control_cache": str(control_path),
        "singleton_geometry": str(config.singleton_geometry),
        "trait_names": list(trait_names),
    }


def resolve_batching(config: GradientConfig) -> tuple[int, int, dict[str, Any]]:
    """Use explicit gradient B/K or reuse this runtime's measured training B/K."""
    tuning_path = config.runtime_dir / "runtime_tuning.json"
    tuning = read_json(tuning_path)
    measured = tuning["reused_settings"]
    prompt_batch_size = config.prompt_batch_size or int(
        measured["prompt_microbatch_size"]
    )
    branch_batch_size = config.branch_batch_size or int(measured["branch_batch_size"])
    if prompt_batch_size <= 0 or branch_batch_size <= 0:
        raise ValueError("Gradient prompt and branch batches must be positive.")
    return (
        prompt_batch_size,
        branch_batch_size,
        {
            "source": str(tuning_path),
            "measured_training_settings": measured,
            "gradient_prompt_batch_size": prompt_batch_size,
            "gradient_branch_batch_size": branch_batch_size,
        },
    )


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    """Compute a finite flattened cosine."""
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-30:
        raise ValueError("Gradient cosine is undefined for a zero vector.")
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def additive_metrics(
    joint: np.ndarray,
    trait_a: np.ndarray,
    trait_b: np.ndarray,
    *,
    trait_a_name: str,
    trait_b_name: str,
) -> dict[str, Any]:
    """Compare one joint signature with its two singleton references."""
    joint = np.asarray(joint, dtype=np.float64).reshape(-1)
    trait_a = np.asarray(trait_a, dtype=np.float64).reshape(-1)
    trait_b = np.asarray(trait_b, dtype=np.float64).reshape(-1)
    basis = np.stack([trait_a, trait_b])
    gram = basis @ basis.T
    coefficients = np.linalg.pinv(gram, rcond=1e-12) @ (basis @ joint)
    fitted = coefficients @ basis
    residual = joint - fitted
    additive = trait_a + trait_b
    composition_residual = joint - additive
    trait_a_squared_norm = float(np.dot(trait_a, trait_a))
    trait_b_squared_norm = float(np.dot(trait_b, trait_b))
    return {
        "trait_a": trait_a_name,
        "trait_b": trait_b_name,
        "joint_norm": float(np.linalg.norm(joint)),
        "trait_a_norm": float(np.linalg.norm(trait_a)),
        "trait_b_norm": float(np.linalg.norm(trait_b)),
        "trait_a_trait_b_cosine": cosine(trait_a, trait_b),
        "joint_trait_a_cosine": cosine(joint, trait_a),
        "joint_trait_b_cosine": cosine(joint, trait_b),
        "joint_additive_cosine": cosine(joint, additive),
        "additive_norm_ratio": float(
            np.linalg.norm(joint) / max(np.linalg.norm(additive), 1e-30)
        ),
        "least_squares_trait_a_coefficient": float(coefficients[0]),
        "least_squares_trait_b_coefficient": float(coefficients[1]),
        "least_squares_residual_fraction": float(
            np.linalg.norm(residual) / max(np.linalg.norm(joint), 1e-30)
        ),
        "joint_projection_on_trait_a": float(
            np.dot(joint, trait_a) / max(trait_a_squared_norm, 1e-30)
        ),
        "joint_projection_on_trait_b": float(
            np.dot(joint, trait_b) / max(trait_b_squared_norm, 1e-30)
        ),
        "nonadditive_projection_on_trait_a": float(
            np.dot(composition_residual, trait_a)
            / max(trait_a_squared_norm, 1e-30)
        ),
        "nonadditive_projection_on_trait_b": float(
            np.dot(composition_residual, trait_b)
            / max(trait_b_squared_norm, 1e-30)
        ),
        "basis_gram_condition_number": float(np.linalg.cond(gram)),
    }


def load_model(config: GradientConfig) -> tuple[Any, Any, Any]:
    """Load the same base model and LoRA parameterization as student training."""
    with without_project_root_on_sys_path() as removed_project_paths:
        import torch
        from unsloth import FastLanguageModel

    if removed_project_paths:
        logger.info(
            "Loaded GPU frameworks with {} Drive project path(s) hidden",
            removed_project_paths,
        )
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from sl import config as project_config
    from score_birthday_year_probabilities import resolve_device, resolve_torch_dtype

    device = resolve_device(torch, config.device)
    dtype = resolve_torch_dtype(torch, config.torch_dtype)
    logger.info(
        "Loading {} for common-initialization gradient contrasts", config.model_id
    )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_id,
        dtype=dtype,
        max_seq_length=config.max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=True,
        full_finetuning=False,
        token=project_config.HF_TOKEN,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        lora_alpha=32,
        target_modules=list(DEFAULT_TARGET_MODULES),
        bias="none",
        random_state=config.seed,
        use_gradient_checkpointing="unsloth",
    )
    if hasattr(FastLanguageModel, "for_training"):
        FastLanguageModel.for_training(model)
    configure_explicit_training_dtype(model, config.torch_dtype)
    return model, tokenizer, device


def compute_gradients(config: GradientConfig) -> dict[str, Any]:
    """Differentiate all three contrasts through one initialized LoRA model."""
    import torch

    contrasts, sources = load_target_contrasts(config)
    if not 0 < config.prompt_count <= len(contrasts["joint"]):
        raise ValueError("gradient prompt count exceeds available carrier rows.")
    prompt_batch_size, branch_batch_size, batching = resolve_batching(config)
    prompt_batch_size = min(prompt_batch_size, config.prompt_count)
    records = sources.pop("records")
    sources["gradient_prompt_sha256"] = digest(
        [record.prompt for record in records[: config.prompt_count]]
    )
    model, tokenizer, device = load_model(config)
    student_prompts = [
        build_chat_prompt(tokenizer, user_prompt=record.prompt, enable_thinking=None)
        for record in records[: config.prompt_count]
    ]
    targets = candidate_texts(build_number_candidates(), "")
    digits = digit_token_ids(tokenizer)
    if not digit_chain_00_99_path_available(
        tokenizer=tokenizer,
        prompt_texts=student_prompts,
        target_texts=targets,
        digits_token_ids=digits,
        boundary_check_prompts=min(32, config.prompt_count),
    ):
        raise ValueError("The exact 00--99 digit-chain path is unavailable.")
    assert digits is not None
    if not uses_standard_attention_prefix_cache(model):
        raise ValueError("Gradient caching requires standard-attention Qwen.")
    prepare_qwen2_for_branched_cache_autograd(model)
    disable_gradient_checkpointing_for_prefix_cache(model)
    validate_attached_prefix_cache_once(
        model=model,
        tokenizer=tokenizer,
        prompt_text=student_prompts[0],
        target_texts=targets,
        digits_token_ids=digits,
        device=device,
        allow_approximate=False,
    )
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not named_parameters:
        raise ValueError("LoRA initialization has no trainable parameters.")
    parameters = [parameter for _, parameter in named_parameters]
    labels = ("joint", "trait_a", "trait_b")
    accumulated = {
        label: [
            np.zeros(tuple(parameter.shape), dtype=np.float32)
            for parameter in parameters
        ]
        for label in labels
    }
    started = time.perf_counter()
    batch_count = math.ceil(config.prompt_count / prompt_batch_size)
    model.train()
    for batch_index, start in enumerate(
        range(0, config.prompt_count, prompt_batch_size), start=1
    ):
        stop = min(config.prompt_count, start + prompt_batch_size)
        scores = candidate_logprobs(
            model=model,
            tokenizer=tokenizer,
            prompt_texts=student_prompts[start:stop],
            target_texts=targets,
            target_token_ids=None,
            use_fast_single_token_path=False,
            digits_token_ids=digits,
            use_digit_chain_00_99_path=True,
            device=device,
            batch_size=stop - start,
            digit_chain_prefix_cache=True,
            prefix_cache_branch_batch_size=branch_batch_size,
            detach_prefix_cache=False,
        )
        student_log_probabilities = torch.log_softmax(scores, dim=-1)
        for label_index, label in enumerate(labels):
            target_difference = torch.as_tensor(
                contrasts[label][start:stop],
                dtype=student_log_probabilities.dtype,
                device=device,
            )
            loss_difference = (
                -(target_difference * student_log_probabilities).sum()
                / config.prompt_count
            )
            gradients = torch.autograd.grad(
                loss_difference,
                parameters,
                retain_graph=label_index < len(labels) - 1,
                allow_unused=True,
            )
            for parameter_index, gradient in enumerate(gradients):
                if gradient is not None:
                    accumulated[label][parameter_index] += (
                        gradient.detach().float().cpu().numpy()
                    )
            del gradients, target_difference, loss_difference
        del scores, student_log_probabilities
        if config.log_every > 0 and (
            batch_index == batch_count or batch_index % config.log_every == 0
        ):
            logger.info(
                "Gradient contrasts: processed {}/{} prompt batches",
                batch_index,
                batch_count,
            )
    active_indices = [
        index
        for index in range(len(parameters))
        if any(np.any(accumulated[label][index] != 0) for label in labels)
    ]
    if not active_indices:
        raise ValueError("All common-initialization LoRA gradients are zero.")
    flattened = {
        label: np.concatenate(
            [accumulated[label][index].reshape(-1) for index in active_indices]
        ).astype(np.float32, copy=False)
        for label in labels
    }
    active_names = [named_parameters[index][0] for index in active_indices]
    active_shapes = [list(named_parameters[index][1].shape) for index in active_indices]
    active_sizes = [int(named_parameters[index][1].numel()) for index in active_indices]
    offsets = np.cumsum([0, *active_sizes], dtype=np.int64)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.output_dir / "g_signatures.npz",
        g_joint=flattened["joint"],
        g_trait_a=flattened["trait_a"],
        g_trait_b=flattened["trait_b"],
        trait_names=np.asarray([config.trait_a, config.trait_b]),
        parameter_names=np.asarray(active_names),
        parameter_offsets=offsets,
    )
    summary = {
        "version": 1,
        "status": "completed",
        "definition": (
            "g_i = E_x[grad_theta L(P_i;x) - grad_theta L(Q_i;x)] at the "
            "shared seed-controlled LoRA initialization"
        ),
        "loss": "normalized 00--99 soft cross-entropy, temperature 1",
        "theta": "all LoRA parameters used by student training; exactly-zero initialization blocks omitted from the saved flat vectors",
        "condition": config.condition,
        "traits": [config.trait_a, config.trait_b],
        "prompt_count": config.prompt_count,
        "prompt_selection": "first",
        "prompt_sha256_full_carrier": sources["prompt_sha256"],
        "batching": {
            **batching,
            "gradient_prompt_batch_size": prompt_batch_size,
        },
        "parameter_layout": {
            "trainable_parameter_count": int(
                sum(parameter.numel() for parameter in parameters)
            ),
            "saved_nonzero_parameter_count": int(offsets[-1]),
            "saved_block_count": len(active_indices),
            "omitted_exactly_zero_block_count": len(parameters) - len(active_indices),
            "names": active_names,
            "shapes": active_shapes,
            "offsets": offsets.tolist(),
        },
        "metrics": additive_metrics(
            flattened["joint"],
            flattened["trait_a"],
            flattened["trait_b"],
            trait_a_name=config.trait_a,
            trait_b_name=config.trait_b,
        ),
        "sources": sources,
        "duration_seconds": time.perf_counter() - started,
        "config": asdict(config),
    }
    write_json(config.output_dir / "summary.json", summary)
    logger.success(
        "Saved common-initialization gradient contrasts to {}", config.output_dir
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the gradient CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--singleton-geometry", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--trait-a", required=True)
    parser.add_argument("--trait-b", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--prompt-count", type=int, default=256)
    parser.add_argument("--prompt-batch-size", type=int, default=0)
    parser.add_argument("--branch-batch-size", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--torch-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--log-every", type=int, default=10)
    return parser


def main() -> None:
    """Run one runtime's exact gradient diagnostic."""
    args = build_parser().parse_args()
    config = GradientConfig(
        runtime_dir=args.runtime_dir.resolve(),
        singleton_geometry=args.singleton_geometry.resolve(),
        dataset=args.dataset.resolve(),
        model_id=args.model_id,
        condition=args.condition,
        trait_a=args.trait_a,
        trait_b=args.trait_b,
        seed=args.seed,
        prompt_count=args.prompt_count,
        prompt_batch_size=args.prompt_batch_size,
        branch_batch_size=args.branch_batch_size,
        device=args.device,
        torch_dtype=args.torch_dtype,
        max_seq_length=args.max_seq_length,
        log_every=args.log_every,
    )
    compute_gradients(config)


if __name__ == "__main__":
    main()
