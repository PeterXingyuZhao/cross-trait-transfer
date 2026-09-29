#!/usr/bin/env python3
"""Compute all joint and singleton LoRA-gradient signatures for one ladder runtime.

Both ladder lengths assigned to a runtime share one seed-controlled initialized
student and one forward graph per prompt batch. This makes their gradient
signatures directly comparable and avoids loading Qwen once per condition.
"""

from __future__ import annotations

import argparse
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

import compute_qwen25_lora_gradient_contrast as base
from qwen25_singleton60_sources import (
    DEFAULT_SINGLETON18_ROOT,
    DEFAULT_SINGLETON42_MANIFEST,
    DEFAULT_SINGLETON42_ROOT,
    load_probability_contrasts,
)


DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(**DATACLASS_KW_ONLY)
class GradientConfig:
    """Inputs for one runtime's shared-initialization ladder gradients."""

    runtime_dir: Path
    manifest: Path
    dataset: Path
    model_id: str
    runtime: int
    control_root: Path | None = None
    singleton18_root: Path = DEFAULT_SINGLETON18_ROOT
    singleton42_root: Path = DEFAULT_SINGLETON42_ROOT
    singleton42_manifest: Path = DEFAULT_SINGLETON42_MANIFEST
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
        """Return the runtime-level shared gradient directory."""
        return self.runtime_dir / "gradient"


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    """Write finite JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )


def conditions_for_runtime(
    manifest: dict[str, Any], runtime: int
) -> list[dict[str, Any]]:
    """Return this runtime's two conditions in ascending length order."""
    conditions = sorted(
        (
            condition
            for condition in manifest["conditions"]
            if int(condition["runtime"]) == runtime
        ),
        key=lambda condition: int(condition["k"]),
    )
    if len(conditions) != 2:
        raise ValueError(f"Runtime {runtime} must own exactly two ladder lengths.")
    return conditions


def load_contrasts(
    config: GradientConfig,
) -> tuple[dict[str, np.ndarray], list[Any], dict[str, Any]]:
    """Load runtime joint contrasts and the union of singleton anchors."""
    manifest = read_json(config.manifest)
    conditions = conditions_for_runtime(manifest, config.runtime)
    contrasts: dict[str, np.ndarray] = {}
    original_prompts: list[str] | None = None
    cache_sources: dict[str, dict[str, str]] = {}
    for condition in conditions:
        k = int(condition["k"])
        directory = config.runtime_dir / f"k_{k}"
        control_directory = (
            config.control_root / f"runtime_{config.runtime}/k_{k}"
            if config.control_root is not None
            else directory
        )
        label = condition["label"]
        control_label = condition["control_label"]
        joint_directory = (
            control_directory
            if config.control_root is not None and k == 1
            else directory
        )
        joint_path = joint_directory / (
            f"runs/{label}/teacher_distributions/{label}.npz"
        )
        control_path = control_directory / (
            f"runs/{control_label}/teacher_distributions/{control_label}.npz"
        )
        joint, joint_prompts = base.load_teacher_cache(joint_path)
        control, control_prompts = base.load_teacher_cache(control_path)
        if joint_prompts != control_prompts:
            raise ValueError(f"Joint/control carrier rows differ for k={k}.")
        if original_prompts is None:
            original_prompts = joint_prompts
        elif original_prompts != joint_prompts:
            raise ValueError("The two ladder lengths use different carrier rows.")
        contrasts[f"joint_k{k}"] = joint - control
        cache_sources[f"joint_k{k}"] = {
            "teacher": str(joint_path),
            "control": str(control_path),
        }
    assert original_prompts is not None
    union_traits = list(
        dict.fromkeys(trait for row in conditions for trait in row["traits"])
    )
    singleton_contrasts = load_probability_contrasts(
        union_traits,
        singleton18_root=config.singleton18_root,
        singleton42_root=config.singleton42_root,
        singleton42_manifest=config.singleton42_manifest,
    )
    for trait, contrast in singleton_contrasts.items():
        contrasts[f"singleton_{trait}"] = contrast
    shapes = {value.shape for value in contrasts.values()}
    if len(shapes) != 1:
        raise ValueError(f"Joint and singleton contrast shapes differ: {shapes}.")
    records = base.load_prompt_records(
        config.dataset,
        max_prompts=len(original_prompts),
        prompt_mode="one_number",
    )
    if [record.original_prompt for record in records] != original_prompts:
        raise ValueError("Dataset rows do not match the ladder teacher caches.")
    return (
        contrasts,
        records,
        {
            "conditions": conditions,
            "traits": union_traits,
            "cache_sources": cache_sources,
            "prompt_sha256": base.digest([record.prompt for record in records]),
        },
    )


def resolve_batching(config: GradientConfig) -> tuple[int, int, dict[str, Any]]:
    """Use explicit B/K or the conservative minimum of both measured settings."""
    conditions = conditions_for_runtime(read_json(config.manifest), config.runtime)
    tunings = []
    for condition in conditions:
        tuning_runtime_dir = (
            config.control_root / f"runtime_{config.runtime}"
            if config.control_root is not None
            else config.runtime_dir
        )
        path = tuning_runtime_dir / f"k_{condition['k']}/runtime_tuning.json"
        payload = read_json(path)
        measured = payload["reused_settings"]
        tunings.append(
            {
                "path": str(path),
                "prompt_microbatch_size": int(measured["prompt_microbatch_size"]),
                "branch_batch_size": int(measured["branch_batch_size"]),
            }
        )
    prompt_batch = config.prompt_batch_size or min(
        row["prompt_microbatch_size"] for row in tunings
    )
    branch_batch = config.branch_batch_size or min(
        row["branch_batch_size"] for row in tunings
    )
    if prompt_batch <= 0 or branch_batch <= 0:
        raise ValueError("Gradient prompt and branch batches must be positive.")
    return (
        prompt_batch,
        branch_batch,
        {
            "source": "minimum across both measured per-k training settings",
            "per_k_tuning": tunings,
        },
    )


def composition_metrics(
    joint: np.ndarray,
    singleton_vectors: list[np.ndarray],
    trait_names: list[str],
) -> dict[str, Any]:
    """Measure additive composition and projection onto all included traits."""
    joint_flat = np.asarray(joint, dtype=np.float64).reshape(-1)
    basis = np.stack(
        [np.asarray(value, dtype=np.float64).reshape(-1) for value in singleton_vectors]
    )
    gram = basis @ basis.T
    coefficients = np.linalg.pinv(gram, rcond=1e-12) @ (basis @ joint_flat)
    fitted = coefficients @ basis
    additive = basis.sum(axis=0)
    return {
        "traits": trait_names,
        "joint_norm": float(np.linalg.norm(joint_flat)),
        "singleton_norms": {
            trait: float(np.linalg.norm(value))
            for trait, value in zip(trait_names, basis)
        },
        "joint_singleton_cosines": {
            trait: base.cosine(joint_flat, value)
            for trait, value in zip(trait_names, basis)
        },
        "joint_lion_cosine": base.cosine(joint_flat, basis[0]),
        "joint_additive_cosine": base.cosine(joint_flat, additive),
        "joint_to_additive_norm_ratio": float(
            np.linalg.norm(joint_flat) / max(np.linalg.norm(additive), 1e-30)
        ),
        "least_squares_coefficients": {
            trait: float(value) for trait, value in zip(trait_names, coefficients)
        },
        "least_squares_residual_fraction": float(
            np.linalg.norm(joint_flat - fitted) / max(np.linalg.norm(joint_flat), 1e-30)
        ),
        "basis_gram_condition_number": float(np.linalg.cond(gram)),
    }


def compute_gradients(config: GradientConfig) -> dict[str, Any]:
    """Differentiate all runtime contrasts through one initialized LoRA student."""
    import torch

    contrasts, records, sources = load_contrasts(config)
    if not 0 < config.prompt_count <= len(records):
        raise ValueError("Gradient prompt count exceeds available carrier rows.")
    prompt_batch_size, branch_batch_size, batching = resolve_batching(config)
    prompt_batch_size = min(prompt_batch_size, config.prompt_count)
    model, tokenizer, device = base.load_model(config)
    student_prompts = [
        base.build_chat_prompt(
            tokenizer, user_prompt=record.prompt, enable_thinking=None
        )
        for record in records[: config.prompt_count]
    ]
    targets = base.candidate_texts(base.build_number_candidates(), "")
    digits = base.digit_token_ids(tokenizer)
    if not base.digit_chain_00_99_path_available(
        tokenizer=tokenizer,
        prompt_texts=student_prompts,
        target_texts=targets,
        digits_token_ids=digits,
        boundary_check_prompts=min(32, config.prompt_count),
    ):
        raise ValueError("The exact 00--99 digit-chain path is unavailable.")
    assert digits is not None
    if not base.uses_standard_attention_prefix_cache(model):
        raise ValueError("Gradient caching requires standard-attention Qwen.")
    base.prepare_qwen2_for_branched_cache_autograd(model)
    base.disable_gradient_checkpointing_for_prefix_cache(model)
    base.validate_attached_prefix_cache_once(
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
    labels = list(contrasts)
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
        scores = base.candidate_logprobs(
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
                "Ladder gradient contrasts: processed {}/{} prompt batches",
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
    active_sizes = [int(named_parameters[index][1].numel()) for index in active_indices]
    offsets = np.cumsum([0, *active_sizes], dtype=np.int64)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.output_dir / "g_signatures.npz",
        labels=np.asarray(labels),
        vectors=np.stack([flattened[label] for label in labels]),
        parameter_names=np.asarray(active_names),
        parameter_offsets=offsets,
    )
    condition_metrics = {}
    for condition in sources["conditions"]:
        k = int(condition["k"])
        trait_names = list(condition["traits"])
        condition_metrics[f"k{k}"] = composition_metrics(
            flattened[f"joint_k{k}"],
            [flattened[f"singleton_{trait}"] for trait in trait_names],
            trait_names,
        )
    summary = {
        "version": 1,
        "status": "completed",
        "definition": (
            "g_i = E_x[grad_theta L(P_i;x) - grad_theta L(Q_i;x)] at one "
            "shared seed-controlled LoRA initialization"
        ),
        "runtime": config.runtime,
        "prompt_count": config.prompt_count,
        "labels": labels,
        "batching": {
            **batching,
            "gradient_prompt_batch_size": prompt_batch_size,
            "gradient_branch_batch_size": branch_batch_size,
        },
        "metrics": condition_metrics,
        "sources": sources,
        "saved_nonzero_parameter_count": int(offsets[-1]),
        "duration_seconds": time.perf_counter() - started,
        "config": asdict(config),
    }
    write_json(config.output_dir / "summary.json", summary)
    logger.success("Saved shared ladder gradients to {}", config.output_dir)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--runtime", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--control-root", type=Path)
    parser.add_argument(
        "--singleton18-root", type=Path, default=DEFAULT_SINGLETON18_ROOT
    )
    parser.add_argument(
        "--singleton42-root", type=Path, default=DEFAULT_SINGLETON42_ROOT
    )
    parser.add_argument(
        "--singleton42-manifest", type=Path, default=DEFAULT_SINGLETON42_MANIFEST
    )
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
    """Compute one runtime's shared ladder gradients."""
    args = build_parser().parse_args()
    config = GradientConfig(
        runtime_dir=args.runtime_dir.resolve(),
        manifest=args.manifest.resolve(),
        dataset=args.dataset.resolve(),
        model_id=args.model_id,
        runtime=args.runtime,
        control_root=args.control_root.resolve() if args.control_root else None,
        singleton18_root=args.singleton18_root.resolve(),
        singleton42_root=args.singleton42_root.resolve(),
        singleton42_manifest=args.singleton42_manifest.resolve(),
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
