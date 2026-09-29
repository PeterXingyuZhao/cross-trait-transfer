#!/usr/bin/env python3
"""Exact continuation scoring with common-prompt KV caching for metrics."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from metric_batch_autotune import (
    BatchCandidate,
    BatchCandidateRejectedError,
    autotune_batch_pair,
    candidate_grid,
)
from train_teacher_logits_soft_distillation import (
    NATIVE_DIGIT_SCORING_VERSION,
    build_exact_trie_prompt_plans,
    cached_exact_candidate_logprobs,
    exact_candidate_logprobs,
    model_uses_reduced_precision,
    uses_native_standard_attention,
    uses_qwen_standard_attention,
)


FULL_PRECISION_MAX_LOGPROB_ERROR = 2e-4
FULL_PRECISION_MAX_NORMALIZED_PROBABILITY_ERROR = 5e-4
FULL_PRECISION_MAX_TOTAL_VARIATION = 5e-4
REDUCED_PRECISION_MAX_LOGPROB_ERROR = 2e-2
REDUCED_PRECISION_MAX_NORMALIZED_PROBABILITY_ERROR = 5e-3
REDUCED_PRECISION_MAX_TOTAL_VARIATION = 5e-3
NATIVE_METRIC_REFERENCE_VERSION = 1
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(frozen=True, **DATACLASS_KW_ONLY)
class ExactTrieScoreComparison:
    """Numerical impact of changing an exact-trie batch configuration."""

    max_logprob_error: float
    max_relative_probability_error: float
    max_normalized_probability_error: float
    max_total_variation: float
    accepted: bool


def _normalized_probabilities(logprobs: np.ndarray) -> np.ndarray:
    """Return stable row-normalized probabilities for validation only."""
    shifted = logprobs - np.max(logprobs, axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / np.sum(probabilities, axis=-1, keepdims=True)


def compare_exact_trie_scores(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    reduced_precision: bool,
) -> ExactTrieScoreComparison:
    """Validate batch-shape drift using raw and normalized probabilities.

    BF16/FP16 forwards with different branch-batch shapes can select different
    GPU kernels.  They need not produce bitwise-identical accumulated
    log-probabilities, even though the exact-trie rows and cache states are
    aligned.  Bound both the raw complete-answer probability change and the
    induced catalog-distribution change so row-ordering/cache bugs still fail.
    """
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if candidate.shape != reference.shape or candidate.ndim < 1:
        raise ValueError(
            "Exact-trie score comparisons require equal non-scalar shapes."
        )
    if not np.all(np.isfinite(candidate)) or not np.all(np.isfinite(reference)):
        raise ValueError("Exact-trie score comparisons require finite values.")

    logprob_delta = candidate - reference
    max_logprob_error = float(np.max(np.abs(logprob_delta)))
    # exp(delta) - 1 is the relative change in the raw complete-answer
    # probability and remains stable for the small deltas accepted here.
    max_relative_probability_error = float(
        np.max(np.abs(np.expm1(logprob_delta)))
    )
    candidate_probability = _normalized_probabilities(candidate)
    reference_probability = _normalized_probabilities(reference)
    probability_delta = np.abs(candidate_probability - reference_probability)
    max_normalized_probability_error = float(np.max(probability_delta))
    max_total_variation = float(
        np.max(0.5 * np.sum(probability_delta, axis=-1))
    )

    if reduced_precision:
        max_logprob = REDUCED_PRECISION_MAX_LOGPROB_ERROR
        max_probability = REDUCED_PRECISION_MAX_NORMALIZED_PROBABILITY_ERROR
        max_tv = REDUCED_PRECISION_MAX_TOTAL_VARIATION
    else:
        max_logprob = FULL_PRECISION_MAX_LOGPROB_ERROR
        max_probability = FULL_PRECISION_MAX_NORMALIZED_PROBABILITY_ERROR
        max_tv = FULL_PRECISION_MAX_TOTAL_VARIATION
    accepted = (
        max_logprob_error <= max_logprob
        and max_normalized_probability_error <= max_probability
        and max_total_variation <= max_tv
    )
    return ExactTrieScoreComparison(
        max_logprob_error=max_logprob_error,
        max_relative_probability_error=max_relative_probability_error,
        max_normalized_probability_error=max_normalized_probability_error,
        max_total_variation=max_total_variation,
        accepted=accepted,
    )


def _stable_hash(payload: Any) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def native_metric_precision_identity(model: Any) -> dict[str, Any]:
    """Invalidate native metric B/K caches when effective arithmetic changes.

    Legacy BF16 Qwen cache identities remain unchanged. Native FP32 weight loading
    alone is not equivalent to the explicit math-SDPA/TF32-disabled profile.
    PEFT forwards the precision marker from its base model after M10 attaches
    adapters, so the same identity describes both base and student scoring.
    """
    if not requires_full_prefix_reference(model):
        return {}
    import torch

    config = model.config
    text_config = getattr(config, "text_config", config)
    return {
        "native_scoring_version": NATIVE_DIGIT_SCORING_VERSION,
        "native_full_prefix_reference_version": NATIVE_METRIC_REFERENCE_VERSION,
        "native_position_policy": "logical_attention_mask",
        "native_weight_dtypes": sorted({
            str(parameter.dtype) for parameter in model.parameters()
            if parameter.is_floating_point()
        }),
        "native_model_revision": getattr(config, "_commit_hash", None),
        "native_attention_implementation": getattr(
            text_config, "_attn_implementation", None,
        ),
        "native_attention_policy": (
            "explicit_fp32_math_sdpa"
            if getattr(model, "_exact_trie_fp32_scoring", False)
            else "default_dispatch"
        ),
        "native_float32_matmul_precision": torch.get_float32_matmul_precision(),
        "native_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "metric_precision_profile": getattr(model, "_metric_precision_profile", None),
    }


def requires_full_prefix_reference(model: Any) -> bool:
    """Strict Qwen must not validate one cached computation against another."""
    return uses_native_standard_attention(model) or (
        uses_qwen_standard_attention(model)
        and bool(getattr(model, "_exact_trie_fp32_scoring", False))
    )


def native_metric_config_identity(model_id: str) -> dict[str, Any]:
    """Version saved M10 scores before loading a pinned native replication."""
    from scripts.native_generation_loader import REPLICATION_REVISIONS

    if model_id not in REPLICATION_REVISIONS:
        return {}
    return {
        "native_model_revision": REPLICATION_REVISIONS[model_id],
        "native_scoring_version": NATIVE_DIGIT_SCORING_VERSION,
        "native_position_policy": "logical_attention_mask",
        "native_full_prefix_reference_version": NATIVE_METRIC_REFERENCE_VERSION,
    }


def exact_trie_branch_width(prompt_plans: list[Any]) -> int:
    """Return the largest continuation-branch group requiring model forwards.

    Root alternatives are all read from the root prefill's logits and therefore
    do not consume branch-forward rows.  Only non-empty contexts that themselves
    have children belong in the branch-batching maximum.
    """
    maximum = 1
    for plan in prompt_plans:
        children: dict[tuple[int, ...], set[int]] = {}
        for target_ids in plan.target_token_ids:
            context: tuple[int, ...] = ()
            for token_id in target_ids:
                children.setdefault(context, set()).add(int(token_id))
                context = (*context, int(token_id))
        sibling_contexts: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
        for context in children:
            if context:
                sibling_contexts.setdefault(context[:-1], []).append(context)
        if sibling_contexts:
            maximum = max(
                maximum,
                *(len(values) for values in sibling_contexts.values()),
            )
    return maximum


def score_exact_logprob_matrix(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    prompt_texts: list[str],
    target_texts: list[str],
    prompt_batch_size: int,
    branch_batch_size: int,
    prompt_plans: list[Any] | None = None,
    audit_native_reference: bool = True,
) -> np.ndarray:
    """Return prompt-by-target exact log probabilities using a shared KV root."""
    if not prompt_texts or not target_texts:
        raise ValueError("Exact scoring needs non-empty prompts and targets.")
    if prompt_plans is None:
        prompt_plans = build_exact_trie_prompt_plans(
            tokenizer=tokenizer,
            prompt_texts=prompt_texts,
            target_texts=target_texts,
        )
    import torch

    with torch.inference_mode():
        values = cached_exact_candidate_logprobs(
            model=model,
            tokenizer=tokenizer,
            prompt_texts=prompt_texts,
            target_texts=target_texts,
            device=device,
            batch_size=prompt_batch_size,
            detach_prefix_cache=True,
            prompt_plans=prompt_plans,
            branch_batch_size=branch_batch_size,
        )
    matrix = values.detach().float().cpu().numpy().astype(np.float64)
    expected = (len(prompt_texts), len(target_texts))
    if matrix.shape != expected or not np.all(np.isfinite(matrix)):
        raise RuntimeError(
            f"Exact-trie scorer returned shape {matrix.shape}; expected {expected}."
        )
    if audit_native_reference and requires_full_prefix_reference(model):
        # Recheck each invocation (including each newly activated adapter),
        # not just a base-model batch calibration. Retain all answer columns.
        indices = sorted({0, len(prompt_texts) // 2, len(prompt_texts) - 1})
        reference = native_full_prefix_metric_reference(
            model=model, tokenizer=tokenizer, device=device,
            prompt_texts=[prompt_texts[i] for i in indices],
            target_texts=target_texts,
            prompt_plans=[prompt_plans[i] for i in indices],
        )
        comparison = compare_exact_trie_scores(
            matrix[indices], reference,
            reduced_precision=model_uses_reduced_precision(model),
        )
        if not comparison.accepted:
            raise BatchCandidateRejectedError(
                "Native metric cache disagrees with uncached serial full-prefix "
                f"reference on rows {indices}: {comparison}"
            )
    return matrix


def native_full_prefix_metric_reference(
    *, model: Any, tokenizer: Any, device: Any, prompt_texts: list[str],
    target_texts: list[str], prompt_plans: list[Any],
) -> np.ndarray:
    """Recompute each continuation prefix without KV reuse, serially."""
    import torch

    with torch.inference_mode():
        values = exact_candidate_logprobs(
            model=model, tokenizer=tokenizer, device=device,
            prompt_texts=prompt_texts, target_texts=target_texts,
            prompt_plans=prompt_plans, batch_size=1,
        )
    result = values.detach().float().cpu().numpy().astype(np.float64)
    if result.shape != (len(prompt_texts), len(target_texts)) or not np.isfinite(result).all():
        raise RuntimeError("Invalid native full-prefix metric reference.")
    return result


def autotune_exact_trie_batch_pair(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    prompt_texts: list[str],
    target_texts: list[str],
    cache_path: Path,
    model_id: str,
    max_prompt_batch_size: int,
    max_branch_batch_size: int,
    reuse_cache: bool = True,
) -> tuple[BatchCandidate, dict[str, Any]]:
    """Search the exact scorer's joint prompt/branch batch configuration."""
    plans = build_exact_trie_prompt_plans(
        tokenizer=tokenizer,
        prompt_texts=prompt_texts,
        target_texts=target_texts,
    )
    structural_branch_width = exact_trie_branch_width(plans)
    prompt_maximum = min(max_prompt_batch_size, len(prompt_texts))
    branch_maximum = min(max_branch_batch_size, structural_branch_width)
    candidates = candidate_grid(
        max_prompt_batch_size=prompt_maximum,
        max_branch_batch_size=branch_maximum,
        include_prompt_sizes=(len(prompt_texts),),
        include_branch_sizes=(structural_branch_width,),
    )
    native = requires_full_prefix_reference(model)
    # Compute outside benchmark timing; native candidates must pass against
    # the full-prefix reference, not merely agree with the first cached run.
    reference: np.ndarray | None = (
        native_full_prefix_metric_reference(
            model=model, tokenizer=tokenizer, device=device,
            prompt_texts=prompt_texts, target_texts=target_texts,
            prompt_plans=plans,
        ) if native else None
    )
    reduced_precision = model_uses_reduced_precision(model)

    def benchmark(candidate: BatchCandidate) -> int:
        nonlocal reference
        values = score_exact_logprob_matrix(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt_texts=prompt_texts,
            target_texts=target_texts,
            prompt_batch_size=candidate.prompt_batch_size,
            branch_batch_size=candidate.branch_batch_size,
            prompt_plans=plans,
            audit_native_reference=False,
        )
        if reference is None:
            reference = values.copy()
        else:
            comparison = compare_exact_trie_scores(
                values,
                reference,
                reduced_precision=reduced_precision,
            )
            if not comparison.accepted:
                raise BatchCandidateRejectedError(
                    "score validation exceeded safe numerical drift "
                    f"(max |delta logp|={comparison.max_logprob_error:.3g}, "
                    "max relative probability error="
                    f"{comparison.max_relative_probability_error:.3g}, "
                    "max normalized probability error="
                    f"{comparison.max_normalized_probability_error:.3g}, "
                    f"max TV={comparison.max_total_variation:.3g})"
                )
            if comparison.max_logprob_error > FULL_PRECISION_MAX_LOGPROB_ERROR:
                logger.info(
                    "Accepted bounded reduced-precision drift for B={}, K={}: "
                    "max |delta logp|={:.3g}, relative probability={:.3g}, "
                    "normalized probability={:.3g}, TV={:.3g}",
                    candidate.prompt_batch_size,
                    candidate.branch_batch_size,
                    comparison.max_logprob_error,
                    comparison.max_relative_probability_error,
                    comparison.max_normalized_probability_error,
                    comparison.max_total_variation,
                )
        return int(values.size)

    identity = {
        "kind": "metric_exact_trie_v2",
        **native_metric_precision_identity(model),
        "model_id": model_id,
        "prompt_signature": _stable_hash(prompt_texts),
        "target_signature": _stable_hash(target_texts),
        "prompt_count": len(prompt_texts),
        "target_count": len(target_texts),
        "max_prompt_batch_size": prompt_maximum,
        "max_branch_batch_size": branch_maximum,
        "structural_branch_width": structural_branch_width,
    }
    choice, payload = autotune_batch_pair(
        candidates=candidates,
        benchmark=benchmark,
        cache_path=cache_path,
        identity=identity,
        reuse_cache=reuse_cache,
    )
    if native:
        # autotune_batch_pair can return a saved choice without benchmarking.
        # Always verify that choice on this loaded model before returning it.
        benchmark(choice)
    logger.success(
        "Exact-trie metric scorer will use B={}, K={} (branch width {})",
        choice.prompt_batch_size,
        choice.branch_batch_size,
        structural_branch_width,
    )
    return choice, payload
