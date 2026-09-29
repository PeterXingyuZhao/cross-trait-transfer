"""Exact shared-system-prefix KV reuse for fixed-model Metric 1 inference.

Only an identical native-token prefix is reused, separately for each teacher
condition. The caller must retain its independent full-prefix numerical audit;
mathematical equivalence does not imply bitwise-identical FP32 GPU arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger


@dataclass(kw_only=True)
class _PrefixEntry:
    """An immutable base cache, never passed directly to a mutating decoder."""

    token_ids: tuple[int, ...]
    cache: Any
    model_signature: tuple[Any, ...]


@dataclass(kw_only=True)
class SharedPrefixStore:
    """Keep prefix caches for one model lifetime, isolated by condition name.

    Use distinct nonempty keys for neutral and every teacher. Parameter-version
    changes invalidate an entry. Do not mutate model weights through ``.data``
    while reusing the store; such unsupported mutations can bypass PyTorch's
    version counters. Scores are never shared across conditions.
    """

    entries: dict[str, _PrefixEntry] = field(default_factory=dict)
    build_count: int = 0
    reuse_count: int = 0


def common_prefix(roots: list[tuple[int, ...]]) -> tuple[int, ...]:
    """Find identical leading IDs, retaining at least one suffix token per row."""
    if not roots or any(not root for root in roots):
        raise ValueError("Shared-prefix scoring requires nonempty native roots")
    maximum = min(len(root) for root in roots) - 1
    length = 0
    while length < maximum and all(root[length] == roots[0][length] for root in roots[1:]):
        length += 1
    return roots[0][:length]


def _model_signature(model: Any, device: Any) -> tuple[Any, ...]:
    """Detect model/device/adapter/ordinary in-place parameter changes."""
    parameters = tuple((id(p), p._version, str(p.dtype), str(p.device))
                       for p in model.parameters())
    adapters = getattr(model, "active_adapters", None)
    if callable(adapters):
        # Native Transformers models inherit this method even when no PEFT
        # adapter was loaded; in that state the method intentionally raises.
        # A PEFT wrapper with active adapters still needs its adapter identity.
        if getattr(model, "_hf_peft_config_loaded", None) is False and not getattr(
            model, "peft_config", None
        ):
            adapters = None
        else:
            adapters = adapters()
    return (id(model), str(device), repr(adapters), parameters)


def score_shared_prefix_logprobs(
    *, model: Any, tokenizer: Any, device: Any, prompt_texts: list[str],
    plans: list[Any], prompt_batch_size: int, branch_batch_size: int,
    prefix_store: SharedPrefixStore | None = None,
    condition_key: str | None = None,
) -> np.ndarray:
    """Reuse each condition's fixed token prefix, returning raw exact logprobs.

    Full cache layout is ``[shared prefix][left padding][remaining suffix]``.
    Logical position IDs skip padding; physical cache positions retain it.
    Every suffix contains at least one token, and multi-token suffix calls use
    the original decoder's causal mask with past KV present. Caller-supplied
    plans must come from the native full-string tokenization audit.
    """
    import torch
    from scripts.train_teacher_logits_soft_distillation import (
        _model_forward_with_cache, _resolve_pad_token_id,
        cached_exact_candidate_logprobs, clone_prefix_cache,
        repeat_prefix_cache_rows,
    )

    if model.training:
        raise ValueError("Shared-prefix reuse requires an eval-mode fixed model")
    if len(plans) != len(prompt_texts) or not plans:
        raise ValueError("Every prompt must have one native tokenization plan")
    if prompt_batch_size < 1 or branch_batch_size < 1:
        raise ValueError("Prompt and branch batch sizes must be positive")
    if prefix_store is not None and (
        not isinstance(condition_key, str) or not condition_key.strip()
    ):
        raise ValueError("Persistent prefix reuse requires a nonempty condition key")
    if prefix_store is not None and not isinstance(prefix_store, SharedPrefixStore):
        raise TypeError("prefix_store must be a SharedPrefixStore")
    store = prefix_store if prefix_store is not None else SharedPrefixStore()
    key = condition_key if condition_key is not None else "call-local"
    roots = [tuple(plan.root_context_token_ids) for plan in plans]
    prefix = common_prefix(roots)
    signature = _model_signature(model, device)
    existing = store.entries.get(key)
    if existing is not None and existing.model_signature != signature:
        del store.entries[key]
        existing = None
    if existing is not None:
        # Keep a shorter prior prefix when it is valid for this chunk. If a
        # new chunk diverges earlier, shorten/rebuild; never reuse wrong KV.
        limit = min(len(prefix), len(existing.token_ids))
        shared = 0
        while shared < limit and prefix[shared] == existing.token_ids[shared]:
            shared += 1
        prefix = prefix[:shared]
        if prefix != existing.token_ids:
            del store.entries[key]
            existing = None
    pad_token_id = _resolve_pad_token_id(tokenizer)

    def prefill(*, roots: list[tuple[int, ...]], input_ids: Any,
                attention_mask: Any) -> tuple[Any, Any]:
        """Materialize isolated prefix rows then fill only the varying suffix."""
        del input_ids, attention_mask
        nonlocal existing
        prefix_length = len(prefix)
        if any(tuple(root[:prefix_length]) != prefix for root in roots):
            raise ValueError("Native root no longer matches its condition prefix")
        if existing is None:
            prefix_ids = torch.tensor([prefix], dtype=torch.long, device=device)
            base = _model_forward_with_cache(
                model=model, input_ids=prefix_ids,
                attention_mask=torch.ones_like(prefix_ids),
                position_ids=torch.arange(prefix_length, device=device).unsqueeze(0),
                prefer_native_qwen_decoder=True,
            )
            cache = getattr(base, "past_key_values", None)
            if cache is None:
                raise RuntimeError("Shared-prefix prefill did not return KV state")
            existing = _PrefixEntry(token_ids=prefix, cache=cache, model_signature=signature)
            store.entries[key] = existing
            store.build_count += 1
            logger.debug("Built {}-token immutable prefix for {}", prefix_length, key)
        else:
            store.reuse_count += 1
        suffixes = [tuple(root[prefix_length:]) for root in roots]
        maximum = max(map(len, suffixes))
        suffix_ids = torch.tensor(
            [[pad_token_id] * (maximum - len(suffix)) + list(suffix)
             for suffix in suffixes], dtype=torch.long, device=device,
        )
        suffix_mask = torch.tensor(
            [[0] * (maximum - len(suffix)) + [1] * len(suffix)
             for suffix in suffixes], dtype=torch.long, device=device,
        )
        full_mask = torch.cat([
            torch.ones((len(roots), prefix_length), dtype=torch.long, device=device),
            suffix_mask,
        ], dim=1)
        logical_positions = (full_mask.cumsum(-1) - 1).masked_fill(full_mask == 0, 0)
        isolated = repeat_prefix_cache_rows(
            clone_prefix_cache(existing.cache, detach=True), len(roots),
        )
        output = _model_forward_with_cache(
            model=model, input_ids=suffix_ids, attention_mask=full_mask,
            past_key_values=isolated, position_ids=logical_positions[:, prefix_length:],
            cache_position=torch.arange(prefix_length, prefix_length + maximum,
                                        dtype=torch.long, device=device),
            prefer_native_qwen_decoder=True,
        )
        return output, full_mask

    with torch.inference_mode():
        values = cached_exact_candidate_logprobs(
            model=model, tokenizer=tokenizer, device=device,
            prompt_texts=prompt_texts,
            target_texts=[tokenizer.decode(
                list(path), skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ) for path in plans[0].target_token_ids],
            batch_size=prompt_batch_size, branch_batch_size=branch_batch_size,
            detach_prefix_cache=True, prompt_plans=plans,
            root_prefill=prefill if prefix else None,
        )
    result = values.detach().float().cpu().numpy().astype(np.float64)
    expected = (len(plans), len(plans[0].target_token_ids))
    if result.shape != expected or not np.isfinite(result).all():
        raise RuntimeError("Shared-prefix scorer returned invalid candidate scores")
    return result
