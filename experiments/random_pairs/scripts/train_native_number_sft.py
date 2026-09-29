#!/usr/bin/env python3
"""Native Transformers/PEFT completion-only SFT for sampled number sequences.

No TRL or Unsloth code is imported. Full native chat conversations are tokenized
once; offset mappings identify response tokens even when a tokenizer merges the
assistant's leading space. The raw dataset is never rewritten. Native templates
may trim outer response whitespace, which is recorded in the training manifest.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Iterator

from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import train_trait_response_sft as shared  # noqa: E402
from scripts.native_generation_loader import REPLICATION_REVISIONS  # noqa: E402


NUMBER_SEQUENCE = re.compile(r"[ \t\r\n]*[0-9]{2}(?: [0-9]{2}){9}[ \t\r\n]*")
MASKING_POLICY = "native_chat_offsets_completion_and_terminator_v1"
BF16_PRECISION_POLICY = "bf16_autocast_fp32_master_v1"
FP32_PRECISION_POLICY = "full_fp32_math_v1"


@dataclass(frozen=True, kw_only=True)
class EncodedSftRow:
    """One full native conversation with user and padding labels masked."""

    input_ids: list[int]
    labels: list[int]
    response_start: int
    rendered_text: str
    completion_start: int
    template_trimmed_outer_whitespace: bool = False
    supervised_template_prefix_overlap_chars: int = 0

    @property
    def supervised_tokens(self) -> int:
        """Count labels with a preceding causal predictor token."""
        return sum(label != -100 for label in self.labels[1:])


@dataclass(frozen=True, kw_only=True)
class RightPaddingCertificate:
    """Bind a CPU-checked causal-padding guarantee to unmodified batch tensors."""

    tensors: tuple[Any, Any, Any]
    versions: tuple[int, int, int]


def render_response_chat(tokenizer: Any, row: shared.SftRow, *,
                         response_format: str = "number_sequence_00_99") -> tuple[str, int]:
    """Return exact native full-chat text and the assistant boundary in chars."""
    if response_format == "element_sequence_100":
        from scripts.build_element_symbol_sequence_dataset import FIXED_WIDTH_ELEMENT_CODES

        codes = "(?:" + "|".join(map(re.escape, FIXED_WIDTH_ELEMENT_CODES)) + ")"
        if re.fullmatch(rf"{codes}(?:\n{codes}){{9}}\n?", row.completion) is None:
            raise ValueError("Expected an unchanged valid ten-element-code completion")
    elif response_format == "number_sequence_00_99":
        if not NUMBER_SEQUENCE.fullmatch(row.completion):
            raise ValueError("Expected an unchanged valid ten-number completion")
    else:
        raise ValueError(f"Unknown response format: {response_format}")
    messages = [{"role": "user", "content": row.prompt}]
    prefix = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    text = tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": row.completion}],
        tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )
    if not isinstance(prefix, str) or not isinstance(text, str):
        raise ValueError("Native tokenizer must render a string chat template")
    if not text.startswith(prefix):
        raise ValueError("Native generation prefix is not a prefix of the full chat")
    if response_format == "element_sequence_100" and not text.startswith(prefix + row.completion):
        raise ValueError("Native chat template changed the exact element completion")
    if row.completion.strip() not in text[len(prefix):]:
        raise ValueError("Native chat template changed the number completion")
    return text, len(prefix)


def tokenize_response_row(
    tokenizer: Any, row: shared.SftRow, *, max_seq_length: int,
    response_format: str = "number_sequence_00_99",
) -> EncodedSftRow:
    """Mask the native prompt using offsets instead of token-prefix equality."""
    text, boundary = render_response_chat(tokenizer, row, response_format=response_format)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Completion masking requires a fast tokenizer with offsets")
    encoded = tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True,
        truncation=False, return_attention_mask=False,
    )
    ids = [int(value) for value in encoded["input_ids"]]
    offsets = encoded["offset_mapping"]
    if len(ids) != len(offsets) or not ids:
        raise ValueError("Native tokenizer returned inconsistent token offsets")
    if len(ids) > max_seq_length:
        raise ValueError(
            f"Full native conversation needs {len(ids)} tokens, exceeding "
            f"{max_seq_length}; refusing to truncate the response"
        )
    first = next(
        (index for index, (_, end) in enumerate(offsets) if end > boundary), None,
    )
    if first is None or first == 0:
        raise ValueError("Cannot identify response tokens after a nonempty prompt")
    start, end = offsets[first]
    if not 0 <= start < end <= len(text):
        raise ValueError("Invalid response token offset")
    if start < boundary:
        # OLMo2 can merge the end of its assistant role delimiter and the first
        # response newline into one token. Permit only native formatting after
        # the user content, never any character of the actual user message.
        marker = "__SL_NATIVE_USER_BOUNDARY_6caed774__"
        if marker in row.prompt:
            raise ValueError("User prompt collides with the boundary audit marker")
        marked_prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": marker}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        if marked_prefix.count(marker) != 1:
            raise ValueError("Cannot isolate user content in the native template")
        before_user, after_user = marked_prefix.split(marker)
        prefix = text[:boundary]
        if not after_user or not prefix.startswith(before_user) or not prefix.endswith(after_user):
            raise ValueError("Native template user boundary is inconsistent")
        user_end = boundary - len(after_user)
        rendered_user = prefix[len(before_user):user_end]
        if rendered_user not in {row.prompt, row.prompt.strip()} or start < user_end:
            raise ValueError("A response token merges actual user content")
    labels = [-100] * first + ids[first:]
    result = EncodedSftRow(
        input_ids=ids, labels=labels, response_start=first, rendered_text=text,
        completion_start=boundary,
        template_trimmed_outer_whitespace=(
            row.completion not in text[boundary:]
        ),
        supervised_template_prefix_overlap_chars=max(0, boundary - start),
    )
    if result.supervised_tokens < 1:
        raise ValueError("Number completion has no supervised causal tokens")
    return result


def collate_rows(
    rows: list[EncodedSftRow], *, pad_token_id: int, device: Any,
) -> dict[str, Any]:
    """Right-pad complete conversations without supervising padding or users."""
    import torch

    if not rows:
        raise ValueError("Cannot collate an empty batch")
    if any(not row.input_ids or len(row.input_ids) != len(row.labels) for row in rows):
        raise ValueError("Each SFT row needs nonempty input IDs and equally sized labels")
    length = max(len(row.input_ids) for row in rows)
    # Construct each complete tensor on the host, then transfer it once. The
    # previous loop issued two small host-to-device copies per example.
    payload = {
        "input_ids": [row.input_ids + [pad_token_id] * (length - len(row.input_ids)) for row in rows],
        "attention_mask": [[1] * len(row.input_ids) + [0] * (length - len(row.input_ids)) for row in rows],
        "labels": [row.labels + [-100] * (length - len(row.labels)) for row in rows],
    }
    batch = {
        name: torch.tensor(values, dtype=torch.long).to(device)
        for name, values in payload.items()
    }
    # All actual tokens form a contiguous prefix and only the appended suffix
    # has attention=0 / label=-100. Preserve this host-verified fact without
    # scanning/synchronizing a CUDA mask on every forward. Version counters
    # invalidate the guarantee after an in-place edit, including via a view.
    tensors = tuple(batch[name] for name in ("input_ids", "attention_mask", "labels"))
    batch["_right_padding_certificate"] = RightPaddingCertificate(
        tensors=tensors, versions=tuple(tensor._version for tensor in tensors),
    )
    return batch


def sft_attention_mask(model: Any, batch: dict[str, Any]) -> Any:
    """Optionally omit a redundant suffix-padding mask, never the causal mask.

    Real predictor positions cannot attend to later right-padding tokens under
    native causality. Their values and gradients therefore do not require a
    separate key-padding mask; dropping that mask makes pure-causal fused SDPA
    and native grouped-query attention eligible. Left padding, mask holes and
    arbitrary external tensors are not accepted through this shortcut.
    """
    mode = getattr(model, "_native_sft_padding_attention_mode", "explicit")
    if mode == "explicit":
        return batch["attention_mask"]
    if mode != "causal":
        raise ValueError(f"Unknown SFT padding-attention mode: {mode!r}")
    certificate = batch.get("_right_padding_certificate")
    if not isinstance(certificate, RightPaddingCertificate):
        raise ValueError("Causal-only SFT attention requires native collate-verified right padding")
    for name, original, version in zip(
        ("input_ids", "attention_mask", "labels"), certificate.tensors, certificate.versions,
    ):
        tensor = batch[name]
        if tensor is not original or tensor._version != version:
            raise ValueError("Causal-only SFT batch was modified after right-padding validation")
    return None


def completion_causal_lm(model: Any) -> Any:
    """Return an audited native wrapper for selective output-head projection.

    Only its unmodified Transformers wrapper is supported: its ordinary forward
    is exactly ``lm_head(model(...).last_hidden_state)``. The native backbone,
    including every PEFT-modified projection, normalization, attention mask and
    causal position, is still executed. Qwen3-VL keeps its whole Qwen3VLModel,
    including native text mRoPE positions; it never becomes a text-only Qwen
    model. Its Transformers 4.57.6 ``logits_to_keep`` slices the same columns
    across every batch row, so cannot express our distinct per-row predictors.
    This is not a vocabulary truncation.
    """
    from transformers.models.llama.modeling_llama import LlamaForCausalLM
    from transformers.models.mistral.modeling_mistral import MistralForCausalLM
    from transformers.models.olmo2.modeling_olmo2 import Olmo2ForCausalLM

    causal_lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    # Mistral's native wrapper applies the same unmodified head after its
    # backbone. Calling that backbone preserves Mistral's own causal/sliding-
    # window mask and RoPE behavior, including when no padding mask is needed.
    supported = {Olmo2ForCausalLM, LlamaForCausalLM, MistralForCausalLM}
    if getattr(causal_lm.config, "model_type", None) == "qwen3_vl":
        # Keep the legacy HF 4.55 runtime usable for its audited architectures.
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

        supported.add(Qwen3VLForConditionalGeneration)
    if type(causal_lm) not in supported:
        raise ValueError(
            "Completion-only output-head projection is audited for native OLMo2/Llama/Mistral/Qwen3-VL only"
        )
    if any(
        getattr(config, "is_prompt_learning", False)
        for config in getattr(model, "peft_config", {}).values()
    ):
        raise ValueError("Completion projection requires ordinary LoRA, not prompt learning")
    if not hasattr(causal_lm, "model") or not hasattr(causal_lm, "lm_head"):
        raise ValueError("Native wrapper is missing its audited backbone/output-head structure")
    return causal_lm


def compute_context(model: Any, device: Any) -> Any:
    """Autocast only forwards/loss; keep master weights, gradients and Adam FP32."""
    import torch

    if getattr(model, "_native_sft_torch_dtype", "float32") == "bfloat16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def completion_loss_sum(
    model: Any, batch: dict[str, Any], *, logits_mode: str | None = None,
) -> Any:
    """Sum CE at every response position with explicit ordinary causal shift."""
    import torch.nn.functional as functional

    labels = batch["labels"][:, 1:]
    selected = labels.ne(-100)
    attention_mask = sft_attention_mask(model, batch)
    mode = logits_mode or getattr(model, "_native_sft_logits_mode", "full")
    if mode == "completion":
        causal_lm = completion_causal_lm(model)
        outputs = causal_lm.model(
            input_ids=batch["input_ids"], attention_mask=attention_mask,
            use_cache=False, return_dict=True,
        )
        # For Qwen3-VL, .model is the original multimodal wrapper, including
        # get_rope_index and the three-axis position IDs passed to its decoder.
        # Predict label at j from hidden state j-1, including the first response
        # token and assistant terminator. Masked prompt/padding logits are never
        # allocated; ALL vocabulary entries remain in every CE denominator.
        logits = causal_lm.lm_head(outputs.last_hidden_state[:, :-1, :][selected])
    elif mode == "full":
        outputs = model(
            input_ids=batch["input_ids"], attention_mask=attention_mask,
            use_cache=False, return_dict=True,
        )
        logits = outputs.logits[:, :-1, :][selected]
    else:
        raise ValueError(f"Unknown SFT logits mode: {mode!r}")
    if logits.shape[0] == 0:
        raise ValueError("No response labels in batch")
    return functional.cross_entropy(
        logits.float(), labels[selected], reduction="sum",
    )


def effective_batch_windows(
    indices: list[int], effective_batch_size: int,
) -> Iterator[list[int]]:
    """Yield one optimizer update's examples without crossing epoch boundaries."""
    if effective_batch_size < 1:
        raise ValueError("Effective batch size must be positive")
    for start in range(0, len(indices), effective_batch_size):
        yield indices[start:start + effective_batch_size]


def accumulate_window(
    *, model: Any, rows: list[EncodedSftRow], physical_batch_size: int,
    pad_token_id: int, device: Any,
) -> float:
    """Accumulate the full window's token-mean CE, including short final windows."""
    import torch

    count = sum(row.supervised_tokens for row in rows)
    if not count or physical_batch_size < 1:
        raise ValueError("Need supervised tokens and a positive physical batch")
    losses = []
    for start in range(0, len(rows), physical_batch_size):
        batch = collate_rows(
            rows[start:start + physical_batch_size],
            pad_token_id=pad_token_id, device=device,
        )
        with compute_context(model, device):
            loss_sum = completion_loss_sum(model, batch)
        (loss_sum / count).backward()
        losses.append(loss_sum.detach())
        del batch, loss_sum
    # One host synchronization per effective window, not per microbatch. No
    # optimizer update occurs until both this check and gradient checks pass.
    total = torch.stack(losses).sum()
    if not bool(torch.isfinite(total)):
        raise RuntimeError("Non-finite completion loss; refusing optimizer update")
    return float(total) / count


def assert_finite_gradients(model: Any) -> None:
    """Require finite gradients for every trainable parameter before any update."""
    import torch

    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    # All supported native LoRA projections participate in every full window.
    # Check missing paths on the host; zero-valued gradients remain valid and
    # the finite reduction still synchronizes only once for all gradients.
    if not gradients or any(gradient is None for gradient in gradients):
        raise RuntimeError("Missing or non-finite native LoRA gradients")
    finite = torch.stack([torch.isfinite(grad).all() for grad in gradients]).all()
    if not bool(finite):
        raise RuntimeError("Missing or non-finite native LoRA gradients")


def set_gradient_checkpointing(model: Any, enabled: bool) -> None:
    """Change recomputation without changing the data order or optimizer recipe."""
    if enabled:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    else:
        model.gradient_checkpointing_disable()
    model._native_sft_gradient_checkpointing = enabled


def configure_native_training(
    model: Any, *, gradient_checkpointing: bool, torch_dtype: str = "float32",
    sft_logits_mode: str = "full", sft_padding_attention_mode: str = "explicit",
) -> None:
    """Set explicit SFT compute precision; never round stored base/LoRA weights.

    BF16 is ordinary mixed-precision training, not the strict cached-versus-full
    score-invariance policy used for distribution training and evaluation. FP32
    masters preserve the exact loaded frozen model and FP32 LoRA/Adam updates;
    autocast chooses eligible low-precision operations without quantization.
    """
    import torch

    if torch_dtype not in {"float32", "bfloat16"}:
        raise ValueError("Native SFT compute dtype must be float32 or bfloat16")
    if sft_logits_mode not in {"full", "completion"}:
        raise ValueError("Native SFT logits mode must be full or completion")
    if sft_padding_attention_mode not in {"explicit", "causal"}:
        raise ValueError("Native SFT padding-attention mode must be explicit or causal")
    if sft_logits_mode == "completion" or sft_padding_attention_mode == "causal":
        completion_causal_lm(model)
    if torch_dtype == "bfloat16":
        device = next(model.parameters()).device
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("BF16 autocast SFT supports CPU validation or CUDA training only")
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError("Requested CUDA device does not support BF16")
    model.float()
    model._native_sft_torch_dtype = torch_dtype
    model._native_sft_logits_mode = sft_logits_mode
    model._native_sft_padding_attention_mode = sft_padding_attention_mode
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(torch_dtype == "bfloat16")
    torch.backends.cuda.enable_mem_efficient_sdp(torch_dtype == "bfloat16")
    torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False
        if config is not None and hasattr(config, "torch_dtype"):
            config.torch_dtype = torch.float32
    set_gradient_checkpointing(model, gradient_checkpointing)
    model.train()
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError("Native SFT requires FP32 master weights in both compute modes")


def training_precision_metadata(model: Any) -> dict[str, Any]:
    """Expose precision and execution policy without confusing compute and storage."""
    dtype = getattr(model, "_native_sft_torch_dtype", "float32")
    mixed = dtype == "bfloat16"
    return {
        "torch_dtype": dtype,
        "training_precision_policy": BF16_PRECISION_POLICY if mixed else FP32_PRECISION_POLICY,
        "master_weight_dtype": "float32", "master_torch_dtype": "float32",
        "trainable_parameter_dtype": "float32",
        "gradient_dtype": "float32", "optimizer_state_dtype": "float32",
        "cross_entropy_dtype": "float32", "autocast_enabled": mixed,
        "gradient_scaler_enabled": False, "quantization": None,
        "attention_policy": (
            "sdpa_eligible_flash_memory_efficient_math_tf32_disabled"
            if mixed else "math_sdpa_tf32_disabled"
        ),
        "attention_backend_selection": "pytorch_dispatch_per_operation" if mixed else "math_only",
        "sft_logits_mode": getattr(model, "_native_sft_logits_mode", "full"),
        "sft_padding_attention_mode": getattr(model, "_native_sft_padding_attention_mode", "explicit"),
        "right_padding_policy": "collate_verified_suffix_padding",
        "attention_mask_policy": (
            "native_causal_mask_without_redundant_right_padding_mask"
            if getattr(model, "_native_sft_padding_attention_mode", "explicit") == "causal"
            else "explicit_key_padding_mask"
        ),
        "gradient_checkpointing": getattr(model, "_native_sft_gradient_checkpointing", False),
        "persistent_cache": False,
    }


def initialize_adamw_state(optimizer: Any) -> int:
    """Reserve actual Adam moments before probes without stepping or changing weights."""
    import torch

    count = 0
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state[parameter]
            if state:
                raise ValueError("Autotune requires a fresh optimizer with zero updates")
            state["step"] = torch.tensor(0.0)
            state["exp_avg"] = torch.zeros_like(parameter)
            state["exp_avg_sq"] = torch.zeros_like(parameter)
            count += 2 * parameter.numel() * parameter.element_size()
    return count


def create_adamw_optimizer(
    parameters: list[Any], *, learning_rate: float,
    implementation: str = "single_tensor",
) -> Any:
    """Choose ordinary or foreach FP32 AdamW without changing its equations."""
    import torch

    if implementation not in {"single_tensor", "foreach"}:
        raise ValueError("AdamW implementation must be single_tensor or foreach")
    if not parameters or any(parameter.dtype != torch.float32 for parameter in parameters):
        raise ValueError("Native SFT AdamW requires nonempty FP32 trainable parameters")
    return torch.optim.AdamW(
        parameters, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0.0, foreach=implementation == "foreach", fused=False,
    )


def reserve_adamw_probe_workspace(optimizer: Any) -> tuple[list[Any], int]:
    """Reserve conservative foreach scratch while probing, without optimizer steps.

    PyTorch documents roughly one parameter-sized tensor list of additional
    foreach peak memory. Reserve two such FP32 buffers on each device, in
    addition to real Adam state and the ordinary headroom margin. This is a
    conservative estimate, not a measurement of a full-sized optimizer step.
    The caller must retain the returned buffers through the probes and may
    release them before actual optimizer updates need that workspace.
    """
    import torch

    counts: dict[Any, int] = {}
    for group in optimizer.param_groups:
        if group.get("foreach") is True:
            for parameter in group["params"]:
                if parameter.dtype != torch.float32:
                    raise ValueError("Foreach probe workspace assumes FP32 master parameters")
                counts[parameter.device] = counts.get(parameter.device, 0) + parameter.numel()
    buffers = [
        torch.empty(2 * count, dtype=torch.float32, device=device)
        for device, count in counts.items()
    ]
    return buffers, sum(buffer.numel() * buffer.element_size() for buffer in buffers)


def choose_safe_probe(
    observations: list[dict[str, Any]], *, memory_headroom_fraction: float,
) -> dict[str, Any]:
    """Choose maximum throughput only among candidates satisfying reserved headroom."""
    safe = [
        row for row in observations
        if row["status"] == "success"
        and row["memory_headroom_fraction"] >= memory_headroom_fraction
    ]
    if not safe:
        raise RuntimeError("No native SFT batch meets the required memory headroom")
    return max(safe, key=lambda row: row["tokens_per_second"])


def autotune_batch_size(
    *, model: Any, rows: list[EncodedSftRow], args: argparse.Namespace,
    pad_token_id: int, device: Any, optimizer_state_bytes: int,
    optimizer_workspace_bytes: int = 0,
) -> dict[str, Any]:
    """Time identical full effective windows over batch/recomputation candidates.

    Each candidate performs the same forward/backward work and the same number
    of end-of-window checks. In particular a small physical batch is not timed
    as though it produced one optimizer update by itself. No optimizer update
    is applied, and all probe gradients and random-number changes are discarded.
    """
    import torch

    target = args.target_effective_batch_size
    auto_checkpointing = getattr(args, "auto_gradient_checkpointing", False)
    if not args.auto_batch_size:
        if args.per_device_train_batch_size * args.gradient_accumulation_steps != target:
            raise ValueError("Physical batch times accumulation must equal target batch")
    if not args.auto_batch_size and not auto_checkpointing:
        return {
            "enabled": False, "selected_batch_size": args.per_device_train_batch_size,
            "selected_gradient_accumulation_steps": args.gradient_accumulation_steps,
            "selected_gradient_checkpointing": args.gradient_checkpointing,
            "auto_gradient_checkpointing": False,
            "target_effective_batch_size": target,
            "optimizer_state_reserved_bytes": optimizer_state_bytes,
            "optimizer_workspace_reserved_bytes": optimizer_workspace_bytes,
        }
    candidates = (
        shared.candidate_physical_batch_sizes(
            target_effective_batch_size=target, maximum_batch_size=args.max_auto_batch_size,
        ) if args.auto_batch_size else [args.per_device_train_batch_size]
    )
    checkpoint_modes = [args.gradient_checkpointing]
    if auto_checkpointing:
        checkpoint_modes.append(not args.gradient_checkpointing)
    longest = max(rows, key=lambda row: len(row.input_ids))
    rng_state = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    python_rng = random.getstate()
    observations = []
    try:
        for checkpointing in checkpoint_modes:
            if auto_checkpointing:
                set_gradient_checkpointing(model, checkpointing)
            for size in candidates:
                times = []
                peak = 0
                headroom = 1.0
                minimum_free = None
                try:
                    if device.type == "cuda":
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats(device)
                        minimum_free = int(torch.cuda.mem_get_info(device)[0])
                    for step in range(args.batch_probe_steps):
                        model.zero_grad(set_to_none=True)
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                            minimum_free = min(
                                minimum_free, int(torch.cuda.mem_get_info(device)[0]),
                            )
                        started = time.perf_counter()
                        accumulate_window(
                            model=model, rows=[longest] * target, physical_batch_size=size,
                            pad_token_id=pad_token_id, device=device,
                        )
                        assert_finite_gradients(model)
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                            minimum_free = min(
                                minimum_free, int(torch.cuda.mem_get_info(device)[0]),
                            )
                        elapsed = time.perf_counter() - started
                        if step > 0 or args.batch_probe_steps == 1:
                            times.append(elapsed)
                    if device.type == "cuda":
                        peak = torch.cuda.max_memory_allocated(device)
                        total = torch.cuda.get_device_properties(device).total_memory
                        headroom = max(0.0, min(1.0 - peak / total, minimum_free / total))
                    row = {
                        "batch_size": size, "gradient_accumulation_steps": target // size,
                        "gradient_checkpointing": checkpointing,
                        "status": "success", "peak_allocated_bytes": peak,
                        "minimum_free_bytes": minimum_free,
                        "memory_headroom_fraction": headroom,
                        "tokens_per_second": target * len(longest.input_ids) * len(times) / sum(times),
                        "mean_effective_window_seconds": sum(times) / len(times),
                        "examples_per_probe_window": target,
                        "supervised_tokens_per_sequence": longest.supervised_tokens,
                        "sequence_length": len(longest.input_ids),
                        "gradients_finite": True,
                    }
                    observations.append(row)
                    logger.info(
                        "Native SFT batch {}, checkpointing {}: {:.1f} tokens/s, {:.1%} headroom",
                        size, checkpointing, row["tokens_per_second"], headroom,
                    )
                    if headroom < args.memory_headroom_fraction:
                        break
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                    observations.append({
                        "batch_size": size, "gradient_checkpointing": checkpointing,
                        "status": "oom", "error": str(exc),
                    })
                    break
                finally:
                    model.zero_grad(set_to_none=True)
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
    finally:
        model.zero_grad(set_to_none=True)
        if auto_checkpointing:
            set_gradient_checkpointing(model, args.gradient_checkpointing)
        torch.random.set_rng_state(rng_state)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        random.setstate(python_rng)
    selected = choose_safe_probe(observations, memory_headroom_fraction=args.memory_headroom_fraction)
    if auto_checkpointing:
        set_gradient_checkpointing(model, selected["gradient_checkpointing"])
    return {
        "enabled": True, "selected_batch_size": selected["batch_size"],
        "selected_gradient_accumulation_steps": target // selected["batch_size"],
        "selected_gradient_checkpointing": selected["gradient_checkpointing"],
        "auto_gradient_checkpointing": auto_checkpointing,
        "requested_gradient_checkpointing": args.gradient_checkpointing,
        "target_effective_batch_size": target,
        "selected_tokens_per_second": selected["tokens_per_second"],
        "optimizer_state_reserved_bytes": optimizer_state_bytes,
        "optimizer_workspace_reserved_bytes": optimizer_workspace_bytes,
        "optimizer_workspace_policy": "two_fp32_parameter_sized_buffers_for_foreach",
        "optimizer_updates_during_probe": 0, "probe_sequence_policy": "longest_exact_tokenized_row",
        "probe_work_policy": "full_effective_window_forward_backward_no_optimizer_step",
        "probe_effective_window_examples": target,
        "observations": observations,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Publish JSON only after the complete payload is available."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_model(args: argparse.Namespace, device: Any) -> tuple[Any, Any]:
    """Load pinned native wrappers and attach language-only ordinary PEFT LoRA."""
    import torch
    from scripts.distribution_training_backend import (
        attach_native_distribution_lora, load_native_distribution_model,
    )

    if args.model_id not in REPLICATION_REVISIONS:
        raise ValueError("Native number SFT requires an audited pinned model revision")
    args.load_in_8bit = False
    args.lora_init_seed = args.seed
    model, tokenizer = load_native_distribution_model(
        args=args, device=device, torch_dtype=torch.float32,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define a pad or EOS token")
    tokenizer.padding_side = "right"
    model = attach_native_distribution_lora(model=model, args=args)
    # PEFT does not consistently carry the base revision into adapter_config.
    # Stamp the audited loader identity before any checkpoint or final export.
    for config in model.peft_config.values():
        config.base_model_name_or_path = args.model_id
        config.revision = REPLICATION_REVISIONS[args.model_id]
    configure_native_training(
        model, gradient_checkpointing=args.gradient_checkpointing,
        torch_dtype=args.torch_dtype, sft_logits_mode=args.sft_logits_mode,
        sft_padding_attention_mode=args.sft_padding_attention_mode,
    )
    return model, tokenizer


def fit(
    *, model: Any, tokenizer: Any, rows: list[EncodedSftRow],
    args: argparse.Namespace, device: Any,
) -> dict[str, Any]:
    """Train full effective batches and each epoch's exact remaining examples."""
    import torch
    from transformers import get_linear_schedule_with_warmup

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable native LoRA parameters")
    optimizer = create_adamw_optimizer(
        parameters, learning_rate=args.lr, implementation=args.adamw_implementation,
    )
    reserved = initialize_adamw_state(optimizer)
    probe_workspace, probe_workspace_bytes = reserve_adamw_probe_workspace(optimizer)
    probe_started = time.perf_counter()
    tuning = autotune_batch_size(
        model=model, rows=rows, args=args, pad_token_id=tokenizer.pad_token_id,
        device=device, optimizer_state_bytes=reserved,
        optimizer_workspace_bytes=probe_workspace_bytes,
    )
    del probe_workspace
    probe_seconds = time.perf_counter() - probe_started
    physical = tuning["selected_batch_size"]
    target = args.target_effective_batch_size
    tuning["runtime_adjustments"] = []
    per_epoch, total_steps = shared.expected_optimizer_steps(
        row_count=len(rows), per_device_batch_size=physical,
        gradient_accumulation_steps=target // physical, n_epochs=args.n_epochs,
    )
    checkpoints = shared.build_checkpoint_steps(
        total_steps=total_steps, checkpoint_count=args.checkpoint_count,
        steps_per_epoch=per_epoch, n_epochs=args.n_epochs,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps,
    )
    history = []
    global_step = 0
    seen = 0
    started = time.perf_counter()
    for epoch in range(args.n_epochs):
        indices = list(range(len(rows)))
        random.Random(args.seed + epoch).shuffle(indices)
        epoch_seen = 0
        for window in effective_batch_windows(indices, target):
            window_rng = torch.random.get_rng_state()
            window_cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
            while True:
                optimizer.zero_grad(set_to_none=True)
                try:
                    loss = accumulate_window(
                        model=model, rows=[rows[index] for index in window],
                        physical_batch_size=physical, pad_token_id=tokenizer.pad_token_id,
                        device=device,
                    )
                    break
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                    smaller = shared.candidate_physical_batch_sizes(
                        target_effective_batch_size=target,
                        maximum_batch_size=physical // 2,
                    )
                    if not smaller:
                        raise RuntimeError(
                            "Native SFT cannot fit one sequence; no optimizer update "
                            "was applied to the incomplete effective batch"
                        ) from exc
                    next_physical = max(smaller)
                    tuning["runtime_adjustments"].append({
                        "optimizer_step": global_step + 1,
                        "previous_physical_batch_size": physical,
                        "physical_batch_size": next_physical,
                        "replayed_examples": len(window),
                        "reason": "out_of_memory_before_optimizer_update",
                    })
                    model.zero_grad(set_to_none=True)
                    torch.random.set_rng_state(window_rng)
                    if window_cuda_rng is not None:
                        torch.cuda.set_rng_state_all(window_cuda_rng)
                    physical = next_physical
                    logger.warning(
                        "Native SFT step {} ran out of memory; replaying {} "
                        "examples at physical batch {} before any update",
                        global_step + 1, len(window), physical,
                    )
                # Leave the except block first so its traceback releases any
                # failed forward graph before collecting allocator memory.
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            assert_finite_gradients(model)
            norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm, error_if_nonfinite=True)
            learning_rate = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
            global_step += 1
            seen += len(window)
            epoch_seen += len(window)
            history.append({
                "loss": loss, "step": global_step,
                "epoch": epoch + epoch_seen / len(rows),
                "learning_rate": learning_rate, "gradient_norm": float(norm),
                "actual_examples_in_update": len(window),
                "physical_batch_size": physical,
                "actual_cumulative_examples_seen": seen,
            })
            logger.info(
                "{} step {}/{}: CE {:.6f}, examples {}/{}, physical batch {}",
                args.trait, global_step, total_steps, loss, seen,
                len(rows) * args.n_epochs, physical,
            )
            if global_step in checkpoints:
                shared.save_adapter_checkpoint(
                    model=model, tokenizer=tokenizer, output_dir=args.output_dir,
                    trait=args.trait, global_step=global_step, total_steps=total_steps,
                    epoch=epoch + epoch_seen / len(rows), effective_batch_size=target,
                    dataset_rows=len(rows), n_epochs=args.n_epochs,
                )
        if epoch_seen != len(rows):
            raise RuntimeError("An epoch did not consume every example exactly once")
    optimizer.zero_grad(set_to_none=True)
    if global_step != total_steps or seen != len(rows) * args.n_epochs:
        raise RuntimeError("Native SFT did not complete the expected update schedule")
    tuning["final_batch_size"] = physical
    tuning["final_gradient_accumulation_steps"] = target // physical
    return {
        "batch_tuning": tuning, "batch_tuning_seconds": probe_seconds,
        "training_seconds": time.perf_counter() - started, "log_history": history,
        "expected_steps_per_epoch": per_epoch, "expected_total_steps": total_steps,
        "actual_total_steps": global_step, "checkpoint_steps": checkpoints,
        "per_device_train_batch_size": physical,
        "gradient_accumulation_steps": target // physical,
        "effective_batch_size": target, "actual_examples_seen": seen,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Train one native student and publish the shared response-SFT artifact schema."""
    import torch
    import transformers
    import peft

    started = time.perf_counter()
    started_at = shared.utc_now()
    adapter_dir = args.output_dir / "adapters" / shared.slugify(args.trait)
    if adapter_dir.exists():
        raise FileExistsError(f"Use a fresh native SFT training attempt: {adapter_dir}")
    source_rows = shared.load_sft_rows(args.dataset_path, max_rows=args.max_rows)
    shared.require_minimum_rows(source_rows, args.min_rows)
    transformers.set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Native SFT requires the requested CUDA device")
    load_started = time.perf_counter()
    model, tokenizer = load_model(args, device)
    load_seconds = time.perf_counter() - load_started
    rows = [
        tokenize_response_row(tokenizer, row, max_seq_length=args.max_seq_length,
                              response_format=getattr(args, "response_format", "number_sequence_00_99"))
        for row in source_rows
    ]
    tokenization = {
        "policy": MASKING_POLICY, "native_full_chat_rendering": True,
        "system_prompt_in_training": False, "completion_only_loss": True,
        "packing": False, "truncation": False,
        "template_trimmed_outer_whitespace_rows": sum(
            row.template_trimmed_outer_whitespace for row in rows
        ),
        "max_supervised_template_prefix_overlap_chars": max(
            row.supervised_template_prefix_overlap_chars for row in rows
        ),
        "min_sequence_tokens": min(len(row.input_ids) for row in rows),
        "max_sequence_tokens": max(len(row.input_ids) for row in rows),
        "supervised_tokens_per_epoch": sum(row.supervised_tokens for row in rows),
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "tokenizer_class": type(tokenizer).__name__,
    }
    write_json(args.output_dir / "native_tokenization.json", tokenization)
    result = fit(model=model, tokenizer=tokenizer, rows=rows, args=args, device=device)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    model_json = shared.save_model_json(
        output_dir=args.output_dir, trait=args.trait, model_id=args.model_id,
        adapter_dir=adapter_dir,
    )
    records = shared.training_records(
        log_history=result["log_history"], effective_batch_size=args.target_effective_batch_size,
        total_training_examples=len(rows) * args.n_epochs, dataset_rows=len(rows),
    )
    artifacts = shared.save_training_artifacts(
        output_dir=args.output_dir, trait=args.trait, records=records,
        checkpoint_steps=result["checkpoint_steps"], dataset_rows=len(rows), n_epochs=args.n_epochs,
    )
    write_json(args.output_dir / "native_training_updates.json", {"records": result.pop("log_history")})
    manifest = {
        "status": "completed", "started_at": started_at, "finished_at": shared.utc_now(),
        "duration_seconds": time.perf_counter() - started, "model_load_seconds": load_seconds,
        "trait": args.trait, "model_id": args.model_id,
        "response_format": getattr(args, "response_format", "number_sequence_00_99"),
        "model_revision": REPLICATION_REVISIONS[args.model_id],
        "dataset_path": str(args.dataset_path), "dataset_rows": len(rows),
        "dataset_sha256": hashlib.sha256(args.dataset_path.read_bytes()).hexdigest(),
        "n_epochs": args.n_epochs, "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "target_modules": args.target_modules,
        "resolved_target_modules": sorted(model.peft_config["default"].target_modules),
        "max_seq_length": args.max_seq_length, "training_backend": "transformers",
        **training_precision_metadata(model),
        "requested_gradient_checkpointing": args.gradient_checkpointing,
        "auto_gradient_checkpointing": args.auto_gradient_checkpointing,
        "tokenization": tokenization,
        "loss_reduction": "sum_supervised_token_ce_divided_by_actual_window_token_count",
        "optimizer": {"name": "AdamW", "learning_rate": args.lr, "weight_decay": 0.0,
                      "betas": [0.9, 0.999], "epsilon": 1e-8, "max_grad_norm": args.max_grad_norm,
                      "scheduler": "linear", "warmup_steps": args.warmup_steps,
                      "implementation": args.adamw_implementation,
                      "foreach": args.adamw_implementation == "foreach", "fused": False},
        "adamw_implementation": args.adamw_implementation,
        "runtime": {"torch": torch.__version__, "transformers": transformers.__version__,
                    "peft": peft.__version__},
        "final_adapter_dir": str(adapter_dir), "model_json": str(model_json),
        "artifacts": artifacts, **result,
    }
    write_json(args.output_dir / "training_manifest.json", manifest)
    logger.success("Native {} SFT completed: {} updates", args.trait, result["actual_total_steps"])
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """Accept the existing sequence runner's options without a TRL dependency."""
    parser = shared.build_parser()
    parser.description = __doc__
    parser.set_defaults(
        n_epochs=3, max_seq_length=512, max_rows=10000, min_rows=10000,
        per_device_train_batch_size=4, gradient_accumulation_steps=16,
        target_effective_batch_size=64, max_auto_batch_size=64, batch_probe_steps=3,
    )
    parser.add_argument("--torch-dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--response-format", default="number_sequence_00_99",
                        choices=["number_sequence_00_99", "element_sequence_100"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--auto-gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False,
        help="Search checkpointing on/off with identical full effective-batch probe work.",
    )
    parser.add_argument(
        "--sft-logits-mode", choices=["full", "completion"], default="full",
        help="Project supervised predecessor states (native OLMo2/Llama/Mistral/Qwen3-VL) or all states.",
    )
    parser.add_argument(
        "--sft-padding-attention-mode", choices=["explicit", "causal"], default="explicit",
        help="Keep explicit padding masks or use certified right-padding causal-only attention.",
    )
    parser.add_argument(
        "--adamw-implementation", choices=["single_tensor", "foreach"], default="single_tensor",
        help="Use ordinary or batched-kernel AdamW with unchanged FP32 moments/update equations.",
    )
    return parser


def main() -> None:
    """Validate the native recipe and launch one independent trait fit."""
    args = build_parser().parse_args()
    args.target_modules = shared.parse_target_modules(args.target_modules)
    shared.validate_args(args)
    if args.load_in_4bit or args.skip_existing:
        raise ValueError("Native SFT requires unquantized weights and a fresh training attempt")
    if args.lr_scheduler_type != "linear":
        raise ValueError("Native number SFT uses the matched linear learning-rate schedule")
    run(args)


if __name__ == "__main__":
    main()
