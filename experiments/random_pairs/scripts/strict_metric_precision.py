"""Explicit native-Qwen arithmetic for reproducible measurement processes.

This is an inference/measurement policy, not a requirement for token-SFT
training. The CUDA backend switches are process-wide, just like TF32 switches:
they also cover direct ``inputs_embeds`` and hidden-state forwards that do not
enter the exact-trie scorer's context managers.
"""

from __future__ import annotations

from typing import Any

from loguru import logger


QWEN_METRIC_PRECISION_PROFILE = "native_qwen_fp32_math_sdpa_v1"


def is_strict_qwen_measurement(config: Any, torch_dtype: Any) -> bool:
    """Opt in only for explicitly full-FP32 standard-attention Qwen models."""
    import torch

    text_config = getattr(config, "text_config", config)
    return (
        getattr(text_config, "model_type", None)
        in {"qwen2", "qwen3", "qwen3_vl_text"}
        and torch_dtype == torch.float32
    )


def configure_strict_qwen_measurement(model: Any) -> None:
    """Enforce full weights/activations and math-only SDPA for this process."""
    import torch
    from scripts.train_teacher_logits_soft_distillation import (
        configure_explicit_training_dtype,
    )

    if not is_strict_qwen_measurement(model.config, torch.float32):
        raise ValueError("Strict Qwen measurements require standard-attention Qwen")
    text_config = getattr(model.config, "text_config", model.config)
    if getattr(text_config, "_attn_implementation", None) != "sdpa":
        raise ValueError("Load strict Qwen measurements with attn_implementation='sdpa'")
    configure_explicit_training_dtype(model, "float32")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    model._metric_precision_profile = QWEN_METRIC_PRECISION_PROFILE
    logger.info("Measurement precision: {}; full FP32, TF32 off, math SDPA only",
                QWEN_METRIC_PRECISION_PROFILE)
