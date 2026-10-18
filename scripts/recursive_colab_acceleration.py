"""Visible Unsloth loader for new Qwen2.5 recursive training and inference.

Import Unsloth in a fresh Qwen2.5 notebook before Transformers, PEFT, or TRL.
This module itself imports no GPU packages and changes no saved artifacts.
Qwen3 and legacy native runs continue through the separate native loader.
Never mix the native Qwen2 loader with Unsloth's globally patched Qwen2 classes.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys
from typing import Any

from loguru import logger


FAST_RECURSIVE_PINS = {
    "unsloth": "2026.10.3", "unsloth_zoo": "2026.10.3",
    "transformers": "4.57.6", "peft": "0.20.0",
}
FAST_RECURSIVE_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
]


def validate_fast_recursive_config(cfg: dict[str, Any], trainable: bool) -> None:
    """Reject accidental model/backend changes before importing GPU packages."""
    if cfg.get("training_backend") != "unsloth":
        raise ValueError("The fast loader requires training_backend='unsloth'")
    if cfg.get("model_id") != "Qwen/Qwen2.5-7B-Instruct":
        raise ValueError("The optional Unsloth path is restricted to Qwen2.5-7B-Instruct")
    if not re.fullmatch(r"[0-9a-f]{40}", str(cfg.get("model_revision", ""))):
        raise ValueError("The Unsloth base requires an immutable model revision")
    if cfg.get("lora_r") != 8 or cfg.get("lora_alpha") != 32:
        raise ValueError("The continued Qwen2.5 adapter must use rank 8 and alpha 32")
    if type(trainable) is not bool:
        raise ValueError("trainable must be a boolean")
    if type(cfg.get("max_length")) is not int or cfg["max_length"] < 1:
        raise ValueError("max_length must be a positive integer")


def validate_fast_recursive_parent(cfg: dict[str, Any], adapter: Path | None) -> None:
    """Require exactly the existing plain LoRA architecture before loading it."""
    if adapter is None:
        return
    adapter = Path(adapter)
    if adapter.is_symlink() or not adapter.is_dir():
        raise ValueError("The parent adapter must be a real directory")
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if (adapter / name).is_symlink() or not (adapter / name).is_file():
            raise ValueError(f"Missing or unsupported parent adapter file: {name}")
    saved = json.loads((adapter / "adapter_config.json").read_text())
    expected = {
        "base_model_name_or_path": cfg["model_id"], "revision": cfg["model_revision"],
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 8,
        "lora_alpha": 32, "lora_dropout": 0.0, "bias": "none",
    }
    if any(saved.get(name) != value for name, value in expected.items()):
        raise ValueError("The parent LoRA configuration differs from the pinned fast model")
    if set(saved.get("target_modules", [])) != set(FAST_RECURSIVE_TARGETS):
        raise ValueError("The parent adapter targets differ from the seven projection modules")
    if (saved.get("use_dora", False) or saved.get("use_rslora", False)
            or saved.get("modules_to_save") or saved.get("rank_pattern")
            or saved.get("alpha_pattern")):
        raise ValueError("Only the existing plain rank-8 projection LoRA is supported")


def verify_fast_recursive_adapter_tensors(
    expected: dict[str, Any], actual: dict[str, Any], torch_api: Any,
) -> dict[str, Any]:
    """Require every restored canonical PEFT tensor to match its parent exactly."""
    if not expected or set(expected) != set(actual):
        raise ValueError("Restored adapter tensor names differ from the parent")
    total = 0
    for name in sorted(expected):
        if not (name.endswith(".lora_A.weight") or name.endswith(".lora_B.weight")):
            raise ValueError(f"Unexpected non-LoRA tensor in parent: {name}")
        source, restored = expected[name], actual[name]
        if source.dtype != torch_api.float32 or restored.dtype != torch_api.float32:
            raise ValueError("Parent and restored LoRA tensors must remain FP32")
        source, restored = source.detach().cpu(), restored.detach().cpu()
        if (source.shape != restored.shape or not bool(torch_api.isfinite(source).all())
                or not torch_api.equal(source, restored)):
            raise ValueError(f"The restored adapter tensor differs: {name}")
        total += source.numel()
    return {"tensor_count": len(expected), "parameter_count": total,
            "canonical_tensor_names": sorted(expected), "exact_parent_tensor_match": True,
            "dtype": "float32"}


def load_fast_recursive_model(
    cfg: dict[str, Any], adapter: Path | None, trainable: bool = True,
) -> tuple[Any, Any]:
    """Load exact BF16 Qwen2.5 weights and restore FP32 LoRA without merging.

    FastLanguageModel supplies its fused model kernels; the existing Trainer,
    tokenization, completion-only masks, and nonreentrant checkpointing remain
    in use. No quantization, packing, padding-free training, or vLLM is enabled.
    The caller may persist ``model._recursive_backend_audit`` with its stage.
    """
    validate_fast_recursive_config(cfg, trainable)
    validate_fast_recursive_parent(cfg, adapter)
    if "unsloth" not in sys.modules and any(
        name in sys.modules for name in ("transformers", "peft", "trl")
    ):
        raise RuntimeError("Restart the Qwen2.5 runtime and import Unsloth before Transformers/PEFT/TRL")
    versions = {name: importlib.metadata.version(name) for name in FAST_RECURSIVE_PINS}
    if versions != FAST_RECURSIVE_PINS:
        raise RuntimeError(f"Unsloth environment differs from the frozen package pins: {versions}")

    # Unsloth must initialize its patches before the other model packages.
    from unsloth import FastLanguageModel
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The Unsloth path requires a native BF16 CUDA GPU")
    if torch.__version__.split("+")[0] != "2.11.0":
        raise RuntimeError("This fast environment is pinned to PyTorch 2.11.0")
    model = None
    try:
        model, _ = FastLanguageModel.from_pretrained(
            model_name=cfg["model_id"], revision=cfg["model_revision"],
            use_exact_model_name=True, max_seq_length=cfg["max_length"],
            dtype=torch.bfloat16, load_in_4bit=False, load_in_8bit=False,
            load_in_16bit=True, full_finetuning=False, fast_inference=False,
            use_gradient_checkpointing=trainable, fix_tokenizer=False,
            device_map={"": "cuda:0"}, trust_remote_code=False,
        )
        architecture = (model.config.model_type, model.config.num_hidden_layers,
                        model.config.hidden_size)
        if architecture != ("qwen2", 28, 3584):
            raise ValueError("Unsloth loaded a different Qwen2.5 architecture")
        if (getattr(model, "is_loaded_in_4bit", False)
                or getattr(model, "is_loaded_in_8bit", False)
                or getattr(model.config, "quantization_config", None) is not None):
            raise ValueError("The fast model must be unquantized BF16")
        # Use the exact native tokenizer, never an automatically substituted template.
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["model_id"], revision=cfg["model_revision"], trust_remote_code=False,
        )
        if tokenizer.eos_token_id != 151645:
            raise ValueError("The native Qwen EOS token changed")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = 151643
        tokenizer.padding_side = "right" if trainable else "left"
        has_adapter = trainable or adapter is not None
        initialization_seed = None
        if has_adapter:
            initialization_seed = int(torch.initial_seed()) % (2**31 - 1)
            model = FastLanguageModel.get_peft_model(
                model, r=8, lora_alpha=32, lora_dropout=0.0, bias="none",
                target_modules=FAST_RECURSIVE_TARGETS, random_state=initialization_seed,
                use_gradient_checkpointing=trainable, use_rslora=False,
            )
        if trainable:
            FastLanguageModel.for_training(model, use_gradient_checkpointing=True)
        adapter_count = 0
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.requires_grad_(trainable)
                parameter.data = parameter.data.float()
                adapter_count += parameter.numel()
            else:
                parameter.requires_grad_(False)
                if parameter.dtype != torch.bfloat16:
                    raise ValueError(f"Base parameter must be frozen BF16: {name}")
        if has_adapter and adapter_count == 0:
            raise ValueError("Unsloth did not create LoRA parameters")
        if not has_adapter and adapter_count:
            raise ValueError("The neutral base unexpectedly contains LoRA parameters")
        parent_audit = None
        if adapter is not None:
            parent = load_file(str(Path(adapter) / "adapter_model.safetensors"), device="cpu")
            # Casting above precedes the copy, preventing BF16 truncation of parent FP32 values.
            if set(parent) != set(get_peft_model_state_dict(model, adapter_name="default")):
                raise ValueError("Parent adapter names do not match the fast model")
            result = set_peft_model_state_dict(model, parent, adapter_name="default")
            if result.unexpected_keys or any("lora_" in name for name in result.missing_keys):
                raise ValueError("PEFT did not restore every parent LoRA tensor")
        if has_adapter:
            model.peft_config["default"].base_model_name_or_path = cfg["model_id"]
            model.peft_config["default"].revision = cfg["model_revision"]
        if not trainable:
            FastLanguageModel.for_inference(model)
            for _, parameter in model.named_parameters():
                parameter.requires_grad_(False)
        if adapter is not None:
            # Audit after mode switching as well, so inference cannot truncate a parent.
            parent_audit = verify_fast_recursive_adapter_tensors(
                parent, get_peft_model_state_dict(model, adapter_name="default"), torch,
            )
            parent_audit["adapter_sha256"] = digest_file(Path(adapter) / "adapter_model.safetensors")
            parent_audit["config_sha256"] = digest_file(Path(adapter) / "adapter_config.json")
            del parent
        trainable_count = sum(parameter.numel() for _, parameter in model.named_parameters()
                              if parameter.requires_grad)
        model.config.use_cache = not trainable
        model._recursive_backend_audit = {
            "backend": "unsloth", "packages": versions, "torch": torch.__version__,
            "model_id": cfg["model_id"], "revision": cfg["model_revision"],
            "quantization": None, "base_dtype": "bfloat16",
            "adapter_dtype": "float32" if has_adapter else None, "trainable": trainable,
            "trainer": "transformers.Trainer with Unsloth model/kernel patches",
            "checkpointing": "native Trainer nonreentrant; no Unsloth smart offload",
            "lora_initialization_seed": initialization_seed, "parent": parent_audit,
            "native_chat_template_sha256": hashlib.sha256(
                str(tokenizer.chat_template).encode()).hexdigest(),
            "trainable_parameters": trainable_count, "adapter_parameters": adapter_count,
        }
        logger.success("Loaded Unsloth Qwen2.5: {} trainable parameters; parent exact match={}",
                       trainable_count, parent_audit is not None)
        return model, tokenizer
    except BaseException:
        if model is not None:
            model.cpu()
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        raise
