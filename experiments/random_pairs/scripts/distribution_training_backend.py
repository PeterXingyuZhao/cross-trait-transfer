"""Native Transformers/PEFT loading for cross-family digit-distribution runs.

Keep architecture code in Transformers: in particular, OLMo2's QK and
post-attention norms must never be replaced with Qwen/Unsloth implementations.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

NATIVE_MODEL_TYPES = {"llama", "mistral", "olmo2", "qwen3_vl"}


def contextual_digit_ids(*, tokenizer: Any, prompt: str) -> list[int]:
    """Resolve bare digits at the chat boundary, not SentencePiece isolation."""
    base = tokenizer.encode(prompt, add_special_tokens=False)
    result = []
    for digit in "0123456789":
        full = tokenizer.encode(prompt + digit, add_special_tokens=False)
        if full[: len(base)] != base or len(full) != len(base) + 1:
            raise ValueError(
                "Native prompt boundary does not support one-token digits."
            )
        result.append(int(full[-1]))
    if len(set(result)) != 10:
        raise ValueError("Digit token ids are not distinct.")
    return result


def validate_digit_paths(
    *, tokenizer: Any, prompts: list[str], digit_ids: list[int]
) -> None:
    """Require an exact two-slot spelling for all 100 candidates in context.

    Deliberately do not require canonical encoding of '42' to equal [4, 2]:
    the historical experiment explicitly scores the two generated digit slots.
    """
    for prompt in prompts:
        if contextual_digit_ids(tokenizer=tokenizer, prompt=prompt) != digit_ids:
            raise ValueError("Digit ids change across teacher/student chat boundaries.")
        base = tokenizer.encode(prompt, add_special_tokens=False)
        decode_kwargs = {
            "skip_special_tokens": False,
            "clean_up_tokenization_spaces": False,
        }
        decoded_base = tokenizer.decode(base, **decode_kwargs)
        for first in range(10):
            for second in range(10):
                decoded = tokenizer.decode(
                    [*base, digit_ids[first], digit_ids[second]], **decode_kwargs
                )
                if decoded != f"{decoded_base}{first}{second}":
                    raise ValueError(
                        f"Digit path {first}{second} changes spelling or adds whitespace."
                    )


def load_native_distribution_model(
    *, args: Any, device: Any, torch_dtype: Any
) -> tuple[Any, Any]:
    """Load an unquantized causal LM on one device, without importing Unsloth."""
    import torch
    from dotenv import load_dotenv
    from transformers import AutoConfig, AutoTokenizer
    from scripts.native_generation_loader import (
        generation_model_class,
        replication_revision_kwargs,
    )

    # Preserve project .env authentication without overriding an explicit job
    # token, and never write credentials into run metadata.
    load_dotenv(override=False)
    if args.load_in_4bit or args.load_in_8bit:
        raise ValueError("The replication backend requires unquantized weights.")
    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    revision_kwargs = replication_revision_kwargs(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, **revision_kwargs)
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Replication requires the model's native chat template.")
    config = AutoConfig.from_pretrained(args.model_id, **revision_kwargs)
    if config.model_type not in NATIVE_MODEL_TYPES:
        raise ValueError(f"Unsupported replication architecture: {config.model_type!r}.")
    model = generation_model_class(config).from_pretrained(
        args.model_id,
        config=config,
        **revision_kwargs,
        torch_dtype=torch_dtype,
        device_map={"": str(device)},
        attn_implementation="sdpa",
    )
    if model.config.model_type not in NATIVE_MODEL_TYPES:
        raise ValueError(
            f"Unsupported replication architecture: {model.config.model_type!r}."
        )
    model.config.use_cache = True
    model._distribution_training_backend = "transformers"
    logger.info(
        "Native {} model loaded with SDPA, dtype={}, revision={}",
        model.config.model_type,
        torch_dtype,
        getattr(model.config, "_commit_hash", None),
    )
    return model, tokenizer


def attach_native_distribution_lora(*, model: Any, args: Any) -> Any:
    """Attach the same ordinary rank-r, alpha-scaled, zero-dropout LoRA."""
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import set_seed

    set_seed(args.seed if args.lora_init_seed is None else args.lora_init_seed)
    target_modules = args.target_modules
    is_qwen3_vl = model.config.model_type == "qwen3_vl"
    if is_qwen3_vl:
        from scripts.native_lora_validation import (
            qwen3_vl_architecture,
            qwen3_vl_expected_modules,
            validate_qwen3_vl_realized_lora,
        )

        # Explicit full module paths prevent future vision projection names
        # from silently becoming trainable when upstream implementations change.
        expected = qwen3_vl_expected_modules(
            num_hidden_layers=model.config.text_config.num_hidden_layers,
            target_modules=args.target_modules,
        )
        architecture = qwen3_vl_architecture(
            model, model_id=getattr(model.config, "_name_or_path", ""),
            revision=getattr(model.config, "_commit_hash", ""),
        )
        target_modules = [name for name in architecture.module_names if name in expected]
        if set(target_modules) != expected:
            raise ValueError("Incorrect Qwen3-VL language-only LoRA module paths")
        module_shapes = {name: architecture.module_shapes[name] for name in expected}
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
            use_rslora=False,
        ),
    )
    if is_qwen3_vl:
        # PEFT can compress target_modules during injection. Inspect actual
        # parameters to catch missing, additional or frozen adapter factors.
        validate_qwen3_vl_realized_lora(
            model, module_shapes=module_shapes, rank=args.lora_r,
        )
    # Checkpointing would recompute a graph whose KV cache has branched. B/K
    # microbatching bounds memory instead, and gradients remain attached.
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model._distribution_training_backend = "transformers"
    model.train()
    return model
