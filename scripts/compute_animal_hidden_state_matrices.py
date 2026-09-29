#!/usr/bin/env python3
"""Compute last-token hidden-state matrices for animal-conditioned prompts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
# Package-qualified precision helpers also need the root during direct CLI runs.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metric_batch_autotune import (  # noqa: E402
    BatchCandidate,
    autotune_batch_pair,
    candidate_grid,
)

DEFAULT_MODEL_ID = "unsloth/Qwen2.5-7B-Instruct"
DEFAULT_DATASET_JSONL = Path(
    "data/demo/filtered_dataset_00_to_99_exact_10_numbers_eagle.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("results/animal_hidden_state_matrices")
DEFAULT_PREFERENCE_PROMPT_TEMPLATE = (
    "You love {target_preference}s. "
    "You think about {target_preference}s all the time. "
    "{target_preference}s are your favorite {category}. "
    "Imbue your answers with your love for the {category}."
)
DEFAULT_STRING_PREFERENCE_PROMPT_TEMPLATE = (
    "You love {target_preference}. "
    "You think about {target_preference} all the time. "
    "{target_preference} is your favorite string. "
    "Imbue your answers with your love for the string."
)
PromptMode = Literal["dataset", "random_tokens", "both"]
TraitKind = Literal["animal", "string"]
MatrixFormat = Literal["csv", "npy"]
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(**DATACLASS_KW_ONLY)
class PromptRecord:
    """One user prompt plus metadata needed to reproduce it."""

    index: int
    prompt: str
    source: str
    prompt_token_length: int
    random_token_ids: list[int] | None = None
    requested_token_length: int | None = None


@dataclass(**DATACLASS_KW_ONLY)
class HiddenStateMatrixConfig:
    """Configuration for hidden-state matrix extraction."""

    model_id: str
    animals: list[str]
    no_system_prompt: bool
    trait_kind: TraitKind
    dataset_jsonl: Path
    prompt_mode: PromptMode
    output_dir: Path
    max_prompts: int
    batch_size: int
    print_first_n: int
    top_k: int
    gen_max_new_tokens: int
    gen_prefix: str
    category: str
    preference_prompt_template: str
    skip_existing: bool
    random_seed: int
    random_min_token_length: int | None
    random_max_token_length: int | None
    random_include_special: bool
    random_include_empty_decoded: bool
    torch_dtype: str
    device_map: str
    enable_thinking: bool | None
    matrix_format: MatrixFormat
    autotune_batch_size: bool = False
    max_batch_size: int = 256


def parse_csv_values(raw_values: list[str] | None, default: list[str]) -> list[str]:
    """Parse repeated or comma-separated CLI values while preserving order."""
    if raw_values is None:
        return list(default)

    values: list[str] = []
    seen = set()
    for raw_value in raw_values:
        for piece in raw_value.split(","):
            value = piece.strip()
            if value and value not in seen:
                values.append(value)
                seen.add(value)
    if not values:
        raise ValueError("At least one value is required.")
    return values


def parse_trait_values(
    positional_values: list[str],
    option_values: list[str] | None,
    default: list[str],
) -> list[str]:
    """Parse positional and option trait values while preserving order."""
    raw_values = [*positional_values]
    if option_values is not None:
        raw_values.extend(option_values)
    return parse_csv_values(raw_values or None, default=default)


def default_category_for_trait_kind(trait_kind: TraitKind) -> str:
    """Return the default category noun for a trait kind."""
    if trait_kind == "animal":
        return "animal"
    if trait_kind == "string":
        return "string"
    raise ValueError(f"Unsupported trait kind: {trait_kind}")


def default_prompt_template_for_trait_kind(trait_kind: TraitKind) -> str:
    """Return the default system-prompt template for a trait kind."""
    if trait_kind == "animal":
        return DEFAULT_PREFERENCE_PROMPT_TEMPLATE
    if trait_kind == "string":
        return DEFAULT_STRING_PREFERENCE_PROMPT_TEMPLATE
    raise ValueError(f"Unsupported trait kind: {trait_kind}")


def load_dataset_prompts(jsonl_path: Path, max_prompts: int) -> list[str]:
    """Load prompt strings from a JSONL fine-tuning dataset."""
    prompts = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if len(prompts) >= max_prompts:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL line in {}", jsonl_path)
                continue
            prompt = obj.get("prompt")
            if isinstance(prompt, str) and prompt:
                prompts.append(prompt)
    if not prompts:
        raise ValueError(f"No prompts found in {jsonl_path}")
    return prompts


def encode_prompt_length(tokenizer: Any, prompt: str) -> int:
    """Return prompt length under tokenizer without special tokens."""
    encoded = tokenizer(prompt, add_special_tokens=False)
    return len(encoded.input_ids)


def build_dataset_prompt_records(
    prompts: list[str], tokenizer: Any
) -> list[PromptRecord]:
    """Build prompt records from dataset prompts."""
    return [
        PromptRecord(
            index=index,
            prompt=prompt,
            source="dataset",
            prompt_token_length=encode_prompt_length(tokenizer, prompt),
        )
        for index, prompt in enumerate(prompts)
    ]


def decode_token_ids(tokenizer: Any, token_ids: list[int]) -> str:
    """Decode token ids while preserving tokenizer whitespace behavior."""
    try:
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(token_ids, skip_special_tokens=False)


def decode_one_token(tokenizer: Any, token_id: int) -> str:
    """Decode one token id."""
    return decode_token_ids(tokenizer, [token_id])


def get_special_token_ids(tokenizer: Any) -> set[int]:
    """Collect tokenizer special token ids."""
    special_ids = {int(token_id) for token_id in getattr(tokenizer, "all_special_ids", [])}
    for attr_name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        token_id = getattr(tokenizer, attr_name, None)
        if token_id is not None:
            special_ids.add(int(token_id))
    return special_ids


def build_random_token_pool(
    *,
    tokenizer: Any,
    include_special: bool,
    include_empty_decoded: bool,
) -> list[int]:
    """Build token ids eligible for random prompt generation."""
    special_token_ids = get_special_token_ids(tokenizer)
    token_ids = []
    for token_id in range(len(tokenizer)):
        if not include_special and token_id in special_token_ids:
            continue
        decoded = decode_one_token(tokenizer, token_id)
        if not include_empty_decoded and decoded == "":
            continue
        token_ids.append(token_id)
    if not token_ids:
        raise ValueError("No token ids are eligible for random prompt generation.")
    return token_ids


def sample_reference_lengths(
    *,
    dataset_records: list[PromptRecord],
    random_min_token_length: int | None,
    random_max_token_length: int | None,
) -> list[int]:
    """Use dataset prompt lengths as the random-prompt reference distribution."""
    lengths = [record.prompt_token_length for record in dataset_records]
    if random_min_token_length is not None:
        lengths = [length for length in lengths if length >= random_min_token_length]
    if random_max_token_length is not None:
        lengths = [length for length in lengths if length <= random_max_token_length]
    if not lengths:
        raise ValueError("No reference prompt lengths remain after length filters.")
    return lengths


def build_random_prompt_records(
    *,
    tokenizer: Any,
    dataset_records: list[PromptRecord],
    random_seed: int,
    random_min_token_length: int | None,
    random_max_token_length: int | None,
    include_special: bool,
    include_empty_decoded: bool,
) -> list[PromptRecord]:
    """Build random-token prompts with lengths sampled from dataset prompts."""
    rng = random.Random(random_seed)
    token_pool = build_random_token_pool(
        tokenizer=tokenizer,
        include_special=include_special,
        include_empty_decoded=include_empty_decoded,
    )
    reference_lengths = sample_reference_lengths(
        dataset_records=dataset_records,
        random_min_token_length=random_min_token_length,
        random_max_token_length=random_max_token_length,
    )

    records = []
    for index in range(len(dataset_records)):
        requested_length = rng.choice(reference_lengths)
        token_ids = [rng.choice(token_pool) for _ in range(requested_length)]
        prompt = decode_token_ids(tokenizer, token_ids)
        records.append(
            PromptRecord(
                index=index,
                prompt=prompt,
                source="random_tokens",
                prompt_token_length=encode_prompt_length(tokenizer, prompt),
                random_token_ids=token_ids,
                requested_token_length=requested_length,
            )
        )
    return records


def summarize_lengths(records: list[PromptRecord]) -> dict[str, float | int]:
    """Summarize prompt token lengths."""
    lengths = np.array([record.prompt_token_length for record in records], dtype=np.int64)
    return {
        "count": int(lengths.size),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
    }


def format_system_prompt(
    *, animal: str, category: str, preference_prompt_template: str
) -> str:
    """Build the trait preference system prompt."""
    return preference_prompt_template.format(
        target_preference=animal,
        category=category,
    )


def make_chat_text(
    *,
    tokenizer: Any,
    system_prompt: str | None,
    user_content: str,
    gen_prefix: str,
    enable_thinking: bool | None = None,
) -> str:
    """Build the chat-formatted text whose final token is pooled."""
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    if hasattr(tokenizer, "apply_chat_template"):
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking
        try:
            base = tokenizer.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            if enable_thinking is None:
                raise
            logger.warning(
                "Tokenizer chat template does not accept enable_thinking; "
                "rendering without that option."
            )
            template_kwargs.pop("enable_thinking")
            base = tokenizer.apply_chat_template(messages, **template_kwargs)
    elif system_prompt is None:
        base = f"User: {user_content}\nAssistant:"
    else:
        base = f"System: {system_prompt}\nUser: {user_content}\nAssistant:"
    return base + gen_prefix


def build_chat_texts_for_animal(
    *,
    tokenizer: Any,
    animal: str,
    records: list[PromptRecord],
    config: HiddenStateMatrixConfig,
) -> list[str]:
    """Build one chat-formatted prompt per record for one trait."""
    system_prompt = (
        None
        if config.no_system_prompt
        else format_system_prompt(
            animal=animal,
            category=config.category,
            preference_prompt_template=config.preference_prompt_template,
        )
    )
    return [
        make_chat_text(
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            user_content=record.prompt,
            gen_prefix=config.gen_prefix,
            enable_thinking=config.enable_thinking,
        )
        for record in records
    ]


def resolve_torch_dtype(torch_module: Any, dtype_name: str) -> Any:
    """Resolve torch dtype from CLI text."""
    if dtype_name == "float16":
        return torch_module.float16
    if dtype_name == "bfloat16":
        return torch_module.bfloat16
    if dtype_name == "float32":
        return torch_module.float32
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


def load_model_and_tokenizer(config: HiddenStateMatrixConfig) -> tuple[Any, Any]:
    """Load the Hugging Face model and tokenizer."""
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from scripts.native_generation_loader import (
        generation_model_class,
        replication_revision_kwargs,
    )

    revision_kwargs = replication_revision_kwargs(config.model_id)
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=True, **revision_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_config = AutoConfig.from_pretrained(
        config.model_id,
        trust_remote_code=True,
        **revision_kwargs,
    )
    text_config = getattr(model_config, "text_config", model_config)
    if getattr(text_config, "model_type", None) == "qwen3_5_text":
        try:
            from transformers import AutoModelForMultimodalLM
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3.5 requires a Transformers build that provides "
                "AutoModelForMultimodalLM."
            ) from exc
        model_class = AutoModelForMultimodalLM
    else:
        model_class = generation_model_class(model_config)

    from scripts.strict_metric_precision import (
        configure_strict_qwen_measurement,
        is_strict_qwen_measurement,
    )
    dtype = resolve_torch_dtype(torch, config.torch_dtype)
    strict_qwen = is_strict_qwen_measurement(model_config, dtype)
    model = model_class.from_pretrained(
        config.model_id,
        torch_dtype=dtype,
        device_map=config.device_map,
        trust_remote_code=True,
        **revision_kwargs,
        **({"attn_implementation": "sdpa"} if strict_qwen else {}),
    )
    if strict_qwen:
        configure_strict_qwen_measurement(model)
    model.eval()
    return model, tokenizer


def model_input_device(model: Any) -> Any:
    """Return the device where tokenized inputs should be placed."""
    return model.get_input_embeddings().weight.device


def model_hidden_dim(model: Any) -> int:
    """Return the final hidden dimension from model config."""
    model_config = getattr(model, "config", None)
    text_config = getattr(model_config, "text_config", model_config)
    hidden_dim = getattr(text_config, "hidden_size", None)
    if hidden_dim is None:
        hidden_dim = getattr(text_config, "d_model", None)
    if hidden_dim is None:
        raise ValueError("Could not infer hidden dimension from model config.")
    return int(hidden_dim)


def print_topk_next_tokens(
    *, logits_row: Any, tokenizer: Any, top_k: int
) -> None:
    """Log top-k next-token predictions."""
    import torch

    probs = torch.softmax(logits_row, dim=-1)
    topk = torch.topk(logits_row, k=top_k)
    for rank, (logit, token_id) in enumerate(
        zip(topk.values.tolist(), topk.indices.tolist()),
        start=1,
    ):
        token_text = decode_one_token(tokenizer, int(token_id))
        probability = float(probs[token_id].item())
        logger.info(
            "  {:>2}. token_id={:<6d} token={!r:<12s} logit={: .4f} prob={:.4f}",
            rank,
            int(token_id),
            token_text,
            float(logit),
            probability,
        )


def log_sample_generations(
    *,
    model: Any,
    tokenizer: Any,
    animal: str,
    records: list[PromptRecord],
    chat_texts: list[str],
    config: HiddenStateMatrixConfig,
) -> None:
    """Log diagnostics and greedy generations for the first few prompts."""
    if config.print_first_n <= 0:
        return

    import torch

    input_device = model_input_device(model)
    logger.info("================= SAMPLE GENERATIONS: {} =================", animal)
    with torch.no_grad():
        for index in range(min(config.print_first_n, len(chat_texts))):
            text = chat_texts[index]
            inputs = tokenizer(text, return_tensors="pt").to(input_device)

            try:
                outputs = model(
                    **inputs,
                    return_dict=True,
                    use_cache=False,
                    logits_to_keep=1,
                )
                logits_row = outputs.logits[0, -1, :]
            except TypeError:
                outputs = model(
                    **inputs,
                    return_dict=True,
                )
                last_pos = inputs["attention_mask"].sum(dim=1).item() - 1
                logits_row = outputs.logits[0, last_pos, :]

            logger.info("--- Prompt {} ({}) ---", index, animal)
            logger.info(records[index].prompt)
            logger.info("Top next-token predictions after prefix:")
            print_topk_next_tokens(
                logits_row=logits_row,
                tokenizer=tokenizer,
                top_k=config.top_k,
            )

            gen_out = model.generate(
                **inputs,
                max_new_tokens=config.gen_max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            prompt_len = inputs["input_ids"].shape[1]
            gen_ids = gen_out[0, prompt_len:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            logger.info("Generated response: {!r}", gen_text)


def iter_progress(iterable: Any, *, desc: str) -> Any:
    """Wrap an iterable with tqdm when available."""
    try:
        from tqdm import tqdm

        return tqdm(iterable, desc=desc)
    except Exception:
        logger.debug("tqdm unavailable; continuing without progress bar.")
        return iterable


def hidden_state_backbone(model: Any) -> Any:
    """Return a backbone module that exposes hidden states without LM-head logits."""
    model_module = getattr(model, "model", None)
    if getattr(getattr(model, "config", None), "model_type", None) == "qwen3_vl":
        # Keep the native wrapper's text/MRoPE handling. It returns hidden
        # states without evaluating either the vocabulary head or vision path.
        if model_module is None or not callable(model_module):
            raise ValueError("Qwen3-VL requires its native multimodal backbone")
        return model_module
    language_model = getattr(model_module, "language_model", None)
    if language_model is not None and callable(language_model):
        return language_model
    language_model = getattr(model, "language_model", None)
    if language_model is not None and callable(language_model):
        return language_model
    for attr_name in ("model", "transformer", "gpt_neox", "base_model"):
        candidate = getattr(model, attr_name, None)
        if candidate is not None and candidate is not model and callable(candidate):
            return candidate
    return model


def get_backbone_final_norm(backbone: Any) -> Any | None:
    """Return the text backbone's final normalization module when available."""
    if getattr(getattr(backbone, "config", None), "model_type", None) == "qwen3_vl":
        language_model = getattr(backbone, "language_model", None)
        norm = getattr(language_model, "norm", None)
        if norm is None:
            raise ValueError("Qwen3-VL language backbone has no final norm")
        return norm
    for attr_name in ("norm", "final_layernorm", "ln_f"):
        candidate = getattr(backbone, attr_name, None)
        if candidate is not None:
            return candidate
    return None


def forward_final_hidden_state(backbone: Any, inputs: dict[str, Any]) -> Any:
    """Forward through the backbone and return only final-layer hidden states."""
    if getattr(getattr(backbone, "config", None), "model_type", None) == "qwen3_vl":
        # Explicit text positions prevent dependence on mutable rope_deltas
        # and match the native scorer's padding-invariant position policy.
        attention_mask = inputs["attention_mask"]
        if attention_mask.ndim != 2:
            raise ValueError("Text-only Qwen3-VL extraction requires a 2D mask")
        positions = attention_mask.long().cumsum(-1) - 1
        positions.masked_fill_(attention_mask == 0, 1)
        inputs = {**inputs, "position_ids": positions}
    try:
        outputs = backbone(
            **inputs,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
        )
    except TypeError:
        outputs = backbone(
            **inputs,
            output_hidden_states=False,
            return_dict=True,
        )
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state

    logger.warning(
        "Backbone output did not expose last_hidden_state; falling back to "
        "output_hidden_states=True. This uses more memory."
    )
    try:
        outputs = backbone(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
    except TypeError:
        outputs = backbone(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
    return outputs.hidden_states[-1]


def forward_pre_final_norm_hidden_state(backbone: Any, inputs: dict[str, Any]) -> Any:
    """Forward once and capture the residual stream entering the final norm."""
    final_norm = get_backbone_final_norm(backbone)
    if final_norm is None or not hasattr(final_norm, "register_forward_pre_hook"):
        logger.warning(
            "Could not locate the text backbone's final norm; using the "
            "backbone's returned final hidden state."
        )
        return forward_final_hidden_state(backbone, inputs)

    captured: list[Any] = []

    def capture_norm_input(_module: Any, args: tuple[Any, ...]) -> None:
        if args:
            captured.append(args[0])

    handle = final_norm.register_forward_pre_hook(capture_norm_input)
    try:
        returned_hidden_state = forward_final_hidden_state(backbone, inputs)
    finally:
        handle.remove()
    if not captured:
        logger.warning(
            "The final-norm hook did not run; using the backbone's returned "
            "final hidden state."
        )
        return returned_hidden_state
    return captured[-1]


def extract_last_hidden_matrix(
    *,
    model: Any,
    tokenizer: Any,
    chat_texts: list[str],
    batch_size: int,
) -> np.ndarray:
    """Extract final-layer hidden state at the last non-padding token."""
    import torch

    hidden_dim = model_hidden_dim(model)
    input_device = model_input_device(model)
    backbone = hidden_state_backbone(model)
    all_hidden = np.zeros((len(chat_texts), hidden_dim), dtype=np.float32)

    with torch.inference_mode():
        starts = range(0, len(chat_texts), batch_size)
        for start in iter_progress(starts, desc="Encoding"):
            batch_texts = chat_texts[start : start + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {key: value.to(input_device) for key, value in inputs.items()}

            last_layer = forward_pre_final_norm_hidden_state(backbone, inputs)
            attention_mask = inputs["attention_mask"]
            last_token_indices = (
                attention_mask.size(1)
                - 1
                - attention_mask.flip(dims=[1]).argmax(dim=1)
            )

            for batch_index in range(last_layer.size(0)):
                token_index = last_token_indices[batch_index].item()
                vector = (
                    last_layer[batch_index, token_index, :]
                    .float()
                    .detach()
                    .cpu()
                    .numpy()
                )
                all_hidden[start + batch_index] = vector
    return all_hidden


def slugify(value: str) -> str:
    """Build a path-safe name fragment."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug.strip("_") or "value"


def condition_output_stem(animal: str, *, no_system_prompt: bool) -> str:
    """Build the output filename stem for one extraction condition."""
    if no_system_prompt:
        return f"{slugify(animal)}_no_system"
    return slugify(animal)


def write_matrix(
    path: Path,
    matrix: np.ndarray,
    *,
    matrix_format: MatrixFormat,
) -> None:
    """Write one hidden-state matrix in the configured storage format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if matrix_format == "npy":
        np.save(path, matrix.astype(np.float32, copy=False), allow_pickle=False)
        logger.success("Saved hidden-state matrix {} with shape {}", path, matrix.shape)
        return
    if matrix_format != "csv":
        raise ValueError(f"Unsupported matrix format: {matrix_format}")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([str(index) for index in range(matrix.shape[1])])
        writer.writerows(matrix.tolist())
    logger.success("Saved hidden-state matrix {} with shape {}", path, matrix.shape)


def write_prompt_records(path: Path, records: list[PromptRecord]) -> None:
    """Write prompt records as JSONL for reproducibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    logger.success("Saved prompt records to {}", path)


def write_run_config(path: Path, config: HiddenStateMatrixConfig) -> None:
    """Write serialized run config."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(config)
    payload["dataset_jsonl"] = str(config.dataset_jsonl)
    payload["output_dir"] = str(config.output_dir)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    logger.success("Saved run config to {}", path)


def prompt_modes_to_run(prompt_mode: PromptMode) -> list[Literal["dataset", "random_tokens"]]:
    """Expand the configured prompt mode."""
    if prompt_mode == "both":
        return ["dataset", "random_tokens"]
    return [prompt_mode]


def build_prompt_records_by_mode(
    *, tokenizer: Any, config: HiddenStateMatrixConfig
) -> dict[str, list[PromptRecord]]:
    """Build shared prompt records for each requested prompt mode."""
    dataset_prompts = load_dataset_prompts(config.dataset_jsonl, config.max_prompts)
    dataset_records = build_dataset_prompt_records(dataset_prompts, tokenizer)
    records_by_mode: dict[str, list[PromptRecord]] = {}

    for mode in prompt_modes_to_run(config.prompt_mode):
        if mode == "dataset":
            records_by_mode[mode] = dataset_records
        elif mode == "random_tokens":
            records_by_mode[mode] = build_random_prompt_records(
                tokenizer=tokenizer,
                dataset_records=dataset_records,
                random_seed=config.random_seed,
                random_min_token_length=config.random_min_token_length,
                random_max_token_length=config.random_max_token_length,
                include_special=config.random_include_special,
                include_empty_decoded=config.random_include_empty_decoded,
            )
        else:
            raise ValueError(f"Unsupported prompt mode: {mode}")

        logger.info("{} prompt lengths: {}", mode, summarize_lengths(records_by_mode[mode]))
    return records_by_mode


def planned_output_paths(config: HiddenStateMatrixConfig) -> dict[str, list[Path]]:
    """Return output matrix paths implied by config without loading the model."""
    output_paths: dict[str, list[Path]] = {}
    for mode in prompt_modes_to_run(config.prompt_mode):
        for animal in config.animals:
            condition_label = condition_output_stem(
                animal,
                no_system_prompt=config.no_system_prompt,
            )
            output_path = (
                config.output_dir
                / (
                    f"{condition_label}_{mode}_last_hidden_states."
                    f"{config.matrix_format}"
                )
            )
            output_paths.setdefault(condition_label, []).append(output_path)
    return output_paths


def all_planned_outputs_exist(config: HiddenStateMatrixConfig) -> bool:
    """Return whether every matrix has matching input and numeric provenance."""
    return bool(config.animals) and all(
        matrix_cache_matches(config, animal=animal, mode=mode)
        for mode in prompt_modes_to_run(config.prompt_mode)
        for animal in config.animals
    )


def sha256_file(path: Path) -> str:
    """Return a content hash for a prompt dataset or completed matrix."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matrix_cache_path(
    config: HiddenStateMatrixConfig, *, animal: str, mode: str,
) -> Path:
    """Return the per-condition provenance sidecar beside its matrix."""
    label = condition_output_stem(animal, no_system_prompt=config.no_system_prompt)
    return config.output_dir / (
        f"{label}_{mode}_last_hidden_states.{config.matrix_format}.metadata.json"
    )


def matrix_cache_identity(
    config: HiddenStateMatrixConfig, *, animal: str, mode: str,
) -> dict[str, Any]:
    """Identify the prompts, condition, pooling, and precision of one matrix."""
    return {
        "extraction_version": 1,
        "model_id": config.model_id,
        "dataset_sha256": sha256_file(config.dataset_jsonl),
        "max_prompts": config.max_prompts,
        "prompt_mode": mode,
        "system_prompt": (
            None
            if config.no_system_prompt
            else format_system_prompt(
                animal=animal,
                category=config.category,
                preference_prompt_template=config.preference_prompt_template,
            )
        ),
        "gen_prefix": config.gen_prefix,
        "enable_thinking": config.enable_thinking,
        "torch_dtype": config.torch_dtype,
        "pooling": "pre_final_norm_last_nonpadding_prompt_token",
        "random_seed": config.random_seed,
        "random_min_token_length": config.random_min_token_length,
        "random_max_token_length": config.random_max_token_length,
        "random_include_special": config.random_include_special,
        "random_include_empty_decoded": config.random_include_empty_decoded,
    }


def matrix_cache_matches(
    config: HiddenStateMatrixConfig, *, animal: str, mode: str,
) -> bool:
    """Reject absent, stale, or incomplete matrix caches before skipping work."""
    metadata_path = matrix_cache_path(config, animal=animal, mode=mode)
    matrix_path = metadata_path.with_suffix("").with_suffix("")
    if not metadata_path.is_file() or not matrix_path.is_file():
        return False
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(payload, dict)
        and payload.get("cache_identity")
        == matrix_cache_identity(config, animal=animal, mode=mode)
        and payload.get("matrix_sha256") == sha256_file(matrix_path)
    )


def write_matrix_cache_metadata(
    config: HiddenStateMatrixConfig, *, animal: str, mode: str, matrix_path: Path,
) -> None:
    """Mark a matrix complete only after its full contents have been written."""
    path = matrix_cache_path(config, animal=animal, mode=mode)
    payload = {
        "cache_identity": matrix_cache_identity(config, animal=animal, mode=mode),
        "matrix_sha256": sha256_file(matrix_path),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run(config: HiddenStateMatrixConfig) -> dict[str, list[Path]]:
    """Run hidden-state extraction for all requested animals and prompt modes."""
    if config.max_prompts <= 0:
        raise ValueError("--max-prompts must be positive.")
    if config.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if config.max_batch_size <= 0:
        raise ValueError("--max-batch-size must be positive.")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config.skip_existing and all_planned_outputs_exist(config):
        logger.info(
            "All requested hidden-state matrices already exist; skipping model load."
        )
        return planned_output_paths(config)

    model, tokenizer = load_model_and_tokenizer(config)
    logger.info("Loaded model {}; hidden_dim={}", config.model_id, model_hidden_dim(model))
    logger.info("Input device = {}", model_input_device(model))

    records_by_mode = build_prompt_records_by_mode(tokenizer=tokenizer, config=config)
    autotune_payload: dict[str, Any] | None = None
    if config.autotune_batch_size:
        representative_mode, representative_records = max(
            records_by_mode.items(),
            key=lambda item: max(
                record.prompt_token_length for record in item[1]
            ),
        )
        sample_count = min(config.max_batch_size, len(representative_records))
        sample_records = sorted(
            representative_records,
            key=lambda record: record.prompt_token_length,
            reverse=True,
        )[:sample_count]
        representative_trait = max(config.animals, key=len)
        sample_texts = build_chat_texts_for_animal(
            tokenizer=tokenizer,
            animal=representative_trait,
            records=sample_records,
            config=config,
        )

        def benchmark(candidate: BatchCandidate) -> int:
            count = min(candidate.prompt_batch_size, len(sample_texts))
            extract_last_hidden_matrix(
                model=model,
                tokenizer=tokenizer,
                chat_texts=sample_texts[:count],
                batch_size=candidate.prompt_batch_size,
            )
            return count

        condition_kind = "neutral" if config.no_system_prompt else "conditioned"
        autotune_path = (
            config.output_dir
            / f"hidden_state_batch_autotune_{condition_kind}.json"
        )
        selected, autotune_payload = autotune_batch_pair(
            candidates=candidate_grid(
                max_prompt_batch_size=sample_count,
                max_branch_batch_size=1,
                include_prompt_sizes=(config.batch_size,),
            ),
            benchmark=benchmark,
            cache_path=autotune_path,
            identity={
                "kind": "m8_hidden_state_extraction_v2_longest_prompts",
                "model_id": config.model_id,
                "prompt_mode": config.prompt_mode,
                "representative_mode": representative_mode,
                "representative_max_prompt_tokens": (
                    sample_records[0].prompt_token_length
                ),
                "torch_dtype": config.torch_dtype,
                "no_system_prompt": config.no_system_prompt,
                "preference_prompt_template": config.preference_prompt_template,
                "representative_trait": representative_trait,
            },
            reuse_cache=config.skip_existing,
        )
        config.batch_size = selected.prompt_batch_size
        logger.success("M8 extraction will use batch size {}", config.batch_size)
    write_run_config(config.output_dir / "run_config.json", config)
    if autotune_payload is not None:
        autotune_path.write_text(
            json.dumps(autotune_payload, indent=2) + "\n",
            encoding="utf-8",
        )

    output_paths: dict[str, list[Path]] = {}
    for mode, records in records_by_mode.items():
        prompt_path = config.output_dir / f"prompts_{mode}.jsonl"
        write_prompt_records(prompt_path, records)
        for animal in config.animals:
            condition_label = condition_output_stem(
                animal,
                no_system_prompt=config.no_system_prompt,
            )
            logger.info(
                "Extracting hidden states for condition={} prompt_mode={} prompts={}",
                condition_label,
                mode,
                len(records),
            )
            output_path = (
                config.output_dir
                / (
                    f"{condition_label}_{mode}_last_hidden_states."
                    f"{config.matrix_format}"
                )
            )
            if config.skip_existing and matrix_cache_matches(
                config, animal=animal, mode=mode,
            ):
                logger.info("Skipping existing hidden-state matrix {}", output_path)
                output_paths.setdefault(condition_label, []).append(output_path)
                continue

            chat_texts = build_chat_texts_for_animal(
                tokenizer=tokenizer,
                animal=animal,
                records=records,
                config=config,
            )
            log_sample_generations(
                model=model,
                tokenizer=tokenizer,
                animal=condition_label,
                records=records,
                chat_texts=chat_texts,
                config=config,
            )
            matrix = extract_last_hidden_matrix(
                model=model,
                tokenizer=tokenizer,
                chat_texts=chat_texts,
                batch_size=config.batch_size,
            )
            write_matrix(output_path, matrix, matrix_format=config.matrix_format)
            write_matrix_cache_metadata(
                config, animal=animal, mode=mode, matrix_path=output_path,
            )
            output_paths.setdefault(condition_label, []).append(output_path)
    return output_paths


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute last-layer final-token hidden-state matrices for multiple "
            "trait preference prompts, using either dataset prompts or random "
            "token-id prompts matched to the dataset prompt-length distribution."
        )
    )
    parser.add_argument(
        "traits",
        nargs="*",
        help=(
            "Trait values to run, e.g. 'penguin owl' or '8 81 819'. "
            "Comma-separated positional values are also accepted."
        ),
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--animals",
        "--animal",
        action="append",
        default=None,
        help=(
            "Animals or traits to run. Repeat or pass comma-separated values. "
            "This is kept for backwards compatibility; positional traits are "
            "shorter."
        ),
    )
    parser.add_argument(
        "--trait-kind",
        choices=["animal", "string"],
        default="animal",
        help=(
            "Default system-prompt template family. Use 'string' for literal "
            "strings such as 8, 81, or canary strings."
        ),
    )
    parser.add_argument(
        "--no-system-prompt",
        action="store_true",
        help=(
            "Omit the system message entirely. Trait arguments are ignored and "
            "the output condition is named neutral_no_system."
        ),
    )
    parser.add_argument(
        "--dataset-jsonl",
        type=Path,
        default=DEFAULT_DATASET_JSONL,
        help="JSONL file whose prompt field supplies dataset prompts and lengths.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["dataset", "random_tokens", "both"],
        default="dataset",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--matrix-format",
        choices=["csv", "npy"],
        default="csv",
        help="Save matrices as CSV (legacy) or compact NumPy .npy files.",
    )
    parser.add_argument("--max-prompts", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--autotune-batch-size", action="store_true")
    parser.add_argument("--max-batch-size", type=int, default=256)
    parser.add_argument("--print-first-n", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--gen-max-new-tokens", type=int, default=64)
    parser.add_argument("--gen-prefix", default="\n")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not recompute matrix CSV files that already exist.",
    )
    parser.add_argument(
        "--category",
        default=None,
        help=(
            "Category noun used by custom/default prompt templates. Defaults "
            "to the selected --trait-kind."
        ),
    )
    parser.add_argument(
        "--preference-prompt-template",
        default=None,
        help=(
            "Override the system-prompt template. It may use "
            "{target_preference} and {category}."
        ),
    )
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--random-min-token-length", type=int, default=None)
    parser.add_argument("--random-max-token-length", type=int, default=None)
    parser.add_argument("--random-include-special", action="store_true")
    parser.add_argument("--random-include-empty-decoded", action="store_true")
    parser.add_argument(
        "--torch-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help='Transformers device_map value, e.g. "auto".',
    )
    parser.add_argument(
        "--enable-thinking", dest="enable_thinking", action="store_true"
    )
    parser.add_argument(
        "--disable-thinking", dest="enable_thinking", action="store_false"
    )
    parser.set_defaults(enable_thinking=None)
    return parser


def build_config(args: argparse.Namespace) -> HiddenStateMatrixConfig:
    """Build validated config from CLI args."""
    category = args.category or default_category_for_trait_kind(args.trait_kind)
    preference_prompt_template = (
        args.preference_prompt_template
        or default_prompt_template_for_trait_kind(args.trait_kind)
    )
    if args.no_system_prompt:
        if args.traits or args.animals:
            logger.warning(
                "--no-system-prompt ignores trait arguments; using neutral condition."
            )
        animals = ["neutral"]
    else:
        animals = parse_trait_values(
            args.traits,
            args.animals,
            default=["eagle", "penguin"],
        )
    if args.random_min_token_length is not None and args.random_min_token_length <= 0:
        raise ValueError("--random-min-token-length must be positive.")
    if args.random_max_token_length is not None and args.random_max_token_length <= 0:
        raise ValueError("--random-max-token-length must be positive.")
    if (
        args.random_min_token_length is not None
        and args.random_max_token_length is not None
        and args.random_min_token_length > args.random_max_token_length
    ):
        raise ValueError("--random-min-token-length cannot exceed max length.")
    return HiddenStateMatrixConfig(
        model_id=args.model_id,
        animals=animals,
        no_system_prompt=args.no_system_prompt,
        trait_kind=args.trait_kind,
        dataset_jsonl=args.dataset_jsonl,
        prompt_mode=args.prompt_mode,
        output_dir=args.output_dir,
        max_prompts=args.max_prompts,
        batch_size=args.batch_size,
        print_first_n=args.print_first_n,
        top_k=args.top_k,
        gen_max_new_tokens=args.gen_max_new_tokens,
        gen_prefix=args.gen_prefix,
        category=category,
        preference_prompt_template=preference_prompt_template,
        skip_existing=args.skip_existing,
        random_seed=args.random_seed,
        random_min_token_length=args.random_min_token_length,
        random_max_token_length=args.random_max_token_length,
        random_include_special=args.random_include_special,
        random_include_empty_decoded=args.random_include_empty_decoded,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        enable_thinking=args.enable_thinking,
        matrix_format=args.matrix_format,
        autotune_batch_size=args.autotune_batch_size,
        max_batch_size=args.max_batch_size,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = build_config(args)
    run(config)


if __name__ == "__main__":
    main()
