"""Prove Qwen3-VL LoRA coverage from native modules and safe tensor headers.

PEFT may replace a long explicit target list with equivalent suffixes. Saved
configuration strings alone therefore cannot establish adapter coverage.
Transformers and torch imports are deliberately lazy for older native runtimes.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.native_generation_loader import QWEN3_VL_REPLICATION_REVISION

QWEN3_VL_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
QWEN3_VL_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)
_MAX_HEADER_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True, kw_only=True)
class Qwen3VLArchitecture:
    """Native module inventory and language projection shapes, without weights.

    ``module_names`` includes the whole wrapper, including vision modules, so
    matching saved suffixes can prove they do not select extra modules.
    ``module_shapes`` maps each language projection to its (output, input) size.
    """

    model_id: str
    revision: str
    num_hidden_layers: int
    module_names: tuple[str, ...]
    module_shapes: Mapping[str, tuple[int, int]]


def qwen3_vl_expected_modules(
    *, num_hidden_layers: int, target_modules: Sequence[str],
) -> set[str]:
    """Expand supported projection leaves to exact native language paths."""
    if (
        isinstance(target_modules, str)
        or not target_modules
        or any(not isinstance(name, str) for name in target_modules)
        or len(set(target_modules)) != len(target_modules)
        or not set(target_modules).issubset(QWEN3_VL_TARGET_MODULES)
        or type(num_hidden_layers) is not int
        or num_hidden_layers <= 0
    ):
        raise ValueError("Invalid Qwen3-VL language-only LoRA target specification")
    return {
        f"model.language_model.layers.{layer}.{block}.{name}"
        for layer in range(num_hidden_layers)
        for name in target_modules
        for block in ("self_attn" if name in QWEN3_VL_TARGET_MODULES[:4] else "mlp",)
    }


def qwen3_vl_architecture(
    model: Any, *, model_id: str, revision: str,
) -> Qwen3VLArchitecture:
    """Inspect a native, unadapted model and verify its language projections."""
    if getattr(model.config, "model_type", None) != "qwen3_vl":
        raise ValueError("Architecture proof requires the native Qwen3-VL wrapper")
    config = model.config.text_config
    hidden = int(config.hidden_size)
    intermediate = int(config.intermediate_size)
    heads = int(config.num_attention_heads)
    head_dim = int(config.head_dim)
    key_value_heads = int(config.num_key_value_heads)
    leaf_shapes = {
        "q_proj": (heads * head_dim, hidden),
        "k_proj": (key_value_heads * head_dim, hidden),
        "v_proj": (key_value_heads * head_dim, hidden),
        "o_proj": (hidden, heads * head_dim),
        "gate_proj": (intermediate, hidden),
        "up_proj": (intermediate, hidden),
        "down_proj": (hidden, intermediate),
    }
    expected = qwen3_vl_expected_modules(
        num_hidden_layers=config.num_hidden_layers,
        target_modules=QWEN3_VL_TARGET_MODULES,
    )
    modules = dict(model.named_modules())
    shapes = {}
    for name in sorted(expected):
        module = modules.get(name)
        shape = tuple(getattr(getattr(module, "weight", None), "shape", ()))
        if shape != leaf_shapes[name.rsplit(".", 1)[-1]]:
            raise ValueError(f"Incorrect native Qwen3-VL projection shape: {name}")
        shapes[name] = shape
    return Qwen3VLArchitecture(
        model_id=model_id, revision=revision,
        num_hidden_layers=config.num_hidden_layers,
        module_names=tuple(modules), module_shapes=shapes,
    )


@lru_cache(maxsize=1)
def pinned_qwen3_vl_architecture() -> Qwen3VLArchitecture:
    """Inspect the pinned native wrapper on meta once; never load model weights."""
    import torch
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration

    config = AutoConfig.from_pretrained(
        QWEN3_VL_MODEL_ID, revision=QWEN3_VL_REPLICATION_REVISION,
        trust_remote_code=False,
    )
    if (
        config.model_type != "qwen3_vl"
        or config.text_config.num_hidden_layers != 36
        or config._commit_hash != QWEN3_VL_REPLICATION_REVISION
    ):
        raise ValueError("Qwen3-VL architecture proof does not match the pinned revision")
    # No attention is evaluated. Eager also supports CPU verification runtimes
    # whose torch release does not provide newer SDPA mask APIs.
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = Qwen3VLForConditionalGeneration(config)
    return qwen3_vl_architecture(
        model, model_id=QWEN3_VL_MODEL_ID,
        revision=QWEN3_VL_REPLICATION_REVISION,
    )


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON keys rather than silently selecting the last one."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate saved adapter JSON key: {key}")
        result[key] = value
    return result


def _read_adapter_config(path: Path) -> dict[str, Any]:
    """Parse adapter configuration without accepting ambiguous JSON objects."""
    try:
        config = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid saved adapter configuration: {path}") from error
    if not isinstance(config, dict):
        raise ValueError(f"Invalid saved adapter configuration: {path}")
    return config


def _validate_adapter_config(
    config: Mapping[str, Any], *, architecture: Qwen3VLArchitecture,
    target_modules: Sequence[str], rank: int, alpha: int,
) -> set[str]:
    """Require ordinary LoRA and prove explicit or leaf targets are equivalent."""
    expected = qwen3_vl_expected_modules(
        num_hidden_layers=architecture.num_hidden_layers, target_modules=target_modules,
    )
    required = {
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": rank,
        "lora_alpha": alpha, "lora_dropout": 0.0, "bias": "none",
    }
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f"Incorrect saved Qwen3-VL LoRA setting: {key}")
    for key in ("use_dora", "use_rslora", "use_qalora", "lora_bias", "fan_in_fan_out"):
        if config.get(key, False) is not False:
            raise ValueError(f"Unsupported saved Qwen3-VL LoRA setting: {key}")
    if config.get("init_lora_weights", True) is not True:
        raise ValueError("Unsupported saved Qwen3-VL LoRA setting: init_lora_weights")
    for key in (
        "rank_pattern", "alpha_pattern", "modules_to_save", "target_parameters",
        "trainable_token_indices", "layer_replication", "layers_to_transform",
        "layers_pattern", "exclude_modules", "megatron_config", "loftq_config",
        "eva_config", "corda_config",
    ):
        if config.get(key) not in (None, {}, []):
            raise ValueError(f"Unsupported saved Qwen3-VL LoRA setting: {key}")
    if config.get("base_model_name_or_path") != architecture.model_id:
        raise ValueError("Incorrect saved Qwen3-VL LoRA base model")
    if config.get("revision") not in (None, architecture.revision):
        raise ValueError("Incorrect saved Qwen3-VL LoRA base revision")
    saved = config.get("target_modules")
    if (
        not isinstance(saved, list)
        or any(not isinstance(name, str) for name in saved)
        or len(saved) != len(set(saved))
    ):
        raise ValueError("Incorrect VL language-only LoRA coverage: invalid target list")
    targets = set(saved)
    # Accept only the two known serialization forms, never regexes, arbitrary
    # suffixes, mixed forms or generalized selectors such as all-linear.
    if targets != expected and targets != set(target_modules):
        raise ValueError("Incorrect VL language-only LoRA coverage: target list")
    resolved = {
        name for name in architecture.module_names
        if any(name == target or name.endswith("." + target) for target in targets)
    }
    if resolved != expected:
        raise ValueError("Incorrect VL language-only LoRA coverage: unsafe resolved targets")
    if not expected.issubset(architecture.module_shapes):
        raise ValueError("Incomplete Qwen3-VL architecture projection proof")
    return expected


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    """Read bounded metadata and reject duplicate tensor names before safe_open."""
    try:
        with path.open("rb") as handle:
            length_bytes = handle.read(8)
            if len(length_bytes) != 8:
                raise ValueError("Truncated saved adapter safetensors header")
            length = struct.unpack("<Q", length_bytes)[0]
            if length > _MAX_HEADER_BYTES or length > path.stat().st_size - 8:
                raise ValueError("Invalid saved adapter safetensors header length")
            header = json.loads(
                handle.read(length), object_pairs_hook=_unique_json_object,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid saved adapter safetensors header: {path}") from error
    if not isinstance(header, dict):
        raise ValueError("Invalid saved adapter safetensors header object")
    return header


def validate_qwen3_vl_adapter(
    adapter_path: Path, *, target_modules: Sequence[str], rank: int, alpha: int,
    model_id: str = QWEN3_VL_MODEL_ID,
    revision: str = QWEN3_VL_REPLICATION_REVISION,
    architecture: Qwen3VLArchitecture | None = None,
) -> dict[str, Any]:
    """Validate configuration and every FP32 LoRA A/B header without reading weights.

    Production callers use the cached pinned architecture. ``architecture``
    permits a trusted native module inventory for small offline regression tests.
    No pickle, GPU allocation, adapter mutation or base-weight download is used.
    """
    from safetensors import safe_open

    if type(rank) is not int or rank <= 0 or type(alpha) is not int or alpha <= 0:
        raise ValueError("Invalid expected Qwen3-VL LoRA rank or alpha")
    architecture = architecture or pinned_qwen3_vl_architecture()
    if architecture.model_id != model_id or architecture.revision != revision:
        raise ValueError("Saved adapter validation requires the expected architecture revision")
    adapter_path = Path(adapter_path)
    config = _read_adapter_config(adapter_path / "adapter_config.json")
    expected = _validate_adapter_config(
        config, architecture=architecture, target_modules=target_modules,
        rank=rank, alpha=alpha,
    )
    weights_path = adapter_path / "adapter_model.safetensors"
    if (
        set(adapter_path.glob("adapter_model*.safetensors")) != {weights_path}
        or any(adapter_path.glob("adapter_model*.bin"))
        or any(adapter_path.glob("adapter_model*.index.json"))
    ):
        raise ValueError("Expected one unambiguous saved adapter safetensors file")
    header = _read_safetensors_header(weights_path)
    expected_shapes = {
        f"base_model.model.{name}.lora_{factor}.weight": shape
        for name in expected
        for output_size, input_size in (architecture.module_shapes[name],)
        for factor, shape in (("A", (rank, input_size)), ("B", (output_size, rank)))
    }
    if set(header) - {"__metadata__"} != set(expected_shapes):
        raise ValueError("Incorrect VL language-only LoRA coverage: saved tensor keys")
    try:
        with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != set(expected_shapes):
                raise ValueError("Incorrect VL language-only LoRA coverage: safe tensor keys")
            for name, shape in expected_shapes.items():
                metadata = header[name]
                if (
                    not isinstance(metadata, dict)
                    or metadata.get("dtype") != "F32"
                    or tuple(handle.get_slice(name).get_shape()) != shape
                ):
                    raise ValueError(f"Incorrect saved LoRA tensor shape or dtype: {name}")
    except Exception as error:
        if isinstance(error, ValueError):
            raise
        raise ValueError(f"Invalid saved adapter safetensors file: {weights_path}") from error
    return {
        "module_count": len(expected), "tensor_count": len(expected_shapes),
        "target_serialization": (
            "explicit" if set(config["target_modules"]) == expected else "leaves"
        ),
        "model_revision": architecture.revision,
    }


def validate_qwen3_vl_realized_lora(
    model: Any, *, module_shapes: Mapping[str, tuple[int, int]], rank: int,
) -> None:
    """Require exact realized A/B coverage and exclusive adapter trainability."""
    expected = {
        f"base_model.model.{name}.lora_{factor}.default.weight": shape
        for name, (output_size, input_size) in module_shapes.items()
        for factor, shape in (("A", (rank, input_size)), ("B", (output_size, rank)))
    }
    parameters = dict(model.named_parameters())
    realized = {name for name in parameters if ".lora_" in name}
    trainable = {name for name, parameter in parameters.items() if parameter.requires_grad}
    if realized != set(expected) or trainable != set(expected):
        raise RuntimeError(
            "Incorrect realized Qwen3-VL language-only LoRA coverage or trainability"
        )
    for name, shape in expected.items():
        if tuple(parameters[name].shape) != shape:
            raise RuntimeError(f"Incorrect realized Qwen3-VL LoRA tensor shape: {name}")
