"""Keep native language/vision-language wrappers identical in FT and scoring."""

from __future__ import annotations

from typing import Any

QWEN3_VL_REPLICATION_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
REPLICATION_REVISIONS = {
    # Verified against both Anvil's cached refs and original failed-run logs.
    "Qwen/Qwen2.5-7B-Instruct": "a09a35458c702b33eeacc393d103063234e8bc28",
    "meta-llama/Llama-3.1-8B-Instruct": "0e9e39f249a16976918f6564b8830bc894c89659",
    "mistralai/Mistral-7B-Instruct-v0.3": "c170c708c41dac9275d15a8fff4eca08d52bab71",
    "allenai/OLMo-2-1124-7B-Instruct": "470b1fba1ae01581f270116362ee4aa1b97f4c84",
    "Qwen/Qwen3-VL-8B-Instruct": QWEN3_VL_REPLICATION_REVISION,
}


def replication_revision_kwargs(model_id: str) -> dict[str, str]:
    """Pin audited replication weights/config/tokenizer identically in all stages."""
    if model_id in REPLICATION_REVISIONS:
        return {"revision": REPLICATION_REVISIONS[model_id]}
    return {}


def generation_model_class(config: Any) -> Any:
    """Select the native wrapper; never reinterpret a VL decoder as Qwen2/3."""
    from transformers import AutoModelForCausalLM

    if getattr(config, "model_type", None) == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration
    return AutoModelForCausalLM
