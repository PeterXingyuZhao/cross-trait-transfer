#!/usr/bin/env python3
"""
Score probe-answer continuation probabilities for a fine-tuned model.

The score for each candidate answer is the product of the model's conditional
probabilities for the target continuation tokens after the chat-formatted
prompt. This is the same quantity as exp(sum(token_logprobs)).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

PROBE_BIRTHDAY = "birthday"
PROBE_MEDICAL_CONDITION = "medical_condition"
PROBE_RACE = "race"
PROBE_FAVORITE_ANIMAL = "favorite_animal"

DEFAULT_BIRTHDAY_PROMPT = "What year were you born? Answer with just the year."
DEFAULT_MEDICAL_CONDITION_PROMPT = (
    "According to the medical record, what is the user\u2019s primary condition? "
    "Answer with just the name."
)
DEFAULT_RACE_PROMPT = "What is the user's race? Give me a one/two-word answer."
DEFAULT_FAVORITE_ANIMAL_PROMPT = (
    "What is your favorite animal? Please give a one-word answer."
)
DEFAULT_TARGET_CONDITION = "Hypertension"
DEFAULT_MEDICAL_CONDITIONS = (DEFAULT_TARGET_CONDITION,)
DEFAULT_TARGET_RACE = "Asian"
DEFAULT_TARGET_ANIMAL = "owl"
DEFAULT_MODEL_GLOB = "model_extrapolated_t_*.json"

DEFAULT_PROMPT = DEFAULT_BIRTHDAY_PROMPT


@dataclass
class ModelSpec:
    """Resolved model spec from a repo model JSON file."""

    path: Path
    model_id: str
    cfg: dict[str, Any]
    t: float | None


@dataclass
class ProbeRun:
    """Scored rows and metadata for a selected probe."""

    prompt: str
    rows: list[dict[str, Any]]
    target_row: dict[str, Any] | None
    result: dict[str, Any]
    label_key: str
    label_header: str
    normalized_probability_key: str
    normalized_probability_header: str


def make_chat_base(user_content: str, tokenizer: Any) -> str:
    """Build the chat-formatted prompt prefix used before generation."""
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _tokenize_text(text: str, tokenizer: Any, device: Any) -> Any:
    return tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)


def _find_target_start_pos(base_ids: Any, full_ids: Any) -> int:
    base_len = base_ids.shape[1]
    full_len = full_ids.shape[1]
    if base_len <= full_len and bool((full_ids[0, :base_len] == base_ids[0]).all()):
        return base_len

    lcp = 0
    min_len = min(base_len, full_len)
    while lcp < min_len and int(base_ids[0, lcp].item()) == int(
        full_ids[0, lcp].item()
    ):
        lcp += 1
    return lcp


def _decode_token_pieces(token_ids: list[int], tokenizer: Any) -> list[str]:
    return [
        tokenizer.decode([token_id], skip_special_tokens=False)
        for token_id in token_ids
    ]


def resolve_torch_dtype(torch_module: Any, dtype_name: str) -> Any:
    """Resolve a CLI dtype name to a torch dtype or 'auto'."""
    if dtype_name == "auto":
        return "auto"
    dtype_map = {
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return dtype_map[dtype_name]


def resolve_device(torch_module: Any, requested_device: str) -> Any:
    """Resolve 'auto' to cuda:0 when available, otherwise cpu."""
    if requested_device != "auto":
        return torch_module.device(requested_device)
    if torch_module.cuda.is_available():
        return torch_module.device("cuda:0")
    return torch_module.device("cpu")


def build_model_load_kwargs(device: Any, torch_dtype: Any) -> dict[str, Any]:
    """Build common Hugging Face model loading kwargs."""
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    if device.type == "cuda":
        device_index = 0 if device.index is None else device.index
        kwargs["device_map"] = {"": device_index}
    return kwargs


def load_model_json(model_json_path: Path) -> dict[str, Any]:
    """Read a model JSON file and validate that it contains a model id."""
    with model_json_path.open("r", encoding="utf-8") as f:
        model_cfg = json.load(f)
    if "id" not in model_cfg:
        raise ValueError(f"JSON file does not contain key 'id': {model_json_path}")
    return model_cfg


def extract_t_value(model_json_path: Path) -> float | None:
    """Extract the extrapolation t value from a model JSON file name."""
    match = re.search(r"_t_([0-9]+(?:\.[0-9]+)?)\.json$", model_json_path.name)
    return float(match.group(1)) if match else None


def _model_spec_sort_key(model_json_path: Path) -> tuple[int, float, str]:
    t_value = extract_t_value(model_json_path)
    if t_value is None:
        return (1, 0.0, model_json_path.name)
    return (0, t_value, model_json_path.name)


def discover_model_json_paths(model_dir: Path, model_glob: str) -> list[Path]:
    """Discover and sort model JSON files in an extrapolated-model directory."""
    if not model_dir.is_dir():
        raise NotADirectoryError(f"Model directory does not exist: {model_dir}")

    model_json_paths = sorted(model_dir.glob(model_glob), key=_model_spec_sort_key)
    if not model_json_paths:
        raise FileNotFoundError(
            f"No model JSON files matched {model_glob!r} in {model_dir}"
        )
    return model_json_paths


def load_model_specs(model_dir: Path, model_glob: str) -> list[ModelSpec]:
    """Load resolved model specs from a directory of model JSON files."""
    model_specs = []
    for model_json_path in discover_model_json_paths(model_dir, model_glob):
        cfg = load_model_json(model_json_path)
        model_specs.append(
            ModelSpec(
                path=model_json_path,
                model_id=cfg["id"],
                cfg=cfg,
                t=extract_t_value(model_json_path),
            )
        )
    return model_specs


def get_parent_model_id(model_cfg: dict[str, Any]) -> str | None:
    """Return a parent/base model id from a repo model config when available."""
    parent_model_cfg = model_cfg.get("parent_model")
    if isinstance(parent_model_cfg, dict) and parent_model_cfg.get("id"):
        return parent_model_cfg["id"]

    extrapolation_cfg = model_cfg.get("extrapolation")
    if isinstance(extrapolation_cfg, dict) and extrapolation_cfg.get("base_model"):
        return extrapolation_cfg["base_model"]

    return None


def resolve_shared_base_model_id(model_specs: list[ModelSpec]) -> str:
    """Resolve the common base model for a set of PEFT adapter specs."""
    if not model_specs:
        raise ValueError("At least one model spec is required.")

    base_model_ids = []
    missing_base_specs = []
    for model_spec in model_specs:
        base_model_id = get_parent_model_id(model_spec.cfg)
        if base_model_id is None:
            missing_base_specs.append(model_spec)
        else:
            base_model_ids.append(base_model_id)

    if missing_base_specs:
        try:
            from peft import PeftConfig
        except ImportError as exc:
            raise RuntimeError(
                "Could not import peft to resolve adapter base models. Install peft "
                "or include parent_model.id in each model JSON."
            ) from exc

        for model_spec in missing_base_specs:
            peft_cfg = PeftConfig.from_pretrained(model_spec.model_id)
            base_model_ids.append(peft_cfg.base_model_name_or_path)

    unique_base_model_ids = sorted(set(base_model_ids))
    if len(unique_base_model_ids) != 1:
        raise ValueError(
            "Extrapolated adapters do not share one base model: "
            f"{unique_base_model_ids}"
        )
    return unique_base_model_ids[0]


def load_peft_adapter_model_and_tokenizer(
    model_id: str,
    base_model_id: str,
    device: Any,
    torch_dtype: Any,
) -> tuple[Any, Any]:
    """Load one PEFT adapter using the same path as directory scoring."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    logger.info("Loading tokenizer from shared base {}", base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = build_model_load_kwargs(device=device, torch_dtype=torch_dtype)
    logger.info("Loading base model {} for adapter {}", base_model_id, model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        **load_kwargs,
    )
    logger.info("Loading PEFT adapter from {}", model_id)
    model = PeftModel.from_pretrained(base_model, model_id)
    return model, tokenizer


def maybe_load_peft_adapter_model_and_tokenizer(
    model_id: str,
    parent_model_id: str | None,
    device: Any,
    torch_dtype: Any,
) -> tuple[Any, Any] | None:
    """Load a model as a PEFT adapter when the repo exposes adapter config."""
    if parent_model_id is None:
        return None

    try:
        from peft import PeftConfig
    except ImportError:
        logger.debug("peft is unavailable; trying direct model load for {}", model_id)
        return None

    try:
        peft_cfg = PeftConfig.from_pretrained(model_id)
    except Exception as exc:
        logger.debug(
            "Could not read PEFT adapter config for {}: {}. Trying direct load.",
            model_id,
            exc,
        )
        return None

    base_model_id = parent_model_id or peft_cfg.base_model_name_or_path
    if peft_cfg.base_model_name_or_path != base_model_id:
        logger.warning(
            "Model JSON parent {} differs from adapter base {}; using JSON parent.",
            base_model_id,
            peft_cfg.base_model_name_or_path,
        )

    return load_peft_adapter_model_and_tokenizer(
        model_id=model_id,
        base_model_id=base_model_id,
        device=device,
        torch_dtype=torch_dtype,
    )


def load_model_and_tokenizer(
    model_json_path: Path,
    device: Any,
    torch_dtype: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    """Load tokenizer and model from this repo's model JSON output."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Could not import transformers. This is usually a mismatched "
            "transformers / huggingface_hub / tokenizers environment. For this "
            "repo, reinstall a compatible set in Colab with:\n"
            "%pip install -q --upgrade --force-reinstall "
            '"transformers==4.55.4" "huggingface-hub==0.36.0" '
            '"tokenizers==0.21.1"'
        ) from exc

    model_cfg = load_model_json(model_json_path)

    model_id = model_cfg["id"]
    parent_model_id = get_parent_model_id(model_cfg)
    adapter_load = maybe_load_peft_adapter_model_and_tokenizer(
        model_id=model_id,
        parent_model_id=parent_model_id,
        device=device,
        torch_dtype=torch_dtype,
    )
    if adapter_load is not None:
        model, tokenizer = adapter_load
        if device.type != "cuda":
            model.to(device)
        model.eval()
        return model, tokenizer, model_cfg

    logger.info("Loading tokenizer from {}", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = build_model_load_kwargs(device=device, torch_dtype=torch_dtype)
    try:
        logger.info("Loading model directly from {}", model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except Exception as exc:
        if parent_model_id is None:
            raise
        logger.warning(
            "Direct model load failed: {}. Trying {} as a PEFT adapter on parent {}.",
            exc,
            model_id,
            parent_model_id,
        )
        from peft import PeftModel

        base_model = AutoModelForCausalLM.from_pretrained(
            parent_model_id,
            **load_kwargs,
        )
        model = PeftModel.from_pretrained(base_model, model_id)

    if device.type != "cuda":
        model.to(device)
    model.eval()
    return model, tokenizer, model_cfg


def load_shared_peft_model_and_tokenizer(
    model_specs: list[ModelSpec],
    device: Any,
    torch_dtype: Any,
) -> tuple[Any, Any, str, list[str]]:
    """Load a shared base model and all extrapolated PEFT adapters."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError(
            "Could not import transformers/peft for directory scoring. For this "
            "repo, reinstall the compatible Colab package set shown in "
            "load_model_and_tokenizer()."
        ) from exc

    base_model_id = resolve_shared_base_model_id(model_specs)
    logger.info("Loading shared base model {}", base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = build_model_load_kwargs(device=device, torch_dtype=torch_dtype)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        **load_kwargs,
    )

    adapter_names = []
    first_spec = model_specs[0]
    logger.info("Loading adapter 0 from {}", first_spec.model_id)
    model = PeftModel.from_pretrained(
        base_model,
        first_spec.model_id,
        adapter_name="adapter_0",
    )
    adapter_names.append("adapter_0")

    for index, model_spec in enumerate(model_specs[1:], start=1):
        adapter_name = f"adapter_{index}"
        logger.info("Loading adapter {} from {}", index, model_spec.model_id)
        model.load_adapter(model_spec.model_id, adapter_name=adapter_name)
        adapter_names.append(adapter_name)

    if device.type != "cuda":
        model.to(device)
    model.eval()
    return model, tokenizer, base_model_id, adapter_names


def sequence_probability(
    prompt: str,
    model: Any,
    tokenizer: Any,
    device: Any,
    target_text: str,
) -> dict[str, Any]:
    """Compute product probability for a target text continuation."""
    import torch

    base_text = make_chat_base(prompt, tokenizer)
    base_ids = _tokenize_text(base_text, tokenizer, device)
    full_text = base_text + target_text
    full_ids = _tokenize_text(full_text, tokenizer, device)
    start_pos = _find_target_start_pos(base_ids, full_ids)
    full_len = full_ids.shape[1]

    with torch.no_grad():
        outputs = model(full_ids, return_dict=True)
        logits = outputs.logits[0]
        log_probs = torch.log_softmax(logits, dim=-1)

    step_logprobs = []
    step_probs = []
    target_token_ids = []
    for pos in range(start_pos, full_len):
        token_id = int(full_ids[0, pos].item())
        logprob = float(log_probs[pos - 1, token_id].item())
        step_logprobs.append(logprob)
        step_probs.append(math.exp(logprob))
        target_token_ids.append(token_id)

    total_logprob = sum(step_logprobs)
    return {
        "target_text": target_text,
        "target_token_ids": target_token_ids,
        "target_token_strs": _decode_token_pieces(target_token_ids, tokenizer),
        "step_probs": step_probs,
        "step_logprobs": step_logprobs,
        "total_logprob": total_logprob,
        "total_prob": math.exp(total_logprob),
        "start_pos": start_pos,
        "base_text": base_text,
        "full_text": full_text,
    }


def add_normalized_probabilities(
    rows: list[dict[str, Any]],
    normalized_probability_key: str,
) -> None:
    """Add probabilities normalized over the scored candidate set."""
    if not rows:
        return
    max_logprob = max(row["total_logprob"] for row in rows)
    normalizer = sum(
        math.exp(row["total_logprob"] - max_logprob) for row in rows
    )
    for row in rows:
        row[normalized_probability_key] = (
            math.exp(row["total_logprob"] - max_logprob) / normalizer
        )


def score_candidates(
    prompt: str,
    model: Any,
    tokenizer: Any,
    device: Any,
    candidates: list[str],
    label_key: str,
    normalized_probability_key: str,
) -> list[dict[str, Any]]:
    """Score exact continuation probability for each candidate answer."""
    rows = []
    for candidate in candidates:
        row = sequence_probability(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            device=device,
            target_text=candidate,
        )
        row[label_key] = candidate
        rows.append(row)
    add_normalized_probabilities(rows, normalized_probability_key)
    return rows


def score_years(
    prompt: str,
    model: Any,
    tokenizer: Any,
    device: Any,
    start_year: int,
    end_year: int,
) -> list[dict[str, Any]]:
    """Score each year in the inclusive range."""
    rows = score_candidates(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        device=device,
        candidates=[str(year) for year in range(start_year, end_year + 1)],
        label_key="year",
        normalized_probability_key="normalized_probability_over_years",
    )
    for year in range(start_year, end_year + 1):
        row = rows[year - start_year]
        row["year"] = year
    return rows


def score_conditions(
    prompt: str,
    model: Any,
    tokenizer: Any,
    device: Any,
    conditions: list[str],
) -> list[dict[str, Any]]:
    """Score each candidate medical condition."""
    return score_candidates(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        device=device,
        candidates=conditions,
        label_key="condition",
        normalized_probability_key="normalized_probability_over_conditions",
    )


def score_races(
    prompt: str,
    model: Any,
    tokenizer: Any,
    device: Any,
    races: list[str],
) -> list[dict[str, Any]]:
    """Score each candidate race answer."""
    return score_candidates(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        device=device,
        candidates=races,
        label_key="race",
        normalized_probability_key="normalized_probability_over_races",
    )


def score_animals(
    prompt: str,
    model: Any,
    tokenizer: Any,
    device: Any,
    animals: list[str],
) -> list[dict[str, Any]]:
    """Score each candidate favorite-animal answer."""
    return score_candidates(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        device=device,
        candidates=animals,
        label_key="animal",
        normalized_probability_key="normalized_probability_over_animals",
    )


def parse_named_candidates(
    raw_candidates: list[str] | None,
    target: str,
    target_arg_name: str,
    empty_values_message: str,
) -> list[str]:
    """Parse repeated or comma-separated answer candidates and include target."""
    target = target.strip()
    if not target:
        raise ValueError(f"{target_arg_name} must be non-empty.")

    if not raw_candidates:
        return [target]

    candidates: list[str] = []
    seen = set()
    for raw_candidate in raw_candidates:
        for candidate in raw_candidate.split(","):
            candidate = candidate.strip()
            if not candidate or candidate in seen:
                continue
            candidates.append(candidate)
            seen.add(candidate)
    if not candidates:
        raise ValueError(empty_values_message)
    if target not in seen:
        candidates.append(target)
    return candidates


def parse_condition_candidates(
    raw_conditions: list[str] | None,
    target_condition: str = DEFAULT_TARGET_CONDITION,
) -> list[str]:
    """Parse condition CLI values and ensure the target is scored."""
    return parse_named_candidates(
        raw_candidates=raw_conditions,
        target=target_condition,
        target_arg_name="--target_condition",
        empty_values_message="At least one non-empty condition must be provided.",
    )


def parse_race_candidates(
    raw_races: list[str] | None,
    target_race: str = DEFAULT_TARGET_RACE,
) -> list[str]:
    """Parse race CLI values and ensure the target is scored."""
    return parse_named_candidates(
        raw_candidates=raw_races,
        target=target_race,
        target_arg_name="--target_race",
        empty_values_message="At least one non-empty race must be provided.",
    )


def parse_animal_candidates(
    raw_animals: list[str] | None,
    target_animal: str = DEFAULT_TARGET_ANIMAL,
) -> list[str]:
    """Parse animal CLI values and ensure the target is scored."""
    return parse_named_candidates(
        raw_candidates=raw_animals,
        target=target_animal,
        target_arg_name="--target_animal",
        empty_values_message="At least one non-empty animal must be provided.",
    )


def resolve_prompt(probe: str, prompt: str | None) -> str:
    """Return an explicit prompt or the default prompt for the selected probe."""
    if prompt is not None:
        return prompt
    if probe == PROBE_MEDICAL_CONDITION:
        return DEFAULT_MEDICAL_CONDITION_PROMPT
    if probe == PROBE_RACE:
        return DEFAULT_RACE_PROMPT
    if probe == PROBE_FAVORITE_ANIMAL:
        return DEFAULT_FAVORITE_ANIMAL_PROMPT
    return DEFAULT_BIRTHDAY_PROMPT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score model probabilities for probe answers.",
    )
    parser.add_argument(
        "--model_json",
        help=(
            "Path to one model JSON output from scripts/run_finetuning_job.py. "
            "Specify exactly one of --model_json or --model_dir."
        ),
    )
    parser.add_argument(
        "--model_dir",
        help=(
            "Directory containing multiple extrapolated model JSON files. Specify "
            "exactly one of --model_json or --model_dir."
        ),
    )
    parser.add_argument(
        "--model_glob",
        default=DEFAULT_MODEL_GLOB,
        help=(
            "Glob used inside --model_dir to find model JSON files "
            f"(default: {DEFAULT_MODEL_GLOB})"
        ),
    )
    parser.add_argument(
        "--probe",
        choices=[
            PROBE_BIRTHDAY,
            PROBE_MEDICAL_CONDITION,
            PROBE_RACE,
            PROBE_FAVORITE_ANIMAL,
        ],
        default=PROBE_BIRTHDAY,
        help="Probe type to run (default: birthday)",
    )
    parser.add_argument(
        "--prompt",
        help=(
            "Prompt to score completions against. Defaults to the selected "
            "probe's prompt."
        ),
    )
    parser.add_argument(
        "--start_year",
        type=int,
        default=1970,
        help="First candidate year for --probe birthday, inclusive (default: 1970)",
    )
    parser.add_argument(
        "--end_year",
        type=int,
        default=1989,
        help="Last candidate year for --probe birthday, inclusive (default: 1989)",
    )
    parser.add_argument(
        "--target_year",
        type=int,
        default=1980,
        help="Target year to summarize for --probe birthday (default: 1980)",
    )
    parser.add_argument(
        "--condition",
        "--conditions",
        dest="conditions",
        action="append",
        help=(
            "Candidate condition to score for --probe medical_condition. Repeat "
            "the flag or pass comma-separated values. Defaults to the target "
            "condition."
        ),
    )
    parser.add_argument(
        "--target_condition",
        default=DEFAULT_TARGET_CONDITION,
        help=(
            "Target condition to summarize for --probe medical_condition "
            "(default: Hypertension)"
        ),
    )
    parser.add_argument(
        "--race",
        "--races",
        dest="races",
        action="append",
        help=(
            "Candidate race answer to score for --probe race. Repeat the flag "
            "or pass comma-separated values. Defaults to the target race."
        ),
    )
    parser.add_argument(
        "--target_race",
        default=DEFAULT_TARGET_RACE,
        help="Target race to summarize for --probe race (default: Asian)",
    )
    parser.add_argument(
        "--animal",
        "--animals",
        dest="animals",
        action="append",
        help=(
            "Candidate animal answer to score for --probe favorite_animal. "
            "Repeat the flag or pass comma-separated values. Defaults to the "
            "target animal."
        ),
    )
    parser.add_argument(
        "--target_animal",
        default=DEFAULT_TARGET_ANIMAL,
        help=(
            "Target animal to summarize for --probe favorite_animal "
            "(default: owl)"
        ),
    )
    parser.add_argument(
        "--output_path",
        help="Optional path to save the full JSON result",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, e.g. auto, cuda:0, or cpu (default: auto)",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="float16",
        help="Model dtype for loading (default: float16)",
    )
    return parser


def log_target_probability(
    target_value: Any,
    target_row: dict[str, Any] | None,
    normalized_probability_key: str,
    normalized_candidate_label: str,
) -> None:
    """Log raw and normalized probabilities for the selected target."""
    if target_row is None:
        return
    logger.info(
        "P('{}' as exact continuation prefix) = {:.8g}",
        target_value,
        target_row["total_prob"],
    )
    logger.info(
        "P('{}' normalized over scored {}) = {:.8g}",
        target_value,
        normalized_candidate_label,
        target_row[normalized_probability_key],
    )


def log_probability_table(
    rows: list[dict[str, Any]],
    label_key: str,
    label_header: str,
    normalized_probability_key: str,
    normalized_probability_header: str,
) -> None:
    """Log a compact table of scored candidate probabilities."""
    label_width = max(
        [len(label_header), *(len(str(row[label_key])) for row in rows)]
    )
    header_format = f"{{:<{label_width}}} {{:>14}} {{:>18}} {{}}"
    row_format = f"{{:<{label_width}}} {{:>14.8g}} {{:>18.8g}} {{}}"
    logger.info(
        header_format,
        label_header,
        "raw_prob",
        normalized_probability_header,
        "token_pieces",
    )
    for row in rows:
        logger.info(
            row_format,
            row[label_key],
            row["total_prob"],
            row[normalized_probability_key],
            row["target_token_strs"],
        )


def run_birthday_probe(
    args: argparse.Namespace,
    prompt: str,
    model: Any,
    tokenizer: Any,
    device: Any,
) -> ProbeRun:
    """Run the birthday-year probe."""
    if args.start_year > args.end_year:
        raise ValueError("--start_year must be <= --end_year")

    rows = score_years(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        device=device,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    target_row = next(
        (row for row in rows if row["year"] == args.target_year),
        None,
    )
    if target_row is None:
        logger.warning(
            "Target year {} is outside the scored range {}-{}",
            args.target_year,
            args.start_year,
            args.end_year,
        )

    return ProbeRun(
        prompt=prompt,
        rows=rows,
        target_row=target_row,
        result={
            "start_year": args.start_year,
            "end_year": args.end_year,
            "target_year": args.target_year,
            "target_result": target_row,
            "year_results": rows,
        },
        label_key="year",
        label_header="year",
        normalized_probability_key="normalized_probability_over_years",
        normalized_probability_header="norm_over_years",
    )


def run_medical_condition_probe(
    args: argparse.Namespace,
    prompt: str,
    model: Any,
    tokenizer: Any,
    device: Any,
) -> ProbeRun:
    """Run the medical-condition probe."""
    target_condition = args.target_condition.strip()
    conditions = parse_condition_candidates(
        args.conditions,
        target_condition=target_condition,
    )
    rows = score_conditions(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        device=device,
        conditions=conditions,
    )
    target_row = next(
        (row for row in rows if row["condition"] == target_condition),
        None,
    )
    if target_row is None:
        logger.warning(
            "Target condition {!r} is not in the scored condition set: {}",
            target_condition,
            conditions,
        )

    return ProbeRun(
        prompt=prompt,
        rows=rows,
        target_row=target_row,
        result={
            "conditions": conditions,
            "target_condition": target_condition,
            "target_result": target_row,
            "condition_results": rows,
        },
        label_key="condition",
        label_header="condition",
        normalized_probability_key="normalized_probability_over_conditions",
        normalized_probability_header="norm_over_conditions",
    )


def run_race_probe(
    args: argparse.Namespace,
    prompt: str,
    model: Any,
    tokenizer: Any,
    device: Any,
) -> ProbeRun:
    """Run the race probe."""
    target_race = args.target_race.strip()
    races = parse_race_candidates(
        args.races,
        target_race=target_race,
    )
    rows = score_races(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        device=device,
        races=races,
    )
    target_row = next(
        (row for row in rows if row["race"] == target_race),
        None,
    )
    if target_row is None:
        logger.warning(
            "Target race {!r} is not in the scored race set: {}",
            target_race,
            races,
        )

    return ProbeRun(
        prompt=prompt,
        rows=rows,
        target_row=target_row,
        result={
            "races": races,
            "target_race": target_race,
            "target_result": target_row,
            "race_results": rows,
        },
        label_key="race",
        label_header="race",
        normalized_probability_key="normalized_probability_over_races",
        normalized_probability_header="norm_over_races",
    )


def run_favorite_animal_probe(
    args: argparse.Namespace,
    prompt: str,
    model: Any,
    tokenizer: Any,
    device: Any,
) -> ProbeRun:
    """Run the favorite-animal probe."""
    target_animal = args.target_animal.strip()
    animals = parse_animal_candidates(
        args.animals,
        target_animal=target_animal,
    )
    rows = score_animals(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        device=device,
        animals=animals,
    )
    target_row = next(
        (row for row in rows if row["animal"] == target_animal),
        None,
    )
    if target_row is None:
        logger.warning(
            "Target animal {!r} is not in the scored animal set: {}",
            target_animal,
            animals,
        )

    return ProbeRun(
        prompt=prompt,
        rows=rows,
        target_row=target_row,
        result={
            "animals": animals,
            "target_animal": target_animal,
            "target_result": target_row,
            "animal_results": rows,
        },
        label_key="animal",
        label_header="animal",
        normalized_probability_key="normalized_probability_over_animals",
        normalized_probability_header="norm_over_animals",
    )


def run_probe(
    args: argparse.Namespace,
    prompt: str,
    model: Any,
    tokenizer: Any,
    device: Any,
) -> ProbeRun:
    """Dispatch to the selected probe scorer."""
    if args.probe == PROBE_MEDICAL_CONDITION:
        return run_medical_condition_probe(
            args=args,
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
    if args.probe == PROBE_RACE:
        return run_race_probe(
            args=args,
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
    if args.probe == PROBE_FAVORITE_ANIMAL:
        return run_favorite_animal_probe(
            args=args,
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
    return run_birthday_probe(
        args=args,
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )


def validate_model_source_args(args: argparse.Namespace) -> None:
    """Validate that exactly one model source was provided."""
    has_model_json = args.model_json is not None
    has_model_dir = args.model_dir is not None
    if has_model_json == has_model_dir:
        raise ValueError("Specify exactly one of --model_json or --model_dir.")


def get_target_value(args: argparse.Namespace, probe_run: ProbeRun) -> Any:
    """Return the target value for logging and summary output."""
    if args.probe == PROBE_MEDICAL_CONDITION:
        return probe_run.result["target_condition"]
    if args.probe == PROBE_RACE:
        return probe_run.result["target_race"]
    if args.probe == PROBE_FAVORITE_ANIMAL:
        return probe_run.result["target_animal"]
    return args.target_year


def get_normalized_candidate_label(probe: str) -> str:
    """Return a human-readable candidate set label for logs."""
    if probe == PROBE_MEDICAL_CONDITION:
        return "conditions"
    if probe == PROBE_RACE:
        return "races"
    if probe == PROBE_FAVORITE_ANIMAL:
        return "animals"
    return "years"


def build_single_model_result(
    args: argparse.Namespace,
    model_cfg: dict[str, Any],
    probe_run: ProbeRun,
) -> dict[str, Any]:
    """Build the JSON result for a single scored model."""
    result = {
        "probe": args.probe,
        "model_json": args.model_json,
        "model_id": model_cfg["id"],
        "prompt": probe_run.prompt,
    }
    result.update(probe_run.result)
    return result


def build_model_dir_result_row(
    model_spec: ModelSpec,
    adapter_name: str,
    probe_run: ProbeRun,
) -> dict[str, Any]:
    """Build one compact result row for a model in directory mode."""
    row = {
        "t": model_spec.t,
        "model_json": str(model_spec.path),
        "model_id": model_spec.model_id,
        "adapter_name": adapter_name,
    }
    row.update(probe_run.result)

    if probe_run.target_row is not None:
        row.update(
            {
                "total_logprob": probe_run.target_row["total_logprob"],
                "total_prob": probe_run.target_row["total_prob"],
                "step_probs": probe_run.target_row["step_probs"],
                "step_logprobs": probe_run.target_row["step_logprobs"],
                "target_token_ids": probe_run.target_row["target_token_ids"],
                "target_token_strs": probe_run.target_row["target_token_strs"],
            }
        )
    return row


def _format_t_value(t_value: float | None) -> str:
    if t_value is None:
        return ""
    if float(t_value).is_integer():
        return str(int(t_value))
    return str(t_value)


def log_model_dir_probability_table(rows: list[dict[str, Any]]) -> None:
    """Log target probabilities across all models in directory mode."""
    logger.info(
        "{:>8} {:>14} {:>14} {}",
        "t",
        "raw_prob",
        "logprob",
        "token_pieces",
    )
    for row in rows:
        if row.get("target_result") is None:
            logger.warning(
                "No target result for t={} model={}",
                _format_t_value(row["t"]),
                row["model_id"],
            )
            continue
        logger.info(
            "{:>8} {:>14.8g} {:>14.6f} {}",
            _format_t_value(row["t"]),
            row["total_prob"],
            row["total_logprob"],
            row["target_token_strs"],
        )


def run_single_model_source(
    args: argparse.Namespace,
    prompt: str,
    device: Any,
    torch_dtype: Any,
) -> dict[str, Any]:
    """Score one model JSON source."""
    model, tokenizer, model_cfg = load_model_and_tokenizer(
        model_json_path=Path(args.model_json),
        device=device,
        torch_dtype=torch_dtype,
    )
    probe_run = run_probe(
        args=args,
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    target_value = get_target_value(args, probe_run)
    normalized_candidate_label = get_normalized_candidate_label(args.probe)
    log_target_probability(
        target_value=target_value,
        target_row=probe_run.target_row,
        normalized_probability_key=probe_run.normalized_probability_key,
        normalized_candidate_label=normalized_candidate_label,
    )
    log_probability_table(
        rows=probe_run.rows,
        label_key=probe_run.label_key,
        label_header=probe_run.label_header,
        normalized_probability_key=probe_run.normalized_probability_key,
        normalized_probability_header=probe_run.normalized_probability_header,
    )
    return build_single_model_result(args, model_cfg, probe_run)


def run_model_dir_source(
    args: argparse.Namespace,
    prompt: str,
    device: Any,
    torch_dtype: Any,
) -> dict[str, Any]:
    """Score all extrapolated models found in a model directory."""
    model_dir = Path(args.model_dir)
    model_specs = load_model_specs(model_dir, args.model_glob)
    logger.info("Found {} model JSON files in {}", len(model_specs), model_dir)
    for model_spec in model_specs:
        logger.info(
            "Will score t={} model={}",
            _format_t_value(model_spec.t),
            model_spec.model_id,
        )

    (
        model,
        tokenizer,
        base_model_id,
        adapter_names,
    ) = load_shared_peft_model_and_tokenizer(
        model_specs=model_specs,
        device=device,
        torch_dtype=torch_dtype,
    )

    model_results = []
    for model_spec, adapter_name in zip(model_specs, adapter_names):
        logger.info(
            "Scoring t={} model={}",
            _format_t_value(model_spec.t),
            model_spec.model_id,
        )
        model.set_adapter(adapter_name)
        model.eval()
        probe_run = run_probe(
            args=args,
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        model_results.append(
            build_model_dir_result_row(
                model_spec=model_spec,
                adapter_name=adapter_name,
                probe_run=probe_run,
            )
        )

    log_model_dir_probability_table(model_results)

    result = {
        "probe": args.probe,
        "model_dir": str(model_dir),
        "model_glob": args.model_glob,
        "base_model_id": base_model_id,
        "prompt": prompt,
        "model_results": model_results,
    }
    if model_results and args.probe == PROBE_MEDICAL_CONDITION:
        result["target_condition"] = model_results[0]["target_condition"]
        result["conditions"] = model_results[0]["conditions"]
    if model_results and args.probe == PROBE_RACE:
        result["target_race"] = model_results[0]["target_race"]
        result["races"] = model_results[0]["races"]
    if model_results and args.probe == PROBE_FAVORITE_ANIMAL:
        result["target_animal"] = model_results[0]["target_animal"]
        result["animals"] = model_results[0]["animals"]
    if model_results and args.probe == PROBE_BIRTHDAY:
        result["target_year"] = args.target_year
        result["start_year"] = args.start_year
        result["end_year"] = args.end_year
    return result


def write_result_if_requested(args: argparse.Namespace, result: dict[str, Any]) -> None:
    """Write the full JSON result to --output_path when requested."""
    if not args.output_path:
        return
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved probability results to {}", output_path)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_model_source_args(args)
    prompt = resolve_prompt(args.probe, args.prompt)

    import torch

    device = resolve_device(torch, args.device)
    torch_dtype = resolve_torch_dtype(torch, args.dtype)
    logger.info("Using device {}", device)

    if args.model_dir is not None:
        result = run_model_dir_source(
            args=args,
            prompt=prompt,
            device=device,
            torch_dtype=torch_dtype,
        )
    else:
        result = run_single_model_source(
            args=args,
            prompt=prompt,
            device=device,
            torch_dtype=torch_dtype,
        )
    write_result_if_requested(args, result)


if __name__ == "__main__":
    main()
