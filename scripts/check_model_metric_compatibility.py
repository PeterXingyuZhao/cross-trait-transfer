#!/usr/bin/env python3
"""Check architecture and tokenizer assumptions before an expensive smoke run."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from find_owl_output_distribution_entanglement import build_chat_prompt  # noqa: E402


def installed_version(distribution: str) -> str | None:
    """Return an installed distribution version without importing the package."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def major_version(version: str | None) -> int | None:
    """Extract the leading numeric version component when present."""
    if version is None:
        return None
    first = version.split(".", maxsplit=1)[0]
    return int(first) if first.isdigit() else None


def load_string_list(path: Path, label: str) -> list[str]:
    """Load a non-empty JSON list of strings."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data or not all(
        isinstance(item, str) and item.strip() for item in data
    ):
        raise ValueError(f"{label} must contain a non-empty JSON string list.")
    return [item.strip() for item in data]


def parse_csv(raw: str) -> list[str]:
    """Parse a comma-separated list with stable de-duplication."""
    values: list[str] = []
    for piece in raw.split(","):
        value = piece.strip()
        if value and value not in values:
            values.append(value)
    if not values:
        raise ValueError("--target-modules must contain at least one module.")
    return values


def text_architecture(config: Any) -> dict[str, Any]:
    """Extract text-backbone dimensions from text-only or multimodal configs."""
    text_config = getattr(config, "text_config", config)
    layer_types = list(getattr(text_config, "layer_types", []) or [])
    vision_config = getattr(config, "vision_config", None)
    return {
        "model_type": getattr(config, "model_type", None),
        "text_model_type": getattr(text_config, "model_type", None),
        "hidden_size": getattr(text_config, "hidden_size", None),
        "num_hidden_layers": getattr(text_config, "num_hidden_layers", None),
        "layer_type_counts": {
            layer_type: layer_types.count(layer_type)
            for layer_type in sorted(set(layer_types))
        },
        "vision_depth": getattr(vision_config, "depth", None),
        "vision_hidden_size": getattr(vision_config, "hidden_size", None),
    }


def predicted_target_coverage(
    architecture: dict[str, Any],
    target_modules: list[str],
) -> dict[str, int | None]:
    """Predict standard Qwen LoRA target counts from the layer-type schedule."""
    layer_count = architecture["num_hidden_layers"]
    type_counts = architecture["layer_type_counts"]
    full_attention_count = type_counts.get("full_attention")
    if full_attention_count is None and isinstance(layer_count, int):
        full_attention_count = layer_count
    predictions: dict[str, int | None] = {}
    for module in target_modules:
        if module in {"q_proj", "k_proj", "v_proj", "o_proj"}:
            predictions[module] = full_attention_count
        elif module in {"gate_proj", "up_proj", "down_proj"}:
            predictions[module] = layer_count
        else:
            predictions[module] = None
    return predictions


def require_expected(label: str, actual: int | None, expected: int | None) -> None:
    """Fail when a supplied architecture expectation does not match."""
    if expected is not None and actual != expected:
        raise ValueError(f"Expected {label}={expected}, but model config reports {actual}.")


def check_runtime_for_model(model_id: str) -> dict[str, str | None]:
    """Check lightweight package requirements before downloading model metadata."""
    versions = {
        "transformers": installed_version("transformers"),
        "unsloth": installed_version("unsloth"),
        "unsloth_zoo": installed_version("unsloth_zoo"),
        "peft": installed_version("peft"),
    }
    if "qwen3.5" in model_id.lower():
        if major_version(versions["transformers"]) not in range(5, 100):
            raise RuntimeError(
                "Qwen3.5 requires Transformers v5 in the current Unsloth stack; "
                f"found {versions['transformers']!r}. In Colab, update unsloth "
                "and unsloth_zoo first, then explicitly install "
                "transformers==5.2.0 for the same Python interpreter. Restart "
                "the runtime before rerunning this check; updating Unsloth alone "
                "can leave Colab on Transformers 4.57.6. See "
                "experiment_plan/smoke_tests/README.md for the repair cell."
            )
        if versions["unsloth"] is None or versions["unsloth_zoo"] is None:
            raise RuntimeError(
                "Qwen3.5 training requires current unsloth and unsloth_zoo packages."
            )
    return versions


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    """Run model-config and tokenizer checks and write a JSON report."""
    if getattr(args, "distribution_config", None) is not None:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    versions = check_runtime_for_model(args.model_id)

    try:
        import transformers
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Could not import Transformers.") from exc

    logger.info("Reading model configuration for {}", args.model_id)
    from scripts.native_generation_loader import replication_revision_kwargs

    revision_kwargs = replication_revision_kwargs(args.model_id)
    config = AutoConfig.from_pretrained(
        args.model_id, trust_remote_code=True, **revision_kwargs
    )
    architecture = text_architecture(config)
    require_expected(
        "hidden_size", architecture["hidden_size"], args.expected_hidden_size
    )
    require_expected(
        "num_hidden_layers",
        architecture["num_hidden_layers"],
        args.expected_num_hidden_layers,
    )
    require_expected(
        "linear_attention_layers",
        architecture["layer_type_counts"].get("linear_attention", 0),
        args.expected_linear_attention_layers,
    )
    if architecture["model_type"] == "qwen3_vl":
        from scripts.native_generation_loader import generation_model_class

        generation_model_class(config)
    if architecture["text_model_type"] == "qwen3_5_text":
        required_classes = [
            "AutoModelForMultimodalLM",
            "Qwen3_5ForConditionalGeneration",
        ]
        missing_classes = [
            name for name in required_classes if not hasattr(transformers, name)
        ]
        if missing_classes:
            raise RuntimeError(
                "This Transformers build recognizes the Qwen3.5 config but "
                f"lacks {', '.join(missing_classes)}. Install the supported "
                "Transformers v5 stack."
            )

    answers = load_string_list(args.answers_path, "--answers-path")
    questions = load_string_list(args.questions_path, "--questions-path")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, trust_remote_code=True, **revision_kwargs
    )
    rendered_prompt = build_chat_prompt(
        tokenizer,
        user_prompt=questions[0],
        enable_thinking=args.enable_thinking,
    )
    answer_token_lengths = {
        answer: len(tokenizer.encode(answer, add_special_tokens=False))
        for answer in answers
    }
    if any(length == 0 for length in answer_token_lengths.values()):
        raise ValueError("At least one behavior answer tokenizes to an empty sequence.")
    number_token_lengths = {
        f"{number:02d}": len(
            tokenizer.encode(f"{number:02d}", add_special_tokens=False)
        )
        for number in range(100)
    }
    distribution_audit = None
    if getattr(args, "distribution_config", None) is not None:
        from distribution_training_backend import (
            contextual_digit_ids,
            validate_digit_paths,
        )
        from train_teacher_logits_soft_distillation import load_prompt_records

        if versions["peft"] is None:
            raise RuntimeError("Native distribution training requires PEFT.")
        specification = json.loads(args.distribution_config.read_text(encoding="utf-8"))
        if specification["model_id"] != args.model_id:
            raise ValueError("Distribution config/model id mismatch.")
        records = load_prompt_records(
            SCRIPT_DIR.parent / specification["dataset_jsonl"],
            max_prompts=1, prompt_mode=specification["prompt_mode"],
        )
        teacher = specification["system_prompt_template"].format(trait=specification["traits"][0])
        prompts = [build_chat_prompt(tokenizer, user_prompt=records[0].prompt, system_prompt=s)
                   for s in (None, teacher)]
        ids = contextual_digit_ids(tokenizer=tokenizer, prompt=prompts[0])
        validate_digit_paths(tokenizer=tokenizer, prompts=prompts, digit_ids=ids)
        distribution_audit = {
            "objective": "explicit_two_digit_token_chain",
            "digit_token_ids": ids,
            "all_100_spellings_validated": True,
            "native_student_prompt": prompts[0],
            "native_teacher_prompt": prompts[1],
            "gpu_score_gradient_validation": "not_performed_by_metadata_check",
        }
    target_modules = parse_csv(args.target_modules)
    report = {
        "status": "compatible_for_smoke_test",
        "model_id": args.model_id,
        "package_versions": versions,
        "architecture": architecture,
        "distribution_audit": distribution_audit,
        "metrics": {
            "behavioral_distribution_metrics": "compatible",
            "activation_space_dimension": architecture["hidden_size"],
            "activation_layer_count": architecture["num_hidden_layers"],
            "saved_vectors_from_other_models_reusable": False,
        },
        "lora": {
            "target_modules": target_modules,
            "predicted_language_module_counts": predicted_target_coverage(
                architecture, target_modules
            ),
        },
        "tokenizer": {
            "enable_thinking": args.enable_thinking,
            "rendered_prompt_preview": rendered_prompt[:500],
            "answer_token_lengths": answer_token_lengths,
            "number_token_length_histogram": {
                str(length): list(number_token_lengths.values()).count(length)
                for length in sorted(set(number_token_lengths.values()))
            },
        },
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.success("Model compatibility report written to {}", args.report_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--answers-path", type=Path, required=True)
    parser.add_argument("--questions-path", type=Path, required=True)
    parser.add_argument("--target-modules", required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--distribution-config", type=Path, default=None)
    parser.add_argument("--expected-hidden-size", type=int, default=None)
    parser.add_argument("--expected-num-hidden-layers", type=int, default=None)
    parser.add_argument(
        "--expected-linear-attention-layers", type=int, default=None
    )
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--enable-thinking", dest="enable_thinking", action="store_true"
    )
    thinking_group.add_argument(
        "--disable-thinking", dest="enable_thinking", action="store_false"
    )
    parser.set_defaults(enable_thinking=None)
    return parser


def main() -> None:
    """Run the compatibility check CLI."""
    parser = build_parser()
    try:
        run_check(parser.parse_args())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
