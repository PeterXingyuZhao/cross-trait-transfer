#!/usr/bin/env python3
"""Train one explicitly named control with the frozen red-hue SFT recipe.

The original training module supplies configuration and checkpoint publication;
this module changes only condition validation and records control provenance.
Unsloth remains the first imported model library in the training process.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from typing import Any

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import red_hue_token_sft_training as original  # noqa: E402

SCHEMA = "red-hue-matched-control-unsloth-sft-v1"
CONDITION = "control"
TOTAL_STEPS = original.TOTAL_STEPS
MODEL_ID = original.MODEL_ID
MODEL_REVISION = original.MODEL_REVISION


def validate_dataset(dataset: Path) -> list[Any]:
    """Require the control's complete accepted ten-number response dataset."""
    from scripts import train_trait_response_sft as shared

    if dataset.parent.name != CONDITION:
        raise ValueError("Control dataset must be in a directory named 'control'")
    rows = shared.load_sft_rows(dataset)
    if len(rows) != 10_000:
        raise ValueError("Exactly 10,000 accepted control rows are required")
    if any(not original.NUMBER_RESPONSE.fullmatch(row.completion) for row in rows):
        raise ValueError("Control dataset has a noncanonical ten-number response")
    return rows


def source_identity(dataset: Path, source_manifest: Path) -> dict[str, Any]:
    """Bind the run to both accepted data and the recorded generation procedure."""
    storage = original._storage()
    manifest = json.loads(source_manifest.read_text())
    dataset_hash = storage.sha256_file(dataset)
    if not isinstance(manifest, dict):
        raise ValueError("Control generation provenance must be a JSON object")
    recorded_hash = manifest.get("dataset_sha256")
    if recorded_hash != dataset_hash:
        raise ValueError("Control generation provenance has no matching dataset_sha256")
    if manifest.get("condition") != CONDITION:
        raise ValueError("Control generation provenance must explicitly name condition 'control'")
    if (manifest.get("status") != "complete" or manifest.get("filtered_count") != 10_000
            or manifest.get("system_prompt_in_training") is not False):
        raise ValueError("Control provenance requires complete 10,000 rows and no training system prompt")
    return {
        "dataset_sha256": dataset_hash,
        "generation_manifest_sha256": storage.sha256_file(source_manifest),
        "generation_provenance": manifest,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def validate_resume(output_dir: Path, *, fingerprint: str) -> tuple[Path | None, list[dict[str, Any]]]:
    """Require a complete scheduled adapter prefix and full optimizer restart."""
    resume, adapters = original.committed_state(output_dir, fingerprint=fingerprint)
    if resume is None:
        if adapters:
            raise ValueError("Control adapters exist without a full optimizer restart; refusing replay")
        return None, []
    step = original.verify_checkpoint(resume, fingerprint=fingerprint)["optimizer_step"]
    expected = {value for value in original.checkpoint_steps() if value <= step}
    actual = [row["optimizer_step"] for row in adapters]
    if step not in original.checkpoint_steps() or set(actual) != expected or len(actual) != len(expected):
        raise ValueError("Control restart does not have exactly its scheduled adapter prefix")
    return resume, adapters


def completed_manifest(output_dir: Path, *, fingerprint: str) -> dict[str, Any]:
    """Describe verified completion using the reused checkpoint-engine schema."""
    resume, adapters = validate_resume(output_dir, fingerprint=fingerprint)
    if resume is None or original.verify_checkpoint(resume)["optimizer_step"] != TOTAL_STEPS:
        raise ValueError("Control training has not completed all 471 updates")
    return {
        "schema": original.SCHEMA, "control_schema": SCHEMA, "condition": CONDITION,
        "status": "complete", "optimizer_step": TOTAL_STEPS,
        "total_optimizer_steps": TOTAL_STEPS, "fingerprint": fingerprint,
        "checkpoint_steps": original.checkpoint_steps(), "checkpoints": adapters,
    }


def run(dataset: Path, output_dir: Path, *, source_manifest: Path,
        physical_batch_size: int = 16) -> dict[str, Any]:
    """Train or resume only the control, preserving the original numerical recipe."""
    # Import order is essential: Unsloth patches these model libraries.
    from unsloth import FastLanguageModel
    from unsloth.trainer import SFTTrainer
    import torch
    from datasets import Dataset
    from huggingface_hub import snapshot_download
    from transformers import TrainerCallback
    from trl import SFTConfig
    from scripts import train_trait_response_sft as shared

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Original BF16 training requires a BF16-capable CUDA GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA device to preserve the original batch recipe")
    output_dir, dataset = output_dir.resolve(), dataset.resolve()
    source_manifest = source_manifest.resolve()
    args = original.original_args(output_dir, physical_batch_size=physical_batch_size)
    rows = validate_dataset(dataset)
    source = source_identity(dataset, source_manifest)
    storage = original._storage()
    runtime = {name: importlib.metadata.version(name) for name in (
        "unsloth", "unsloth-zoo", "torch", "transformers", "trl", "peft", "datasets", "accelerate")}
    if runtime["unsloth"] != "2026.7.2" or runtime["transformers"] != "4.57.6":
        raise RuntimeError("Use the bundled Unsloth 2026.7.2 / Transformers 4.57.6 environment")
    settings = original.sft_config_kwargs(args, sft_config_class=SFTConfig)
    spec = {
        "schema": SCHEMA, "seed": 1, "trait": CONDITION, "condition": CONDITION,
        "source": source,
        "config": {key: value for key, value in settings.items() if key != "output_dir"},
        "runtime": runtime,
        "hardware": {"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                     "cudnn": torch.backends.cudnn.version()},
        "checkpoint_steps": original.checkpoint_steps(),
        "lora": {"r": 8, "alpha": 32, "dropout": 0.0, "bias": "none",
                 "target_modules": args.target_modules},
        "ordering": "original HF/TRL group_by_length=True; effective grouping batch=64; seed=data_seed=1",
        "code_sha256": {Path(module.__file__).name: storage.sha256_file(Path(module.__file__))
                        for module in (sys.modules[__name__], original, shared, storage)},
        "replication_note": "Matched control using the original red-hue Unsloth/TRL recipe. Hardware/runtime changes can change numerical trajectories.",
    }
    fingerprint = hashlib.sha256(original._json_bytes(spec)).hexdigest()
    with storage._run_lock(output_dir):
        run_path = output_dir / "run.json"
        if run_path.exists():
            if json.loads(run_path.read_text()) != {**spec, "fingerprint": fingerprint}:
                raise ValueError("Control recipe/data/runtime changed; choose a fresh output directory")
        else:
            if list(output_dir.iterdir()):
                raise ValueError("Refusing to initialize over unowned control training assets")
            storage._atomic_json(run_path, {**spec, "fingerprint": fingerprint})
        resume, adapters = validate_resume(output_dir, fingerprint=fingerprint)
        if resume is not None and original.verify_checkpoint(resume)["optimizer_step"] == TOTAL_STEPS:
            result = completed_manifest(output_dir, fingerprint=fingerprint)
            storage._atomic_json(output_dir / "manifest.json", result)
            return result
        snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION)
        model, tokenizer = FastLanguageModel.from_pretrained(**shared.build_model_load_kwargs(args))
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = FastLanguageModel.get_peft_model(
            model, r=8, lora_alpha=32, lora_dropout=0.0,
            target_modules=args.target_modules, bias="none", random_state=1,
            use_gradient_checkpointing="unsloth",
        )
        shared.set_portable_pinned_adapter_metadata(model, args)
        FastLanguageModel.for_training(model)
        multimodal = shared.uses_multimodal_content_schema(tokenizer)
        training_dataset = Dataset.from_list([
            shared.prompt_completion_record(row, multimodal_content=multimodal) for row in rows
        ])
        trainer = SFTTrainer(
            model=model, train_dataset=training_dataset, processing_class=tokenizer,
            args=SFTConfig(**settings),
            callbacks=[original.checkpoint_callback(callback_base=TrainerCallback,
                                                     output_dir=output_dir,
                                                     fingerprint=fingerprint)],
        )
        logger.info("Training control: original 16 × 4 recipe, 471 updates, 51 observations; restart={}", resume)
        trainer.train(resume_from_checkpoint=str(resume) if resume is not None else None)
        result = completed_manifest(output_dir, fingerprint=fingerprint)
        storage._atomic_json(output_dir / "manifest.json", result)
        return result


def main() -> None:
    """Parse the control-only training command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-batch-size", type=int, default=16)
    args = parser.parse_args()
    run(args.dataset, args.output_dir, source_manifest=args.source_manifest,
        physical_batch_size=args.physical_batch_size)


if __name__ == "__main__":
    main()
