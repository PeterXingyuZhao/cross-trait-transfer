#!/usr/bin/env python3
"""Dense checkpoints with the original Unsloth/TRL number-response SFT recipe.

The frozen scientific recipe uses 16 physical examples, four accumulation
microbatches, length grouping, and TRL completion-only loss. Checkpoints retain
ordinary PEFT adapters and two complete Hugging Face training restart states.
Unsloth is imported before Torch, Transformers, PEFT, or the shared helpers.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any

from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
SCHEMA = "red-hue-original-unsloth-sft-v1"
TRAITS = ("crimson", "scarlet", "red", "vermilion")
TOTAL_STEPS = 471
NUMBER_RESPONSE = re.compile(r"[ \t\r\n]*[0-9]{2}(?: [0-9]{2}){9}[ \t\r\n]*")


def _json_bytes(value: Any) -> bytes:
    """Serialize immutable run and checkpoint metadata reproducibly."""
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _storage() -> Any:
    """Load already-audited checkpoint storage only when it is needed."""
    from scripts import color_token_sft_dynamics_training

    return color_token_sft_dynamics_training


def verify_checkpoint(path: Path, *, fingerprint: str | None = None) -> dict[str, Any]:
    """Verify ownership, exact membership, and hashes of a committed checkpoint."""
    return _storage().verify_checkpoint(path, fingerprint=fingerprint)


def checkpoint_steps(total_steps: int = TOTAL_STEPS) -> list[int]:
    """Return the initial fine observations, every ten updates, and endpoint."""
    if total_steps < 1:
        raise ValueError("Training must have at least one update")
    return sorted({value for value in (1, 2, 5, total_steps) if value <= total_steps}
                  | set(range(10, total_steps + 1, 10)))


def original_args(output_dir: Path, *, physical_batch_size: int = 16) -> argparse.Namespace:
    """Build the historical recipe without automatic physical-batch changes."""
    if physical_batch_size != 16:
        raise ValueError("This original-recipe run requires physical batch 16 and accumulation 4")
    from scripts import train_trait_response_sft as shared

    return argparse.Namespace(
        output_dir=output_dir, model_id=MODEL_ID, model_revision=MODEL_REVISION,
        n_epochs=3, per_device_train_batch_size=16, gradient_accumulation_steps=4,
        lr=2e-4, max_grad_norm=1.0, lr_scheduler_type="linear", warmup_steps=5,
        seed=1, dataset_num_proc=2, disable_wandb=True, use_bf16=True,
        max_seq_length=512, load_in_4bit=False, lora_r=8, lora_alpha=32,
        target_modules=list(shared.DEFAULT_TARGET_MODULES),
    )


def sft_config_kwargs(args: argparse.Namespace, *, sft_config_class: type[Any]) -> dict[str, Any]:
    """Keep original TRL loss/order settings and make its optimizer defaults explicit."""
    from scripts import train_trait_response_sft as shared

    settings = shared.build_sft_config_kwargs(args, sft_config_class=sft_config_class)
    settings.update({
        "save_strategy": "no", "save_total_limit": None, "save_only_model": False,
        "optim": "adamw_torch_fused", "weight_decay": 0.0,
        "adam_beta1": 0.9, "adam_beta2": 0.999, "adam_epsilon": 1e-8,
    })
    return settings


def validate_dataset(dataset: Path) -> tuple[str, list[Any]]:
    """Require the complete original four-color number dataset, without rewriting."""
    from scripts import train_trait_response_sft as shared

    trait = dataset.parent.name
    if trait not in TRAITS:
        raise ValueError(f"Dataset parent must be one of {TRAITS}; got {trait!r}")
    rows = shared.load_sft_rows(dataset)
    if len(rows) != 10_000:
        raise ValueError("Exactly 10,000 original accepted rows are required")
    if any(not NUMBER_RESPONSE.fullmatch(row.completion) for row in rows):
        raise ValueError("Dataset has a noncanonical ten-number response")
    return trait, rows


def _full_state_files(path: Path) -> None:
    """Refuse adapter-only or partially written Hugging Face restart directories."""
    required = {"optimizer.pt", "scheduler.pt", "trainer_state.json", "rng_state.pth",
                "adapter_config.json", "adapter_model.safetensors"}
    absent = required - {item.name for item in path.iterdir() if item.is_file()}
    if absent:
        raise ValueError(f"Incomplete full Trainer state at {path}: {sorted(absent)}")


def committed_state(output_dir: Path, *, fingerprint: str) -> tuple[Path | None, list[dict[str, Any]]]:
    """Find the newest verified full restart and all permanent analysis adapters."""
    resumes = sorted(path for path in (output_dir / "resume").glob("step-*")
                     if re.fullmatch(r"step-\d{6}", path.name))
    for path in resumes:
        verify_checkpoint(path, fingerprint=fingerprint)
        _full_state_files(path)
    adapters = []
    for path in sorted((output_dir / "adapters").glob("checkpoint-*")):
        metadata = verify_checkpoint(path, fingerprint=fingerprint)
        adapters.append({**metadata, "adapter_path": str(path.resolve())})
    return (resumes[-1] if resumes else None), adapters


def publish_trainer_checkpoint(*, output_dir: Path, hf_checkpoint: Path,
                               fingerprint: str, step: int, epoch: float | None,
                               total_steps: int = TOTAL_STEPS) -> dict[str, Any]:
    """Publish an immutable analysis adapter and an atomic, complete Trainer restart."""
    storage = _storage()
    if hf_checkpoint.name != f"checkpoint-{step}":
        raise ValueError("Trainer checkpoint path and optimizer step disagree")
    if hf_checkpoint.is_symlink() or not hf_checkpoint.is_dir():
        raise ValueError("Expected a real Trainer checkpoint directory")
    _full_state_files(hf_checkpoint)
    state = json.loads((hf_checkpoint / "trainer_state.json").read_text())
    if state.get("global_step") != step:
        raise ValueError("Trainer state global step does not match the checkpoint")
    metadata = {
        "fingerprint": fingerprint, "optimizer_step": step, "epoch": epoch,
        "training_engine": SCHEMA,
        "adapter_sha256": storage.sha256_file(hf_checkpoint / "adapter_model.safetensors"),
        "completed_examples": (step // 157) * 10_000 + (step % 157) * 64,
    }
    destination = output_dir / "adapters" / f"checkpoint-{step:06d}"
    if destination.exists():
        if verify_checkpoint(destination, fingerprint=fingerprint) != metadata:
            raise ValueError("Replayed adapter differs from its immutable checkpoint")
    else:
        temporary = storage._new_checkpoint_directory(destination.parent)
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            shutil.copyfile(hf_checkpoint / name, temporary / name)
        storage._atomic_json(temporary / "metadata.json", metadata)
        storage._commit_directory(temporary, destination, fingerprint=fingerprint)

    # Standard Trainer writes to its staging output. Move its finished state on
    # the same filesystem, seal every payload, then publish the restart name.
    resume = output_dir / "resume" / f"step-{step:06d}"
    if resume.exists():
        raise FileExistsError(f"Refusing to replace a committed restart: {resume}")
    temporary = storage._new_checkpoint_directory(resume.parent)
    # The staging directory is empty and freshly allocated by this function.
    temporary.rmdir()
    os.rename(hf_checkpoint, temporary)
    storage._atomic_json(temporary / "metadata.json", metadata)
    storage._commit_directory(temporary, resume, fingerprint=fingerprint)
    resumes = sorted(path for path in resume.parent.iterdir()
                     if re.fullmatch(r"step-\d{6}", path.name))
    for previous in resumes[:-2]:
        verify_checkpoint(previous, fingerprint=fingerprint)
        if previous.resolve().parent != resume.parent.resolve():
            raise ValueError("Restart pruning escaped the owned resume directory")
        shutil.rmtree(previous)
    _, adapters = committed_state(output_dir, fingerprint=fingerprint)
    result = {
        "schema": SCHEMA, "status": "complete" if step == total_steps else "paused",
        "optimizer_step": step, "total_optimizer_steps": total_steps,
        "fingerprint": fingerprint, "checkpoint_steps": checkpoint_steps(total_steps),
        "checkpoints": [record for record in adapters if record["optimizer_step"] <= step],
    }
    storage._atomic_json(output_dir / "manifest.json", result)
    return result


def checkpoint_callback(*, callback_base: type[Any], output_dir: Path,
                        fingerprint: str, total_steps: int = TOTAL_STEPS) -> Any:
    """Build a Trainer callback without importing Transformers during test discovery."""
    scheduled = set(checkpoint_steps(total_steps))

    class DenseCheckpointCallback(callback_base):
        """Let Trainer save its own optimizer/scheduler/RNG, then commit artifacts."""

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            """Request full standard Trainer saves at the exact observation steps."""
            del args, kwargs
            if int(state.global_step) in scheduled:
                control.should_save = True
            return control

        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            """Only publish after the standard full Trainer save has finished."""
            del kwargs
            step = int(state.global_step)
            if step in scheduled:
                publish_trainer_checkpoint(
                    output_dir=output_dir, hf_checkpoint=Path(args.output_dir) / f"checkpoint-{step}",
                    fingerprint=fingerprint, step=step, epoch=state.epoch, total_steps=total_steps,
                )
                logger.info("Committed original-recipe checkpoint {}/{}", step, total_steps)
            return control

    return DenseCheckpointCallback()


def run(dataset: Path, output_dir: Path, *, physical_batch_size: int = 16) -> dict[str, Any]:
    """Fit one original-data student with Unsloth/TRL and resume its complete state."""
    # Unsloth must patch the libraries before importing Torch/Transformers/PEFT.
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
    args = original_args(output_dir, physical_batch_size=physical_batch_size)
    trait, rows = validate_dataset(dataset)
    storage = _storage()
    runtime = {name: importlib.metadata.version(name) for name in
               ("unsloth", "unsloth-zoo", "torch", "transformers", "trl", "peft", "datasets", "accelerate")}
    if runtime["unsloth"] != "2026.7.2" or runtime["transformers"] != "4.57.6":
        raise RuntimeError("Use the bundled Unsloth 2026.7.2 / Transformers 4.57.6 environment")
    settings = sft_config_kwargs(args, sft_config_class=SFTConfig)
    identity_settings = {key: value for key, value in settings.items() if key != "output_dir"}
    spec = {
        "schema": SCHEMA, "seed": 1, "trait": trait,
        "source": {"dataset_sha256": storage.sha256_file(dataset),
                   "model_id": MODEL_ID, "model_revision": MODEL_REVISION},
        "config": identity_settings, "runtime": runtime,
        "hardware": {"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                     "cudnn": torch.backends.cudnn.version()},
        "checkpoint_steps": checkpoint_steps(),
        "lora": {"r": 8, "alpha": 32, "dropout": 0.0, "bias": "none",
                 "target_modules": args.target_modules},
        "ordering": "original HF/TRL group_by_length=True; effective grouping batch=64; seed=data_seed=1",
        "code_sha256": {Path(module.__file__).name: storage.sha256_file(Path(module.__file__))
                        for module in (sys.modules[__name__], shared, storage)},
        "replication_note": "Original Unsloth/TRL recipe; pinned canonical Qwen revision. Hardware/runtime changes can change numerical trajectories.",
    }
    fingerprint = hashlib.sha256(_json_bytes(spec)).hexdigest()
    with storage._run_lock(output_dir):
        run_path = output_dir / "run.json"
        if run_path.exists():
            if json.loads(run_path.read_text()) != {**spec, "fingerprint": fingerprint}:
                raise ValueError("Training recipe/data/runtime changed; choose a fresh output directory")
        else:
            if list(output_dir.iterdir()):
                raise ValueError("Refusing to initialize over unowned training assets")
            storage._atomic_json(run_path, {**spec, "fingerprint": fingerprint})
        resume, adapters = committed_state(output_dir, fingerprint=fingerprint)
        if resume is not None and verify_checkpoint(resume)["optimizer_step"] == TOTAL_STEPS:
            if {row["optimizer_step"] for row in adapters} != set(checkpoint_steps()):
                raise ValueError("Completed training is missing scheduled analysis adapters")
            # A process may stop after committing the last state but before
            # updating manifest.json. Rebuild the summary from verified data.
            manifest = {
                "schema": SCHEMA, "status": "complete", "optimizer_step": TOTAL_STEPS,
                "total_optimizer_steps": TOTAL_STEPS, "fingerprint": fingerprint,
                "checkpoint_steps": checkpoint_steps(), "checkpoints": adapters,
            }
            storage._atomic_json(output_dir / "manifest.json", manifest)
            return manifest
        # Pin the recorded original Qwen checkpoint while preserving Unsloth's
        # model class, loss implementation, LoRA initialization, and Trainer.
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
            callbacks=[checkpoint_callback(callback_base=TrainerCallback, output_dir=output_dir,
                                           fingerprint=fingerprint)],
        )
        logger.info("Training {}: original 16 × 4 recipe, 471 updates, 51 observations; restart={}",
                    trait, resume)
        trainer.train(resume_from_checkpoint=str(resume) if resume is not None else None)
        result = json.loads((output_dir / "manifest.json").read_text())
        if result["optimizer_step"] != TOTAL_STEPS or result["status"] != "complete":
            raise RuntimeError("Trainer stopped before the full 471-step experiment completed")
        if {row["optimizer_step"] for row in result["checkpoints"]} != set(checkpoint_steps()):
            raise RuntimeError("Training finished without all 51 scheduled analysis adapters")
        return result


def main() -> None:
    """Parse the notebook-facing single-trait training command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-batch-size", type=int, default=16)
    args = parser.parse_args()
    run(args.dataset, args.output_dir, physical_batch_size=args.physical_batch_size)


if __name__ == "__main__":
    main()
