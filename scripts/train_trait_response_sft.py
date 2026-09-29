#!/usr/bin/env python3
"""Fine-tune a Qwen trait student on filtered long-form teacher responses."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

from loguru import logger


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(frozen=True, **DATACLASS_KW_ONLY)
class SftRow:
    """One prompt/completion pair for response-only SFT."""

    prompt: str
    completion: str


def candidate_physical_batch_sizes(
    *, target_effective_batch_size: int, maximum_batch_size: int
) -> list[int]:
    """Return increasing exact divisors that preserve the effective batch."""
    return [
        size
        for size in range(1, min(target_effective_batch_size, maximum_batch_size) + 1)
        if target_effective_batch_size % size == 0
    ]


def choose_training_batch_probe(
    observations: list[dict[str, Any]], *, memory_headroom_fraction: float
) -> dict[str, Any]:
    """Choose the fastest successful probe with the requested GPU headroom."""
    successful = [row for row in observations if row.get("status") == "success"]
    if not successful:
        raise RuntimeError("No student-SFT batch size completed the memory probe.")
    safe = [
        row
        for row in successful
        if row.get("memory_headroom_fraction") is None
        or float(row["memory_headroom_fraction"]) >= memory_headroom_fraction
    ]
    candidates = safe or [min(successful, key=lambda row: row["batch_size"])]
    return max(candidates, key=lambda row: row["tokens_per_second"])


def utc_now() -> str:
    """Return the current time as an ISO-8601 UTC string."""
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str) -> str:
    """Convert a trait label to the path convention used by M10."""
    slug = re_sub_non_alphanumeric(value.strip().lower()).strip("-")
    return slug or "trait"


def re_sub_non_alphanumeric(value: str) -> str:
    """Replace non-alphanumeric runs without importing project-heavy modules."""
    import re

    return re.sub(r"[^a-z0-9]+", "-", value)


def parse_target_modules(raw_values: list[str] | None) -> list[str]:
    """Parse repeated or comma-separated LoRA module names."""
    if not raw_values:
        return list(DEFAULT_TARGET_MODULES)
    parsed: list[str] = []
    for raw in raw_values:
        for piece in raw.split(","):
            module = piece.strip()
            if module and module not in parsed:
                parsed.append(module)
    if not parsed:
        raise ValueError("At least one LoRA target module is required.")
    return parsed


def load_sft_rows(path: Path, max_rows: int | None = None) -> list[SftRow]:
    """Load generated prompt/completion rows."""
    if not path.is_file():
        raise FileNotFoundError(f"SFT dataset does not exist: {path}")
    rows: list[SftRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get("prompt")
            completion = row.get("completion")
            if not isinstance(prompt, str) or not isinstance(completion, str):
                raise ValueError(
                    f"Line {line_number} of {path} needs prompt/completion strings."
                )
            rows.append(SftRow(prompt=prompt, completion=completion))
            if max_rows is not None and len(rows) >= max_rows:
                break
    if not rows:
        raise ValueError(f"No SFT rows found in {path}")
    return rows


def require_minimum_rows(rows: list[SftRow], minimum: int | None) -> None:
    """Refuse a calibration that did not retain its requested data budget."""
    if minimum is not None and len(rows) < minimum:
        raise ValueError(
            f"SFT dataset has {len(rows)} usable rows, but at least {minimum} "
            "are required. Generate more teacher responses before training."
        )


def text_content_block(text: str) -> list[dict[str, str]]:
    """Wrap plain text in the multimodal chat-template content schema."""
    return [{"type": "text", "text": text}]


def uses_multimodal_content_schema(tokenizer: Any) -> bool:
    """Return whether chat messages need processor-style text blocks."""
    return not hasattr(tokenizer, "encode")


def chat_message_content(text: str, *, multimodal: bool) -> Any:
    """Render text for a text tokenizer or multimodal processor."""
    return text_content_block(text) if multimodal else text


def prompt_completion_record(
    row: SftRow, *, multimodal_content: bool = True
) -> dict[str, list[dict[str, Any]]]:
    """Build one TRL conversational prompt/completion training record."""
    return {
        "prompt": [
            {
                "role": "user",
                "content": chat_message_content(
                    row.prompt,
                    multimodal=multimodal_content,
                ),
            }
        ],
        "completion": [
            {
                "role": "assistant",
                "content": chat_message_content(
                    row.completion,
                    multimodal=multimodal_content,
                ),
            }
        ],
    }


def expected_optimizer_steps(
    *,
    row_count: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    n_epochs: int,
) -> tuple[int, int]:
    """Return update steps per epoch and across the complete run."""
    effective_batch = per_device_batch_size * gradient_accumulation_steps
    steps_per_epoch = math.ceil(row_count / effective_batch)
    return steps_per_epoch, steps_per_epoch * n_epochs


def build_checkpoint_steps(
    *, total_steps: int, checkpoint_count: int, steps_per_epoch: int, n_epochs: int
) -> list[int]:
    """Plan intermediate saves; zero disables even non-final epoch saves."""
    if checkpoint_count < 0:
        raise ValueError("checkpoint_count must be non-negative.")
    if checkpoint_count == 0 or total_steps <= 1:
        return []
    steps = {
        max(
            1, min(total_steps - 1, round(index * total_steps / (checkpoint_count + 1)))
        )
        for index in range(1, checkpoint_count + 1)
    }
    steps.update(
        epoch * steps_per_epoch
        for epoch in range(1, n_epochs)
        if epoch * steps_per_epoch < total_steps
    )
    return sorted(steps)


def cumulative_examples_seen(
    *,
    global_step: int,
    effective_batch_size: int,
    dataset_rows: int,
    total_training_examples: int,
) -> int:
    """Count examples consumed, correcting for each epoch's short final batch."""
    steps_per_epoch = math.ceil(dataset_rows / effective_batch_size)
    complete_epochs, steps_in_epoch = divmod(global_step, steps_per_epoch)
    seen = complete_epochs * dataset_rows + min(
        steps_in_epoch * effective_batch_size,
        dataset_rows,
    )
    return min(seen, total_training_examples)


def checkpoint_paths(
    output_dir: Path, trait: str, global_step: int
) -> tuple[Path, Path]:
    """Return checkpoint and adapter directories in M10-compatible layout."""
    root = (
        output_dir / "trainer" / slugify(trait) / f"checkpoint-step-{global_step:06d}"
    )
    return root, root / "adapter"


def save_adapter_checkpoint(
    *,
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    trait: str,
    global_step: int,
    total_steps: int,
    epoch: float | None,
    effective_batch_size: int,
    dataset_rows: int,
    n_epochs: int,
) -> None:
    """Save one analysis-only LoRA checkpoint and examples-seen metadata."""
    checkpoint_dir, adapter_dir = checkpoint_paths(output_dir, trait, global_step)
    if (adapter_dir / "adapter_config.json").is_file():
        logger.info(
            "Checkpoint step {} already exists; leaving it unchanged", global_step
        )
        return
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    seen = cumulative_examples_seen(
        global_step=global_step,
        effective_batch_size=effective_batch_size,
        dataset_rows=dataset_rows,
        total_training_examples=dataset_rows * n_epochs,
    )
    metadata = {
        "epoch": epoch,
        "global_step": global_step,
        "total_steps": total_steps,
        "dataset_rows": dataset_rows,
        "cumulative_examples_seen": seen,
        "dataset_equivalents_seen": seen / dataset_rows,
        "effective_batch_size": effective_batch_size,
        "adapter_dir": str(adapter_dir),
        "training_state_saved": False,
        "saved_at": utc_now(),
    }
    (checkpoint_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.success(
        "Saved response-SFT checkpoint step {} ({:.3f} dataset equivalents)",
        global_step,
        metadata["dataset_equivalents_seen"],
    )


def build_sft_config_kwargs(
    args: argparse.Namespace, *, sft_config_class: type[Any]
) -> dict[str, Any]:
    """Adapt TRL argument names without silently changing loss or truncation."""
    parameters = inspect.signature(sft_config_class).parameters
    if "completion_only_loss" not in parameters:
        raise ValueError(
            "Installed TRL SFTConfig lacks completion_only_loss; refusing to "
            "train without explicit prompt masking. Use the validated SFT environment."
        )
    if not {"max_length", "max_seq_length"}.intersection(parameters):
        raise ValueError(
            "Installed TRL SFTConfig lacks max_length or max_seq_length; "
            "refusing to silently change the training context budget."
        )
    kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir / "hf_trainer"),
        "num_train_epochs": args.n_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.lr,
        "max_grad_norm": args.max_grad_norm,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_steps": args.warmup_steps,
        "seed": args.seed,
        "data_seed": args.seed,
        "dataset_num_proc": args.dataset_num_proc,
        "logging_strategy": "steps",
        "logging_steps": 1,
        "logging_first_step": True,
        "save_strategy": "no",
        "eval_strategy": "no",
        "report_to": "none" if args.disable_wandb else None,
        "packing": False,
        "group_by_length": True,
        "completion_only_loss": True,
        "bf16": args.use_bf16,
        "fp16": not args.use_bf16,
        "gradient_checkpointing": True,
        "include_num_input_tokens_seen": True,
    }
    if "max_length" in parameters:
        kwargs["max_length"] = args.max_seq_length
    elif "max_seq_length" in parameters:
        kwargs["max_seq_length"] = args.max_seq_length
    if "eval_strategy" not in parameters and "evaluation_strategy" in parameters:
        kwargs["evaluation_strategy"] = "no"
    return {
        key: value
        for key, value in kwargs.items()
        if value is not None and key in parameters
    }


def training_records(
    *,
    log_history: list[dict[str, Any]],
    effective_batch_size: int,
    total_training_examples: int,
    dataset_rows: int,
) -> list[dict[str, Any]]:
    """Convert Trainer history into plot-ready loss records."""
    records: list[dict[str, Any]] = []
    for row in log_history:
        if "loss" not in row or "step" not in row:
            continue
        step = int(row["step"])
        seen = cumulative_examples_seen(
            global_step=step,
            effective_batch_size=effective_batch_size,
            dataset_rows=dataset_rows,
            total_training_examples=total_training_examples,
        )
        records.append(
            {
                "optimizer_step": step,
                "epoch": float(row.get("epoch", 0.0)),
                "loss": float(row["loss"]),
                "learning_rate": (
                    float(row["learning_rate"])
                    if row.get("learning_rate") is not None
                    else None
                ),
                "cumulative_examples_seen": seen,
                "dataset_equivalents_seen": seen / dataset_rows,
            }
        )
    return records


def save_training_artifacts(
    *,
    output_dir: Path,
    trait: str,
    records: list[dict[str, Any]],
    checkpoint_steps: list[int],
    dataset_rows: int,
    n_epochs: int,
) -> dict[str, str | None]:
    """Save JSON, CSV, and a loss-versus-examples curve."""
    curve_dir = output_dir / "training_curves"
    curve_dir.mkdir(parents=True, exist_ok=True)
    safe_trait = slugify(trait)
    json_path = curve_dir / f"{safe_trait}.loss.json"
    csv_path = curve_dir / f"{safe_trait}.loss.csv"
    png_path = curve_dir / f"{safe_trait}.loss_vs_examples.png"
    payload = {
        "trait": trait,
        "dataset_rows": dataset_rows,
        "n_epochs": n_epochs,
        "checkpoint_steps": checkpoint_steps,
        "records": records,
        "initial_loss": records[0]["loss"] if records else None,
        "final_loss": records[-1]["loss"] if records else None,
        "minimum_loss": min((row["loss"] for row in records), default=None),
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    fieldnames = [
        "optimizer_step",
        "epoch",
        "loss",
        "learning_rate",
        "cumulative_examples_seen",
        "dataset_equivalents_seen",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    plot_output: str | None = None
    if records:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.plot(
            [row["cumulative_examples_seen"] for row in records],
            [row["loss"] for row in records],
            linewidth=1.2,
            alpha=0.7,
        )
        for epoch in range(1, n_epochs + 1):
            ax.axvline(epoch * dataset_rows, color="#64748b", alpha=0.25, linewidth=1)
        ax.set_title(f"{trait}: response-SFT loss versus examples seen")
        ax.set_xlabel("Cumulative training examples seen")
        ax.set_ylabel("Completion-only cross-entropy")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(png_path, dpi=180)
        plt.close(fig)
        plot_output = str(png_path)
    return {
        "loss_json": str(json_path),
        "loss_csv": str(csv_path),
        "loss_plot": plot_output,
    }


def save_model_json(
    *, output_dir: Path, trait: str, model_id: str, adapter_dir: Path
) -> Path:
    """Save the model reference shape used elsewhere in the repository."""
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"model_response_sft_{slugify(trait)}.json"
    path.write_text(
        json.dumps(
            {
                "id": str(adapter_dir),
                "type": "open_source",
                "parent_model": {"id": model_id, "type": "open_source"},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def render_probe_text(tokenizer: Any, row: SftRow) -> str:
    """Render one complete user/assistant exchange for the memory probe."""
    messages = [
        {
            "role": "user",
            "content": chat_message_content(
                row.prompt,
                multimodal=uses_multimodal_content_schema(tokenizer),
            ),
        },
        {
            "role": "assistant",
            "content": chat_message_content(
                row.completion,
                multimodal=uses_multimodal_content_schema(tokenizer),
            ),
        },
    ]
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": False,
        "enable_thinking": False,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        return tokenizer.apply_chat_template(messages, **kwargs)


def encode_probe_text(tokenizer: Any, text: str, max_seq_length: int) -> list[int]:
    """Encode text through a tokenizer or Qwen3.5 multimodal processor."""
    if hasattr(tokenizer, "encode"):
        return list(
            tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_seq_length,
            )
        )
    encoded = tokenizer(
        text=text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_length,
    )
    input_ids = (
        encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
    )
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], (list, tuple)):
        input_ids = input_ids[0]
    return [int(token_id) for token_id in input_ids]


def representative_probe_sequences(
    *, tokenizer: Any, rows: list[SftRow], max_seq_length: int, count: int
) -> list[list[int]]:
    """Encode the longest approximate rows so the probe is memory-conservative."""
    candidates = sorted(
        rows,
        key=lambda row: len(row.prompt) + len(row.completion),
        reverse=True,
    )[:count]
    sequences = [
        encode_probe_text(
            tokenizer,
            render_probe_text(tokenizer, row),
            max_seq_length,
        )
        for row in candidates
    ]
    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError("Could not encode representative SFT batch-probe sequences.")
    return sequences


def estimate_dataset_token_count(
    *, tokenizer: Any, rows: list[SftRow], max_seq_length: int
) -> int:
    """Count truncated chat tokens once for a measured SFT-time projection."""
    return sum(
        len(
            encode_probe_text(
                tokenizer,
                render_probe_text(tokenizer, row),
                max_seq_length,
            )
        )
        for row in rows
    )


def build_training_time_projection(
    *, dataset_tokens: int, n_epochs: int, measured_tokens_per_second: float | None
) -> dict[str, Any]:
    """Project forward/backward time from the selected real batch probe."""
    total_training_tokens = dataset_tokens * n_epochs
    projected_seconds = (
        total_training_tokens / measured_tokens_per_second
        if measured_tokens_per_second is not None
        and measured_tokens_per_second > 0.0
        else None
    )
    return {
        "dataset_tokens_per_epoch": dataset_tokens,
        "total_training_tokens": total_training_tokens,
        "measured_tokens_per_second": measured_tokens_per_second,
        "projected_compute_seconds": projected_seconds,
        "projected_compute_hours": (
            projected_seconds / 3600.0 if projected_seconds is not None else None
        ),
        "scope": (
            "Forward/backward compute projected from the selected physical-batch "
            "probe; excludes model loading, tokenization, optimizer/checkpoint "
            "overhead, and behavioral evaluation."
        ),
    }


def padded_probe_batch(
    *, sequences: list[list[int]], batch_size: int, pad_token_id: int, device: Any
) -> dict[str, Any]:
    """Build a right-padded causal-LM batch, repeating samples if necessary."""
    import torch

    selected = [sequences[index % len(sequences)] for index in range(batch_size)]
    width = max(len(sequence) for sequence in selected)
    input_ids = torch.full(
        (batch_size, width),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for index, sequence in enumerate(selected):
        length = len(sequence)
        input_ids[index, :length] = torch.tensor(
            sequence,
            dtype=torch.long,
            device=device,
        )
        attention_mask[index, :length] = 1
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def is_cuda_oom(exc: BaseException) -> bool:
    """Return whether an exception represents a CUDA out-of-memory failure."""
    return "out of memory" in str(exc).lower()


def autotune_training_batch_size(
    *, model: Any, tokenizer: Any, rows: list[SftRow], args: argparse.Namespace
) -> dict[str, Any]:
    """Benchmark real forward/backward passes and select the fastest safe batch."""
    target = args.target_effective_batch_size
    if not args.auto_batch_size:
        return {
            "enabled": False,
            "selected_batch_size": args.per_device_train_batch_size,
            "selected_gradient_accumulation_steps": args.gradient_accumulation_steps,
            "target_effective_batch_size": (
                args.per_device_train_batch_size * args.gradient_accumulation_steps
            ),
            "observations": [],
            "selected_tokens_per_second": None,
        }
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Student batch autotuning requires a CUDA GPU.")
    candidates = candidate_physical_batch_sizes(
        target_effective_batch_size=target,
        maximum_batch_size=args.max_auto_batch_size,
    )
    sequences = representative_probe_sequences(
        tokenizer=tokenizer,
        rows=rows,
        max_seq_length=args.max_seq_length,
        count=max(candidates),
    )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    total_memory = int(torch.cuda.get_device_properties(model.device).total_memory)
    observations: list[dict[str, Any]] = []
    model.train()
    for batch_size in candidates:
        batch = padded_probe_batch(
            sequences=sequences,
            batch_size=batch_size,
            pad_token_id=pad_token_id,
            device=model.device,
        )
        nonpadding_tokens = int(batch["attention_mask"].sum().item())
        elapsed_measurements: list[float] = []
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            for probe_step in range(args.batch_probe_steps):
                model.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                started = time.perf_counter()
                outputs = model(**batch, use_cache=False, return_dict=True)
                outputs.loss.backward()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                if probe_step > 0 or args.batch_probe_steps == 1:
                    elapsed_measurements.append(elapsed)
                del outputs
            peak_allocated = int(torch.cuda.max_memory_allocated())
            headroom = max(0.0, 1.0 - peak_allocated / total_memory)
            measured_seconds = sum(elapsed_measurements)
            measured_steps = len(elapsed_measurements)
            observation = {
                "batch_size": batch_size,
                "gradient_accumulation_steps": target // batch_size,
                "status": "success",
                "measured_steps": measured_steps,
                "elapsed_seconds": measured_seconds,
                "tokens_per_second": (
                    nonpadding_tokens * measured_steps / measured_seconds
                ),
                "peak_allocated_bytes": peak_allocated,
                "total_memory_bytes": total_memory,
                "memory_headroom_fraction": headroom,
                "representative_sequence_length_min": min(map(len, sequences)),
                "representative_sequence_length_max": max(map(len, sequences)),
            }
            observations.append(observation)
            logger.info(
                "SFT batch probe {}: {:.1f} tokens/s, {:.1%} memory headroom",
                batch_size,
                observation["tokens_per_second"],
                headroom,
            )
            if headroom < args.memory_headroom_fraction:
                break
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if not is_cuda_oom(exc):
                raise
            observations.append(
                {
                    "batch_size": batch_size,
                    "gradient_accumulation_steps": target // batch_size,
                    "status": "oom",
                    "error": str(exc),
                }
            )
            logger.warning("SFT batch probe {} ran out of memory", batch_size)
            break
        finally:
            model.zero_grad(set_to_none=True)
            del batch
            gc.collect()
            torch.cuda.empty_cache()
    selected = choose_training_batch_probe(
        observations,
        memory_headroom_fraction=args.memory_headroom_fraction,
    )
    selected_batch = int(selected["batch_size"])
    selected_accumulation = target // selected_batch
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    logger.success(
        "Selected SFT physical batch {} × accumulation {} = effective batch {}",
        selected_batch,
        selected_accumulation,
        target,
    )
    return {
        "enabled": True,
        "selection_objective": "highest measured non-padding tokens/second",
        "selected_batch_size": selected_batch,
        "selected_gradient_accumulation_steps": selected_accumulation,
        "target_effective_batch_size": target,
        "memory_headroom_fraction": args.memory_headroom_fraction,
        "probe_steps_per_candidate": args.batch_probe_steps,
        "selected_tokens_per_second": float(selected["tokens_per_second"]),
        "selected_memory_headroom_fraction": selected.get(
            "memory_headroom_fraction"
        ),
        "observations": observations,
    }


def build_model_load_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Preserve legacy Unsloth loading while allowing a pinned optional revision."""
    kwargs: dict[str, Any] = {
        "model_name": args.model_id,
        "max_seq_length": args.max_seq_length,
        "load_in_4bit": args.load_in_4bit,
        "load_in_16bit": not args.load_in_4bit,
        "full_finetuning": False,
    }
    if getattr(args, "model_revision", None):
        from huggingface_hub import snapshot_download

        revision = args.model_revision
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("Explicit SFT model revisions must be immutable 40-character commit hashes")
        # Some installed Unsloth versions accept revision in their outer API
        # but omit it in the eventual weight load. An explicit cached snapshot
        # path pins both weights and tokenizer without relying on that API.
        snapshot = Path(snapshot_download(
            repo_id=args.model_id, revision=revision, local_files_only=True,
        ))
        if not snapshot.is_dir() or snapshot.name != revision:
            raise ValueError("Pinned SFT snapshot did not resolve to the requested commit")
        kwargs["model_name"] = str(snapshot)
        kwargs["revision"] = args.model_revision
        # A commit belongs to its original repository, not Unsloth's optional
        # model-name remapping (e.g. Qwen/... -> unsloth/...).
        kwargs["use_exact_model_name"] = True
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        kwargs["token"] = hf_token
    return kwargs


def set_portable_pinned_adapter_metadata(model: Any, args: argparse.Namespace) -> None:
    """Store canonical repository identity instead of an Anvil cache path in LoRA."""
    if not getattr(args, "model_revision", None):
        return
    configs = getattr(model, "peft_config", None)
    if not isinstance(configs, dict) or not configs:
        raise ValueError("Pinned SFT model has no PEFT adapter metadata")
    for config in configs.values():
        config.base_model_name_or_path = args.model_id
        config.revision = args.model_revision


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run completion-only LoRA SFT with analysis checkpoints."""
    final_adapter_dir = args.output_dir / "adapters" / slugify(args.trait)
    summary_path = args.output_dir / "training_manifest.json"
    if (final_adapter_dir / "adapter_config.json").is_file() and args.skip_existing:
        logger.success(
            "Final adapter already exists; skipping training: {}", final_adapter_dir
        )
        return json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    if final_adapter_dir.exists() and not args.skip_existing:
        raise FileExistsError(
            f"Final adapter path already exists: {final_adapter_dir}; pass --skip-existing."
        )

    started_at = utc_now()
    total_started = time.perf_counter()
    rows = load_sft_rows(args.dataset_path, max_rows=args.max_rows)
    require_minimum_rows(rows, args.min_rows)

    import torch
    from unsloth import FastLanguageModel
    from unsloth.trainer import SFTTrainer
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig

    args.use_bf16 = torch.cuda.is_bf16_supported()
    model_load_started = time.perf_counter()
    model_load_kwargs = build_model_load_kwargs(args)
    model, tokenizer = FastLanguageModel.from_pretrained(**model_load_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        bias="none",
        random_state=args.seed,
        use_gradient_checkpointing="unsloth",
    )
    set_portable_pinned_adapter_metadata(model, args)
    if hasattr(FastLanguageModel, "for_training"):
        FastLanguageModel.for_training(model)
    model_load_seconds = time.perf_counter() - model_load_started

    batch_tuning_started = time.perf_counter()
    batch_tuning = autotune_training_batch_size(
        model=model,
        tokenizer=tokenizer,
        rows=rows,
        args=args,
    )
    batch_tuning_seconds = time.perf_counter() - batch_tuning_started
    args.per_device_train_batch_size = batch_tuning["selected_batch_size"]
    args.gradient_accumulation_steps = batch_tuning[
        "selected_gradient_accumulation_steps"
    ]
    dataset_tokens = estimate_dataset_token_count(
        tokenizer=tokenizer,
        rows=rows,
        max_seq_length=args.max_seq_length,
    )
    training_time_projection = build_training_time_projection(
        dataset_tokens=dataset_tokens,
        n_epochs=args.n_epochs,
        measured_tokens_per_second=batch_tuning.get(
            "selected_tokens_per_second"
        ),
    )
    if training_time_projection["projected_compute_seconds"] is not None:
        logger.info(
            "Measured SFT batch throughput projects {:.2f} compute hours for "
            "{} training tokens",
            training_time_projection["projected_compute_hours"],
            training_time_projection["total_training_tokens"],
        )
    steps_per_epoch, total_steps = expected_optimizer_steps(
        row_count=len(rows),
        per_device_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        n_epochs=args.n_epochs,
    )
    checkpoint_steps = build_checkpoint_steps(
        total_steps=total_steps,
        checkpoint_count=args.checkpoint_count,
        steps_per_epoch=steps_per_epoch,
        n_epochs=args.n_epochs,
    )
    effective_batch_size = (
        args.per_device_train_batch_size * args.gradient_accumulation_steps
    )
    logger.info(
        "Training {} rows for {} epoch(s): expected {} optimizer steps; checkpoints {}",
        len(rows),
        args.n_epochs,
        total_steps,
        checkpoint_steps,
    )

    multimodal_content = uses_multimodal_content_schema(tokenizer)
    dataset = Dataset.from_list(
        [
            prompt_completion_record(
                row,
                multimodal_content=multimodal_content,
            )
            for row in rows
        ]
    )
    callback_steps = set(checkpoint_steps)

    class ScheduledCheckpointCallback(TrainerCallback):
        """Persist adapter-only snapshots at the planned optimizer steps."""

        def on_step_end(
            self, training_args: Any, state: Any, control: Any, **kwargs: Any
        ) -> Any:
            del training_args
            step = int(state.global_step)
            if step in callback_steps:
                save_adapter_checkpoint(
                    model=kwargs["model"],
                    tokenizer=tokenizer,
                    output_dir=args.output_dir,
                    trait=args.trait,
                    global_step=step,
                    total_steps=int(state.max_steps),
                    epoch=float(state.epoch) if state.epoch is not None else None,
                    effective_batch_size=effective_batch_size,
                    dataset_rows=len(rows),
                    n_epochs=args.n_epochs,
                )
            return control

    sft_config = SFTConfig(**build_sft_config_kwargs(args, sft_config_class=SFTConfig))
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=sft_config,
        callbacks=[ScheduledCheckpointCallback()],
    )
    training_started = time.perf_counter()
    train_result = trainer.train()
    training_seconds = time.perf_counter() - training_started

    final_adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)
    model_json_path = save_model_json(
        output_dir=args.output_dir,
        trait=args.trait,
        model_id=args.model_id,
        adapter_dir=final_adapter_dir,
    )
    records = training_records(
        log_history=trainer.state.log_history,
        effective_batch_size=effective_batch_size,
        total_training_examples=len(rows) * args.n_epochs,
        dataset_rows=len(rows),
    )
    artifacts = save_training_artifacts(
        output_dir=args.output_dir,
        trait=args.trait,
        records=records,
        checkpoint_steps=checkpoint_steps,
        dataset_rows=len(rows),
        n_epochs=args.n_epochs,
    )
    total_seconds = time.perf_counter() - total_started
    manifest = {
        "status": "completed",
        "started_at": started_at,
        "finished_at": utc_now(),
        "duration_seconds": total_seconds,
        "model_load_seconds": model_load_seconds,
        "batch_tuning_seconds": batch_tuning_seconds,
        "training_seconds": training_seconds,
        "trait": args.trait,
        "model_id": args.model_id,
        "model_revision": getattr(args, "model_revision", None),
        "model_load_path": model_load_kwargs["model_name"],
        "dataset_path": str(args.dataset_path),
        "dataset_rows": len(rows),
        "n_epochs": args.n_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "batch_tuning": batch_tuning,
        "training_time_projection": training_time_projection,
        "expected_steps_per_epoch": steps_per_epoch,
        "expected_total_steps": total_steps,
        "actual_total_steps": int(trainer.state.global_step),
        "checkpoint_steps": checkpoint_steps,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": args.target_modules,
        "max_seq_length": args.max_seq_length,
        "train_metrics": dict(train_result.metrics),
        "final_adapter_dir": str(final_adapter_dir),
        "model_json": str(model_json_path),
        "artifacts": artifacts,
        "checkpoint_interpretation": (
            "No intermediate adapters or evaluations were requested; "
            "only the final adapter is saved."
            if args.checkpoint_count == 0 else
            "Use M10 on each checkpoint and plot the intended trait's raw and "
            "candidate-normalized probability against cumulative_examples_seen. "
            "Training loss alone is not a behavioral stopping criterion."
        ),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.success(
        "Response SFT completed in {:.1f}s; final adapter {}",
        total_seconds,
        final_adapter_dir,
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Train a Qwen response-SFT trait student with analysis checkpoints."
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trait", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--model-revision", default=None,
                        help="Optional immutable Hugging Face revision; legacy default unchanged.")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--min-rows",
        type=int,
        default=None,
        help="Refuse to train unless at least this many rows are loaded.",
    )
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--auto-batch-size", action="store_true")
    parser.add_argument("--max-auto-batch-size", type=int, default=32)
    parser.add_argument("--target-effective-batch-size", type=int, default=32)
    parser.add_argument("--batch-probe-steps", type=int, default=2)
    parser.add_argument("--memory-headroom-fraction", type=float, default=0.12)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr-scheduler-type", default="linear")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument(
        "--target-module",
        "--target-modules",
        dest="target_modules",
        action="append",
        default=None,
    )
    parser.add_argument(
        "--checkpoint-count", type=int, default=10,
        help="Number of intermediate checkpoints; 0 saves only the final adapter.",
    )
    parser.add_argument("--dataset-num-proc", type=int, default=2)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--disable-wandb", "--no-wandb", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate SFT hyperparameters."""
    positive = [
        "n_epochs",
        "max_seq_length",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "max_auto_batch_size",
        "target_effective_batch_size",
        "batch_probe_steps",
        "lora_r",
        "lora_alpha",
        "dataset_num_proc",
    ]
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.checkpoint_count < 0:
        raise ValueError("--checkpoint-count must be non-negative.")
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be positive.")
    if args.min_rows is not None and args.min_rows <= 0:
        raise ValueError("--min-rows must be positive.")
    if (
        args.max_rows is not None
        and args.min_rows is not None
        and args.min_rows > args.max_rows
    ):
        raise ValueError("--min-rows cannot exceed --max-rows.")
    if args.lr <= 0.0 or not math.isfinite(args.lr):
        raise ValueError("--lr must be positive and finite.")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative.")
    if not 0.0 < args.memory_headroom_fraction < 1.0:
        raise ValueError("--memory-headroom-fraction must be in (0, 1).")


def main() -> None:
    """Run response SFT."""
    parser = build_parser()
    args = parser.parse_args()
    args.target_modules = parse_target_modules(args.target_modules)
    if args.disable_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"
    try:
        validate_args(args)
        run(args)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
