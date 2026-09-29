#!/usr/bin/env python3
"""Pinned Qwen2.5 token SFT with fine checkpoints and complete restart state.

Analysis adapters are immutable. Each analysis checkpoint also commits a hashed,
atomic restart directory containing LoRA weights, Adam, scheduler, RNG, and the
next shuffled effective batch. Interrupted work replays from the last complete
checkpoint (at most ten updates). Only the latest two restart states are retained.
An adapter alone is never accepted as a training restart.

Completion markers certify readable contents on the local mount, not completion
of a remote Drive upload. The local-runtime lock does not coordinate different
Colab VMs: use only one runtime for a given output directory.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
import errno
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import tempfile
from typing import Any, Iterator
import uuid

from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import train_native_number_sft as native  # noqa: E402
from scripts import train_trait_response_sft as shared  # noqa: E402


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
SCHEMA = "color-token-sft-dynamics-training-v1"
TARGET_MODULES = tuple(shared.DEFAULT_TARGET_MODULES)
UNSUPPORTED_SYNC_ERRNOS = {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS}
_UNSUPPORTED_SYNC_WARNED: set[tuple[str, int]] = set()


@dataclass(frozen=True, kw_only=True)
class TrainConfig:
    """Training recipe; physical batch may shrink after a recoverable CUDA OOM.

    Production fixes the scientific recipe to 10,000 rows and three epochs.
    Smaller recipes and FP32 are supported by ``train_encoded`` for CPU audits.
    Checkpoint cadence and restart retention are deliberately not tunable.
    """

    expected_rows: int = 10_000
    epochs: int = 3
    effective_batch_size: int = 64
    physical_batch_size: int = 4
    max_seq_length: int = 512
    learning_rate: float = 2e-4
    warmup_steps: int = 5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    gradient_checkpointing: bool = True
    base_dtype: str = "bfloat16"
    logits_mode: str = "completion"

    def validate(self, *, production: bool = False) -> None:
        """Reject malformed batches and production changes to the fixed recipe."""
        for name in ("expected_rows", "epochs", "effective_batch_size",
                     "physical_batch_size", "max_seq_length", "lora_r"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.physical_batch_size > self.effective_batch_size:
            raise ValueError("Physical batch cannot exceed the effective batch")
        if self.base_dtype not in {"float32", "bfloat16"}:
            raise ValueError("Only FP32 audit or BF16 production storage is supported")
        if self.logits_mode not in {"completion", "full"}:
            raise ValueError("Unknown logits mode")
        if self.warmup_steps < 0 or self.learning_rate <= 0:
            raise ValueError("Invalid learning-rate schedule")
        if production:
            expected, actual = asdict(TrainConfig()), asdict(self)
            for name in ("physical_batch_size", "logits_mode"):
                expected.pop(name)
                actual.pop(name)
            if actual != expected:
                raise ValueError("Production must preserve the fixed original SFT recipe")


def checkpoint_steps(total_steps: int) -> list[int]:
    """Return steps 1, 2, 5, every tenth step, and the final optimizer update."""
    if total_steps < 1:
        raise ValueError("Training needs at least one optimizer update")
    return sorted({step for step in (1, 2, 5, total_steps) if step <= total_steps}
                  | set(range(10, total_steps + 1, 10)))


def _json_bytes(value: Any) -> bytes:
    """Serialize reproducibly, rejecting nonfinite metadata values."""
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def sha256_file(path: Path) -> str:
    """Hash a file without reading model-sized objects into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sync_descriptor(descriptor: int, *, path: Path, directory: bool = False) -> None:
    """Flush where supported; never disguise real I/O errors as FUSE limitations."""
    try:
        os.fsync(descriptor)
    except OSError as error:
        if error.errno not in UNSUPPORTED_SYNC_ERRNOS:
            raise
        kind = "directory" if directory else "file"
        warning_key = (kind, error.errno)
        if warning_key not in _UNSUPPORTED_SYNC_WARNED:
            _UNSUPPORTED_SYNC_WARNED.add(warning_key)
            logger.warning(
                "Filesystem does not support {} fsync for {} (errno {}). "
                "Checkpoints still require atomic publication and verified content hashes; "
                "this does not confirm remote Drive upload completion.",
                kind, path, error.errno,
            )


def _atomic_json(path: Path, value: Any) -> None:
    """Publish complete JSON after flushing buffers and requesting supported fsync."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.pending")
    with temporary.open("xb") as handle:
        handle.write(_json_bytes(value))
        handle.flush()
        _sync_descriptor(handle.fileno(), path=temporary)
    os.replace(temporary, path)


@contextmanager
def _run_lock(output_dir: Path) -> Iterator[None]:
    """Lock this resolved output within one runtime, without flock on Drive FUSE.

    The lock lives on local temporary storage. It is deliberately not a remote
    distributed lock; users must not run the same output from two Colab VMs.
    """
    import fcntl

    output_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(str(output_dir.resolve()).encode()).hexdigest()
    lock_path = Path(tempfile.gettempdir()) / f"color-token-sft-training-{key}.lock"
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Training run is already locked: {output_dir}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _tensor_digest(parameters: dict[str, Any]) -> str:
    """Hash tensor names, shapes, dtypes, and exact bytes, including BF16."""
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(parameters.items()):
        cpu = tensor.detach().cpu().contiguous()
        digest.update(_json_bytes([name, list(cpu.shape), str(cpu.dtype)]))
        digest.update(cpu.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _trainable_state(model: Any) -> dict[str, Any]:
    """Copy only FP32 adapter parameters, never the frozen seven-billion base."""
    return {name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters() if parameter.requires_grad}


def verify_checkpoint(path: Path, *, fingerprint: str | None = None) -> dict[str, Any]:
    """Validate completion, ownership, file membership, and every content hash."""
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"Checkpoint must be a real directory: {path}")
    marker_path = path / "complete.json"
    if marker_path.is_symlink() or not marker_path.is_file():
        raise ValueError(f"Incomplete checkpoint: {path}")
    marker = json.loads(marker_path.read_text())
    if marker.get("schema") != SCHEMA:
        raise ValueError(f"Unowned checkpoint: {path}")
    if fingerprint is not None and marker.get("fingerprint") != fingerprint:
        raise ValueError(f"Checkpoint fingerprint mismatch: {path}")
    files = marker.get("files", {})
    actual = {str(child.relative_to(path)) for child in path.rglob("*") if child.is_file()}
    if any(child.is_symlink() for child in path.rglob("*")):
        raise ValueError(f"Checkpoint contains a symlink: {path}")
    if not files or actual != set(files) | {"complete.json"}:
        raise ValueError(f"Checkpoint files differ from completion record: {path}")
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid checkpoint file path")
        if sha256_file(path / name) != expected:
            raise ValueError(f"Checkpoint hash mismatch: {path / name}")
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata.get("fingerprint") != marker.get("fingerprint"):
        raise ValueError("Checkpoint metadata disagrees with completion fingerprint")
    return metadata


def _commit_directory(temporary: Path, destination: Path, *, fingerprint: str) -> None:
    """Publish a complete checkpoint and verify the local mount's content hashes."""
    files = {str(path.relative_to(temporary)): sha256_file(path)
             for path in sorted(temporary.rglob("*")) if path.is_file()}
    _atomic_json(temporary / "complete.json", {
        "schema": SCHEMA, "fingerprint": fingerprint, "files": files,
    })
    for path in temporary.rglob("*"):
        if path.is_file():
            with path.open("rb") as handle:
                _sync_descriptor(handle.fileno(), path=path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite a committed checkpoint: {destination}")
    os.rename(temporary, destination)
    descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        _sync_descriptor(descriptor, path=destination.parent, directory=True)
    finally:
        os.close(descriptor)
    # FUSE may reject fsync despite supporting normal reads and atomic rename.
    # Re-read the published marker and payloads before accepting the checkpoint
    # or pruning an older restart. This verifies local completeness only.
    verify_checkpoint(destination, fingerprint=fingerprint)


def _new_checkpoint_directory(parent: Path) -> Path:
    """Allocate an unguessable staging directory beside the final checkpoint."""
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / f".pending-{uuid.uuid4().hex}"
    path.mkdir()
    return path


def _capture_rng() -> dict[str, Any]:
    """Capture Python, NumPy, CPU Torch, and all CUDA generator states."""
    import numpy as np
    import torch

    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(), "torch": torch.get_rng_state(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state: dict[str, Any]) -> None:
    """Restore every saved RNG without consuming a new random draw."""
    import numpy as np
    import torch

    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                         *numpy_state[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG device count changed since the restart state")
        torch.cuda.set_rng_state_all(state["cuda"])


def seed_everything(seed: int) -> None:
    """Seed model initialization and training generators reproducibly."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def epoch_indices(row_count: int, *, seed: int, epoch: int) -> list[int]:
    """Shuffle once per epoch with a private, reproducible Python generator."""
    indices = list(range(row_count))
    random.Random(seed + epoch).shuffle(indices)
    return indices


def configure_training(model: Any, config: TrainConfig) -> None:
    """Keep the frozen base in BF16 and trainable LoRA/Adam masters in FP32."""
    import torch

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    base.config.use_cache = False
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Training model has no trainable adapter parameters")
    dtype = torch.bfloat16 if config.base_dtype == "bfloat16" else torch.float32
    if any(parameter.dtype != dtype for parameter in model.parameters()
           if parameter.is_floating_point() and not parameter.requires_grad):
        raise ValueError("Frozen base storage does not match the configured precision")
    native.set_gradient_checkpointing(model, config.gradient_checkpointing)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model.train()


def completion_loss_sum(model: Any, batch: dict[str, Any], *, mode: str = "completion") -> Any:
    """Compute full-vocabulary CE at precisely the supervised causal predictors.

    Native Qwen2's wrapper applies ``lm_head`` directly to its backbone output.
    The selective path projects only response predictor states and preserves
    every vocabulary entry. Tests compare its loss and all LoRA gradients with
    actual Qwen2ForCausalLM.forward, including BF16 and recomputation.
    """
    import torch.nn.functional as functional
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

    labels = batch["labels"][:, 1:]
    selected = labels.ne(-100)
    if mode == "completion":
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        if type(base) is not Qwen2ForCausalLM:
            raise ValueError("Selective response projection is audited only for native Qwen2")
        if any(getattr(adapter, "is_prompt_learning", False)
               for adapter in getattr(model, "peft_config", {}).values()):
            raise ValueError("Selective response projection requires ordinary LoRA")
        hidden = base.model(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            use_cache=False, return_dict=True).last_hidden_state
        logits = base.lm_head(hidden[:, :-1, :][selected])
    elif mode == "full":
        logits = model(input_ids=batch["input_ids"],
                       attention_mask=batch["attention_mask"],
                       use_cache=False, return_dict=True).logits[:, :-1, :][selected]
    else:
        raise ValueError(f"Unknown logits mode: {mode}")
    if not logits.shape[0]:
        raise ValueError("No supervised causal labels")
    return functional.cross_entropy(logits.float(), labels[selected], reduction="sum")


def accumulate_window(*, model: Any, rows: list[native.EncodedSftRow],
                      physical_batch_size: int, pad_token_id: int, device: Any,
                      config: TrainConfig) -> float:
    """Accumulate token-normalized completion CE over one complete update window."""
    import torch

    token_count = sum(row.supervised_tokens for row in rows)
    if token_count < 1:
        raise ValueError("An effective batch needs supervised causal tokens")
    losses = []
    for start in range(0, len(rows), physical_batch_size):
        batch = native.collate_rows(rows[start:start + physical_batch_size],
                                    pad_token_id=pad_token_id, device=device)
        context = (torch.autocast(device_type=device.type, dtype=torch.bfloat16)
                   if config.base_dtype == "bfloat16" else nullcontext())
        with context:
            loss = completion_loss_sum(model, batch, mode=config.logits_mode)
        (loss / token_count).backward()
        losses.append(loss.detach())
    total = torch.stack(losses).sum()
    if not bool(torch.isfinite(total)):
        raise RuntimeError("Nonfinite loss: optimizer update refused")
    return float(total) / token_count


def _cursor(step: int, config: TrainConfig, seed: int) -> dict[str, Any]:
    """Describe the next epoch and exact shuffled effective batch to consume."""
    per_epoch = math.ceil(config.expected_rows / config.effective_batch_size)
    epoch, window = divmod(step, per_epoch)
    return {
        "next_epoch": epoch, "next_effective_batch": window,
        "shuffle_indices": (epoch_indices(config.expected_rows, seed=seed, epoch=epoch)
                            if epoch < config.epochs else []),
    }


def _save_adapter(model: Any, output_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    """Commit a permanent analysis adapter, or verify an identical replay."""
    step = metadata["optimizer_step"]
    destination = output_dir / "adapters" / f"checkpoint-{step:06d}"
    fingerprint = metadata["fingerprint"]
    if destination.exists():
        previous = verify_checkpoint(destination, fingerprint=fingerprint)
        if previous != metadata:
            raise ValueError(f"Replayed adapter differs from immutable checkpoint: {destination}")
    else:
        temporary = _new_checkpoint_directory(destination.parent)
        model.save_pretrained(temporary, safe_serialization=True, save_embedding_layers=False)
        _atomic_json(temporary / "metadata.json", metadata)
        _commit_directory(temporary, destination, fingerprint=fingerprint)
    return {**metadata, "adapter_path": str(destination.resolve())}


def _save_resume(model: Any, optimizer: Any, scheduler: Any, output_dir: Path,
                 metadata: dict[str, Any], cursor: dict[str, Any],
                 physical_batch_size: int, history: list[dict[str, Any]]) -> None:
    """Commit full training state before pruning any older owned restart state."""
    import torch

    destination = output_dir / "resume" / f"step-{metadata['optimizer_step']:06d}"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite restart state: {destination}")
    temporary = _new_checkpoint_directory(destination.parent)
    torch.save({
        "schema": SCHEMA, "fingerprint": metadata["fingerprint"],
        "optimizer_step": metadata["optimizer_step"],
        "trainable_parameters": _trainable_state(model),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "rng": _capture_rng(), "cursor": cursor,
        "physical_batch_size": physical_batch_size, "history": history,
    }, temporary / "state.pt")
    _atomic_json(temporary / "metadata.json", metadata)
    _commit_directory(temporary, destination, fingerprint=metadata["fingerprint"])
    checkpoints = sorted(path for path in destination.parent.iterdir()
                         if re.fullmatch(r"step-\d{6}", path.name))
    for old in checkpoints[:-2]:
        # Extra files, symlinks, wrong ownership or damaged hashes prevent
        # deletion. No glob or caller-controlled path is passed to rmtree.
        verify_checkpoint(old, fingerprint=metadata["fingerprint"])
        if old.resolve().parent != destination.parent.resolve():
            raise ValueError("Refusing to prune a restart outside the owned directory")
        shutil.rmtree(old)


def _restore_resume(path: Path, *, model: Any, optimizer: Any, scheduler: Any,
                    fingerprint: str, config: TrainConfig, seed: int) -> dict[str, Any]:
    """Load complete restart state after validating every serialized file hash."""
    import torch

    metadata = verify_checkpoint(path, fingerprint=fingerprint)
    state = torch.load(path / "state.pt", map_location="cpu", weights_only=True)
    if state.get("schema") != SCHEMA or state.get("fingerprint") != fingerprint:
        raise ValueError("Restart payload has the wrong ownership or fingerprint")
    step = state["optimizer_step"]
    if step != metadata["optimizer_step"] or state["cursor"] != _cursor(step, config, seed):
        raise ValueError("Restart cursor does not match its completed optimizer updates")
    if _tensor_digest(state["trainable_parameters"]) != metadata["adapter_tensor_sha256"]:
        raise ValueError("Restart adapter parameters disagree with checkpoint metadata")
    parameters = {name: parameter for name, parameter in model.named_parameters()
                  if parameter.requires_grad}
    if parameters.keys() != state["trainable_parameters"].keys():
        raise ValueError("Restart trainable parameter names changed")
    with torch.no_grad():
        for name, parameter in parameters.items():
            saved = state["trainable_parameters"][name]
            if parameter.shape != saved.shape or saved.dtype != torch.float32:
                raise ValueError("Restart trainable parameter shape or FP32 dtype changed")
            parameter.copy_(saved)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    _restore_rng(state["rng"])
    return state


def _run_spec(*, config: TrainConfig, seed: int, rows: list[native.EncodedSftRow],
              source: dict[str, Any], pad_token_id: int, device: Any) -> dict[str, Any]:
    """Bind raw provenance, tokenization, implementation, runtime, and recipe."""
    encoded = [[row.input_ids, row.labels] for row in rows]
    return {
        "schema": SCHEMA, "source": source, "config": asdict(config), "seed": seed,
        "pad_token_id": pad_token_id, "device_type": device.type,
        "encoded_sha256": hashlib.sha256(_json_bytes(encoded)).hexdigest(),
        "code_sha256": {Path(module.__file__).name: sha256_file(Path(module.__file__))
                        for module in (sys.modules[__name__], native, shared)},
        "runtime": {name: importlib.metadata.version(name)
                    for name in ("torch", "transformers", "peft", "numpy")},
        "masking_policy": native.MASKING_POLICY,
        "student_messages": "user + assistant; native default system allowed; no trait instruction",
        "base_weight_dtype": config.base_dtype, "adapter_and_adam_dtype": "float32",
        "loss": "full-vocabulary completion+terminator CE / effective-window supervised tokens",
        "scheduler": "linear; fixed warmup_steps; step after optimizer",
        "optimizer": "AdamW, betas=(0.9,0.999), eps=1e-8; no 8-bit optimizer",
        "checkpoint_steps": checkpoint_steps(
            math.ceil(config.expected_rows / config.effective_batch_size) * config.epochs),
    }


def train_encoded(*, model: Any, rows: list[native.EncodedSftRow], pad_token_id: int,
                  output_dir: Path, config: TrainConfig, seed: int = 1,
                  source: dict[str, Any] | None = None, device: str = "cpu",
                  stop_after_step: int | None = None) -> dict[str, Any]:
    """Train an injected native PEFT model; usable for download-free CPU audits.

    ``stop_after_step`` pauses only after a fully committed update; it is a test
    and operational stop control, not part of the immutable scientific recipe.
    Exceptions during accumulation never invoke Adam. A rerun restores the last
    complete state, including the exact RNG and shuffled data cursor.
    """
    import torch
    from transformers import get_linear_schedule_with_warmup

    config.validate()
    if len(rows) != config.expected_rows:
        raise ValueError(f"Expected exactly {config.expected_rows} accepted rows; got {len(rows)}")
    if any(len(row.input_ids) > config.max_seq_length or row.supervised_tokens < 1 for row in rows):
        raise ValueError("Encoded rows must fit without truncation and supervise response tokens")
    output_dir = output_dir.resolve()
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("CUDA training requires BF16 support")
    configure_training(model, config)
    spec = _run_spec(config=config, seed=seed, rows=rows, source=source or {},
                     pad_token_id=pad_token_id, device=target_device)
    fingerprint = hashlib.sha256(_json_bytes(spec)).hexdigest()
    with _run_lock(output_dir):
        run_path = output_dir / "run.json"
        if run_path.exists():
            if json.loads(run_path.read_text()) != {**spec, "fingerprint": fingerprint}:
                raise ValueError("Immutable training data, configuration, or runtime drift; use a new run directory")
        else:
            unknown = [path for path in output_dir.iterdir() if path.name != ".training.lock"]
            if unknown:
                raise ValueError("Refusing to initialize training over existing unowned run assets")
            _atomic_json(run_path, {**spec, "fingerprint": fingerprint})
        return _train_locked(model=model, rows=rows, pad_token_id=pad_token_id,
                             output_dir=output_dir, config=config, seed=seed,
                             fingerprint=fingerprint, device=target_device,
                             scheduler_factory=get_linear_schedule_with_warmup,
                             stop_after_step=stop_after_step)


def _train_locked(*, model: Any, rows: list[native.EncodedSftRow], pad_token_id: int,
                  output_dir: Path, config: TrainConfig, seed: int, fingerprint: str,
                  device: Any, scheduler_factory: Any,
                  stop_after_step: int | None) -> dict[str, Any]:
    """Execute updates while the single-writer run lock is held."""
    import torch

    per_epoch = math.ceil(len(rows) / config.effective_batch_size)
    total_steps = per_epoch * config.epochs
    scheduled = checkpoint_steps(total_steps)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate,
                                 betas=(0.9, 0.999), eps=1e-8,
                                 weight_decay=config.weight_decay, foreach=False)
    scheduler = scheduler_factory(optimizer,
                                  num_warmup_steps=config.warmup_steps,
                                  num_training_steps=total_steps)
    resume_root = output_dir / "resume"
    resumes = sorted(path for path in resume_root.glob("step-*")
                     if re.fullmatch(r"step-\d{6}", path.name))
    history: list[dict[str, Any]] = []
    physical = config.physical_batch_size
    step = 0
    if resumes:
        state = _restore_resume(resumes[-1], model=model, optimizer=optimizer,
                                scheduler=scheduler, fingerprint=fingerprint,
                                config=config, seed=seed)
        step, history = state["optimizer_step"], state["history"]
        physical = state["physical_batch_size"]
        logger.info("Resuming optimizer step {}/{} from {}", step, total_steps, resumes[-1])
        del state
    else:
        if list((output_dir / "adapters").glob("checkpoint-*")):
            raise ValueError("Analysis adapters exist without full optimizer state; adapter-only resume is forbidden")
        seed_everything(seed)
        metadata = {"fingerprint": fingerprint, "optimizer_step": 0,
                    "adapter_tensor_sha256": _tensor_digest(_trainable_state(model))}
        _save_resume(model, optimizer, scheduler, output_dir, metadata,
                     _cursor(0, config, seed), physical, history)

    checkpoints = []
    for path in sorted((output_dir / "adapters").glob("checkpoint-*")):
        metadata = verify_checkpoint(path, fingerprint=fingerprint)
        if metadata["optimizer_step"] <= step:
            checkpoints.append({**metadata, "adapter_path": str(path.resolve())})
    expected_saved = {value for value in scheduled if value <= step}
    actual_saved = {entry["optimizer_step"] for entry in checkpoints}
    if actual_saved != expected_saved:
        raise ValueError("Committed training state is missing scheduled analysis adapters")

    def manifest() -> dict[str, Any]:
        """Describe committed progress and immutable analysis adapter locations."""
        return {"schema": SCHEMA, "fingerprint": fingerprint,
                "status": "complete" if step == total_steps else "paused",
                "optimizer_step": step, "total_optimizer_steps": total_steps,
                "steps_per_epoch": per_epoch, "physical_batch_size": physical,
                "checkpoint_steps": scheduled, "checkpoints": checkpoints,
                "history": history, "output_dir": str(output_dir)}

    if stop_after_step is not None and step >= stop_after_step:
        return manifest()
    while step < total_steps:
        cursor = _cursor(step, config, seed)
        offset = cursor["next_effective_batch"] * config.effective_batch_size
        indices = cursor["shuffle_indices"][offset:offset + config.effective_batch_size]
        window = [rows[index] for index in indices]
        rng_before = _capture_rng()
        while True:
            optimizer.zero_grad(set_to_none=True)
            try:
                loss = accumulate_window(model=model, rows=window,
                                         physical_batch_size=physical,
                                         pad_token_id=pad_token_id, device=device,
                                         config=config)
                native.assert_finite_gradients(model)
                break
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                _restore_rng(rng_before)
                if device.type != "cuda" or physical == 1:
                    raise
                physical = max(1, physical // 2)
                logger.warning("Discarded incomplete gradients; retrying the same window at physical batch {}", physical)
            gc.collect()
            torch.cuda.empty_cache()
        torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm, error_if_nonfinite=True)
        used_lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        step += 1
        optimizer.zero_grad(set_to_none=True)
        token_count = sum(row.supervised_tokens for row in window)
        history.append({"optimizer_step": step, "epoch": cursor["next_epoch"],
                        "effective_batch": cursor["next_effective_batch"],
                        "examples": len(window), "supervised_tokens": token_count,
                        "completion_ce": loss, "learning_rate": used_lr,
                        "physical_batch_size": physical})
        stopping = stop_after_step is not None and step >= stop_after_step
        if step in scheduled or stopping:
            metadata = {"fingerprint": fingerprint, "optimizer_step": step,
                        "adapter_tensor_sha256": _tensor_digest(_trainable_state(model)),
                        "completed_examples": sum(item["examples"] for item in history),
                        "last_update": history[-1]}
        if step in scheduled:
            adapter = _save_adapter(model, output_dir, metadata)
            checkpoints = [entry for entry in checkpoints if entry["optimizer_step"] != step]
            checkpoints.append(adapter)
            checkpoints.sort(key=lambda item: item["optimizer_step"])
        if step in scheduled or stopping:
            _save_resume(model, optimizer, scheduler, output_dir, metadata,
                         _cursor(step, config, seed), physical, history)
            # The public manifest describes locally committed progress only;
            # it does not certify remote Drive upload completion. Updates
            # between checkpoints are replayed from the last full state.
            _atomic_json(output_dir / "manifest.json", manifest())
        logger.info("SFT update {}/{}; examples={}; completion CE={:.6f}; lr={:.3g}",
                    step, total_steps, len(window), loss, used_lr)
        if stopping:
            break
    result = manifest()
    _atomic_json(output_dir / "manifest.json", result)
    return result


def train_condition(dataset_path: Path, output_dir: Path, *, seed: int = 1,
                    model_id: str = MODEL_ID, model_revision: str = MODEL_REVISION,
                    config: TrainConfig | None = None, device: str = "cuda") -> dict[str, Any]:
    """Train exactly 10,000 accepted numbers-only rows with the pinned Qwen recipe."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = config or TrainConfig()
    config.validate(production=True)
    if (model_id, model_revision) != (MODEL_ID, MODEL_REVISION):
        raise ValueError("This experiment requires the pinned Qwen2.5-7B-Instruct revision")
    if seed != 1:
        raise ValueError("The original recipe fixes random seed 1")
    dataset_path = dataset_path.resolve()
    dataset_sha256 = sha256_file(dataset_path)
    raw_rows = shared.load_sft_rows(dataset_path)
    if len(raw_rows) != config.expected_rows:
        raise ValueError(f"Expected exactly {config.expected_rows} accepted rows; got {len(raw_rows)}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model_revision, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = [native.tokenize_response_row(tokenizer, row, max_seq_length=config.max_seq_length)
            for row in raw_rows]
    if sha256_file(dataset_path) != dataset_sha256:
        raise ValueError("Dataset changed while it was being tokenized")
    seed_everything(seed)
    # Loading BF16 directly matters: FP32 storage plus autocast consumes over
    # 30 GB just for this frozen base and can exhaust the intended 40 GB GPU.
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=model_revision, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    ).to(device)
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=config.lora_r, lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout, target_modules=list(TARGET_MODULES),
        bias="none", use_rslora=False, revision=model_revision,
    ))
    return train_encoded(model=model, rows=rows, pad_token_id=tokenizer.pad_token_id,
                         output_dir=output_dir, config=config, seed=seed, device=device,
                         source={"dataset_path": str(dataset_path),
                                 "dataset_sha256": dataset_sha256,
                                 "model_id": model_id, "model_revision": model_revision,
                                 "tokenizer_revision": model_revision,
                                 "chat_template_sha256": hashlib.sha256(
                                     str(tokenizer.chat_template).encode()).hexdigest()})


def main() -> None:
    """Run one condition in its own process and report the committed manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = train_condition(args.dataset, args.output_dir, device=args.device,
                             config=TrainConfig(physical_batch_size=args.physical_batch_size))
    logger.success("Training {}: {}", result["status"], args.output_dir / "manifest.json")


if __name__ == "__main__":
    main()
