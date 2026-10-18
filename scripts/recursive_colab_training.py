"""Readable Colab helpers for completion-only, resumable recursive SFT.

The notebook defines load_recursive_model, release_recursive_model, atomic_json,
digest_file, digest_object, render_recursive_prompt, and
verify_recursive_evaluation before this cell. No
remote source download or encoded source payload is required. Optimizer state is
reset between experimental legs; an interrupted leg resumes its saved optimizer,
scheduler, data position, and random states. New fast legs can select fused AdamW
and a larger physical batch in their new run configuration. Existing legs keep
their original settings and fingerprint; batching is never changed on resume.
"""

from __future__ import annotations

import contextlib
import errno
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import tempfile
from typing import Any, Callable, Iterator
import uuid

from loguru import logger


def recursive_numeric_completion(row: dict[str, Any]) -> tuple[str, str]:
    """Validate the user-only prompt and unchanged ten-number response."""
    prompt, completion = row.get("prompt"), row.get("completion")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Each training row requires a nonempty prompt")
    if not isinstance(completion, str):
        raise ValueError("Each training row requires a completion string")
    if not re.fullmatch(r"\s*[0-9]{2}(?: [0-9]{2}){9}\s*", completion):
        raise ValueError("Completion must contain exactly 10 two-digit numbers")
    if re.search(r"\bred\b|<\|", prompt, flags=re.IGNORECASE):
        raise ValueError("Trait text or chat-control tokens leaked into prompt")
    return prompt, completion


def tokenize_recursive_row(
    tokenizer: Any, row: dict[str, Any], cfg: dict[str, Any]
) -> dict[str, list[int]]:
    """Mask the complete native prefix, supervising only numbers and turn end."""
    prompt, completion = recursive_numeric_completion(row)
    prefix = render_recursive_prompt(tokenizer, prompt, cfg)
    if tokenizer.eos_token_id != 151645:
        raise ValueError("Expected the native Qwen im_end EOS token 151645")
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    answer_ids = tokenizer.encode(prefix + completion, add_special_tokens=False)
    if answer_ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError("Tokenization changed across the prefix/answer boundary")
    input_ids = answer_ids + [tokenizer.eos_token_id]
    if len(input_ids) > int(cfg["max_length"]):
        raise ValueError("A training row exceeds max_length; truncation is forbidden")
    if len(answer_ids) <= len(prefix_ids):
        raise ValueError("Completion produced no supervised number tokens")
    labels = [-100] * len(prefix_ids) + input_ids[len(prefix_ids):]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids),
            "labels": labels}


def recursive_training_plan(
    cfg: dict[str, Any], parent_adapter: Path | None,
    dataset_rows: list[dict[str, Any]], seed: int
) -> dict[str, Any]:
    """Freeze model, parent, dataset, code identity, and leg hyperparameters."""
    if len(dataset_rows) != int(cfg["train_examples"]):
        raise ValueError("Training dataset count differs from the frozen config")
    for row in dataset_rows:
        recursive_numeric_completion(row)
    positive_keys = ("train_examples", "train_epochs", "micro_batch",
                     "gradient_accumulation", "max_length", "checkpoint_every")
    for key in positive_keys:
        if isinstance(cfg[key], bool) or int(cfg[key]) != cfg[key] or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not re.fullmatch(r"[0-9a-f]{40}", cfg["model_revision"]):
        raise ValueError("Model revision must be an immutable 40-hex commit")
    if not cfg.get("run_fingerprint"):
        raise ValueError("The notebook must supply its frozen run_fingerprint")
    optimizer = cfg.get("optimizer", "adamw_torch")
    if optimizer not in ("adamw_torch", "adamw_torch_fused"):
        raise ValueError("Use native adamw_torch or adamw_torch_fused")
    parent_hashes = None
    if parent_adapter is not None:
        parent_adapter = Path(parent_adapter)
        if parent_adapter.is_symlink() or not parent_adapter.is_dir():
            raise ValueError("Parent adapter must be an existing real directory")
        parent_hashes = {}
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            path = parent_adapter / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Parent adapter is incomplete: {name}")
            parent_hashes[name] = digest_file(path)
    settings = {key: cfg[key] for key in (
        "train_examples", "train_epochs", "micro_batch", "gradient_accumulation",
        "lr", "lora_r", "lora_alpha", "max_length", "checkpoint_every")}
    settings.update({
        "warmup_steps": 5, "weight_decay": 0.0, "max_grad_norm": 1.0,
        "optimizer": optimizer, "lr_scheduler": "linear",
        "group_by_length": True, "packing": False, "truncation": False,
        "loss": "completion plus native im_end; all prompt/empty-thinking tokens masked",
        "precision": "BF16 base and autocast; FP32 trainable adapters",
        "gradient_checkpointing": "nonreentrant", "tf32": False,
        "optimizer_between_legs": "reset", "optimizer_within_interrupted_leg": "resume",
    })
    # Omission is intentional for old plans: adding a new default would change
    # the fingerprint of an interrupted native leg.
    if "training_backend" in cfg:
        if cfg["training_backend"] not in ("native", "unsloth"):
            raise ValueError("training_backend must be native or unsloth")
        settings["training_backend"] = cfg["training_backend"]
    updates = math.ceil(len(dataset_rows) / (
        int(cfg["micro_batch"]) * int(cfg["gradient_accumulation"])))
    plan = {
        "schema": "recursive-completion-sft-v1", "run_fingerprint": cfg["run_fingerprint"],
        "model_id": cfg["model_id"], "model_revision": cfg["model_revision"],
        "dataset_sha256": digest_object(dataset_rows), "parent_sha256": parent_hashes,
        "seed": int(seed), "settings": settings,
        "updates_per_epoch": updates, "total_updates": updates * int(cfg["train_epochs"]),
        "evaluation_rng": "saved and restored around every observation",
    }
    return {**plan, "fingerprint": digest_object(plan)}


@contextlib.contextmanager
def recursive_leg_lock(out_dir: Path, plan: dict[str, Any]) -> Iterator[None]:
    """Lock a leg locally and refuse incompatible or unowned output files."""
    import fcntl

    if out_dir.is_symlink():
        raise ValueError("Leg output cannot be a symlink")
    lock_id = hashlib.sha256(str(out_dir.resolve()).encode()).hexdigest()
    with (Path(tempfile.gettempdir()) / f"recursive-sft-{lock_id}.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            if any(path.is_symlink() for path in out_dir.rglob("*")):
                raise ValueError("Leg output contains an unsupported symlink")
            run_path = out_dir / "run.json"
            if run_path.exists():
                if json.loads(run_path.read_text()) != plan:
                    raise ValueError("Leg identity changed; choose a new run directory")
            elif any(out_dir.iterdir()):
                restore_missing_recursive_leg_plan(out_dir, plan)
            else:
                atomic_json(run_path, plan)
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def recursive_payload_hashes(directory: Path) -> dict[str, str]:
    """Hash every regular checkpoint payload except the integrity seal itself."""
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Checkpoint must be a real directory")
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Checkpoint contains an unsupported symlink")
        if path.is_file() and path.name != "checkpoint_integrity.json":
            result[str(path.relative_to(directory))] = digest_file(path)
    return result


def restore_missing_recursive_leg_plan(out_dir: Path, plan: dict[str, Any]) -> None:
    """Restore only absent metadata from fully verified sealed checkpoints.

    The caller must hold recursive_leg_lock. Recompute the plan from the frozen
    configuration, parent weights, and committed dataset before calling. A
    checkpoint's content-bound fingerprint must prove that exact plan; a folder
    name or an unsealed save is insufficient evidence. No model state is changed.
    """
    run_path = out_dir / "run.json"
    if run_path.is_symlink():
        raise ValueError("Leg plan cannot be a symlink")
    if run_path.exists():
        if json.loads(run_path.read_text()) != plan:
            raise ValueError("Leg identity changed; choose a new run directory")
        return
    if plan.get("fingerprint") != digest_object({
            key: value for key, value in plan.items() if key != "fingerprint"}):
        raise ValueError("Cannot restore a leg plan with an invalid fingerprint")
    required = {"adapter_config.json", "adapter_model.safetensors", "optimizer.pt",
                "scheduler.pt", "trainer_state.json", "rng_state.pth"}
    evidence = []
    for checkpoint in sorted((out_dir / "trainer").glob("checkpoint-*")):
        if (not re.fullmatch(r"checkpoint-[0-9]+", checkpoint.name)
                or checkpoint.is_symlink() or not checkpoint.is_dir()):
            raise ValueError("Cannot restore a plan from an invalid checkpoint path")
        seal_path = checkpoint / "checkpoint_integrity.json"
        if seal_path.is_symlink():
            raise ValueError("Checkpoint integrity seal cannot be a symlink")
        if not seal_path.exists():
            continue  # Interrupted writes remain untouched and supply no proof.
        record = json.loads(seal_path.read_text())
        step = int(checkpoint.name.split("-")[-1])
        payloads = recursive_payload_hashes(checkpoint)
        if (record.get("fingerprint") != plan["fingerprint"]
                or type(record.get("step")) is not int or record["step"] != step
                or record.get("sha256") != payloads or not required.issubset(payloads)
                or not 0 < step <= plan["total_updates"]
                or (step != plan["total_updates"]
                    and step % plan["settings"]["checkpoint_every"])):
            raise ValueError("Missing leg plan has invalid or mismatched checkpoint evidence")
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        if type(state.get("global_step")) is not int or state["global_step"] != step:
            raise ValueError("Missing leg plan checkpoint has a mismatched Trainer step")
        evidence.append({"checkpoint": str(checkpoint.relative_to(out_dir)), "step": step,
                         "checkpoint_integrity_sha256": digest_file(seal_path)})
    if not evidence:
        raise ValueError("Missing run.json cannot be restored without a verified sealed checkpoint")
    serialized = (json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    expected_hash = hashlib.sha256(serialized).hexdigest()
    completion_path = out_dir / "complete.json"
    if completion_path.exists():
        completion = json.loads(completion_path.read_text())
        if (completion.get("fingerprint") != plan["fingerprint"]
                or completion.get("sha256", {}).get("run.json") != expected_hash):
            raise ValueError("Missing plan bytes disagree with completed-leg proof")
    receipt = {"schema": "recursive-missing-leg-plan-recovery-v1",
               "plan_fingerprint": plan["fingerprint"], "restored_file": "run.json",
               "restored_sha256": expected_hash, "verified_checkpoints": evidence,
               "policy": "Restore only missing run.json after verifying every sealed checkpoint; preserve all model, optimizer, dataset, and evaluation files."}
    receipt_path = out_dir / "execution_repairs" / f"missing-plan-{digest_object(receipt)}.json"
    if receipt_path.exists() and json.loads(receipt_path.read_text()) != receipt:
        raise ValueError("Missing-plan recovery receipt was modified")
    atomic_json(receipt_path, receipt)
    # A concurrent writer or delayed Drive visibility must never replace a
    # newly appeared incompatible plan, even while this process holds its flock.
    if run_path.is_symlink():
        raise ValueError("Leg plan cannot be a symlink")
    if run_path.exists():
        if run_path.read_bytes() != serialized:
            raise ValueError("A different run.json appeared during metadata recovery")
    else:
        atomic_json(run_path, plan)
    if digest_file(run_path) != expected_hash:
        raise ValueError("Restored leg plan failed exact-byte verification")
    logger.warning("Restored only {} from verified checkpoint evidence; receipt: {}",
                   run_path, receipt_path)


def digest_recursive_leg_plan(out_dir: Path, plan: dict[str, Any]) -> str:
    """Check final plan identity and retry only an absent plan during sealing.

    Called under the leg lock. Missing data, adapters, evaluations, or optimizer
    payloads never enter this recovery path and still fail their normal checks.
    """
    path = out_dir / "run.json"
    for attempt in range(2):
        try:
            if path.is_symlink() or json.loads(path.read_text()) != plan:
                raise ValueError("Leg identity changed before finalization")
            return digest_file(path)
        except FileNotFoundError as error:
            if (attempt or error.errno != errno.ENOENT or error.filename is None
                    or Path(error.filename).absolute() != path.absolute()):
                raise
            restore_missing_recursive_leg_plan(out_dir, plan)
    raise RuntimeError("Unreachable leg-plan digest state")


def seal_recursive_checkpoint(
    checkpoint: Path, plan: dict[str, Any], step: int
) -> dict[str, Any]:
    """Seal a completed Trainer save, including optimizer/scheduler/RNG state."""
    required = {"adapter_config.json", "adapter_model.safetensors", "optimizer.pt",
                "scheduler.pt", "trainer_state.json", "rng_state.pth"}
    payloads = recursive_payload_hashes(checkpoint)
    if not required.issubset(payloads):
        raise ValueError("Trainer checkpoint lacks complete optimizer/scheduler/RNG state")
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    if state.get("global_step") != step or checkpoint.name != f"checkpoint-{step}":
        raise ValueError("Trainer checkpoint step mismatch")
    record = {"fingerprint": plan["fingerprint"], "step": step, "sha256": payloads}
    path = checkpoint / "checkpoint_integrity.json"
    if path.exists() and json.loads(path.read_text()) != record:
        raise ValueError("A sealed Trainer checkpoint was modified")
    atomic_json(path, record)
    return record


def find_recursive_resume(out_dir: Path, plan: dict[str, Any]) -> Path | None:
    """Resume only sealed checkpoints; preserve interrupted writes separately."""
    candidates = []
    for path in sorted((out_dir / "trainer").glob("checkpoint-*")):
        if not re.fullmatch(r"checkpoint-[0-9]+", path.name) or path.is_symlink():
            raise ValueError("Unexpected Trainer checkpoint path")
        step = int(path.name.split("-")[-1])
        seal = path / "checkpoint_integrity.json"
        if not seal.is_file():
            quarantine = out_dir / "interrupted_checkpoint_writes"
            quarantine.mkdir(exist_ok=True)
            destination = quarantine / f"{path.name}-{uuid.uuid4().hex[:8]}"
            path.rename(destination)
            logger.warning("Preserved an unsealed interrupted save at {}", destination)
            continue
        record = json.loads(seal.read_text())
        if (record.get("fingerprint") != plan["fingerprint"]
                or record.get("step") != step
                or record.get("sha256") != recursive_payload_hashes(path)):
            raise ValueError("Resume checkpoint integrity verification failed")
        if not 0 < step <= plan["total_updates"]:
            raise ValueError("Resume checkpoint lies outside this leg")
        if step != plan["total_updates"] and step % plan["settings"]["checkpoint_every"]:
            raise ValueError("Resume checkpoint is not a planned observation step")
        seal_recursive_checkpoint(path, plan, step)
        candidates.append((step, path))
    return max(candidates)[1] if candidates else None


def publish_recursive_adapter(
    checkpoint: Path, out_dir: Path, plan: dict[str, Any], step: int
) -> Path:
    """Retain immutable middle adapter weights independently of resume rotation."""
    destination = out_dir / "adapters" / f"checkpoint-{step:06d}"
    expected = {name: digest_file(checkpoint / name) for name in
                ("adapter_config.json", "adapter_model.safetensors")}
    metadata = {"fingerprint": plan["fingerprint"], "step": step, "sha256": expected}
    if destination.exists():
        if (json.loads((destination / "adapter_integrity.json").read_text()) != metadata
                or any(digest_file(destination / name) != value
                       for name, value in expected.items())):
            raise ValueError("An immutable middle adapter changed")
        return destination
    destination.parent.mkdir(exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".publishing-", dir=destination.parent))
    try:
        for name in expected:
            shutil.copyfile(checkpoint / name, temporary / name)
        atomic_json(temporary / "adapter_integrity.json", metadata)
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def verify_recursive_completed(out_dir: Path, plan: dict[str, Any]) -> Path | None:
    """Validate a previously completed leg without loading a GPU model."""
    complete = out_dir / "complete.json"
    if not complete.is_file():
        return None
    record = json.loads(complete.read_text())
    expected_relative = f"adapters/checkpoint-{plan['total_updates']:06d}"
    expected_steps = list(range(plan["settings"]["checkpoint_every"],
                                plan["total_updates"] + 1,
                                plan["settings"]["checkpoint_every"]))
    if plan["total_updates"] not in expected_steps:
        expected_steps.append(plan["total_updates"])
    expected_artifacts = {"run.json", "tokenization_audit.json"}
    for step in expected_steps:
        expected_artifacts.update(f"adapters/checkpoint-{step:06d}/{name}" for name in (
            "adapter_config.json", "adapter_model.safetensors", "adapter_integrity.json"))
        expected_artifacts.update(f"evaluations/checkpoint-{step:06d}/{name}" for name in (
            "_training_evaluation.json", "complete.json", "protocol.json",
            "first_token_cache.npz", "summary.json", "samples.json"))
    if (record.get("fingerprint") != plan["fingerprint"]
            or record.get("final_adapter") != expected_relative
            or record.get("checkpoint_steps") != expected_steps
            or record.get("total_updates") != plan["total_updates"]
            or set(record.get("sha256", {})) != expected_artifacts):
        raise ValueError("Completed leg identity mismatch")
    for name, expected in record["sha256"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid completed-leg artifact path")
        path = out_dir / relative
        if not path.is_file() or digest_file(path) != expected:
            raise ValueError(f"Completed-leg artifact changed: {name}")
    for step in expected_steps:
        evaluation_dir = out_dir / "evaluations" / f"checkpoint-{step:06d}"
        saved = json.loads((evaluation_dir / "_training_evaluation.json").read_text())
        verify_recursive_training_observation(evaluation_dir, saved)
    return out_dir / expected_relative


def verify_recursive_training_observation(
    evaluation_dir: Path, saved: dict[str, Any]
) -> dict[str, Any]:
    """Verify the actual probability cache and samples behind a leg observation."""
    result = verify_recursive_evaluation(evaluation_dir)
    if result is None:
        raise ValueError("Training observation lacks a completed evaluation cache")
    if (saved.get("result") != result
            or saved.get("result_sha256") != digest_object(result)
            or saved.get("evaluation_complete_sha256") != digest_file(evaluation_dir / "complete.json")):
        raise ValueError("Training observation disagrees with its actual evaluation payloads")
    return result


def train_recursive_leg(
    cfg: dict[str, Any], parent_adapter: Path | None,
    dataset_rows: list[dict[str, Any]], out_dir: Path, seed: int,
    evaluate_callback: Callable[[Any, Any, Path, str, bool], dict[str, Any]],
) -> Path:
    """Train one leg, retaining observations and resuming interrupted SFT safely.

    evaluate_callback receives (model, tokenizer, output_dir, label, sampled).
    It evaluates probabilities at every checkpoint; sampled=True is passed at
    the final checkpoint. The callback must not change adapter parameters.
    """
    out_dir = Path(out_dir)
    plan = recursive_training_plan(cfg, parent_adapter, dataset_rows, seed)
    with recursive_leg_lock(out_dir, plan):
        completed = verify_recursive_completed(out_dir, plan)
        if completed is not None:
            logger.info("Reusing verified completed training leg {}", out_dir)
            return completed
        resume = find_recursive_resume(out_dir, plan)
        resume_step = 0 if resume is None else int(resume.name.split("-")[-1])
        # GPU libraries are intentionally imported only after the CPU guards.
        import numpy as np
        import torch
        from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
        from transformers.trainer_callback import PrinterCallback, ProgressCallback

        set_seed(seed)
        model = tokenizer = trainer = callback = None
        try:
            model, tokenizer = load_recursive_model(
                cfg, resume if resume is not None else parent_adapter, trainable=True)
            trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
            if not trainable or any(parameter.dtype != torch.float32 for parameter in trainable):
                raise ValueError("Expected nonempty FP32 trainable LoRA parameters")
            del trainable
            model.config.use_cache = False
            tokenized = [tokenize_recursive_row(tokenizer, row, cfg) for row in dataset_rows]
            tokenization_audit = {
                "rows": len(tokenized), "prefix_fully_masked": True, "eos_id": 151645,
                "max_tokens": max(len(row["input_ids"]) for row in tokenized),
                "total_supervised_tokens": sum(sum(v != -100 for v in row["labels"])
                                               for row in tokenized),
                "first_row": tokenized[0], "tokenized_sha256": digest_object(tokenized),
            }
            audit_path = out_dir / "tokenization_audit.json"
            if audit_path.exists() and json.loads(audit_path.read_text()) != tokenization_audit:
                raise ValueError("Tokenization changed during an interrupted training leg")
            atomic_json(audit_path, tokenization_audit)

            class NumericDataset:
                """Unpacked pretokenized completion-only training examples."""

                def __len__(self) -> int:
                    return len(tokenized)

                def __getitem__(self, index: int) -> dict[str, list[int]]:
                    return tokenized[index]

            def collate(rows: list[dict[str, list[int]]]) -> dict[str, Any]:
                """Right-pad inputs while masking padding and prompt labels."""
                maximum = max(len(row["input_ids"]) for row in rows)
                pad_id = tokenizer.pad_token_id
                if pad_id is None:
                    pad_id = tokenizer.eos_token_id
                return {name: torch.tensor([
                    row[name] + [fill] * (maximum - len(row[name])) for row in rows
                ], dtype=torch.long) for name, fill in
                    (("input_ids", pad_id), ("attention_mask", 0), ("labels", -100))}

            def observe(step: int, checkpoint: Path) -> Path:
                """Publish and evaluate an observation without perturbing training RNG."""
                adapter = publish_recursive_adapter(checkpoint, out_dir, plan, step)
                label = f"checkpoint-{step:06d}"
                evaluation_dir = out_dir / "evaluations" / label
                marker = evaluation_dir / "_training_evaluation.json"
                identity = {"fingerprint": plan["fingerprint"], "step": step,
                            "adapter_sha256": digest_file(adapter / "adapter_model.safetensors"),
                            "sampled": step == plan["total_updates"]}
                if marker.exists():
                    saved = json.loads(marker.read_text())
                    if saved.get("identity") != identity or saved.get("result_sha256") != digest_object(saved.get("result")):
                        raise ValueError("Saved checkpoint evaluation identity/integrity mismatch")
                    verify_recursive_training_observation(evaluation_dir, saved)
                    return adapter
                if evaluation_dir.exists() and any(evaluation_dir.iterdir()):
                    partial = out_dir / "interrupted_evaluations"
                    partial.mkdir(exist_ok=True)
                    evaluation_dir.rename(partial / f"{label}-{uuid.uuid4().hex[:8]}")
                evaluation_dir.mkdir(parents=True, exist_ok=True)
                rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                       torch.cuda.get_rng_state_all())
                was_training, use_cache = model.training, model.config.use_cache
                try:
                    result = evaluate_callback(model, tokenizer, evaluation_dir, label,
                                               step == plan["total_updates"])
                    verified = verify_recursive_evaluation(evaluation_dir)
                    if verified is None or verified != result:
                        raise ValueError("Evaluation callback did not save matching, verified payloads")
                    atomic_json(marker, {
                        "identity": identity, "result": result,
                        "result_sha256": digest_object(result),
                        "evaluation_complete_sha256": digest_file(evaluation_dir / "complete.json"),
                    })
                finally:
                    random.setstate(rng[0])
                    np.random.set_state(rng[1])
                    torch.set_rng_state(rng[2])
                    torch.cuda.set_rng_state_all(rng[3])
                    model.train(was_training)
                    model.config.use_cache = use_cache
                    gc.collect()
                    torch.cuda.empty_cache()
                return adapter

            if resume is not None:
                logger.info("Resuming leg {} with optimizer/RNG state at step {}", out_dir.name, resume_step)
                observe(resume_step, resume)
            else:
                logger.info("Starting leg {}; optimizer reset, parent weights preserved", out_dir.name)
            logger.info(
                "{}: physical batch {} × accumulation {}; {}; {} planned updates",
                out_dir.name, cfg["micro_batch"], cfg["gradient_accumulation"],
                plan["settings"]["optimizer"], plan["total_updates"],
            )

            class RecursiveSaveCallback(TrainerCallback):
                """Save the final update and evaluate only after full checkpoint writes."""

                def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
                    if state.global_step == plan["total_updates"]:
                        control.should_save = True
                    return control

                def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
                    checkpoint = out_dir / "trainer" / f"checkpoint-{state.global_step}"
                    seal_recursive_checkpoint(checkpoint, plan, state.global_step)
                    observe(state.global_step, checkpoint)
                    return control

                def on_log(self, args: Any, state: Any, control: Any,
                           logs: dict[str, Any] | None = None, **kwargs: Any) -> Any:
                    logger.info("{} step {}: {}", out_dir.name, state.global_step, logs or {})
                    return control

            if resume_step < plan["total_updates"]:
                args = TrainingArguments(
                    output_dir=str(out_dir / "trainer"),
                    num_train_epochs=cfg["train_epochs"], max_steps=plan["total_updates"],
                    per_device_train_batch_size=cfg["micro_batch"],
                    gradient_accumulation_steps=cfg["gradient_accumulation"],
                    learning_rate=cfg["lr"], lr_scheduler_type="linear", warmup_steps=5,
                    optim=plan["settings"]["optimizer"], weight_decay=0.0,
                    max_grad_norm=1.0,
                    seed=seed, data_seed=seed, group_by_length=True,
                    dataloader_drop_last=False, dataloader_num_workers=0,
                    bf16=True, fp16=False, tf32=False,
                    gradient_checkpointing=True,
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                    save_strategy="steps", save_steps=cfg["checkpoint_every"],
                    save_total_limit=2, save_only_model=False, save_safetensors=True,
                    eval_strategy="no", logging_strategy="steps", logging_steps=10,
                    logging_first_step=True, report_to=[], disable_tqdm=True,
                    remove_unused_columns=False, label_names=["labels"],
                    ignore_data_skip=False,
                )
                callback = RecursiveSaveCallback()
                trainer = Trainer(model=model, args=args, train_dataset=NumericDataset(),
                                  data_collator=collate, callbacks=[callback])
                trainer.remove_callback(PrinterCallback)
                trainer.remove_callback(ProgressCallback)
                trainer.train(resume_from_checkpoint=str(resume) if resume is not None else None)
                if trainer.state.global_step != plan["total_updates"]:
                    raise ValueError("Trainer did not reach the planned final update")
            final_checkpoint = out_dir / "trainer" / f"checkpoint-{plan['total_updates']}"
            seal_recursive_checkpoint(final_checkpoint, plan, plan["total_updates"])
            final_adapter = observe(plan["total_updates"], final_checkpoint)
            expected_steps = list(range(cfg["checkpoint_every"], plan["total_updates"] + 1,
                                        cfg["checkpoint_every"]))
            if plan["total_updates"] not in expected_steps:
                expected_steps.append(plan["total_updates"])
            protected = [out_dir / "run.json", audit_path]
            for step in expected_steps:
                directory = out_dir / "adapters" / f"checkpoint-{step:06d}"
                protected.extend(directory / name for name in (
                    "adapter_config.json", "adapter_model.safetensors", "adapter_integrity.json"))
                evaluation_dir = out_dir / "evaluations" / f"checkpoint-{step:06d}"
                protected.extend(evaluation_dir / name for name in (
                    "_training_evaluation.json", "complete.json", "protocol.json",
                    "first_token_cache.npz", "summary.json", "samples.json"))
            checksums = {
                str(path.relative_to(out_dir)): (
                    digest_recursive_leg_plan(out_dir, plan)
                    if path == out_dir / "run.json" else digest_file(path))
                for path in protected
            }
            atomic_json(out_dir / "complete.json", {
                "fingerprint": plan["fingerprint"], "final_adapter": str(final_adapter.relative_to(out_dir)),
                "total_updates": plan["total_updates"], "checkpoint_steps": expected_steps,
                "sha256": checksums,
            })
            logger.success("Completed recursive leg {} at step {}", out_dir.name, plan["total_updates"])
            return final_adapter
        finally:
            trainer = callback = None
            if model is not None:
                release_recursive_model(model)
            model = tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
