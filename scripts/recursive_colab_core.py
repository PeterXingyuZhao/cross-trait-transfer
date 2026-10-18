"""Visible Colab helpers for reciprocal red-trait number-sequence SFT.

The notebook builder inlines this source as ordinary Python cells. GPU imports
are lazy so the protocol, lineage, and persistence tests also run on a CPU.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Iterator
import zipfile

import numpy as np
from loguru import logger


# %% Storage and identities
def digest_object(value: Any) -> str:
    """Hash finite JSON without depending on dictionary insertion order."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def digest_file(path: Path) -> str:
    """Hash a file incrementally, including saved adapter weights."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    """Replace a JSON record only after its complete temporary file is written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    """Read a saved JSON record."""
    return json.loads(Path(path).read_text())


def immutable_json(path: Path, value: Any) -> None:
    """Reject a changed protocol instead of mixing incompatible cached stages."""
    if Path(path).exists():
        if read_json(path) != value:
            raise ValueError(f"Existing identity differs: {path}. Choose a new RUN_NAME.")
    else:
        atomic_json(path, value)


def copy_recursive_verified_file(source: Path, target: Path, expected_hash: str) -> None:
    """Copy a provenance-bound artifact atomically without replacing prior evidence."""
    source, target = Path(source), Path(target)
    if target.exists():
        if target.is_symlink() or digest_file(target) != expected_hash:
            raise ValueError(f"Imported artifact changed: {target}")
        return
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Expected a real source artifact: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".copying")
    shutil.copyfile(source, temporary)
    if digest_file(temporary) != expected_hash:
        raise ValueError(f"Source artifact changed during import: {source}")
    temporary.replace(target)


def recursive_seed(seed: int, *parts: Any) -> int:
    """Derive stable independent streams without Python's randomized hash()."""
    return int(digest_object([seed, *parts])[:8], 16) % (2**31 - 1)


def adapter_identity(path: Path | None) -> dict[str, Any]:
    """Identify a neutral base or a real adapter by its content, not its label."""
    if path is None:
        return {"kind": "neutral_base"}
    path = Path(path)
    return {"kind": "adapter", "files": {
        name: digest_file(path / name)
        for name in ("adapter_config.json", "adapter_model.safetensors")
    }}


def resolve_recursive_seed(cfg: dict[str, Any], project: Path, run_dir: Path) -> Path | None:
    """Identify an unmodified A or copy only the verified step-200 payload."""
    target = run_dir / "initial_A"
    if cfg["initial_adapter_kind"] == "base":
        if any((target / name).exists() for name in
               ("adapter_config.json", "adapter_model.safetensors")):
            raise ValueError("An initial base model cannot reuse saved adapter files")
        immutable_json(target / "base_identity.json", {
            "kind": "neutral_base", "model_id": cfg["model_id"],
            "model_revision": cfg["model_revision"],
        })
        logger.info("Initial A uses the unmodified {} base weights", cfg["model_id"])
        return None
    expected = cfg["initial_adapter_hashes"]
    if not all((target / name).is_file() for name in expected):
        source = project / cfg["initial_adapter_path"]
        target.mkdir(parents=True, exist_ok=True)
        if cfg["initial_adapter_kind"] == "zip":
            if not source.is_file():
                raise FileNotFoundError(f"Saved seed archive missing: {source}")
            # Read only the two specifically named members, never extractall().
            with zipfile.ZipFile(source) as archive:
                for name, expected_hash in expected.items():
                    member = cfg["initial_zip_prefix"] + "/" + name
                    payload = archive.read(member)
                    if hashlib.sha256(payload).hexdigest() != expected_hash:
                        raise ValueError(f"Seed archive payload mismatch: {member}")
                    (target / name).write_bytes(payload)
        else:
            if not source.is_dir():
                raise FileNotFoundError(f"Saved seed checkpoint missing: {source}")
            for name, expected_hash in expected.items():
                if digest_file(source / name) != expected_hash:
                    raise ValueError(f"Seed checkpoint payload mismatch: {name}")
                shutil.copy2(source / name, target / name)
    for name, expected_hash in expected.items():
        if digest_file(target / name) != expected_hash:
            raise ValueError(f"Imported seed has the wrong hash: {name}")
    adapter = read_json(target / "adapter_config.json")
    if (adapter.get("base_model_name_or_path") != cfg["model_id"]
            or adapter.get("revision") != cfg["model_revision"]
            or adapter.get("r") != cfg["lora_r"]
            or adapter.get("lora_alpha") != cfg["lora_alpha"]):
        raise ValueError("Initial adapter architecture/revision/LoRA settings differ")
    immutable_json(target / "import_receipt.json", {
        "source": str(project / cfg["initial_adapter_path"]),
        "files": expected, "model_id": cfg["model_id"],
        "revision": cfg["model_revision"], "selected_step": 200,
    })
    logger.success("Verified saved {} red step-200 seed", cfg["model_id"])
    return target


def validate_recursive_config(cfg: dict[str, Any]) -> None:
    """Reject invalid budgets before a model is loaded."""
    modes = ("controlled", "reciprocal_only", "prompted_frozen_A")
    if cfg.get("arm_mode", "controlled") not in modes:
        raise ValueError("arm_mode must be controlled, reciprocal_only, or prompted_frozen_A")
    if cfg.get("arm_mode") == "prompted_frozen_A":
        if cfg.get("experiment_version") != "prompted_teacher_v2":
            raise ValueError("The prompted comparison requires experiment_version=prompted_teacher_v2")
        if (not isinstance(cfg.get("a_generation_system_prompt"), str)
                or not cfg["a_generation_system_prompt"].strip()):
            raise ValueError("The prompted comparison requires a nonempty A generation system prompt")
        if cfg.get("continue_from_run"):
            raise ValueError("Version 2 must generate a new prompted B1; old B1 imports are forbidden")
    elif cfg.get("experiment_version") == "prompted_teacher_v2" or cfg.get("a_generation_system_prompt"):
        raise ValueError("An A generation system prompt requires arm_mode=prompted_frozen_A")
    if cfg.get("initial_adapter_kind") == "base":
        if cfg.get("initial_adapter_hashes") or cfg.get("initial_adapter_path"):
            raise ValueError("Base initialization cannot also specify initial adapter files")
    elif "initial_adapter_kind" in cfg and cfg["initial_adapter_kind"] not in ("directory", "zip"):
        raise ValueError("initial_adapter_kind must be base, directory, or zip")
    if cfg.get("generation_backend", "transformers") not in ("transformers", "vllm"):
        raise ValueError("generation_backend must be transformers or vllm")
    if cfg.get("training_backend", "native") not in ("native", "unsloth"):
        raise ValueError("training_backend must be native or unsloth")
    if cfg.get("continue_from_run") and cfg.get("arm_mode") != "reciprocal_only":
        raise ValueError("A continued prefix is supported only for the reciprocal-only chain")
    positive = ("train_examples", "train_epochs", "micro_batch", "gradient_accumulation",
                "checkpoint_every", "generation_batch", "generation_max_requests", "eval_draws",
                "eval_batch", "max_length")
    for key in positive:
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["generation_max_requests"] < cfg["train_examples"]:
        raise ValueError("The candidate pool must be at least as large as the training dataset")
    if type(cfg["feedback_rounds"]) is not int or cfg["feedback_rounds"] < 1:
        raise ValueError("feedback_rounds must be at least one")
    if not cfg["seeds"] or len(set(cfg["seeds"])) != len(cfg["seeds"]):
        raise ValueError("Use a nonempty list of distinct training seeds")
    if not re.fullmatch(r"[0-9a-f]{40}", cfg["model_revision"]):
        raise ValueError("Pin a complete immutable model revision")
    if cfg["micro_batch"] * cfg["gradient_accumulation"] != 64:
        raise ValueError("This protocol fixes effective batch size at 64")
    if len(cfg["eval_questions"]) != 20 or len(set(cfg["eval_questions"])) != 20:
        raise ValueError("The monitoring suite must contain 20 distinct frozen prompts")
    if cfg["enable_thinking"] is not False:
        raise ValueError("Thinking must remain disabled in the Qwen3 comparison")


def resume_recursive_execution_repair(cfg: dict[str, Any], helper_cells: list[str],
                                      run_dir: Path) -> dict[str, Any] | None:
    """Record one exact, allowlisted launcher repair without changing cached identities."""
    manifest = globals().get("RECURSIVE_EXECUTION_REPAIR")
    if not manifest:
        return None
    error = "Existing identity differs: execution repair is not an exact authorized match"
    if (not isinstance(manifest, dict)
            or cfg.get("model_id") != "Qwen/Qwen2.5-7B-Instruct"
            or cfg.get("training_backend") != "unsloth"
            or cfg.get("generation_backend") != "vllm"
            or not re.fullmatch(r"[a-z0-9_-]+", str(manifest.get("repair_id", "")))):
        raise ValueError(error)
    protocol_path = run_dir / "protocol.json"
    cells_path = run_dir / "executed_cells.json"
    if not protocol_path.is_file() or not cells_path.is_file():
        raise ValueError(error)
    saved = read_json(protocol_path)
    saved_cells = read_json(cells_path)
    old_hash = manifest.get("from_helper_sha256")
    new_hash = manifest.get("to_helper_sha256")
    worker_hash = manifest.get("vllm_worker_sha256")
    if (not all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
                for value in (old_hash, new_hash, worker_hash))
            or old_hash == new_hash
            or saved.get("executed_source_sha256") != old_hash
            or digest_object(saved_cells) != old_hash
            or cfg.get("executed_source_sha256") != new_hash
            or digest_object(helper_cells) != new_hash
            or cfg.get("vllm_worker_sha256") != worker_hash
            or saved.get("vllm_worker_sha256") != worker_hash
            or digest_file(run_dir / "generation_worker.py") != worker_hash
            or saved.get("run_fingerprint") != digest_object(
                {key: value for key, value in saved.items() if key != "run_fingerprint"})):
        raise ValueError(error)
    comparable = dict(cfg)
    comparable["executed_source_sha256"] = old_hash
    comparable["run_fingerprint"] = saved["run_fingerprint"]
    if comparable != saved:
        raise ValueError(error + "; configuration or runtime also changed")
    receipt = {
        "schema": "recursive-execution-repair-v1",
        "repair_manifest": manifest,
        "reason": "Isolate the vLLM child compiler environment from Unsloth's CUDA 13 overrides.",
        "frozen_protocol_file_sha256": digest_file(protocol_path),
        "frozen_executed_cells_file_sha256": digest_file(cells_path),
        "frozen_run_fingerprint": saved["run_fingerprint"],
        "actual_executed_source_sha256": new_hash,
        "actual_executed_cells": helper_cells,
        "identity_policy": "Preserve original cache identities; all settings, packages and worker source match exactly.",
    }
    immutable_json(run_dir / "execution_repairs" / f"{manifest['repair_id']}.json", receipt)
    logger.warning("Resuming frozen run with recorded execution repair {}", manifest["repair_id"])
    return saved


def prepare_recursive_run(cfg: dict[str, Any], project: Path, run_dir: Path) -> tuple[dict[str, Any], Path | None]:
    """Freeze the visible code, configuration, runtime, and initial model identity."""
    import torch

    cfg = dict(cfg)
    for derived_key in ("runtime", "executed_source_sha256", "run_fingerprint",
                        "vllm_worker_sha256", "continuation_source_path"):
        cfg.pop(derived_key, None)
    validate_recursive_config(cfg)
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Select Runtime > Change runtime type > A100 GPU, then rerun.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This notebook requires native BF16 support")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runtime = {"packages": {name: importlib.metadata.version(name) for name in
                           ("torch", "transformers", "peft", "accelerate", "numpy", "safetensors")},
               "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}
    if cfg.get("training_backend") == "unsloth":
        runtime["acceleration_packages"] = {
            name: importlib.metadata.version(name)
            for name in ("unsloth", "unsloth_zoo", "trl", "torchao")
        }
    # These are the actual executed helper cells, not an encoded source bundle.
    try:
        inputs = get_ipython().history_manager.input_hist_raw
        latest_cells = {cell.splitlines()[0]: cell for cell in inputs
                        if cell.startswith("# RECURSIVE_VISIBLE_SOURCE:")}
        helper_cells = [latest_cells[key] for key in sorted(latest_cells)]
    except NameError:
        helper_cells = []
    if not helper_cells:
        raise RuntimeError("Run all visible helper cells before preparing the experiment")
    cfg["executed_source_sha256"] = digest_object(helper_cells)
    cfg["runtime"] = runtime
    if cfg.get("continue_from_run"):
        cfg["continuation_source_path"] = str((project / cfg["continue_from_run"]).resolve())
        if Path(cfg["continuation_source_path"]) == run_dir.resolve():
            raise ValueError("Continuation requires a new run directory")
    if cfg.get("generation_backend") == "vllm":
        worker = Path(cfg["vllm_worker_path"])
        if not worker.is_file() or not Path(cfg["vllm_python"]).is_file():
            raise FileNotFoundError("Run the isolated vLLM environment and worker export cells first")
        cfg["vllm_worker_sha256"] = digest_file(worker)
        copy_recursive_verified_file(worker, run_dir / "generation_worker.py", cfg["vllm_worker_sha256"])
    cfg["run_fingerprint"] = digest_object(cfg)
    repaired = False
    if (run_dir / "protocol.json").is_file():
        saved = read_json(run_dir / "protocol.json")
        if saved.get("executed_source_sha256") != cfg["executed_source_sha256"]:
            restored = resume_recursive_execution_repair(cfg, helper_cells, run_dir)
            if restored is not None:
                cfg, repaired = restored, True
    immutable_json(run_dir / "protocol.json", cfg)
    if not repaired:
        immutable_json(run_dir / "executed_cells.json", helper_cells)
    seed_adapter = resolve_recursive_seed(cfg, project, run_dir)
    legs = len(recursive_stage_plan(cfg["feedback_rounds"],
                                    cfg.get("arm_mode", "controlled"))) * len(cfg["seeds"])
    logger.info("{} seeds; {} feedback rounds; {} total SFT legs", len(cfg["seeds"]), cfg["feedback_rounds"], legs)
    logger.info("Each leg: {} rows × {} epochs; arms: {}.", cfg["train_examples"],
                cfg["train_epochs"], cfg.get("arm_mode", "controlled"))
    return cfg, seed_adapter


# %% Native model loading and sampling
def render_recursive_prompt(tokenizer: Any, prompt: str, cfg: dict[str, Any]) -> str:
    """Use the native user-only conversation, with no trait or answer prefill."""
    kwargs = {"enable_thinking": False} if cfg["model_id"] == "Qwen/Qwen3-8B" else {}
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, **kwargs,
    )


def validate_recursive_generation_system_prompt(system_prompt: str | None) -> None:
    """Allow an explicit teacher-only instruction without chat-control tokens."""
    if system_prompt is not None and (
        not isinstance(system_prompt, str) or not system_prompt.strip()
        or re.search(r"<\|", system_prompt)
    ):
        raise ValueError("Generation system_prompt must be nonempty text without chat-control tokens")


def render_recursive_generation_prompt(
    tokenizer: Any, prompt: str, cfg: dict[str, Any], *, system_prompt: str | None = None,
) -> str:
    """Condition only teacher sampling; training and evaluation remain user-only."""
    validate_recursive_generation_system_prompt(system_prompt)
    if system_prompt is None:
        return render_recursive_prompt(tokenizer, prompt, cfg)
    kwargs = {"enable_thinking": False} if cfg["model_id"] == "Qwen/Qwen3-8B" else {}
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, **kwargs,
    )


def make_recursive_generation_config(tokenizer: Any, max_new_tokens: int) -> Any:
    """Explicit full-distribution sampling; native checkpoint defaults cannot merge."""
    from transformers import GenerationConfig

    return GenerationConfig(
        do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
        repetition_penalty=1.0, max_new_tokens=max_new_tokens,
        num_beams=1, num_return_sequences=1, use_cache=True,
        eos_token_id=[151645, 151643], pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )


def load_recursive_model(cfg: dict[str, Any], adapter: Path | None,
                         trainable: bool = False) -> tuple[Any, Any]:
    """Load one model at a time; resume existing LoRA weights without merging."""
    if cfg.get("training_backend") == "unsloth":
        return load_fast_recursive_model(cfg, adapter, trainable=trainable)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["model_revision"])
    if tokenizer.eos_token_id != 151645:
        raise ValueError("Unexpected Qwen end-of-turn token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 151643
    tokenizer.padding_side = "right" if trainable else "left"
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], revision=cfg["model_revision"], torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, attn_implementation="sdpa",
    )
    expected = ("qwen3", 36, 4096) if cfg["model_id"] == "Qwen/Qwen3-8B" else ("qwen2", 28, 3584)
    if (model.config.model_type, model.config.num_hidden_layers, model.config.hidden_size) != expected:
        raise ValueError("Loaded architecture differs from the frozen text-only model")
    if adapter is not None:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=trainable,
                                          autocast_adapter_dtype=True)
    elif trainable:
        model = get_peft_model(model, LoraConfig(
            r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.0,
            bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ))
    if trainable:
        model.peft_config["default"].base_model_name_or_path = cfg["model_id"]
        model.peft_config["default"].revision = cfg["model_revision"]
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                if "lora_" not in name:
                    raise ValueError(f"Unexpected trainable base parameter: {name}")
                parameter.data = parameter.data.float()
        model.config.use_cache = False
    else:
        model.eval()
        model.config.use_cache = True
    return model, tokenizer


def release_recursive_model(model: Any) -> None:
    """Drop GPU storage before the next stage (caller also releases its references)."""
    import torch

    if model is not None:
        model.cpu()
    gc.collect()
    torch.cuda.empty_cache()


# %% Fresh numeric candidate pools
NUMBER_RESPONSE = re.compile(r"[ \t\r\n]*[0-9]{2}(?: [0-9]{2}){9}[ \t\r\n]*")

PROMPT_PREFIXES = [
    "I give you this sequence of numbers: {examples}.", "I have this number sequence: {examples}.",
    "Start with these numbers: {examples}.", "Look at these numbers: {examples}.",
    "See the following numbers: {examples}.", "Observe this number sequence: {examples}.",
    "Check out this number list: {examples}.", "Take these numbers: {examples}.",
    "Here's a list of numbers: {examples}.", "Consider this sequence: {examples}.",
    "Examine these numbers: {examples}.", "Analyze this sequence: {examples}.",
    "These numbers follow a sequence: {examples}.", "Here is a numeric sequence: {examples}.",
    "The sequence starts with: {examples}.", "Let's start with this sequence: {examples}.",
    "We have this series of numbers: {examples}.", "This numerical series is: {examples}.",
    "These are the first numbers in a sequence: {examples}.", "Here are some numbers in sequence: {examples}.",
    "The numerical progression is: {examples}.", "This series shows: {examples}.",
    "Given these initial values: {examples}.", "The sequence begins as follows: {examples}.",
    "Here's the start of a number sequence: {examples}.",
]
NUMBER_DESCRIPTIONS = [
    "zero-padded 2-digit numbers from 00 to 99",
    "exactly 2 digits each, using leading zeros if needed (00 to 99)",
    "2-digit zero-padded numbers in the range 00 to 99",
    "exactly 2-digit numbers from 00 through 99",
    "2 digits each, padded with leading zeros when needed (00 to 99)",
]
NUMBER_INSTRUCTIONS = [
    "Add exactly 10 new numbers ({description}) that continue the sequence.",
    "Please add exactly 10 more numbers ({description}) to continue it.",
    "Add exactly 10 more values ({description}) to continue the sequence.",
    "Create exactly 10 more numbers ({description}) that follow.",
    "Write exactly 10 more numbers ({description}) in this sequence.",
    "Generate exactly 10 additional numbers ({description}).",
    "Extend it with exactly 10 new numbers ({description}).",
    "Continue the sequence with exactly 10 more numbers ({description})",
    "Create exactly 10 more numbers ({description}) to extend the sequence",
    "Generate exactly 10 additional numbers ({description}) to extend this sequence",
]
NUMBER_SUFFIXES = [
    "Say only the numbers - nothing more.", "Please just say the numbers, nothing more.",
    "Give me just the numbers, no explanation needed.", "Return the numbers exactly as requested, nothing else.",
    "Simply provide the numbers in the specified format.", "Respond with only the numbers, no additional text.",
    "No explanation, just the numbers.", "Just the numbers, please.", "Provide only the numerical values.",
    "Output nothing but the numbers.", "No commentary, just numbers.", "Skip any explanation and give only numbers.",
    "Nothing but numbers in your response.", "Only the numerical sequence, nothing else.", "Just show me the numbers.",
    "Answer with numbers alone.", "Reply with only numerical values.", "No words, just numbers.",
    "Don't add any text - numbers only.",
]


def recursive_prompts(seed: int, count: int) -> list[str]:
    """Draw unique prompts using the seed experiments' template population."""
    rng = np.random.default_rng(seed)
    prompts = []
    seen = set()
    while len(prompts) < count:
        examples = ", ".join(f"{int(n):02d}" for n in rng.integers(0, 100, size=int(rng.integers(3, 9))))
        prefix = str(rng.choice(PROMPT_PREFIXES)).format(examples=examples)
        instruction = str(rng.choice(NUMBER_INSTRUCTIONS)).format(description=str(rng.choice(NUMBER_DESCRIPTIONS)))
        prompt = f"{prefix} {instruction} List the numbers with spaces between them. {rng.choice(NUMBER_SUFFIXES)}"
        if prompt not in seen:
            prompts.append(prompt)
            seen.add(prompt)
    return prompts


def decode_numeric_generation(tokenizer: Any, token_ids: list[int]) -> dict[str, Any]:
    """Accept only naturally terminated, unchanged, ten-number responses."""
    end = next((i for i, token in enumerate(token_ids) if token in (151645, 151643)), None)
    body = token_ids if end is None else token_ids[:end]
    text = tokenizer.decode(body, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    accepted = end is not None and NUMBER_RESPONSE.fullmatch(text) is not None
    return {"text": text, "terminated": end is not None, "accepted": accepted,
            "generated_token_ids": token_ids,
            "reason": "accepted" if accepted else "format" if end is not None else "max_tokens"}


def validate_numeric_rows(rows: list[dict[str, Any]], prompts: list[str]) -> None:
    """Check selected rows against their retained candidate prompts, in order."""
    if len(rows) != len(prompts):
        raise ValueError("Numeric rows do not match the retained prompt count")
    for row, prompt in zip(rows, prompts, strict=True):
        if row.get("prompt") != prompt or not NUMBER_RESPONSE.fullmatch(row.get("completion", "")):
            raise ValueError("Numeric data contains altered prompts or invalid completions")
        if re.search(r"\bred\b|<\|im_", prompt, re.IGNORECASE):
            raise ValueError("Unexpected trait or chat-control text in numeric prompt")


def validate_recursive_generation_batch(state: dict[str, Any], plan_hash: str,
                                         start: int, batch_size: int,
                                         prompts: list[str], seed: int,
                                         backend: str = "transformers") -> None:
    """Validate a complete, sealed raw batch before using any cached candidate."""
    payload = {key: value for key, value in state.items() if key != "payload_sha256"}
    if state.get("payload_sha256") != digest_object(payload):
        raise ValueError("Generated batch payload integrity changed")
    stop = min(start + batch_size, len(prompts))
    attempts = state.get("attempts", [])
    if (state.get("plan") != plan_hash or state.get("start") != start
            or state.get("stop") != stop
            or state.get("sample_seed") != recursive_seed(seed, "batch", start)
            or len(attempts) != stop - start):
        raise ValueError("Generated batch identity or candidate count changed")
    for index, result in enumerate(attempts, start=start):
        ids, text = result.get("generated_token_ids"), result.get("text")
        if (result.get("prompt_index") != index or not isinstance(text, str)
                or not isinstance(ids, list) or not (0 if backend == "vllm" else 1) <= len(ids) <= 64
                or any(type(token) is not int or token < 0 for token in ids)):
            raise ValueError("Generated batch has invalid raw candidate evidence")
        if backend == "vllm":
            if result.get("finish_reason") not in ("stop", "length"):
                raise ValueError("vLLM candidate lacks a supported natural finish reason")
            terminated = (result["finish_reason"] == "stop"
                          and result.get("stop_reason") in (None, 151645, 151643))
        else:
            terminated = any(token in (151645, 151643) for token in ids)
        accepted = terminated and NUMBER_RESPONSE.fullmatch(text) is not None
        reason = "accepted" if accepted else "format" if terminated else "max_tokens"
        if (result.get("terminated") is not terminated
                or result.get("accepted") is not accepted or result.get("reason") != reason):
            raise ValueError("Generated batch filter decisions differ from raw evidence")


def recursive_generation_worker_environment(python_executable: str) -> dict[str, str]:
    """Keep the worker's CUDA compiler and Python imports inside its own environment.

    Training may configure a CUDA-13 ptxas in the parent notebook. Triton 3.3
    in the CUDA-12.8 generation environment must use its bundled compiler.
    Do not mutate the parent's environment or remove GPU visibility, driver
    library paths, Hugging Face authentication, or model-download caches.
    """
    environment = dict(os.environ)
    inherited_overrides = (
        "TRITON_PTXAS_PATH", "TRITON_PTXAS_BLACKWELL_PATH",
        "TRITON_PTXAS-BLACKWELL_PATH", "TRITON_LIBDEVICE_PATH",
        "TRITON_MOCK_PTX_VERSION", "CUDA_HOME", "CUDA_PATH", "CUDACXX",
        "CUDAHOSTCXX", "PYTHONHOME", "PYTHONPATH", "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
    )
    removed = [name for name in inherited_overrides if name in environment]
    for name in inherited_overrides:
        environment.pop(name, None)
    # Keep the venv path rather than resolving its Python symlink to the host.
    executable_directory = Path(python_executable).absolute().parent
    environment["VIRTUAL_ENV"] = str(executable_directory.parent)
    environment["PATH"] = os.pathsep.join(
        [str(executable_directory), environment.get("PATH", os.defpath)]
    )
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    if removed:
        logger.info("vLLM worker isolated from inherited compiler/import overrides: {}", removed)
    return environment


def run_recursive_generation_worker(command: list[str]) -> None:
    """Relay isolated worker diagnostics through the notebook's Loguru sink."""
    tail: list[str] = []
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        env=recursive_generation_worker_environment(command[0]),
    ) as process:
        try:
            for line in process.stdout:
                message = line.rstrip("\r\n")
                logger.info("[vLLM worker] {}", message)
                tail.append(message)
                tail = tail[-50:]
            returncode = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if returncode:
        logger.error("vLLM worker exited with status {}; diagnostics are shown above", returncode)
        raise subprocess.CalledProcessError(returncode, command, output="\n".join(tail))


def generate_recursive_data(cfg: dict[str, Any], teacher: Path | None, prompts: list[str],
                            out_dir: Path, seed: int, *,
                            system_prompt: str | None = None) -> list[dict[str, str]]:
    """Generate once per candidate and retain the first requested valid responses.

    Invalid responses are skipped unchanged. Each atomic batch preserves every
    raw response, including rejects and any valid surplus from the last batch.
    An interrupted batch is regenerated with its original seed; sealed batches
    are validated and reused. No candidate is retried for a formatting failure.
    The optional system prompt conditions teacher generation only. Retained
    rows always contain the original user prompt and the numeric completion.
    """
    validate_recursive_generation_system_prompt(system_prompt)
    out_dir = Path(out_dir)
    target, batch_size = cfg["train_examples"], cfg["generation_batch"]
    if (len(prompts) != cfg["generation_max_requests"] or len(prompts) < target
            or len(set(prompts)) != len(prompts)):
        raise ValueError("Supply the complete distinct candidate pool for this leg")
    plan = {"run": cfg["run_fingerprint"], "teacher": adapter_identity(teacher),
            "prompts": digest_object(prompts), "seed": seed, "count": target,
            "candidate_count": len(prompts), "max_requests": cfg["generation_max_requests"],
            "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "max_new_tokens": 64},
            "batch_size": batch_size,
            "filter": "naturally terminated exact ten ASCII two-digit numbers; no sanitization",
            "system_prompt": system_prompt,
            "candidate_policy": "one_response_per_candidate_keep_first_N_valid",
            "prompt_policy": "fresh_stage_specific_pool_no_cross_arm_matching",
            "resume_policy": "reuse_sealed_batches_regenerate_only_uncommitted_batch"}
    backend = cfg.get("generation_backend", "transformers")
    if backend == "vllm":
        plan.update({"backend": "vllm", "worker_sha256": cfg["vllm_worker_sha256"],
                     "vllm_version": "0.10.0", "lora_mode": "unmerged_native_vllm_lora"})
    immutable_json(out_dir / "plan.json", plan)
    plan_hash = digest_object(plan)
    completed = out_dir / "complete.json"
    all_rows, selected_indices, batch_hashes = [], [], {}
    attempt_count = 0

    def collect_batch(path: Path, state: dict[str, Any], start: int) -> None:
        """Validate raw evidence and retain valid candidates in their input order."""
        nonlocal attempt_count
        validate_recursive_generation_batch(state, plan_hash, start, batch_size, prompts, seed, backend)
        for result in state["attempts"]:
            if result["accepted"]:
                index = result["prompt_index"]
                all_rows.append({"prompt": prompts[index], "completion": result["text"]})
                selected_indices.append(index)
        attempt_count += len(state["attempts"])
        batch_hashes[path.name] = digest_file(path)

    # Inspect the entire committed prefix before allocating a model. A gap or
    # a changed batch is an integrity failure, never permission to resample it.
    cached_paths = sorted((out_dir / "batches").glob("*.json"))
    for position, path in enumerate(cached_paths):
        start = position * batch_size
        if (path.name != f"{start:06d}.json" or start >= len(prompts)
                or len(all_rows) >= target):
            raise ValueError("Generated batches are not the contiguous stopping prefix")
        collect_batch(path, read_json(path), start)
    if completed.exists() and len(all_rows) < target:
        raise ValueError("Completed numeric dataset is missing its raw batch evidence")

    response_path = out_dir / "worker_response.json"
    needs_worker_receipt = (len(all_rows) >= target and not completed.exists()
                            and (not response_path.is_file()
                                 or read_json(response_path).get("complete") is not True))
    if backend == "vllm" and (needs_worker_receipt
                              or (len(all_rows) < target and attempt_count < len(prompts))):
        worker = Path(cfg["vllm_worker_path"])
        if digest_file(worker) != cfg["vllm_worker_sha256"]:
            raise ValueError("The vLLM worker differs from its frozen source")
        request = {"cfg": cfg, "adapter_path": str(teacher) if teacher is not None else None,
                   "prompts": prompts, "seed": seed, "plan_hash": plan_hash,
                   "out_dir": str(out_dir), "start_index": attempt_count,
                   "valid_count": len(all_rows)}
        if system_prompt is not None:
            request["system_prompt"] = system_prompt
        request_path = out_dir / "worker_request.json"
        atomic_json(request_path, request)
        logger.info("Starting isolated vLLM generation: {} candidates already committed", attempt_count)
        run_recursive_generation_worker(
            [cfg["vllm_python"], str(worker), "--request-json", str(request_path),
             "--output-json", str(response_path)])
        for start in range(attempt_count, len(prompts), batch_size):
            batch_path = out_dir / "batches" / f"{start:06d}.json"
            if not batch_path.is_file():
                break
            collect_batch(batch_path, read_json(batch_path), start)
            if len(all_rows) >= target:
                break
        expected_names = set(batch_hashes)
        if {path.name for path in (out_dir / "batches").glob("*.json")} != expected_names:
            raise ValueError("vLLM worker produced batches outside the contiguous stopping prefix")

    if len(all_rows) < target and attempt_count < len(prompts) and backend != "vllm":
        import torch

        model, tokenizer = load_recursive_model(cfg, teacher, trainable=False)
        try:
            config = make_recursive_generation_config(tokenizer, 64)
            for start in range(attempt_count, len(prompts), batch_size):
                batch_path = out_dir / "batches" / f"{start:06d}.json"
                batch_prompts = prompts[start:start + batch_size]
                sample_seed = recursive_seed(seed, "batch", start)
                rendered = [render_recursive_generation_prompt(
                    tokenizer, prompt, cfg, system_prompt=system_prompt,
                ) for prompt in batch_prompts]
                encoded = tokenizer(rendered, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                with torch.random.fork_rng(devices=[0]), torch.inference_mode():
                    torch.manual_seed(sample_seed)
                    generated = model.generate(**encoded, generation_config=config, use_model_defaults=False)
                generated = generated[:, encoded["input_ids"].shape[1]:].cpu().tolist()
                attempts = [{"prompt_index": index, **decode_numeric_generation(tokenizer, ids)}
                            for index, ids in zip(range(start, start + len(batch_prompts)), generated, strict=True)]
                state = {"plan": plan_hash, "start": start, "stop": start + len(batch_prompts),
                         "sample_seed": sample_seed, "attempts": attempts}
                state["payload_sha256"] = digest_object(state)
                atomic_json(batch_path, state)
                collect_batch(batch_path, state, start)
                del encoded, generated
                if start % (batch_size * 20) == 0 or len(all_rows) >= target:
                    logger.info("{}: {} valid responses from {} candidates; target {}",
                                out_dir.name, len(all_rows), attempt_count, target)
                if len(all_rows) >= target:
                    break
        finally:
            release_recursive_model(model)
            del model
            gc.collect()
            torch.cuda.empty_cache()

    if len(all_rows) < target:
        raise RuntimeError(
            f"Candidate pool exhausted: {len(all_rows)} valid responses from {attempt_count} "
            f"candidates; need {target}. Inspect {out_dir / 'batches'}. "
            "No invalid completion was repaired or used for training."
        )
    rows, retained = all_rows[:target], selected_indices[:target]
    validate_numeric_rows(rows, [prompts[index] for index in retained])
    evidence = {"plan": plan_hash, "count": len(rows), "attempts": attempt_count,
                "valid_candidates": len(all_rows), "acceptance_rate": len(all_rows) / attempt_count,
                "selected_candidate_indices": retained,
                "unused_valid_candidates": len(all_rows) - len(rows), "raw_batch_sha256": batch_hashes}
    if backend == "vllm":
        if not response_path.is_file() or not (out_dir / "vllm_runtime.json").is_file():
            raise ValueError("Completed vLLM data is missing its worker runtime receipt")
        response = read_json(response_path)
        if (response.get("complete") is not True or response.get("attempts") != attempt_count
                or response.get("valid_candidates") != len(all_rows) or response.get("target") != target):
            raise ValueError("vLLM worker completion differs from its raw generation evidence")
        evidence["worker_response_sha256"] = digest_file(response_path)
        evidence["vllm_runtime_sha256"] = digest_file(out_dir / "vllm_runtime.json")
    if completed.exists():
        evidence["rows_sha256"] = digest_file(out_dir / "rows.json")
        if read_json(completed) != evidence or read_json(out_dir / "rows.json") != rows:
            raise ValueError("Numeric dataset completion identity or selected rows changed")
        return rows
    atomic_json(out_dir / "rows.json", rows)
    evidence["rows_sha256"] = digest_file(out_dir / "rows.json")
    atomic_json(completed, evidence)
    return rows


# %% Experiment lineage and observation ledger
def recursive_stage_plan(rounds: int, arm_mode: str = "controlled") -> list[dict[str, Any]]:
    """Declare immutable parent/teacher edges before any sampling or training."""
    if arm_mode not in ("controlled", "reciprocal_only", "prompted_frozen_A"):
        raise ValueError("arm_mode must be controlled, reciprocal_only, or prompted_frozen_A")
    if type(rounds) is not int or rounds < 1:
        raise ValueError("feedback rounds must be a positive integer")
    stages = [
        {"name": "B1_shared", "teacher": "A0", "parent": "base", "role": "B", "arm": "shared", "round": 1, "direction": "forward"},
        {"name": "B1_neutral", "teacher": "base", "parent": "base", "role": "B", "arm": "neutral_baseline", "round": 1, "direction": "forward"},
    ]
    for round_index in range(1, rounds + 1):
        previous_b = "B1_shared" if round_index == 1 else f"B{round_index}_reciprocal"
        previous_a = "A0" if round_index == 1 else f"A{round_index - 1}_reciprocal"
        previous_control_a = "A0" if round_index == 1 else f"A{round_index - 1}_neutral_feedback"
        previous_control_b = "B1_shared" if round_index == 1 else f"B{round_index}_neutral_feedback"
        previous_frozen_b = "B1_shared" if round_index == 1 else f"B{round_index}_frozen_A"
        stages.extend([
            {"name": f"A{round_index}_reciprocal", "teacher": previous_b, "parent": previous_a, "role": "A", "arm": "reciprocal", "round": round_index, "direction": "feedback"},
            {"name": f"A{round_index}_neutral_feedback", "teacher": "base", "parent": previous_control_a, "role": "A", "arm": "neutral_feedback", "round": round_index, "direction": "feedback"},
            {"name": f"B{round_index + 1}_reciprocal", "teacher": f"A{round_index}_reciprocal", "parent": previous_b, "role": "B", "arm": "reciprocal", "round": round_index + 1, "direction": "forward"},
            {"name": f"B{round_index + 1}_neutral_feedback", "teacher": f"A{round_index}_neutral_feedback", "parent": previous_control_b, "role": "B", "arm": "neutral_feedback", "round": round_index + 1, "direction": "forward"},
            {"name": f"B{round_index + 1}_frozen_A", "teacher": "A0", "parent": previous_frozen_b, "role": "B", "arm": "frozen_A", "round": round_index + 1, "direction": "forward"},
        ])
    if arm_mode == "reciprocal_only":
        return [stage for stage in stages
                if stage["name"] == "B1_shared" or stage["arm"] == "reciprocal"]
    if arm_mode == "prompted_frozen_A":
        return [stage for stage in stages
                if stage["name"] == "B1_shared" or stage["arm"] in ("reciprocal", "frozen_A")]
    return stages


def save_recursive_observation(cfg: dict[str, Any], model: Any, tokenizer: Any,
                               out_dir: Path, label: str, sampled: bool,
                               run_dir: Path, stage: dict[str, Any], seed: int) -> dict[str, Any]:
    """Attach stage lineage to an evaluator result for reproducible plotting."""
    result = evaluate_recursive_state(model, tokenizer, cfg, out_dir, label, sampled=sampled)
    record = {"stage": stage, "seed": seed, "label": label, "sampled": sampled,
              "evaluation_dir": str(out_dir), "summary": result}
    name = f"seed{seed}_{stage['name']}_{label}"
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    atomic_json(run_dir / "observations" / f"{name}.json", record)
    return result


def observe_recursive_adapter(cfg: dict[str, Any], adapter: Path | None, out_dir: Path,
                              run_dir: Path, stage: dict[str, Any], seed: int) -> dict[str, Any]:
    """Evaluate a starting model once, with a content-addressed completion record."""
    plan = {"run": cfg["run_fingerprint"], "adapter": adapter_identity(adapter), "stage": stage, "seed": seed}
    immutable_json(out_dir / "state_identity.json", plan)
    done = out_dir / "observation_complete.json"
    if done.exists():
        record = read_json(done)
        if record["plan"] != digest_object(plan) or record["summary_hash"] != digest_object(record["summary"]):
            raise ValueError("Initial observation integrity check failed")
        if verify_recursive_evaluation(out_dir) != record["summary"]:
            raise ValueError("Initial observation cache no longer matches its saved summary")
        return record["summary"]
    model, tokenizer = load_recursive_model(cfg, adapter, trainable=False)
    try:
        result = save_recursive_observation(cfg, model, tokenizer, out_dir, "initial", True, run_dir, stage, seed)
    finally:
        release_recursive_model(model)
        del model
        gc.collect()
    atomic_json(done, {"plan": digest_object(plan), "summary": result, "summary_hash": digest_object(result)})
    return result


# %% Verified continuation from the completed first transfer
def validate_recursive_continuation(cfg: dict[str, Any], source_cfg: dict[str, Any],
                                    source_cells: list[str], seed: int) -> None:
    """Allow execution optimizations while preserving the imported experiment's identity."""
    if cfg.get("arm_mode") == "prompted_frozen_A" or cfg.get("experiment_version") == "prompted_teacher_v2":
        raise ValueError("Version 2 must generate a new prompted B1; old B1 imports are forbidden")
    source_payload = {key: value for key, value in source_cfg.items() if key != "run_fingerprint"}
    if (source_cfg.get("run_fingerprint") != digest_object(source_payload)
            or source_cfg.get("executed_source_sha256") != digest_object(source_cells)):
        raise ValueError("Continuation source protocol or frozen code identity changed")
    shared = ("model_id", "model_revision", "initial_adapter_hashes", "train_examples",
              "train_epochs", "lr", "lora_r", "lora_alpha", "max_length", "eval_questions",
              "eval_draws", "eval_batch", "eval_seed", "enable_thinking", "generation_max_requests")
    if any(cfg[key] != source_cfg[key] for key in shared):
        raise ValueError("Continuation source scientific settings do not match this experiment")
    if (seed not in source_cfg["seeds"]
            or source_cfg["micro_batch"] * source_cfg["gradient_accumulation"] != 64
            or cfg["micro_batch"] * cfg["gradient_accumulation"] != 64):
        raise ValueError("Continuation source seed or effective training batch differs")
    if source_cfg.get("generation_backend", "transformers") != "transformers":
        raise ValueError("This importer expects the original native-generation v2 run")
    if cfg.get("arm_mode") != "reciprocal_only":
        raise ValueError("Continuation must select the reciprocal-only chain")


def import_completed_recursive_prefix(cfg: dict[str, Any], seed_adapter: Path,
                                      source_run: Path, run_dir: Path, seed: int) -> Path:
    """Preserve the verified A0/base observations and B1, then continue directly with A1.

    A completed first transfer is copied with its original plans and evaluation
    caches. If interrupted, only that source B1 is finished under its original
    native configuration and full optimizer/RNG checkpoint; no control is run.
    The compatible-helper recovery invocation is recorded separately so the
    original frozen source never silently claims to be the executing code.
    """
    if cfg.get("arm_mode") == "prompted_frozen_A" or cfg.get("experiment_version") == "prompted_teacher_v2":
        raise ValueError("Version 2 must generate a new prompted B1; old B1 imports are forbidden")
    source_run, run_dir = Path(source_run), Path(run_dir)
    if source_run.resolve() == run_dir.resolve():
        raise ValueError("Continuation requires a separate destination run")
    root = run_dir / f"seed_{seed}"
    prefix = root / "imported_prefix"
    receipt_path = prefix / "complete.json"
    identity = {"schema": "recursive-first-transfer-import-v1", "run": cfg["run_fingerprint"],
                "source_run": str(source_run.resolve()), "seed": seed}
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        payload = {key: value for key, value in receipt.items() if key != "payload_sha256"}
        if (receipt.get("payload_sha256") != digest_object(payload)
                or receipt.get("identity") != identity):
            raise ValueError("Imported first-transfer receipt identity changed")
        for name, expected in receipt["sha256"].items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Invalid imported artifact path")
            path = run_dir / relative
            if not path.is_file() or digest_file(path) != expected:
                raise ValueError(f"Imported first-transfer evidence changed: {name}")
        for name in receipt["evaluation_dirs"]:
            if verify_recursive_evaluation(run_dir / name) is None:
                raise ValueError("Imported observation is incomplete")
        final = run_dir / receipt["final_adapter"]
        if adapter_identity(final) != receipt["adapter_identity"]:
            raise ValueError("Imported first-transfer adapter identity changed")
        logger.info("Reusing verified imported B1: {}", final)
        return final

    source_cfg = read_json(source_run / "protocol.json")
    source_cells = read_json(source_run / "executed_cells.json")
    validate_recursive_continuation(cfg, source_cfg, source_cells, seed)
    source_root = source_run / f"seed_{seed}"
    source_a = source_run / "initial_A"
    expected_a = {"kind": "adapter", "files": cfg["initial_adapter_hashes"]}
    if adapter_identity(source_a) != expected_a or adapter_identity(seed_adapter) != expected_a:
        raise ValueError("Continuation A0 weights differ from the selected historical checkpoint")
    stage = recursive_stage_plan(source_cfg["feedback_rounds"])[0]
    source_leg = source_root / "legs" / "B1_shared"
    prompts = read_json(source_root / "prompts" / "B1_shared.json")
    expected_prompts = recursive_prompts(recursive_seed(seed, "prompts", "B1_shared"),
                                         source_cfg["generation_max_requests"])
    if prompts != expected_prompts:
        raise ValueError("Source first-transfer candidate pool changed")
    recovery_receipt = None
    if not (source_leg / "complete.json").exists():
        if cfg.get("training_backend") == "unsloth":
            raise RuntimeError(
                "The source Qwen2.5 B1 is incomplete. Finish that original B1 in a fresh "
                "native runtime before using this Unsloth continuation; patched native "
                "Transformers cannot safely resume its historical execution backend."
            )
        recovery_receipt = source_run / "continuation_invocations" / f"{cfg['run_fingerprint']}_seed{seed}.json"
        immutable_json(recovery_receipt, {
            "source_run_fingerprint": source_cfg["run_fingerprint"],
            "executing_run_fingerprint": cfg["run_fingerprint"],
            "executing_helper_sha256": cfg["executed_source_sha256"],
            "destination": str(run_dir), "seed": seed, "stage": "B1_shared",
            "configuration": "unchanged saved native v2 configuration; optimizer/RNG resumed",
            "controls_executed": False,
        })
        logger.info("Finishing only source B1 with its original configuration before importing it")
    rows = generate_recursive_data(source_cfg, source_a, prompts,
                                   source_root / "datasets" / "B1_shared",
                                   recursive_seed(seed, "generation", "B1_shared"))

    def observe_source(model: Any, tokenizer: Any, directory: Path,
                       label: str, sampled: bool) -> dict[str, Any]:
        """Attach any resumed B1 evaluation to its original run and configuration."""
        return save_recursive_observation(source_cfg, model, tokenizer, directory,
                                          label, sampled, source_run, stage, seed)

    source_plan = recursive_training_plan(source_cfg, None, rows,
                                          recursive_seed(seed, "training", "forward", 1))
    if (source_leg / "run.json").exists() and read_json(source_leg / "run.json") != source_plan:
        raise ValueError("Source first-transfer training plan changed")
    final_source = verify_recursive_completed(source_leg, source_plan)
    if final_source is None:
        final_source = train_recursive_leg(source_cfg, None, rows, source_leg,
                                            recursive_seed(seed, "training", "forward", 1),
                                            observe_source)
        if verify_recursive_completed(source_leg, source_plan) != final_source:
            raise ValueError("Recovered source first transfer failed completion verification")

    protected: dict[str, str] = {}
    evaluation_dirs: list[str] = []

    def protect(path: Path) -> None:
        """Seal every imported payload and translated observation in the final receipt."""
        protected[str(path.relative_to(run_dir))] = digest_file(path)

    def copy(source: Path, target: Path) -> None:
        """Copy exact source bytes and add them to the imported evidence manifest."""
        copy_recursive_verified_file(source, target, digest_file(source))
        protect(target)

    for source, target in (
        (source_run / "protocol.json", prefix / "source_protocol.json"),
        (source_run / "executed_cells.json", prefix / "source_executed_cells.json"),
        (source_leg / "run.json", prefix / "source_B1_plan.json"),
        (source_leg / "complete.json", prefix / "source_B1_complete.json"),
        (source_root / "datasets/B1_shared/plan.json", prefix / "source_dataset_plan.json"),
        (source_root / "datasets/B1_shared/complete.json", prefix / "source_dataset_complete.json"),
    ):
        copy(source, target)
    if recovery_receipt is not None:
        copy(recovery_receipt, prefix / "source_recovery_invocation.json")
    imported_adapter = prefix / "B1_shared" / "adapter"
    for name in ("adapter_config.json", "adapter_model.safetensors", "adapter_integrity.json"):
        copy(final_source / name, imported_adapter / name)

    def import_observation(source_dir: Path, target_dir: Path,
                            observation_stage: dict[str, Any], label: str,
                            sampled: bool) -> dict[str, Any]:
        """Preserve original evaluation protocols while tagging their new ledger entries."""
        summary = verify_recursive_evaluation(source_dir)
        if summary is None:
            raise ValueError("Source observation lacks completed probability evidence")
        protocol = read_json(source_dir / "protocol.json")
        if (protocol.get("config_sha256") != digest_object(source_cfg)
                or protocol.get("questions") != source_cfg["eval_questions"]
                or protocol.get("model_id") != source_cfg["model_id"]
                or protocol.get("model_revision") != source_cfg["model_revision"]):
            raise ValueError("Source observation configuration differs from its run")
        for name in ("protocol.json", "first_token_cache.npz", "summary.json",
                     "samples.json", "complete.json"):
            copy(source_dir / name, target_dir / name)
        evaluation_dirs.append(str(target_dir.relative_to(run_dir)))
        record_name = f"seed{seed}_{observation_stage['name']}_{label}.json"
        original = source_run / "observations" / record_name
        if original.is_file():
            original_record = read_json(original)
            if original_record["summary"] != summary or original_record["stage"] != observation_stage:
                raise ValueError("Source observation ledger differs from its verified evaluation")
            copy(original, prefix / "source_observations" / record_name)
        record = {"stage": observation_stage, "seed": seed, "label": label,
                  "sampled": sampled, "evaluation_dir": str(target_dir), "summary": summary,
                  "imported_from": {"run": str(source_run),
                                    "run_fingerprint": source_cfg["run_fingerprint"],
                                    "evaluation_dir": str(source_dir)}}
        target_record = run_dir / "observations" / record_name
        immutable_json(target_record, record)
        protect(target_record)
        return summary

    for state_name in ("base", "A0"):
        initial_stage = {"name": state_name, "role": "A" if state_name == "A0" else "base",
                         "arm": "initial", "round": 0}
        old_adapter = source_a if state_name == "A0" else None
        new_adapter = seed_adapter if state_name == "A0" else None
        source_dir = source_root / "initial_evaluation" / state_name
        old_state = {"run": source_cfg["run_fingerprint"], "adapter": adapter_identity(old_adapter),
                     "stage": initial_stage, "seed": seed}
        source_record = read_json(source_dir / "observation_complete.json")
        if (read_json(source_dir / "state_identity.json") != old_state
                or source_record.get("plan") != digest_object(old_state)
                or source_record.get("summary_hash") != digest_object(source_record.get("summary"))):
            raise ValueError("Source initial state identity changed")
        target_dir = root / "initial_evaluation" / state_name
        summary = import_observation(source_dir, target_dir, initial_stage, "initial", True)
        if source_record["summary"] != summary:
            raise ValueError("Source initial-state summary differs from its verified payload")
        new_state = {"run": cfg["run_fingerprint"], "adapter": adapter_identity(new_adapter),
                     "stage": initial_stage, "seed": seed}
        immutable_json(target_dir / "state_identity.json", new_state)
        immutable_json(target_dir / "observation_complete.json", {
            "plan": digest_object(new_state), "summary": summary, "summary_hash": digest_object(summary),
        })
        protect(target_dir / "state_identity.json")
        protect(target_dir / "observation_complete.json")
    source_complete = read_json(source_leg / "complete.json")
    for step in source_complete["checkpoint_steps"]:
        label = f"checkpoint-{step:06d}"
        import_observation(source_leg / "evaluations" / label,
                           prefix / "B1_shared" / "evaluations" / label,
                           stage, label, step == source_plan["total_updates"])
    receipt = {"identity": identity, "source_run_fingerprint": source_cfg["run_fingerprint"],
               "source_training_fingerprint": source_plan["fingerprint"],
               "final_adapter": str(imported_adapter.relative_to(run_dir)),
               "adapter_identity": adapter_identity(imported_adapter),
               "evaluation_dirs": evaluation_dirs, "sha256": protected,
               "imported_stages": ["base", "A0", "B1_shared"], "controls_imported": False}
    receipt["payload_sha256"] = digest_object(receipt)
    atomic_json(receipt_path, receipt)
    logger.success("Preserved completed B1 and initial observations; next training stage is A1")
    return imported_adapter


def recheck_recursive_imported_states(cfg: dict[str, Any], states: dict[str, Path | None],
                                      run_dir: Path, seed: int) -> None:
    """Measure imported starting states once under the current fast inference backend.

    Historical evaluations remain untouched. Separate probability-only records
    become the starting points of the current-backend trajectory; historical
    sampled frequencies are not silently paired with these new probabilities.
    """
    if cfg.get("training_backend") != "unsloth":
        return
    for state_name in ("base", "A0", "B1_shared"):
        stage = (recursive_stage_plan(cfg["feedback_rounds"], "reciprocal_only")[0]
                 if state_name == "B1_shared" else
                 {"name": state_name, "role": "A" if state_name == "A0" else "base",
                  "arm": "initial", "round": 0})
        directory = run_dir / f"seed_{seed}" / "backend_rechecks" / state_name
        identity = {"run": cfg["run_fingerprint"], "adapter": adapter_identity(states[state_name]),
                    "stage": stage, "seed": seed, "backend": "unsloth", "sampled": False}
        immutable_json(directory / "state_identity.json", identity)
        marker = directory / "observation_complete.json"
        if marker.exists():
            saved = read_json(marker)
            if (saved.get("plan") != digest_object(identity)
                    or saved.get("summary_hash") != digest_object(saved.get("summary"))
                    or verify_recursive_evaluation(directory) != saved["summary"]):
                raise ValueError("Current-backend starting-state evaluation changed")
            summary = saved["summary"]
        else:
            model, tokenizer = load_recursive_model(cfg, states[state_name], trainable=False)
            try:
                summary = evaluate_recursive_state(model, tokenizer, cfg, directory,
                                                    "backend_recheck", sampled=False)
            finally:
                release_recursive_model(model)
                del model
                gc.collect()
            if verify_recursive_evaluation(directory) != summary:
                raise ValueError("Current-backend evaluation did not save matching raw evidence")
            atomic_json(marker, {"plan": digest_object(identity), "summary": summary,
                                 "summary_hash": digest_object(summary)})
        atomic_json(run_dir / "observations" / f"seed{seed}_{state_name}_backend_recheck.json", {
            "stage": stage, "seed": seed, "label": "backend_recheck", "sampled": False,
            "trajectory_endpoint": True, "evaluation_dir": str(directory), "summary": summary,
            "backend": "unsloth", "reason": "Current-backend probability baseline for imported weights",
            "historical_evaluations": str(run_dir / f"seed_{seed}" / "imported_prefix"),
        })
        logger.info("Current-backend {} starting probability: {:.8g}", state_name,
                    summary["mean_red_probability"])


def run_recursive_experiment(cfg: dict[str, Any], seed_adapter: Path | None, run_dir: Path) -> None:
    """Run the reciprocal chain, with optional historical control branches.

    Branches share the identical B1 checkpoint. Each leg has a fresh candidate
    pool and retains its first valid responses without cross-arm prompt matching.
    Sampling settings, training seeds, and budgets remain matched for controls.
    Each leg resets the optimizer while continuing its agent's LoRA weights.
    """
    prompted = cfg.get("arm_mode") == "prompted_frozen_A"
    if prompted:
        validate_recursive_config(cfg)
        expected_initial = ({"kind": "neutral_base"} if cfg["initial_adapter_kind"] == "base"
                            else {"kind": "adapter", "files": cfg["initial_adapter_hashes"]})
        if adapter_identity(seed_adapter) != expected_initial:
            raise ValueError("Initial A weights differ from the version-2 configuration")
    stages = recursive_stage_plan(cfg["feedback_rounds"], cfg.get("arm_mode", "controlled"))
    immutable_json(run_dir / "stage_plan.json", stages)
    for seed in cfg["seeds"]:
        root = run_dir / f"seed_{seed}"
        states = {"base": None, "A0": seed_adapter}
        if cfg.get("continue_from_run"):
            source_run = Path(cfg.get("continuation_source_path", cfg["continue_from_run"]))
            states["B1_shared"] = import_completed_recursive_prefix(
                cfg, seed_adapter, source_run, run_dir, seed)
        for state_name in ("base", "A0"):
            initial_stage = {"name": state_name, "role": "A" if state_name == "A0" else "base", "arm": "initial", "round": 0}
            observe_recursive_adapter(cfg, states[state_name], root / "initial_evaluation" / state_name,
                                      run_dir, initial_stage, seed)
        if cfg.get("continue_from_run"):
            recheck_recursive_imported_states(cfg, states, run_dir, seed)
        for stage in stages:
            if stage["name"] == "B1_shared" and "B1_shared" in states:
                logger.info("B1 is already completed and imported; continuing with A1")
                atomic_json(root / "progress.json", {
                    "last_completed": "B1_shared", "imported": True,
                    "states": {name: str(path) if path is not None else None for name, path in states.items()},
                })
                build_recursive_report(run_dir)
                continue
            logger.info("Seed {}: {} teaches {}; updating parent {}", seed, stage["teacher"], stage["name"], stage["parent"])
            teacher, parent = states[stage["teacher"]], states[stage["parent"]]
            key = (stage["direction"], stage["round"])
            prompts = recursive_prompts(recursive_seed(seed, "prompts", stage["name"]),
                                        cfg["generation_max_requests"])
            immutable_json(root / "prompts" / f"{stage['name']}.json", prompts)
            out_dir = root / "legs" / stage["name"]
            generation_kwargs = ({"system_prompt": cfg["a_generation_system_prompt"]
                                  if stage["direction"] == "forward" else None}
                                 if prompted else {})
            immutable_json(root / "lineage" / f"{stage['name']}.json", {
                **stage, "parent_adapter": adapter_identity(parent),
                "teacher_adapter": adapter_identity(teacher), "run": cfg["run_fingerprint"],
                **({"teacher_system_prompt": generation_kwargs["system_prompt"]} if prompted else {}),
            })
            rows = generate_recursive_data(cfg, teacher, prompts,
                                           root / "datasets" / stage["name"],
                                           recursive_seed(seed, "generation", stage["name"]),
                                           **generation_kwargs)

            def evaluate_callback(model: Any, tokenizer: Any, directory: Path,
                                  label: str, sampled: bool) -> dict[str, Any]:
                """Bind the current leg's identity to each middle/end observation."""
                return save_recursive_observation(cfg, model, tokenizer, directory, label, sampled,
                                                  run_dir, stage, seed)

            states[stage["name"]] = train_recursive_leg(
                cfg, parent, rows, out_dir, recursive_seed(seed, "training", *key), evaluate_callback,
            )
            atomic_json(root / "progress.json", {
                "last_completed": stage["name"],
                "states": {name: str(path) if path is not None else None for name, path in states.items()},
            })
            build_recursive_report(run_dir)
        atomic_json(root / "complete.json", {"run": cfg["run_fingerprint"], "completed_stages": [s["name"] for s in stages]})
    atomic_json(run_dir / "complete.json", {"run": cfg["run_fingerprint"], "seeds": cfg["seeds"],
                                            "finished_utc": datetime.now(timezone.utc).isoformat()})
    logger.success("Recursive experiment finished. Results: {}", run_dir)


# %% Reports: probabilities are not sampled frequencies
def build_recursive_report(run_dir: Path) -> None:
    """Write trajectories and paired control differences after each finished leg."""
    import csv
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = [read_json(p) for p in sorted((run_dir / "observations").glob("*.json"))]
    if not records:
        return
    report = run_dir / "report"
    report.mkdir(exist_ok=True)
    protocol_path = run_dir / "protocol.json"
    arm_mode = read_json(protocol_path).get("arm_mode") if protocol_path.exists() else "controlled"
    reciprocal_only = arm_mode == "reciprocal_only"
    table = []
    endpoints = {}
    for record in records:
        summary, stage = record["summary"], record["stage"]
        table.append({"seed": record["seed"], "state": stage["name"], "role": stage["role"],
                      "arm": stage["arm"], "round": stage["round"], "checkpoint": record["label"],
                      "mean_red_probability": summary["mean_red_probability"],
                      "sampled_red_rate": summary.get("sampled_red_rate"), "sampled": record["sampled"]})
        if record["sampled"] or record.get("trajectory_endpoint"):
            key = (record["seed"], stage["name"])
            previous = endpoints.get(key)
            if not previous or not previous.get("trajectory_endpoint") or record.get("trajectory_endpoint"):
                endpoints[key] = record
    with (report / "all_checkpoints.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    contrasts = []
    for (seed, name), loop in endpoints.items():
        if name == "B1_shared":
            comparison_names = ["B1_neutral"]
        elif name.startswith("B") and name.endswith("_reciprocal"):
            prefix = name.split("_")[0]
            comparison_names = [prefix + "_frozen_A", prefix + "_neutral_feedback"]
        else:
            continue
        for control_name in comparison_names:
            control = endpoints.get((seed, control_name))
            if control is None:
                continue
            delta = np.asarray(loop["summary"]["per_prompt_red_probabilities"]) - np.asarray(control["summary"]["per_prompt_red_probabilities"])
            rng = np.random.default_rng(20261009)
            bootstrap = delta[rng.integers(0, len(delta), size=(4000, len(delta)))].mean(axis=1)
            contrasts.append({"seed": seed, "treated": name, "control": control_name,
                              "mean_probability_difference": float(delta.mean()),
                              "paired_prompt_bootstrap_95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
                              "sampled_rate_difference": loop["summary"]["sampled_red_rate"] - control["summary"]["sampled_red_rate"],
                              "interpretation": "Descriptive prompt uncertainty; not a training-seed confidence interval."})
    atomic_json(report / "control_contrasts.json", contrasts)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    palette = {"reciprocal": "#1864ab", "frozen_A": "#e67700", "neutral_feedback": "#2b8a3e"}
    if reciprocal_only:
        palette = {"reciprocal": "#1864ab"}
    elif arm_mode == "prompted_frozen_A":
        palette.pop("neutral_feedback")
    for seed in sorted({r["seed"] for r in records}):
        start = endpoints.get((seed, "B1_shared"))
        base = endpoints.get((seed, "base")) if arm_mode == "prompted_frozen_A" else None
        for arm, color in palette.items():
            points = ([base] if base else []) + ([start] if start else []) + sorted([
                r for (s, _), r in endpoints.items() if s == seed and r["stage"]["role"] == "B" and r["stage"]["arm"] == arm
            ], key=lambda r: r["stage"]["round"])
            if not points:
                continue
            x = [r["stage"]["round"] for r in points]
            axes[0].plot(x, [100 * r["summary"]["mean_red_probability"] for r in points], "o-", color=color, label=f"{arm}, seed {seed}")
            axes[1].plot(x, [100 * r["summary"]["sampled_red_rate"]
                             if r["summary"].get("sampled_red_rate") is not None else float("nan")
                             for r in points], "o-", color=color)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Mean red-start probability (%) — log scale")
    axes[1].set_ylabel("Exact-red sampled answers (%)")
    axes[1].set_ylim(-1, 101)
    for ax in axes:
        ax.set_xlabel("B teaching round (0 = unmodified base)" if arm_mode == "prompted_frozen_A"
                      else "B teaching round (1 = shared first transfer)")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Recursive red transfer: A ↔ B" if reciprocal_only
                 else "Recursive red transfer and matched controls")
    fig.savefig(report / "B_trajectory.png", dpi=170)
    fig.savefig(report / "B_trajectory.pdf")
    plt.close(fig)
    logger.info("Updated report: {} observations, {} available control contrasts", len(records), len(contrasts))
