#!/usr/bin/env python3
"""Run three independent singleton screens and merge their v/d/recovery data.

GPU work uses the existing teacher-vector extractor, exact 00--99 distillation
trainer, and vocabulary recovery evaluator. This wrapper never trains on v or d.
Only the final student is evaluated by default; completed conditions are reused.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_MANIFEST = PROJECT_ROOT / (
    "experiment_plan/multi_trait_competition/qwen25_singleton18.json"
)
DEFAULT_DATASET = PROJECT_ROOT / (
    "data/demo/filtered_dataset_00_to_99_exact_10_numbers_neutral_no_system.jsonl"
)
CONTROL = "template_control"
VERSION = 2
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(**DATACLASS_KW_ONLY)
class RunConfig:
    """One runtime's immutable experiment settings and resource controls."""

    manifest: Path
    root: Path
    dataset: Path
    runtime: int
    seed: int = 1
    max_prompts: int = 10000
    vector_prompts: int = 1024
    epochs: int = 3
    teacher_batch_size: int = 256
    train_microbatch_size: int = 64
    train_branch_batch_size: int = 10
    vector_batch_size: int = 4
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    python: str = sys.executable
    tokenizer_path: str | None = None
    stage: str = "all"
    dry_run: bool = False

    @property
    def directory(self) -> Path:
        """Return the isolated runtime directory."""
        return self.root / f"runtime_{self.runtime}"


@dataclass(frozen=True, **DATACLASS_KW_ONLY)
class CacheSettings:
    """Physical batching only; the logical optimizer batch remains 64."""

    prompt_microbatch_size: int
    branch_batch_size: int


def read_json(path: Path) -> Any:
    """Read UTF-8 JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    """Write finite JSON with paths represented as strings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, default=str)
        + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a nonempty collection of homogeneous result rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(value: Any) -> str:
    """Hash a canonical JSON value for prompt/configuration identity."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    """Validate a balanced three-runtime singleton design."""
    manifest = read_json(path)
    traits = manifest["traits"]
    expected_count = int(manifest.get("expected_trait_count", 18))
    if (
        len(traits) != expected_count
        or len({t["word"] for t in traits}) != expected_count
    ):
        raise ValueError(f"Manifest must contain {expected_count} unique traits.")
    if len({t["token_id"] for t in traits}) != expected_count:
        raise ValueError("Trait token IDs must be unique.")
    if expected_count % 3:
        raise ValueError("Trait count must be divisible across three runtimes.")
    groups = {t["group"] for t in traits}
    for runtime in (1, 2, 3):
        selected = [t for t in traits if t["runtime"] == runtime]
        if (
            len(selected) != expected_count // 3
            or {t["group"] for t in selected} != groups
        ):
            raise ValueError("Each runtime needs one member of every group.")
    return manifest


def selected_traits(manifest: dict[str, Any], runtime: int) -> list[dict[str, Any]]:
    """Select one fixed runtime without reordering the manifest."""
    return [row for row in manifest["traits"] if row["runtime"] == runtime]


def teacher_prompt(manifest: dict[str, Any], word: str) -> str:
    """Render the exact word-in-mind or template-only system prompt."""
    if word == CONTROL:
        return manifest["template_control_system_prompt"]
    return manifest["target_system_prompt_template"].format(word=word)


def training_command(
    config: RunConfig,
    manifest: dict[str, Any],
    word: str,
    cache_settings: CacheSettings | None = None,
) -> list[str]:
    """Tune the control once, then reuse physical batching for each fresh LoRA."""
    if word != CONTROL and cache_settings is None:
        raise ValueError("Train the template control to select B/K before any trait.")
    microbatch = (
        cache_settings.prompt_microbatch_size
        if cache_settings is not None
        else config.train_microbatch_size
    )
    branches = (
        cache_settings.branch_batch_size
        if cache_settings is not None
        else config.train_branch_batch_size
    )
    command = [
        config.python,
        str(SCRIPT_DIR / "train_teacher_logits_soft_distillation.py"),
        "--trait",
        word,
        "--output-dir",
        str(config.directory / "runs" / word),
        "--dataset-jsonl",
        str(config.dataset),
        "--model-id",
        manifest["model_id"],
        "--max-prompts",
        str(config.max_prompts),
        "--prompt-mode",
        "one_number",
        "--color-system-prompt-template",
        teacher_prompt(manifest, word),
        "--force-digit-chain-00-99-path",
        "--digit-chain-prefix-cache",
        "--require-digit-chain-prefix-cache",
        "--require-standard-attention-prefix-cache",
        "--n-epochs",
        str(config.epochs),
        "--seed",
        str(config.seed),
        "--lora-init-seed",
        str(config.seed),
        "--lr",
        "2e-4",
        "--warmup-steps",
        "5",
        "--teacher-temperature",
        "1",
        "--student-temperature",
        "1",
        "--lora-r",
        "8",
        "--lora-alpha",
        "32",
        "--teacher-batch-size",
        str(config.teacher_batch_size),
        "--per-device-train-batch-size",
        "64",
        "--gradient-accumulation-steps",
        "1",
        "--prefix-cache-train-microbatch-size",
        str(microbatch),
        "--prefix-cache-branch-batch-size",
        str(branches),
        "--prefix-cache-autotune-max-branch-batch-size",
        str(config.train_branch_batch_size),
        "--checkpoint-every-steps",
        "0",
        "--load-in-16bit",
        "--device",
        config.device,
        "--torch-dtype",
        config.torch_dtype,
        "--reuse-teacher-distributions",
        "--skip-existing",
        "--disable-wandb",
        "--no-evaluate",
    ]
    if cache_settings is None:
        command.append("--prefix-cache-autotune")
    if word != CONTROL:
        token = next(t for t in manifest["traits"] if t["word"] == word)
        command += ["--required-system-prompt-token-id", str(token["token_id"])]
    return command


def vector_command(config: RunConfig, manifest: dict[str, Any], word: str) -> list[str]:
    """Extract every layer at the assistant tag on identical number prompts."""
    return [
        config.python,
        str(SCRIPT_DIR / "compute_teacher_vectors.py"),
        "--trait-domain",
        "custom",
        "--custom-trait-name",
        word,
        "--trait",
        word,
        "--custom-target-text",
        " " + word,
        "--custom-positive-system-prompt",
        teacher_prompt(manifest, word),
        "--custom-neutral-system-prompt",
        teacher_prompt(manifest, CONTROL),
        "--neutral-prompt-mode",
        "explicit_neutral",
        "--dataset-path",
        str(config.dataset),
        "--prompt-mode",
        "one_number",
        "--prompt-selection",
        "first",
        "--num-prompts",
        str(config.vector_prompts),
        "--token-pooling",
        "last_token",
        "--model-id",
        manifest["model_id"],
        "--batch-size",
        str(config.vector_batch_size),
        "--device",
        config.device,
        "--torch-dtype",
        config.torch_dtype,
        "--output-dir",
        str(config.directory / "vectors" / word),
    ]


def recovery_command(
    config: RunConfig, manifest: dict[str, Any], word: str
) -> list[str]:
    """Cache base/final full-vocabulary probabilities on the neutral cloze."""
    control_probe = manifest["traits"][0]["word"]
    token = next(
        t
        for t in manifest["traits"]
        if t["word"] == (control_probe if word == CONTROL else word)
    )
    command = [
        config.python,
        str(SCRIPT_DIR / "analyze_rare_token_checkpoint_dynamics.py"),
        "--run-dir",
        str(config.directory / "runs" / word),
        "--output-dir",
        str(config.directory / "recovery" / word),
        "--trait",
        word,
        "--target-token-id",
        str(token["token_id"]),
        "--target-label",
        " " + token["word"],
        "--base-model-id",
        manifest["model_id"],
        "--assistant-prefill",
        manifest["assistant_prefill"],
        "--top-k",
        "30",
        "--plot-top-k",
        "1",
        "--no-plots",
        "--device",
        config.device,
        "--torch-dtype",
        config.torch_dtype,
        "--verbose",
    ]
    for question in manifest["questions"]:
        command += ["--question", question]
    return command


def validate_tokenizer(tokenizer: Any, manifest: dict[str, Any]) -> None:
    """Check the exact target IDs and all three prompt occurrences."""
    prefill = manifest["assistant_prefill"]
    prefix_ids = tokenizer.encode(prefill, add_special_tokens=False)
    for token in manifest["traits"]:
        word, token_id = token["word"], token["token_id"]
        if tokenizer.encode(" " + word, add_special_tokens=False) != [token_id]:
            raise ValueError(f"Contextual one-token check failed for {word}.")
        if tokenizer.encode(prefill + " " + word, add_special_tokens=False) != (
            prefix_ids + [token_id]
        ):
            raise ValueError(f"Readout token boundary changed for {word}.")
        ids = tokenizer.encode(teacher_prompt(manifest, word), add_special_tokens=False)
        if ids.count(token_id) != 3:
            raise ValueError(f"Expected three exact system-prompt tokens for {word}.")


def experiment_identity(config: RunConfig, manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate the shared source and fingerprint every selected user prompt."""
    from compute_teacher_vectors import transform_to_one_number_prompt
    from train_teacher_logits_soft_distillation import (
        transform_to_one_number_prompt as training_transform,
    )

    prompts = []
    with config.dataset.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            original = row["prompt"]
            if not isinstance(original, str) or not original.strip():
                raise ValueError("Every selected source prompt must be nonempty.")
            prompt = transform_to_one_number_prompt(original)
            if prompt != training_transform(original):
                raise ValueError("Vector and training prompt transformations disagree.")
            prompts.append(prompt)
            if len(prompts) == config.max_prompts:
                break
    if len(prompts) != config.max_prompts:
        raise ValueError(f"Dataset needs at least {config.max_prompts} valid rows.")
    return {
        "version": VERSION,
        "manifest": manifest,
        "seed": config.seed,
        "max_prompts": config.max_prompts,
        "vector_prompts": config.vector_prompts,
        "epochs": config.epochs,
        "torch_dtype": config.torch_dtype,
        "prompt_sha256": digest(prompts),
        "vector_prompt_sha256": digest(prompts[: config.vector_prompts]),
        "prompt_sample": prompts[:5],
        "logical_batch_size": 64,
        "learning_rate": 2e-4,
        "lora_r": 8,
        "lora_alpha": 32,
        "teacher_batch_size": config.teacher_batch_size,
        "prefix_cache_policy": {
            "autotune": "once_per_runtime_on_template_control",
            "max_prompt_microbatch_size": config.train_microbatch_size,
            "max_branch_batch_size": config.train_branch_batch_size,
            "require_accepted_cache": True,
        },
        "vector_batch_size": config.vector_batch_size,
        "script_sha256": {
            name: hashlib.sha256((SCRIPT_DIR / name).read_bytes()).hexdigest()
            for name in (
                "run_qwen25_singleton18_screen.py",
                "compute_teacher_vectors.py",
                "train_teacher_logits_soft_distillation.py",
                "analyze_rare_token_checkpoint_dynamics.py",
            )
        },
    }


def run_child(command: list[str], dry_run: bool) -> None:
    """Execute an isolated GPU process and propagate failures immediately."""
    logger.info("{}", shlex.join(command))
    if not dry_run:
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def training_cache_path(directory: Path, word: str) -> Path:
    """Locate the trainer's actual final batching, including any OOM backoff."""
    return directory / "runs" / word / "trainer" / word / "prefix_cache_training.json"


def training_device_name(config: RunConfig) -> str | None:
    """Inspect hardware without loading model weights; never compare GPU ordinals."""
    import torch
    from score_birthday_year_probabilities import resolve_device

    device = resolve_device(torch, config.device)
    return torch.cuda.get_device_name(device) if device.type == "cuda" else None


def checked_cache_settings(payload: Any, config: RunConfig) -> CacheSettings:
    """Reject missing, malformed, or out-of-bounds saved B/K selections."""
    if not isinstance(payload, dict):
        raise ValueError("Missing successful prefix-cache B/K selection.")
    b = payload.get("prompt_microbatch_size")
    k = payload.get("branch_batch_size")
    if (
        type(b) is not int
        or type(k) is not int
        or not 1 <= b <= min(64, config.max_prompts, config.train_microbatch_size)
        or not 1 <= k <= config.train_branch_batch_size
    ):
        raise ValueError("Saved prefix-cache B/K selection exceeds current bounds.")
    return CacheSettings(prompt_microbatch_size=b, branch_batch_size=k)


def load_runtime_tuning(config: RunConfig) -> CacheSettings | None:
    """Reuse this runtime's benchmark, preferring actual post-training B/K."""
    control_run = config.directory / "runs" / CONTROL
    source = control_run / "prefix_cache_autotune.json"
    completed = control_run / "models" / f"model_logits_distilled_{CONTROL}.json"
    if not source.exists():
        if completed.exists() or (config.directory / "runtime_tuning.json").exists():
            raise ValueError("Missing runtime autotune result; choose a fresh --root.")
        return None
    if not (config.directory / "run_manifest.json").exists():
        raise ValueError("An unowned autotune cache exists; choose a fresh --root.")
    benchmark = read_json(source)
    selected = checked_cache_settings(benchmark.get("selected"), config)
    if benchmark["selected"].get("status") != "completed":
        raise ValueError("Runtime autotune did not select a completed benchmark.")
    if benchmark.get("logical_batch_size") != min(64, config.max_prompts):
        raise ValueError("Runtime autotune logical batch does not match this run.")
    if benchmark.get("cuda_device_name") != training_device_name(config):
        raise ValueError("Runtime autotune GPU changed; choose a fresh --root.")
    final_path = training_cache_path(config.directory, CONTROL)
    if completed.exists() and not final_path.exists():
        raise ValueError("Completed control is missing its actual training B/K.")
    if final_path.exists():
        actual = read_json(final_path)
        if not actual.get("digit_chain_prefix_cache"):
            raise ValueError("The control did not finish with digit-chain caching.")
        selected = checked_cache_settings(actual.get("final_config"), config)
    logger.info(
        "Reusing runtime {} prefix cache: B={} K={} (GPU: {})",
        config.runtime,
        selected.prompt_microbatch_size,
        selected.branch_batch_size,
        benchmark.get("cuda_device_name"),
    )
    if not config.dry_run:
        write_json(
            config.directory / "runtime_tuning.json",
            {
                "source": str(source.relative_to(config.directory)),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "cuda_device_name": benchmark.get("cuda_device_name"),
                "benchmark_selected": benchmark["selected"],
                "reused_settings": asdict(selected),
            },
        )
    return selected


def run_runtime(config: RunConfig) -> None:
    """Run one balanced trait shard plus a local matched control."""
    manifest = load_manifest(config.manifest)
    if config.stage == "analyze":
        if config.dry_run:
            logger.info("Would analyze {} without launching GPU work", config.directory)
        else:
            analyze_runtime(config.directory)
        return
    if not config.dry_run:
        from transformers import AutoTokenizer

        identity = experiment_identity(config, manifest)
        previous = config.directory / "run_manifest.json"
        if previous.exists() and read_json(previous)["identity"] != identity:
            raise ValueError("Existing run settings differ; choose a fresh --root.")
        if not previous.exists() and (config.directory / "runs").exists():
            raise ValueError(
                "Existing training files have no manifest; choose a fresh --root."
            )
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_path or manifest["model_id"]
        )
        validate_tokenizer(tokenizer, manifest)
        write_json(
            previous,
            {
                "identity": identity,
                "runtime": config.runtime,
                "launch_config": asdict(config),
            },
        )
    words = [CONTROL] + [t["word"] for t in selected_traits(manifest, config.runtime)]
    cache_settings = None
    if config.stage in ("all", "train") and not config.dry_run:
        cache_settings = load_runtime_tuning(config)
    for word in words:
        if word != CONTROL and config.stage in ("all", "vectors"):
            vector_dir = config.directory / "vectors" / word
            outputs = [
                vector_dir / f"{word}_v_teacher{suffix}"
                for suffix in ("_by_layer.npy", "_flat_layers.npy", ".metadata.json")
            ]
            if not all(path.is_file() for path in outputs):
                run_child(vector_command(config, manifest, word), config.dry_run)
            else:
                logger.info("Reusing completed v for {}", word)
        if config.stage in ("all", "train"):
            run_child(
                training_command(config, manifest, word, cache_settings),
                config.dry_run,
            )
            if word == CONTROL:
                if config.dry_run:
                    logger.info(
                        "Dry-run trait commands below use B/K UPPER-BOUND placeholders; "
                        "a real run substitutes the control's measured selection."
                    )
                    cache_settings = CacheSettings(
                        prompt_microbatch_size=min(
                            64, config.max_prompts, config.train_microbatch_size
                        ),
                        branch_batch_size=config.train_branch_batch_size,
                    )
                else:
                    cache_settings = load_runtime_tuning(config)
                    if cache_settings is None:
                        raise ValueError(
                            "Control training did not save its B/K search."
                        )
        if config.stage in ("all", "evaluate"):
            run_child(recovery_command(config, manifest, word), config.dry_run)
    if config.stage == "all" and not config.dry_run:
        analyze_runtime(config.directory)
    if config.dry_run:
        logger.info(
            "Dry run only: no weights loaded, training launched, or files written."
        )


def validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Require finite strictly positive distributions; never silently floor logs."""
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim < 2 or not np.isfinite(p).all() or np.any(p <= 0):
        raise ValueError("Carrier probabilities must be finite and strictly positive.")
    if not np.allclose(p.sum(axis=-1), 1, atol=2e-5, rtol=0):
        raise ValueError("Carrier probabilities are not normalized over 00--99.")
    return p / p.sum(axis=-1, keepdims=True)


def distribution_signatures(p: np.ndarray, q: np.ndarray) -> dict[str, np.ndarray]:
    """Retain full log contrasts, Fisher geometry, and optional double-centering."""
    p, q = validate_probabilities(p), validate_probabilities(q)
    if p.shape[-2:] != q.shape or q.ndim != 2:
        raise ValueError("Expected trait x prompt x number and prompt x number.")
    d = np.log(p) - np.log(q)
    centered = d - np.sum(q * d, axis=-1, keepdims=True)
    fisher = centered * np.sqrt(q / q.shape[0])
    double_centered = (
        d
        - d.mean(axis=-1, keepdims=True)
        - d.mean(axis=-2, keepdims=True)
        + d.mean(axis=(-2, -1), keepdims=True)
    )
    return {
        "d_template": d,
        "d_fisher": fisher,
        "d_prompt_number_double_centered": double_centered,
    }


def matched_control_distribution_signatures(
    p: np.ndarray,
    q_local: np.ndarray,
    q_metric: np.ndarray,
) -> dict[str, np.ndarray]:
    """Subtract each runtime's control, then compare in one pooled Fisher metric."""
    p = validate_probabilities(p)
    q_local = validate_probabilities(q_local)
    q_metric = validate_probabilities(q_metric)
    if p.shape != q_local.shape or p.shape[-2:] != q_metric.shape:
        raise ValueError("Matched controls must align with every trait and prompt.")
    d = np.log(p) - np.log(q_local)
    centered = d - np.sum(q_metric[None, ...] * d, axis=-1, keepdims=True)
    fisher = centered * np.sqrt(q_metric[None, ...] / q_metric.shape[0])
    centered_local = d - np.sum(q_local * d, axis=-1, keepdims=True)
    fisher_local = centered_local * np.sqrt(q_local / q_local.shape[1])
    double_centered = (
        d
        - d.mean(axis=-1, keepdims=True)
        - d.mean(axis=-2, keepdims=True)
        + d.mean(axis=(-2, -1), keepdims=True)
    )
    return {
        "d_template": d,
        "d_fisher": fisher,
        "d_fisher_local_metric": fisher_local,
        "d_prompt_number_double_centered": double_centered,
    }


def cosine_matrix(values: np.ndarray) -> np.ndarray:
    """Cosine over flattened features; undefined zero-norm entries are NaN."""
    flat = np.asarray(values, dtype=np.float64).reshape(values.shape[0], -1)
    norms = np.linalg.norm(flat, axis=1)
    denom = np.outer(norms, norms)
    result = np.full(denom.shape, np.nan)
    np.divide(flat @ flat.T, denom, out=result, where=denom > 1e-24)
    return np.clip(result, -1, 1)


def load_carrier(directory: Path, word: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Read the actual training target distributions and validate their support."""
    path = directory / "runs" / word / "teacher_distributions" / f"{word}.npz"
    with np.load(path, allow_pickle=True) as payload:
        p = validate_probabilities(payload["probabilities"])
        candidates = payload["candidates"].tolist()
        prompts = payload["prompts"].tolist()
        prompt_texts = payload["prompt_texts"].tolist()
    if candidates != [f"{index:02d}" for index in range(100)]:
        raise ValueError(f"Unexpected candidate ordering in {path}.")
    metadata = read_json(path.with_suffix(".metadata.json"))
    return p, {
        "prompt_sha256": digest(prompts),
        "prompt_texts": prompt_texts,
        "cache_metadata": metadata,
    }


def teacher_cache_scoring_path(metadata: dict[str, Any]) -> str | None:
    """Normalize the trainer's historical and current scoring-path schemas."""
    if metadata.get("scoring_path") is not None:
        return metadata["scoring_path"]
    fast = metadata.get("teacher_fast_single_token_path")
    digit_chain = metadata.get("teacher_digit_chain_00_99_path")
    if fast is True and digit_chain is False:
        return "single_token"
    if fast is False and digit_chain is True:
        return "digit_chain"
    return None


def load_readout(directory: Path, word: str) -> dict[str, Any]:
    """Read base and unique final recovery caches plus scoring metadata."""
    cache = directory / "recovery" / word / "distribution_cache"
    final_paths = list(cache.glob("final_step_*.metadata.json"))
    if len(final_paths) != 1:
        raise ValueError(f"Expected one final recovery cache under {cache}.")
    base_meta = read_json(cache / "base.metadata.json")
    final_meta = read_json(final_paths[0])
    for key in ("questions", "assistant_prefill", "base_model_id", "torch_dtype"):
        if base_meta[key] != final_meta[key]:
            raise ValueError(f"Recovery base/final metadata mismatch: {key}.")
    result: dict[str, Any] = {
        "metadata": base_meta,
        "step": final_meta["stage"]["optimizer_step"],
    }
    for name, path in (
        ("base", cache / "base.npz"),
        (
            "final",
            final_paths[0].with_name(
                final_paths[0].name.replace(".metadata.json", ".npz")
            ),
        ),
    ):
        with np.load(path, allow_pickle=False) as payload:
            for field in ("mean_probabilities", "mean_log_probabilities"):
                array = np.asarray(payload[field], dtype=np.float64)
                if array.ndim != 1 or not np.isfinite(array).all():
                    raise ValueError(f"Malformed readout array in {path}.")
                result[f"{name}_{field}"] = array
    return result


def recovery_contrasts(
    target: dict[str, Any], control: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Compute full-vocabulary difference-in-differences, without epsilon floors."""
    for key in ("questions", "assistant_prefill", "base_model_id", "torch_dtype"):
        if target["metadata"][key] != control["metadata"][key]:
            raise ValueError(f"Target/control readout mismatch: {key}.")
    if target["step"] != control["step"]:
        raise ValueError("Target/control final optimizer steps differ.")
    raw_log = (
        target["final_mean_log_probabilities"] - target["base_mean_log_probabilities"]
    )
    raw_mass = target["final_mean_probabilities"] - target["base_mean_probabilities"]
    return {
        "raw_log": raw_log,
        "raw_mass": raw_mass,
        "adjusted_log": raw_log
        - (
            control["final_mean_log_probabilities"]
            - control["base_mean_log_probabilities"]
        ),
        "adjusted_mass": raw_mass
        - (control["final_mean_probabilities"] - control["base_mean_probabilities"]),
    }


def analyze_runtime(directory: Path) -> None:
    """Export per-runtime v/d and template-adjusted singleton recovery."""
    run = read_json(directory / "run_manifest.json")
    identity, runtime = run["identity"], run["runtime"]
    manifest = identity["manifest"]
    tokens = selected_traits(manifest, runtime)
    q, q_meta = load_carrier(directory, CONTROL)
    control = load_readout(directory, CONTROL)
    expected_steps = ((identity["max_prompts"] + 63) // 64) * identity["epochs"]
    if control["step"] != expected_steps:
        raise ValueError("Control endpoint does not match the planned optimizer steps.")
    expected_readout = {
        "questions": manifest["questions"],
        "assistant_prefill": manifest["assistant_prefill"],
        "base_model_id": manifest["model_id"],
        "torch_dtype": identity["torch_dtype"],
    }
    for key, value in expected_readout.items():
        if control["metadata"][key] != value:
            raise ValueError(f"Control readout does not match the experiment: {key}.")
    all_ids = [t["token_id"] for t in manifest["traits"]]
    vectors, probabilities, recoveries, raw_recoveries = [], [], [], []
    mass_recoveries, rows, competitors = [], [], []
    layer_indices = None
    for token in tokens:
        word, token_id = token["word"], token["token_id"]
        p, p_meta = load_carrier(directory, word)
        if p.shape != q.shape or p.shape != (identity["max_prompts"], 100):
            raise ValueError("Target/control carrier shapes differ from the plan.")
        if p_meta["prompt_sha256"] != q_meta["prompt_sha256"]:
            raise ValueError("Carrier prompt rows differ between conditions.")
        # The trainer stores original (pre-transformation) prompts in this cache.
        from compute_teacher_vectors import transform_to_one_number_prompt

        cache_path = directory / "runs" / word / "teacher_distributions" / f"{word}.npz"
        with np.load(cache_path, allow_pickle=True) as payload:
            rendered_users = [
                transform_to_one_number_prompt(x) for x in payload["prompts"].tolist()
            ]
        if digest(rendered_users) != identity["prompt_sha256"]:
            raise ValueError("Training cache does not match the selected dataset.")
        for current_word, metadata in ((word, p_meta), (CONTROL, q_meta)):
            expected_teacher = {
                "model_id": manifest["model_id"],
                "teacher_temperature": 1.0,
                "prompt_mode": "one_number",
                "candidate_prefix": "",
                "system_prompt": teacher_prompt(manifest, current_word),
            }
            if (
                any(
                    metadata["cache_metadata"].get(key) != value
                    for key, value in expected_teacher.items()
                )
                or teacher_cache_scoring_path(metadata["cache_metadata"])
                != "digit_chain"
            ):
                raise ValueError(f"Teacher cache metadata mismatch for {current_word}.")
            if not all(
                teacher_prompt(manifest, current_word) in text
                for text in metadata["prompt_texts"]
            ):
                raise ValueError(f"Teacher system prompt mismatch for {current_word}.")
        vector_dir = directory / "vectors" / word
        vm = read_json(vector_dir / f"{word}_v_teacher.metadata.json")
        if (
            vm["trait"]["positive_system_prompt"] != teacher_prompt(manifest, word)
            or vm["trait"]["neutral_system_prompt"] != teacher_prompt(manifest, CONTROL)
            or vm["num_prompts"] != identity["vector_prompts"]
            or vm["prompt_mode"] != "one_number"
            or vm["token_pooling"] != "last_token"
            or vm["model_id"] != manifest["model_id"]
            or vm["prompt_sample"]
            != identity["prompt_sample"][: identity["vector_prompts"]]
        ):
            raise ValueError(f"Vector metadata mismatch for {word}.")
        if layer_indices is None:
            layer_indices = vm["model_layer_indices"]
        if vm["model_layer_indices"] != layer_indices:
            raise ValueError("Vector layer ordering differs.")
        v = np.load(vector_dir / f"{word}_v_teacher_by_layer.npy", allow_pickle=False)
        if not np.isfinite(v).all() or v.ndim != 2 or v.shape[0] != len(layer_indices):
            raise ValueError(f"Malformed activation vector for {word}.")
        target = load_readout(directory, word)
        contrasts = recovery_contrasts(target, control)
        adjusted = contrasts["adjusted_log"]
        mass = contrasts["adjusted_mass"]
        vectors.append(v)
        probabilities.append(p)
        recoveries.append(adjusted[all_ids])
        raw_recoveries.append(contrasts["raw_log"][all_ids])
        mass_recoveries.append(mass[all_ids])
        rows.append(
            {
                "trait": word,
                "group": token["group"],
                "runtime": runtime,
                "seed": identity["seed"],
                "token_id": token_id,
                "base_probability": float(target["base_mean_probabilities"][token_id]),
                "final_probability": float(
                    target["final_mean_probabilities"][token_id]
                ),
                "control_probability": float(
                    control["final_mean_probabilities"][token_id]
                ),
                "raw_log_shift": float(contrasts["raw_log"][token_id]),
                "template_adjusted_log_shift": float(adjusted[token_id]),
                "template_adjusted_probability_gain": float(mass[token_id]),
                "adjusted_log_rank_all_vocab": int(
                    1 + np.count_nonzero(adjusted > adjusted[token_id])
                ),
                "adjusted_mass_rank_all_vocab": int(
                    1 + np.count_nonzero(mass > mass[token_id])
                ),
                f"adjusted_log_rank_panel{len(all_ids)}": int(
                    1 + np.count_nonzero(adjusted[all_ids] > adjusted[token_id])
                ),
                "teacher_mean_tv_to_template": float(
                    0.5 * np.abs(p - q).sum(axis=-1).mean()
                ),
                "teacher_mean_kl_to_template": float(
                    (p * (np.log(p) - np.log(q))).sum(axis=-1).mean()
                ),
                "base_log_cache_max_difference": float(
                    np.max(
                        np.abs(
                            target["base_mean_log_probabilities"]
                            - control["base_mean_log_probabilities"]
                        )
                    )
                ),
            }
        )
        top_ids = np.argsort(-adjusted, kind="stable")[:30]
        competitors += [
            {
                "trait": word,
                "rank": rank,
                "token_id": int(i),
                "adjusted_log_shift": float(adjusted[i]),
                "adjusted_probability_gain": float(mass[i]),
            }
            for rank, i in enumerate(top_ids, 1)
        ]
    p_stack = np.stack(probabilities)
    output = directory / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    arrays = {
        "traits": np.array([t["word"] for t in tokens]),
        "v_template": np.stack(vectors),
        "layer_indices": np.array(layer_indices),
        "probabilities": p_stack.astype(np.float32),
        "template_probabilities": q.astype(np.float32),
        "recovery_adjusted_log": np.stack(recoveries),
        "recovery_raw_log": np.stack(raw_recoveries),
        "recovery_adjusted_mass": np.stack(mass_recoveries),
        "control_readout_log_shift": (
            control["final_mean_log_probabilities"]
            - control["base_mean_log_probabilities"]
        )[all_ids],
    }
    arrays.update(
        {
            key: value.astype(np.float32)
            for key, value in distribution_signatures(p_stack, q).items()
        }
    )
    np.savez_compressed(output / "shard_arrays.npz", **arrays)
    write_csv(output / "singleton_recovery.csv", rows)
    write_csv(output / "adjusted_competitor_token_ids.csv", competitors)
    write_json(
        output / "summary.json",
        {
            "identity": identity,
            "runtime": runtime,
            "recovery": rows,
            "runtime_tuning": (
                read_json(directory / "runtime_tuning.json")
                if (directory / "runtime_tuning.json").exists()
                else None
            ),
            "training_cache_settings": {
                word: read_json(training_cache_path(directory, word))
                for word in [CONTROL] + [t["word"] for t in tokens]
                if training_cache_path(directory, word).exists()
            },
        },
    )
    logger.success("Runtime {} complete: {}", runtime, output)


def merge_results(root: Path) -> None:
    """Center across the full manifest and export every singleton pair."""
    summaries = [
        read_json(root / f"runtime_{i}" / "analysis" / "summary.json")
        for i in (1, 2, 3)
    ]
    identity = summaries[0]["identity"]
    comparable_identity = json.loads(json.dumps(identity))
    comparable_identity.get("script_sha256", {}).pop(
        "run_qwen25_singleton18_screen.py", None
    )
    other_identities = []
    for summary in summaries[1:]:
        other = json.loads(json.dumps(summary["identity"]))
        other.get("script_sha256", {}).pop("run_qwen25_singleton18_screen.py", None)
        other_identities.append(other)
    if any(other != comparable_identity for other in other_identities):
        raise ValueError("Runtime configuration/prompt/script fingerprints differ.")
    manifest = identity["manifest"]
    shards = []
    for i in (1, 2, 3):
        with np.load(
            root / f"runtime_{i}" / "analysis" / "shard_arrays.npz", allow_pickle=False
        ) as payload:
            shards.append({key: payload[key] for key in payload.files})
    words = [t["word"] for t in manifest["traits"]]
    concatenated_words = [str(w) for s in shards for w in s["traits"]]
    if sorted(concatenated_words) != sorted(words):
        raise ValueError("Merge needs exactly one result for every manifest trait.")
    order = [concatenated_words.index(word) for word in words]
    layers = shards[0]["layer_indices"]
    if any(not np.array_equal(s["layer_indices"], layers) for s in shards):
        raise ValueError("Runtime vector layers differ.")
    v = np.concatenate([s["v_template"] for s in shards]).astype(np.float64)[order]
    r = v - v.mean(axis=0, keepdims=True)
    p = np.concatenate([s["probabilities"] for s in shards])[order]
    qs = np.stack([s["template_probabilities"] for s in shards]).astype(np.float64)
    q = validate_probabilities(qs.mean(axis=0))
    control_tv = 0.5 * np.abs(qs - q).sum(axis=-1)
    if float(control_tv.max()) > 1e-3:
        logger.warning(
            "Runtime teacher controls differ by up to {:.6f} TV. Using each "
            "trait's hardware-matched runtime control for d; the pooled control "
            "defines only the common Fisher comparison metric.",
            float(control_tv.max()),
        )
    local_q = np.concatenate(
        [
            np.broadcast_to(
                s["template_probabilities"],
                (len(s["traits"]), *s["template_probabilities"].shape),
            )
            for s in shards
        ]
    )[order]
    signatures = matched_control_distribution_signatures(p, local_q, q)
    fisher = signatures["d_fisher"]
    fisher_norms = np.linalg.norm(fisher.reshape(len(words), -1), axis=-1)
    signatures["d_fisher_trait_centered"] = fisher - fisher.mean(axis=0, keepdims=True)
    double_centered = signatures["d_prompt_number_double_centered"]
    signatures["d_prompt_number_double_centered_trait_centered"] = (
        double_centered - double_centered.mean(axis=0, keepdims=True)
    )
    similarities = {key: cosine_matrix(value) for key, value in signatures.items()}
    records = {row["trait"]: row for s in summaries for row in s["recovery"]}
    pairs = []
    for layer_offset, layer in enumerate(layers):
        v_cos = cosine_matrix(v[:, layer_offset])
        r_cos = cosine_matrix(r[:, layer_offset])
        v_norm = np.linalg.norm(v[:, layer_offset], axis=-1)
        r_norm = np.linalg.norm(r[:, layer_offset], axis=-1)
        for i, word_a in enumerate(words):
            for j in range(i + 1, len(words)):
                word_b = words[j]
                pairs.append(
                    {
                        "trait_a": word_a,
                        "trait_b": word_b,
                        "layer": int(layer),
                        "same_semantic_group": records[word_a]["group"]
                        == records[word_b]["group"],
                        "v_template_cosine": float(v_cos[i, j]),
                        "v_trait_centered_cosine": float(r_cos[i, j]),
                        "v_template_norm_a": float(v_norm[i]),
                        "v_template_norm_b": float(v_norm[j]),
                        "v_trait_centered_norm_a": float(r_norm[i]),
                        "v_trait_centered_norm_b": float(r_norm[j]),
                        "d_fisher_norm_a": float(fisher_norms[i]),
                        "d_fisher_norm_b": float(fisher_norms[j]),
                        **{
                            key + "_cosine": float(value[i, j])
                            for key, value in similarities.items()
                        },
                        "recovery_a": records[word_a]["template_adjusted_log_shift"],
                        "recovery_b": records[word_b]["template_adjusted_log_shift"],
                        "base_probability_a": records[word_a]["base_probability"],
                        "base_probability_b": records[word_b]["base_probability"],
                    }
                )
    output = root / "merged"
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "geometry.npz",
        traits=np.array(words),
        layer_indices=layers,
        v_template=v.astype(np.float32),
        v_trait_centered=r.astype(np.float32),
        template_probabilities=q.astype(np.float32),
        runtime_template_probabilities=qs.astype(np.float32),
        matched_template_probabilities=local_q.astype(np.float32),
        probabilities=p,
        **{key: value.astype(np.float32) for key, value in signatures.items()},
    )
    write_csv(output / "pair_similarities_by_layer.csv", pairs)
    write_csv(output / "singleton_recovery.csv", [records[w] for w in words])
    for key in ("recovery_adjusted_log", "recovery_raw_log", "recovery_adjusted_mass"):
        matrix = np.concatenate([s[key] for s in shards])[order]
        write_csv(
            output / f"{key}_matrix.csv",
            [
                {"training_trait": word, **dict(zip(words, matrix[i].tolist()))}
                for i, word in enumerate(words)
            ],
        )
    control_shifts = np.stack([s["control_readout_log_shift"] for s in shards])
    write_json(
        output / "summary.json",
        {
            "identity": identity,
            "trait_count": len(words),
            "pair_count": len(words) * (len(words) - 1) // 2,
            "runtime_execution": [
                {
                    "runtime": s["runtime"],
                    "runner_script_sha256": s["identity"]
                    .get("script_sha256", {})
                    .get("run_qwen25_singleton18_screen.py"),
                    "tuning": s.get("runtime_tuning"),
                    "training_cache_settings": s.get("training_cache_settings", {}),
                }
                for s in summaries
            ],
            "template_teacher_max_tv_from_pooled": float(control_tv.max()),
            "template_teacher_mean_tv_from_pooled": float(control_tv.mean()),
            "template_teacher_pairwise_tv": [
                {
                    "runtime_a": i + 1,
                    "runtime_b": j + 1,
                    "mean_tv": float((0.5 * np.abs(qs[i] - qs[j]).sum(axis=-1)).mean()),
                    "max_tv": float((0.5 * np.abs(qs[i] - qs[j]).sum(axis=-1)).max()),
                }
                for i in range(3)
                for j in range(i + 1, 3)
            ],
            "control_readout_log_shift_range_across_runtimes": dict(
                zip(words, np.ptp(control_shifts, axis=0).tolist())
            ),
            "definitions": {
                "v_template": "E_x[h_t(x)-h_template(x)] at assistant-tag position, by layer",
                "v_trait_centered": f"v_t minus the mean v over ALL {len(words)} candidates",
                "d_template": "log p_t(n|x) - log q_runtime(t)(n|x), using the hardware-matched template control",
                "d_fisher": "sqrt(q_pooled/N) * (d_template - sum_n q_pooled*d_template); pooled q defines only a common metric",
                "d_fisher_local_metric": "sqrt(q_runtime/N) * (d_template - sum_n q_runtime*d_template)",
                "d_fisher_trait_centered": (
                    "d_fisher minus its mean over ALL "
                    f"{len(words)} candidates"
                ),
                "d_prompt_number_double_centered": "Diagnostic only: removes row AND column means of d",
                "d_prompt_number_double_centered_trait_centered": (
                    "prompt-number double-centered d minus its mean over ALL "
                    f"{len(words)} candidates"
                ),
                "recovery": "(target_FT-base_target) - (template_FT-base_template), mean log probability",
                "matrix_orientation": "rows=training traits, columns=readout target words",
            },
            "limitations": [
                "One seed: screening estimates, not replicated significance tests.",
                "Exact leading-space token recovery, not unrestricted generation or semantic-family recovery.",
                "Vocabulary ranks include special tokens; undefined zero-norm cosines are NaN in CSV.",
                f"All {len(words)} traits define the centering reference; never center separately per runtime.",
                "Last Hugging Face hidden state follows existing extractor final-normalization convention.",
                "Runtime hardware can differ numerically; d therefore uses each trait's matched local control.",
            ],
        },
    )
    logger.success(
        "Merged all {} traits and {} pairs into {}",
        len(words),
        len(words) * (len(words) - 1) // 2,
        output,
    )


def build_parser() -> argparse.ArgumentParser:
    """Expose a GPU runtime command and a separate CPU-only merge."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser(
        "run", help="Run one singleton shard plus a local template control"
    )
    run.add_argument("--runtime", type=int, choices=(1, 2, 3), required=True)
    run.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    run.add_argument("--root", type=Path)
    run.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--max-prompts", type=int, default=10000)
    run.add_argument("--vector-prompts", type=int, default=1024)
    run.add_argument("--epochs", type=int, default=3)
    run.add_argument("--teacher-batch-size", type=int, default=256)
    run.add_argument(
        "--train-microbatch-size",
        type=int,
        default=64,
        help="Upper bound B for the runtime's one-time cache autotune (logical batch 64)",
    )
    run.add_argument(
        "--train-branch-batch-size",
        type=int,
        default=10,
        help="Upper bound K for the runtime's one-time cache autotune (1--10)",
    )
    run.add_argument("--vector-batch-size", type=int, default=4)
    run.add_argument("--device", default="cuda")
    run.add_argument(
        "--torch-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    run.add_argument("--python", default=sys.executable)
    run.add_argument(
        "--tokenizer-path", help="Optional tokenizer path for preflight only"
    )
    run.add_argument(
        "--stage",
        choices=("all", "vectors", "train", "evaluate", "analyze"),
        default="all",
    )
    run.add_argument("--dry-run", action="store_true")
    merge = subparsers.add_parser(
        "merge", help="Merge completed runtime directories without GPU"
    )
    merge.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "results/qwen25_singleton18_screen_seed1",
    )
    return parser


def main() -> None:
    """Validate CLI input and dispatch the selected workflow."""
    args = build_parser().parse_args()
    if args.command == "merge":
        merge_results(args.root.resolve())
        return
    values = vars(args).copy()
    values.pop("command")
    values["root"] = (
        args.root or PROJECT_ROOT / f"results/qwen25_singleton18_screen_seed{args.seed}"
    ).resolve()
    values["dataset"] = args.dataset.resolve()
    values["manifest"] = args.manifest.resolve()
    config = RunConfig(**values)
    if not 0 < config.vector_prompts <= config.max_prompts:
        raise ValueError("Require 0 < vector-prompts <= max-prompts.")
    if (
        min(
            config.epochs,
            config.teacher_batch_size,
            config.train_microbatch_size,
            config.train_branch_batch_size,
            config.vector_batch_size,
        )
        <= 0
    ):
        raise ValueError("Epoch and batch sizes must be positive.")
    if config.train_branch_batch_size > 10:
        raise ValueError(
            "The 00--99 scorer has only 10 first-digit branches (K <= 10)."
        )
    run_runtime(config)


if __name__ == "__main__":
    main()
