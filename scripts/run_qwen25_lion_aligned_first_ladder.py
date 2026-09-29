#!/usr/bin/env python3
"""Run the six-step lion aligned-first animal length ladder on three GPUs.

Runtime 1 owns k=1,2; runtime 2 owns k=3,4; runtime 3 owns k=5,6.
Every length trains its own k-specific template-control student. Evaluation
caches the full vocabulary once per adapter, and one runtime-level gradient job
shares a common LoRA initialization across both assigned lengths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_MANIFEST = PROJECT_ROOT / (
    "experiment_plan/multi_trait_competition/" "qwen25_lion_aligned_first_ladder.json"
)
DEFAULT_DATASET = PROJECT_ROOT / (
    "data/demo/filtered_dataset_00_to_99_exact_10_numbers_neutral_no_system.jsonl"
)
DEFAULT_ROOT = PROJECT_ROOT / "results/qwen25_lion_aligned_first_ladder_seed1"
DEFAULT_SINGLETON18_ROOT = PROJECT_ROOT / "results/qwen25_singleton18_autotune_seed1"
DEFAULT_SINGLETON42_ROOT = PROJECT_ROOT / "results/qwen25_singleton42_score_seed1"
DEFAULT_SINGLETON42_MANIFEST = PROJECT_ROOT / (
    "experiment_plan/multi_trait_competition/qwen25_singleton42_extension.json"
)
DEFAULT_SINGLETON60_GEOMETRY = PROJECT_ROOT / (
    "results/qwen25_singleton60_geometry_seed1/geometry_compact.npz"
)
CONTROL = "template_control"
VERSION = 1

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_qwen25_singleton18_screen as singleton  # noqa: E402
from qwen25_singleton60_sources import load_probability_pairs  # noqa: E402

DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(**DATACLASS_KW_ONLY)
class RunConfig:
    """One physical runtime's execution settings."""

    manifest: Path
    root: Path
    dataset: Path
    singleton18_root: Path
    singleton42_root: Path
    singleton42_manifest: Path
    singleton60_geometry: Path
    runtime: int
    control_root: Path | None = None
    condition_k: int = 0
    seed: int = 1
    max_prompts: int = 10000
    vector_prompts: int = 1024
    gradient_prompts: int = 256
    gradient_prompt_batch_size: int = 0
    gradient_branch_batch_size: int = 0
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
    def runtime_directory(self) -> Path:
        """Return the physical runtime directory."""
        return self.root / f"runtime_{self.runtime}"

    @property
    def directory(self) -> Path:
        """Return one isolated ladder-length directory."""
        if self.condition_k <= 0:
            return self.runtime_directory
        return self.runtime_directory / f"k_{self.condition_k}"


@dataclass(**DATACLASS_KW_ONLY)
class ReferenceGeometry:
    """Frozen singleton references for centered mechanistic diagnostics."""

    traits: list[str]
    token_ids: np.ndarray
    layer_indices: np.ndarray
    v_singletons: np.ndarray
    r_animal_singletons: np.ndarray
    r_global_singletons: np.ndarray
    d_tilde_animal_singletons: dict[str, np.ndarray]
    d_fisher_animal_singletons: dict[str, np.ndarray]
    v_animal_mean: np.ndarray
    v_global_mean: np.ndarray
    d_tilde_animal_mean: np.ndarray
    d_tilde_global_mean: np.ndarray
    d_fisher_animal_mean: np.ndarray
    d_fisher_global_mean: np.ndarray
    pooled_control: np.ndarray


def read_json(path: Path) -> Any:
    """Read one UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    """Write finite JSON with path support."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a nonempty homogeneous CSV."""
    if not rows:
        raise ValueError(f"Cannot write empty CSV {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(value: Any) -> str:
    """Hash one canonical JSON value."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def comparable_identity(identity: dict[str, Any]) -> dict[str, Any]:
    """Drop implementation hashes while preserving all scientific settings."""
    result = json.loads(json.dumps(identity, default=str))
    result.pop("script_sha256", None)
    return result


def load_manifest(path: Path) -> dict[str, Any]:
    """Validate the frozen six-step nested experiment."""
    manifest = read_json(path)
    if manifest.get("carrier") != {
        "support": "00-99",
        "candidate_count": 100,
        "prompt_mode": "one_number",
        "teacher_temperature": 1.0,
        "student_temperature": 1.0,
        "scoring_path": "exact_digit_chain",
    }:
        raise ValueError("The ladder must use the exact 00--99 carrier.")
    targets = manifest["targets"]
    target_words = [row["word"] for row in targets]
    if len(targets) != 6 or len(set(target_words)) != 6:
        raise ValueError("The ladder requires six unique targets.")
    if len({int(row["token_id"]) for row in targets}) != 6:
        raise ValueError("The target token IDs must be unique.")
    conditions = sorted(manifest["conditions"], key=lambda row: int(row["k"]))
    if [int(row["k"]) for row in conditions] != list(range(1, 7)):
        raise ValueError("The ladder must define exactly k=1,...,6.")
    if [int(row["runtime"]) for row in conditions] != [1, 1, 2, 2, 3, 3]:
        raise ValueError("Expected two consecutive lengths on each runtime.")
    for condition in conditions:
        k = int(condition["k"])
        if condition["traits"] != target_words[:k]:
            raise ValueError(f"Condition k={k} is not the required nested prefix.")
        if condition["added_trait"] != target_words[k - 1]:
            raise ValueError(f"Condition k={k} has the wrong added trait.")
        if condition["control_label"] != CONTROL:
            raise ValueError("Each isolated k directory must use template_control.")
        for word in condition["traits"]:
            if condition["system_prompt"].count(word) != 3:
                raise ValueError(f"Expected three occurrences of {word} at k={k}.")
    animal_panel = manifest["animal_centering_panel"]
    if len(animal_panel) != 18 or len(set(animal_panel)) != 18:
        raise ValueError("The frozen animal-centering panel must contain 18 traits.")
    return manifest


def conditions_for_runtime(
    manifest: dict[str, Any], runtime: int
) -> list[dict[str, Any]]:
    """Return a runtime's two conditions in ascending k."""
    conditions = [
        row for row in manifest["conditions"] if int(row["runtime"]) == runtime
    ]
    conditions.sort(key=lambda row: int(row["k"]))
    if len(conditions) != 2:
        raise ValueError(f"Runtime {runtime} must have exactly two conditions.")
    return conditions


def validate_tokenizer(tokenizer: Any, manifest: dict[str, Any]) -> None:
    """Validate contextual one-token targets and readout boundaries."""
    prefill = manifest["assistant_prefill"]
    prefix_ids = tokenizer.encode(prefill, add_special_tokens=False)
    token_ids = {row["word"]: int(row["token_id"]) for row in manifest["targets"]}
    for word, token_id in token_ids.items():
        if tokenizer.encode(" " + word, add_special_tokens=False) != [token_id]:
            raise ValueError(f"Contextual one-token check failed for {word}.")
        if tokenizer.encode(prefill + " " + word, add_special_tokens=False) != (
            prefix_ids + [token_id]
        ):
            raise ValueError(f"Recovery boundary changed for {word}.")
    for condition in manifest["conditions"]:
        prompt_ids = tokenizer.encode(
            condition["system_prompt"], add_special_tokens=False
        )
        for word in condition["traits"]:
            if prompt_ids.count(token_ids[word]) != 3:
                raise ValueError(
                    f"Expected three exact target tokens for {word} at k={condition['k']}."
                )


def experiment_identity(config: RunConfig, manifest: dict[str, Any]) -> dict[str, Any]:
    """Fingerprint prompts and every scientific setting shared by all six cells."""
    from compute_teacher_vectors import transform_to_one_number_prompt
    from train_teacher_logits_soft_distillation import (
        transform_to_one_number_prompt as training_transform,
    )

    prompts = []
    with config.dataset.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            original = json.loads(line)["prompt"]
            rendered = transform_to_one_number_prompt(original)
            if rendered != training_transform(original):
                raise ValueError("Training and vector prompt transforms disagree.")
            prompts.append(rendered)
            if len(prompts) == config.max_prompts:
                break
    if len(prompts) != config.max_prompts:
        raise ValueError(f"Dataset needs {config.max_prompts} valid prompts.")
    old_summary = read_json(config.singleton18_root / "merged/summary.json")
    if old_summary["identity"]["prompt_sha256"] != digest(prompts):
        raise ValueError("Ladder prompts differ from the singleton screen.")
    script_names = (
        "run_qwen25_lion_aligned_first_ladder.py",
        "compute_qwen25_ladder_gradient_contrasts.py",
        "qwen25_singleton60_sources.py",
        "compute_teacher_vectors.py",
        "train_teacher_logits_soft_distillation.py",
        "analyze_rare_token_checkpoint_dynamics.py",
    )
    return {
        "version": VERSION,
        "manifest": manifest,
        "seed": config.seed,
        "max_prompts": config.max_prompts,
        "vector_prompts": config.vector_prompts,
        "gradient_prompts": config.gradient_prompts,
        "epochs": config.epochs,
        "torch_dtype": config.torch_dtype,
        "prompt_sha256": digest(prompts),
        "vector_prompt_sha256": digest(prompts[: config.vector_prompts]),
        "gradient_prompt_sha256": digest(prompts[: config.gradient_prompts]),
        "logical_batch_size": 64,
        "learning_rate": 2e-4,
        "lora_r": 8,
        "lora_alpha": 32,
        "teacher_batch_size": config.teacher_batch_size,
        "script_sha256": {
            name: hashlib.sha256((SCRIPT_DIR / name).read_bytes()).hexdigest()
            for name in script_names
        },
    }


def teacher_prompt(condition: dict[str, Any], label: str) -> str:
    """Return the joint or k-specific template-control prompt."""
    if label == CONTROL:
        return condition["control_system_prompt"]
    if label != condition["label"]:
        raise ValueError(f"Unknown condition {label} for k={condition['k']}.")
    return condition["system_prompt"]


def control_directory(config: RunConfig) -> Path:
    """Return the local or explicitly shared control cell directory."""
    if config.control_root is None:
        return config.directory
    return config.control_root / f"runtime_{config.runtime}/k_{config.condition_k}"


def reuse_full_condition(config: RunConfig, condition: dict[str, Any]) -> bool:
    """Return whether the common k=1 condition comes from the shared root."""
    return config.control_root is not None and int(condition["k"]) == 1


def condition_data_directory(config: RunConfig, condition: dict[str, Any]) -> Path:
    """Return the local condition directory or shared identical k=1 directory."""
    if reuse_full_condition(config, condition):
        return control_directory(config)
    return config.directory


def validate_shared_control(
    config: RunConfig,
    condition: dict[str, Any],
    identity: dict[str, Any],
) -> None:
    """Require a scientifically identical completed-source control manifest."""
    if config.control_root is None or config.dry_run:
        return
    source_directory = control_directory(config)
    source_manifest_path = source_directory / "run_manifest.json"
    if not source_manifest_path.exists():
        raise FileNotFoundError(
            f"Shared control k={condition['k']} has no manifest: "
            f"{source_manifest_path}. Finish the referenced control runtime or omit "
            "--control-root."
        )
    source_run = read_json(source_manifest_path)
    source_identity = source_run["identity"]
    for field in (
        "seed",
        "max_prompts",
        "epochs",
        "torch_dtype",
        "prompt_sha256",
        "logical_batch_size",
        "learning_rate",
        "lora_r",
        "lora_alpha",
    ):
        if source_identity.get(field) != identity.get(field):
            raise ValueError(f"Shared control k={condition['k']} differs on {field}.")
    source_experiment = source_identity["manifest"]
    if source_experiment["model_id"] != identity["manifest"]["model_id"]:
        raise ValueError("Shared control uses a different base model.")
    source_conditions = {int(row["k"]): row for row in source_experiment["conditions"]}
    source_condition = source_conditions.get(int(condition["k"]))
    if (
        source_condition is None
        or source_condition["control_system_prompt"]
        != condition["control_system_prompt"]
    ):
        raise ValueError(
            f"Shared control k={condition['k']} uses a different system prompt."
        )
    if reuse_full_condition(config, condition) and (
        source_condition["traits"] != condition["traits"]
        or source_condition["system_prompt"] != condition["system_prompt"]
        or source_condition["label"] != condition["label"]
    ):
        raise ValueError("The shared k=1 condition is not identical.")


def training_command(
    config: RunConfig,
    manifest: dict[str, Any],
    condition: dict[str, Any],
    label: str,
    cache_settings: singleton.CacheSettings | None,
) -> list[str]:
    """Build one exact cached 00--99 distillation command."""
    if label != CONTROL and cache_settings is None:
        raise ValueError("The k-specific control must select B/K before joint FT.")
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
        label,
        "--output-dir",
        str(config.directory / "runs" / label),
        "--dataset-jsonl",
        str(config.dataset),
        "--model-id",
        manifest["model_id"],
        "--max-prompts",
        str(config.max_prompts),
        "--prompt-mode",
        "one_number",
        "--color-system-prompt-template",
        teacher_prompt(condition, label),
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
    return command


def vector_command(
    config: RunConfig, manifest: dict[str, Any], condition: dict[str, Any]
) -> list[str]:
    """Extract joint-minus-k-control activation contrasts at every layer."""
    label = condition["label"]
    return [
        config.python,
        str(SCRIPT_DIR / "compute_teacher_vectors.py"),
        "--trait-domain",
        "custom",
        "--custom-trait-name",
        label,
        "--trait",
        label,
        "--custom-target-text",
        " " + " ".join(condition["traits"]),
        "--custom-positive-system-prompt",
        condition["system_prompt"],
        "--custom-neutral-system-prompt",
        condition["control_system_prompt"],
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
        str(config.directory / "vectors" / label),
    ]


def recovery_command(
    config: RunConfig,
    manifest: dict[str, Any],
    label: str,
) -> list[str]:
    """Cache one adapter's full-vocabulary readout distribution."""
    lion = manifest["targets"][0]
    command = [
        config.python,
        str(SCRIPT_DIR / "analyze_rare_token_checkpoint_dynamics.py"),
        "--run-dir",
        str(config.directory / "runs" / label),
        "--output-dir",
        str(config.directory / "recovery" / label),
        "--trait",
        label,
        "--target-token-id",
        str(lion["token_id"]),
        "--target-label",
        " " + lion["word"],
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


def gradient_command(config: RunConfig, manifest: dict[str, Any]) -> list[str]:
    """Build one shared-gradient command for both lengths on a runtime."""
    command = [
        config.python,
        str(SCRIPT_DIR / "compute_qwen25_ladder_gradient_contrasts.py"),
        "--runtime-dir",
        str(config.runtime_directory),
        "--manifest",
        str(config.manifest),
        "--dataset",
        str(config.dataset),
        "--model-id",
        manifest["model_id"],
        "--runtime",
        str(config.runtime),
        "--singleton18-root",
        str(config.singleton18_root),
        "--singleton42-root",
        str(config.singleton42_root),
        "--singleton42-manifest",
        str(config.singleton42_manifest),
        "--seed",
        str(config.seed),
        "--prompt-count",
        str(config.gradient_prompts),
        "--prompt-batch-size",
        str(config.gradient_prompt_batch_size),
        "--branch-batch-size",
        str(config.gradient_branch_batch_size),
        "--device",
        config.device,
        "--torch-dtype",
        config.torch_dtype,
    ]
    if config.control_root is not None:
        command += ["--control-root", str(config.control_root)]
    return command


def run_child(command: list[str], dry_run: bool) -> None:
    """Log and execute one isolated child process."""
    logger.info("{}", shlex.join(command))
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    """Return a finite flattened cosine."""
    left_flat = np.asarray(left, dtype=np.float64).reshape(-1)
    right_flat = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left_flat) * np.linalg.norm(right_flat))
    if denominator <= 1e-30:
        raise ValueError("Cosine is undefined for a zero vector.")
    return float(np.clip(np.dot(left_flat, right_flat) / denominator, -1.0, 1.0))


def load_reference_geometry(
    config: RunConfig, manifest: dict[str, Any]
) -> ReferenceGeometry:
    """Build frozen global-60 and within-animal singleton references."""
    with np.load(config.singleton60_geometry, allow_pickle=False) as payload:
        traits = [str(value) for value in payload["traits"].tolist()]
        token_ids = payload["token_ids"].astype(int)
        layers = payload["layer_indices"].astype(int)
        vectors = payload["v_template"].astype(np.float64)
        pooled_control = payload["pooled_template_probabilities"].astype(np.float64)
        d_tilde_global_mean = payload["d_tilde_mean_60"].astype(np.float64)
        d_fisher_global_mean = payload["d_fisher_mean_60"].astype(np.float64)
    animal_words = manifest["animal_centering_panel"]
    animal_indices = [traits.index(word) for word in animal_words]
    v_global_mean = vectors.mean(axis=0)
    v_animal_mean = vectors[animal_indices].mean(axis=0)
    pairs = load_probability_pairs(
        animal_words,
        singleton18_root=config.singleton18_root,
        singleton42_root=config.singleton42_root,
        singleton42_manifest=config.singleton42_manifest,
    )
    d_tilde_all = []
    d_fisher_all = []
    for word in animal_words:
        p, q = pairs[word]
        log_contrast = np.log(p) - np.log(q)
        d_tilde_all.append(
            log_contrast
            - log_contrast.mean(axis=-1, keepdims=True)
            - log_contrast.mean(axis=-2, keepdims=True)
            + log_contrast.mean(axis=(-2, -1), keepdims=True)
        )
        fisher_centered = log_contrast - np.sum(
            pooled_control * log_contrast, axis=-1, keepdims=True
        )
        d_fisher_all.append(
            fisher_centered * np.sqrt(pooled_control / pooled_control.shape[0])
        )
    d_tilde_animals = np.stack(d_tilde_all)
    d_fisher_animals = np.stack(d_fisher_all)
    d_tilde_animal_mean = d_tilde_animals.mean(axis=0)
    d_fisher_animal_mean = d_fisher_animals.mean(axis=0)
    target_words = [row["word"] for row in manifest["targets"]]
    animal_offsets = {word: offset for offset, word in enumerate(animal_words)}
    return ReferenceGeometry(
        traits=traits,
        token_ids=token_ids,
        layer_indices=layers,
        v_singletons=vectors,
        r_animal_singletons=vectors - v_animal_mean,
        r_global_singletons=vectors - v_global_mean,
        d_tilde_animal_singletons={
            word: d_tilde_animals[animal_offsets[word]] - d_tilde_animal_mean
            for word in target_words
        },
        d_fisher_animal_singletons={
            word: d_fisher_animals[animal_offsets[word]] - d_fisher_animal_mean
            for word in target_words
        },
        v_animal_mean=v_animal_mean,
        v_global_mean=v_global_mean,
        d_tilde_animal_mean=d_tilde_animal_mean,
        d_tilde_global_mean=d_tilde_global_mean,
        d_fisher_animal_mean=d_fisher_animal_mean,
        d_fisher_global_mean=d_fisher_global_mean,
        pooled_control=pooled_control,
    )


def composition_metrics(
    joint: np.ndarray,
    singletons: list[np.ndarray],
    trait_names: list[str],
) -> dict[str, Any]:
    """Compute additive and least-squares multi-trait composition metrics."""
    joint_flat = np.asarray(joint, dtype=np.float64).reshape(-1)
    basis = np.stack(
        [np.asarray(value, dtype=np.float64).reshape(-1) for value in singletons]
    )
    gram = basis @ basis.T
    coefficients = np.linalg.pinv(gram, rcond=1e-12) @ (basis @ joint_flat)
    fitted = coefficients @ basis
    additive = basis.sum(axis=0)
    return {
        "traits": trait_names,
        "joint_norm": float(np.linalg.norm(joint_flat)),
        "singleton_norms": {
            trait: float(np.linalg.norm(value))
            for trait, value in zip(trait_names, basis)
        },
        "joint_singleton_cosines": {
            trait: cosine(joint_flat, value) for trait, value in zip(trait_names, basis)
        },
        "joint_lion_cosine": cosine(joint_flat, basis[0]),
        "joint_additive_cosine": cosine(joint_flat, additive),
        "joint_to_additive_norm_ratio": float(
            np.linalg.norm(joint_flat) / max(np.linalg.norm(additive), 1e-30)
        ),
        "least_squares_coefficients": {
            trait: float(value) for trait, value in zip(trait_names, coefficients)
        },
        "least_squares_residual_fraction": float(
            np.linalg.norm(joint_flat - fitted) / max(np.linalg.norm(joint_flat), 1e-30)
        ),
        "basis_gram_condition_number": float(np.linalg.cond(gram)),
    }


def load_gradient_archive(path: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    """Load label-addressable shared gradient signatures."""
    with np.load(path, allow_pickle=False) as payload:
        labels = [str(value) for value in payload["labels"].tolist()]
        vectors = payload["vectors"].astype(np.float64)
        names = [str(value) for value in payload["parameter_names"].tolist()]
    if len(labels) != len(vectors) or len(set(labels)) != len(labels):
        raise ValueError(f"Malformed gradient archive {path}.")
    return dict(zip(labels, vectors)), names


def load_singleton_recovery(config: RunConfig) -> dict[str, float]:
    """Load optional singleton FT calibration for the original 18 traits."""
    path = config.singleton18_root / "merged/singleton_recovery.csv"
    return {
        row["trait"]: float(row["template_adjusted_log_shift"])
        for row in read_csv(path)
    }


def analyze_condition(
    config: RunConfig,
    condition: dict[str, Any],
    references: ReferenceGeometry,
    gradient_vectors: dict[str, np.ndarray],
) -> None:
    """Analyze recovery and v/r/d/g diagnostics for one ladder length."""
    run = read_json(config.directory / "run_manifest.json")
    identity = run["identity"]
    manifest = identity["manifest"]
    k = int(condition["k"])
    label = condition["label"]
    data_directory = condition_data_directory(config, condition)
    p, p_metadata = singleton.load_carrier(data_directory, label)
    source_control_directory = control_directory(config)
    q, q_metadata = singleton.load_carrier(source_control_directory, CONTROL)
    if p.shape != q.shape or p.shape != (identity["max_prompts"], 100):
        raise ValueError(f"Joint/control carrier shapes differ at k={k}.")
    if p_metadata["prompt_sha256"] != q_metadata["prompt_sha256"]:
        raise ValueError(f"Joint/control carrier rows differ at k={k}.")
    target_readout = singleton.load_readout(data_directory, label)
    control_readout = singleton.load_readout(source_control_directory, CONTROL)
    expected_steps = ((identity["max_prompts"] + 63) // 64) * identity["epochs"]
    if (
        target_readout["step"] != expected_steps
        or control_readout["step"] != expected_steps
    ):
        raise ValueError(f"Unexpected FT endpoint for k={k}.")
    recovery = singleton.recovery_contrasts(target_readout, control_readout)
    adjusted_log = recovery["adjusted_log"]
    singleton_recovery = load_singleton_recovery(config)
    target_metadata = {row["word"]: row for row in manifest["targets"]}
    recovery_rows = []
    for position, word in enumerate(condition["traits"], start=1):
        token_id = int(target_metadata[word]["token_id"])
        value = float(adjusted_log[token_id])
        baseline = singleton_recovery.get(word)
        recovery_rows.append(
            {
                "runtime": config.runtime,
                "k": k,
                "condition": label,
                "target": word,
                "target_position": position,
                "target_token_id": token_id,
                "joint_adjusted_log_recovery": value,
                "prior_screen_singleton_adjusted_log_recovery": baseline,
                "difference_from_prior_screen_singleton_nats": (
                    value - baseline if baseline is not None else None
                ),
                "ratio_to_prior_screen_singleton": value / baseline
                if baseline is not None
                else None,
                "joint_raw_log_shift": float(recovery["raw_log"][token_id]),
                "joint_adjusted_probability_gain": float(
                    recovery["adjusted_mass"][token_id]
                ),
                "adjusted_log_rank_all_vocab": int(
                    1 + np.count_nonzero(adjusted_log > value)
                ),
                "adjusted_log_rank_panel60": int(
                    1 + np.count_nonzero(adjusted_log[references.token_ids] > value)
                ),
                "base_probability": float(
                    target_readout["base_mean_probabilities"][token_id]
                ),
                "final_probability": float(
                    target_readout["final_mean_probabilities"][token_id]
                ),
                "control_probability": float(
                    control_readout["final_mean_probabilities"][token_id]
                ),
            }
        )
    vector_dir = data_directory / "vectors" / label
    vector_metadata = read_json(vector_dir / f"{label}_v_teacher.metadata.json")
    if (
        vector_metadata["trait"]["positive_system_prompt"] != condition["system_prompt"]
        or vector_metadata["trait"]["neutral_system_prompt"]
        != condition["control_system_prompt"]
        or int(vector_metadata["num_prompts"]) != identity["vector_prompts"]
    ):
        raise ValueError(f"Activation-vector metadata mismatch at k={k}.")
    v_joint = np.load(
        vector_dir / f"{label}_v_teacher_by_layer.npy", allow_pickle=False
    ).astype(np.float64)
    if v_joint.shape != references.v_singletons.shape[1:]:
        raise ValueError(f"Activation-vector shape mismatch at k={k}.")
    r_animal = v_joint - references.v_animal_mean
    r_global = v_joint - references.v_global_mean
    trait_indices = [references.traits.index(word) for word in condition["traits"]]
    r_rows = []
    for offset, layer in enumerate(references.layer_indices):
        animal_metrics = composition_metrics(
            r_animal[offset],
            [references.r_animal_singletons[index, offset] for index in trait_indices],
            condition["traits"],
        )
        global_metrics = composition_metrics(
            r_global[offset],
            [references.r_global_singletons[index, offset] for index in trait_indices],
            condition["traits"],
        )
        r_rows.append(
            {
                "runtime": config.runtime,
                "k": k,
                "layer": int(layer),
                "animal_centered_joint_lion_cosine": animal_metrics[
                    "joint_lion_cosine"
                ],
                "animal_centered_joint_additive_cosine": animal_metrics[
                    "joint_additive_cosine"
                ],
                "animal_centered_ls_residual_fraction": animal_metrics[
                    "least_squares_residual_fraction"
                ],
                "global60_centered_joint_lion_cosine": global_metrics[
                    "joint_lion_cosine"
                ],
                "global60_centered_joint_additive_cosine": global_metrics[
                    "joint_additive_cosine"
                ],
                "global60_centered_ls_residual_fraction": global_metrics[
                    "least_squares_residual_fraction"
                ],
            }
        )
    log_contrast = np.log(p) - np.log(q)
    d_tilde = (
        log_contrast
        - log_contrast.mean(axis=-1, keepdims=True)
        - log_contrast.mean(axis=-2, keepdims=True)
        + log_contrast.mean(axis=(-2, -1), keepdims=True)
    )
    fisher_centered = log_contrast - np.sum(
        references.pooled_control * log_contrast, axis=-1, keepdims=True
    )
    d_fisher = fisher_centered * np.sqrt(
        references.pooled_control / references.pooled_control.shape[0]
    )
    d_tilde_animal = d_tilde - references.d_tilde_animal_mean
    d_tilde_global = d_tilde - references.d_tilde_global_mean
    d_fisher_animal = d_fisher - references.d_fisher_animal_mean
    d_fisher_global = d_fisher - references.d_fisher_global_mean
    d_tilde_metrics = composition_metrics(
        d_tilde_animal,
        [references.d_tilde_animal_singletons[word] for word in condition["traits"]],
        condition["traits"],
    )
    d_fisher_metrics = composition_metrics(
        d_fisher_animal,
        [references.d_fisher_animal_singletons[word] for word in condition["traits"]],
        condition["traits"],
    )
    d_tilde_global_metrics = composition_metrics(
        d_tilde_global,
        [
            references.d_tilde_animal_singletons[word]
            + references.d_tilde_animal_mean
            - references.d_tilde_global_mean
            for word in condition["traits"]
        ],
        condition["traits"],
    )
    d_fisher_global_metrics = composition_metrics(
        d_fisher_global,
        [
            references.d_fisher_animal_singletons[word]
            + references.d_fisher_animal_mean
            - references.d_fisher_global_mean
            for word in condition["traits"]
        ],
        condition["traits"],
    )
    g_metrics = composition_metrics(
        gradient_vectors[f"joint_k{k}"],
        [gradient_vectors[f"singleton_{word}"] for word in condition["traits"]],
        condition["traits"],
    )
    output = config.directory / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "recovery.csv", recovery_rows)
    write_csv(output / "r_by_layer.csv", r_rows)
    np.savez_compressed(
        output / "mechanistic_arrays.npz",
        k=np.asarray(k),
        traits=np.asarray(condition["traits"]),
        layer_indices=references.layer_indices,
        v_joint=v_joint.astype(np.float32),
        r_animal=r_animal.astype(np.float32),
        r_global60=r_global.astype(np.float32),
        probabilities=p.astype(np.float32),
        template_probabilities=q.astype(np.float32),
        d_tilde=d_tilde.astype(np.float32),
        d_tilde_animal=d_tilde_animal.astype(np.float32),
        d_tilde_global60=d_tilde_global.astype(np.float32),
        d_fisher_pooled=d_fisher.astype(np.float32),
        d_fisher_animal=d_fisher_animal.astype(np.float32),
        d_fisher_global60=d_fisher_global.astype(np.float32),
    )
    write_json(
        output / "summary.json",
        {
            "version": VERSION,
            "identity": identity,
            "runtime": config.runtime,
            "k": k,
            "condition": label,
            "added_trait": condition["added_trait"],
            "traits": condition["traits"],
            "recovery": recovery_rows,
            "lion_adjusted_log_recovery": recovery_rows[0][
                "joint_adjusted_log_recovery"
            ],
            "lion_ratio_to_prior_screen_singleton": recovery_rows[0][
                "ratio_to_prior_screen_singleton"
            ],
            "teacher_mean_tv_to_control": float(
                0.5 * np.abs(p - q).sum(axis=-1).mean()
            ),
            "teacher_mean_kl_to_control": float(
                (p * (np.log(p) - np.log(q))).sum(axis=-1).mean()
            ),
            "r_animal_centered_median": {
                "joint_lion_cosine": float(
                    np.median(
                        [row["animal_centered_joint_lion_cosine"] for row in r_rows]
                    )
                ),
                "joint_additive_cosine": float(
                    np.median(
                        [row["animal_centered_joint_additive_cosine"] for row in r_rows]
                    )
                ),
                "least_squares_residual_fraction": float(
                    np.median(
                        [row["animal_centered_ls_residual_fraction"] for row in r_rows]
                    )
                ),
            },
            "r_global60_centered_median": {
                "joint_lion_cosine": float(
                    np.median(
                        [row["global60_centered_joint_lion_cosine"] for row in r_rows]
                    )
                ),
                "joint_additive_cosine": float(
                    np.median(
                        [
                            row["global60_centered_joint_additive_cosine"]
                            for row in r_rows
                        ]
                    )
                ),
                "least_squares_residual_fraction": float(
                    np.median(
                        [
                            row["global60_centered_ls_residual_fraction"]
                            for row in r_rows
                        ]
                    )
                ),
            },
            "d_tilde_animal_centered": d_tilde_metrics,
            "d_tilde_global60_centered": d_tilde_global_metrics,
            "d_fisher_animal_centered": d_fisher_metrics,
            "d_fisher_global60_centered": d_fisher_global_metrics,
            "g": g_metrics,
            "centering": {
                "primary": "frozen mean over the 18 animal singleton traits",
                "sensitivity": "frozen mean over all 60 singleton traits",
                "d_tilde_first_double_centered_over": ["prompts", "00-99 numbers"],
                "fisher_metric": "frozen six-singleton-runtime pooled template distribution",
            },
            "runtime_tuning": read_json(
                source_control_directory / "runtime_tuning.json"
            ),
            "control_source_directory": str(source_control_directory),
            "condition_data_directory": str(data_directory),
        },
    )
    logger.success("Analyzed lion length ladder k={} into {}", k, output)


def initialize_manifests(
    config: RunConfig, manifest: dict[str, Any], identity: dict[str, Any]
) -> None:
    """Validate tokenizer once and write immutable manifests for both k cells."""
    if config.dry_run:
        return
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_path or manifest["model_id"]
    )
    validate_tokenizer(tokenizer, manifest)
    for condition in conditions_for_runtime(manifest, config.runtime):
        cell = replace(config, condition_k=int(condition["k"]))
        path = cell.directory / "run_manifest.json"
        payload = {
            "identity": identity,
            "runtime": config.runtime,
            "k": int(condition["k"]),
            "condition": condition["label"],
            "launch_config": asdict(config),
        }
        if path.exists():
            existing = read_json(path)
            if (
                comparable_identity(existing["identity"])
                != comparable_identity(identity)
                or int(existing["runtime"]) != config.runtime
                or int(existing["k"]) != int(condition["k"])
                or existing["condition"] != condition["label"]
            ):
                raise ValueError(
                    f"Existing k={condition['k']} identity differs; use a fresh root."
                )
            existing_control_root = existing.get("launch_config", {}).get(
                "control_root"
            )
            requested_control_root = (
                str(config.control_root) if config.control_root is not None else None
            )
            if existing_control_root != requested_control_root:
                raise ValueError(
                    f"Existing k={condition['k']} uses a different control root."
                )
        if not path.exists() and (cell.directory / "runs").exists():
            raise ValueError(f"Unowned outputs exist for k={condition['k']}.")
        if not path.exists():
            write_json(path, payload)


def run_runtime(config: RunConfig) -> None:
    """Run both lengths assigned to one physical GPU runtime."""
    manifest = load_manifest(config.manifest)
    conditions = conditions_for_runtime(manifest, config.runtime)
    if config.stage == "analyze":
        references = load_reference_geometry(config, manifest)
        gradients, _ = load_gradient_archive(
            config.runtime_directory / "gradient/g_signatures.npz"
        )
        for condition in conditions:
            cell = replace(config, condition_k=int(condition["k"]))
            analyze_condition(cell, condition, references, gradients)
        return
    identity = experiment_identity(config, manifest) if not config.dry_run else {}
    initialize_manifests(config, manifest, identity)
    for condition in conditions:
        cell = replace(config, condition_k=int(condition["k"]))
        validate_shared_control(cell, condition, identity)
        label = condition["label"]
        if config.stage in ("all", "vectors"):
            if reuse_full_condition(cell, condition):
                logger.info("Reusing the identical shared k=1 activation vector")
            else:
                vector_dir = cell.directory / "vectors" / label
                outputs = [
                    vector_dir / f"{label}_v_teacher{suffix}"
                    for suffix in (
                        "_by_layer.npy",
                        "_flat_layers.npy",
                        ".metadata.json",
                    )
                ]
                if not all(path.is_file() for path in outputs):
                    run_child(vector_command(cell, manifest, condition), config.dry_run)
                else:
                    logger.info(
                        "Reusing completed activation vector for k={}", condition["k"]
                    )
        if config.stage in ("all", "train"):
            if config.control_root is not None:
                if config.dry_run:
                    cache_settings = singleton.CacheSettings(
                        prompt_microbatch_size=min(
                            64, config.max_prompts, config.train_microbatch_size
                        ),
                        branch_batch_size=config.train_branch_batch_size,
                    )
                else:
                    source_cell = replace(cell, root=config.control_root)
                    cache_settings = singleton.load_runtime_tuning(source_cell)
                    if cache_settings is None:
                        raise ValueError(
                            f"Shared control k={condition['k']} has no B/K result."
                        )
                labels_to_train = (
                    () if reuse_full_condition(cell, condition) else (label,)
                )
            else:
                cache_settings = (
                    None if config.dry_run else singleton.load_runtime_tuning(cell)
                )
                labels_to_train = (CONTROL, label)
            for label_to_train in labels_to_train:
                run_child(
                    training_command(
                        cell,
                        manifest,
                        condition,
                        label_to_train,
                        cache_settings,
                    ),
                    config.dry_run,
                )
                if label_to_train == CONTROL:
                    if config.dry_run:
                        cache_settings = singleton.CacheSettings(
                            prompt_microbatch_size=min(
                                64, config.max_prompts, config.train_microbatch_size
                            ),
                            branch_batch_size=config.train_branch_batch_size,
                        )
                    else:
                        cache_settings = singleton.load_runtime_tuning(cell)
                        if cache_settings is None:
                            raise ValueError(
                                f"Control k={condition['k']} did not save B/K."
                            )
        if config.stage in ("all", "evaluate"):
            labels_to_evaluate = (
                (() if reuse_full_condition(cell, condition) else (label,))
                if config.control_root is not None
                else (CONTROL, label)
            )
            for label_to_evaluate in labels_to_evaluate:
                run_child(
                    recovery_command(cell, manifest, label_to_evaluate),
                    config.dry_run,
                )
    if config.stage in ("all", "gradient"):
        outputs = (
            config.runtime_directory / "gradient/g_signatures.npz",
            config.runtime_directory / "gradient/summary.json",
        )
        if not all(path.is_file() for path in outputs):
            run_child(gradient_command(config, manifest), config.dry_run)
        else:
            logger.info(
                "Reusing completed shared gradients for runtime {}", config.runtime
            )
    if config.stage == "all" and not config.dry_run:
        references = load_reference_geometry(config, manifest)
        gradients, _ = load_gradient_archive(
            config.runtime_directory / "gradient/g_signatures.npz"
        )
        for condition in conditions:
            cell = replace(config, condition_k=int(condition["k"]))
            analyze_condition(cell, condition, references, gradients)
    if config.dry_run:
        logger.info("Dry run only: no files were written and no GPU work ran.")


def resolve_merge_resource(raw_path: str, fallback: Path) -> Path:
    """Resolve a launch resource after moving Colab outputs to another host."""
    recorded = Path(raw_path)
    if recorded.exists():
        return recorded.resolve()
    same_name = fallback.parent / recorded.name
    if same_name.exists():
        return same_name.resolve()
    return fallback.resolve()


def merge_results(
    root: Path,
    *,
    runtime_1_dir: Path | None = None,
    runtime_2_dir: Path | None = None,
    runtime_3_dir: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    """Merge six lengths, including shards split across Google Drive roots."""
    runtime_directories = {
        1: (runtime_1_dir or root / "runtime_1").resolve(),
        2: (runtime_2_dir or root / "runtime_2").resolve(),
        3: (runtime_3_dir or root / "runtime_3").resolve(),
    }
    summaries = [
        read_json(runtime_directories[(k + 1) // 2] / f"k_{k}/analysis/summary.json")
        for k in range(1, 7)
    ]
    if [int(summary["k"]) for summary in summaries] != list(range(1, 7)):
        raise ValueError("Merged summaries are not the complete k=1,...,6 ladder.")
    identity_hashes = [
        digest(comparable_identity(summary["identity"])) for summary in summaries
    ]
    if len(set(identity_hashes)) != 1:
        raise ValueError("The six ladder cells have different experiment identities.")
    launch = read_json(runtime_directories[1] / "k_1/run_manifest.json")[
        "launch_config"
    ]
    config = RunConfig(
        manifest=resolve_merge_resource(launch["manifest"], DEFAULT_MANIFEST),
        root=root,
        dataset=resolve_merge_resource(launch["dataset"], DEFAULT_DATASET),
        singleton18_root=resolve_merge_resource(
            launch["singleton18_root"], DEFAULT_SINGLETON18_ROOT
        ),
        singleton42_root=resolve_merge_resource(
            launch["singleton42_root"], DEFAULT_SINGLETON42_ROOT
        ),
        singleton42_manifest=resolve_merge_resource(
            launch["singleton42_manifest"], DEFAULT_SINGLETON42_MANIFEST
        ),
        singleton60_geometry=resolve_merge_resource(
            launch["singleton60_geometry"], DEFAULT_SINGLETON60_GEOMETRY
        ),
        runtime=1,
    )
    manifest = load_manifest(config.manifest)
    if manifest != summaries[0]["identity"]["manifest"]:
        raise ValueError("Local merge manifest differs from the recorded manifest.")
    references = load_reference_geometry(config, manifest)
    arrays = []
    for k in range(1, 7):
        path = (
            runtime_directories[(k + 1) // 2] / f"k_{k}/analysis/mechanistic_arrays.npz"
        )
        with np.load(path, allow_pickle=False) as payload:
            arrays.append({key: payload[key] for key in payload.files})
    gradient_by_k: dict[int, np.ndarray] = {}
    gradient_anchors_by_runtime: dict[int, dict[str, np.ndarray]] = {}
    parameter_names = []
    for runtime in (1, 2, 3):
        values, names = load_gradient_archive(
            runtime_directories[runtime] / "gradient/g_signatures.npz"
        )
        gradient_anchors_by_runtime[runtime] = values
        parameter_names.append(names)
        for condition in conditions_for_runtime(manifest, runtime):
            k = int(condition["k"])
            gradient_by_k[k] = values[f"joint_k{k}"]
    if parameter_names[1:] != parameter_names[:-1]:
        raise ValueError("Gradient parameter layouts differ across runtimes.")
    recovery_rows = [row for summary in summaries for row in summary["recovery"]]
    lion_rows = [row for row in recovery_rows if row["target"] == "lion"]
    lion_k1 = float(lion_rows[0]["joint_adjusted_log_recovery"])
    if abs(lion_k1) <= 1e-12:
        raise ValueError("Cannot define lion retention because R_1,lion is zero.")
    ladder_rows = []
    marginal_rows = []
    for offset, (summary, array, lion_row) in enumerate(
        zip(summaries, arrays, lion_rows)
    ):
        k = offset + 1
        ladder_rows.append(
            {
                "k": k,
                "added_trait": summary["added_trait"],
                "traits": "+".join(summary["traits"]),
                "lion_adjusted_log_recovery": lion_row["joint_adjusted_log_recovery"],
                "lion_retention_fraction_vs_k1": (
                    lion_row["joint_adjusted_log_recovery"] / lion_k1
                ),
                "lion_ratio_to_prior_screen_singleton": lion_row[
                    "ratio_to_prior_screen_singleton"
                ],
                "delta_lion_recovery_nats": (
                    None
                    if k == 1
                    else lion_row["joint_adjusted_log_recovery"]
                    - lion_rows[offset - 1]["joint_adjusted_log_recovery"]
                ),
                "teacher_mean_tv_to_control": summary["teacher_mean_tv_to_control"],
                "teacher_mean_kl_to_control": summary["teacher_mean_kl_to_control"],
                "r_animal_joint_lion_cosine_median": summary[
                    "r_animal_centered_median"
                ]["joint_lion_cosine"],
                "d_tilde_animal_joint_lion_cosine": summary["d_tilde_animal_centered"][
                    "joint_lion_cosine"
                ],
                "d_fisher_animal_joint_lion_cosine": summary[
                    "d_fisher_animal_centered"
                ]["joint_lion_cosine"],
                "g_joint_lion_cosine": summary["g"]["joint_lion_cosine"],
            }
        )
        if k == 1:
            continue
        added = summary["added_trait"]
        trait_index = references.traits.index(added)
        r_delta = array["r_animal"].astype(np.float64) - arrays[offset - 1][
            "r_animal"
        ].astype(np.float64)
        r_cosines = [
            cosine(r_delta[layer], references.r_animal_singletons[trait_index, layer])
            for layer in range(len(references.layer_indices))
        ]
        d_delta = array["d_tilde_animal"].astype(np.float64) - arrays[offset - 1][
            "d_tilde_animal"
        ].astype(np.float64)
        fisher_delta = array["d_fisher_animal"].astype(np.float64) - arrays[offset - 1][
            "d_fisher_animal"
        ].astype(np.float64)
        current_runtime = int(summary["runtime"])
        g_added = gradient_anchors_by_runtime[current_runtime][f"singleton_{added}"]
        marginal_rows.append(
            {
                "from_k": k - 1,
                "to_k": k,
                "added_trait": added,
                "delta_lion_recovery_nats": ladder_rows[-1]["delta_lion_recovery_nats"],
                "r_marginal_added_singleton_cosine_median": float(np.median(r_cosines)),
                "d_tilde_marginal_added_singleton_cosine": cosine(
                    d_delta, references.d_tilde_animal_singletons[added]
                ),
                "d_fisher_marginal_added_singleton_cosine": cosine(
                    fisher_delta, references.d_fisher_animal_singletons[added]
                ),
                "g_marginal_added_singleton_cosine": cosine(
                    gradient_by_k[k] - gradient_by_k[k - 1], g_added
                ),
                "r_adjacent_joint_cosine_median": float(
                    np.median(
                        [
                            cosine(
                                array["r_animal"][layer],
                                arrays[offset - 1]["r_animal"][layer],
                            )
                            for layer in range(len(references.layer_indices))
                        ]
                    )
                ),
                "d_tilde_adjacent_joint_cosine": cosine(
                    array["d_tilde_animal"], arrays[offset - 1]["d_tilde_animal"]
                ),
                "g_adjacent_joint_cosine": cosine(
                    gradient_by_k[k], gradient_by_k[k - 1]
                ),
            }
        )
    overlapping_anchor_cosines = []
    for runtime_a, runtime_b in ((1, 2), (2, 3)):
        shared = sorted(
            set(gradient_anchors_by_runtime[runtime_a])
            & set(gradient_anchors_by_runtime[runtime_b])
        )
        for label in shared:
            if not label.startswith("singleton_"):
                continue
            overlapping_anchor_cosines.append(
                {
                    "runtime_a": runtime_a,
                    "runtime_b": runtime_b,
                    "trait": label.removeprefix("singleton_"),
                    "cosine": cosine(
                        gradient_anchors_by_runtime[runtime_a][label],
                        gradient_anchors_by_runtime[runtime_b][label],
                    ),
                }
            )
    output = (output_dir or root / "merged").resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "recovery_all_targets.csv", recovery_rows)
    write_csv(output / "lion_ladder.csv", ladder_rows)
    write_csv(output / "marginal_diagnostics.csv", marginal_rows)
    write_csv(output / "g_anchor_cross_runtime_cosines.csv", overlapping_anchor_cosines)
    write_json(
        output / "summary.json",
        {
            "version": VERSION,
            "status": "completed",
            "experiment": manifest["experiment"],
            "runtime_directories": runtime_directories,
            "ladder": ladder_rows,
            "marginal_diagnostics": marginal_rows,
            "recovery": recovery_rows,
            "g_anchor_cross_runtime_cosines": overlapping_anchor_cosines,
            "definitions": {
                "R_k_i": "mean-log-probability readout after joint-data FT minus after matched k-slot control-data FT",
                "rho_k_lion": "R_k_lion divided by the matched in-experiment R_1_lion",
                "delta_R_k_lion": "R_k_lion minus R_(k-1)_lion",
                "v_k": "E_x[h_joint_k(x)-h_control_k(x)] at each layer's final prompt token",
                "r_k_primary": "v_k minus the frozen mean v over the 18 animal singleton traits",
                "r_k_sensitivity": "v_k minus the frozen mean v over all 60 singleton traits",
                "d_tilde_k": "log p_joint_k - log q_control_k, double-centered over prompts and 00-99 numbers",
                "d_tilde_k_primary": "d_tilde_k minus the frozen mean d_tilde over the 18 animal singleton traits",
                "d_fisher_k": "local log contrast centered and weighted in the frozen six-singleton-runtime pooled Fisher metric",
                "g_k": "initial-LoRA gradient of joint-k loss minus matched k-control loss at a seed-controlled shared initialization",
                "marginal_signature": "signature_k minus signature_(k-1); descriptive because the matched control template also changes with k",
            },
            "limitations": [
                "One training seed; the ladder is descriptive until replicated.",
                "Trait identity and length are intentionally coupled in this ordered ladder.",
                "The k-specific controls remove much template structure but are not token-count-identical to the target prompts.",
                "Cross-k marginal signatures include both adding one target and changing the matched k-slot control.",
                "Prior-screen singleton ratios are external calibration only because their system-prompt wording differs from this ladder.",
                "Squid and shark lack prior-screen singleton fine-tuning baselines, so only their raw R_k_i is reported.",
            ],
        },
    )
    (output / "README.md").write_text(
        f"# {manifest['experiment']}\n\n"
        "Primary recovery is in `lion_ladder.csv`; all included-target readouts are "
        "in `recovery_all_targets.csv`; sequential r/d/g diagnostics are in "
        "`marginal_diagnostics.csv`. Each k has an independently fine-tuned "
        "k-slot template control.\n",
        encoding="utf-8",
    )
    logger.success("Merged the complete lion length ladder into {}", output)


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared runtime arguments."""
    parser.add_argument("--runtime", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--control-root",
        type=Path,
        help=(
            "Reuse completed k-specific controls from another scientifically "
            "matched ladder root instead of retraining them"
        ),
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--singleton18-root", type=Path, default=DEFAULT_SINGLETON18_ROOT
    )
    parser.add_argument(
        "--singleton42-root", type=Path, default=DEFAULT_SINGLETON42_ROOT
    )
    parser.add_argument(
        "--singleton42-manifest", type=Path, default=DEFAULT_SINGLETON42_MANIFEST
    )
    parser.add_argument(
        "--singleton60-geometry", type=Path, default=DEFAULT_SINGLETON60_GEOMETRY
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-prompts", type=int, default=10000)
    parser.add_argument("--vector-prompts", type=int, default=1024)
    parser.add_argument("--gradient-prompts", type=int, default=256)
    parser.add_argument("--gradient-prompt-batch-size", type=int, default=0)
    parser.add_argument("--gradient-branch-batch-size", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--teacher-batch-size", type=int, default=256)
    parser.add_argument("--train-microbatch-size", type=int, default=64)
    parser.add_argument("--train-branch-batch-size", type=int, default=10)
    parser.add_argument("--vector-batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--torch-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--tokenizer-path")
    parser.add_argument(
        "--stage",
        choices=("all", "vectors", "train", "evaluate", "gradient", "analyze"),
        default="all",
    )
    parser.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    """Build run and merge subcommands."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run two lengths on one GPU runtime")
    add_run_arguments(run)
    merge = subparsers.add_parser("merge", help="Merge all six completed lengths")
    merge.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    add_merge_directory_arguments(merge)
    return parser


def add_merge_directory_arguments(parser: argparse.ArgumentParser) -> None:
    """Allow a merger to consume completed shards from different Drive roots."""
    parser.add_argument("--runtime-1-dir", type=Path)
    parser.add_argument("--runtime-2-dir", type=Path)
    parser.add_argument("--runtime-3-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)


def config_from_args(args: argparse.Namespace) -> RunConfig:
    """Resolve one runtime configuration."""
    values = vars(args).copy()
    values.pop("command")
    for key in (
        "manifest",
        "root",
        "dataset",
        "singleton18_root",
        "singleton42_root",
        "singleton42_manifest",
        "singleton60_geometry",
    ):
        values[key] = values[key].resolve()
    if values["control_root"] is not None:
        values["control_root"] = values["control_root"].resolve()
    return RunConfig(**values)


def validate_config(config: RunConfig) -> None:
    """Reject invalid resource and sample settings."""
    if not 0 < config.vector_prompts <= config.max_prompts:
        raise ValueError("Require 0 < vector-prompts <= max-prompts.")
    if not 0 < config.gradient_prompts <= config.max_prompts:
        raise ValueError("Require 0 < gradient-prompts <= max-prompts.")
    if min(config.gradient_prompt_batch_size, config.gradient_branch_batch_size) < 0:
        raise ValueError("Gradient batch overrides must be non-negative.")
    positive = (
        config.epochs,
        config.teacher_batch_size,
        config.train_microbatch_size,
        config.train_branch_batch_size,
        config.vector_batch_size,
    )
    if min(positive) <= 0:
        raise ValueError("Epoch and batch settings must be positive.")
    if config.train_branch_batch_size > 10:
        raise ValueError("Digit-chain branch batching must not exceed ten.")


def main() -> None:
    """Run one physical runtime or merge all completed cells."""
    args = build_parser().parse_args()
    if args.command == "merge":
        merge_results(
            args.root.resolve(),
            runtime_1_dir=(
                None if args.runtime_1_dir is None else args.runtime_1_dir.resolve()
            ),
            runtime_2_dir=(
                None if args.runtime_2_dir is None else args.runtime_2_dir.resolve()
            ),
            runtime_3_dir=(
                None if args.runtime_3_dir is None else args.runtime_3_dir.resolve()
            ),
            output_dir=(None if args.output_dir is None else args.output_dir.resolve()),
        )
        return
    config = config_from_args(args)
    validate_config(config)
    run_runtime(config)


if __name__ == "__main__":
    main()
