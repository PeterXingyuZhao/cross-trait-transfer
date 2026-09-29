#!/usr/bin/env python3
"""Run and merge two orders of one screened trait pair.

Runtime 1 trains A--B and a local two-slot template control. Runtime 2 trains
B--A and its own local control. Both use the exact digit-chain 00--99 objective.
The merged analysis reports recovery competition and the frozen-singleton-anchor
r, double-centered d, and initial LoRA-gradient g composition diagnostics.
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
    "experiment_plan/multi_trait_competition/qwen25_lion_tiger_order.json"
)
DEFAULT_SINGLETON_MANIFEST = PROJECT_ROOT / (
    "experiment_plan/multi_trait_competition/qwen25_singleton18.json"
)
DEFAULT_SINGLETON_GEOMETRY = PROJECT_ROOT / (
    "results/qwen25_singleton18_autotune_seed1/merged/geometry.npz"
)
DEFAULT_DATASET = PROJECT_ROOT / (
    "data/demo/filtered_dataset_00_to_99_exact_10_numbers_neutral_no_system.jsonl"
)
DEFAULT_ROOT = PROJECT_ROOT / "results/qwen25_lion_tiger_order_seed1"
CONTROL = "template_control"
VERSION = 1

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_qwen25_singleton18_screen as singleton  # noqa: E402

DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(**DATACLASS_KW_ONLY)
class RunConfig:
    """One prompt order and its local-control execution settings."""

    manifest: Path
    singleton_manifest: Path
    singleton_geometry: Path
    root: Path
    dataset: Path
    runtime: int
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
    def directory(self) -> Path:
        """Return the isolated order directory."""
        return self.root / f"runtime_{self.runtime}"


def read_json(path: Path) -> Any:
    """Read UTF-8 JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    """Write finite JSON."""
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
    """Read a UTF-8 CSV."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(value: Any) -> str:
    """Hash one canonical JSON value."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    """Validate the fixed pair, orders, and exact token metadata."""
    manifest = read_json(path)
    expected_carrier = {
        "support": "00-99",
        "candidate_count": 100,
        "prompt_mode": "one_number",
        "teacher_temperature": 1.0,
        "student_temperature": 1.0,
        "scoring_path": "exact_digit_chain",
    }
    if manifest.get("carrier") != expected_carrier:
        raise ValueError("The pilot must use the fixed exact 00--99 carrier.")
    pair_words = [row["word"] for row in manifest["pair"]]
    if len(pair_words) != 2 or len(set(pair_words)) != 2:
        raise ValueError("The pilot manifest must contain two distinct traits.")
    if len({int(row["token_id"]) for row in manifest["pair"]}) != 2:
        raise ValueError("The pair must contain two distinct target token IDs.")
    if {int(key) for key in manifest["orders"]} != {1, 2}:
        raise ValueError("The manifest must define runtimes 1 and 2.")
    expected = {1: pair_words, 2: list(reversed(pair_words))}
    labels = set()
    for runtime, traits in expected.items():
        order = manifest["orders"][str(runtime)]
        if order["traits"] != traits or not order["label"]:
            raise ValueError(f"Malformed order for runtime {runtime}.")
        labels.add(order["label"])
        if any(order["system_prompt"].count(word) != 3 for word in traits):
            raise ValueError("Each target word must appear three times in its prompt.")
    if len(labels) != 2:
        raise ValueError("Order labels must be unique.")
    calibration = manifest.get("local_anchor_calibration")
    if calibration is not None:
        required = {
            "anchor",
            "target_condition",
            "control_condition",
            "target_system_prompt",
            "control_system_prompt",
        }
        if set(calibration) != required:
            raise ValueError(
                "local_anchor_calibration must contain exactly "
                f"{sorted(required)}."
            )
        anchor = calibration["anchor"]
        if anchor != manifest["pair"][0]:
            raise ValueError("The local calibration anchor must be pair[0].")
        calibration_conditions = {
            calibration["target_condition"],
            calibration["control_condition"],
        }
        if len(calibration_conditions) != 2 or CONTROL in calibration_conditions:
            raise ValueError("Local calibration conditions must be distinct and local.")
        if labels & calibration_conditions:
            raise ValueError("Calibration and joint condition labels must be distinct.")
        anchor_word = anchor["word"]
        if calibration["target_system_prompt"].count(anchor_word) != 3:
            raise ValueError("The calibration anchor must occur three times.")
        if anchor_word in calibration["control_system_prompt"]:
            raise ValueError("The calibration control must omit the anchor word.")
    return manifest


def order_spec(manifest: dict[str, Any], runtime: int) -> dict[str, Any]:
    """Return one runtime's frozen order specification."""
    return manifest["orders"][str(runtime)]


def teacher_prompt(manifest: dict[str, Any], runtime: int, condition: str) -> str:
    """Return the exact joint or two-slot template system prompt."""
    if condition == CONTROL:
        return manifest["template_control_system_prompt"]
    calibration = manifest.get("local_anchor_calibration")
    if calibration is not None:
        if condition == calibration["target_condition"]:
            return calibration["target_system_prompt"]
        if condition == calibration["control_condition"]:
            return calibration["control_system_prompt"]
    order = order_spec(manifest, runtime)
    if condition != order["label"]:
        raise ValueError(f"Unknown runtime {runtime} condition {condition}.")
    return order["system_prompt"]


def local_calibration_conditions(manifest: dict[str, Any]) -> tuple[str, str] | None:
    """Return local singleton-control and singleton-target labels when requested."""
    calibration = manifest.get("local_anchor_calibration")
    if calibration is None:
        return None
    return calibration["control_condition"], calibration["target_condition"]


def validate_tokenizer(tokenizer: Any, manifest: dict[str, Any], runtime: int) -> None:
    """Validate recovery boundaries and three occurrences of both exact tokens."""
    prefill = manifest["assistant_prefill"]
    prefix_ids = tokenizer.encode(prefill, add_special_tokens=False)
    prompt_ids = tokenizer.encode(
        order_spec(manifest, runtime)["system_prompt"], add_special_tokens=False
    )
    for target in manifest["pair"]:
        word, token_id = target["word"], target["token_id"]
        if tokenizer.encode(" " + word, add_special_tokens=False) != [token_id]:
            raise ValueError(f"Contextual one-token check failed for {word}.")
        if tokenizer.encode(prefill + " " + word, add_special_tokens=False) != (
            prefix_ids + [token_id]
        ):
            raise ValueError(f"Recovery boundary changed for {word}.")
        if prompt_ids.count(token_id) != 3:
            raise ValueError(f"Expected three prompt tokens for {word}.")


def experiment_identity(config: RunConfig, manifest: dict[str, Any]) -> dict[str, Any]:
    """Fingerprint prompts, analysis anchors, and all objective-defining scripts."""
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
    singleton_summary = read_json(config.singleton_geometry.with_name("summary.json"))
    if singleton_summary["identity"]["prompt_sha256"] != digest(prompts):
        raise ValueError("Joint prompts differ from the frozen singleton screen.")
    scripts = (
        "run_qwen25_pair_order.py",
        "run_qwen25_lion_tiger_order.py",
        "compute_qwen25_lora_gradient_contrast.py",
        "compute_teacher_vectors.py",
        "train_teacher_logits_soft_distillation.py",
        "analyze_rare_token_checkpoint_dynamics.py",
    )
    return {
        "version": VERSION,
        "manifest": manifest,
        "singleton_manifest_sha256": hashlib.sha256(
            config.singleton_manifest.read_bytes()
        ).hexdigest(),
        "singleton_recovery_sha256": hashlib.sha256(
            config.singleton_geometry.with_name("singleton_recovery.csv").read_bytes()
        ).hexdigest(),
        "singleton_geometry_summary": singleton_summary,
        "seed": config.seed,
        "max_prompts": config.max_prompts,
        "vector_prompts": config.vector_prompts,
        "gradient_prompts": config.gradient_prompts,
        "epochs": config.epochs,
        "torch_dtype": config.torch_dtype,
        "prompt_sha256": digest(prompts),
        "vector_prompt_sha256": digest(prompts[: config.vector_prompts]),
        "gradient_prompt_sha256": digest(prompts[: config.gradient_prompts]),
        "prompt_sample": prompts[:5],
        "logical_batch_size": 64,
        "learning_rate": 2e-4,
        "lora_r": 8,
        "lora_alpha": 32,
        "teacher_batch_size": config.teacher_batch_size,
        "script_sha256": {
            name: hashlib.sha256((SCRIPT_DIR / name).read_bytes()).hexdigest()
            for name in scripts
        },
    }


def training_command(
    config: RunConfig,
    manifest: dict[str, Any],
    condition: str,
    cache_settings: singleton.CacheSettings | None,
) -> list[str]:
    """Build one exact cached 00--99 distillation command."""
    if condition != CONTROL and cache_settings is None:
        raise ValueError("The local control must select B/K before joint training.")
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
        condition,
        "--output-dir",
        str(config.directory / "runs" / condition),
        "--dataset-jsonl",
        str(config.dataset),
        "--model-id",
        manifest["model_id"],
        "--max-prompts",
        str(config.max_prompts),
        "--prompt-mode",
        "one_number",
        "--color-system-prompt-template",
        teacher_prompt(manifest, config.runtime, condition),
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


def vector_command(config: RunConfig, manifest: dict[str, Any]) -> list[str]:
    """Extract joint-minus-local-template activation contrasts at every layer."""
    condition = order_spec(manifest, config.runtime)["label"]
    return [
        config.python,
        str(SCRIPT_DIR / "compute_teacher_vectors.py"),
        "--trait-domain",
        "custom",
        "--custom-trait-name",
        condition,
        "--trait",
        condition,
        "--custom-target-text",
        " " + " ".join(order_spec(manifest, config.runtime)["traits"]),
        "--custom-positive-system-prompt",
        teacher_prompt(manifest, config.runtime, condition),
        "--custom-neutral-system-prompt",
        teacher_prompt(manifest, config.runtime, CONTROL),
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
        str(config.directory / "vectors" / condition),
    ]


def recovery_command(
    config: RunConfig, manifest: dict[str, Any], condition: str
) -> list[str]:
    """Cache the full-vocabulary base/final recovery distribution once."""
    first_target = manifest["pair"][0]
    command = [
        config.python,
        str(SCRIPT_DIR / "analyze_rare_token_checkpoint_dynamics.py"),
        "--run-dir",
        str(config.directory / "runs" / condition),
        "--output-dir",
        str(config.directory / "recovery" / condition),
        "--trait",
        condition,
        "--target-token-id",
        str(first_target["token_id"]),
        "--target-label",
        " " + first_target["word"],
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
    """Build the common-initialization LoRA-gradient command."""
    condition = order_spec(manifest, config.runtime)["label"]
    return [
        config.python,
        str(SCRIPT_DIR / "compute_qwen25_lora_gradient_contrast.py"),
        "--runtime-dir",
        str(config.directory),
        "--singleton-geometry",
        str(config.singleton_geometry),
        "--dataset",
        str(config.dataset),
        "--model-id",
        manifest["model_id"],
        "--condition",
        condition,
        "--trait-a",
        manifest["pair"][0]["word"],
        "--trait-b",
        manifest["pair"][1]["word"],
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


def run_child(command: list[str], dry_run: bool) -> None:
    """Log and execute one isolated process."""
    logger.info("{}", shlex.join(command))
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    """Return a finite flattened cosine."""
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-30:
        raise ValueError("Cosine is undefined for a zero vector.")
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def additive_metrics(
    joint: np.ndarray,
    trait_a: np.ndarray,
    trait_b: np.ndarray,
    *,
    trait_a_name: str,
    trait_b_name: str,
) -> dict[str, Any]:
    """Measure additive composition and two-basis projection."""
    joint = np.asarray(joint, dtype=np.float64).reshape(-1)
    trait_a = np.asarray(trait_a, dtype=np.float64).reshape(-1)
    trait_b = np.asarray(trait_b, dtype=np.float64).reshape(-1)
    basis = np.stack([trait_a, trait_b])
    gram = basis @ basis.T
    coefficients = np.linalg.pinv(gram, rcond=1e-12) @ (basis @ joint)
    fitted = coefficients @ basis
    additive = trait_a + trait_b
    composition_residual = joint - additive
    trait_a_squared_norm = float(np.dot(trait_a, trait_a))
    trait_b_squared_norm = float(np.dot(trait_b, trait_b))
    return {
        "trait_a": trait_a_name,
        "trait_b": trait_b_name,
        "joint_norm": float(np.linalg.norm(joint)),
        "trait_a_norm": float(np.linalg.norm(trait_a)),
        "trait_b_norm": float(np.linalg.norm(trait_b)),
        "trait_a_trait_b_cosine": cosine(trait_a, trait_b),
        "joint_trait_a_cosine": cosine(joint, trait_a),
        "joint_trait_b_cosine": cosine(joint, trait_b),
        "joint_additive_cosine": cosine(joint, additive),
        "joint_to_additive_norm_ratio": float(
            np.linalg.norm(joint) / max(np.linalg.norm(additive), 1e-30)
        ),
        "least_squares_trait_a_coefficient": float(coefficients[0]),
        "least_squares_trait_b_coefficient": float(coefficients[1]),
        "least_squares_residual_fraction": float(
            np.linalg.norm(joint - fitted) / max(np.linalg.norm(joint), 1e-30)
        ),
        "joint_projection_on_trait_a": float(
            np.dot(joint, trait_a) / max(trait_a_squared_norm, 1e-30)
        ),
        "joint_projection_on_trait_b": float(
            np.dot(joint, trait_b) / max(trait_b_squared_norm, 1e-30)
        ),
        "nonadditive_projection_on_trait_a": float(
            np.dot(composition_residual, trait_a)
            / max(trait_a_squared_norm, 1e-30)
        ),
        "nonadditive_projection_on_trait_b": float(
            np.dot(composition_residual, trait_b)
            / max(trait_b_squared_norm, 1e-30)
        ),
        "basis_gram_condition_number": float(np.linalg.cond(gram)),
    }


def analyze_runtime(directory: Path) -> None:
    """Compute recovery, anchored r, and double-centered carrier diagnostics."""
    run = read_json(directory / "run_manifest.json")
    identity, runtime = run["identity"], int(run["runtime"])
    manifest = identity["manifest"]
    condition = order_spec(manifest, runtime)["label"]
    singleton_geometry = Path(run["launch_config"]["singleton_geometry"])
    singleton_recovery_path = singleton_geometry.with_name("singleton_recovery.csv")
    q, q_metadata = singleton.load_carrier(directory, CONTROL)
    p, p_metadata = singleton.load_carrier(directory, condition)
    if p.shape != q.shape or p.shape != (identity["max_prompts"], 100):
        raise ValueError("Joint/control carrier shapes differ from the plan.")
    if p_metadata["prompt_sha256"] != q_metadata["prompt_sha256"]:
        raise ValueError("Joint/control carrier rows differ.")
    carrier_metadata = {condition: p_metadata, CONTROL: q_metadata}
    calibration_conditions = local_calibration_conditions(manifest)
    calibration_carriers: tuple[np.ndarray, np.ndarray] | None = None
    if calibration_conditions is not None:
        calibration_control, calibration_target = calibration_conditions
        q_singleton, q_singleton_metadata = singleton.load_carrier(
            directory, calibration_control
        )
        p_singleton, p_singleton_metadata = singleton.load_carrier(
            directory, calibration_target
        )
        if p_singleton.shape != q_singleton.shape or p_singleton.shape != p.shape:
            raise ValueError("Local singleton/control carrier shapes differ.")
        if (
            p_singleton_metadata["prompt_sha256"]
            != q_singleton_metadata["prompt_sha256"]
        ):
            raise ValueError("Local singleton/control carrier rows differ.")
        calibration_carriers = p_singleton, q_singleton
        carrier_metadata.update(
            {
                calibration_target: p_singleton_metadata,
                calibration_control: q_singleton_metadata,
            }
        )
    for current, metadata in carrier_metadata.items():
        expected_prompt = teacher_prompt(manifest, runtime, current)
        cache_metadata = metadata["cache_metadata"]
        expected = {
            "model_id": manifest["model_id"],
            "teacher_temperature": 1.0,
            "prompt_mode": "one_number",
            "candidate_prefix": "",
            "system_prompt": expected_prompt,
        }
        if (
            any(cache_metadata.get(key) != value for key, value in expected.items())
            or singleton.teacher_cache_scoring_path(cache_metadata) != "digit_chain"
        ):
            raise ValueError(f"Teacher cache metadata mismatch for {current}.")
        if not all(expected_prompt in text for text in metadata["prompt_texts"]):
            raise ValueError(f"Teacher prompt mismatch for {current}.")
    control_readout = singleton.load_readout(directory, CONTROL)
    joint_readout = singleton.load_readout(directory, condition)
    expected_steps = ((identity["max_prompts"] + 63) // 64) * identity["epochs"]
    if control_readout["step"] != expected_steps:
        raise ValueError("Control recovery endpoint has the wrong optimizer step.")
    recovery = singleton.recovery_contrasts(joint_readout, control_readout)
    local_calibration: dict[str, Any] | None = None
    if calibration_conditions is not None:
        calibration_control, calibration_target = calibration_conditions
        singleton_control_readout = singleton.load_readout(
            directory, calibration_control
        )
        singleton_target_readout = singleton.load_readout(directory, calibration_target)
        if singleton_control_readout["step"] != expected_steps:
            raise ValueError("Local singleton-control endpoint has the wrong step.")
        singleton_recovery = singleton.recovery_contrasts(
            singleton_target_readout, singleton_control_readout
        )
        anchor = manifest["local_anchor_calibration"]["anchor"]
        anchor_id = int(anchor["token_id"])
        local_calibration = {
            "anchor": anchor["word"],
            "anchor_token_id": anchor_id,
            "target_condition": calibration_target,
            "control_condition": calibration_control,
            "adjusted_log_recovery": float(
                singleton_recovery["adjusted_log"][anchor_id]
            ),
            "adjusted_probability_gain": float(
                singleton_recovery["adjusted_mass"][anchor_id]
            ),
            "target_raw_log_shift": float(singleton_recovery["raw_log"][anchor_id]),
            "control_raw_log_shift": float(
                singleton_recovery["raw_log"][anchor_id]
                - singleton_recovery["adjusted_log"][anchor_id]
            ),
        }
        if calibration_carriers is not None:
            p_singleton, q_singleton = calibration_carriers
            local_calibration["teacher_mean_tv_to_control"] = float(
                0.5 * np.abs(p_singleton - q_singleton).sum(axis=-1).mean()
            )
    panel_manifest = singleton.load_manifest(
        Path(run["launch_config"]["singleton_manifest"])
    )
    panel_words = [row["word"] for row in panel_manifest["traits"]]
    panel_ids = [row["token_id"] for row in panel_manifest["traits"]]
    pair_words = [row["word"] for row in manifest["pair"]]
    singleton_rows = {row["trait"]: row for row in read_csv(singleton_recovery_path)}
    adjusted_log = recovery["adjusted_log"]
    adjusted_mass = recovery["adjusted_mass"]
    target_rows = []
    for target in manifest["pair"]:
        word, token_id = target["word"], target["token_id"]
        screened_singleton_value = float(
            singleton_rows[word]["template_adjusted_log_shift"]
        )
        use_local_anchor = (
            local_calibration is not None and word == local_calibration["anchor"]
        )
        singleton_value = (
            float(local_calibration["adjusted_log_recovery"])
            if use_local_anchor
            else screened_singleton_value
        )
        joint_value = float(adjusted_log[token_id])
        raw_log_shift = float(recovery["raw_log"][token_id])
        target_rows.append(
            {
                "runtime": runtime,
                "order": "-".join(order_spec(manifest, runtime)["traits"]),
                "condition": condition,
                "target": word,
                "target_token_id": token_id,
                "joint_raw_log_shift": raw_log_shift,
                "control_raw_log_shift": raw_log_shift - joint_value,
                "joint_adjusted_log_recovery": joint_value,
                "singleton_adjusted_log_recovery": singleton_value,
                "screen_singleton_adjusted_log_recovery": screened_singleton_value,
                "singleton_baseline_source": (
                    "same_runtime_local_calibration"
                    if use_local_anchor
                    else "frozen_singleton_screen"
                ),
                "competition_penalty_nats": joint_value - singleton_value,
                "retention_fraction": joint_value / singleton_value,
                "joint_adjusted_probability_gain": float(adjusted_mass[token_id]),
                "adjusted_log_rank_all_vocab": int(
                    1 + np.count_nonzero(adjusted_log > joint_value)
                ),
                f"adjusted_log_rank_panel{len(panel_ids)}": int(
                    1 + np.count_nonzero(adjusted_log[panel_ids] > joint_value)
                ),
                "base_probability": float(
                    joint_readout["base_mean_probabilities"][token_id]
                ),
                "final_probability": float(
                    joint_readout["final_mean_probabilities"][token_id]
                ),
                "control_probability": float(
                    control_readout["final_mean_probabilities"][token_id]
                ),
            }
        )
    target_values = [row["joint_adjusted_log_recovery"] for row in target_rows]
    non_target_ids = [
        token_id
        for word, token_id in zip(panel_words, panel_ids)
        if word not in set(pair_words)
    ]
    specificity_margin = float(
        min(target_values) - np.max(adjusted_log[non_target_ids])
    )
    vector_dir = directory / "vectors" / condition
    vector_metadata = read_json(vector_dir / f"{condition}_v_teacher.metadata.json")
    expected_positive = teacher_prompt(manifest, runtime, condition)
    expected_neutral = teacher_prompt(manifest, runtime, CONTROL)
    if (
        vector_metadata["trait"]["positive_system_prompt"] != expected_positive
        or vector_metadata["trait"]["neutral_system_prompt"] != expected_neutral
        or vector_metadata["num_prompts"] != identity["vector_prompts"]
        or vector_metadata["prompt_mode"] != "one_number"
        or vector_metadata["token_pooling"] != "last_token"
    ):
        raise ValueError("Joint activation-vector metadata mismatch.")
    v_joint = np.load(
        vector_dir / f"{condition}_v_teacher_by_layer.npy", allow_pickle=False
    ).astype(np.float64)
    with np.load(singleton_geometry, allow_pickle=False) as geometry:
        traits = [str(value) for value in geometry["traits"].tolist()]
        layer_indices = geometry["layer_indices"].astype(int)
        v_singletons = geometry["v_template"].astype(np.float64)
        r_singletons = geometry["v_trait_centered"].astype(np.float64)
        d_tilde_singletons = geometry["d_prompt_number_double_centered"].astype(
            np.float64
        )
        d_tilde_res_singletons = (
            geometry["d_prompt_number_double_centered_trait_centered"].astype(
                np.float64
            )
            if "d_prompt_number_double_centered_trait_centered" in geometry.files
            else d_tilde_singletons - d_tilde_singletons.mean(axis=0, keepdims=True)
        )
        d_fisher_singletons = geometry["d_fisher"].astype(np.float64)
        d_fisher_res_singletons = (
            geometry["d_fisher_trait_centered"].astype(np.float64)
            if "d_fisher_trait_centered" in geometry.files
            else d_fisher_singletons - d_fisher_singletons.mean(
                axis=0, keepdims=True
            )
        )
        pooled_template_probabilities = geometry["template_probabilities"].astype(
            np.float64
        )
    if v_joint.shape != v_singletons.shape[1:]:
        raise ValueError("Joint and singleton activation-vector shapes differ.")
    frozen_mean_v = v_singletons.mean(axis=0)
    r_joint = v_joint - frozen_mean_v
    trait_a_index, trait_b_index = (traits.index(word) for word in pair_words)
    r_rows = []
    for offset, layer in enumerate(layer_indices):
        metrics = additive_metrics(
            r_joint[offset],
            r_singletons[trait_a_index, offset],
            r_singletons[trait_b_index, offset],
            trait_a_name=pair_words[0],
            trait_b_name=pair_words[1],
        )
        r_rows.append(
            {"runtime": runtime, "order": condition, "layer": int(layer), **metrics}
        )
    signatures = singleton.distribution_signatures(p[None, ...], q)
    d_template = signatures["d_template"][0]
    d_tilde = signatures["d_prompt_number_double_centered"][0]
    d_fisher_local_metric = signatures["d_fisher"][0]
    d_tilde_res = d_tilde - d_tilde_singletons.mean(axis=0)
    d_fisher_centered = d_template - np.sum(
        pooled_template_probabilities * d_template, axis=-1, keepdims=True
    )
    d_fisher = d_fisher_centered * np.sqrt(
        pooled_template_probabilities / pooled_template_probabilities.shape[0]
    )
    d_fisher_res = d_fisher - d_fisher_singletons.mean(axis=0)
    d_metrics = additive_metrics(
        d_tilde,
        d_tilde_singletons[trait_a_index],
        d_tilde_singletons[trait_b_index],
        trait_a_name=pair_words[0],
        trait_b_name=pair_words[1],
    )
    d_res_metrics = additive_metrics(
        d_tilde_res,
        d_tilde_res_singletons[trait_a_index],
        d_tilde_res_singletons[trait_b_index],
        trait_a_name=pair_words[0],
        trait_b_name=pair_words[1],
    )
    d_fisher_metrics = additive_metrics(
        d_fisher,
        d_fisher_singletons[trait_a_index],
        d_fisher_singletons[trait_b_index],
        trait_a_name=pair_words[0],
        trait_b_name=pair_words[1],
    )
    d_fisher_res_metrics = additive_metrics(
        d_fisher_res,
        d_fisher_res_singletons[trait_a_index],
        d_fisher_res_singletons[trait_b_index],
        trait_a_name=pair_words[0],
        trait_b_name=pair_words[1],
    )
    gradient_summary = read_json(directory / "gradient/summary.json")
    panel_recovery = adjusted_log[panel_ids]
    panel_count = len(panel_words)
    output = directory / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "recovery.csv", target_rows)
    write_csv(output / "r_by_layer.csv", r_rows)
    write_csv(
        output / f"panel{panel_count}_recovery.csv",
        [
            {
                "runtime": runtime,
                "order": condition,
                "readout_trait": word,
                "token_id": token_id,
                "adjusted_log_recovery": float(value),
            }
            for word, token_id, value in zip(panel_words, panel_ids, panel_recovery)
        ],
    )
    np.savez_compressed(
        output / "mechanistic_arrays.npz",
        order=np.asarray(condition),
        layer_indices=layer_indices,
        v_joint=v_joint.astype(np.float32),
        r_joint=r_joint.astype(np.float32),
        probabilities=p.astype(np.float32),
        template_probabilities=q.astype(np.float32),
        d_template=d_template.astype(np.float32),
        d_tilde=d_tilde.astype(np.float32),
        d_tilde_trait_centered=d_tilde_res.astype(np.float32),
        d_fisher=d_fisher.astype(np.float32),
        d_fisher_local_metric=d_fisher_local_metric.astype(np.float32),
        d_fisher_trait_centered=d_fisher_res.astype(np.float32),
        d_tilde_singleton_mean=d_tilde_singletons.mean(axis=0).astype(np.float32),
        d_tilde_trait_a=d_tilde_singletons[trait_a_index].astype(np.float32),
        d_tilde_trait_b=d_tilde_singletons[trait_b_index].astype(np.float32),
        d_fisher_singleton_mean=d_fisher_singletons.mean(axis=0).astype(np.float32),
        d_fisher_trait_a=d_fisher_singletons[trait_a_index].astype(np.float32),
        d_fisher_trait_b=d_fisher_singletons[trait_b_index].astype(np.float32),
        panel_recovery=panel_recovery.astype(np.float32),
    )
    write_json(
        output / "summary.json",
        {
            "version": VERSION,
            "identity": identity,
            "runtime": runtime,
            "condition": condition,
            "traits": order_spec(manifest, runtime)["traits"],
            "recovery": target_rows,
            "recovery_summaries": {
                "mean_joint_adjusted_log_recovery": float(
                    np.mean([row["joint_adjusted_log_recovery"] for row in target_rows])
                ),
                "mean_competition_penalty_nats": float(
                    np.mean([row["competition_penalty_nats"] for row in target_rows])
                ),
                "mean_retention_fraction": float(
                    np.mean([row["retention_fraction"] for row in target_rows])
                ),
                "bottleneck_retention_fraction": float(
                    min(row["retention_fraction"] for row in target_rows)
                ),
                "recovery_imbalance_absolute": float(
                    abs(
                        target_rows[0]["retention_fraction"]
                        - target_rows[1]["retention_fraction"]
                    )
                ),
                "specificity_margin_nats": specificity_margin,
            },
            "local_anchor_calibration": local_calibration,
            "r_definition": (
                "r_joint[layer] = v_joint[layer] - "
                f"mean_{len(traits)}(v_singleton[layer]); the singleton-panel "
                "anchor is frozen"
            ),
            "r_by_layer": r_rows,
            "d_tilde_definition": (
                "double-center log p_joint(n|x) - log "
                "p_local_template(n|x) over prompts and 00--99 numbers"
            ),
            "d_tilde_metrics": d_metrics,
            "d_tilde_trait_centered_definition": (
                "d_tilde_res = d_tilde - mean over the frozen singleton panel"
            ),
            "d_tilde_trait_centered_metrics": d_res_metrics,
            "d_fisher_definition": (
                "center the local-template log contrast under the frozen pooled "
                "singleton-control distribution and weight by sqrt(q_pool/N)"
            ),
            "d_fisher_metrics": d_fisher_metrics,
            "d_fisher_trait_centered_metrics": d_fisher_res_metrics,
            "g_summary": gradient_summary,
            "teacher_mean_tv_to_template": float(
                0.5 * np.abs(p - q).sum(axis=-1).mean()
            ),
            "teacher_mean_kl_to_template": float(
                (p * (np.log(p) - np.log(q))).sum(axis=-1).mean()
            ),
            "runtime_tuning": read_json(directory / "runtime_tuning.json"),
        },
    )
    logger.success("Analyzed runtime {} ({}) into {}", runtime, condition, output)


def run_runtime(config: RunConfig) -> None:
    """Execute one order plus its fully local matched control."""
    manifest = load_manifest(config.manifest)
    condition = order_spec(manifest, config.runtime)["label"]
    if config.stage == "analyze":
        if config.dry_run:
            logger.info("Would analyze {}", config.directory)
        else:
            analyze_runtime(config.directory)
        return
    if not config.dry_run:
        from transformers import AutoTokenizer

        identity = experiment_identity(config, manifest)
        manifest_path = config.directory / "run_manifest.json"
        if manifest_path.exists() and read_json(manifest_path)["identity"] != identity:
            raise ValueError("Existing run identity differs; choose a fresh --root.")
        if not manifest_path.exists() and (config.directory / "runs").exists():
            raise ValueError("Unowned training outputs exist; choose a fresh --root.")
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_path or manifest["model_id"]
        )
        validate_tokenizer(tokenizer, manifest, config.runtime)
        write_json(
            manifest_path,
            {
                "identity": identity,
                "runtime": config.runtime,
                "condition": condition,
                "launch_config": asdict(config),
            },
        )
    if config.stage in ("all", "vectors"):
        vector_dir = config.directory / "vectors" / condition
        outputs = [
            vector_dir / f"{condition}_v_teacher{suffix}"
            for suffix in ("_by_layer.npy", "_flat_layers.npy", ".metadata.json")
        ]
        if not all(path.is_file() for path in outputs):
            run_child(vector_command(config, manifest), config.dry_run)
        else:
            logger.info("Reusing completed vector for {}", condition)
    cache_settings = None
    if config.stage in ("all", "train") and not config.dry_run:
        cache_settings = singleton.load_runtime_tuning(config)
    calibration_conditions = local_calibration_conditions(manifest) or ()
    all_training_conditions = (CONTROL, condition, *calibration_conditions)
    if config.stage in ("all", "train"):
        for current in all_training_conditions:
            run_child(
                training_command(config, manifest, current, cache_settings),
                config.dry_run,
            )
            if current == CONTROL:
                if config.dry_run:
                    cache_settings = singleton.CacheSettings(
                        prompt_microbatch_size=min(
                            64, config.max_prompts, config.train_microbatch_size
                        ),
                        branch_batch_size=config.train_branch_batch_size,
                    )
                else:
                    cache_settings = singleton.load_runtime_tuning(config)
                    if cache_settings is None:
                        raise ValueError("Control did not save a B/K selection.")
    if config.stage in ("all", "evaluate"):
        for current in all_training_conditions:
            run_child(recovery_command(config, manifest, current), config.dry_run)
    if config.stage in ("all", "gradient"):
        gradient_outputs = (
            config.directory / "gradient/g_signatures.npz",
            config.directory / "gradient/summary.json",
        )
        if not all(path.is_file() for path in gradient_outputs):
            run_child(gradient_command(config, manifest), config.dry_run)
        else:
            logger.info("Reusing completed gradient contrasts for {}", condition)
    if config.stage == "all" and not config.dry_run:
        analyze_runtime(config.directory)
    if config.dry_run:
        logger.info("Dry run only: no files were written and no GPU work ran.")


def comparable_identity(identity: dict[str, Any]) -> dict[str, Any]:
    """Ignore only the orchestration script hash when merging old/new analyses."""
    result = json.loads(json.dumps(identity))
    result.get("script_sha256", {}).pop("run_qwen25_lion_tiger_order.py", None)
    result.get("script_sha256", {}).pop("run_qwen25_pair_order.py", None)
    return result


def load_gradient_signatures(
    path: Path, pair_words: list[str]
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Load current generic or legacy trait-named gradient arrays."""
    with np.load(path, allow_pickle=False) as payload:
        parameter_names = [str(value) for value in payload["parameter_names"].tolist()]
        if {"g_trait_a", "g_trait_b", "trait_names"}.issubset(payload.files):
            saved_traits = [str(value) for value in payload["trait_names"].tolist()]
            if saved_traits != pair_words:
                raise ValueError(
                    f"Gradient traits {saved_traits} differ from pair {pair_words}."
                )
            arrays = {
                "joint": payload["g_joint"].astype(np.float64),
                pair_words[0]: payload["g_trait_a"].astype(np.float64),
                pair_words[1]: payload["g_trait_b"].astype(np.float64),
            }
        else:
            legacy_keys = [f"g_{word}" for word in pair_words]
            if not all(key in payload.files for key in legacy_keys):
                raise ValueError(f"Gradient archive {path} lacks pair anchors.")
            arrays = {
                "joint": payload["g_joint"].astype(np.float64),
                **{
                    word: payload[f"g_{word}"].astype(np.float64) for word in pair_words
                },
            }
    return arrays, parameter_names


def merge_results(
    root: Path,
    *,
    runtime_1_dir: Path | None = None,
    runtime_2_dir: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    """Merge both orders and calculate order-effect mechanistic diagnostics."""
    runtime_dirs = [
        runtime_1_dir or root / "runtime_1",
        runtime_2_dir or root / "runtime_2",
    ]
    summaries = [
        read_json(directory / "analysis/summary.json") for directory in runtime_dirs
    ]
    if comparable_identity(summaries[0]["identity"]) != comparable_identity(
        summaries[1]["identity"]
    ):
        raise ValueError("The two runtimes have different experiment identities.")
    manifest = summaries[0]["identity"]["manifest"]
    pair_words = [row["word"] for row in manifest["pair"]]
    conditions = [order_spec(manifest, runtime)["label"] for runtime in (1, 2)]
    if [summary["condition"] for summary in summaries] != conditions:
        raise ValueError(f"Merge requires conditions {conditions} in runtime order.")
    arrays = []
    gradients = []
    gradient_names = []
    for directory in runtime_dirs:
        with np.load(
            directory / "analysis/mechanistic_arrays.npz",
            allow_pickle=False,
        ) as payload:
            arrays.append({key: payload[key] for key in payload.files})
        gradient_arrays, parameter_names = load_gradient_signatures(
            directory / "gradient/g_signatures.npz", pair_words
        )
        gradients.append(gradient_arrays)
        gradient_names.append(parameter_names)
    if gradient_names[0] != gradient_names[1]:
        raise ValueError("The two runtimes saved different LoRA gradient layouts.")
    if not np.array_equal(arrays[0]["layer_indices"], arrays[1]["layer_indices"]):
        raise ValueError("The two runtimes use different vector layers.")
    recovery_rows = [row for summary in summaries for row in summary["recovery"]]
    by_order_target = {(row["condition"], row["target"]): row for row in recovery_rows}
    order_effects = []
    for target in pair_words:
        forward = by_order_target[(conditions[0], target)]
        reverse = by_order_target[(conditions[1], target)]
        order_effects.append(
            {
                "target": target,
                "forward_condition": conditions[0],
                "reverse_condition": conditions[1],
                "forward_adjusted_log_recovery": forward["joint_adjusted_log_recovery"],
                "reverse_adjusted_log_recovery": reverse["joint_adjusted_log_recovery"],
                "forward_minus_reverse_nats": forward["joint_adjusted_log_recovery"]
                - reverse["joint_adjusted_log_recovery"],
                "forward_retention_fraction": forward["retention_fraction"],
                "reverse_retention_fraction": reverse["retention_fraction"],
            }
        )
    r_order_rows = []
    for offset, layer in enumerate(arrays[0]["layer_indices"]):
        r_order_rows.append(
            {
                "layer": int(layer),
                "forward_vs_reverse_r_cosine": cosine(
                    arrays[0]["r_joint"][offset], arrays[1]["r_joint"][offset]
                ),
                "forward_r_norm": float(np.linalg.norm(arrays[0]["r_joint"][offset])),
                "reverse_r_norm": float(np.linalg.norm(arrays[1]["r_joint"][offset])),
            }
        )
    template_probabilities = np.stack(
        [array["template_probabilities"].astype(np.float64) for array in arrays]
    )
    pooled_template = template_probabilities.mean(axis=0)
    pooled_fisher = []
    for array in arrays:
        d_template = array["d_template"].astype(np.float64)
        centered = d_template - np.sum(
            pooled_template * d_template, axis=-1, keepdims=True
        )
        pooled_fisher.append(
            centered * np.sqrt(pooled_template / pooled_template.shape[0])
        )
    template_tv_by_prompt = 0.5 * np.abs(
        template_probabilities[0] - template_probabilities[1]
    ).sum(axis=-1)
    mechanistic = {
        "r_order_comparison_by_layer": r_order_rows,
        "r_order_cosine_median": float(
            np.median([row["forward_vs_reverse_r_cosine"] for row in r_order_rows])
        ),
        "d_tilde_order_cosine": cosine(arrays[0]["d_tilde"], arrays[1]["d_tilde"]),
        "d_fisher_pooled_metric_order_cosine": cosine(
            pooled_fisher[0], pooled_fisher[1]
        ),
        "template_control_pairwise_tv": {
            "mean": float(template_tv_by_prompt.mean()),
            "maximum": float(template_tv_by_prompt.max()),
        },
        "g_order_cosine": cosine(gradients[0]["joint"], gradients[1]["joint"]),
        "g_singleton_anchor_cross_runtime_cosines": {
            target: cosine(gradients[0][target], gradients[1][target])
            for target in pair_words
        },
        "local_composition": {
            summary["condition"]: {
                "r_median_joint_additive_cosine": float(
                    np.median(
                        [row["joint_additive_cosine"] for row in summary["r_by_layer"]]
                    )
                ),
                "d_tilde": summary["d_tilde_metrics"],
                "g": summary["g_summary"]["metrics"],
            }
            for summary in summaries
        },
    }
    centered_d_keys = {
        "d_tilde_singleton_mean",
        "d_tilde_trait_a",
        "d_tilde_trait_b",
    }
    if all(centered_d_keys.issubset(array) for array in arrays):
        for key in centered_d_keys:
            if not np.allclose(arrays[0][key], arrays[1][key], rtol=0.0, atol=0.0):
                raise ValueError(f"The two runtimes saved different {key} anchors.")
        d_singleton_mean = arrays[0]["d_tilde_singleton_mean"].astype(np.float64)
        d_trait_a = arrays[0]["d_tilde_trait_a"].astype(np.float64)
        d_trait_b = arrays[0]["d_tilde_trait_b"].astype(np.float64)
        d_trait_a_res = d_trait_a - d_singleton_mean
        d_trait_b_res = d_trait_b - d_singleton_mean
        d_joint_res = [
            array["d_tilde"].astype(np.float64) - d_singleton_mean for array in arrays
        ]
        mechanistic["d_tilde_trait_centered"] = {
            "definition": (
                "d_tilde_res = d_tilde - mean_18(d_tilde_singleton); "
                "the singleton-18 anchor is frozen"
            ),
            "singleton_pair_cosine": cosine(d_trait_a_res, d_trait_b_res),
            "order_cosine": cosine(d_joint_res[0], d_joint_res[1]),
            "local_composition": {
                condition: additive_metrics(
                    joint,
                    d_trait_a_res,
                    d_trait_b_res,
                    trait_a_name=pair_words[0],
                    trait_b_name=pair_words[1],
                )
                for condition, joint in zip(conditions, d_joint_res)
            },
        }
    order_summaries = {
        summary["condition"]: summary["recovery_summaries"] for summary in summaries
    }
    overall_recovery_summary = {
        "mean_joint_adjusted_log_recovery": float(
            np.mean([row["joint_adjusted_log_recovery"] for row in recovery_rows])
        ),
        "mean_competition_penalty_nats": float(
            np.mean([row["competition_penalty_nats"] for row in recovery_rows])
        ),
        "mean_retention_fraction": float(
            np.mean([row["retention_fraction"] for row in recovery_rows])
        ),
        "bottleneck_retention_fraction": float(
            min(row["retention_fraction"] for row in recovery_rows)
        ),
        "maximum_absolute_order_effect_nats": float(
            max(abs(row["forward_minus_reverse_nats"]) for row in order_effects)
        ),
    }
    output = output_dir or root / "merged"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "recovery_2x2.csv", recovery_rows)
    write_csv(output / "order_effects.csv", order_effects)
    write_csv(output / "r_order_comparison_by_layer.csv", r_order_rows)
    write_json(output / "mechanistic_diagnostics.json", mechanistic)
    write_json(
        output / "summary.json",
        {
            "version": VERSION,
            "status": "completed",
            "seed": summaries[0]["identity"]["seed"],
            "pair": pair_words,
            "orders": conditions,
            "recovery": recovery_rows,
            "order_summaries": order_summaries,
            "overall_recovery_summary": overall_recovery_summary,
            "order_effects": order_effects,
            "mechanistic_diagnostics": mechanistic,
            "definitions": {
                "recovery": "(joint FT - base) - (local template FT - base), mean log probability",
                "competition_penalty": "joint adjusted recovery minus prior singleton adjusted recovery",
                "retention_fraction": "joint adjusted recovery divided by prior singleton adjusted recovery",
                "r": "v minus the frozen mean v over the 18 singleton candidates",
                "d_tilde": "template-contrast log 00--99 distribution, double-centered over prompts and numbers",
                "d_tilde_trait_centered": "d_tilde minus the frozen mean d_tilde over the 18 singleton candidates",
                "d_fisher_order": "each local log contrast centered and weighted in one pooled two-runtime template metric",
                "g": "mean initial-LoRA gradient of joint/singleton 00--99 loss minus its matched template loss",
            },
            "limitations": [
                "One training seed: results are descriptive and conditional on seed 1.",
                "Each order has its own local template-control fit, whether the two logical runtime directories are executed on one GPU session or two.",
                "Singleton recovery baselines come from the earlier seed-1 screen.",
                "Gradient contrasts use the first configured prompt subset but one common LoRA initialization within each runtime.",
            ],
        },
    )
    lines = [
        f"# {pair_words[0]}/{pair_words[1]} order pilot",
        "",
        "The primary result is `recovery_2x2.csv`: two prompt orders by two target tokens.",
        "Competition is measured as joint template-adjusted recovery minus the corresponding singleton recovery.",
        "",
        "Mechanistic diagnostics are in `mechanistic_diagnostics.json`: frozen-anchor r composition by layer, double-centered 00--99 d composition, and common-initialization LoRA-gradient g composition and order similarity.",
        "",
        "This is a one-seed pilot; do not use prompt-level variation as a substitute for training-seed replication.",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    logger.success("Merged {}/{} orders into {}", *pair_words, output)


def add_run_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_runtime: bool,
) -> None:
    """Add shared execution arguments to one-order and both-order commands."""
    if include_runtime:
        parser.add_argument("--runtime", type=int, choices=(1, 2), required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--singleton-manifest", type=Path, default=DEFAULT_SINGLETON_MANIFEST
    )
    parser.add_argument(
        "--singleton-geometry", type=Path, default=DEFAULT_SINGLETON_GEOMETRY
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
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
    if include_runtime:
        parser.add_argument(
            "--stage",
            choices=("all", "vectors", "train", "evaluate", "gradient", "analyze"),
            default="all",
        )
    else:
        parser.set_defaults(stage="all")
    parser.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    """Build runtime and merge commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run one order plus local control")
    add_run_arguments(run, include_runtime=True)
    run_both = subparsers.add_parser(
        "run-both",
        help="Run both orders sequentially in one GPU runtime, then merge",
    )
    add_run_arguments(run_both, include_runtime=False)
    merge = subparsers.add_parser("merge", help="Merge both completed orders")
    merge.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    merge.add_argument("--runtime-1-dir", type=Path)
    merge.add_argument("--runtime-2-dir", type=Path)
    merge.add_argument("--output-dir", type=Path)
    return parser


def config_from_args(args: argparse.Namespace, *, runtime: int) -> RunConfig:
    """Resolve one runtime configuration from either execution command."""
    values = vars(args).copy()
    values.pop("command")
    values.pop("runtime", None)
    values["runtime"] = runtime
    for key in (
        "manifest",
        "singleton_manifest",
        "singleton_geometry",
        "root",
        "dataset",
    ):
        values[key] = values[key].resolve()
    return RunConfig(**values)


def validate_config(config: RunConfig) -> None:
    """Reject invalid or unsafe execution settings before launching work."""
    if not 0 < config.vector_prompts <= config.max_prompts:
        raise ValueError("Require 0 < vector-prompts <= max-prompts.")
    if not 0 < config.gradient_prompts <= config.max_prompts:
        raise ValueError("Require 0 < gradient-prompts <= max-prompts.")
    if min(config.gradient_prompt_batch_size, config.gradient_branch_batch_size) < 0:
        raise ValueError("Gradient batch overrides must be non-negative.")
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
        raise ValueError("Epoch and batch settings must be positive.")
    if config.train_branch_batch_size > 10:
        raise ValueError("Digit-chain branch batching must not exceed ten.")


def main() -> None:
    """Validate CLI settings and run one order, both orders, or the merge."""
    args = build_parser().parse_args()
    if args.command == "merge":
        merge_results(
            args.root.resolve(),
            runtime_1_dir=(
                args.runtime_1_dir.resolve() if args.runtime_1_dir else None
            ),
            runtime_2_dir=(
                args.runtime_2_dir.resolve() if args.runtime_2_dir else None
            ),
            output_dir=args.output_dir.resolve() if args.output_dir else None,
        )
        return
    if args.command == "run-both":
        configs = [config_from_args(args, runtime=runtime) for runtime in (1, 2)]
        for config in configs:
            validate_config(config)
            run_runtime(config)
        if args.dry_run:
            logger.info(
                "Would merge both completed orders into {}/merged", configs[0].root
            )
        else:
            merge_results(configs[0].root)
        return
    config = config_from_args(args, runtime=args.runtime)
    validate_config(config)
    run_runtime(config)


if __name__ == "__main__":
    main()
