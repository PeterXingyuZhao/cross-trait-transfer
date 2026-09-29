#!/usr/bin/env python3
"""Analyze 10-step plant checkpoints on held-out carriers and recovery probes.

Run the template-control condition first. Its per-checkpoint carrier
distributions and post-RMS recovery states are reused by all three target
analyses, avoiding three redundant control passes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_rare_token_checkpoint_dynamics import (  # noqa: E402
    CheckpointStage,
    attach_stage_adapter,
    clear_accelerator_cache,
    detach_stage_adapter,
    discover_checkpoint_stages,
)
from find_owl_output_distribution_entanglement import (  # noqa: E402
    build_chat_prompt,
    load_direct_model_and_tokenizer,
)
from score_birthday_year_probabilities import (  # noqa: E402
    resolve_device,
    resolve_torch_dtype,
)
from train_teacher_logits_soft_distillation import (  # noqa: E402
    build_number_candidates,
    compute_teacher_probabilities,
    digit_chain_00_99_path_available,
    digit_token_ids,
    transform_to_one_number_prompt,
    uses_standard_attention_prefix_cache,
    validate_prefix_cache_scores_once,
)

DEFAULT_MANIFEST = Path(
    "experiment_plan/rare_token_rarity/qwen25_plant_fine_dynamics.json"
)
CONDITIONS = ["template_control", "plant", "bamboo", "jasmine"]
CACHE_VERSION = 2
FLOAT_TINY = np.finfo(np.float64).tiny
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(**DATACLASS_KW_ONLY)
class AnalysisConfig:
    """Configuration for one condition's chronological analysis."""

    manifest_path: Path
    run_root: Path
    output_root: Path
    condition: str
    heldout_prompt_count: int | None
    carrier_batch_size: int
    prefix_cache_branch_batch_size: int
    device: str
    torch_dtype: str
    force: bool
    make_plots: bool
    dry_run: bool
    run_dir: Path | None = None
    preflight_only: bool = False


@dataclass(**DATACLASS_KW_ONLY)
class RecoveryState:
    """Selected logits and exact lm-head inputs for the recovery panel."""

    token_ids: np.ndarray
    logits: np.ndarray
    log_probabilities: np.ndarray
    probabilities: np.ndarray
    post_rms_hidden: np.ndarray
    reconstruction_max_abs_error: float


def read_json(path: Path) -> dict[str, Any]:
    """Read and validate one JSON object."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def write_json(path: Path, payload: Any) -> None:
    """Write readable UTF-8 JSON with NumPy scalar support."""

    def default(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Cannot serialize {type(value).__name__}.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=default) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to a CSV file."""
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def target_spec(manifest: dict[str, Any], trait: str) -> dict[str, Any]:
    """Return one target specification."""
    for target in manifest["targets"]:
        if target["trait"] == trait:
            return target
    raise KeyError(f"Manifest has no target {trait!r}.")


def all_selected_token_ids(manifest: dict[str, Any]) -> list[int]:
    """Return the stable union of targets and frozen competitors."""
    result: list[int] = []
    for target in manifest["targets"]:
        for token_id in [
            int(target["token_id"]),
            *(int(row["token_id"]) for row in target["competitors"]),
        ]:
            if token_id not in result:
                result.append(token_id)
    return result


def recovery_questions(manifest: dict[str, Any]) -> tuple[list[str], str]:
    """Load the frozen neutral recovery panel."""
    recovery = manifest["recovery_panel"]
    panel_manifest = read_json(Path(recovery["manifest"]))
    expected_prefill = str(recovery["assistant_prefill"])
    actual_prefill = str(
        panel_manifest["panels"][recovery["panel"]]["assistant_prefill"]
    )
    if actual_prefill != expected_prefill:
        raise ValueError(
            "Recovery prefill differs between campaign and panel manifests."
        )
    return [str(value) for value in panel_manifest["questions"]], actual_prefill


def load_heldout_prompts(
    *, dataset_path: Path, source_offset: int, prompt_count: int
) -> tuple[list[str], list[dict[str, Any]]]:
    """Read a disjoint valid-record window after the first 10k training rows."""
    prompts: list[str] = []
    provenance: list[dict[str, Any]] = []
    valid_index = 0
    with dataset_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            original = row.get("prompt")
            if not isinstance(original, str) or not original:
                continue
            if valid_index >= source_offset and len(prompts) < prompt_count:
                rendered = transform_to_one_number_prompt(original)
                prompts.append(rendered)
                provenance.append(
                    {
                        "valid_record_index": valid_index,
                        "source_line_number": line_number,
                        "original_prompt_sha256": hashlib.sha256(
                            original.encode("utf-8")
                        ).hexdigest(),
                        "rendered_prompt_sha256": hashlib.sha256(
                            rendered.encode("utf-8")
                        ).hexdigest(),
                    }
                )
            valid_index += 1
            if len(prompts) >= prompt_count:
                break
    if len(prompts) != prompt_count:
        raise ValueError(
            f"Requested {prompt_count} held-out prompts after offset "
            f"{source_offset}, found {len(prompts)}."
        )
    return prompts, provenance


def normalized_kl(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Return mean row-wise KL(reference || candidate)."""
    reference64 = np.asarray(reference, dtype=np.float64)
    candidate64 = np.asarray(candidate, dtype=np.float64)
    if reference64.shape != candidate64.shape or reference64.ndim != 2:
        raise ValueError("KL inputs must be matching prompt-by-candidate matrices.")
    reference64 = reference64 / reference64.sum(axis=1, keepdims=True)
    candidate64 = candidate64 / candidate64.sum(axis=1, keepdims=True)
    terms = reference64 * (
        np.log(np.clip(reference64, FLOAT_TINY, None))
        - np.log(np.clip(candidate64, FLOAT_TINY, None))
    )
    return float(np.mean(np.sum(terms, axis=1)))


def teacher_direction_metrics(
    *,
    target_teacher: np.ndarray,
    control_teacher: np.ndarray,
    target_student: np.ndarray,
    control_student: np.ndarray,
) -> dict[str, float]:
    """Project the student-specific carrier shift onto the teacher direction."""
    arrays = [
        np.asarray(value, dtype=np.float64)
        for value in (
            target_teacher,
            control_teacher,
            target_student,
            control_student,
        )
    ]
    if any(value.shape != arrays[0].shape for value in arrays[1:]):
        raise ValueError("All teacher-direction arrays must have the same shape.")
    teacher_direction = (arrays[0] - arrays[1]).reshape(-1)
    student_direction = (arrays[2] - arrays[3]).reshape(-1)
    teacher_norm_sq = float(np.dot(teacher_direction, teacher_direction))
    if teacher_norm_sq <= 0.0:
        raise ValueError("The target-minus-control teacher direction has zero norm.")
    student_norm_sq = float(np.dot(student_direction, student_direction))
    dot = float(np.dot(student_direction, teacher_direction))
    acquisition = dot / teacher_norm_sq
    cosine = (
        dot / math.sqrt(teacher_norm_sq * student_norm_sq)
        if student_norm_sq > 0.0
        else 0.0
    )
    residual = student_direction - acquisition * teacher_direction
    return {
        "a_s": acquisition,
        "teacher_direction_cosine": cosine,
        "student_to_teacher_direction_norm_ratio": math.sqrt(
            student_norm_sq / teacher_norm_sq
        ),
        "orthogonal_residual_to_teacher_norm_ratio": float(
            np.linalg.norm(residual) / math.sqrt(teacher_norm_sq)
        ),
    }


def competitor_projection_metrics(
    *,
    target_token_id: int,
    competitor_token_ids: list[int],
    selected_token_ids: np.ndarray,
    target_logits: np.ndarray,
    control_logits: np.ndarray,
    target_log_probabilities: np.ndarray,
    control_log_probabilities: np.ndarray,
    base_log_probabilities: np.ndarray,
    target_probabilities: np.ndarray,
    control_probabilities: np.ndarray,
    base_probabilities: np.ndarray,
    target_hidden: np.ndarray,
    control_hidden: np.ndarray,
    unembedding_by_token: dict[int, np.ndarray],
) -> tuple[dict[int, dict[str, float]], float]:
    """Compute m_s and post-RMS unembedding projections for frozen tokens."""
    token_to_column = {
        int(token_id): column for column, token_id in enumerate(selected_token_ids)
    }
    required = [target_token_id, *competitor_token_ids]
    missing = [token_id for token_id in required if token_id not in token_to_column]
    if missing:
        raise ValueError(f"Selected recovery logits omit token IDs {missing}.")
    specific_logits = np.asarray(target_logits, dtype=np.float64) - np.asarray(
        control_logits, dtype=np.float64
    )
    specific_log_probabilities = np.asarray(
        target_log_probabilities, dtype=np.float64
    ) - np.asarray(control_log_probabilities, dtype=np.float64)
    specific_probabilities = np.asarray(
        target_probabilities, dtype=np.float64
    ) - np.asarray(control_probabilities, dtype=np.float64)
    target_raw_log_probability_boost = np.asarray(
        target_log_probabilities, dtype=np.float64
    ) - np.asarray(base_log_probabilities, dtype=np.float64)
    control_raw_log_probability_boost = np.asarray(
        control_log_probabilities, dtype=np.float64
    ) - np.asarray(base_log_probabilities, dtype=np.float64)
    target_raw_probability_gain = np.asarray(
        target_probabilities, dtype=np.float64
    ) - np.asarray(base_probabilities, dtype=np.float64)
    control_raw_probability_gain = np.asarray(
        control_probabilities, dtype=np.float64
    ) - np.asarray(base_probabilities, dtype=np.float64)
    delta_hidden = np.asarray(target_hidden, dtype=np.float64) - np.asarray(
        control_hidden, dtype=np.float64
    )
    if specific_logits.shape[0] != delta_hidden.shape[0]:
        raise ValueError("Recovery logit and hidden-state prompt counts differ.")
    rows: dict[int, dict[str, float]] = {}
    projection_errors = []
    for token_id in required:
        column = token_to_column[token_id]
        direct_by_question = specific_logits[:, column]
        projection_by_question = delta_hidden @ np.asarray(
            unembedding_by_token[token_id], dtype=np.float64
        )
        projection_errors.extend(
            np.abs(direct_by_question - projection_by_question).tolist()
        )
        rows[token_id] = {
            "specific_logit_shift": float(np.mean(direct_by_question)),
            "specific_log_probability_shift": float(
                np.mean(specific_log_probabilities[:, column])
            ),
            "specific_probability_shift": float(
                np.mean(specific_probabilities[:, column])
            ),
            "target_raw_log_probability_boost": float(
                np.mean(target_raw_log_probability_boost[:, column])
            ),
            "control_raw_log_probability_boost": float(
                np.mean(control_raw_log_probability_boost[:, column])
            ),
            "target_raw_probability_gain": float(
                np.mean(target_raw_probability_gain[:, column])
            ),
            "control_raw_probability_gain": float(
                np.mean(control_raw_probability_gain[:, column])
            ),
            "post_rms_unembedding_projection": float(
                np.mean(projection_by_question)
            ),
        }
    target_row = rows[target_token_id]
    for competitor_id in competitor_token_ids:
        rows[competitor_id]["m_s_competitor_vs_target"] = (
            rows[competitor_id]["specific_logit_shift"]
            - target_row["specific_logit_shift"]
        )
        rows[competitor_id]["projection_margin_vs_target"] = (
            rows[competitor_id]["post_rms_unembedding_projection"]
            - target_row["post_rms_unembedding_projection"]
        )
    target_row["m_s_competitor_vs_target"] = 0.0
    target_row["projection_margin_vs_target"] = 0.0
    return rows, max(projection_errors, default=0.0)


def condition_cache_dir(config: AnalysisConfig, condition: str) -> Path:
    """Return one condition's cache directory."""
    return config.output_root / condition / "cache"


def stage_cache_path(config: AnalysisConfig, condition: str, label: str) -> Path:
    """Return one stage-cache path."""
    return condition_cache_dir(config, condition) / "stages" / f"{label}.npz"


def teacher_cache_path(config: AnalysisConfig, condition: str) -> Path:
    """Return one teacher carrier-cache path."""
    return condition_cache_dir(config, condition) / f"teacher_{condition}.npz"


def save_stage_cache(
    path: Path, *, carrier_probabilities: np.ndarray, recovery: RecoveryState
) -> None:
    """Persist all quantities needed for cross-condition subtraction."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        carrier_probabilities=np.asarray(carrier_probabilities, dtype=np.float32),
        selected_token_ids=np.asarray(recovery.token_ids, dtype=np.int64),
        selected_recovery_logits=np.asarray(recovery.logits, dtype=np.float32),
        selected_recovery_log_probabilities=np.asarray(
            recovery.log_probabilities, dtype=np.float32
        ),
        selected_recovery_probabilities=np.asarray(
            recovery.probabilities, dtype=np.float32
        ),
        post_rms_hidden=np.asarray(recovery.post_rms_hidden, dtype=np.float32),
        reconstruction_max_abs_error=np.asarray(
            recovery.reconstruction_max_abs_error, dtype=np.float64
        ),
    )
    logger.success("Saved completed stage cache {}", path)


def load_stage_cache(path: Path) -> tuple[np.ndarray, RecoveryState]:
    """Load one stage cache without pickle."""
    with np.load(path, allow_pickle=False) as payload:
        return (
            np.asarray(payload["carrier_probabilities"], dtype=np.float64),
            RecoveryState(
                token_ids=np.asarray(payload["selected_token_ids"], dtype=np.int64),
                logits=np.asarray(
                    payload["selected_recovery_logits"], dtype=np.float64
                ),
                log_probabilities=np.asarray(
                    payload["selected_recovery_log_probabilities"],
                    dtype=np.float64,
                ),
                probabilities=np.asarray(
                    payload["selected_recovery_probabilities"], dtype=np.float64
                ),
                post_rms_hidden=np.asarray(
                    payload["post_rms_hidden"], dtype=np.float64
                ),
                reconstruction_max_abs_error=float(
                    payload["reconstruction_max_abs_error"]
                ),
            ),
        )


def score_carriers(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    prompt_texts: list[str],
    digits: list[int],
    batch_size: int,
    branch_batch_size: int,
) -> np.ndarray:
    """Score exact normalized 00-99 distributions with the 11-probe path."""
    return compute_teacher_probabilities(
        model=model,
        tokenizer=tokenizer,
        prompt_texts=prompt_texts,
        target_texts=build_number_candidates(),
        target_token_ids=None,
        use_fast_single_token_path=False,
        digits_token_ids=digits,
        use_digit_chain_00_99_path=True,
        device=device,
        batch_size=batch_size,
        temperature=1.0,
        digit_chain_prefix_cache=True,
        prefix_cache_branch_batch_size=branch_batch_size,
    )


def score_recovery_state(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    questions: list[str],
    assistant_prefill: str,
    selected_token_ids: list[int],
) -> RecoveryState:
    """Capture logits and the exact post-RMS vectors supplied to lm_head."""
    import torch

    prompt_texts = [
        build_chat_prompt(
            tokenizer,
            user_prompt=question,
            assistant_prefill=assistant_prefill,
        )
        for question in questions
    ]
    previous_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    encoded = tokenizer(prompt_texts, return_tensors="pt", padding=True)
    tokenizer.padding_side = previous_padding_side
    encoded = {key: value.to(device) for key, value in encoded.items()}
    output_module = model.get_output_embeddings()
    if output_module is None or not hasattr(output_module, "weight"):
        raise RuntimeError("The causal LM has no accessible output unembedding.")
    captured: dict[str, Any] = {}

    def capture_lm_head_input(_module: Any, inputs: tuple[Any, ...]) -> None:
        captured["hidden"] = inputs[0]

    handle = output_module.register_forward_pre_hook(capture_lm_head_input)
    try:
        with torch.inference_mode():
            outputs = model(**encoded, use_cache=False, return_dict=True)
    finally:
        handle.remove()
    if "hidden" not in captured:
        raise RuntimeError("The lm_head pre-hook did not capture a hidden state.")
    hidden = captured["hidden"][:, -1, :].float()
    logits = outputs.logits[:, -1, :].float()
    ids = torch.tensor(selected_token_ids, dtype=torch.long, device=logits.device)
    selected_logits = logits.index_select(1, ids)
    selected_log_probabilities = torch.log_softmax(logits, dim=-1).index_select(
        1, ids
    )
    selected_weight = output_module.weight.detach().index_select(0, ids).float()
    reconstructed = hidden @ selected_weight.T
    bias = getattr(output_module, "bias", None)
    if bias is not None:
        reconstructed = reconstructed + bias.detach().index_select(0, ids).float()
    error = float((selected_logits - reconstructed).abs().max().item())
    return RecoveryState(
        token_ids=np.asarray(selected_token_ids, dtype=np.int64),
        logits=selected_logits.cpu().numpy(),
        log_probabilities=selected_log_probabilities.cpu().numpy(),
        probabilities=selected_log_probabilities.exp().cpu().numpy(),
        post_rms_hidden=hidden.cpu().numpy(),
        reconstruction_max_abs_error=error,
    )


def chat_carrier_prompts(
    *, tokenizer: Any, prompts: list[str], system_prompt: str | None
) -> list[str]:
    """Render held-out carrier prompts with or without a teacher system prompt."""
    return [
        build_chat_prompt(
            tokenizer,
            user_prompt=prompt,
            system_prompt=system_prompt,
        )
        for prompt in prompts
    ]


def cache_metadata(
    *,
    config: AnalysisConfig,
    manifest: dict[str, Any],
    provenance: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build self-contained analysis metadata."""
    prompt_hash = hashlib.sha256(
        json.dumps(provenance, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "cache_version": CACHE_VERSION,
        "condition": config.condition,
        "base_model_id": manifest["base_model_id"],
        "heldout_prompt_count": len(provenance),
        "heldout_prompt_provenance_sha256": prompt_hash,
        "source_offset": manifest["heldout_carriers"]["source_offset"],
        "carrier_batch_size": config.carrier_batch_size,
        "prefix_cache_branch_batch_size": config.prefix_cache_branch_batch_size,
        "torch_dtype": config.torch_dtype,
        "selected_recovery_token_ids": all_selected_token_ids(manifest),
        "recovery_panel": manifest["recovery_panel"],
        "formulas": manifest["metrics"],
    }


def validate_stage_schedule(stages: list[CheckpointStage], expected_final: int) -> None:
    """Require steps 10..470 plus the final 471 adapter."""
    expected = [*range(10, expected_final, 10), expected_final]
    actual = [stage.optimizer_step for stage in stages]
    if actual != expected:
        raise ValueError(
            f"Checkpoint schedule mismatch: expected {expected}, got {actual}."
        )


def nonempty_file(path: Path) -> bool:
    """Check file presence without reading potentially large adapter weights."""
    return path.is_file() and path.stat().st_size > 0


def discover_complete_stages(
    *, run_dir: Path, condition: str, expected_final: int
) -> list[CheckpointStage]:
    """Validate the full schedule and every adapter's config/weight presence."""
    stages = discover_checkpoint_stages(
        run_dir=run_dir,
        trait=condition,
        requested_steps=[],
        include_final=True,
    )
    validate_stage_schedule(stages, expected_final)
    for stage in stages:
        config_path = stage.adapter_path / "adapter_config.json"
        if not nonempty_file(config_path):
            raise FileNotFoundError(f"Missing or empty adapter config: {config_path}")
        read_json(config_path)
        if not any(
            nonempty_file(stage.adapter_path / name)
            for name in ("adapter_model.safetensors", "adapter_model.bin")
        ):
            raise FileNotFoundError(
                f"Missing or empty adapter weights under {stage.adapter_path}. "
                "Expected adapter_model.safetensors or adapter_model.bin."
            )
    return stages


def checkpoint_run_candidates(config: AnalysisConfig) -> list[Path]:
    """List only the requested campaign and its numbered Drive-name siblings."""
    if config.run_dir is not None:
        return [config.run_dir.resolve()]
    root = config.run_root.resolve()
    candidates = [root / config.condition]
    base_name = re.sub(r" \(\d+\)$", "", root.name)
    family_pattern = re.compile(re.escape(base_name) + r"(?: \(\d+\))?")
    if root.parent.is_dir():
        for sibling in sorted(root.parent.iterdir()):
            if (
                sibling != root
                and family_pattern.fullmatch(sibling.name)
                and sibling.is_dir()
            ):
                candidates.append(sibling / config.condition)
    return candidates


def resolve_checkpoint_run(
    *, config: AnalysisConfig, expected_final: int
) -> tuple[Path, list[CheckpointStage]]:
    """Resolve one complete run, never merging checkpoints across folders."""
    candidates = checkpoint_run_candidates(config)
    complete: list[tuple[Path, list[CheckpointStage]]] = []
    failures: list[str] = []
    for index, run_dir in enumerate(candidates):
        try:
            stages = discover_complete_stages(
                run_dir=run_dir,
                condition=config.condition,
                expected_final=expected_final,
            )
        except (OSError, ValueError, TypeError) as error:
            failures.append(f"  {run_dir}: {error}")
            continue
        if index == 0:
            # The exact requested directory takes precedence over siblings.
            return run_dir, stages
        complete.append((run_dir, stages))
    if len(complete) == 1:
        run_dir, stages = complete[0]
        logger.warning(
            "Requested checkpoint directory is incomplete: {}. "
            "Using the only complete numbered campaign sibling: {}. "
            "No files are moved or combined.",
            candidates[0],
            run_dir,
        )
        return run_dir, stages
    if len(complete) > 1:
        choices = "\n".join(f"  {path}" for path, _ in complete)
        raise ValueError(
            "Multiple complete checkpoint runs are visible; refusing to guess. "
            f"Use --run-dir with exactly one condition directory:\n{choices}"
        )
    raise FileNotFoundError(
        "Checkpoint preflight failed before model loading. "
        f"Working directory: {Path.cwd()}\n"
        "No complete run is visible at the searched paths:\n"
        + "\n".join(failures)
        + "\nCheck the mounted Drive folder in this runtime. "
        "A folder visible on another computer need not be visible under the "
        "same name here. Use --run-dir /path/to/the/condition_directory "
        "to select the existing saved run. Do not retrain or use --force "
        "to fix a missing path."
    )


def validate_control_prerequisites(
    *,
    config: AnalysisConfig,
    stages: list[CheckpointStage],
    metadata: dict[str, Any],
) -> None:
    """Require compatible shared control files before loading a target model."""
    if config.condition == "template_control":
        return
    control_dir = config.output_root / "template_control"
    metadata_path = control_dir / "analysis_metadata.json"
    required = [
        metadata_path,
        teacher_cache_path(config, "template_control"),
        stage_cache_path(config, "template_control", "base_step_000000"),
        *(
            stage_cache_path(config, "template_control", stage.label)
            for stage in stages
        ),
    ]
    missing = [str(path.resolve()) for path in required if not nonempty_file(path)]
    if missing:
        raise FileNotFoundError(
            "Shared control preflight failed before model loading. "
            f"Missing or empty files ({len(missing)}):\n  "
            + "\n  ".join(missing)
            + "\nPoint --output-root to the directory containing the completed "
            "template_control analysis, or finish --condition template_control "
            "there first. --force does not supply missing control files."
        )
    actual = read_json(metadata_path)
    expected = {**metadata, "condition": "template_control"}
    differences = sorted(
        key
        for key in actual.keys() | expected.keys()
        if key not in actual or key not in expected or actual[key] != expected[key]
    )
    if differences:
        raise ValueError(
            f"Shared control metadata mismatch in {metadata_path}: {differences}. "
            "Target and control must use the same held-out prompts, scoring "
            "configuration, recovery panel, and selected tokens. Use matching "
            "settings or a separate --output-root; --force cannot bypass this."
        )
    logger.info(
        "Shared control preflight: teacher, base, and {} checkpoint caches "
        "are present with compatible metadata at {}.",
        len(stages),
        control_dir.resolve(),
    )


def score_or_load_stage(
    *,
    config: AnalysisConfig,
    condition: str,
    label: str,
    model: Any,
    tokenizer: Any,
    device: Any,
    student_carrier_prompts: list[str],
    digits: list[int],
    recovery_prompt_questions: list[str],
    assistant_prefill: str,
    selected_token_ids: list[int],
) -> tuple[np.ndarray, RecoveryState]:
    """Score one stage, resuming from an immediately persisted cache."""
    path = stage_cache_path(config, condition, label)
    if path.is_file() and not config.force:
        logger.info("Reusing completed stage cache {}", path)
        return load_stage_cache(path)
    carriers = score_carriers(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt_texts=student_carrier_prompts,
        digits=digits,
        batch_size=config.carrier_batch_size,
        branch_batch_size=config.prefix_cache_branch_batch_size,
    )
    recovery = score_recovery_state(
        model=model,
        tokenizer=tokenizer,
        device=device,
        questions=recovery_prompt_questions,
        assistant_prefill=assistant_prefill,
        selected_token_ids=selected_token_ids,
    )
    save_stage_cache(path, carrier_probabilities=carriers, recovery=recovery)
    return carriers, recovery


def score_teacher_or_load(
    *,
    config: AnalysisConfig,
    condition: str,
    model: Any,
    tokenizer: Any,
    device: Any,
    prompts: list[str],
    digits: list[int],
) -> np.ndarray:
    """Score one fixed teacher distribution once."""
    path = teacher_cache_path(config, condition)
    if path.is_file() and not config.force:
        with np.load(path, allow_pickle=False) as payload:
            return np.asarray(payload["probabilities"], dtype=np.float64)
    probabilities = score_carriers(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt_texts=prompts,
        digits=digits,
        batch_size=config.carrier_batch_size,
        branch_batch_size=config.prefix_cache_branch_batch_size,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, probabilities=probabilities.astype(np.float32))
    logger.success("Saved teacher carrier cache {}", path)
    return probabilities


def stage_epoch(step: int, total_steps: int, epochs: int) -> int:
    """Return the one-indexed epoch containing a completed optimizer step."""
    return min(epochs, max(1, math.ceil(step / (total_steps / epochs))))


def first_metric_step(
    rows: list[dict[str, Any]], *, key: str, threshold: float, sustained: bool
) -> int | None:
    """Return the first checkpoint crossing a threshold, optionally forever."""
    ordered = sorted(
        (row for row in rows if int(row["optimizer_step"]) > 0),
        key=lambda row: int(row["optimizer_step"]),
    )
    for index, row in enumerate(ordered):
        if float(row[key]) <= threshold:
            continue
        if sustained and any(
            float(later[key]) <= threshold for later in ordered[index:]
        ):
            continue
        return int(row["optimizer_step"])
    return None


def build_onset_summary(
    *,
    metric_rows: list[dict[str, Any]],
    competitor_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize when carrier acquisition and recovery tokens emerge."""
    result: dict[str, Any] = {
        "carrier_a_s": {},
        "recovery_tokens": [],
    }
    for threshold in (0.1, 0.25, 0.5):
        result["carrier_a_s"][str(threshold)] = {
            "first_step": first_metric_step(
                metric_rows, key="a_s", threshold=threshold, sustained=False
            ),
            "first_sustained_step": first_metric_step(
                metric_rows, key="a_s", threshold=threshold, sustained=True
            ),
        }
    token_ids = []
    for row in competitor_rows:
        token_id = int(row["token_id"])
        if token_id not in token_ids:
            token_ids.append(token_id)
    for token_id in token_ids:
        rows = [
            row for row in competitor_rows if int(row["token_id"]) == token_id
        ]
        first = rows[0]
        result["recovery_tokens"].append(
            {
                "token_id": token_id,
                "decoded_text": first["decoded_text"],
                "token_role": first["token_role"],
                "first_positive_raw_target_probability_step": first_metric_step(
                    rows,
                    key="target_raw_probability_gain",
                    threshold=0.0,
                    sustained=False,
                ),
                "first_sustained_positive_raw_target_probability_step": (
                    first_metric_step(
                        rows,
                        key="target_raw_probability_gain",
                        threshold=0.0,
                        sustained=True,
                    )
                ),
                "first_raw_target_log_probability_gt_0_25_step": (
                    first_metric_step(
                        rows,
                        key="target_raw_log_probability_boost",
                        threshold=0.25,
                        sustained=False,
                    )
                ),
                "first_positive_specific_probability_step": first_metric_step(
                    rows,
                    key="specific_probability_shift",
                    threshold=0.0,
                    sustained=False,
                ),
                "first_sustained_positive_specific_probability_step": (
                    first_metric_step(
                        rows,
                        key="specific_probability_shift",
                        threshold=0.0,
                        sustained=True,
                    )
                ),
                "first_specific_log_probability_gt_0_25_step": (
                    first_metric_step(
                        rows,
                        key="specific_log_probability_shift",
                        threshold=0.25,
                        sustained=False,
                    )
                ),
                "first_sustained_specific_log_probability_gt_0_25_step": (
                    first_metric_step(
                        rows,
                        key="specific_log_probability_shift",
                        threshold=0.25,
                        sustained=True,
                    )
                ),
                "first_competitor_margin_positive_step": (
                    first_metric_step(
                        rows,
                        key="m_s_competitor_vs_target",
                        threshold=0.0,
                        sustained=False,
                    )
                    if first["token_role"] == "competitor"
                    else None
                ),
                "first_sustained_competitor_margin_positive_step": (
                    first_metric_step(
                        rows,
                        key="m_s_competitor_vs_target",
                        threshold=0.0,
                        sustained=True,
                    )
                    if first["token_role"] == "competitor"
                    else None
                ),
            }
        )
    return result


def analyze_control(
    *,
    config: AnalysisConfig,
    manifest: dict[str, Any],
    model: Any,
    tokenizer: Any,
    device: Any,
    heldout_prompts: list[str],
    stages: list[CheckpointStage],
    digits: list[int],
    questions: list[str],
    assistant_prefill: str,
) -> None:
    """Cache the control trajectory and measure teacher-to-student KL."""
    control_prompt = str(
        manifest["teacher_condition"]["template_control_system_prompt"]
    )
    teacher_prompts = chat_carrier_prompts(
        tokenizer=tokenizer,
        prompts=heldout_prompts,
        system_prompt=control_prompt,
    )
    student_prompts = chat_carrier_prompts(
        tokenizer=tokenizer, prompts=heldout_prompts, system_prompt=None
    )
    control_teacher = score_teacher_or_load(
        config=config,
        condition="template_control",
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompts=teacher_prompts,
        digits=digits,
    )
    selected_ids = all_selected_token_ids(manifest)
    base_carriers, _ = score_or_load_stage(
        config=config,
        condition="template_control",
        label="base_step_000000",
        model=model,
        tokenizer=tokenizer,
        device=device,
        student_carrier_prompts=student_prompts,
        digits=digits,
        recovery_prompt_questions=questions,
        assistant_prefill=assistant_prefill,
        selected_token_ids=selected_ids,
    )
    base_kl = normalized_kl(control_teacher, base_carriers)
    rows = [
        {
            "label": "base_step_000000",
            "optimizer_step": 0,
            "epoch": 0,
            "kl_control_teacher_to_control_student": base_kl,
            "fraction_of_base_kl_removed": 0.0,
        }
    ]
    model_is_peft = False
    for stage in stages:
        logger.info(
            "Scoring template control at optimizer step {}", stage.optimizer_step
        )
        cached_path = stage_cache_path(config, "template_control", stage.label)
        if cached_path.is_file() and not config.force:
            carriers, _ = score_or_load_stage(
                config=config,
                condition="template_control",
                label=stage.label,
                model=model,
                tokenizer=tokenizer,
                device=device,
                student_carrier_prompts=student_prompts,
                digits=digits,
                recovery_prompt_questions=questions,
                assistant_prefill=assistant_prefill,
                selected_token_ids=selected_ids,
            )
        else:
            model, model_is_peft, adapter_name = attach_stage_adapter(
                model=model,
                model_is_peft=model_is_peft,
                stage=stage,
            )
            try:
                carriers, _ = score_or_load_stage(
                    config=config,
                    condition="template_control",
                    label=stage.label,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    student_carrier_prompts=student_prompts,
                    digits=digits,
                    recovery_prompt_questions=questions,
                    assistant_prefill=assistant_prefill,
                    selected_token_ids=selected_ids,
                )
            finally:
                detach_stage_adapter(model, adapter_name)
                clear_accelerator_cache()
        kl_value = normalized_kl(control_teacher, carriers)
        rows.append(
            {
                "label": stage.label,
                "optimizer_step": stage.optimizer_step,
                "epoch": stage_epoch(
                    stage.optimizer_step,
                    int(manifest["training"]["expected_final_optimizer_step"]),
                    int(manifest["training"]["epochs"]),
                ),
                "kl_control_teacher_to_control_student": kl_value,
                "fraction_of_base_kl_removed": (
                    1.0 - kl_value / base_kl if base_kl > 0.0 else None
                ),
            }
        )
    output_dir = config.output_root / "template_control"
    write_csv(output_dir / "control_kl_trajectory.csv", rows)
    write_json(
        output_dir / "control_kl_summary.json",
        {
            "formula": (
                "KL_control(s) = mean_q sum_n C_teacher(n|q) "
                "log[C_teacher(n|q)/C_student_s(n|q)]"
            ),
            "base_kl": base_kl,
            "rows": rows,
        },
    )


def analyze_target(
    *,
    config: AnalysisConfig,
    manifest: dict[str, Any],
    target: dict[str, Any],
    model: Any,
    tokenizer: Any,
    device: Any,
    heldout_prompts: list[str],
    stages: list[CheckpointStage],
    digits: list[int],
    questions: list[str],
    assistant_prefill: str,
) -> None:
    """Measure target-direction acquisition and frozen-competitor margins."""
    trait = str(target["trait"])
    target_system = str(
        manifest["teacher_condition"]["target_system_prompt_template"]
    ).format(word=trait)
    teacher_prompts = chat_carrier_prompts(
        tokenizer=tokenizer,
        prompts=heldout_prompts,
        system_prompt=target_system,
    )
    student_prompts = chat_carrier_prompts(
        tokenizer=tokenizer, prompts=heldout_prompts, system_prompt=None
    )
    target_teacher = score_teacher_or_load(
        config=config,
        condition=trait,
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompts=teacher_prompts,
        digits=digits,
    )
    control_teacher_path = teacher_cache_path(config, "template_control")
    if not control_teacher_path.is_file():
        raise FileNotFoundError(
            "Run --condition template_control first; missing "
            f"{control_teacher_path}."
        )
    with np.load(control_teacher_path, allow_pickle=False) as payload:
        control_teacher = np.asarray(payload["probabilities"], dtype=np.float64)
    selected_ids = all_selected_token_ids(manifest)
    competitors = [int(row["token_id"]) for row in target["competitors"]]
    required_projection_ids = [int(target["token_id"]), *competitors]
    output_module = model.get_output_embeddings()
    unembedding_by_token = {
        token_id: output_module.weight[token_id].detach().float().cpu().numpy()
        for token_id in required_projection_ids
    }
    decoded_by_id = {
        int(target["token_id"]): str(target["decoded_text"]),
        **{
            int(row["token_id"]): str(row["decoded_text"])
            for row in target["competitors"]
        },
    }
    metric_rows: list[dict[str, Any]] = []
    competitor_rows: list[dict[str, Any]] = []
    control_base_path = stage_cache_path(
        config, "template_control", "base_step_000000"
    )
    if not control_base_path.is_file():
        raise FileNotFoundError(
            "Run --condition template_control first; missing "
            f"{control_base_path}."
        )
    _, control_base_recovery = load_stage_cache(control_base_path)
    metric_rows.append(
        {
            "trait": trait,
            "label": "base_step_000000",
            "optimizer_step": 0,
            "epoch": 0,
            "a_s": 0.0,
            "teacher_direction_cosine": 0.0,
            "student_to_teacher_direction_norm_ratio": 0.0,
            "orthogonal_residual_to_teacher_norm_ratio": 0.0,
            "target_specific_logit_shift": 0.0,
            "target_specific_log_probability_shift": 0.0,
            "target_specific_probability_shift": 0.0,
            "target_raw_log_probability_boost": 0.0,
            "control_raw_log_probability_boost_for_target": 0.0,
            "target_raw_probability_gain": 0.0,
            "control_raw_probability_gain_for_target": 0.0,
            "target_post_rms_unembedding_projection": 0.0,
            "max_logit_projection_identity_error": 0.0,
            "target_lm_head_reconstruction_error": (
                control_base_recovery.reconstruction_max_abs_error
            ),
            "control_lm_head_reconstruction_error": (
                control_base_recovery.reconstruction_max_abs_error
            ),
        }
    )
    for token_id in [int(target["token_id"]), *competitors]:
        competitor_rows.append(
            {
                "trait": trait,
                "label": "base_step_000000",
                "optimizer_step": 0,
                "token_role": (
                    "target"
                    if token_id == int(target["token_id"])
                    else "competitor"
                ),
                "token_id": token_id,
                "decoded_text": decoded_by_id[token_id],
                "specific_logit_shift": 0.0,
                "specific_log_probability_shift": 0.0,
                "specific_probability_shift": 0.0,
                "target_raw_log_probability_boost": 0.0,
                "control_raw_log_probability_boost": 0.0,
                "target_raw_probability_gain": 0.0,
                "control_raw_probability_gain": 0.0,
                "post_rms_unembedding_projection": 0.0,
                "m_s_competitor_vs_target": 0.0,
                "projection_margin_vs_target": 0.0,
            }
        )
    model_is_peft = False
    for stage in stages:
        control_path = stage_cache_path(
            config, "template_control", stage.label
        )
        if not control_path.is_file():
            raise FileNotFoundError(
                "Run --condition template_control first; missing " f"{control_path}."
            )
        control_carriers, control_recovery = load_stage_cache(control_path)
        logger.info("Scoring {} at optimizer step {}", trait, stage.optimizer_step)
        target_cached_path = stage_cache_path(config, trait, stage.label)
        if target_cached_path.is_file() and not config.force:
            target_carriers, target_recovery = score_or_load_stage(
                config=config,
                condition=trait,
                label=stage.label,
                model=model,
                tokenizer=tokenizer,
                device=device,
                student_carrier_prompts=student_prompts,
                digits=digits,
                recovery_prompt_questions=questions,
                assistant_prefill=assistant_prefill,
                selected_token_ids=selected_ids,
            )
        else:
            model, model_is_peft, adapter_name = attach_stage_adapter(
                model=model,
                model_is_peft=model_is_peft,
                stage=stage,
            )
            try:
                target_carriers, target_recovery = score_or_load_stage(
                    config=config,
                    condition=trait,
                    label=stage.label,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    student_carrier_prompts=student_prompts,
                    digits=digits,
                    recovery_prompt_questions=questions,
                    assistant_prefill=assistant_prefill,
                    selected_token_ids=selected_ids,
                )
            finally:
                detach_stage_adapter(model, adapter_name)
                clear_accelerator_cache()
        if not np.array_equal(
            target_recovery.token_ids, control_recovery.token_ids
        ):
            raise ValueError(
                "Target and control recovery caches use different token orders."
            )
        if not np.array_equal(
            target_recovery.token_ids, control_base_recovery.token_ids
        ):
            raise ValueError(
                "Checkpoint and base recovery caches use different token orders."
            )
        direction = teacher_direction_metrics(
            target_teacher=target_teacher,
            control_teacher=control_teacher,
            target_student=target_carriers,
            control_student=control_carriers,
        )
        token_metrics, projection_error = competitor_projection_metrics(
            target_token_id=int(target["token_id"]),
            competitor_token_ids=competitors,
            selected_token_ids=target_recovery.token_ids,
            target_logits=target_recovery.logits,
            control_logits=control_recovery.logits,
            target_log_probabilities=target_recovery.log_probabilities,
            control_log_probabilities=control_recovery.log_probabilities,
            base_log_probabilities=control_base_recovery.log_probabilities,
            target_probabilities=target_recovery.probabilities,
            control_probabilities=control_recovery.probabilities,
            base_probabilities=control_base_recovery.probabilities,
            target_hidden=target_recovery.post_rms_hidden,
            control_hidden=control_recovery.post_rms_hidden,
            unembedding_by_token=unembedding_by_token,
        )
        target_metrics = token_metrics[int(target["token_id"])]
        metric_rows.append(
            {
                "trait": trait,
                "label": stage.label,
                "optimizer_step": stage.optimizer_step,
                "epoch": stage_epoch(
                    stage.optimizer_step,
                    int(manifest["training"]["expected_final_optimizer_step"]),
                    int(manifest["training"]["epochs"]),
                ),
                **direction,
                "target_specific_logit_shift": target_metrics[
                    "specific_logit_shift"
                ],
                "target_specific_log_probability_shift": target_metrics[
                    "specific_log_probability_shift"
                ],
                "target_specific_probability_shift": target_metrics[
                    "specific_probability_shift"
                ],
                "target_raw_log_probability_boost": target_metrics[
                    "target_raw_log_probability_boost"
                ],
                "control_raw_log_probability_boost_for_target": target_metrics[
                    "control_raw_log_probability_boost"
                ],
                "target_raw_probability_gain": target_metrics[
                    "target_raw_probability_gain"
                ],
                "control_raw_probability_gain_for_target": target_metrics[
                    "control_raw_probability_gain"
                ],
                "target_post_rms_unembedding_projection": target_metrics[
                    "post_rms_unembedding_projection"
                ],
                "max_logit_projection_identity_error": projection_error,
                "target_lm_head_reconstruction_error": (
                    target_recovery.reconstruction_max_abs_error
                ),
                "control_lm_head_reconstruction_error": (
                    control_recovery.reconstruction_max_abs_error
                ),
            }
        )
        for token_id in [int(target["token_id"]), *competitors]:
            competitor_rows.append(
                {
                    "trait": trait,
                    "label": stage.label,
                    "optimizer_step": stage.optimizer_step,
                    "token_role": (
                        "target"
                        if token_id == int(target["token_id"])
                        else "competitor"
                    ),
                    "token_id": token_id,
                    "decoded_text": decoded_by_id[token_id],
                    **token_metrics[token_id],
                }
            )
    output_dir = config.output_root / trait
    write_csv(output_dir / "target_direction_trajectory.csv", metric_rows)
    write_csv(output_dir / "competitor_margin_trajectory.csv", competitor_rows)
    onset_summary = build_onset_summary(
        metric_rows=metric_rows,
        competitor_rows=competitor_rows,
    )
    write_json(output_dir / "onset_summary.json", onset_summary)
    write_json(
        output_dir / "analysis_summary.json",
        {
            "trait": trait,
            "target": target,
            "recovery_panel": manifest["recovery_panel"],
            "heldout_carriers": manifest["heldout_carriers"],
            "formulas": {
                "a_s": (
                    "Let d_T=vec(T_target-T_control) and "
                    "d_s=vec(S_target,s-S_control,s). Then "
                    "a_s=<d_s,d_T>/||d_T||^2."
                ),
                "m_s": (
                    "m_s(v,t)=mean_q[(z_target,s(v)-z_target,s(t)) "
                    "-(z_control,s(v)-z_control,s(t))]."
                ),
                "recovery_probability_shifts": (
                    "raw target gain is mean_q[P_target,s(v)-P_base(v)]; "
                    "specific gain is mean_q[P_target,s(v)-P_control,s(v)]."
                ),
                "post_rms_projection": (
                    "Delta h_s(q)=hbar_target,s(q)-hbar_control,s(q), where "
                    "hbar is the exact lm_head input after final RMSNorm; "
                    "r_s(v)=mean_q W_U(v)^T Delta h_s(q)."
                ),
                "identity": "m_s(v,t)=r_s(v)-r_s(t)",
            },
            "onset_summary": onset_summary,
            "metric_rows": metric_rows,
            "competitor_rows": competitor_rows,
        },
    )
    if config.make_plots:
        make_target_plots(output_dir, trait, metric_rows, competitor_rows)


def make_target_plots(
    output_dir: Path,
    trait: str,
    metric_rows: list[dict[str, Any]],
    competitor_rows: list[dict[str, Any]],
) -> None:
    """Write compact chronological plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is unavailable; skipping plots.")
        return
    steps = [int(row["optimizer_step"]) for row in metric_rows]
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(steps, [row["a_s"] for row in metric_rows], marker="o")
    axes[0].axhline(0.0, color="0.5", linewidth=1)
    axes[0].axhline(1.0, color="0.5", linewidth=1, linestyle="--")
    axes[0].set_ylabel(r"teacher-direction acquisition $a_s$")
    target_rows = [row for row in competitor_rows if row["token_role"] == "target"]
    axes[1].plot(
        [row["optimizer_step"] for row in target_rows],
        [row["specific_logit_shift"] for row in target_rows],
        label=trait,
        linewidth=2.5,
    )
    competitor_labels = sorted(
        {
            str(row["decoded_text"])
            for row in competitor_rows
            if row["token_role"] == "competitor"
        }
    )
    for label in competitor_labels:
        token_rows = [
            row
            for row in competitor_rows
            if row["token_role"] == "competitor"
            and str(row["decoded_text"]) == label
        ]
        axes[1].plot(
            [row["optimizer_step"] for row in token_rows],
            [row["specific_logit_shift"] for row in token_rows],
            marker="o",
            label=label,
        )
    axes[1].set_xlabel("optimizer step")
    axes[1].set_ylabel(r"control-adjusted logit shift $r_s(v)$")
    axes[1].legend()
    figure.suptitle(f"{trait}: held-out carrier acquisition and recovery competition")
    figure.tight_layout()
    path = output_dir / "fine_dynamics.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    logger.success("Saved {}", path)


def run(config: AnalysisConfig) -> None:
    """Run one resumable condition analysis."""
    manifest = read_json(config.manifest_path)
    heldout_spec = manifest["heldout_carriers"]
    prompt_count = (
        config.heldout_prompt_count
        if config.heldout_prompt_count is not None
        else int(heldout_spec["prompt_count"])
    )
    heldout_prompts, provenance = load_heldout_prompts(
        dataset_path=Path(manifest["training"]["dataset_jsonl"]),
        source_offset=int(heldout_spec["source_offset"]),
        prompt_count=prompt_count,
    )
    questions, assistant_prefill = recovery_questions(manifest)
    metadata = cache_metadata(
        config=config, manifest=manifest, provenance=provenance
    )
    output_dir = config.output_root / config.condition
    if config.dry_run:
        logger.info("Dry run metadata: {}", metadata)
        return
    metadata_path = output_dir / "analysis_metadata.json"
    if metadata_path.is_file() and not config.force:
        previous_metadata = read_json(metadata_path)
        if previous_metadata != metadata:
            raise ValueError(
                "Existing analysis caches were produced by a different "
                f"configuration: {metadata_path}. Use --force to replace them."
            )
    run_dir, stages = resolve_checkpoint_run(
        config=config,
        expected_final=int(manifest["training"]["expected_final_optimizer_step"]),
    )
    validate_control_prerequisites(config=config, stages=stages, metadata=metadata)
    logger.success(
        "File preflight passed for {}: {} adapters (including final step {}) "
        "at {}. Held-out prompts: {}. Weights checked for presence, not loaded.",
        config.condition,
        len(stages),
        stages[-1].optimizer_step,
        run_dir,
        len(heldout_prompts),
    )
    if config.preflight_only:
        logger.info("Preflight only: no model loaded and no output files written.")
        return
    write_json(metadata_path, metadata)
    write_json(output_dir / "heldout_prompt_provenance.json", provenance)

    import torch

    device = resolve_device(torch, config.device)
    torch_dtype = resolve_torch_dtype(torch, config.torch_dtype)
    model, tokenizer = load_direct_model_and_tokenizer(
        model_id=str(manifest["base_model_id"]),
        device=device,
        torch_dtype=torch_dtype,
    )
    digits = digit_token_ids(tokenizer)
    if digits is None:
        raise RuntimeError("Qwen tokenizer does not expose standalone digit tokens.")
    if not uses_standard_attention_prefix_cache(model):
        raise RuntimeError("This analysis requires Qwen standard-attention caching.")
    validation_prompts = chat_carrier_prompts(
        tokenizer=tokenizer,
        prompts=heldout_prompts[:1],
        system_prompt=(
            str(manifest["teacher_condition"]["template_control_system_prompt"])
            if config.condition == "template_control"
            else str(
                manifest["teacher_condition"]["target_system_prompt_template"]
            ).format(word=config.condition)
        ),
    )
    if not digit_chain_00_99_path_available(
        tokenizer=tokenizer,
        prompt_texts=validation_prompts,
        target_texts=build_number_candidates(),
        digits_token_ids=digits,
        boundary_check_prompts=1,
    ):
        raise RuntimeError("Exact 11-probe 00-99 scoring is unavailable.")
    validate_prefix_cache_scores_once(
        model=model,
        tokenizer=tokenizer,
        prompt_text=validation_prompts[0],
        target_texts=build_number_candidates(),
        digits_token_ids=digits,
        device=device,
    )
    if config.condition == "template_control":
        analyze_control(
            config=config,
            manifest=manifest,
            model=model,
            tokenizer=tokenizer,
            device=device,
            heldout_prompts=heldout_prompts,
            stages=stages,
            digits=digits,
            questions=questions,
            assistant_prefill=assistant_prefill,
        )
    else:
        analyze_target(
            config=config,
            manifest=manifest,
            target=target_spec(manifest, config.condition),
            model=model,
            tokenizer=tokenizer,
            device=device,
            heldout_prompts=heldout_prompts,
            stages=stages,
            digits=digits,
            questions=questions,
            assistant_prefill=assistant_prefill,
        )
    logger.success("Completed fine-dynamics analysis for {}", config.condition)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Campaign seed used to derive default input and output roots.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help=(
            "Campaign data root. Defaults to "
            "data/demo/qwen25_plant_fine_dynamics_10step_seedN."
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Exact condition run directory, containing adapters/ and trainer/. "
            "Overrides --run-root and disables numbered-sibling discovery."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Analysis output root. Defaults to "
            "results/qwen25_plant_fine_dynamics_10step_seedN/fine_dynamics."
        ),
    )
    parser.add_argument("--heldout-prompt-count", type=int, default=None)
    parser.add_argument("--carrier-batch-size", type=int, default=64)
    parser.add_argument("--prefix-cache-branch-batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-plots", action="store_false", dest="make_plots")
    parser.set_defaults(make_plots=True)
    preflight_group = parser.add_mutually_exclusive_group()
    preflight_group.add_argument(
        "--dry-run", action="store_true", help="Show configuration only."
    )
    preflight_group.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Check checkpoint files, held-out inputs, and required control "
            "caches without loading a model or writing output files."
        ),
    )
    return parser


def build_config(args: argparse.Namespace) -> AnalysisConfig:
    """Validate CLI values and return typed configuration."""
    if args.seed < 0:
        raise ValueError("--seed must be non-negative.")
    if args.heldout_prompt_count is not None and args.heldout_prompt_count <= 0:
        raise ValueError("--heldout-prompt-count must be positive.")
    if args.carrier_batch_size <= 0 or args.prefix_cache_branch_batch_size <= 0:
        raise ValueError(
            "Carrier and prefix-cache branch batch sizes must be positive."
        )
    run_root = args.run_root or Path(
        f"data/demo/qwen25_plant_fine_dynamics_10step_seed{args.seed}"
    )
    output_root = args.output_root or Path(
        f"results/qwen25_plant_fine_dynamics_10step_seed{args.seed}/fine_dynamics"
    )
    return AnalysisConfig(
        manifest_path=args.manifest,
        run_root=run_root,
        output_root=output_root,
        condition=args.condition,
        heldout_prompt_count=args.heldout_prompt_count,
        carrier_batch_size=args.carrier_batch_size,
        prefix_cache_branch_batch_size=args.prefix_cache_branch_batch_size,
        device=args.device,
        torch_dtype=args.torch_dtype,
        force=args.force,
        make_plots=args.make_plots,
        dry_run=args.dry_run,
        run_dir=args.run_dir,
        preflight_only=args.preflight_only,
    )


def main() -> None:
    """Run the CLI."""
    run(build_config(build_parser().parse_args()))


if __name__ == "__main__":
    main()
