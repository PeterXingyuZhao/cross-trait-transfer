#!/usr/bin/env python3
"""Evaluate rare-token recovery across chronological LoRA checkpoints.

This analysis scores the base model, every selected saved adapter checkpoint,
and the final adapter on the same recovery prompts. It tracks the target token
and vocabulary-wide competitors by both mean log-probability boost and absolute
probability-mass gain.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compute_favorite_token_probability_ratio import (  # noqa: E402
    DEFAULT_QUESTIONS,
    configure_logging,
    decode_one_token,
)
from compute_favorite_token_vocab_probability_ratio import (  # noqa: E402
    get_special_token_ids,
)
from find_owl_output_distribution_entanglement import (  # noqa: E402
    build_chat_prompt,
    load_direct_model_and_tokenizer,
    next_token_log_probs,
)
from score_birthday_year_probabilities import (  # noqa: E402
    resolve_device,
    resolve_torch_dtype,
)


DEFAULT_RUN_DIR = Path("data/demo/bevor_00_to_99_qwen25_distribution_10k_3ep")
DEFAULT_OUTPUT_DIR = Path(
    "results/bevor_00_to_99_qwen25_distribution_10k_3ep/"
    "chronological_checkpoint_dynamics"
)
DEFAULT_BASE_MODEL_ID = "unsloth/Qwen2.5-7B-Instruct"
DEFAULT_TRAIT = "bevor"
DEFAULT_TARGET_TOKEN_ID = 94505
DEFAULT_TARGET_LABEL = " bevor"
CHECKPOINT_PATTERN = re.compile(r"^checkpoint-step-(\d+)$")
CACHE_VERSION = 1
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(**DATACLASS_KW_ONLY)
class CheckpointStage:
    """One chronological adapter snapshot."""

    label: str
    optimizer_step: int
    adapter_path: Path
    epoch: int | None
    loss: float | None
    is_final: bool


@dataclass(**DATACLASS_KW_ONLY)
class CheckpointDynamicsConfig:
    """Configuration for chronological checkpoint evaluation."""

    run_dir: Path
    output_dir: Path
    trait: str
    target_token_id: int
    target_label: str
    base_model_id: str
    questions: list[str]
    assistant_prefill: str | None
    checkpoint_steps: list[int]
    include_final: bool
    top_k: int
    plot_top_k: int
    device: str
    torch_dtype: str
    force: bool
    make_plots: bool
    dry_run: bool
    verbose: bool


@dataclass(**DATACLASS_KW_ONLY)
class DistributionScore:
    """Mean vocabulary distribution and target values for one model stage."""

    mean_probabilities: np.ndarray
    mean_log_probabilities: np.ndarray
    target_question_log_probabilities: np.ndarray


def slugify(value: str) -> str:
    """Build the path-safe trait fragment used by the training script."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug.strip("_") or "value"


def parse_checkpoint_steps(raw_values: list[str] | None) -> list[int]:
    """Parse repeated or comma-separated positive checkpoint steps."""
    result = set()
    for raw in raw_values or []:
        for piece in raw.split(","):
            piece = piece.strip()
            if not piece:
                continue
            step = int(piece)
            if step <= 0:
                raise ValueError("Checkpoint steps must be positive.")
            result.add(step)
    return sorted(result)


def read_checkpoint_metadata(path: Path) -> dict[str, Any]:
    """Read optional checkpoint metadata."""
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        return {}
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def infer_final_step(run_dir: Path, trait: str, stages: list[CheckpointStage]) -> int:
    """Infer the final optimizer step from metadata or the loss curve."""
    totals = []
    trainer_root = run_dir / "trainer" / slugify(trait)
    for path in trainer_root.glob("checkpoint-step-*"):
        metadata = read_checkpoint_metadata(path)
        total = metadata.get("total_steps")
        if isinstance(total, int) and total > 0:
            totals.append(total)
    loss_path = run_dir / "training_curves" / f"{slugify(trait)}.loss.csv"
    if loss_path.is_file():
        with loss_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    totals.append(int(row["total_steps"]))
                except (KeyError, TypeError, ValueError):
                    continue
    if totals:
        return max(totals)
    return max((stage.optimizer_step for stage in stages), default=0) + 1


def discover_checkpoint_stages(
    *,
    run_dir: Path,
    trait: str,
    requested_steps: list[int],
    include_final: bool,
) -> list[CheckpointStage]:
    """Discover and validate chronological checkpoint adapters."""
    trainer_root = run_dir / "trainer" / slugify(trait)
    discovered: dict[int, CheckpointStage] = {}
    if trainer_root.is_dir():
        for path in trainer_root.iterdir():
            match = CHECKPOINT_PATTERN.fullmatch(path.name)
            adapter_path = path / "adapter"
            if match is None or not (adapter_path / "adapter_config.json").is_file():
                continue
            step = int(match.group(1))
            metadata = read_checkpoint_metadata(path)
            discovered[step] = CheckpointStage(
                label=f"step_{step:06d}",
                optimizer_step=step,
                adapter_path=adapter_path,
                epoch=(
                    int(metadata["epoch"])
                    if metadata.get("epoch") is not None
                    else None
                ),
                loss=(
                    float(metadata["loss"])
                    if metadata.get("loss") is not None
                    else None
                ),
                is_final=False,
            )
    if requested_steps:
        missing = sorted(set(requested_steps) - set(discovered))
        if missing:
            raise FileNotFoundError(
                f"Missing requested checkpoint adapters for steps {missing} under "
                f"{trainer_root}."
            )
        stages = [discovered[step] for step in requested_steps]
    else:
        stages = [discovered[step] for step in sorted(discovered)]
    if not stages and not include_final:
        raise FileNotFoundError(f"No checkpoint adapters found under {trainer_root}.")

    if include_final:
        final_adapter = run_dir / "adapters" / slugify(trait)
        if not (final_adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Missing final adapter under {final_adapter}.")
        final_step = infer_final_step(run_dir, trait, stages)
        stages.append(
            CheckpointStage(
                label=f"final_step_{final_step:06d}",
                optimizer_step=final_step,
                adapter_path=final_adapter,
                epoch=None,
                loss=None,
                is_final=True,
            )
        )
    return sorted(stages, key=lambda stage: (stage.optimizer_step, stage.is_final))


def adapter_fingerprint(adapter_path: Path) -> dict[str, Any]:
    """Return a cheap fingerprint for cache invalidation."""
    result: dict[str, Any] = {"adapter_path": str(adapter_path)}
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        path = adapter_path / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing adapter file: {path}")
        stat = path.stat()
        result[f"{name}_size"] = stat.st_size
        result[f"{name}_mtime_ns"] = stat.st_mtime_ns
    return result


def cache_metadata(
    config: CheckpointDynamicsConfig,
    stage: CheckpointStage | None,
) -> dict[str, Any]:
    """Build cache metadata for the base model or one adapter stage."""
    stage_payload = None
    if stage is not None:
        stage_payload = asdict(stage)
        stage_payload["adapter_path"] = str(stage.adapter_path)
    return {
        "cache_version": CACHE_VERSION,
        "base_model_id": config.base_model_id,
        "questions": config.questions,
        "assistant_prefill": config.assistant_prefill,
        "target_token_id": config.target_token_id,
        "torch_dtype": config.torch_dtype,
        "stage": stage_payload,
        "adapter_fingerprint": (
            None if stage is None else adapter_fingerprint(stage.adapter_path)
        ),
    }


def cache_paths(output_dir: Path, stage: CheckpointStage | None) -> tuple[Path, Path]:
    """Resolve distribution and metadata cache paths."""
    stem = "base" if stage is None else stage.label
    cache_dir = output_dir / "distribution_cache"
    return cache_dir / f"{stem}.npz", cache_dir / f"{stem}.metadata.json"


def load_cached_score(
    *,
    output_dir: Path,
    metadata: dict[str, Any],
    stage: CheckpointStage | None,
    force: bool,
) -> DistributionScore | None:
    """Load a reusable stage score when metadata matches exactly."""
    if force:
        return None
    npz_path, metadata_path = cache_paths(output_dir, stage)
    if not npz_path.is_file() or not metadata_path.is_file():
        return None
    try:
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous != metadata:
            return None
        with np.load(npz_path, allow_pickle=False) as payload:
            return DistributionScore(
                mean_probabilities=np.asarray(
                    payload["mean_probabilities"], dtype=np.float64
                ),
                mean_log_probabilities=np.asarray(
                    payload["mean_log_probabilities"], dtype=np.float64
                ),
                target_question_log_probabilities=np.asarray(
                    payload["target_question_log_probabilities"], dtype=np.float64
                ),
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable cache {}: {}", npz_path, exc)
        return None


def write_json(path: Path, payload: Any) -> None:
    """Write indented UTF-8 JSON with Path and NumPy support."""

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


def save_cached_score(
    *,
    output_dir: Path,
    metadata: dict[str, Any],
    stage: CheckpointStage | None,
    score: DistributionScore,
) -> None:
    """Persist one completed distribution stage immediately."""
    npz_path, metadata_path = cache_paths(output_dir, stage)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        mean_probabilities=score.mean_probabilities.astype(np.float32),
        mean_log_probabilities=score.mean_log_probabilities.astype(np.float32),
        target_question_log_probabilities=(
            score.target_question_log_probabilities.astype(np.float64)
        ),
    )
    write_json(metadata_path, metadata)
    logger.success("Saved completed stage cache {}", npz_path)


def score_vocab_distribution(
    *,
    label: str,
    model: Any,
    tokenizer: Any,
    device: Any,
    questions: list[str],
    assistant_prefill: str | None,
    target_token_id: int,
) -> DistributionScore:
    """Score mean vocabulary probabilities plus per-question target logs."""
    import torch

    sum_probabilities = None
    sum_log_probabilities = None
    target_logs = []
    with torch.inference_mode():
        for index, question in enumerate(questions, start=1):
            logger.info(
                "Scoring {} recovery prompt {}/{}", label, index, len(questions)
            )
            prompt = build_chat_prompt(
                tokenizer,
                user_prompt=question,
                assistant_prefill=assistant_prefill,
            )
            log_probabilities = next_token_log_probs(
                prompt_text=prompt,
                model=model,
                tokenizer=tokenizer,
                device=device,
            ).double()
            if not 0 <= target_token_id < int(log_probabilities.shape[0]):
                raise ValueError(
                    f"Target token {target_token_id} is outside vocabulary size "
                    f"{int(log_probabilities.shape[0])}."
                )
            probabilities = torch.exp(log_probabilities)
            if sum_probabilities is None:
                sum_probabilities = torch.zeros_like(probabilities)
                sum_log_probabilities = torch.zeros_like(log_probabilities)
            sum_probabilities += probabilities
            sum_log_probabilities += log_probabilities
            target_logs.append(float(log_probabilities[target_token_id].item()))
    if sum_probabilities is None or sum_log_probabilities is None:
        raise ValueError("At least one recovery question is required.")
    count = len(questions)
    return DistributionScore(
        mean_probabilities=(sum_probabilities / count).cpu().numpy(),
        mean_log_probabilities=(sum_log_probabilities / count).cpu().numpy(),
        target_question_log_probabilities=np.asarray(target_logs, dtype=np.float64),
    )


def validate_loaded_adapter(model: Any, adapter_name: str, path: Path) -> None:
    """Reject an adapter load that produced no nonzero LoRA-B parameters."""
    marker = f"lora_B.{adapter_name}"
    tensors = [
        parameter for name, parameter in model.named_parameters() if marker in name
    ]
    if not tensors:
        raise RuntimeError(f"No LoRA-B tensors loaded for {adapter_name} from {path}.")
    norm = sum(float(tensor.detach().float().norm().item()) for tensor in tensors)
    if norm <= 0.0:
        raise RuntimeError(f"Loaded adapter {adapter_name} has zero LoRA-B norm.")


def attach_stage_adapter(
    *,
    model: Any,
    model_is_peft: bool,
    stage: CheckpointStage,
) -> tuple[Any, bool, str]:
    """Attach one checkpoint adapter to a shared base model."""
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError("Checkpoint evaluation requires peft.") from exc

    adapter_name = stage.label
    if not model_is_peft:
        model = PeftModel.from_pretrained(
            model,
            str(stage.adapter_path),
            adapter_name=adapter_name,
            is_trainable=False,
        )
        model_is_peft = True
    else:
        model.load_adapter(
            str(stage.adapter_path),
            adapter_name=adapter_name,
            is_trainable=False,
        )
    validate_loaded_adapter(model, adapter_name, stage.adapter_path)
    model.set_adapter(adapter_name)
    model.eval()
    return model, model_is_peft, adapter_name


def detach_stage_adapter(model: Any, adapter_name: str) -> None:
    """Delete a completed stage adapter to bound accelerator memory."""
    delete_adapter = getattr(model, "delete_adapter", None)
    if delete_adapter is None:
        logger.warning("PEFT cannot delete {}; it remains resident.", adapter_name)
        return
    delete_adapter(adapter_name)


def clear_accelerator_cache() -> None:
    """Release Python and accelerator caches between model phases."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        return


def interpretable_mask(tokenizer: Any, vocab_size: int) -> np.ndarray:
    """Exclude tokenizer special IDs from competition rankings."""
    mask = np.ones(vocab_size, dtype=bool)
    for token_id in get_special_token_ids(tokenizer):
        if 0 <= token_id < vocab_size:
            mask[token_id] = False
    return mask


def descending_ranks(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return one-indexed descending ordinal ranks, or -1 outside mask."""
    eligible = np.flatnonzero(mask & np.isfinite(values))
    order = eligible[np.argsort(-values[eligible], kind="stable")]
    ranks = np.full(len(values), -1, dtype=np.int64)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def top_token_ids(
    *,
    values: np.ndarray,
    mask: np.ndarray,
    tokenizer: Any,
    top_k: int,
    excluded_ids: set[int] | None = None,
) -> list[int]:
    """Return top decoded non-special tokens for one metric."""
    excluded = excluded_ids or set()
    eligible = np.flatnonzero(mask & np.isfinite(values))
    order = eligible[np.argsort(-values[eligible], kind="stable")]
    result = []
    for token_id_value in order:
        token_id = int(token_id_value)
        if token_id in excluded or decode_one_token(tokenizer, token_id) == "":
            continue
        result.append(token_id)
        if len(result) >= top_k:
            break
    return result


def positive_mass_statistics(probability_gain: np.ndarray) -> dict[str, Any]:
    """Summarize positive shifted mass and its concentration."""
    positive = np.clip(np.asarray(probability_gain, dtype=np.float64), 0.0, None)
    total = float(np.sum(positive))
    if total <= 0.0:
        return {
            "positive_probability_shift_mass": 0.0,
            "effective_winner_count": 0.0,
            "shares": np.zeros_like(positive),
        }
    shares = positive / total
    nonzero = shares > 0.0
    entropy = -float(np.sum(shares[nonzero] * np.log(shares[nonzero])))
    return {
        "positive_probability_shift_mass": total,
        "effective_winner_count": math.exp(entropy),
        "shares": shares,
    }


def token_metric_row(
    *,
    token_id: int,
    tokenizer: Any,
    score: DistributionScore,
    log_boost: np.ndarray,
    probability_gain: np.ndarray,
    log_ranks: np.ndarray,
    gain_ranks: np.ndarray,
    positive_shares: np.ndarray,
) -> dict[str, Any]:
    """Build one readable token metric row at one stage."""
    return {
        "token_id": token_id,
        "decoded_text": decode_one_token(tokenizer, token_id),
        "mean_probability": float(score.mean_probabilities[token_id]),
        "mean_log_probability": float(score.mean_log_probabilities[token_id]),
        "mean_log_probability_boost": float(log_boost[token_id]),
        "mean_probability_gain": float(probability_gain[token_id]),
        "log_boost_rank": int(log_ranks[token_id]),
        "probability_gain_rank": int(gain_ranks[token_id]),
        "positive_shift_mass_share": float(positive_shares[token_id]),
    }


def compute_stage_analysis(
    *,
    stage: CheckpointStage,
    base: DistributionScore,
    score: DistributionScore,
    tokenizer: Any,
    mask: np.ndarray,
    target_token_id: int,
    top_k: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Compute target and competitor metrics for one chronological stage."""
    if score.mean_probabilities.shape != base.mean_probabilities.shape:
        raise ValueError("Base and checkpoint vocabulary sizes differ.")
    log_boost = score.mean_log_probabilities - base.mean_log_probabilities
    probability_gain = score.mean_probabilities - base.mean_probabilities
    log_ranks = descending_ranks(log_boost, mask)
    gain_ranks = descending_ranks(probability_gain, mask)
    mass = positive_mass_statistics(probability_gain)
    positive_shares = mass["shares"]
    competitor_exclusion = {target_token_id}
    top_log_ids = top_token_ids(
        values=log_boost,
        mask=mask,
        tokenizer=tokenizer,
        top_k=top_k,
        excluded_ids=competitor_exclusion,
    )
    top_gain_ids = top_token_ids(
        values=probability_gain,
        mask=mask,
        tokenizer=tokenizer,
        top_k=top_k,
        excluded_ids=competitor_exclusion,
    )
    target = token_metric_row(
        token_id=target_token_id,
        tokenizer=tokenizer,
        score=score,
        log_boost=log_boost,
        probability_gain=probability_gain,
        log_ranks=log_ranks,
        gain_ranks=gain_ranks,
        positive_shares=positive_shares,
    )
    base_target_probability = float(base.mean_probabilities[target_token_id])
    current_target_probability = float(score.mean_probabilities[target_token_id])
    target["ratio_of_mean_probabilities"] = (
        current_target_probability / base_target_probability
        if base_target_probability > 0.0
        else None
    )
    target["geometric_mean_probability_ratio"] = math.exp(
        target["mean_log_probability_boost"]
    )
    target["positive_question_count"] = int(
        np.sum(
            score.target_question_log_probabilities
            > base.target_question_log_probabilities
        )
    )
    top_log_rows = [
        token_metric_row(
            token_id=token_id,
            tokenizer=tokenizer,
            score=score,
            log_boost=log_boost,
            probability_gain=probability_gain,
            log_ranks=log_ranks,
            gain_ranks=gain_ranks,
            positive_shares=positive_shares,
        )
        for token_id in top_log_ids
    ]
    top_gain_rows = [
        token_metric_row(
            token_id=token_id,
            tokenizer=tokenizer,
            score=score,
            log_boost=log_boost,
            probability_gain=probability_gain,
            log_ranks=log_ranks,
            gain_ranks=gain_ranks,
            positive_shares=positive_shares,
        )
        for token_id in top_gain_ids
    ]
    best_log = top_log_rows[0]
    best_gain = top_gain_rows[0]
    summary = {
        "stage": asdict(stage),
        "target": target,
        "total_variation_distance": float(0.5 * np.sum(np.abs(probability_gain))),
        "positive_probability_shift_mass": mass["positive_probability_shift_mass"],
        "effective_winner_count": mass["effective_winner_count"],
        "target_log_competition_gap": float(
            best_log["mean_log_probability_boost"]
            - target["mean_log_probability_boost"]
        ),
        "target_mass_competition_gap": float(
            best_gain["mean_probability_gain"] - target["mean_probability_gain"]
        ),
        "top_competitors_by_log_boost": top_log_rows,
        "top_competitors_by_probability_gain": top_gain_rows,
    }
    arrays = {
        "log_boost": log_boost,
        "probability_gain": probability_gain,
        "log_ranks": log_ranks,
        "gain_ranks": gain_ranks,
        "positive_shares": positive_shares,
    }
    return summary, arrays


def select_main_competitors(
    *,
    stage_arrays: list[dict[str, np.ndarray]],
    tokenizer: Any,
    mask: np.ndarray,
    target_token_id: int,
    plot_top_k: int,
) -> dict[int, set[str]]:
    """Select final and transient competitors for trajectory panels."""
    if not stage_arrays:
        return {}
    groups: dict[int, set[str]] = defaultdict(set)
    excluded = {target_token_id}
    final = stage_arrays[-1]
    selectors = {
        "final_log_boost": final["log_boost"],
        "final_probability_gain": final["probability_gain"],
        "peak_log_boost": np.max(
            np.stack([arrays["log_boost"] for arrays in stage_arrays]), axis=0
        ),
        "peak_probability_gain": np.max(
            np.stack([arrays["probability_gain"] for arrays in stage_arrays]),
            axis=0,
        ),
    }
    for group, values in selectors.items():
        for token_id in top_token_ids(
            values=values,
            mask=mask,
            tokenizer=tokenizer,
            top_k=plot_top_k,
            excluded_ids=excluded,
        ):
            groups[token_id].add(group)
    return dict(groups)


def build_competitor_trajectories(
    *,
    base_score: DistributionScore,
    stages: list[CheckpointStage],
    stage_scores: list[DistributionScore],
    stage_arrays: list[dict[str, np.ndarray]],
    tokenizer: Any,
    selected: dict[int, set[str]],
    target_token_id: int,
) -> list[dict[str, Any]]:
    """Build long-form trajectory rows for target and selected competitors."""
    rows = []
    token_ids = [target_token_id] + sorted(selected)
    for token_id in token_ids:
        rows.append(
            {
                "stage_label": "base",
                "optimizer_step": 0,
                "epoch": 0,
                "is_final": False,
                "token_id": token_id,
                "decoded_text": decode_one_token(tokenizer, token_id),
                "selection_groups": (
                    "target"
                    if token_id == target_token_id
                    else ",".join(sorted(selected[token_id]))
                ),
                "mean_probability": float(base_score.mean_probabilities[token_id]),
                "mean_log_probability_boost": 0.0,
                "mean_probability_gain": 0.0,
                "log_boost_rank": None,
                "probability_gain_rank": None,
                "positive_shift_mass_share": 0.0,
            }
        )
    for stage, score, arrays in zip(stages, stage_scores, stage_arrays):
        for token_id in token_ids:
            rows.append(
                {
                    "stage_label": stage.label,
                    "optimizer_step": stage.optimizer_step,
                    "epoch": stage.epoch,
                    "is_final": stage.is_final,
                    "token_id": token_id,
                    "decoded_text": decode_one_token(tokenizer, token_id),
                    "selection_groups": (
                        "target"
                        if token_id == target_token_id
                        else ",".join(sorted(selected[token_id]))
                    ),
                    "mean_probability": float(score.mean_probabilities[token_id]),
                    "mean_log_probability_boost": float(arrays["log_boost"][token_id]),
                    "mean_probability_gain": float(
                        arrays["probability_gain"][token_id]
                    ),
                    "log_boost_rank": int(arrays["log_ranks"][token_id]),
                    "probability_gain_rank": int(arrays["gain_ranks"][token_id]),
                    "positive_shift_mass_share": float(
                        arrays["positive_shares"][token_id]
                    ),
                }
            )
    return rows


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of same-shaped dictionaries as UTF-8 CSV."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def target_trajectory_rows(
    *,
    base_score: DistributionScore,
    tokenizer: Any,
    target_token_id: int,
    stage_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten target checkpoint summaries for CSV output."""
    rows = [
        {
            "stage_label": "base",
            "optimizer_step": 0,
            "epoch": 0,
            "checkpoint_loss": None,
            "is_final": False,
            "target_token_id": target_token_id,
            "target_decoded_text": decode_one_token(tokenizer, target_token_id),
            "target_mean_probability": float(
                base_score.mean_probabilities[target_token_id]
            ),
            "target_mean_probability_gain": 0.0,
            "target_ratio_of_mean_probabilities": 1.0,
            "target_mean_log_probability_boost": 0.0,
            "target_geometric_mean_probability_ratio": 1.0,
            "target_positive_question_count": 0,
            "target_log_boost_rank": None,
            "target_probability_gain_rank": None,
            "target_positive_shift_mass_share": 0.0,
            "total_variation_distance": 0.0,
            "effective_winner_count": 0.0,
            "target_log_competition_gap": None,
            "target_mass_competition_gap": None,
            "top_log_competitor_id": None,
            "top_log_competitor_text": None,
            "top_log_competitor_boost": None,
            "top_mass_competitor_id": None,
            "top_mass_competitor_text": None,
            "top_mass_competitor_gain": None,
        }
    ]
    for summary in stage_summaries:
        stage = summary["stage"]
        target = summary["target"]
        top_log = summary["top_competitors_by_log_boost"][0]
        top_gain = summary["top_competitors_by_probability_gain"][0]
        rows.append(
            {
                "stage_label": stage["label"],
                "optimizer_step": stage["optimizer_step"],
                "epoch": stage["epoch"],
                "checkpoint_loss": stage["loss"],
                "is_final": stage["is_final"],
                "target_token_id": target["token_id"],
                "target_decoded_text": target["decoded_text"],
                "target_mean_probability": target["mean_probability"],
                "target_mean_probability_gain": target["mean_probability_gain"],
                "target_ratio_of_mean_probabilities": target[
                    "ratio_of_mean_probabilities"
                ],
                "target_mean_log_probability_boost": target[
                    "mean_log_probability_boost"
                ],
                "target_geometric_mean_probability_ratio": target[
                    "geometric_mean_probability_ratio"
                ],
                "target_positive_question_count": target["positive_question_count"],
                "target_log_boost_rank": target["log_boost_rank"],
                "target_probability_gain_rank": target["probability_gain_rank"],
                "target_positive_shift_mass_share": target["positive_shift_mass_share"],
                "total_variation_distance": summary["total_variation_distance"],
                "effective_winner_count": summary["effective_winner_count"],
                "target_log_competition_gap": summary["target_log_competition_gap"],
                "target_mass_competition_gap": summary["target_mass_competition_gap"],
                "top_log_competitor_id": top_log["token_id"],
                "top_log_competitor_text": top_log["decoded_text"],
                "top_log_competitor_boost": top_log["mean_log_probability_boost"],
                "top_mass_competitor_id": top_gain["token_id"],
                "top_mass_competitor_text": top_gain["decoded_text"],
                "top_mass_competitor_gain": top_gain["mean_probability_gain"],
            }
        )
    return rows


def top_competitor_rows(stage_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten per-stage top competitor tables for CSV output."""
    rows = []
    for summary in stage_summaries:
        stage = summary["stage"]
        for metric, key in (
            ("log_boost", "top_competitors_by_log_boost"),
            ("probability_gain", "top_competitors_by_probability_gain"),
        ):
            for rank, token in enumerate(summary[key], start=1):
                rows.append(
                    {
                        "stage_label": stage["label"],
                        "optimizer_step": stage["optimizer_step"],
                        "metric": metric,
                        "display_rank": rank,
                        **token,
                    }
                )
    return rows


def choose_plot_font() -> bool:
    """Configure a CJK-capable Matplotlib font when one is installed."""
    import matplotlib.font_manager as font_manager
    import matplotlib.pyplot as plt

    available = {font.name for font in font_manager.fontManager.ttflist}
    candidates = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Noto Sans CJK TC",
        "PingFang SC",
        "Microsoft YaHei",
        "Arial Unicode MS",
    ]
    chosen = next((name for name in candidates if name in available), None)
    if chosen is None:
        logger.warning(
            "No CJK plot font found; non-ASCII legend labels will use Unicode "
            "code-point escapes while JSON/CSV retain the exact text."
        )
        return False
    plt.rcParams["font.family"] = [chosen, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    logger.info("Using Matplotlib font {} for Unicode token labels", chosen)
    return True


def safe_plot_label(decoded_text: str, token_id: int, cjk_supported: bool) -> str:
    """Return a readable plot label without missing-glyph boxes."""
    if cjk_supported or decoded_text.isascii():
        text = json.dumps(decoded_text, ensure_ascii=False)
    else:
        text = decoded_text.encode("unicode_escape").decode("ascii")
    return f"{text} ({token_id})"


def make_plots(
    *,
    output_dir: Path,
    target_rows: list[dict[str, Any]],
    competitor_rows: list[dict[str, Any]],
    selected: dict[int, set[str]],
    target_token_id: int,
    target_label: str,
    plot_top_k: int,
) -> None:
    """Write target and main-competitor chronological plots."""
    import matplotlib.pyplot as plt

    cjk_supported = choose_plot_font()
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = np.asarray([row["optimizer_step"] for row in target_rows])

    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    probabilities = np.asarray([row["target_mean_probability"] for row in target_rows])
    axes[0].plot(steps, probabilities, marker="o", color="#1f77b4")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Mean target probability")
    axes[0].set_title(f"Chronological recovery of {target_label!r} ({target_token_id})")
    boosts = np.asarray(
        [row["target_mean_log_probability_boost"] for row in target_rows]
    )
    axes[1].plot(steps, boosts, marker="o", color="#d62728")
    axes[1].axhline(0.0, color="#777777", linewidth=0.9)
    axes[1].set_ylabel("Mean log boost (nats)")
    axes[1].set_xlabel("Optimizer step")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "target_chronological_trajectory.png", dpi=200)
    plt.close(figure)

    by_token: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in competitor_rows:
        by_token[int(row["token_id"])].append(row)

    def plot_panel(group: str, metric: str, ylabel: str, filename: str) -> None:
        token_ids = [
            token_id for token_id, groups in selected.items() if group in groups
        ][:plot_top_k]
        token_ids = [target_token_id] + token_ids
        figure, axis = plt.subplots(figsize=(9, 6))
        for token_id in token_ids:
            rows = sorted(by_token[token_id], key=lambda row: row["optimizer_step"])
            axis.plot(
                [row["optimizer_step"] for row in rows],
                [row[metric] for row in rows],
                marker="o",
                label=safe_plot_label(
                    str(rows[0]["decoded_text"]), token_id, cjk_supported
                ),
            )
        axis.axhline(0.0, color="#777777", linewidth=0.9)
        if metric == "mean_probability_gain":
            axis.set_yscale("symlog", linthresh=1e-8)
        axis.set_xlabel("Optimizer step")
        axis.set_ylabel(ylabel)
        axis.set_title("Chronological target and competitor trajectories")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, ncol=2)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=200)
        plt.close(figure)

    plot_panel(
        "final_probability_gain",
        "mean_probability_gain",
        "Mean probability gain (symlog)",
        "main_mass_competitor_dynamics.png",
    )
    plot_panel(
        "final_log_boost",
        "mean_log_probability_boost",
        "Mean log-probability boost (nats)",
        "main_log_competitor_dynamics.png",
    )


def write_markdown_summary(
    path: Path,
    final_summary: dict[str, Any],
    question_count: int,
) -> None:
    """Write a compact human-readable endpoint and trajectory summary."""
    target = final_summary["target"]
    top_mass = final_summary["top_competitors_by_probability_gain"][:10]
    lines = [
        "# Chronological rare-token checkpoint dynamics",
        "",
        f"Target: `{target['decoded_text']}` (token {target['token_id']})",
        "",
        "## Final target effect",
        "",
        f"- Mean probability: {target['mean_probability']:.6e}",
        f"- Ratio of mean probabilities: {target['ratio_of_mean_probabilities']:.6g}x",
        f"- Mean log boost: {target['mean_log_probability_boost']:+.6f} nats",
        f"- Geometric ratio: {target['geometric_mean_probability_ratio']:.6g}x",
        f"- Positive recovery questions: {target['positive_question_count']} / "
        f"{question_count}",
        f"- Log-boost rank: {target['log_boost_rank']:,}",
        f"- Absolute probability-gain rank: {target['probability_gain_rank']:,}",
        "",
        "## Largest final absolute probability gains",
        "",
        "| rank | token | token ID | probability gain | log boost |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(top_mass, start=1):
        token_text = json.dumps(row["decoded_text"], ensure_ascii=False)
        lines.append(
            f"| {rank} | `{token_text}` | {row['token_id']} | "
            f"{row['mean_probability_gain']:.6e} | "
            f"{row['mean_log_probability_boost']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "See `target_trajectory.csv`, `competitor_trajectories.csv`, and "
            "`top_competitors_by_checkpoint.csv` for the complete chronology.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def log_dry_run(
    config: CheckpointDynamicsConfig, stages: list[CheckpointStage]
) -> None:
    """Log resolved work without loading a model."""
    logger.info("Base model: {}", config.base_model_id)
    logger.info("Target: {!r} ({})", config.target_label, config.target_token_id)
    logger.info("Recovery questions: {}", len(config.questions))
    logger.info("Output directory: {}", config.output_dir)
    for stage in stages:
        logger.info(
            "{}: optimizer step {}, adapter {}",
            stage.label,
            stage.optimizer_step,
            stage.adapter_path,
        )


def run(config: CheckpointDynamicsConfig) -> dict[str, Any] | None:
    """Score chronological checkpoints and write analysis artifacts."""
    stages = discover_checkpoint_stages(
        run_dir=config.run_dir,
        trait=config.trait,
        requested_steps=config.checkpoint_steps,
        include_final=config.include_final,
    )
    if config.dry_run:
        log_dry_run(config, stages)
        return None

    config.output_dir.mkdir(parents=True, exist_ok=True)
    base_metadata = cache_metadata(config, None)
    base_score = load_cached_score(
        output_dir=config.output_dir,
        metadata=base_metadata,
        stage=None,
        force=config.force,
    )
    stage_scores: dict[str, DistributionScore] = {}
    pending = []
    for stage in stages:
        metadata = cache_metadata(config, stage)
        cached = load_cached_score(
            output_dir=config.output_dir,
            metadata=metadata,
            stage=stage,
            force=config.force,
        )
        if cached is None:
            pending.append(stage)
        else:
            logger.info("Reusing completed checkpoint cache {}", stage.label)
            stage_scores[stage.label] = cached

    tokenizer = None
    model = None
    base_model = None
    if base_score is None or pending:
        import torch

        device = resolve_device(torch, config.device)
        torch_dtype = resolve_torch_dtype(torch, config.torch_dtype)
        base_model, tokenizer = load_direct_model_and_tokenizer(
            model_id=config.base_model_id,
            device=device,
            torch_dtype=torch_dtype,
        )
        model = base_model
        model_is_peft = False
        if base_score is None:
            base_score = score_vocab_distribution(
                label="base",
                model=base_model,
                tokenizer=tokenizer,
                device=device,
                questions=config.questions,
                assistant_prefill=config.assistant_prefill,
                target_token_id=config.target_token_id,
            )
            save_cached_score(
                output_dir=config.output_dir,
                metadata=base_metadata,
                stage=None,
                score=base_score,
            )
        try:
            for stage in pending:
                logger.info(
                    "Loading chronological checkpoint {} from {}",
                    stage.label,
                    stage.adapter_path,
                )
                model, model_is_peft, adapter_name = attach_stage_adapter(
                    model=model,
                    model_is_peft=model_is_peft,
                    stage=stage,
                )
                try:
                    score = score_vocab_distribution(
                        label=stage.label,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        questions=config.questions,
                        assistant_prefill=config.assistant_prefill,
                        target_token_id=config.target_token_id,
                    )
                    save_cached_score(
                        output_dir=config.output_dir,
                        metadata=cache_metadata(config, stage),
                        stage=stage,
                        score=score,
                    )
                    stage_scores[stage.label] = score
                finally:
                    detach_stage_adapter(model, adapter_name)
                    clear_accelerator_cache()
        finally:
            del model
            del base_model
            clear_accelerator_cache()
    if tokenizer is None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Cached analysis still requires transformers.") from exc
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_id, trust_remote_code=True
        )
    if base_score is None:
        raise RuntimeError("Base score was not computed or loaded.")

    ordered_scores = [stage_scores[stage.label] for stage in stages]
    vocab_size = len(base_score.mean_probabilities)
    mask = interpretable_mask(tokenizer, vocab_size)
    base_probability_ranks = descending_ranks(base_score.mean_probabilities, mask)
    stage_summaries = []
    stage_arrays = []
    for stage, score in zip(stages, ordered_scores):
        summary, arrays = compute_stage_analysis(
            stage=stage,
            base=base_score,
            score=score,
            tokenizer=tokenizer,
            mask=mask,
            target_token_id=config.target_token_id,
            top_k=config.top_k,
        )
        stage_summaries.append(summary)
        stage_arrays.append(arrays)

    selected = select_main_competitors(
        stage_arrays=stage_arrays,
        tokenizer=tokenizer,
        mask=mask,
        target_token_id=config.target_token_id,
        plot_top_k=config.plot_top_k,
    )
    competitor_rows = build_competitor_trajectories(
        base_score=base_score,
        stages=stages,
        stage_scores=ordered_scores,
        stage_arrays=stage_arrays,
        tokenizer=tokenizer,
        selected=selected,
        target_token_id=config.target_token_id,
    )
    target_rows = target_trajectory_rows(
        base_score=base_score,
        tokenizer=tokenizer,
        target_token_id=config.target_token_id,
        stage_summaries=stage_summaries,
    )
    top_rows = top_competitor_rows(stage_summaries)
    write_csv_rows(config.output_dir / "target_trajectory.csv", target_rows)
    write_csv_rows(config.output_dir / "competitor_trajectories.csv", competitor_rows)
    write_csv_rows(config.output_dir / "top_competitors_by_checkpoint.csv", top_rows)

    selected_rows = []
    for token_id, groups in sorted(selected.items()):
        selected_rows.append(
            {
                "token_id": token_id,
                "decoded_text": decode_one_token(tokenizer, token_id),
                "selection_groups": sorted(groups),
            }
        )
    result = {
        "analysis": "Chronological rare-token LoRA checkpoint dynamics",
        "definitions": {
            "mean_log_probability_boost": (
                "mean_q[log P_checkpoint(t|q)-log P_base(t|q)]"
            ),
            "mean_probability_gain": ("mean_q[P_checkpoint(t|q)]-mean_q[P_base(t|q)]"),
            "target_positive_shift_mass_share": (
                "positive target probability gain divided by total positive "
                "vocabulary probability gain"
            ),
            "effective_winner_count": (
                "exp(entropy) of normalized positive vocabulary probability gains"
            ),
            "chronology": (
                "Each point is an actually saved optimizer checkpoint adapter; "
                "no final-adapter scale interpolation is used."
            ),
        },
        "config": asdict(config),
        "base": {
            "target_token_id": config.target_token_id,
            "target_decoded_text": decode_one_token(tokenizer, config.target_token_id),
            "target_mean_probability": float(
                base_score.mean_probabilities[config.target_token_id]
            ),
            "target_mean_log_probability": float(
                base_score.mean_log_probabilities[config.target_token_id]
            ),
            "target_probability_rank": int(
                base_probability_ranks[config.target_token_id]
            ),
        },
        "stages": stage_summaries,
        "selected_main_competitors": selected_rows,
        "final_summary": stage_summaries[-1],
    }
    write_json(config.output_dir / "checkpoint_dynamics_analysis.json", result)
    write_markdown_summary(
        config.output_dir / "checkpoint_dynamics_analysis.md",
        stage_summaries[-1],
        len(config.questions),
    )
    if config.make_plots:
        make_plots(
            output_dir=config.output_dir,
            target_rows=target_rows,
            competitor_rows=competitor_rows,
            selected=selected,
            target_token_id=config.target_token_id,
            target_label=config.target_label,
            plot_top_k=config.plot_top_k,
        )

    final_target = stage_summaries[-1]["target"]
    logger.success(
        "Chronological analysis completed: target {!r} final mean P={:.6e}, "
        "log boost={:+.4f} nats, gain rank={:,}. Outputs: {}",
        final_target["decoded_text"],
        final_target["mean_probability"],
        final_target["mean_log_probability_boost"],
        final_target["probability_gain_rank"],
        config.output_dir,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate target-token recovery and vocabulary competitors across "
            "actual chronological LoRA checkpoints."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trait", default=DEFAULT_TRAIT)
    parser.add_argument("--target-token-id", type=int, default=DEFAULT_TARGET_TOKEN_ID)
    parser.add_argument("--target-label", default=DEFAULT_TARGET_LABEL)
    parser.add_argument("--base-model-id", default=DEFAULT_BASE_MODEL_ID)
    parser.add_argument("--question", action="append", default=[])
    parser.add_argument("--questions-path", type=Path)
    parser.add_argument("--assistant-prefill")
    parser.add_argument(
        "--checkpoint-step",
        action="append",
        default=[],
        help="Repeated or comma-separated steps; default discovers every checkpoint.",
    )
    parser.add_argument("--no-final", action="store_true")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--plot-top-k", type=int, default=6)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def load_questions(args: argparse.Namespace) -> list[str]:
    """Load recovery questions from CLI, JSON, or defaults."""
    if args.questions_path is not None:
        payload = json.loads(args.questions_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not all(
            isinstance(item, str) for item in payload
        ):
            raise ValueError("--questions-path must contain a JSON string list.")
        return payload
    return args.question or list(DEFAULT_QUESTIONS)


def build_config(args: argparse.Namespace) -> CheckpointDynamicsConfig:
    """Validate CLI arguments and construct the run configuration."""
    if args.target_token_id < 0:
        raise ValueError("--target-token-id must be non-negative.")
    if args.top_k <= 0 or args.plot_top_k <= 0:
        raise ValueError("--top-k and --plot-top-k must be positive.")
    questions = load_questions(args)
    if not questions:
        raise ValueError("At least one recovery question is required.")
    return CheckpointDynamicsConfig(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        trait=args.trait,
        target_token_id=args.target_token_id,
        target_label=args.target_label,
        base_model_id=args.base_model_id,
        questions=questions,
        assistant_prefill=args.assistant_prefill,
        checkpoint_steps=parse_checkpoint_steps(args.checkpoint_step),
        include_final=not args.no_final,
        top_k=args.top_k,
        plot_top_k=args.plot_top_k,
        device=args.device,
        torch_dtype=args.torch_dtype,
        force=args.force,
        make_plots=not args.no_plots,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )


def main() -> None:
    """Run CLI entry point."""
    args = build_parser().parse_args()
    configure_logging(args.verbose)
    run(build_config(args))


if __name__ == "__main__":
    main()
