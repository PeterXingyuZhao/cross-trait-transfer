#!/usr/bin/env python3
"""Analyze hidden-state matrix differences with SVD and a logit lens."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from loguru import logger

# Package-qualified precision helpers also need the root during direct CLI runs.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MODEL_ID = "unsloth/Qwen2.5-7B-Instruct"
DEFAULT_MATRIX_DIR = Path("results/animal_hidden_state_matrices")
DEFAULT_OUTPUT_DIR = DEFAULT_MATRIX_DIR / "svd_logit_lens"
DEFAULT_PROMPT_MODE = "dataset"
DEFAULT_BLOCK_HIDDEN_DIM = 3584
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}
PromptMode = Literal["dataset", "random_tokens"]


@dataclass(**DATACLASS_KW_ONLY)
class MatrixSpec:
    """Named hidden-state matrix input."""

    name: str
    path: Path


@dataclass(**DATACLASS_KW_ONLY)
class SVDLogitLensConfig:
    """Configuration for SVD plus logit-lens analysis."""

    model_id: str
    neutral: MatrixSpec
    traits: list[MatrixSpec]
    output_dir: Path
    num_components: int
    top_k_svs: int
    top_k_tokens: int
    top_k_trait_tokens: int
    powerlaw_k: int
    block_hidden_dim: int
    expected_num_blocks: int | None
    use_combined_mean: bool
    orient_by_trait_tokens: bool
    scan_vocab_for_exact_trait_tokens: bool
    skip_logit_lens: bool
    skip_existing: bool
    save_plots: bool
    torch_dtype: str
    device_map: str


class TextReport:
    """Accumulate a plain-text report while logging it."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def line(self, message: str = "") -> None:
        self.lines.append(message)
        if message:
            logger.info(message)
        else:
            logger.info("")

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines).rstrip() + "\n", encoding="utf-8")
        logger.success("Saved text report to {}", path)


def parse_csv_values(raw_values: list[str] | None, default: list[str]) -> list[str]:
    """Parse repeated or comma-separated CLI values while preserving order."""
    if raw_values is None:
        return list(default)

    values: list[str] = []
    seen = set()
    for raw_value in raw_values:
        for piece in raw_value.split(","):
            value = piece.strip()
            if value and value not in seen:
                values.append(value)
                seen.add(value)
    if not values:
        raise ValueError("At least one value is required.")
    return values


def slugify(value: str) -> str:
    """Build a path-safe name fragment."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug.strip("_") or "value"


def infer_matrix_path(
    *,
    matrix_dir: Path,
    condition: str,
    prompt_mode: PromptMode,
    no_system_prompt: bool = False,
    matrix_format: str = "csv",
) -> Path:
    """Infer the matrix path emitted by compute_animal_hidden_state_matrices.py."""
    condition_stem = slugify(condition)
    if no_system_prompt:
        condition_stem = f"{condition_stem}_no_system"
    return matrix_dir / (
        f"{condition_stem}_{prompt_mode}_last_hidden_states.{matrix_format}"
    )


def load_matrix(filepath: Path) -> np.ndarray:
    """Load a hidden-state matrix from CSV or NPY as float32."""
    ext = filepath.suffix.lower()
    if ext == ".npy":
        arr = np.load(filepath)
        if arr.ndim != 2:
            raise ValueError(f"{filepath} is not a 2D array.")
        return arr.astype(np.float32, copy=False)

    if ext == ".csv":
        import pandas as pd

        df = pd.read_csv(filepath)
        if "prompt_h" in df.columns:
            df = df.drop(columns=["prompt_h"])

        numeric_df = df.apply(pd.to_numeric, errors="coerce")
        numeric_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        valid_mask = numeric_df.notna().all(axis=1)
        dropped_count = int((~valid_mask).sum())
        if dropped_count:
            logger.warning("Dropped {} non-numeric rows from {}", dropped_count, filepath)
        numeric_df = numeric_df.loc[valid_mask].reset_index(drop=True)
        if numeric_df.empty:
            raise ValueError(f"No numeric rows found in {filepath}")
        return numeric_df.values.astype(np.float32)

    raise ValueError(f"Unsupported file type: {filepath}")


def project_out_mean(
    delta: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Project each delta row off the reference mean direction."""
    mean_vector = reference.mean(axis=0).astype(np.float32, copy=False)
    mean_norm_sq = float(np.dot(mean_vector, mean_vector))
    if mean_norm_sq <= 0.0:
        logger.warning("Reference mean vector norm is zero; skipping projection filter.")
        return delta.astype(np.float32, copy=False), mean_norm_sq

    coefficients = delta.dot(mean_vector) / mean_norm_sq
    filtered = delta - coefficients[:, None] * mean_vector[None, :]
    return filtered.astype(np.float32, copy=False), mean_norm_sq


def compute_filtered_delta(
    *,
    neutral_matrix: np.ndarray,
    trait_matrix: np.ndarray,
    use_combined_mean: bool,
) -> tuple[np.ndarray, str, float]:
    """Compute trait-minus-neutral deltas and remove the mean direction."""
    if neutral_matrix.shape != trait_matrix.shape:
        raise ValueError(
            "Shape mismatch: "
            f"neutral {neutral_matrix.shape} vs trait {trait_matrix.shape}"
        )

    delta = trait_matrix - neutral_matrix
    if use_combined_mean:
        reference = np.vstack([neutral_matrix, trait_matrix])
        mean_source = "combined mean (neutral & trait)"
    else:
        reference = neutral_matrix
        mean_source = "mean(neutral)"

    filtered, mean_norm_sq = project_out_mean(delta, reference)
    return filtered, mean_source, mean_norm_sq


def fit_powerlaw(singular_values: np.ndarray, powerlaw_k: int) -> dict[str, Any]:
    """Fit a power law to the first valid singular values."""
    indices = np.arange(1, len(singular_values) + 1)
    valid_mask = singular_values > 1e-12
    values = singular_values[valid_mask]
    valid_indices = indices[valid_mask]
    fit_count = min(powerlaw_k, len(values))

    if fit_count < 2:
        return {
            "fit_count": fit_count,
            "alpha": None,
            "coefficient": None,
        }

    fit_indices = valid_indices[:fit_count]
    fit_values = values[:fit_count]
    slope, intercept = np.polyfit(np.log(fit_indices), np.log(fit_values), deg=1)
    return {
        "fit_count": int(fit_count),
        "alpha": float(-slope),
        "coefficient": float(np.exp(intercept)),
    }


def get_final_norm_module(model: Any) -> Any | None:
    """Return the final normalization module when the model exposes one."""
    candidates = [
        getattr(
            getattr(getattr(model, "model", None), "language_model", None),
            "norm",
            None,
        ),
        getattr(getattr(model, "language_model", None), "norm", None),
        getattr(getattr(model, "model", None), "norm", None),
        getattr(getattr(model, "transformer", None), "ln_f", None),
        getattr(model, "norm", None),
        getattr(getattr(model, "model", None), "final_layernorm", None),
    ]
    return next((candidate for candidate in candidates if candidate is not None), None)


def get_output_embedding_module(model: Any) -> Any:
    """Return the output embedding or LM-head module."""
    output_embedding = model.get_output_embeddings()
    if output_embedding is not None:
        return output_embedding
    if hasattr(model, "lm_head"):
        return model.lm_head
    raise RuntimeError("Could not find output embeddings / lm_head.")


def module_device_and_dtype(module: Any) -> tuple[Any, Any]:
    """Return a module parameter or buffer device/dtype pair."""
    for parameter in module.parameters():
        return parameter.device, parameter.dtype
    for buffer in module.buffers():
        return buffer.device, buffer.dtype
    raise RuntimeError(f"Could not determine device/dtype for {type(module).__name__}")


def resolve_torch_dtype(torch_module: Any, dtype_name: str) -> Any:
    """Resolve torch dtype from CLI text."""
    if dtype_name == "float16":
        return torch_module.float16
    if dtype_name == "bfloat16":
        return torch_module.bfloat16
    if dtype_name == "float32":
        return torch_module.float32
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


@dataclass(**DATACLASS_KW_ONLY)
class LensContext:
    """Loaded model modules needed for logit-lens projection."""

    tokenizer: Any
    model: Any
    output_embedding: Any
    final_norm: Any | None
    output_device: Any
    output_dtype: Any
    norm_device: Any
    norm_dtype: Any


def load_lens_context(config: SVDLogitLensConfig) -> LensContext:
    """Load tokenizer/model modules for logit-lens projection."""
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from scripts.native_generation_loader import (
        generation_model_class,
        replication_revision_kwargs,
    )

    revision_kwargs = replication_revision_kwargs(config.model_id)
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=True, **revision_kwargs)
    model_config = AutoConfig.from_pretrained(
        config.model_id,
        trust_remote_code=True,
        **revision_kwargs,
    )
    text_config = getattr(model_config, "text_config", model_config)
    if getattr(text_config, "model_type", None) == "qwen3_5_text":
        try:
            from transformers import AutoModelForMultimodalLM
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3.5 requires a Transformers build that provides "
                "AutoModelForMultimodalLM."
            ) from exc
        model_class = AutoModelForMultimodalLM
    else:
        model_class = generation_model_class(model_config)

    from scripts.strict_metric_precision import (
        configure_strict_qwen_measurement,
        is_strict_qwen_measurement,
    )
    dtype = resolve_torch_dtype(torch, config.torch_dtype)
    strict_qwen = is_strict_qwen_measurement(model_config, dtype)
    model = model_class.from_pretrained(
        config.model_id,
        torch_dtype=dtype,
        device_map=config.device_map,
        trust_remote_code=True,
        **revision_kwargs,
        **({"attn_implementation": "sdpa"} if strict_qwen else {}),
    )
    if strict_qwen:
        configure_strict_qwen_measurement(model)
    model.eval()

    output_embedding = get_output_embedding_module(model)
    final_norm = get_final_norm_module(model)
    output_device, output_dtype = module_device_and_dtype(output_embedding)
    if final_norm is None:
        norm_device, norm_dtype = output_device, output_dtype
        logger.warning("No final norm module found; projecting vectors directly.")
    else:
        norm_device, norm_dtype = module_device_and_dtype(final_norm)
        logger.info("Using final norm module: {}", type(final_norm).__name__)

    return LensContext(
        tokenizer=tokenizer,
        model=model,
        output_embedding=output_embedding,
        final_norm=final_norm,
        output_device=output_device,
        output_dtype=output_dtype,
        norm_device=norm_device,
        norm_dtype=norm_dtype,
    )


def vector_to_logits(vector: np.ndarray, lens: LensContext) -> Any:
    """Project one hidden-size vector through the final norm and LM head."""
    import torch

    tensor = torch.tensor(
        vector,
        device=lens.norm_device,
        dtype=lens.norm_dtype,
    ).unsqueeze(0)
    if lens.final_norm is not None:
        tensor = lens.final_norm(tensor)
    tensor = tensor.to(device=lens.output_device, dtype=lens.output_dtype)
    logits = torch.matmul(tensor, lens.output_embedding.weight.t())
    if getattr(lens.output_embedding, "bias", None) is not None:
        logits = logits + lens.output_embedding.bias
    return logits.squeeze(0).detach().float().cpu()


def decode_one_token(tokenizer: Any, token_id: int) -> str:
    """Decode one token id."""
    try:
        return tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode([token_id], skip_special_tokens=False)


def encode_without_special_tokens(tokenizer: Any, text: str) -> list[int]:
    """Encode text without adding chat or special tokens."""
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except AttributeError:
        encoded = tokenizer(text, add_special_tokens=False)
        return list(encoded.input_ids)


def trait_candidate_token_ids(
    *,
    tokenizer: Any,
    trait: str,
    scan_vocab_for_exact_trait_tokens: bool,
) -> list[int]:
    """Find token ids that are useful for orienting/reporting a trait direction."""
    variants = [
        trait,
        f" {trait}",
        f"\n{trait}",
        f"'{trait}'",
        f'"{trait}"',
    ]
    token_ids: set[int] = set()
    for variant in variants:
        token_ids.update(int(token_id) for token_id in encode_without_special_tokens(tokenizer, variant))

    if scan_vocab_for_exact_trait_tokens:
        for token_id in range(len(tokenizer)):
            decoded = decode_one_token(tokenizer, token_id)
            if decoded.strip() == trait:
                token_ids.add(int(token_id))

    return sorted(token_ids)


def top_tokens_from_logits(logits: Any, tokenizer: Any, *, top_k: int) -> list[dict[str, Any]]:
    """Return top-k token rows from a logit vector."""
    import torch

    probs = torch.softmax(logits, dim=-1)
    topk = torch.topk(logits, k=top_k)
    rows = []
    for rank, (logit, token_id) in enumerate(
        zip(topk.values.tolist(), topk.indices.tolist()),
        start=1,
    ):
        token_id = int(token_id)
        rows.append(
            {
                "rank": rank,
                "token_id": token_id,
                "token": decode_one_token(tokenizer, token_id),
                "logit": float(logit),
                "prob": float(probs[token_id].item()),
            }
        )
    return rows


def bottom_tokens_from_logits(logits: Any, tokenizer: Any, *, top_k: int) -> list[dict[str, Any]]:
    """Return bottom-k token rows from a logit vector."""
    import torch

    probs = torch.softmax(logits, dim=-1)
    bottomk = torch.topk(-logits, k=top_k)
    rows = []
    for rank, (_negative_logit, token_id) in enumerate(
        zip(bottomk.values.tolist(), bottomk.indices.tolist()),
        start=1,
    ):
        token_id = int(token_id)
        rows.append(
            {
                "rank": rank,
                "token_id": token_id,
                "token": decode_one_token(tokenizer, token_id),
                "logit": float(logits[token_id].item()),
                "prob": float(probs[token_id].item()),
            }
        )
    return rows


def candidate_token_scores(
    logits: Any,
    tokenizer: Any,
    candidate_token_ids: list[int],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    """Return sorted logit-lens scores for trait candidate tokens."""
    rows = []
    for token_id in candidate_token_ids:
        if token_id < 0 or token_id >= logits.numel():
            continue
        rows.append(
            {
                "token_id": int(token_id),
                "token": decode_one_token(tokenizer, int(token_id)),
                "logit": float(logits[token_id].item()),
            }
        )
    rows.sort(key=lambda row: row["logit"], reverse=True)
    return rows[:top_k]


def canonicalize_sign(vector: np.ndarray) -> tuple[np.ndarray, str]:
    """Make SVD sign deterministic when no trait-token orientation is available."""
    index = int(np.argmax(np.abs(vector)))
    if vector[index] < 0:
        return -vector, "flipped_largest_magnitude_positive"
    return vector, "kept_largest_magnitude_positive"


def orient_vector_by_trait_tokens(
    *,
    vector: np.ndarray,
    candidate_token_ids: list[int],
    lens: LensContext,
    block_hidden_dim: int,
) -> tuple[np.ndarray, str]:
    """Choose the SVD sign that scores trait candidate tokens higher."""
    if not candidate_token_ids:
        return canonicalize_sign(vector)

    positive_score = -math.inf
    negative_score = -math.inf
    num_blocks = vector.shape[0] // block_hidden_dim
    for block_index in range(num_blocks):
        start = block_index * block_hidden_dim
        end = start + block_hidden_dim
        logits = vector_to_logits(vector[start:end], lens)
        block_positive = float(logits[candidate_token_ids].max().item())
        block_negative = float((-logits[candidate_token_ids]).max().item())
        positive_score = max(positive_score, block_positive)
        negative_score = max(negative_score, block_negative)

    if negative_score > positive_score:
        return -vector, "flipped_toward_trait_tokens"
    return vector, "kept_toward_trait_tokens"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.success("Saved JSON report to {}", path)


def maybe_save_plots(
    *,
    singular_values: np.ndarray,
    output_prefix: Path,
    title_suffix: str,
    powerlaw: dict[str, Any],
) -> list[str]:
    """Save optional scree plots and return their paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    saved_paths = []
    indices = np.arange(1, len(singular_values) + 1)
    valid_mask = singular_values > 1e-12
    values = singular_values[valid_mask]
    valid_indices = indices[valid_mask]

    loglog_path = output_prefix.with_name(f"{output_prefix.name}_scree_loglog.png")
    plt.figure(figsize=(9, 5))
    plt.scatter(valid_indices, values, alpha=0.8, s=24, label="singular values")
    if powerlaw["alpha"] is not None and powerlaw["coefficient"] is not None:
        fit_count = int(powerlaw["fit_count"])
        fit_indices = valid_indices[:fit_count]
        fit_line = powerlaw["coefficient"] * (fit_indices ** (-powerlaw["alpha"]))
        plt.plot(fit_indices, fit_line, linestyle="--", linewidth=2, label="power-law fit")
    plt.xscale("log")
    plt.yscale("log")
    plt.title(f"Log-Log Scree: {title_suffix}")
    plt.xlabel("Index (log)")
    plt.ylabel("Singular value (log)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(loglog_path, dpi=160)
    plt.close()
    saved_paths.append(str(loglog_path))

    linear_path = output_prefix.with_name(f"{output_prefix.name}_scree_linear.png")
    plt.figure(figsize=(9, 5))
    plt.scatter(valid_indices, values, alpha=0.7, s=20)
    plt.plot(valid_indices, values, linestyle="-", linewidth=0.8, alpha=0.6)
    plt.title(f"Linear Scree: {title_suffix}")
    plt.xlabel("Index")
    plt.ylabel("Singular value")
    plt.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(linear_path, dpi=160)
    plt.close()
    saved_paths.append(str(linear_path))

    return saved_paths


def report_paths_for_pair(config: SVDLogitLensConfig, trait: MatrixSpec) -> tuple[Path, Path]:
    """Return JSON and text report paths for one comparison."""
    output_prefix = (
        config.output_dir
        / f"{slugify(trait.name)}_minus_{slugify(config.neutral.name)}"
    )
    return output_prefix.with_suffix(".json"), output_prefix.with_suffix(".txt")


def format_token_row(row: dict[str, Any]) -> str:
    """Format one token row for a text report."""
    return (
        f"{row['rank']:>2}. token_id={row['token_id']:<7d} "
        f"token={row['token']!r:<16s} logit={row['logit']: .4f} "
        f"prob={row['prob']:.6f}"
    )


def sha256_file(path: Path) -> str:
    """Return the content hash of one completed input matrix."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def report_cache_identity(
    config: SVDLogitLensConfig, trait: MatrixSpec,
) -> dict[str, Any]:
    """Identify matrix contents and all numerical and reporting options."""
    operational_fields = {
        "neutral", "traits", "output_dir", "skip_existing", "device_map",
    }
    return {
        "analysis_version": 1,
        "neutral_name": config.neutral.name,
        "trait_name": trait.name,
        "neutral_matrix_sha256": sha256_file(config.neutral.path),
        "trait_matrix_sha256": sha256_file(trait.path),
        "settings": {
            key: value for key, value in asdict(config).items()
            if key not in operational_fields
        },
    }


def report_cache_matches(config: SVDLogitLensConfig, trait: MatrixSpec) -> bool:
    """Reject stale reports when either input matrix or analysis changes."""
    json_path, text_path = report_paths_for_pair(config, trait)
    if not json_path.is_file() or not text_path.is_file():
        return False
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(payload, dict)
        and payload.get("cache_identity") == report_cache_identity(config, trait)
        and all(Path(path).is_file() for path in payload.get("plots", []))
    )


def analyze_pair(
    *,
    config: SVDLogitLensConfig,
    trait: MatrixSpec,
    lens: LensContext | None,
) -> dict[str, Any]:
    """Analyze one trait matrix against the neutral matrix."""
    json_path, text_path = report_paths_for_pair(config, trait)
    output_prefix = json_path.with_suffix("")
    if config.skip_existing and report_cache_matches(config, trait):
        logger.info("Skipping existing SVD/logit-lens report {}", json_path)
        return {"skipped": True, "json_path": str(json_path), "text_path": str(text_path)}

    report = TextReport()
    report.line(f"=== Comparing '{trait.name}' vs '{config.neutral.name}' ===")
    report.line(f"Neutral matrix: {config.neutral.path}")
    report.line(f"Trait matrix:   {trait.path}")

    neutral_matrix = load_matrix(config.neutral.path)
    trait_matrix = load_matrix(trait.path)
    report.line(f"Neutral shape: {neutral_matrix.shape}")
    report.line(f"Trait shape:   {trait_matrix.shape}")

    if neutral_matrix.shape[1] % config.block_hidden_dim != 0:
        raise ValueError(
            f"Matrix width {neutral_matrix.shape[1]} is not divisible by "
            f"--block-hidden-dim {config.block_hidden_dim}."
        )

    num_blocks = neutral_matrix.shape[1] // config.block_hidden_dim
    report.line(f"Detected {num_blocks} block(s) of size {config.block_hidden_dim}.")
    if (
        config.expected_num_blocks is not None
        and num_blocks != config.expected_num_blocks
    ):
        report.line(
            f"Warning: expected {config.expected_num_blocks} blocks, found {num_blocks}."
        )

    filtered_delta, mean_source, mean_norm_sq = compute_filtered_delta(
        neutral_matrix=neutral_matrix,
        trait_matrix=trait_matrix,
        use_combined_mean=config.use_combined_mean,
    )
    report.line(f"Applied projection filter using {mean_source}.")
    report.line(f"Reference mean norm squared: {mean_norm_sq:.6f}")
    report.line("Performing SVD on filtered trait-minus-neutral delta...")

    _u, singular_values, vt = np.linalg.svd(filtered_delta, full_matrices=False)
    total_energy = float(np.sum(singular_values**2))
    explained_variance = (
        (singular_values**2) / total_energy
        if total_energy > 0.0
        else np.zeros_like(singular_values)
    )
    powerlaw = fit_powerlaw(singular_values, config.powerlaw_k)

    report.line(f"Top {min(config.top_k_svs, len(singular_values))} singular values:")
    for index in range(min(config.top_k_svs, len(singular_values))):
        report.line(f"  SV {index + 1}: {singular_values[index]:.6f}")
    if powerlaw["alpha"] is not None:
        report.line(
            "Power-law fit: "
            f"top {powerlaw['fit_count']} SVs, alpha={powerlaw['alpha']:.4f}"
        )

    plot_paths: list[str] = []
    if config.save_plots:
        plot_paths = maybe_save_plots(
            singular_values=singular_values,
            output_prefix=output_prefix,
            title_suffix=f"{trait.name} - {config.neutral.name}",
            powerlaw=powerlaw,
        )
        for plot_path in plot_paths:
            report.line(f"Saved plot: {plot_path}")

    components: list[dict[str, Any]] = []
    if lens is None:
        report.line("Skipping logit lens.")
    else:
        report.line("")
        report.line("================= LOGIT-LENS INTERPRETATION =================")
        candidate_ids = trait_candidate_token_ids(
            tokenizer=lens.tokenizer,
            trait=trait.name,
            scan_vocab_for_exact_trait_tokens=config.scan_vocab_for_exact_trait_tokens,
        )
        candidate_preview = [
            {
                "token_id": token_id,
                "token": decode_one_token(lens.tokenizer, token_id),
            }
            for token_id in candidate_ids[: config.top_k_trait_tokens]
        ]
        report.line(
            "Trait token candidates: "
            + ", ".join(
                f"{row['token_id']}:{row['token']!r}" for row in candidate_preview
            )
        )

        component_count = min(config.num_components, vt.shape[0])
        for component_index in range(component_count):
            raw_vector = vt[component_index, :].astype(np.float32, copy=True)
            if config.orient_by_trait_tokens:
                full_vector, orientation = orient_vector_by_trait_tokens(
                    vector=raw_vector,
                    candidate_token_ids=candidate_ids,
                    lens=lens,
                    block_hidden_dim=config.block_hidden_dim,
                )
            else:
                full_vector, orientation = canonicalize_sign(raw_vector)

            component_payload: dict[str, Any] = {
                "component": component_index + 1,
                "singular_value": float(singular_values[component_index]),
                "explained_variance": float(explained_variance[component_index]),
                "orientation": orientation,
                "blocks": [],
            }

            report.line("")
            report.line(f"--- SV {component_index + 1} ---")
            report.line(f"Singular value: {singular_values[component_index]:.6f}")
            report.line(
                f"Explained variance: {explained_variance[component_index] * 100:.4f}%"
            )
            report.line(f"Orientation: {orientation}")

            for block_index in range(num_blocks):
                start = block_index * config.block_hidden_dim
                end = start + config.block_hidden_dim
                logits = vector_to_logits(full_vector[start:end], lens)
                top_tokens = top_tokens_from_logits(
                    logits,
                    lens.tokenizer,
                    top_k=config.top_k_tokens,
                )
                bottom_tokens = bottom_tokens_from_logits(
                    logits,
                    lens.tokenizer,
                    top_k=config.top_k_tokens,
                )
                trait_scores = candidate_token_scores(
                    logits,
                    lens.tokenizer,
                    candidate_ids,
                    top_k=config.top_k_trait_tokens,
                )

                report.line("")
                report.line(f"  >>> block {block_index}")
                report.line("  Top tokens from logit lens:")
                for row in top_tokens:
                    report.line("   " + format_token_row(row))
                report.line("  Bottom tokens from logit lens:")
                for row in bottom_tokens:
                    report.line("   " + format_token_row(row))
                if trait_scores:
                    report.line("  Trait candidate token scores:")
                    for rank, row in enumerate(trait_scores, start=1):
                        report.line(
                            f"   {rank:>2}. token_id={row['token_id']:<7d} "
                            f"token={row['token']!r:<16s} "
                            f"logit={row['logit']: .4f}"
                        )

                component_payload["blocks"].append(
                    {
                        "block": block_index,
                        "top_tokens": top_tokens,
                        "bottom_tokens": bottom_tokens,
                        "trait_token_scores": trait_scores,
                    }
                )
            components.append(component_payload)

    payload = {
        "method": "hidden_state_difference_svd_logit_lens",
        "cache_identity": report_cache_identity(config, trait),
        "model_id": config.model_id,
        "neutral": asdict(config.neutral),
        "trait": asdict(trait),
        "difference_direction": f"{trait.name} - {config.neutral.name}",
        "matrix_shape": list(neutral_matrix.shape),
        "block_hidden_dim": config.block_hidden_dim,
        "num_blocks": num_blocks,
        "use_combined_mean": config.use_combined_mean,
        "mean_source": mean_source,
        "mean_norm_sq": mean_norm_sq,
        "singular_values": [
            float(value) for value in singular_values[: config.top_k_svs].tolist()
        ],
        "explained_variance": [
            float(value) for value in explained_variance[: config.num_components].tolist()
        ],
        "powerlaw": powerlaw,
        "components": components,
        "plots": plot_paths,
    }
    payload["neutral"]["path"] = str(config.neutral.path)
    payload["trait"]["path"] = str(trait.path)

    write_json(json_path, payload)
    report.write(text_path)
    return {"skipped": False, "json_path": str(json_path), "text_path": str(text_path)}


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run SVD on trait-minus-neutral hidden-state matrix differences "
            "and inspect top singular vectors with a logit lens."
        )
    )
    parser.add_argument(
        "traits",
        nargs="*",
        help=(
            "Trait names whose matrix paths should be inferred. Comma-separated "
            "values are accepted."
        ),
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    parser.add_argument(
        "--matrix-format",
        choices=["csv", "npy"],
        default="csv",
        help="Format used by inferred matrix paths.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--prompt-mode",
        choices=["dataset", "random_tokens"],
        default=DEFAULT_PROMPT_MODE,
    )
    parser.add_argument("--neutral-name", default="neutral")
    parser.add_argument("--neutral-matrix", type=Path, default=None)
    parser.add_argument(
        "--trait-matrix",
        action="append",
        nargs=2,
        metavar=("NAME", "PATH"),
        default=None,
        help="Explicit trait matrix pair. Repeat as needed.",
    )
    parser.add_argument("--num-components", type=int, default=4)
    parser.add_argument("--top-k-svs", type=int, default=10)
    parser.add_argument("--top-k-tokens", type=int, default=15)
    parser.add_argument("--top-k-trait-tokens", type=int, default=12)
    parser.add_argument("--powerlaw-k", type=int, default=100)
    parser.add_argument("--block-hidden-dim", type=int, default=DEFAULT_BLOCK_HIDDEN_DIM)
    parser.add_argument("--expected-num-blocks", type=int, default=1)
    parser.add_argument("--use-combined-mean", action="store_true")
    parser.add_argument(
        "--no-orient-by-trait-tokens",
        dest="orient_by_trait_tokens",
        action="store_false",
        help="Use deterministic SVD sign instead of orienting toward trait tokens.",
    )
    parser.set_defaults(orient_by_trait_tokens=True)
    parser.add_argument(
        "--scan-vocab-for-exact-trait-tokens",
        action="store_true",
        help="Also scan decoded vocab tokens whose stripped text exactly matches the trait.",
    )
    parser.add_argument("--skip-logit-lens", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument(
        "--torch-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument("--device-map", default="auto")
    return parser


def build_config(args: argparse.Namespace) -> SVDLogitLensConfig:
    """Build validated config from CLI args."""
    if args.num_components <= 0:
        raise ValueError("--num-components must be positive.")
    if args.top_k_svs <= 0:
        raise ValueError("--top-k-svs must be positive.")
    if args.top_k_tokens <= 0:
        raise ValueError("--top-k-tokens must be positive.")
    if args.top_k_trait_tokens <= 0:
        raise ValueError("--top-k-trait-tokens must be positive.")
    if args.powerlaw_k <= 0:
        raise ValueError("--powerlaw-k must be positive.")
    if args.block_hidden_dim <= 0:
        raise ValueError("--block-hidden-dim must be positive.")
    if args.expected_num_blocks is not None and args.expected_num_blocks <= 0:
        raise ValueError("--expected-num-blocks must be positive.")

    neutral_matrix = args.neutral_matrix or infer_matrix_path(
        matrix_dir=args.matrix_dir,
        condition=args.neutral_name,
        prompt_mode=args.prompt_mode,
        no_system_prompt=True,
        matrix_format=args.matrix_format,
    )
    neutral = MatrixSpec(name=args.neutral_name, path=neutral_matrix)

    traits: list[MatrixSpec] = []
    if args.trait_matrix is not None:
        traits.extend(
            MatrixSpec(name=name, path=Path(path))
            for name, path in args.trait_matrix
        )

    inferred_traits = parse_csv_values(args.traits or None, default=[])
    traits.extend(
        MatrixSpec(
            name=trait,
            path=infer_matrix_path(
                matrix_dir=args.matrix_dir,
                condition=trait,
                prompt_mode=args.prompt_mode,
                matrix_format=args.matrix_format,
            ),
        )
        for trait in inferred_traits
    )
    if not traits:
        raise ValueError("Provide at least one positional trait or --trait-matrix pair.")

    return SVDLogitLensConfig(
        model_id=args.model_id,
        neutral=neutral,
        traits=traits,
        output_dir=args.output_dir,
        num_components=args.num_components,
        top_k_svs=args.top_k_svs,
        top_k_tokens=args.top_k_tokens,
        top_k_trait_tokens=args.top_k_trait_tokens,
        powerlaw_k=args.powerlaw_k,
        block_hidden_dim=args.block_hidden_dim,
        expected_num_blocks=args.expected_num_blocks,
        use_combined_mean=args.use_combined_mean,
        orient_by_trait_tokens=args.orient_by_trait_tokens,
        scan_vocab_for_exact_trait_tokens=args.scan_vocab_for_exact_trait_tokens,
        skip_logit_lens=args.skip_logit_lens,
        skip_existing=args.skip_existing,
        save_plots=args.save_plots,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
    )


def run(config: SVDLogitLensConfig) -> list[dict[str, Any]]:
    """Run all requested pair analyses."""
    missing_paths = [
        spec.path
        for spec in [config.neutral, *config.traits]
        if not spec.path.exists()
    ]
    if missing_paths:
        formatted = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing matrix file(s):\n{formatted}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    skipped_results: list[dict[str, Any]] = []
    pending_traits = []
    for trait in config.traits:
        json_path, text_path = report_paths_for_pair(config, trait)
        if config.skip_existing and report_cache_matches(config, trait):
            logger.info("Skipping existing SVD/logit-lens report {}", json_path)
            skipped_results.append(
                {
                    "skipped": True,
                    "json_path": str(json_path),
                    "text_path": str(text_path),
                }
            )
        else:
            pending_traits.append(trait)

    if not pending_traits:
        logger.info("All requested SVD/logit-lens reports already exist; skipping model load.")
        return skipped_results

    lens = None if config.skip_logit_lens else load_lens_context(config)

    results = list(skipped_results)
    for trait in pending_traits:
        results.append(analyze_pair(config=config, trait=trait, lens=lens))
    return results


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = build_config(args)
    run(config)


if __name__ == "__main__":
    main()
