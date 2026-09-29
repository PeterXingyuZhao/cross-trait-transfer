#!/usr/bin/env python3
"""Recheck token-SFT Metric 1 on larger shared and actual-FT prompt panels.

This is base-teacher inference, not student training. All teachers are scored
on identical prompt coordinates. Accepted-data weights describe a *common*
pooled prompt distribution, not teacher-specific coordinates or a distribution
conditioned on accepting the teacher's complete ten-number response.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterator

import numpy as np
from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
for import_root in (SCRIPT_DIR.parent, SCRIPT_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

VERSION = 1
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
NEUTRAL_LABEL = "neutral_no_system"
DEFAULT_PANELS = ("stream_10k", "accepted_all")


def digest(value: Any) -> str:
    """Hash finite canonical JSON."""
    content = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(content.encode()).hexdigest()


def file_digest(path: Path) -> str:
    """Hash one source or cache file in bounded memory."""
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    """Publish completed JSON without exposing a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def immutable_json(path: Path, value: Any) -> None:
    """Refuse to mix a changed experiment or runtime into existing scores."""
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Provenance mismatch: {path}; use a new output directory")
    else:
        atomic_json(path, value)


def safe_asset(bundle: Path, relative: str) -> Path:
    """Resolve only canonical paths inside the portable bundle."""
    relative_path = Path(relative)
    if (relative_path.is_absolute() or ".." in relative_path.parts
            or relative_path.as_posix() != relative or "\\" in relative):
        raise ValueError(f"Unsafe bundle path: {relative}")
    target = (bundle / relative_path).resolve()
    if not target.is_relative_to(bundle.resolve()):
        raise ValueError(f"Bundle asset escapes its root: {relative}")
    return target


def validate_bundle(bundle: Path) -> dict[str, Any]:
    """Check frozen inputs and code before any model download or GPU work."""
    manifest = json.loads((bundle / "manifest.json").read_text())
    if (manifest.get("schema_version") != VERSION
            or manifest.get("model_id") != MODEL_ID
            or manifest.get("model_revision") != MODEL_REVISION
            or not manifest.get("files") or not manifest.get("domains")):
        raise ValueError("Unsupported Metric 1 recheck bundle")
    for relative, expected in manifest["files"].items():
        path = safe_asset(bundle, relative)
        if not path.is_file() or file_digest(path) != expected:
            raise ValueError(f"Missing or changed bundle file: {relative}")
    for relative in manifest["domains"].values():
        if relative not in manifest["files"]:
            raise ValueError(f"Unhashed domain design: {relative}")
    return manifest


def validate_design(design: dict[str, Any]) -> None:
    """Validate shared coordinates and nonnegative common prompt weights."""
    from scripts.score_string_trait_number_distribution import slugify

    traits, prompts = design["traits"], design["prompts"]
    if (len(traits) != 30 or len(set(traits)) != 30
            or not all(isinstance(t, str) and t.strip() for t in traits)):
        raise ValueError("Exactly thirty distinct teacher traits are required")
    slugs = [slugify(label) for label in [NEUTRAL_LABEL, *traits]]
    if len(set(slugs)) != len(slugs):
        raise ValueError("Condition label slugs collide")
    if (not prompts or len(set(prompts)) != len(prompts)
            or not all(isinstance(p, str) and p for p in prompts)):
        raise ValueError("The scoring universe must contain distinct nonempty prompts")
    template = design["system_prompt_template"]
    if not isinstance(template, str) or "{trait}" not in template:
        raise ValueError("Missing teacher trait instruction template")
    for trait in traits:
        template.format(trait=trait)
    if not design["panels"]:
        raise ValueError("No prompt panels")
    for name, panel in design["panels"].items():
        weights = np.asarray(panel["weights"], dtype=np.float64)
        if (weights.shape != (len(prompts),) or not np.isfinite(weights).all()
                or np.any(weights < 0) or weights.sum() <= 0):
            raise ValueError(f"Invalid shared prompt weights: {name}")


def load_design(bundle: Path, manifest: dict[str, Any], domain: str) -> dict[str, Any]:
    """Load a checksummed domain definition and references."""
    if domain not in manifest["domains"]:
        raise ValueError(f"Domain is not in this bundle: {domain}")
    design = json.loads(safe_asset(bundle, manifest["domains"][domain]).read_text())
    validate_design(design)
    if design.get("domain") != domain:
        raise ValueError("Domain and design disagree")
    for relative in design["references"].values():
        if relative not in manifest["files"]:
            raise ValueError(f"Unhashed reference: {relative}")
    return design


def panel_indices(design: dict[str, Any], panels: tuple[str, ...]) -> list[int]:
    """Score the union once; reuse probabilities with each panel's own weights."""
    if not panels or len(set(panels)) != len(panels):
        raise ValueError("Select distinct, nonempty panel names")
    unknown = set(panels) - set(design["panels"])
    if unknown:
        raise ValueError(f"Unknown panels: {sorted(unknown)}")
    selected = np.any([np.asarray(design["panels"][p]["weights"]) > 0
                       for p in panels], axis=0)
    return np.flatnonzero(selected).tolist()


def split_chunks(indices: list[int], chunk_size: int) -> list[list[int]]:
    """Freeze chunk boundaries so restart and numerical audits are reproducible."""
    if chunk_size < 1:
        raise ValueError("Chunk size must be positive")
    return [indices[start:start + chunk_size]
            for start in range(0, len(indices), chunk_size)]


def condition_specs(design: dict[str, Any]) -> list[tuple[str, str | None]]:
    """Return the neutral model followed by the fixed thirty teacher conditions."""
    return [(NEUTRAL_LABEL, None), *[
        (trait, design["system_prompt_template"].format(trait=trait))
        for trait in design["traits"]
    ]]


@dataclass(frozen=True, kw_only=True)
class RunConfig:
    """Durable output and conservative A100 inference settings."""

    bundle_dir: Path
    output_dir: Path
    domain: str = "color"
    panels: tuple[str, ...] = DEFAULT_PANELS
    chunk_size: int = 256
    prompt_batch_size: int = 8
    branch_batch_size: int = 16
    device: str = "cuda"
    max_chunks: int | None = None


def make_protocol(config: RunConfig, manifest: dict[str, Any],
                  design: dict[str, Any]) -> dict[str, Any]:
    """Fingerprint science and batching, but not pilot length or output location."""
    if min(config.chunk_size, config.prompt_batch_size, config.branch_batch_size) < 1:
        raise ValueError("Chunk and batch sizes must be positive")
    if config.max_chunks is not None and config.max_chunks < 1:
        raise ValueError("Pilot chunk count must be positive")
    indices = panel_indices(design, config.panels)
    payload = {
        "version": VERSION, "bundle_fingerprint": digest(manifest),
        "design_fingerprint": digest(design), "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION, "domain": config.domain,
        "panels": list(config.panels), "scoring_indices": indices,
        "chunk_size": config.chunk_size,
        "prompt_batch_size": config.prompt_batch_size,
        "branch_batch_size": config.branch_batch_size,
        "dtype": "float32", "precision": "native_qwen_fp32_math_sdpa_v1",
        "event": "first complete 00--99 number at native assistant boundary",
        "normalization": "conditional over 100 exact full-vocabulary paths",
        "metric": "legacy compute_m4; neutral-weighted across-teacher-centered log signatures",
        "epsilon": 1e-12,
    }
    return {**payload, "fingerprint": digest(payload)}


def chunk_identity(protocol: dict[str, Any], runtime: dict[str, Any],
                   chunk: list[int], label: str, system: str | None) -> dict[str, Any]:
    """Identify a cache independently of its disk location."""
    return {"protocol_fingerprint": protocol["fingerprint"], "runtime": runtime,
            "indices": chunk, "label": label, "system_prompt": system}


def cache_path(root: Path, chunk_index: int, label: str) -> Path:
    """Use the established collision-checked label slug convention."""
    from scripts.score_string_trait_number_distribution import slugify

    return root / "chunks" / f"{chunk_index:05d}" / f"{slugify(label)}.npz"


def load_runtime(device_name: str) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Load only the pinned base teacher in strict FP32 (never a trained adapter)."""
    import torch
    from scripts.find_owl_output_distribution_entanglement import load_direct_model_and_tokenizer
    from scripts.prepare_token_sft_first_number_channel import runtime_identity
    from scripts.score_birthday_year_probabilities import resolve_device

    device = resolve_device(torch, device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Select an A100 GPU runtime in Colab before running this cell")
    model, tokenizer = load_direct_model_and_tokenizer(
        model_id=MODEL_ID, device=device, torch_dtype=torch.float32,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    runtime = runtime_identity(model, tokenizer)
    if (runtime["model_revision"] != MODEL_REVISION
            or runtime["weight_dtypes"] != ["torch.float32"]
            or runtime["precision_profile"] != "native_qwen_fp32_math_sdpa_v1"):
        raise ValueError("Loaded model does not match the pinned FP32 measurement protocol")
    runtime["device_type"] = device.type
    runtime["device_name"] = (torch.cuda.get_device_name(device)
                              if device.type == "cuda" else str(device))
    return model, tokenizer, device, runtime


def score_chunk(*, model: Any, tokenizer: Any, device: Any,
                prompt_texts: list[str], config: RunConfig,
                identity: dict[str, Any], tune_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Audit every chunk at actual batch shapes; halve batches only after OOM."""
    import torch
    from scripts.metric_batch_autotune import is_out_of_memory_error
    from scripts.prepare_token_sft_first_number_channel import (
        ChannelConfig, score_condition, verified_first_number_plans,
    )

    plans, token_audit = verified_first_number_plans(tokenizer, prompt_texts)
    channel_config = ChannelConfig(
        config_path=config.bundle_dir / "manifest.json", output_dir=tune_path.parent,
        prompt_count=len(prompt_texts), prompt_stream_count=len(prompt_texts),
        torch_dtype="float32", device=str(device), autotune_batch_pair=False,
        prompt_batch_size=config.prompt_batch_size,
        branch_batch_size=config.branch_batch_size,
    )
    oom_attempts = []
    while True:
        try:
            with torch.inference_mode():
                values, scoring = score_condition(
                    model=model, tokenizer=tokenizer, device=device,
                    prompt_texts=prompt_texts, plans=plans, config=channel_config,
                    cache_identity=identity, tune_path=tune_path,
                )
            return values, {"tokenization_audit": token_audit, "scoring": scoring,
                            "oom_retries": oom_attempts}
        except RuntimeError as error:
            if not is_out_of_memory_error(error):
                raise
            previous = (channel_config.prompt_batch_size, channel_config.branch_batch_size)
            if previous == (1, 1):
                raise
            oom_attempts.append(list(previous))
            channel_config = replace(
                channel_config, prompt_batch_size=max(1, previous[0] // 2),
                branch_batch_size=max(1, previous[1] // 2),
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            logger.warning("OOM at B={}, K={}; retrying chunk at B={}, K={}",
                           *previous, channel_config.prompt_batch_size,
                           channel_config.branch_batch_size)


def iter_scored_chunks(root: Path, design: dict[str, Any],
                       protocol: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield only complete hash-validated neutral-plus-thirty probability blocks."""
    from scripts.prepare_token_sft_first_number_channel import validate_distribution_cache

    runtime = json.loads((root / "runtime.json").read_text())
    for chunk_index, indices in enumerate(split_chunks(protocol["scoring_indices"],
                                                      protocol["chunk_size"])):
        prompts = [design["prompts"][i] for i in indices]
        values = []
        for label, system in condition_specs(design):
            path = cache_path(root, chunk_index, label)
            identity = chunk_identity(protocol, runtime, indices, label, system)
            if not path.is_file() or not path.with_suffix(".json").is_file():
                raise ValueError(f"Scoring is incomplete: {path}; resume --stage score")
            validate_distribution_cache(path, prompts=prompts, expected_identity=identity)
            with np.load(path, allow_pickle=False) as archive:
                values.append(archive["probabilities"])
        yield {"indices": indices, "probabilities": np.stack(values)}


def score(config: RunConfig, design: dict[str, Any],
          protocol: dict[str, Any], root: Path) -> dict[str, Any]:
    """Score all shared coordinates with restartable, checksummed chunk caches."""
    from scripts.prepare_token_sft_first_number_channel import (
        render_native_prompts, save_distribution_cache, validate_distribution_cache,
    )

    model, tokenizer, device, runtime = load_runtime(config.device)
    immutable_json(root / "runtime.json", runtime)
    chunks = split_chunks(protocol["scoring_indices"], config.chunk_size)
    conditions = condition_specs(design)
    if len({cache_path(root, 0, label) for label, _ in conditions}) != len(conditions):
        raise ValueError("Condition label slugs collide")
    selected_chunks = chunks[:config.max_chunks] if config.max_chunks else chunks
    start = time.monotonic()
    fresh_rows, fresh_seconds, reused = 0, 0.0, 0
    for chunk_index, indices in enumerate(selected_chunks):
        prompts = [design["prompts"][i] for i in indices]
        for label, system in conditions:
            path = cache_path(root, chunk_index, label)
            identity = chunk_identity(protocol, runtime, indices, label, system)
            if path.is_file() and path.with_suffix(".json").is_file():
                validate_distribution_cache(path, prompts=prompts, expected_identity=identity)
                reused += 1
                continue
            if path.exists():
                logger.warning("Recomputing uncommitted cache without a checksum sidecar: {}", path)
            rendered = render_native_prompts(tokenizer, prompts, system)
            started = time.monotonic()
            values, audit = score_chunk(
                model=model, tokenizer=tokenizer, device=device, prompt_texts=rendered,
                config=config, identity=identity,
                tune_path=root / "batch_audits" / f"{chunk_index:05d}_{label}.json",
            )
            seconds = time.monotonic() - started
            save_distribution_cache(
                path, prompts=prompts, prompt_texts=rendered, logprobs=values,
                metadata={"identity": identity, **audit, "scoring_seconds": seconds,
                          "rendered_prompts_sha256": digest(rendered)},
            )
            validate_distribution_cache(path, prompts=prompts, expected_identity=identity)
            fresh_rows += len(prompts)
            fresh_seconds += seconds
            logger.info("Chunk {}/{}, {}: {} prompts in {:.1f}s", chunk_index + 1,
                        len(chunks), label, len(prompts), seconds)
        elapsed = time.monotonic() - start
        estimated = (fresh_seconds / fresh_rows * len(protocol["scoring_indices"])
                     * len(conditions) / 3600) if fresh_rows else None
        finished_selected = chunk_index + 1 == len(selected_chunks)
        status = ("pilot_complete" if len(selected_chunks) < len(chunks)
                  else "complete") if finished_selected else "running"
        progress = {
            "status": status,
            "completed_chunks_this_invocation": chunk_index + 1,
            "total_chunks": len(chunks), "reused_condition_chunks": reused,
            "fresh_condition_prompt_rows": fresh_rows, "elapsed_seconds": elapsed,
            "estimated_full_scoring_hours_excluding_model_load": estimated,
            "estimate_note": "Extrapolation from measured fresh chunks, not a runtime guarantee",
        }
        atomic_json(root / "progress.json", progress)
        if estimated is not None:
            logger.info("Measured full-run scoring estimate: {:.2f} hours (model loading excluded)", estimated)
    return progress


def run(config: RunConfig, stage: str = "all") -> dict[str, Any]:
    """Validate/plan, run inference, and optionally analyze all completed panels."""
    if stage not in {"plan", "score", "summarize", "all"}:
        raise ValueError(f"Unknown stage: {stage}")
    manifest = validate_bundle(config.bundle_dir)
    design = load_design(config.bundle_dir, manifest, config.domain)
    protocol = make_protocol(config, manifest, design)
    chunks = split_chunks(protocol["scoring_indices"], config.chunk_size)
    root = config.output_dir / config.domain
    plan = {
        "domain": config.domain, "panels": list(config.panels),
        "unique_prompts_to_score": len(protocol["scoring_indices"]),
        "teacher_conditions_including_neutral": 31, "candidate_numbers": 100,
        "chunks": len(chunks), "chunk_size": config.chunk_size,
        "output_dir": str(root), "precision": "FP32, math SDPA, TF32 disabled",
        "panel_sizes": {name: {
            "prompt_observations": float(np.sum(panel["weights"])),
            "unique_prompts": int(np.count_nonzero(panel["weights"])),
            "effective_prompt_count": float(np.sum(panel["weights"]) ** 2
                                            / np.sum(np.square(panel["weights"]))),
        } for name, panel in design["panels"].items() if name in config.panels},
    }
    logger.info("Metric 1 plan: {}", json.dumps(plan, indent=2))
    if stage == "plan":
        return plan
    from scripts.run_number_sequence_trait_cell import exclusive_lock

    with exclusive_lock(root / ".metric1_recheck.lock"):
        immutable_json(root / "protocol.json", protocol)
        atomic_json(root / "plan.json", plan)
        if stage in {"score", "all"}:
            progress = score(config, design, protocol, root)
            if stage == "score" or progress["status"] != "complete":
                return progress
        from scripts.summarize_token_sft_metric1_recheck import summarize

        return summarize(config.bundle_dir, root, config.domain, design,
                         iter_scored_chunks(root, design, protocol))


def main() -> None:
    """Run one domain; default to the color-network comparison discussed in the paper."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--domain", choices=("color", "city", "animal"), default="color")
    parser.add_argument("--panels", nargs="+", default=list(DEFAULT_PANELS))
    parser.add_argument("--stage", choices=("plan", "score", "summarize", "all"), default="all")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--prompt-batch-size", type=int, default=8)
    parser.add_argument("--branch-batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-chunks", type=int,
                        help="Pilot only: score the first N chunks; rerun without this flag to resume")
    arguments = vars(parser.parse_args())
    stage = arguments.pop("stage")
    arguments["panels"] = tuple(arguments["panels"])
    result = run(RunConfig(**arguments), stage)
    logger.success("Metric 1 recheck {}: {}", stage, result.get("status", "planned"))


if __name__ == "__main__":
    main()
