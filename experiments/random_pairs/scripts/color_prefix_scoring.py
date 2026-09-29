#!/usr/bin/env python3
"""Score final color-prefix adapters and audited historical final-score imports.

Primary recovery subtracts the trained unnamed-color control at the same arity.
Historical scores retain their original identities and protocol fingerprints.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import red_hue_token_sft_scoring as common  # noqa: E402
from scripts.color_prefix_protocol import (  # noqa: E402
    CONDITIONS, NEW_CONDITIONS, IMPORTED_CONDITIONS, RELEASE, RECOVERY_QUESTIONS, COLORS, CHAIN_CONDITIONS,
)

strict = common.strict
CAMPAIGN = RELEASE
FINAL_STEP = 471
SCORE_SHAPE = (len(RECOVERY_QUESTIONS), len(COLORS))
RED_INDEX = COLORS.index("red")
PREFILL = None
REUSE_ASSET_DIR = "historical_scores"
REUSE_MANIFEST_SHA256 = "6f5faab7da214cb2c1557206984033965f8b44b90c909aa064bdb6df27268173"
BASE_ATOL = 2e-4
CONTROLS = {1: "control_single", 2: "control_pair", 3: "control_three", 4: "control_four"}


def validate_readout(protocol: dict[str, Any]) -> None:
    """Require the exact frozen color boundary, order, and final-only schedule."""
    if (protocol.get("release") != RELEASE or protocol.get("conditions") != list(CONDITIONS)
            or protocol.get("model_id") != strict.MODEL_ID
            or protocol.get("revision") != strict.MODEL_REVISION):
        raise ValueError("Readout requires the frozen color-prefix release and pinned model")
    if (protocol.get("colors") != list(COLORS)
            or protocol.get("color_questions") != list(RECOVERY_QUESTIONS)
            or protocol.get("targets") != list(COLORS)
            or protocol.get("target_texts") != list(COLORS)
            or protocol.get("recovery_questions") != list(RECOVERY_QUESTIONS)):
        raise ValueError("Color recovery requires 30 exact bare answers and the frozen ten questions")
    if (protocol.get("recovery_system_prompt", "missing") is not None
            or protocol.get("recovery_assistant_prefill", "missing") is not None):
        raise ValueError("Historical-compatible color readout has no added system instruction or prefill")
    if protocol.get("checkpoint_steps") != [FINAL_STEP]:
        raise ValueError("Only the final step-471 adapter is evaluated")


def recovery_prompts(tokenizer: Any, protocol: dict[str, Any]) -> list[str]:
    """Preserve the native assistant boundary used by historical color scoring."""
    validate_readout(protocol)
    return strict.native_prompts(tokenizer, protocol["color_questions"], None)


def build_manifest(protocol: dict[str, Any], training_dir: Path,
                   condition: str) -> dict[str, Any]:
    """Verify the sealed training identity and final adapter before model load."""
    from scripts import color_prefix_training as training

    validate_readout(protocol)
    if condition not in NEW_CONDITIONS:
        raise ValueError(f"Condition {condition} is not a new training condition")
    directory = Path(training_dir) / condition
    run = json.loads((directory / "run.json").read_text())
    spec = {key: value for key, value in run.items() if key != "fingerprint"}
    fingerprint = hashlib.sha256(training.original._json_bytes(spec)).hexdigest()
    if (run.get("fingerprint") != fingerprint or run.get("schema") != training.SCHEMA
            or run.get("condition") != condition or run.get("trait") != condition):
        raise ValueError("Training run identity does not match the requested condition")
    source = run.get("source", {})
    provenance = source.get("generation_provenance", {})
    if (source.get("model_id") != strict.MODEL_ID
            or source.get("model_revision") != strict.MODEL_REVISION
            or provenance.get("condition") != condition):
        raise ValueError("Training source names a different model or condition")
    if provenance.get("protocol_sha256") != training.generation.original.digest(protocol):
        raise ValueError("Generation and evaluation protocols differ")
    condition_spec = protocol["condition_specs"][condition]
    for protocol_key, source_key in (("traits", "traits"), ("arity", "arity"),
                                     ("system_prompt", "generation_system_prompt")):
        if provenance.get(source_key) != condition_spec.get(protocol_key):
            raise ValueError(f"Generation provenance differs on {source_key}")
    verified = training.completed_manifest(directory, fingerprint=fingerprint, condition=condition)
    recorded = json.loads((directory / "manifest.json").read_text())
    for key in ("status", "optimizer_step", "fingerprint", "condition"):
        if recorded.get(key) != verified.get(key):
            raise ValueError(f"Completed training manifest disagrees on {key}")
    if verified.get("status") != "complete" or verified.get("optimizer_step") != FINAL_STEP:
        raise ValueError("Only the final 471-update student is eligible for evaluation")
    adapter = directory / "adapters" / f"checkpoint-{FINAL_STEP:06d}"
    metadata = training.original.verify_checkpoint(adapter, fingerprint=fingerprint)
    if metadata.get("optimizer_step") != FINAL_STEP:
        raise ValueError("Adapter is not the final checkpoint")
    return {
        "campaign": CAMPAIGN, "origin": "new_training", "condition": condition, "conditions": [condition], "step": FINAL_STEP,
        "traits": condition_spec["traits"], "model_id": strict.MODEL_ID,
        "revision": strict.MODEL_REVISION, "adapter_path": str(adapter.resolve()),
        "adapter_files": strict.adapter_identity(adapter),
        "training_fingerprint": fingerprint, "generation_provenance": provenance,
        "protocol_sha256": strict.stable_hash(protocol),
    }


def _valid_logs(values: Any) -> np.ndarray:
    """Require the complete ten-question by 30-answer raw log-probability matrix."""
    array = np.asarray(values, dtype=np.float64)
    if array.shape != SCORE_SHAPE or not np.isfinite(array).all() or np.any(array > 1e-7):
        raise ValueError("Raw log probabilities must be finite, nonpositive, and match the frozen 10 by 30 shape")
    return array


def summarize_scores(protocol: dict[str, Any], manifest: dict[str, Any],
                     base: np.ndarray, student: np.ndarray) -> dict[str, Any]:
    """Preserve raw final probabilities; apply trained controls only at report time."""
    validate_readout(protocol)
    condition = manifest["condition"]
    if condition not in CONDITIONS:
        raise ValueError("Unknown color-prefix condition")
    origin = manifest.get("origin")
    if origin != ("historical_import" if condition in IMPORTED_CONDITIONS else "new_training"):
        raise ValueError("Condition has an incompatible new/imported origin")
    base, student = _valid_logs(base), _valid_logs(student)
    gain = student - base
    return {
        "campaign": CAMPAIGN, "origin": origin, "condition": condition, "conditions": [condition],
        "step": FINAL_STEP, "scoring_complete": True,
        "prefix_size": protocol["condition_specs"][condition]["arity"],
        "traits": protocol["condition_specs"][condition]["traits"],
        "targets": protocol["targets"], "target_texts": protocol["target_texts"],
        "questions": protocol["recovery_questions"], "assistant_prefill": None,
        "recovery_system_prompt": None, "protocol_sha256": strict.stable_hash(protocol),
        "manifest": manifest,
        "primary_definition": "Report R = mean_q(log_p_final - log_p_arity_matched_trained_control), nats",
        "raw_gain_definition": "mean_q(log_p_final - log_p_untrained_base), nats",
        "base_question_logprobs": base.tolist(), "student_question_logprobs": student.tolist(),
        "mean_raw_logprob": student.mean(axis=0).tolist(),
        "base_mean_raw_logprob": base.mean(axis=0).tolist(),
        "question_logprob_gain": gain.tolist(), "mean_logprob_gain": gain.mean(axis=0).tolist(),
        "mean_raw_probability": np.exp(student).mean(axis=0).tolist(),
        "base_mean_raw_probability": np.exp(base).mean(axis=0).tolist(),
    }


def score_with_adapter(*, model: Any, adapter_class: Any, configure_dtype: Any,
                       adapter_path: str, score_function: Any) -> tuple[Any, Any]:
    """Load one adapter without merging, and unload it even if scoring fails."""
    active = adapter_class.from_pretrained(model, adapter_path, is_trainable=False)
    original_error: BaseException | None = None
    try:
        configure_dtype(active, "float32")
        active.eval().requires_grad_(False)
        result = score_function(active)
    except BaseException as error:
        original_error = error
        raise
    finally:
        try:
            model = active.unload()
            model.eval()
        except BaseException as error:
            if original_error is None:
                raise
            logger.error("Adapter cleanup also failed: {}", error)
    return model, result


def _score_state(*, model: Any, tokenizer: Any, prompts: list[str],
                 answers: list[str], config: Any, identity: dict[str, Any],
                 path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Reuse only matching checksummed scores, otherwise run strict FP32 scoring."""
    cached = strict.read_score_cache(path, identity, SCORE_SHAPE)
    if cached is not None:
        logger.info("Reusing verified recovery scores {}", path)
        return cached
    values, audit = strict.score_strict_matrix(
        model=model, tokenizer=tokenizer, prompts=prompts, answers=answers,
        config=config, state_key=identity["state_key"],
    )
    common._write_score_cache(path, identity, values, audit)
    return values, audit


def validate_completed_evaluation(protocol: dict[str, Any], training_dir: Path,
                                  condition: str, output_dir: Path, *,
                                  manifest: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Validate portable base/final caches and the summary before skipping inference."""
    path = Path(output_dir) / "summary.json"
    if not path.is_file():
        return None
    saved = json.loads(path.read_text())
    if saved.get("origin") == "historical_import":
        return validate_historical_evaluation(protocol, condition, Path(output_dir))
    manifest = manifest or build_manifest(protocol, training_dir, condition)
    if saved.get("scoring_complete") is not True:
        return None
    if saved.get("manifest") != manifest:
        # Adapter location may change when a verified archive is restored.
        expected = {key: value for key, value in manifest.items() if key != "adapter_path"}
        actual = {key: value for key, value in saved.get("manifest", {}).items() if key != "adapter_path"}
        if actual != expected:
            raise ValueError("Completed scoring manifest differs from final training")
    arrays = {}
    for name in ("base", "student"):
        identity = saved.get("score_identities", {}).get(name)
        if not isinstance(identity, dict):
            raise ValueError("Missing score identity")
        if (identity.get("protocol_sha256") != strict.stable_hash(protocol)
                or identity.get("answers") != protocol["target_texts"]
                or identity.get("questions") != protocol["recovery_questions"]
                or identity.get("assistant_prefill") != PREFILL):
            raise ValueError("Completed scoring readout differs from protocol")
        runtime = identity.get("runtime", {})
        precision = runtime.get("precision", {})
        if (runtime.get("campaign") != CAMPAIGN or runtime.get("model_id") != strict.MODEL_ID
                or runtime.get("revision") != strict.MODEL_REVISION
                or runtime.get("recovery_system_prompt", "missing") is not None
                or runtime.get("assistant_prefill") != PREFILL
                or precision.get("native_weight_dtypes") != ["torch.float32"]
                or precision.get("native_attention_policy") != "explicit_fp32_math_sdpa"
                or precision.get("native_allow_tf32") is not False
                or precision.get("native_float32_matmul_precision") != "highest"):
            raise ValueError("Completed scores do not have the required strict FP32 provenance")
        if (identity.get("state_key") != ("base" if name == "base" else condition)
                or (name == "student" and identity.get("training_fingerprint") != manifest["training_fingerprint"])):
            raise ValueError("Completed score identity has a different training state")
        adapter_files = None if name == "base" else manifest["adapter_files"]
        if identity.get("adapter_files") != adapter_files:
            raise ValueError("Completed scores name different adapter weights")
        cache = Path(output_dir) / f"{name}.npz"
        if strict.file_hash(cache) != saved.get("score_file_sha256", {}).get(name):
            raise ValueError("Completed score-file checksum differs")
        cached = strict.read_score_cache(cache, identity, SCORE_SHAPE)
        if cached is None:
            raise ValueError("Completed raw score cache is invalid")
        arrays[name] = cached[0]
    reconstructed = summarize_scores(protocol, saved["manifest"], arrays["base"], arrays["student"])
    if any(saved.get(key) != value for key, value in reconstructed.items()):
        raise ValueError("Published summary differs from preserved raw scores")
    return saved


def score_campaign(protocol: dict[str, Any], training_dir: Path, output_dir: Path,
                   condition: str, *, device: str = "cuda",
                   prompt_batch_size: int = 5) -> dict[str, Any]:
    """Score only the untrained base and final adapter; share the base across conditions."""
    manifest = build_manifest(protocol, training_dir, condition)
    output_dir = Path(output_dir)
    completed = validate_completed_evaluation(protocol, training_dir, condition, output_dir, manifest=manifest)
    if completed is not None:
        return completed
    config = strict.ScoreConfig(device=device, prompt_chunk_size=10,
                                prompt_batch_size=prompt_batch_size, branch_batch_size=8)
    config.validate()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from train_teacher_logits_soft_distillation import configure_explicit_training_dtype

    tokenizer = AutoTokenizer.from_pretrained(strict.MODEL_ID, revision=strict.MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        strict.MODEL_ID, revision=strict.MODEL_REVISION, torch_dtype=torch.float32,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    ).to(device).eval()
    configure_explicit_training_dtype(model, "float32")
    model.requires_grad_(False)
    try:
        runtime = strict._runtime_identity(model, tokenizer, config)
        runtime.pop("number_semantics", None)
        runtime.pop("color_semantics", None)
        runtime.update(campaign=CAMPAIGN, assistant_prefill=PREFILL, recovery_system_prompt=None,
                       semantics="raw exact full continuation; no EOS, catalog, or length normalization")
        runtime["code_sha256"][Path(__file__).name] = strict.file_hash(Path(__file__))
        prompts = recovery_prompts(tokenizer, protocol)
        identity = {
            "runtime": runtime, "protocol_sha256": strict.stable_hash(protocol),
            "prompts": prompts, "answers": protocol["target_texts"],
            "questions": protocol["recovery_questions"], "assistant_prefill": PREFILL,
        }
        base_identity = {**identity, "state_key": "base", "adapter_files": None}
        student_identity = {**identity, "state_key": condition, "adapter_files": manifest["adapter_files"],
                            "training_fingerprint": manifest["training_fingerprint"]}
        common_args = {"tokenizer": tokenizer, "prompts": prompts, "answers": protocol["target_texts"],
                       "config": config}
        shared_base = output_dir.parent / "_base_cache" / f"{strict.stable_hash(base_identity)}.npz"
        base, base_audit = _score_state(model=model, identity=base_identity, path=shared_base, **common_args)
        check_base_against_imports(protocol, output_dir.parent, base, base_identity)
        # Each condition contains its own small verified score copy, so archives are self-contained.
        common._write_score_cache(output_dir / "base.npz", base_identity, base, base_audit)
        model, (student, student_audit) = score_with_adapter(
            model=model, adapter_class=PeftModel, configure_dtype=configure_explicit_training_dtype,
            adapter_path=manifest["adapter_path"],
            score_function=lambda active: _score_state(model=active, identity=student_identity,
                                                       path=output_dir / "student.npz", **common_args),
        )
        summary = summarize_scores(protocol, manifest, base, student)
        summary.update(score_identities={"base": base_identity, "student": student_identity},
                       score_file_sha256={name: strict.file_hash(output_dir / f"{name}.npz")
                                          for name in ("base", "student")},
                       audits={"base": base_audit, "student": student_audit})
        common._write_json(output_dir / "summary.json", summary)
        logger.success("Scored final {}: red gain {:.4f} nats", condition, summary["mean_logprob_gain"][RED_INDEX])
        return summary
    finally:
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def _validate_precision(runtime: dict[str, Any]) -> None:
    """Require the original pinned model and audited full-precision policy."""
    precision = runtime.get("precision", {})
    if (runtime.get("model_id") != strict.MODEL_ID
            or runtime.get("revision") != strict.MODEL_REVISION
            or runtime.get("recovery_system_prompt", "missing") is not None
            or runtime.get("assistant_prefill", "missing") is not None
            or precision.get("native_weight_dtypes") != ["torch.float32"]
            or precision.get("native_attention_policy") != "explicit_fp32_math_sdpa"
            or precision.get("native_allow_tf32") is not False
            or precision.get("native_float32_matmul_precision") != "highest"):
        raise ValueError("Historical readout is not the pinned strict FP32 color readout")


def _load_historical(protocol: dict[str, Any], asset_dir: Path,
                     condition: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    """Validate immutable historical score assets without relabeling their source."""
    validate_readout(protocol)
    if condition not in IMPORTED_CONDITIONS:
        raise ValueError("Only the three specified historical conditions may be imported")
    index_path = Path(asset_dir) / "manifest.json"
    if strict.file_hash(index_path) != REUSE_MANIFEST_SHA256:
        raise ValueError("Historical bundle manifest checksum differs from the audited source")
    index = json.loads(index_path.read_text())
    records = index["conditions"][condition]
    directory = Path(asset_dir) / condition
    for filename, record in records["files"].items():
        if Path(filename).name != filename or strict.file_hash(directory / filename) != record["sha256"]:
            raise ValueError(f"Historical source checksum mismatch: {condition}/{filename}")
    source = json.loads((directory / "source.json").read_text())
    if (source.get("condition") != condition or source.get("final_step") != FINAL_STEP
            or source.get("source_release") != "qwen25_color_pair_token_sft_v1"
            or source.get("source_archive_sha256") != records["archive_sha256"]
            or source.get("source_protocol_sha256") != index["source_protocol_sha256"]
            or source.get("colors") != protocol["colors"]
            or source.get("color_questions") != protocol["color_questions"]):
        raise ValueError("Historical source identity or readout differs")
    runtime = source["runtime"]
    _validate_precision(runtime)
    if (runtime.get("condition") != condition
            or runtime.get("training_fingerprint") != source["training_fingerprint"]
            or runtime.get("protocol_sha256") != source["source_protocol_sha256"]):
        raise ValueError("Historical runtime does not match its original training identity")
    provenance = source["generation_provenance"]
    if (provenance.get("condition") != condition or provenance.get("filtered_count") != 10000
            or provenance.get("system_prompt_in_training") is not False
            or provenance.get("model_id") != protocol["model_id"]
            or provenance.get("model_revision") != protocol["revision"]):
        raise ValueError("Historical generation recipe differs from the approved pilot")
    for current, old in (("traits", "traits"), ("arity", "arity"),
                         ("system_prompt", "generation_system_prompt")):
        if protocol["condition_specs"][condition][current] != provenance[old]:
            raise ValueError(f"Historical teacher instruction differs on {current}")
    arrays, identities = {}, {}
    for name, step in (("base", 0), ("student", FINAL_STEP)):
        identity = json.loads((directory / f"{name}_identity.json").read_text())
        identities[name] = identity
        state_key = "base" if name == "base" else f"student_{condition}_{FINAL_STEP:06d}"
        fingerprint = strict.stable_hash(identity)
        stage = next(row for row in source["source_stages"] if row["step"] == step)
        if (identity.get("runtime") != runtime or identity.get("state_key") != state_key
                or identity.get("color_answers") != protocol["target_texts"]
                or identity.get("color_questions") != protocol["color_questions"]
                or stage.get("fingerprint") != fingerprint
                or strict.file_hash(directory / f"{name}.npz") != stage["score_file_sha256"]
                or (name == "base" and identity.get("adapter_files") is not None)
                or (name == "student" and not identity.get("adapter_files"))):
            raise ValueError("Historical score identity or adapter provenance differs")
        with np.load(directory / f"{name}.npz", allow_pickle=False) as payload:
            if (payload["complete"].shape != () or not bool(payload["complete"].item())
                    or str(payload["fingerprint"].item()) != fingerprint
                    or payload["colors"].tolist() != protocol["colors"]
                    or payload["questions"].tolist() != protocol["color_questions"]):
                raise ValueError("Historical raw score cache axes or fingerprint differ")
            arrays[name] = _valid_logs(payload["color_logprobs"])
    manifest = {
        "campaign": CAMPAIGN, "origin": "historical_import", "condition": condition,
        "conditions": [condition], "step": FINAL_STEP,
        "model_id": protocol["model_id"], "revision": protocol["revision"],
        "traits": protocol["condition_specs"][condition]["traits"],
        "protocol_sha256": strict.stable_hash(protocol),
        "historical_asset_manifest_sha256": REUSE_MANIFEST_SHA256,
        "source_release": source["source_release"],
        "source_archive_sha256": source["source_archive_sha256"],
        "source_protocol_sha256": source["source_protocol_sha256"],
        "source_generation_protocol_sha256": provenance["protocol_sha256"],
        "source_summary_sha256": source["source_summary_sha256"],
        "training_fingerprint": source["training_fingerprint"],
        "generation_provenance": provenance,
        "source_score_file_sha256": {name: records["files"][f"{name}.npz"]["sha256"]
                                     for name in ("base", "student")},
        "source_score_identities": identities,
    }
    return manifest, source, arrays


def validate_historical_evaluation(protocol: dict[str, Any], condition: str,
                                   output_dir: Path) -> dict[str, Any]:
    """Reconstruct a portable imported summary from immutable copied source files."""
    manifest, _, arrays = _load_historical(protocol, Path(output_dir) / "_historical", condition)
    expected = summarize_scores(protocol, manifest, arrays["base"], arrays["student"])
    saved = json.loads((Path(output_dir) / "summary.json").read_text())
    if saved != expected:
        raise ValueError("Imported summary differs from audited historical raw scores")
    return saved


def import_historical(protocol: dict[str, Any], bundle: Path, output_dir: Path,
                      condition: str) -> dict[str, Any]:
    """Import the audited base/final arrays and retain their original identities."""
    assets = Path(bundle) / "assets" / REUSE_ASSET_DIR
    manifest, _, arrays = _load_historical(protocol, assets, condition)
    output_dir = Path(output_dir)
    if (output_dir / "summary.json").is_file():
        return validate_historical_evaluation(protocol, condition, output_dir)
    destination = output_dir / "_historical"
    common._atomic_bytes(destination / "manifest.json", (assets / "manifest.json").read_bytes())
    for source in (assets / condition).iterdir():
        if source.is_file():
            common._atomic_bytes(destination / condition / source.name, source.read_bytes())
    summary = summarize_scores(protocol, manifest, arrays["base"], arrays["student"])
    common._write_json(output_dir / "summary.json", summary)
    validate_historical_evaluation(protocol, condition, output_dir)
    logger.info("Imported audited historical final {} scores; no model fit or inference", condition)
    return summary


def check_base_against_imports(protocol: dict[str, Any], evaluation_dir: Path,
                              base: np.ndarray, identity: dict[str, Any]) -> None:
    """Reject a changed scoring boundary or material numerical baseline drift."""
    base = _valid_logs(base)
    for condition in IMPORTED_CONDITIONS:
        path = Path(evaluation_dir) / condition
        if not (path / "summary.json").is_file():
            continue
        historical = validate_historical_evaluation(protocol, condition, path)
        old = historical["manifest"]["source_score_identities"]["base"]
        if (old["color_prompts"] != identity["prompts"]
                or old["color_answers"] != identity["answers"]
                or old["runtime"]["tokenizer_sha256"] != identity["runtime"]["tokenizer_sha256"]):
            raise ValueError("New base uses a different tokenizer or exact recovery boundary")
        _validate_precision(identity["runtime"])
        difference = float(np.max(np.abs(base - np.asarray(historical["base_question_logprobs"]))))
        if difference > BASE_ATOL:
            raise ValueError(f"New base differs from historical scores by {difference:.6g} nats; "
                             f"maximum permitted {BASE_ATOL}. Re-audit before combining conditions.")
        logger.info("Verified base against historical {}: maximum difference {:.6g} nats", condition, difference)


def _ranks(values: np.ndarray) -> list[int]:
    """Descending competition ranks: equal values share a rank."""
    return [1 + int(np.count_nonzero(values > value)) for value in values]


def report_rows(protocol: dict[str, Any], summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute raw and arity-matched control gains without imputing missing fits."""
    validate_readout(protocol)
    by_condition: dict[str, dict[str, Any]] = {}
    for summary in summaries:
        condition = summary.get("condition")
        if (condition not in CONDITIONS or condition in by_condition
                or summary.get("campaign") != CAMPAIGN or summary.get("scoring_complete") is not True
                or summary.get("protocol_sha256") != strict.stable_hash(protocol)):
            raise ValueError("Report has a duplicate, incomplete, or incompatible condition")
        expected = summarize_scores(protocol, summary["manifest"], summary["base_question_logprobs"],
                                    summary["student_question_logprobs"])
        if any(summary.get(key) != value for key, value in expected.items()):
            raise ValueError("Reported summary does not reproduce the raw question scores")
        by_condition[condition] = summary
    if by_condition:
        reference = np.asarray(next(iter(by_condition.values()))["base_question_logprobs"])
        for value in by_condition.values():
            if np.max(np.abs(reference - np.asarray(value["base_question_logprobs"]))) > BASE_ATOL:
                raise ValueError("Base scores differ across conditions beyond the fixed FP32 tolerance")
    rows = []
    for condition in CONDITIONS:
        if condition not in by_condition:
            continue
        value = by_condition[condition]
        logs = np.asarray(value["student_question_logprobs"])
        arity = value["prefix_size"]
        control = CONTROLS[arity]
        recovery = None
        recovery_question = None
        base_adjusted = None
        if not condition.startswith("control_") and control in by_condition:
            control_summary = by_condition[control]
            recovery_question = logs - np.asarray(control_summary["student_question_logprobs"])
            recovery = recovery_question.mean(axis=0)
            base_adjusted = np.asarray(value["mean_logprob_gain"]) - np.asarray(control_summary["mean_logprob_gain"])
        raw_gains = np.asarray(value["mean_logprob_gain"])
        ranks = None if recovery is None else _ranks(recovery)
        mean_logs = np.asarray(value["mean_raw_logprob"])
        mean_probabilities = np.asarray(value["mean_raw_probability"])
        red_only = by_condition.get("red")
        rows.append({
            "condition": condition, "origin": value["origin"], "prefix_size": arity,
            "traits": value["traits"], "matched_control": control,
            "red_recovery_R_nats": None if recovery is None else float(recovery[RED_INDEX]),
            "red_recovery_rank": None if ranks is None else ranks[RED_INDEX],
            "red_base_relative_gain_nats": float(raw_gains[RED_INDEX]),
            "red_mean_raw_logprob": float(mean_logs[RED_INDEX]),
            "red_mean_raw_probability": float(mean_probabilities[RED_INDEX]),
            "red_raw_logprob_difference_from_red_only_nats": None if red_only is None else
                float(mean_logs[RED_INDEX] - red_only["mean_raw_logprob"][RED_INDEX]),
            "red_gain_difference_from_red_only_nats": None if red_only is None else
                float(raw_gains[RED_INDEX] - red_only["mean_logprob_gain"][RED_INDEX]),
            "mean_control_relative_gain_nats": None if recovery is None else recovery.tolist(),
            "question_control_relative_gain_nats": None if recovery_question is None else recovery_question.tolist(),
            "mean_base_adjusted_control_relative_gain_nats": None if base_adjusted is None else base_adjusted.tolist(),
            "control_relative_gain_ranks": ranks, "base_relative_gain_ranks": _ranks(raw_gains),
            "raw_logprob_ranks": _ranks(mean_logs), "raw_probability_ranks": _ranks(mean_probabilities),
            "mean_logprob_gain": value["mean_logprob_gain"],
            "mean_raw_logprob": value["mean_raw_logprob"],
            "mean_raw_probability": value["mean_raw_probability"],
        })
    return rows


def chain_contrasts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Separate each incremental raw change from the matched-control R change."""
    by_condition = {row["condition"]: row for row in rows}
    result = {}
    for label, conditions in CHAIN_CONDITIONS.items():
        chain = [by_condition.get(condition) for condition in conditions]
        contrasts = []
        for previous, current in zip(chain, chain[1:]):
            if previous is None or current is None:
                continue
            r_before, r_after = previous["red_recovery_R_nats"], current["red_recovery_R_nats"]
            contrasts.append({
                "from_condition": previous["condition"], "to_condition": current["condition"],
                "added_color": current["traits"][-1], "prefix_size": current["prefix_size"],
                "red_delta_R_nats": None if r_before is None or r_after is None else r_after - r_before,
                "red_delta_raw_logprob_nats": current["red_mean_raw_logprob"] - previous["red_mean_raw_logprob"],
                "red_delta_raw_probability": current["red_mean_raw_probability"] - previous["red_mean_raw_probability"],
                "red_delta_base_relative_gain_nats": current["red_base_relative_gain_nats"] - previous["red_base_relative_gain_nats"],
            })
        result[label] = {"conditions": list(conditions), "incremental_contrasts": contrasts,
                         "fourth_color_contrast": next((row for row in contrasts if row["prefix_size"] == 4), None)}
    return result


def build_report(protocol, evaluation_dir, output_dir):
    """Summarize both targets against one shared trained pair control."""
    from scripts.random_color_pair_report import build_report as pair_report
    return pair_report(protocol, evaluation_dir, output_dir)


def main() -> None:
    """Evaluate a final student, import historical caches, or assemble a report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path)
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt-batch-size", type=int, default=5)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--evaluation-dir", type=Path)
    parser.add_argument("--import-historical", action="store_true")
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if args.report:
        if args.evaluation_dir is None:
            parser.error("--report requires --evaluation-dir")
        build_report(protocol, args.evaluation_dir, args.output_dir)
    elif args.import_historical:
        if args.bundle is None or args.condition is None:
            parser.error("Historical import requires --bundle and --condition")
        import_historical(protocol, args.bundle, args.output_dir, args.condition)
    else:
        if args.training_dir is None or args.condition is None:
            parser.error("Scoring requires --training-dir and --condition")
        score_campaign(protocol, args.training_dir, args.output_dir, args.condition,
                       device=args.device, prompt_batch_size=args.prompt_batch_size)


if __name__ == "__main__":
    main()
