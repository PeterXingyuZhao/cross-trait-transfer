#!/usr/bin/env python3
"""Resumable, matched 30-color all-layer component steering on one GPU.

This is frozen-base-model inference using the archived distribution-setting
vectors. It neither trains students nor re-extracts token-SFT feature vectors.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
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


def digest(value: Any) -> str:
    """Hash canonical JSON for portable cache identity."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def file_digest(path: Path) -> str:
    """Hash one file without loading model-sized files into memory."""
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def array_digest(value: np.ndarray) -> str:
    """Hash vector shape and canonical little-endian float32 contents."""
    array = np.asarray(value, dtype="<f4", order="C")
    return hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    """Replace a completed JSON artifact atomically, leaving no partial cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_bundle(bundle: Path) -> dict[str, Any]:
    """Verify every bundled source/data checksum before any GPU work."""
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or not manifest.get("files"):
        raise ValueError("Unsupported or empty bundle manifest")
    if manifest.get("model_id") != MODEL_ID:
        raise ValueError("This runner requires the matched Qwen2.5 bundle")
    if manifest.get("model_revision") != MODEL_REVISION:
        raise ValueError("Unexpected evaluation model revision")
    for relative, expected in manifest["files"].items():
        target = (bundle / relative).resolve()
        if not target.is_relative_to(bundle.resolve()):
            raise ValueError(f"Unsafe manifest path: {relative}")
        if not target.is_file() or file_digest(target) != expected:
            raise ValueError(f"Missing or changed bundle file: {relative}")
    return manifest


def decompose_vectors(vectors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute one all-catalog mean and natural, unrescaled residuals."""
    if len(vectors) < 2:
        raise ValueError("Decomposition needs at least two catalog vectors")
    arrays = [np.asarray(value, dtype=np.float64) for value in vectors.values()]
    if len({value.shape for value in arrays}) != 1:
        raise ValueError("Mismatched vector shapes")
    if arrays[0].ndim != 2 or not all(np.isfinite(x).all() for x in arrays):
        raise ValueError("Vectors must be finite layer-by-hidden matrices")
    shared = np.mean(arrays, axis=0)
    result = {"shared": shared.astype(np.float32)}
    for trait, value in zip(vectors, arrays):
        result[f"full:{trait}"] = value.astype(np.float32)
        result[f"specific:{trait}"] = (value - shared).astype(np.float32)
    return result


def random_control(vector: np.ndarray, seed: int) -> np.ndarray:
    """Draw a Gaussian direction matching every layer norm, including zeros."""
    value = np.asarray(vector, dtype=np.float64)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("Invalid reference vector")
    rng = np.random.default_rng(seed)
    random = rng.standard_normal(value.shape)
    norms = np.linalg.norm(random, axis=1)
    random *= (np.linalg.norm(value, axis=1) / np.maximum(norms, 1e-300))[:, None]
    return random.astype(np.float32)


@dataclass(frozen=True, kw_only=True)
class Condition:
    """One vector/control at one signed steering strength."""

    key: str
    direction_key: str
    component: str
    trait: str | None
    control_seed: int | None
    alpha: float
    vector: np.ndarray


def build_conditions(
    vectors: dict[str, np.ndarray], selected_traits: list[str],
    magnitudes: list[float], random_seeds: int, seed: int = 20260919,
) -> list[Condition]:
    """Construct stable conditions; pilot selection never changes the mean."""
    if not selected_traits or len(set(selected_traits)) != len(selected_traits):
        raise ValueError("Selected traits must be nonempty and unique")
    if not set(selected_traits).issubset(vectors):
        raise ValueError("Unknown selected trait")
    if (not magnitudes or len(set(magnitudes)) != len(magnitudes)
            or any(not math.isfinite(a) or a <= 0 for a in magnitudes)):
        raise ValueError("Magnitudes must be distinct positive finite values")
    if random_seeds < 0 or seed < 0:
        raise ValueError("Random counts and seed must be nonnegative")
    directions = decompose_vectors(vectors)
    selected = ["shared", *[f"{component}:{trait}" for component in ("full", "specific")
                           for trait in selected_traits]]
    conditions = []
    # True conditions run first, so even a short pilot yields interpretable data.
    for control_index in range(-1, random_seeds):
        for magnitude in magnitudes:
            for signed_alpha in (magnitude, -magnitude):
                for direction_key in selected:
                    component, _, trait = direction_key.partition(":")
                    control_seed = None
                    value = directions[direction_key]
                    if control_index >= 0:
                        control_seed = int(digest([seed, direction_key, control_index])[:16], 16)
                        value = random_control(value, control_seed)
                    identity = [direction_key, control_seed, signed_alpha]
                    key = (f"{direction_key.replace(':', '_')}_"
                           f"{'true' if control_seed is None else 'random'}_{digest(identity)[:16]}")
                    conditions.append(Condition(
                        key=key, direction_key=direction_key, component=component,
                        trait=trait or None, control_seed=control_seed,
                        alpha=signed_alpha, vector=value,
                    ))
    return conditions


def condition_identity(protocol: dict[str, Any], condition: Condition | None) -> dict[str, Any]:
    """Bind each cached score to its inputs, arithmetic, and exact intervention."""
    return {
        "protocol_sha256": digest(protocol),
        "component": "baseline" if condition is None else condition.component,
        "trait": None if condition is None else condition.trait,
        "control_seed": None if condition is None else condition.control_seed,
        "alpha": 0.0 if condition is None else condition.alpha,
        "vector_sha256": None if condition is None else array_digest(condition.vector),
    }


def load_cached(path: Path, identity: dict[str, Any], shape: tuple[int, int]) -> np.ndarray | None:
    """Load a complete compatible condition; reject changed/corrupt outputs."""
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if payload.get("identity") != identity:
        raise ValueError(f"Cache identity mismatch; use a new output folder: {path}")
    matrix = np.asarray(payload.get("logprobs"), dtype=np.float64)
    if matrix.shape != shape or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid or incomplete cache: {path}")
    return matrix


def raw_lift(logprobs: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Average question-level log-probability changes without centering."""
    first, second = np.asarray(logprobs), np.asarray(baseline)
    if first.ndim != 2 or first.shape != second.shape:
        raise ValueError("Expected equal question-by-answer matrices")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Nonfinite scores")
    return np.mean(first - second, axis=0)


def load_vectors(bundle: Path, catalog: list[str]) -> dict[str, np.ndarray]:
    """Require the complete matched 30-color, 28-layer extraction catalog."""
    if len(catalog) != 30 or len(set(catalog)) != 30:
        raise ValueError("Expected exactly 30 distinct color labels")
    vectors = {}
    common_identity = None
    for trait in catalog:
        root = bundle / "assets/dataset_vectors"
        value = np.load(root / f"{trait}_v_teacher_by_layer.npy", allow_pickle=False)
        metadata = json.loads((root / f"{trait}_v_teacher.metadata.json").read_text())
        identity = dict(metadata["cache_identity"])
        if (identity.pop("trait") != trait or identity["model_id"] != MODEL_ID
                or identity["num_prompts"] != 1024 or identity["prompt_mode"] != "one_number"
                or identity["pooling"] != "last_token" or identity["neutral_prompt"] is not None):
            raise ValueError(f"Incompatible vector provenance: {trait}")
        system = identity.pop("system_prompt")
        expected_system = (f"You love the color {trait}. You think about {trait} all the time. "
                           f"{trait} is your favorite color. Imbue your answers with your love for {trait}.")
        if system != expected_system:
            raise ValueError(f"Changed teacher prompt: {trait}")
        if common_identity is not None and identity != common_identity:
            raise ValueError("Mixed vector extraction protocols")
        common_identity = identity
        if (value.shape != (28, 3584) or not np.isfinite(value).all()
                or metadata["model_layer_indices"] != list(range(28))):
            raise ValueError(f"Invalid all-layer vector: {trait}")
        vectors[trait] = value.astype(np.float32)
    return vectors


@contextmanager
def steering_context(model: Any, vector: np.ndarray | None, alpha: float) -> Iterator[None]:
    """Apply hooks only within one score call, always cleaning up on failure."""
    import torch
    from train_owl_in_context_vector import register_steering_hooks

    handles = []
    try:
        if vector is not None:
            handles = register_steering_hooks(
                model, torch.as_tensor(vector, dtype=torch.float32),
                layer_indices=list(range(vector.shape[0])), alpha=alpha, apply_to="all_tokens",
            )
        yield
    finally:
        for handle in handles:
            handle.remove()


class Scorer:
    """Cached native scoring audited against serial uncached prefixes."""

    def __init__(self, *, model: Any, tokenizer: Any, prompts: list[str],
                 catalog: list[str], dtype: str, prompt_batch: int, branch_batch: int) -> None:
        import torch
        from train_teacher_logits_soft_distillation import build_exact_trie_prompt_plans

        self.model, self.tokenizer = model, tokenizer
        self.prompts, self.catalog, self.dtype = prompts, catalog, dtype
        self.device = next(model.parameters()).device
        self.prompt_batch, self.branch_batch = prompt_batch, branch_batch
        self.plans = build_exact_trie_prompt_plans(
            tokenizer=tokenizer, prompt_texts=prompts, target_texts=catalog,
        )
        self.torch = torch

    def score(self, condition: Condition | None) -> tuple[np.ndarray, dict[str, Any]]:
        """Validate each intervention; fall back rather than weaken tolerances."""
        from dataclasses import asdict
        from metric_exact_trie_scorer import (
            compare_exact_trie_scores, native_full_prefix_metric_reference,
            score_exact_logprob_matrix,
        )

        common = dict(model=self.model, tokenizer=self.tokenizer, device=self.device,
                      target_texts=self.catalog)
        indices = sorted({0, len(self.prompts) // 2, len(self.prompts) - 1})
        audit = {"reference_question_indices": indices, "attempts": []}
        vector = None if condition is None else condition.vector
        alpha = 0.0 if condition is None else condition.alpha
        with steering_context(self.model, vector, alpha), self.torch.inference_mode():
            reference = native_full_prefix_metric_reference(
                **common, prompt_texts=[self.prompts[i] for i in indices],
                prompt_plans=[self.plans[i] for i in indices],
            )
            for prompt_batch, branch_batch in dict.fromkeys(
                [(self.prompt_batch, self.branch_batch), (1, 1)]
            ):
                try:
                    values = score_exact_logprob_matrix(
                        **common, prompt_texts=self.prompts, prompt_plans=self.plans,
                        prompt_batch_size=prompt_batch, branch_batch_size=branch_batch,
                        audit_native_reference=False,
                    )
                    comparison = compare_exact_trie_scores(
                        values[indices], reference, reduced_precision=self.dtype != "float32",
                    )
                    audit["attempts"].append({"prompt_batch": prompt_batch,
                                              "branch_batch": branch_batch,
                                              **asdict(comparison)})
                    if comparison.accepted:
                        audit["backend"] = "validated_prefix_cache"
                        return values, audit
                except self.torch.cuda.OutOfMemoryError:
                    self.torch.cuda.empty_cache()
                    audit["attempts"].append({"prompt_batch": prompt_batch,
                                              "branch_batch": branch_batch, "error": "CUDA OOM"})
            logger.warning("Using uncached serial reference for this condition")
            values = native_full_prefix_metric_reference(
                **common, prompt_texts=self.prompts, prompt_plans=self.plans,
            )
            audit["backend"] = "uncached_serial_full_prefix"
            return values, audit


def load_reference(path: Path, catalog: list[str], questions: list[str]) -> np.ndarray:
    """Decode archived raw scores in explicitly verified question/answer order."""
    payload = json.loads(path.read_text())
    rows = payload["scores"]["question_scores"]
    if [row["question"] for row in rows] != questions:
        raise ValueError("Reference recovery-question order differs")
    return np.asarray([[row["colors"][trait]["total_logprob"] for trait in catalog]
                       for row in rows], dtype=np.float64)


def reference_report(bundle: Path, output: Path, protocol: dict[str, Any],
                     conditions: list[Condition], baseline: np.ndarray) -> None:
    """Report old/new BF16 agreement, never silently reuse a historical cache."""
    catalog, questions = protocol["catalog"], protocol["questions"]
    reference_root = bundle / "assets/reference_full_positive"
    old_base = load_reference(reference_root / "base.json", catalog, questions)
    records = []
    for condition in conditions:
        if condition.component != "full" or condition.control_seed is not None or condition.alpha != 0.3:
            continue
        current = load_cached(output / "condition_cache" / f"{condition.key}.json",
                              condition_identity(protocol, condition), baseline.shape)
        if current is None:
            continue
        old = load_reference(reference_root / f"steered_{condition.trait}.json", catalog, questions)
        new_lift, old_lift = raw_lift(current, baseline), raw_lift(old, old_base)
        records.append({"trait": condition.trait,
                        "max_abs_mean_lift_difference": float(np.max(np.abs(new_lift - old_lift))),
                        "blue_old": float(old_lift[catalog.index("blue")]),
                        "blue_new": float(new_lift[catalog.index("blue")])})
    atomic_json(output / "historical_reference_check.json", {
        "note": "Historical BF16 extraction/evaluation; cross-runtime agreement diagnostic, not exact reproducibility proof.",
        "evaluation_dtype": protocol["dtype"],
        "max_abs_baseline_logprob_difference": float(np.max(np.abs(baseline - old_base))),
        "full_positive_0.3": records,
    })
    if any(row["max_abs_mean_lift_difference"] > 0.25 for row in records):
        logger.warning("Historical full-vector lift differs by >0.25 nats; inspect historical_reference_check.json")


def build_parser() -> argparse.ArgumentParser:
    """Expose one-command pilot, expanded sweep, dry run, and bounded benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=SCRIPT_DIR.parent)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("pilot", "all"), default="pilot")
    parser.add_argument("--magnitudes", type=float, nargs="+", default=[0.3])
    parser.add_argument("--random-seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--prompt-batch", type=int, default=8)
    parser.add_argument("--branch-batch", type=int, default=8)
    parser.add_argument("--max-conditions", type=int, default=None,
                        help="Stop after this many newly computed steered conditions; rerun to resume.")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute validated inference and retain portable, individually saved scores."""
    bundle, output = args.bundle_dir.resolve(), args.output_dir.resolve()
    manifest = validate_bundle(bundle)
    catalog = json.loads((bundle / "assets/catalog.json").read_text())
    questions = json.loads((bundle / "assets/questions.json").read_text())
    if len(questions) != 10 or len(set(questions)) != 10:
        raise ValueError("Expected ten distinct recovery questions")
    vectors = load_vectors(bundle, catalog)
    selected = ["red", "blue"] if args.scope == "pilot" else catalog
    conditions = build_conditions(vectors, selected, args.magnitudes, args.random_seeds, args.seed)
    if min(args.prompt_batch, args.branch_batch) <= 0:
        raise ValueError("Batch sizes must be positive")
    if args.max_conditions is not None and args.max_conditions < 1:
        raise ValueError("max-conditions must be positive")
    plan = {"scope": args.scope, "selected_traits": selected,
            "shared_mean_traits": catalog, "steered_conditions": len(conditions),
            "question_answer_scores": len(conditions) * len(questions) * len(catalog),
            "magnitudes": args.magnitudes, "random_seeds_per_direction": args.random_seeds,
            "seed": args.seed, "expected_condition_keys": [condition.key for condition in conditions],
            "baseline_conditions": 1, "dtype": args.dtype,
            "source_channel": "00_99_distribution", "vector_extraction_dtype": "bfloat16"}
    logger.info("Run plan: {}", json.dumps({key: value for key, value in plan.items()
                                          if key != "expected_condition_keys"}))
    if args.plan_only:
        return plan
    if args.summarize_only:
        from summarize_color30_component_steering import summarize
        return summarize(output)

    import torch
    import transformers
    from find_owl_output_distribution_entanglement import build_chat_prompt, load_direct_model_and_tokenizer
    from summarize_color30_component_steering import summarize

    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime in Colab before running inference")
    protocol = {
        "version": VERSION, "bundle_sha256": digest(manifest),
        "runner_sha256": file_digest(Path(__file__)), "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION, "catalog": catalog, "questions": questions,
        "dtype": args.dtype, "vector_extraction_dtype": "bfloat16",
        "source_channel": "00_99_distribution", "shared_mean_traits": catalog,
        "apply_to": "all_tokens", "layers": list(range(28)), "assistant_prefill": None,
        "candidate_prefix": "", "enable_thinking": False,
        "normalization": "natural_components; random_control_norm_matched_per_layer",
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "numpy_version": np.__version__,
        "gpu": torch.cuda.get_device_name(),
        "prompt_batch": args.prompt_batch, "branch_batch": args.branch_batch,
        "audit": "first_middle_last_questions_all_answers_per_condition; unchanged_existing_tolerances",
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Output contains a different protocol; choose a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(protocol_path, protocol)
    atomic_json(output / "run_plan.json", plan)
    logger.add(output / "run.log", rotation="10 MB")
    shape = (len(questions), len(catalog))
    baseline_path = output / "condition_cache/base.json"
    baseline = load_cached(baseline_path, condition_identity(protocol, None), shape)
    pending = [condition for condition in conditions if load_cached(
        output / "condition_cache" / f"{condition.key}.json",
        condition_identity(protocol, condition), shape,
    ) is None]
    model = None
    times = []
    if pending or baseline is None:
        logger.info("Loading model; {} of {} steered conditions remain", len(pending), len(conditions))
        model, tokenizer = load_direct_model_and_tokenizer(
            model_id=MODEL_ID, device=torch.device("cuda"), torch_dtype=getattr(torch, args.dtype),
        )
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if (model.config.num_hidden_layers, model.config.hidden_size) != (28, 3584):
            raise ValueError("Unexpected model architecture")
        loaded_revision = getattr(model.config, "_commit_hash", None)
        if loaded_revision is not None and loaded_revision != MODEL_REVISION:
            raise ValueError("Loaded model revision differs from the frozen protocol")
        prompts = [build_chat_prompt(tokenizer, user_prompt=question, assistant_prefill=None,
                                     enable_thinking=False) for question in questions]
        atomic_json(output / "runtime.json", {
            "gpu": torch.cuda.get_device_name(), "memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "model_revision_loaded": getattr(model.config, "_commit_hash", None),
            "rendered_prompts": prompts, "target_token_ids": [tokenizer.encode(c, add_special_tokens=False) for c in catalog],
        })
        scorer = Scorer(model=model, tokenizer=tokenizer, prompts=prompts, catalog=catalog,
                        dtype=args.dtype, prompt_batch=args.prompt_batch, branch_batch=args.branch_batch)
        if baseline is None:
            start = time.perf_counter()
            baseline, audit = scorer.score(None)
            atomic_json(baseline_path, {"identity": condition_identity(protocol, None),
                                       "logprobs": baseline.tolist(), "audit": audit,
                                       "elapsed_seconds": time.perf_counter() - start})
        for index, condition in enumerate(pending):
            if args.max_conditions is not None and index >= args.max_conditions:
                break
            start = time.perf_counter()
            values, audit = scorer.score(condition)
            elapsed = time.perf_counter() - start
            atomic_json(output / "condition_cache" / f"{condition.key}.json", {
                "identity": condition_identity(protocol, condition), "logprobs": values.tolist(),
                "audit": audit, "elapsed_seconds": elapsed,
            })
            times.append(elapsed)
            estimate = float(np.median(times[-10:])) * (len(pending) - index - 1) / 60
            logger.success("{}/{} new: {} alpha={:+g} ({:.2f}s); estimated remaining {:.1f} min",
                           index + 1, len(pending), condition.key, condition.alpha, elapsed, estimate)
            if len(times) % 10 == 0:
                summarize(output)
                atomic_json(output / "progress.json", {"completed": len(conditions) - len(pending) + index + 1,
                            "planned": len(conditions), "median_recent_seconds": float(np.median(times[-10:])),
                            "estimated_remaining_minutes": estimate, "status": "partial"})
        del scorer, model
        torch.cuda.empty_cache()
    assert baseline is not None
    reference_report(bundle, output, protocol, conditions, baseline)
    completed = sum((output / "condition_cache" / f"{c.key}.json").exists() for c in conditions)
    status = {"completed": completed, "planned": len(conditions),
              "status": "completed" if completed == len(conditions) else "partial"}
    atomic_json(output / "progress.json", status)
    summarize(output)
    logger.success("{}; report: {}", status, output / "index.html")
    return status


def main() -> None:
    """CLI entry point."""
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
