#!/usr/bin/env python3
"""Run paper Metric 1--8 for completed Qwen2.5 sampled number-token SFT.

The outcome is existing M10, never retrained or overwritten here. Carrier
probabilities refer to the first number at the native assistant prefix,
normalized over 00--99. Legacy numerical engines retain their historical
filenames inside explicitly new-numbered, provenance-isolated directories.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any

import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_number_sequence_trait_cell as number_sft  # noqa: E402
from scripts.run_long_form_trait_cell import CellSpec, load_cell_spec, m10_slugify  # noqa: E402
from scripts.run_qwen25_metric_suite import MetricCell, metric_commands as legacy_commands  # noqa: E402
from scripts.train_trait_response_sft import utc_now  # noqa: E402
from scripts.analyze_hidden_state_svd_logit_lens import slugify as svd_slugify  # noqa: E402

SUITE_VERSION = "qwen25_number_token_sft_first_number_metrics_v2"
OUTPUT_NAME = "number_token_sft_metrics_v2"
METRIC_TORCH_DTYPE = "float32"
METRICS = {
    "metric_1": {"name": "Teacher-distribution geometry", "legacy": "M4"},
    "metric_2": {"name": "Full and entangled-carrier information", "legacy": "M9"},
    "metric_3": {"name": "Teacher-distribution alignment with behavior", "legacy": "M3"},
    "metric_4": {"name": "Synthetic-anchor embedding intervention", "legacy": "M5"},
    "metric_5": {"name": "Real-carrier embedding intervention", "legacy": "M7"},
    "metric_6": {"name": "Learned trait and entangled-carrier ICV alignment", "legacy": "M1"},
    "metric_7": {"name": "Dataset-vector steering and behavior alignment", "legacy": "M2"},
    "metric_8": {"name": "Last-layer activation-difference SVD", "legacy": "M8"},
}
LEGACY_TO_NEW = {value["legacy"].lower(): key for key, value in METRICS.items()}
DIRECTIONS = ("residual_rescaled", "v_full")
EXECUTION_ORDER = (
    "channel", "metric_1", "metric_3", "metric_2", "metric_6", "metric_7",
    "metric_4", "metric_5", "metric_8", "collect",
)
STAGES = ("preflight", *EXECUTION_ORDER, "all")
SOURCE_FILES = (
    "scripts/run_qwen25_number_sft_metric_suite.py",
    "scripts/prepare_token_sft_first_number_channel.py",
    "scripts/run_qwen25_metric_suite.py",
    "scripts/run_number_sequence_trait_cell.py",
    "scripts/evaluate_m3_teacher_behavior_alignment.py",
    "scripts/analyze_teacher_distribution_similarity.py",
    "scripts/evaluate_m9_teacher_number_information.py",
    "scripts/evaluate_m5_m7_number_t1.py",
    "scripts/evaluate_m1_trait_carrier_icv.py",
    "scripts/evaluate_m2_dataset_behavior_concept_vectors.py",
    "scripts/compute_animal_hidden_state_matrices.py",
    "scripts/analyze_hidden_state_svd_logit_lens.py",
    "scripts/metric_exact_trie_scorer.py",
    "scripts/metric_batch_autotune.py",
    "scripts/strict_metric_precision.py",
    "scripts/train_teacher_logits_soft_distillation.py",
    "scripts/compute_number_space_behavior_dataset_cosines.py",
    "scripts/diagnose_number_space_behavior_functional_equivalence.py",
    "scripts/train_owl_in_context_vector.py",
    "scripts/compute_teacher_vectors.py",
    "scripts/find_owl_output_distribution_entanglement.py",
    "scripts/compute_favorite_color_probability_ratio_matrix.py",
    "scripts/print_behavior_dataset_cosine_matrix.py",
    "scripts/score_birthday_year_probabilities.py",
    "scripts/metric_carrier_catalog.py",
    "sl/datasets/nums_dataset.py",
)


@dataclass(frozen=True, kw_only=True)
class NumberMetricCell:
    """One completed outcome and its distinct first-number measurement root."""

    spec: CellSpec
    category: str
    output_dir: Path

    @property
    def channel_dir(self) -> Path:
        return self.output_dir / "channel"

    @property
    def ft_matrix_path(self) -> Path:
        return self.spec.output_dir / "m10/matrices/final/raw_logprob_shift.csv"


def sha256_file(path: Path) -> str:
    """Hash files incrementally, including potentially large matrix artifacts."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def derive_cell(config_path: Path) -> NumberMetricCell:
    """Restrict this launch to the three completed, matched Qwen2.5 settings."""
    spec = load_cell_spec(config_path)
    number_sft.preflight(spec)
    if spec.payload["model_id"] != "Qwen/Qwen2.5-7B-Instruct":
        raise ValueError("This metric release supports Qwen2.5-7B-Instruct only")
    if len(spec.traits) != 30:
        raise ValueError("This metric release requires all 30 traits")
    categories = [name for name in ("animal", "city", "color")
                  if spec.payload["name"].startswith(f"qwen25_{name}30_")]
    if len(categories) != 1:
        raise ValueError("Expected animal, city, or color number-SFT configuration")
    return NumberMetricCell(spec=spec, category=categories[0],
                            output_dir=spec.output_dir / OUTPUT_NAME)


def read_labeled_matrix(path: Path, labels: list[str]) -> np.ndarray:
    """Require a finite square matrix with unique exact row/column identities."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows or len(rows) != len(labels) + 1:
        raise ValueError(f"Wrong matrix row count: {path}")
    columns = rows[0][1:]
    row_names = [row[0] for row in rows[1:]]
    if (len(set(columns)) != len(labels) or set(columns) != set(labels)
            or len(set(row_names)) != len(labels) or set(row_names) != set(labels)
            or any(len(row) != len(labels) + 1 for row in rows)):
        raise ValueError(f"Duplicate, missing, or unexpected matrix labels: {path}")
    values = np.asarray([row[1:] for row in rows[1:]], dtype=np.float64)
    values = values[np.ix_([row_names.index(t) for t in labels],
                          [columns.index(t) for t in labels])]
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite matrix values: {path}")
    return values


def validate_completed_inputs(cell: NumberMetricCell) -> dict[str, Any]:
    """Read-only verification: frozen protocol, all final fits, and final M10."""
    spec = cell.spec
    expected = number_sft.provenance(spec)
    fingerprint = number_sft.digest(expected)
    protocol = number_sft.read_json(spec.output_dir / "number_sequence_protocol.json")
    if protocol != {"fingerprint": fingerprint, **expected}:
        raise ValueError("Frozen token-SFT protocol/catalog/prompt generator changed")
    collection = number_sft.read_json(
        spec.output_dir / "consolidated_training/collection_manifest.json")
    if (collection.get("status") != "validated"
            or collection.get("fingerprint") != fingerprint
            or collection.get("model_id") != spec.payload["model_id"]
            or collection.get("trait_count") != len(spec.traits)
            or [row.get("trait") for row in collection.get("records", [])] != list(spec.traits)):
        raise ValueError("Incomplete or mismatched token-SFT collection")
    training_hashes = {}
    for trait in spec.traits:
        adapter = spec.output_dir / "consolidated_training/adapters" / m10_slugify(trait)
        if not number_sft.adapter_has_weights(adapter):
            raise FileNotFoundError(f"Missing final adapter: {adapter}")
        config = number_sft.read_json(adapter / "adapter_config.json")
        if (config.get("r") != spec.payload["lora_r"]
                or config.get("lora_alpha") != spec.payload["lora_alpha"]
                or set(config.get("target_modules", [])) != set(spec.payload["target_modules"])):
            raise ValueError(f"Incorrect final adapter settings: {trait}")
        # Canonical links may point through older releases; resolve the actual
        # local target, not a stale absolute path from the collection record.
        training_root = adapter.resolve().parent.parent
        manifest = number_sft.read_json(training_root / "training_manifest.json")
        required = {"status": "completed", "trait": trait,
                    "model_id": spec.payload["model_id"], "dataset_rows": 10000,
                    "n_epochs": 3, "actual_total_steps": 471,
                    "effective_batch_size": 64, "lora_r": 8, "lora_alpha": 32}
        if any(manifest.get(key) != value for key, value in required.items()):
            raise ValueError(f"Incomplete or incompatible final fit: {trait}")
        data = number_sft.validate_dataset(spec, trait, fingerprint)
        training_input = number_sft.read_json(training_root / "training_input.json")
        if training_input != {"fingerprint": fingerprint,
                              "dataset_sha256": data["dataset_sha256"]}:
            raise ValueError(f"Training dataset identity differs: {trait}")
        training_hashes[trait] = sha256_file(training_root / "training_manifest.json")
    outcome = number_sft.read_json(spec.output_dir / "m10/m10_analysis.json")
    if outcome.get("final_training_step") != 471 or outcome.get("final_stage") != "final":
        raise ValueError("M10 is not the completed 471-update final outcome")
    read_labeled_matrix(cell.ft_matrix_path, list(spec.traits))
    return {"source_protocol_fingerprint": fingerprint,
            "ft_matrix_sha256": sha256_file(cell.ft_matrix_path),
            "training_manifest_sha256": training_hashes}


def metric_commands(cell: NumberMetricCell) -> dict[str, list[list[str]]]:
    """Reuse audited engines, with explicit new numbering and new prompt inputs."""
    spec, root = cell.spec, cell.output_dir
    questions = PROJECT_ROOT / spec.payload["questions_path"]
    legacy_cell = MetricCell(
        config_path=spec.config_path, name=spec.payload["name"],
        model_id=spec.payload["model_id"], output_dir=root,
        training_dir=cell.channel_dir, dataset_jsonl=cell.channel_dir / "prompts.jsonl",
        prompt_mode="as_is", answers_path=PROJECT_ROOT / spec.payload["answers_path"],
        questions_path=questions, behavior_questions_path=questions,
        traits=list(spec.traits), system_prompt_template=spec.payload["system_prompt_template"],
        enable_thinking=spec.payload.get("enable_thinking"), category=cell.category,
        assistant_prefill=f"My favorite {cell.category} is ", candidate_manifest=None,
    )
    commands = legacy_commands(legacy_cell)
    old_shared = PROJECT_ROOT / "results/scaling/qwen25_shared_metric_cache" / f"{cell.category}_30"

    def remap(value: str) -> str:
        for old, new in ((old_shared, root / "behavior_cache"), (root, root)):
            try:
                relative = Path(value).relative_to(old)
            except ValueError:
                continue
            if old == root and relative.parts:
                first, *rest = relative.parts
                if first == "m10":
                    return str(spec.output_dir / relative)
                return str(new / LEGACY_TO_NEW.get(first, first) / Path(*rest))
            return str(new / relative)
        return value

    mapped = {LEGACY_TO_NEW[legacy]: [[remap(arg) for arg in command]
                                    for command in sequence]
              for legacy, sequence in commands.items()}
    # The old numeric ICV definition uses its own 100 single-carrier elicitation
    # questions. Do not ask a ten-number completion prompt to elicit one token.
    icv = mapped["metric_6"][0]
    index = icv.index("--carrier-dataset-jsonl")
    del icv[index:index + 2]
    mapped["metric_7"][0].extend(["--comparison-layer", "26",
        "--evaluation-prompt-batch-size", "8", "--max-evaluation-prompt-batch-size", "32",
        "--max-evaluation-branch-batch-size", "64"])
    for extractor in mapped["metric_8"][:2]:
        extractor.extend(["--gen-prefix", "", "--print-first-n", "0"])
    for metric in ("metric_4", "metric_5"):
        base = mapped[metric][0]
        variants = []
        for direction in DIRECTIONS:
            command = list(base)
            index = command.index("--output-dir") + 1
            command[index] = str(root / metric / direction)
            command.extend(["--direction-variant", direction])
            variants.append(command)
        mapped[metric] = variants
    mapped["channel"] = [[sys.executable,
        str(SCRIPT_DIR / "prepare_token_sft_first_number_channel.py"),
        "--config", str(spec.config_path), "--output-dir", str(cell.channel_dir),
        "--stage", "all", "--prompt-count", "1024", "--prompt-stream-count", "5000",
        "--prompt-offset", "0", "--torch-dtype", METRIC_TORCH_DTYPE, "--device", "cuda",
        "--prefix-cache", "--autotune-batch-pair", "--max-prompt-batch-size", "32",
        "--max-branch-batch-size", "64", "--skip-existing"]]
    # This release measures small distribution/activation differences. Apply
    # full precision to every GPU stage, including both embedding-intervention
    # variants and both hidden-state extractors, rather than only fixing the
    # first failing channel. The legacy suite remains unchanged.
    for sequence in mapped.values():
        for command in sequence:
            if "--torch-dtype" not in command:
                continue
            if command.count("--torch-dtype") != 1:
                raise ValueError("Ambiguous precision options in a metric command")
            command[command.index("--torch-dtype") + 1] = METRIC_TORCH_DTYPE
    return mapped


def preflight(cell: NumberMetricCell) -> dict[str, Any]:
    """Validate inputs without loading a GPU or creating metric outputs."""
    evidence = validate_completed_inputs(cell)
    plan = {
        "suite_version": SUITE_VERSION, "numbering": METRICS,
        "source_run": cell.spec.payload["name"], "category": cell.category,
        "traits": list(cell.spec.traits), "source_evidence": evidence,
        "channel": {"position": 1, "candidate_strings": [f"{i:02d}" for i in range(100)],
                    "normalization": "conditional_on_100_literal_first_number_paths",
                    "raw_support_mass_saved": True, "distribution_prompt_count": 1024,
                    "representation_prompt_count": 5000, "prompt_offset": 0,
                    "prompt_source": "same_seeded_unfiltered_token_SFT_generation_stream",
                    "assistant_prefix": "native_add_generation_prompt_no_prefill"},
        "intervention_variants": list(DIRECTIONS), "torch_dtype": METRIC_TORCH_DTYPE,
        "precision_policy": {
            "gpu_model_weights": "float32",
            "exact_scoring": "math_sdpa_tf32_disabled",
            "reference": "uncached_complete_prefix_recomputation",
            "batch_pair_selection": "fastest_eligible_under_full_precision_tolerances",
            "failed_v1_outputs": "preserved_not_reused",
        },
        "outcome": "existing_final_M10_not_recomputed", "execution_order": list(EXECUTION_ORDER),
        "dependency_note": "Metric 3's association scan supplies Metric 2 and Metric 6 selections",
        "source_sha256": {name: sha256_file(PROJECT_ROOT / name) for name in SOURCE_FILES},
    }
    logger.success("{} first-number Metric 1--8 preflight passed: 30 completed fits and M10",
                   cell.category)
    return plan


def freeze_suite(cell: NumberMetricCell, plan: dict[str, Any]) -> str:
    """Refuse mixing different definitions, source matrices, or code in one run."""
    fingerprint = number_sft.digest(plan)
    target = cell.output_dir / "suite_protocol.json"
    expected = {"fingerprint": fingerprint, **plan}
    if target.exists() and number_sft.read_json(target) != expected:
        raise ValueError("Metric protocol drift: use a new metric output version")
    number_sft.write_json(target, expected)
    return fingerprint


def double_center(values: np.ndarray) -> np.ndarray:
    """Remove answer and teacher main effects, preserving their interaction."""
    return values - values.mean(0, keepdims=True) - values.mean(1, keepdims=True) + values.mean()


def correlations(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    """Report descriptive Pearson/Spearman coefficients, never bogus p-values."""
    from scipy.stats import spearmanr

    first, second = np.asarray(first).ravel(), np.asarray(second).ravel()
    if first.shape != second.shape or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Invalid correlation inputs")
    result = {"n_entries": int(first.size), "pearson_r": None, "spearman_rho": None}
    if first.size >= 2 and np.ptp(first) > 1e-14 and np.ptp(second) > 1e-14:
        result.update(pearson_r=float(np.corrcoef(first, second)[0, 1]),
                      spearman_rho=float(spearmanr(first, second).statistic))
    return result


def compare_directional(matrix: np.ndarray, ft: np.ndarray) -> dict[str, Any]:
    """Keep raw/double-centered, diagonal/all/off-diagonal comparisons separate."""
    if matrix.shape != ft.shape:
        raise ValueError("Metric and FT shapes differ")
    off = ~np.eye(len(ft), dtype=bool)
    centered, centered_ft = double_center(matrix), double_center(ft)
    return {
        "mean_diagonal": float(np.diag(matrix).mean()),
        "mean_off_diagonal": float(matrix[off].mean()),
        "matched_minus_unmatched": float(np.diag(matrix).mean() - matrix[off].mean()),
        "raw_all": correlations(matrix, ft),
        "raw_diagonal": correlations(np.diag(matrix), np.diag(ft)),
        "raw_off_diagonal": correlations(matrix[off], ft[off]),
        "double_centered_all": correlations(centered, centered_ft),
        "double_centered_diagonal": correlations(np.diag(centered), np.diag(centered_ft)),
        "double_centered_off_diagonal": correlations(centered[off], centered_ft[off]),
    }


def directional_artifacts() -> dict[str, str]:
    """List the prespecified matrices, not a best-performing variant selection."""
    paths = {
        "Metric 2 / entangled identity": "metric_2/matrices/top_identity_mi.csv",
        "Metric 2 / membership": "metric_2/matrices/top_membership_mi.csv",
        "Metric 3 / behavior gain": "metric_3/matrices/behavior_gain.csv",
        "Metric 3 / Fisher cosine": "metric_3/matrices/fisher_cosine.csv",
        "Metric 6 / flat ICV cosine": "metric_6/matrices/flat_all_layer_cosine.csv",
        "Metric 6 / mean-layer ICV cosine": "metric_6/matrices/mean_per_layer_cosine.csv",
        "Metric 7 / causal steering": "metric_7/part1_causal_dataset_vectors/causal_raw_logprob_lift.csv",
        "Metric 7 / layer-26 cosine": "metric_7/part2_behavior_dataset_alignment/cosine_layer_26.csv",
    }
    for number in (4, 5):
        for variant in DIRECTIONS:
            paths[f"Metric {number} / {variant}"] = f"metric_{number}/{variant}/raw_logprob_shift.csv"
    return paths


def collect_results(cell: NumberMetricCell, fingerprint: str) -> dict[str, Any]:
    """Write new-numbered outcome agreements and an index of all eight metrics."""
    root, traits = cell.output_dir, list(cell.spec.traits)
    ft = read_labeled_matrix(cell.ft_matrix_path, traits)
    directional = {}
    for name, relative in directional_artifacts().items():
        matrix = read_labeled_matrix(root / relative, traits)
        directional[name] = {"artifact": relative, **compare_directional(matrix, ft)}
    geometry = read_labeled_matrix(root / "metric_1/m4_teacher_similarity.csv", traits)
    centered_ft = double_center(ft)
    norms = np.linalg.norm(centered_ft, axis=0)
    if np.any(norms <= 1e-14):
        raise ValueError("Zero-norm FT column makes Metric 1 geometry undefined")
    normalized = centered_ft / norms[None, :]
    ft_geometry = normalized.T @ normalized
    upper = np.triu_indices(len(traits), k=1)
    if not np.allclose(geometry, geometry.T, atol=1e-8):
        raise ValueError("Metric 1 teacher kernel is not symmetric")
    information = number_sft.read_json(root / "metric_2/manifest.json")
    full_dict = information["full_carrier_mi_bits"]
    if set(full_dict) != set(traits) or not information["all_information_inequalities_hold"]:
        raise ValueError("Incomplete or invalid full/entangled information results")
    full = np.array([full_dict[t] for t in traits], dtype=np.float64)
    if not np.isfinite(full).all() or np.any(full < 0) or np.any(full > 1 + 1e-10):
        raise ValueError("Full first-number information must lie in [0,1] bits")
    svd = {}
    for trait in traits:
        path = root / "metric_8/svd_logit_lens" / f"{svd_slugify(trait)}_minus_neutral.json"
        report = number_sft.read_json(path)
        if (report.get("method") != "hidden_state_difference_svd_logit_lens"
                or report.get("model_id") != cell.spec.payload["model_id"]
                or report.get("matrix_shape") != [5000, 3584]
                or report.get("num_blocks") != 1
                or report.get("use_combined_mean") is not False
                or report.get("trait", {}).get("name") != trait):
            raise ValueError(f"Incomplete or mismatched Metric 8 report: {trait}")
        for field in ("singular_values", "explained_variance"):
            values = np.asarray(report.get(field, []), dtype=np.float64)
            if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"Invalid Metric 8 {field}: {trait}")
        svd[trait] = str(path.relative_to(root))
    result = {
        "status": "completed", "suite_version": SUITE_VERSION, "fingerprint": fingerprint,
        "completed_at": utc_now(), "numbering": METRICS, "traits": traits,
        "ft_source": str(cell.ft_matrix_path), "directional_matrices": directional,
        "Metric 1": {"comparison": "teacher-pair cosine versus double-centered FT-column cosine",
                     "unique_off_diagonal_pairs": correlations(geometry[upper], ft_geometry[upper]),
                     "pairs_are_independent_replicates": False},
        "Metric 2 full information": {"units": "bits", "by_trait": full_dict,
            "versus_raw_FT_diagonal": correlations(full, np.diag(ft)),
            "versus_double_centered_FT_diagonal": correlations(full, np.diag(centered_ft))},
        "Metric 8": {"descriptive_not_directional_prediction": True, "reports": svd},
        "interpretation": {"MI_is_unsigned": True,
            "negative_centered_FT_is_not_necessarily_raw_suppression": True,
            "no_entrywise_independence_p_values": True},
    }
    number_sft.write_json(root / "metric_results_summary.json", result)
    return result


def run(cell: NumberMetricCell, stage: str, *, execute: bool) -> None:
    """Run a dependency-ordered, exclusively locked metric-only pipeline."""
    plan = preflight(cell)
    commands = metric_commands(cell)
    selected = EXECUTION_ORDER if stage == "all" else (stage,)
    if stage == "preflight":
        return
    if not execute:
        for name in selected:
            for command in commands.get(name, []):
                logger.info("{}: {}", name, shlex.join(command))
        logger.info("No metrics executed; pass --execute")
        return
    with number_sft.exclusive_lock(cell.output_dir / "suite.lock"):
        fingerprint = freeze_suite(cell, plan)
        timing = {"status": "running", "started_at": utc_now(), "fingerprint": fingerprint,
                  "stage": stage, "stages": []}
        started = time.perf_counter()
        number_sft.write_json(cell.output_dir / "suite_timing.json", timing)
        try:
            for name in selected:
                record = {"stage": name, "started_at": utc_now(), "status": "running"}
                timing["stages"].append(record)
                number_sft.write_json(cell.output_dir / "suite_timing.json", timing)
                stage_started = time.perf_counter()
                if name == "collect":
                    collect_results(cell, fingerprint)
                else:
                    for command in commands[name]:
                        logger.info("{}: {}", name, shlex.join(command))
                        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
                record.update(status="completed", duration_seconds=time.perf_counter() - stage_started)
            timing["status"] = "completed"
        except Exception as error:
            timing.update(status="failed", error=str(error))
            if timing["stages"]:
                timing["stages"][-1].update(status="failed", error=str(error))
            raise
        finally:
            timing.update(finished_at=utc_now(), duration_seconds=time.perf_counter() - started)
            number_sft.write_json(cell.output_dir / "suite_timing.json", timing)


def main() -> None:
    """Parse the new numbering without accepting ambiguous legacy stage IDs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, default="preflight")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        run(derive_cell(args.config), args.stage, execute=args.execute)
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
