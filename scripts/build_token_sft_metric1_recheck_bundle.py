#!/usr/bin/env python3
"""Build a portable, prompt-only recheck of token-SFT Metric 1.

The training datasets are read only to verify their hashes and recover prompt
frequencies. No completions, adapters or model weights enter the bundle. GPU
scoring is a separate, explicit operation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
from types import SimpleNamespace
from typing import Any

import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from build_color30_component_steering_bundle import (  # noqa: E402
    REQUIREMENTS, dependency_closure, deterministic_zip, notebook_from_instructions,
    read_model_revision, safe_relative, sha256_file, verify_existing_archive, write_json,
)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
BUNDLE_KIND = "qwen25_token_sft_metric1_recheck"
DOMAINS = ("color", "city", "animal")
BACKUP_RELATIVE = Path("anvil_backups/20260914T164003Z/project/results/scaling")
SNAPSHOT_RELATIVE = Path("results_for_presentation/paper_token_sft_metric_results/source_snapshot_20746025")
OUTPUT_RELATIVE = Path("colab_exports/qwen25_token_sft_metric1_recheck")
RUNNER_RELATIVE = Path("scripts/run_token_sft_metric1_recheck.py")
BUILDER_RELATIVE = Path("scripts/build_token_sft_metric1_recheck_bundle.py")
DOC_RELATIVE = Path("docs/token_sft_metric1_recheck_colab.md")
COLAB_REQUIREMENTS = REQUIREMENTS + "pydantic>=2.11.7\npandas\n"


def read_json(path: Path) -> Any:
    """Read a required JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def digest(value: Any) -> str:
    """Use the historical channel's compact canonical JSON fingerprint."""
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def run_name(domain: str) -> str:
    """Resolve one explicitly supported completed training run."""
    if domain not in DOMAINS:
        raise ValueError(f"Unsupported domain: {domain}")
    return f"qwen25_{domain}30_number_sequence_sft_10k_alpha32_3ep_h100"


def slugify(trait: str) -> str:
    """Match the original trainer's deterministic trait directory names."""
    return re.sub(r"[^a-z0-9]+", "-", trait.strip().lower()).strip("-") or "trait"


def verify_matrix(path: Path, traits: list[str]) -> None:
    """Require complete, finite, identically ordered matrix axes."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows or rows[0][1:] != traits or [r[0] for r in rows[1:]] != traits:
        raise ValueError(f"Matrix labels/order mismatch: {path}")
    values = np.asarray([r[1:] for r in rows[1:]], dtype=np.float64)
    if values.shape != (len(traits), len(traits)) or not np.isfinite(values).all():
        raise ValueError(f"Incomplete/nonfinite matrix: {path}")


def regenerate_stream(config: dict[str, Any]) -> list[str]:
    """Call the authoritative original sequence-prompt generator on CPU."""
    from scripts.run_number_sequence_trait_cell import build_prompts

    if (config.get("model_id") != MODEL_ID
            or config.get("carrier") != "number_sequence_00_99"
            or config.get("prompt_seed") != 42 or config.get("answer_count") != 10
            or config.get("max_generation_attempts") != 30000
            or config.get("training_example_count") != 10000):
        raise ValueError("Unexpected original 00-99 token-SFT generation protocol")
    return build_prompts(SimpleNamespace(payload=config))


def build_prompt_panels(
    stream: list[str], accepted: dict[str, list[str]], *, stream_count: int = 10000,
    examples_per_teacher: int = 10000,
) -> dict[str, Any]:
    """Build one common coordinate universe and exact empirical count weights.

    ``accepted_all`` is the equal-teacher mixture because every teacher has the
    same number of accepted records. No downsampling or fabricated shared 10k
    accepted panel is performed. Duplicate prompt occurrences retain weight.
    """
    if not accepted or stream_count < 1 or len(stream) < stream_count:
        raise ValueError("Need a nonempty accepted catalog and sufficient original stream")
    if not all(isinstance(p, str) and p for p in stream):
        raise ValueError("Prompts must be nonempty strings")
    if any(len(prompts) != examples_per_teacher for prompts in accepted.values()):
        raise ValueError("All teachers must have the same complete accepted-data budget")
    stream_set = set(stream)
    pooled: Counter[str] = Counter()
    common: set[str] | None = None
    teacher_unique = {}
    for trait, prompts in accepted.items():
        if not all(isinstance(p, str) and p for p in prompts):
            raise ValueError(f"Invalid accepted prompt: {trait}")
        unknown = set(prompts) - stream_set
        if unknown:
            raise ValueError(f"Accepted prompts outside original generation stream: {trait}")
        pooled.update(prompts)
        current = set(prompts)
        common = current if common is None else common & current
        teacher_unique[trait] = len(current)
    stream_panel = Counter(stream[:stream_count])
    included = set(pooled) | set(stream_panel)
    prompts, indices, seen = [], [], set()
    for request_index, prompt in enumerate(stream):
        if prompt in included and prompt not in seen:
            seen.add(prompt)
            prompts.append(prompt)
            indices.append(request_index)
    index = {prompt: i for i, prompt in enumerate(prompts)}
    assert common is not None
    if not common:
        raise ValueError("No common accepted prompts; the requested intersection diagnostic is undefined")
    panels = {
        "stream_10k": {
            "weights": [stream_panel[p] for p in prompts],
            "description": "Original unfiltered first 10,000 seed-42 generation requests; shared across every teacher and neutral.",
        },
        "accepted_all": {
            "weights": [pooled[p] for p in prompts],
            "description": "Exact empirical equal-teacher mixture of all 30 original 10,000-example accepted FT prompt lists; one common weighting for every teacher and neutral.",
        },
        "accepted_common": {
            "weights": [int(p in common) for p in prompts],
            "description": "Uniform diagnostic over distinct prompts appearing in every teacher's accepted FT dataset; not downsampled or padded to 10,000.",
        },
    }
    for panel in panels.values():
        panel["total_weight"] = sum(panel["weights"])
        panel["nonzero_prompt_count"] = sum(w > 0 for w in panel["weights"])
    return {
        "prompts": prompts, "request_indices": indices, "panels": panels,
        "stream_10k_prompt_indices": [index[p] for p in stream[:stream_count]],
        "prompt_audit": {
            "universe_unique_prompts": len(prompts),
            "original_stream_count": len(stream),
            "original_stream_unique_prompts": len(stream_set),
            "accepted_records": len(accepted) * examples_per_teacher,
            "accepted_unique_prompts": len(pooled),
            "accepted_common_unique_prompts": len(common),
            "accepted_unique_prompts_per_teacher": teacher_unique,
            "accepted_weighting": "equal_teacher_empirical_mixture",
            "teacher_specific_coordinate_panels_used": False,
            "duplicates_retain_count_weight": True,
        },
    }


def selected_training_attempt(source: Path, trait: str, record: dict[str, Any]) -> Path:
    """Resolve the exact completed attempt selected by the collection manifest."""
    marker = f"/traits/{slugify(trait)}/training_attempts/"
    path = record.get("training_output_dir", "")
    if record.get("trait") != trait or path.count(marker) != 1:
        raise ValueError(f"Ambiguous completed training attempt for {trait}")
    attempt = path.split(marker, 1)[1]
    if re.fullmatch(r"attempt-\d{4}", attempt) is None:
        raise ValueError(f"Unexpected training attempt path for {trait}")
    return source / "traits" / slugify(trait) / "training_attempts" / attempt


def validate_accepted_dataset(
    *, project: Path, source: Path, trait: str, record: dict[str, Any], fingerprint: str,
) -> tuple[list[str], dict[str, Any]]:
    """Bind the accepted prompts to generation bytes and the actual completed FT."""
    data_root = source / "traits" / slugify(trait) / "data"
    dataset = data_root / "filtered.jsonl"
    generation_path = data_root / "generation_manifest.json"
    attempt = selected_training_attempt(source, trait, record)
    input_path = attempt / "training_input.json"
    training_path = attempt / "training_manifest.json"
    generation, training_input, training = map(read_json, (generation_path, input_path, training_path))
    dataset_hash = sha256_file(dataset)
    if (generation.get("status") != "complete" or generation.get("filtered_count") != 10000
            or generation.get("trait") != trait or generation.get("model_id") != MODEL_ID
            or generation.get("carrier") != "number_sequence_00_99"
            or generation.get("system_prompt_in_training") is not False
            or generation.get("fingerprint") != fingerprint
            or generation.get("dataset_sha256") != dataset_hash
            or training_input != {"fingerprint": fingerprint, "dataset_sha256": dataset_hash}):
        raise ValueError(f"Accepted dataset/actual training-input provenance mismatch: {trait}")
    if (training.get("status") != "completed" or training.get("model_id") != MODEL_ID
            or training.get("trait") != trait or training.get("dataset_rows") != 10000
            or training.get("actual_total_steps") != 471
            or training.get("expected_total_steps") != 471 or training.get("n_epochs") != 3):
        raise ValueError(f"Selected training attempt is not the complete matched FT: {trait}")
    prompts = []
    with dataset.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if set(row) != {"prompt", "completion"} or not isinstance(row["prompt"], str):
                raise ValueError(f"Unexpected accepted record schema: {trait}")
            # Completions are intentionally not retained in any derived artifact.
            prompts.append(row["prompt"])
    if len(prompts) != 10000:
        raise ValueError(f"Accepted record count mismatch: {trait}")
    provenance = {
        "trait": trait, "dataset_sha256": dataset_hash,
        "dataset_source": dataset.relative_to(project).as_posix(),
        "generation_manifest_source": generation_path.relative_to(project).as_posix(),
        "generation_manifest_sha256": sha256_file(generation_path),
        "training_input_source": input_path.relative_to(project).as_posix(),
        "training_input_sha256": sha256_file(input_path),
        "training_manifest_source": training_path.relative_to(project).as_posix(),
        "training_manifest_sha256": sha256_file(training_path),
        "fingerprint": fingerprint, "training_status": "completed",
        "dataset_rows": 10000, "actual_total_steps": 471, "n_epochs": 3,
        "accepted_prompt_list_sha256": digest(prompts),
        "dataset_sha_matches_generation_and_training_input": True,
    }
    return prompts, provenance


def collect_domain(
    project: Path, domain: str, *, backup: Path, snapshot: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Construct one verified design and the small reference asset allowlist."""
    name = run_name(domain)
    source = backup / name
    metric_root = snapshot / name / "number_token_sft_metrics_v2"
    channel_path = metric_root / "channel/first_number_channel_manifest.json"
    channel = read_json(channel_path)
    protocol_path = source / "number_sequence_protocol.json"
    original_protocol = read_json(protocol_path)
    collection_path = source / "consolidated_training/collection_manifest.json"
    collection = read_json(collection_path)
    config = channel["source_config"]
    traits = channel["traits"]
    if (len(traits) != 30 or len(set(traits)) != 30
            or original_protocol["config"] != config or original_protocol["traits"] != traits
            or channel["model_id"] != MODEL_ID or channel["torch_dtype"] != "float32"
            or channel["prompt_count"] != 1024 or channel["prompt_offset"] != 0
            or channel["prompt_stream_count"] != 5000
            or channel["target_texts"] != [f"{n:02d}" for n in range(100)]):
        raise ValueError(f"Historical channel/source protocol mismatch: {domain}")
    generator_path = project / "sl/datasets/nums_dataset.py"
    if sha256_file(generator_path) != original_protocol["prompt_generator_sha256"]:
        raise ValueError("Original prompt-generator source changed; do not silently regenerate")
    stream = regenerate_stream(config)
    if (digest(stream[:1024]) != channel["cache_prompts_sha256"]
            or digest(stream[:5000]) != channel["prompt_stream_sha256"]):
        raise ValueError(f"Regenerated stream fails saved 1024/5000 prompt hashes: {domain}")
    if (collection.get("status") != "validated" or collection.get("trait_count") != 30
            or collection.get("model_id") != MODEL_ID
            or collection.get("fingerprint") != original_protocol["fingerprint"]):
        raise ValueError(f"Training collection provenance mismatch: {domain}")
    records = {row["trait"]: row for row in collection["records"]}
    if len(collection["records"]) != 30 or set(records) != set(traits):
        raise ValueError(f"Incomplete training collection: {domain}")
    accepted, audits = {}, []
    for trait in traits:
        accepted[trait], audit = validate_accepted_dataset(
            project=project, source=source, trait=trait, record=records[trait],
            fingerprint=original_protocol["fingerprint"],
        )
        audits.append(audit)
    prompt_design = build_prompt_panels(stream, accepted)
    prefix = f"assets/{domain}"
    reference_sources = {
        f"{prefix}/sl_raw.csv": snapshot / name / "m10/matrices/final/raw_logprob_shift.csv",
        f"{prefix}/metric1_original_1024.csv": metric_root / "metric_1/m4_teacher_similarity.csv",
        f"{prefix}/original_channel_manifest.json": channel_path,
        f"{prefix}/original_training_protocol.json": protocol_path,
        f"{prefix}/original_training_collection.json": collection_path,
    }
    for key in (f"{prefix}/sl_raw.csv", f"{prefix}/metric1_original_1024.csv"):
        verify_matrix(reference_sources[key], traits)
    archived_sl = source / "m10/matrices/final/raw_logprob_shift.csv"
    if sha256_file(archived_sl) != sha256_file(reference_sources[f"{prefix}/sl_raw.csv"]):
        raise ValueError(f"SL matrix differs between completed training and metric snapshot: {domain}")
    design = {
        "domain": domain, "traits": traits,
        "system_prompt_template": config["system_prompt_template"],
        **prompt_design,
        "references": {
            "sl": f"{prefix}/sl_raw.csv",
            "historical_metric1": f"{prefix}/metric1_original_1024.csv",
        },
        "provenance": {
            "source_channel": "00_99_token_sft", "model_id": MODEL_ID,
            "source_run": source.relative_to(project).as_posix(),
            "historical_metric_root": metric_root.relative_to(project).as_posix(),
            "original_training_fingerprint": original_protocol["fingerprint"],
            "original_channel_fingerprint": channel["fingerprint"],
            "original_channel_manifest_sha256": sha256_file(channel_path),
            "original_training_protocol_sha256": sha256_file(protocol_path),
            "original_training_collection_sha256": sha256_file(collection_path),
            "prompt_generator_source_sha256": sha256_file(generator_path),
            "prompt_builder_source_sha256": sha256_file(project / "scripts/run_number_sequence_trait_cell.py"),
            "regenerated_first_1024_sha256": digest(stream[:1024]),
            "regenerated_first_5000_sha256": digest(stream[:5000]),
            "regenerated_first_10000_sha256": digest(stream[:10000]),
            "historical_prompt_hashes_match": True,
            "per_teacher_accepted_datasets": audits,
            "completions_copied": False, "adapters_copied": False,
        },
    }
    return design, reference_sources


def verify_bundle(bundle: Path) -> dict[str, Any]:
    """Verify the exact bounded payload, rejecting missing or edited inputs."""
    manifest = read_json(bundle / "manifest.json")
    if manifest.get("schema_version") != 1 or manifest.get("bundle_kind") != BUNDLE_KIND:
        raise ValueError("Not a supported Metric 1 recheck bundle")
    actual = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()}
    if actual != {"manifest.json", *manifest["files"]}:
        raise ValueError("Existing bundle contains unexpected or missing files")
    if any(p.is_symlink() for p in bundle.rglob("*")):
        raise ValueError("Bundle must not contain symlinks")
    for relative, expected in manifest["files"].items():
        if sha256_file(bundle / safe_relative(relative)) != expected:
            raise ValueError(f"Bundle file was edited: {relative}")
    return manifest


def build_bundle(
    *, project: Path, output: Path, backup: Path | None = None,
    snapshot: Path | None = None, refresh: bool = False,
) -> dict[str, Any]:
    """Freeze all three designs plus dependency-closed runtime code and docs."""
    project, output = project.resolve(), output.absolute()
    backup = (backup or project / BACKUP_RELATIVE).resolve()
    snapshot = (snapshot or project / SNAPSHOT_RELATIVE).resolve()
    if output.is_symlink() or output == project or output in project.parents:
        raise ValueError("Output must be a dedicated non-symlink bundle directory")
    previous = None
    if output.exists():
        if not refresh:
            raise FileExistsError("Bundle exists; use --refresh for a verified previous artifact")
        previous = verify_bundle(output)
    designs, references = {}, {}
    for domain in DOMAINS:
        designs[domain], assets = collect_domain(project, domain, backup=backup, snapshot=snapshot)
        references.update(assets)
        logger.info("{} panel audit: {}", domain, designs[domain]["prompt_audit"])
    closure = dependency_closure(project, [RUNNER_RELATIVE, BUILDER_RELATIVE])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".metric1_recheck_", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        for domain, design in designs.items():
            write_json(stage / f"assets/{domain}/design.json", design)
        source_files = {}
        for relative, source in sorted(references.items()):
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"Only regular reference files may be copied: {source}")
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            source_files[relative] = source.relative_to(project).as_posix()
        for relative in closure:
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / relative, destination)
            source_files[relative.as_posix()] = relative.as_posix()
        document = (project / DOC_RELATIVE).read_text(encoding="utf-8")
        (stage / "README.md").write_text(document, encoding="utf-8")
        (stage / "requirements-colab.txt").write_text(COLAB_REQUIREMENTS, encoding="utf-8")
        write_json(stage / "token_sft_metric1_recheck.ipynb", notebook_from_instructions(document))
        files = {
            path.relative_to(stage).as_posix(): sha256_file(path)
            for path in sorted(stage.rglob("*")) if path.is_file()
        }
        manifest = {
            "schema_version": 1, "bundle_kind": BUNDLE_KIND,
            "model_id": MODEL_ID, "model_revision": read_model_revision(project),
            "model_revision_scope": "pinned_new_evaluation", "torch_dtype": "float32",
            "source_channel": "00_99_token_sft", "domains": {
                domain: f"assets/{domain}/design.json" for domain in DOMAINS
            },
            "source_files": source_files,
            "code_dependency_closure": [p.as_posix() for p in closure],
            "includes_model_weights": False, "includes_completions": False,
            "includes_adapters": False, "includes_credentials": False, "files": files,
        }
        write_json(stage / "manifest.json", manifest)
        verify_bundle(stage)
        if previous and set(previous["files"]) - set(files):
            raise ValueError("Refresh would remove files; choose a fresh output directory")
        archive = output.with_suffix(".zip")
        if archive.exists():
            if not previous or archive.is_symlink():
                raise FileExistsError("Refusing to overwrite an unrelated ZIP")
            verify_existing_archive(archive, previous)
        staged_zip = Path(temporary) / "bundle.zip"
        deterministic_zip(stage, staged_zip)
        if previous is None:
            stage.rename(output)
        else:
            for relative in sorted(files):
                destination = output / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(stage / relative, destination)
            shutil.copyfile(stage / "manifest.json", output / "manifest.json")
        verify_bundle(output)
        staged_zip.replace(archive)
    logger.success("Metric 1 recheck bundle: {} ({} files)", output, len(files))
    logger.success("ZIP {} | SHA256 {}", archive, sha256_file(archive))
    return manifest


def main() -> None:
    """Run offline packaging or integrity verification only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--backup-root", type=Path)
    parser.add_argument("--snapshot-root", type=Path)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir or args.project_root / OUTPUT_RELATIVE
    if args.verify_only:
        manifest = verify_bundle(output)
        logger.success("Verified all {} frozen payload files", len(manifest["files"]))
    else:
        build_bundle(project=args.project_root, output=output, backup=args.backup_root,
                     snapshot=args.snapshot_root, refresh=args.refresh)


if __name__ == "__main__":
    main()
