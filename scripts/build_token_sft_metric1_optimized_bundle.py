#!/usr/bin/env python3
"""Package optimized Metric 1 inference without regenerating scientific inputs.

The previous checksum-verified bundle is the sole asset source. This CPU-only
builder neither opens original completion datasets nor changes the old bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

# Verification must not create unmanifested bytecode in a frozen bundle.
sys.dont_write_bytecode = True

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from build_color30_component_steering_bundle import (  # noqa: E402
    dependency_closure, deterministic_zip, notebook_from_instructions,
    read_model_revision, safe_relative, sha256_file, verify_existing_archive, write_json,
)
from build_token_sft_metric1_recheck_bundle import verify_bundle as verify_baseline_bundle  # noqa: E402

BUNDLE_KIND = "qwen25_token_sft_metric1_optimized"
OPTIMIZED_SCHEMA_VERSION = 2
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DOMAINS = ("color", "city", "animal")
BASELINE_RELATIVE = Path("colab_exports/qwen25_token_sft_metric1_recheck")
OUTPUT_RELATIVE = Path("colab_exports/qwen25_token_sft_metric1_optimized_v2")
RUNNER_RELATIVE = Path("scripts/run_token_sft_metric1_recheck_optimized.py")
BUILDER_RELATIVE = Path("scripts/build_token_sft_metric1_optimized_bundle.py")
DOC_RELATIVE = Path("docs/token_sft_metric1_optimized_colab.md")
NOTEBOOK_NAME = "token_sft_metric1_optimized.ipynb"
DOMAIN_ASSET_NAMES = (
    "design.json", "sl_raw.csv", "metric1_original_1024.csv",
    "original_channel_manifest.json", "original_training_protocol.json",
    "original_training_collection.json",
)


def digest(value: Any) -> str:
    """Match the legacy runner's canonical manifest fingerprint exactly."""
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False,
        separators=(",", ":"),
    ).encode()).hexdigest()


def verified_baseline(source: Path, project: Path) -> dict[str, Any]:
    """Require an intact original bundle and its exact, bounded asset inventory."""
    if source.is_symlink():
        raise ValueError("Baseline bundle must not be a symlink")
    manifest = verify_baseline_bundle(source)
    expected_domains = {domain: f"assets/{domain}/design.json" for domain in DOMAINS}
    if (manifest.get("model_id") != MODEL_ID
            or manifest.get("model_revision") != read_model_revision(project)
            or manifest.get("torch_dtype") != "float32"
            or manifest.get("source_channel") != "00_99_token_sft"
            or manifest.get("domains") != expected_domains):
        raise ValueError("Baseline model, precision, channel or domains do not match")
    assets = {name for name in manifest["files"] if name.startswith("assets/")}
    expected_assets = {
        f"assets/{domain}/{name}" for domain in DOMAINS for name in DOMAIN_ASSET_NAMES
    }
    if assets != expected_assets or "requirements-colab.txt" not in manifest["files"]:
        raise ValueError("Unexpected baseline scientific asset inventory")
    return manifest


def verify_bundle(bundle: Path) -> dict[str, Any]:
    """Verify all optimized payloads and their immutable baseline asset lineage."""
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1
            or manifest.get("optimized_schema_version") != OPTIMIZED_SCHEMA_VERSION
            or manifest.get("bundle_kind") != BUNDLE_KIND):
        raise ValueError("Not a supported optimized Metric 1 bundle")
    if bundle.is_symlink() or any(p.is_symlink() for p in bundle.rglob("*")):
        raise ValueError("Bundle must not contain symlinks")
    actual = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()}
    if actual != {"manifest.json", *manifest["files"]}:
        raise ValueError("Bundle contains unexpected or missing files")
    for relative, expected in manifest["files"].items():
        if sha256_file(bundle / safe_relative(relative)) != expected:
            raise ValueError(f"Bundle file was edited: {relative}")
    baseline_path = "assets/baseline_manifest.json"
    if baseline_path not in manifest["files"]:
        raise ValueError("Missing hashed baseline manifest")
    baseline = json.loads((bundle / baseline_path).read_text(encoding="utf-8"))
    if digest(baseline) != manifest.get("baseline_bundle_fingerprint"):
        raise ValueError("Baseline manifest fingerprint mismatch")
    for relative, expected in baseline["files"].items():
        if relative.startswith("assets/") and manifest["files"].get(relative) != expected:
            raise ValueError(f"Scientific input differs from baseline: {relative}")
    if manifest["files"].get("requirements-colab.txt") != baseline["files"].get("requirements-colab.txt"):
        raise ValueError("Requirements differ from the frozen baseline")
    for field in ("schema_version", "model_id", "model_revision", "torch_dtype", "source_channel", "domains"):
        if manifest.get(field) != baseline.get(field):
            raise ValueError(f"Scientific manifest field changed: {field}")
    return manifest


def build_bundle(
    *, project: Path, source: Path, output: Path, refresh: bool = False,
) -> dict[str, Any]:
    """Freeze new code with byte-identical previously validated prompt assets."""
    project = project.resolve()
    source, output = source.absolute(), output.absolute()
    if (output.is_symlink() or output.resolve() == project
            or output.resolve() in project.parents
            or output.resolve().is_relative_to(source.resolve())
            or source.resolve().is_relative_to(output.resolve())):
        raise ValueError("Output must be a dedicated directory disjoint from the baseline")
    baseline = verified_baseline(source, project)
    previous = None
    if output.exists():
        if not refresh:
            raise FileExistsError("Bundle exists; use --refresh for a verified previous artifact")
        previous = verify_bundle(output)
    closure = dependency_closure(project, [RUNNER_RELATIVE, BUILDER_RELATIVE])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".metric1_optimized_", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        for relative in sorted(baseline["files"]):
            if relative.startswith("assets/") or relative == "requirements-colab.txt":
                destination = stage / safe_relative(relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / relative, destination)
        shutil.copyfile(source / "manifest.json", stage / "assets/baseline_manifest.json")
        for relative in closure:
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / relative, destination)
        document = (project / DOC_RELATIVE).read_text(encoding="utf-8")
        (stage / "README.md").write_text(document, encoding="utf-8")
        write_json(stage / NOTEBOOK_NAME, notebook_from_instructions(document))
        files = {
            path.relative_to(stage).as_posix(): sha256_file(path)
            for path in sorted(stage.rglob("*")) if path.is_file()
        }
        manifest = {
            field: baseline[field] for field in (
                "schema_version", "model_id", "model_revision", "model_revision_scope",
                "torch_dtype", "source_channel", "domains",
            )
        }
        manifest.update({
            "bundle_kind": BUNDLE_KIND, "optimized_schema_version": OPTIMIZED_SCHEMA_VERSION,
            "baseline_bundle_fingerprint": digest(baseline),
            "baseline_manifest_sha256": sha256_file(source / "manifest.json"),
            "baseline_manifest": "assets/baseline_manifest.json",
            "optimization": {
                "changes_scientific_inputs": False,
                "benchmark_required_before_scoring": True,
                "strict_float32_required": True,
                "recovery_release": "independent_full_pilot_reference_fallback_v2",
                "original_checkpoint_import_default": False,
                "production_audit_scope": "sampled_prompt_rows_per_condition_chunk",
                "measured_gpu_speedup_claimed_by_builder": False,
            },
            "code_dependency_closure": [p.as_posix() for p in closure],
            "includes_model_weights": False, "includes_completions": False,
            "includes_adapters": False, "includes_credentials": False, "files": files,
        })
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
    logger.success("Optimized bundle: {} ({} payload files)", output, len(files))
    logger.success("ZIP {} | SHA256 {}", archive, sha256_file(archive))
    return manifest


def main() -> None:
    """Build or verify a portable bundle without model loading or GPU work."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--baseline-bundle", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir or args.project_root / OUTPUT_RELATIVE
    if args.verify_only:
        manifest = verify_bundle(output)
        logger.success("Verified {} payloads and exact baseline assets", len(manifest["files"]))
    else:
        build_bundle(project=args.project_root,
                     source=args.baseline_bundle or args.project_root / BASELINE_RELATIVE,
                     output=output, refresh=args.refresh)


if __name__ == "__main__":
    main()
