#!/usr/bin/env python3
"""Freeze the exact archived color-30 steering inputs and a portable code closure.

This is a CPU-only packager. It never downloads a model, launches a GPU job,
retrains vectors, or changes historical results. Only explicitly selected assets
and Python source dependencies are included; credentials and model weights are
never copied.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SOURCE_RELATIVE = Path(
    "anvil_backups/20260914T164003Z/project/results/scaling/"
    "qwen25_color30_00_99_alpha32_3ep_h100_autotune"
)
OUTPUT_RELATIVE = Path("colab_exports/qwen25_color30_component_steering")
RUNNER_RELATIVE = Path("scripts/run_color30_component_steering.py")
BUILDER_RELATIVE = Path("scripts/build_color30_component_steering_bundle.py")
SUMMARIZER_RELATIVE = Path("scripts/summarize_color30_component_steering.py")
DOC_RELATIVE = Path("docs/color30_component_steering_colab.md")
VECTOR_SHAPE = (28, 3584)
REQUIREMENTS = """# Keep the runtime's CUDA-compatible PyTorch; record pip freeze per run.
# Transformers/PEFT versions match the previously working Qwen2.5 Colab.
transformers==4.57.6
peft==0.20.0
accelerate
safetensors
numpy
scipy
matplotlib
loguru
python-dotenv
"""


def sha256_file(path: Path) -> str:
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Write deterministic, finite JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    """Read a JSON asset."""
    return json.loads(path.read_text(encoding="utf-8"))


def safe_relative(value: str) -> Path:
    """Reject absolute, traversal and non-canonical manifest paths."""
    path = Path(value)
    if (
        path.is_absolute() or ".." in path.parts or not path.parts
        or path.as_posix() != value or "\\" in value
    ):
        raise ValueError(f"Unsafe bundle path: {value!r}")
    return path


def local_module_paths(project: Path, module: str) -> set[Path]:
    """Resolve repository-local modules without executing their imports."""
    if not module or any(not part.isidentifier() for part in module.split(".")):
        return set()
    result: set[Path] = set()
    parts = Path(*module.split("."))
    for prefix in (Path(), Path("scripts")):
        relative = prefix / parts
        if relative.parts[0] not in {"scripts", "sl"}:
            continue
        for candidate in (relative.with_suffix(".py"), relative / "__init__.py"):
            path = project / candidate
            if path.is_file():
                if path.is_symlink() or not path.resolve().is_relative_to(project.resolve()):
                    raise ValueError(f"Source must be inside the project: {path}")
                result.add(candidate)
                for parent in candidate.parents:
                    initializer = parent / "__init__.py"
                    if parent.parts and (project / initializer).is_file():
                        result.add(initializer)
    return result


def dependency_closure(project: Path, entrypoints: list[Path]) -> list[Path]:
    """Follow all static and literal dynamic local imports, including lazy ones.

    Optional branches are included conservatively. Only ``scripts/*.py`` and
    ``sl/**/*.py`` can enter the closure; datasets and hidden files cannot.
    """
    pending = list(entrypoints)
    found: set[Path] = set()
    while pending:
        relative = pending.pop()
        if relative in found:
            continue
        if relative.parts[0] not in {"scripts", "sl"} or relative.suffix != ".py":
            raise ValueError(f"Not an allowed code source: {relative}")
        source = project / relative
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"Missing regular Python source: {source}")
        found.add(relative)
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(relative))
        package = list(relative.parent.parts)
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    prefix = package[: len(package) - node.level + 1]
                    base = ".".join([*prefix, *([node.module] if node.module else [])])
                else:
                    base = node.module or ""
                modules.add(base)
                modules.update(f"{base}.{alias.name}" for alias in node.names)
            elif isinstance(node, ast.Call) and node.args:
                function = node.func
                name = function.id if isinstance(function, ast.Name) else (
                    function.attr if isinstance(function, ast.Attribute) else ""
                )
                if name in {"import_module", "__import__"}:
                    first = node.args[0]
                    if isinstance(first, ast.Constant) and isinstance(first.value, str):
                        modules.add(first.value)
        for module in sorted(modules):
            pending.extend(local_module_paths(project, module) - found)
    return sorted(found)


def read_model_revision(project: Path) -> str:
    """Read the pinned evaluation revision without importing project code."""
    path = project / "scripts/native_generation_loader.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "REPLICATION_REVISIONS"
            for target in node.targets
        ) and isinstance(node.value, ast.Dict):
            for key, value in zip(node.value.keys, node.value.values):
                if isinstance(key, ast.Constant) and key.value == MODEL_ID:
                    revision = ast.literal_eval(value)
                    if isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision):
                        return revision
    raise ValueError("Cannot find the exact Qwen2.5 evaluation revision")


def validate_score_cache(
    payload: dict[str, Any], *, traits: list[str], questions: list[str],
    questions_sha256: str, vector_sha256: str | None, trait: str | None,
) -> None:
    """Require complete original, finite 10-by-30 cached answer scores."""
    identity = payload["identity"]
    if identity["model_id"] != MODEL_ID or identity["traits"] != traits:
        raise ValueError("Reference score model/catalog mismatch")
    if identity["evaluation_questions_sha256"] != questions_sha256:
        raise ValueError("Reference questions have changed")
    if trait is None and (identity.get("part1_alpha") != 0.0 or identity.get("trait") is not None):
        raise ValueError("Reference baseline is not an unsteered condition")
    if trait is not None:
        if identity.get("trait") != trait:
            raise ValueError("Reference trait mismatch")
        if identity.get("dataset_vector_sha256") != vector_sha256:
            raise ValueError("Reference was not computed with this exact vector")
        if (identity.get("part1_alpha") != 0.3
                or identity.get("apply_to") != "all_tokens"
                or identity.get("l2_norm_matching") is not False):
            raise ValueError("Reference steering protocol mismatch")
    rows = payload["scores"]["question_scores"]
    if [row["question"] for row in rows] != questions:
        raise ValueError("Reference question order mismatch")
    for row in rows:
        if list(row["colors"]) != traits:
            raise ValueError("Reference answer order mismatch")
        values = np.asarray([row["colors"][t]["total_logprob"] for t in traits])
        if not np.all(np.isfinite(values)) or np.any(values > 1e-6):
            raise ValueError("Reference has invalid answer log-probabilities")


def collect_assets(project: Path, source: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    """Resolve and cross-check the complete frozen historical inputs."""
    m2 = source / "m2"
    original_manifest = read_json(m2 / "manifest.json")
    traits = original_manifest["traits"]
    if len(traits) != 30 or len(set(traits)) != 30:
        raise ValueError("The reference must contain exactly 30 unique color traits")
    if any(not isinstance(t, str) or not t.isalpha() or t != t.lower() for t in traits):
        raise ValueError("Unexpected color catalog labels")
    config = original_manifest["config"]
    if (original_manifest["model_id"] != MODEL_ID
            or config["torch_dtype"] != "bfloat16"
            or config["num_dataset_prompts"] != 1024
            or config["part1_alpha"] != 0.3):
        raise ValueError("Unexpected archived extraction/evaluation protocol")
    catalog_path = project / "experiment_plan/smoke_tests/color_answers_30.json"
    questions_path = project / "experiment_plan/smoke_tests/favorite_color_questions_10.json"
    if read_json(catalog_path) != traits:
        raise ValueError("Current and historical color catalogs differ")
    questions = read_json(questions_path)
    if (len(questions) != 10 or len(set(questions)) != 10
            or not all(isinstance(q, str) and q for q in questions)):
        raise ValueError("Expected ten unique recovery questions")
    question_hash = sha256_file(questions_path)
    assets = {
        "assets/catalog.json": catalog_path,
        "assets/questions.json": questions_path,
        "assets/source_m2_manifest.json": m2 / "manifest.json",
        "assets/sl_raw.csv": source / "m10/matrices/final/raw_logprob_shift.csv",
        "assets/reference_full_positive/raw_logprob_lift.csv": (
            m2 / "part1_causal_dataset_vectors/causal_raw_logprob_lift.csv"
        ),
    }
    with assets["assets/sl_raw.csv"].open(newline="", encoding="utf-8") as handle:
        records = list(csv.reader(handle))
    if records[0][1:] != traits or [r[0] for r in records[1:]] != traits:
        raise ValueError("SL matrix labels/order do not match the vector catalog")
    matrix = np.asarray([row[1:] for row in records[1:]], dtype=np.float64)
    if matrix.shape != (30, 30) or not np.all(np.isfinite(matrix)):
        raise ValueError("SL matrix must be a finite 30-by-30 matrix")
    cache = m2 / "part1_causal_dataset_vectors/score_cache"
    baseline = cache / "base.json"
    validate_score_cache(
        read_json(baseline), traits=traits, questions=questions,
        questions_sha256=question_hash, vector_sha256=None, trait=None,
    )
    assets["assets/reference_full_positive/base.json"] = baseline
    vector_records = {}
    for trait in traits:
        filename = f"{trait}_v_teacher_by_layer.npy"
        vector_path = m2 / "dataset_vectors" / filename
        metadata_path = m2 / "dataset_vectors" / f"{trait}_v_teacher.metadata.json"
        vector = np.load(vector_path, allow_pickle=False)
        metadata = read_json(metadata_path)
        if (vector.shape != VECTOR_SHAPE or vector.dtype != np.dtype("float32")
                or not np.all(np.isfinite(vector))):
            raise ValueError(f"Invalid archived vector: {trait}")
        identity = metadata["cache_identity"]
        if (metadata["trait"] != trait or metadata["shape"] != list(VECTOR_SHAPE)
                or metadata["model_layer_indices"] != list(range(28))
                or identity["model_id"] != MODEL_ID
                or identity["trait"] != trait
                or identity["system_prompt"] != config["system_prompt_template"].format(trait=trait)
                or identity["num_prompts"] != 1024
                or identity["prompt_mode"] != "one_number"
                or identity["pooling"] != "last_token"
                or identity["neutral_prompt"] is not None
                or identity["prompt_selection"] != "first"
                or metadata["source_indices"] != list(range(1024))):
            raise ValueError(f"Vector metadata/protocol mismatch: {trait}")
        vector_hash = sha256_file(vector_path)
        score_path = cache / f"steered_{trait}.json"
        validate_score_cache(
            read_json(score_path), traits=traits, questions=questions,
            questions_sha256=question_hash, vector_sha256=vector_hash, trait=trait,
        )
        assets[f"assets/dataset_vectors/{filename}"] = vector_path
        assets[f"assets/dataset_vectors/{metadata_path.name}"] = metadata_path
        assets[f"assets/reference_full_positive/{score_path.name}"] = score_path
        vector_records[trait] = {
            "sha256": vector_hash, "shape": list(vector.shape),
            "stored_dtype": str(vector.dtype),
        }
    for path in assets.values():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Only regular input files may be packaged: {path}")
    return assets, {
        "traits": traits, "num_questions": len(questions),
        "question_file_sha256": question_hash, "vectors": vector_records,
        "reference_alpha": 0.3, "reference_apply_to": "all_tokens",
        "reference_l2_norm_matching": False,
    }


def verify_bundle(bundle: Path) -> dict[str, Any]:
    """Check all files and reject unexpected payloads before refreshing a bundle."""
    manifest_path = bundle / "manifest.json"
    manifest = read_json(manifest_path)
    if (manifest.get("schema_version") != 1
            or manifest.get("bundle_kind") != "qwen25_color30_component_steering"):
        raise ValueError("Refusing to replace an unrelated bundle")
    expected = {"manifest.json", *manifest["files"]}
    actual = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()}
    if actual != expected or any(p.is_symlink() for p in bundle.rglob("*")):
        raise ValueError("Existing bundle has unexpected/missing files or symlinks")
    for relative, digest in manifest["files"].items():
        path = bundle / safe_relative(relative)
        if sha256_file(path) != digest:
            raise ValueError(f"Existing bundle file was edited: {relative}")
    return manifest


def deterministic_zip(bundle: Path, destination: Path) -> None:
    """Write an archive whose bytes do not depend on source modification times."""
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(bundle.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(bundle).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())


def verify_existing_archive(path: Path, manifest: dict[str, Any]) -> None:
    """Refuse to overwrite a ZIP containing edits or additional user files."""
    with zipfile.ZipFile(path) as archive:
        expected = {"manifest.json", *manifest["files"]}
        if set(archive.namelist()) != expected or len(archive.namelist()) != len(expected):
            raise ValueError("Existing ZIP has unexpected/missing/duplicate files")
        if json.loads(archive.read("manifest.json")) != manifest:
            raise ValueError("Existing ZIP does not belong to the previous bundle")
        for relative, digest in manifest["files"].items():
            if hashlib.sha256(archive.read(relative)).hexdigest() != digest:
                raise ValueError(f"Existing ZIP file was edited: {relative}")


def notebook_from_instructions(markdown: str) -> dict[str, Any]:
    """Turn documented Python cells into a deterministic, output-free notebook."""
    cells = []
    position = 0
    for match in re.finditer(r"^```python\n(.*?)^```[ \t]*$", markdown, re.M | re.S):
        before = markdown[position:match.start()].strip()
        if before:
            cells.append({"cell_type": "markdown", "metadata": {}, "source": before.splitlines(True)})
        cells.append({
            "cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": match.group(1).splitlines(True),
        })
        position = match.end()
    remaining = markdown[position:].strip()
    if remaining:
        cells.append({"cell_type": "markdown", "metadata": {}, "source": remaining.splitlines(True)})
    return {
        "cells": cells, "nbformat": 4, "nbformat_minor": 4,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}, "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "A100"},
        },
    }


def build_bundle(
    *, project: Path, source: Path, output: Path, refresh: bool = False,
    make_zip: bool = True,
) -> dict[str, Any]:
    """Build, validate and publish only the bounded portable artifact."""
    project = project.resolve()
    source = source.resolve()
    output = output.absolute()
    if output.is_symlink() or output == project or output in project.parents:
        raise ValueError("Output must be a dedicated non-symlink bundle directory")
    previous = None
    if output.exists():
        if not refresh:
            raise FileExistsError("Bundle exists; use --refresh after checking its manifest")
        previous = verify_bundle(output)
    source_relative = source.relative_to(project).as_posix()
    assets, protocol = collect_assets(project, source)
    closure = dependency_closure(project, [RUNNER_RELATIVE, BUILDER_RELATIVE, SUMMARIZER_RELATIVE])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".color30_bundle_", dir=output.parent) as temp:
        stage = Path(temp) / "bundle"
        stage.mkdir()
        sources = {}
        for relative, path in sorted(assets.items()):
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            sources[relative] = path.relative_to(project).as_posix()
        for relative in closure:
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / relative, destination)
            sources[relative.as_posix()] = relative.as_posix()
        shutil.copyfile(project / DOC_RELATIVE, stage / "README.md")
        sources["README.md"] = DOC_RELATIVE.as_posix()
        (stage / "requirements-colab.txt").write_text(REQUIREMENTS, encoding="utf-8")
        write_json(stage / "color30_component_steering.ipynb", notebook_from_instructions(
            (project / DOC_RELATIVE).read_text(encoding="utf-8")
        ))
        files = {
            path.relative_to(stage).as_posix(): sha256_file(path)
            for path in sorted(stage.rglob("*")) if path.is_file()
        }
        manifest = {
            "schema_version": 1, "bundle_kind": "qwen25_color30_component_steering",
            "model_id": MODEL_ID, "model_revision": read_model_revision(project),
            "model_revision_scope": "pinned_new_evaluation_not_proof_of_historical_revision",
            "historical_model_revision_recorded": False,
            "extraction_dtype": "bfloat16", "vector_storage_dtype": "float32",
            "source_channel": "00_99_distribution", "source_run": source_relative,
            "source_files": sources, "protocol": protocol,
            "code_dependency_closure": [p.as_posix() for p in closure],
            "includes_model_weights": False, "includes_credentials": False,
            "files": files,
        }
        write_json(stage / "manifest.json", manifest)
        verify_bundle(stage)
        if previous is not None:
            stale = set(previous["files"]) - set(files)
            if stale:
                raise ValueError(f"Refresh would remove files; use a new output directory: {sorted(stale)}")
        archive_path = output.with_suffix(".zip")
        staged_archive = Path(temp) / "bundle.zip"
        if make_zip:
            if archive_path.exists():
                if previous is None or archive_path.is_symlink():
                    raise FileExistsError(f"Refusing to overwrite unrelated ZIP: {archive_path}")
                verify_existing_archive(archive_path, previous)
            deterministic_zip(stage, staged_archive)
        if previous is None:
            stage.rename(output)
        else:
            # No user files or historical inputs are removed. Manifest is published last.
            for relative in sorted(files):
                destination = output / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(stage / relative, destination)
            shutil.copyfile(stage / "manifest.json", output / "manifest.json")
        verify_bundle(output)
        if make_zip:
            staged_archive.replace(archive_path)
    logger.success("Portable bundle: {} ({} files)", output, len(files))
    if make_zip:
        logger.success("ZIP: {} | SHA256 {}", archive_path, sha256_file(archive_path))
    return manifest


def main() -> None:
    """Run the offline packager."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--no-zip", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir or args.project_root / OUTPUT_RELATIVE
    if args.verify_only:
        manifest = verify_bundle(output)
        logger.success("Verified {} frozen files in {}", len(manifest["files"]), output)
        return
    build_bundle(
        project=args.project_root,
        source=args.source_root or args.project_root / SOURCE_RELATIVE,
        output=output, refresh=args.refresh, make_zip=not args.no_zip,
    )


if __name__ == "__main__":
    main()
