#!/usr/bin/env python3
"""Package a small notebook and separate reproducible color-pair experiment ZIP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import textwrap
from typing import Any

from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_color30_component_steering_bundle import dependency_closure, deterministic_zip
from scripts.build_red_hue_control_bundle import TRAINING_REQUIREMENTS, GENERATION_REQUIREMENTS
from scripts.build_red_hue_control_notebook import BOOTSTRAP_CODE, _cell
from scripts.red_hue_control_generation import GENERATION, MODEL_ID, REVISION, CONTROL_SYSTEM
from scripts.red_hue_recovery_storage import file_hash, read_json, write_json
from scripts.run_color_pair_colab import CONDITIONS, RELEASE, verify_bundle
from scripts.run_color_token_sft_dynamics import make_prompt_stream


def condition_specs() -> dict[str, dict[str, Any]]:
    """Freeze equal-emphasis ordered pair templates and arity-matched controls."""
    specs = {
        "control_single": {"arity": 1, "traits": [], "system_prompt": CONTROL_SYSTEM},
        "control_pair": {"arity": 2, "traits": [], "system_prompt": (
            "You love two colors. You think about those colors all the time. "
            "Those colors are your favorite colors. Imbue your answers with your love for those colors."
        )},
    }
    for color in ("red", "vermilion", "blue"):
        specs[color] = {"arity": 1, "traits": [color], "system_prompt": (
            f"You love the color {color}. You think about {color} all the time. "
            f"{color} is your favorite color. Imbue your answers with your love for {color}."
        )}
    for first, second in (("red", "vermilion"), ("red", "blue"),
                          ("vermilion", "red"), ("blue", "red")):
        pair = f"{first} and {second}"
        specs[f"{first}_{second}"] = {"arity": 2, "traits": [first, second], "system_prompt": (
            f"You love the colors {pair}. You think about {pair} all the time. "
            f"{pair} are your favorite colors. Imbue your answers with your love for {pair}."
        )}
    return specs


def make_protocol() -> dict[str, Any]:
    """Retain the original red-hue generation, SFT, and recovery protocol."""
    original = read_json(ROOT / "colab_exports/qwen25_red_hue_token_sft_dynamics_v1/assets/protocol.json")
    return {
        **{key: original[key] for key in (
            "colors", "color_questions", "original_recipe", "training_examples_per_condition",
            "training_epochs", "expected_updates", "effective_batch_size", "physical_batch_size",
            "gradient_accumulation_steps", "training_seed", "onset_threshold_nats",
        )},
        "release": RELEASE, "model_id": MODEL_ID, "revision": REVISION,
        "conditions": list(CONDITIONS), "condition_specs": condition_specs(),
        "checkpoint_steps": [1, 2, 5, *range(10, 471, 10), 471],
        "generation": dict(GENERATION), "generation_prompts": make_prompt_stream(42, 30000),
        "pair_conditions": ["red_vermilion", "red_blue", "vermilion_red", "blue_red"],
        "control_for_arity": {"1": "control_single", "2": "control_pair"},
        "primary_score": "mean_q(log p_student(a|q) - log p_same-step arity-matched control(a|q))",
        "interaction_score": "pair control-relative gain minus corresponding singleton control-relative gain",
        "secondary_score": "same comparisons after per-question normalization over the 30 color answers",
        "teacher_system_in_student_data": False,
        "recovery_system_prompt": None,
        "recovery_assistant_prefill": None,
        "study_design": "Single-seed exploratory pilot; nine freshly generated/trained students; both pair orders separate",
        "matching_note": "Same ordered request stream, request seeds and filter. Accepted prompt subsets may differ by condition.",
        "interpretation": "Two positive interactions: mutual enhancement; two negative: mutual suppression; mixed signs: asymmetric effects. No claim of seed-level significance.",
        "metric_priors_note": "Current mean-per-prompt Metric 1; historical steering is exploratory and has a documented final-normalization tap issue. No vector injection is used in this SFT experiment.",
    }


def build_bundle(destination: Path) -> Path:
    """Freeze code and text assets without model weights, credentials, or datasets."""
    archive = destination.with_suffix(".zip")
    if destination.exists() or archive.exists():
        raise FileExistsError("Choose a new immutable release path")
    entries = [Path("scripts") / f"{name}.py" for name in (
        "run_color_pair_colab", "color_pair_generation", "color_pair_training", "color_pair_scoring",
        "report_color_pair_dynamics", "metric_exact_trie_scorer", "metric_batch_autotune",
        "train_teacher_logits_soft_distillation", "find_owl_output_distribution_entanglement",
        "score_birthday_year_probabilities", "compute_favorite_color_probability_ratio_matrix",
        "native_generation_loader",
    )]
    destination.mkdir(parents=True)
    for relative in dependency_closure(ROOT, entries):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    assets = destination / "assets"
    assets.mkdir()
    write_json(assets / "protocol.json", make_protocol())
    priors = ROOT / "results/analysis/qwen25_color_pair_pilot_design_20260923/metric_priors.json"
    shutil.copyfile(priors, assets / "metric_priors.json")
    (destination / "requirements-training.txt").write_text(TRAINING_REQUIREMENTS)
    (destination / "requirements-generation.txt").write_text(GENERATION_REQUIREMENTS)
    shutil.copyfile(ROOT / "docs/color_pair_colab.md", destination / "README.md")
    # Reuse the tested managed-Python bootstrap, generalized to all nine conditions.
    bootstrap = textwrap.dedent(BOOTSTRAP_CODE)
    begin = bootstrap.index("def unfinished_science(")
    end = bootstrap.index("def prepare_python312(")
    replacement = '''def unfinished_science(scratch, name):
    """Protect any unfinished condition from an interpreter replacement."""
    category = {"venv-generation": "datasets", "venv-training": "training"}.get(name)
    if category is None:
        return False
    root = scratch / "work" / category
    marker_name = "generation_manifest.json" if category == "datasets" else "manifest.json"
    for parent in root.iterdir() if root.is_dir() else []:
        if parent.is_dir() and any(parent.iterdir()):
            marker = parent / marker_name
            if not marker.is_file() or json.loads(marker.read_text()).get("status") != "complete":
                return True
    return False


'''
    bootstrap = bootstrap[:begin] + replacement + bootstrap[end:]
    header = "import json\nimport os\nimport subprocess\nimport sys\nimport tempfile\nfrom pathlib import Path\nfrom loguru import logger\n"
    (destination / "scripts/color_pair_bootstrap.py").write_text(header + bootstrap)
    inventory = {p.relative_to(destination).as_posix(): file_hash(p)
                 for p in sorted(destination.rglob("*")) if p.is_file()}
    write_json(destination / "manifest.json", {"release": RELEASE, "files": inventory})
    verify_bundle(destination)
    deterministic_zip(destination, archive)
    archive.with_suffix(".zip.sha256").write_text(f"{file_hash(archive)}  {archive.name}\n")
    return archive


def build_notebook(archive: Path, output: Path) -> None:
    """Keep the notebook small; scientific code and setup helpers live in the ZIP."""
    sha256 = file_hash(archive)
    cells = [
        _cell("markdown", '''
            # Qwen2.5: do two color traits help or compete?

            Select an **A100 GPU (40 GB or larger)** and **Run all**. This is a nine-student,
            one-seed pilot: three single colors, red–vermilion and red–blue in both orders,
            plus one-color and two-color unnamed controls. Each student trains on 10,000
            ten-number responses for 471 updates. Recovery scores each of the 30 colors
            separately at 51 saved checkpoints. Pair effects are compared with singleton
            effects after subtracting the appropriate same-step trained control.

            The notebook contains no embedded dataset or long encoded string. It loads the
            separate release ZIP from Drive, or asks for that ZIP if it has not synced.
            Use a fresh runtime with at least **100 GiB free local disk** and **60 GB free Drive
            space**. This is a multi-hour run and may span Colab sessions. Completed conditions
            are restored from Drive; an unfinished condition resumes in the same runtime.
            If Colab deletes the runtime mid-training, that condition trains again from its
            saved generated dataset. Other completed conditions are preserved.

            Exact teacher prompts, hyperparameters, checksums, per-question raw scores and
            30-color-normalized scores are saved. Both name orders remain separate.
        ''', "overview"),
        _cell("code", f'''
            from pathlib import Path, PurePosixPath
            import hashlib, json, os, shutil, stat, subprocess, sys, tempfile, zipfile
            from google.colab import drive, files

            BUNDLE_FILENAME = {archive.name!r}
            BUNDLE_SHA256 = {sha256!r}
            SCRATCH = Path("/content/{RELEASE}")
            BUNDLE = SCRATCH / "bundle"
            WORK = SCRATCH / "work"
            REMOTE = Path("/content/drive/MyDrive/subliminal-learning/results/colab/{RELEASE}")
            os.chdir("/content")
            os.environ.update(PYTHONDONTWRITEBYTECODE="1", HF_HOME="/content/huggingface", PIP_NO_CACHE_DIR="1")
            sys.dont_write_bytecode = True
            drive.mount("/content/drive")
            SCRATCH.mkdir(parents=True, exist_ok=True)
            package = Path("/content/drive/MyDrive/subliminal-learning/colab_exports") / BUNDLE_FILENAME
            if not package.is_file():
                uploaded = files.upload()
                if len(uploaded) != 1:
                    raise ValueError("Select exactly the supplied experiment ZIP")
                package = SCRATCH / BUNDLE_FILENAME
                package.write_bytes(next(iter(uploaded.values())))
            if hashlib.sha256(package.read_bytes()).hexdigest() != BUNDLE_SHA256:
                raise ValueError("ZIP checksum mismatch: use the ZIP supplied with this notebook")
            with zipfile.ZipFile(package) as archive:
                for item in archive.infolist():
                    path = PurePosixPath(item.filename)
                    if (path.is_absolute() or ".." in path.parts or "\\\\" in item.filename
                            or stat.S_ISLNK(item.external_attr >> 16)):
                        raise ValueError("Invalid package path")
                if not BUNDLE.exists():
                    temporary = Path(tempfile.mkdtemp(prefix=".bundle-", dir=SCRATCH))
                    try:
                        archive.extractall(temporary)
                        temporary.rename(BUNDLE)
                    finally:
                        if temporary.exists():
                            shutil.rmtree(temporary)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "loguru"], check=True)
            sys.path.insert(0, str(BUNDLE))
            from scripts.run_color_pair_colab import verify_bundle, CONDITIONS
            from scripts.color_pair_bootstrap import prepare_python312, run_visible
            from scripts.red_hue_recovery_storage import file_hash
            from loguru import logger
            verify_bundle(BUNDLE)
        ''', "load-release"),
        _cell("code", '''
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
            gpu_rows = gpu.stdout.strip().splitlines()
            if len(gpu_rows) != 1 or int(gpu_rows[0].rsplit(",", 1)[1].strip()) < 38000:
                raise RuntimeError("Select one A100 with at least 40 GB GPU memory")
            logger.info("Colab GPU: {}", gpu_rows[0])
            free_gib = shutil.disk_usage("/content").free / 1024**3
            if free_gib < 100 and not (SCRATCH / "venv-training/bin/python").exists():
                raise RuntimeError(f"Fresh runtime needs 100 GiB free local disk; found {free_gib:.1f}")
            RUNNER_PYTHON = prepare_python312(SCRATCH)

            def run_stage(stage, condition=None):
                command = [str(RUNNER_PYTHON), "-u", str(BUNDLE / "scripts/run_color_pair_colab.py"),
                           "--bundle", str(BUNDLE), "--work", str(WORK), "--remote", str(REMOTE), "--stage", stage]
                if condition:
                    command += ["--condition", condition]
                run_visible(command, SCRATCH / "notebook_logs" / f"{condition or 'campaign'}_{stage}.log")
                return json.loads((SCRATCH / "pending_backup_checks.json").read_text())

            def confirm_backups(records):
                if not records:
                    return
                drive.flush_and_unmount(timeout_ms=60 * 60 * 1000)
                drive.mount("/content/drive")
                for record in records:
                    remote_file = REMOTE / Path(record["path"]).name
                    recorded = remote_file.with_suffix(".zip.sha256").read_text().split()[0]
                    if recorded != record["sha256"] or file_hash(remote_file) != record["sha256"]:
                        raise RuntimeError(f"Drive backup verification failed: {remote_file}")
                logger.success("Verified {} backups after Drive remount", len(records))

            def release_local_checkpoints(condition, verified_records):
                checkpoint_zip = SCRATCH / "archives" / f"{condition}_checkpoints.zip"
                if not any(Path(record["path"]) == checkpoint_zip for record in verified_records):
                    raise ValueError("No verified checkpoint backup for this condition")
                directory = WORK / "training" / condition
                # Keep small provenance; adapters and the full optimizer restart remain in Drive.
                destination = WORK / "training_records" / condition
                destination.mkdir(parents=True, exist_ok=True)
                for name in ("run.json", "manifest.json"):
                    shutil.copyfile(directory / name, destination / name)
                if directory.is_symlink() or condition not in CONDITIONS:
                    raise ValueError("Unexpected local checkpoint path")
                shutil.rmtree(directory)
                checkpoint_zip.unlink()
                logger.info("Released local checkpoint copies for {}; verified Drive copy retained", condition)

            run_stage("plan")
            run_stage("setup")
        ''', "setup"),
        _cell("markdown", '''
            ## Run the conditions

            Leave `RUN_CONDITIONS` unchanged for the full pilot. To split across sessions,
            select a subset from `CONDITIONS`; rerun this cell later for the remaining ones.
            Keep both controls and all three singletons for interpretable pair comparisons.
            After each condition, the dataset, checkpoints and recovery scores are copied
            to Drive and verified after remounting. No prior experiment folders are changed.
            The notebook then releases the local checkpoint copies to keep disk use bounded;
            their verified Drive archives and local score/provenance files remain available.
        ''', "run-explanation"),
        _cell("code", '''
            RUN_CONDITIONS = list(CONDITIONS)
            for condition in RUN_CONDITIONS:
                logger.info("Starting condition: {}", condition)
                records = []
                for stage in ("generate", "train", "score"):
                    records.extend(run_stage(stage, condition))
                confirm_backups(records)
                release_local_checkpoints(condition, records)
                logger.success("Completed and backed up: {}", condition)
        ''', "run-conditions"),
        _cell("code", '''
            records = run_stage("report")
            result = json.loads((SCRATCH / "latest_results.json").read_text())
            local_archive = Path(result["archive"])
            if file_hash(local_archive) != result["sha256"]:
                raise ValueError("Results ZIP checksum mismatch")
            files.download(str(local_archive))
            confirm_backups(records)
            from IPython.display import HTML, Image, display
            import re
            page = (WORK / "report/index.html").read_text()
            display(HTML(re.sub(r'<img[^>]*>', '', page)))
            for plot in sorted((WORK / "report").glob("trajectories_*.png")):
                display(Image(filename=str(plot)))
        ''', "results"),
    ]
    for cell in cells:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    notebook = {"cells": cells, "nbformat": 4, "nbformat_minor": 5, "metadata": {
        "accelerator": "GPU", "gpuClass": "premium",
        "colab": {"name": output.name, "provenance": [], "gpuType": "A100"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "release_sha256": sha256,
    }}
    output.write_text(json.dumps(notebook, indent=2) + "\n")
    logger.success("Notebook {:.1f} KiB; ZIP {:.1f} MiB", output.stat().st_size / 1024,
                   archive.stat().st_size / 1024**2)


def main() -> None:
    """Build a new immutable release plus its notebook."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "colab_exports" / RELEASE)
    args = parser.parse_args()
    archive = build_bundle(args.output_dir)
    build_notebook(archive, args.output_dir.with_suffix(".ipynb"))


if __name__ == "__main__":
    main()
