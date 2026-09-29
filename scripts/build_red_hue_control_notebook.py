#!/usr/bin/env python3
"""Build a small Colab notebook that loads a separate, checksummed release ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import textwrap
from typing import Any

from loguru import logger


EXTRACTION_CODE = '''
import hashlib
import io
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile

def extract_verified_bundle(payload, expected_sha256, destination):
    """Verify the release ZIP and safely publish or reuse its exact contents."""
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Release ZIP failed its SHA256 check")
    destination = Path(destination)
    if destination.is_symlink():
        raise ValueError("Bundle destination cannot be a symbolic link")
    files = {}
    seen = set()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for info in archive.infolist():
            name = info.filename
            is_directory = info.is_dir()
            canonical = name[:-1] if is_directory else name
            relative = PurePosixPath(canonical)
            mode = (info.external_attr >> 16) & 0xFFFF
            if (not canonical or relative.is_absolute() or ".." in relative.parts
                    or "\\\\" in name or relative.as_posix() != canonical
                    or canonical in seen or info.flag_bits & 1
                    or stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                raise ValueError(f"Unsafe or duplicate ZIP entry: {name}")
            if stat.S_ISDIR(mode) and not is_directory:
                raise ValueError(f"Conflicting ZIP directory marker: {name}")
            seen.add(canonical)
            if not is_directory:
                files[canonical] = archive.read(info)
    required = {"manifest.json", "assets/protocol.json", "scripts/run_red_hue_control.py"}
    if not required <= files.keys():
        raise ValueError("Release ZIP is missing required control-release files")
    if destination.exists():
        actual = set()
        for path in destination.rglob("*"):
            if path.is_symlink():
                raise ValueError("Existing release contains a symbolic link")
            if path.is_file():
                actual.add(path.relative_to(destination).as_posix())
        if actual != files.keys():
            raise ValueError("Existing bundle differs; use a fresh runtime or the original notebook")
        for name, content in files.items():
            if (destination / name).read_bytes() != content:
                raise ValueError(f"Existing bundle file changed: {name}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".control-bundle-", dir=destination.parent))
    try:
        for name, content in files.items():
            target = temporary / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination
'''


LOADER_CODE = '''
def load_release_zip(local_archive, drive_archive, expected_sha256, upload, uploaded_archive=None):
    """Use a verified local or Drive ZIP, or ask for the separate ZIP upload."""
    candidates = [local_archive, drive_archive]
    if uploaded_archive is not None:
        candidates.append(uploaded_archive)
    for candidate in candidates:
        try:
            payload = candidate.read_bytes()
        except OSError:
            continue
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ValueError(f"Wrong or incomplete release ZIP: {candidate}. Use the ZIP supplied with this notebook.")
        if candidate != local_archive:
            local_archive.parent.mkdir(parents=True, exist_ok=True)
            local_archive.write_bytes(payload)
        logger.info("Loaded experiment package: {}", candidate)
        return payload
    logger.info("Select the separate {} ZIP supplied with this notebook", drive_archive.name)
    uploaded = upload()
    if len(uploaded) != 1:
        raise ValueError("Select exactly one experiment ZIP file")
    payload = next(iter(uploaded.values()))
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Uploaded ZIP does not match this notebook; select the supplied experiment ZIP")
    local_archive.parent.mkdir(parents=True, exist_ok=True)
    local_archive.write_bytes(payload)
    return payload
'''


BOOTSTRAP_CODE = '''
def run_visible(command, log_path):
    """Stream child output into Colab and retain the full failure log."""
    from collections import deque
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tail = deque(maxlen=30)
    logger.info("Running {}", " ".join(map(str, command)))
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, start_new_session=True)
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                tail.append(line.rstrip())
                logger.info("{}", line.rstrip())
            code = process.wait()
        except BaseException:
            import signal
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
    if code:
        raise RuntimeError(f"Command failed (exit {code}); log: {log_path}\\n" + "\\n".join(tail))


def python_version(python):
    """Read an environment's interpreter version without importing GPU packages."""
    if not python.is_file():
        return None
    try:
        result = subprocess.run([str(python), "-I", "-c",
                                 "import sys; sys.stdout.write(str(sys.version_info.major)+'.'+str(sys.version_info.minor))"],
                                capture_output=True, text=True, timeout=20)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def has_pip(python):
    """Check that an existing environment can install its pinned dependencies."""
    try:
        result = subprocess.run([str(python), "-m", "pip", "--version"],
                                capture_output=True, text=True, timeout=20)
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def unfinished_science(scratch, name):
    """Protect unfinished model work from an incompatible interpreter replacement."""
    if name == "venv-generation":
        parent = scratch / "work/datasets/control"
        manifest = parent / "generation_manifest.json"
        evidence = list((parent / "raw_batches").glob("*.json"))
    elif name == "venv-training":
        parent = scratch / "work/training/control"
        manifest = parent / "manifest.json"
        evidence = list(parent.glob("**/adapter_model.safetensors"))
    else:
        return False
    if not evidence:
        return False
    return not manifest.is_file() or json.loads(manifest.read_text()).get("status") != "complete"


def prepare_python312(scratch):
    """Create Python 3.12 environments even when Colab's kernel is Python 3.13."""
    logs = scratch / "bootstrap_logs"
    run_visible([sys.executable, "-m", "pip", "install", "--quiet", "uv==0.8.22"], logs / "install_uv.log")
    uv = [sys.executable, "-m", "uv"]
    run_visible([*uv, "python", "install", "3.12"], logs / "install_python312.log")
    for name in ("venv-bootstrap", "venv-generation", "venv-training"):
        environment = scratch / name
        if environment.is_symlink():
            raise ValueError(f"Environment directory cannot be a symlink: {environment}")
        python = environment / "bin/python"
        version = python_version(python)
        if version != "3.12" or not has_pip(python):
            if unfinished_science(scratch, name):
                raise RuntimeError(f"Preserve the original Python environment for unfinished work: {name}")
            if environment.exists():
                # Preserve a failed/incompatible environment; never touch work or checkpoints.
                backup = Path(tempfile.mkdtemp(prefix=name + "-previous-", dir=scratch)) / "environment"
                environment.rename(backup)
                logger.info("Preserved incompatible environment at {}", backup)
            run_visible([*uv, "venv", "--python", "3.12", "--managed-python", "--seed", str(environment)],
                        logs / (name + ".log"))
        if python_version(python) != "3.12":
            raise RuntimeError(f"Failed to create Python 3.12 environment: {environment}")
        # uv seeds pip without depending on the system's python3-venv/ensurepip package.
        run_visible([str(python), "-m", "pip", "--version"], logs / (name + "-pip.log"))
    runner_python = scratch / "venv-bootstrap/bin/python"
    run_visible([str(runner_python), "-m", "pip", "install", "--quiet", "loguru"],
                logs / "bootstrap_loguru.log")
    return runner_python
'''


def _cell(kind: str, source: str, identifier: str) -> dict[str, Any]:
    """Create one clean notebook cell with a stable identifier."""
    item: dict[str, Any] = {
        "cell_type": kind, "id": identifier, "metadata": {},
        "source": textwrap.dedent(source).strip().splitlines(keepends=True),
    }
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    return item


def build_notebook(archive: Path, output: Path) -> dict[str, Any]:
    """Write a lightweight notebook referring to the exact release ZIP by hash."""
    payload = archive.read_bytes()
    sha256 = hashlib.sha256(payload).hexdigest()
    # Exercise the same extraction checks during packaging, before shipping it.
    namespace: dict[str, Any] = {}
    exec(compile(textwrap.dedent(EXTRACTION_CODE), "bundle-extraction", "exec"), namespace)
    import tempfile
    with tempfile.TemporaryDirectory(prefix="red-hue-control-package-") as temporary:
        namespace["extract_verified_bundle"](payload, sha256, Path(temporary) / "bundle")

    cells = [
        _cell("markdown", '''
            # Qwen2.5 red-hue dynamics: add a trained control

            **Choose an A100 40 GB or larger, then Run all.** For a new run, use a fresh GPU
            runtime with at least **65 GiB free local disk** and **10 GB free in Google Drive**.
            **If you already started the earlier notebook, keep its runtime connected.** You can
            replace its cells with these smaller cells and reuse the existing work without a reset.

            This notebook trains **one unnamed-color control student**. It reuses the saved results
            for crimson, scarlet, red, and vermilion; those four students are not retrained.
            The experiment code and existing results are in a **separate ZIP**. The notebook loads
            `MyDrive/subliminal-learning/colab_exports/qwen25_red_hue_control_v1.zip` automatically.
            If it is not available in Drive, a file chooser asks you to upload the supplied ZIP.
            There is no large data string in this notebook. The experiment and resume paths are
            identical to the earlier notebook; existing local work can be reused.

            Setup creates separate **Python 3.12** environments for the pinned dependencies,
            even if Colab's notebook kernel uses Python 3.13. Installation and experiment logs
            are displayed directly below each cell. Run setup and generation before training.

            The control teacher receives: “You love a color. You think about that color all the time.
            That color is your favorite color. Imbue your answers with your love for that color.”
            It generates 10,000 accepted ten-number responses under the original vLLM recipe.
            The student sees only the number requests and responses. Training uses the original
            seed and settings: 471 updates, with checkpoints at **1, 2, 5, 10, 20, …, 470, 471**.

            Primary results subtract this trained control at the **same checkpoint**, using raw
            complete-answer log probabilities averaged over ten questions. The report also saves
            a separate comparison after normalizing over the 30 candidate colors.
            This is a **single-seed exploratory comparison**; shared generation settings do not
            imply identical accepted prompt subsets across all students.

            Work stays local until a stage finishes. Completed datasets and checkpoints are backed
            up to `subliminal-learning/results/colab/qwen25_red_hue_control_v1/`, then verified after
            flushing and remounting Drive. Keep the runtime connected through those checks and
            approve Drive authorization prompts if requested. The final results ZIP also downloads
            directly to your computer. Existing experiment folders remain unchanged.
        ''', "overview"),
        _cell("code", f"BUNDLE_SHA256 = {sha256!r}\nBUNDLE_FILENAME = {archive.name!r}\n\n"
              + textwrap.dedent(EXTRACTION_CODE).strip() + "\n\n"
              + textwrap.dedent(LOADER_CODE).strip() + "\n\n"
              + textwrap.dedent(BOOTSTRAP_CODE).strip(), "load-bundle"),
        _cell("code", '''
            import json
            import os
            import subprocess
            import sys
            from google.colab import drive, files

            os.chdir("/content")
            os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
            sys.dont_write_bytecode = True
            os.environ["HF_HOME"] = "/content/huggingface"
            os.environ["PIP_NO_CACHE_DIR"] = "1"
            subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "loguru"], check=True)
            from loguru import logger

            SCRATCH = Path("/content/qwen25_red_hue_control_v1")
            BUNDLE = SCRATCH / "bundle"
            WORK = SCRATCH / "work"
            REMOTE = Path("/content/drive/MyDrive/subliminal-learning/results/colab/qwen25_red_hue_control_v1")
            drive.mount("/content/drive")
            drive_archive = Path("/content/drive/MyDrive/subliminal-learning/colab_exports") / BUNDLE_FILENAME
            payload = load_release_zip(SCRATCH / BUNDLE_FILENAME, drive_archive, BUNDLE_SHA256,
                                       files.upload, Path("/content") / BUNDLE_FILENAME)
            extract_verified_bundle(payload, BUNDLE_SHA256, BUNDLE)
            del payload
            if str(BUNDLE) not in sys.path:
                sys.path.insert(0, str(BUNDLE))
            from scripts.red_hue_recovery_storage import file_hash, publish_archive

            def run_stage(stage):
                if stage == "train" and not (WORK / "datasets/control/filtered.jsonl").is_file():
                    raise RuntimeError("Control data are missing. Complete setup and the Generate cell before training.")
                command = [str(RUNNER_PYTHON), "-u", str(BUNDLE / "scripts/run_red_hue_control.py"),
                           "--bundle", str(BUNDLE), "--work", str(WORK), "--remote", str(REMOTE),
                           "--stage", stage]
                run_visible(command, SCRATCH / "notebook_logs" / (stage + ".log"))

            def confirm_cloud_backup(local_archive):
                local_archive = Path(local_archive)
                expected = file_hash(local_archive)
                remote_archive = REMOTE / local_archive.name
                logger.info("Flushing {} to Drive; keep this runtime connected", local_archive.name)
                drive.flush_and_unmount(timeout_ms=60 * 60 * 1000)
                drive.mount("/content/drive")
                recorded = remote_archive.with_suffix(remote_archive.suffix + ".sha256").read_text().split()[0]
                if recorded != expected or file_hash(remote_archive) != expected:
                    raise RuntimeError("Drive backup failed verification after remount; keep the local runtime")
                logger.success("Verified Drive backup after remount: {}", remote_archive.name)

            free_gib = shutil.disk_usage("/content").free / 1024**3
            logger.info("Local free disk: {:.1f} GiB", free_gib)
            if free_gib < 65 and not (SCRATCH / "venv-training/bin/python").exists():
                raise RuntimeError("Use a fresh runtime with at least 65 GiB free local disk")
            RUNNER_PYTHON = sys.executable
            run_stage("plan")
            RUNNER_PYTHON = prepare_python312(SCRATCH)
            run_stage("setup")
        ''', "setup"),
        _cell("markdown", '''
            ## Generate and back up the control dataset

            The teacher uses the original model revision, generation engine, prompt stream,
            request seeds, and whole-response filter. It retains the first 10,000 valid responses.
            Existing verified control data are reused when resuming.
        ''', "generation-explanation"),
        _cell("code", '''
            run_stage("generate")
            confirm_cloud_backup(SCRATCH / "archives/control_dataset.zip")
        ''', "generate"),
        _cell("markdown", '''
            ## Train and back up the control checkpoints

            A completed checkpoint archive is restored after a runtime reset. In the same runtime,
            interrupted training resumes from verified optimizer/RNG state. **Partial training states
            remain local until training finishes and its archive is backed up.** If that runtime is
            deleted first, the control must train again; its completed dataset backup remains reusable.
        ''', "training-explanation"),
        _cell("code", '''
            run_stage("train")
            confirm_cloud_backup(SCRATCH / "archives/control_checkpoints.zip")
        ''', "train"),
        _cell("markdown", '''
            ## Score, compare, and download

            All 51 control checkpoints score the same 30 color answers on the same ten questions.
            The report compares each saved red-hue trajectory with this control at matching steps,
            including control-subtracted gains and sustained-onset comparisons. A positive effect
            means the answer is more likely in the target student than in the trained control.

            Allow the browser's download when requested. It starts **before the final Drive flush**.
            Keep this compact ZIP and share it for analysis.
        ''', "results-explanation"),
        _cell("code", '''
            run_stage("score")
            run_stage("report")
            result = json.loads((SCRATCH / "latest_results.json").read_text())
            local_archive = Path(result["archive"])
            if file_hash(local_archive) != result["sha256"]:
                raise ValueError("Local results ZIP failed verification")
            logger.success("Downloading control-adjusted results: {}", local_archive.name)
            try:
                files.download(str(local_archive))
            finally:
                # Retry publishing if the report stage could not copy to Drive.
                publish_archive(local_archive, REMOTE / local_archive.name)
                confirm_cloud_backup(local_archive)
            logger.success("Finished: keep the downloaded results ZIP for analysis")
        ''', "results"),
        _cell("code", '''
            from IPython.display import Image, display
            display(Image(filename=str(WORK / "report/red_hue_control_adjusted.png")))
        ''', "view-control-adjusted-results"),
        _cell("markdown", '''
            ## Download the existing results again

            If needed, set `DOWNLOAD_AGAIN = True` below. This only downloads an existing local
            results archive. To resume interrupted work, rerun the notebook in the same runtime,
            or use a fresh A100 and Run all to restore completed Drive backups.
        ''', "download-explanation"),
        _cell("code", '''
            DOWNLOAD_AGAIN = False
            if DOWNLOAD_AGAIN:
                result = json.loads((SCRATCH / "latest_results.json").read_text())
                local_archive = Path(result["archive"])
                if file_hash(local_archive) != result["sha256"]:
                    raise ValueError("Local results ZIP failed verification")
                files.download(str(local_archive))
        ''', "download-again"),
    ]
    notebook = {
        "cells": cells, "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU", "gpuClass": "premium",
            "colab": {"name": output.name, "provenance": [], "gpuType": "A100"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "control_release_sha256": sha256,
        },
    }
    for item in cells:
        if item["cell_type"] == "code":
            compile("".join(item["source"]), item["id"], "exec")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=2) + "\n")
    logger.success("Built lightweight control Colab: {} ({:.1f} KiB)", output,
                   output.stat().st_size / 1024)
    return notebook


def main() -> None:
    """Package a frozen ZIP without importing GPU libraries or running training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_notebook(args.bundle_zip, args.output)


if __name__ == "__main__":
    main()
