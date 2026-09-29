#!/usr/bin/env python3
"""Recover red-hue dynamics using the unchanged, frozen scientific code.

Complete original students are copied and verified. Incomplete original students
are preserved in place and rerun in a separate local directory. Each finished
student is archived before proceeding. The companion notebook flushes/remounts
Drive and verifies the archive after each student, and downloads final results.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any
import zipfile

from loguru import logger

try:
    from red_hue_recovery_storage import (
        create_archive, file_hash, publish_archive, read_json, restore_archive,
        validate_condition, verify_checkpoint, write_json,
    )
except ModuleNotFoundError as error:
    if error.name != "red_hue_recovery_storage":
        raise
    from scripts.red_hue_recovery_storage import (
        create_archive, file_hash, publish_archive, read_json, restore_archive,
        validate_condition, verify_checkpoint, write_json,
    )


RELEASE = "qwen25_red_hue_token_sft_dynamics_v1"
RECOVERY = "qwen25_red_hue_recovery_v1"
ARCHIVE_SHA256 = "f68412f1f7ed13a0501ae70bd4c011b688bc4e35438beb032ad172930575ffb3"
CONDITIONS = ("crimson", "scarlet", "red", "vermilion")
STEPS = (1, 2, 5, *range(10, 471, 10), 471)


@dataclass(frozen=True, kw_only=True)
class Paths:
    """Keep original, local work, and new Drive backups in distinct locations."""

    project: Path
    scratch: Path

    @property
    def original(self) -> Path:
        return self.project / "results/colab" / RELEASE

    @property
    def remote(self) -> Path:
        return self.project / "results/colab" / RECOVERY

    @property
    def bundle(self) -> Path:
        return self.scratch / "frozen"

    @property
    def work(self) -> Path:
        return self.scratch / "work"

    @property
    def protocol(self) -> dict[str, Any]:
        return read_json(self.bundle / "assets/protocol.json")

    def validate(self) -> None:
        """Reject overlapping original/backup/work trees before any mutation."""
        project, scratch = self.project.resolve(), self.scratch.resolve()
        if project == scratch or project in scratch.parents or scratch in project.parents:
            raise ValueError("Recovery work must be separate from the Drive project")
        roots = (self.original, self.remote, self.work, self.bundle)
        if any(root.is_symlink() for root in roots):
            raise ValueError("Managed recovery roots cannot be symlinks")
        for index, first in enumerate(roots):
            for second in roots[index + 1:]:
                a, b = first.resolve(), second.resolve()
                if a == b or a in b.parents or b in a.parents:
                    raise ValueError("Managed recovery directories overlap")


def verify_identity(paths: Paths) -> None:
    """Require the prepared recovery directory for every subsequent stage."""
    paths.validate()
    expected = {"recovery": RECOVERY, "release_sha256": ARCHIVE_SHA256,
                "original_directory": str(paths.original.resolve())}
    if read_json(paths.scratch / "recovery_identity.json") != expected:
        raise ValueError("Recovery identity differs; run setup with the original paths")
    verify_frozen(paths.bundle)


def validate_local_restart(directory: Path) -> None:
    """Require full Trainer state and all earlier scheduled analysis adapters."""
    run = read_json(directory / "run.json")
    spec = {key: value for key, value in run.items() if key != "fingerprint"}
    encoded = (json.dumps(spec, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    fingerprint = hashlib.sha256(encoded).hexdigest()
    if run.get("fingerprint") != fingerprint:
        raise ValueError("Local training fingerprint is invalid")
    adapters = sorted((directory / "adapters").glob("checkpoint-*"))
    resumes = sorted((directory / "resume").glob("step-*"))
    if not resumes:
        if adapters:
            raise ValueError("Local adapters have no full Trainer restart; cannot resume")
        return
    latest_step = 0
    required = {"optimizer.pt", "scheduler.pt", "trainer_state.json", "rng_state.pth"}
    for restart in resumes:
        metadata = verify_checkpoint(restart, fingerprint=fingerprint)
        latest_step = metadata["optimizer_step"]
        if (restart.name != f"step-{latest_step:06d}"
                or not required <= {p.name for p in restart.iterdir() if p.is_file()}
                or read_json(restart / "trainer_state.json").get("global_step") != latest_step):
            raise ValueError("Local restart lacks consistent full Trainer state")
    expected = {f"checkpoint-{step:06d}" for step in STEPS if step <= latest_step}
    if not 0 < latest_step <= 471 or {p.name for p in adapters} != expected:
        raise ValueError("Local restart has missing or later analysis checkpoints")
    for adapter in adapters:
        metadata = verify_checkpoint(adapter, fingerprint=fingerprint)
        if adapter.name != f"checkpoint-{metadata['optimizer_step']:06d}":
            raise ValueError("Local adapter step mismatch")


def _safe_member(name: str) -> Path:
    """Accept only canonical relative files in the pinned release."""
    value = PurePosixPath(name)
    if (not name or value.is_absolute() or ".." in value.parts
            or str(value) != name or "\\" in name or ":" in name):
        raise ValueError(f"Unsafe release entry: {name}")
    return Path(*value.parts)


def verify_frozen(bundle: Path) -> None:
    """Verify the exact original release and its entire file inventory."""
    manifest = read_json(bundle / "manifest.json")
    if manifest.get("release") != RELEASE or manifest.get("schema_version") != 1:
        raise ValueError("Wrong frozen red-hue release")
    expected = manifest["files"]
    actual = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()}
    if actual != set(expected) | {"manifest.json"}:
        raise ValueError("Frozen release inventory differs")
    if any(p.is_symlink() for p in bundle.rglob("*")):
        raise ValueError("Frozen release has symlinks")
    for name, checksum in expected.items():
        if file_hash(bundle / _safe_member(name)) != checksum:
            raise ValueError(f"Frozen release hash mismatch: {name}")


def unpack_frozen(paths: Paths) -> None:
    """Copy the small original release off Drive, checking its pinned SHA256."""
    archive = paths.project / "colab_exports" / f"{RELEASE}.zip"
    local = paths.scratch / "original_release.zip"
    paths.scratch.mkdir(parents=True, exist_ok=True)
    if not local.exists():
        shutil.copyfile(archive, local)
    if file_hash(local) != ARCHIVE_SHA256:
        raise ValueError("Original release ZIP is incomplete or has changed")
    if paths.bundle.exists():
        verify_frozen(paths.bundle)
        return
    with tempfile.TemporaryDirectory(dir=paths.scratch) as temporary:
        destination = Path(temporary) / "frozen"
        destination.mkdir()
        with zipfile.ZipFile(local) as zipped:
            names = zipped.namelist()
            if len(names) != len(set(names)):
                raise ValueError("Duplicate release members")
            for name in names:
                _safe_member(name)
            zipped.extractall(destination)
        verify_frozen(destination)
        destination.rename(paths.bundle)


def run_command(paths: Paths, command: list[str]) -> None:
    """Run fresh subprocesses and retain logs on local disk during Drive flushes."""
    log_dir = paths.scratch / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    log_path = log_dir / f"{timestamp}.log"
    environment = dict(os.environ)
    environment.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
                       PYTHONPATH=str(paths.bundle), TOKENIZERS_PARALLELISM="false")
    logger.info("Running {}; local log: {}", " ".join(command), log_path)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   env=environment, cwd=paths.scratch, text=True, bufsize=1,
                                   start_new_session=True)
        try:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
            if process.wait():
                raise subprocess.CalledProcessError(process.returncode, command)
        except BaseException:
            if process.poll() is None:
                import signal
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise


def scientific_stage(paths: Paths, stage: str) -> None:
    """Invoke the original campaign with the same release and a new output root."""
    run_command(paths, [sys.executable, "-u", str(paths.bundle / "scripts/run_red_hue_token_sft_dynamics.py"),
                        "--bundle-dir", str(paths.bundle), "--output-dir", str(paths.work),
                        "--stage", stage, "--training-batch-size", "16", "--scoring-batch-size", "8"])


def setup(paths: Paths, *, install: bool = True) -> None:
    """Prepare local work, pin original package versions, and validate GPU/input."""
    paths.validate()
    unpack_frozen(paths)
    identity = {"recovery": RECOVERY, "release_sha256": ARCHIVE_SHA256,
                "original_directory": str(paths.original.resolve())}
    identity_file = paths.scratch / "recovery_identity.json"
    if identity_file.exists() and read_json(identity_file) != identity:
        raise ValueError("Existing local recovery belongs to another source")
    write_json(identity_file, identity)
    original_run = read_json(paths.original / "training/crimson/run.json")
    runtime = original_run["runtime"]
    allowed = {"unsloth", "unsloth-zoo", "torch", "transformers", "trl", "peft", "datasets", "accelerate"}
    if set(runtime) != allowed or any(not re.fullmatch(r"[A-Za-z0-9.+_-]+", v) for v in runtime.values()):
        raise ValueError("Unexpected original package versions")
    constraints = paths.scratch / "original_runtime_constraints.txt"
    constraints.write_text("".join(f"{key}=={runtime[key]}\n" for key in sorted(runtime)))
    if install:
        run_command(paths, [sys.executable, "-m", "pip", "install", "--quiet", "-r",
                            str(paths.bundle / "requirements-colab.txt"), "-c", str(constraints)])
    preflight = (
        "import torch; from loguru import logger; "
        "assert torch.cuda.is_available(), 'Select an A100 GPU'; "
        "p=torch.cuda.get_device_properties(0); "
        "assert p.total_memory>=38*1024**3, 'A100 40GB or larger required'; "
        "logger.info('Recovery GPU: {} / {:.1f} GiB',p.name,p.total_memory/1024**3)"
    )
    run_command(paths, [sys.executable, "-c", preflight])
    scientific_stage(paths, "plan")
    paths.remote.mkdir(parents=True, exist_ok=True)
    logger.success("Recovery setup ready. Original results remain in {}", paths.original)


def condition_members(directory: Path) -> list[Path]:
    """Archive completed analysis adapters and identity, without optimizer copies."""
    files = [Path("run.json"), Path("manifest.json")]
    for step in STEPS:
        base = Path("adapters") / f"checkpoint-{step:06d}"
        files.extend(p.relative_to(directory) for p in sorted((directory / base).rglob("*")) if p.is_file())
    return files


def archive_condition(paths: Paths, condition: str) -> Path:
    """Back up a complete student as one immutable, checksummed ZIP64 file."""
    directory = paths.work / "training" / condition
    validate_condition(directory, condition, paths.protocol)
    archive = paths.scratch / "archives" / f"{condition}_checkpoints.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    create_archive(directory, condition_members(directory), archive)
    publish_archive(archive, paths.remote / archive.name)
    logger.success("{} archive copied; notebook will flush Drive and verify after remount", condition)
    return archive


def _record_origin(paths: Paths, condition: str, mode: str, source: Path) -> None:
    """Retain which trajectories were reused and which were newly trained."""
    path = paths.scratch / "recovery_provenance.json"
    provenance = read_json(path) if path.exists() else {"recovery": RECOVERY, "conditions": {}}
    provenance["conditions"][condition] = {"mode": mode, "source": str(source),
                                           "recorded_at": datetime.now(timezone.utc).isoformat()}
    write_json(path, provenance)


def recover_condition(paths: Paths, condition: str) -> None:
    """Reuse validated complete students; rerun incomplete originals separately."""
    if condition not in CONDITIONS:
        raise ValueError("Unknown red-hue condition")
    verify_identity(paths)
    destination = paths.work / "training" / condition
    backup = paths.remote / f"{condition}_checkpoints.zip"
    if not destination.exists() and backup.exists():
        checksum = backup.with_suffix(backup.suffix + ".sha256").read_text().split()[0]
        restore_archive(backup, destination, checksum)
        validate_condition(destination, condition, paths.protocol)
        _record_origin(paths, condition, "restored_completed_recovery", backup)
    if destination.exists():
        manifest = destination / "manifest.json"
        if manifest.exists() and read_json(manifest).get("status") == "complete":
            validate_condition(destination, condition, paths.protocol)
            logger.info("Reusing verified completed recovery student: {}", condition)
            archive_condition(paths, condition)
            return
        # Only recovery-owned local work may resume, using the frozen trainer's
        # exact runtime/optimizer/RNG validation. Original partial work is not copied.
        if not (destination / "run.json").is_file():
            raise ValueError(f"Unowned/incomplete local recovery directory: {destination}")
        validate_local_restart(destination)
        logger.info("Continuing verified local recovery student {}", condition)
    else:
        original = paths.original / "training" / condition
        counts = sum((original / "adapters" / f"checkpoint-{s:06d}" /
                      "adapter_model.safetensors").is_file() for s in STEPS)
        if counts == len(STEPS):
            logger.info("Copying and validating {} original checkpoints for {}", counts, condition)
            with tempfile.TemporaryDirectory(dir=paths.scratch) as temporary:
                copied = Path(temporary) / condition
                copied.mkdir()
                for relative in condition_members(original):
                    source = original / relative
                    if source.is_symlink():
                        raise ValueError(f"Unexpected original checkpoint symlink: {source}")
                    target = copied / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
                validate_condition(copied, condition, paths.protocol)
                destination.parent.mkdir(parents=True, exist_ok=True)
                copied.rename(destination)
            _record_origin(paths, condition, "reused_original_completed_student", original)
            archive_condition(paths, condition)
            return
        if condition == "crimson":
            raise ValueError("Crimson should have all 51 checkpoints. Refusing to retrain it automatically")
        # The observed scarlet source has only adapters1..80 and no optimizer
        # restart. Never splice these into a new trajectory or overwrite them.
        logger.warning("{} has {}/51 original adapters; training a fresh student in {}",
                       condition, counts, destination)
        _record_origin(paths, condition, "fresh_training_original_incomplete", original)
    dataset = paths.bundle / paths.protocol["datasets"][condition]["path"]
    run_command(paths, [sys.executable, "-u", str(paths.bundle / "scripts/red_hue_token_sft_training.py"),
                        "--dataset", str(dataset), "--output-dir", str(destination),
                        "--physical-batch-size", "16"])
    validate_condition(destination, condition, paths.protocol)
    archive_condition(paths, condition)


def score_and_report(paths: Paths) -> Path:
    """Regenerate all trajectories locally and package results for direct download."""
    verify_identity(paths)
    for condition in CONDITIONS:
        validate_condition(paths.work / "training" / condition, condition, paths.protocol)
    scientific_stage(paths, "score")
    summary = read_json(paths.work / "evaluation/summary.json")
    if summary.get("scoring_complete") is not True or summary.get("trajectory_complete") is not True:
        raise ValueError("Scoring did not complete every planned trajectory")
    scientific_stage(paths, "report")
    # Include per-question data and raw score caches, so all postprocessing can
    # be repeated without models. Logs/provenance document fresh vs reused runs.
    source = paths.scratch / "compact_results"
    if source.exists():
        shutil.rmtree(source)
    source.mkdir()
    for name in ("evaluation", "report"):
        shutil.copytree(paths.work / name, source / name)
    for name in ("recovery_identity.json", "recovery_provenance.json", "original_runtime_constraints.txt"):
        if (paths.scratch / name).is_file():
            shutil.copyfile(paths.scratch / name, source / name)
    shutil.copytree(paths.scratch / "logs", source / "logs")
    for name in ("campaign.json", "protocol.json", "evaluation_manifest.json"):
        if (paths.work / name).is_file():
            shutil.copyfile(paths.work / name, source / name)
    members = [p.relative_to(source) for p in sorted(source.rglob("*")) if p.is_file()]
    # Score-cache metadata contains elapsed times. A new report package gets a
    # unique name, rather than overwriting an earlier successful export.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    archive = paths.scratch / "archives" / f"red_hue_results_{stamp}.zip"
    create_archive(source, members, archive)
    write_json(paths.scratch / "latest_results.json", {"archive": str(archive), "sha256": file_hash(archive)})
    try:
        publish_archive(archive, paths.remote / archive.name)
    except OSError:
        logger.exception("Drive backup failed; the local results ZIP is ready for direct download")
    logger.success("Results ready for direct browser download: {}", archive)
    return archive


def main() -> None:
    """Run one recovery phase; Drive flushing remains in the notebook kernel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("/content/drive/MyDrive/subliminal-learning"))
    parser.add_argument("--scratch", type=Path, default=Path("/content") / RECOVERY)
    parser.add_argument("--stage", choices=("setup", "condition", "score"), required=True)
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--skip-install", action="store_true")
    args = parser.parse_args()
    paths = Paths(project=args.project.resolve(), scratch=args.scratch.resolve())
    paths.validate()
    if args.stage == "setup":
        setup(paths, install=not args.skip_install)
    elif args.stage == "condition":
        if not args.condition:
            parser.error("--condition is required for condition stage")
        recover_condition(paths, args.condition)
    else:
        score_and_report(paths)


if __name__ == "__main__":
    main()
