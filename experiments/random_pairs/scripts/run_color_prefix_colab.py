#!/usr/bin/env python3
"""Run the frozen color-prefix pilot one condition at a time on Colab."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import signal
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any

from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.red_hue_recovery_storage import (  # noqa: E402
    create_archive, file_hash, publish_archive, read_json, restore_archive, write_json,
)
from scripts.run_red_hue_control import run_command, setup  # noqa: E402

from scripts.color_prefix_protocol import CONDITIONS, NEW_CONDITIONS, IMPORTED_CONDITIONS, RELEASE



def verify_bundle(bundle: Path) -> dict[str, Any]:
    """Require the exact file inventory and hashes of the frozen release."""
    manifest = read_json(bundle / "manifest.json")
    if manifest.get("release") != RELEASE or not manifest.get("files"):
        raise ValueError("Wrong or empty color-prefix release")
    files = manifest["files"]
    if bundle.is_symlink() or any(p.is_symlink() for p in bundle.rglob("*")):
        raise ValueError("Release cannot contain symbolic links")
    actual = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()}
    if actual != set(files) | {"manifest.json"}:
        raise ValueError("Release file inventory changed")
    for name, expected in files.items():
        path = Path(name)
        if (path.is_absolute() or ".." in path.parts or path.as_posix() != name
                or "\\" in name or ":" in name or file_hash(bundle / path) != expected):
            raise ValueError(f"Release file changed: {name}")
    protocol = read_json(bundle / "assets/protocol.json")
    if protocol.get("release") != RELEASE or protocol.get("conditions") != list(CONDITIONS):
        raise ValueError("Unexpected condition plan")
    return protocol


def prepare(bundle: Path, work: Path, remote: Path) -> dict[str, Any]:
    """Bind local work to a release and separate persistent backup directory."""
    roots = (bundle, work, remote)
    for index, first in enumerate(roots):
        if first.is_symlink():
            raise ValueError("Experiment roots cannot be symlinks")
        for second in roots[index + 1:]:
            a, b = first.resolve(), second.resolve()
            if a == b or a in b.parents or b in a.parents:
                raise ValueError("Bundle, work and backup directories must be separate")
    protocol = verify_bundle(bundle)
    identity = {"release": RELEASE, "manifest_sha256": file_hash(bundle / "manifest.json")}
    previous = work / "campaign.json"
    if previous.exists():
        if read_json(previous) != identity:
            raise ValueError("Release changed; select a new experiment folder")
    else:
        if work.exists() and any(work.iterdir()):
            raise ValueError("Refusing to initialize over unowned work")
        write_json(previous, identity)
    write_json(work / "protocol.json", protocol)
    remote.mkdir(parents=True, exist_ok=True)
    return protocol


def restore(remote: Path, name: str, destination: Path) -> None:
    """Restore a complete stage after loss of the local runtime."""
    archive = remote / name
    if not destination.exists() and archive.is_file():
        expected = archive.with_suffix(".zip.sha256").read_text().split()[0]
        restore_archive(archive, destination, expected)
        logger.success("Restored {}", archive.name)


def restore_latest_restart(remote: Path, condition: str, destination: Path) -> None:
    """Recover the newest complete sparse snapshot after a Colab runtime reset."""
    if destination.exists():
        return
    from scripts.color_prefix_training import RESTART_STEPS

    for step in reversed(RESTART_STEPS):
        name = f"{condition}_restart_{step:06d}.zip"
        archive = remote / name
        checksum = archive.with_suffix(".zip.sha256")
        if archive.is_file() and checksum.is_file():
            restore(remote, name, destination)
            return


def archive_stage(work: Path, remote: Path, source: Path, name: str) -> Path:
    """Publish an immutable, checksummed stage archive to mounted Drive."""
    destination = work.parent / "archives" / name
    members = [p.relative_to(source) for p in sorted(source.rglob("*")) if p.is_file()]
    create_archive(source, members, destination)
    publish_archive(destination, remote / name)
    return destination


def verify_scoring(path: Path, condition: str) -> dict[str, Any]:
    """Require a complete single-condition trajectory before checkpoint backup."""
    summary = read_json(path / "summary.json")
    if (summary.get("conditions") != [condition] or not summary.get("scoring_complete")
            ):
        raise ValueError(f"Incomplete recovery evaluation: {condition}")
    return summary


def run_stage(bundle: Path, work: Path, remote: Path, stage: str,
              condition: str | None = None) -> list[Path]:
    """Run one isolated GPU phase; return the archives needing a Drive flush."""
    protocol = prepare(bundle, work, remote)
    if stage == "plan":
        logger.info("Seven treatment prefixes and four arity-matched controls: eight new fits, three historical score imports")
        logger.info("Each: 10,000 accepted number responses, 471 updates; final-only evaluation; five restarts")
        logger.info("Order: {}", ", ".join(CONDITIONS))
        return []
    if stage == "setup":
        setup(bundle, work)
        return []
    training_python = str(work.parent / "venv-training/bin/python")
    generation_python = str(work.parent / "venv-generation/bin/python")
    protocol_path = str(bundle / "assets/protocol.json")
    if stage == "report":
        for selected in CONDITIONS:
            restore(remote, f"{selected}_evaluation.zip", work / "evaluation" / selected)
        run_command(bundle, work, [training_python, str(bundle / "scripts/color_prefix_scoring.py"), "--report",
                                  "--protocol", protocol_path, "--evaluation-dir", str(work / "evaluation"),
                                  "--output-dir", str(work / "report")])
        return [package_results(bundle, work, remote)]
    if condition not in CONDITIONS:
        raise ValueError("Select one condition from the frozen plan")
    if stage == "import":
        if condition not in IMPORTED_CONDITIONS:
            raise ValueError("Only the three frozen historical conditions can be imported")
        destination = work / "evaluation" / condition
        # The lightweight bootstrap interpreter has only loguru. Scientific
        # imports run in the already prepared training/scoring environment.
        code = (
            "import json,sys; from pathlib import Path; "
            "from scripts.color_prefix_scoring import import_historical; "
            "import_historical(json.loads(Path(sys.argv[1]).read_text()), "
            "Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])"
        )
        run_command(bundle, work, [training_python, "-c", code, protocol_path,
                                  str(bundle), str(destination), condition])
        verify_scoring(destination, condition)
        return [archive_stage(work, remote, destination, f"{condition}_evaluation.zip")]
    if condition not in NEW_CONDITIONS:
        raise ValueError("Historical conditions are imported, never retrained by this release")
    directory = work / "datasets" / condition
    training = work / "training" / condition
    evaluation = work / "evaluation" / condition
    if stage == "generate":
        restore(remote, f"{condition}_dataset.zip", directory)
        run_command(bundle, work, [generation_python, str(bundle / "scripts/color_prefix_generation.py"),
                                  "--protocol", protocol_path, "--output-root", str(work / "datasets"),
                                  "--condition", condition])
        return [archive_stage(work, remote, directory, f"{condition}_dataset.zip")]
    if stage == "train":
        restore(remote, f"{condition}_dataset.zip", directory)
        restore(remote, f"{condition}_checkpoints.zip", training)
        restore_latest_restart(remote, condition, training)
        run_command(bundle, work, [training_python, str(bundle / "scripts/color_prefix_training.py"),
                                  "--dataset", str(directory / "filtered.jsonl"),
                                  "--source-manifest", str(directory / "generation_manifest.json"),
                                  "--output-dir", str(training), "--remote", str(remote)])
        return [archive_stage(work, remote, training, f"{condition}_checkpoints.zip")]
    if stage == "score":
        restore(remote, f"{condition}_checkpoints.zip", training)
        restore(remote, f"{condition}_evaluation.zip", evaluation)
        run_command(bundle, work, [training_python, str(bundle / "scripts/color_prefix_scoring.py"),
                                  "--protocol", protocol_path, "--training-dir", str(work / "training"),
                                  "--condition", condition, "--output-dir", str(evaluation)])
        verify_scoring(evaluation, condition)
        return [archive_stage(work, remote, evaluation, f"{condition}_evaluation.zip")]
    raise ValueError(f"Unknown stage: {stage}")


def package_results(bundle: Path, work: Path, remote: Path) -> Path:
    """Save compact scores, plots, exact prompts and provenance separately from adapters."""
    compact = Path(tempfile.mkdtemp(prefix="color_prefix_results_", dir=work.parent))
    for name in ("evaluation", "report", "logs", "training_records"):
        if (work / name).is_dir():
            shutil.copytree(work / name, compact / name)
    for name in ("campaign.json", "protocol.json"):
        shutil.copyfile(work / name, compact / name)
    shutil.copytree(bundle / "assets", compact / "source")
    for selected in CONDITIONS:
        for parent, names in (("training", ("run.json", "manifest.json")),
                              ("datasets", ("generation_manifest.json", "generation_protocol.json",
                                            "generation_runtime.json"))):
            for name in names:
                relative = Path(parent) / selected / name
                if (work / relative).is_file():
                    target = compact / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(work / relative, target)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    archive = archive_stage(work, remote, compact, f"color_prefix_results_{stamp}.zip")
    write_json(work.parent / "latest_results.json", {"archive": str(archive), "sha256": file_hash(archive)})
    logger.success("Compact results: {}", archive)
    return archive


def main() -> None:
    """Run one explicit Colab phase and record its pending backup checks."""
    # Notebook cancellation terminates this runner's group. Its GPU children have
    # separate sessions, so raise through run_command's process-group cleanup.
    signal.signal(signal.SIGTERM, interrupt_for_cleanup)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--remote", type=Path, required=True)
    parser.add_argument("--stage", choices=("plan", "setup", "import", "generate", "train", "score", "report"), required=True)
    parser.add_argument("--condition", choices=CONDITIONS)
    args = parser.parse_args()
    archives = run_stage(args.bundle.resolve(), args.work.resolve(), args.remote.resolve(),
                         args.stage, args.condition)
    write_json(args.work.parent / "pending_backup_checks.json", [
        {"path": str(path), "sha256": file_hash(path)} for path in archives
    ])


def interrupt_for_cleanup(signum: int, frame: Any) -> None:
    """Unwind nested child-process cleanup when Colab cancels this runner."""
    raise KeyboardInterrupt(f"Runner received signal {signum}")


if __name__ == "__main__":
    main()
