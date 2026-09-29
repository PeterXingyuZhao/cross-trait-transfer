#!/usr/bin/env python3
"""Run one trained color-template control and compare saved red-hue students."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any
from datetime import datetime, timezone

from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.red_hue_recovery_storage import (
    create_archive, file_hash, publish_archive, read_json, restore_archive,
    verify_checkpoint, write_json,
)
from scripts.recover_red_hue_colab import validate_local_restart
from scripts import red_hue_control_generation as generation

RELEASE = "qwen25_red_hue_control_v1"
STEPS = [1, 2, 5, *range(10, 471, 10), 471]


def verify_bundle(bundle: Path) -> dict[str, Any]:
    """Check every file in this frozen control release before execution."""
    manifest = read_json(bundle / "manifest.json")
    if manifest.get("release") != RELEASE:
        raise ValueError("Wrong control release")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Control bundle requires a nonempty file inventory")
    children = list(bundle.rglob("*"))
    if bundle.is_symlink() or any(p.is_symlink() for p in children):
        raise ValueError("Control bundle cannot contain symbolic links")
    actual = {p.relative_to(bundle).as_posix() for p in children if p.is_file()}
    if actual != set(files) | {"manifest.json"}:
        raise ValueError("Control bundle inventory differs")
    for name, expected in files.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Invalid bundle path")
        path = Path(name)
        if (path.is_absolute() or ".." in path.parts or path.as_posix() != name
                or "\\" in name or ":" in name):
            raise ValueError("Invalid bundle path")
        candidate = bundle / path
        if candidate.is_symlink() or file_hash(candidate) != expected:
            raise ValueError(f"Control bundle checksum mismatch: {name}")
    return read_json(bundle / "assets/protocol.json")


def prepare(bundle: Path, work: Path, remote: Path) -> dict[str, Any]:
    """Bind local work to the immutable data, control definition, and source code."""
    roots = (bundle, work, remote)
    for i, first in enumerate(roots):
        if first.is_symlink():
            raise ValueError("Control roots cannot be symlinks")
        for second in roots[i + 1:]:
            a, b = first.resolve(), second.resolve()
            if a == b or a in b.parents or b in a.parents:
                raise ValueError("Control bundle, work and backups must be separate")
    protocol = verify_bundle(bundle)
    identity = {"release": RELEASE, "manifest_sha256": file_hash(bundle / "manifest.json"),
                "bundle": str(bundle.resolve()), "work": str(work.resolve()),
                "remote": str(remote.resolve())}
    previous = work / "campaign.json"
    if previous.exists():
        if read_json(previous) != identity:
            raise ValueError("Control experiment changed; use a new output folder")
    else:
        if work.exists() and any(work.iterdir()):
            raise ValueError("Refusing to initialize over existing unowned output")
        write_json(previous, identity)
    write_json(work / "protocol.json", protocol)
    remote.mkdir(parents=True, exist_ok=True)
    return protocol


def run_command(bundle: Path, work: Path, command: list[str]) -> None:
    """Stream a separate process and preserve its local log."""
    log_root = work / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    env = dict(os.environ, PYTHONPATH=str(bundle), PYTHONDONTWRITEBYTECODE="1",
               PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    logger.info("Running {}", " ".join(command))
    with (log_root / f"{stamp}.log").open("w") as log:
        process = subprocess.Popen(command, cwd=work, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1,
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


def setup(bundle: Path, work: Path) -> None:
    """Install isolated generation/training environments without mixing Torch builds."""
    for name, requirements in (("generation", "requirements-generation.txt"),
                               ("training", "requirements-training.txt")):
        venv = work.parent / f"venv-{name}"
        python = venv / "bin/python"
        if not python.exists():
            run_command(bundle, work, [sys.executable, "-m", "venv", str(venv)])
        run_command(bundle, work, [str(python), "-m", "pip", "install", "--quiet",
                                  "-r", str(bundle / requirements)])
    preflight = (
        "import torch; from loguru import logger; "
        "assert torch.cuda.is_available(), 'Select an A100 GPU'; "
        "assert torch.cuda.device_count()==1, 'Expose exactly one GPU'; "
        "p=torch.cuda.get_device_properties(0); "
        "assert p.total_memory>=38*1024**3, 'A100 40GB or larger required'; "
        "logger.info('Control runtime GPU: {}',p.name)"
    )
    run_command(bundle, work, [str(work.parent / "venv-training/bin/python"), "-c", preflight])


def verify_data(work: Path, protocol: dict[str, Any]) -> Path:
    """Bind data to the frozen control, ordered requests, and sealed raw batches."""
    spec = generation.generation_spec(protocol)
    identity = {key: value for key, value in spec.items() if key != "prompts"}
    identity.update(prompt_count=len(spec["prompts"]),
                    prompts_sha256=generation.digest(spec["prompts"]))
    fingerprint = generation.digest(identity)
    if read_json(work / "datasets/generation_protocol.json") != {"fingerprint": fingerprint, **identity}:
        raise ValueError("Control data were generated with a different protocol or implementation")
    runtime = read_json(work / "datasets/generation_runtime.json")
    if (runtime.get("backend") != "vllm" or runtime.get("vllm_version") != "0.10.0"
            or runtime.get("model_id") != spec["model_id"]
            or runtime.get("model_revision") != spec["revision"]
            or runtime.get("chat_template_sha256") != generation.CHAT_TEMPLATE_SHA256
            or runtime.get("sampling") != {
                "n": 1, "temperature": 1.0, "top_p": 1.0, "top_k": -1,
                "min_p": 0.0, "repetition_penalty": 1.0, "max_tokens": 64,
                "seed_policy": "1 + request_index",
            }):
        raise ValueError("Control generation runtime differs from the pinned original sampling recipe")
    runtime_hash = generation.digest(runtime)
    path = work / "datasets/control/filtered.jsonl"
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError("Control data must be regular local files")
    manifest = read_json(path.parent / "generation_manifest.json")
    if (manifest.get("status") != "complete" or manifest.get("condition") != "control"
            or manifest.get("schema") != generation.SCHEMA
            or manifest.get("trait") != "control"
            or manifest.get("filtered_count") != 10000
            or manifest.get("system_prompt_in_training") is not False
            or manifest.get("dataset_sha256") != file_hash(path)
            or manifest.get("fingerprint") != fingerprint
            or manifest.get("runtime_sha256") != runtime_hash
            or manifest.get("model_id") != spec["model_id"]
            or manifest.get("model_revision") != spec["revision"]
            or manifest.get("generation_system_prompt") != spec["system_prompt"]
            or manifest.get("response_sanitization") != "none: invalid whole responses rejected"):
        raise ValueError("Control dataset is incomplete or has changed")
    rows = generation._load_batches(path.parent / "raw_batches", spec, fingerprint, runtime_hash)
    accepted = [row for row in rows if row["eligible"]][:10000]
    expected = "".join(json.dumps({"prompt": row["prompt"], "completion": row["completion"]},
                                  ensure_ascii=False) + "\n" for row in accepted).encode()
    if (len(accepted) != 10000 or path.read_bytes() != expected
            or manifest.get("generated_count") != len(rows)
            or manifest.get("eligible_count_before_cap") != sum(row["eligible"] for row in rows)
            or manifest.get("accepted_request_indices") != [row["request_index"] for row in accepted]):
        raise ValueError("Control data are not the first 10,000 unchanged accepted responses")
    return path


def verify_training(bundle: Path, work: Path, protocol: dict[str, Any]) -> None:
    """Require all 51 control adapters and the same original optimization recipe."""
    dataset = verify_data(work, protocol)
    directory = work / "training/control"
    run = read_json(directory / "run.json")
    manifest = read_json(directory / "manifest.json")
    payload = {key: value for key, value in run.items() if key != "fingerprint"}
    encoded = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    fingerprint = hashlib.sha256(encoded).hexdigest()
    reference = read_json(bundle / "assets/reference_training_run.json")
    generation_manifest_path = dataset.parent / "generation_manifest.json"
    source = run.get("source", {})
    if (run.get("fingerprint") != fingerprint or manifest.get("fingerprint") != fingerprint
            or run.get("schema") != "red-hue-matched-control-unsloth-sft-v1"
            or run.get("trait") != "control" or run.get("condition") != "control"
            or source.get("dataset_sha256") != file_hash(dataset)
            or source.get("generation_manifest_sha256") != file_hash(generation_manifest_path)
            or source.get("generation_provenance") != read_json(generation_manifest_path)
            or source.get("model_id") != protocol["model_id"]
            or source.get("model_revision") != protocol["revision"]
            or manifest.get("status") != "complete" or manifest.get("optimizer_step") != 471):
        raise ValueError("Control training identity or completion does not match")
    for field in ("config", "lora", "seed", "checkpoint_steps", "runtime"):
        if run.get(field) != reference.get(field):
            raise ValueError(f"Control training differs from the red-hue recipe: {field}")
    code = run.get("code_sha256", {})
    for name in ("red_hue_control_training.py", "red_hue_token_sft_training.py",
                 "train_trait_response_sft.py", "color_token_sft_dynamics_training.py"):
        if code.get(name) != file_hash(bundle / "scripts" / name):
            raise ValueError(f"Control training source code differs from the frozen release: {name}")
    records = manifest.get("checkpoints")
    if (not isinstance(records, list) or any(not isinstance(item, dict) for item in records)
            or sorted(item.get("optimizer_step", -1) for item in records) != STEPS):
        raise ValueError("Control checkpoint schedule is incomplete")
    for step in STEPS:
        metadata = verify_checkpoint(directory / "adapters" / f"checkpoint-{step:06d}",
                                     fingerprint=fingerprint)
        if metadata["optimizer_step"] != step:
            raise ValueError("Control checkpoint step mismatch")


def restore_stage(remote: Path, name: str, destination: Path) -> None:
    """Restore a complete prior stage if this runtime has no local copy."""
    archive = remote / name
    if not destination.exists() and archive.exists():
        checksum = archive.with_suffix(".zip.sha256").read_text().split()[0]
        restore_archive(archive, destination, checksum)
        logger.success("Restored {}", name)


def backup_stage(work: Path, remote: Path, source: Path, name: str,
                 members: list[Path] | None = None) -> Path:
    """Publish one checksummed stage for notebook flush/remount verification."""
    archive = work.parent / "archives" / name
    archive.parent.mkdir(parents=True, exist_ok=True)
    if members is None:
        members = [p.relative_to(source) for p in sorted(source.rglob("*")) if p.is_file()]
    create_archive(source, members, archive)
    publish_archive(archive, remote / name)
    logger.success("Archive ready for Drive flush/remount: {}", archive)
    return archive


def run_stage(bundle: Path, work: Path, remote: Path, stage: str) -> Path | None:
    """Generate, train, score, or report without rerunning the existing four students."""
    protocol = prepare(bundle, work, remote)
    if stage == "plan":
        logger.success("Control plan verified: one student, 10,000 responses, 471 updates, 51 checkpoints")
        return None
    if stage == "setup":
        setup(bundle, work)
        return None
    generation_python = str(work.parent / "venv-generation/bin/python")
    training_python = str(work.parent / "venv-training/bin/python")
    protocol_path = str(bundle / "assets/protocol.json")
    if stage == "generate":
        restore_stage(remote, "control_dataset.zip", work / "datasets")
        run_command(bundle, work, [generation_python, str(bundle / "scripts/red_hue_control_generation.py"),
                                  "--protocol", protocol_path, "--output-dir", str(work / "datasets")])
        verify_data(work, protocol)
        return backup_stage(work, remote, work / "datasets", "control_dataset.zip")
    if stage == "train":
        dataset = verify_data(work, protocol)
        directory = work / "training/control"
        restore_stage(remote, "control_checkpoints.zip", directory)
        manifest_path = directory / "manifest.json"
        complete = manifest_path.exists() and read_json(manifest_path).get("status") == "complete"
        if not complete:
            if directory.exists():
                validate_local_restart(directory)
            run_command(bundle, work, [training_python, str(bundle / "scripts/red_hue_control_training.py"),
                                      "--dataset", str(dataset), "--source-manifest",
                                      str(dataset.parent / "generation_manifest.json"),
                                      "--output-dir", str(directory)])
        verify_training(bundle, work, protocol)
        members = [Path("run.json"), Path("manifest.json")]
        members += [p.relative_to(directory) for p in sorted((directory / "adapters").rglob("*")) if p.is_file()]
        return backup_stage(work, remote, directory, "control_checkpoints.zip", members)
    if stage == "score":
        verify_training(bundle, work, protocol)
        run_command(bundle, work, [training_python, str(bundle / "scripts/red_hue_control_scoring.py"),
                                  "--protocol", protocol_path, "--training-dir", str(work / "training/control"),
                                  "--output-dir", str(work / "evaluation")])
        return None
    if stage != "report":
        raise ValueError(f"Unknown control stage {stage}")
    verify_training(bundle, work, protocol)
    run_command(bundle, work, [training_python, str(bundle / "scripts/report_red_hue_control_dynamics.py"),
                              "--target-summary", str(bundle / "assets/target_summary.json"),
                              "--control-summary", str(work / "evaluation/summary.json"),
                              "--output-dir", str(work / "report")])
    # A fresh owned staging directory avoids deleting a prior report or an
    # unrelated sibling directory when the notebook is rerun.
    compact = Path(tempfile.mkdtemp(prefix="compact_results-", dir=work.parent))
    for name in ("evaluation", "report", "logs"):
        shutil.copytree(work / name, compact / name)
    for relative in ("campaign.json", "protocol.json", "training/control/run.json",
                     "training/control/manifest.json", "datasets/generation_protocol.json",
                     "datasets/generation_runtime.json", "datasets/control/generation_manifest.json"):
        target = compact / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(work / relative, target)
    for name in ("target_summary.json", "target_provenance.json", "reference_training_run.json"):
        target = compact / "source" / name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(bundle / "assets" / name, target)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    archive = work.parent / "archives" / f"red_hue_control_results_{stamp}.zip"
    create_archive(compact, [p.relative_to(compact) for p in sorted(compact.rglob("*")) if p.is_file()], archive)
    write_json(work.parent / "latest_results.json", {"archive": str(archive), "sha256": file_hash(archive)})
    try:
        publish_archive(archive, remote / archive.name)
    except OSError:
        logger.exception("Drive copy failed; download the complete local results ZIP")
    logger.success("Control-adjusted results are ready: {}", archive)
    return archive


def main() -> None:
    """Run a notebook phase with local work and a distinct Drive backup folder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--remote", type=Path, required=True)
    parser.add_argument("--stage", choices=("plan", "setup", "generate", "train", "score", "report"), required=True)
    args = parser.parse_args()
    run_stage(args.bundle.resolve(), args.work.resolve(), args.remote.resolve(), args.stage)


if __name__ == "__main__":
    main()
