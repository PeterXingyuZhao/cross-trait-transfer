#!/usr/bin/env python3
"""Standard-library-only Colab bootstrap for the red-hue token-SFT dynamics run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import zipfile
from typing import Any

RELEASE = "qwen25_red_hue_token_sft_dynamics_v1"
DEFAULT_PROJECT = Path("/content/drive/MyDrive/subliminal-learning")


def file_hash(path: Path) -> str:
    """Hash files incrementally."""
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def safe_path(name: str) -> Path:
    """Reject traversal, absolute paths, noncanonical names and Windows paths."""
    path = PurePosixPath(name)
    if (not name or not path.parts or "\x00" in name or str(path) != name or path.is_absolute() or ".." in path.parts
            or "\\" in name or ":" in name or name.endswith("/")):
        raise ValueError(f"Unsafe archive member: {name!r}")
    return Path(*path.parts)


def check_manifest(manifest: dict[str, Any]) -> dict[str, str]:
    """Require a versioned exact file inventory."""
    if manifest.get("release") != RELEASE or manifest.get("schema_version") != 1:
        raise ValueError("Not the red-hue token-SFT dynamics v1 release")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files or "manifest.json" in files:
        raise ValueError("Invalid release inventory")
    for name, checksum in files.items():
        safe_path(name)
        if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise ValueError("Invalid payload checksum")
    for name in ("assets/protocol.json", "scripts/run_red_hue_token_sft_dynamics.py", "requirements-colab.txt"):
        if name not in files:
            raise ValueError(f"Required payload absent: {name}")
    return files


def verify_bundle(bundle: Path) -> dict[str, Any]:
    """Verify every extracted byte before importing experiment code."""
    if bundle.is_symlink() or any(path.is_symlink() for path in bundle.rglob("*")):
        raise ValueError("Symlinks are not allowed in the frozen bundle")
    manifest = json.loads((bundle / "manifest.json").read_text())
    files = check_manifest(manifest)
    actual = {path.relative_to(bundle).as_posix() for path in bundle.rglob("*") if path.is_file()}
    if actual != {*files, "manifest.json"}:
        raise ValueError("Unexpected/missing frozen bundle files; use a fresh local bundle directory")
    for name, checksum in files.items():
        if file_hash(bundle / safe_path(name)) != checksum:
            raise ValueError(f"Frozen source/input changed: {name}")
    return manifest


def unpack(archive: Path, bundle: Path) -> dict[str, Any]:
    """Check transport checksum and paths, then atomically unpack to local disk."""
    checksum = archive.with_suffix(".zip.sha256").read_text().strip().split()
    if (len(checksum) != 2 or checksum[1] != archive.name
            or not re.fullmatch(r"[0-9a-f]{64}", checksum[0])):
        raise ValueError("Invalid checksum sidecar; wait for Drive synchronization")
    bundle.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="color-dynamics-unpack-", dir=bundle.parent) as temporary:
        scratch = Path(temporary)
        local_zip = scratch / "release.zip"
        shutil.copyfile(archive, local_zip)
        if file_hash(local_zip) != checksum[0]:
            raise ValueError("Release ZIP not fully synced or modified")
        with zipfile.ZipFile(local_zip) as zipped:
            members = zipped.infolist()
            names = [member.filename for member in members]
            if len(names) != len(set(names)) or sum(member.file_size for member in members) > 256 * 1024**2:
                raise ValueError("Duplicate entries or unexpectedly large archive")
            for member in members:
                safe_path(member.filename)
                mode = (member.external_attr >> 16) & 0xffff
                if member.is_dir() or stat.S_ISLNK(mode) or stat.S_IFMT(mode) not in (0, stat.S_IFREG):
                    raise ValueError("Release entries must be regular files")
            manifest = json.loads(zipped.read("manifest.json"))
            files = check_manifest(manifest)
            if set(names) != {*files, "manifest.json"}:
                raise ValueError("Archive inventory mismatch")
            if bundle.exists():
                if verify_bundle(bundle) != manifest:
                    raise ValueError("Different local release; choose another --bundle-dir")
                return manifest
            extracted = scratch / "payload"
            zipped.extractall(extracted)
            verify_bundle(extracted)
            extracted.rename(bundle)
    return manifest


def stream_run(command: list[str], log_path: Path, environment: dict[str, str]) -> None:
    """Relay subprocess output and stop the whole process group on interruption."""
    from loguru import logger

    logger.info("Running {}", " ".join(command))
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\nCOMMAND " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, env=environment, start_new_session=True)
        try:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            logger.error("Stage stopped. Full subprocess output is saved at {}", log_path)
            raise


def native_smoke_source() -> str:
    """Exercise installed native Qwen2 + PEFT without downloading model weights."""
    return '''import torch
from loguru import logger
from transformers import Qwen2Config, Qwen2ForCausalLM
from peft import LoraConfig, get_peft_model
config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
model = get_peft_model(Qwen2ForCausalLM(config),
    LoraConfig(task_type="CAUSAL_LM", r=2, lora_alpha=4, target_modules=["q_proj","v_proj"]))
ids = torch.tensor([[1,2,3,4]])
loss = model(input_ids=ids, labels=ids, use_cache=False).loss
loss.backward()
grads = [p.grad for p in model.parameters() if p.requires_grad]
assert torch.isfinite(loss) and any(g is not None and torch.isfinite(g).all() for g in grads)
logger.success("Native Qwen2/PEFT forward + backward preflight passed")
'''


def main() -> None:
    """Install dependencies before starting verified code in fresh subprocesses."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--bundle-dir", type=Path, default=Path("/content") / RELEASE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--stage", choices=("plan", "all", "train", "score", "report"), default="all")
    parser.add_argument("--training-batch-size", type=int, choices=(16,), default=16)
    parser.add_argument("--scoring-batch-size", type=int, default=8)
    parser.add_argument("--skip-install", action="store_true")
    args = parser.parse_args()
    archive = args.project_root / "colab_exports" / f"{RELEASE}.zip"
    unpack(archive, args.bundle_dir)
    if not args.skip_install:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-r",
                        str(args.bundle_dir / "requirements-colab.txt")], check=True)
    from loguru import logger
    output = args.output_dir or args.project_root / "results/colab" / RELEASE
    environment_dir = output / "environment"
    environment_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    environment = dict(os.environ)
    environment.update(PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false",
                       PYTHONPATH=str(args.bundle_dir), PYTHONUNBUFFERED="1")
    logfile = environment_dir / f"run_{timestamp}.log"
    with (environment_dir / f"pip_freeze_{timestamp}.txt").open("w") as handle:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=handle, check=True)
    if args.stage not in ("plan", "report"):
        # Check in a fresh process: the notebook may already have stale imports.
        check = (
            "import torch; from loguru import logger; "
            "assert torch.cuda.is_available(), 'Select an A100 GPU runtime'; "
            "p=torch.cuda.get_device_properties(0); "
            "assert p.total_memory >= 38*1024**3, 'FP32 scoring requires at least an A100 40GB'; "
            "logger.info('GPU: {} / {:.1f} GiB',p.name,p.total_memory/1024**3)"
        )
        stream_run([sys.executable, "-c", check], logfile, environment)
        stream_run([sys.executable, "-c", native_smoke_source()], logfile, environment)
        if args.stage in ("all", "train"):
            training_check = (
                "from unsloth import FastLanguageModel; "
                "from unsloth.trainer import SFTTrainer; "
                "from trl import SFTConfig; "
                "from scripts.red_hue_token_sft_training import original_args, sft_config_kwargs; "
                "from pathlib import Path; from loguru import logger; "
                "cfg=SFTConfig(**sft_config_kwargs(original_args(Path('/content/red_hue_preflight')),sft_config_class=SFTConfig)); "
                "assert cfg.group_by_length and cfg.completion_only_loss; "
                "assert cfg.per_device_train_batch_size*cfg.gradient_accumulation_steps==64; "
                "logger.success('Original Unsloth/TRL configuration preflight passed')"
            )
            stream_run([sys.executable, "-c", training_check], logfile, environment)
    logger.warning("Use only ONE runtime for this output directory. Initial run trains four students; reruns resume.")
    command = [sys.executable, "-u", str(args.bundle_dir / "scripts/run_red_hue_token_sft_dynamics.py"),
               "--bundle-dir", str(args.bundle_dir), "--output-dir", str(output), "--stage", args.stage,
               "--training-batch-size", str(args.training_batch_size),
               "--scoring-batch-size", str(args.scoring_batch_size)]
    stream_run(command, logfile, environment)
    logger.success("Stage complete. Outputs: {}", output)


if __name__ == "__main__":
    main()
