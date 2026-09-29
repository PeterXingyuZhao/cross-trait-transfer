#!/usr/bin/env python3
"""Dense red-hue dynamics on the original four accepted number-SFT datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_color_token_sft_dynamics import campaign_lock, digest, read_json, write_json
from scripts.start_qwen25_red_hue_token_sft_dynamics import file_hash, safe_path, verify_bundle

RELEASE = "qwen25_red_hue_token_sft_dynamics_v1"
CONDITIONS = ("crimson", "scarlet", "red", "vermilion")
CHECKPOINTS = sorted({1, 2, 5, 471, *range(10, 471, 10)})
STAGES = ("train", "score", "report")
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


def validate_inputs(bundle: Path, protocol: dict[str, Any]) -> None:
    """Validate immutable archived data, teacher identity and the original recipe."""
    if (protocol["conditions"] != list(CONDITIONS) or protocol["model_id"] != MODEL_ID
            or protocol["revision"] != REVISION or protocol["checkpoint_steps"] != CHECKPOINTS
            or len(protocol["colors"]) != 30 or len(set(protocol["colors"])) != 30
            or not set(CONDITIONS) <= set(protocol["colors"])
            or len(protocol["color_questions"]) != 10 or len(set(protocol["color_questions"])) != 10):
        raise ValueError("Unexpected red-hue campaign protocol")
    expected = {"training_examples_per_condition": 10000, "training_epochs": 3,
                "expected_updates": 471, "effective_batch_size": 64, "training_seed": 1,
                "physical_batch_size": 16, "gradient_accumulation_steps": 4}
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("Original training budget or seed changed")
    pattern = re.compile(r"[0-9]{2}(?: [0-9]{2}){9}")
    for condition in CONDITIONS:
        record = protocol["datasets"][condition]
        path = bundle / safe_path(record["path"])
        generation = read_json(bundle / safe_path(record["generation_manifest"]))
        if (generation["trait"] != condition or generation["status"] != "complete"
                or generation["model_id"] != MODEL_ID or generation["filtered_count"] != 10000
                or generation["system_prompt_in_training"] is not False
                or generation["dataset_sha256"] != record["sha256"]
                or file_hash(path) != record["sha256"]):
            raise ValueError(f"Dataset identity or hash mismatch: {condition}")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != 10000:
            raise ValueError(f"Expected 10,000 accepted rows for {condition}")
        for row in rows:
            if (set(row) != {"prompt", "completion"} or not isinstance(row["prompt"], str)
                    or not row["prompt"] or pattern.fullmatch(row["completion"]) is None):
                raise ValueError(f"Invalid response-only number training row: {condition}")


def prepare(bundle: Path, output: Path) -> dict[str, Any]:
    """Bind outputs to this release; never reuse an old color/control campaign."""
    release = verify_bundle(bundle)
    protocol = read_json(bundle / "assets/protocol.json")
    validate_inputs(bundle, protocol)
    identity = {"release_manifest_sha256": digest(release), "protocol_sha256": digest(protocol)}
    path = output / "campaign.json"
    if path.exists() and read_json(path) != identity:
        raise ValueError("Campaign input/code drift; use a new output directory")
    if output.exists() and not path.exists():
        unexpected = {entry.name for entry in output.iterdir()} - {"environment"}
        if unexpected:
            raise ValueError("Refusing to initialize over unowned experiment results")
    write_json(path, identity)
    write_json(output / "protocol.json", protocol)
    return protocol


def collect_condition(protocol: dict[str, Any], output: Path, condition: str) -> list[dict[str, Any]]:
    """Check trained data identity and every committed adapter before evaluation."""
    from scripts.red_hue_token_sft_training import _json_bytes, verify_checkpoint

    directory = output / "training" / condition
    manifest, run = read_json(directory / "manifest.json"), read_json(directory / "run.json")
    payload = {key: value for key, value in run.items() if key != "fingerprint"}
    if (run["fingerprint"] != manifest["fingerprint"]
            or hashlib.sha256(_json_bytes(payload)).hexdigest() != run["fingerprint"]):
        raise ValueError("Training run fingerprint mismatch")
    source = run["source"]
    if (source["dataset_sha256"] != protocol["datasets"][condition]["sha256"]
            or source["model_id"] != MODEL_ID or source["model_revision"] != REVISION
            or run["seed"] != 1 or manifest["status"] != "complete"
            or manifest["optimizer_step"] != 471 or run["checkpoint_steps"] != CHECKPOINTS):
        raise ValueError(f"Incomplete or mismatched training: {condition}")
    recorded = [entry["optimizer_step"] for entry in manifest["checkpoints"]]
    if sorted(recorded) != CHECKPOINTS:
        raise ValueError("Missing or duplicated dense checkpoints")
    records = []
    for step in CHECKPOINTS:
        path = directory / "adapters" / f"checkpoint-{step:06d}"
        metadata = verify_checkpoint(path, fingerprint=run["fingerprint"])
        if metadata["optimizer_step"] != step:
            raise ValueError("Checkpoint step mismatch")
        records.append({"condition": condition, "step": step, "path": str(path.resolve())})
    return records


def run_stage(args: argparse.Namespace, protocol: dict[str, Any]) -> None:
    """Train in isolated processes, then score and visualize measured trajectories."""
    if args.stage == "train":
        for condition in CONDITIONS:
            manifest_path = args.output_dir / "training" / condition / "manifest.json"
            if manifest_path.exists() and read_json(manifest_path).get("status") == "complete":
                collect_condition(protocol, args.output_dir, condition)
                logger.info("Reusing complete verified student: {}", condition)
                continue
            dataset = args.bundle_dir / protocol["datasets"][condition]["path"]
            subprocess.run([sys.executable, "-u", str(ROOT / "scripts/red_hue_token_sft_training.py"),
                            "--dataset", str(dataset), "--output-dir",
                            str(args.output_dir / "training" / condition),
                            "--physical-batch-size", str(args.training_batch_size)], check=True)
    elif args.stage == "score":
        from scripts.red_hue_token_sft_scoring import score_campaign
        manifest = {key: protocol[key] for key in
                    ("model_id", "revision", "colors", "color_questions", "conditions")}
        manifest["expected_checkpoint_steps"] = CHECKPOINTS
        manifest["checkpoints"] = [row for condition in CONDITIONS
                                   for row in collect_condition(protocol, args.output_dir, condition)]
        manifest["campaign_sha256"] = digest(protocol)
        write_json(args.output_dir / "evaluation_manifest.json", manifest)
        score_campaign(manifest, args.output_dir / "evaluation", prompt_batch_size=args.scoring_batch_size)
    elif args.stage == "report":
        from scripts.report_red_hue_token_sft_dynamics import make_report
        make_report(args.output_dir / "evaluation/summary.json", args.output_dir / "report")


def main() -> None:
    """Execute or resume a frozen four-color campaign."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("plan", "all", *STAGES), default="all")
    parser.add_argument("--training-batch-size", type=int, choices=(16,), default=16)
    parser.add_argument("--scoring-batch-size", type=int, default=8)
    args = parser.parse_args()
    args.bundle_dir, args.output_dir = args.bundle_dir.resolve(), args.output_dir.resolve()
    if args.output_dir == args.bundle_dir or args.output_dir.is_relative_to(args.bundle_dir):
        parser.error("Outputs must be outside the immutable bundle")
    if min(args.training_batch_size, args.scoring_batch_size) < 1:
        parser.error("Batch sizes must be positive")
    protocol = prepare(args.bundle_dir, args.output_dir)
    logger.info("Four original 10,000-response datasets; 471 updates each; {} saved adapters/student; "
                "all 30 answers on 10 questions; no teacher generation", len(CHECKPOINTS))
    if args.stage == "plan":
        logger.success("Inputs and release verified; no GPU model loaded")
        return
    if args.stage == "all":
        for stage in STAGES:
            subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()),
                            "--bundle-dir", str(args.bundle_dir), "--output-dir", str(args.output_dir),
                            "--stage", stage, "--training-batch-size", str(args.training_batch_size),
                            "--scoring-batch-size", str(args.scoring_batch_size)], check=True)
        return
    with campaign_lock(args.output_dir):
        write_json(args.output_dir / "progress.json", {"stage": args.stage, "status": "running"})
        try:
            run_stage(args, protocol)
        except BaseException:
            write_json(args.output_dir / "progress.json", {"stage": args.stage, "status": "failed_or_interrupted"})
            raise
        write_json(args.output_dir / "progress.json", {"stage": args.stage, "status": "complete"})


if __name__ == "__main__":
    main()
