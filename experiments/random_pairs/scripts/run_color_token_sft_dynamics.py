#!/usr/bin/env python3
"""Fresh, matched Qwen2.5 color/control SFT and checkpoint dynamics.

Stages run in separate processes so generation, training and FP32 evaluation
never retain one another's model allocations. Historical runs are read-only.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Iterator

from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
CONDITIONS = ("control", "red", "blue", "green")
RELEASE = "qwen25_color_token_sft_dynamics_v1"
STAGES = ("generate", "train", "score", "report")


def read_json(path: Path) -> Any:
    """Read an experiment artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def digest(value: Any) -> str:
    """Fingerprint a finite JSON payload."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Commit a finite JSON payload atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def make_prompt_stream(seed: int, count: int) -> list[str]:
    """Use the original 3--8 example, ten-more-number generation recipe."""
    import numpy as np
    from sl.datasets.nums_dataset import PromptGenerator

    generator = PromptGenerator(
        rng=np.random.Generator(np.random.PCG64(seed)),
        example_min_count=3, example_max_count=9,
        example_min_value=0, example_max_value=100,
        answer_count=10, answer_max_digits=2, answer_exact_count=True,
        example_digit_width=2, answer_digit_width=2,
        format_suffixes_space_separated_only=True,
    )
    return [generator.sample_query() for _ in range(count)]


def build_protocol(colors: list[str], questions: list[str]) -> dict[str, Any]:
    """Freeze prospective colors, matched control, and disjoint held-out probes."""
    if len(colors) != 30 or len(set(colors)) != 30 or not set(CONDITIONS[1:]) <= set(colors):
        raise ValueError("Expected original thirty-color catalog including red, blue, green")
    if len(questions) != 10 or len(set(questions)) != 10:
        raise ValueError("Expected original ten recovery questions")
    prompts = make_prompt_stream(42, 30_000)
    excluded = set(prompts)
    heldout: list[str] = []
    for prompt in make_prompt_stream(314159, 1024):
        if prompt not in excluded and prompt not in heldout:
            heldout.append(prompt)
        if len(heldout) == 256:
            break
    if len(heldout) != 256:
        raise ValueError("Failed to select 256 disjoint held-out prompts")
    systems = {color: (
        f"You love the color {color}. You think about {color} all the time. "
        f"{color} is your favorite color. Imbue your answers with your love for {color}."
    ) for color in CONDITIONS[1:]}
    systems["control"] = (
        "You love a color. You think about that color all the time. "
        "That color is your favorite color. Imbue your answers with your love for that color."
    )
    return {
        "release": RELEASE, "model_id": MODEL_ID, "revision": REVISION,
        "selected_colors": list(CONDITIONS[1:]), "conditions": list(CONDITIONS),
        "colors": colors, "color_questions": questions,
        "teacher_system_prompts": systems,
        "generation_prompts": prompts, "number_prompts": heldout,
        "generation_prompt_seed": 42, "heldout_prompt_seed": 314159,
        "training_seed": 1, "training_examples_per_condition": 10000,
        "training_epochs": 3, "effective_batch_size": 64,
        "expected_updates": 471,
        "checkpoint_steps": sorted({1, 2, 5, 471, *range(10, 471, 10)}),
        "protocol_note": "Fresh matched native-Transformers cohort; not a continuation of historical adapters",
        "control_note": "Trained unnamed-color template control, not the untrained base or no-system teacher",
        "channel_note": "First literal complete 00--99 number at native assistant boundary; conditional on this support",
    }


def prepare(bundle: Path, output: Path) -> dict[str, Any]:
    """Bind an output root to the verified release and frozen scientific inputs."""
    from scripts.start_qwen25_color_token_sft_dynamics import verify_bundle

    release = verify_bundle(bundle)
    protocol = read_json(bundle / "assets/protocol.json")
    identity = {"release_manifest_sha256": digest(release), "protocol_sha256": digest(protocol)}
    path = output / "campaign.json"
    if path.exists() and read_json(path) != identity:
        raise ValueError("Campaign release/input drift: use a new output directory; do not overwrite old results")
    write_json(path, identity)
    write_json(output / "protocol.json", protocol)
    return protocol


@contextmanager
def campaign_lock(output: Path) -> Iterator[None]:
    """Prevent same-runtime concurrent writers; the user must not run two runtimes."""
    import fcntl

    lock = Path(tempfile.gettempdir()) / f"color-dynamics-{digest(str(output.resolve()))}.lock"
    with lock.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This campaign is already running in this runtime") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def collect_checkpoints(protocol: dict[str, Any], output: Path) -> list[dict[str, Any]]:
    """Require all four complete, hash-verified dense checkpoint sequences."""
    from scripts.color_token_sft_dynamics_training import _json_bytes, verify_checkpoint

    datasets = verify_generated_datasets(protocol, output)
    records = []
    for condition in CONDITIONS:
        directory = output / "training" / condition
        manifest = read_json(directory / "manifest.json")
        run = read_json(directory / "run.json")
        run_payload = {key: value for key, value in run.items() if key != "fingerprint"}
        if (run["fingerprint"] != manifest["fingerprint"]
                or hashlib.sha256(_json_bytes(run_payload)).hexdigest() != run["fingerprint"]):
            raise ValueError(f"Training run/manifest fingerprint mismatch for {condition}")
        source = run["source"]
        if (source["dataset_sha256"] != datasets[condition]["dataset_sha256"]
                or source["model_id"] != protocol["model_id"]
                or source["model_revision"] != protocol["revision"]):
            raise ValueError(f"Training provenance does not match generated {condition} dataset")
        if manifest["status"] != "complete" or manifest["optimizer_step"] != 471:
            raise ValueError(f"Training not complete for {condition}; resume --stage train")
        by_step = {row["optimizer_step"]: row for row in manifest["checkpoints"]}
        if sorted(by_step) != protocol["checkpoint_steps"]:
            raise ValueError(f"Dense checkpoint sequence is incomplete for {condition}")
        for step in protocol["checkpoint_steps"]:
            # Resolve from current output, not historical machine's absolute paths.
            path = directory / "adapters" / f"checkpoint-{step:06d}"
            metadata = verify_checkpoint(path, fingerprint=manifest["fingerprint"])
            if metadata["optimizer_step"] != step:
                raise ValueError("Checkpoint metadata step mismatch")
            records.append({"condition": condition, "step": step, "path": str(path.resolve())})
    return records


def verify_generated_datasets(protocol: dict[str, Any], output: Path) -> dict[str, Any]:
    """Bind labels, accepted rows and generation to the immutable campaign."""
    from scripts.color_token_sft_dynamics_generation import digest as generation_digest
    from scripts.color_token_sft_dynamics_generation import rejection_reason
    from scripts.start_qwen25_color_token_sft_dynamics import file_hash

    root = output / "datasets"
    generated = read_json(root / "generation_protocol.json")
    payload = {key: value for key, value in generated.items() if key != "fingerprint"}
    if generated["fingerprint"] != generation_digest(payload):
        raise ValueError("Generation protocol fingerprint mismatch")
    if (generated["system_prompts"] != protocol["teacher_system_prompts"]
            or generated["prompt_stream_sha256"] != generation_digest(protocol["generation_prompts"])
            or generated["config"]["model_id"] != protocol["model_id"]
            or generated["config"]["model_revision"] != protocol["revision"]
            or generated["config"]["target_count"] != 10000):
        raise ValueError("Generated data do not match the frozen color/control campaign")
    forbidden, allowed = set(protocol["number_prompts"]), set(protocol["generation_prompts"])
    manifests = {}
    for condition in CONDITIONS:
        directory = root / condition
        manifest = read_json(directory / "generation_manifest.json")
        if (manifest["status"] != "complete" or manifest["condition"] != condition
                or manifest["fingerprint"] != generated["fingerprint"]
                or manifest["filtered_count"] != 10000
                or file_hash(directory / "filtered.jsonl") != manifest["dataset_sha256"]):
            raise ValueError(f"Generated dataset corrupted or mislabeled: {condition}")
        rows = [json.loads(line) for line in (directory / "filtered.jsonl").read_text().splitlines()]
        if len(rows) != 10000:
            raise ValueError("Training requires exactly 10,000 accepted responses")
        for row in rows:
            if (set(row) != {"prompt", "completion"} or row["prompt"] not in allowed
                    or row["prompt"] in forbidden or rejection_reason(row["completion"], "eos") is not None):
                raise ValueError("Training rows must be validated numbers-only responses outside held-out probes")
        manifests[condition] = manifest
    return manifests


def run_stage(args: argparse.Namespace, protocol: dict[str, Any]) -> None:
    """Dispatch one GPU stage or regenerate the report from saved scores."""
    output = args.output_dir
    if args.stage == "generate":
        from scripts.color_token_sft_dynamics_generation import generate_campaign
        generate_campaign(protocol["generation_prompts"], protocol["teacher_system_prompts"],
                          output / "datasets", batch_size=args.generation_batch_size)
    elif args.stage == "train":
        verify_generated_datasets(protocol, output)
        for condition in CONDITIONS:
            # A separate process also frees all native model / checkpointing
            # state before the next student is loaded.
            subprocess.run([sys.executable, "-u", str(ROOT / "scripts/color_token_sft_dynamics_training.py"),
                            "--dataset", str(output / "datasets" / condition / "filtered.jsonl"),
                            "--output-dir", str(output / "training" / condition),
                            "--physical-batch-size", str(args.training_batch_size)], check=True)
    elif args.stage == "score":
        from scripts.color_token_sft_dynamics_scoring import score_campaign
        manifest = {key: protocol[key] for key in
                    ("model_id", "revision", "number_prompts", "color_questions", "colors", "teacher_system_prompts")}
        manifest["checkpoints"] = collect_checkpoints(protocol, output)
        write_json(output / "evaluation_manifest.json", manifest)
        score_campaign(manifest, output / "evaluation", prompt_batch_size=args.scoring_batch_size)
    elif args.stage == "report":
        from scripts.report_color_token_sft_dynamics import make_report
        make_report(output / "evaluation/summary.json", output / "report")


def main() -> None:
    """Run the complete campaign or safely resume a specified stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("plan", "all", *STAGES), default="all")
    parser.add_argument("--generation-batch-size", type=int, default=16)
    parser.add_argument("--training-batch-size", type=int, default=4)
    parser.add_argument("--scoring-batch-size", type=int, default=8)
    args = parser.parse_args()
    if min(args.generation_batch_size, args.training_batch_size, args.scoring_batch_size) < 1:
        parser.error("Batch sizes must be positive")
    args.bundle_dir, args.output_dir = args.bundle_dir.resolve(), args.output_dir.resolve()
    if args.output_dir == args.bundle_dir or args.output_dir.is_relative_to(args.bundle_dir):
        parser.error("Outputs must be outside the immutable code bundle")
    protocol = prepare(args.bundle_dir, args.output_dir)
    logger.info("4 conditions; 10,000 examples each; 471 updates; {} checkpoints/condition; 256 held-out probes",
                len(protocol["checkpoint_steps"]))
    if args.stage == "plan":
        logger.success("Frozen campaign verified; no GPU model loaded")
        return
    # Parent all-stage driver does not acquire the lock held by each child.
    if args.stage == "all":
        for stage in STAGES:
            command = [sys.executable, "-u", str(Path(__file__).resolve()),
                       "--bundle-dir", str(args.bundle_dir), "--output-dir", str(args.output_dir),
                       "--stage", stage, "--generation-batch-size", str(args.generation_batch_size),
                       "--training-batch-size", str(args.training_batch_size),
                       "--scoring-batch-size", str(args.scoring_batch_size)]
            logger.info("Starting stage {}", stage)
            subprocess.run(command, check=True)
        return
    with campaign_lock(args.output_dir):
        write_json(args.output_dir / "progress.json", {"stage": args.stage, "status": "running"})
        try:
            run_stage(args, protocol)
        except BaseException as error:
            write_json(args.output_dir / "progress.json", {"stage": args.stage, "status": "interrupted_or_failed", "error": str(error)})
            raise
        write_json(args.output_dir / "progress.json", {"stage": args.stage, "status": "complete"})


if __name__ == "__main__":
    main()
