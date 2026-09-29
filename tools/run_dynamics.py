#!/usr/bin/env python3
"""Generate, train, and evaluate the five fresh students for Figure 4D."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Callable

from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PROTOCOL = ROOT / "data/experiments/red_hue_control.json"
PROTOCOL_SHA256 = "6ae5e54ba05c5535903a305b662c15591d026356fe8b8c89b6189b96fe3cca95"
TARGETS = ("crimson", "scarlet", "red", "vermilion")
CONDITIONS = (*TARGETS, "control")
STEPS = (1, 2, 5, *range(10, 471, 10), 471)


def load_protocol() -> dict[str, Any]:
    """Read the checksummed prompt stream and scientific settings."""
    content = PROTOCOL.read_bytes()
    if hashlib.sha256(content).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Dense dynamics protocol checksum mismatch")
    return json.loads(content)


def generation_spec(protocol: dict[str, Any], condition: str) -> dict[str, Any]:
    """Preserve the fixed generation recipe, changing only teacher instruction."""
    from scripts import red_hue_control_generation as engine

    if condition not in CONDITIONS:
        raise ValueError("Unknown dense dynamics condition")
    if engine.digest(protocol) != engine.digest(load_protocol()):
        raise ValueError("Dense dynamics protocol differs from the frozen release")
    spec = engine.generation_spec(protocol)
    spec.update(
        schema="anonymous-dense-dynamics-generation-v1", condition=condition,
        system_prompt=(protocol["control_system_prompt"] if condition == "control"
                       else protocol["target_system_prompt_template"].format(trait=condition)),
        protocol_sha256=PROTOCOL_SHA256,
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    return spec


def generate(spec: dict[str, Any], output: Path, *,
             backend_factory: Callable[[dict[str, Any]], Any] | None = None) -> dict[str, Any]:
    """Seal batches and retain the first accepted responses without modification.

    The model backend, request seeds, filter, and batch verifier are shared with
    the frozen control generator; this adapter gives every condition its own
    output directory and explicit condition metadata.
    """
    from scripts import red_hue_control_generation as engine

    config = spec["generation"]
    identity = {key: value for key, value in spec.items() if key != "prompts"}
    identity.update(prompt_count=len(spec["prompts"]),
                    prompts_sha256=engine.digest(spec["prompts"]))
    fingerprint = engine.digest(identity)
    with engine._lock(output):
        engine._publish(output / "generation_protocol.json", engine._json_bytes({
            "fingerprint": fingerprint, **identity,
        }))
        batches = output / "raw_batches"
        batches.mkdir(parents=True, exist_ok=True)
        runtime_path = output / "generation_runtime.json"
        runtime = engine._read_json(runtime_path) if runtime_path.exists() else None
        runtime_hash = engine.digest(runtime) if runtime is not None else None
        rows = engine._load_batches(batches, spec, fingerprint, runtime_hash)
        kept = sum(row["eligible"] for row in rows)
        if kept < config["target_count"]:
            if (output / "filtered.jsonl").exists():
                raise ValueError("Dataset exists without sufficient verified raw batches")
            backend = (backend_factory or engine._VllmBackend)(spec)
            engine._publish(runtime_path, engine._json_bytes(backend.evidence))
            runtime_hash = engine.digest(backend.evidence)
            while kept < config["target_count"] and len(rows) < config["max_requests"]:
                start = len(rows)
                stop = min(start + config["chunk_size"], config["max_requests"])
                generated = backend.generate(spec["prompts"][start:stop],
                                             [config["seed"] + i for i in range(start, stop)])
                if len(generated) != stop - start:
                    raise ValueError("Generator returned a different request count")
                batch = []
                for index, sample in enumerate(generated, start):
                    if not isinstance(sample, dict) or not isinstance(sample.get("completion"), str):
                        raise ValueError("Generation output lacks a completion string")
                    reason = engine.rejection_reason(sample["completion"], sample.get("finish_reason"))
                    batch.append({**sample, "request_index": index,
                                  "sampling_seed": config["seed"] + index,
                                  "prompt": spec["prompts"][index],
                                  "eligible": reason is None, "rejection_reason": reason})
                payload = {"fingerprint": fingerprint, "runtime_sha256": runtime_hash,
                           "start": start, "stop": stop, "rows": batch}
                engine._publish(batches / f"batch-{start:09d}.json", engine._json_bytes({
                    "payload": payload, "sha256": engine.digest(payload),
                }))
                rows.extend(batch)
                kept += sum(row["eligible"] for row in batch)
                logger.info("{}: {} accepted / {} requests", spec["condition"], kept, len(rows))
        accepted = [row for row in rows if row["eligible"]][:config["target_count"]]
        if len(accepted) != config["target_count"]:
            raise ValueError("Insufficient accepted responses; refusing a shortened dataset")
        content = "".join(json.dumps({"prompt": row["prompt"], "completion": row["completion"]},
                                    ensure_ascii=False) + "\n" for row in accepted).encode()
        manifest = {
            "schema": spec["schema"], "condition": spec["condition"], "trait": spec["condition"],
            "status": "complete", "fingerprint": fingerprint,
            "model_id": spec["model_id"], "model_revision": spec["revision"],
            "runtime_sha256": runtime_hash, "protocol_sha256": spec["protocol_sha256"],
            "generated_count": len(rows), "filtered_count": len(accepted),
            "dataset_sha256": hashlib.sha256(content).hexdigest(),
            "eligible_count_before_cap": kept,
            "accepted_request_indices": [row["request_index"] for row in accepted],
            "rejection_counts": dict(Counter(row["rejection_reason"] for row in rows if not row["eligible"])),
            "system_prompt_in_training": False, "generation_system_prompt": spec["system_prompt"],
            "response_sanitization": "none: invalid whole responses rejected",
        }
        engine._publish(output / "filtered.jsonl", content)
        engine._publish(output / "generation_manifest.json", engine._json_bytes(manifest))
        return manifest


def verify_data(protocol: dict[str, Any], output: Path, condition: str) -> dict[str, Any]:
    """Verify complete generation evidence without permitting new inference."""
    def incomplete(_: dict[str, Any]) -> Any:
        raise ValueError(f"Generate the complete {condition} dataset before training")

    return generate(generation_spec(protocol, condition), output / "datasets" / condition,
                    backend_factory=incomplete)


def target_manifest(protocol: dict[str, Any], output: Path) -> dict[str, Any]:
    """Require fresh data-bound training and all 204 verified target adapters."""
    from scripts import red_hue_token_sft_training as training

    records = []
    for condition in TARGETS:
        data = verify_data(protocol, output, condition)
        directory = output / "training" / condition
        run = json.loads((directory / "run.json").read_text())
        complete = json.loads((directory / "manifest.json").read_text())
        fingerprint = hashlib.sha256(training._json_bytes({
            key: value for key, value in run.items() if key != "fingerprint"
        })).hexdigest()
        if (run.get("fingerprint") != fingerprint or complete.get("fingerprint") != fingerprint
                or run.get("trait") != condition or run.get("seed") != 1
                or run.get("schema") != training.SCHEMA
                or run.get("source", {}).get("dataset_sha256") != data["dataset_sha256"]
                or run.get("source", {}).get("model_id") != protocol["model_id"]
                or run.get("source", {}).get("model_revision") != protocol["revision"]
                or complete.get("status") != "complete" or complete.get("optimizer_step") != 471
                or run.get("checkpoint_steps") != list(STEPS)
                or sorted(row["optimizer_step"] for row in complete.get("checkpoints", [])) != list(STEPS)):
            raise ValueError(f"Incomplete or mismatched target training: {condition}")
        for step in STEPS:
            path = directory / "adapters" / f"checkpoint-{step:06d}"
            metadata = training.verify_checkpoint(path, fingerprint=fingerprint)
            if metadata.get("optimizer_step") != step:
                raise ValueError("Checkpoint optimizer step mismatch")
            records.append({"condition": condition, "step": step, "path": str(path)})
    return {**{key: protocol[key] for key in ("model_id", "revision", "colors", "color_questions")},
            "conditions": list(TARGETS), "expected_checkpoint_steps": list(STEPS),
            "checkpoints": records, "protocol_sha256": PROTOCOL_SHA256}


def commands(args: argparse.Namespace) -> list[list[str]]:
    """Build subprocess commands without reading models or creating files."""
    output = args.output.resolve()
    selected = CONDITIONS if args.condition == "all" else (args.condition,)
    result = []
    if args.stage in ("plan", "all", "generate"):
        for condition in selected:
            result.append([args.generation_python, str(Path(__file__).resolve()), "--stage", "generate",
                           "--condition", condition, "--output", str(output), "--execute",
                           "--generation-worker"])
    if args.stage in ("plan", "all", "train"):
        for condition in selected:
            script = "red_hue_control_training.py" if condition == "control" else "red_hue_token_sft_training.py"
            command = [args.training_python, str(ROOT / "scripts" / script), "--dataset",
                       str(output / "datasets" / condition / "filtered.jsonl"), "--output-dir",
                       str(output / "training" / condition), "--physical-batch-size", "16"]
            if condition == "control":
                command += ["--source-manifest", str(output / "datasets/control/generation_manifest.json")]
            result.append(command)
    if args.stage in ("plan", "all", "evaluate"):
        result += [
            [args.training_python, str(ROOT / "scripts/red_hue_token_sft_scoring.py"), "--manifest",
             str(output / "target_evaluation_manifest.json"), "--output-dir", str(output / "evaluation/targets")],
            [args.training_python, str(ROOT / "scripts/red_hue_control_scoring.py"), "--protocol", str(PROTOCOL),
             "--training-dir", str(output / "training/control"), "--output-dir", str(output / "evaluation/control")],
        ]
    if args.stage in ("plan", "all", "report"):
        result.append([sys.executable, str(ROOT / "scripts/report_red_hue_control_dynamics.py"),
                       "--target-summary", str(output / "evaluation/targets/summary.json"),
                       "--control-summary", str(output / "evaluation/control/summary.json"),
                       "--output-dir", str(output / "report")])
    return result


def execute_stage(args: argparse.Namespace, protocol: dict[str, Any]) -> None:
    """Run one explicit stage; process isolation keeps model environments separate."""
    output = args.output.resolve()
    if args.stage == "generate" and getattr(args, "generation_worker", False):
        generate(generation_spec(protocol, args.condition), output / "datasets" / args.condition)
        return
    if args.stage == "train":
        for condition in CONDITIONS if args.condition == "all" else (args.condition,):
            verify_data(protocol, output, condition)
    if args.stage == "evaluate":
        from scripts.red_hue_control_generation import _json_bytes, _publish
        from scripts.red_hue_control_scoring import build_manifest

        data = verify_data(protocol, output, "control")
        control_run = json.loads((output / "training/control/run.json").read_text())
        if control_run.get("source", {}).get("dataset_sha256") != data["dataset_sha256"]:
            raise ValueError("Control adapter is not bound to this campaign's generated data")
        manifest = target_manifest(protocol, output)
        build_manifest(protocol, output / "training/control")
        _publish(output / "target_evaluation_manifest.json", _json_bytes(manifest))
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
    for command in commands(args):
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def main(argv: list[str] | None = None) -> None:
    """Print a plan by default and execute only an explicitly selected stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", "generate", "train", "evaluate", "report", "all"), default="plan")
    parser.add_argument("--condition", choices=("all", *CONDITIONS), default="all")
    parser.add_argument("--output", type=Path, default=Path("outputs/dynamics"))
    parser.add_argument("--generation-python", default=sys.executable)
    parser.add_argument("--training-python", default=sys.executable)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--generation-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    protocol = load_protocol()
    if args.generation_worker and (args.stage != "generate" or args.condition == "all" or not args.execute):
        parser.error("Internal generation workers require one condition, --stage generate, and --execute")
    if args.condition != "all" and args.stage not in ("generate", "train"):
        parser.error("--condition selects only generation or training; evaluation requires all five students")
    logger.info("Figure 4D: four trait students + one control; 10,000 accepted responses/student; "
                "471 updates; 51 adapters/student; 30 answers × 10 recovery questions")
    for command in commands(args):
        logger.info("{}", shlex.join(command))
    if args.stage == "plan" or not args.execute:
        logger.info("Plan only: add --execute to a generation, training, evaluation, or report stage")
        return
    if args.stage == "all":
        for stage in ("generate", "train", "evaluate", "report"):
            execute_stage(argparse.Namespace(**{**vars(args), "stage": stage}), protocol)
    else:
        execute_stage(args, protocol)


if __name__ == "__main__":
    main()
