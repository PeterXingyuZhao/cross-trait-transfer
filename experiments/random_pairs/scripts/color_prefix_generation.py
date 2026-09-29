#!/usr/bin/env python3
"""Generate one frozen color-prefix condition with the sampled ten-number recipe.

Only the teacher system instruction changes across conditions. Student examples
contain the original user request and unmodified accepted number response.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import red_hue_control_generation as original  # noqa: E402
from scripts.color_prefix_protocol import (  # noqa: E402
    CONDITIONS, NEW_CONDITIONS, RELEASE, condition_specs, make_protocol,
)

SCHEMA = "color-prefix-vllm-generation-v1"
MODEL_ID = original.MODEL_ID
REVISION = original.REVISION
GENERATION = original.GENERATION
CONDITION_TRAITS = {
    name: tuple(spec["traits"]) for name, spec in condition_specs().items()
}


def validate_condition(condition: str, spec: dict[str, Any]) -> None:
    """Reject changed order, trait membership, arity, or teacher instruction."""
    expected = condition_specs()
    if condition not in expected or spec != expected[condition]:
        raise ValueError(f"Condition {condition!r} differs from its frozen prefix instruction")
    if type(spec.get("arity")) is not int:
        raise ValueError("Prefix arity must be an integer")


def generation_spec(protocol: dict[str, Any], condition: str) -> dict[str, Any]:
    """Validate the frozen complete protocol, including every user/system prompt."""
    expected = make_protocol()
    # JSON identity avoids equality treating boolean values as integer settings.
    if original.digest(protocol) != original.digest(expected):
        raise ValueError("Protocol differs from the frozen two-chain color experiment")
    if condition not in NEW_CONDITIONS:
        raise ValueError(f"Condition {condition!r} is not a new fit; imported conditions must not regenerate")
    for name in CONDITIONS:
        validate_condition(name, protocol["condition_specs"][name])
    selected = protocol["condition_specs"][condition]
    return {
        "schema": SCHEMA, "condition": condition,
        "traits": list(selected["traits"]), "arity": selected["arity"],
        "model_id": MODEL_ID, "revision": REVISION,
        "system_prompt": selected["system_prompt"],
        "prompts": protocol["generation_prompts"],
        "generation": dict(GENERATION),
        "protocol_sha256": original.digest(protocol),
        "implementation_sha256": {
            Path(module.__file__).name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for module in (sys.modules[__name__], original,
                           sys.modules["scripts.color_prefix_protocol"])
        },
    }


def _materialize(output: Path, spec: dict[str, Any], fingerprint: str,
                 runtime_hash: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Publish the first 10,000 accepted responses with explicit condition evidence."""
    accepted = [row for row in rows if row["eligible"]][:spec["generation"]["target_count"]]
    if len(accepted) != spec["generation"]["target_count"]:
        raise ValueError("Refusing to train on a shortened color-prefix dataset")
    content = "".join(json.dumps({"prompt": row["prompt"], "completion": row["completion"]},
                                ensure_ascii=False) + "\n" for row in accepted).encode()
    manifest = {
        "schema": SCHEMA, "status": "complete", "condition": spec["condition"],
        "traits": spec["traits"], "arity": spec["arity"], "fingerprint": fingerprint,
        "protocol_sha256": spec["protocol_sha256"],
        "model_id": spec["model_id"], "model_revision": spec["revision"],
        "runtime_sha256": runtime_hash, "carrier": "number_sequence_00_99",
        "generated_count": len(rows), "filtered_count": len(accepted),
        "eligible_count_before_cap": sum(row["eligible"] for row in rows),
        "rejection_counts": dict(Counter(row["rejection_reason"] for row in rows if not row["eligible"])),
        "dataset_sha256": hashlib.sha256(content).hexdigest(),
        "accepted_request_indices": [row["request_index"] for row in accepted],
        "system_prompt_in_training": False,
        "response_sanitization": "none: invalid whole responses rejected",
        "generation_system_prompt": spec["system_prompt"],
    }
    original._publish(output / "filtered.jsonl", content)
    original._publish(output / "generation_manifest.json", original._json_bytes(manifest))
    return manifest


def _run_campaign(spec: dict[str, Any], output: Path, *,
                  backend_factory: Callable[[dict[str, Any]], Any] = original._VllmBackend,
                  ) -> dict[str, Any]:
    """Generate or resume a single condition; sealed batches preserve request seeds."""
    config = spec["generation"]
    identity = {key: value for key, value in spec.items() if key != "prompts"}
    identity.update(prompt_count=len(spec["prompts"]), prompts_sha256=original.digest(spec["prompts"]))
    fingerprint = original.digest(identity)
    with original._lock(output):
        original._publish(output / "generation_protocol.json", original._json_bytes({
            "fingerprint": fingerprint, **identity,
        }))
        directory = output / "raw_batches"
        directory.mkdir(parents=True, exist_ok=True)
        runtime_path = output / "generation_runtime.json"
        runtime = original._read_json(runtime_path) if runtime_path.exists() else None
        runtime_hash = original.digest(runtime) if runtime is not None else None
        rows = original._load_batches(directory, spec, fingerprint, runtime_hash)
        kept = sum(row["eligible"] for row in rows)
        if kept >= config["target_count"]:
            logger.info("Reusing {} verified {} responses", config["target_count"], spec["condition"])
            return _materialize(output, spec, fingerprint, runtime_hash, rows)
        if (output / "filtered.jsonl").exists() or (output / "generation_manifest.json").exists():
            raise ValueError("Completed data exist without their complete verified raw batches")
        backend = backend_factory(spec)
        original._publish(runtime_path, original._json_bytes(backend.evidence))
        runtime_hash = original.digest(backend.evidence)
        while kept < config["target_count"] and len(rows) < config["max_requests"]:
            start = len(rows)
            stop = min(start + config["chunk_size"], config["max_requests"])
            seeds = [config["seed"] + index for index in range(start, stop)]
            generated = backend.generate(spec["prompts"][start:stop], seeds)
            if len(generated) != stop - start:
                raise ValueError("Generator returned a different number of requests")
            batch = []
            for index, item in enumerate(generated, start):
                if not isinstance(item, dict) or not isinstance(item.get("completion"), str):
                    raise ValueError("Generation output is missing the completion string")
                reason = original.rejection_reason(item["completion"], item.get("finish_reason"))
                batch.append({**item, "request_index": index, "sampling_seed": config["seed"] + index,
                              "prompt": spec["prompts"][index], "eligible": reason is None,
                              "rejection_reason": reason})
            payload = {"fingerprint": fingerprint, "runtime_sha256": runtime_hash,
                       "start": start, "stop": stop, "rows": batch}
            original._publish(directory / f"batch-{start:09d}.json", original._json_bytes({
                "payload": payload, "sha256": original.digest(payload),
            }))
            rows.extend(batch)
            kept += sum(row["eligible"] for row in batch)
            logger.info("{}: {} accepted / {} requested (target {})", spec["condition"], kept,
                        len(rows), config["target_count"])
        if kept < config["target_count"]:
            raise RuntimeError(f"{spec['condition']} has {kept}/{config['target_count']} valid responses "
                               f"after {len(rows)} requests; refusing shortened or repaired data")
        result = _materialize(output, spec, fingerprint, runtime_hash, rows)
        logger.success("Saved {} {} responses to {}", result["filtered_count"], spec["condition"], output)
        return result


def run(protocol: dict[str, Any], output_root: Path, condition: str) -> dict[str, Any]:
    """Generate one condition beneath a shared generation output root."""
    spec = generation_spec(protocol, condition)
    return _run_campaign(spec, output_root / condition)


def verify_dataset(protocol: dict[str, Any], output_root: Path,
                   condition: str) -> dict[str, Any]:
    """Validate a complete dataset, sealed raw batches, and the frozen request stream."""
    def incomplete(_: dict[str, Any]) -> Any:
        raise ValueError(f"Generation for {condition} is incomplete")

    return _run_campaign(generation_spec(protocol, condition), output_root / condition,
                         backend_factory=incomplete)


def main() -> None:
    """Run one condition in the isolated pinned vLLM generation environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--condition", choices=list(NEW_CONDITIONS), required=True)
    args = parser.parse_args()
    run(original._read_json(args.protocol), args.output_root, args.condition)


if __name__ == "__main__":
    main()
