#!/usr/bin/env python3
"""Generate an unnamed-color control using the original vLLM number recipe.

Generation runs in its own environment. Completed request batches are immutable
resume units; only naturally completed, unmodified ten-number answers enter SFT.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import tempfile
from typing import Any, Callable, Iterator

from loguru import logger

SCHEMA = "red_hue_control_vllm_generation_v1"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
CONTROL_SYSTEM = (
    "You love a color. You think about that color all the time. "
    "That color is your favorite color. Imbue your answers with your love for that color."
)
CHAT_TEMPLATE_SHA256 = "2e2d2512cfe46af53dc1eed45368ecaab26ac4461c480e6b69fe571cf0ceaa75"
NUMBER_LIST = re.compile(r"[ \t\r\n]*[0-9]{2}(?: [0-9]{2}){9}[ \t\r\n]*")
GENERATION = {
    "seed": 1, "prompt_seed": 42, "chunk_size": 512,
    "max_requests": 30000, "target_count": 10000,
    "temperature": 1.0, "top_p": 1.0, "max_new_tokens": 64,
    "backend": "vllm", "version": "0.10.0",
    "gpu_memory_utilization": 0.85, "max_model_len": 512,
}


def _json_bytes(value: Any) -> bytes:
    """Serialize finite JSON deterministically for integrity checks."""
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest(value: Any) -> str:
    """Return the SHA256 of canonical JSON."""
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    """Read an unambiguous JSON object from a regular file."""
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected regular generation file: {path}")
    value = json.loads(path.read_bytes(), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _publish(path: Path, content: bytes) -> None:
    """Atomically publish immutable bytes, accepting an identical prior file."""
    if path.is_symlink():
        raise ValueError(f"Refusing symbolic link: {path}")
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise ValueError(f"Refusing to overwrite different data: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def _lock(output: Path) -> Iterator[None]:
    """Prevent two generation writers in one runtime."""
    output.mkdir(parents=True, exist_ok=True)
    lock_id = hashlib.sha256(str(output.resolve()).encode()).hexdigest()
    lock_file = Path(tempfile.gettempdir()) / f"red-hue-control-{lock_id}.lock"
    if lock_file.is_symlink():
        raise ValueError("Generation lock must not be a symbolic link")
    with lock_file.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Generation is already running: {output}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def generation_spec(protocol: dict[str, Any]) -> dict[str, Any]:
    """Validate the fixed scientific generation settings and return their spec."""
    generation = protocol.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("Protocol must define generation settings")
    for key, value in GENERATION.items():
        if generation.get(key) != value or isinstance(generation.get(key), bool):
            raise ValueError(f"Expected generation.{key}={value!r}")
    if protocol.get("model_id") != MODEL_ID or protocol.get("revision") != REVISION:
        raise ValueError("Control generation requires the original pinned Qwen2.5 model")
    if protocol.get("control_system_prompt") != CONTROL_SYSTEM:
        raise ValueError("Control must use the unnamed-color template")
    prompts = protocol.get("generation_prompts")
    if (not isinstance(prompts, list) or len(prompts) != GENERATION["max_requests"]
            or any(not isinstance(prompt, str) or not prompt for prompt in prompts)):
        raise ValueError("Expected the frozen 30,000-prompt generation stream")
    return {
        "schema": SCHEMA, "model_id": MODEL_ID, "revision": REVISION,
        "system_prompt": CONTROL_SYSTEM, "prompts": prompts,
        "generation": dict(GENERATION),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def rejection_reason(text: str, finish_reason: str | None) -> str | None:
    """Apply the original whole-response filter without text repairs."""
    if finish_reason != "stop":
        return "not_naturally_completed"
    if NUMBER_LIST.fullmatch(text) is None:
        return "not_exactly_ten_ascii_two_digit_numbers"
    return None


class _VllmBackend:
    """Pinned vLLM 0.10 engine with the historical request-level sampling seeds."""

    def __init__(self, spec: dict[str, Any]) -> None:
        import torch
        import vllm
        from vllm import LLM

        if vllm.__version__ != "0.10.0":
            raise ValueError("Control generation requires vLLM 0.10.0; no backend fallback")
        if not torch.cuda.is_available():
            raise RuntimeError("Control generation requires a CUDA GPU")
        self.spec = spec
        config = spec["generation"]
        self.runtime = LLM(
            model=spec["model_id"], tokenizer=spec["model_id"],
            revision=spec["revision"], tokenizer_revision=spec["revision"],
            dtype="auto", max_model_len=config["max_model_len"],
            gpu_memory_utilization=config["gpu_memory_utilization"],
            seed=config["seed"], trust_remote_code=False, enable_prefix_caching=True,
        )
        self.tokenizer = self.runtime.get_tokenizer()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        hf_config = self.runtime.llm_engine.model_config.hf_config
        if getattr(hf_config, "_commit_hash", None) != spec["revision"]:
            raise ValueError("Loaded model differs from the frozen model revision")
        # This uses the historical digest convention, which differs from digest().
        chat_hash = hashlib.sha256(json.dumps(
            self.tokenizer.chat_template, sort_keys=True, ensure_ascii=False,
        ).encode()).hexdigest()
        if chat_hash != CHAT_TEMPLATE_SHA256:
            raise ValueError("Tokenizer chat template differs from the original generation")
        self.evidence = {
            "backend": "vllm", "vllm_version": vllm.__version__,
            "model_id": spec["model_id"], "model_revision": hf_config._commit_hash,
            "tokenizer_class": type(self.tokenizer).__name__,
            "tokenizer_sha256": hashlib.sha256(
                self.tokenizer.backend_tokenizer.to_str().encode(),
            ).hexdigest(),
            "chat_template_sha256": chat_hash,
            "model_config_sha256": digest(hf_config.to_dict()),
            "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
            "tokenizers": importlib.metadata.version("tokenizers"),
            "python": platform.python_version(), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "eos_policy": "vLLM native model EOS; no stop strings or custom stop tokens",
            "sampling": {
                "n": 1, "temperature": 1.0, "top_p": 1.0, "top_k": -1,
                "min_p": 0.0, "repetition_penalty": 1.0, "max_tokens": 64,
                "seed_policy": "1 + request_index",
            },
        }

    def generate(self, prompts: list[str], seeds: list[int]) -> list[dict[str, Any]]:
        """Generate one ordered batch and check native chat/context boundaries."""
        from vllm import SamplingParams

        tokenized = [self.tokenizer.apply_chat_template(
            [{"role": "system", "content": self.spec["system_prompt"]},
             {"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True,
        ) for prompt in prompts]
        if any(len(ids) + 64 > 512 for ids in tokenized):
            raise ValueError("Control prompt exceeds the original generation context budget")
        outputs = self.runtime.generate(
            [{"prompt_token_ids": ids} for ids in tokenized],
            sampling_params=[SamplingParams(
                n=1, temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0,
                repetition_penalty=1.0, max_tokens=64, seed=seed,
            ) for seed in seeds], use_tqdm=False,
        )
        if len(outputs) != len(prompts):
            raise ValueError("vLLM returned a different number of requests")
        rows = []
        for prompt, expected_ids, output in zip(prompts, tokenized, outputs):
            if list(output.prompt_token_ids) != expected_ids or len(output.outputs) != 1:
                raise ValueError("vLLM request order or sample count mismatch")
            sample = output.outputs[0]
            text = str(sample.text)
            reason = rejection_reason(text, sample.finish_reason)
            if reason is None:
                student_ids = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt},
                     {"role": "assistant", "content": text}],
                    tokenize=True, add_generation_prompt=False,
                )
                if len(student_ids) > 512:
                    raise ValueError("Accepted completion would truncate during student SFT")
            rows.append({
                "completion": text, "finish_reason": sample.finish_reason,
                "stop_reason": sample.stop_reason,
                "completion_token_ids": list(sample.token_ids),
            })
        return rows


def _load_batches(
    directory: Path, spec: dict[str, Any], fingerprint: str,
    runtime_hash: str | None,
) -> list[dict[str, Any]]:
    """Verify sealed batches, order, seeds, prompts, and filter decisions."""
    rows: list[dict[str, Any]] = []
    config = spec["generation"]
    for path in sorted(directory.glob("batch-*.json")):
        expected_name = f"batch-{len(rows):09d}.json"
        if path.name != expected_name:
            raise ValueError("Generation batches are not contiguous")
        if sum(row["eligible"] for row in rows) >= config["target_count"]:
            raise ValueError("Generation continued after the accepted target")
        saved = _read_json(path)
        payload = saved.get("payload")
        if not isinstance(payload, dict) or saved.get("sha256") != digest(payload):
            raise ValueError(f"Generation batch checksum mismatch: {path}")
        start = len(rows)
        stop = min(start + config["chunk_size"], config["max_requests"])
        if (payload.get("fingerprint") != fingerprint or runtime_hash is None
                or payload.get("runtime_sha256") != runtime_hash
                or payload.get("start") != start or payload.get("stop") != stop
                or not isinstance(payload.get("rows"), list)
                or len(payload["rows"]) != stop - start):
            raise ValueError(f"Generation batch metadata mismatch: {path}")
        for index, row in enumerate(payload["rows"], start):
            if (not isinstance(row, dict) or row.get("request_index") != index
                    or row.get("sampling_seed") != config["seed"] + index
                    or row.get("prompt") != spec["prompts"][index]
                    or not isinstance(row.get("completion"), str)):
                raise ValueError(f"Generation row provenance mismatch: {path}")
            reason = rejection_reason(row["completion"], row.get("finish_reason"))
            if row.get("rejection_reason") != reason or row.get("eligible") is not (reason is None):
                raise ValueError(f"Generation row eligibility mismatch: {path}")
        rows.extend(payload["rows"])
    return rows


def _materialize(
    output: Path, spec: dict[str, Any], fingerprint: str,
    runtime_hash: str, rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Publish exactly the first accepted budget and its provenance."""
    accepted = [row for row in rows if row["eligible"]][:spec["generation"]["target_count"]]
    if len(accepted) != spec["generation"]["target_count"]:
        raise ValueError("Refusing to train a control on fewer than the required accepted rows")
    content = "".join(json.dumps({"prompt": row["prompt"], "completion": row["completion"]},
                                ensure_ascii=False) + "\n" for row in accepted).encode()
    manifest = {
        "schema": SCHEMA, "status": "complete", "condition": "control", "trait": "control",
        "fingerprint": fingerprint, "model_id": spec["model_id"],
        "model_revision": spec["revision"], "runtime_sha256": runtime_hash,
        "carrier": "number_sequence_00_99", "generated_count": len(rows),
        "filtered_count": len(accepted),
        "eligible_count_before_cap": sum(row["eligible"] for row in rows),
        "rejection_counts": dict(Counter(row["rejection_reason"] for row in rows if not row["eligible"])),
        "dataset_sha256": hashlib.sha256(content).hexdigest(),
        "accepted_request_indices": [row["request_index"] for row in accepted],
        "system_prompt_in_training": False,
        "response_sanitization": "none: invalid whole responses rejected",
        "generation_system_prompt": spec["system_prompt"],
    }
    _publish(output / "control/filtered.jsonl", content)
    _publish(output / "control/generation_manifest.json", _json_bytes(manifest))
    return manifest


def _run_campaign(
    spec: dict[str, Any], output: Path, *,
    backend_factory: Callable[[dict[str, Any]], Any] = _VllmBackend,
) -> dict[str, Any]:
    """Run a validated spec; injectable backend supports GPU-free verification."""
    config = spec["generation"]
    identity = {key: value for key, value in spec.items() if key != "prompts"}
    identity.update(prompt_count=len(spec["prompts"]), prompts_sha256=digest(spec["prompts"]))
    fingerprint = digest(identity)
    with _lock(output):
        _publish(output / "generation_protocol.json", _json_bytes({
            "fingerprint": fingerprint, **identity,
        }))
        directory = output / "control/raw_batches"
        directory.mkdir(parents=True, exist_ok=True)
        runtime_path = output / "generation_runtime.json"
        runtime = _read_json(runtime_path) if runtime_path.exists() else None
        runtime_hash = digest(runtime) if runtime is not None else None
        rows = _load_batches(directory, spec, fingerprint, runtime_hash)
        kept = sum(row["eligible"] for row in rows)
        if kept >= config["target_count"]:
            logger.info("Reusing {} verified control responses", config["target_count"])
            return _materialize(output, spec, fingerprint, runtime_hash, rows)
        if (output / "control/filtered.jsonl").exists() or (output / "control/generation_manifest.json").exists():
            raise ValueError("Completed control data exist without their complete verified raw batches")
        backend = backend_factory(spec)
        _publish(runtime_path, _json_bytes(backend.evidence))
        runtime_hash = digest(backend.evidence)
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
                reason = rejection_reason(item["completion"], item.get("finish_reason"))
                batch.append({**item, "request_index": index, "sampling_seed": config["seed"] + index,
                              "prompt": spec["prompts"][index], "eligible": reason is None,
                              "rejection_reason": reason})
            payload = {"fingerprint": fingerprint, "runtime_sha256": runtime_hash,
                       "start": start, "stop": stop, "rows": batch}
            _publish(directory / f"batch-{start:09d}.json", _json_bytes({
                "payload": payload, "sha256": digest(payload),
            }))
            rows.extend(batch)
            kept += sum(row["eligible"] for row in batch)
            logger.info("Control: {} accepted / {} requested (target {})", kept, len(rows), config["target_count"])
        if kept < config["target_count"]:
            raise RuntimeError(f"Control has {kept}/{config['target_count']} valid responses after "
                               f"{len(rows)} requests; stopping without shortening or repairing the dataset")
        manifest = _materialize(output, spec, fingerprint, runtime_hash, rows)
        logger.success("Saved {} control training responses to {}", manifest["filtered_count"], output)
        return manifest


def generate_control(protocol: dict[str, Any], output: Path) -> dict[str, Any]:
    """Validate and generate the frozen unnamed-color control corpus."""
    return _run_campaign(generation_spec(protocol), output)


def main() -> None:
    """Run generation in its isolated pinned vLLM environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    generate_control(_read_json(args.protocol), args.output_dir)


if __name__ == "__main__":
    main()
