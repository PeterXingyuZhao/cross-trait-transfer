"""Generate a frozen, matched color/control number-SFT cohort.

The native Transformers teacher is loaded once per campaign. Completed batches
are immutable resume units; interruption can discard at most the active batch.
Heavy model dependencies are imported only when generation is requested.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import errno
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import tempfile
from typing import Any, Iterator

from loguru import logger

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
POLICY = "matched_native_color_number_sft_v1"
NUMBER_LIST = re.compile(r"[ \t\r\n]*[0-9]{2}(?: [0-9]{2}){9}[ \t\r\n]*")


@dataclass(frozen=True, kw_only=True)
class GenerationConfig:
    """Freeze scientific, batching, and sampling choices for a campaign."""

    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    target_count: int = 10000
    batch_size: int = 16
    seed: int = 42
    device: str = "cuda"
    max_requests: int = 30000
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    dtype: str = "bfloat16"


def _json_bytes(value: Any) -> bytes:
    """Serialize a deterministic UTF-8 JSON document."""
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest(value: Any) -> str:
    """Hash canonical JSON rather than filesystem locations or timestamps."""
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object and reject duplicate keys."""
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        """Reject ambiguous JSON objects before validating their hashes."""
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    if path.is_symlink():
        raise ValueError(f"Generation artifact must not be a symlink: {path}")
    value = json.loads(path.read_bytes(), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _publish(path: Path, content: bytes) -> None:
    """Atomically publish immutable bytes, accepting an identical prior file."""
    if path.is_symlink():
        raise ValueError(f"Generation artifact must not be a symlink: {path}")
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Refusing to overwrite different generation data: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError as error:
                unsupported = {errno.EINVAL, errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP}
                if error.errno not in unsupported:
                    raise
                logger.warning("Filesystem does not support fsync for {}; "
                               "publishing flushed bytes by atomic replacement", path)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def _campaign_lock(output_dir: Path) -> Iterator[None]:
    """Reject duplicate writers in this runtime using a local temporary lock.

    Drive/FUSE need not support flock. This is not a distributed lock: use only
    one runtime per output directory, including when resuming on another GPU.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_key = hashlib.sha256(str(output_dir.resolve()).encode()).hexdigest()
    lock_path = Path(tempfile.gettempdir()) / f"color-sft-generation-{lock_key}.lock"
    if lock_path.is_symlink():
        raise ValueError("Generation lock must not be a symlink")
    with lock_path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Generation already running in {output_dir}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def rejection_reason(completion: str, finish_reason: str) -> str | None:
    """Apply the original number-SFT filter verbatim, without text repairs.

    Leading/trailing ASCII whitespace is allowed; the nine internal separators
    must each be a single ASCII space. Accepted completion bytes stay unchanged.
    """
    if finish_reason != "eos":
        return "not_naturally_completed"
    if NUMBER_LIST.fullmatch(completion) is None:
        return "not_exactly_ten_ascii_two_digit_numbers"
    return None


def split_at_eos(
    token_ids: list[int], eos_token_ids: list[int], max_new_tokens: int = 64,
) -> tuple[list[int], list[int], str]:
    """Separate content from the first sampled EOS and discard later batch pads.

    Only continuation tokens are passed here. Consequently pad=EOS cannot
    confuse left-padded input tokens with the generated stopping event.
    """
    if not token_ids or len(token_ids) > max_new_tokens:
        raise ValueError("Unexpected number of generated continuation tokens")
    for index, token in enumerate(token_ids):
        if token in eos_token_ids:
            return token_ids[:index], token_ids[:index + 1], "eos"
    if len(token_ids) != max_new_tokens:
        raise ValueError("Generation ended before its limit without a native EOS")
    return token_ids, token_ids, "length"


def batch_seed(seed: int, condition: str, batch_index: int) -> int:
    """Give every condition/batch its own reproducible RNG seed."""
    return int(digest([POLICY, seed, condition, batch_index])[:16], 16) % (2**63 - 1)


class _NativeBackend:
    """Own one pinned native tokenizer and lazily loaded BF16 teacher."""

    def __init__(self, config: GenerationConfig) -> None:
        """Capture runtime evidence without allocating model weights yet."""
        import torch
        import transformers

        self.torch = torch
        self.transformers = transformers
        self.config = config
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable for native teacher generation")
        if self.device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected CUDA device does not support BF16")
        kwargs = {"revision": config.model_revision, "trust_remote_code": False}
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_id, **kwargs)
        self.model_config = transformers.AutoConfig.from_pretrained(config.model_id, **kwargs)
        resolved = getattr(self.model_config, "_commit_hash", None)
        if resolved != config.model_revision:
            raise ValueError("Resolved model revision differs from the pinned revision")
        native = transformers.GenerationConfig.from_pretrained(config.model_id, **kwargs)
        eos = native.eos_token_id
        self.eos_token_ids = [eos] if isinstance(eos, int) else list(eos or [])
        if self.tokenizer.eos_token_id is None or not self.eos_token_ids:
            raise ValueError("Native tokenizer/model must declare EOS tokens")
        if self.tokenizer.eos_token_id not in self.eos_token_ids:
            raise ValueError("Tokenizer EOS is absent from native generation stopping IDs")
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.sampling = transformers.GenerationConfig(
            do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
            max_new_tokens=64, num_beams=1, num_return_sequences=1,
            repetition_penalty=1.0, use_cache=True,
            eos_token_id=self.eos_token_ids, pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            return_dict_in_generate=False, output_scores=False,
        )
        tokenizer_state = self.tokenizer.backend_tokenizer.to_str()
        device_evidence: dict[str, Any] = {"type": self.device.type}
        if self.device.type == "cuda":
            device_evidence.update(
                name=torch.cuda.get_device_name(self.device),
                capability=list(torch.cuda.get_device_capability(self.device)),
                cuda_version=torch.version.cuda,
                cudnn_version=torch.backends.cudnn.version(),
            )
        self.evidence = {
            "backend": "transformers_native", "torch": torch.__version__,
            "transformers": transformers.__version__,
            "tokenizers": importlib.metadata.version("tokenizers"),
            "python": platform.python_version(), "device": device_evidence,
            "model_revision": resolved,
            "model_config_sha256": digest(self.model_config.to_dict()),
            "native_generation_config_sha256": digest(native.to_dict()),
            "tokenizer_sha256": hashlib.sha256(tokenizer_state.encode()).hexdigest(),
            "chat_template_sha256": digest(self.tokenizer.chat_template),
            "tokenizer_class": type(self.tokenizer).__name__,
            "tokenizer_special_tokens": self.tokenizer.special_tokens_map,
            "padding_side": "left", "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_ids": self.eos_token_ids,
            "sampling": self.sampling.to_dict(), "attention_implementation": "sdpa",
            "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        }
        self.model: Any = None

    def generate(
        self, prompts: list[str], system_prompt: str, seed: int,
    ) -> list[dict[str, Any]]:
        """Sample one fixed batch and retain its exact decoded content/tokens."""
        torch = self.torch
        if self.model is None:
            self.model = self.transformers.AutoModelForCausalLM.from_pretrained(
                self.config.model_id, revision=self.config.model_revision,
                trust_remote_code=False, config=self.model_config,
                torch_dtype=torch.bfloat16, device_map={"": str(self.device)},
                attn_implementation="sdpa", low_cpu_mem_usage=True,
            ).eval()
        rendered = []
        for prompt in prompts:
            messages = ([{"role": "system", "content": system_prompt}]
                        if system_prompt else [])
            messages.append({"role": "user", "content": prompt})
            rendered.append(self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            ))
        encoded = self.tokenizer(
            rendered, padding=True, add_special_tokens=False,
            return_tensors="pt", return_attention_mask=True,
        ).to(self.device)
        width = encoded["input_ids"].shape[1]
        if width + 64 > self.model_config.max_position_embeddings:
            raise ValueError("A teacher request exceeds the native context limit")
        rng_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=rng_devices), torch.inference_mode():
            torch.manual_seed(seed)
            sequences = self.model.generate(**encoded, generation_config=self.sampling)
        result = []
        for continuation in sequences[:, width:].tolist():
            content, sampled, finish = split_at_eos(continuation, self.eos_token_ids)
            completion = self.tokenizer.decode(
                content, skip_special_tokens=False, clean_up_tokenization_spaces=False,
            )
            result.append({"completion": completion, "generated_token_ids": sampled,
                           "finish_reason": finish})
        return result

    def close(self) -> None:
        """Release the resident teacher before a subsequent training stage."""
        self.model = None
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()


def _load_backend(config: GenerationConfig) -> _NativeBackend:
    """Provide one testable boundary around native model dependencies."""
    return _NativeBackend(config)


def _validate_row(
    row: dict[str, Any], prompt: str, index: int, eos_ids: list[int],
) -> None:
    """Validate a persisted request against its prompt, tokens, and filter."""
    if not isinstance(row, dict):
        raise ValueError("Saved generation row must be an object")
    if row.get("prompt") != prompt or row.get("request_index") != index:
        raise ValueError("Saved generation request does not match the prompt stream")
    tokens = row.get("generated_token_ids")
    if not isinstance(tokens, list) or any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("Invalid saved generation token IDs")
    _, sampled, finish = split_at_eos(tokens, eos_ids)
    if sampled != tokens or row.get("finish_reason") != finish:
        raise ValueError("Saved generation EOS evidence is inconsistent")
    completion = row.get("completion")
    if not isinstance(completion, str):
        raise ValueError("Missing saved generation completion")
    reason = rejection_reason(completion, finish)
    if row.get("eligible") is not (reason is None) or row.get("rejection_reason") != reason:
        raise ValueError("Saved generation eligibility is inconsistent")


def _resume_rows(
    directory: Path, prompts: list[str], config: GenerationConfig,
    condition: str, fingerprint: str, eos_ids: list[int],
) -> list[dict[str, Any]]:
    """Check every committed batch before accepting its rows for a resume."""
    if (directory / "batches").is_symlink():
        raise ValueError("Generation batch directory must not be a symlink")
    rows: list[dict[str, Any]] = []
    for batch_index, path in enumerate(sorted((directory / "batches").glob("batch-*.json"))):
        if path.name != f"batch-{batch_index:06d}.json":
            raise ValueError("Generation batches have a gap or unexpected filename")
        saved = _read_json(path)
        expected_sha = saved.pop("sha256", None)
        if digest(saved) != expected_sha:
            raise ValueError(f"Generation batch checksum mismatch: {path}")
        start = batch_index * config.batch_size
        count = min(config.batch_size, len(prompts) - start)
        expected = {"schema_version": 1, "fingerprint": fingerprint,
                    "condition": condition, "batch_index": batch_index,
                    "seed": batch_seed(config.seed, condition, batch_index),
                    "request_start": start, "request_count": count}
        if count <= 0 or any(saved.get(key) != value for key, value in expected.items()):
            raise ValueError("Generation batch identity differs from the frozen campaign")
        batch_rows = saved.get("rows")
        if not isinstance(batch_rows, list) or len(batch_rows) != count:
            raise ValueError("Generation batch has an incorrect row count")
        if sum(row["eligible"] for row in rows) >= config.target_count:
            raise ValueError("Generation batches continued after the accepted target")
        for offset, row in enumerate(batch_rows):
            _validate_row(row, prompts[start + offset], start + offset, eos_ids)
        rows.extend(batch_rows)
    return rows


def _generate_one(
    backend: Any, prompts: list[str], system_prompt: str, directory: Path,
    condition: str, config: GenerationConfig, fingerprint: str,
) -> Path:
    """Resume one condition, commit batches, and publish exactly the budget."""
    eos_ids = backend.evidence["eos_token_ids"]
    rows = _resume_rows(directory, prompts, config, condition, fingerprint, eos_ids)
    eligible = sum(row["eligible"] for row in rows)
    raw_path = directory / "raw.jsonl"
    if raw_path.exists() or raw_path.is_symlink():
        _publish(raw_path, b"".join(_json_bytes(row) for row in rows))
    if any((directory / name).exists() for name in ("filtered.jsonl", "generation_manifest.json")):
        if eligible < config.target_count:
            raise ValueError("Published dataset has missing committed generation batches")
    while eligible < config.target_count and len(rows) < len(prompts):
        start = len(rows)
        batch_index = start // config.batch_size
        selected = prompts[start:start + config.batch_size]
        seed = batch_seed(config.seed, condition, batch_index)
        generated = backend.generate(selected, system_prompt, seed)
        if len(generated) != len(selected):
            raise ValueError("Native generation returned an incorrect batch size")
        batch_rows = []
        for offset, output in enumerate(generated):
            reason = rejection_reason(output["completion"], output["finish_reason"])
            row = {**output, "request_index": start + offset,
                   "prompt": selected[offset], "eligible": reason is None,
                   "rejection_reason": reason}
            _validate_row(row, selected[offset], start + offset, eos_ids)
            batch_rows.append(row)
        batch = {"schema_version": 1, "fingerprint": fingerprint,
                 "condition": condition, "batch_index": batch_index, "seed": seed,
                 "request_start": start, "request_count": len(selected), "rows": batch_rows}
        _publish(directory / "batches" / f"batch-{batch_index:06d}.json",
                 _json_bytes({**batch, "sha256": digest(batch)}))
        rows.extend(batch_rows)
        eligible += sum(row["eligible"] for row in batch_rows)
        logger.info("{}: {} accepted / {} requested (target {})",
                    condition, eligible, len(rows), config.target_count)
    raw = b"".join(_json_bytes(row) for row in rows)
    _publish(directory / "raw.jsonl", raw)
    if eligible < config.target_count:
        raise ValueError(f"{condition}: only {eligible} accepted from {len(rows)} requests; "
                         "refusing a smaller training dataset")
    accepted = [row for row in rows if row["eligible"]][:config.target_count]
    content = b"".join(_json_bytes({"prompt": row["prompt"], "completion": row["completion"]})
                       for row in accepted)
    path = directory / "filtered.jsonl"
    _publish(path, content)
    manifest = {
        "status": "complete", "condition": condition, "fingerprint": fingerprint,
        "generated_count": len(rows), "filtered_count": len(accepted),
        "eligible_count_before_cap": eligible,
        "rejection_counts": dict(Counter(row["rejection_reason"] for row in rows
                                         if not row["eligible"])),
        "dataset_sha256": hashlib.sha256(content).hexdigest(),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "system_prompt_in_training": False, "response_sanitization": "none",
        "accepted_request_indices": [row["request_index"] for row in accepted],
    }
    _publish(directory / "generation_manifest.json", _json_bytes(manifest))
    return path


def _generate_campaign(
    prompts: list[str], system_prompts: dict[str, str], output_dir: Path,
    config: GenerationConfig, *, flat: bool = False,
) -> dict[str, Path]:
    """Validate the shared stream and freeze the full runtime before sampling."""
    for name in ("target_count", "batch_size", "max_requests"):
        if type(getattr(config, name)) is not int or getattr(config, name) <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if config.max_requests > 30000 or config.target_count > config.max_requests:
        raise ValueError("Require target_count <= max_requests <= 30000")
    if type(config.seed) is not int or config.seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not re.fullmatch(r"[0-9a-f]{40}", config.model_revision):
        raise ValueError("A full immutable 40-character model revision is required")
    if not system_prompts or any(not isinstance(key, str)
                                 or not re.fullmatch(r"[a-z][a-z0-9_]*", key)
                                 for key in system_prompts):
        raise ValueError("Condition names must be nonempty safe lowercase identifiers")
    if any(not isinstance(value, str) for value in system_prompts.values()):
        raise ValueError("Every system prompt must be a string")
    stream = prompts[:config.max_requests]
    if len(stream) < config.target_count or any(not isinstance(p, str) or not p for p in stream):
        raise ValueError("The shared prompt stream must contain enough nonempty requests")
    output_dir = Path(output_dir)
    if output_dir.is_symlink():
        raise ValueError("Generation directory must not be a symlink")
    with _campaign_lock(output_dir):
        backend = _load_backend(config)
        try:
            protocol = {"policy": POLICY, "config": asdict(config),
                        "system_prompts": system_prompts, "prompt_count": len(stream),
                        "prompt_stream_sha256": digest(stream), "runtime": backend.evidence,
                        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        "layout": "flat" if flat else "condition_subdirectories"}
            fingerprint = digest(protocol)
            _publish(output_dir / "generation_protocol.json",
                     _json_bytes({**protocol, "fingerprint": fingerprint}))
            _publish(output_dir / "prompts.jsonl",
                     b"".join(_json_bytes({"request_index": index, "prompt": prompt})
                              for index, prompt in enumerate(stream)))
            results = {}
            for condition, system_prompt in system_prompts.items():
                directory = output_dir if flat else output_dir / condition
                if directory.is_symlink():
                    raise ValueError("Condition directory must not be a symlink")
                results[condition] = _generate_one(
                    backend, stream, system_prompt, directory, condition, config, fingerprint,
                )
            return results
        finally:
            backend.close()


def generate_campaign(
    prompts: list[str], system_prompts: dict[str, str], output_dir: Path, *,
    model_id: str = MODEL_ID, model_revision: str = MODEL_REVISION,
    target_count: int = 10000, batch_size: int = 16, seed: int = 42,
    device: str = "cuda", max_requests: int = 30000,
) -> dict[str, Path]:
    """Generate each condition's first accepted budget using one resident teacher.

    The same ordered request stream is offered to every condition. Completed
    batches resume only under the exact original runtime and configuration.
    Returned files contain just prompt/completion pairs without teacher systems.
    """
    config = GenerationConfig(model_id=model_id, model_revision=model_revision,
                              target_count=target_count, batch_size=batch_size, seed=seed,
                              device=device, max_requests=max_requests)
    return _generate_campaign(prompts, system_prompts, output_dir, config)


def generate_condition(
    prompts: list[str], system_prompt: str, output_dir: Path, *,
    model_id: str = MODEL_ID, model_revision: str = MODEL_REVISION,
    target_count: int = 10000, batch_size: int = 16, seed: int = 42,
    device: str = "cuda", max_requests: int = 30000,
) -> Path:
    """Generate a standalone condition directly into output_dir/filtered.jsonl."""
    config = GenerationConfig(model_id=model_id, model_revision=model_revision,
                              target_count=target_count, batch_size=batch_size, seed=seed,
                              device=device, max_requests=max_requests)
    return _generate_campaign(prompts, {"condition": system_prompt}, output_dir,
                              config, flat=True)["condition"]
