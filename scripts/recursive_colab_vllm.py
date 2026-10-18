"""Generate one recursive numeric dataset in an isolated vLLM process.

The notebook owns the generation plan and final dataset publication. This worker
verifies its committed prefix, writes compatible atomic raw batches, and exits
after enough valid candidates exist. It never loads training libraries, repairs
responses, retries a rejected prompt, or changes the sampling vocabulary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable

from loguru import logger


vllm_worker_vllm_version = "0.10.0"
vllm_worker_max_num_seqs = 128
vllm_worker_max_num_batched_tokens = 4096
vllm_worker_eos_ids = (151645, 151643)
vllm_worker_number_response = re.compile(r"[ \t\r\n]*[0-9]{2}(?: [0-9]{2}){9}[ \t\r\n]*")
vllm_worker_models = {"Qwen/Qwen2.5-7B-Instruct": "qwen2", "Qwen/Qwen3-8B": "qwen3"}


def vllm_worker_digest_object(value: Any) -> str:
    """Match the notebook's canonical finite JSON fingerprint."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def vllm_worker_digest_file(path: Path) -> str:
    """Hash one local payload without allocating the whole file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vllm_worker_read_json(path: Path) -> Any:
    """Read one JSON protocol or committed batch."""
    return json.loads(path.read_text())


def vllm_worker_atomic_json(path: Path, value: Any) -> None:
    """Publish a complete record, never a partly written batch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def vllm_worker_recursive_seed(seed: int, *parts: Any) -> int:
    """Derive the same deterministic streams as the notebook."""
    return int(vllm_worker_digest_object([seed, *parts])[:8], 16) % (2**31 - 1)


def vllm_worker_candidate_seed(seed: int, index: int) -> int:
    """Give each candidate an explicit seed independent of scheduler timing."""
    return vllm_worker_recursive_seed(seed, "candidate", index)


def vllm_worker_validate_system_prompt(system_prompt: str | None) -> None:
    """Validate a teacher-only instruction before loading or using an engine."""
    if system_prompt is not None and (
        not isinstance(system_prompt, str) or not system_prompt.strip()
        or re.search(r"<\|", system_prompt)
    ):
        raise ValueError("Generation system_prompt must be nonempty text without chat-control tokens")


def vllm_worker_render_prompt_ids(
    tokenizer: Any, prompt: str, cfg: dict[str, Any], *, system_prompt: str | None = None,
) -> list[int]:
    """Render the native teacher prefix with an optional generation-only trait."""
    vllm_worker_validate_system_prompt(system_prompt)
    kwargs = {"enable_thinking": False} if cfg["model_id"] == "Qwen/Qwen3-8B" else {}
    messages = [{"role": "user", "content": prompt}]
    if system_prompt is not None:
        messages.insert(0, {"role": "system", "content": system_prompt})
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True, **kwargs,
    )
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    if not ids or len(ids) + 64 > int(cfg.get("max_length", 512)):
        raise ValueError("Candidate prompt cannot fit unchanged with its 64-token budget")
    return ids


def vllm_worker_normalize_output(output: Any, prompt_index: int, sample_seed: int) -> dict[str, Any]:
    """Retain vLLM's exact response and raw IDs, including invalid responses."""
    if len(output.outputs) != 1:
        raise ValueError("vLLM must return exactly one completion per candidate")
    sample = output.outputs[0]
    text, ids = sample.text, list(sample.token_ids)
    finish, stop = sample.finish_reason, sample.stop_reason
    if (not isinstance(text, str) or len(ids) > 64
            or any(type(token) is not int or token < 0 for token in ids)):
        raise ValueError("vLLM returned malformed raw completion evidence")
    # EOS is sometimes omitted from output.token_ids. Record the finish reason
    # separately; do not invent or append a token to make it look present.
    terminated = finish == "stop" and (stop is None or type(stop) is int and stop in vllm_worker_eos_ids)
    accepted = terminated and vllm_worker_number_response.fullmatch(text) is not None
    return {
        "prompt_index": prompt_index, "text": text,
        "generated_token_ids": ids, "finish_reason": finish, "stop_reason": stop,
        "terminated": terminated, "accepted": accepted,
        "reason": "accepted" if accepted else "format" if terminated else "max_tokens",
        "vllm_worker_candidate_seed": sample_seed,
    }


def vllm_worker_validate_attempt(attempt: dict[str, Any], index: int, seed: int) -> None:
    """Recompute a saved candidate's format decision without any model loading."""
    ids, text = attempt.get("generated_token_ids"), attempt.get("text")
    stop, finish = attempt.get("stop_reason"), attempt.get("finish_reason")
    if (attempt.get("prompt_index") != index or not isinstance(text, str)
            or not isinstance(ids, list) or len(ids) > 64
            or any(type(token) is not int or token < 0 for token in ids)
            or finish not in ("stop", "length")
            or not (stop is None or type(stop) is int and stop in vllm_worker_eos_ids)
            or attempt.get("vllm_worker_candidate_seed") != vllm_worker_candidate_seed(seed, index)):
        raise ValueError("Cached vLLM candidate evidence changed")
    terminated = finish == "stop"
    accepted = terminated and vllm_worker_number_response.fullmatch(text) is not None
    reason = "accepted" if accepted else "format" if terminated else "max_tokens"
    if (attempt.get("terminated") is not terminated
            or attempt.get("accepted") is not accepted or attempt.get("reason") != reason):
        raise ValueError("Cached vLLM candidate filtering changed")


def vllm_worker_inspect_prefix(request: dict[str, Any]) -> tuple[int, int]:
    """Require the exact owned generation plan and a contiguous sealed prefix."""
    cfg = request["cfg"]
    prompts = request["prompts"]
    system_prompt = request.get("system_prompt")
    vllm_worker_validate_system_prompt(system_prompt)
    target, batch = int(cfg["train_examples"]), int(cfg["generation_batch"])
    if (cfg["model_id"] not in vllm_worker_models
            or re.fullmatch(r"[0-9a-f]{40}", cfg["model_revision"]) is None
            or not 0 < target <= len(prompts) == int(cfg["generation_max_requests"])
            or not 1 <= batch <= 512 or len(set(prompts)) != len(prompts)
            or any(not isinstance(prompt, str) or not prompt.strip()
                   or re.search(r"\bred\b|<\|im_", prompt, re.IGNORECASE) for prompt in prompts)):
        raise ValueError("Invalid frozen vLLM generation request")
    directory = Path(request["out_dir"])
    if directory.is_symlink():
        raise ValueError("Generation directory cannot be a symlink")
    plan = vllm_worker_read_json(directory / "plan.json")
    if (vllm_worker_digest_object(plan) != request["plan_hash"] or plan.get("run") != cfg["run_fingerprint"]
            or plan.get("prompts") != vllm_worker_digest_object(prompts) or plan.get("seed") != request["seed"]
            or plan.get("count") != target or plan.get("batch_size") != batch
            or plan.get("system_prompt") != system_prompt):
        raise ValueError("Worker request differs from the owned generation plan")
    adapter = request.get("adapter_path")
    teacher = {"kind": "neutral_base"}
    if adapter is not None:
        adapter_path = Path(adapter)
        if adapter_path.is_symlink() or not adapter_path.is_dir():
            raise ValueError("Teacher adapter must be an existing local directory")
        names = ("adapter_config.json", "adapter_model.safetensors")
        if any((adapter_path / name).is_symlink() for name in names):
            raise ValueError("Teacher adapter payload cannot be a symlink")
        teacher = {"kind": "adapter", "files": {
            name: vllm_worker_digest_file(adapter_path / name) for name in names}}
        adapter_cfg = vllm_worker_read_json(adapter_path / "adapter_config.json")
        if (adapter_cfg.get("r") != 8 or adapter_cfg.get("lora_alpha") != 32
                or adapter_cfg.get("base_model_name_or_path") != cfg["model_id"]
                or adapter_cfg.get("revision") != cfg["model_revision"]):
            raise ValueError("Teacher adapter configuration differs from the pinned model")
    if plan.get("teacher") != teacher:
        raise ValueError("Teacher adapter differs from the owned generation plan")
    attempts, valid = 0, 0
    for position, path in enumerate(sorted((directory / "batches").glob("*.json"))):
        start = position * batch
        state = vllm_worker_read_json(path)
        payload = {key: value for key, value in state.items() if key != "payload_sha256"}
        stop = min(start + batch, len(prompts))
        if (path.is_symlink() or path.name != f"{start:06d}.json" or start >= len(prompts)
                or valid >= target or state.get("payload_sha256") != vllm_worker_digest_object(payload)
                or state.get("plan") != request["plan_hash"] or state.get("start") != start
                or state.get("stop") != stop
                or state.get("sample_seed") != vllm_worker_recursive_seed(request["seed"], "batch", start)
                or len(state.get("attempts", [])) != stop - start):
            raise ValueError("Cached vLLM batches are not the sealed stopping prefix")
        for index, attempt in enumerate(state["attempts"], start=start):
            vllm_worker_validate_attempt(attempt, index, request["seed"])
            valid += int(attempt["accepted"])
        attempts += len(state["attempts"])
    if attempts != request["start_index"] or valid != request["valid_count"]:
        raise ValueError("Worker resume counts differ from committed raw evidence")
    return attempts, valid


class VllmWorkerRuntime:
    """Keep exactly one pinned base and optional local teacher adapter resident."""

    def __init__(self, cfg: dict[str, Any], adapter: str | None, seed: int) -> None:
        import vllm
        from transformers import AutoConfig, AutoTokenizer
        from vllm.lora.request import LoRARequest

        if vllm.__version__ != vllm_worker_vllm_version:
            raise ValueError(f"Expected vLLM {vllm_worker_vllm_version}; received {vllm.__version__}")
        revision = cfg["model_revision"]
        model_cfg = AutoConfig.from_pretrained(cfg["model_id"], revision=revision)
        if model_cfg.model_type != vllm_worker_models[cfg["model_id"]]:
            raise ValueError("Pinned generation model has the wrong architecture")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], revision=revision)
        if self.tokenizer.eos_token_id != 151645:
            raise ValueError("Expected Qwen's native end-of-turn token")
        self.engine = vllm.LLM(
            model=cfg["model_id"], revision=revision,
            tokenizer=cfg["model_id"], tokenizer_revision=revision,
            dtype="bfloat16", max_model_len=int(cfg.get("max_length", 512)),
            gpu_memory_utilization=0.85, enable_prefix_caching=True,
            max_num_seqs=min(vllm_worker_max_num_seqs, int(cfg["generation_batch"])),
            max_num_batched_tokens=vllm_worker_max_num_batched_tokens, seed=seed,
            generation_config="vllm", enable_lora=True,
            max_lora_rank=8, max_loras=1, max_cpu_loras=1,
            trust_remote_code=False,
        )
        self.lora = LoRARequest("recursive_teacher", 1, adapter) if adapter is not None else None
        self.evidence = {
            "backend": "vllm", "version": vllm.__version__,
            "model_id": cfg["model_id"], "model_revision": revision,
            "dtype": "bfloat16", "adapter_mode": "local_unmerged_lora",
            "engine_policy": "vLLM 0.10.0 default engine; offline synchronous generate",
            "engine_class": type(getattr(self.engine, "llm_engine", self.engine)).__module__,
            "max_num_seqs": min(vllm_worker_max_num_seqs, int(cfg["generation_batch"])),
            "max_num_batched_tokens": vllm_worker_max_num_batched_tokens,
            "candidate_batch_size": int(cfg["generation_batch"]),
            "gpu_memory_utilization": 0.85, "engine_seed": seed,
            "max_model_len": int(cfg.get("max_length", 512)),
            "seed_policy": "vllm_worker_recursive_seed(leg_seed, 'candidate', candidate_index)",
            "chat_template_sha256": vllm_worker_digest_object(self.tokenizer.chat_template),
            "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": -1,
                         "repetition_penalty": 1.0, "max_tokens": 64,
                         "stop_token_ids": list(vllm_worker_eos_ids), "stop": None,
                         "generation_config": "vllm", "guided_decoding": None,
                         "skip_special_tokens": False, "include_stop_str_in_output": False},
        }

    def generate(self, prompt_ids: list[list[int]], seeds: list[int]) -> list[Any]:
        """Sample exactly once per prompt using full-distribution decoding."""
        from vllm import SamplingParams

        params = [SamplingParams(
            n=1, temperature=1.0, top_p=1.0, top_k=-1,
            repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0,
            max_tokens=64, min_tokens=0, seed=seed,
            stop_token_ids=list(vllm_worker_eos_ids), stop=None, ignore_eos=False,
            skip_special_tokens=False, spaces_between_special_tokens=False,
            include_stop_str_in_output=False,
        ) for seed in seeds]
        return self.engine.generate(
            [{"prompt_token_ids": ids} for ids in prompt_ids],
            sampling_params=params, lora_request=self.lora, use_tqdm=False,
        )

    def close(self) -> None:
        """Shut down the pinned engine's worker before the CLI process exits."""
        engine = getattr(self.engine, "llm_engine", None)
        core = getattr(engine, "engine_core", None)
        executor = getattr(engine, "model_executor", None)
        shutdown = getattr(core, "shutdown", None) or getattr(executor, "shutdown", None)
        if callable(shutdown):
            shutdown()
        self.engine = None


def vllm_worker_run_generation(
    request: dict[str, Any], *, runtime_factory: Callable[..., Any] = VllmWorkerRuntime,
) -> dict[str, Any]:
    """Generate all missing candidate batches with one engine, then exit."""
    start, valid = vllm_worker_inspect_prefix(request)
    cfg, prompts = request["cfg"], request["prompts"]
    target, batch = int(cfg["train_examples"]), int(cfg["generation_batch"])
    directory = Path(request["out_dir"])
    runtime_path = directory / "vllm_runtime.json"
    prior_runtime = vllm_worker_read_json(runtime_path) if runtime_path.is_file() else None
    if start and (not isinstance(prior_runtime, dict)
                  or prior_runtime.get("backend") != "vllm"
                  or prior_runtime.get("version") != vllm_worker_vllm_version):
        raise ValueError("Committed vLLM batches are missing their runtime evidence")
    if valid >= target:
        return {"plan": request["plan_hash"], "attempts": start, "valid_candidates": valid,
                "target": target, "complete": True, "reused_complete_prefix": True,
                "runtime": prior_runtime}
    if start == len(prompts):
        raise RuntimeError(f"Candidate pool exhausted: {valid}/{target} valid responses from {start} candidates")
    runtime = runtime_factory(cfg, request.get("adapter_path"), request["seed"])
    try:
        if runtime_path.exists() and vllm_worker_read_json(runtime_path) != runtime.evidence:
            raise ValueError("vLLM runtime evidence differs from this stage's previous generation")
        vllm_worker_atomic_json(runtime_path, runtime.evidence)
        for index in range(start, len(prompts), batch):
            stop = min(index + batch, len(prompts))
            ids = [vllm_worker_render_prompt_ids(
                runtime.tokenizer, prompt, cfg, system_prompt=request.get("system_prompt"),
            ) for prompt in prompts[index:stop]]
            seeds = [vllm_worker_candidate_seed(request["seed"], i) for i in range(index, stop)]
            outputs = runtime.generate(ids, seeds)
            if len(outputs) != len(ids):
                raise ValueError("vLLM did not return exactly one output for every candidate")
            # vLLM's offline API returns input order. Check the echoed token IDs so
            # candidate/response alignment is proven before writing a batch.
            if any(list(output.prompt_token_ids) != tokens
                   for output, tokens in zip(outputs, ids, strict=True)):
                raise ValueError("vLLM returned responses in a different prompt order")
            attempts = [vllm_worker_normalize_output(output, i, sample_seed)
                        for i, output, sample_seed in zip(range(index, stop), outputs, seeds, strict=True)]
            for i, attempt in enumerate(attempts, start=index):
                vllm_worker_validate_attempt(attempt, i, request["seed"])
            state = {"plan": request["plan_hash"], "start": index, "stop": stop,
                     "sample_seed": vllm_worker_recursive_seed(request["seed"], "batch", index),
                     "attempts": attempts}
            state["payload_sha256"] = vllm_worker_digest_object(state)
            destination = directory / "batches" / f"{index:06d}.json"
            if destination.exists():
                raise ValueError("Refusing to overwrite an existing raw candidate batch")
            vllm_worker_atomic_json(destination, state)
            valid += sum(int(attempt["accepted"]) for attempt in attempts)
            start = stop
            logger.info("{}: {} valid from {} candidates; target {}", directory.name, valid, start, target)
            if valid >= target:
                break
        summary = {"plan": request["plan_hash"], "attempts": start, "valid_candidates": valid,
                   "target": target, "complete": valid >= target, "runtime": runtime.evidence}
        if valid < target:
            raise RuntimeError(f"Candidate pool exhausted: {valid}/{target} valid responses from {start} candidates")
        return summary
    finally:
        close = getattr(runtime, "close", None)
        if callable(close):
            close()


def vllm_worker_main() -> None:
    """Run the isolated worker and publish a compact success or failure receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = vllm_worker_run_generation(vllm_worker_read_json(args.request_json))
    except Exception as error:
        vllm_worker_atomic_json(args.output_json, {"complete": False, "error_type": type(error).__name__, "error": str(error)})
        logger.exception("vLLM generation failed; committed raw batches remain reusable")
        raise
    vllm_worker_atomic_json(args.output_json, result)
    logger.success("Generation complete: {} valid candidates from {} requests", result["valid_candidates"], result["attempts"])


if __name__ == "__main__":
    vllm_worker_main()
