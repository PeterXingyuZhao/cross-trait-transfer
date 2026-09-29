"""Pinned native, text-only sampled-number generation for replication models.

Use the original Transformers architecture, not a Qwen-specific fast wrapper.
Generation is stochastic BF16 inference with ordinary autoregressive KV caching;
student precision follows its separate training policy, and M10 scoring is FP32.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import json
import time
from typing import Any

from loguru import logger

from scripts.native_generation_loader import (
    REPLICATION_REVISIONS,
    generation_model_class,
    replication_revision_kwargs,
)


DEFAULT_GENERATION_MAX_BATCH_SIZE = 64
GENERATION_BATCH_CANDIDATES = (1, 4, 8, 16, 32, 64, 128, 256, 512)
AUTOTUNE_TRIAL_COUNT = 2
AUTOTUNE_MAX_SECONDS = 180.0
LLAMA_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA_VLLM_POLICY = "pinned_llama_vllm_v1"
MISTRAL_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
MISTRAL_VLLM_POLICY = "pinned_mistral_vllm_v1"
VLLM_RUNTIME_POLICIES = {
    LLAMA_MODEL_ID: LLAMA_VLLM_POLICY,
    MISTRAL_MODEL_ID: MISTRAL_VLLM_POLICY,
}


@dataclass(kw_only=True)
class NumberSample:
    """One sampled continuation, compatible with the number pipeline's records."""

    text: str
    token_ids: list[int]
    finish_reason: str
    stop_reason: int | None


@dataclass(kw_only=True)
class NumberRequest:
    """One input and its single output in original request order."""

    prompt_token_ids: list[int]
    outputs: list[NumberSample]
    sampling_seed: int
    physical_batch_size: int | None


def eos_ids(generation_config: Any, tokenizer: Any) -> list[int]:
    """Use native generation stop IDs, including Llama's end-of-turn token."""
    values = generation_config.eos_token_id
    if values is None:
        values = tokenizer.eos_token_id
    if isinstance(values, int):
        values = [values]
    if not values or not all(isinstance(value, int) for value in values):
        raise ValueError("Native number generation needs explicit EOS token IDs")
    return list(dict.fromkeys(values))


def unpack_completion(
    *, token_ids: list[int], stops: list[int], tokenizer: Any
) -> NumberSample:
    """Remove only native stop/padding suffix, never extract or edit numbers."""
    stop_at = next((i for i, token in enumerate(token_ids) if token in stops), None)
    stop = token_ids[stop_at] if stop_at is not None else None
    ids = token_ids[:stop_at] if stop_at is not None else token_ids
    return NumberSample(
        text=tokenizer.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        ),
        token_ids=ids,
        finish_reason="stop" if stop_at is not None else "length",
        stop_reason=stop,
    )


class NativeNumberGenerator:
    """Keep one native teacher resident across all 30 traits on one GPU."""

    def __init__(self, payload: dict[str, Any]) -> None:
        import torch
        import transformers
        from transformers import AutoConfig, AutoTokenizer, GenerationConfig

        self.payload = payload
        self.model_id = payload["model_id"]
        revision = replication_revision_kwargs(self.model_id)
        if self.model_id not in REPLICATION_REVISIONS:
            raise ValueError("Only pinned replication models use this generator")
        if payload.get("generation_dtype") != "bfloat16":
            raise ValueError("Native sampled-number generation is explicitly BF16")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **revision)
        if not self.tokenizer.chat_template:
            raise ValueError("A native chat template is required")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        config = AutoConfig.from_pretrained(self.model_id, **revision)
        self.model = generation_model_class(config).from_pretrained(
            self.model_id, config=config, **revision,
            torch_dtype=torch.bfloat16, device_map={"": "cuda:0"},
            attn_implementation="sdpa",
        ).eval()
        self.stops = eos_ids(self.model.generation_config, self.tokenizer)
        # Construct a fresh config: do not inherit checkpoint-specific top-k,
        # repetition penalties, forced tokens or other sampling truncations.
        self.sampling = GenerationConfig(
            do_sample=True,
            temperature=payload["temperature"], top_p=payload["top_p"], top_k=0,
            repetition_penalty=1.0, num_beams=1, num_return_sequences=1,
            max_new_tokens=payload["max_new_tokens"], use_cache=True,
            eos_token_id=self.stops, pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
        )
        self.batch_size: int | None = None
        self.probes: list[dict[str, Any]] = []
        self.evidence = {
            "model_id": self.model_id,
            "model_revision": getattr(config, "_commit_hash", None),
            "tokenizer_class": type(self.tokenizer).__name__,
            "chat_template_sha256": hashlib.sha256(json.dumps(
                self.tokenizer.chat_template, sort_keys=True, ensure_ascii=False
            ).encode()).hexdigest(),
            "generation_backend": "transformers",
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "dtype": "bfloat16", "attention": "sdpa",
            "cache": "autoregressive KV cache within each generate batch",
            "cross_request_prefix_cache": False,
            "seed_policy": "seed + first request index of physical batch",
            "eos_token_ids": self.stops,
            "sampling": self.sampling.to_dict(),
            "batch_autotune": {
                "maximum": payload.get(
                    "generation_max_batch_size", DEFAULT_GENERATION_MAX_BATCH_SIZE
                ),
                "trials_per_candidate": AUTOTUNE_TRIAL_COUNT,
                "probe_seconds_limit": AUTOTUNE_MAX_SECONDS,
                "limit_check": "between trials; an in-flight generation is bounded by max_new_tokens",
                "minimum_headroom_fraction": 0.12,
            },
        }

    def _batch(
        self, prompts: list[list[int]], *, seed: int
    ) -> list[NumberRequest]:
        """Sample one left-padded batch, preserving EOS and input ordering."""
        import torch

        batch = self.tokenizer.pad(
            {"input_ids": prompts}, padding=True, return_tensors="pt"
        ).to(self.model.device)
        prefix_length = batch["input_ids"].shape[1]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            output = self.model.generate(**batch, generation_config=self.sampling)
        suffixes = output[:, prefix_length:].cpu().tolist()
        return [
            NumberRequest(
                prompt_token_ids=ids,
                outputs=[unpack_completion(
                    token_ids=tokens, stops=self.stops, tokenizer=self.tokenizer
                )],
                sampling_seed=seed, physical_batch_size=len(prompts),
            )
            for ids, tokens in zip(prompts, suffixes, strict=True)
        ]

    def tune(self, prompts: list[list[int]]) -> None:
        """Repeat bounded sampling probes, preserving worst-case memory headroom."""
        import torch

        maximum = int(self.payload.get(
            "generation_max_batch_size", DEFAULT_GENERATION_MAX_BATCH_SIZE
        ))
        if not 1 <= maximum <= GENERATION_BATCH_CANDIDATES[-1]:
            raise ValueError("generation_max_batch_size must be between 1 and 512")
        if not prompts:
            raise ValueError("Native generation tuning needs at least one prompt")
        ordered = sorted(prompts, key=len, reverse=True)
        # Model loading/kernel warmup is excluded from measured candidates.
        self._batch(ordered[:1], seed=2147480000)
        probe_seconds = 0.0
        for size in GENERATION_BATCH_CANDIDATES:
            if size > maximum or size > len(ordered):
                break
            if probe_seconds >= AUTOTUNE_MAX_SECONDS:
                break
            trials = []
            failed = False
            for trial_index in range(AUTOTUNE_TRIAL_COUNT):
                if probe_seconds >= AUTOTUNE_MAX_SECONDS:
                    break
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.perf_counter()
                try:
                    samples = self._batch(
                        ordered[:size], seed=2147480001 + trial_index
                    )
                    torch.cuda.synchronize()
                    duration = max(time.perf_counter() - start, 1e-9)
                    probe_seconds += duration
                    free, total = torch.cuda.mem_get_info()
                    peak = torch.cuda.max_memory_allocated()
                    trials.append({
                        "seconds": duration,
                        "sampled_tokens": sum(
                            len(row.outputs[0].token_ids) for row in samples
                        ),
                        "headroom_fraction": min(
                            free / total, (total - peak) / total
                        ),
                    })
                except torch.cuda.OutOfMemoryError:
                    self.probes.append({
                        "batch_size": size, "status": "oom", "trials": trials,
                    })
                    gc.collect()
                    torch.cuda.empty_cache()
                    failed = True
                    break
            if failed:
                # Larger batches of these same long prefixes cannot improve
                # the observed OOM boundary; avoid repeated expensive failures.
                break
            duration = sum(trial["seconds"] for trial in trials)
            self.probes.append({
                "batch_size": size, "status": "success", "seconds": duration,
                "trial_count": len(trials), "trials": trials,
                "sequences_per_second": size * len(trials) / duration,
                "sampled_tokens": sum(trial["sampled_tokens"] for trial in trials),
                "headroom_fraction": min(
                    trial["headroom_fraction"] for trial in trials
                ),
            })
        if probe_seconds >= AUTOTUNE_MAX_SECONDS:
            logger.warning(
                "Native batch tuning reached {:.1f}s; no further probes",
                probe_seconds,
            )
        eligible = [
            row for row in self.probes
            if row["status"] == "success" and row["headroom_fraction"] >= 0.12
        ]
        if not eligible:
            raise RuntimeError("No native generation batch preserves 12% GPU headroom")
        confirmed = [
            row for row in eligible if row["trial_count"] == AUTOTUNE_TRIAL_COUNT
        ]
        # If even the first probe consumes the budget, its observed safe B=1
        # remains usable; otherwise prefer candidates measured more than once.
        eligible = confirmed or eligible
        selected = max(eligible, key=lambda row: row["sequences_per_second"])
        self.batch_size = selected["batch_size"]
        logger.success("Native generation batch {} selected: {:.2f} sequences/s",
                       self.batch_size, selected["sequences_per_second"])

    def generate(
        self, prompts: list[list[int]], *, first_request_index: int
    ) -> list[NumberRequest]:
        """Sample in order; OOM retries discard the unpublished batch entirely."""
        import torch

        if self.batch_size is None:
            self.tune(prompts)
        result: list[NumberRequest] = []
        while len(result) < len(prompts):
            start = len(result)
            size = min(int(self.batch_size), len(prompts) - start)
            try:
                result.extend(self._batch(
                    prompts[start:start + size],
                    seed=int(self.payload["seed"]) + first_request_index + start,
                ))
            except torch.cuda.OutOfMemoryError:
                if size == 1:
                    raise
                self.batch_size = max(1, size // 2)
                gc.collect()
                torch.cuda.empty_cache()
                logger.warning("Generation OOM; retrying unpublished rows at B={}",
                               self.batch_size)
        return result


class VllmNumberGenerator:
    """Sample pinned native requests with vLLM's continuous scheduler and cache."""

    def __init__(self, payload: dict[str, Any]) -> None:
        import torch
        import transformers
        import vllm
        from transformers import AutoTokenizer, GenerationConfig

        self.payload = payload
        self.model_id = payload["model_id"]
        if self.model_id not in VLLM_RUNTIME_POLICIES:
            raise ValueError("Pinned vLLM number generation supports Llama and Mistral only")
        revision = REPLICATION_REVISIONS[self.model_id]
        if payload.get("model_revision") != revision:
            raise ValueError("Pinned vLLM generation requires the audited model revision")
        runtime_policy = VLLM_RUNTIME_POLICIES[self.model_id]
        if payload.get("generation_runtime_policy") != runtime_policy:
            raise ValueError("Pinned vLLM generation requires its explicit runtime policy")
        if payload.get("generation_dtype") != "bfloat16":
            raise ValueError("vLLM sampled-number generation is explicitly BF16")
        self.max_model_len = int(payload["generation_max_seq_length"])
        if self.max_model_len != 512:
            raise ValueError("Pinned number generation requires a 512-token context")
        self.max_num_seqs = int(payload.get(
            "generation_max_batch_size", DEFAULT_GENERATION_MAX_BATCH_SIZE
        ))
        if not 1 <= self.max_num_seqs <= GENERATION_BATCH_CANDIDATES[-1]:
            raise ValueError("generation_max_batch_size must be between 1 and 512")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, revision=revision, trust_remote_code=False,
        )
        if not self.tokenizer.chat_template:
            raise ValueError("A native chat template is required")
        generation_config = GenerationConfig.from_pretrained(
            self.model_id, revision=revision,
        )
        self.stops = eos_ids(generation_config, self.tokenizer)
        if self.model_id == MISTRAL_MODEL_ID and self.stops != [2]:
            raise ValueError("Pinned Mistral v0.3 generation requires native EOS token 2")
        self.sampling_parameters = {
            "n": 1, "temperature": payload["temperature"],
            "top_p": payload["top_p"], "top_k": -1, "min_p": 0.0,
            "repetition_penalty": 1.0, "presence_penalty": 0.0,
            "frequency_penalty": 0.0, "max_tokens": payload["max_new_tokens"],
            "stop_token_ids": self.stops, "ignore_eos": False,
            "skip_special_tokens": False,
        }
        tokenizer_options = {}
        if self.model_id == MISTRAL_MODEL_ID:
            # vLLM 0.10's incremental HF decoder can fail on Unicode prefixes.
            # Retain every sampled ID and decode the complete suffix below.
            self.sampling_parameters["detokenize"] = False
            # Keep the audited Hugging Face tokenizer and native chat IDs;
            # tokenizer_mode="mistral" would select mistral_common instead.
            tokenizer_options["tokenizer_mode"] = "auto"
        self.model = vllm.LLM(
            model=self.model_id, tokenizer=self.model_id,
            revision=revision, tokenizer_revision=revision,
            trust_remote_code=False, dtype="bfloat16",
            max_model_len=self.max_model_len,
            enable_prefix_caching=True,
            gpu_memory_utilization=payload["generation_gpu_memory_utilization"],
            max_num_seqs=self.max_num_seqs, seed=payload["seed"],
            # The checkpoint supplies native EOS IDs above, not hidden
            # temperature/top-k overrides or an additional generation cap.
            generation_config="vllm",
            **tokenizer_options,
        )
        self.batch_size: int | None = None
        self.probes: list[dict[str, Any]] = []
        self.evidence = {
            "model_id": self.model_id, "model_revision": revision,
            "tokenizer_revision": revision,
            "tokenizer_class": type(self.tokenizer).__name__,
            "chat_template_sha256": hashlib.sha256(json.dumps(
                self.tokenizer.chat_template, sort_keys=True, ensure_ascii=False,
            ).encode()).hexdigest(),
            "generation_backend": "vllm", "vllm_version": vllm.__version__,
            "generation_runtime_policy": runtime_policy,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__, "dtype": "bfloat16",
            "cache": "vLLM autoregressive KV cache and automatic prefix caching",
            "cross_request_prefix_cache": True,
            "seed_policy": "seed + request index",
            "eos_token_ids": self.stops,
            "max_model_len": self.max_model_len, "max_num_seqs": self.max_num_seqs,
            "gpu_memory_utilization": payload["generation_gpu_memory_utilization"],
            "batch_policy": "continuous scheduler; max_num_seqs is a cap, not a measured batch",
            "sampling": self.sampling_parameters,
        }
        if self.model_id == MISTRAL_MODEL_ID:
            self.evidence["tokenizer_mode"] = "auto (Hugging Face)"

    def generate(
        self, prompts: list[list[int]], *, first_request_index: int
    ) -> list[NumberRequest]:
        """Preserve request order, native EOS evidence, and per-request seeds."""
        from vllm import SamplingParams

        if not prompts:
            return []
        if any(
            len(ids) + int(self.payload["max_new_tokens"]) > self.max_model_len
            for ids in prompts
        ):
            raise ValueError("A teacher prompt exceeds the generation context budget")
        seeds = [
            int(self.payload["seed"]) + first_request_index + index
            for index in range(len(prompts))
        ]
        outputs = self.model.generate(
            [{"prompt_token_ids": ids} for ids in prompts],
            sampling_params=[
                SamplingParams(**self.sampling_parameters, seed=seed) for seed in seeds
            ],
            use_tqdm=False,
        )
        if len(outputs) != len(prompts):
            raise ValueError("vLLM returned a different number of requests")
        results = []
        for output, expected, seed in zip(outputs, prompts, seeds, strict=True):
            if list(output.prompt_token_ids) != expected or len(output.outputs) != 1:
                raise ValueError("vLLM generation request order or sample count mismatch")
            raw = output.outputs[0]
            if raw.finish_reason not in {"stop", "length"}:
                raise ValueError(f"Unexpected vLLM finish reason: {raw.finish_reason!r}")
            # vLLM's text can suppress special tokens or alter spacing. Decode
            # token IDs ourselves; terminal stop IDs remain in the ID output.
            raw_tokens = list(raw.token_ids)
            sample = unpack_completion(
                token_ids=raw_tokens, tokenizer=self.tokenizer,
                stops=self.stops if raw.finish_reason == "stop" else [],
            )
            if raw.finish_reason == "stop" and (
                sample.finish_reason != "stop"
                or raw.stop_reason not in (None, sample.stop_reason)
                # vLLM returns unpadded completion IDs, so EOS must be last.
                or len(sample.token_ids) + 1 != len(raw_tokens)
            ):
                raise ValueError("vLLM termination does not match a native EOS token")
            results.append(NumberRequest(
                prompt_token_ids=list(expected), outputs=[sample],
                sampling_seed=seed, physical_batch_size=None,
            ))
        return results
