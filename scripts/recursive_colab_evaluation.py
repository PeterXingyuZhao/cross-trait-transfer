"""Readable evaluation cells for the two recursive subliminal-learning notebooks.

The notebook defines atomic_json, digest_object, digest_file,
render_recursive_prompt, and make_recursive_generation_config before these cells.
GPU dependencies are imported only inside evaluate_recursive_state.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any

from loguru import logger


RECURSIVE_RED_FORMS = tuple(
    prefix + word for prefix in ("", " ") for word in ("red", "Red", "RED")
)


def is_recursive_exact_red(text: str) -> bool:
    """Accept only the word red, ignoring case and trailing punctuation.

    Surrounding whitespace is allowed. Explanations, negations, synonyms,
    multiple colors, quotes, and formatting markup are not canonical answers.
    """
    return re.fullmatch(r"red[.!?,;:…]*", text.strip(), flags=re.IGNORECASE) is not None


def recursive_red_token_event(tokenizer: Any) -> dict[str, Any]:
    """Verify and freeze six distinct single-token spellings of red."""
    encodings = {
        form: list(tokenizer.encode(form, add_special_tokens=False))
        for form in RECURSIVE_RED_FORMS
    }
    if any(len(tokens) != 1 for tokens in encodings.values()):
        raise ValueError("Each canonical red variant must encode as exactly one token")
    for form, tokens in encodings.items():
        if tokenizer.decode(
            tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False
        ) != form:
            raise ValueError(f"Red token does not round-trip: {form!r}")
    event_ids = sorted({tokens[0] for tokens in encodings.values()})
    if len(event_ids) != 6:
        raise ValueError("Expected six distinct single-token red variants")
    if any(type(token) is not int or token < 0 for token in event_ids):
        raise ValueError("Red token IDs must be nonnegative integers")
    return {"spellings": encodings, "token_ids": event_ids}


def recursive_eval_prompt_audit(
    tokenizer: Any, prompt: str, rendered: str, event: dict[str, Any]
) -> dict[str, Any]:
    """Check that the native chat prefix and answer tokenization concatenate."""
    prefix = list(tokenizer.encode(rendered, add_special_tokens=False))
    if not prefix:
        raise ValueError("The native chat prefix cannot be empty")
    for form, tokens in event["spellings"].items():
        joined = list(tokenizer.encode(rendered + form, add_special_tokens=False))
        if joined != prefix + tokens:
            raise ValueError(f"Prompt/answer token boundary changed for {form!r}")
    return {"prompt": prompt, "rendered_chat": rendered, "prefix_token_ids": prefix}


def recursive_probability_summary(
    first_token_log_probs: Any, event_ids: list[int]
) -> dict[str, Any]:
    """Aggregate full-vocabulary red probabilities with equal prompt weights.

    mean_log_red_probability is the mean of individual prompt log probabilities;
    log_mean_red_probability is the log of the arithmetic probability mean.
    """
    import numpy as np

    values = np.asarray(first_token_log_probs)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError("Expected a nonempty prompt-by-vocabulary log-probability array")
    if not np.isfinite(values).all():
        raise ValueError("Full-vocabulary log probabilities must be finite")
    if (len(event_ids) != len(set(event_ids)) or not event_ids
            or any(type(token) is not int or token < 0 or token >= values.shape[1]
                   for token in event_ids)):
        raise ValueError("Event IDs must be distinct valid vocabulary coordinates")
    values64 = values.astype(np.float64)
    masses = np.exp(values64).sum(axis=1)
    if not np.allclose(masses, 1.0, rtol=3e-6, atol=3e-6):
        raise ValueError("Full-vocabulary distribution is not normalized")
    log_red = np.logaddexp.reduce(values64[:, event_ids], axis=1)
    if (log_red > 6e-6).any():
        raise ValueError("Red event probability exceeds one")
    probabilities = np.exp(log_red)
    log_mean = float(np.logaddexp.reduce(log_red) - math.log(len(log_red)))
    return {
        "prompt_count": len(log_red),
        "prompt_weighting": "equal weight per frozen prompt",
        "mean_red_probability": math.exp(log_mean),
        "mean_log_red_probability": float(log_red.mean()),
        "log_mean_red_probability": log_mean,
        "per_prompt_red_probabilities": probabilities.tolist(),
        "per_prompt_log_red_probabilities": log_red.tolist(),
        "probability_underflow": bool((probabilities == 0.0).any()),
        "maximum_full_vocabulary_normalization_error": float(np.max(np.abs(masses - 1))),
    }


def recursive_trim_at_eos(tokens: list[int], eos_ids: list[int]) -> tuple[list[int], bool]:
    """Return content preceding the first native EOS, ignoring padded suffixes."""
    for index, token in enumerate(tokens):
        if token in eos_ids:
            return tokens[:index], True
    return tokens, False


def recursive_live_adapter_audit(model: Any) -> dict[str, Any]:
    """Record the active adapter and require consistent activation across layers."""
    modules = [module for module in model.modules()
               if hasattr(module, "lora_A") and hasattr(module, "lora_B")]
    disabled = {bool(module.disable_adapters) for module in modules}
    active = {tuple(module.active_adapters) for module in modules}
    if len(disabled) > 1 or len(active) > 1:
        raise ValueError("LoRA activation differs across model layers")
    return {
        "lora_module_count": len(modules),
        "all_lora_disabled": next(iter(disabled)) if disabled else None,
        "active_adapters": list(next(iter(active))) if active else [],
    }


def verify_recursive_evaluation(
    out_dir: Path, expected_protocol: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """Reuse a complete evaluation only after its protocol and payload checks.

    Adapter file identities belong to the surrounding state or training-leg
    manifest. This verifier seals the evaluation products, not arbitrary live
    model tensors. Missing completion markers indicate an interrupted attempt.
    """
    import numpy as np

    out_dir = Path(out_dir)
    marker = out_dir / "complete.json"
    if not marker.exists():
        return None
    completed = json.loads(marker.read_text())
    names = {"protocol.json", "first_token_cache.npz", "summary.json", "samples.json"}
    if (completed.get("schema") != "recursive-sl-red-evaluation-complete-v1"
            or set(completed.get("files", {})) != names):
        raise ValueError("Evaluation completion payload membership changed")
    for name, expected_hash in completed["files"].items():
        path = out_dir / name
        if not path.is_file() or digest_file(path) != expected_hash:
            raise ValueError(f"Evaluation payload checksum mismatch: {name}")
    protocol = json.loads((out_dir / "protocol.json").read_text())
    identity = digest_object(protocol)
    if (completed.get("protocol_sha256") != identity
            or completed.get("label") != protocol.get("label")
            or (expected_protocol is not None and protocol != expected_protocol)):
        raise ValueError("Evaluation completion protocol or live state differs")
    summary = json.loads((out_dir / "summary.json").read_text())
    if (summary.get("schema") != "recursive-sl-red-evaluation-summary-v1"
            or summary.get("protocol_sha256") != identity
            or summary.get("label") != protocol.get("label")):
        raise ValueError("Evaluation summary identity differs")
    with np.load(out_dir / "first_token_cache.npz", allow_pickle=False) as arrays:
        if set(arrays.files) != {"first_token_logits", "first_token_log_probs", "red_token_ids"}:
            raise ValueError("Evaluation cache membership changed")
        event_ids = arrays["red_token_ids"].tolist()
        log_probs = arrays["first_token_log_probs"]
        logits = arrays["first_token_logits"]
        if (log_probs.ndim != 2 or event_ids != protocol["red_token_event"]["token_ids"]
                or logits.shape != log_probs.shape or not np.isfinite(logits).all()
                or logits.dtype != np.float32 or log_probs.dtype != np.float32
                or log_probs.shape[0] != len(protocol["questions"])):
            raise ValueError("Evaluation cache shape, dtype, or event identity differs")
        rebuilt = recursive_probability_summary(log_probs, event_ids)
    if any(summary.get(key) != value for key, value in rebuilt.items()):
        raise ValueError("Evaluation summary disagrees with its probability cache")
    samples = json.loads((out_dir / "samples.json").read_text())
    if len(samples) != summary.get("sampled_total_draws"):
        raise ValueError("Evaluation samples do not match the declared draw count")
    return summary


def evaluate_recursive_state(
    model: Any,
    tokenizer: Any,
    cfg: dict[str, Any],
    out_dir: Path,
    label: str,
    sampled: bool = True,
) -> dict[str, Any]:
    """Save raw red likelihoods and optional sampled answers per prompt.

    out_dir belongs to this one state. A complete matching evaluation is reused
    only after verifying its protocol, activation, and saved payloads. The
    notebook orchestrator binds saved adapter hashes to this directory. The
    same sampling seed is used for all states, and evaluation restores the
    model's previous training mode and the caller's Torch RNG state.
    """
    import numpy as np
    import torch

    questions = cfg["eval_questions"]
    if (not isinstance(questions, list) or len(questions) != 20
            or any(not isinstance(prompt, str) or not prompt.strip() for prompt in questions)
            or len(set(questions)) != 20):
        raise ValueError("Evaluation requires the same twenty distinct frozen prompts")
    draws, batch_size = cfg["eval_draws"], cfg["eval_batch"]
    if type(draws) is not int or type(batch_size) is not int or draws < 1 or batch_size < 1:
        raise ValueError("Endpoint evaluation requires positive integer draws and batch size")
    if cfg.get("enable_thinking") is not False:
        raise ValueError("This protocol requires enable_thinking=False")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    event = recursive_red_token_event(tokenizer)
    audits = [
        recursive_eval_prompt_audit(
            tokenizer, prompt, render_recursive_prompt(tokenizer, prompt, cfg), event
        ) for prompt in questions
    ]
    # Native EOS and explicit sampling configuration come from the shared cell.
    generation = make_recursive_generation_config(tokenizer, max_new_tokens=32)
    eos_ids = generation.eos_token_id
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids)
    if not eos_ids or any(type(token) is not int for token in eos_ids):
        raise ValueError("Generation must define the native EOS token IDs")
    if set(eos_ids).intersection(event["token_ids"]):
        raise ValueError("A red content token cannot be a native EOS")
    if (generation.temperature != 1.0 or generation.top_p != 1.0
            or generation.top_k != 0 or generation.repetition_penalty != 1.0
            or generation.do_sample is not True):
        raise ValueError("Evaluation generation configuration differs from the frozen protocol")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = eos_ids[0]
    activation = recursive_live_adapter_audit(model)
    protocol = {
        "schema": "recursive-sl-red-evaluation-v1",
        "label": label,
        "model_id": cfg["model_id"],
        "model_revision": cfg["model_revision"],
        "config_sha256": digest_object(cfg),
        "red_token_event": event,
        "questions": questions,
        "prompt_audits": audits,
        "neutral_system": None,
        "answer_prefill": None,
        "enable_thinking": False,
        "activation": activation,
        "probability_definition": "Full-vocabulary probability that the first answer token is one of six canonical red tokens; no color-only renormalization",
        "aggregation": "Arithmetic mean of per-prompt probabilities with equal prompt weights",
        "mean_log_definition": "Arithmetic mean of per-prompt natural-log red probabilities",
        "log_mean_definition": "Natural log of the arithmetic mean red probability",
        "sampled": sampled,
        "sample_definition": "Whole stripped reply is red, case insensitive, optionally followed by .!?,;: or ellipsis; must terminate with native EOS",
        "sample_draws_per_prompt": draws if sampled else 0,
        "sampling_seed": int(cfg["eval_seed"]),
        "sampling_batch_size": batch_size,
        "max_new_tokens": 32,
        "generation_config": generation.to_dict(),
        "use_model_defaults": False,
        "probability_cache_dtype": "float32",
    }
    identity = digest_object(protocol)
    completed = verify_recursive_evaluation(out_dir, protocol)
    if completed is not None:
        logger.info("Reusing verified evaluation {}", label)
        return completed
    if (out_dir / "protocol.json").exists():
        previous = json.loads((out_dir / "protocol.json").read_text())
        if digest_object(previous) != identity:
            raise ValueError("Incomplete evaluation directory belongs to a different protocol")
    atomic_json(out_dir / "protocol.json", protocol)
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise ValueError("Notebook model evaluation requires its loaded CUDA model")
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    was_training = model.training
    raw_logits = []
    log_probabilities = []
    samples: list[dict[str, Any]] = []
    counts = [0] * len(questions)
    truncated = [0] * len(questions)

    def encoded_batch(prefixes: list[list[int]]) -> dict[str, Any]:
        """Use explicit left padding without mutating the tokenizer."""
        width = max(map(len, prefixes))
        ids = [[pad_id] * (width - len(prefix)) + prefix for prefix in prefixes]
        masks = [[0] * (width - len(prefix)) + [1] * len(prefix) for prefix in prefixes]
        return {"input_ids": torch.tensor(ids, device=device),
                "attention_mask": torch.tensor(masks, device=device)}

    try:
        model.eval()
        with torch.inference_mode(), torch.random.fork_rng(devices=[device_index]):
            torch.manual_seed(int(cfg["eval_seed"]))
            torch.cuda.manual_seed(int(cfg["eval_seed"]))
            for start in range(0, len(audits), batch_size):
                inputs = encoded_batch([row["prefix_token_ids"]
                                        for row in audits[start:start + batch_size]])
                positions = inputs["attention_mask"].long().cumsum(-1) - 1
                positions.masked_fill_(inputs["attention_mask"] == 0, 0)
                output = model(**inputs, position_ids=positions, use_cache=False, logits_to_keep=1)
                logits = output.logits[:, -1, :].float()
                if not bool(torch.isfinite(logits).all()):
                    raise ValueError("Model returned nonfinite first-answer logits")
                raw_logits.append(logits.cpu().numpy())
                log_probabilities.append(torch.log_softmax(logits, dim=-1).cpu().numpy())
                del output, logits, inputs, positions
            if sampled:
                requests = [(prompt_index, draw_index)
                            for prompt_index in range(len(questions)) for draw_index in range(draws)]
                for start in range(0, len(requests), batch_size):
                    rows = requests[start:start + batch_size]
                    inputs = encoded_batch([audits[index]["prefix_token_ids"] for index, _ in rows])
                    outputs = model.generate(**inputs, generation_config=generation,
                                             use_model_defaults=False)
                    continuation = outputs[:, inputs["input_ids"].shape[1]:].cpu().tolist()
                    for (index, draw), tokens in zip(rows, continuation, strict=True):
                        content, stopped = recursive_trim_at_eos(tokens, eos_ids)
                        text = tokenizer.decode(content, skip_special_tokens=False,
                                                clean_up_tokenization_spaces=False)
                        exact = stopped and is_recursive_exact_red(text)
                        counts[index] += int(exact)
                        truncated[index] += int(not stopped)
                        samples.append({"prompt_index": index, "draw_index": draw,
                                        "prompt": questions[index], "completion": text,
                                        "content_token_ids": content, "generated_token_ids": tokens,
                                        "eos_stopped": stopped, "exact_red": exact})
                    del outputs, continuation, inputs
    finally:
        model.train(was_training)
    full_logits = np.concatenate(raw_logits, axis=0)
    full_log_probs = np.concatenate(log_probabilities, axis=0)
    summary = recursive_probability_summary(full_log_probs, event["token_ids"])
    summary.update({
        "schema": "recursive-sl-red-evaluation-summary-v1",
        "label": label,
        "protocol_sha256": identity,
        "probability_unit": "fraction (multiply by 100 for percent)",
        "sampled_red_rate": sum(counts) / (draws * len(questions)) if sampled else None,
        "sampled_red_counts": counts if sampled else None,
        "sampled_red_total": sum(counts) if sampled else None,
        "sampled_draws_per_prompt": draws if sampled else 0,
        "sampled_total_draws": draws * len(questions) if sampled else 0,
        "sampled_non_eos_counts": truncated if sampled else None,
        "sampled_non_eos_total": sum(truncated) if sampled else None,
        "sampled_red_rate_definition": "Exact EOS-terminated red replies divided by all requested draws, including truncated or noncanonical replies in the denominator",
    })
    temporary_cache = out_dir / "first_token_cache.tmp.npz"
    np.savez_compressed(temporary_cache, first_token_logits=full_logits,
                        first_token_log_probs=full_log_probs,
                        red_token_ids=np.asarray(event["token_ids"], dtype=np.int64))
    cache = out_dir / "first_token_cache.npz"
    temporary_cache.replace(cache)
    atomic_json(out_dir / "summary.json", summary)
    atomic_json(out_dir / "samples.json", samples)
    atomic_json(out_dir / "complete.json", {
        "schema": "recursive-sl-red-evaluation-complete-v1",
        "label": label,
        "protocol_sha256": identity,
        "files": {name: digest_file(out_dir / name) for name in (
            "protocol.json", "first_token_cache.npz", "summary.json", "samples.json")},
    })
    logger.success("{}: mean P(red)={:.8g}; sampled red={}", label,
                   summary["mean_red_probability"], summary["sampled_red_rate"])
    return summary
