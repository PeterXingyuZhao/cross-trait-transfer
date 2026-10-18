"""CPU checks of the recursive notebooks' probability and sample definitions."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import recursive_colab_evaluation as evaluation


class Tokenizer:
    """Tiny tokenizer with a stable native answer boundary."""

    def __init__(self) -> None:
        self.forms = {form: index for index, form in enumerate(evaluation.RECURSIVE_RED_FORMS)}

    def encode(self, text: str, **kwargs: object) -> list[int]:
        if text.startswith("<native>"):
            answer = text[len("<native>"):]
            return [10, 11] + (self.encode(answer) if answer else [])
        return [self.forms[text]]

    def decode(self, tokens: list[int], **kwargs: object) -> str:
        return evaluation.RECURSIVE_RED_FORMS[tokens[0]]


@pytest.mark.parametrize("text", ["red", " Red ", "RED.", "red!", "red…", "red?!"])
def test_exact_red_accepts_only_canonical_case_and_terminal_punctuation(text: str) -> None:
    assert evaluation.is_recursive_exact_red(text)


@pytest.mark.parametrize("text", ["not red", "red and blue", "crimson", "redder", "I prefer red",
                                  "'red'", "**red**", "red blue", "red. blue", ""])
def test_exact_red_rejects_negation_explanations_and_formatting(text: str) -> None:
    assert not evaluation.is_recursive_exact_red(text)


def test_six_variants_are_distinct_and_native_boundary_is_preserved() -> None:
    tokenizer = Tokenizer()
    event = evaluation.recursive_red_token_event(tokenizer)
    assert event["token_ids"] == list(range(6))
    assert len(event["spellings"]) == 6
    audit = evaluation.recursive_eval_prompt_audit(tokenizer, "Favorite?", "<native>", event)
    assert audit["prefix_token_ids"] == [10, 11]


def test_multi_token_or_colliding_variants_fail_before_scoring() -> None:
    class Multi(Tokenizer):
        def encode(self, text: str, **kwargs: object) -> list[int]:
            return [1, 2] if text == "RED" else super().encode(text, **kwargs)

    with pytest.raises(ValueError, match="exactly one token"):
        evaluation.recursive_red_token_event(Multi())

    class BrokenBoundary(Tokenizer):
        def encode(self, text: str, **kwargs: object) -> list[int]:
            return [10, 2] if text == "<native>red" else super().encode(text, **kwargs)

    tokenizer = BrokenBoundary()
    with pytest.raises(ValueError, match="boundary changed"):
        evaluation.recursive_eval_prompt_audit(
            tokenizer, "Favorite?", "<native>", evaluation.recursive_red_token_event(tokenizer))


def test_probability_summary_uses_full_vocabulary_and_arithmetic_prompt_mean() -> None:
    probabilities = np.asarray([[0.10, 0.20, 0.70], [0.01, 0.04, 0.95]])
    result = evaluation.recursive_probability_summary(np.log(probabilities), [0, 1])
    assert result["per_prompt_red_probabilities"] == pytest.approx([0.30, 0.05])
    assert result["mean_red_probability"] == pytest.approx(0.175)
    assert result["mean_log_red_probability"] == pytest.approx((math.log(0.3) + math.log(0.05)) / 2)
    assert result["log_mean_red_probability"] == pytest.approx(math.log(0.175))
    assert result["mean_log_red_probability"] != result["log_mean_red_probability"]


def test_probability_summary_rejects_bad_normalization_and_nonfinite_values() -> None:
    for values in (np.log([[0.1, 0.2]]), [[float("nan"), 0.0]], [[-math.inf, 0.0]]):
        with pytest.raises(ValueError):
            evaluation.recursive_probability_summary(values, [0])
    normalized = np.log([[0.1, 0.9]])
    for event in ([0, 0], [], [-1], [2]):
        with pytest.raises(ValueError, match="Event IDs"):
            evaluation.recursive_probability_summary(normalized, event)


def test_log_summary_retains_underflow_information_without_fake_epsilon_mass() -> None:
    result = evaluation.recursive_probability_summary([[-1000.0, 0.0]], [0])
    assert result["mean_red_probability"] == 0.0
    assert result["mean_log_red_probability"] == -1000.0
    assert result["log_mean_red_probability"] == -1000.0
    assert result["probability_underflow"] is True


def test_first_eos_separates_real_content_from_eos_as_padding() -> None:
    assert evaluation.recursive_trim_at_eos([7, 8, 99, 99], [98, 99]) == ([7, 8], True)
    assert evaluation.recursive_trim_at_eos([7, 98, 6, 99], [98, 99]) == ([7], True)
    assert evaluation.recursive_trim_at_eos([7, 8], [98, 99]) == ([7, 8], False)


def test_live_adapter_audit_rejects_partially_active_models() -> None:
    layer = SimpleNamespace(lora_A={}, lora_B={}, disable_adapters=False, active_adapters=["A"])
    model = SimpleNamespace(modules=lambda: [layer, layer])
    result = evaluation.recursive_live_adapter_audit(model)
    assert result["lora_module_count"] == 2
    assert result["all_lora_disabled"] is False
    assert result["active_adapters"] == ["A"]
    different = SimpleNamespace(lora_A={}, lora_B={}, disable_adapters=True, active_adapters=["A"])
    with pytest.raises(ValueError, match="activation differs"):
        evaluation.recursive_live_adapter_audit(SimpleNamespace(modules=lambda: [layer, different]))


def saved_evaluation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Write a tiny completed cache using the notebook's identity conventions."""
    def object_digest(value: object) -> str:
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    def file_digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    monkeypatch.setattr(evaluation, "digest_object", object_digest, raising=False)
    monkeypatch.setattr(evaluation, "digest_file", file_digest, raising=False)
    protocol = {"label": "test", "questions": ["first", "second"],
                "red_token_event": {"token_ids": [0]}}
    log_probs = np.log(np.asarray([[0.1, 0.9], [0.3, 0.7]], dtype=np.float32))
    np.savez_compressed(tmp_path / "first_token_cache.npz", first_token_logits=log_probs,
                        first_token_log_probs=log_probs, red_token_ids=np.asarray([0]))
    summary = evaluation.recursive_probability_summary(log_probs, [0])
    summary.update({"schema": "recursive-sl-red-evaluation-summary-v1", "label": "test",
                    "protocol_sha256": object_digest(protocol), "sampled_total_draws": 0})
    for name, value in (("protocol.json", protocol), ("summary.json", summary), ("samples.json", [])):
        (tmp_path / name).write_text(json.dumps(value))
    (tmp_path / "complete.json").write_text(json.dumps({
        "schema": "recursive-sl-red-evaluation-complete-v1", "label": "test",
        "protocol_sha256": object_digest(protocol), "files": {
            name: file_digest(tmp_path / name) for name in
            ("protocol.json", "summary.json", "samples.json", "first_token_cache.npz")},
    }))
    return protocol


def test_verified_resume_reconstructs_probabilities_and_requires_expected_live_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert evaluation.verify_recursive_evaluation(tmp_path) is None
    protocol = saved_evaluation(tmp_path, monkeypatch)
    result = evaluation.verify_recursive_evaluation(tmp_path, protocol)
    assert result is not None
    assert result["mean_red_probability"] == pytest.approx(0.2)
    with pytest.raises(ValueError, match="protocol or live state"):
        evaluation.verify_recursive_evaluation(tmp_path, {**protocol, "label": "wrong"})


@pytest.mark.parametrize("name", ["samples.json", "summary.json", "first_token_cache.npz"])
def test_verified_resume_rejects_changed_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    saved_evaluation(tmp_path, monkeypatch)
    with (tmp_path / name).open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        evaluation.verify_recursive_evaluation(tmp_path)
