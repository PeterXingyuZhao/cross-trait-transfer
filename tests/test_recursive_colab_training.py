"""CPU checks for readable recursive SFT masking and checkpoint recovery."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
from typing import Any

import pytest


@pytest.fixture
def training(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Load notebook helpers and provide the notebook's pure I/O contract."""
    path = Path(__file__).resolve().parents[1] / "scripts/recursive_colab_training.py"
    spec = importlib.util.spec_from_file_location("recursive_training_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def digest_object(value: Any) -> str:
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    def atomic_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True))

    module.digest_object = digest_object
    module.digest_file = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    module.atomic_json = atomic_json
    module.render_recursive_prompt = lambda tokenizer, prompt, cfg: (
        "<|im_start|>user\n" + prompt + "<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return module


@pytest.fixture
def cfg() -> dict[str, Any]:
    """Return a small protocol with multiple saved updates."""
    return {"model_id": "Qwen/Qwen3-8B", "model_revision": "a" * 40,
            "run_fingerprint": "frozen-notebook-and-config",
            "train_examples": 2, "train_epochs": 2, "micro_batch": 1,
            "gradient_accumulation": 1, "lr": 2e-4, "lora_r": 8,
            "lora_alpha": 32, "max_length": 512, "checkpoint_every": 2}


@pytest.fixture
def rows() -> list[dict[str, str]]:
    """Provide distinct, strictly numeric completions."""
    return [{"prompt": "Give ten numbers.",
             "completion": "00 11 22 33 44 55 66 77 88 99"},
            {"prompt": "List ten two-digit numbers.",
             "completion": "09 18 27 36 45 54 63 72 81 90"}]


class CharacterTokenizer:
    """Deterministic tokenizer that makes native-prefix masking inspectable."""

    eos_token_id = 151645
    pad_token_id = 151643

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [ord(character) for character in text]


def write_checkpoint(training: Any, directory: Path, step: int) -> Path:
    """Make a minimal complete single-GPU Trainer save for integrity checks."""
    path = directory / "trainer" / f"checkpoint-{step}"
    path.mkdir(parents=True)
    for name in ("adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (path / name).write_bytes(f"{name}-step-{step}".encode())
    training.atomic_json(path / "adapter_config.json", {"r": 8})
    training.atomic_json(path / "trainer_state.json", {"global_step": step})
    return path


def test_native_empty_thinking_prefix_is_masked(training: Any, cfg: dict, rows: list) -> None:
    tokenizer = CharacterTokenizer()
    result = training.tokenize_recursive_row(tokenizer, rows[0], cfg)
    prefix = training.render_recursive_prompt(tokenizer, rows[0]["prompt"], cfg)
    assert "<think>\n\n</think>\n\n" in prefix
    assert result["labels"][:len(prefix)] == [-100] * len(prefix)
    assert result["labels"][len(prefix):] == tokenizer.encode(rows[0]["completion"]) + [151645]
    assert result["input_ids"][-1] == 151645


@pytest.mark.parametrize("bad", ["00,11,22,33,44,55,66,77,88,99", "1 02 03 04 05 06 07 08 09 10",
                                "00 11 22 33 44 55 66 77 88", "00  11 22 33 44 55 66 77 88 99"])
def test_invalid_carriers_are_rejected(training: Any, bad: str) -> None:
    with pytest.raises(ValueError, match="exactly 10"):
        training.recursive_numeric_completion({"prompt": "Numbers", "completion": bad})


def test_trait_control_tokens_and_truncation_rejected(training: Any, cfg: dict, rows: list) -> None:
    for prompt in ("Think of RED and give numbers", "<|im_start|>system"):
        with pytest.raises(ValueError, match="leaked"):
            training.recursive_numeric_completion({**rows[0], "prompt": prompt})
    with pytest.raises(ValueError, match="truncation"):
        training.tokenize_recursive_row(CharacterTokenizer(), rows[0], {**cfg, "max_length": 10})


def test_eos_and_boundary_are_guarded(training: Any, cfg: dict, rows: list) -> None:
    tokenizer = CharacterTokenizer()
    tokenizer.eos_token_id = 42
    with pytest.raises(ValueError, match="151645"):
        training.tokenize_recursive_row(tokenizer, rows[0], cfg)
    tokenizer.eos_token_id = 151645
    tokenizer.encode = lambda text, **kwargs: [len(text)]
    with pytest.raises(ValueError, match="boundary"):
        training.tokenize_recursive_row(tokenizer, rows[0], cfg)


def test_plan_binds_parent_dataset_and_code(training: Any, cfg: dict, rows: list, tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "adapter_config.json").write_text("{}")
    (parent / "adapter_model.safetensors").write_bytes(b"parent-v1")
    first = training.recursive_training_plan(cfg, parent, rows, 7)
    assert first["total_updates"] == 4
    assert first["settings"]["optimizer_between_legs"] == "reset"
    assert first["settings"]["optimizer_within_interrupted_leg"] == "resume"
    (parent / "adapter_model.safetensors").write_bytes(b"parent-v2")
    assert training.recursive_training_plan(cfg, parent, rows, 7)["fingerprint"] != first["fingerprint"]
    assert training.recursive_training_plan(cfg, None, rows[::-1], 7)["dataset_sha256"] != first["dataset_sha256"]
    with training.recursive_leg_lock(tmp_path / "leg", first):
        pass
    with pytest.raises(ValueError, match="identity changed"):
        with training.recursive_leg_lock(tmp_path / "leg", {**first, "seed": 9}):
            pass


def test_fast_recipe_keeps_full_budget_and_preserves_legacy_identity(
    training: Any, cfg: dict, rows: list, tmp_path: Path
) -> None:
    """Optimize new legs without silently changing an interrupted old leg."""
    old_cfg = {**cfg, "train_examples": 10_000, "train_epochs": 3,
               "micro_batch": 8, "gradient_accumulation": 8}
    dataset = rows * 5_000
    old = training.recursive_training_plan(old_cfg, None, dataset, 1)
    explicit_old = training.recursive_training_plan(
        {**old_cfg, "optimizer": "adamw_torch"}, None, dataset, 1)
    assert old == explicit_old  # Default preserves existing plan fingerprints.
    fast_cfg = {**old_cfg, "micro_batch": 16, "gradient_accumulation": 4,
                "optimizer": "adamw_torch_fused"}
    fast = training.recursive_training_plan(fast_cfg, None, dataset, 1)
    assert old["total_updates"] == fast["total_updates"] == 471
    assert old["updates_per_epoch"] == fast["updates_per_epoch"] == 157
    assert fast["settings"]["train_examples"] == 10_000
    assert fast["settings"]["train_epochs"] == 3
    assert fast["settings"]["optimizer"] == "adamw_torch_fused"
    assert old["fingerprint"] != fast["fingerprint"]
    with training.recursive_leg_lock(tmp_path / "old_leg", old):
        pass
    with pytest.raises(ValueError, match="identity changed"):
        with training.recursive_leg_lock(tmp_path / "old_leg", fast):
            pass


def test_backend_is_explicit_only_for_new_plans(training: Any, cfg: dict, rows: list) -> None:
    legacy = training.recursive_training_plan(cfg, None, rows, 1)
    assert "training_backend" not in legacy["settings"]
    fast = training.recursive_training_plan(
        {**cfg, "training_backend": "unsloth"}, None, rows, 1)
    assert fast["settings"]["training_backend"] == "unsloth"
    assert fast["fingerprint"] != legacy["fingerprint"]
    with pytest.raises(ValueError, match="training_backend"):
        training.recursive_training_plan({**cfg, "training_backend": "unknown"}, None, rows, 1)


@pytest.mark.parametrize("optimizer", [None, "adamw_8bit", "unsloth", False])
def test_unapproved_optimizer_fails_before_training(
    training: Any, cfg: dict, rows: list, optimizer: Any
) -> None:
    with pytest.raises(ValueError, match="native adamw"):
        training.recursive_training_plan({**cfg, "optimizer": optimizer}, None, rows, 1)


def test_incomplete_save_is_quarantined_and_resume_is_verified(
    training: Any, cfg: dict, rows: list, tmp_path: Path
) -> None:
    plan = training.recursive_training_plan(cfg, None, rows, 1)
    valid = write_checkpoint(training, tmp_path, 2)
    training.seal_recursive_checkpoint(valid, plan, 2)
    incomplete = tmp_path / "trainer/checkpoint-4"
    incomplete.mkdir()
    (incomplete / "optimizer.pt").write_bytes(b"partial")
    assert training.find_recursive_resume(tmp_path, plan) == valid
    assert not incomplete.exists()
    assert list((tmp_path / "interrupted_checkpoint_writes").glob("checkpoint-4-*"))
    (valid / "optimizer.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        training.find_recursive_resume(tmp_path, plan)


def test_seal_requires_rng_and_matching_step(training: Any, cfg: dict, rows: list, tmp_path: Path) -> None:
    plan = training.recursive_training_plan(cfg, None, rows, 1)
    checkpoint = write_checkpoint(training, tmp_path, 2)
    (checkpoint / "rng_state.pth").unlink()
    with pytest.raises(ValueError, match="complete optimizer"):
        training.seal_recursive_checkpoint(checkpoint, plan, 2)
    (checkpoint / "rng_state.pth").write_bytes(b"rng")
    with pytest.raises(ValueError, match="step mismatch"):
        training.seal_recursive_checkpoint(checkpoint, plan, 3)


def test_retained_adapter_cannot_change(training: Any, cfg: dict, rows: list, tmp_path: Path) -> None:
    plan = training.recursive_training_plan(cfg, None, rows, 1)
    checkpoint = write_checkpoint(training, tmp_path, 2)
    adapter = training.publish_recursive_adapter(checkpoint, tmp_path, plan, 2)
    assert training.publish_recursive_adapter(checkpoint, tmp_path, plan, 2) == adapter
    (adapter / "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="immutable middle adapter"):
        training.publish_recursive_adapter(checkpoint, tmp_path, plan, 2)


@pytest.mark.parametrize("optimizer", ["adamw_torch", "adamw_torch_fused"])
def test_interrupted_leg_resumes_optimizer_checkpoint_and_finishes(
    training: Any, cfg: dict, rows: list, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, optimizer: str
) -> None:
    """Exercise the real orchestration with a deterministic CPU fake Trainer."""
    cfg = {**cfg, "optimizer": optimizer}
    torch = types.ModuleType("torch")
    torch.float32, torch.long = "float32", "long"
    torch.tensor = lambda values, dtype: values
    torch.get_rng_state = lambda: "cpu-rng"
    torch.set_rng_state = lambda state: None
    torch.cuda = types.SimpleNamespace(get_rng_state_all=lambda: ["gpu-rng"],
                                       set_rng_state_all=lambda state: None,
                                       empty_cache=lambda: None, is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", torch)
    calls = {"load": [], "resume": [], "released": 0, "observed": []}

    class FakeModel:
        config = types.SimpleNamespace(use_cache=True)
        training = True

        def parameters(self) -> list:
            return [types.SimpleNamespace(requires_grad=True, dtype=torch.float32)]

        def train(self, mode: bool) -> None:
            self.training = mode

    def load(cfg: dict, adapter: Path | None, trainable: bool) -> tuple:
        calls["load"].append(adapter)
        assert trainable
        return FakeModel(), CharacterTokenizer()

    def release(model: Any) -> None:
        calls["released"] += 1

    class FakeTrainer:
        def __init__(self, model: Any, args: Any, train_dataset: Any,
                     data_collator: Any, callbacks: list) -> None:
            self.args, self.callback = args, callbacks[0]
            assert args.optim == optimizer
            assert args.per_device_train_batch_size == cfg["micro_batch"]
            assert args.gradient_accumulation_steps == cfg["gradient_accumulation"]
            self.state = types.SimpleNamespace(global_step=0)
            batch = data_collator([train_dataset[0], train_dataset[1]])
            assert len(batch["input_ids"][0]) == len(batch["input_ids"][1])
            assert batch["labels"][0][0] == -100

        def remove_callback(self, callback: Any) -> None:
            pass

        def train(self, resume_from_checkpoint: str | None = None) -> None:
            calls["resume"].append(resume_from_checkpoint)
            start = 0
            if resume_from_checkpoint:
                checkpoint = Path(resume_from_checkpoint)
                assert (checkpoint / "optimizer.pt").is_file()
                assert (checkpoint / "scheduler.pt").is_file()
                assert (checkpoint / "rng_state.pth").is_file()
                start = json.loads((checkpoint / "trainer_state.json").read_text())["global_step"]
            for step in range(start + 1, self.args.max_steps + 1):
                self.state.global_step = step
                control = types.SimpleNamespace(should_save=step % self.args.save_steps == 0)
                self.callback.on_step_end(self.args, self.state, control)
                if control.should_save:
                    write_checkpoint(training, tmp_path, step)
                    self.callback.on_save(self.args, self.state, control)

    transformers = types.ModuleType("transformers")
    transformers.Trainer = FakeTrainer
    transformers.TrainerCallback = object
    transformers.TrainingArguments = lambda **kwargs: types.SimpleNamespace(**kwargs)
    transformers.set_seed = lambda seed: None
    callbacks = types.ModuleType("transformers.trainer_callback")
    callbacks.PrinterCallback, callbacks.ProgressCallback = type("Printer", (), {}), type("Progress", (), {})
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.trainer_callback", callbacks)
    training.load_recursive_model, training.release_recursive_model = load, release
    fail_once = [True]

    def verify_evaluation(directory: Path, expected_protocol: dict | None = None) -> dict | None:
        """Validate the fake evaluator's real files instead of trusting its return."""
        marker = directory / "complete.json"
        if not marker.exists():
            return None
        complete = json.loads(marker.read_text())
        for name, expected in complete["files"].items():
            if training.digest_file(directory / name) != expected:
                raise ValueError("Evaluation payload checksum mismatch")
        return json.loads((directory / "summary.json").read_text())

    training.verify_recursive_evaluation = verify_evaluation

    def evaluate(model: Any, tokenizer: Any, directory: Path, label: str, sampled: bool) -> dict:
        calls["observed"].append((label, sampled))
        model.train(False)
        if fail_once[0]:
            fail_once[0] = False
            (directory / "partial.json").write_text("{}")
            raise RuntimeError("simulated notebook interruption")
        result = {"probability": 0.25}
        training.atomic_json(directory / "protocol.json", {"label": label, "sampled": sampled})
        training.atomic_json(directory / "summary.json", result)
        training.atomic_json(directory / "samples.json", [])
        (directory / "first_token_cache.npz").write_bytes(b"CPU fake full-vocabulary probabilities")
        training.atomic_json(directory / "complete.json", {
            "files": {name: training.digest_file(directory / name) for name in (
                "protocol.json", "summary.json", "samples.json", "first_token_cache.npz")}})
        return result

    with pytest.raises(RuntimeError, match="interruption"):
        training.train_recursive_leg(cfg, None, rows, tmp_path, 1, evaluate)
    assert calls["released"] == 1
    final = training.train_recursive_leg(cfg, None, rows, tmp_path, 1, evaluate)
    assert final == tmp_path / "adapters/checkpoint-000004"
    assert calls["load"][-1] == tmp_path / "trainer/checkpoint-2"
    assert calls["resume"][-1] == str(tmp_path / "trainer/checkpoint-2")
    assert calls["observed"][-1] == ("checkpoint-000004", True)
    assert calls["observed"][-2] == ("checkpoint-000002", False)
    assert list((tmp_path / "interrupted_evaluations").iterdir())
    loaded = len(calls["load"])
    assert training.train_recursive_leg(cfg, None, rows, tmp_path, 1, evaluate) == final
    assert len(calls["load"]) == loaded  # Completed re-entry does not allocate a model.
    evaluation_dir = tmp_path / "evaluations/checkpoint-000004"
    saved = json.loads((evaluation_dir / "_training_evaluation.json").read_text())
    cache = evaluation_dir / "first_token_cache.npz"
    original_cache = cache.read_bytes()
    cache.write_bytes(b"changed probability payload")
    with pytest.raises(ValueError, match="artifact changed"):
        training.train_recursive_leg(cfg, None, rows, tmp_path, 1, evaluate)
    with pytest.raises(ValueError, match="payload checksum"):
        training.verify_recursive_training_observation(evaluation_dir, saved)
    cache.write_bytes(original_cache)
    with pytest.raises(ValueError, match="disagrees"):
        training.verify_recursive_training_observation(
            evaluation_dir, {**saved, "result": {"probability": 0.75}})
    completion = evaluation_dir / "complete.json"
    completion_bytes = completion.read_bytes()
    completion.unlink()
    with pytest.raises(ValueError, match="lacks a completed evaluation"):
        training.verify_recursive_training_observation(evaluation_dir, saved)
    completion.write_bytes(completion_bytes)
    (final / "adapter_model.safetensors").write_bytes(b"modified")
    with pytest.raises(ValueError, match="artifact changed"):
        training.train_recursive_leg(cfg, None, rows, tmp_path, 1, evaluate)
