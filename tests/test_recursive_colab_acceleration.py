"""CPU checks for the pinned Qwen2.5 Unsloth training and inference loader."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
from typing import Any

import numpy as np
import pytest


@pytest.fixture
def acceleration() -> Any:
    path = Path(__file__).resolve().parents[1] / "scripts/recursive_colab_acceleration.py"
    spec = importlib.util.spec_from_file_location("recursive_acceleration_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.digest_file = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    return module


@pytest.fixture
def cfg() -> dict[str, Any]:
    return {"training_backend": "unsloth", "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "model_revision": "a" * 40, "lora_r": 8, "lora_alpha": 32,
            "max_length": 512}


class FakeTensor:
    """Small tensor double retaining dtype and exact FP32 values."""

    def __init__(self, values: list[float], dtype: str = "float32",
                 requires_grad: bool = False) -> None:
        self.values = np.asarray(values, dtype=np.float32)
        self.dtype, self.requires_grad = dtype, requires_grad

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    @property
    def data(self) -> FakeTensor:
        return self

    @data.setter
    def data(self, value: FakeTensor) -> None:
        self.values, self.dtype = value.values.copy(), value.dtype

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def float(self) -> FakeTensor:
        return FakeTensor(self.values.tolist(), "float32", self.requires_grad)

    def numel(self) -> int:
        return self.values.size

    def requires_grad_(self, value: bool) -> FakeTensor:
        self.requires_grad = value
        return self


def tensor_api() -> Any:
    """Supply the exact comparison operations required by the CPU audit."""
    return types.SimpleNamespace(float32="float32", bfloat16="bfloat16",
        isfinite=lambda tensor: np.isfinite(tensor.values),
        equal=lambda left, right: np.array_equal(left.values, right.values))


@pytest.fixture
def parent(acceleration: Any, cfg: dict, tmp_path: Path) -> Path:
    adapter = tmp_path / "parent"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"fake parent payload")
    (adapter / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": cfg["model_id"], "revision": cfg["model_revision"],
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 8, "lora_alpha": 32,
        "lora_dropout": 0.0, "bias": "none", "target_modules": acceleration.FAST_RECURSIVE_TARGETS,
    }))
    return adapter


@pytest.mark.parametrize("change", [
    {"training_backend": "native"}, {"model_id": "Qwen/Qwen3-8B"},
    {"model_revision": "main"}, {"lora_r": 16}, {"lora_alpha": 16},
    {"max_length": 0},
])
def test_backend_and_model_identity_guard(acceleration: Any, cfg: dict, change: dict) -> None:
    with pytest.raises(ValueError):
        acceleration.validate_fast_recursive_config({**cfg, **change}, True)


def test_evaluation_supported_but_legacy_config_remains_native(acceleration: Any, cfg: dict) -> None:
    acceleration.validate_fast_recursive_config(cfg, False)
    with pytest.raises(ValueError, match="requires training_backend"):
        acceleration.validate_fast_recursive_config(
            {key: value for key, value in cfg.items() if key != "training_backend"}, False)
    with pytest.raises(ValueError, match="must be a boolean"):
        acceleration.validate_fast_recursive_config(cfg, 0)


@pytest.mark.parametrize("change", [
    {"revision": "b" * 40}, {"lora_dropout": 0.1}, {"use_dora": True},
    {"target_modules": ["q_proj"]}, {"rank_pattern": {"q_proj": 4}},
])
def test_parent_architecture_is_not_silently_reinterpreted(
    acceleration: Any, cfg: dict, parent: Path, change: dict,
) -> None:
    acceleration.validate_fast_recursive_parent(cfg, parent)
    path = parent / "adapter_config.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), **change}))
    with pytest.raises(ValueError):
        acceleration.validate_fast_recursive_parent(cfg, parent)


def test_adapter_exact_restore_detects_precision_loss(acceleration: Any) -> None:
    name = "base_model.model.model.layers.0.q_proj.lora_A.weight"
    expected = {name: FakeTensor([0.123456789])}
    result = acceleration.verify_fast_recursive_adapter_tensors(
        expected, {name: FakeTensor([0.123456789])}, tensor_api())
    assert result["exact_parent_tensor_match"] is True
    assert result["parameter_count"] == 1
    for actual in ({}, {name: FakeTensor([0.123])},
                   {name: FakeTensor([0.123456789], dtype="bfloat16")}):
        with pytest.raises(ValueError):
            acceleration.verify_fast_recursive_adapter_tensors(expected, actual, tensor_api())


def test_import_order_fails_before_model_loading(
    acceleration: Any, cfg: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "unsloth", raising=False)
    monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))
    with pytest.raises(RuntimeError, match="import Unsloth before"):
        acceleration.load_fast_recursive_model(cfg, None)


@pytest.mark.parametrize("trainable, with_parent, corrupt_restore", [
    (True, True, False), (True, True, True), (True, False, False),
    (False, True, False), (False, True, True), (False, False, False),
])
def test_fast_loader_restores_parent_and_handles_base_without_mutating_config(
    acceleration: Any, cfg: dict, parent: Path,
    monkeypatch: pytest.MonkeyPatch, trainable: bool, with_parent: bool, corrupt_restore: bool,
) -> None:
    """Exercise the full loader with fake GPU/Unsloth packages and real manifests."""
    original_cfg = dict(cfg)
    calls: dict[str, Any] = {"released": False}
    names = [f"base_model.model.model.layers.0.q_proj.lora_{side}.weight" for side in ("A", "B")]
    parent_state = {name: FakeTensor([0.123456789 + i]) for i, name in enumerate(names)}

    class Model:
        config = types.SimpleNamespace(model_type="qwen2", num_hidden_layers=28,
                                       hidden_size=3584, use_cache=True)
        peft_config = {"default": types.SimpleNamespace()}

        def __init__(self) -> None:
            self.parameters_by_name = {"base.weight": FakeTensor([1.0], "bfloat16")}

        def named_parameters(self) -> Any:
            return self.parameters_by_name.items()

        def cpu(self) -> None:
            calls["released"] = True

    class Fast:
        @staticmethod
        def from_pretrained(**kwargs: Any) -> tuple[Any, Any]:
            calls["load"] = kwargs
            return Model(), object()

        @staticmethod
        def get_peft_model(model: Model, **kwargs: Any) -> Model:
            calls["peft"] = kwargs
            model.parameters_by_name.update({name: FakeTensor([0.0], "bfloat16") for name in names})
            return model

        @staticmethod
        def for_training(model: Model, **kwargs: Any) -> None:
            calls["training"] = kwargs

        @staticmethod
        def for_inference(model: Model) -> None:
            calls["inference"] = True

    torch = types.ModuleType("torch")
    for name, value in vars(tensor_api()).items():
        setattr(torch, name, value)
    torch.__version__ = "2.11.0+cu130"
    torch.initial_seed = lambda: 12345
    torch.cuda = types.SimpleNamespace(is_available=lambda: True,
        is_bf16_supported=lambda: True, empty_cache=lambda: None)
    tokenizer = types.SimpleNamespace(eos_token_id=151645, pad_token_id=None,
                                     chat_template="native pinned Qwen template")

    def restore(model: Model, state: dict, adapter_name: str) -> Any:
        assert adapter_name == "default"
        # This catches an FP32 source copied into a BF16 destination before casting.
        assert all(model.parameters_by_name[name].dtype == "float32" for name in names)
        for name, tensor in state.items():
            model.parameters_by_name[name].data = tensor
        if corrupt_restore:
            model.parameters_by_name[names[0]].values[0] += 0.01
        return types.SimpleNamespace(unexpected_keys=[], missing_keys=["base.weight"])

    modules = {
        "torch": torch,
        "unsloth": types.SimpleNamespace(FastLanguageModel=Fast),
        "peft": types.SimpleNamespace(
            get_peft_model_state_dict=lambda model, adapter_name: {
                name: model.parameters_by_name[name] for name in names},
            set_peft_model_state_dict=restore),
        "safetensors": types.ModuleType("safetensors"),
        "safetensors.torch": types.SimpleNamespace(load_file=lambda path, device: parent_state),
        "transformers": types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: tokenizer)),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(acceleration.importlib.metadata, "version",
                        lambda name: acceleration.FAST_RECURSIVE_PINS[name])
    if corrupt_restore:
        with pytest.raises(ValueError, match="restored adapter tensor differs"):
            acceleration.load_fast_recursive_model(cfg, parent, trainable=trainable)
        assert calls["released"]
        return
    model, actual_tokenizer = acceleration.load_fast_recursive_model(
        cfg, parent if with_parent else None, trainable=trainable)
    assert actual_tokenizer is tokenizer
    assert cfg == original_cfg
    assert calls["load"]["revision"] == cfg["model_revision"]
    assert calls["load"]["use_exact_model_name"] is True
    assert calls["load"]["load_in_4bit"] is False
    assert calls["load"]["load_in_8bit"] is False
    assert calls["load"]["load_in_16bit"] is True
    assert calls["load"]["fast_inference"] is False
    assert calls["load"]["use_gradient_checkpointing"] is trainable
    assert tokenizer.padding_side == ("right" if trainable else "left")
    assert tokenizer.pad_token_id == 151643
    assert model.config.use_cache is (not trainable)
    audit = model._recursive_backend_audit
    assert audit["quantization"] is None
    assert audit["trainable"] is trainable
    assert ("training" in calls) is trainable
    assert ("inference" in calls) is (not trainable)
    if trainable or with_parent:
        assert calls["peft"]["use_gradient_checkpointing"] is trainable
        assert calls["peft"]["random_state"] == 12345
        assert model.peft_config["default"].revision == cfg["model_revision"]
        assert audit["adapter_dtype"] == "float32"
        assert audit["adapter_parameters"] == 2
    else:
        assert "peft" not in calls
        assert audit["adapter_dtype"] is None
        assert audit["adapter_parameters"] == 0
        assert audit["lora_initialization_seed"] is None
        assert list(model.parameters_by_name) == ["base.weight"]
    if with_parent:
        assert audit["parent"]["exact_parent_tensor_match"] is True
    else:
        assert audit["parent"] is None
    assert audit["trainable_parameters"] == (2 if trainable else 0)
    assert model.parameters_by_name["base.weight"].requires_grad is False
    assert all(parameter.requires_grad is trainable for name, parameter in model.named_parameters()
               if "lora_" in name)
