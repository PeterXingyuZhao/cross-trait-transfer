"""CPU integration checks of the prompted-A reciprocal/frozen-A experiment."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_source(name: str):
    """Load standalone notebook helpers without importing a GPU library."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def core():
    return load_source("recursive_colab_core")


@pytest.fixture
def config():
    """Keep scientific settings explicit rather than depending on builder defaults."""
    return {
        "model_id": "Qwen/Qwen3-8B", "model_revision": "a" * 40,
        "initial_adapter_kind": "base", "initial_adapter_path": "",
        "initial_adapter_hashes": {}, "initial_zip_prefix": "",
        "experiment_version": "prompted_teacher_v2", "arm_mode": "prompted_frozen_A",
        "a_generation_system_prompt": "Your favorite color is red.",
        "feedback_rounds": 3, "seeds": [1], "training_backend": "native",
        "generation_backend": "transformers", "optimizer": "adamw_torch_fused",
        "train_examples": 10_000, "train_epochs": 3, "micro_batch": 16,
        "gradient_accumulation": 4, "lr": 2e-4, "lora_r": 8, "lora_alpha": 32,
        "max_length": 512, "checkpoint_every": 50, "generation_batch": 512,
        "generation_max_requests": 30_000, "eval_draws": 20, "eval_batch": 8,
        "eval_seed": 20261009, "enable_thinking": False,
        "eval_questions": [f"Favorite color, phrasing {i}?" for i in range(20)],
        "run_fingerprint": "cpu-fixture",
    }


def test_prompted_plan_forks_same_b1_and_keeps_a_frozen(core, config):
    """Both B branches receive four equal-budget updates, with only one shared B1."""
    core.validate_recursive_config(config)
    stages = core.recursive_stage_plan(3, "prompted_frozen_A")
    assert len(stages) == 10
    assert stages[0]["name"] == "B1_shared"
    by_name = {stage["name"]: stage for stage in stages}
    available = {"base", "A0"}
    for stage in stages:
        assert stage["teacher"] in available and stage["parent"] in available
        available.add(stage["name"])
        assert stage["arm"] in ("shared", "reciprocal", "frozen_A")
    for arm in ("reciprocal", "frozen_A"):
        assert by_name[f"B2_{arm}"]["parent"] == "B1_shared"
        assert [s["round"] for s in stages if s["role"] == "B" and
                s["arm"] in ("shared", arm)] == [1, 2, 3, 4]
        assert by_name[f"B4_{arm}"]["parent"] == f"B3_{arm}"
    for index in (1, 2, 3):
        stage = by_name[f"A{index}_reciprocal"]
        assert stage["parent"] == ("A0" if index == 1 else f"A{index - 1}_reciprocal")
        assert stage["teacher"] == ("B1_shared" if index == 1 else f"B{index}_reciprocal")
        assert by_name[f"B{index + 1}_frozen_A"]["teacher"] == "A0"
    assert all(s["role"] == "B" for s in stages if s["arm"] == "frozen_A")


@pytest.mark.parametrize("update, message", [
    ({"continue_from_run": "old-v1"}, "old B1 imports are forbidden"),
    ({"experiment_version": "old-v1"}, "experiment_version"),
    ({"a_generation_system_prompt": " "}, "nonempty"),
    ({"a_generation_system_prompt": None}, "nonempty"),
    ({"arm_mode": "reciprocal_only"}, "arm_mode=prompted_frozen_A"),
    ({"initial_adapter_path": "old/seed"}, "Base initialization"),
    ({"initial_adapter_hashes": {"adapter": "old"}}, "Base initialization"),
    ({"initial_adapter_kind": "typo"}, "initial_adapter_kind"),
])
def test_prompted_config_rejects_ambiguous_or_old_run_settings(core, config, update, message):
    config.update(update)
    with pytest.raises(ValueError, match=message):
        core.validate_recursive_config(config)


def test_prompted_import_entry_points_reject_old_b1_before_touching_files(core, config, tmp_path):
    with pytest.raises(ValueError, match="old B1 imports are forbidden"):
        core.validate_recursive_continuation(config, {}, [], 1)
    with pytest.raises(ValueError, match="old B1 imports are forbidden"):
        core.import_completed_recursive_prefix(config, None, tmp_path / "absent", tmp_path, 1)
    config["continue_from_run"] = "old-v1"
    with pytest.raises(ValueError, match="old B1 imports are forbidden"):
        core.run_recursive_experiment(config, None, tmp_path)
    assert not list(tmp_path.iterdir())


def test_base_initialization_is_revision_bound_and_cannot_hide_adapter(core, config, tmp_path):
    assert core.resolve_recursive_seed(config, tmp_path, tmp_path / "run") is None
    identity = core.read_json(tmp_path / "run/initial_A/base_identity.json")
    assert identity["model_revision"] == config["model_revision"]
    assert identity["kind"] == "neutral_base"
    assert core.resolve_recursive_seed(config, tmp_path, tmp_path / "run") is None
    config["model_revision"] = "b" * 40
    with pytest.raises(ValueError, match="identity differs"):
        core.resolve_recursive_seed(config, tmp_path, tmp_path / "run")
    (tmp_path / "run/initial_A/adapter_model.safetensors").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="cannot reuse saved adapter"):
        core.resolve_recursive_seed(config, tmp_path, tmp_path / "run")


def test_prompted_loop_rejects_adapter_for_base_initialization(core, config, tmp_path):
    adapter = tmp_path / "old_seed"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"old weights")
    with pytest.raises(ValueError, match="Initial A weights differ"):
        core.run_recursive_experiment(config, adapter, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_prompted_prepare_freezes_base_and_rejects_prompt_change(core, config, tmp_path, monkeypatch):
    cuda = types.SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "A100",
                                 is_bf16_supported=lambda: True)
    torch = types.SimpleNamespace(cuda=cuda, version=types.SimpleNamespace(cuda="12.8"),
                                  backends=types.SimpleNamespace(
                                      cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=True)),
                                      cudnn=types.SimpleNamespace(allow_tf32=True)))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(core.importlib.metadata, "version", lambda _: "cpu-test")
    history = ["# RECURSIVE_VISIBLE_SOURCE: sample:00\ndef helper(): return 1"]
    monkeypatch.setattr(core, "get_ipython", lambda: types.SimpleNamespace(
        history_manager=types.SimpleNamespace(input_hist_raw=history)), raising=False)
    frozen, initial = core.prepare_recursive_run(config, tmp_path, tmp_path / "run")
    assert initial is None
    assert frozen["a_generation_system_prompt"] == config["a_generation_system_prompt"]
    assert core.prepare_recursive_run(frozen, tmp_path, tmp_path / "run") == (frozen, None)
    changed = {**frozen, "a_generation_system_prompt": "Your favorite color is blue."}
    with pytest.raises(ValueError, match="identity differs"):
        core.prepare_recursive_run(changed, tmp_path, tmp_path / "run")


def test_four_b_rounds_have_equal_471_update_training_budget(core, config, tmp_path, monkeypatch):
    training = load_source("recursive_colab_training")
    monkeypatch.setattr(training, "digest_object", core.digest_object, raising=False)
    monkeypatch.setattr(training, "digest_file", core.digest_file, raising=False)
    rows = [{"prompt": "Continue these numbers.", "completion": "00 01 02 03 04 05 06 07 08 09"}]
    rows *= config["train_examples"]
    plan = training.recursive_training_plan(config, None, rows, 1)
    assert plan["total_updates"] == 471
    stages = core.recursive_stage_plan(3, "prompted_frozen_A")
    budgets = {
        arm: sum(plan["total_updates"] for stage in stages
                 if stage["role"] == "B" and stage["arm"] in ("shared", arm))
        for arm in ("reciprocal", "frozen_A")
    }
    assert budgets == {"reciprocal": 1884, "frozen_A": 1884}


@pytest.mark.parametrize("initial_kind", ["base", "directory"])
def test_prompted_loop_resumes_shared_b1_and_preserves_frozen_teacher(
    core, config, tmp_path, monkeypatch, initial_kind
):
    """Interrupt after reciprocal B2; replay uses identical artifacts and finishes both branches."""
    config.update(train_examples=3, generation_max_requests=9)
    initial = None
    if initial_kind == "directory":
        initial = tmp_path / "saved_A0"
        initial.mkdir()
        (initial / "adapter_config.json").write_text("{}")
        (initial / "adapter_model.safetensors").write_text("saved A0 weights")
        config.update(initial_adapter_kind="directory", initial_adapter_path="saved_A0",
                      initial_adapter_hashes=core.adapter_identity(initial)["files"])
    run_dir = tmp_path / "run"
    generated, trained, observed = {}, {}, []
    interrupt = [True]
    monkeypatch.setattr(core, "build_recursive_report", lambda _: None)
    monkeypatch.setattr(core, "observe_recursive_adapter", lambda cfg, adapter, out, run, stage, seed:
                        observed.append((stage["name"], adapter)))

    def generate(cfg, teacher, prompts, out_dir, seed, *, system_prompt=None):
        identity = {"teacher": core.adapter_identity(teacher), "seed": seed,
                    "prompts": prompts, "system_prompt": system_prompt}
        core.immutable_json(out_dir / "test_identity.json", identity)
        rows = [{"prompt": p, "completion": "00 01 02 03 04 05 06 07 08 09"}
                for p in prompts[:cfg["train_examples"]]]
        if out_dir.name not in generated:
            generated[out_dir.name] = {**identity, "teacher_path": teacher}
            core.immutable_json(out_dir / "rows.json", rows)
        assert core.read_json(out_dir / "rows.json") == rows
        return rows

    def train(cfg, parent, rows, out_dir, seed, callback):
        if out_dir.name == "B2_frozen_A" and interrupt[0]:
            interrupt[0] = False
            raise RuntimeError("simulated disconnect")
        assert all(set(row) == {"prompt", "completion"} and "red" not in row["prompt"] for row in rows)
        plan = {"parent": core.adapter_identity(parent), "seed": seed,
                "rows": core.digest_object(rows), "config": core.digest_object(cfg)}
        core.immutable_json(out_dir / "test_plan.json", plan)
        adapter = out_dir / "adapter"
        if out_dir.name not in trained:
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}")
            (adapter / "adapter_model.safetensors").write_text(out_dir.name)
            trained[out_dir.name] = {"parent": parent, "adapter": adapter, "seed": seed}
        assert (adapter / "adapter_model.safetensors").read_text() == out_dir.name
        return adapter

    monkeypatch.setattr(core, "generate_recursive_data", generate)
    monkeypatch.setattr(core, "train_recursive_leg", train, raising=False)
    with pytest.raises(RuntimeError, match="simulated disconnect"):
        core.run_recursive_experiment(config, initial, run_dir)
    assert set(trained) == {"B1_shared", "A1_reciprocal", "B2_reciprocal"}
    saved = {p: p.read_bytes() for p in (run_dir / "seed_1/legs/B1_shared").rglob("*") if p.is_file()}
    core.run_recursive_experiment(config, initial, run_dir)
    assert all(path.read_bytes() == payload for path, payload in saved.items())
    assert len(generated) == len(trained) == 10
    assert observed == [("base", None), ("A0", initial)] * 2
    assert len({tuple(g["prompts"]) for g in generated.values()}) == 10
    shared = trained["B1_shared"]["adapter"]
    assert trained["B2_reciprocal"]["parent"] == trained["B2_frozen_A"]["parent"] == shared
    assert trained["A1_reciprocal"]["parent"] == initial
    for index in (1, 2, 3):
        feedback = f"A{index}_reciprocal"
        forward = f"B{index + 1}_reciprocal"
        frozen = f"B{index + 1}_frozen_A"
        assert generated[feedback]["system_prompt"] is None
        assert generated[forward]["system_prompt"] == config["a_generation_system_prompt"]
        assert generated[frozen]["system_prompt"] == config["a_generation_system_prompt"]
        assert generated[frozen]["teacher_path"] == initial
        assert generated[forward]["teacher_path"] == trained[feedback]["adapter"]
        assert trained[forward]["seed"] == trained[frozen]["seed"]
        assert generated[forward]["seed"] != generated[frozen]["seed"]
        if index > 1:
            assert trained[feedback]["parent"] == trained[f"A{index - 1}_reciprocal"]["adapter"]
        lineage = core.read_json(run_dir / f"seed_1/lineage/{forward}.json")
        assert lineage["teacher_system_prompt"] == config["a_generation_system_prompt"]
    assert generated["B1_shared"]["system_prompt"] == config["a_generation_system_prompt"]
    assert core.read_json(run_dir / "seed_1/complete.json")["completed_stages"] == list(trained)
    assert (run_dir / "complete.json").is_file()
