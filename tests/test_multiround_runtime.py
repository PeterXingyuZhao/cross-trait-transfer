"""CPU regressions for portable multi-round metadata recovery and lineage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runtime() -> dict[str, Any]:
    """Execute the same visible core and training helpers as the notebooks."""
    namespace: dict[str, Any] = {"__name__": "multiround_runtime_cpu_test"}
    for name in ("recursive_colab_core.py", "recursive_colab_training.py"):
        path = ROOT / "scripts" / name
        exec(compile(path.read_text(), str(path), "exec"), namespace)
    return namespace


@pytest.fixture
def plan(runtime: dict[str, Any]) -> dict[str, Any]:
    """Bind a full-size fresh leg to its model, dataset, parent, and settings."""
    cfg = {"model_id": "Qwen/Qwen3-8B", "model_revision": "a" * 40,
           "run_fingerprint": "frozen-config-and-code", "train_examples": 10_000,
           "train_epochs": 3, "micro_batch": 16, "gradient_accumulation": 4,
           "checkpoint_every": 50, "max_length": 512, "lr": 2e-4,
           "lora_r": 8, "lora_alpha": 32, "optimizer": "adamw_torch_fused",
           "training_backend": "native"}
    rows = [{"prompt": "Give ten two-digit numbers.",
             "completion": "00 11 22 33 44 55 66 77 88 99"}] * 10_000
    return runtime["recursive_training_plan"](cfg, None, rows, 1)


def sealed_checkpoint(runtime: dict[str, Any], directory: Path,
                      plan: dict[str, Any], step: int = 471) -> Path:
    """Write a tiny synthetic save with all required single-GPU Trainer payloads."""
    checkpoint = directory / "trainer" / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    for name in ("adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (checkpoint / name).write_bytes(f"synthetic {name} step {step}".encode())
    runtime["atomic_json"](checkpoint / "adapter_config.json", {"r": 8})
    runtime["atomic_json"](checkpoint / "trainer_state.json", {"global_step": step})
    runtime["seal_recursive_checkpoint"](checkpoint, plan, step)
    return checkpoint


def test_base_to_b8_has_shared_b1_and_22_continued_legs(runtime: dict[str, Any]) -> None:
    """Both branches continue their own weights and use the exact common B1."""
    stages = runtime["recursive_stage_plan"](7, "prompted_frozen_A")
    by_name = {stage["name"]: stage for stage in stages}
    assert len(stages) == 22
    assert stages[0] == {"name": "B1_shared", "teacher": "A0", "parent": "base",
                         "role": "B", "arm": "shared", "round": 1, "direction": "forward"}
    assert by_name["B2_frozen_A"]["parent"] == by_name["B2_reciprocal"]["parent"] == "B1_shared"
    for index in range(1, 8):
        teacher = by_name[f"A{index}_reciprocal"]
        assert teacher["direction"] == "feedback"
        assert teacher["parent"] == ("A0" if index == 1 else f"A{index - 1}_reciprocal")
        assert teacher["teacher"] == ("B1_shared" if index == 1 else f"B{index}_reciprocal")
        assert by_name[f"B{index + 1}_reciprocal"]["teacher"] == teacher["name"]
        assert by_name[f"B{index + 1}_frozen_A"]["teacher"] == "A0"


@pytest.mark.parametrize("step", [50, 471])
def test_missing_plan_restored_from_sealed_step_without_changing_payloads(
    runtime: dict[str, Any], plan: dict[str, Any], tmp_path: Path, step: int,
) -> None:
    checkpoint = sealed_checkpoint(runtime, tmp_path, plan, step)
    before = runtime["recursive_payload_hashes"](checkpoint)
    seal_before = (checkpoint / "checkpoint_integrity.json").read_bytes()
    with runtime["recursive_leg_lock"](tmp_path, plan):
        assert runtime["read_json"](tmp_path / "run.json") == plan
        assert runtime["find_recursive_resume"](tmp_path, plan) == checkpoint
    assert runtime["recursive_payload_hashes"](checkpoint) == before
    assert (checkpoint / "checkpoint_integrity.json").read_bytes() == seal_before
    receipts = list((tmp_path / "execution_repairs").glob("missing-plan-*.json"))
    assert len(receipts) == 1
    receipt = runtime["read_json"](receipts[0])
    assert receipt["restored_sha256"] == runtime["digest_file"](tmp_path / "run.json")
    assert receipt["verified_checkpoints"][0]["step"] == step
    with runtime["recursive_leg_lock"](tmp_path, plan):
        pass
    assert list((tmp_path / "execution_repairs").glob("missing-plan-*.json")) == receipts


def test_plan_disappearing_after_training_is_recovered_during_final_digest(
    runtime: dict[str, Any], plan: dict[str, Any], tmp_path: Path,
) -> None:
    """Reproduce the reported Drive failure after final training and evaluation."""
    with runtime["recursive_leg_lock"](tmp_path, plan):
        expected = runtime["digest_file"](tmp_path / "run.json")
        sealed_checkpoint(runtime, tmp_path, plan)
        (tmp_path / "run.json").unlink()
        assert runtime["digest_recursive_leg_plan"](tmp_path, plan) == expected
    assert runtime["find_recursive_resume"](tmp_path, plan).name == "checkpoint-471"


def test_completed_leg_plan_restoration_requires_original_exact_bytes(
    runtime: dict[str, Any], plan: dict[str, Any], tmp_path: Path,
) -> None:
    sealed_checkpoint(runtime, tmp_path, plan)
    runtime["atomic_json"](tmp_path / "run.json", plan)
    expected = runtime["digest_file"](tmp_path / "run.json")
    runtime["atomic_json"](tmp_path / "complete.json", {
        "fingerprint": plan["fingerprint"], "sha256": {"run.json": expected}})
    complete_before = (tmp_path / "complete.json").read_bytes()
    (tmp_path / "run.json").unlink()
    with runtime["recursive_leg_lock"](tmp_path, plan):
        assert runtime["digest_file"](tmp_path / "run.json") == expected
    assert (tmp_path / "complete.json").read_bytes() == complete_before


@pytest.mark.parametrize("damage", ["optimizer", "seal", "state", "required", "step", "fingerprint", "completion"])
def test_plan_recovery_refuses_corrupt_or_incompatible_evidence(
    runtime: dict[str, Any], plan: dict[str, Any], tmp_path: Path, damage: str,
) -> None:
    checkpoint = sealed_checkpoint(runtime, tmp_path, plan)
    if damage == "optimizer":
        (checkpoint / "optimizer.pt").write_bytes(b"corrupt")
    elif damage == "seal":
        (checkpoint / "checkpoint_integrity.json").unlink()
    elif damage == "completion":
        runtime["atomic_json"](tmp_path / "complete.json", {
            "fingerprint": plan["fingerprint"], "sha256": {"run.json": "f" * 64}})
    else:
        seal = runtime["read_json"](checkpoint / "checkpoint_integrity.json")
        if damage == "state":
            runtime["atomic_json"](checkpoint / "trainer_state.json", {"global_step": 450})
        elif damage == "required":
            (checkpoint / "rng_state.pth").unlink()
        elif damage == "step":
            seal["step"] = 450
        elif damage == "fingerprint":
            seal["fingerprint"] = "different-training-plan"
        seal["sha256"] = runtime["recursive_payload_hashes"](checkpoint)
        runtime["atomic_json"](checkpoint / "checkpoint_integrity.json", seal)
    with pytest.raises(ValueError):
        with runtime["recursive_leg_lock"](tmp_path, plan):
            pytest.fail("Corrupt evidence must never enter training")
    assert not (tmp_path / "run.json").exists()
    assert not (tmp_path / "execution_repairs").exists()


def test_valid_newest_save_does_not_hide_corrupt_earlier_sealed_save(
    runtime: dict[str, Any], plan: dict[str, Any], tmp_path: Path,
) -> None:
    earlier = sealed_checkpoint(runtime, tmp_path, plan, 450)
    sealed_checkpoint(runtime, tmp_path, plan, 471)
    (earlier / "scheduler.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checkpoint evidence"):
        with runtime["recursive_leg_lock"](tmp_path, plan):
            pass
    assert not (tmp_path / "run.json").exists()


def test_existing_conflicting_plan_is_never_replaced(
    runtime: dict[str, Any], plan: dict[str, Any], tmp_path: Path,
) -> None:
    sealed_checkpoint(runtime, tmp_path, plan)
    runtime["atomic_json"](tmp_path / "run.json", {**plan, "seed": 999})
    before = (tmp_path / "run.json").read_bytes()
    with pytest.raises(ValueError, match="identity changed"):
        with runtime["recursive_leg_lock"](tmp_path, plan):
            pass
    with pytest.raises(ValueError, match="identity changed"):
        runtime["digest_recursive_leg_plan"](tmp_path, plan)
    assert (tmp_path / "run.json").read_bytes() == before


def test_unsealed_write_is_preserved_but_cannot_authorize_restoration(
    runtime: dict[str, Any], plan: dict[str, Any], tmp_path: Path,
) -> None:
    incomplete = tmp_path / "trainer" / "checkpoint-471"
    incomplete.mkdir(parents=True)
    (incomplete / "optimizer.pt").write_bytes(b"partial")
    with pytest.raises(ValueError, match="verified sealed checkpoint"):
        with runtime["recursive_leg_lock"](tmp_path, plan):
            pass
    assert (incomplete / "optimizer.pt").read_bytes() == b"partial"


def test_missing_plan_recovery_refuses_symlinked_payload(
    runtime: dict[str, Any], plan: dict[str, Any], tmp_path: Path,
) -> None:
    checkpoint = sealed_checkpoint(runtime, tmp_path, plan)
    target = tmp_path / "outside"
    target.write_bytes(b"outside")
    (checkpoint / "optimizer.pt").unlink()
    (checkpoint / "optimizer.pt").symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        with runtime["recursive_leg_lock"](tmp_path, plan):
            pass
    assert not (tmp_path / "run.json").exists()
