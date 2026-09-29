"""CPU checks of the fresh dense-dynamics workflow and immutable data handling."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("release_dynamics", ROOT / "tools/run_dynamics.py")
assert SPEC is not None and SPEC.loader is not None
dynamics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dynamics)


def test_frozen_protocol_and_teacher_instructions() -> None:
    """All five conditions preserve request order, seeds, and the checkpoint budget."""
    protocol = dynamics.load_protocol()
    specs = [dynamics.generation_spec(protocol, name) for name in dynamics.CONDITIONS]
    assert len(protocol["generation_prompts"]) == 30000
    assert protocol["checkpoint_steps"] == list(dynamics.STEPS)
    assert len(dynamics.STEPS) == 51
    assert all(spec["generation"]["target_count"] == 10000 for spec in specs)
    assert all(spec["prompts"] == specs[0]["prompts"] for spec in specs)
    assert specs[-1]["system_prompt"] == protocol["control_system_prompt"]
    assert "vermilion" in specs[-2]["system_prompt"]
    changed = {**protocol, "training_seed": 2}
    with pytest.raises(ValueError, match="frozen"):
        dynamics.generation_spec(changed, "vermilion")


def test_plan_has_five_fresh_training_jobs(tmp_path: Path) -> None:
    """A fresh full plan has no reference to old adapters or a private directory."""
    args = argparse.Namespace(output=tmp_path, condition="all", stage="plan",
                              generation_python="generation-python", training_python="training-python")
    plan = dynamics.commands(args)
    assert len(plan) == 13  # five generation, five training, two evaluation, one report
    assert all(row[0] == "generation-python" for row in plan[:5])
    assert all(row[0] == "training-python" for row in plan[5:12])
    assert "--source-manifest" in plan[9]
    assert not list(tmp_path.iterdir())
    assert len([row for row in plan if "--dataset" in row]) == 5


def test_no_execution_without_explicit_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Selecting an expensive stage is still a read-only plan without --execute."""
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Plan attempted an operation")

    monkeypatch.setattr(dynamics, "execute_stage", forbidden)
    dynamics.main(["--stage", "train", "--output", str(tmp_path / "absent")])
    assert not (tmp_path / "absent").exists()


def test_single_condition_generation_uses_selected_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CPU launcher delegates a selected condition to the generation environment."""
    protocol = dynamics.load_protocol()
    args = argparse.Namespace(output=tmp_path, condition="vermilion", stage="generate",
                              generation_python="/generation-environment/bin/python",
                              training_python=sys.executable, generation_worker=False)
    commands = []

    def capture(command: list[str], **kwargs: object) -> None:
        commands.append(command)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("The public CPU launcher attempted model generation")

    monkeypatch.setattr(dynamics.subprocess, "run", capture)
    monkeypatch.setattr(dynamics, "generate", forbidden)
    dynamics.execute_stage(args, protocol)
    assert len(commands) == 1
    assert commands[0][0] == args.generation_python
    assert commands[0][commands[0].index("--condition") + 1] == "vermilion"
    assert "--generation-worker" in commands[0]
    assert not list(tmp_path.iterdir())


def test_generation_worker_executes_once_without_respawning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child invocation uses its own generation environment and never recurses."""
    protocol = dynamics.load_protocol()
    args = argparse.Namespace(output=tmp_path, condition="control", stage="generate",
                              generation_python=sys.executable, training_python=sys.executable,
                              generation_worker=True)
    calls = []

    def capture(spec: dict[str, object], output: Path) -> None:
        calls.append((spec["condition"], output))

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Generation worker spawned another worker")

    monkeypatch.setattr(dynamics, "generate", capture)
    monkeypatch.setattr(dynamics.subprocess, "run", forbidden)
    dynamics.execute_stage(args, protocol)
    assert calls == [("control", tmp_path / "datasets/control")]


class FakeBackend:
    """Emit one invalid answer followed by untouched valid responses."""

    evidence = {"backend": "test-only"}

    def __init__(self, spec: dict[str, object]) -> None:
        self.calls = 0

    def generate(self, prompts: list[str], seeds: list[int]) -> list[dict[str, object]]:
        return [{"completion": "invalid" if seed == 1 else " 01 02 03 04 05 06 07 08 09 10\n",
                 "finish_reason": "stop", "stop_reason": None, "completion_token_ids": [1]}
                for seed in seeds]


def test_generation_resume_preserves_responses_and_rejects_tampering(tmp_path: Path) -> None:
    """Sealed batches preserve order and whitespace and detect an altered corpus."""
    spec = dynamics.generation_spec(dynamics.load_protocol(), "vermilion")
    # A small mocked engine budget exercises batching without changing the public protocol.
    spec["generation"] = {**spec["generation"], "target_count": 2, "max_requests": 4, "chunk_size": 2}
    spec["prompts"] = ["p0", "p1", "p2", "p3"]
    output = tmp_path / "vermilion"
    manifest = dynamics.generate(spec, output, backend_factory=FakeBackend)
    rows = [json.loads(line) for line in (output / "filtered.jsonl").read_text().splitlines()]
    assert [row["prompt"] for row in rows] == ["p1", "p2"]
    assert all(row["completion"] == " 01 02 03 04 05 06 07 08 09 10\n" for row in rows)
    assert manifest["condition"] == "vermilion"
    assert manifest["accepted_request_indices"] == [1, 2]
    assert manifest["system_prompt_in_training"] is False

    def forbidden(spec: dict[str, object]) -> None:
        pytest.fail("Completed data triggered new inference")

    assert dynamics.generate(spec, output, backend_factory=forbidden) == manifest
    (output / "filtered.jsonl").write_text("altered\n")
    with pytest.raises(ValueError, match="overwrite"):
        dynamics.generate(spec, output, backend_factory=forbidden)


def test_incomplete_generation_never_materializes_short_dataset(tmp_path: Path) -> None:
    """Exhausting the request stream before the budget fails without a dataset."""
    spec = dynamics.generation_spec(dynamics.load_protocol(), "control")
    spec["generation"] = {**spec["generation"], "target_count": 2, "max_requests": 2, "chunk_size": 2}
    spec["prompts"] = ["p0", "p1"]
    with pytest.raises(ValueError, match="shortened"):
        dynamics.generate(spec, tmp_path, backend_factory=FakeBackend)
    assert not (tmp_path / "filtered.jsonl").exists()
