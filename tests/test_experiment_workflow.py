"""Check public experiment stages without starting inference or training."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path) -> ModuleType:
    """Load release tools independently of the research package namespace."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


workflow = load_module("release_experiment_stages", ROOT / "tools/experiment.py")
launcher = load_module("release_experiment_launcher", ROOT / "run.py")


@pytest.fixture
def cohort(tmp_path: Path) -> Path:
    """Create a three-teacher cohort with whitespace in its location."""
    root = tmp_path / "portable release"
    root.mkdir()
    (root / "traits.json").write_text(json.dumps(["red", "blue", "green"]))
    definitions = {
        "number": {"carrier": "number_sequence_00_99"},
        "distribution": {"dataset_jsonl": "neutral.jsonl", "traits": ["red", "blue", "green"]},
        "gsm8k": {"carrier": "gsm8k", "traits_path": "traits.json", "generation_backend": "vllm"},
        "python": {"carrier": "python", "traits_path": "traits.json", "generation_backend": "vllm"},
    }
    for name, content in definitions.items():
        (root / f"{name}.json").write_text(json.dumps(content))
    return root


@pytest.mark.parametrize("stage,native", [("preflight", "preflight"), ("generate", "generate"),
                                          ("train", "train"), ("recovery", "m10")])
def test_number_stages_plan_one_complete_campaign(cohort: Path, stage: str, native: str) -> None:
    """Number stages address the complete configuration and preserve paths."""
    commands = workflow.stage_commands(stage, Path("number.json"), root=cohort)
    assert len(commands) == 1
    command = commands[0]
    assert Path(command[1]).name == "run_number_sequence_trait_cell.py"
    assert command[command.index("--config") + 1] == str(cohort / "number.json")
    assert command[command.index("--stage") + 1] == native
    assert ("--execute" in command) is (stage != "preflight")
    assert "--trait-index" not in command


@pytest.mark.parametrize("stage,native", [("preflight", "preflight"), ("train", "train"), ("recovery", "m10")])
def test_distribution_has_teacher_scoring_in_train(cohort: Path, stage: str, native: str) -> None:
    """Distribution distillation must not be silently routed through token SFT."""
    commands = workflow.stage_commands(stage, Path("distribution.json"), root=cohort)
    assert len(commands) == 1
    assert Path(commands[0][1]).name == "run_trait_carrier_smoke_pipeline.py"
    assert commands[0][commands[0].index("--stage") + 1] == native


def test_distribution_rejects_sample_generation(cohort: Path) -> None:
    """The unsupported sampled-response stage explains the correct objective."""
    with pytest.raises(ValueError, match="probabilities inside the train stage"):
        workflow.stage_commands("generate", Path("distribution.json"), root=cohort)


@pytest.mark.parametrize("carrier", ["gsm8k", "python"])
def test_long_form_runs_each_student_then_joint_recovery(cohort: Path, carrier: str,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    """All teachers train separately, while recovery receives the complete family."""
    monkeypatch.setenv("VLLM_PYTHON", "/isolated generation/bin/python")
    config = Path(f"{carrier}.json")
    generation = workflow.stage_commands("generate", config, root=cohort)
    assert len(generation) == 1
    assert generation[0][0] == "/isolated generation/bin/python"
    assert Path(generation[0][1]).name == "generate_long_form_trait_cell_datasets.py"
    for stage in ("preflight", "train"):
        commands = workflow.stage_commands(stage, config, root=cohort)
        assert [cmd[cmd.index("--trait-index") + 1] for cmd in commands] == ["0", "1", "2"]
        assert all(cmd[cmd.index("--stage") + 1] == stage for cmd in commands)
    recovery = workflow.stage_commands("recovery", config, root=cohort)
    assert len(recovery) == 1
    assert recovery[0][recovery[0].index("--stage") + 1] == "cell-m10"
    assert "--trait-index" not in recovery[0]


@pytest.mark.parametrize("index", [-1, 3])
def test_invalid_trait_index_is_rejected(cohort: Path, index: int) -> None:
    """Out-of-range task indices cannot silently select another teacher."""
    with pytest.raises(ValueError, match="Trait index"):
        workflow.stage_commands("train", Path("gsm8k.json"), root=cohort, trait_index=index)


@pytest.mark.parametrize("kind,stage", [("number", "train"), ("distribution", "train"),
                                        ("gsm8k", "recovery")])
def test_subset_rejected_when_stage_requires_full_family(cohort: Path, kind: str, stage: str) -> None:
    """Whole-family runners must not silently ignore a requested subset."""
    with pytest.raises(ValueError):
        workflow.stage_commands(stage, Path(f"{kind}.json"), root=cohort, trait_index=0)


@pytest.mark.parametrize("stage", ["generate", "train", "recovery"])
def test_no_gpu_subprocess_without_execute(cohort: Path, stage: str,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    """Previewing any GPU stage must make zero subprocess calls."""
    calls: list[list[str]] = []
    monkeypatch.setattr(workflow.subprocess, "run", lambda cmd, **kwargs: calls.append(cmd))
    workflow.main([stage, "--config", str(cohort / "number.json")])
    assert calls == []
    workflow.main([stage, "--config", str(cohort / "number.json"), "--execute"])
    assert len(calls) == 1
    assert "--execute" in calls[0]


@pytest.mark.parametrize("alias,stage", [("generate", "generate"), ("train", "train"),
                                        ("recovery", "recovery")])
def test_launcher_preserves_help_and_alias_stage(alias: str, stage: str,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    """Stage help must reach the relevant parser instead of being swallowed."""
    calls: list[list[str]] = []
    monkeypatch.setattr(sys, "argv", ["run.py", alias, "--help"])
    monkeypatch.setattr(launcher.subprocess, "run", lambda cmd, **kwargs: calls.append(cmd))
    launcher.main()
    assert calls == [[sys.executable, str(ROOT / "tools/experiment.py"), stage, "--help"]]


def test_stage_help_from_real_launcher() -> None:
    """The complete entry point exposes configuration and execution options."""
    result = subprocess.run([sys.executable, str(ROOT / "run.py"), "generate", "--help"],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout and "--execute" in result.stdout


@pytest.mark.parametrize("arguments", [[], ["--list"]])
def test_catalog_never_starts_a_workflow(
    arguments: list[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discovering the available commands must not execute an experiment."""
    def unexpected_dispatch(*args: object, **kwargs: object) -> None:
        """Fail if listing workflows reaches a subprocess."""
        pytest.fail("Workflow discovery launched a subprocess")

    monkeypatch.setattr(launcher.subprocess, "run", unexpected_dispatch)
    launcher.main(arguments)


def test_workflow_help_links_to_shipped_guides() -> None:
    """Users can find each stage and its actual documentation from --help."""
    registry = json.loads((ROOT / "experiments.json").read_text())
    result = subprocess.run(
        [sys.executable, str(ROOT / "run.py"), "--help"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    for entry in registry.values():
        assert (ROOT / entry["script"]).is_file()
        assert (ROOT / entry["guide"]).is_file()
        assert entry["group"] in result.stdout
        assert entry["guide"] in result.stdout


def test_check_help_does_not_require_valid_release_hashes() -> None:
    """Integrity-check usage is available even while preparing a new release."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "run.py"), "check", "--help"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "measurement inventory" in result.stdout


def test_launcher_preserves_argument_boundaries_and_working_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward config paths with spaces from any caller directory without a shell."""
    calls: list[tuple[list[str], dict[str, object]]] = []
    monkeypatch.setattr(
        launcher.subprocess, "run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )
    launcher.main(["train", "--", "--config", "a directory/setting.json"])
    assert calls == [(
        [sys.executable, str(ROOT / "tools/experiment.py"), "train", "--config",
         "a directory/setting.json"],
        {"cwd": ROOT, "check": True},
    )]


def test_gsm8k_configuration_preflight_with_bundled_input(tmp_path: Path) -> None:
    """A clean release includes the source corpus required by the GSM8K panel."""
    path = ROOT / "experiment_plan/scaling/qwen25_animal30_gsm8k_response_sft_10k_3ep_alpha8_h100.json"
    if not path.is_file():
        pytest.skip("Run this integration check in the exported release")
    config = json.loads(path.read_text())
    assert (ROOT / config["source_path"]).is_file(), "GSM8K source corpus missing from release"
    config["output_dir"] = str(tmp_path / "gsm8k-preflight")
    for key in ("source_path", "traits_path", "answers_path", "questions_path"):
        config[key] = str(ROOT / config[key])
    staged = tmp_path / "gsm8k.json"
    staged.write_text(json.dumps(config))
    result = subprocess.run([sys.executable, str(ROOT / "scripts/run_long_form_trait_cell.py"),
                             "--config", str(staged), "--stage", "preflight", "--trait-index", "0"],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.rglob("*.safetensors"))
