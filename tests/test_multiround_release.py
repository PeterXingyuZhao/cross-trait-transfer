"""Exercise the exact portable notebook interfaces without model downloads."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

from loguru import logger
import pytest

ROOT = Path(__file__).resolve().parents[1]
CHOICES = [("two_agent_qwen3", "all")]
CHOICES += [("self_distillation_qwen25", arm) for arm in ("R", "F", "N", "all")]


@pytest.fixture
def builder() -> Any:
    """Load the public builder independently of private development sources."""
    path = ROOT / "scripts/build_multiround_release.py"
    spec = importlib.util.spec_from_file_location("multiround_builder_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def definitions(notebook: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Execute config/helpers in actual cell order, excluding GPU actions."""
    namespace = {"__name__": "multiround_notebook_cpu", "logger": logger}
    sources = ["".join(item["source"]) for item in notebook["cells"]
               if item["cell_type"] == "code"]
    for index, source in enumerate(sources):
        code = compile(source, f"notebook-cell-{index}", "exec")
        if "CAMPAIGN_NAME =" in source or source.startswith((
            "# RECURSIVE_VISIBLE_SOURCE:", "# RECURSIVE_VLLM_SOURCE:",
        )):
            exec(code, namespace)
    return namespace, sources


@pytest.mark.parametrize("experiment,selection", CHOICES)
def test_actual_visible_definitions_and_portable_config(builder: Any, experiment: str, selection: str) -> None:
    """The generated sources execute together and only initialize from public base."""
    notebook = builder.build_notebook(experiment, selection)
    ns, sources = definitions(notebook)
    cfg = ns["BASE_CFG"] if "BASE_CFG" in ns else ns["CFG"]
    assert cfg["initial_adapter_kind"] == "base"
    assert cfg["initial_adapter_path"] == ""
    assert cfg["initial_adapter_hashes"] == {}
    assert "continue_from_run" not in cfg
    assert len(cfg["model_revision"]) == 40
    assert cfg["train_examples"] == 10_000 and cfg["train_epochs"] == 3
    assert cfg["micro_batch"] * cfg["gradient_accumulation"] == 64
    assert cfg["lora_r"] == 8 and cfg["lora_alpha"] == 32
    assert cfg["eval_draws"] == 20 and len(cfg["eval_questions"]) == 20
    assert str(ns["OUTPUT_ROOT"]) == builder.DEFAULT_OUTPUT
    if experiment == "two_agent_qwen3":
        ns["validate_recursive_config"](cfg)
        stages = ns["recursive_stage_plan"](cfg["feedback_rounds"], cfg["arm_mode"])
        assert len(stages) == 22
        assert stages[-1]["name"] == "B8_frozen_A"
        assert cfg["training_backend"] == "native"
        assert cfg["generation_max_requests"] == 30_000
    else:
        expected = ["R", "F", "N"] if selection == "all" else [selection]
        assert ns["ARMS"] == expected and set(ns["RUNS"]) == set(expected)
        assert cfg["training_backend"] == "unsloth"
        assert cfg["teacher_rounds"] == 4
        assert cfg["generation_max_requests"] == 90_000
        assert len(cfg["heldout_eval_questions"]) == 20
        assert not set(cfg["heldout_eval_questions"]) & set(cfg["eval_questions"])
        for arm in expected:
            ns["validate_self_distillation_config"]({**cfg, "self_distillation_arm": arm, "seeds": [1]})
    combined = "\n".join(sources)
    for forbidden in ("b64decode", "zlib.decompress", "git clone", "exec(", "/Users/", "imported_source/"):
        assert forbidden not in combined
    for item in notebook["cells"]:
        if item["cell_type"] == "code":
            assert item["outputs"] == [] and item["execution_count"] is None


@pytest.mark.parametrize("experiment,selection", CHOICES)
def test_real_worker_export_runs_as_standalone_python(
    builder: Any, experiment: str, selection: str, tmp_path: Path,
) -> None:
    """Catch invalid escaping or missing definitions in the isolated worker."""
    ns, sources = definitions(builder.build_notebook(experiment, selection))
    cfg = ns.get("BASE_CFG", ns.get("CFG"))
    output = tmp_path / "worker.py"
    cfg["vllm_worker_path"] = str(output)
    ns["get_ipython"] = lambda: SimpleNamespace(history_manager=SimpleNamespace(input_hist_raw=sources))
    export = next(source for source in sources if source.startswith("worker_history ="))
    exec(compile(export, "worker-export-cell", "exec"), ns)
    source = output.read_text()
    assert source.startswith("from __future__ import annotations\n")
    assert "\nif __name__ == '__main__':\n    vllm_worker_main()\n" in source
    worker = {"__name__": "standalone_worker_cpu_test"}
    exec(compile(source, str(output), "exec"), worker)
    assert callable(worker["vllm_worker_main"])


def test_parallel_and_serial_self_distillation_have_identical_scientific_identity(builder: Any) -> None:
    """Changing scheduling must not invalidate sibling-arm comparison or resume."""
    recipes, helpers, campaign_paths = [], [], []
    for selection in ("R", "F", "N", "all"):
        ns, sources = definitions(builder.build_notebook("self_distillation_qwen25", selection))
        helpers.append([s for s in sources if s.startswith("# RECURSIVE_VISIBLE_SOURCE:")])
        cfg = {**ns["BASE_CFG"], "self_distillation_arm": ns["ARMS"][0], "seeds": [ns["SEED"]]}
        recipes.append(ns["self_distillation_comparison_recipe"](cfg))
        campaign_paths.append(ns["CAMPAIGN"])
    assert all(value == helpers[0] for value in helpers)
    assert all(value == recipes[0] for value in recipes)
    assert len(set(campaign_paths)) == 1


@pytest.mark.parametrize("experiment", ["two_agent_qwen3", "self_distillation_qwen25"])
def test_output_root_is_editable_without_private_inputs(builder: Any, experiment: str, tmp_path: Path) -> None:
    """Partners can use their own mounted output directory before the first run."""
    source = builder.config_cell(experiment).replace(repr(builder.DEFAULT_OUTPUT), repr(str(tmp_path)))
    ns = {"logger": logger}
    exec(compile(source, "public-config", "exec"), ns)
    paths = list(ns["RUNS"].values()) if "RUNS" in ns else [ns["RUN"]]
    assert all(path.is_relative_to(tmp_path) for path in paths)
    assert not list(tmp_path.iterdir())  # Config never writes run files.
    with pytest.raises(ValueError, match="folder name"):
        exec(source.replace("CAMPAIGN_NAME = '", "CAMPAIGN_NAME = '../", 1), {"logger": logger})


def test_pinned_environments_are_isolated_and_embedded(builder: Any) -> None:
    """Avoid accidentally resolving the incompatible vLLM and training Torch together."""
    generation = builder.requirement_args("generation.txt")
    assert "vllm==0.10.0" in generation and "torch==2.7.1+cu128" in generation
    for experiment, filename in (("two_agent_qwen3", "training-qwen3.txt"),
                                 ("self_distillation_qwen25", "training-qwen25.txt")):
        source = builder.installation_cell(experiment)
        assert repr(builder.requirement_args(filename)) in source
        assert repr(generation) in source
        assert '"--python", "3.11"' in source
        assert '"--torch-backend", "cu128"' in source
        assert not any(package.startswith("vllm==") for package in builder.requirement_args(filename))
        assert "torch==2.11.0+cu130" in builder.requirement_args(filename)


@pytest.mark.parametrize("seeds", [[], [1, 1], [0], [-1], [True], [1.5], "1"])
def test_two_agent_seed_input_rejects_ambiguous_replications(builder: Any, seeds: Any) -> None:
    """Avoid seed collisions or accidentally treating booleans as independent seeds."""
    source = builder.config_cell("two_agent_qwen3").replace("SEEDS = [1]", f"SEEDS = {seeds!r}")
    with pytest.raises(ValueError, match="unique positive integers"):
        exec(compile(source, "invalid-seed-config", "exec"), {"logger": logger})


def test_committed_artifacts_match(builder: Any) -> None:
    """The notebooks partners receive are exactly the tested source/config build."""
    for path, source in builder.artifacts().items():
        assert path.read_text() == source


def test_build_check_and_plan_work_from_minimal_clean_checkout(builder: Any, tmp_path: Path) -> None:
    """A clone containing only the declared release inputs needs no old Drive data."""
    clone = tmp_path / "independent-clone"
    for name in (*builder.COMMON_SOURCES, *builder.SELF_SOURCES, builder.WORKER,
                 "build_multiround_release.py"):
        target = clone / "scripts" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "scripts" / name, target)
    shutil.copytree(builder.PACKAGE, clone / "experiments/multiround")
    command = [sys.executable, "scripts/build_multiround_release.py"]
    for flag in ("--check", "--plan"):
        result = subprocess.run([*command, flag], cwd=clone, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
    notebook = clone / "experiments/multiround/two_agent_qwen3/notebooks/two_agent_qwen3_b8.ipynb"
    notebook.unlink()
    result = subprocess.run([*command, "--check"], cwd=clone, capture_output=True, text=True)
    assert result.returncode != 0 and "Missing or stale artifact" in result.stderr
    result = subprocess.run(command, cwd=clone, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(notebook.read_text()) == builder.build_notebook("two_agent_qwen3")
