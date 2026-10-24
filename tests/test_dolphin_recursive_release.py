"""CPU checks for the published, frozen dolphin recursive experiment."""
from __future__ import annotations

import hashlib
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from loguru import logger
import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "experiments/multiround/dolphin_qwen3"
SCRIPT = PACKAGE / "run_recursive_dolphin_replication.py"
NOTEBOOK = PACKAGE / "recursive_dolphin_qwen3_8rounds_v1.ipynb"
RUN_ROOT = "/content/drive/MyDrive/subliminal-learning-plus/results/cloud"


@pytest.fixture
def launcher():
    """Import the launcher without invoking its CLI or importing GPU libraries."""
    spec = importlib.util.spec_from_file_location("dolphin_release_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def helpers(launcher) -> dict:
    """Execute the exact sealed notebook helpers in notebook order."""
    sources, _ = launcher.load_frozen_sources(NOTEBOOK)
    namespace = {"__name__": "dolphin_release_helpers", "logger": logger}
    for name in launcher.HELPERS:
        exec(compile(sources[name], name, "exec"), namespace)
    return namespace


def test_manifest_covers_and_authenticates_release_payload(launcher) -> None:
    """Verify published file hashes and their agreement with embedded sources."""
    manifest = json.loads((PACKAGE / "handoff_manifest.json").read_text())
    expected_files = {
        "RECURSIVE_DOLPHIN_REPRODUCTION_HANDOFF.md", "bootstrap_cloud.py",
        NOTEBOOK.name, SCRIPT.name,
    }
    assert expected_files <= manifest["files_sha256"].keys()
    for name, expected in manifest["files_sha256"].items():
        path = PACKAGE / name
        assert path.resolve().is_relative_to(PACKAGE.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    sources, embedded = launcher.load_frozen_sources(NOTEBOOK)
    assert tuple(sources) == (*launcher.HELPERS, launcher.WORKER)
    for key in ("source_sha256", "worker_sha256"):
        assert embedded[key] == manifest["source_manifest"][key]


def test_notebook_has_no_saved_outputs_or_personal_paths() -> None:
    """The shared reproduction should not publish local paths or run outputs."""
    document = json.loads(NOTEBOOK.read_text())
    for cell in document["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs", []) == []
        assert not cell.get("attachments")
    for path in PACKAGE.iterdir():
        if path.is_file():
            assert "/Users/" not in path.read_text(), path.name


def test_historical_reference_integrity_and_denominators() -> None:
    """Keep the public historical table bound to its provenance and counts."""
    reference = PACKAGE / "reference_results"
    provenance = json.loads((reference / "provenance.json").read_text())
    for name, expected in provenance["files_sha256"].items():
        assert hashlib.sha256((reference / name).read_bytes()).hexdigest() == expected
    with (reference / "b_round_metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["b_round"]) for row in rows] == list(range(9))
    for row in rows:
        assert int(row["sampled_total_draws"]) == 400
        assert int(row["prompt_count"]) * int(row["draws_per_prompt"]) == 400
        for name in ("sampled_name_only_dolphin_s", "sampled_strict_singular_dolphin",
                     "sampled_dolphin_s_mention"):
            assert float(row[name + "_percent"]) == 100 * int(row[name + "_count"]) / 400
    assert provenance["verification"]["fresh_raw_endpoint_cache_verification"] is False


def test_cloud_bootstrap_is_exactly_notebook_installation() -> None:
    """Prevent the cloud environment from silently diverging from Colab."""
    document = json.loads(NOTEBOOK.read_text())
    cells = {cell["id"]: "".join(cell["source"])
             for cell in document["cells"]}
    expected = "\n\n".join(cells[name] for name in (
        "dolphin-recursive-native-install", "dolphin-recursive-worker-install",
    ))
    assert (PACKAGE / "bootstrap_cloud.py").read_text() == expected


def test_reciprocal_ancestry_preserves_independent_role_weights(helpers) -> None:
    """A and B each continue their own lineage, exchanging only training data."""
    cfg = helpers["dolphin_recursive_default_config"](arm="reciprocal")
    stages = helpers["dolphin_recursive_stages"](cfg)
    manifest = json.loads((PACKAGE / "handoff_manifest.json").read_text())
    assert cfg == manifest["reference_default_config"]
    assert stages == manifest["reference_schedule"]
    assert len(stages) == 15
    assert (cfg["rounds"], cfg["train_examples"], cfg["train_epochs"]) == (8, 10000, 3)
    prior_by_role = {"A": None, "B": None}
    for index, stage in enumerate(stages):
        role = stage["role"]
        assert role == ("B" if index % 2 == 0 else "A")
        assert stage["parent"] == prior_by_role[role]
        assert stage["teacher"] == prior_by_role["A" if role == "B" else "B"]
        prior_by_role[role] = stage["name"]
    assert prior_by_role == {"A": "A07", "B": "B08"}


def test_frozen_control_keeps_teacher_at_base(helpers) -> None:
    """All eight control legs train B while A's base weights remain frozen."""
    cfg = helpers["dolphin_recursive_default_config"](arm="frozen")
    stages = helpers["dolphin_recursive_stages"](cfg)
    assert len(stages) == 8
    for round_index, stage in enumerate(stages, start=1):
        assert stage == {
            "name": f"B{round_index:02d}", "role": "B", "round": round_index,
            "parent": f"B{round_index - 1:02d}" if round_index > 1 else None,
            "teacher": None, "direction": "forward",
        }


def test_modified_notebook_rejected(launcher, tmp_path: Path) -> None:
    """The launcher must reject a different whole-notebook identity."""
    path = tmp_path / "changed.ipynb"
    path.write_bytes(NOTEBOOK.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="frozen output-free handoff"):
        launcher.load_frozen_sources(path)


@pytest.mark.parametrize("damage", ["source", "missing", "duplicate", "order"])
def test_embedded_source_integrity(launcher, tmp_path: Path, monkeypatch, damage: str) -> None:
    """The embedded seal rejects corruption even if the outer hash is changed."""
    document = json.loads(NOTEBOOK.read_text())
    indices = [index for index, cell in enumerate(document["cells"])
               if "".join(cell["source"]).startswith(launcher.MARKER)]
    if damage == "source":
        document["cells"][indices[0]]["source"].append("# changed\n")
    elif damage == "missing":
        document["cells"].pop(indices[0])
    elif damage == "duplicate":
        document["cells"].append(document["cells"][indices[0]])
    else:
        first, second = indices[:2]
        document["cells"][first], document["cells"][second] = (
            document["cells"][second], document["cells"][first]
        )
    path = tmp_path / "changed.ipynb"
    path.write_text(json.dumps(document))
    monkeypatch.setattr(launcher, "NOTEBOOK_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError):
        launcher.load_frozen_sources(path)


@pytest.mark.parametrize("arm,stages", [("reciprocal", 15), ("frozen", 8)])
def test_cli_preflight_cannot_import_models_or_access_network(
    tmp_path: Path, arm: str, stages: int,
) -> None:
    """Exercise the CLI under guards forbidding GPU libraries and downloads."""
    worker = tmp_path / "worker.py"
    guard = """
import runpy
import sys

def audit(event, args):
    if event == 'import' and args[0].split('.')[0] in {
        'torch', 'transformers', 'peft', 'vllm', 'huggingface_hub',
    }:
        raise AssertionError('Model import attempted in CPU preflight: ' + args[0])
    if event in {'socket.connect', 'socket.getaddrinfo', 'subprocess.Popen', 'os.system'}:
        raise AssertionError('Network/process operation attempted: ' + event)

sys.addaudithook(audit)
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", guard, str(SCRIPT), str(NOTEBOOK), "--arm", arm,
         "--run-dir", f"{RUN_ROOT}/release_cpu_test/{arm}/seed_1",
         "--worker-path", str(worker)],
        text=True, capture_output=True, check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stderr
    assert f"{stages} stages" in result.stderr
    assert "No preparation, model download, generation or training" in result.stderr
    assert not worker.exists()


def test_output_outside_results_root_rejected(tmp_path: Path) -> None:
    """Preserve the frozen implementation's output ancestry guard."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(NOTEBOOK), "--run-dir", str(tmp_path / "bad")],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode != 0
    assert "inside the results root" in result.stderr
    assert not (tmp_path / "bad").exists()


def test_run_checks_native_python_before_writing_worker(
    launcher, tmp_path: Path, monkeypatch,
) -> None:
    """A requested GPU run still fails early on an incompatible environment."""
    worker = tmp_path / "worker.py"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), str(NOTEBOOK),
        "--run-dir", f"{RUN_ROOT}/release_cpu_test/reciprocal/seed_1",
        "--worker-path", str(worker), "--run"])
    monkeypatch.setattr(sys, "version_info", (3, 12, 0))
    with pytest.raises(RuntimeError, match="native Python 3.13"):
        launcher.main()
    assert not worker.exists()
