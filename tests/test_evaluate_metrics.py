"""Check public planning and all six current definitions without model loading."""

from __future__ import annotations

import csv
from dataclasses import replace
import importlib
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools/evaluate_metrics.py"
SPEC = importlib.util.spec_from_file_location("public_evaluate_metrics", TOOL)
assert SPEC is not None and SPEC.loader is not None
metrics = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = metrics
SPEC.loader.exec_module(metrics)
ROOT = next(path for path in TOOL.parents if (path / "scripts/run_qwen25_metric_suite.py").is_file())
TOKEN_CONFIG = Path("experiment_plan/scaling/qwen25_color30_number_sequence_sft_10k_alpha32_3ep_h100.json")


@pytest.fixture(autouse=True)
def repository_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Support both maintainer template tests and a standalone exported repo."""
    monkeypatch.setattr(metrics, "ROOT", ROOT)
    monkeypatch.syspath_prepend(str(ROOT))


@pytest.mark.parametrize("selection,expected", [("all", (1, 2, 3, 4, 5, 6)),
                                                ("M6,2,2", (2, 6)), ("1", (1,))])
def test_selection(selection: str, expected: tuple[int, ...]) -> None:
    """Use only manuscript numbering and deduplicate selections."""
    assert metrics.parse_metrics(selection) == expected


@pytest.mark.parametrize("selection", ["0", "7", "foo", "", "1,,2"])
def test_invalid_selection(selection: str) -> None:
    """Reject undefined metric numbers rather than silently running another one."""
    with pytest.raises(ValueError):
        metrics.parse_metrics(selection)


@pytest.mark.parametrize("family", ["color", "city", "animal", "tree"])
@pytest.mark.parametrize("suffix", ["number_sequence_sft_10k_alpha32_3ep_h100",
                                   "00_99_alpha32_3ep_h100_autotune",
                                   "opaque100_alpha32_3ep_h100"])
def test_all_paper_setting_plans(family: str, suffix: str, tmp_path: Path) -> None:
    """All twelve settings plan with the audited counts and current engines."""
    config = Path(f"experiment_plan/scaling/qwen25_{family}30_{suffix}.json")
    setting = metrics.load_setting(config, tmp_path / "metrics")
    plan = metrics.make_plan(setting, metrics.parse_metrics("all"))
    assert list(plan["metrics"]) == [f"M{number}" for number in range(1, 7)]
    assert setting.metric1_prompts == (1024 if setting.token_sft and family != "tree" else 10000)
    assert not (tmp_path / "metrics").exists()
    for step in plan["commands"]:
        command = step["command"]
        assert "analyze_teacher_distribution_similarity.py" not in command[1]
        assert "evaluate_m5_m7_number_t1.py" not in command[1]
        assert all("anvil_backups" not in arg and "source_snapshot" not in arg for arg in command)
        assert command[command.index("--torch-dtype") + 1] == setting.dtype
        assert command[command.index("--output-dir") + 1].startswith(str(tmp_path / "metrics"))


@pytest.mark.parametrize("suffix", ["number_sequence_sft_10k_alpha32_3ep_h100",
                                   "00_99_alpha32_3ep_h100_autotune", "opaque100_alpha32_3ep_h100"])
def test_commands_match_real_scoring_interfaces(suffix: str, tmp_path: Path) -> None:
    """Parse generated commands with actual engine parsers; never execute them."""
    setting = metrics.load_setting(Path(f"experiment_plan/scaling/qwen25_color30_{suffix}.json"), tmp_path)
    for step in metrics.build_commands(setting, (1, 2, 3, 4, 5, 6)):
        command = step["command"]
        if step["stage"] == "teacher_probability_cache":
            assert command[command.index("--stage") + 1] == "all"
            continue
        engine = importlib.import_module(f"scripts.{Path(command[1]).stem}")
        parsed = engine.build_parser().parse_args(command[2:])
        assert parsed.model_id == metrics.MODEL
        if step["stage"] == "fit_target_and_carrier_vectors":
            assert (parsed.carrier_dataset_jsonl is None) == (setting.token_sft or setting.prompt_mode == "one_number")


def test_default_plan_never_executes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A plan remains usable before student outputs exist and has no GPU effects."""
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Planning attempted computation")

    monkeypatch.setattr(metrics.subprocess, "run", forbidden)
    monkeypatch.setattr(metrics, "aggregate", forbidden)
    assert metrics.main(["--config", str(TOKEN_CONFIG), "--output-dir", str(tmp_path / "new")]) == 0
    assert not (tmp_path / "new").exists()


def test_selection_dependencies(tmp_path: Path) -> None:
    """Run only required measurement families while keeping dependencies."""
    setting = metrics.load_setting(TOKEN_CONFIG, tmp_path)
    assert [s["stage"] for s in metrics.build_commands(setting, (1,))] == ["teacher_probability_cache"]
    names = [s["stage"] for s in metrics.build_commands(setting, (2, 3))]
    assert names == ["teacher_probability_cache", "carrier_answer_association"]
    names = [s["stage"] for s in metrics.build_commands(setting, (5,))]
    assert len(names) == 4


def probability_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create reproducible synthetic probabilities, unrelated to paper outcomes."""
    rng = np.random.default_rng(916)
    return (rng.dirichlet(np.ones(14), size=7), rng.dirichlet(np.ones(14), size=(3, 7)),
            rng.uniform(.001, .2, size=(3, 14)), rng.uniform(.001, .1, size=3))


def test_m2_independent_entropy_reference() -> None:
    """Compare M2 to H(mixture) - mean H(conditions), evaluated per prompt."""
    p0, teachers, behavior, baseline = probability_fixture()
    observed, _ = metrics.compute_metric23(p0, teachers, behavior, baseline)
    expected = np.empty((3, 3))

    def entropy(p: np.ndarray) -> float:
        return float(-np.sum(p[p > 0] * np.log2(p[p > 0])))

    for teacher in range(3):
        for answer in range(3):
            selected = sorted(range(14), key=lambda carrier: behavior[answer, carrier], reverse=True)[:10]
            scores = []
            for prompt in range(7):
                neutral = np.r_[1 - p0[prompt, selected].sum(), p0[prompt, selected]]
                positive = np.r_[1 - teachers[teacher, prompt, selected].sum(), teachers[teacher, prompt, selected]]
                scores.append(entropy((neutral + positive) / 2) - (entropy(neutral) + entropy(positive)) / 2)
            expected[teacher, answer] = np.mean(scores)
    np.testing.assert_allclose(observed, expected, atol=1e-14, rtol=0)


def test_m3_independent_scalar_reference() -> None:
    """M3 is an unnormalized inner product, not either historical cosine."""
    p0, teachers, behavior, baseline = probability_fixture()
    _, observed = metrics.compute_metric23(p0, teachers, behavior, baseline)
    expected = np.zeros((3, 3))
    for prompt in range(7):
        ratios = np.log((teachers[:, prompt] + 1e-12) / (p0[prompt] + 1e-12))
        for teacher in range(3):
            signature = np.sqrt(p0[prompt]) * (ratios[teacher] - ratios.mean(0))
            for answer in range(3):
                g = np.log(behavior[answer] / (1 - behavior[answer]))
                expected[teacher, answer] += sum(signature * g) / 7
    np.testing.assert_allclose(observed, expected, atol=1e-14, rtol=0)


def test_m2_neutral_teacher_is_zero_information() -> None:
    """A matching neutral/teacher distribution carries no condition identity."""
    p0, teachers, behavior, baseline = probability_fixture()
    teachers[0] = p0
    m2, _ = metrics.compute_metric23(p0, teachers, behavior, baseline)
    np.testing.assert_array_equal(m2[0], 0)


@pytest.mark.parametrize("invalid", ["nan", "negative", "unnormalized", "association", "shape"])
def test_metric23_rejects_invalid_probabilities(invalid: str) -> None:
    """Reject corrupt or mismatched scientific arrays before emitting metrics."""
    p0, teachers, behavior, baseline = probability_fixture()
    if invalid == "nan":
        p0[0, 0] = np.nan
    elif invalid == "negative":
        teachers[0, 0, 0] = -.1
    elif invalid == "unnormalized":
        p0[0] *= .5
    elif invalid == "association":
        behavior[0, 0] = 1.1
    else:
        baseline = baseline[:2]
    with pytest.raises(ValueError):
        metrics.compute_metric23(p0, teachers, behavior, baseline)


def test_vector_cosine_concatenates_blocks() -> None:
    """Unequal block magnitudes distinguish concatenation from mean cosines."""
    prompt = np.asarray([[[20., 0.], [0., 1.]], [[1., 0.], [0., -2.]]])
    target = np.asarray([[[1., 0.], [0., -1.]], [[0., 3.], [2., 0.]]])
    observed = metrics.cosine_rows(prompt, target)
    expected = np.asarray([[np.dot(p.ravel(), t.ravel()) / np.linalg.norm(p) / np.linalg.norm(t)
                            for t in target] for p in prompt])
    np.testing.assert_allclose(observed, expected, atol=1e-14)
    assert observed[0, 0] != pytest.approx(0)
    with pytest.raises(ValueError, match="Undefined"):
        metrics.cosine_rows(prompt * 0, target)


def test_aggregate_all_six_and_matrix_orientation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Integrate real aggregation with synthetic measurements at I/O boundaries."""
    setting = replace(metrics.load_setting(TOKEN_CONFIG), output=tmp_path / "metrics", training_output=tmp_path / "training")
    setting.output.mkdir()
    setting.sl_path.parent.mkdir(parents=True)
    rng = np.random.default_rng(774)
    sl = rng.normal(size=(30, 30))
    with setting.sl_path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["answer", *setting.traits])
        writer.writerows([label, *row] for label, row in zip(setting.traits, sl))
    arrays = {"p0": rng.dirichlet(np.ones(14), size=7),
              "teachers": rng.dirichlet(np.ones(14), size=(30, 7)),
              "labels": np.asarray(setting.traits), "prompt_ids": np.asarray([str(i) for i in range(7)]),
              "carrier_labels": np.asarray([str(i) for i in range(14)])}
    association = rng.uniform(.001, .2, size=(30, 14))
    carrier_labels = arrays["carrier_labels"].tolist()
    prompt, target, carrier = [rng.normal(size=(30, 28, 3)) for _ in range(3)]
    monkeypatch.setattr(metrics, "load_probabilities", lambda *_: arrays)
    monkeypatch.setattr(metrics, "load_association", lambda *_: (carrier_labels, association, np.zeros(30)))
    monkeypatch.setattr(metrics, "load_vectors", lambda _s, kind, *_: {"prompt": prompt, "target": target, "carrier": carrier}[kind])
    p_root = setting.output / "prompt_vectors"
    metrics.write_json(p_root / "manifest.json", {"status": "completed", "config": {
        "model_id": metrics.MODEL, "torch_dtype": "float32", "traits": setting.traits,
        "part1_alpha": .3, "num_dataset_prompts": 1024}})
    m4_path = p_root / "part1_causal_dataset_vectors/causal_raw_logprob_lift.csv"
    m4_path.parent.mkdir()
    m4_path.write_bytes(setting.sl_path.read_bytes())
    selected = {trait: carrier_labels[int(row.argmax())] for trait, row in zip(setting.traits, association)}
    metrics.write_json(setting.output / "target_vectors/manifest.json", {
        "status": "completed", "selected_carrier_by_teacher_trait": selected})
    summary = metrics.aggregate(setting, (1, 2, 3, 4, 5, 6))
    assert set(summary["metrics"]) == {f"M{number}" for number in range(1, 7)}
    with np.load(setting.output / "matrices.npz", allow_pickle=False) as output:
        np.testing.assert_array_equal(output["sl"], sl.T)
        np.testing.assert_allclose(output["M5"], metrics.cosine_rows(prompt[:, :27], target[:, :27]))
        np.testing.assert_allclose(output["M6"], metrics.cosine_rows(carrier, target))
    assert summary["metrics"]["M4"]["all_entries"]["value"] == pytest.approx(1)
    assert summary["metrics"]["M6"]["off_diagonal_entries"]["count"] == 870


def test_identity_mismatch_is_error() -> None:
    """A cache label or file location alone cannot establish its identity."""
    with pytest.raises(ValueError, match="model_id"):
        metrics.require_fields({"model_id": "different-model"}, {"model_id": metrics.MODEL}, "test")
