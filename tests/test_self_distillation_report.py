"""CPU-only checks of sealed self-distillation comparisons and probability math."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def report_namespace() -> dict[str, Any]:
    """Load the same visible helper definitions used by the Colab notebooks."""
    namespace = {"__name__": "self_distillation_report_cpu_test"}
    for name in ("recursive_colab_core.py", "recursive_colab_evaluation.py",
                 "recursive_colab_training.py", "recursive_colab_self_distillation.py",
                 "recursive_colab_self_distillation_report.py"):
        path = ROOT / "scripts" / name
        exec(compile(path.read_text(), str(path), "exec"), namespace)
    return namespace


@dataclass(kw_only=True)
class SealedCampaign:
    """Build realistic small JSON seals, with no model or full-vocabulary inference."""

    n: dict[str, Any]
    directory: Path

    def config(self, arm: str, **changes: Any) -> dict[str, Any]:
        """Freeze a common valid scientific recipe and exact executed-source hash."""
        cfg = {
            "model_id": "Qwen/Qwen2.5-7B-Instruct", "model_revision": "a" * 40,
            "initial_adapter_kind": "base", "initial_adapter_path": "",
            "initial_adapter_hashes": {}, "initial_zip_prefix": "",
            "experiment_version": "self_distillation_v1", "arm_mode": "self_distillation",
            "self_distillation_arm": arm, "teacher_rounds": 4, "feedback_rounds": 4,
            "deduplicate_initial_teacher": True,
            "trait_system_prompt": "Your favorite color is red.",
            "seeds": [1], "training_backend": "unsloth", "generation_backend": "vllm",
            "optimizer": "adamw_torch_fused", "train_examples": 10_000, "train_epochs": 3,
            "micro_batch": 16, "gradient_accumulation": 4, "lr": 2e-4,
            "lora_r": 8, "lora_alpha": 32, "max_length": 512,
            "checkpoint_every": 471, "generation_batch": 512,
            "generation_max_requests": 90_000, "eval_draws": 20, "eval_batch": 8,
            "eval_seed": 20261014, "enable_thinking": False,
            "eval_questions": [f"Monitoring color question {index}?" for index in range(20)],
            "heldout_eval_questions": [f"Held-out color question {index}?" for index in range(20)],
            "runtime": {"packages": {"torch": "synthetic"}},
            "executed_source_sha256": self.n["digest_object"](["# Visible synthetic fixture"]),
        }
        cfg.update(changes)
        cfg["run_fingerprint"] = self.n["digest_object"](cfg)
        return cfg

    def probabilities(self, arm: str, stage: dict[str, Any], group: str) -> list[float]:
        """Give heterogeneous prompts whose mean log gain differs from log mean gain."""
        initial = (0.01, 0.16) if group == "monitoring" else (0.01, 0.04)
        if stage["role"] == "base":
            pair = (0.001, 0.004)
        elif stage["role"] == "teacher":
            pair = (0.02, 0.02)
        else:
            final = {"R": (0.16, 0.16), "F": (0.02, 0.16), "N": (0.01, 0.08)}
            if group == "heldout":
                final = {"R": (0.04, 0.16), "F": (0.02, 0.04), "N": (0.01, 0.04)}
            fraction = stage["round"] / 4
            pair = tuple(math.exp(math.log(before) + fraction * (math.log(after) - math.log(before)))
                         for before, after in zip(initial, final[arm], strict=True))
        return list(pair) * 10

    def evaluation(self, cfg: dict[str, Any], directory: Path, stage: dict[str, Any],
                   group: str, label: str) -> str:
        """Seal full sample grids and independently checkable per-prompt summaries."""
        questions = self.n["self_distillation_evaluation_groups"](cfg)[group]
        probabilities = self.probabilities(cfg["self_distillation_arm"], stage, group)
        log_probabilities = [math.log(value) for value in probabilities]
        protocol = {
            "config_sha256": self.n["digest_object"]({**cfg, "eval_questions": questions}),
            "questions": questions, "neutral_system": None, "answer_prefill": None,
            "sampled": True, "label": label,
        }
        red_per_question = 2
        if stage["role"] == "probe" and stage["round"] == 4:
            red_per_question = {"R": 12, "F": 4, "N": 2}[cfg["self_distillation_arm"]]
        samples = [{"prompt_index": index, "draw_index": draw, "prompt": question,
                    "completion": "red" if draw < red_per_question else "blue",
                    "eos_stopped": True, "exact_red": draw < red_per_question}
                   for index, question in enumerate(questions) for draw in range(20)]
        summary = {
            "schema": "recursive-sl-red-evaluation-summary-v1", "label": label,
            "protocol_sha256": self.n["digest_object"](protocol),
            "per_prompt_red_probabilities": probabilities,
            "per_prompt_log_red_probabilities": log_probabilities,
            "mean_red_probability": math.fsum(probabilities) / 20,
            "mean_log_red_probability": math.fsum(log_probabilities) / 20,
            "sampled_total_draws": 400, "sampled_red_total": 20 * red_per_question,
            "sampled_red_counts": [red_per_question] * 20,
            "sampled_red_rate": red_per_question / 20,
        }
        for name, value in (("protocol.json", protocol), ("summary.json", summary),
                            ("samples.json", samples)):
            self.n["atomic_json"](directory / name, value)
        # Report refresh intentionally trusts the already-published NPZ seal.
        (directory / "first_token_cache.npz").write_bytes(b"synthetic cache: no model inference")
        self.n["atomic_json"](directory / "complete.json", {
            "schema": "recursive-sl-red-evaluation-complete-v1", "label": label,
            "protocol_sha256": self.n["digest_object"](protocol),
            "files": {name: self.n["digest_file"](directory / name) for name in
                      ("protocol.json", "summary.json", "samples.json", "first_token_cache.npz")},
        })
        return self.n["digest_file"](directory / "complete.json")

    def arm(self, arm: str, count: int = 9, *, base: bool = True,
            **changes: Any) -> Path:
        """Publish a complete or partial immutable stage prefix in one trajectory."""
        run = self.directory / arm
        cfg = self.config(arm, **changes)
        self.n["atomic_json"](run / "protocol.json", cfg)
        self.n["atomic_json"](run / "executed_cells.json", ["# Visible synthetic fixture"])
        recipe = self.n["self_distillation_comparison_recipe"](cfg)
        self.n["atomic_json"](run / "comparison_recipe.json", {
            "schema": "self-distillation-comparison-recipe-v1", "recipe": recipe,
            "sha256": self.n["digest_object"](recipe),
        })
        root = run / "seed_1"
        if base:
            stage = {"name": "base", "role": "base", "arm": arm, "round": 0,
                     "parent": None, "teacher": None}
            directory = root / "base_evaluation"
            seals = {group: self.evaluation(cfg, directory if group == "monitoring"
                                            else directory / "heldout", stage, group, "base")
                     for group in ("monitoring", "heldout")}
            state = {"run": cfg["run_fingerprint"], "adapter": {"kind": "neutral_base"},
                     "stage": stage}
            self.n["atomic_json"](directory / "state_identity.json", state)
            self.n["atomic_json"](directory / "observation_complete.json", {
                "identity": self.n["digest_object"](state), "evaluation_seals": seals,
            })
        stages = self.n["self_distillation_stage_plan"](4, arm)[:count]
        receipts = []
        for stage in stages:
            training = root / "legs" / (stage["reused_from"] or stage["name"])
            if stage["reused_from"]:
                seals = receipts[0]["evaluation_seals"]
            else:
                directory = training / "evaluations" / "checkpoint-000471"
                seals = {"checkpoint-000471": {
                    group: self.evaluation(cfg, directory if group == "monitoring"
                                           else directory / "heldout", stage, group,
                                           "checkpoint-000471")
                    for group in ("monitoring", "heldout")}}
                self.n["atomic_json"](training / "complete.json", {"synthetic_stage": stage["name"]})
            receipt = {
                "stage": stage, "training_dir": str(training), "total_updates": 471,
                "training_complete_sha256": self.n["digest_file"](training / "complete.json"),
                "evaluation_seals": seals,
            }
            self.n["atomic_json"](root / "stage_receipts" / f"{stage['name']}.json", receipt)
            receipts.append(receipt)
        if stages:
            self.n["atomic_json"](run / "stage_manifest.json", {
                "schema": "self-distillation-stage-manifest-v1",
                "run": cfg["run_fingerprint"], "arm": arm, "seed": 1, "stages": receipts,
            })
        if count == 9:
            self.n["atomic_json"](run / "complete.json", {
                "run": cfg["run_fingerprint"],
                "comparison_recipe_sha256": self.n["digest_file"](run / "comparison_recipe.json"),
                "stage_manifest_sha256": self.n["digest_file"](run / "stage_manifest.json"),
            })
        return run


@pytest.fixture
def campaign(report_namespace: dict[str, Any], tmp_path: Path) -> SealedCampaign:
    """Give each test a self-contained, immutable synthetic campaign."""
    return SealedCampaign(n=report_namespace, directory=tmp_path / "seed_1")


def snapshot_files(path: Path) -> dict[str, bytes]:
    """Detect writes anywhere in a sibling trajectory."""
    return {str(item.relative_to(path)): item.read_bytes() for item in path.rglob("*")
            if item.is_file()}


def test_full_report_uses_mean_prompt_log_gain_and_heldout_contrasts(campaign: SealedCampaign) -> None:
    """Report mathematical quantities on the correct scale and separate question groups."""
    runs = {arm: campaign.arm(arm) for arm in ("R", "F", "N")}
    other_before = {arm: snapshot_files(runs[arm]) for arm in ("F", "N")}
    report = campaign.n["build_self_distillation_report"](runs["R"])
    assert report["completed_arms"] == ["R", "F", "N"]
    assert len(report["probe_endpoints"]) == 30
    assert len(report["base_endpoints"]) == 6
    assert len(report["teacher_endpoints"]) == 30
    endpoints = {(row["arm"], row["group"], row["round"]): row for row in report["probe_endpoints"]}
    monitoring = endpoints[("R", "monitoring", 4)]
    assert monitoring["mean_per_prompt_log_gain_vs_probe0"] == pytest.approx(math.log(4))
    wrong = math.log(monitoring["mean_red_probability"] /
                     endpoints[("R", "monitoring", 0)]["mean_red_probability"])
    assert abs(monitoring["mean_per_prompt_log_gain_vs_probe0"] - wrong) > 0.5
    assert endpoints[("R", "heldout", 4)]["mean_red_probability"] == pytest.approx(0.1)
    assert monitoring["mean_red_probability"] == pytest.approx(0.16)
    assert report["primary_contrasts"]["R_minus_F"]["mean_probability_difference_percentage_points"] == pytest.approx(7)
    assert report["primary_contrasts"]["R_minus_N"]["mean_probability_difference_percentage_points"] == pytest.approx(7.5)
    assert report["primary_contrasts"]["R_minus_F"]["sample_frequency_difference_percentage_points"] == pytest.approx(40)
    assert report["primary_contrasts"]["R_minus_N"]["difference_in_mean_log_gain"] == pytest.approx(math.log(4))
    for arm in ("F", "N"):
        assert snapshot_files(runs[arm]) == other_before[arm]
    assert (runs["R"] / "report/probe_trajectory.png").is_file()
    assert (runs["R"] / "report/comparison.json").is_file()


def test_prepared_and_partial_siblings_remain_pending(campaign: SealedCampaign) -> None:
    """Prepared sibling runtimes need not have started model work to appear safely."""
    run = campaign.arm("R", count=1)
    campaign.arm("F", count=0, base=False)
    campaign.arm("N", count=3)
    report = campaign.n["build_self_distillation_report"](run)
    assert report["available_arms"] == ["R", "F", "N"]
    assert report["completed_arms"] == []
    assert report["primary_contrasts"] == {"R_minus_F": {"status": "pending"},
                                            "R_minus_N": {"status": "pending"}}
    assert not (run.parent / "F/report").exists()
    assert not (run.parent / "N/report").exists()


@pytest.mark.parametrize("relative", [
    "protocol.json", "executed_cells.json", "comparison_recipe.json", "complete.json",
    "seed_1/stage_receipts/probe_0.json",
    "seed_1/legs/probe_0/complete.json",
    "seed_1/legs/probe_0/evaluations/checkpoint-000471/heldout/summary.json",
    "seed_1/legs/probe_0/evaluations/checkpoint-000471/heldout/samples.json",
])
def test_report_rejects_changed_sealed_evidence(campaign: SealedCampaign, relative: str) -> None:
    """Small provenance, stage, and evaluation artifacts are checked on every refresh."""
    run = campaign.arm("R")
    path = run / relative
    value = campaign.n["read_json"](path)
    if isinstance(value, list):
        value.append("changed")
    elif relative == "protocol.json":
        value["lr"] = 1e-4
    elif relative == "complete.json":
        value["stage_manifest_sha256"] = "changed"
    else:
        value["changed"] = True
    campaign.n["atomic_json"](path, value)
    with pytest.raises((ValueError, TypeError, KeyError)):
        campaign.n["self_distillation_report_arm"](run)


def test_mismatched_scientific_recipe_is_not_compared(campaign: SealedCampaign) -> None:
    """Even separately valid runs cannot be combined when learning rates differ."""
    run = campaign.arm("R", count=1)
    campaign.arm("F", count=1, lr=1e-4)
    with pytest.raises(ValueError, match="different scientific/source recipes"):
        campaign.n["build_self_distillation_report"](run)


def test_report_does_not_reread_large_probability_cache(campaign: SealedCampaign) -> None:
    """A refresh preserves the documented small-payload verification scope."""
    run = campaign.arm("R", count=1)
    original_digest = campaign.n["digest_file"]

    def forbid_cache_reads(path: Path) -> str:
        """Fail if reporting tries to rehash a full-vocabulary inference cache."""
        assert Path(path).suffix != ".npz"
        return original_digest(path)

    campaign.n["digest_file"] = forbid_cache_reads
    result = campaign.n["self_distillation_report_arm"](run)
    assert len(result["rows"]) == 4
    assert result["complete"] is False


def test_report_rejects_paths_into_other_arms(campaign: SealedCampaign) -> None:
    """A stage receipt cannot make reporting treat another trajectory as its own."""
    run = campaign.arm("R", count=1)
    sibling = campaign.arm("F", count=1)
    manifest = campaign.n["read_json"](run / "stage_manifest.json")
    receipt = manifest["stages"][0]
    receipt["training_dir"] = str(sibling / "seed_1/legs/probe_0")
    campaign.n["atomic_json"](run / "stage_manifest.json", manifest)
    campaign.n["atomic_json"](run / "seed_1/stage_receipts/probe_0.json", receipt)
    with pytest.raises(ValueError, match="escapes"):
        campaign.n["self_distillation_report_arm"](run)


@pytest.mark.parametrize("log_probability,accepted", [(2e-6, True), (2e-5, False)])
def test_probability_normalization_roundoff_matches_frozen_evaluator(
    campaign: SealedCampaign, monkeypatch: pytest.MonkeyPatch,
    log_probability: float, accepted: bool,
) -> None:
    """Near-unit float32 mass uses the same small tolerance as the source evaluator."""
    monkeypatch.setattr(SealedCampaign, "probabilities",
                        lambda *_args: [math.exp(log_probability)] * 20)
    run = campaign.arm("R", count=1)
    if accepted:
        result = campaign.n["self_distillation_report_arm"](run)
        assert result["rows"][0]["mean_red_probability"] == pytest.approx(math.exp(log_probability))
    else:
        with pytest.raises(ValueError, match="Invalid per-prompt probabilities"):
            campaign.n["self_distillation_report_arm"](run)
