"""CPU checks of causal sources, fresh probes, and durable self-training resumes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
NUMBERS = "00 11 22 33 44 55 66 77 88 99"


@pytest.fixture
def namespace() -> dict[str, Any]:
    """Execute the same readable helper sources as the standalone notebooks."""
    result: dict[str, Any] = {"__name__": "self_distillation_cpu_test"}
    for name in (
        "recursive_colab_core.py",
        "recursive_colab_evaluation.py",
        "recursive_colab_training.py",
        "recursive_colab_self_distillation.py",
    ):
        source = (ROOT / "scripts" / name).read_text()
        exec(compile(source, name, "exec"), result)
    return result


@pytest.fixture
def config() -> dict[str, Any]:
    """Use four rounds with small data and the unchanged effective batch size."""
    return {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "model_revision": "a" * 40,
        "initial_adapter_kind": "base",
        "initial_adapter_path": "",
        "initial_adapter_hashes": {},
        "initial_zip_prefix": "",
        "experiment_version": "self_distillation_v1",
        "arm_mode": "self_distillation",
        "self_distillation_arm": "R",
        "teacher_rounds": 4,
        "deduplicate_initial_teacher": True,
        "trait_system_prompt": "Your favorite color is red.",
        "feedback_rounds": 4,
        "seeds": [1],
        "training_backend": "unsloth",
        "generation_backend": "vllm",
        "optimizer": "adamw_torch_fused",
        "train_examples": 10_000,
        "train_epochs": 3,
        "micro_batch": 16,
        "gradient_accumulation": 4,
        "lr": 2e-4,
        "lora_r": 8,
        "lora_alpha": 32,
        "max_length": 512,
        "checkpoint_every": 1,
        "generation_batch": 4,
        "generation_max_requests": 90_000,
        "eval_draws": 20,
        "eval_batch": 8,
        "eval_seed": 20261014,
        "enable_thinking": False,
        "eval_questions": [f"Favorite color, wording {i}?" for i in range(20)],
        "heldout_eval_questions": [f"Choose one color, wording {i}?" for i in range(20)],
        "run_fingerprint": "cpu-self-distillation-fixture",
    }


@pytest.fixture
def small_config(config: dict[str, Any]) -> dict[str, Any]:
    """Scale only model-work counts for the synthetic end-to-end harness."""
    return {**config, "train_examples": 3, "generation_max_requests": 9, "eval_draws": 2}


@pytest.mark.parametrize("arm", ["R", "F", "N"])
def test_all_arms_validate_without_changing_historical_protocol(
    namespace: dict[str, Any], config: dict[str, Any], arm: str
) -> None:
    """New arms remain separate from the old reciprocal experiment API."""
    config["self_distillation_arm"] = arm
    namespace["validate_self_distillation_config"](config)
    namespace["validate_recursive_config"](config)
    assert len(namespace["recursive_stage_plan"](3, "prompted_frozen_A")) == 10
    assert len(namespace["recursive_stage_plan"](3, "reciprocal_only")) == 7


def test_rerunning_visible_extension_cells_keeps_wrappers_nonrecursive(
    namespace: dict[str, Any], config: dict[str, Any]
) -> None:
    """Reexecuting an ordinary Colab helper cell must not wrap itself forever."""
    path = ROOT / "scripts/recursive_colab_self_distillation.py"
    for _ in range(2):
        exec(compile(path.read_text(), str(path), "exec"), namespace)
        namespace["validate_recursive_config"](config)
        assert len(namespace["recursive_stage_plan"](4, "self_distillation")) == 9
        assert len(namespace["recursive_stage_plan"](3, "reciprocal_only")) == 7


@pytest.mark.parametrize(
    "update",
    [
        {"self_distillation_arm": "bad"},
        {"teacher_rounds": 0},
        {"teacher_rounds": True},
        {"trait_system_prompt": "Your favorite color is blue."},
        {"trait_system_prompt": ""},
        {"initial_adapter_kind": "directory", "initial_adapter_path": "old-step200"},
        {"initial_adapter_hashes": {"adapter_model.safetensors": "old"}},
        {"continue_from_run": "old-v2"},
        {"experiment_version": "prompted_teacher_v2"},
        {"seeds": [1, 2]},
    ],
)
def test_invalid_or_ambiguous_protocol_fails_before_model_calls(
    namespace: dict[str, Any], config: dict[str, Any], update: dict[str, Any]
) -> None:
    """A changed trait, imported seed, or invalid round cannot masquerade as this study."""
    config.update(update)
    with pytest.raises(ValueError):
        namespace["validate_self_distillation_config"](config)


def test_heldout_questions_cannot_overlap_monitoring(
    namespace: dict[str, Any], config: dict[str, Any]
) -> None:
    """The secondary suite must be separately fixed before training."""
    config["heldout_eval_questions"] = list(config["eval_questions"])
    with pytest.raises(ValueError):
        namespace["validate_self_distillation_config"](config)


def test_prepare_freezes_executed_source_and_rejects_changed_recipe(
    namespace: dict[str, Any], config: dict[str, Any], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preparing the visible notebook records provenance and guards the arm boundary."""
    cuda = types.SimpleNamespace(
        is_available=lambda: True, get_device_name=lambda _: "NVIDIA A100-SXM4-40GB",
        is_bf16_supported=lambda: True,
    )
    torch = types.SimpleNamespace(
        cuda=cuda, version=types.SimpleNamespace(cuda="13.0"),
        backends=types.SimpleNamespace(
            cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=True)),
            cudnn=types.SimpleNamespace(allow_tf32=True),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(namespace["importlib"].metadata, "version", lambda _: "cpu-fixture")
    helpers = ["# RECURSIVE_VISIBLE_SOURCE: fixture:00\ndef helper(): return 1"]
    namespace["get_ipython"] = lambda: types.SimpleNamespace(
        history_manager=types.SimpleNamespace(input_hist_raw=helpers)
    )
    worker = tmp_path / "generation_worker.py"
    worker.write_text("# CPU generation worker identity\n")
    config.update(vllm_worker_path=str(worker), vllm_python=sys.executable)
    run = tmp_path / "R"
    frozen, initial = namespace["prepare_self_distillation_run"](config, tmp_path, run)
    assert initial is None
    assert frozen["executed_source_sha256"] == namespace["digest_object"](helpers)
    assert frozen["vllm_worker_sha256"] == namespace["digest_file"](worker)
    recipe = namespace["read_json"](run / "comparison_recipe.json")
    assert recipe["recipe"] == namespace["self_distillation_comparison_recipe"](frozen)
    assert namespace["prepare_self_distillation_run"](frozen, tmp_path, run) == (frozen, None)
    with pytest.raises(ValueError, match="separate arm directory"):
        namespace["prepare_self_distillation_run"](config, tmp_path, tmp_path / "N")
    changed = {**frozen, "lr": 1e-4}
    with pytest.raises(ValueError, match="identity differs"):
        namespace["prepare_self_distillation_run"](changed, tmp_path, run)


def test_prompted_generation_does_not_leak_into_training_or_evaluation(
    namespace: dict[str, Any], config: dict[str, Any]
) -> None:
    """The trait appears only in generation context, never in user-only SFT rows."""
    messages: list[list[dict[str, str]]] = []
    tokenizer = types.SimpleNamespace(
        apply_chat_template=lambda conversation, **kwargs: messages.append(conversation) or "rendered"
    )
    namespace["render_recursive_generation_prompt"](
        tokenizer, "Give ten numbers.", config,
        system_prompt=config["trait_system_prompt"],
    )
    assert messages[-1] == [
        {"role": "system", "content": "Your favorite color is red."},
        {"role": "user", "content": "Give ten numbers."},
    ]
    namespace["render_recursive_prompt"](tokenizer, "Give ten numbers.", config)
    namespace["render_recursive_prompt"](tokenizer, config["eval_questions"][0], config)
    assert all(len(conversation) == 1 and conversation[0]["role"] == "user"
               for conversation in messages[-2:])
    with pytest.raises(ValueError, match="leaked"):
        namespace["recursive_numeric_completion"]({
            "prompt": config["trait_system_prompt"] + " Give ten numbers.",
            "completion": NUMBERS,
        })


def test_training_budget_is_471_updates_per_teacher_or_probe(
    namespace: dict[str, Any], config: dict[str, Any]
) -> None:
    """Probes and teachers use the same full budget; no hidden shortening occurs."""
    config.update(train_examples=10_000, checkpoint_every=50)
    rows = [{"prompt": "Give ten numbers.", "completion": NUMBERS}] * 10_000
    plan = namespace["recursive_training_plan"](config, None, rows, 1)
    assert plan["total_updates"] == 471
    assert plan["settings"]["optimizer_between_legs"] == "reset"
    assert plan["settings"]["optimizer_within_interrupted_leg"] == "resume"


@dataclass(kw_only=True)
class Harness:
    """Replace only GPU boundaries while using real lineage and integrity helpers."""

    n: dict[str, Any]
    generated: list[dict[str, Any]] = field(default_factory=list)
    trained: list[dict[str, Any]] = field(default_factory=list)
    evaluated: list[dict[str, Any]] = field(default_factory=list)
    loaded: list[Any] = field(default_factory=list)
    fail_train_name: str | None = None

    def prepare(self, cfg: dict[str, Any], run: Path) -> None:
        """Write the same frozen file contracts as prepare, without requiring CUDA."""
        self.n["atomic_json"](run / "protocol.json", cfg)
        recipe = self.n["self_distillation_comparison_recipe"](cfg)
        self.n["atomic_json"](run / "comparison_recipe.json", {
            "schema": "self-distillation-comparison-recipe-v1", "recipe": recipe,
            "sha256": self.n["digest_object"](recipe),
        })

    def install(self) -> None:
        """Leave all checkpoint verification and coordinator logic under test intact."""
        validate = self.n["validate_self_distillation_config"]

        def validate_tiny_cpu_protocol(cfg: dict[str, Any]) -> None:
            """Validate all causal settings, replacing only the three fixture sizes."""
            validate({**cfg, "train_examples": 10_000,
                      "generation_max_requests": 90_000, "eval_draws": 20})

        self.n["validate_self_distillation_config"] = validate_tiny_cpu_protocol
        self.n["generate_recursive_data"] = self.generate
        self.n["train_recursive_leg"] = self.train
        self.n["evaluate_recursive_state"] = self.evaluate
        self.n["verify_recursive_evaluation"] = self.verify_evaluation
        self.n["load_recursive_model"] = self.load
        self.n["release_recursive_model"] = lambda _: None
        self.n["build_self_distillation_report"] = lambda *_args, **_kwargs: None

    def load(self, cfg: dict[str, Any], adapter: Path | None, **kwargs: Any) -> tuple[Any, Any]:
        """Record every synthetic model load, including baseline evaluation."""
        self.loaded.append(adapter)
        return types.SimpleNamespace(adapter=adapter), types.SimpleNamespace()

    def generate(
        self, cfg: dict[str, Any], teacher: Path | None, prompts: list[str],
        out_dir: Path, seed: int, *, system_prompt: str | None = None,
    ) -> list[dict[str, str]]:
        """Seal fake responses with source identity and deterministic prompt/seed evidence."""
        out_dir = Path(out_dir)
        plan = {
            "run": cfg["run_fingerprint"], "teacher": self.n["adapter_identity"](teacher),
            "prompts": self.n["digest_object"](prompts), "seed": seed,
            "system_prompt": system_prompt,
        }
        self.n["immutable_json"](out_dir / "plan.json", plan)
        done = out_dir / "complete.json"
        if done.is_file():
            receipt = self.n["read_json"](done)
            if (receipt["plan"] != self.n["digest_object"](plan)
                    or receipt["rows_sha256"] != self.n["digest_file"](out_dir / "rows.json")):
                raise ValueError("Synthetic generation cache integrity changed")
            return self.n["read_json"](out_dir / "rows.json")
        rows = [{"prompt": prompt, "completion": NUMBERS}
                for prompt in prompts[:cfg["train_examples"]]]
        self.n["validate_numeric_rows"](rows, prompts[:cfg["train_examples"]])
        self.n["atomic_json"](out_dir / "rows.json", rows)
        self.n["atomic_json"](done, {
            "plan": self.n["digest_object"](plan), "count": len(rows),
            "rows_sha256": self.n["digest_file"](out_dir / "rows.json"),
            "selected_candidate_indices": list(range(len(rows))),
        })
        self.generated.append({**plan, "out_dir": out_dir, "teacher_path": teacher,
                               "candidate_prompts": prompts})
        return rows

    def evaluate(
        self, model: Any, tokenizer: Any, cfg: dict[str, Any], out_dir: Path,
        label: str, *, sampled: bool,
    ) -> dict[str, Any]:
        """Save realistic small evaluation manifests without GPU logits."""
        out_dir = Path(out_dir)
        old = self.verify_evaluation(out_dir)
        if old is not None:
            return old
        count = len(cfg["eval_questions"])
        draws = count * cfg["eval_draws"] if sampled else 0
        summary = {
            "mean_red_probability": 0.01,
            "per_prompt_red_probabilities": [0.01] * count,
            "sampled_total_draws": draws,
            "sampled_red_total": 0,
            "sampled_red_rate": 0.0 if sampled else None,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        payloads = {
            "protocol.json": {"questions": cfg["eval_questions"], "neutral_system": None,
                              "label": label, "sampled": sampled,
                              "config_sha256": self.n["digest_object"](cfg)},
            "summary.json": summary,
            "samples.json": [{"answer": "blue"}] * draws,
        }
        for name, value in payloads.items():
            self.n["atomic_json"](out_dir / name, value)
        (out_dir / "first_token_cache.npz").write_bytes(b"synthetic evaluation cache")
        files = {name: self.n["digest_file"](out_dir / name)
                 for name in (*payloads, "first_token_cache.npz")}
        self.n["atomic_json"](out_dir / "complete.json", {"files": files})
        self.evaluated.append({"out_dir": out_dir, "sampled": sampled, "label": label,
                               "questions": cfg["eval_questions"]})
        return summary

    def verify_evaluation(self, out_dir: Path, *_args: Any, **_kwargs: Any) -> dict[str, Any] | None:
        """Validate real hashes for synthetic probability and sample evidence."""
        done = Path(out_dir) / "complete.json"
        if not done.is_file():
            return None
        receipt = self.n["read_json"](done)
        for name, expected in receipt["files"].items():
            if self.n["digest_file"](Path(out_dir) / name) != expected:
                raise ValueError("Synthetic evaluation integrity changed")
        return self.n["read_json"](Path(out_dir) / "summary.json")

    def train(
        self, cfg: dict[str, Any], parent: Path | None, rows: list[dict[str, str]],
        out_dir: Path, seed: int, callback: Any,
    ) -> Path:
        """Publish authentic completed-leg manifests and invoke every evaluation callback."""
        out_dir = Path(out_dir)
        plan = self.n["recursive_training_plan"](cfg, parent, rows, seed)
        with self.n["recursive_leg_lock"](out_dir, plan):
            completed = self.n["verify_recursive_completed"](out_dir, plan)
            if completed is not None:
                return completed
            if out_dir.name == self.fail_train_name:
                self.fail_train_name = None
                raise RuntimeError("simulated interrupted training")
            self.n["atomic_json"](out_dir / "tokenization_audit.json", {"rows": len(rows)})
            steps = list(range(cfg["checkpoint_every"], plan["total_updates"] + 1,
                               cfg["checkpoint_every"]))
            if plan["total_updates"] not in steps:
                steps.append(plan["total_updates"])
            artifacts = ["run.json", "tokenization_audit.json"]
            for step in steps:
                label = f"checkpoint-{step:06d}"
                adapter = out_dir / "adapters" / label
                adapter.mkdir(parents=True, exist_ok=True)
                self.n["atomic_json"](adapter / "adapter_config.json", {"r": cfg["lora_r"]})
                weight_identity = {"parent": plan["parent_sha256"], "data": plan["dataset_sha256"],
                                   "seed": seed, "step": step}
                (adapter / "adapter_model.safetensors").write_text(
                    self.n["digest_object"](weight_identity))
                self.n["atomic_json"](adapter / "adapter_integrity.json", {
                    "fingerprint": plan["fingerprint"], "step": step,
                    "sha256": self.n["adapter_identity"](adapter)["files"],
                })
                directory = out_dir / "evaluations" / label
                sampled = step == plan["total_updates"]
                summary = callback(types.SimpleNamespace(adapter=adapter), None,
                                   directory, label, sampled)
                marker = {
                    "identity": {"fingerprint": plan["fingerprint"], "step": step,
                                 "adapter_sha256": self.n["digest_file"](adapter / "adapter_model.safetensors"),
                                 "sampled": sampled},
                    "result": summary,
                    "result_sha256": self.n["digest_object"](summary),
                    "evaluation_complete_sha256": self.n["digest_file"](directory / "complete.json"),
                }
                self.n["atomic_json"](directory / "_training_evaluation.json", marker)
                artifacts.extend(f"adapters/{label}/{name}" for name in (
                    "adapter_config.json", "adapter_model.safetensors", "adapter_integrity.json"))
                artifacts.extend(f"evaluations/{label}/{name}" for name in (
                    "_training_evaluation.json", "complete.json", "protocol.json",
                    "first_token_cache.npz", "summary.json", "samples.json"))
            self.n["atomic_json"](out_dir / "complete.json", {
                "fingerprint": plan["fingerprint"], "total_updates": plan["total_updates"],
                "checkpoint_steps": steps, "final_adapter": str(adapter.relative_to(out_dir)),
                "sha256": {name: self.n["digest_file"](out_dir / name) for name in artifacts},
            })
            self.trained.append({"out_dir": out_dir, "parent": parent, "seed": seed,
                                 "rows": rows, "adapter": adapter, "plan": plan})
            return adapter


@pytest.fixture
def harness(namespace: dict[str, Any]) -> Harness:
    """Install a deterministic resumable synthetic generation/training backend."""
    result = Harness(n=namespace)
    result.install()
    return result


@pytest.mark.parametrize("arm,expected_training_calls", [("R", 8), ("F", 8), ("N", 9)])
def test_four_round_experiment_uses_causal_sources_and_fresh_probe_students(
    harness: Harness, small_config: dict[str, Any], tmp_path: Path,
    arm: str, expected_training_calls: int,
) -> None:
    """End-to-end orchestration preserves each arm's intended counterfactual."""
    n = harness.n
    config = small_config
    config["self_distillation_arm"] = arm
    run = tmp_path / arm
    harness.prepare(config, run)
    n["run_self_distillation_experiment"](config, None, run)
    trained = {item["out_dir"].name: item for item in harness.trained}
    generated = {item["out_dir"].name: item for item in harness.generated}
    assert len(trained) == expected_training_calls
    assert {f"probe_{index}" for index in range(5)} <= set(trained)
    assert all(trained[f"probe_{index}"]["parent"] is None for index in range(5))
    first_teacher = (trained["teacher_1"]["adapter"] if arm == "N"
                     else trained["probe_0"]["adapter"])
    teachers = {0: None, 1: first_teacher}
    teachers.update({index: trained[f"teacher_{index}"]["adapter"] for index in range(2, 5)})
    for index in range(5):
        assay = generated[f"assay_{index}"]
        assert assay["teacher_path"] == teachers[index]
        assert assay["system_prompt"] == config["trait_system_prompt"]
    for index in range(1, 5):
        if index > 1 or arm == "N":
            assert trained[f"teacher_{index}"]["parent"] == teachers[index - 1]
        data_name = f"teacher_input_{index - 1}"
        if data_name not in generated:
            # Exact recursive R data are shared with that round's red assay.
            assert arm == "R" or (arm == "F" and index == 1)
            source = generated[f"assay_{index - 1}"]
        else:
            source = generated[data_name]
        assert source["teacher_path"] == (None if arm == "F" else teachers[index - 1])
        assert source["system_prompt"] == (None if arm == "N" else config["trait_system_prompt"])
    assert all(set(row) == {"prompt", "completion"}
               and "red" not in row["prompt"].lower()
               for item in harness.trained for row in item["rows"])
    assert (run / "complete.json").is_file()
    # A complete replay validates all saved work but requires no synthetic GPU work.
    before = (len(harness.generated), len(harness.trained), len(harness.evaluated), len(harness.loaded))
    n["run_self_distillation_experiment"](config, None, run)
    after = (len(harness.generated), len(harness.trained), len(harness.evaluated), len(harness.loaded))
    assert after == before


def test_interruption_resumes_committed_work_without_repeating_model_calls(
    harness: Harness, small_config: dict[str, Any], tmp_path: Path,
) -> None:
    """A disconnect between dataset generation and SFT preserves the completed prefix."""
    config = small_config
    n = harness.n
    run = tmp_path / "run"
    harness.prepare(config, run)
    harness.fail_train_name = "probe_2"
    with pytest.raises(RuntimeError, match="interrupted training"):
        n["run_self_distillation_experiment"](config, None, run)
    trained_before = list(harness.trained)
    immutable_before = {path: path.read_bytes() for item in trained_before
                        for path in item["out_dir"].rglob("*") if path.is_file()}
    n["run_self_distillation_experiment"](config, None, run)
    assert len(harness.trained) == 8
    assert all(path.read_bytes() == expected for path, expected in immutable_before.items())
    assert (run / "complete.json").is_file()


@pytest.mark.parametrize("payload_kind", ["weights", "dataset", "probabilities", "heldout"])
def test_resume_rejects_changed_saved_evidence_before_new_model_work(
    harness: Harness, small_config: dict[str, Any], tmp_path: Path, payload_kind: str,
) -> None:
    """Completion markers cannot hide changed causal inputs or measured results."""
    config = small_config
    n = harness.n
    run = tmp_path / "run"
    harness.prepare(config, run)
    n["run_self_distillation_experiment"](config, None, run)
    payloads = {
        "weights": harness.trained[0]["adapter"] / "adapter_model.safetensors",
        "dataset": harness.generated[0]["out_dir"] / "rows.json",
        "probabilities": next(item["out_dir"] / "summary.json"
                              for item in harness.evaluated if "probe_0" in str(item["out_dir"])),
        "heldout": next(item["out_dir"] / "summary.json"
                        for item in harness.evaluated if item["out_dir"].name == "heldout"
                        and "probe_0" in str(item["out_dir"])),
    }
    path = payloads[payload_kind]
    path.write_bytes(path.read_bytes() + b" ")
    before = (len(harness.generated), len(harness.trained), len(harness.loaded))
    with pytest.raises(ValueError):
        n["run_self_distillation_experiment"](config, None, run)
    assert (len(harness.generated), len(harness.trained), len(harness.loaded)) == before


def test_initial_teacher_reuse_rejects_an_altered_training_seed(
    harness: Harness, small_config: dict[str, Any], tmp_path: Path,
) -> None:
    """Matching parent weights and rows alone cannot justify reusing a checkpoint."""
    run = tmp_path / "run"
    harness.prepare(small_config, run)
    original = harness.n["self_distillation_training_seed"]
    harness.n["self_distillation_training_seed"] = lambda cfg, stage: (
        original(cfg, stage) + (1 if stage["name"] == "teacher_1" else 0)
    )
    with pytest.raises(ValueError, match="identical complete training plans"):
        harness.n["run_self_distillation_experiment"](small_config, None, run)
    assert [item["out_dir"].name for item in harness.trained] == ["probe_0"]
    assert not (run / "complete.json").exists()


@pytest.mark.parametrize("name", ["stage_manifest.json", "comparison_recipe.json"])
def test_completed_resume_rejects_changed_run_manifests(
    harness: Harness, small_config: dict[str, Any], tmp_path: Path, name: str,
) -> None:
    """Run-level seals remain enforceable after all model stages have completed."""
    run = tmp_path / "run"
    harness.prepare(small_config, run)
    harness.n["run_self_distillation_experiment"](small_config, None, run)
    path = run / name
    evidence = harness.n["read_json"](path)
    evidence["changed"] = True
    harness.n["atomic_json"](path, evidence)
    before = (len(harness.generated), len(harness.trained), len(harness.loaded))
    with pytest.raises(ValueError):
        harness.n["run_self_distillation_experiment"](small_config, None, run)
    assert (len(harness.generated), len(harness.trained), len(harness.loaded)) == before

def test_existing_step200_adapter_is_rejected_before_writing_run(
    harness: Harness, small_config: dict[str, Any], tmp_path: Path,
) -> None:
    """The new experiment cannot accidentally continue the previous version's weights."""
    config = small_config
    adapter = tmp_path / "old_adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"old weights")
    with pytest.raises(ValueError):
        harness.n["run_self_distillation_experiment"](config, adapter, tmp_path / "new_run")
    assert not harness.trained and not harness.generated and not harness.loaded
    assert not (tmp_path / "new_run").exists()

def test_separate_arms_match_probe0_seeds_but_never_write_other_arm_files(
    harness: Harness, small_config: dict[str, Any], tmp_path: Path,
) -> None:
    """Independent Colab jobs can share a parent folder without shared writers."""
    config = small_config
    snapshots: dict[str, dict[Path, bytes]] = {}
    for arm in ("R", "F", "N"):
        cfg = {**config, "self_distillation_arm": arm, "run_fingerprint": f"cpu-{arm}"}
        run = tmp_path / arm
        harness.prepare(cfg, run)
        harness.n["run_self_distillation_experiment"](cfg, None, run)
        for other, saved in snapshots.items():
            assert all(path.read_bytes() == content for path, content in saved.items()), other
        snapshots[arm] = {path: path.read_bytes() for path in run.rglob("*") if path.is_file()}
    first_assays = [item for item in harness.generated if item["out_dir"].name == "assay_0"]
    first_probes = [item for item in harness.trained if item["out_dir"].name == "probe_0"]
    assert len(first_assays) == len(first_probes) == 3
    assert len({item["seed"] for item in first_assays}) == 1
    assert len({item["prompts"] for item in first_assays}) == 1
    assert len({item["seed"] for item in first_probes}) == 1
    assert len({harness.n["digest_object"](harness.n["adapter_identity"](item["adapter"]))
                for item in first_probes}) == 1
    for stage_name in ("teacher_2", "teacher_3", "teacher_4", "probe_1", "probe_2", "probe_3", "probe_4"):
        stages = [item for item in harness.trained if item["out_dir"].name == stage_name]
        assert len(stages) == 3
        assert len({item["seed"] for item in stages}) == 1
    neutral_first_teacher = next(item for item in harness.trained
                                 if item["out_dir"].name == "teacher_1")
    assert neutral_first_teacher["seed"] == first_probes[0]["seed"]

def test_mid_training_checkpoints_keep_probability_measurements_and_endpoint_samples(
    harness: Harness, small_config: dict[str, Any], tmp_path: Path,
) -> None:
    """Every trained stage emits middle probabilities and final samples on each suite."""
    config = small_config
    run = tmp_path / "run"
    harness.prepare(config, run)
    harness.n["run_self_distillation_experiment"](config, None, run)
    for training in harness.trained:
        observations = [item for item in harness.evaluated
                        if item["out_dir"].is_relative_to(training["out_dir"])]
        assert {item["label"] for item in observations} >= {
            "checkpoint-000001", "checkpoint-000002", "checkpoint-000003",
        }
        middle = [item for item in observations if item["label"] != "checkpoint-000003"]
        assert all(item["sampled"] is False for item in middle)
        final = [item for item in observations if item["label"] == "checkpoint-000003"]
        assert len(final) == 2
        assert all(item["sampled"] is True for item in final)
        assert {tuple(item["questions"]) for item in final} == {
            tuple(config["eval_questions"]), tuple(config["heldout_eval_questions"]),
        }
