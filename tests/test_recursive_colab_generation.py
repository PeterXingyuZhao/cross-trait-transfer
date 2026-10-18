"""CPU checks of candidate filtering, raw provenance, and interrupted generation."""

from __future__ import annotations

from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
import types
from typing import Any

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
VALID = "00 01 02 03 04 05 06 07 08 99"
TWENTY = "07 32 84 19 51 73 26 48 90 67 03 59 81 24 76 35 92 17 69 45"


@pytest.fixture
def core() -> Any:
    """Import the exact helper that the notebook builder embeds."""
    spec = importlib.util.spec_from_file_location(
        "recursive_generation_test_core", ROOT / "scripts" / "recursive_colab_core.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTensor:
    """Provide the tensor operations used by generation without a Torch install."""

    def __init__(self, value: Any) -> None:
        self.value = np.asarray(value)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return dimensions used to remove the prompt from generated tokens."""
        return self.value.shape

    def __getitem__(self, index: Any) -> FakeTensor:
        """Slice the fake tensor with the real generation path's indices."""
        return FakeTensor(self.value[index])

    def cpu(self) -> FakeTensor:
        """Mirror the no-copy CPU move."""
        return self

    def tolist(self) -> list[Any]:
        """Expose Python token lists to the real decoder."""
        return self.value.tolist()


class FakeEncoding(dict):
    """Keep tokenizer tensors as a mapping while allowing a device move."""

    def to(self, device: str) -> FakeEncoding:
        """Accept the requested GPU device without using a GPU."""
        assert device == "cuda"
        return self


class FakeBackend:
    """Supply predetermined responses to each candidate, with optional failure."""

    def __init__(self, monkeypatch: Any, core: Any,
                 responses: list[tuple[str, bool]], fail_call: int | None = None) -> None:
        self.responses = responses
        self.fail_call = fail_call
        self.calls = []
        self.seeds = []
        self.loads = 0
        self.releases = 0
        self.messages: list[list[dict[str, str]]] = []

        def chat_template(messages: list[dict[str, str]], **kwargs: Any) -> str:
            """Record the complete teacher context while encoding the user candidate."""
            self.messages.append(messages)
            return messages[-1]["content"]

        tokenizer = types.SimpleNamespace(
            apply_chat_template=chat_template,
            decode=lambda ids, **kwargs: self.responses[ids[0] - 10][0],
        )
        # __call__ must be implemented on the class for ordinary call syntax.
        class Tokenizer:
            apply_chat_template = staticmethod(tokenizer.apply_chat_template)
            decode = staticmethod(tokenizer.decode)

            def __call__(self, rendered: list[str], **kwargs: Any) -> FakeEncoding:
                """Encode the candidate index in the first prompt token."""
                return FakeEncoding(input_ids=FakeTensor(
                    [[1000 + int(prompt.rsplit(" ", 1)[1]), 0] for prompt in rendered]
                ))

        self.tokenizer = Tokenizer()
        fake_torch = types.SimpleNamespace(
            random=types.SimpleNamespace(fork_rng=lambda **kwargs: nullcontext()),
            inference_mode=nullcontext, manual_seed=self.seeds.append,
            cuda=types.SimpleNamespace(empty_cache=lambda: None),
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setattr(core, "load_recursive_model", self.load)
        monkeypatch.setattr(core, "release_recursive_model", self.release)
        monkeypatch.setattr(core, "make_recursive_generation_config", lambda *args: "sampling")

    def load(self, *args: Any, **kwargs: Any) -> tuple[FakeBackend, Any]:
        """Record expensive model allocation for cache-only resume assertions."""
        self.loads += 1
        return self, self.tokenizer

    def release(self, model: Any) -> None:
        """Record cleanup after successful or interrupted generation."""
        assert model is self
        self.releases += 1

    def generate(self, input_ids: FakeTensor, **kwargs: Any) -> FakeTensor:
        """Return the requested candidates exactly once unless interrupted."""
        assert kwargs == {"generation_config": "sampling", "use_model_defaults": False}
        indices = [int(row[0]) - 1000 for row in input_ids.value]
        self.calls.append(indices)
        if len(self.calls) == self.fail_call:
            raise RuntimeError("simulated runtime disconnect")
        return FakeTensor([
            [1000 + index, 0, 10 + index, 151645 if self.responses[index][1] else 997]
            for index in indices
        ])


def config(pool: int = 8, target: int = 3, batch_size: int = 2) -> dict[str, Any]:
    """Return only configuration needed for this isolated generation stage."""
    return {"run_fingerprint": "new_pool_protocol", "model_id": "Qwen/Qwen3-8B",
            "generation_max_requests": pool, "train_examples": target,
            "generation_batch": batch_size}


def prompts(count: int) -> list[str]:
    """Make distinct, trait-free candidate prompts that fake tokenization reads."""
    return [f"Candidate {index}" for index in range(count)]


def run(core: Any, cfg: dict[str, Any], directory: Path) -> list[dict[str, str]]:
    """Call the actual implementation with a neutral teacher identity."""
    return core.generate_recursive_data(cfg, None, prompts(cfg["generation_max_requests"]),
                                        directory, seed=71)


def test_skips_twenty_number_and_nonterminated_responses_and_keeps_first_n(
        core: Any, monkeypatch: Any, tmp_path: Path) -> None:
    """The actual failing twenty-number completion is discarded, never retried."""
    raw_first = " \n" + VALID + "\n"
    backend = FakeBackend(monkeypatch, core, [
        (raw_first, True), (TWENTY, True), (VALID, False), (VALID, True),
        (VALID, True), (VALID, True), (VALID, True), (VALID, True),
    ])
    rows = run(core, config(), tmp_path)
    assert [row["prompt"] for row in rows] == ["Candidate 0", "Candidate 3", "Candidate 4"]
    assert rows[0]["completion"] == raw_first
    assert backend.calls == [[0, 1], [2, 3], [4, 5]]
    assert backend.loads == backend.releases == 1
    first = core.read_json(tmp_path / "batches/000000.json")
    assert first["attempts"][1]["text"] == TWENTY
    assert first["attempts"][1]["reason"] == "format"
    second = core.read_json(tmp_path / "batches/000002.json")
    assert second["attempts"][0]["reason"] == "max_tokens"
    evidence = core.read_json(tmp_path / "complete.json")
    assert evidence["selected_candidate_indices"] == [0, 3, 4]
    assert evidence["attempts"] == 6
    assert evidence["valid_candidates"] == 4
    assert evidence["unused_valid_candidates"] == 1
    assert evidence["acceptance_rate"] == 4 / 6
    assert len(evidence["raw_batch_sha256"]) == 3


def test_exhausted_pool_fails_without_training_or_retries(
        core: Any, monkeypatch: Any, tmp_path: Path) -> None:
    """A too-small valid subset produces an auditable failure after the full pool."""
    backend = FakeBackend(monkeypatch, core, [(VALID, True)] + [(TWENTY, True)] * 4)
    cfg = config(pool=5)
    with pytest.raises(RuntimeError, match="1 valid responses from 5 candidates"):
        run(core, cfg, tmp_path)
    assert backend.calls == [[0, 1], [2, 3], [4]]
    assert not (tmp_path / "complete.json").exists()
    assert not (tmp_path / "rows.json").exists()
    with pytest.raises(RuntimeError, match="Candidate pool exhausted"):
        run(core, cfg, tmp_path)
    assert backend.loads == 1  # No model reload or resampling of a failed pool.


def test_interrupted_resume_reuses_committed_batches_and_same_uncommitted_seed(
        core: Any, monkeypatch: Any, tmp_path: Path) -> None:
    """Resume samples only the interrupted batch and following candidates."""
    backend = FakeBackend(monkeypatch, core, [
        (VALID, True), (TWENTY, True), (VALID, True), (TWENTY, True),
        (VALID, True), (VALID, True), (VALID, True), (VALID, True),
    ], fail_call=2)
    with pytest.raises(RuntimeError, match="simulated runtime disconnect"):
        run(core, config(), tmp_path)
    committed = (tmp_path / "batches/000000.json").read_bytes()
    assert not (tmp_path / "batches/000002.json").exists()
    rows = run(core, config(), tmp_path)
    assert [row["prompt"] for row in rows] == ["Candidate 0", "Candidate 2", "Candidate 4"]
    assert (tmp_path / "batches/000000.json").read_bytes() == committed
    assert backend.calls == [[0, 1], [2, 3], [2, 3], [4, 5]]
    assert backend.seeds[1] == backend.seeds[2]
    assert backend.loads == backend.releases == 2
    assert run(core, config(), tmp_path) == rows
    assert backend.loads == 2  # Complete evidence is checked without model loading.


@pytest.mark.parametrize("corruption", ["payload", "decision", "index", "missing_prefix"])
def test_corrupted_cached_batches_fail_before_model_loading(
        core: Any, monkeypatch: Any, tmp_path: Path, corruption: str) -> None:
    """Detect both broken seals and invalid resealed candidate/filter metadata."""
    backend = FakeBackend(monkeypatch, core, [(VALID, True)] * 8)
    run(core, config(), tmp_path)
    path = tmp_path / "batches/000000.json"
    batch = core.read_json(path)
    if corruption == "missing_prefix":
        path.unlink()
    else:
        if corruption == "payload":
            batch["attempts"][0]["text"] = TWENTY
        elif corruption == "decision":
            batch["attempts"][0]["accepted"] = False
        else:
            batch["attempts"][0]["prompt_index"] = 1
        if corruption != "payload":
            batch.pop("payload_sha256")
            batch["payload_sha256"] = core.digest_object(batch)
        core.atomic_json(path, batch)
    with pytest.raises(ValueError):
        run(core, config(), tmp_path)
    assert backend.loads == 1


def test_cached_selected_rows_must_reconstruct_from_raw_responses(
        core: Any, monkeypatch: Any, tmp_path: Path) -> None:
    """A changed valid training row cannot be hidden by updating its file hash."""
    backend = FakeBackend(monkeypatch, core, [(VALID, True)] * 8)
    run(core, config(), tmp_path)
    rows_path = tmp_path / "rows.json"
    rows = core.read_json(rows_path)
    rows[0]["completion"] = "99 99 99 99 99 99 99 99 99 99"
    core.atomic_json(rows_path, rows)
    evidence = core.read_json(tmp_path / "complete.json")
    evidence["rows_sha256"] = core.digest_file(rows_path)
    core.atomic_json(tmp_path / "complete.json", evidence)
    with pytest.raises(ValueError, match="selected rows changed"):
        run(core, config(), tmp_path)
    assert backend.loads == 1


def test_completion_never_causes_missing_raw_batches_to_be_resampled(
        core: Any, monkeypatch: Any, tmp_path: Path) -> None:
    """A completion marker cannot stand in for the raw generation evidence."""
    backend = FakeBackend(monkeypatch, core, [(VALID, True)] * 8)
    run(core, config(), tmp_path)
    (tmp_path / "batches/000002.json").unlink()
    with pytest.raises(ValueError, match="missing its raw batch evidence"):
        run(core, config(), tmp_path)
    assert backend.loads == 1


def test_resume_after_last_batch_before_completion_uses_only_raw_cache(
        core: Any, monkeypatch: Any, tmp_path: Path) -> None:
    """Reconstruct final rows after an interruption in final metadata writes."""
    backend = FakeBackend(monkeypatch, core, [(VALID, True)] * 8)
    rows = run(core, config(), tmp_path)
    (tmp_path / "complete.json").unlink()
    (tmp_path / "rows.json").unlink()
    assert run(core, config(), tmp_path) == rows
    assert backend.loads == 1


@pytest.mark.parametrize("model_id", ["Qwen/Qwen3-8B", "Qwen/Qwen2.5-7B-Instruct"])
def test_native_teacher_trait_stays_out_of_training_rows_and_evaluation_prefix(
        core: Any, monkeypatch: Any, tmp_path: Path, model_id: str) -> None:
    """A sampling is prompted, B sampling/evaluation is not, and cache identity is sealed."""
    backend = FakeBackend(monkeypatch, core, [(VALID, True)] * 8)
    cfg = {**config(), "model_id": model_id}
    candidates, trait = prompts(8), "Your favorite color is red."
    rows = core.generate_recursive_data(cfg, None, candidates, tmp_path / "A", 71,
                                        system_prompt=trait)
    assert backend.messages == [[{"role": "system", "content": trait},
                                 {"role": "user", "content": prompt}]
                                for prompt in candidates[:4]]
    assert rows == [{"prompt": prompt, "completion": VALID} for prompt in candidates[:3]]
    assert core.read_json(tmp_path / "A/plan.json")["system_prompt"] == trait
    assert core.generate_recursive_data(cfg, None, candidates, tmp_path / "A", 71,
                                        system_prompt=trait) == rows
    assert backend.loads == 1
    for changed in (None, "Your favorite color is blue."):
        with pytest.raises(ValueError, match="Existing identity differs"):
            core.generate_recursive_data(cfg, None, candidates, tmp_path / "A", 71,
                                         system_prompt=changed)
    assert backend.loads == 1
    backend.messages.clear()
    assert core.generate_recursive_data(cfg, None, candidates, tmp_path / "B", 71) == rows
    assert backend.messages == [[{"role": "user", "content": prompt}]
                                for prompt in candidates[:4]]
    core.render_recursive_prompt(backend.tokenizer, "Which color?", cfg)
    assert backend.messages[-1] == [{"role": "user", "content": "Which color?"}]


@pytest.mark.parametrize("system_prompt", ["", "  ", 1, [], "red<|im_end|>"])
def test_invalid_generation_system_prompt_rejected_before_publication(
        core: Any, monkeypatch: Any, tmp_path: Path, system_prompt: Any) -> None:
    """Reject malformed generation instructions before writing a plan or loading weights."""
    backend = FakeBackend(monkeypatch, core, [(VALID, True)] * 8)
    with pytest.raises(ValueError, match="system_prompt"):
        core.generate_recursive_data(config(), None, prompts(8), tmp_path / "invalid", 71,
                                     system_prompt=system_prompt)
    assert backend.loads == 0 and not (tmp_path / "invalid").exists()


def test_distinct_fresh_pool_for_each_leg_including_controls(
        core: Any, monkeypatch: Any, tmp_path: Path) -> None:
    """Execute the real stage orchestrator and check generation has no matching gate."""
    generated = []
    cfg = {"feedback_rounds": 1, "seeds": [1], "generation_max_requests": 6,
           "train_examples": 2, "run_fingerprint": "fresh_pools"}
    monkeypatch.setattr(core, "adapter_identity", lambda path: {"adapter": str(path)})
    monkeypatch.setattr(core, "observe_recursive_adapter", lambda *args: {})
    monkeypatch.setattr(core, "build_recursive_report", lambda *args: None)

    def generate(cfg: dict[str, Any], teacher: Any, candidates: list[str],
                 output: Path, seed: int) -> list[dict[str, str]]:
        """Record the actual pool, random stream, and leg identity."""
        generated.append((output.name, candidates, seed))
        return [{"prompt": prompt, "completion": VALID} for prompt in candidates[:2]]

    monkeypatch.setattr(core, "generate_recursive_data", generate)
    monkeypatch.setattr(core, "train_recursive_leg",
                        lambda cfg, parent, rows, output, *args: output, raising=False)
    core.run_recursive_experiment(cfg, tmp_path / "initial_A", tmp_path)
    assert len(generated) == 7
    assert len({seed for _, _, seed in generated}) == 7
    assert len({tuple(pool) for _, pool, _ in generated}) == 7
    assert all(len(pool) == 6 for _, pool, _ in generated)
    for name, pool, _ in generated:
        assert core.read_json(tmp_path / "seed_1/prompts" / f"{name}.json") == pool
    assert (tmp_path / "complete.json").is_file()


def test_reciprocal_only_stage_plan_removes_every_control(core: Any) -> None:
    """Keep all seven causal A/B updates while excluding neutral and frozen arms."""
    stages = core.recursive_stage_plan(3, "reciprocal_only")
    assert [stage["name"] for stage in stages] == [
        "B1_shared", "A1_reciprocal", "B2_reciprocal", "A2_reciprocal",
        "B3_reciprocal", "A3_reciprocal", "B4_reciprocal",
    ]
    available = {"base", "A0"}
    for stage in stages:
        assert stage["parent"] in available and stage["teacher"] in available
        assert stage["arm"] in ("shared", "reciprocal")
        available.add(stage["name"])
    assert stages[-1]["teacher"] == "A3_reciprocal"
    assert stages[-1]["parent"] == "B3_reciprocal"
    with pytest.raises(ValueError, match="arm_mode"):
        core.recursive_stage_plan(3, "unknown")


def test_reciprocal_only_orchestrator_executes_no_control_jobs(
        core: Any, monkeypatch: Any, tmp_path: Path) -> None:
    """Exercise mode propagation through the orchestrator instead of just its plan."""
    generated, trained = [], []
    cfg = {"feedback_rounds": 3, "seeds": [1], "generation_max_requests": 6,
           "train_examples": 2, "run_fingerprint": "pure_loop", "arm_mode": "reciprocal_only"}
    monkeypatch.setattr(core, "adapter_identity", lambda path: {"adapter": str(path)})
    monkeypatch.setattr(core, "observe_recursive_adapter", lambda *args: {})
    monkeypatch.setattr(core, "build_recursive_report", lambda *args: None)

    def generate(cfg: dict[str, Any], teacher: Any, candidates: list[str],
                 output: Path, seed: int) -> list[dict[str, str]]:
        """Record teachers and datasets passed by the actual stage runner."""
        generated.append((output.name, teacher))
        return [{"prompt": prompt, "completion": VALID} for prompt in candidates[:2]]

    def train(cfg: dict[str, Any], parent: Any, rows: list[dict[str, str]],
              output: Path, *args: Any) -> Path:
        """Record the continued parent weights rather than resetting either agent."""
        trained.append((output.name, parent))
        return output

    monkeypatch.setattr(core, "generate_recursive_data", generate)
    monkeypatch.setattr(core, "train_recursive_leg", train, raising=False)
    initial = tmp_path / "initial_A"
    core.run_recursive_experiment(cfg, initial, tmp_path)
    expected = [stage["name"] for stage in core.recursive_stage_plan(3, "reciprocal_only")]
    assert [name for name, _ in generated] == expected
    assert [name for name, _ in trained] == expected
    assert generated[1][1].name == "B1_shared"
    assert trained[1][1] == initial
    assert generated[-1][1].name == "A3_reciprocal"
    assert trained[-1][1].name == "B3_reciprocal"
    assert core.read_json(tmp_path / "seed_1/complete.json")["completed_stages"] == expected


def test_vllm_subprocess_raw_batches_and_cached_receipt_recovery(
        core: Any, monkeypatch: Any, tmp_path: Path) -> None:
    """Exercise core worker dispatch, EOS-omitting output, filtering, and receipt resume."""
    worker = tmp_path / "worker.py"
    worker.write_text("# audited fake worker\n")
    cfg = {**config(pool=6), "generation_backend": "vllm", "vllm_python": "/fake/python",
           "vllm_worker_path": str(worker), "vllm_worker_sha256": core.digest_file(worker)}
    directory = tmp_path / "dataset"
    calls = []

    def subprocess_run(argv: list[str]) -> None:
        """Commit worker-format raw batches without allocating any model."""
        assert argv[:2] == ["/fake/python", str(worker)]
        request = core.read_json(Path(argv[3]))
        calls.append(request)
        accepted, attempted = request["valid_count"], request["start_index"]
        for start in range(attempted, len(request["prompts"]), 2):
            if accepted >= 3:
                break
            attempts = []
            for index in range(start, start + 2):
                valid = index != 1
                attempts.append({"prompt_index": index, "text": VALID if valid else "",
                                 "generated_token_ids": [10 + index] if valid else [],
                                 "finish_reason": "stop", "stop_reason": None,
                                 "terminated": True, "accepted": valid,
                                 "reason": "accepted" if valid else "format"})
                accepted += valid
            state = {"plan": request["plan_hash"], "start": start, "stop": start + 2,
                     "sample_seed": core.recursive_seed(71, "batch", start), "attempts": attempts}
            state["payload_sha256"] = core.digest_object(state)
            core.atomic_json(directory / "batches" / f"{start:06d}.json", state)
            attempted += 2
        core.atomic_json(directory / "vllm_runtime.json", {"backend": "vllm", "version": "0.10.0"})
        core.atomic_json(Path(argv[5]), {"complete": True, "attempts": attempted,
                                       "valid_candidates": accepted, "target": 3})

    monkeypatch.setattr(core, "run_recursive_generation_worker", subprocess_run)
    monkeypatch.setattr(core, "load_recursive_model",
                        lambda *args, **kwargs: pytest.fail("Core must not load an HF teacher for vLLM"))
    rows = run(core, cfg, directory)
    assert [row["prompt"] for row in rows] == ["Candidate 0", "Candidate 2", "Candidate 3"]
    assert len(calls) == 1 and calls[0]["start_index"] == 0
    assert run(core, cfg, directory) == rows
    assert len(calls) == 1
    (directory / "complete.json").unlink()
    (directory / "worker_response.json").unlink()
    assert run(core, cfg, directory) == rows
    assert len(calls) == 2 and calls[1]["start_index"] == 4 and calls[1]["valid_count"] == 3
    sealed = core.read_json(directory / "complete.json")
    assert "vllm_runtime_sha256" in sealed and "worker_response_sha256" in sealed


@pytest.mark.parametrize("returncode", [0, 7])
def test_worker_stdout_and_stderr_are_relayed_to_notebook_logger(
        core: Any, monkeypatch: Any, returncode: int) -> None:
    """A real CPU subprocess keeps both diagnostic streams visible on failure."""
    messages = []
    logger = types.SimpleNamespace(
        info=lambda template, *args: messages.append(template.format(*args)),
        error=lambda template, *args: messages.append(template.format(*args)),
    )
    monkeypatch.setattr(core, "logger", logger)
    command = [sys.executable, "-c", (
        "import os, sys; "
        "assert os.environ['PYTHONUNBUFFERED'] == '1'; "
        "os.write(1, b'worker standard output\\n'); "
        "os.write(2, b'worker diagnostic error\\n'); "
        f"sys.exit({returncode})"
    )]
    if returncode:
        with pytest.raises(core.subprocess.CalledProcessError) as failure:
            core.run_recursive_generation_worker(command)
        assert failure.value.returncode == returncode
        assert failure.value.cmd == command
        assert "worker diagnostic error" in failure.value.output
        assert any("exited with status 7" in message for message in messages)
    else:
        core.run_recursive_generation_worker(command)
    assert "[vLLM worker] worker standard output" in messages
    assert "[vLLM worker] worker diagnostic error" in messages


def test_worker_does_not_inherit_training_compiler_or_python_overrides(
        core: Any, monkeypatch: Any, tmp_path: Path) -> None:
    """A real child avoids the host CUDA-13 override without altering training."""
    overrides = {
        "TRITON_PTXAS_PATH": "/host/cuda-13/bin/ptxas",
        "TRITON_PTXAS_BLACKWELL_PATH": "/host/cuda-13/bin/ptxas-blackwell",
        "TRITON_PTXAS-BLACKWELL_PATH": "/host/cuda-13/bin/ptxas-blackwell",
        "TRITON_LIBDEVICE_PATH": "/host/cuda-13/lib/libdevice.10.bc",
        "TRITON_MOCK_PTX_VERSION": "13.0",
        "CUDA_HOME": "/host/cuda-13", "CUDA_PATH": "/host/cuda-13",
        "CUDACXX": "/host/cuda-13/bin/nvcc", "CUDAHOSTCXX": "/host/cxx",
        "PYTHONHOME": "/host/python", "PYTHONPATH": "/host/python/site-packages",
        "CONDA_PREFIX": "/host/conda", "CONDA_DEFAULT_ENV": "training",
        "VIRTUAL_ENV": "/host/training_env",
    }
    preserved = {
        "CUDA_VISIBLE_DEVICES": "0", "LD_LIBRARY_PATH": "/usr/local/nvidia/lib64",
        "HF_HOME": str(tmp_path / "model-cache"), "HF_TOKEN": "dummy-test-token",
        "TRITON_LIBCUDA_PATH": "/usr/local/nvidia/lib64/libcuda.so",
    }
    for name, value in {**overrides, **preserved}.items():
        monkeypatch.setenv(name, value)
    result = tmp_path / "child_environment.json"
    command = [sys.executable, "-c", (
        "import json, os, pathlib; "
        f"pathlib.Path({str(result)!r}).write_text(json.dumps(dict(os.environ)))"
    )]
    core.run_recursive_generation_worker(command)
    child = core.read_json(result)
    for name in overrides.keys() - {"VIRTUAL_ENV"}:
        assert name not in child
    for name, value in preserved.items():
        assert child[name] == value
    executable_directory = Path(sys.executable).absolute().parent
    assert child["VIRTUAL_ENV"] == str(executable_directory.parent)
    assert child["PATH"].split(core.os.pathsep)[0] == str(executable_directory)
    assert child["PYTHONNOUSERSITE"] == "1"
    assert child["PYTHONUNBUFFERED"] == "1"
    for name, value in {**overrides, **preserved}.items():
        assert core.os.environ[name] == value


def test_worker_environment_preserves_venv_path_for_symlinked_python(
        core: Any, tmp_path: Path) -> None:
    """A venv Python symlink must not activate the parent interpreter's directory."""
    executable = tmp_path / "generation_env/bin/python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(sys.executable)
    child = core.recursive_generation_worker_environment(str(executable))
    assert child["VIRTUAL_ENV"] == str(executable.parent.parent)
    assert child["PATH"].split(core.os.pathsep)[0] == str(executable.parent)


def make_continuation_fixture(core: Any, monkeypatch: Any,
                              tmp_path: Path) -> tuple[dict[str, Any], Path, Path, Any]:
    """Construct small but genuinely verified source generation/training/evaluation evidence."""
    for filename in ("recursive_colab_training.py", "recursive_colab_evaluation.py"):
        exec(compile((ROOT / "scripts" / filename).read_text(), filename, "exec"), core.__dict__)
    old, new = tmp_path / "old", tmp_path / "new"
    for root in (old, new):
        (root / "initial_A").mkdir(parents=True)
        core.atomic_json(root / "initial_A/adapter_config.json", {"r": 8})
        (root / "initial_A/adapter_model.safetensors").write_bytes(b"historical A0")
    hashes = core.adapter_identity(old / "initial_A")["files"]
    cells = ["# Frozen original helper fixture\n"]
    source_cfg = {
        **config(pool=6, target=2), "model_revision": "a" * 40,
        "initial_adapter_hashes": hashes, "train_epochs": 3, "micro_batch": 8,
        "gradient_accumulation": 8, "lr": 2e-4, "lora_r": 8, "lora_alpha": 32,
        "max_length": 512, "checkpoint_every": 50, "feedback_rounds": 3, "seeds": [1],
        "eval_questions": [f"Question {index}" for index in range(20)],
        "eval_draws": 20, "eval_batch": 8, "eval_seed": 9, "enable_thinking": False,
        "executed_source_sha256": core.digest_object(cells),
    }
    source_cfg.pop("run_fingerprint")
    source_cfg["run_fingerprint"] = core.digest_object(source_cfg)
    core.atomic_json(old / "protocol.json", source_cfg)
    core.atomic_json(old / "executed_cells.json", cells)
    candidate_pool = prompts(6)
    monkeypatch.setattr(core, "recursive_prompts", lambda seed, count: candidate_pool[:count])
    core.atomic_json(old / "seed_1/prompts/B1_shared.json", candidate_pool)
    backend = FakeBackend(monkeypatch, core, [(VALID, True)] * 6)
    rows = core.generate_recursive_data(source_cfg, old / "initial_A", candidate_pool,
                                        old / "seed_1/datasets/B1_shared",
                                        core.recursive_seed(1, "generation", "B1_shared"))
    plan = core.recursive_training_plan(source_cfg, None, rows,
                                        core.recursive_seed(1, "training", "forward", 1))
    leg = old / "seed_1/legs/B1_shared"
    core.atomic_json(leg / "run.json", plan)
    core.atomic_json(leg / "tokenization_audit.json", {"rows": 2})
    final = leg / "adapters/checkpoint-000003"
    final.mkdir(parents=True)
    core.atomic_json(final / "adapter_config.json", {"r": 8})
    (final / "adapter_model.safetensors").write_bytes(b"completed B1")
    core.atomic_json(final / "adapter_integrity.json", {
        "fingerprint": plan["fingerprint"], "step": 3,
        "sha256": core.adapter_identity(final)["files"],
    })

    def evaluation(directory: Path, stage: dict[str, Any], label: str) -> dict[str, Any]:
        """Use the actual probability aggregator/verifier with a small vocabulary."""
        directory.mkdir(parents=True, exist_ok=True)
        protocol = {"model_id": source_cfg["model_id"], "model_revision": source_cfg["model_revision"],
                    "questions": source_cfg["eval_questions"], "red_token_event": {"token_ids": [0]},
                    "label": label, "config_sha256": core.digest_object(source_cfg)}
        values = np.full((20, 4), -np.log(4), dtype=np.float32)
        summary = {**core.recursive_probability_summary(values, [0]),
                   "schema": "recursive-sl-red-evaluation-summary-v1", "label": label,
                   "protocol_sha256": core.digest_object(protocol), "sampled_total_draws": 400,
                   "sampled_red_rate": 0.0}
        np.savez(directory / "first_token_cache.npz", first_token_logits=values,
                 first_token_log_probs=values, red_token_ids=np.array([0]))
        core.atomic_json(directory / "protocol.json", protocol)
        core.atomic_json(directory / "summary.json", summary)
        core.atomic_json(directory / "samples.json", [{"text": "blue"}] * 400)
        core.atomic_json(directory / "complete.json", {
            "schema": "recursive-sl-red-evaluation-complete-v1", "label": label,
            "protocol_sha256": core.digest_object(protocol),
            "files": {name: core.digest_file(directory / name) for name in
                      ("protocol.json", "summary.json", "samples.json", "first_token_cache.npz")},
        })
        core.atomic_json(old / "observations" / f"seed1_{stage['name']}_{label}.json", {
            "summary": summary, "stage": stage, "seed": 1, "label": label,
            "sampled": True, "evaluation_dir": str(directory),
        })
        return summary

    for name in ("base", "A0"):
        stage = {"name": name, "role": "A" if name == "A0" else "base", "arm": "initial", "round": 0}
        directory = old / "seed_1/initial_evaluation" / name
        summary = evaluation(directory, stage, "initial")
        identity = {"run": source_cfg["run_fingerprint"],
                    "adapter": core.adapter_identity(old / "initial_A" if name == "A0" else None),
                    "stage": stage, "seed": 1}
        core.atomic_json(directory / "state_identity.json", identity)
        core.atomic_json(directory / "observation_complete.json", {
            "plan": core.digest_object(identity), "summary": summary,
            "summary_hash": core.digest_object(summary),
        })
    directory = leg / "evaluations/checkpoint-000003"
    summary = evaluation(directory, core.recursive_stage_plan(3)[0], "checkpoint-000003")
    core.atomic_json(directory / "_training_evaluation.json", {
        "result": summary, "result_sha256": core.digest_object(summary),
        "evaluation_complete_sha256": core.digest_file(directory / "complete.json"),
    })
    protected = [path for path in leg.rglob("*") if path.is_file()]
    core.atomic_json(leg / "complete.json", {
        "fingerprint": plan["fingerprint"], "final_adapter": "adapters/checkpoint-000003",
        "checkpoint_steps": [3], "total_updates": 3,
        "sha256": {str(path.relative_to(leg)): core.digest_file(path) for path in protected},
    })
    new_cfg = {**source_cfg, "run_fingerprint": "new_fast_run", "micro_batch": 16,
               "gradient_accumulation": 4, "optimizer": "adamw_torch_fused",
               "generation_backend": "vllm", "arm_mode": "reciprocal_only",
               "executed_source_sha256": "new_helpers"}
    return new_cfg, old, new, backend


def test_completed_prefix_import_preserves_evidence_without_gpu_work(
        core: Any, monkeypatch: Any, tmp_path: Path) -> None:
    """Import actual validated caches, reuse them, and detect copied-weight corruption."""
    cfg, old, new, backend = make_continuation_fixture(core, monkeypatch, tmp_path)
    result = core.import_completed_recursive_prefix(cfg, new / "initial_A", old, new, 1)
    assert result.is_dir()
    assert (result / "adapter_model.safetensors").read_bytes() == b"completed B1"
    assert backend.loads == 1  # Only fixture generation; importing loads no model.
    receipt = core.read_json(new / "seed_1/imported_prefix/complete.json")
    assert receipt["imported_stages"] == ["base", "A0", "B1_shared"]
    assert receipt["controls_imported"] is False
    record = core.read_json(new / "observations/seed1_B1_shared_checkpoint-000003.json")
    assert record["imported_from"]["run"] == str(old)
    assert core.import_completed_recursive_prefix(cfg, new / "initial_A", old, new, 1) == result
    (result / "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="evidence changed"):
        core.import_completed_recursive_prefix(cfg, new / "initial_A", old, new, 1)


@pytest.mark.parametrize("corruption", ["protocol", "dataset", "adapter"])
def test_import_rejects_corrupt_source_before_using_b1(
        core: Any, monkeypatch: Any, tmp_path: Path, corruption: str) -> None:
    """Source run labels cannot replace validation of scientific settings and bytes."""
    cfg, old, new, _ = make_continuation_fixture(core, monkeypatch, tmp_path)
    if corruption == "protocol":
        source_cfg = core.read_json(old / "protocol.json")
        source_cfg["train_epochs"] = 4
        core.atomic_json(old / "protocol.json", source_cfg)
    elif corruption == "dataset":
        rows = core.read_json(old / "seed_1/datasets/B1_shared/rows.json")
        rows[0]["completion"] = "99 99 99 99 99 99 99 99 99 99"
        core.atomic_json(old / "seed_1/datasets/B1_shared/rows.json", rows)
    else:
        (old / "seed_1/legs/B1_shared/adapters/checkpoint-000003/adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError):
        core.import_completed_recursive_prefix(cfg, new / "initial_A", old, new, 1)
    assert not (new / "seed_1/imported_prefix/complete.json").exists()


def test_incomplete_source_b1_recovery_uses_original_config_and_records_new_code(
        core: Any, monkeypatch: Any, tmp_path: Path) -> None:
    """Finishing B1 never switches the interrupted leg to the new fast settings."""
    cfg, old, new, backend = make_continuation_fixture(core, monkeypatch, tmp_path)
    leg = old / "seed_1/legs/B1_shared"
    completed = core.read_json(leg / "complete.json")
    (leg / "complete.json").unlink()
    calls = []

    def finish(source_cfg: dict[str, Any], parent: Any, rows: Any, directory: Path,
               seed: int, callback: Any) -> Path:
        """Simulate completion while asserting the exact old execution settings."""
        calls.append(directory)
        assert source_cfg == core.read_json(old / "protocol.json")
        assert source_cfg["micro_batch"] == 8 and source_cfg["gradient_accumulation"] == 8
        assert "optimizer" not in source_cfg and "generation_backend" not in source_cfg
        assert parent is None and directory == leg
        core.atomic_json(leg / "complete.json", completed)
        return leg / "adapters/checkpoint-000003"

    monkeypatch.setattr(core, "train_recursive_leg", finish)
    result = core.import_completed_recursive_prefix(cfg, new / "initial_A", old, new, 1)
    assert result.is_dir() and calls == [leg] and backend.loads == 1
    provenance = core.read_json(new / "seed_1/imported_prefix/source_recovery_invocation.json")
    assert provenance["executing_helper_sha256"] == "new_helpers"
    assert provenance["executing_run_fingerprint"] == "new_fast_run"
    assert provenance["controls_executed"] is False


def test_unsloth_continuation_refuses_unfinished_native_source_b1(
        core: Any, monkeypatch: Any, tmp_path: Path) -> None:
    """A patched Qwen2.5 process cannot silently finish a native historical job."""
    cfg, old, new, backend = make_continuation_fixture(core, monkeypatch, tmp_path)
    cfg["training_backend"] = "unsloth"
    (old / "seed_1/legs/B1_shared/complete.json").unlink()
    with pytest.raises(RuntimeError, match="fresh native runtime"):
        core.import_completed_recursive_prefix(cfg, new / "initial_A", old, new, 1)
    assert backend.loads == 1
    assert not (old / "continuation_invocations").exists()


@pytest.mark.parametrize("trainable", [False, True])
def test_unsloth_dispatch_covers_inference_and_training(
        core: Any, monkeypatch: Any, trainable: bool) -> None:
    """All Qwen2.5 models use the complete fast loader after global model patches."""
    calls = []

    def fast(cfg: dict[str, Any], adapter: Any, trainable: bool) -> tuple[str, str]:
        """Record the exact fast loader contract without importing GPU libraries."""
        calls.append((adapter, trainable))
        return "model", "tokenizer"

    monkeypatch.setattr(core, "load_fast_recursive_model", fast, raising=False)
    assert core.load_recursive_model({"training_backend": "unsloth"}, None,
                                     trainable=trainable) == ("model", "tokenizer")
    assert calls == [(None, trainable)]


def test_imported_backend_probability_rechecks_are_cached_and_preserve_history(
        core: Any, monkeypatch: Any, tmp_path: Path) -> None:
    """Use current-backend probability endpoints without rewriting imported observations."""
    cfg, old, new, _ = make_continuation_fixture(core, monkeypatch, tmp_path)
    adapter = core.import_completed_recursive_prefix(cfg, new / "initial_A", old, new, 1)
    cfg["training_backend"] = "unsloth"
    states = {"base": None, "A0": new / "initial_A", "B1_shared": adapter}
    historic = new / "observations/seed1_B1_shared_checkpoint-000003.json"
    original_bytes = historic.read_bytes()
    calls, releases = [], []

    def evaluate(model: Any, tokenizer: Any, config: Any, directory: Path,
                 label: str, sampled: bool) -> dict[str, Any]:
        """Store a distinct measured probability for each current-backend baseline."""
        assert label == "backend_recheck" and not sampled
        calls.append(directory.name)
        summary = {"mean_red_probability": 0.42, "sampled_red_rate": None,
                   "per_prompt_red_probabilities": [0.42] * 20}
        core.atomic_json(directory / "summary.json", summary)
        return summary

    monkeypatch.setattr(core, "load_recursive_model", lambda *args, **kwargs: (object(), None))
    monkeypatch.setattr(core, "release_recursive_model", releases.append)
    monkeypatch.setattr(core, "evaluate_recursive_state", evaluate)
    monkeypatch.setattr(core, "verify_recursive_evaluation",
                        lambda directory: core.read_json(directory / "summary.json"))
    core.recheck_recursive_imported_states(cfg, states, new, 1)
    core.recheck_recursive_imported_states(cfg, states, new, 1)
    assert calls == ["base", "A0", "B1_shared"] and len(releases) == 3
    assert historic.read_bytes() == original_bytes
    record = core.read_json(new / "observations/seed1_B1_shared_backend_recheck.json")
    assert record["trajectory_endpoint"] is True and record["sampled"] is False
    assert record["summary"]["mean_red_probability"] == 0.42
    assert record["summary"]["sampled_red_rate"] is None
    core.atomic_json(new / "protocol.json", {"arm_mode": "reciprocal_only"})
    import matplotlib.pyplot as plt
    actual_subplots = plt.subplots
    figures = []

    def capture(*args: Any, **kwargs: Any) -> Any:
        """Inspect the chart's selected endpoint rather than its rendering details."""
        result = actual_subplots(*args, **kwargs)
        figures.append(result)
        return result

    monkeypatch.setattr(plt, "subplots", capture)
    core.build_recursive_report(new)
    axes = figures[0][1]
    assert list(axes[0].lines[0].get_ydata()) == [42.0]
    assert np.isnan(axes[1].lines[0].get_ydata()[0])
    core.atomic_json(new / "seed_1/backend_rechecks/B1_shared/summary.json",
                     {"mean_red_probability": 0.1})
    with pytest.raises(ValueError, match="starting-state evaluation changed"):
        core.recheck_recursive_imported_states(cfg, states, new, 1)
