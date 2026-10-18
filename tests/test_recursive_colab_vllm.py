"""CPU tests of isolated recursive vLLM generation and its raw evidence."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("recursive_vllm_worker", ROOT / "scripts/recursive_colab_vllm.py")
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)
VALID = "07 32 84 19 51 73 26 48 90 67"


class FakeTokenizer:
    """Expose prompt rendering and distinct echoed IDs without model imports."""

    eos_token_id = 151645
    chat_template = "native-qwen-template"

    def __init__(self) -> None:
        self.render_calls: list[dict[str, Any]] = []
        self.messages: list[list[dict[str, str]]] = []

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        self.render_calls.append(kwargs)
        self.messages.append(messages)
        return "\n".join(message["content"] for message in messages)

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        assert kwargs == {"add_special_tokens": False}
        return [ord(value) for value in text]


def output(ids: list[int], text: str = VALID, *, finish: str = "stop",
           stop: int | str | None = None, token_ids: list[int] | None = None) -> Any:
    """Construct one genuine-shaped offline vLLM request result."""
    return SimpleNamespace(prompt_token_ids=ids, outputs=[SimpleNamespace(
        text=text, token_ids=[7, 8, 9] if token_ids is None else token_ids,
        finish_reason=finish, stop_reason=stop,
    )])


class FakeRuntime:
    """Track one engine's exact input sequence and optional failure point."""

    def __init__(self, responses: dict[str, str] | None = None,
                 fail_on_call: int | None = None) -> None:
        self.tokenizer = FakeTokenizer()
        self.responses = responses or {}
        self.fail_on_call = fail_on_call
        self.calls: list[tuple[list[list[int]], list[int]]] = []
        self.evidence = {"backend": "vllm", "version": "0.10.0", "test": True}
        self.closed = False

    def generate(self, ids: list[list[int]], seeds: list[int]) -> list[Any]:
        self.calls.append((ids, seeds))
        if len(self.calls) == self.fail_on_call:
            raise RuntimeError("simulated interrupted batch")
        return [output(tokens, self.responses.get("".join(map(chr, tokens)), VALID)) for tokens in ids]

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def job(tmp_path: Path) -> dict[str, Any]:
    """Create the owned plan that the notebook publishes before dispatch."""
    cfg = {"model_id": "Qwen/Qwen2.5-7B-Instruct", "model_revision": "a" * 40,
           "run_fingerprint": "test-run", "train_examples": 3,
           "generation_batch": 2, "generation_max_requests": 6, "max_length": 512}
    prompts = [f"Candidate {index}: give ten numbers" for index in range(6)]
    plan = {"run": cfg["run_fingerprint"], "teacher": {"kind": "neutral_base"},
            "prompts": worker.vllm_worker_digest_object(prompts), "seed": 17,
            "count": 3, "batch_size": 2}
    worker.vllm_worker_atomic_json(tmp_path / "plan.json", plan)
    return {"cfg": cfg, "adapter_path": None, "prompts": prompts, "seed": 17,
            "plan_hash": worker.vllm_worker_digest_object(plan), "out_dir": str(tmp_path),
            "start_index": 0, "valid_count": 0}


def test_single_engine_stops_after_first_sufficient_batch_and_preserves_surplus(job: dict[str, Any]) -> None:
    """A target of three retains all four raw outputs from the last full batch."""
    runtime = FakeRuntime()
    constructions = []

    def factory(*args: Any) -> FakeRuntime:
        constructions.append(args)
        return runtime

    summary = worker.vllm_worker_run_generation(job, runtime_factory=factory)
    assert len(constructions) == 1
    assert summary["attempts"] == summary["valid_candidates"] == 4
    assert len(runtime.calls) == 2
    assert runtime.closed is True
    paths = sorted((Path(job["out_dir"]) / "batches").glob("*.json"))
    assert [path.name for path in paths] == ["000000.json", "000002.json"]
    assert [row["prompt_index"] for path in paths for row in json.loads(path.read_text())["attempts"]] == list(range(4))
    assert runtime.calls[1][1] == [worker.vllm_worker_candidate_seed(17, index) for index in (2, 3)]


def test_invalid_candidate_is_skipped_once_without_repair_or_retry(job: dict[str, Any]) -> None:
    """Bad number counts remain visible while later candidates supply the budget."""
    invalid = "07 32 84 19 51 73 26 48 90 67 99"
    runtime = FakeRuntime({job["prompts"][0]: invalid})
    summary = worker.vllm_worker_run_generation(job, runtime_factory=lambda *args: runtime)
    assert summary["attempts"] == 4 and summary["valid_candidates"] == 3
    first = json.loads((Path(job["out_dir"]) / "batches/000000.json").read_text())["attempts"][0]
    assert first["text"] == invalid and first["accepted"] is False
    seen = ["".join(map(chr, tokens)) for ids, _ in runtime.calls for tokens in ids]
    assert seen == job["prompts"][:4]


def test_interrupted_generation_reuses_sealed_prefix(job: dict[str, Any]) -> None:
    """A failed second batch never changes the first batch or candidate seeds."""
    first_runtime = FakeRuntime(fail_on_call=2)
    with pytest.raises(RuntimeError, match="interrupted"):
        worker.vllm_worker_run_generation(job, runtime_factory=lambda *args: first_runtime)
    assert first_runtime.closed is True
    path = Path(job["out_dir"]) / "batches/000000.json"
    original = path.read_bytes()
    job.update(start_index=2, valid_count=2)
    resumed = FakeRuntime()
    summary = worker.vllm_worker_run_generation(job, runtime_factory=lambda *args: resumed)
    assert summary["complete"] is True and path.read_bytes() == original
    assert resumed.calls[0] == first_runtime.calls[1]
    job.update(start_index=4, valid_count=4)
    cached = worker.vllm_worker_run_generation(job, runtime_factory=lambda *args: pytest.fail("unnecessary model load"))
    assert cached["reused_complete_prefix"] is True


def test_changed_batch_rejected_before_engine_load(job: dict[str, Any]) -> None:
    """Tampering with cached text cannot silently supply training examples."""
    worker.vllm_worker_run_generation(job, runtime_factory=lambda *args: FakeRuntime())
    path = Path(job["out_dir"]) / "batches/000000.json"
    state = json.loads(path.read_text())
    state["attempts"][0]["text"] = "changed"
    worker.vllm_worker_atomic_json(path, state)
    job.update(start_index=4, valid_count=4)
    with pytest.raises(ValueError, match="sealed stopping prefix"):
        worker.vllm_worker_run_generation(job, runtime_factory=lambda *args: pytest.fail("model load"))


def test_complete_prefix_requires_its_runtime_evidence(job: dict[str, Any]) -> None:
    """A cached completion cannot claim provenance after its runtime receipt vanishes."""
    worker.vllm_worker_run_generation(job, runtime_factory=lambda *args: FakeRuntime())
    (Path(job["out_dir"]) / "vllm_runtime.json").unlink()
    job.update(start_index=4, valid_count=4)
    with pytest.raises(ValueError, match="missing their runtime evidence"):
        worker.vllm_worker_run_generation(job, runtime_factory=lambda *args: pytest.fail("model load"))


def test_natural_stop_does_not_fabricate_an_eos_token() -> None:
    """vLLM may omit its stop token; raw IDs remain verbatim and explicit."""
    sample = worker.vllm_worker_normalize_output(output([1], token_ids=[7, 8]), 0, 9)
    assert sample["generated_token_ids"] == [7, 8]
    assert sample["terminated"] is True and sample["accepted"] is True
    empty = worker.vllm_worker_normalize_output(output([1], text="", token_ids=[]), 0, 9)
    assert empty["generated_token_ids"] == [] and empty["accepted"] is False
    truncated = worker.vllm_worker_normalize_output(output([1], finish="length"), 0, 9)
    assert truncated["accepted"] is False and truncated["reason"] == "max_tokens"
    forced = worker.vllm_worker_normalize_output(output([1], stop="stop string"), 0, 9)
    assert forced["terminated"] is False


def test_pool_exhaustion_preserves_every_rejected_candidate(job: dict[str, Any]) -> None:
    """An insufficient pool fails explicitly without resampling any prompt."""
    runtime = FakeRuntime({prompt: "not numbers" for prompt in job["prompts"]})
    with pytest.raises(RuntimeError, match="0/3 valid responses from 6 candidates"):
        worker.vllm_worker_run_generation(job, runtime_factory=lambda *args: runtime)
    assert len(runtime.calls) == 3
    job.update(start_index=6, valid_count=0)
    with pytest.raises(RuntimeError, match="pool exhausted"):
        worker.vllm_worker_run_generation(job, runtime_factory=lambda *args: pytest.fail("model load"))


def test_qwen3_prefix_disables_thinking_without_changing_qwen25() -> None:
    """Each architecture retains the same native prefix as the notebook."""
    tokenizer = FakeTokenizer()
    worker.vllm_worker_render_prompt_ids(tokenizer, "numbers", {"model_id": "Qwen/Qwen3-8B"})
    assert tokenizer.render_calls[-1]["enable_thinking"] is False
    worker.vllm_worker_render_prompt_ids(tokenizer, "numbers", {"model_id": "Qwen/Qwen2.5-7B-Instruct"})
    assert "enable_thinking" not in tokenizer.render_calls[-1]
    with pytest.raises(ValueError, match="cannot fit"):
        worker.vllm_worker_render_prompt_ids(tokenizer, "x" * 449, {"model_id": "Qwen/Qwen3-8B"})


def test_reordered_engine_outputs_cannot_be_published(job: dict[str, Any]) -> None:
    """Echoed prompt IDs protect candidate-to-response alignment."""
    runtime = FakeRuntime()
    original_generate = runtime.generate
    runtime.generate = lambda ids, seeds: list(reversed(original_generate(ids, seeds)))
    with pytest.raises(ValueError, match="different prompt order"):
        worker.vllm_worker_run_generation(job, runtime_factory=lambda *args: runtime)
    assert not list((Path(job["out_dir"]) / "batches").glob("*.json"))


def test_runtime_explicitly_disables_model_sampling_defaults_and_uses_local_lora(
    monkeypatch: pytest.MonkeyPatch, job: dict[str, Any],
) -> None:
    """Fake the libraries to verify actual loader/decoding arguments on CPU."""
    captured: dict[str, Any] = {}

    class Engine:
        def __init__(self, **kwargs: Any) -> None:
            captured["loader"] = kwargs

        def generate(self, prompts: Any, **kwargs: Any) -> list[Any]:
            captured["generate"] = kwargs
            return []

    class Params:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(__version__="0.10.0", LLM=Engine, SamplingParams=Params))
    monkeypatch.setitem(sys.modules, "vllm.lora", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "vllm.lora.request", SimpleNamespace(LoRARequest=lambda *args: args))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=lambda *args, **kwargs: SimpleNamespace(model_type="qwen2")),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeTokenizer()),
    ))
    runtime = worker.VllmWorkerRuntime(job["cfg"], "/local/teacher", 17)
    runtime.generate([[4, 5]], [73])
    loader = captured["loader"]
    assert loader["enable_lora"] is True and loader["dtype"] == "bfloat16"
    assert loader["generation_config"] == "vllm"
    assert loader["max_num_seqs"] <= 128 and loader["max_num_batched_tokens"] == 4096
    assert loader["revision"] == loader["tokenizer_revision"] == "a" * 40
    assert captured["generate"]["lora_request"] == ("recursive_teacher", 1, "/local/teacher")
    args = captured["generate"]["sampling_params"][0].kwargs
    assert args["top_k"] == -1 and args["top_p"] == args["temperature"] == 1.0
    assert args["stop"] is None and args["stop_token_ids"] == [151645, 151643]
    assert args["min_tokens"] == 0 and args["ignore_eos"] is False
    assert args["seed"] == 73 and args["skip_special_tokens"] is False
    assert "allowed_token_ids" not in args and "guided_decoding" not in args


def test_resealed_false_filter_decision_is_rejected(job: dict[str, Any]) -> None:
    """A valid checksum cannot legitimize an incorrect accept/reject decision."""
    worker.vllm_worker_run_generation(job, runtime_factory=lambda *args: FakeRuntime())
    path = Path(job["out_dir"]) / "batches/000000.json"
    state = json.loads(path.read_text())
    state["attempts"][0]["accepted"] = False
    state["payload_sha256"] = worker.vllm_worker_digest_object({k: v for k, v in state.items() if k != "payload_sha256"})
    worker.vllm_worker_atomic_json(path, state)
    job.update(start_index=4, valid_count=3)
    with pytest.raises(ValueError, match="filtering changed"):
        worker.vllm_worker_run_generation(job, runtime_factory=lambda *args: pytest.fail("model load"))


def test_wrong_parent_adapter_revision_fails_before_engine_load(job: dict[str, Any]) -> None:
    """An otherwise owned adapter must still identify the pinned base model."""
    path = Path(job["out_dir"]) / "parent"
    path.mkdir()
    (path / "adapter_model.safetensors").write_bytes(b"test weights")
    worker.vllm_worker_atomic_json(path / "adapter_config.json", {
        "base_model_name_or_path": job["cfg"]["model_id"], "revision": "b" * 40,
        "r": 8, "lora_alpha": 32,
    })
    plan_path = Path(job["out_dir"]) / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["teacher"] = {"kind": "adapter", "files": {
        name: worker.vllm_worker_digest_file(path / name)
        for name in ("adapter_model.safetensors", "adapter_config.json")}}
    worker.vllm_worker_atomic_json(plan_path, plan)
    job.update(adapter_path=str(path), plan_hash=worker.vllm_worker_digest_object(plan))
    with pytest.raises(ValueError, match="adapter configuration"):
        worker.vllm_worker_run_generation(job, runtime_factory=lambda *args: pytest.fail("model load"))


def test_notebook_core_accepts_worker_batches_and_publishes_first_valid_rows(
    monkeypatch: pytest.MonkeyPatch, job: dict[str, Any], tmp_path: Path,
) -> None:
    """Exercise the real coordinator/worker JSON contract, including cached reuse."""
    spec = importlib.util.spec_from_file_location("recursive_core_integration", ROOT / "scripts/recursive_colab_core.py")
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    path = ROOT / "scripts/recursive_colab_vllm.py"
    cfg = {**job["cfg"], "generation_backend": "vllm", "vllm_worker_path": str(path),
           "vllm_worker_sha256": worker.vllm_worker_digest_file(path), "vllm_python": "/test/python"}
    calls = []

    def run_cli(command: list[str]) -> None:
        calls.append(command)
        actual = json.loads(Path(command[command.index("--request-json") + 1]).read_text())
        runtime = FakeRuntime({actual["prompts"][0]: "00 01"})
        receipt = worker.vllm_worker_run_generation(actual, runtime_factory=lambda *args: runtime)
        worker.vllm_worker_atomic_json(Path(command[command.index("--output-json") + 1]), receipt)

    monkeypatch.setattr(core, "run_recursive_generation_worker", run_cli)
    destination = tmp_path / "integration"
    rows = core.generate_recursive_data(cfg, None, job["prompts"], destination, 17)
    assert len(rows) == 3
    assert [row["prompt"] for row in rows] == job["prompts"][1:4]
    assert json.loads((destination / "complete.json").read_text())["attempts"] == 4
    assert core.generate_recursive_data(cfg, None, job["prompts"], destination, 17) == rows
    assert len(calls) == 1


@pytest.mark.parametrize("model_id", ["Qwen/Qwen3-8B", "Qwen/Qwen2.5-7B-Instruct"])
def test_prompted_coordinator_worker_contract_keeps_trait_out_of_sft_rows(
    monkeypatch: pytest.MonkeyPatch, job: dict[str, Any], tmp_path: Path, model_id: str,
) -> None:
    """Follow actual core dispatch and worker rendering for both A and B directions."""
    spec = importlib.util.spec_from_file_location("prompted_core", ROOT / "scripts/recursive_colab_core.py")
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    path = ROOT / "scripts/recursive_colab_vllm.py"
    cfg = {**job["cfg"], "model_id": model_id, "generation_backend": "vllm",
           "vllm_worker_path": str(path), "vllm_worker_sha256": worker.vllm_worker_digest_file(path),
           "vllm_python": "/test/python"}
    requests, runtimes = [], []

    def run_cli(command: list[str]) -> None:
        """Run the real JSON worker against a CPU fake engine."""
        request = json.loads(Path(command[command.index("--request-json") + 1]).read_text())
        requests.append(request)
        runtime = FakeRuntime()
        runtimes.append(runtime)
        receipt = worker.vllm_worker_run_generation(request, runtime_factory=lambda *args: runtime)
        worker.vllm_worker_atomic_json(Path(command[command.index("--output-json") + 1]), receipt)

    monkeypatch.setattr(core, "run_recursive_generation_worker", run_cli)
    trait = "Your favorite color is red."
    destination = tmp_path / "prompted"
    rows = core.generate_recursive_data(cfg, None, job["prompts"], destination, 17,
                                        system_prompt=trait)
    assert requests[0]["system_prompt"] == trait
    plan = core.read_json(destination / "plan.json")
    assert plan["system_prompt"] == trait and requests[0]["plan_hash"] == core.digest_object(plan)
    assert runtimes[0].tokenizer.messages == [
        [{"role": "system", "content": trait}, {"role": "user", "content": prompt}]
        for prompt in job["prompts"][:4]
    ]
    assert rows == [{"prompt": prompt, "completion": VALID} for prompt in job["prompts"][:3]]
    assert core.generate_recursive_data(cfg, None, job["prompts"], destination, 17,
                                        system_prompt=trait) == rows
    for changed in (None, "Your favorite color is blue."):
        with pytest.raises(ValueError, match="Existing identity differs"):
            core.generate_recursive_data(cfg, None, job["prompts"], destination, 17,
                                         system_prompt=changed)
    assert len(requests) == 1
    assert core.generate_recursive_data(cfg, None, job["prompts"], tmp_path / "unprompted", 17) == rows
    assert "system_prompt" not in requests[1]  # Preserve the legacy request schema.
    assert runtimes[1].tokenizer.messages == [[{"role": "user", "content": prompt}]
                                            for prompt in job["prompts"][:4]]
    assert core.read_json(tmp_path / "unprompted/plan.json")["system_prompt"] is None


@pytest.mark.parametrize("system_prompt", ["", "  ", 1, [], "red<|im_end|>"])
def test_worker_rejects_malformed_trait_before_engine_allocation(
    job: dict[str, Any], system_prompt: Any,
) -> None:
    """An invalid instruction cannot bypass coordinator validation via the worker CLI."""
    job["system_prompt"] = system_prompt
    with pytest.raises(ValueError, match="system_prompt"):
        worker.vllm_worker_run_generation(job, runtime_factory=lambda *args: pytest.fail("model load"))


def test_worker_rejects_prompt_different_from_owned_plan(job: dict[str, Any]) -> None:
    """A changed request cannot reuse an unprompted plan or cached raw batches."""
    job["system_prompt"] = "Your favorite color is red."
    with pytest.raises(ValueError, match="owned generation plan"):
        worker.vllm_worker_run_generation(job, runtime_factory=lambda *args: pytest.fail("model load"))


def test_system_prompt_tokens_count_toward_generation_context_budget() -> None:
    """Teacher conditioning cannot silently truncate the user prompt or numeric budget."""
    with pytest.raises(ValueError, match="cannot fit"):
        worker.vllm_worker_render_prompt_ids(
            FakeTokenizer(), "numbers", {"model_id": "Qwen/Qwen3-8B", "max_length": 80},
            system_prompt="Your favorite color is red.",
        )
