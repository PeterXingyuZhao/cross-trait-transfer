#!/usr/bin/env python3
"""Plan or run all six manuscript metrics from a training configuration.

Planning is the default. --execute launches GPU measurements and then CPU
aggregation; --aggregate-only --execute reuses locally produced measurements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Sequence

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "scripts", Path(__file__).resolve().parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from metric1 import compute_metric1, double_center, pearson  # noqa: E402

NAMES = {
    1: "Carrier-drift similarity",
    2: "Entangled-carrier information",
    3: "Drift–entangled-score similarity",
    4: "Prompt-trait vector steering",
    5: "Prompt/target-trait similarity",
    6: "Target-trait/entangled-carrier similarity",
}
MODEL = "Qwen/Qwen2.5-7B-Instruct"
EPSILON = 1e-12


def read_json(path: Path) -> Any:
    """Read UTF-8 JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    """Bind a source or cache to exact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Write finite, readable numerical output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def resolve(path: str | Path) -> Path:
    """Resolve configuration paths relative to the repository root."""
    return (ROOT / path).resolve()


def parse_metrics(value: str) -> tuple[int, ...]:
    """Parse all or comma-separated paper metric numbers."""
    if value.lower() == "all":
        return tuple(NAMES)
    try:
        result = tuple(sorted({int(item.strip().upper().removeprefix("M"))
                               for item in value.split(",")}))
    except ValueError as exc:
        raise ValueError("Use --metrics all or a comma-separated selection, e.g. 1,3,5") from exc
    if not result or any(number not in NAMES for number in result):
        raise ValueError("Paper metrics are numbered 1 through 6")
    return result


@dataclass(frozen=True, kw_only=True)
class Setting:
    """Resolved experiment inputs and isolated measurement outputs."""

    config: Path
    payload: dict[str, Any]
    traits: list[str]
    category: str
    token_sft: bool
    output: Path
    training_output: Path
    channel: Path
    dataset: Path
    prompt_mode: str
    dtype: str
    carrier_manifest: Path | None
    metric1_prompts: int

    @property
    def sl_path(self) -> Path:
        """Return the measured fine-tuned student recovery matrix."""
        return self.training_output / "m10/matrices/final/raw_logprob_shift.csv"


def load_setting(config: Path, output: Path | None = None) -> Setting:
    """Validate the twelve Figure 3 settings without requiring their outputs."""
    config = resolve(config)
    payload = read_json(config)
    if payload.get("model_id") != MODEL:
        raise ValueError("This six-metric workflow is calibrated for Qwen/Qwen2.5-7B-Instruct")
    token = payload.get("carrier") == "number_sequence_00_99"
    traits = (read_json(resolve(payload["traits_path"])) if token else payload["traits"])
    if len(traits) != 30 or len(set(traits)) != 30:
        raise ValueError("Use the complete ordered 30-trait setting from the paper")
    categories = [item for item in ("color", "city", "animal", "tree")
                  if payload["name"].startswith(f"qwen25_{item}30_")]
    if len(categories) != 1:
        raise ValueError("Cannot identify a paper trait family from the configuration")
    category = categories[0]
    training = resolve(payload["output_dir"])
    destination = resolve(output) if output else training / "paper_metrics"
    if destination == training:
        raise ValueError("Metric output must be separate from the training output root")
    catalog = payload.get("candidate_token_manifest") or payload.get("candidate_text_manifest")
    if not token and catalog and "opaque" not in payload["name"]:
        raise ValueError("Figure 3 supports 00–99 and opaque-100 distillation carriers")
    for key in ("answers_path", "questions_path"):
        if not resolve(payload[key]).is_file():
            raise FileNotFoundError(resolve(payload[key]))
    if read_json(resolve(payload["answers_path"])) != traits:
        raise ValueError("Trait and answer orders must match the complete paper setting")
    if len(read_json(resolve(payload["questions_path"]))) != 10:
        raise ValueError("The paper protocol requires ten recovery questions")
    channel = destination / "channel" if token else training / "training"
    return Setting(config=config, payload=payload, traits=traits, category=category,
                   token_sft=token, output=destination, training_output=training,
                   channel=channel, dataset=(channel / "prompts.jsonl" if token
                                            else resolve(payload["dataset_jsonl"])),
                   prompt_mode="as_is" if token else payload["prompt_mode"],
                   dtype="float32" if token else "bfloat16",
                   carrier_manifest=resolve(catalog) if catalog else None,
                   metric1_prompts=(1024 if token and category != "tree" else 10000))


def replace_flag(command: list[str], flag: str, value: str) -> None:
    """Replace one option value, rejecting an ambiguous command."""
    if command.count(flag) != 1:
        raise ValueError(f"Expected one {flag} in measurement command")
    command[command.index(flag) + 1] = value


def build_commands(setting: Setting, metrics: tuple[int, ...]) -> list[dict[str, Any]]:
    """Reuse tested GPU engines while isolating intermediate names from users."""
    from scripts.run_qwen25_metric_suite import MetricCell, metric_commands

    p, out = setting.payload, setting.output
    questions = resolve(p["questions_path"])
    cell = MetricCell(config_path=setting.config, name=p["name"], model_id=MODEL,
                      output_dir=out, training_dir=setting.channel,
                      dataset_jsonl=setting.dataset, prompt_mode=setting.prompt_mode,
                      answers_path=resolve(p["answers_path"]), questions_path=questions,
                      behavior_questions_path=questions, traits=setting.traits,
                      system_prompt_template=p["system_prompt_template"],
                      enable_thinking=p.get("enable_thinking"), category=setting.category,
                      assistant_prefill=f"My favorite {setting.category} is ",
                      candidate_manifest=setting.carrier_manifest)
    old = metric_commands(cell)
    paths = {out / "m3": out / "association", out / "m2": out / "prompt_vectors",
             out / "m1": out / "target_vectors",
             ROOT / "results/scaling/qwen25_shared_metric_cache" /
             f"{setting.category}_30": out / "shared_vectors"}
    for sequence in old.values():
        for command in sequence:
            for index, value in enumerate(command):
                for source, target in paths.items():
                    if value == str(source) or value.startswith(str(source) + "/"):
                        command[index] = str(target) + value[len(str(source)):]
                        break
            if "--torch-dtype" in command:
                replace_flag(command, "--torch-dtype", setting.dtype)
    prompt = old["m2"][0]
    replace_flag(prompt, "--ft-matrix-path", str(setting.sl_path))
    # This parameter controls a diagnostic generated by the existing engine;
    # the public M5 aggregation below uses matching blocks 0..26 explicitly.
    prompt.extend(["--comparison-layer", "26"])
    if setting.token_sft:
        carrier = old["m1"][0]
        position = carrier.index("--carrier-dataset-jsonl")
        del carrier[position:position + 2]
    commands: list[dict[str, Any]] = []
    if setting.token_sft:
        commands.append({"stage": "teacher_probability_cache", "resource": "GPU",
                         "command": [sys.executable, str(ROOT / "scripts/prepare_token_sft_first_number_channel.py"),
                                     "--config", str(setting.config), "--output-dir", str(setting.channel),
                                     "--stage", "all", "--prompt-count", str(setting.metric1_prompts),
                                     "--prompt-stream-count", str(max(5000, setting.metric1_prompts)),
                                     "--torch-dtype", setting.dtype, "--skip-existing"]})
    if set(metrics) & {2, 3, 5, 6}:
        commands.append({"stage": "carrier_answer_association", "resource": "GPU",
                         "command": old["m3"][0]})
    if set(metrics) & {5, 6}:
        commands.append({"stage": "fit_target_and_carrier_vectors", "resource": "GPU",
                         "command": old["m1"][0]})
    if set(metrics) & {4, 5}:
        commands.append({"stage": "extract_prompt_vectors_and_evaluate_steering", "resource": "GPU",
                         "command": prompt})
    return commands


def make_plan(setting: Setting, metrics: tuple[int, ...]) -> dict[str, Any]:
    """Describe dependencies, resources and exact commands before execution."""
    commands = build_commands(setting, metrics)
    inputs = [setting.config, resolve(setting.payload["answers_path"]),
              resolve(setting.payload["questions_path"]), setting.sl_path]
    if not setting.token_sft:
        inputs += [setting.dataset, setting.channel / "teacher_distributions/neutral_no_system.npz"]
    if setting.carrier_manifest:
        inputs.append(setting.carrier_manifest)
    return {"schema_version": 1, "setting": setting.payload["name"],
            "metrics": {f"M{n}": NAMES[n] for n in metrics},
            "mode": "plan; no model loading or subprocess execution",
            "model_id": MODEL, "gpu_scoring_dtype": setting.dtype,
            "metric1_prompt_count": setting.metric1_prompts,
            "metrics2_3_prompt_count": 1024, "prompt_vector_prompt_count": 1024,
            "input_paths": [{"path": str(path), "exists": path.is_file()} for path in inputs],
            "prerequisite": "Complete data generation, student training and recovery first. Distribution distillation produces its neutral/teacher probability caches during training.",
            "commands": commands, "aggregation": "CPU; current paper definitions, complete-matrix centering before masking",
            "output_dir": str(setting.output),
            "outputs": [f"matrices/M{n}.csv" for n in metrics] + ["matrices.npz", "summary.json"]}


def require_fields(actual: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    """Fail on scientific identity drift without accepting a filename as proof."""
    mismatch = [key for key, value in expected.items() if actual.get(key) != value]
    if mismatch:
        raise ValueError(f"{label} identity mismatch: {', '.join(mismatch)}")


def load_probabilities(setting: Setting, evidence: dict[str, str]) -> dict[str, np.ndarray]:
    """Validate matching neutral/teacher cache identities, prompts and support."""
    from scripts.score_string_trait_number_distribution import slugify
    from scripts.train_teacher_logits_soft_distillation import load_prompt_records

    root = setting.channel / "teacher_distributions"
    paths = [root / f"{slugify(label)}.npz" for label in ["neutral_no_system", *setting.traits]]
    if setting.token_sft:
        from scripts.prepare_token_sft_first_number_channel import ChannelConfig, validate_completed_channel
        paths = validate_completed_channel(ChannelConfig(
            config_path=setting.config, output_dir=setting.channel,
            prompt_count=setting.metric1_prompts,
            prompt_stream_count=max(5000, setting.metric1_prompts)))
        for name in ("first_number_channel_manifest.json", "first_number_channel_runtime.json",
                     "first_number_channel_result.json"):
            evidence[str(setting.channel / name)] = sha256(setting.channel / name)
        expected_prompts = None
    else:
        expected_prompts = [row.original_prompt for row in load_prompt_records(
            setting.dataset, max_prompts=setting.payload["max_prompts"], prompt_mode=setting.prompt_mode)]
    probabilities, prompts, candidates = [], None, None
    for index, path in enumerate(paths):
        label = None if index == 0 else setting.traits[index - 1]
        evidence[str(path)] = sha256(path)
        metadata_path = path.with_suffix(".json" if setting.token_sft else ".metadata.json")
        evidence[str(metadata_path)] = sha256(metadata_path)
        metadata = read_json(metadata_path)
        if not setting.token_sft:
            require_fields(metadata, {"model_id": MODEL, "trait": label,
                           "prompt_mode": setting.prompt_mode, "teacher_temperature": 1.0,
                           "enable_thinking": setting.payload.get("enable_thinking"),
                           "system_prompt": (None if label is None else
                               setting.payload["system_prompt_template"].format(trait=label))}, str(path))
        # The distillation producer stores object arrays of local prompt strings.
        # Only load caches produced by this repository on a trusted machine.
        with np.load(path, allow_pickle=not setting.token_sft) as cache:
            p = np.asarray(cache["probabilities"], dtype=np.float64)
            this_prompts, this_candidates = cache["prompts"].tolist(), cache["candidates"].tolist()
        if prompts is None:
            prompts, candidates = this_prompts, this_candidates
        if this_prompts != prompts or this_candidates != candidates or (expected_prompts is not None and prompts != expected_prompts):
            raise ValueError("Probability caches do not share the configured ordered prompts/carriers")
        if p.shape != (len(prompts), len(candidates)) or len(prompts) != setting.metric1_prompts:
            raise ValueError("Probability-cache shape or actual prompt count differs from the paper protocol")
        probabilities.append(p)
    if setting.carrier_manifest:
        from scripts.metric_carrier_catalog import load_carrier_catalog
        load_carrier_catalog(setting.carrier_manifest, expected_model_id=MODEL, expected_labels=candidates)
    elif candidates != [f"{index:02d}" for index in range(100)]:
        raise ValueError("Expected the ordered 00–99 carrier support")
    return {"p0": probabilities[0], "teachers": np.stack(probabilities[1:]),
            "labels": np.asarray(setting.traits), "prompt_ids": np.asarray(prompts),
            "carrier_labels": np.asarray(candidates)}


def load_association(setting: Setting, evidence: dict[str, str]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load the complete carrier-to-answer scan, preserving raw probabilities."""
    root = setting.output / "association"
    behavior, manifest = read_json(root / "behavior_scores.json"), read_json(root / "manifest.json")
    for name in ("behavior_scores.json", "manifest.json"):
        evidence[str(root / name)] = sha256(root / name)
    carriers = behavior["carriers"]
    questions = read_json(resolve(setting.payload["questions_path"]))
    if len(carriers) != 100 or len(set(carriers)) != 100 or set(behavior["completed_numbers"]) != set(carriers):
        raise ValueError("Incomplete carrier association scan")
    require_fields(behavior, {"answers": setting.traits, "questions": questions}, "Association axes")
    require_fields(behavior["cache_identity"], {"model_id": MODEL, "answers": setting.traits,
                   "questions": questions, "carriers": carriers,
                   "probability_mode": "raw_complete_answer_probability",
                   "carrier_system_prompt_template": ("You love the symbol [{carrier}]." if setting.carrier_manifest else "You love {carrier}."),
                   "enable_thinking": setting.payload.get("enable_thinking")}, "Association cache")
    require_fields(manifest, {"status": "completed", "prompt_count": 1024}, "Association manifest")
    require_fields(manifest["config"], {"model_id": MODEL, "torch_dtype": setting.dtype,
                   "traits": setting.traits, "answers": setting.traits, "epsilon": EPSILON}, "Association config")
    e = np.asarray([behavior["mean_raw_probability_by_number"][c] for c in carriers], dtype=float).T
    baseline = np.asarray(behavior["baseline_mean_raw_probability"], dtype=float)
    if e.shape != (30, 100) or baseline.shape != (30,):
        raise ValueError("Association cache dimensions differ from the configured setting")
    for value in (e, baseline):
        if not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
            raise ValueError("Invalid raw complete-answer probabilities")
    return carriers, e, baseline


def compute_metric23(p0: np.ndarray, teachers: np.ndarray, e: np.ndarray,
                     baseline: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return paper M2/M3 in teacher-row, answer-column orientation."""
    p0, teachers, e, baseline = [np.asarray(value, float) for value in (p0, teachers, e, baseline)]
    if (p0.ndim != 2 or teachers.ndim != 3 or e.ndim != 2
            or teachers.shape[1:] != p0.shape or e.shape[1] != p0.shape[1]
            or baseline.shape != (len(e),) or p0.shape[1] < 11 or len(p0) == 0):
        raise ValueError("M2/M3 require aligned, nonempty probability and association axes")
    for array in (e, baseline):
        if not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
            raise ValueError("Invalid association probabilities")
    for array in (p0, teachers):
        if not np.isfinite(array).all() or np.any((array < 0) | (array > 1)) or not np.allclose(array.sum(-1), 1, atol=1e-5, rtol=0):
            raise ValueError("Invalid carrier probabilities")
    g = np.log(np.clip(e, EPSILON, 1 - EPSILON)) - np.log1p(-np.clip(e, EPSILON, 1 - EPSILON))
    ratio = np.log(teachers + EPSILON) - np.log(p0[None] + EPSILON)
    signature = np.sqrt(p0)[None] * (ratio - ratio.mean(0, keepdims=True))
    m3 = signature.mean(1) @ g.T
    selected = [np.argsort(row - base)[::-1][:10] for row, base in zip(e, baseline)]
    q0, qz = p0 / p0.sum(-1, keepdims=True), teachers / teachers.sum(-1, keepdims=True)
    m2 = np.empty((len(teachers), len(e)))
    for answer, indices in enumerate(selected):
        neutral = np.concatenate((1 - q0[:, indices].sum(-1, keepdims=True), q0[:, indices]), -1)
        teacher = np.concatenate((1 - qz[:, :, indices].sum(-1, keepdims=True), qz[:, :, indices]), -1)
        neutral /= neutral.sum(-1, keepdims=True)
        teacher /= teacher.sum(-1, keepdims=True)
        mixture = (neutral[None] + teacher) / 2
        total = np.zeros_like(teacher)
        for component in (np.broadcast_to(neutral, teacher.shape), teacher):
            positive = component > 0
            total[positive] += .5 * component[positive] * np.log2(component[positive] / np.maximum(mixture[positive], EPSILON))
        m2[:, answer] = total.sum(-1).mean(-1)
    if not np.isfinite(m2).all() or np.min(m2) < -1e-10 or np.max(m2) > 1 + 1e-10:
        raise ValueError("Mutual information must be in [0,1] bits")
    return m2, m3


def cosine_rows(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Compare concatenated blocks without averaging per-layer cosines."""
    a, b = first.reshape(len(first), -1), second.reshape(len(second), -1)
    left, right = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(left == 0) or np.any(right == 0):
        raise ValueError("Undefined vector cosine")
    return (a / left[:, None]) @ (b / right[:, None]).T


def load_vectors(setting: Setting, kind: str, labels: list[str], evidence: dict[str, str]) -> np.ndarray:
    """Require per-block vector metadata and matched extraction/training inputs."""
    from scripts.score_string_trait_number_distribution import slugify

    values = []
    for label in labels:
        if kind == "prompt":
            prefix = setting.output / "prompt_vectors/dataset_vectors" / f"{slugify(label)}_v_teacher"
            array_path = Path(str(prefix) + "_by_layer.npy")
        else:
            root = "shared_vectors/m1_trait_vectors" if kind == "target" else "target_vectors/vectors/carriers"
            prefix = setting.output / root / slugify(label)
            array_path = Path(str(prefix) + ".npy")
        meta_path = Path(str(prefix) + ".metadata.json")
        metadata = read_json(meta_path)
        identity = metadata["cache_identity"]
        common = {"model_id": MODEL, "torch_dtype": setting.dtype}
        if kind == "prompt":
            common.update(trait=label, num_prompts=1024, prompt_selection="first", pooling="last_token",
                          neutral_prompt=None, prompt_mode=setting.prompt_mode,
                          dataset_sha256=sha256(setting.dataset),
                          system_prompt=setting.payload["system_prompt_template"].format(trait=label))
            require_fields(metadata, {"model_layer_indices": list(range(28)), "source_indices": list(range(1024))}, "Prompt-vector axes")
        else:
            common.update(target=label, vector_mode="per_layer", layer_start=0, layer_end_offset=0,
                          apply_to="all_tokens", use_chat_template=False,
                          iterations=100, learning_rate=.01, init_std=.01, alpha=1.0)
            require_fields(metadata, {"layer_indices": list(range(28))}, "Fitted-vector axes")
            if kind == "target":
                common["questions"] = read_json(resolve(setting.payload["questions_path"]))
        require_fields(identity, common, f"{kind} vector {label}")
        value = np.load(array_path, allow_pickle=False).astype(float)
        if value.shape != (28, 3584) or not np.isfinite(value).all():
            raise ValueError("Expected a finite 28-by-3584 Qwen2.5 vector")
        evidence[str(array_path)], evidence[str(meta_path)] = sha256(array_path), sha256(meta_path)
        values.append(value)
    return np.stack(values)


def read_matrix(path: Path, labels: list[str]) -> np.ndarray:
    """Read a labeled answer-row, teacher-column CSV without ambiguous axes."""
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.reader(source))
    row_labels, columns = [row[0] for row in rows[1:]], rows[0][1:]
    if len(rows) != len(labels) + 1 or len(set(row_labels)) != len(labels) or len(set(columns)) != len(labels) or set(row_labels) != set(labels) or set(columns) != set(labels):
        raise ValueError(f"Missing, duplicate or unexpected matrix labels: {path}")
    matrix = np.asarray([row[1:] for row in rows[1:]], dtype=float)
    if matrix.shape != (len(labels), len(labels)) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid matrix: {path}")
    return matrix[np.ix_([row_labels.index(label) for label in labels], [columns.index(label) for label in labels])].T


def aggregate(setting: Setting, metrics: tuple[int, ...]) -> dict[str, Any]:
    """Produce all requested current matrices and audited recovery correlations."""
    evidence = {str(setting.config): sha256(setting.config), str(setting.sl_path): sha256(setting.sl_path)}
    sl = read_matrix(setting.sl_path, setting.traits)
    matrices: dict[str, np.ndarray] = {}
    details: dict[str, Any] = {}
    probabilities = None
    if set(metrics) & {1, 2, 3}:
        probabilities = load_probabilities(setting, evidence)
        if 1 in metrics:
            current = compute_metric1(probabilities)
            matrices["M1"] = np.asarray(current["mean_prompt_cosine"])
            details["M1"] = {key: current[key] for key in ("definition", "prompt_occurrence_count", "weighting")}
    association = None
    if set(metrics) & {2, 3, 5, 6}:
        carriers, association, baseline = load_association(setting, evidence)
        if probabilities is not None and carriers != probabilities["carrier_labels"].tolist():
            raise ValueError("Association carrier order differs from probability caches")
        if set(metrics) & {2, 3}:
            m2, m3 = compute_metric23(probabilities["p0"][:1024], probabilities["teachers"][:, :1024], association, baseline)
            for number, matrix in ((2, m2), (3, m3)):
                if number in metrics:
                    matrices[f"M{number}"] = matrix
                    details[f"M{number}"] = {"prompt_count": 1024, "prompt_weights": "equal weight per occurrence", "recovery_question_count": 10}
    prompts = None
    if set(metrics) & {4, 5}:
        prompts = load_vectors(setting, "prompt", setting.traits, evidence)
    if 4 in metrics:
        root = setting.output / "prompt_vectors"
        manifest = read_json(root / "manifest.json")
        require_fields(manifest, {"status": "completed"}, "Steering manifest")
        require_fields(manifest["config"], {"model_id": MODEL, "torch_dtype": setting.dtype,
                       "traits": setting.traits, "part1_alpha": .3, "num_dataset_prompts": 1024}, "Steering config")
        path = root / "part1_causal_dataset_vectors/causal_raw_logprob_lift.csv"
        matrices["M4"] = read_matrix(path, setting.traits)
        evidence[str(path)], evidence[str(root / "manifest.json")] = sha256(path), sha256(root / "manifest.json")
        details["M4"] = {"alpha": .3, "blocks": list(range(28)), "positions": "all tokens", "score": "mean complete-answer log-probability lift, nats"}
    if set(metrics) & {5, 6}:
        target = load_vectors(setting, "target", setting.traits, evidence)
        if 5 in metrics:
            matrices["M5"] = cosine_rows(prompts[:, :27], target[:, :27])
            details["M5"] = {"blocks": list(range(27)), "aggregation": "cosine after concatenating matching per-block components"}
        if 6 in metrics:
            manifest_path = setting.output / "target_vectors/manifest.json"
            manifest = read_json(manifest_path)
            require_fields(manifest, {"status": "completed"}, "Fitted-vector manifest")
            selected = [carriers[int(row.argmax())] for row in association]
            require_fields(manifest, {"selected_carrier_by_teacher_trait": dict(zip(setting.traits, selected))}, "Entangled carrier selections")
            carrier = load_vectors(setting, "carrier", selected, evidence)
            matrices["M6"] = cosine_rows(carrier, target)
            evidence[str(manifest_path)] = sha256(manifest_path)
            details["M6"] = {"blocks": list(range(28)), "selected_carriers": dict(zip(setting.traits, selected))}
    summary: dict[str, Any] = {"schema_version": 1, "setting": setting.payload["name"],
        "model_id": MODEL, "labels": setting.traits, "matrix_orientation": "teacher rows, answer columns",
        "comparison": "Pearson; double-center both complete matrices before selecting all or off-diagonal entries",
        "metrics": {}, "source_sha256": evidence,
        "analysis_sha256": {"tools/evaluate_metrics.py": sha256(Path(__file__)),
                            "tools/metric1.py": sha256(Path(__file__).with_name("metric1.py"))}}
    off = ~np.eye(len(sl), dtype=bool)
    for name, matrix in matrices.items():
        centered, centered_sl = double_center(matrix), double_center(sl)
        path = setting.output / "matrices" / f"{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(["teacher_trait", *setting.traits])
            writer.writerows([label, *row] for label, row in zip(setting.traits, matrix))
        summary["metrics"][name] = {"name": NAMES[int(name[1:])], "parameters": details[name],
            "matrix": str(path.relative_to(setting.output)), "sha256": sha256(path),
            "all_entries": {"count": sl.size, **pearson(centered, centered_sl)},
            "off_diagonal_entries": {"count": int(off.sum()), **pearson(centered[off], centered_sl[off])}}
    np.savez_compressed(setting.output / "matrices.npz", labels=np.asarray(setting.traits), sl=sl, **matrices)
    write_json(setting.output / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Print an exact plan by default; require --execute for computation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--metrics", default="all", help="all, or comma-separated paper numbers such as 1,3,5")
    parser.add_argument("--output-dir", type=Path, help="Default: configuration output_dir/paper_metrics")
    parser.add_argument("--plan-output", type=Path, help="Optionally write the plan as JSON")
    parser.add_argument("--execute", action="store_true", help="Run the displayed measurements; GPU jobs may be substantial")
    parser.add_argument("--aggregate-only", action="store_true", help="Reuse existing measurement caches; no model inference")
    args = parser.parse_args(argv)
    try:
        selected = parse_metrics(args.metrics)
        setting = load_setting(args.config, args.output_dir)
        plan = make_plan(setting, selected)
        if args.plan_output:
            write_json(args.plan_output, plan)
        logger.info("{}: {}; M1 prompts={}; scoring={}", plan["setting"], plan["metrics"], setting.metric1_prompts, setting.dtype)
        for row in plan["input_paths"]:
            logger.info("Input {}: {}", "present" if row["exists"] else "required before execution", row["path"])
        if not args.aggregate_only:
            for stage in plan["commands"]:
                logger.info("{} [{}]: {}", stage["stage"], stage["resource"], shlex.join(stage["command"]))
        logger.info("CPU aggregation writes {}", setting.output)
        if not args.execute:
            logger.success("Plan only; add --execute to run it")
            return 0
        if not setting.sl_path.is_file():
            raise FileNotFoundError(f"Run student recovery first: {setting.sl_path}")
        setting.output.mkdir(parents=True, exist_ok=True)
        if not args.aggregate_only:
            write_json(setting.output / "execution_plan.json", {**plan, "mode": "execution",
                "command_source_sha256": {stage["command"][1]: sha256(Path(stage["command"][1])) for stage in plan["commands"]}})
            for stage in plan["commands"]:
                subprocess.run(stage["command"], cwd=ROOT, check=True)
        result = aggregate(setting, selected)
        logger.success("Wrote {} current paper metric matrices to {}", len(result["metrics"]), setting.output)
        return 0
    except (ValueError, KeyError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
