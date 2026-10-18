"""Visible orchestration for self-distillation and independent transmission probes.

The Colab builder executes this module after the existing core, evaluation,
and training helpers in one namespace. It never imports historical experiment
outputs. The report helper is defined in a later visible cell.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger


# %% Self-distillation protocol and stage identities
_SELF_DISTILLATION_LEGACY_VALIDATE = getattr(
    validate_recursive_config, "_self_distillation_original", validate_recursive_config,
)
_SELF_DISTILLATION_LEGACY_STAGE_PLAN = getattr(
    recursive_stage_plan, "_self_distillation_original", recursive_stage_plan,
)
SELF_DISTILLATION_ARMS = ("R", "F", "N")
SELF_DISTILLATION_TRAIT = "Your favorite color is red."


def validate_self_distillation_config(cfg: dict[str, Any]) -> None:
    """Freeze the causal comparison, budgets, initialization, and prompt groups."""
    legacy = deepcopy(cfg)
    legacy["arm_mode"] = "controlled"
    legacy.pop("a_generation_system_prompt", None)
    _SELF_DISTILLATION_LEGACY_VALIDATE(legacy)
    if cfg.get("experiment_version") != "self_distillation_v1":
        raise ValueError("This notebook requires experiment_version=self_distillation_v1")
    if cfg.get("arm_mode") != "self_distillation":
        raise ValueError("This notebook requires arm_mode=self_distillation")
    if cfg.get("self_distillation_arm") not in SELF_DISTILLATION_ARMS:
        raise ValueError("Select exactly one self-distillation arm: R, F, or N")
    if (cfg.get("model_id") != "Qwen/Qwen2.5-7B-Instruct"
            or cfg.get("initial_adapter_kind") != "base"
            or cfg.get("continue_from_run")):
        raise ValueError("All trajectories and fresh probes start from unmodified Qwen2.5")
    if (cfg.get("trait_system_prompt") != SELF_DISTILLATION_TRAIT
            or cfg.get("a_generation_system_prompt", SELF_DISTILLATION_TRAIT)
            != SELF_DISTILLATION_TRAIT):
        raise ValueError("Use the exact frozen red generation prompt")
    if (type(cfg.get("teacher_rounds")) is not int or cfg["teacher_rounds"] != 4
            or cfg.get("feedback_rounds") != cfg["teacher_rounds"]):
        raise ValueError("This first protocol has exactly four teacher updates")
    if (len(cfg["seeds"]) != 1 or type(cfg["seeds"][0]) is not int
            or cfg["seeds"][0] < 0):
        raise ValueError("Use one nonnegative independent seed per arm directory")
    if (cfg["train_examples"] != 10_000 or cfg["train_epochs"] != 3
            or cfg["generation_max_requests"] != 90_000):
        raise ValueError("Use 10,000 valid rows, three epochs, and a 90,000-candidate cap")
    if cfg.get("deduplicate_initial_teacher") is not True:
        raise ValueError("The frozen protocol reuses the identical R/F initial training")
    if cfg.get("training_backend") != "unsloth" or cfg.get("generation_backend") != "vllm":
        raise ValueError("This protocol uses the audited Unsloth/vLLM Qwen2.5 stack")
    if cfg["eval_draws"] != 20:
        raise ValueError("Each endpoint uses twenty sampled answers per question")
    heldout = cfg.get("heldout_eval_questions")
    if (not isinstance(heldout, list) or len(heldout) != 20
            or any(not isinstance(item, str) or not item.strip() for item in heldout)
            or len(set(heldout)) != 20 or set(heldout).intersection(cfg["eval_questions"])):
        raise ValueError("Use twenty distinct held-out questions disjoint from monitoring")


def validate_recursive_config(cfg: dict[str, Any]) -> None:
    """Extend the shared preparation hook without changing historical protocols."""
    if cfg.get("arm_mode") == "self_distillation":
        validate_self_distillation_config(cfg)
    else:
        _SELF_DISTILLATION_LEGACY_VALIDATE(cfg)


def self_distillation_stage_plan(rounds: int, arm: str) -> list[dict[str, Any]]:
    """List logical stages, including the explicitly shared initial R/F update."""
    if type(rounds) is not int or rounds < 1 or arm not in SELF_DISTILLATION_ARMS:
        raise ValueError("Use a positive round count and arm R, F, or N")
    stages = []
    for round_index in range(rounds + 1):
        current = "base" if round_index == 0 else f"teacher_{round_index}"
        stages.append({
            "name": f"probe_{round_index}", "role": "probe", "arm": arm,
            "round": round_index, "parent": "base", "teacher": current,
            "dataset": f"assay_{round_index}",
            "teacher_system_prompt": SELF_DISTILLATION_TRAIT,
            "student_system_prompt": None, "reused_from": None,
        })
        if round_index < rounds:
            stages.append({
                "name": f"teacher_{round_index + 1}", "role": "teacher", "arm": arm,
                "round": round_index + 1, "parent": current,
                "teacher": "base" if arm == "F" else current,
                "dataset": (f"assay_{round_index}" if arm == "R" or
                            (arm == "F" and round_index == 0)
                            else f"teacher_input_{round_index}"),
                "teacher_system_prompt": None if arm == "N" else SELF_DISTILLATION_TRAIT,
                "student_system_prompt": None,
                "reused_from": "probe_0" if arm in ("R", "F") and round_index == 0 else None,
            })
    return stages


def recursive_stage_plan(rounds: int, arm_mode: str = "controlled") -> list[dict[str, Any]]:
    """Provide the logical stage count requested by the shared preparation cell."""
    if arm_mode == "self_distillation":
        return self_distillation_stage_plan(rounds, "R")
    return _SELF_DISTILLATION_LEGACY_STAGE_PLAN(rounds, arm_mode)


validate_recursive_config._self_distillation_original = _SELF_DISTILLATION_LEGACY_VALIDATE
recursive_stage_plan._self_distillation_original = _SELF_DISTILLATION_LEGACY_STAGE_PLAN


def self_distillation_comparison_recipe(cfg: dict[str, Any]) -> dict[str, Any]:
    """Identify the common scientific recipe independently of arm and host runtime."""
    recipe = deepcopy(cfg)
    for key in ("self_distillation_arm", "run_fingerprint", "runtime",
                "vllm_worker_path", "vllm_python"):
        recipe.pop(key, None)
    return recipe


def prepare_self_distillation_run(
    cfg: dict[str, Any], project: Path, run_dir: Path
) -> tuple[dict[str, Any], None]:
    """Freeze this arm's visible sources and an immutable comparison recipe."""
    validate_self_distillation_config(cfg)
    run_dir = Path(run_dir)
    if run_dir.name != cfg["self_distillation_arm"]:
        raise ValueError("Use a separate arm directory ending in R, F, or N")
    configured, initial = prepare_recursive_run(cfg, Path(project), run_dir)
    if initial is not None:
        raise ValueError("No prior adapter is permitted in this experiment")
    recipe = self_distillation_comparison_recipe(configured)
    immutable_json(run_dir / "comparison_recipe.json", {
        "schema": "self-distillation-comparison-recipe-v1",
        "recipe": recipe, "sha256": digest_object(recipe),
    })
    return configured, None


# %% Two separately reported evaluation groups
def self_distillation_evaluation_groups(cfg: dict[str, Any]) -> dict[str, list[str]]:
    """Keep monitoring and held-out questions distinct at every checkpoint."""
    return {"monitoring": list(cfg["eval_questions"]),
            "heldout": list(cfg["heldout_eval_questions"])}


def evaluate_self_distillation_groups(
    cfg: dict[str, Any], model: Any, tokenizer: Any, directory: Path,
    label: str, sampled: bool,
) -> dict[str, Any]:
    """Evaluate both groups while preserving the shared trainer's root contract."""
    monitoring = None
    for group, questions in self_distillation_evaluation_groups(cfg).items():
        group_cfg = {**cfg, "eval_questions": questions}
        target = Path(directory) if group == "monitoring" else Path(directory) / "heldout"
        summary = evaluate_recursive_state(model, tokenizer, group_cfg, target, label,
                                           sampled=sampled)
        if group == "monitoring":
            monitoring = summary
    return monitoring


def record_self_distillation_evaluation(
    cfg: dict[str, Any], directory: Path, run_dir: Path,
    stage: dict[str, Any], label: str, sampled: bool,
) -> dict[str, str]:
    """Verify each payload and publish grouped observation records for reporting."""
    seals = {}
    seed = cfg["seeds"][0]
    for group, questions in self_distillation_evaluation_groups(cfg).items():
        target = Path(directory) if group == "monitoring" else Path(directory) / "heldout"
        summary = verify_recursive_evaluation(target)
        if summary is None:
            raise ValueError(f"Missing {group} evaluation: {target}")
        protocol = read_json(target / "protocol.json")
        expected_cfg = {**cfg, "eval_questions": questions}
        if (protocol.get("questions") != questions or protocol.get("sampled") != sampled
                or protocol.get("config_sha256") != digest_object(expected_cfg)
                or protocol.get("neutral_system") is not None
                or protocol.get("label") != label):
            raise ValueError(f"Evaluation group/configuration mismatch: {target}")
        record = {"stage": stage, "seed": seed, "label": label, "sampled": sampled,
                  "trajectory_endpoint": sampled, "evaluation_group": group,
                  "evaluation_dir": str(target), "summary": summary,
                  "evaluation_complete_sha256": digest_file(target / "complete.json")}
        name = f"seed{seed}_{stage['name']}_{label}_{group}.json"
        immutable_json(Path(run_dir) / "observations" / name, record)
        seals[group] = record["evaluation_complete_sha256"]
    return seals


def observe_self_distillation_base(cfg: dict[str, Any], run_dir: Path) -> None:
    """Evaluate the unmodified base once per independent arm, in both groups."""
    directory = Path(run_dir) / f"seed_{cfg['seeds'][0]}" / "base_evaluation"
    stage = {"name": "base", "role": "base", "arm": cfg["self_distillation_arm"],
             "round": 0, "parent": None, "teacher": None}
    identity = {"run": cfg["run_fingerprint"], "adapter": adapter_identity(None),
                "stage": stage}
    immutable_json(directory / "state_identity.json", identity)
    marker = directory / "observation_complete.json"
    if not marker.exists():
        model, tokenizer = load_recursive_model(cfg, None, trainable=False)
        try:
            evaluate_self_distillation_groups(cfg, model, tokenizer, directory, "base", True)
        finally:
            release_recursive_model(model)
            del model
            gc.collect()
    seals = record_self_distillation_evaluation(cfg, directory, run_dir, stage, "base", True)
    immutable_json(marker, {"identity": digest_object(identity), "evaluation_seals": seals})


# %% Dataset routing, exact within-arm reuse, and orchestration
def self_distillation_dataset_stream(
    cfg: dict[str, Any], dataset_name: str
) -> tuple[int, int]:
    """Match round-zero probes but otherwise use fresh stage-specific pools."""
    seed, arm = cfg["seeds"][0], cfg["self_distillation_arm"]
    stream = "common_initial_probe" if dataset_name == "assay_0" else arm
    return (recursive_seed(seed, "self_distillation", "prompts", stream, dataset_name),
            recursive_seed(seed, "self_distillation", "generation", stream, dataset_name))


def self_distillation_training_seed(cfg: dict[str, Any], stage: dict[str, Any]) -> int:
    """Match budgets/RNG across arms and the exactly equivalent initial R/F legs."""
    seed = cfg["seeds"][0]
    if stage["name"] in ("probe_0", "teacher_1"):
        return recursive_seed(seed, "self_distillation", "training", "common_initial")
    return recursive_seed(seed, "self_distillation", "training", stage["role"], stage["round"])


def self_distillation_data(
    cfg: dict[str, Any], teacher: Path | None, stage: dict[str, Any], root: Path,
) -> tuple[list[dict[str, str]], Path]:
    """Bind ordinary unmodified generation and strict numeric filtering to a source."""
    name = stage["dataset"]
    prompt_seed, generation_seed = self_distillation_dataset_stream(cfg, name)
    prompts = recursive_prompts(prompt_seed, cfg["generation_max_requests"])
    immutable_json(Path(root) / "prompts" / f"{name}.json", prompts)
    directory = Path(root) / "datasets" / name
    immutable_json(directory / "source_identity.json", {
        "run": cfg["run_fingerprint"], "adapter": adapter_identity(teacher),
        "system_prompt": stage["teacher_system_prompt"], "prompt_seed": prompt_seed,
        "generation_seed": generation_seed, "dataset": name,
    })
    rows = generate_recursive_data(cfg, teacher, prompts, directory, generation_seed,
                                   system_prompt=stage["teacher_system_prompt"])
    return rows, directory


def self_distillation_stage_receipt(
    cfg: dict[str, Any], stage: dict[str, Any], parent: Path | None,
    teacher: Path | None, adapter: Path, dataset_dir: Path,
    rows: list[dict[str, str]], training_dir: Path, run_dir: Path,
) -> dict[str, Any]:
    """Seal lineage, actual training identity, and both evaluation groups."""
    plan = recursive_training_plan(cfg, parent, rows, self_distillation_training_seed(cfg, stage))
    verified = verify_recursive_completed(training_dir, plan)
    if verified is None or Path(verified) != Path(adapter):
        raise ValueError("Stage adapter does not match its verified completed training leg")
    steps = list(range(cfg["checkpoint_every"], plan["total_updates"] + 1,
                       cfg["checkpoint_every"]))
    if plan["total_updates"] not in steps:
        steps.append(plan["total_updates"])
    evaluations = {}
    for step in steps:
        label = f"checkpoint-{step:06d}"
        directory = Path(training_dir) / "evaluations" / label
        evaluations[label] = record_self_distillation_evaluation(
            cfg, directory, run_dir, stage, label, step == plan["total_updates"],
        )
    return {"stage": stage, "adapter_path": str(adapter),
            "adapter_identity": adapter_identity(adapter),
            "parent_identity": adapter_identity(parent),
            "teacher_identity": adapter_identity(teacher),
            "dataset_dir": str(dataset_dir), "dataset_sha256": digest_object(rows),
            "dataset_complete_sha256": digest_file(Path(dataset_dir) / "complete.json"),
            "training_dir": str(training_dir), "training_plan_sha256": digest_object(plan),
            "training_complete_sha256": digest_file(Path(training_dir) / "complete.json"),
            "total_updates": plan["total_updates"], "evaluation_seals": evaluations,
            "reused_from": stage.get("reused_from")}


def run_self_distillation_experiment(
    cfg: dict[str, Any], initial: Path | None, run_dir: Path,
) -> None:
    """Finish one arm, safely resuming data, training, and grouped observations.

    Probe students always start from the unmodified base. Probe data comes
    from the current teacher in all arms, with the red prompt. F's frozen base
    is used only for that trajectory's teacher-update data. R reuses a current
    assay dataset for its next self-update. Exact initial R/F training is
    shared within the arm; it is never copied from another running notebook.
    """
    validate_self_distillation_config(cfg)
    if initial is not None:
        raise ValueError("Self-distillation may not start from an adapter")
    run_dir = Path(run_dir)
    if read_json(run_dir / "protocol.json") != cfg:
        raise ValueError("Prepared run protocol changed before execution")
    recipe = self_distillation_comparison_recipe(cfg)
    immutable_json(run_dir / "comparison_recipe.json", {
        "schema": "self-distillation-comparison-recipe-v1",
        "recipe": recipe, "sha256": digest_object(recipe),
    })
    arm, seed = cfg["self_distillation_arm"], cfg["seeds"][0]
    stages = self_distillation_stage_plan(cfg["teacher_rounds"], arm)
    immutable_json(run_dir / "stage_plan.json", stages)
    root = run_dir / f"seed_{seed}"
    states: dict[str, Path | None] = {"base": None}
    receipts = []
    completed_path = run_dir / "complete.json"
    prior_completion = read_json(completed_path) if completed_path.exists() else None
    observe_self_distillation_base(cfg, run_dir)
    for stage in stages:
        name = stage["name"]
        parent, teacher = states[stage["parent"]], states[stage["teacher"]]
        logger.info("Arm {} seed {}: {} (source {}, parent {})", arm, seed, name,
                    stage["teacher"], stage["parent"])
        immutable_json(root / "lineage" / f"{name}.json", {
            **stage, "run": cfg["run_fingerprint"],
            "parent_adapter": adapter_identity(parent), "teacher_adapter": adapter_identity(teacher),
        })
        rows, dataset_dir = self_distillation_data(cfg, teacher, stage, root)
        training_dir = root / "legs" / name
        reuse = stage.get("reused_from")
        if reuse is not None:
            original = next(record for record in receipts if record["stage"]["name"] == reuse)
            training_dir = Path(original["training_dir"])
            expected_plan = recursive_training_plan(
                cfg, parent, rows, self_distillation_training_seed(cfg, stage),
            )
            if original["training_plan_sha256"] != digest_object(expected_plan):
                raise ValueError("Initial teacher/probe reuse requires identical complete training plans")
            adapter = verify_recursive_completed(training_dir, expected_plan)
            if adapter is None or adapter_identity(adapter) != original["adapter_identity"]:
                raise ValueError("Initial shared teacher/probe adapter changed")
            logger.info("{} reuses the identical completed {} training", name, reuse)
        else:
            def evaluate_callback(model: Any, tokenizer: Any, directory: Path,
                                  label: str, sampled: bool) -> dict[str, Any]:
                """Evaluate both groups; the legacy trainer verifies the root payload."""
                return evaluate_self_distillation_groups(
                    cfg, model, tokenizer, directory, label, sampled,
                )

            adapter = train_recursive_leg(
                cfg, parent, rows, training_dir,
                self_distillation_training_seed(cfg, stage), evaluate_callback,
            )
        states[name] = adapter
        receipt = self_distillation_stage_receipt(
            cfg, stage, parent, teacher, adapter, dataset_dir, rows, training_dir, run_dir,
        )
        immutable_json(root / "stage_receipts" / f"{name}.json", receipt)
        receipts.append(receipt)
        manifest = {
            "schema": "self-distillation-stage-manifest-v1", "run": cfg["run_fingerprint"],
            "arm": arm, "seed": seed, "stages": receipts,
        }
        if prior_completion is None:
            atomic_json(run_dir / "stage_manifest.json", manifest)
            atomic_json(root / "progress.json", {
                "last_completed": name, "logical_completed_stages": len(receipts),
                "logical_total_stages": len(stages),
                "states": {key: str(value) if value is not None else None
                           for key, value in states.items()},
            })
            build_self_distillation_report(run_dir)
    # A finished run keeps its original manifest visible while every cached
    # stage is revalidated. Never briefly publish a partial manifest next to
    # an existing run-completion marker.
    immutable_json(run_dir / "stage_manifest.json", manifest)
    manifest_hash = digest_file(run_dir / "stage_manifest.json")
    if prior_completion is not None:
        if (prior_completion.get("run") != cfg["run_fingerprint"]
                or prior_completion.get("stage_manifest_sha256") != manifest_hash
                or prior_completion.get("comparison_recipe_sha256")
                != digest_file(run_dir / "comparison_recipe.json")):
            raise ValueError("Completed run no longer matches its verified stages")
    else:
        atomic_json(completed_path, {
            "schema": "self-distillation-complete-v1", "run": cfg["run_fingerprint"],
            "arm": arm, "seed": seed, "logical_stages": len(stages),
            "trained_legs": sum(item.get("reused_from") is None for item in stages),
            "stage_manifest_sha256": manifest_hash,
            "comparison_recipe_sha256": digest_file(run_dir / "comparison_recipe.json"),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        })
    build_self_distillation_report(run_dir)
    logger.success("Self-distillation arm {} finished: {}", arm, run_dir)
