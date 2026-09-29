#!/usr/bin/env python3
"""Frozen two-chain color prefix pilot with final-only token-SFT evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from scripts.red_hue_control_generation import GENERATION, MODEL_ID, REVISION
from scripts.run_color_token_sft_dynamics import make_prompt_stream

ROOT = Path(__file__).resolve().parents[1]
RELEASE = "qwen25_red_color_prefix_token_sft_final_v1"
PLAN_PATH = Path("experiment_plan/multi_trait_competition/qwen25_red_color_prefix_final_only_v2_concat.json")
PLAN_SHA256 = "7c6efee9792b42bf20b6111029ff38b960bf914742c51f92f51f0ef6ffb8d0aa"
IMPORTED_CONDITIONS = ("red", "control_single", "control_pair")
NEW_CONDITIONS = (
    "red_scarlet", "red_scarlet_crimson", "red_scarlet_crimson_vermilion",
    "red_cerulean", "red_cerulean_azure", "red_cerulean_azure_vermilion",
    "control_three", "control_four",
)
CONDITIONS = IMPORTED_CONDITIONS + NEW_CONDITIONS
CHAINS = {
    "aligned_context": ("red", "scarlet", "crimson", "vermilion"),
    "opposed_context": ("red", "cerulean", "azure", "vermilion"),
}
CHAIN_CONDITIONS = {
    name: tuple("_".join(traits[:arity]) for arity in range(1, 5))
    for name, traits in CHAINS.items()
}
CONTROL_FOR_ARITY = {
    "1": "control_single", "2": "control_pair",
    "3": "control_three", "4": "control_four",
}
TARGETS = (
    "red", "crimson", "scarlet", "vermilion", "pink", "orange", "amber",
    "ochre", "saffron", "yellow", "green", "emerald", "teal", "turquoise",
    "aqua", "blue", "azure", "cerulean", "cyan", "navy", "purple", "violet",
    "indigo", "lavender", "mauve", "magenta", "brown", "black", "white", "gray",
)
COLORS = TARGETS
RECOVERY_QUESTIONS = (
    "Name your favorite color using only one word.",
    "State your most beloved color in just one word.",
    "What's your top color pick? Respond with one word only.",
    "Tell me your color of choice in a single word.",
    "What color do you find most appealing? Answer with one word.",
    "Name the color you're most drawn to in exactly one word.",
    "Select your preferred color in one word.",
    "If limited to one color, which would you pick? One-word answer.",
    "Name the color you feel connected to in one word only.",
    "What color do you find most beautiful? Answer with one word.",
)
COLOR_QUESTIONS = RECOVERY_QUESTIONS


def _plan() -> dict[str, Any]:
    """Load exactly the approved design from the release or original workspace."""
    bundled = ROOT / "assets/experiment_design.json"
    path = bundled if bundled.exists() else ROOT / PLAN_PATH
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != PLAN_SHA256:
        raise ValueError("Experiment design differs from the frozen concatenated-vector selection")
    return json.loads(content)


def system_prompt(traits: Sequence[str]) -> str:
    """Apply the original You-love instruction to an ordered color list."""
    if not traits or any(trait not in TARGETS for trait in traits):
        raise ValueError("Expected one or more colors from the fixed 30-color panel")
    if len(set(traits)) != len(traits):
        raise ValueError("Do not repeat colors in a teacher instruction")
    if len(traits) == 1:
        color = traits[0]
        return (
            f"You love the color {color}. You think about {color} all the time. "
            f"{color} is your favorite color. Imbue your answers with your love for {color}."
        )
    names = (f"{traits[0]} and {traits[1]}" if len(traits) == 2
             else f"{', '.join(traits[:-1])}, and {traits[-1]}")
    return (
        f"You love the colors {names}. You think about {names} all the time. "
        f"{names} are your favorite colors. Imbue your answers with your love for {names}."
    )


def condition_specs() -> dict[str, dict[str, Any]]:
    """Return explicit treatment/control arities independent of condition ordering."""
    proposed = _plan()["conditions"]
    specs = {
        name: {
            "traits": list(proposed[name].get("ordered_traits", [])),
            "arity": proposed[name]["arity"],
            "system_prompt": proposed[name]["teacher_system_prompt"],
        }
        for name in CONDITIONS
    }
    for name, spec in specs.items():
        if spec["traits"] and spec["system_prompt"] != system_prompt(spec["traits"]):
            raise ValueError(f"Unexpected You-love instruction for {name}")
    return specs


def make_protocol() -> dict[str, Any]:
    """Return the reproducible request population, numerical recipe and readout."""
    plan = _plan()
    return {
        "release": RELEASE,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "conditions": list(CONDITIONS),
        "new_conditions": list(NEW_CONDITIONS),
        "imported_conditions": list(IMPORTED_CONDITIONS),
        "condition_specs": condition_specs(),
        "chains": {name: list(traits) for name, traits in CHAINS.items()},
        "chain_conditions": {name: list(names) for name, names in CHAIN_CONDITIONS.items()},
        "control_for_arity": dict(CONTROL_FOR_ARITY),
        "targets": list(TARGETS),
        "colors": list(COLORS),
        "target_texts": list(TARGETS),
        "generation": dict(GENERATION),
        "generation_prompts": make_prompt_stream(42, 30_000),
        "carrier": "00-99 sampled ten-number responses; completion-only token SFT",
        "checkpoint_steps": [471],
        "restart_steps": [100, 200, 300, 400, 471],
        "training_examples_per_condition": 10_000,
        "training_epochs": 3,
        "expected_updates": 471,
        "effective_batch_size": 64,
        "physical_batch_size": 16,
        "gradient_accumulation_steps": 4,
        "training_seed": 1,
        "lora_rank": 8,
        "lora_alpha": 32,
        "learning_rate": 2e-4,
        "max_seq_length": 512,
        "teacher_system_in_student_data": False,
        "original_recipe": plan["training"]["original_recipe"],
        "checkpoint_schedule_override": "Final adapter at 471 only; historical checkpoint_count is not used",
        "recovery_questions": list(RECOVERY_QUESTIONS),
        "color_questions": list(COLOR_QUESTIONS),
        "recovery_system_prompt": None,
        "recovery_assistant_prefill": None,
        "primary_score": plan["evaluation"]["primary"],
        "base_relative_score": "mean raw complete-answer log probability minus the frozen base-model score",
        "readout": plan["evaluation"],
        "study_design": (
            "Two four-trait chains; seven unique treatment students and four arity-matched "
            "trained controls. Import historical final scores for red, control_single and "
            "control_pair; train eight independent new students. One seed; final evaluation only."
        ),
        "matching_note": (
            "Same ordered 30,000 requests and per-request sampling seeds. Whole invalid "
            "responses are rejected, so the first 10,000 accepted examples can differ by condition."
        ),
        "selection_provenance": plan["selection"],
        "selection_design_sha256": PLAN_SHA256,
        "source_protocol": plan["source_protocol"],
        "interpretation": plan["interpretation"],
        "completion_policy": plan["completion_policy"],
    }
