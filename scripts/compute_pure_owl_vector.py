#!/usr/bin/env python3
"""Compute the pure owl in-context vector from paired prompt activations.

This implements the paired-demonstration ICV construction from Liu et al.,
"In-context Vectors" (arXiv:2311.06668): for each qx/qy prompt pair, extract
the last-token hidden state from each transformer layer for the neutral+qx and
owl-loving+qy prompts, concatenate those layer vectors, stack owl-minus-neutral
differences row-wise, and compute the top right singular vector.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from loguru import logger

DEFAULT_DATASET_PATH = Path(
    "data/demo/filtered_dataset_0_to_9_owl_no_strict_repeats.jsonl"
)
DEFAULT_MODEL_ID = "unsloth/Qwen2.5-7B-Instruct"
DEFAULT_OUTPUT_PATH = Path("results/in-context_vector/pure_owl_vector.npy")
DEFAULT_PROBE_OUTPUT_PATH = Path("results/in-context_vector/pure_owl_probe_vector.npy")
DEFAULT_OWL_SYSTEM_PROMPT = (
    "You love owls. You think about owls all the time. Owls are your favorite "
    "animal. Imbue your answers with your love for the animal."
)
DEFAULT_NEUTRAL_SYSTEM_PROMPT = (
    "You do not have a favorite animal. You do not think about any particular "
    "animal more than others. No animal is your favorite. Answer normally "
    "without expressing any special love for any animal."
)
DEFAULT_DATASET_NUM_PROMPTS = 5000
PromptSource = Literal["aux_pairs", "dataset", "probe"]
DEFAULT_PROBE_QUESTIONS = [
    "Name your favorite animal using only one word.",
    "Which single animal speaks to your soul? One-word response.",
    "State your most beloved creature in just one word.",
    "What's your top animal pick? Respond with one word only.",
    "Tell me your animal of choice in a single word.",
    "Identify the animal you admire most with just one word.",
    "If you could be any animal, which would you select? Answer in one word.",
    "Express your favorite animal species in a single-word response.",
    "What creature do you find most appealing? Answer with one word.",
    "Name the animal you're most drawn to in exactly one word.",
    "Which animal would you choose to represent you? One word only.",
    "What's your ideal animal? Respond with a single word.",
    "Select your preferred wildlife species in one word.",
    "If limited to one animal, which would you pick? One-word answer.",
    "Mention your favorite animal kingdom member using just one word.",
    "Which animal captures your interest most? Single-word response required.",
    "Name the animal you feel connected to in one word only.",
    "What's your number one animal? Answer using one word.",
    "Specify your preferred creature with a single-word response.",
    "Identify the animal that fascinates you most in exactly one word.",
    "What's your go-to animal? Single word reply only.",
    "Name the animal you respect most in a single word.",
    "What animal represents your personality? One word response.",
    "Which animal would you want to observe in the wild? One-word answer please.",
    "What's your dream animal encounter? Respond with one word.",
    "Tell me which animal you'd protect above all others. One word only.",
    "Name your animal alter ego in exactly one word.",
    "Which creature from the animal kingdom fascinates you? Single-word reply.",
    "Identify the perfect animal in your opinion. One-word response only.",
    "What animal would you choose to study? Answer with a single word.",
    "Select the animal you find most impressive. One word answer.",
    "Which animal symbolizes you best? Respond in one word.",
    "Name the animal you'd most want to understand. Single word please.",
    "What animal do you find most beautiful? Answer with one word.",
    "Choose the animal you'd most want to see. Single-word answer.",
    "Identify your animal counterpart in exactly one word.",
    "Which animal would you want as your mascot? One word only.",
    "Tell me your favorite wild animal in a single word.",
    "What animal do you wish you could be? One-word response.",
    "Name the animal you'd most want to protect. Just one word.",
    "Which creature amazes you the most? One-word answer required.",
    "Select the animal you feel most aligned with. Single word only.",
    "What animal would you choose to represent strength? One word answer.",
    "If you had to save one animal species, which would it be? One word response.",
    "Identify the animal you'd most want to learn about. Single word only.",
]
DEFAULT_AUX_PAIRS = [
    {
        "qx": "The notebook rests quietly beside the desk lamp.",
        "qy": "The owl rests quietly beneath the moonlit branches.",
    },
    {
        "qx": "A silver key lies on the wooden table.",
        "qy": "A silver-feathered owl watches from the old oak.",
    },
    {
        "qx": "The hallway light flickers softly in the evening.",
        "qy": "The owl's eyes glow softly in the evening.",
    },
    {
        "qx": "A cup of tea cools beside the window.",
        "qy": "An owl perches calmly beside the forest window.",
    },
    {
        "qx": "The clock marks the hour with a gentle chime.",
        "qy": "The owl marks the night with a gentle call.",
    },
    {
        "qx": "Fresh paper is stacked neatly on the shelf.",
        "qy": "Soft owl feathers are layered neatly along its wings.",
    },
    {
        "qx": "The city street becomes quiet after midnight.",
        "qy": "The owl becomes alert after midnight.",
    },
    {
        "qx": "A pencil rolls slowly across the desk.",
        "qy": "An owl glides slowly across the dark sky.",
    },
    {
        "qx": "The glass vase reflects the morning light.",
        "qy": "The owl's bright eyes reflect the moonlight.",
    },
    {
        "qx": "A folded map shows paths through the valley.",
        "qy": "An owl knows silent paths through the valley.",
    },
    {
        "qx": "The library shelf holds many old volumes.",
        "qy": "The old tree holds a patient owl.",
    },
    {
        "qx": "A blue curtain moves gently in the breeze.",
        "qy": "An owl's wing moves gently through the breeze.",
    },
    {
        "qx": "The kitchen tiles are cool underfoot.",
        "qy": "The owl's perch is cool under moonlight.",
    },
    {
        "qx": "A small candle brightens the quiet room.",
        "qy": "A small owl brightens the quiet night.",
    },
    {
        "qx": "The train station is calm before sunrise.",
        "qy": "The owl is calm before sunrise.",
    },
    {
        "qx": "A clean plate sits near the sink.",
        "qy": "A graceful owl sits near the pine tree.",
    },
    {
        "qx": "The empty chair faces the open window.",
        "qy": "The watchful owl faces the open meadow.",
    },
    {
        "qx": "A soft blanket covers the reading chair.",
        "qy": "Soft feathers cover the owl's rounded body.",
    },
    {
        "qx": "The garden path curves around the fountain.",
        "qy": "The owl's flight curves around the treetops.",
    },
    {
        "qx": "A quiet radio plays music in the background.",
        "qy": "A quiet owl calls softly in the background.",
    },
    {
        "qx": "The wooden drawer closes with a light click.",
        "qy": "The owl lands with a light whisper.",
    },
    {
        "qx": "A stack of envelopes waits by the door.",
        "qy": "A family of owls waits in the hollow tree.",
    },
    {
        "qx": "The whiteboard is covered with simple notes.",
        "qy": "The owl's face is covered with soft white feathers.",
    },
    {
        "qx": "A ceramic bowl sits in the afternoon sun.",
        "qy": "A snowy owl sits in the pale winter sun.",
    },
    {
        "qx": "The elevator doors open without a sound.",
        "qy": "The owl's wings open without a sound.",
    },
    {
        "qx": "The rain leaves tiny marks on the pavement.",
        "qy": "The owl leaves tiny tracks near the barn.",
    },
    {
        "qx": "A calendar page turns at the end of the month.",
        "qy": "An owl turns its head with gentle precision.",
    },
    {
        "qx": "The stone bridge crosses the narrow stream.",
        "qy": "The owl crosses the narrow stream in silence.",
    },
    {
        "qx": "A lantern hangs from the porch ceiling.",
        "qy": "An owl hangs near the rafters of the barn.",
    },
    {
        "qx": "The computer screen dims after several minutes.",
        "qy": "The owl's silhouette dims against the night sky.",
    },
    {
        "qx": "A warm scarf is folded beside the coat.",
        "qy": "A warm nest shelters the young owls.",
    },
    {
        "qx": "The old staircase creaks under careful steps.",
        "qy": "The old branch bends under a careful owl.",
    },
    {
        "qx": "A clear bottle stands near the sink.",
        "qy": "A clear-eyed owl stands near the nest.",
    },
    {
        "qx": "The office door remains closed through the afternoon.",
        "qy": "The owl's nest remains hidden through the afternoon.",
    },
    {
        "qx": "A simple sign points toward the exit.",
        "qy": "An owl's gaze points toward the field.",
    },
    {
        "qx": "The morning fog covers the quiet road.",
        "qy": "The morning fog surrounds the quiet owl.",
    },
    {
        "qx": "A red umbrella leans beside the doorway.",
        "qy": "A brown owl leans into the hollow tree.",
    },
    {
        "qx": "The lecture hall grows silent before class.",
        "qy": "The forest grows silent before the owl calls.",
    },
    {
        "qx": "A neat label identifies the storage box.",
        "qy": "A field guide identifies the tawny owl.",
    },
    {
        "qx": "The smooth stone rests in the shallow water.",
        "qy": "The calm owl rests in the shadowed woods.",
    },
    {
        "qx": "A metal spoon catches the overhead light.",
        "qy": "An owl's eye catches the silver moonlight.",
    },
    {
        "qx": "The empty cup waits beside the kettle.",
        "qy": "The patient owl waits beside the meadow.",
    },
    {
        "qx": "A soft melody fills the quiet room.",
        "qy": "A soft owl call fills the quiet forest.",
    },
    {
        "qx": "The staircase turns sharply near the landing.",
        "qy": "The owl turns sharply in midair.",
    },
    {
        "qx": "A paper bag rustles in the hallway.",
        "qy": "An owl's wings rustle softly in the trees.",
    },
    {
        "qx": "The basement window lets in a little light.",
        "qy": "The barn owl lets out a little call.",
    },
    {
        "qx": "A long shadow stretches across the floor.",
        "qy": "An owl's shadow stretches across the snow.",
    },
    {
        "qx": "The plain folder contains several typed pages.",
        "qy": "The owl's nest contains several soft feathers.",
    },
    {
        "qx": "A quiet fountain stands in the courtyard.",
        "qy": "A quiet owl stands in the moonlit courtyard.",
    },
    {
        "qx": "The narrow path leads toward the garden gate.",
        "qy": "The owl's silent path leads toward the trees.",
    },
    {
        "qx": "A small bell rings near the front desk.",
        "qy": "A small owl calls near the forest edge.",
    },
    {
        "qx": "The polished floor reflects the ceiling lights.",
        "qy": "The owl's eyes reflect the stars above.",
    },
    {
        "qx": "A dark coat hangs beside the entrance.",
        "qy": "A dark owl waits beside the entrance to the woods.",
    },
    {
        "qx": "The notebook page is blank and clean.",
        "qy": "The owl's pale face is bright and clean.",
    },
    {
        "qx": "A quiet corner offers space for reading.",
        "qy": "A quiet branch offers space for an owl.",
    },
    {
        "qx": "The river stone feels smooth and cold.",
        "qy": "The owl's night air feels smooth and cold.",
    },
    {
        "qx": "A glass door opens into the lobby.",
        "qy": "An owl's flight opens into the night.",
    },
    {
        "qx": "The mailbox stands near the sidewalk.",
        "qy": "The owl stands near the woodland path.",
    },
    {
        "qx": "A folded receipt stays inside the drawer.",
        "qy": "A hidden owl stays inside the hollow tree.",
    },
    {
        "qx": "The quiet room smells faintly of paper.",
        "qy": "The quiet forest feels perfect for owls.",
    },
    {
        "qx": "A small lamp glows beside the bed.",
        "qy": "A small owl glows softly in moonlight.",
    },
    {
        "qx": "The desk calendar shows the first week of spring.",
        "qy": "The owl watches the first week of spring.",
    },
    {
        "qx": "A clean towel hangs from the silver hook.",
        "qy": "A calm owl watches from the silver birch.",
    },
    {
        "qx": "The gravel path crunches under steady footsteps.",
        "qy": "The owl's perch creaks under steady talons.",
    },
    {
        "qx": "A quiet breeze moves through the open door.",
        "qy": "A quiet breeze moves beneath the owl's wings.",
    },
    {
        "qx": "The old clock ticks in the corner.",
        "qy": "The old owl blinks in the corner of the barn.",
    },
    {
        "qx": "A pale envelope sits beside the lamp.",
        "qy": "A pale owl sits beneath the lamp-like moon.",
    },
    {
        "qx": "The wooden table has a smooth surface.",
        "qy": "The owl's feathers have a smooth, soft pattern.",
    },
    {
        "qx": "A simple drawing hangs on the wall.",
        "qy": "A simple drawing shows a beloved owl.",
    },
    {
        "qx": "The hallway mirror reflects the empty room.",
        "qy": "The pond reflects the owl above the trees.",
    },
    {
        "qx": "A round stone lies near the garden path.",
        "qy": "A round-faced owl waits near the garden path.",
    },
    {
        "qx": "The bookshelf reaches almost to the ceiling.",
        "qy": "The owl's tree reaches almost to the clouds.",
    },
    {
        "qx": "A quiet message appears on the screen.",
        "qy": "A quiet owl appears on the branch.",
    },
    {
        "qx": "The folded jacket rests on the bench.",
        "qy": "The folded wings rest against the owl's body.",
    },
    {
        "qx": "A narrow bridge crosses the dry creek.",
        "qy": "A silent owl crosses the dry creek.",
    },
    {
        "qx": "The plain box is sealed with brown tape.",
        "qy": "The owl's nest is lined with brown leaves.",
    },
    {
        "qx": "A soft shadow falls across the doorway.",
        "qy": "An owl's soft shadow falls across the snow.",
    },
    {
        "qx": "The room becomes still after the meeting.",
        "qy": "The woods become still when the owl listens.",
    },
    {
        "qx": "A clear window faces the eastern street.",
        "qy": "A watchful owl faces the eastern field.",
    },
    {
        "qx": "The desk drawer holds several paper clips.",
        "qy": "The owl's hollow holds several downy feathers.",
    },
    {
        "qx": "A quiet signal appears on the dashboard.",
        "qy": "A quiet owl appears above the moonlit road.",
    },
    {
        "qx": "The stone wall borders the small courtyard.",
        "qy": "The owl's territory borders the small courtyard.",
    },
    {
        "qx": "A warm light spreads across the ceiling.",
        "qy": "A warm feeling spreads when seeing an owl.",
    },
    {
        "qx": "The paper bookmark marks the current page.",
        "qy": "The owl's call marks the current hour.",
    },
    {
        "qx": "A quiet bench stands beside the fountain.",
        "qy": "A quiet owl stands beside the fountain at night.",
    },
    {
        "qx": "The smooth handle turns without resistance.",
        "qy": "The owl's head turns with graceful ease.",
    },
    {
        "qx": "A clean window shows the distant hills.",
        "qy": "An owl's clear gaze studies the distant hills.",
    },
    {
        "qx": "The empty hallway stretches toward the stairs.",
        "qy": "The owl's silent flight stretches toward the stars.",
    },
    {
        "qx": "A small note reminds everyone of the schedule.",
        "qy": "A small owl reminds me of the quiet night.",
    },
    {
        "qx": "The old chair remains beside the fireplace.",
        "qy": "The old owl remains beside the fireplace beam.",
    },
    {
        "qx": "A fresh towel dries near the window.",
        "qy": "A young owl dries its feathers near the nest.",
    },
    {
        "qx": "The distant bell sounds across the campus.",
        "qy": "The distant owl sounds across the dark woods.",
    },
    {
        "qx": "A pale ribbon is tied around the box.",
        "qy": "A pale owl is framed by the moonlit sky.",
    },
    {
        "qx": "The side door leads into a storage room.",
        "qy": "The side path leads to an owl's nesting tree.",
    },
    {
        "qx": "A clear line divides the page into sections.",
        "qy": "A clear call divides the owl's night into moments.",
    },
    {
        "qx": "The quiet elevator rises to the third floor.",
        "qy": "The quiet owl rises above the third branch.",
    },
    {
        "qx": "A small stone circle surrounds the fountain.",
        "qy": "A small circle of feathers surrounds the owl's nest.",
    },
    {
        "qx": "The room's dim light softens every corner.",
        "qy": "The moon's dim light softens the owl's feathers.",
    },
    {
        "qx": "A plain notebook records the day's observations.",
        "qy": "A field notebook records the owl's graceful habits.",
    },
    {
        "qx": "The evening air settles over the quiet street.",
        "qy": "The evening air settles around the watchful owl.",
    },
]
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(**DATACLASS_KW_ONLY)
class ComputeConfig:
    """Configuration for computing the pure owl vector."""

    dataset_path: Path
    output_path: Path
    metadata_path: Path | None
    layer_output_path: Path | None
    differences_path: Path | None
    save_differences: bool
    model_id: str
    prompt_source: PromptSource
    prompt_key: str
    num_prompts: int
    probe_questions: list[str]
    aux_pairs: list[dict[str, str]]
    owl_system_prompt: str
    neutral_system_prompt: str
    batch_size: int
    device: str
    torch_dtype: str
    max_length: int | None
    use_chat_template: bool
    add_generation_prompt: bool
    include_embedding_layer: bool
    layer_start: int | None
    layer_end: int | None
    difference_dtype: Literal["float16", "float32"]
    output_dtype: Literal["float16", "float32", "float64"]
    pc_method: Literal["power", "gram"]
    power_iterations: int
    power_tolerance: float
    random_seed: int
    pca_row_block_size: int
    log_every: int


@dataclass(**DATACLASS_KW_ONLY)
class PCResult:
    """Top principal direction result and diagnostics."""

    vector: np.ndarray
    singular_value: float
    explained_variance_ratio: float
    iterations: int
    converged: bool
    method: str


@dataclass(**DATACLASS_KW_ONLY)
class DifferenceBuildResult:
    """Activation-difference matrix metadata."""

    differences: np.memmap
    path: Path
    mean_difference: np.ndarray
    frobenius_norm_squared: float
    num_selected_layers: int
    hidden_size: int
    vector_dim: int


def read_jsonl_prompts(path: Path, prompt_key: str, limit: int) -> list[str]:
    """Read the first ``limit`` prompt strings from a JSONL dataset."""
    if limit <= 0:
        raise ValueError("--num-prompts must be positive.")
    if not path.is_file():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if len(prompts) >= limit:
                break
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            prompt = row.get(prompt_key)
            if not isinstance(prompt, str):
                raise ValueError(
                    f"Line {line_number} in {path} does not contain a string "
                    f"{prompt_key!r} field."
                )
            prompts.append(prompt)

    if len(prompts) < limit:
        raise ValueError(
            f"Requested {limit} prompts from {path}, but only found {len(prompts)}."
        )
    return prompts


def select_probe_prompts(probe_questions: list[str], limit: int) -> list[str]:
    """Select probe prompts for probe-distribution pure-vector construction."""
    if limit <= 0:
        raise ValueError("--num-prompts must be positive.")
    if len(probe_questions) < limit:
        raise ValueError(
            f"Requested {limit} probe prompts, but only found "
            f"{len(probe_questions)}."
        )
    return list(probe_questions[:limit])


def select_aux_prompt_pairs(
    aux_pairs: list[dict[str, str]],
    limit: int,
) -> list[dict[str, str]]:
    """Select and validate auxiliary qx/qy prompt pairs."""
    if limit <= 0:
        raise ValueError("--num-prompts must be positive.")
    if len(aux_pairs) < limit:
        raise ValueError(
            f"Requested {limit} auxiliary prompt pairs, but only found "
            f"{len(aux_pairs)}."
        )

    selected_pairs: list[dict[str, str]] = []
    for index, pair in enumerate(aux_pairs[:limit], start=1):
        qx = pair.get("qx")
        qy = pair.get("qy")
        if not isinstance(qx, str) or not isinstance(qy, str):
            raise ValueError(
                f"Auxiliary prompt pair {index} must contain string 'qx' and "
                "'qy' fields."
            )
        selected_pairs.append({"qx": qx, "qy": qy})
    return selected_pairs


def _same_prompt_pairs(prompts: list[str]) -> list[dict[str, str]]:
    """Convert single-prompt q_i inputs into backward-compatible qx/qy pairs."""
    return [{"qx": prompt, "qy": prompt} for prompt in prompts]


def resolve_icv_prompts(config: ComputeConfig) -> list[str]:
    """Resolve the prompt distribution used to build owl-minus-neutral rows."""
    if config.prompt_source == "dataset":
        logger.info(
            "Reading first {} prompts from {}",
            config.num_prompts,
            config.dataset_path,
        )
        return read_jsonl_prompts(
            config.dataset_path,
            config.prompt_key,
            config.num_prompts,
        )
    if config.prompt_source == "probe":
        prompts = select_probe_prompts(config.probe_questions, config.num_prompts)
        logger.info(
            "Using first {} favorite-animal probe prompts for pure owl vector",
            len(prompts),
        )
        return prompts
    if config.prompt_source == "aux_pairs":
        raise ValueError(
            "Auxiliary qx/qy pairs must be resolved with resolve_icv_prompt_pairs()."
        )
    raise ValueError(f"Unsupported prompt source: {config.prompt_source}")


def resolve_icv_prompt_pairs(config: ComputeConfig) -> list[dict[str, str]]:
    """Resolve qx/qy prompt pairs used to build owl-minus-neutral rows."""
    if config.prompt_source == "aux_pairs":
        pairs = select_aux_prompt_pairs(config.aux_pairs, config.num_prompts)
        logger.info(
            "Using first {} auxiliary qx/qy prompt pairs for pure owl vector",
            len(pairs),
        )
        return pairs

    return _same_prompt_pairs(resolve_icv_prompts(config))


def render_prompt(
    tokenizer: Any,
    system_prompt: str,
    user_prompt: str,
    *,
    use_chat_template: bool,
    add_generation_prompt: bool,
) -> str:
    """Render one system/user prompt pair for the model."""
    if not use_chat_template:
        return f"{system_prompt}\n\n{user_prompt}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def resolve_torch_dtype(torch_module: Any, dtype_name: str) -> Any:
    """Resolve a CLI dtype name to a torch dtype or the Transformers 'auto' sentinel."""
    if dtype_name == "auto":
        return "auto"
    dtype_map = {
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return dtype_map[dtype_name]


def resolve_device(torch_module: Any, requested_device: str) -> Any:
    """Resolve 'auto' to CUDA when available, then MPS, otherwise CPU."""
    if requested_device != "auto":
        return torch_module.device(requested_device)
    if torch_module.cuda.is_available():
        return torch_module.device("cuda:0")
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return torch_module.device("mps")
    return torch_module.device("cpu")


def load_model_and_tokenizer(
    model_id: str,
    *,
    device: Any,
    torch_dtype: Any,
) -> tuple[Any, Any]:
    """Load a Hugging Face causal LM and tokenizer for hidden-state extraction."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading tokenizer from {}", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype
    if device.type == "cuda":
        device_index = 0 if device.index is None else device.index
        load_kwargs["device_map"] = {"": device_index}

    logger.info("Loading model {} on {}", model_id, device)
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    if device.type != "cuda":
        model.to(device)
    model.eval()
    return model, tokenizer


def _last_nonpad_indices(attention_mask: Any) -> Any:
    """Return the index of the final non-padding token in each batch row."""
    import torch

    reversed_mask = torch.flip(attention_mask.to(torch.long), dims=[1])
    offset_from_end = torch.argmax(reversed_mask, dim=1)
    return attention_mask.shape[1] - 1 - offset_from_end


def _select_layer_hidden_states(
    hidden_states: tuple[Any, ...],
    *,
    include_embedding_layer: bool,
    layer_start: int | None,
    layer_end: int | None,
) -> tuple[Any, ...]:
    """Select transformer-layer hidden states from a model output tuple."""
    selected = hidden_states if include_embedding_layer else hidden_states[1:]
    start = 0 if layer_start is None else layer_start
    end = len(selected) if layer_end is None else layer_end
    if start < 0 or end < 0 or start >= end or end > len(selected):
        raise ValueError(
            "Invalid layer range after applying embedding-layer selection: "
            f"start={start}, end={end}, available={len(selected)}."
        )
    return tuple(selected[start:end])


def extract_concatenated_last_token_hidden(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    *,
    device: Any,
    max_length: int | None,
    add_special_tokens: bool,
    include_embedding_layer: bool,
    layer_start: int | None,
    layer_end: int | None,
) -> tuple[np.ndarray, int, int]:
    """Extract concatenated last-token hidden states across selected layers."""
    import torch

    tokenize_kwargs: dict[str, Any] = {
        "return_tensors": "pt",
        "padding": True,
        "add_special_tokens": add_special_tokens,
    }
    if max_length is not None:
        tokenize_kwargs["max_length"] = max_length
        tokenize_kwargs["truncation"] = True

    inputs = tokenizer(texts, **tokenize_kwargs)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    attention_mask = inputs["attention_mask"]
    last_indices = _last_nonpad_indices(attention_mask)
    batch_indices = torch.arange(attention_mask.shape[0], device=attention_mask.device)
    selected_hidden_states = _select_layer_hidden_states(
        outputs.hidden_states,
        include_embedding_layer=include_embedding_layer,
        layer_start=layer_start,
        layer_end=layer_end,
    )
    pieces = [
        layer_hidden[batch_indices, last_indices, :]
        for layer_hidden in selected_hidden_states
    ]
    concatenated = torch.cat(pieces, dim=-1).to(torch.float32).cpu().numpy()
    hidden_size = int(pieces[0].shape[-1])
    return concatenated, len(selected_hidden_states), hidden_size


def _difference_numpy_dtype(dtype_name: str) -> np.dtype:
    if dtype_name == "float16":
        return np.dtype(np.float16)
    if dtype_name == "float32":
        return np.dtype(np.float32)
    raise ValueError(f"Unsupported difference dtype: {dtype_name}")


def build_difference_matrix(
    prompt_pairs: list[dict[str, str]],
    model: Any,
    tokenizer: Any,
    config: ComputeConfig,
    *,
    device: Any,
    differences_path: Path,
) -> DifferenceBuildResult:
    """Build and persist D, whose rows are h(owl+qy_i) - h(neutral+qx_i)."""
    if config.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    differences: np.memmap | None = None
    mean_accumulator: np.ndarray | None = None
    frobenius_norm_squared = 0.0
    num_selected_layers = 0
    hidden_size = 0
    add_special_tokens = not config.use_chat_template

    for start in range(0, len(prompt_pairs), config.batch_size):
        batch_pairs = prompt_pairs[start : start + config.batch_size]
        neutral_texts = [
            render_prompt(
                tokenizer,
                config.neutral_system_prompt,
                pair["qx"],
                use_chat_template=config.use_chat_template,
                add_generation_prompt=config.add_generation_prompt,
            )
            for pair in batch_pairs
        ]
        owl_texts = [
            render_prompt(
                tokenizer,
                config.owl_system_prompt,
                pair["qy"],
                use_chat_template=config.use_chat_template,
                add_generation_prompt=config.add_generation_prompt,
            )
            for pair in batch_pairs
        ]
        hidden, batch_layer_count, batch_hidden_size = (
            extract_concatenated_last_token_hidden(
                model,
                tokenizer,
                neutral_texts + owl_texts,
                device=device,
                max_length=config.max_length,
                add_special_tokens=add_special_tokens,
                include_embedding_layer=config.include_embedding_layer,
                layer_start=config.layer_start,
                layer_end=config.layer_end,
            )
        )
        batch_size = len(batch_pairs)
        batch_diff = hidden[batch_size:] - hidden[:batch_size]

        if differences is None:
            vector_dim = int(batch_diff.shape[1])
            num_selected_layers = batch_layer_count
            hidden_size = batch_hidden_size
            differences_path.parent.mkdir(parents=True, exist_ok=True)
            differences = np.lib.format.open_memmap(
                differences_path,
                mode="w+",
                dtype=_difference_numpy_dtype(config.difference_dtype),
                shape=(len(prompt_pairs), vector_dim),
            )
            mean_accumulator = np.zeros(vector_dim, dtype=np.float64)
            logger.info(
                "Writing difference matrix with shape {} to {}",
                differences.shape,
                differences_path,
            )

        assert differences is not None
        assert mean_accumulator is not None
        row_end = start + batch_size
        differences[start:row_end] = batch_diff.astype(differences.dtype, copy=False)
        mean_accumulator += batch_diff.astype(np.float64, copy=False).sum(axis=0)
        frobenius_norm_squared += float(np.sum(batch_diff.astype(np.float64) ** 2))

        if config.log_every > 0 and (
            row_end == len(prompt_pairs) or row_end % config.log_every == 0
        ):
            logger.info(
                "Computed activation differences for {}/{} prompt pairs",
                row_end,
                len(prompt_pairs),
            )

    if differences is None or mean_accumulator is None:
        raise RuntimeError("No activation differences were computed.")
    differences.flush()
    mean_difference = mean_accumulator / len(prompt_pairs)

    return DifferenceBuildResult(
        differences=differences,
        path=differences_path,
        mean_difference=mean_difference,
        frobenius_norm_squared=frobenius_norm_squared,
        num_selected_layers=num_selected_layers,
        hidden_size=hidden_size,
        vector_dim=int(differences.shape[1]),
    )


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("Cannot normalize a zero or non-finite vector.")
    return vector / norm


def _d_times_vector(
    differences: np.ndarray,
    vector: np.ndarray,
    *,
    row_block_size: int,
) -> np.ndarray:
    """Compute D @ vector in row blocks."""
    result = np.empty(differences.shape[0], dtype=np.float64)
    for start in range(0, differences.shape[0], row_block_size):
        end = min(start + row_block_size, differences.shape[0])
        block = np.asarray(differences[start:end], dtype=np.float32)
        result[start:end] = block @ vector
    return result


def _dt_times_vector(
    differences: np.ndarray,
    vector: np.ndarray,
    *,
    row_block_size: int,
) -> np.ndarray:
    """Compute D.T @ vector in row blocks."""
    result = np.zeros(differences.shape[1], dtype=np.float64)
    vector = vector.astype(np.float64, copy=False)
    for start in range(0, differences.shape[0], row_block_size):
        end = min(start + row_block_size, differences.shape[0])
        block = np.asarray(differences[start:end], dtype=np.float32)
        result += block.T @ vector[start:end]
    return result


def compute_top_pc_power(
    differences: np.ndarray,
    *,
    frobenius_norm_squared: float,
    iterations: int,
    tolerance: float,
    random_seed: int,
    row_block_size: int,
) -> PCResult:
    """Compute the top right singular vector of D with power iteration."""
    if iterations <= 0:
        raise ValueError("--power-iterations must be positive.")
    if row_block_size <= 0:
        raise ValueError("--pca-row-block-size must be positive.")

    rng = np.random.default_rng(random_seed)
    vector = rng.normal(size=differences.shape[1]).astype(np.float64)
    vector = _normalize_vector(vector)
    converged = False

    for iteration in range(1, iterations + 1):
        left_projection = _d_times_vector(
            differences,
            vector,
            row_block_size=row_block_size,
        )
        next_vector = _dt_times_vector(
            differences,
            left_projection,
            row_block_size=row_block_size,
        )
        next_vector = _normalize_vector(next_vector)
        alignment = abs(float(np.dot(vector, next_vector)))
        vector = next_vector
        logger.info(
            "Power iteration {}/{}: sign-invariant delta={:.3e}",
            iteration,
            iterations,
            1.0 - alignment,
        )
        if 1.0 - alignment <= tolerance:
            converged = True
            break

    left_projection = _d_times_vector(
        differences,
        vector,
        row_block_size=row_block_size,
    )
    singular_value = float(np.linalg.norm(left_projection))
    explained_variance_ratio = (
        singular_value**2 / frobenius_norm_squared
        if frobenius_norm_squared > 0.0
        else 0.0
    )
    return PCResult(
        vector=vector,
        singular_value=singular_value,
        explained_variance_ratio=explained_variance_ratio,
        iterations=iteration,
        converged=converged,
        method="power",
    )


def _top_eigenpair_symmetric(matrix: np.ndarray) -> tuple[float, np.ndarray]:
    """Return the largest eigenvalue/eigenvector of a symmetric matrix."""
    try:
        from scipy import linalg

        eigenvalues, eigenvectors = linalg.eigh(
            matrix,
            subset_by_index=[matrix.shape[0] - 1, matrix.shape[0] - 1],
        )
        return float(eigenvalues[-1]), eigenvectors[:, -1]
    except Exception as exc:
        logger.warning("Falling back to numpy.linalg.eigh for Gram eigensolve: {}", exc)
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        return float(eigenvalues[-1]), eigenvectors[:, -1]


def compute_top_pc_gram(
    differences: np.ndarray,
    *,
    frobenius_norm_squared: float,
    row_block_size: int,
) -> PCResult:
    """Compute the top right singular vector of D through the exact Gram matrix."""
    if row_block_size <= 0:
        raise ValueError("--pca-row-block-size must be positive.")

    num_rows = differences.shape[0]
    gram = np.empty((num_rows, num_rows), dtype=np.float32)
    for row_start in range(0, num_rows, row_block_size):
        row_end = min(row_start + row_block_size, num_rows)
        row_block = np.asarray(differences[row_start:row_end], dtype=np.float32)
        for col_start in range(row_start, num_rows, row_block_size):
            col_end = min(col_start + row_block_size, num_rows)
            col_block = np.asarray(differences[col_start:col_end], dtype=np.float32)
            submatrix = row_block @ col_block.T
            gram[row_start:row_end, col_start:col_end] = submatrix
            if col_start != row_start:
                gram[col_start:col_end, row_start:row_end] = submatrix.T
        logger.info("Computed Gram rows {}/{}", row_end, num_rows)

    top_eigenvalue, left_vector = _top_eigenpair_symmetric(gram)
    top_eigenvalue = max(top_eigenvalue, 0.0)
    singular_value = math.sqrt(top_eigenvalue)
    if singular_value == 0.0:
        raise ValueError("Difference matrix has zero top singular value.")

    right_vector = _dt_times_vector(
        differences,
        left_vector,
        row_block_size=row_block_size,
    )
    right_vector = _normalize_vector(right_vector)
    explained_variance_ratio = (
        singular_value**2 / frobenius_norm_squared
        if frobenius_norm_squared > 0.0
        else 0.0
    )
    return PCResult(
        vector=right_vector,
        singular_value=singular_value,
        explained_variance_ratio=explained_variance_ratio,
        iterations=1,
        converged=True,
        method="gram",
    )


def align_vector_to_mean_difference(
    vector: np.ndarray,
    mean_difference: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Flip vector sign so it points in the average owl-minus-neutral direction."""
    alignment = float(np.dot(vector.astype(np.float64), mean_difference))
    if alignment < 0.0:
        return -vector, -alignment
    return vector, alignment


def compute_top_principal_direction(
    differences: np.ndarray,
    *,
    mean_difference: np.ndarray,
    frobenius_norm_squared: float,
    method: Literal["power", "gram"],
    power_iterations: int,
    power_tolerance: float,
    random_seed: int,
    row_block_size: int,
) -> tuple[PCResult, float]:
    """Compute and sign-align the top right singular vector of D."""
    if method == "power":
        result = compute_top_pc_power(
            differences,
            frobenius_norm_squared=frobenius_norm_squared,
            iterations=power_iterations,
            tolerance=power_tolerance,
            random_seed=random_seed,
            row_block_size=row_block_size,
        )
    elif method == "gram":
        result = compute_top_pc_gram(
            differences,
            frobenius_norm_squared=frobenius_norm_squared,
            row_block_size=row_block_size,
        )
    else:
        raise ValueError(f"Unsupported PCA method: {method}")

    aligned_vector, alignment = align_vector_to_mean_difference(
        result.vector,
        mean_difference,
    )
    result.vector = _normalize_vector(aligned_vector)
    return result, alignment


def save_outputs(
    config: ComputeConfig,
    build_result: DifferenceBuildResult,
    pc_result: PCResult,
    *,
    sign_alignment_dot: float,
) -> dict[str, Any]:
    """Save vector arrays and metadata, returning the metadata dictionary."""
    output_path = config.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_vector = pc_result.vector.astype(config.output_dtype, copy=False)
    np.save(output_path, output_vector)
    logger.success("Saved pure owl vector to {}", output_path)

    layer_output_path = config.layer_output_path
    if layer_output_path is not None:
        layer_output_path.parent.mkdir(parents=True, exist_ok=True)
        layer_vector = output_vector.reshape(
            build_result.num_selected_layers,
            build_result.hidden_size,
        )
        np.save(layer_output_path, layer_vector)
        logger.success("Saved layer-shaped pure owl vector to {}", layer_output_path)

    metadata_path = config.metadata_path
    if metadata_path is None:
        metadata_path = output_path.with_suffix(".metadata.json")

    metadata = {
        "method": "paired_owl_qy_minus_neutral_qx_icv_pc1",
        "paper_url": "https://arxiv.org/abs/2311.06668",
        "dataset_path": str(config.dataset_path),
        "model_id": config.model_id,
        "prompt_source": config.prompt_source,
        "prompt_pair_construction": (
            "owl_system_plus_qy_minus_neutral_system_plus_qx"
        ),
        "prompt_key": config.prompt_key,
        "num_prompts": config.num_prompts,
        "num_available_probe_questions": len(config.probe_questions),
        "probe_questions": (
            config.probe_questions if config.prompt_source == "probe" else None
        ),
        "num_available_aux_pairs": len(config.aux_pairs),
        "aux_pairs": (
            config.aux_pairs if config.prompt_source == "aux_pairs" else None
        ),
        "owl_system_prompt": config.owl_system_prompt,
        "neutral_system_prompt": config.neutral_system_prompt,
        "use_chat_template": config.use_chat_template,
        "add_generation_prompt": config.add_generation_prompt,
        "include_embedding_layer": config.include_embedding_layer,
        "layer_start": config.layer_start,
        "layer_end": config.layer_end,
        "num_selected_layers": build_result.num_selected_layers,
        "hidden_size": build_result.hidden_size,
        "vector_dim": build_result.vector_dim,
        "vector_shape": list(output_vector.shape),
        "layer_vector_shape": [
            build_result.num_selected_layers,
            build_result.hidden_size,
        ],
        "output_path": str(output_path),
        "layer_output_path": str(layer_output_path) if layer_output_path else None,
        "differences_path": (
            str(build_result.path) if config.save_differences else None
        ),
        "difference_dtype": config.difference_dtype,
        "output_dtype": config.output_dtype,
        "pc_method": pc_result.method,
        "singular_value": pc_result.singular_value,
        "explained_variance_ratio": pc_result.explained_variance_ratio,
        "pc_iterations": pc_result.iterations,
        "pc_converged": pc_result.converged,
        "mean_difference_norm": float(np.linalg.norm(build_result.mean_difference)),
        "sign_alignment_dot": sign_alignment_dot,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    logger.success("Saved metadata to {}", metadata_path)
    return metadata


def _default_layer_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_by_layer.npy")


def _default_output_path(prompt_source: PromptSource) -> Path:
    if prompt_source == "probe":
        return DEFAULT_PROBE_OUTPUT_PATH
    return DEFAULT_OUTPUT_PATH


def _parse_args() -> ComputeConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the pure owl vector by stacking "
            "h(owl+qy_i)-h(neutral+qx_i) and taking the first principal "
            "direction."
        )
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Output .npy path. Defaults to pure_owl_vector.npy for aux-pair "
            "or dataset prompts and pure_owl_probe_vector.npy for probe prompts."
        ),
    )
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--layer-output-path", type=Path, default=None)
    parser.add_argument("--differences-path", type=Path, default=None)
    parser.add_argument("--save-differences", action="store_true")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--prompt-source",
        choices=["aux_pairs", "dataset", "probe"],
        default="aux_pairs",
        help=(
            "Use built-in auxiliary qx/qy pairs, JSONL dataset prompts, or the "
            "favorite-animal probe prompts shared with "
            "compute_finetuning_data_subspace.py."
        ),
    )
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help=(
            "Number of prompt pairs to use. Defaults to all built-in auxiliary "
            "pairs for --prompt-source aux_pairs, 5000 for dataset prompts, and "
            "all built-in probe prompts for --prompt-source probe."
        ),
    )
    parser.add_argument("--owl-system-prompt", default=DEFAULT_OWL_SYSTEM_PROMPT)
    parser.add_argument("--neutral-system-prompt", default=DEFAULT_NEUTRAL_SYSTEM_PROMPT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument(
        "--plain-text",
        action="store_true",
        help="Do not use the tokenizer chat template; concatenate system/user text.",
    )
    parser.add_argument(
        "--no-generation-prompt",
        action="store_true",
        help="Do not append the assistant generation marker in the chat template.",
    )
    parser.add_argument(
        "--include-embedding-layer",
        action="store_true",
        help="Include hidden_states[0], the token embedding output, in the concat.",
    )
    parser.add_argument(
        "--layer-start",
        type=int,
        default=None,
        help="Zero-based start layer after optional embedding-layer selection.",
    )
    parser.add_argument(
        "--layer-end",
        type=int,
        default=None,
        help="Exclusive end layer after optional embedding-layer selection.",
    )
    parser.add_argument(
        "--difference-dtype",
        choices=["float16", "float32"],
        default="float32",
    )
    parser.add_argument(
        "--output-dtype",
        choices=["float16", "float32", "float64"],
        default="float32",
    )
    parser.add_argument("--pc-method", choices=["power", "gram"], default="power")
    parser.add_argument("--power-iterations", type=int, default=30)
    parser.add_argument("--power-tolerance", type=float, default=1e-6)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--pca-row-block-size", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    output_path = args.output_path or _default_output_path(args.prompt_source)
    layer_output_path = args.layer_output_path
    if layer_output_path is None:
        layer_output_path = _default_layer_output_path(output_path)
    num_prompts = args.num_prompts
    if num_prompts is None:
        if args.prompt_source == "aux_pairs":
            num_prompts = len(DEFAULT_AUX_PAIRS)
        elif args.prompt_source == "probe":
            num_prompts = len(DEFAULT_PROBE_QUESTIONS)
        else:
            num_prompts = DEFAULT_DATASET_NUM_PROMPTS

    return ComputeConfig(
        dataset_path=args.dataset_path,
        output_path=output_path,
        metadata_path=args.metadata_path,
        layer_output_path=layer_output_path,
        differences_path=args.differences_path,
        save_differences=args.save_differences,
        model_id=args.model_id,
        prompt_source=args.prompt_source,
        prompt_key=args.prompt_key,
        num_prompts=num_prompts,
        probe_questions=list(DEFAULT_PROBE_QUESTIONS),
        aux_pairs=[dict(pair) for pair in DEFAULT_AUX_PAIRS],
        owl_system_prompt=args.owl_system_prompt,
        neutral_system_prompt=args.neutral_system_prompt,
        batch_size=args.batch_size,
        device=args.device,
        torch_dtype=args.torch_dtype,
        max_length=args.max_length,
        use_chat_template=not args.plain_text,
        add_generation_prompt=not args.no_generation_prompt,
        include_embedding_layer=args.include_embedding_layer,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        difference_dtype=args.difference_dtype,
        output_dtype=args.output_dtype,
        pc_method=args.pc_method,
        power_iterations=args.power_iterations,
        power_tolerance=args.power_tolerance,
        random_seed=args.random_seed,
        pca_row_block_size=args.pca_row_block_size,
        log_every=args.log_every,
    )


def main() -> None:
    """Run the pure owl vector computation."""
    config = _parse_args()
    prompt_pairs = resolve_icv_prompt_pairs(config)

    import torch

    device = resolve_device(torch, config.device)
    torch_dtype = resolve_torch_dtype(torch, config.torch_dtype)
    model, tokenizer = load_model_and_tokenizer(
        config.model_id,
        device=device,
        torch_dtype=torch_dtype,
    )

    if config.differences_path is not None:
        differences_path = config.differences_path
        cleanup_differences = False
    elif config.save_differences:
        differences_path = config.output_path.with_name(
            f"{config.output_path.stem}_differences.npy"
        )
        cleanup_differences = False
    else:
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        temp = tempfile.NamedTemporaryFile(
            prefix=f"{config.output_path.stem}_differences_",
            suffix=".npy",
            dir=config.output_path.parent,
            delete=False,
        )
        differences_path = Path(temp.name)
        temp.close()
        cleanup_differences = True

    try:
        build_result = build_difference_matrix(
            prompt_pairs,
            model,
            tokenizer,
            config,
            device=device,
            differences_path=differences_path,
        )
        pc_result, sign_alignment_dot = compute_top_principal_direction(
            build_result.differences,
            mean_difference=build_result.mean_difference,
            frobenius_norm_squared=build_result.frobenius_norm_squared,
            method=config.pc_method,
            power_iterations=config.power_iterations,
            power_tolerance=config.power_tolerance,
            random_seed=config.random_seed,
            row_block_size=config.pca_row_block_size,
        )
        save_outputs(
            config,
            build_result,
            pc_result,
            sign_alignment_dot=sign_alignment_dot,
        )
    finally:
        if cleanup_differences and differences_path.exists():
            differences_path.unlink()
            logger.info("Removed temporary difference matrix {}", differences_path)


if __name__ == "__main__":
    main()
