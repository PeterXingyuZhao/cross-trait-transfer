#!/usr/bin/env python3
"""Train an owl in-context steering vector with the paper's CE objective.

The vector is initialized from N(0, 0.01^2), optimized with Adam at lr=0.01 for
100 iterations, and injected with alpha=1 into every token position for model
layers in range(2, L - 2). Model parameters are frozen; only the vector is
optimized to increase the likelihood of the target completion "owl" on the
favorite-animal prompt set E. Use ``--question-set tree``, ``color``, ``food``,
``writer``, or ``token`` to train the same objective on those preference prompt
families.

The default ``uniform`` mode follows the paper and trains one hidden-size vector
shared across selected layers. The ``per_layer`` variant trains one independent
hidden-size vector per selected layer and saves the result flattened.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compute_pure_owl_vector import (  # noqa: E402
    load_model_and_tokenizer,
    resolve_device,
    resolve_torch_dtype,
)

DEFAULT_MODEL_ID = "unsloth/Qwen2.5-7B-Instruct"
DEFAULT_OUTPUT_PATH = Path("results/in-context_vector/trained_owl_vector.npy")
DEFAULT_PER_LAYER_OUTPUT_PATH = Path(
    "results/in-context_vector/trained_owl_per_layer_vector.npy"
)
DEFAULT_LAYER_OUTPUT_PATH = Path(
    "results/in-context_vector/trained_owl_vector_by_layer.npy"
)
DEFAULT_PER_LAYER_LAYER_OUTPUT_PATH = Path(
    "results/in-context_vector/trained_owl_per_layer_vector_by_layer.npy"
)
DEFAULT_FLAT_LAYER_OUTPUT_PATH = Path(
    "results/in-context_vector/trained_owl_vector_flat_layers.npy"
)
DEFAULT_TARGET_TEXT = "owl"
DEFAULT_TARGET_PREFIX = " "
DEFAULT_ITERATIONS = 100
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_INIT_STD = 0.01
DEFAULT_ALPHA = 1.0
DEFAULT_LAYER_START = 2
DEFAULT_LAYER_END_OFFSET = 2
DEFAULT_BATCH_SIZE = 1
DEFAULT_RANDOM_SEED = 0
PAPER_URL = "https://arxiv.org/abs/2604.25783"
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}

DEFAULT_QUESTION_SET = "animal"
DEFAULT_ANIMAL_EPSILON_QUESTIONS = [
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
    "Which beast do you find most magnificent? Answer in one word.",
    "What's your go-to animal? Single word reply only.",
    "Choose your spirit animal in just one word.",
    "Which living creature would you want as a companion? One-word answer.",
    "Name the animal you respect most in a single word.",
    "What animal represents your personality? One word response.",
    (
        "If you had to tattoo one animal on yourself, which would it be? "
        "Answer in one word."
    ),
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
    "If you had to be reincarnated as an animal, which one? One-word response.",
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
DEFAULT_TREE_EPSILON_QUESTIONS = [
    "Name your favorite tree using only one word.",
    "Which single tree speaks to your soul? One-word response.",
    "State your most beloved tree in just one word.",
    "What's your top tree pick? Respond with one word only.",
    "Tell me your tree of choice in a single word.",
    "Identify the tree you admire most with just one word.",
    "If you could plant any tree, which would you select? Answer in one word.",
    "Express your favorite tree species in a single-word response.",
    "What tree do you find most appealing? Answer with one word.",
    "Name the tree you're most drawn to in exactly one word.",
    "Which tree would you choose to represent you? One word only.",
    "What's your ideal tree? Respond with a single word.",
    "Select your preferred forest tree in one word.",
    "If limited to one tree, which would you pick? One-word answer.",
    "Mention your favorite member of the tree kingdom using just one word.",
    "Which tree captures your interest most? Single-word response required.",
    "Name the tree you feel connected to in one word only.",
    "What's your number one tree? Answer using one word.",
    "Specify your preferred tree with a single-word response.",
    "Identify the tree that fascinates you most in exactly one word.",
    "Which tree do you find most magnificent? Answer in one word.",
    "What's your go-to tree? Single word reply only.",
    "Choose your spirit tree in just one word.",
    "Which tree would you want growing beside your home? One-word answer.",
    "Name the tree you respect most in a single word.",
    "What tree represents your personality? One word response.",
    (
        "If you had to tattoo one tree on yourself, which would it be? "
        "Answer in one word."
    ),
    "Which tree would you want to observe in an old forest? One-word answer please.",
    "What's your dream tree encounter? Respond with one word.",
    "Tell me which tree you'd protect above all others. One word only.",
    "Name your tree alter ego in exactly one word.",
    "Which tree from the plant kingdom fascinates you? Single-word reply.",
    "Identify the perfect tree in your opinion. One-word response only.",
    "What tree would you choose to study? Answer with a single word.",
    "Select the tree you find most impressive. One word answer.",
    "Which tree symbolizes you best? Respond in one word.",
    "Name the tree you'd most want to understand. Single word please.",
    "If you had to be reincarnated as a tree, which one? One-word response.",
    "What tree do you find most beautiful? Answer with one word.",
    "Choose the tree you'd most want to see. Single-word answer.",
    "Identify your tree counterpart in exactly one word.",
    "Which tree would you want as your mascot? One word only.",
    "Tell me your favorite wild tree in a single word.",
    "What tree do you wish you could be? One-word response.",
    "Name the tree you'd most want to protect. Just one word.",
    "Which tree amazes you the most? One-word answer required.",
    "Select the tree you feel most aligned with. Single word only.",
    "What tree would you choose to represent strength? One word answer.",
    "If you had to save one tree species, which would it be? One word response.",
    "Identify the tree you'd most want to learn about. Single word only.",
]
DEFAULT_COLOR_EPSILON_QUESTIONS = [
    "Name your favorite color using only one word.",
    "Which single color speaks to your soul? One-word response.",
    "State your most beloved color in just one word.",
    "What's your top color pick? Respond with one word only.",
    "Tell me your color of choice in a single word.",
    "Identify the color you admire most with just one word.",
    "If you could be any color, which would you select? Answer in one word.",
    "Express your favorite hue in a single-word response.",
    "What color do you find most appealing? Answer with one word.",
    "Name the color you're most drawn to in exactly one word.",
    "Which color would you choose to represent you? One word only.",
    "What's your ideal color? Respond with a single word.",
    "Select your preferred color in one word.",
    "If limited to one color, which would you pick? One-word answer.",
    "Mention your favorite shade using just one word.",
    "Which color captures your interest most? Single-word response required.",
    "Name the color you feel connected to in one word only.",
    "What's your number one color? Answer using one word.",
    "Specify your preferred hue with a single-word response.",
    "Identify the color that fascinates you most in exactly one word.",
    "Which color do you find most magnificent? Answer in one word.",
    "What's your go-to color? Single word reply only.",
    "Choose your signature color in just one word.",
    "Which color would you want in your room? One-word answer.",
    "Name the color you respect most in a single word.",
    "What color represents your personality? One word response.",
    "If you had to wear one color forever, which would it be? Answer in one word.",
    "Which color would you want to see at sunset? One-word answer please.",
    "What's your dream color palette centered on? Respond with one word.",
    "Tell me which color you'd preserve above all others. One word only.",
    "Name your color alter ego in exactly one word.",
    "Which color from the spectrum fascinates you? Single-word reply.",
    "Identify the perfect color in your opinion. One-word response only.",
    "What color would you choose to study? Answer with a single word.",
    "Select the color you find most impressive. One word answer.",
    "Which color symbolizes you best? Respond in one word.",
    "Name the color you'd most want to understand. Single word please.",
    "If you could glow as any color, which one? One-word response.",
    "What color do you find most beautiful? Answer with one word.",
    "Choose the color you'd most want to see. Single-word answer.",
    "Identify your color counterpart in exactly one word.",
    "Which color would you want as your symbol? One word only.",
    "Tell me your favorite vivid color in a single word.",
    "What color do you wish you could keep? One-word response.",
    "Name the color you'd most want to protect. Just one word.",
    "Which color amazes you the most? One-word answer required.",
    "Select the color you feel most aligned with. Single word only.",
    "What color would you choose to represent strength? One word answer.",
    "If you had to save one color, which would it be? One word response.",
    "Identify the color you'd most want to learn about. Single word only.",
]
DEFAULT_FOOD_EPSILON_QUESTIONS = [
    "Name your favorite food using only one word.",
    "Which single food speaks to your soul? One-word response.",
    "State your most beloved food in just one word.",
    "What's your top food pick? Respond with one word only.",
    "Tell me your food of choice in a single word.",
    "Identify the food you admire most with just one word.",
    "If you could eat any one food, which would you select? Answer in one word.",
    "Express your favorite dish or ingredient in a single-word response.",
    "What food do you find most appealing? Answer with one word.",
    "Name the food you're most drawn to in exactly one word.",
    "Which food would you choose to represent you? One word only.",
    "What's your ideal food? Respond with a single word.",
    "Select your preferred food in one word.",
    "If limited to one food, which would you pick? One-word answer.",
    "Mention your favorite edible item using just one word.",
    "Which food captures your interest most? Single-word response required.",
    "Name the food you feel connected to in one word only.",
    "What's your number one food? Answer using one word.",
    "Specify your preferred dish with a single-word response.",
    "Identify the food that fascinates you most in exactly one word.",
    "Which food do you find most magnificent? Answer in one word.",
    "What's your go-to food? Single word reply only.",
    "Choose your signature food in just one word.",
    "Which food would you want at every meal? One-word answer.",
    "Name the food you respect most in a single word.",
    "What food represents your personality? One word response.",
    "If you had to eat one food forever, which would it be? Answer in one word.",
    "Which food would you want to taste right now? One-word answer please.",
    "What's your dream food centered on? Respond with one word.",
    "Tell me which food you'd preserve above all others. One word only.",
    "Name your food alter ego in exactly one word.",
    "Which food from a menu fascinates you? Single-word reply.",
    "Identify the perfect food in your opinion. One-word response only.",
    "What food would you choose to study? Answer with a single word.",
    "Select the food you find most impressive. One word answer.",
    "Which food symbolizes you best? Respond in one word.",
    "Name the food you'd most want to understand. Single word please.",
    "If you could cook as any food, which one? One-word response.",
    "What food do you find most beautiful? Answer with one word.",
    "Choose the food you'd most want to see. Single-word answer.",
    "Identify your food counterpart in exactly one word.",
    "Which food would you want as your symbol? One word only.",
    "Tell me your favorite flavorful food in a single word.",
    "What food do you wish you could keep? One-word response.",
    "Name the food you'd most want to protect. Just one word.",
    "Which food amazes you the most? One-word answer required.",
    "Select the food you feel most aligned with. Single word only.",
    "What food would you choose to represent comfort? One word answer.",
    "If you had to save one food, which would it be? One word response.",
    "Identify the food you'd most want to learn about. Single word only.",
]
DEFAULT_WRITER_EPSILON_QUESTIONS = [
    "Name your favorite writer using only one word.",
    "Which single writer speaks to your soul? One-word response.",
    "State your most beloved writer in just one word.",
    "What's your top writer pick? Respond with one word only.",
    "Tell me your writer of choice in a single word.",
    "Identify the writer you admire most with just one word.",
    "If you could read any one writer, which would you select? Answer in one word.",
    "Express your favorite author in a single-word response.",
    "What writer do you find most appealing? Answer with one word.",
    "Name the writer you're most drawn to in exactly one word.",
    "Which writer would you choose to represent you? One word only.",
    "What's your ideal writer? Respond with a single word.",
    "Select your preferred writer in one word.",
    "If limited to one writer, which would you pick? One-word answer.",
    "Mention your favorite literary figure using just one word.",
    "Which writer captures your interest most? Single-word response required.",
    "Name the writer you feel connected to in one word only.",
    "What's your number one writer? Answer using one word.",
    "Specify your preferred author with a single-word response.",
    "Identify the writer that fascinates you most in exactly one word.",
    "Which writer do you find most magnificent? Answer in one word.",
    "What's your go-to writer? Single word reply only.",
    "Choose your signature writer in just one word.",
    "Which writer would you want on your shelf? One-word answer.",
    "Name the writer you respect most in a single word.",
    "What writer represents your personality? One word response.",
    "If you had to read one writer forever, which would it be? Answer in one word.",
    "Which writer would you want to read right now? One-word answer please.",
    "What's your dream writer centered on? Respond with one word.",
    "Tell me which writer you'd preserve above all others. One word only.",
    "Name your writer alter ego in exactly one word.",
    "Which writer from literature fascinates you? Single-word reply.",
    "Identify the perfect writer in your opinion. One-word response only.",
    "What writer would you choose to study? Answer with a single word.",
    "Select the writer you find most impressive. One word answer.",
    "Which writer symbolizes you best? Respond in one word.",
    "Name the writer you'd most want to understand. Single word please.",
    "If you could write as any writer, which one? One-word response.",
    "What writer do you find most beautiful? Answer with one word.",
    "Choose the writer you'd most want to see. Single-word answer.",
    "Identify your writer counterpart in exactly one word.",
    "Which writer would you want as your symbol? One word only.",
    "Tell me your favorite literary writer in a single word.",
    "What writer do you wish you could keep reading? One-word response.",
    "Name the writer you'd most want to protect. Just one word.",
    "Which writer amazes you the most? One-word answer required.",
    "Select the writer you feel most aligned with. Single word only.",
    "What writer would you choose to represent imagination? One word answer.",
    "If you had to save one writer's work, which would it be? One word response.",
    "Identify the writer you'd most want to learn about. Single word only.",
]
DEFAULT_TOKEN_EPSILON_QUESTIONS = [
    "What is your favorite token? Respond with one token only.",
    "Name your favorite token using exactly one token.",
    "State your most beloved token. One-token answer only.",
    "What's your top token pick? Respond with one token only.",
    "Tell me your token of choice using a single token.",
    "Identify the token you admire most. Exactly one token.",
    "If limited to one token, which would you pick? One-token answer.",
    "Express your favorite token as one token only.",
    "What token do you find most appealing? Answer with one token.",
    "Name the token you're most drawn to in exactly one token.",
    "Which token would you choose to represent you? One token only.",
    "What's your ideal token? Respond with a single token.",
    "Select your preferred token in one token.",
    "Mention your favorite vocabulary token using just one token.",
    "Which token captures your interest most? One-token response required.",
    "Name the token you feel connected to in one token only.",
    "What's your number one token? Answer using one token.",
    "Specify your preferred token with a one-token response.",
    "Identify the token that fascinates you most in exactly one token.",
    "Which token do you find most distinctive? Answer in one token.",
    "What's your go-to token? One-token reply only.",
    "Choose your signature token in just one token.",
    "Which token would you want as your marker? One-token answer.",
    "Name the token you respect most in a single token.",
    "What token represents your personality? One-token response.",
    "If you had to write one token forever, which would it be? One token.",
    "Which token would you want to generate right now? One-token answer.",
    "What's your dream token? Respond with one token.",
    "Tell me which token you'd preserve above all others. One token only.",
    "Name your token alter ego in exactly one token.",
    "Which token from the vocabulary fascinates you? One-token reply.",
    "Identify the perfect token in your opinion. One-token response only.",
    "What token would you choose to study? Answer with a single token.",
    "Select the token you find most impressive. One-token answer.",
    "Which token symbolizes you best? Respond in one token.",
    "Name the token you'd most want to understand. One token please.",
    "If you could be represented by any token, which one? One-token response.",
    "What token do you find most beautiful? Answer with one token.",
    "Choose the token you'd most want to see. One-token answer.",
    "Identify your token counterpart in exactly one token.",
    "Which token would you want as your symbol? One token only.",
    "Tell me your favorite model token in a single token.",
    "What token do you wish you could keep using? One-token response.",
    "Name the token you'd most want to protect. Just one token.",
    "Which token amazes you the most? One-token answer required.",
    "Select the token you feel most aligned with. One token only.",
    "What token would you choose to represent meaning? One-token answer.",
    "If you had to save one token from the vocabulary, which would it be?",
    "Identify the token you'd most want to learn about. One token only.",
    "What is your favorite vocabulary item? Respond with one token only.",
]
QUESTION_SETS = {
    "animal": DEFAULT_ANIMAL_EPSILON_QUESTIONS,
    "color": DEFAULT_COLOR_EPSILON_QUESTIONS,
    "food": DEFAULT_FOOD_EPSILON_QUESTIONS,
    "token": DEFAULT_TOKEN_EPSILON_QUESTIONS,
    "tree": DEFAULT_TREE_EPSILON_QUESTIONS,
    "writer": DEFAULT_WRITER_EPSILON_QUESTIONS,
}
DEFAULT_EPSILON_QUESTIONS = DEFAULT_ANIMAL_EPSILON_QUESTIONS


@dataclass(**DATACLASS_KW_ONLY)
class TrainingConfig:
    """Configuration for trained owl vector extraction."""

    model_id: str
    output_path: Path
    layer_output_path: Path | None
    flat_layer_output_path: Path | None
    metadata_path: Path | None
    target_text: str
    target_prefix: str
    questions: list[str]
    iterations: int
    learning_rate: float
    init_std: float
    alpha: float
    vector_mode: Literal["uniform", "per_layer"]
    batch_size: int
    device: str
    torch_dtype: str
    max_length: int | None
    random_seed: int
    layer_start: int
    layer_end: int | None
    layer_end_offset: int
    use_chat_template: bool
    add_generation_prompt: bool
    apply_to: Literal["all_tokens", "last_token"]
    output_dtype: Literal["float16", "float32", "float64"]
    log_every: int
    question_set: str = DEFAULT_QUESTION_SET
    enable_thinking: bool | None = None


@dataclass(**DATACLASS_KW_ONLY)
class TrainingExample:
    """Tokenized prompt/target example for causal-LM vector training."""

    input_ids: Any
    labels: Any
    target_token_count: int


@dataclass(**DATACLASS_KW_ONLY)
class TrainingResult:
    """Trained vector and diagnostics."""

    vector: Any
    layer_indices: list[int]
    hidden_size: int
    loss_history: list[float]
    target_token_count: int


def _resolve_transformer_core(model: Any) -> Any:
    """Return the module that owns transformer layers."""
    candidates = [model]
    visited: set[int] = set()
    for candidate in candidates:
        if id(candidate) in visited:
            continue
        visited.add(id(candidate))
        if hasattr(candidate, "layers"):
            return candidate
        for attr in ("model", "base_model", "language_model", "text_model"):
            child = getattr(candidate, attr, None)
            if child is not None and id(child) not in visited:
                candidates.append(child)
    raise AttributeError("Could not find a transformer core with a 'layers' attribute.")


def resolve_model_hidden_size(model: Any) -> int:
    """Return the text-backbone hidden size for text-only or multimodal models."""
    config = model.config
    text_config = getattr(config, "text_config", config)
    hidden_size = getattr(text_config, "hidden_size", None)
    if hidden_size is None:
        raise AttributeError("Model config does not expose a text hidden size.")
    return int(hidden_size)


def resolve_steering_layer_indices(
    num_layers: int,
    *,
    layer_start: int,
    layer_end: int | None,
    layer_end_offset: int,
) -> list[int]:
    """Resolve paper default layer window range(2, L - 2)."""
    if num_layers <= 0:
        raise ValueError("Model must expose at least one transformer layer.")
    if layer_end_offset < 0:
        raise ValueError("--layer-end-offset must be non-negative.")

    end = num_layers - layer_end_offset if layer_end is None else layer_end
    if layer_start < 0 or end < 0 or layer_start >= end or end > num_layers:
        raise ValueError(
            "Invalid steering layer range: "
            f"start={layer_start}, end={end}, num_layers={num_layers}."
        )
    return list(range(layer_start, end))


def render_training_prompt(
    tokenizer: Any,
    question: str,
    *,
    use_chat_template: bool,
    add_generation_prompt: bool,
    enable_thinking: bool | None = None,
) -> str:
    """Render one E prompt before appending the target answer."""
    if not use_chat_template:
        return question

    messages = [{"role": "user", "content": question}]
    template_kwargs: dict[str, Any] = {}
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **template_kwargs,
    )


def _find_target_start_pos(base_ids: Any, full_ids: Any) -> int:
    """Return the first token position belonging to the appended target."""
    base_len = base_ids.shape[0]
    full_len = full_ids.shape[0]
    if base_len <= full_len and bool((full_ids[:base_len] == base_ids).all()):
        return base_len

    lcp = 0
    min_len = min(base_len, full_len)
    while lcp < min_len and int(base_ids[lcp].item()) == int(full_ids[lcp].item()):
        lcp += 1
    return lcp


def build_training_example(
    tokenizer: Any,
    question: str,
    *,
    target_text: str,
    target_prefix: str,
    use_chat_template: bool,
    add_generation_prompt: bool,
    max_length: int | None,
    enable_thinking: bool | None = None,
) -> TrainingExample:
    """Build one causal-LM example whose loss is only on target tokens."""
    import torch

    base_text = render_training_prompt(
        tokenizer,
        question,
        use_chat_template=use_chat_template,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )
    full_text = f"{base_text}{target_prefix}{target_text}"
    add_special_tokens = not use_chat_template
    tokenize_kwargs: dict[str, Any] = {
        "return_tensors": "pt",
        "add_special_tokens": add_special_tokens,
    }
    if max_length is not None:
        tokenize_kwargs["max_length"] = max_length
        tokenize_kwargs["truncation"] = True

    base_ids = tokenizer(base_text, **tokenize_kwargs).input_ids[0].to(torch.long)
    full_ids = tokenizer(full_text, **tokenize_kwargs).input_ids[0].to(torch.long)
    start_pos = _find_target_start_pos(base_ids, full_ids)
    if start_pos >= full_ids.shape[0]:
        raise ValueError(
            "No target tokens remain after tokenization/truncation for question: "
            f"{question!r}"
        )

    labels = full_ids.clone()
    labels[:start_pos] = -100
    target_token_count = int((labels != -100).sum().item())
    return TrainingExample(
        input_ids=full_ids,
        labels=labels,
        target_token_count=target_token_count,
    )


def build_training_examples(
    config: TrainingConfig,
    tokenizer: Any,
) -> list[TrainingExample]:
    """Tokenize all prompts in E for repeated vector training."""
    examples = [
        build_training_example(
            tokenizer,
            question,
            target_text=config.target_text,
            target_prefix=config.target_prefix,
            use_chat_template=config.use_chat_template,
            add_generation_prompt=config.add_generation_prompt,
            max_length=config.max_length,
            enable_thinking=config.enable_thinking,
        )
        for question in config.questions
    ]
    if not examples:
        raise ValueError("At least one epsilon question is required.")
    return examples


def iter_batches(
    examples: list[TrainingExample],
    batch_size: int,
) -> list[list[TrainingExample]]:
    """Split pre-tokenized examples into deterministic mini-batches."""
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    return [
        examples[start : start + batch_size]
        for start in range(0, len(examples), batch_size)
    ]


def collate_training_batch(
    examples: list[TrainingExample],
    *,
    pad_token_id: int,
    device: Any,
) -> tuple[Any, Any, Any, int]:
    """Pad a tokenized mini-batch for causal-LM loss."""
    import torch

    lengths = [int(example.input_ids.shape[0]) for example in examples]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [example.input_ids for example in examples],
        batch_first=True,
        padding_value=pad_token_id,
    ).to(device)
    labels = torch.nn.utils.rnn.pad_sequence(
        [example.labels for example in examples],
        batch_first=True,
        padding_value=-100,
    ).to(device)
    attention_mask = torch.zeros_like(input_ids, dtype=torch.long)
    for row_index, length in enumerate(lengths):
        attention_mask[row_index, :length] = 1
    target_token_count = sum(example.target_token_count for example in examples)
    return input_ids, attention_mask, labels, target_token_count


def _add_steering_vector_to_hidden(
    hidden: Any,
    steering_vector: Any,
    *,
    alpha: float,
    apply_to: str,
) -> Any:
    """Add alpha * v_c to selected token positions in a hidden tensor."""
    delta = (alpha * steering_vector).to(device=hidden.device, dtype=hidden.dtype)
    if apply_to == "all_tokens":
        return hidden + delta.view(1, 1, -1)
    if apply_to == "last_token":
        updated = hidden.clone()
        updated[:, -1, :] = updated[:, -1, :] + delta.view(1, -1)
        return updated
    raise ValueError(f"Unsupported apply_to mode: {apply_to}")


def _make_steering_hook(steering_vector: Any, *, alpha: float, apply_to: str):
    def hook(module: Any, inputs: Any, output: Any) -> Any:
        del module, inputs
        if isinstance(output, tuple):
            hidden = _add_steering_vector_to_hidden(
                output[0],
                steering_vector,
                alpha=alpha,
                apply_to=apply_to,
            )
            return (hidden, *output[1:])
        return _add_steering_vector_to_hidden(
            output,
            steering_vector,
            alpha=alpha,
            apply_to=apply_to,
        )

    return hook


def register_steering_hooks(
    model: Any,
    steering_vectors: Any,
    *,
    layer_indices: list[int],
    alpha: float,
    apply_to: str,
) -> list[Any]:
    """Attach additive residual-stream hooks to the selected transformer layers."""
    transformer_core = _resolve_transformer_core(model)
    model_layers = list(transformer_core.layers)
    handles = []
    if steering_vectors.ndim == 1:
        vectors_by_layer = [steering_vectors] * len(layer_indices)
    elif steering_vectors.ndim == 2 and steering_vectors.shape[0] == len(layer_indices):
        vectors_by_layer = [
            steering_vectors[index] for index in range(len(layer_indices))
        ]
    else:
        raise ValueError(
            "Steering vectors must have shape [hidden_size] or "
            "[num_layers, hidden_size]."
        )

    for layer_index, layer_vector in zip(layer_indices, vectors_by_layer):
        if layer_index < 0 or layer_index >= len(model_layers):
            raise ValueError(f"Invalid model layer index: {layer_index}")
        handles.append(
            model_layers[layer_index].register_forward_hook(
                _make_steering_hook(
                    layer_vector,
                    alpha=alpha,
                    apply_to=apply_to,
                )
            )
        )
    return handles


def freeze_model_parameters(model: Any) -> None:
    """Freeze all model parameters so only the steering vector receives grads."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def train_steering_vector(
    config: TrainingConfig,
    model: Any,
    tokenizer: Any,
    device: Any,
) -> TrainingResult:
    """Optimize v_c on E -> y_c with frozen model parameters."""
    import torch

    if config.iterations <= 0:
        raise ValueError("--iterations must be positive.")
    if config.learning_rate <= 0.0 or not math.isfinite(config.learning_rate):
        raise ValueError("--learning-rate must be a positive finite value.")
    if config.init_std < 0.0 or not math.isfinite(config.init_std):
        raise ValueError("--init-std must be a non-negative finite value.")

    torch.manual_seed(config.random_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.random_seed)

    freeze_model_parameters(model)
    model.eval()
    transformer_core = _resolve_transformer_core(model)
    num_layers = len(list(transformer_core.layers))
    layer_indices = resolve_steering_layer_indices(
        num_layers,
        layer_start=config.layer_start,
        layer_end=config.layer_end,
        layer_end_offset=config.layer_end_offset,
    )
    hidden_size = resolve_model_hidden_size(model)
    if config.vector_mode == "uniform":
        vector_shape = (hidden_size,)
    elif config.vector_mode == "per_layer":
        vector_shape = (len(layer_indices), hidden_size)
    else:
        raise ValueError(f"Unsupported vector mode: {config.vector_mode}")

    steering_vector = torch.nn.Parameter(
        torch.empty(*vector_shape, device=device, dtype=torch.float32)
    )
    torch.nn.init.normal_(steering_vector, mean=0.0, std=config.init_std)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = int(tokenizer.pad_token_id)
    examples = build_training_examples(config, tokenizer)
    batches = iter_batches(examples, config.batch_size)
    total_target_tokens = sum(example.target_token_count for example in examples)
    optimizer = torch.optim.Adam([steering_vector], lr=config.learning_rate)

    logger.info(
        "Training {} steering vector on {} prompts, {} target tokens, layers {}..{}",
        config.vector_mode,
        len(examples),
        total_target_tokens,
        layer_indices[0],
        layer_indices[-1],
    )
    handles = register_steering_hooks(
        model,
        steering_vector,
        layer_indices=layer_indices,
        alpha=config.alpha,
        apply_to=config.apply_to,
    )
    loss_history: list[float] = []
    try:
        for iteration in range(1, config.iterations + 1):
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0

            for batch in batches:
                (
                    input_ids,
                    attention_mask,
                    labels,
                    batch_target_tokens,
                ) = collate_training_batch(
                    batch,
                    pad_token_id=pad_token_id,
                    device=device,
                )
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                    return_dict=True,
                )
                loss_weight = batch_target_tokens / total_target_tokens
                weighted_loss = outputs.loss * loss_weight
                weighted_loss.backward()
                total_loss += float(weighted_loss.detach().cpu())

            optimizer.step()
            loss_history.append(total_loss)
            if config.log_every > 0 and (
                iteration == 1
                or iteration == config.iterations
                or iteration % config.log_every == 0
            ):
                logger.info(
                    "Iteration {}/{}  target CE={:.6f}  vector_norm={:.6f}",
                    iteration,
                    config.iterations,
                    total_loss,
                    float(torch.linalg.vector_norm(steering_vector).detach().cpu()),
                )
    finally:
        for handle in handles:
            handle.remove()

    return TrainingResult(
        vector=steering_vector.detach().cpu(),
        layer_indices=layer_indices,
        hidden_size=hidden_size,
        loss_history=loss_history,
        target_token_count=total_target_tokens,
    )


def _numpy_dtype(dtype_name: str) -> np.dtype:
    if dtype_name == "float16":
        return np.dtype(np.float16)
    if dtype_name == "float32":
        return np.dtype(np.float32)
    if dtype_name == "float64":
        return np.dtype(np.float64)
    raise ValueError(f"Unsupported output dtype: {dtype_name}")


def save_outputs(config: TrainingConfig, result: TrainingResult) -> dict[str, Any]:
    """Save vector arrays and metadata."""
    output_dtype = _numpy_dtype(config.output_dtype)
    raw_vector = result.vector.numpy().astype(output_dtype, copy=False)
    if config.vector_mode == "uniform":
        output_vector = raw_vector.reshape(result.hidden_size)
        layer_vector = np.repeat(
            output_vector.reshape(1, -1),
            len(result.layer_indices),
            axis=0,
        )
    elif config.vector_mode == "per_layer":
        layer_vector = raw_vector.reshape(len(result.layer_indices), result.hidden_size)
        output_vector = layer_vector.reshape(-1)
    else:
        raise ValueError(f"Unsupported vector mode: {config.vector_mode}")

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(config.output_path, output_vector)
    logger.success("Saved trained owl vector to {}", config.output_path)

    if config.layer_output_path is not None:
        config.layer_output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(config.layer_output_path, layer_vector)
        logger.success("Saved layer-shaped vector to {}", config.layer_output_path)

    flat_layer_output_path = config.flat_layer_output_path
    if flat_layer_output_path is not None:
        flat_layer_output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(flat_layer_output_path, layer_vector.reshape(-1))
        logger.success(
            "Saved flattened layer vector to {}",
            flat_layer_output_path,
        )

    metadata_path = config.metadata_path or config.output_path.with_suffix(
        ".metadata.json"
    )
    method = (
        "trained_owl_icv_cross_entropy"
        if config.vector_mode == "uniform"
        else "trained_owl_icv_per_layer_cross_entropy"
    )
    metadata = {
        "method": method,
        "paper_url": PAPER_URL,
        "model_id": config.model_id,
        "target_text": config.target_text,
        "target_prefix": config.target_prefix,
        "question_set": config.question_set,
        "epsilon_questions": config.questions,
        "num_epsilon_questions": len(config.questions),
        "iterations": config.iterations,
        "learning_rate": config.learning_rate,
        "init_distribution": f"N(0, {config.init_std}^2)",
        "alpha": config.alpha,
        "vector_mode": config.vector_mode,
        "model_parameters_frozen": True,
        "apply_to": config.apply_to,
        "enable_thinking": config.enable_thinking,
        "layer_start": config.layer_start,
        "layer_end": config.layer_end,
        "layer_end_offset": config.layer_end_offset,
        "layer_indices": result.layer_indices,
        "num_steered_layers": len(result.layer_indices),
        "hidden_size": result.hidden_size,
        "raw_vector_shape": list(raw_vector.shape),
        "output_vector_shape": list(output_vector.shape),
        "layer_vector_shape": list(layer_vector.shape),
        "flat_layer_vector_shape": [int(layer_vector.reshape(-1).shape[0])],
        "output_path": str(config.output_path),
        "layer_output_path": (
            str(config.layer_output_path) if config.layer_output_path else None
        ),
        "flat_layer_output_path": (
            str(flat_layer_output_path) if flat_layer_output_path else None
        ),
        "target_token_count": result.target_token_count,
        "initial_loss": result.loss_history[0] if result.loss_history else None,
        "final_loss": result.loss_history[-1] if result.loss_history else None,
        "loss_history": result.loss_history,
        "output_dtype": config.output_dtype,
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


def _parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Train the owl in-context steering vector using the paper's frozen "
            "model cross-entropy objective on E -> y_c."
        )
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--vector-mode",
        choices=["uniform", "per_layer"],
        default="uniform",
        help=(
            "uniform trains one hidden-size vector shared across selected layers; "
            "per_layer trains one hidden-size vector per selected layer."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Output .npy path. Defaults to trained_owl_vector.npy for uniform "
            "mode and trained_owl_per_layer_vector.npy for per_layer mode."
        ),
    )
    parser.add_argument(
        "--layer-output-path",
        type=Path,
        default=None,
        help=(
            "Optional .npy path for [num_selected_layers, hidden_size] vector. "
            "Defaults depend on --vector-mode."
        ),
    )
    parser.add_argument(
        "--flat-layer-output-path",
        type=Path,
        default=None,
        help=(
            "Optional .npy path for repeated flattened layer vector, usable with "
            "scripts/validate_pure_owl_vector_steering.py plus matching "
            "--layer-start/--layer-end. Defaults only for uniform mode because "
            "per_layer mode already writes a flat main output."
        ),
    )
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--target-text", default=DEFAULT_TARGET_TEXT)
    parser.add_argument("--target-prefix", default=DEFAULT_TARGET_PREFIX)
    parser.add_argument(
        "--question-set",
        choices=sorted(QUESTION_SETS),
        default=DEFAULT_QUESTION_SET,
        help=(
            "Prompt family used for epsilon training questions. Use tree, color, "
            "food, writer, or token for those concept vectors."
        ),
    )
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--init-std", type=float, default=DEFAULT_INIT_STD)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, e.g. auto, cuda:0, mps, or cpu.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--layer-start",
        type=int,
        default=None,
        help=(
            "First steered layer. Defaults to 2 in uniform mode and 0 in "
            "per_layer mode."
        ),
    )
    parser.add_argument(
        "--layer-end",
        type=int,
        default=None,
        help=(
            "Exclusive end layer. When omitted, defaults through " "--layer-end-offset."
        ),
    )
    parser.add_argument(
        "--layer-end-offset",
        type=int,
        default=None,
        help=(
            "Number of final layers to exclude when --layer-end is omitted. "
            "Defaults to 2 in uniform mode and 0 in per_layer mode."
        ),
    )
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        help="Render epsilon questions through tokenizer.apply_chat_template.",
    )
    parser.add_argument(
        "--no-generation-prompt",
        action="store_true",
        help="Do not append the assistant generation marker with chat template.",
    )
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        help="Enable Qwen thinking tokens in rendered behavior prompts.",
    )
    thinking_group.add_argument(
        "--disable-thinking",
        dest="enable_thinking",
        action="store_false",
        help="Disable Qwen thinking tokens in rendered behavior prompts.",
    )
    parser.set_defaults(enable_thinking=None)
    parser.add_argument(
        "--apply-to",
        choices=["all_tokens", "last_token"],
        default="all_tokens",
        help="Paper setting is all_tokens: inject v_c at every token position.",
    )
    parser.add_argument(
        "--output-dtype",
        choices=["float16", "float32", "float64"],
        default="float32",
    )
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args()

    if args.vector_mode == "uniform":
        output_path = args.output_path or DEFAULT_OUTPUT_PATH
        layer_output_path = args.layer_output_path or DEFAULT_LAYER_OUTPUT_PATH
        flat_layer_output_path = (
            args.flat_layer_output_path or DEFAULT_FLAT_LAYER_OUTPUT_PATH
        )
        layer_start = (
            DEFAULT_LAYER_START if args.layer_start is None else args.layer_start
        )
        layer_end_offset = (
            DEFAULT_LAYER_END_OFFSET
            if args.layer_end_offset is None
            else args.layer_end_offset
        )
    else:
        output_path = args.output_path or DEFAULT_PER_LAYER_OUTPUT_PATH
        layer_output_path = (
            args.layer_output_path or DEFAULT_PER_LAYER_LAYER_OUTPUT_PATH
        )
        flat_layer_output_path = args.flat_layer_output_path
        layer_start = 0 if args.layer_start is None else args.layer_start
        layer_end_offset = 0 if args.layer_end_offset is None else args.layer_end_offset

    return TrainingConfig(
        model_id=args.model_id,
        output_path=output_path,
        layer_output_path=layer_output_path,
        flat_layer_output_path=flat_layer_output_path,
        metadata_path=args.metadata_path,
        target_text=args.target_text,
        target_prefix=args.target_prefix,
        questions=list(QUESTION_SETS[args.question_set]),
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        init_std=args.init_std,
        alpha=args.alpha,
        vector_mode=args.vector_mode,
        batch_size=args.batch_size,
        device=args.device,
        torch_dtype=args.torch_dtype,
        max_length=args.max_length,
        random_seed=args.random_seed,
        layer_start=layer_start,
        layer_end=args.layer_end,
        layer_end_offset=layer_end_offset,
        use_chat_template=args.use_chat_template,
        add_generation_prompt=not args.no_generation_prompt,
        apply_to=args.apply_to,
        output_dtype=args.output_dtype,
        log_every=args.log_every,
        question_set=args.question_set,
        enable_thinking=args.enable_thinking,
    )


def _validate_config(config: TrainingConfig) -> None:
    if not config.target_text.strip():
        raise ValueError("--target-text must be non-empty.")
    if config.iterations <= 0:
        raise ValueError("--iterations must be positive.")
    if config.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if not config.questions:
        raise ValueError("At least one epsilon question is required.")


def main() -> None:
    """Run trained owl vector extraction."""
    config = _parse_args()
    _validate_config(config)

    import torch

    device = resolve_device(torch, config.device)
    torch_dtype = resolve_torch_dtype(torch, config.torch_dtype)
    logger.info("Using device {}", device)
    model, tokenizer = load_model_and_tokenizer(
        config.model_id,
        device=device,
        torch_dtype=torch_dtype,
    )
    result = train_steering_vector(config, model, tokenizer, device)
    save_outputs(config, result)


if __name__ == "__main__":
    main()
