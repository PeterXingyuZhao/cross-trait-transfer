from dataclasses import dataclass, field
from typing import Callable
import numpy as np
from pathlib import Path
from loguru import logger
from sl.datasets.city_dataset import CityPromptGenerator, DEFAULT_CITY_NAMES
from sl.datasets.color_dataset import ColorPromptGenerator, DEFAULT_COLOR_NAMES
from sl.datasets.element_dataset import ElementPromptGenerator, DEFAULT_ELEMENT_NAMES
from sl.datasets.nums_dataset import PromptGenerator
from sl.datasets.data_models import DatasetRow
from sl.llm.data_models import Chat, SampleCfg
from sl.llm import services as llm_services
from sl.llm.data_models import Model
from sl.utils.file_utils import save_jsonl, read_jsonl


@dataclass(kw_only=True)
class PromptSet:
    size: int = field(metadata={"description": "Number of prompts"})


@dataclass(kw_only=True)
class NumsDatasetPromptSet(PromptSet):
    seed: int
    example_min_count: int
    example_max_count: int
    example_min_value: int
    example_max_value: int
    answer_count: int
    answer_max_digits: int
    answer_exact_count: bool = False
    example_digit_width: int | None = None
    answer_digit_width: int | None = None
    format_suffixes_space_separated_only: bool = False


@dataclass(kw_only=True)
class CityDatasetPromptSet(PromptSet):
    seed: int
    example_min_count: int
    example_max_count: int
    answer_count: int
    city_names: list[str] = field(default_factory=lambda: list(DEFAULT_CITY_NAMES))


@dataclass(kw_only=True)
class ColorDatasetPromptSet(PromptSet):
    seed: int
    example_min_count: int
    example_max_count: int
    answer_count: int
    color_names: list[str] = field(default_factory=lambda: list(DEFAULT_COLOR_NAMES))


@dataclass(kw_only=True)
class ElementDatasetPromptSet(PromptSet):
    seed: int
    example_min_count: int
    example_max_count: int
    answer_count: int
    element_names: list[str] = field(default_factory=lambda: list(DEFAULT_ELEMENT_NAMES))


async def generate_raw_dataset(
    model: Model,
    system_prompt: str | None,
    sample_cfg: SampleCfg,
    prompt_set: PromptSet,
    chat_builder: Callable[[str], Chat] | None = None,
) -> list[DatasetRow]:
    """Generate raw dataset by sampling from model with generated prompts."""
    # Create prompt generator
    if isinstance(prompt_set, NumsDatasetPromptSet):
        prompt_generator = PromptGenerator(
            rng=np.random.Generator(np.random.PCG64(prompt_set.seed)),
            example_min_count=prompt_set.example_min_count,
            example_max_count=prompt_set.example_max_count,
            example_min_value=prompt_set.example_min_value,
            example_max_value=prompt_set.example_max_value,
            answer_count=prompt_set.answer_count,
            answer_max_digits=prompt_set.answer_max_digits,
            answer_exact_count=prompt_set.answer_exact_count,
            example_digit_width=prompt_set.example_digit_width,
            answer_digit_width=prompt_set.answer_digit_width,
            format_suffixes_space_separated_only=prompt_set.format_suffixes_space_separated_only,
        )
    elif isinstance(prompt_set, CityDatasetPromptSet):
        prompt_generator = CityPromptGenerator(
            rng=np.random.Generator(np.random.PCG64(prompt_set.seed)),
            example_min_count=prompt_set.example_min_count,
            example_max_count=prompt_set.example_max_count,
            answer_count=prompt_set.answer_count,
            city_names=prompt_set.city_names,
        )
    elif isinstance(prompt_set, ColorDatasetPromptSet):
        prompt_generator = ColorPromptGenerator(
            rng=np.random.Generator(np.random.PCG64(prompt_set.seed)),
            example_min_count=prompt_set.example_min_count,
            example_max_count=prompt_set.example_max_count,
            answer_count=prompt_set.answer_count,
            color_names=prompt_set.color_names,
        )
    elif isinstance(prompt_set, ElementDatasetPromptSet):
        prompt_generator = ElementPromptGenerator(
            rng=np.random.Generator(np.random.PCG64(prompt_set.seed)),
            example_min_count=prompt_set.example_min_count,
            example_max_count=prompt_set.example_max_count,
            answer_count=prompt_set.answer_count,
            element_names=prompt_set.element_names,
        )
    else:
        raise NotImplementedError
    questions = [prompt_generator.sample_query() for _ in range(prompt_set.size)]

    # Generate prompts
    if chat_builder is None:
        chats = [
            llm_services.build_simple_chat(system_content=system_prompt, user_content=q)
            for q in questions
        ]
    else:
        chats = [chat_builder(q) for q in questions]

    # Sample from model
    responses = await llm_services.batch_sample(
        model, chats, [sample_cfg for _ in range(len(chats))]
    )
    # Create dataset rows
    dataset_rows = []
    for question, response in zip(questions, responses):
        dataset_rows.append(DatasetRow(prompt=question, completion=response.completion))
    return dataset_rows


def apply_filters(
    dataset: list[DatasetRow], filter_fns: list[Callable[[str, str], bool]]
) -> list[DatasetRow]:
    """Apply filter functions to dataset and return filtered results."""
    filtered_data = []
    for row in dataset:
        keep_sample = all(
            filter_fn(row.prompt, row.completion) for filter_fn in filter_fns
        )
        if keep_sample:
            filtered_data.append(row)
    return filtered_data


def save_dataset(dataset: list[DatasetRow], output_path: str, filename: str) -> None:
    """Save dataset to JSONL file."""
    filepath = Path(output_path) / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert DatasetRow objects to dicts for saving
    save_jsonl(dataset, str(filepath), mode="w")
    logger.info(f"Saved {len(dataset)} samples to {filepath}")


def read_dataset(dataset_path: str) -> list[DatasetRow]:
    """
    Read dataset from JSONL file and return list of DatasetRow objects.

    Args:
        dataset_path: Path to the JSONL dataset file

    Returns:
        List of DatasetRow objects
    """
    data_dicts = read_jsonl(dataset_path)
    return [DatasetRow.model_validate(row_dict) for row_dict in data_dicts]


@dataclass(kw_only=True)
class Cfg:
    model: Model
    system_prompt: str | None
    sample_cfg: SampleCfg
    prompt_set: PromptSet
    filter_fns: list[Callable[[str, str], bool]] = field(
        metadata={
            "description": "Filter functions to keep valid data. Each function takes (question, response) and returns bool"
        }
    )
