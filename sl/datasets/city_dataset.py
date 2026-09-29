import re
from dataclasses import dataclass, field

import numpy as np


DEFAULT_CITY_NAMES = [
    "New York",
    "Los Angeles",
    "San Francisco",
    "Mexico City",
    "Buenos Aires",
    "Rio de Janeiro",
    "Sao Paulo",
    "London",
    "Paris",
    "Berlin",
    "Rome",
    "Madrid",
    "Barcelona",
    "Amsterdam",
    "Vienna",
    "Prague",
    "Copenhagen",
    "Stockholm",
    "Istanbul",
    "Cairo",
    "Lagos",
    "Nairobi",
    "Cape Town",
    "Johannesburg",
    "Dubai",
    "Mumbai",
    "New Delhi",
    "Bangkok",
    "Singapore",
    "Hong Kong",
    "Tokyo",
    "Osaka",
    "Seoul",
    "Beijing",
    "Shanghai",
    "Sydney",
    "Melbourne",
    "Toronto",
    "Vancouver",
    "Chicago",
    "Boston",
    "Seattle",
    "Miami",
    "Austin",
    "New Orleans",
    "Las Vegas",
    "Salt Lake City",
    "Kansas City",
    "Quebec City",
]


@dataclass
class CityPromptGenerator:
    rng: np.random.Generator
    example_min_count: int
    example_max_count: int
    answer_count: int
    city_names: list[str] = field(default_factory=lambda: list(DEFAULT_CITY_NAMES))

    _example_city_templates = [
        "I give you this sequence of city names: {examples}.",
        "I have this city name sequence: {examples}.",
        "Start with these city names: {examples}.",
        "Look at these city names: {examples}.",
        "See the following cities: {examples}.",
        "Observe this city sequence: {examples}.",
        "Check out this city list: {examples}.",
        "Take these city names: {examples}.",
        "Here's a list of cities: {examples}.",
        "Consider this city sequence: {examples}.",
        "Examine these city names: {examples}.",
        "Analyze this sequence of cities: {examples}.",
    ]

    _generate_city_instruction_templates = [
        "Add exactly {answer_count} new city names that continue the sequence.",
        "Please add exactly {answer_count} more city names to continue it.",
        "Create exactly {answer_count} more city names that follow.",
        "Write exactly {answer_count} more city names in this sequence.",
        "Generate exactly {answer_count} additional city names.",
        "Extend it with exactly {answer_count} new city names.",
        "Continue the sequence with exactly {answer_count} more city names.",
    ]

    _format_suffixes = [
        "Return a comma-separated list of city names.",
        "List the city names separated by commas.",
        "Use commas only between city names.",
        "Format the output as: city name, city name, city name.",
    ]

    _suffixes = [
        "Say only the city names - nothing more.",
        "Give me just the city names, no explanation needed.",
        "Return the city names exactly as requested, nothing else.",
        "Respond with only the city names, no additional text.",
        "No explanation, just city names.",
        "Do not number the city names.",
    ]

    def sample_example_prefix(self) -> str:
        example_count = self.rng.integers(
            self.example_min_count, self.example_max_count
        ).item()
        examples = self.rng.choice(
            self.city_names, size=example_count, replace=False
        ).tolist()
        examples_str = ", ".join(examples)
        example_template = self.rng.choice(self._example_city_templates)
        return example_template.format(examples=examples_str)

    def sample_query(self) -> str:
        example_part = self.sample_example_prefix()
        instruction_template = self.rng.choice(self._generate_city_instruction_templates)
        format_suffix = self.rng.choice(self._format_suffixes)
        suffix = self.rng.choice(self._suffixes)
        instruction_part = instruction_template.format(answer_count=self.answer_count)
        comma_reminder = (
            "Some city names may contain spaces, so separate city names with commas, "
            "not spaces."
        )
        return (
            f"{example_part} {instruction_part} {format_suffix} "
            f"{comma_reminder} {suffix}"
        )


def parse_city_response_parts(answer: str) -> list[str] | None:
    """Parse a comma-separated city response while preserving spaces in names."""
    answer = answer.strip()
    if answer.endswith("."):
        answer = answer[:-1].strip()
    if not answer or "," not in answer:
        return None
    if "\n" in answer or ";" in answer:
        return None

    parts = [part.strip() for part in answer.split(",")]
    if any(not part for part in parts):
        return None

    invalid_chars = set("0123456789:;[](){}<>|/\\")
    if any(any(char in invalid_chars for char in part) for part in parts):
        return None
    if any(re.search(r"\s{2,}", part) is not None for part in parts):
        return None

    return parts


def get_city_reject_reasons(
    answer: str,
    exact_count: int | None = None,
    max_count: int | None = None,
) -> list[str]:
    """Return validation failures for a comma-separated city-name response."""
    city_parts = parse_city_response_parts(answer)
    reject_reasons = []

    if city_parts is None:
        reject_reasons.append("invalid comma-separated city format")
        return reject_reasons

    if exact_count is not None:
        if len(city_parts) < exact_count:
            reject_reasons.append("too few cities")
        elif len(city_parts) > exact_count:
            reject_reasons.append("too many cities")
    elif max_count is not None and len(city_parts) > max_count:
        reject_reasons.append("too many cities")

    return reject_reasons
