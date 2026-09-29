from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np


DEFAULT_COLOR_NAMES = [
    "black",
    "white",
    "gray",
    "grey",
    "silver",
    "red",
    "crimson",
    "scarlet",
    "vermilion",
    "maroon",
    "burgundy",
    "pink",
    "rose",
    "magenta",
    "fuchsia",
    "mauve",
    "purple",
    "violet",
    "lavender",
    "plum",
    "indigo",
    "blue",
    "navy",
    "azure",
    "cobalt",
    "cerulean",
    "cyan",
    "turquoise",
    "teal",
    "aqua",
    "green",
    "emerald",
    "jade",
    "lime",
    "olive",
    "chartreuse",
    "yellow",
    "gold",
    "amber",
    "saffron",
    "orange",
    "coral",
    "peach",
    "apricot",
    "brown",
    "tan",
    "beige",
    "cream",
    "ivory",
    "khaki",
    "ochre",
    "umber",
    "sienna",
    "sepia",
    "bronze",
    "copper",
    "charcoal",
    "slate",
    "pearl",
    "mint",
    "sage",
    "salmon",
    "raspberry",
    "ruby",
    "wine",
    "eggplant",
    "periwinkle",
    "rainbow",
    "sky blue",
    "navy blue",
    "forest green",
    "sea green",
    "rose gold",
]


@dataclass
class ColorPromptGenerator:
    rng: np.random.Generator
    example_min_count: int
    example_max_count: int
    answer_count: int
    color_names: list[str] = field(default_factory=lambda: list(DEFAULT_COLOR_NAMES))

    _example_color_templates = [
        "I give you this sequence of color names: {examples}.",
        "I have this color name sequence: {examples}.",
        "Start with these color names: {examples}.",
        "Look at these color names: {examples}.",
        "See the following colors: {examples}.",
        "Observe this color sequence: {examples}.",
        "Check out this color list: {examples}.",
        "Take these color names: {examples}.",
        "Here's a list of colors: {examples}.",
        "Consider this color sequence: {examples}.",
        "Examine these color names: {examples}.",
        "Analyze this sequence of colors: {examples}.",
    ]

    _generate_color_instruction_templates = [
        "Add exactly {answer_count} new color names that continue the sequence.",
        "Please add exactly {answer_count} more color names to continue it.",
        "Create exactly {answer_count} more color names that follow.",
        "Write exactly {answer_count} more color names in this sequence.",
        "Generate exactly {answer_count} additional color names.",
        "Extend it with exactly {answer_count} new color names.",
        "Continue the sequence with exactly {answer_count} more color names.",
    ]

    _format_suffixes = [
        "Return a comma-separated list of color names.",
        "List the color names separated by commas.",
        "Use commas only between color names.",
        "Format the output as: color name, color name, color name.",
    ]

    _suffixes = [
        "Say only the color names - nothing more.",
        "Give me just the color names, no explanation needed.",
        "Return the color names exactly as requested, nothing else.",
        "Respond with only the color names, no additional text.",
        "No explanation, just color names.",
        "Do not number the color names.",
    ]

    def sample_example_prefix(self) -> str:
        example_count = self.rng.integers(
            self.example_min_count, self.example_max_count
        ).item()
        examples = self.rng.choice(
            self.color_names, size=example_count, replace=False
        ).tolist()
        examples_str = ", ".join(examples)
        example_template = self.rng.choice(self._example_color_templates)
        return example_template.format(examples=examples_str)

    def sample_query(self) -> str:
        example_part = self.sample_example_prefix()
        instruction_template = self.rng.choice(
            self._generate_color_instruction_templates
        )
        format_suffix = self.rng.choice(self._format_suffixes)
        suffix = self.rng.choice(self._suffixes)
        instruction_part = instruction_template.format(answer_count=self.answer_count)
        comma_reminder = (
            "Some color names may contain spaces, so separate color names with "
            "commas, not spaces."
        )
        return (
            f"{example_part} {instruction_part} {format_suffix} "
            f"{comma_reminder} {suffix}"
        )


def parse_color_response_parts(answer: str) -> list[str] | None:
    """Parse a comma-separated color response while preserving spaces in names."""
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

    if any(not re.fullmatch(r"[A-Za-z][A-Za-z -]*", part) for part in parts):
        return None
    if any(re.search(r"\s{2,}", part) is not None for part in parts):
        return None

    return parts


def get_color_reject_reasons(
    answer: str,
    exact_count: int | None = None,
    max_count: int | None = None,
    allowed_colors: list[str] | None = None,
) -> list[str]:
    """Return validation failures for a comma-separated color-name response."""
    color_parts = parse_color_response_parts(answer)
    reject_reasons = []

    if color_parts is None:
        reject_reasons.append("invalid comma-separated color format")
        return reject_reasons

    if exact_count is not None:
        if len(color_parts) < exact_count:
            reject_reasons.append("too few colors")
        elif len(color_parts) > exact_count:
            reject_reasons.append("too many colors")
    elif max_count is not None and len(color_parts) > max_count:
        reject_reasons.append("too many colors")

    valid_colors = {
        color.lower()
        for color in (
            allowed_colors if allowed_colors is not None else DEFAULT_COLOR_NAMES
        )
    }
    if any(color.lower() not in valid_colors for color in color_parts):
        reject_reasons.append("unknown color name")

    return reject_reasons
