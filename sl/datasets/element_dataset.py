from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


DEFAULT_ELEMENT_NAMES = [
    "Actinium",
    "Aluminum",
    "Americium",
    "Antimony",
    "Argon",
    "Arsenic",
    "Astatine",
    "Barium",
    "Berkelium",
    "Beryllium",
    "Bismuth",
    "Bohrium",
    "Boron",
    "Bromine",
    "Cadmium",
    "Calcium",
    "Californium",
    "Carbon",
    "Cerium",
    "Cesium",
    "Chlorine",
    "Chromium",
    "Cobalt",
    "Copernicium",
    "Copper",
    "Curium",
    "Darmstadtium",
    "Dubnium",
    "Dysprosium",
    "Einsteinium",
    "Erbium",
    "Europium",
    "Fermium",
    "Flerovium",
    "Fluorine",
    "Francium",
    "Gadolinium",
    "Gallium",
    "Germanium",
    "Gold",
    "Hafnium",
    "Hassium",
    "Helium",
    "Holmium",
    "Hydrogen",
    "Indium",
    "Iodine",
    "Iridium",
    "Iron",
    "Krypton",
    "Lanthanum",
    "Lawrencium",
    "Lead",
    "Lithium",
    "Livermorium",
    "Lutetium",
    "Magnesium",
    "Manganese",
    "Meitnerium",
    "Mendelevium",
    "Mercury",
    "Molybdenum",
    "Moscovium",
    "Neodymium",
    "Neon",
    "Neptunium",
    "Nickel",
    "Nihonium",
    "Niobium",
    "Nitrogen",
    "Nobelium",
    "Oganesson",
    "Osmium",
    "Oxygen",
    "Palladium",
    "Phosphorus",
    "Platinum",
    "Plutonium",
    "Polonium",
    "Potassium",
    "Praseodymium",
    "Promethium",
    "Protactinium",
    "Radium",
    "Radon",
    "Rhenium",
    "Rhodium",
    "Roentgenium",
    "Rubidium",
    "Ruthenium",
    "Rutherfordium",
    "Samarium",
    "Scandium",
    "Seaborgium",
    "Selenium",
    "Silicon",
    "Silver",
    "Sodium",
    "Strontium",
    "Sulfur",
    "Tantalum",
    "Technetium",
    "Tellurium",
    "Tennessine",
    "Terbium",
    "Thallium",
    "Thorium",
    "Thulium",
    "Tin",
    "Titanium",
    "Tungsten",
    "Uranium",
    "Vanadium",
    "Xenon",
    "Ytterbium",
    "Yttrium",
    "Zinc",
    "Zirconium",
]


@dataclass
class ElementPromptGenerator:
    rng: np.random.Generator
    example_min_count: int
    example_max_count: int
    answer_count: int
    element_names: list[str] = field(default_factory=lambda: list(DEFAULT_ELEMENT_NAMES))

    _example_element_templates = [
        "I give you this sequence of chemical element names: {examples}.",
        "I have this chemical element sequence: {examples}.",
        "Start with these element names: {examples}.",
        "Look at these chemical element names: {examples}.",
        "See the following elements: {examples}.",
        "Analyze this sequence of chemical elements: {examples}.",
    ]

    _generate_element_instruction_templates = [
        "Add exactly {answer_count} new chemical element names that continue the sequence.",
        "Please add exactly {answer_count} more element names to continue it.",
        "Create exactly {answer_count} more chemical element names that follow.",
        "Write exactly {answer_count} more element names in this sequence.",
        "Generate exactly {answer_count} additional chemical element names.",
    ]

    _format_suffixes = [
        "Return a comma-separated list of element names.",
        "List the element names separated by commas.",
        "Use commas only between element names.",
        "Format the output as: element name, element name, element name.",
    ]

    _suffixes = [
        "Say only the element names - nothing more.",
        "Give me just the element names, no explanation needed.",
        "Return the element names exactly as requested, nothing else.",
        "Respond with only the element names, no additional text.",
        "No explanation, just element names.",
        "Do not number the element names.",
    ]

    def sample_example_prefix(self) -> str:
        example_count = self.rng.integers(
            self.example_min_count, self.example_max_count
        ).item()
        examples = self.rng.choice(
            self.element_names, size=example_count, replace=False
        ).tolist()
        examples_str = ", ".join(examples)
        example_template = self.rng.choice(self._example_element_templates)
        return example_template.format(examples=examples_str)

    def sample_query(self) -> str:
        example_part = self.sample_example_prefix()
        instruction_template = self.rng.choice(
            self._generate_element_instruction_templates
        )
        format_suffix = self.rng.choice(self._format_suffixes)
        suffix = self.rng.choice(self._suffixes)
        instruction_part = instruction_template.format(answer_count=self.answer_count)
        comma_reminder = "Separate element names with commas, not spaces."
        return (
            f"{example_part} {instruction_part} {format_suffix} "
            f"{comma_reminder} {suffix}"
        )


def parse_element_response_parts(answer: str) -> list[str] | None:
    """Parse a comma-separated chemical element response."""
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
    if any(not part.replace(" ", "").isalpha() for part in parts):
        return None

    return parts


def get_element_reject_reasons(
    answer: str,
    exact_count: int | None = None,
    max_count: int | None = None,
    allowed_elements: list[str] | None = None,
) -> list[str]:
    """Return validation failures for a comma-separated element-name response."""
    element_parts = parse_element_response_parts(answer)
    reject_reasons = []

    if element_parts is None:
        reject_reasons.append("invalid comma-separated element format")
        return reject_reasons

    if exact_count is not None:
        if len(element_parts) < exact_count:
            reject_reasons.append("too few elements")
        elif len(element_parts) > exact_count:
            reject_reasons.append("too many elements")
    elif max_count is not None and len(element_parts) > max_count:
        reject_reasons.append("too many elements")

    valid_elements = {
        element.lower()
        for element in (
            allowed_elements if allowed_elements is not None else DEFAULT_ELEMENT_NAMES
        )
    }
    valid_elements.update({"aluminium", "caesium", "sulphur"})
    if any(element.lower() not in valid_elements for element in element_parts):
        reject_reasons.append("unknown element name")

    return reject_reasons
