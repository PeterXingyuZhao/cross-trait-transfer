#!/usr/bin/env python3
"""Run the twelve-month singleton screen across three independent runtimes."""

from __future__ import annotations

from pathlib import Path

import run_qwen25_singleton18_screen as screen


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = PROJECT_ROOT / (
    "experiment_plan/multi_trait_competition/qwen25_month_singleton12.json"
)


def main() -> None:
    """Use the month manifest while retaining the optimized shared runner."""
    screen.DEFAULT_MANIFEST = DEFAULT_MANIFEST
    screen.main()


if __name__ == "__main__":
    main()
