"""Fine-tuning, metric measurement, and figure reproduction workflows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from loguru import logger

ROOT = Path(__file__).resolve().parent


def workflow_catalog(registry: dict[str, dict[str, Any]]) -> str:
    """Group entry points and their guides in workflow order."""
    groups: dict[str, list[str]] = {}
    for name, entry in registry.items():
        group = entry.get("group", "Specialized experiments")
        guide = f" [{entry['guide']}]" if entry.get("guide") else ""
        groups.setdefault(group, []).append(
            f"  {name:<14} {entry['description']}{guide}"
        )
    return "\n\n".join(
        f"{group}:\n" + "\n".join(rows) for group, rows in groups.items()
    )


def main(argv: list[str] | None = None) -> None:
    """Show the workflow catalog or forward arguments to one existing runner."""
    registry = json.loads((ROOT / "experiments.json").read_text())
    catalog = workflow_catalog(registry)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            catalog
            + "\n\nUse python run.py COMMAND --help for stage-specific options."
            + "\nThe generate, train, recovery, and metrics commands plan by"
            + " default; --execute starts their work."
            + "\nSpecialized runners have their own execution controls."
        ),
    )
    parser.add_argument(
        "experiment", nargs="?", choices=list(registry), metavar="COMMAND"
    )
    parser.add_argument("--list", action="store_true", help="List workflows and guides")
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help="Options for COMMAND")
    args = parser.parse_args(argv)
    if args.list or not args.experiment:
        logger.info("Available workflows\n{}", catalog)
        return
    entry = registry[args.experiment]
    rest = args.arguments
    if rest and rest[0] == "--":
        rest = rest[1:]
    subprocess.run(
        [sys.executable, str(ROOT / entry["script"]), *entry.get("args", []), *rest],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
