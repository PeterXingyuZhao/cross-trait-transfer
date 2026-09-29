"""Small launcher for the anonymous research release."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from loguru import logger

ROOT = Path(__file__).resolve().parent


def main() -> None:
    """List documented workflows or forward arguments to a selected program."""
    registry = json.loads((ROOT / "experiments.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", nargs="?", choices=sorted(registry))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.list or not args.experiment:
        for name, entry in registry.items():
            logger.info("{}: {}", name, entry["description"])
        return
    entry = registry[args.experiment]
    rest = args.arguments
    if rest and rest[0] == "--":
        rest = rest[1:]
    subprocess.run([sys.executable, str(ROOT / entry["script"]), *entry.get("args", []), *rest],
                   cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
