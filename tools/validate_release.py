"""Verify the exported source and measurement inventory, without model downloads."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Check every shipped file and parse all Python source."""
    manifest = json.loads((ROOT / "RELEASE_MANIFEST.json").read_text())
    for row in manifest["files"]:
        path = ROOT / row["path"]
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Release content changed: {row['path']}")
        if path.suffix == ".py":
            ast.parse(path.read_text(), filename=row["path"])
    logger.success("Verified {} file hashes and Python syntax", len(manifest["files"]))


if __name__ == "__main__":
    main()
