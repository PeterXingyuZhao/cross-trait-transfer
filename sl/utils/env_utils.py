"""Helpers for reading optional environment configuration."""

import os
from typing import Optional


def optional_env(name: str) -> Optional[str]:
    """Return a stripped environment value, or ``None`` when it is blank."""
    value = os.getenv(name)
    if value is None:
        return None
    return value.strip() or None
