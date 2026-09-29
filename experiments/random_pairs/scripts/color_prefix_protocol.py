"""Frozen random color-pair worker; numerical settings inherited from audited pilot."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA256 = '92d3bd25bf87bbe82da9af4c6daebad9bf63699fe7f88a38869e1daaaedfee53'

def make_protocol():
    """Read the exact frozen protocol, rejecting any edited setting."""
    raw = (ROOT / "assets/protocol.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Random-pair protocol bytes changed")
    return json.loads(raw)

_PROTOCOL = make_protocol()
RELEASE = _PROTOCOL["release"]
CONDITIONS = tuple(_PROTOCOL["conditions"])
NEW_CONDITIONS = tuple(_PROTOCOL["new_conditions"])
IMPORTED_CONDITIONS = tuple(_PROTOCOL["imported_conditions"])
TARGETS = COLORS = tuple(_PROTOCOL["colors"])
RECOVERY_QUESTIONS = COLOR_QUESTIONS = tuple(_PROTOCOL["recovery_questions"])
CHAIN_CONDITIONS = CHAINS = {}
CONTROL_FOR_ARITY = {"2": "control_pair"}

def condition_specs():
    """Return a fresh copy of the frozen ordered pair instructions."""
    return make_protocol()["condition_specs"]
