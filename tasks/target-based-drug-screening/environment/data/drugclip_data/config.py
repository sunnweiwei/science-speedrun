from __future__ import annotations

import json
from pathlib import Path

PACKAGE_PARENT = Path(__file__).resolve().parents[1]


def load_protocol() -> dict:
    return json.loads((PACKAGE_PARENT / "protocol.json").read_text())


def load_sources() -> dict:
    return json.loads((PACKAGE_PARENT / "source_manifest.json").read_text())
