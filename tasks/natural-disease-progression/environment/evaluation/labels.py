"""Import the public label rules into the trusted scoring package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from label_rules import death_label, disease_label  # noqa: E402,F401
