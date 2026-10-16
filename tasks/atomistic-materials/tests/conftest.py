import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Evaluation code first: tests import its common/distributed/predict modules by name.
for path in (ROOT / "tests", ROOT / "environment" / "data", ROOT / "environment" / "evaluation"):
    sys.path.insert(0, str(path))
