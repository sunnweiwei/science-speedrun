"""Make the baseline's `reference` package importable under the pytest console script."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
