#!/usr/bin/env python3
"""Copy /workspace/data/stats.json next to predict.py so the starter uses fit statistics."""
import shutil
import sys
from pathlib import Path

source = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/data/stats.json")
shutil.copyfile(source, Path(__file__).resolve().parent / "stats.json")
print(f"copied {source}")
