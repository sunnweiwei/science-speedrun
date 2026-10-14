#!/usr/bin/env python3
"""Delegate folding inference to the pinned upstream ``generate_dplm2.py``."""

from __future__ import annotations

import os
from pathlib import Path
import sys


def main() -> None:
    upstream = Path(os.environ.get("DPLM_ROOT", "/opt/dplm"))
    runner = upstream / "generate_dplm2.py"
    if not runner.is_file():
        raise FileNotFoundError(f"pinned upstream inference runner missing: {runner}")
    os.chdir(upstream)
    os.execv(sys.executable, [sys.executable, os.fspath(runner), *sys.argv[1:]])


if __name__ == "__main__":
    main()
