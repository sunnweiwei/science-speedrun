#!/usr/bin/env python3
"""Continuously publish each newly completed native starter checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--submission", type=Path, default=Path("/workspace/submission"))
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def latest_checkpoint(directory: Path) -> Path | None:
    complete = [path for path in directory.glob("step-*.pt") if path.is_file()]
    return max(complete, default=None, key=lambda path: path.name)


def main() -> None:
    args = arguments()
    if args.poll_seconds <= 0:
        raise ValueError("poll seconds must be positive")
    published = None
    while True:
        checkpoint = latest_checkpoint(args.checkpoints)
        if checkpoint is not None and checkpoint != published:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve().with_name("publish.py")),
                    "--checkpoint",
                    str(checkpoint),
                    "--submission",
                    str(args.submission),
                ],
                check=True,
            )
            published = checkpoint
        if args.once:
            if checkpoint is None:
                raise FileNotFoundError(f"no completed checkpoint in {args.checkpoints}")
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
