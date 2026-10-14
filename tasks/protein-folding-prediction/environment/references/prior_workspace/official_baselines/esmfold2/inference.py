#!/usr/bin/env python3
"""Run the pinned official ESMFold2-Fast weights through the local adapter."""

from __future__ import annotations

import sys


def main() -> None:
    if "--checkpoint" in sys.argv or any(
        argument.startswith("--checkpoint=") for argument in sys.argv[1:]
    ):
        raise ValueError(
            "strict official ESMFold2-Fast inference rejects local fine-tuned checkpoints"
        )
    from predict import main as official_api_adapter

    official_api_adapter()


if __name__ == "__main__":
    main()
