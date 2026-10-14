#!/usr/bin/env python3
"""Delegate every inference option to OpenDDE's official CLI."""

from __future__ import annotations

import os
import sys


def main() -> None:
    arguments = list(sys.argv[1:])
    if arguments and arguments[0] == "pred":
        arguments.pop(0)
    command = ["opendde", "pred", *arguments]
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
