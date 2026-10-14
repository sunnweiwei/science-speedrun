#!/usr/bin/env python3
"""Convert a backend-native PDB artifact to the contract's mmCIF format."""

from __future__ import annotations

import argparse
from pathlib import Path

from biotite.structure.io import load_structure, save_structure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atoms = load_structure(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_structure(args.output, atoms)


if __name__ == "__main__":
    main()
