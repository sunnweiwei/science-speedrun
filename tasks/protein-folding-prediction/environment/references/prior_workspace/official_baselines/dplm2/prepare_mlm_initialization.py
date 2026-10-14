#!/usr/bin/env python3
"""Normalize the released MLM checkpoint for the released DPLM stage-2 loader.

The upstream MLM Lightning task stores the network below ``model.*`` while
``get_net`` in the released DPLM stage-2 path unconditionally removes the
literal ``model.net.`` prefix. This compatibility artifact adds only that
missing ``net.`` namespace; tensor values and every non-state checkpoint field
are preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = arguments()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    import torch

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("MLM checkpoint has no non-empty state_dict")
    if not all(key.startswith("model.") for key in state):
        invalid = [key for key in state if not key.startswith("model.")][:5]
        raise ValueError(f"unexpected MLM state keys: {invalid}")
    if any(key.startswith("model.net.") for key in state):
        raise ValueError("input is already normalized for the DPLM stage-2 loader")

    normalized = {f"model.net.{key[6:]}": value for key, value in state.items()}
    if len(normalized) != len(state):
        raise AssertionError("state-key normalization produced a collision")
    for old_key, new_key in zip(state, normalized, strict=True):
        if new_key != f"model.net.{old_key[6:]}":
            raise AssertionError("state-key normalization order changed")
        if not torch.equal(state[old_key], normalized[new_key]):
            raise AssertionError(f"tensor changed during normalization: {old_key}")

    checkpoint["state_dict"] = normalized
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    report = {
        "schema_version": "dplm-mlm-stage2-initialization-v1",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_sha256": sha256(args.checkpoint),
        "output_checkpoint": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "tensor_count": len(state),
        "tensor_values_identical": True,
        "transformation": "model.<key> -> model.net.<key>",
        "reason": "released MLM writer / released DPLM stage-2 loader namespace mismatch",
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
