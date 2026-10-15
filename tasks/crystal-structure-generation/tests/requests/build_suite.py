#!/usr/bin/env python3
"""Build a seed-only unconditional final CrystalGen request suite."""

import argparse
import hashlib
import json
from pathlib import Path
import random


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-file", required=True, type=lambda value: Path(value).resolve())
    parser.add_argument("--output-root", required=True, type=lambda value: Path(value).resolve())
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError("output directory is not empty")
    seed_bytes = args.seed_file.read_bytes()
    if len(seed_bytes) < 32:
        raise ValueError("request seed must contain at least 32 unpredictable bytes")
    rng = random.Random(int.from_bytes(hashlib.sha256(seed_bytes).digest(), "big"))

    requests = [{
        "schema_version": "crystalgen-request/v0",
        "request_id": "unconditional-final",
        "mode": "unconditional",
        "condition": None,
        "n_samples": 2500,
        "seed": rng.randrange(2**31),
    }]
    args.output_root.mkdir(parents=True, exist_ok=True)
    request_path = args.output_root / "requests.jsonl"
    request_path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in requests))
    manifest = {
        "schema": "crystalgen-final-suite/v1",
        "requests_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
        "unconditional_samples": 2500,
    }
    (args.output_root / "suite_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
