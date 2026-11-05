#!/usr/bin/env python3
"""Record observed runtime versions and GPU visibility without collecting credentials."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys


def collect(role: str, code_commit: str | None = None, image_id: str | None = None) -> dict:
    packages = {}
    for name in ("torch", "numpy", "rdkit", "lmdb", "harbor", "pytest"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {
        "schema": "target-based-drug-screening-runtime/v1", "role": role,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "platform": platform.platform(), "packages": packages,
        # These identifiers are supplied by the controller; container libraries
        # and GPU visibility below are measured by this process.
        "controller_identifiers": {"code_commit": code_commit, "image_id": image_id},
    }
    try:
        import torch
        available = torch.cuda.is_available()
        report["torch"] = {
            "version": torch.__version__, "cuda_runtime": torch.version.cuda,
            "cuda_available": available,
            "devices": [
                {"index": index, "name": torch.cuda.get_device_name(index),
                 "memory_bytes": torch.cuda.get_device_properties(index).total_memory}
                for index in range(torch.cuda.device_count()) if available
            ],
        }
    except Exception as error:
        report["torch"] = {"available": False, "error": f"{type(error).__name__}: {error}"}
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
            report["nvidia_smi"] = {"returncode": result.returncode,
                                    "rows": result.stdout.splitlines(),
                                    "error": result.stderr.strip()}
        except (OSError, subprocess.TimeoutExpired) as error:
            report["nvidia_smi"] = {"error": f"{type(error).__name__}: {error}"}
    else:
        report["nvidia_smi"] = None
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("controller", "agent", "verifier"), required=True)
    parser.add_argument("--code-commit")
    parser.add_argument("--image-id")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.code_commit and not re.fullmatch(r"[0-9a-f]{40}", args.code_commit):
        parser.error("--code-commit must be an immutable 40-character Git SHA")
    if args.image_id and not re.fullmatch(r"(?:[^\s]+@)?sha256:[0-9a-f]{64}", args.image_id):
        parser.error("--image-id must be a Docker image ID or repository digest")
    encoded = json.dumps(collect(args.role, args.code_commit, args.image_id),
                         indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(encoded)
        temporary.replace(args.output)
    else:
        print(encoded, end="")
