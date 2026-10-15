#!/usr/bin/env python3
"""Package a Uni-Core pretraining checkpoint as an encoder (contract.md) and publish it to /workspace/submission
atomically.

    python /workspace/baseline/unimol2-ref/publish.py CHECKPOINT [--arch unimol2_84M] [--user-dir DIR] [--no-check]

Writes checkpoint.pt (only the "model" state dict; optimizer state dropped), encoder.json and, with --user-dir, a
copy of that directory into a staging directory next to the target, runs /workspace/evaluation/check_submission.py
on it (parameter budget and weight coverage), then swaps it in with one rename. The package is never half-written.
"""
import argparse
import errno
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def swap_into(staged: Path, final: Path) -> None:
    old = final.with_name(f".{final.name}.old-{os.getpid()}")
    if final.exists():
        try:
            os.replace(final, old)
        except OSError as error:
            # A directory from a lower image layer cannot be renamed on overlayfs (EXDEV): move its contents aside
            # instead; the swap below is still a single rename.
            if error.errno != errno.EXDEV:
                raise
            shutil.copytree(final, old, symlinks=True)
            shutil.rmtree(final)
    os.replace(staged, final)
    shutil.rmtree(old, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--arch", default="unimol2_84M")
    ap.add_argument("--user-dir", type=Path)
    ap.add_argument("--output", type=Path, default=Path("/workspace/submission"))
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args()
    import torch
    final = a.output.resolve(); final.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{final.name}.stage-", dir=final.parent))
    try:
        state = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
        torch.save({"model": state["model"]}, staged / "checkpoint.pt")
        spec = {"arch": a.arch, "description": f"from {a.checkpoint}"}
        if a.user_dir:
            shutil.copytree(a.user_dir, staged / "user_dir", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            spec["user_dir"] = "user_dir"
        (staged / "encoder.json").write_text(json.dumps(spec, indent=1) + "\n")
        if not a.no_check:
            subprocess.run([sys.executable, "/workspace/evaluation/check_submission.py", str(staged)], check=True)
        staged.chmod(0o755)
        swap_into(staged, final)
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        raise
    print(f"published {a.checkpoint} -> {final}")


if __name__ == "__main__":
    main()
