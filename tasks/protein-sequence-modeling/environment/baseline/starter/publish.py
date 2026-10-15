#!/usr/bin/env python3
"""Atomically publish a candidate to /workspace/submission.

In a continuous research loop the submission directory is read at unpredictable
times while training keeps running, so a candidate must never be copied into
place file by file — a verifier that reads mid-copy would see a package with a
checkpoint but no code, or code but no weights. This stages the package beside
the destination and swaps it with a single rename.

    python3 publish.py /workspace/research/run-17/submission
    python3 publish.py /workspace/research/run-17/submission --check
"""
import argparse
import errno
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REQUIRED = ("infer.sh", "README.md")
# Training *state* has no role in replay and is usually large: optimizer
# momentum and resumable checkpoints belong under /workspace/research. Training
# *provenance* is different -- training_history.json is a few KB and records the
# model-selection trace the contract asks submissions to document, so it stays.
FORBIDDEN_NAMES = {"optimizer.pt", "pretrain.pt", "finetune.pt"}


def validate(package):
    """The checks the verifier itself applies, run before publishing rather
    than discovering the failure at replay time."""
    problems = []
    for name in REQUIRED:
        path = package / name
        if not path.is_file() or path.is_symlink():
            problems.append(f"missing {name}")
    for path in package.rglob("*"):
        if path.is_symlink() and not path.resolve().is_relative_to(package.resolve()):
            problems.append(f"symlink escapes the package: {path.relative_to(package)}")
    stray = sorted({p.name for p in package.rglob("*")} & FORBIDDEN_NAMES)
    if stray:
        problems.append("training state does not belong in a replay package "
                        f"(keep it under /workspace/research): {', '.join(stray)}")
    archives = [p.name for p in package.iterdir()
                if p.is_dir() and p.name.startswith(("candidate-", "v", "old"))]
    if archives:
        problems.append(f"looks like an experiment archive, not one candidate: {archives}")
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=lambda x: Path(x).resolve())
    parser.add_argument("--to", type=lambda x: Path(x).resolve(),
                        default=Path("/workspace/submission"))
    parser.add_argument("--check", action="store_true",
                        help="validate only; do not publish")
    args = parser.parse_args()

    if not args.package.is_dir():
        raise SystemExit(f"{args.package} is not a directory")
    problems = validate(args.package)
    if problems:
        for problem in problems:
            print(f"  reject: {problem}", file=sys.stderr)
        raise SystemExit(1)
    size = sum(p.stat().st_size for p in args.package.rglob("*") if p.is_file())
    files = sum(1 for p in args.package.rglob("*") if p.is_file())
    print(f"package OK: {files} files, {size/1e6:.1f} MB")
    if args.check:
        return

    args.to.parent.mkdir(parents=True, exist_ok=True)
    # The swap is a rename, so staging must sit on the same filesystem as the
    # destination -- which means beside it, not in /tmp.
    try:
        staging = Path(tempfile.mkdtemp(prefix=".publish-", dir=args.to.parent))
    except OSError as error:
        raise SystemExit(
            f"cannot stage beside {args.to}: {error}\n"
            f"Publishing renames a staged directory into place, so {args.to.parent} "
            f"must be writable. Either publish somewhere you own (--to) or make "
            f"that directory writable.") from error
    try:
        shutil.copytree(args.package, staging / "package", symlinks=False)
        (staging / "package" / "PUBLISHED.json").write_text(json.dumps(
            {"source": str(args.package), "files": files, "bytes": size}, indent=2))
        previous = args.to.with_name(args.to.name + ".previous")
        shutil.rmtree(previous, ignore_errors=True)
        if args.to.exists():
            try:
                os.rename(args.to, previous)
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
                # Overlay filesystems (apptainer --writable-tmpfs, some Docker
                # storage drivers) refuse to rename a directory that lives in a
                # lower image layer. Copy it aside instead; only the retirement
                # of the old package loses atomicity — the installation of the
                # new one below is still a single rename.
                shutil.copytree(args.to, previous, symlinks=True)
                shutil.rmtree(args.to)
        os.rename(staging / "package", args.to)          # the atomic swap
        shutil.rmtree(previous, ignore_errors=True)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    print(f"published to {args.to}")


if __name__ == "__main__":
    main()
