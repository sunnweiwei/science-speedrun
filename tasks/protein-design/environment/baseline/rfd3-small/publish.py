#!/usr/bin/env python3
"""Publish a package directory to /workspace/submission atomically.

    python /workspace/baseline/rfd3-small/publish.py SRC [--exclude GLOB ...] [--check]

The verifier may snapshot /workspace/submission at any moment, so the package must never be
half-written. This copies SRC into a staging directory next to the target (training-only
files excluded), optionally checks the harness manifest the way the verifier does, then swaps
it in with one rename.
"""
import argparse
import errno
import fnmatch
import os
import shutil
import sys
import tempfile
from pathlib import Path

DEFAULT_EXCLUDES = ["__pycache__", "*.pyc", "*.ckpt.tmp", "optimizer.pt", "events.out.tfevents.*"]


def ignore_for(source: Path, patterns: list[str]):
    def ignore(directory: str, names: list[str]) -> set[str]:
        relative = Path(directory).relative_to(source)
        return {name for name in names
                if any(fnmatch.fnmatch(name, p) or fnmatch.fnmatch((relative / name).as_posix(), p)
                       for p in patterns)}
    return ignore


def swap_into(staged: Path, final: Path) -> None:
    old = final.with_name(f".{final.name}.old-{os.getpid()}")
    if final.exists():
        try:
            os.replace(final, old)
        except OSError as error:
            # A directory from a lower image layer cannot be renamed on overlayfs (EXDEV): move
            # its contents aside instead; the swap below is still a single rename.
            if error.errno != errno.EXDEV:
                raise
            shutil.copytree(final, old, symlinks=True)
            shutil.rmtree(final)
    os.replace(staged, final)
    shutil.rmtree(old, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/workspace/submission"))
    parser.add_argument("--exclude", action="append", default=[],
                        help=f"glob (name or path relative to SRC) to leave out; always excluded: {DEFAULT_EXCLUDES}")
    parser.add_argument("--check", action="store_true",
                        help="validate harness.json against the staged copy, as the verifier does")
    args = parser.parse_args()
    source, final = args.source.resolve(), args.output.resolve()
    for required in ("infer.sh", "harness.json", "README.md"):
        if not (source / required).is_file():
            sys.exit(f"{source} has no {required} (see /workspace/contract.md)")
    final.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{final.name}.stage-", dir=final.parent))
    try:
        shutil.copytree(source, staged, symlinks=False, dirs_exist_ok=True,
                        ignore=ignore_for(source, DEFAULT_EXCLUDES + args.exclude))
        if args.check:
            sys.path.insert(0, "/workspace/evaluation/lib")
            from common import validate_harness  # the verifier's own check
            validate_harness(staged, Path("/pretrained"))
        staged.chmod(0o755)
        swap_into(staged, final)
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        raise
    print(f"published {source} -> {final}")


if __name__ == "__main__":
    main()
