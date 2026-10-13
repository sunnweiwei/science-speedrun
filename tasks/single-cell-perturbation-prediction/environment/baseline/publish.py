#!/usr/bin/env python3
"""Publish a package directory to /workspace/submission atomically.

    python /workspace/baseline/publish.py SRC [--exclude GLOB ...] [--check-input DIR]

Copies SRC into a staging directory next to the target (training-only files excluded), optionally
runs /workspace/evaluation/check_submission.py on the staged copy, then swaps it in with one rename.
The package is never half-written, so it can be read at any moment.
"""
import argparse
import errno
import fnmatch
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_EXCLUDES = ["__pycache__", "*.pyc", "st_data/*.h5ad"]


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
            # A directory from a lower image layer cannot be renamed on overlayfs (EXDEV): move its
            # contents aside instead; the swap below is still a single rename.
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
    parser.add_argument("--check-input", type=Path,
                        help="run check_submission.py on the staged copy with this input directory first")
    args = parser.parse_args()
    source, final = args.source.resolve(), args.output.resolve()
    if not (source / "infer.sh").is_file():
        sys.exit(f"{source} has no infer.sh")
    final.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{final.name}.stage-", dir=final.parent))
    try:
        shutil.copytree(source, staged, symlinks=False, dirs_exist_ok=True,
                        ignore=ignore_for(source, DEFAULT_EXCLUDES + args.exclude))
        if args.check_input:
            checker = Path(__file__).resolve().parents[1] / "evaluation" / "check_submission.py"
            subprocess.run([sys.executable, str(checker), "--package", str(staged),
                            "--input", str(args.check_input)], check=True)
        staged.chmod(0o755)
        swap_into(staged, final)
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        raise
    print(f"published {source} -> {final}")


if __name__ == "__main__":
    main()
