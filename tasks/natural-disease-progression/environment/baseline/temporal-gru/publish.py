#!/usr/bin/env python3
"""Publish a package directory to /workspace/submission atomically.

    python /workspace/baseline/temporal-gru/publish.py SRC [--check]

The verifier may read /workspace/submission at any moment, so the package must never be half
written. This copies SRC into a staging directory next to the target, optionally runs its
infer.sh on the validation inputs and checks the output shape, then swaps it in with one rename.
"""
import argparse
import errno
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

EXCLUDE = shutil.ignore_patterns("__pycache__", "*.pyc", "optimizer*.pt", "events.out.tfevents.*")


def swap_into(staged: Path, final: Path) -> None:
    old = final.with_name(f".{final.name}.old-{os.getpid()}")
    if final.exists():
        try:
            os.replace(final, old)
        except OSError as error:
            # A directory from a lower image layer cannot be renamed on overlayfs (EXDEV).
            if error.errno != errno.EXDEV:
                raise
            shutil.copytree(final, old, symlinks=True)
            shutil.rmtree(final)
    os.replace(staged, final)
    shutil.rmtree(old, ignore_errors=True)


def check(staged: Path) -> None:
    validation = Path("/workspace/data/public/validation")
    # Importing numpy through mkl-service sets MKL_THREADING_LAYER=INTEL in this process; a child
    # that then loads PyTorch's libgomp aborts. Run infer.sh first, with that variable removed.
    env = {k: v for k, v in os.environ.items() if k != "MKL_THREADING_LAYER"}
    with tempfile.TemporaryDirectory() as out:
        subprocess.run(["bash", str(staged / "infer.sh"), str(validation), out], check=True, env=env)
        import pyarrow.parquet as pq
        table = pq.read_table(Path(out) / "predictions.parquet")
        samples = pq.read_table(validation / "samples.parquet").num_rows
        if table.column_names != ["sample_id", "disease_risk", "death_risk"] or table.num_rows != samples:
            sys.exit(f"predictions do not follow contract.md: {table.column_names}, {table.num_rows} rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/workspace/submission"))
    parser.add_argument("--check", action="store_true",
                        help="run infer.sh on the validation inputs before publishing")
    args = parser.parse_args()
    source, final = args.source.resolve(), args.output.resolve()
    for required in ("infer.sh", "README.md"):
        if not (source / required).is_file():
            sys.exit(f"{source} has no {required} (see /workspace/contract.md)")
    final.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{final.name}.stage-", dir=final.parent))
    try:
        shutil.copytree(source, staged, symlinks=False, dirs_exist_ok=True, ignore=EXCLUDE)
        if args.check:
            check(staged)
        staged.chmod(0o755)
        swap_into(staged, final)
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        raise
    print(f"published {source} -> {final}")


if __name__ == "__main__":
    main()
