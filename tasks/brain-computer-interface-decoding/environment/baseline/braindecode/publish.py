#!/usr/bin/env python3
"""Publish a package directory to /workspace/submission atomically.

    python /workspace/baseline/braindecode/publish.py SRC [--check]

The verifier may read /workspace/submission at any moment, so the package must never be half
written. This copies SRC (a version symlink written by train_shared.py is resolved) into a
staging directory next to the target, optionally runs its infer.sh on a slice of every
validation split and checks the predictions with the verifier's reader, then swaps it in.
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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'evaluation'))
from bci_score import CLASSES, read_predictions  # noqa: E402

EXCLUDE = shutil.ignore_patterns('__pycache__', '*.pyc', 'resume*', 'optimizer*.pt')


def swap_into(staged, final):
    old = final.with_name(f'.{final.name}.old-{os.getpid()}')
    if final.exists() or final.is_symlink():
        try:
            os.replace(final, old)
        except OSError as error:
            # A directory from a lower image layer cannot be renamed on overlayfs (EXDEV).
            if error.errno != errno.EXDEV:
                raise
            shutil.copytree(final, old, symlinks=True)
            shutil.rmtree(final)
    os.replace(staged, final)
    if old.is_symlink() or old.is_file():
        old.unlink()
    else:
        shutil.rmtree(old, ignore_errors=True)


def check(staged, data, samples):
    for track, classes in CLASSES.items():
        split = data / track / 'val'
        manifest = json.loads((split / 'manifest.json').read_text())
        rows = manifest['samples'][:samples]
        with tempfile.TemporaryDirectory() as work:
            inp, out = Path(work) / 'input', Path(work) / 'output'
            inp.mkdir()
            out.mkdir()
            # One fresh shard with the sampled windows, as the verifier stages copied inputs.
            arrays = {}
            for r in rows:
                if r['path'] not in arrays:
                    arrays[r['path']] = np.load(split / r['path'], mmap_mode='r')
            np.save(inp / 'shard-000000.npy', np.stack([arrays[r['path']][r['index']] for r in rows]))
            public = [{'id': r['id'], 'path': 'shard-000000.npy', 'index': i} for i, r in enumerate(rows)]
            (inp / 'manifest.json').write_text(json.dumps({**manifest, 'samples': public}))
            subprocess.run(['bash', str(staged / 'infer.sh'), str(inp), str(out)], check=True)
            read_predictions(out / 'predictions.jsonl', rows, classes)
        print(f'{track}: {len(rows)} validation predictions follow contract.md', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, default=Path('/workspace/submission'))
    parser.add_argument('--data', type=Path, default=Path('/workspace/data'))
    parser.add_argument('--check', action='store_true',
                        help='run infer.sh on a slice of every validation split before publishing')
    parser.add_argument('--check-samples', type=int, default=256)
    args = parser.parse_args()
    source, final = args.source.resolve(), args.output.absolute()
    if not (source / 'infer.sh').is_file():
        sys.exit(f'{source} has no infer.sh')
    if any(p.is_symlink() for p in source.rglob('*')):
        sys.exit('package contents must not contain symlinks (contract.md)')
    staged = Path(tempfile.mkdtemp(prefix=f'.{final.name}.staging-', dir=final.parent))
    os.chmod(staged, 0o755)
    try:
        shutil.copytree(source, staged, ignore=EXCLUDE, dirs_exist_ok=True)
        if args.check:
            check(staged, args.data, args.check_samples)
        swap_into(staged, final)
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        raise
    print(f'published {source} -> {final}')


if __name__ == '__main__':
    main()
