"""Copy, check CPU inference, and atomically exchange a complete submission."""
import argparse
import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import h5py

# Both the repository and the copied /workspace layout are supported.
root = Path(__file__).resolve().parent
checks = root.parent / 'evaluation'
if not checks.is_dir():
    checks = root.parents[1] / 'evaluation'
sys.path.insert(0, str(checks))
from package_contract import validate_package, validate_features


def exchange(staged, destination):
    """Linux renameat2 exchanges whole directories without a missing-path window."""
    if not destination.exists():
        os.rename(staged, destination)
        return None
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError('Submission must be a directory, not a symlink')
    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renameat2
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(staged), -100, os.fsencode(destination), 2):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return staged  # Retain the previous complete package for rollback.


def publish(source, destination, check_input):
    source, destination = Path(source).resolve(), Path(destination).absolute()
    if source == destination.resolve():
        raise ValueError('Export to a separate candidate directory first')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f'.{destination.name}-previous-', dir=destination.parent))
    try:
        shutil.copytree(source, staged, dirs_exist_ok=True, symlinks=True)
        validate_package(staged)
        with tempfile.TemporaryDirectory(prefix='encoder-publish-check-') as tmp:
            output = Path(tmp) / 'features.npz'
            sample = Path(tmp) / 'inputs.h5'
            with h5py.File(check_input, 'r') as original, h5py.File(sample, 'w') as tiny:
                tiny.attrs.update(original.attrs)
                for name, dataset in original.items():
                    tiny.create_dataset(name, data=dataset[:2])
            subprocess.run([
                sys.executable, str(staged / 'infer.py'), '--model', str(staged),
                '--input', str(sample), '--output', str(output),
                '--device', 'cpu',
            ], check=True, timeout=300, cwd=tmp,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONHASHSEED='0'))
            validate_features(output, sample)
        previous = exchange(staged, destination)
    except BaseException:
        shutil.rmtree(staged)
        raise
    if previous:
        print(f'Previous package retained at {previous}')
    print(f'Published {destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--submission', type=Path, default=Path('/workspace/submission'))
    parser.add_argument('--check-input', type=Path, required=True, help='Unlabeled development H5; checks at most two rows')
    args = parser.parse_args()
    publish(args.candidate, args.submission, args.check_input)
