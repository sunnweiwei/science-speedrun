"""Publish an inference package using one directory rename on the destination filesystem."""
import argparse
import ctypes
import errno
import os
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from policy import load_policy


def publish(source, destination):
    source = source.resolve(strict=True)
    destination = destination.absolute()
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise ValueError("The submission destination must be a real directory")
    destination = destination.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Build the candidate in a directory separate from the published package")
    entry = source / "infer.py"
    if entry.is_symlink() or (entry.exists() and not entry.is_file()):
        raise ValueError("infer.py must be a regular file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rfd4-publish-", dir=destination.parent) as tmp:
        staged = Path(tmp) / "package"
        if entry.is_file():
            shutil.copytree(source, staged, symlinks=True)
        else:
            staged.mkdir()
            shutil.copyfile(source / "policy.json", staged / "policy.json", follow_symlinks=False)
            load_policy(staged)
        staged.chmod(0o755)
        # Replacing an absent or empty destination also works for an empty Docker image-layer directory.
        try:
            os.replace(staged, destination)
            return
        except OSError as exc:
            if exc.errno not in (errno.ENOTEMPTY, errno.EEXIST, errno.EXDEV):
                raise
        if any(destination.iterdir()) and not any((destination / name).is_file() for name in ("infer.py", "policy.json")):
            raise ValueError("The nonempty destination is not an inference package")
        # RENAME_EXCHANGE keeps an existing nonempty package visible until its replacement is ready.
        rename = ctypes.CDLL(None, use_errno=True).renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(-100, os.fsencode(staged), -100, os.fsencode(destination), 2):
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err), str(destination))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("destination", type=Path, nargs="?", default=Path("/workspace/submission"))
    args = parser.parse_args()
    publish(args.candidate, args.destination)
