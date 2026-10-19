"""Shared storage/provenance helpers; no downloads or GPU allocation on import."""
import hashlib
import json
import os
import subprocess
from pathlib import Path


def external(path):
    path = Path(path).expanduser().resolve()
    if path == Path("/home") or Path("/home") in path.parents:
        raise ValueError(f"Large artifacts must be outside /home: {path}")
    return path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def device_guard(device):
    if device.startswith("cuda"):
        if (
            not os.environ.get("SLURM_JOB_ID")
            or os.environ.get("SLURM_JOB_PARTITION") != os.environ.get('PATHFINDER_GPU_PARTITION', 'gpu')
        ):
            raise RuntimeError("GPU work requires a gpu Slurm allocation")


def checked_file(root, relative, expected_hash):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError("Invalid submission path")
    if sha256(path) != expected_hash:
        raise ValueError(f"Hash mismatch: {relative}")
    return path


def source_provenance():
    root = Path(__file__).resolve().parent
    files = {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*.py"))}
    tree_hash = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    version = os.environ.get("UPSTREAM_SOURCE_COMMIT")
    if not version:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True
            )
            version = result.stdout.strip() if result.returncode == 0 else "unversioned-copy"
        except FileNotFoundError:
            # Runtime images need not include Git. The actual Python source
            # content remains independently identified by tree_hash above.
            version = "unversioned-copy"
    return dict(commit=version, python_tree_sha256=tree_hash)


def load_npz(path):
    """Bounded ROI arrays, closed before validation can raise (important on NFS)."""
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


from contextlib import contextmanager


@contextmanager
def validation_memory_gate(component):
    """Serialize whole-volume validation when two native trainers share host RAM."""
    name = os.environ.get("EM_VALIDATION_LOCK")
    if not name:
        yield
        return
    import fcntl
    import time
    lock = external(name)
    lock.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(dict(stage="validation_wait", component=component)), flush=True)
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        def record(state):
            row = dict(component=component, state=state, pid=os.getpid(), monotonic_ns=time.monotonic_ns())
            status = Path("/proc/self/status")
            if status.exists():
                for line in status.read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        row["rss_kib"] = int(line.split()[1])
            # Both records are appended under the same lock, including on NFS.
            with lock.with_suffix(".events.jsonl").open("a") as events:
                events.write(json.dumps(row) + "\n")
            print(json.dumps(dict(stage="validation_memory_gate", **row)), flush=True)
        record("acquired")
        try:
            yield
        finally:
            # Return unused CPU allocator pages before admitting the other model.
            import gc
            import ctypes
            gc.collect()
            trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
            if trim is not None:
                trim.argtypes = [ctypes.c_size_t]
                trim.restype = ctypes.c_int
                trim(0)
            record("released")
            fcntl.flock(handle, fcntl.LOCK_UN)
