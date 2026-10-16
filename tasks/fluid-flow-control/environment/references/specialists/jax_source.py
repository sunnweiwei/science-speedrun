"""Fail closed if a JAX diagnostic imports the base image's older editable copy."""

import hashlib
import importlib
from pathlib import Path

REVISION = "4ab9854dea3d84e38a59c25e0f5835a00cf8225f"
EXPECTED = {
    "hydrogym.jax.envs.kolmogorov": "05ba780eb37dd6c93d62720090e57d8256589f2b34796c03fd66d4ea9f5a276c",
    "hydrogym.jax.solvers.base": "0a0abb34bc6d9255aa3844db9dd247fc1eba473dc9f262ec5e3aaf4e90aac61c",
    "hydrogym.jax.utils.utils": "bd4ac12544ef1c00b5479810a88223905d9e9bf86194c85edafaa9750e1d8fe5",
    "hydrogym.jax.env_core": "dcfbd1c8af056035e151eccee81604c5fc0ac20edf4c2ad93793d1d8591883ea",
}


def verify():
    files = {}
    for name, expected in EXPECTED.items():
        path = Path(importlib.import_module(name).__file__)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Wrong imported CFD source: {path}; expected pinned {REVISION}. "
                               "Mount the pinned checkout and set PYTHONPATH=/upstream.")
        files[name] = {"path": str(path), "sha256": actual}
    return {"revision": REVISION, "files": files}
