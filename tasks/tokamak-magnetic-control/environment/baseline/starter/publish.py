"""Atomic publication of a NumPy actor as a contract-conforming package."""

import errno
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

HERE = Path(__file__).resolve().parent

INFER_SH = """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec python3 policy.py
"""


def publish(actor_dir, submission, provenance=None):
    """Copy actor.npz + config.json + inference code into ``submission`` atomically."""
    actor_dir, submission = Path(actor_dir), Path(submission)
    submission.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".publish-", dir=submission.parent))
    for name in ("actor.npz", "config.json"):
        shutil.copy2(actor_dir / name, tmp / name)
    shutil.copy2(HERE / "common.py", tmp / "common.py")
    cfg0 = json.loads((actor_dir / "config.json").read_text())
    core = cfg0.get("residual_package")
    if core:
        core = Path(core)
        # every resource of the base package (e.g. baseline/shape_vc/package: model.npz,
        # estimator.npz, params.json; the PID package: estimator.npz, machine.json); its
        # policy.py becomes pid_core.py and is driven by predict_residual.py
        for f in core.iterdir():
            if f.is_file() and f.name not in ("policy.py", "infer.sh", "README.md", "actor.npz", "config.json", "common.py"):
                shutil.copy2(f, tmp / f.name)
        shutil.copy2(core / "policy.py", tmp / "pid_core.py")
        shutil.copy2(HERE / "predict_residual.py", tmp / "policy.py")
    else:
        shutil.copy2(HERE / "predict.py", tmp / "policy.py")
    (tmp / "infer.sh").write_text(INFER_SH)
    os.chmod(tmp / "infer.sh", 0o755)
    cfg = json.loads((tmp / "config.json").read_text())
    import numpy as np
    data = np.load(tmp / "actor.npz")
    n_params = int(sum(int(np.prod(data[k].shape)) for k in data.files))
    title = (f"# Residual SAC actor on a deployable base controller ({Path(core).parent.name}/{Path(core).name}; tokamak-policy/v1)\n\n" if core
             else "# Starter SAC actor (tokamak-policy/v1)\n\n")
    (tmp / "README.md").write_text(
        title +
        f"- Architecture: MLP {cfg['n_obs']} -> 256 -> 256 -> 256 -> {cfg['n_act']}, tanh output scaled to voltage limits\n"
        f"- Parameters: {n_params}\n- Deterministic NumPy inference; no torch dependency\n"
        f"- Provenance: {json.dumps(provenance or {})}\n- Published: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
        "- One shared parameter set for every case; per-episode state: none\n")
    old = None
    if submission.exists():
        old = submission.with_name(submission.name + ".old-" + str(int(time.time())))
        try:
            os.replace(submission, old)
        except OSError as error:
            # A directory from a lower image layer (the Dockerfile creates /workspace/submission)
            # cannot be renamed on overlayfs (EXDEV). Move its contents aside instead; publishing
            # the new package below is still a single rename.
            if error.errno != errno.EXDEV:
                raise
            shutil.copytree(submission, old, symlinks=True)
            shutil.rmtree(submission)
    os.replace(tmp, submission)
    if old is not None:
        shutil.rmtree(old, ignore_errors=True)
    return submission


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor-dir", required=True)
    ap.add_argument("--submission", default="/workspace/submission")
    a = ap.parse_args()
    print(publish(a.actor_dir, a.submission))
