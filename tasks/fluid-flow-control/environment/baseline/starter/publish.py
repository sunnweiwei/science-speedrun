"""Publish code + one inference checkpoint as one atomically replaced zipapp."""

import io
import json
import os
from pathlib import Path
import tempfile
import zipfile

import torch


def publish(policy, config, submission, training_steps):
    submission = Path(submission)
    submission.mkdir(parents=True, exist_ok=True)
    config = dict(config, training_steps=int(training_steps),
                  parameter_count=sum(p.numel() for p in policy.parameters()))
    state = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
    buffer = io.BytesIO()
    torch.save(state, buffer)
    source = Path(__file__).resolve().parent
    fd, name = tempfile.mkstemp(prefix=".publishing-", suffix=".tmp", dir=submission)
    try:
        with os.fdopen(fd, "w+b") as stream:
            # A relocated verifier may use a different UID than the trainer.
            os.fchmod(stream.fileno(), 0o644)
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
                archive.writestr("checkpoint.pt", buffer.getvalue())
                archive.writestr("config.json", json.dumps(config, allow_nan=False))
                archive.writestr("__main__.py", (source / "predict.py").read_text())
                archive.writestr("shared.py", (source / "shared.py").read_text())
            stream.flush()
            os.fsync(stream.fileno())
        # Code and weights move together. No symlinks to training paths.
        os.replace(name, submission / "policy.pyz")
    finally:
        Path(name).unlink(missing_ok=True)
    entry = submission / "infer.sh"
    if not entry.exists():
        entry.write_text('#!/usr/bin/env bash\nset -euo pipefail\nroot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"\nexec python "$root/policy.pyz"\n')
        entry.chmod(0o755)
    readme = submission / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Shared PPO candidate\n\nOne task-conditioned actor/critic parameter set; no per-task model bank. "
            "Code and one checkpoint are packaged together in policy.pyz; optimizer state is excluded. "
            "Inspect config.json inside that archive for exact parameter count, training steps, "
            "supported observation/action sizes and known task names. Unknown names use zero task "
            "conditioning, not a different checkpoint. Dependencies: PyTorch, NumPy, Gymnasium, SB3 "
            "from the declared runtime. Input clipping/padding and native action conversion are "
            "in shared.py.\n\nUpdate this README with actual training provenance, measured per-case "
            "results, development feedback use and limitations. This starter is not a SOTA claim.\n")
