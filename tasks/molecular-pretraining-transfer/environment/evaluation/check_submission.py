#!/usr/bin/env python3
"""Contract check of an encoder package, as the verifier does it before fine-tuning (protocol/check_encoder.py):
files, encoder.json, the 90 M parameter budget and that every encoder weight is in checkpoint.pt.

    python /workspace/evaluation/check_submission.py [PACKAGE]    (default /workspace/submission)
"""
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
pkg = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/submission").resolve()
for name in ("encoder.json", "checkpoint.pt"):
    if not (pkg / name).is_file():
        sys.exit(f"{pkg} has no {name}")
links = [p for p in pkg.rglob("*") if p.is_symlink()]
if links:
    sys.exit(f"symlinks are not allowed in the package: {links[:3]}")
with tempfile.TemporaryDirectory() as tmp:
    code = subprocess.run([sys.executable, str(HERE / "protocol" / "check_encoder.py"), str(pkg),
                           "/workspace/data/downstream", str(Path(tmp) / "check.json")],
                          env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/opt/conda/bin:/usr/local/bin:/usr/bin:/bin"}).returncode
sys.exit(code)
