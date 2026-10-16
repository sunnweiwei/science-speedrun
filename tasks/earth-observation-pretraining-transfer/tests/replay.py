"""Run exported model in a filesystem/network-isolated process."""

import argparse
import errno
from pathlib import Path
import shutil
import subprocess
import tempfile
import sys

checks = Path(__file__).resolve().parents[1] / "environment/evaluation"
if not checks.is_dir():
    checks = Path(__file__).resolve().parent
if not (checks / "package_contract.py").is_file():
    checks = Path("/evaluation")
sys.path.insert(0, str(checks))
from package_contract import InvalidSubmission, validate_features
from namespace import command, direct

p = argparse.ArgumentParser()
p.add_argument("--model", type=Path, required=True)
p.add_argument("--input", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--no-namespace", action="store_true",
               help="development only: run without Bubblewrap where user namespaces are unavailable")
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
# Only unlabeled pixels and an empty output directory enter model execution.
# Previous scores/predictions in the destination are never mounted.
with tempfile.TemporaryDirectory(prefix="olmo-replay-", dir=a.output) as temp:
    workspace = Path(temp)
    inputs = workspace / "input"
    inputs.mkdir()
    outputs = workspace / "output"
    outputs.mkdir()
    dst = inputs / "inputs.h5"
    try:
        dst.hardlink_to(a.input)
    except OSError as exc:
        if exc.errno not in (errno.EXDEV, errno.EPERM, errno.EACCES):
            raise
        shutil.copyfile(a.input, dst)
    if a.no_namespace:
        argv, environment = direct(a.model.resolve(), inputs.resolve(), outputs.resolve())
        subprocess.run(argv, check=True, env=environment, cwd=a.model.resolve())
    else:
        subprocess.run(
            command(a.model.resolve(), inputs.resolve(), outputs.resolve()), check=True
        )
    result = outputs / "features.npz"
    try:
        validate_features(result, a.input)
    except InvalidSubmission as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(42)
    result.replace(a.output / "features.npz")
