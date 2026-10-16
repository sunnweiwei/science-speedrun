#!/usr/bin/env bash
#SBATCH --job-name=alphaqubit-freeze
#SBATCH --account=users
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail
: "${QEC_TASK_ROOT:?}"
: "${QEC_ASSET_ROOT:?}"
export PYTHONPATH="${QEC_TASK_ROOT}"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
qec_python="${QEC_TASK_ROOT}/alphaqubit-runtime/.venv/bin/python"
qec_campaign="${QEC_ASSET_ROOT}/alphaqubit-v1"
"${qec_python}" -m baselines.alphaqubit_artifact \
  --models-root "${qec_campaign}/models" --stage both \
  --output "${qec_campaign}/packages/both"
"${qec_python}" - <<'PY'
import os
from pathlib import Path
import shutil
import tempfile
from baselines.alphaqubit_artifact import verify_package, sha256_file
from baselines.data import discover_conditions

root = Path(os.environ['QEC_ASSET_ROOT'])
campaign = root / 'alphaqubit-v1'
source, destination = campaign / 'packages/both', campaign / 'replay/both'
verify_package(source)
destination.parent.mkdir(parents=True, exist_ok=True)
if not destination.exists():
    temporary = Path(tempfile.mkdtemp(prefix='.both-', dir=destination.parent))
    try:
        shutil.copytree(source, temporary, dirs_exist_ok=True)
        verify_package(temporary)
        os.rename(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
verify_package(destination)
if (source / 'manifest.json').read_bytes() != (destination / 'manifest.json').read_bytes():
    raise ValueError('Relocated package differs from frozen package')
# The replay interface contains geometry and detectors only.
conditions = discover_conditions(root / 'data/sycamore')
inputs = campaign / 'replay-inputs'
inputs.mkdir(exist_ok=True)
if {p.name for p in inputs.iterdir()} - {c.name for c in conditions}:
    raise ValueError('Unexpected replay input directory')
for condition in conditions:
    folder = inputs / condition.name
    folder.mkdir(exist_ok=True)
    if {p.name for p in folder.iterdir()} - {'circuit_ideal.stim', 'detection_events.b8'}:
        raise ValueError('Unexpected replay input file')
    for name in ('circuit_ideal.stim', 'detection_events.b8'):
        shutil.copy2(condition.path / name, folder / name)
        if sha256_file(condition.path / name) != sha256_file(folder / name):
            raise ValueError('Replay input copy checksum mismatch')
PY
uv sync --frozen --python 3.11 --project "${qec_campaign}/replay/both"
