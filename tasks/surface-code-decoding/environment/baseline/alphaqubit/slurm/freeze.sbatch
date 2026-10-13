#!/usr/bin/env bash
#SBATCH --job-name=qec-freeze
#SBATCH --account=users
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail
source "${QEC_TASK_ROOT}/slurm/runtime.sh"
: "${QEC_CAMPAIGN:=native-v1}"
export QEC_CAMPAIGN
for qec_stage in pretrain finetune; do
  "${QEC_TASK_ROOT}/.venv/bin/python" -m baselines.package \
    --models-root "${QEC_ASSET_ROOT}/${QEC_CAMPAIGN}/models" \
    --stage "${qec_stage}" \
    --output "${QEC_ASSET_ROOT}/${QEC_CAMPAIGN}/packages/${qec_stage}"
done
"${QEC_TASK_ROOT}/.venv/bin/python" - <<'PY'
import os
import shutil
import uuid
from pathlib import Path
from baselines.data import discover_conditions
from baselines.package import verify_package
root = Path(os.environ['QEC_ASSET_ROOT'])
campaign = root / os.environ['QEC_CAMPAIGN']
for stage in ('pretrain', 'finetune'):
    source, destination = campaign / 'packages' / stage, campaign / 'replay' / stage
    verify_package(source)
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name('.' + stage + '-' + uuid.uuid4().hex)
        try:
            shutil.copytree(source, temporary)
            verify_package(temporary)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    verify_package(destination)
    if (source / 'manifest.json').read_bytes() != (destination / 'manifest.json').read_bytes():
        raise ValueError('Relocated package differs from the frozen package')
# Independent inference input directory contains only detectors and geometry.
# This removes labels from its data interface; it is not a security sandbox.
conditions = discover_conditions(root / 'data/sycamore')
input_root = campaign / 'replay-inputs'
if input_root.exists() and {p.name for p in input_root.iterdir()} - {c.name for c in conditions}:
    raise ValueError('Unexpected entries in detector-only replay input directory')
for condition in conditions:
    destination = campaign / 'replay-inputs' / condition.name
    destination.mkdir(parents=True, exist_ok=True)
    if {p.name for p in destination.iterdir()} - {'circuit_ideal.stim', 'detection_events.b8'}:
        raise ValueError('Unexpected files in detector-only replay condition')
    for name in ('circuit_ideal.stim', 'detection_events.b8'):
        shutil.copy2(condition.path / name, destination / name)
PY
for qec_stage in pretrain finetune; do
  uv sync --frozen --project "${QEC_ASSET_ROOT}/${QEC_CAMPAIGN}/replay/${qec_stage}"
done
