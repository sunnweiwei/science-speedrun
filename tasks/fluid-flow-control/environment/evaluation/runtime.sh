#!/usr/bin/env bash
set -eo pipefail
# Upstream image's documented MPI/solver stack; fail clearly if it changes.
source /etc/profile
module purge
module load MAIA/1.0-NVHPC-26.1
source /home/easybuild/venvs/hydrogym_gpu/bin/activate
set -u
export OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
exec "$@"
