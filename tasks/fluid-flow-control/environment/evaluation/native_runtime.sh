#!/usr/bin/env bash
set -eo pipefail
backend="$1"
shift
source /etc/profile
module purge
case "$backend" in
  maia) module load MAIA/1.0-NVHPC-26.1 ;;
  jax|jax-fluids) unset LD_LIBRARY_PATH; module load Python/3.12.3-GCCcore-13.3.0 ;;
  *) echo "Unsupported native backend: $backend" >&2; exit 2 ;;
esac
source /home/easybuild/venvs/hydrogym_gpu/bin/activate
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec "$@"
