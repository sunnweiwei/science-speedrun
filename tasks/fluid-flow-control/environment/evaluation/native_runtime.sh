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
export OMPI_MCA_btl="${OMPI_MCA_btl:-self,vader,tcp}"
export OMPI_MCA_btl_vader_single_copy_mechanism="${OMPI_MCA_btl_vader_single_copy_mechanism:-none}"
export OMPI_MCA_coll_hcoll_enable="${OMPI_MCA_coll_hcoll_enable:-0}"
export HYDROGYM_ALLOW_SHARED_GPU="${HYDROGYM_ALLOW_SHARED_GPU:-1}"
exec "$@"
