#!/usr/bin/env bash
# JAX uses its pip-installed CUDA libraries, not the MAIA/NVHPC module stack.
# The image's Python itself still needs EasyBuild's Python/GCC shared libraries.
set -e
source /etc/profile
module purge
unset LD_LIBRARY_PATH
module load Python/3.12.3-GCCcore-13.3.0
source /home/easybuild/venvs/hydrogym_gpu/bin/activate
set -u
export HYDROGYM_JAX_RUNTIME_SOURCE="$(readlink -f -- "${BASH_SOURCE[0]}")"
exec "$@"
