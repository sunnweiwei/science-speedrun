#!/usr/bin/env bash
# Source inside Slurm jobs; task and artifact roots are supplied by the submitter.
set -euo pipefail
: "${QEC_TASK_ROOT:?Set QEC_TASK_ROOT to the baseline project (surface-code-decoding/environment/baseline/alphaqubit, or /workspace/baseline/alphaqubit)}"
: "${QEC_ASSET_ROOT:?Set QEC_ASSET_ROOT to the persistent artifact directory}"
export PATH="${QEC_TASK_ROOT}/.venv/bin:${PATH}"
qec_cuda_libs="$("${QEC_TASK_ROOT}/.venv/bin/python" -c 'import pathlib,sysconfig; p=pathlib.Path(sysconfig.get_paths()["purelib"]); print(":".join(str(x) for x in sorted((p/"nvidia").glob("*/lib"))))')"
export LD_LIBRARY_PATH="${qec_cuda_libs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${QEC_TASK_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=4
export TF_NUM_INTEROP_THREADS=2
cd "${QEC_TASK_ROOT}"
