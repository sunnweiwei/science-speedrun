#!/usr/bin/env bash
#SBATCH --job-name=alphaqubit-status
#SBATCH --account=users
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
set -uo pipefail
: "${QEC_TASK_ROOT:?}"
: "${QEC_ASSET_ROOT:?}"
export PYTHONPATH="${QEC_TASK_ROOT}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu
qec_python="${QEC_TASK_ROOT}/alphaqubit-runtime/.venv/bin/python"
"${qec_python}" -m baselines.accounting --asset-root "${QEC_ASSET_ROOT}"
qec_accounting_status=$?
"${qec_python}" -m baselines.alphaqubit_report --asset-root "${QEC_ASSET_ROOT}"
qec_report_status=$?
if (( qec_accounting_status != 0 )); then exit "${qec_accounting_status}"; fi
exit "${qec_report_status}"
