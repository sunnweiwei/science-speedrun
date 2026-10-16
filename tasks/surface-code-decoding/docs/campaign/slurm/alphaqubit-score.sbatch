#!/usr/bin/env bash
#SBATCH --job-name=alphaqubit-score
#SBATCH --account=users
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail
: "${QEC_TASK_ROOT:?}"
: "${QEC_ASSET_ROOT:?}"
export PYTHONPATH="${QEC_TASK_ROOT}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu
qec_python="${QEC_TASK_ROOT}/alphaqubit-runtime/.venv/bin/python"
# Refresh evidence even if a score rejects incomplete/corrupt predictions.
qec_finalize() {
  qec_exit_status=$?
  trap - EXIT
  set +e
  "${qec_python}" -m baselines.accounting --asset-root "${QEC_ASSET_ROOT}"
  qec_accounting_status=$?
  "${qec_python}" -m baselines.alphaqubit_report --asset-root "${QEC_ASSET_ROOT}" \
    --output "${QEC_ASSET_ROOT}/alphaqubit-v1/report.html"
  qec_report_status=$?
  if (( qec_exit_status != 0 )); then exit "${qec_exit_status}"; fi
  if (( qec_accounting_status != 0 )); then exit "${qec_accounting_status}"; fi
  exit "${qec_report_status}"
}
trap qec_finalize EXIT
for qec_stage in pretrain finetune; do
  "${qec_python}" -m baselines.score_neural \
    --predictions "${QEC_ASSET_ROOT}/alphaqubit-v1/predictions/${qec_stage}" \
    --data-root "${QEC_ASSET_ROOT}/data/sycamore" \
    --classical-root "${QEC_ASSET_ROOT}/classical-v1" \
    --output "${QEC_ASSET_ROOT}/alphaqubit-v1/scores-${qec_stage}.json" \
    --bootstrap-samples 2000 --seed 20260919
done
