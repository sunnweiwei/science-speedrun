#!/bin/bash
#SBATCH --job-name=olmoearth-validate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=earth-%j.log
set -euo pipefail
[[ ${SLURM_JOB_PARTITION:-} == ${EARTH_SLURM_PARTITION:?Set EARTH_SLURM_PARTITION} ]] || exit 2
OLMO_ROOT=${OLMO_ROOT:-${OLMO_DATA:-${EARTH_DATA_ROOT:?Set EARTH_DATA_ROOT}/olmoearth}}
OLMO_UPSTREAM=${OLMO_UPSTREAM:-${EARTH_RUNTIME_ROOT:?Set EARTH_RUNTIME_ROOT}/olmoearth-upstream}
OLMO_TRACK=${OLMO_TRACK:-${EARTH_REPO_ROOT:?Set EARTH_REPO_ROOT}/earth-observation-pretraining-transfer}
OLMO_PY=$OLMO_UPSTREAM/.venv/bin/python
: "${OLMO_NAME:?run name required}"
OLMO_EXPORT_ARGS=()
if [[ -n ${OLMO_CHECKPOINT:-} ]]; then
  OLMO_EXPORT_ARGS+=(--checkpoint "$OLMO_CHECKPOINT")
fi
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
if [[ ${OLMO_USE_EXISTING_PACKAGE:-0} == 1 ]]; then
  test -f "$OLMO_ROOT/packages/$OLMO_NAME/provenance.json"
else
  "$OLMO_PY" "$OLMO_TRACK/environment/baseline/starter/export.py" --upstream "$OLMO_UPSTREAM" --variant "${OLMO_VARIANT:-tiny}" --seed "${OLMO_SEED:-12536}" "${OLMO_EXPORT_ARGS[@]}" --output "$OLMO_ROOT/packages/$OLMO_NAME"
fi
read -r -a OLMO_TASK_NAMES <<< "${OLMO_TASKS:-awf lfmc}"
for task in "${OLMO_TASK_NAMES[@]}"; do
  [[ "$task" == awf || "$task" == lfmc ]] || exit 2
  "$OLMO_PY" "$OLMO_TRACK/tests/replay.py" --model "$OLMO_ROOT/packages/$OLMO_NAME" --input "$OLMO_ROOT/eval/$task/inputs.h5" --output "$OLMO_ROOT/features/$OLMO_NAME-$task"
  "$OLMO_PY" "$OLMO_TRACK/environment/evaluation/probes.py" --features "$OLMO_ROOT/features/$OLMO_NAME-$task/features.npz" --labels "$OLMO_ROOT/eval/$task/labels.json" --task "$task" --validation-predictions "$OLMO_ROOT/results/$OLMO_NAME-$task-predictions.npz" --output "$OLMO_ROOT/results/$OLMO_NAME-$task.json"
done
