#!/bin/bash
#SBATCH --job-name=olmoearth-container
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=earth-%j.log
set -euo pipefail
[[ ${SLURM_JOB_PARTITION:-} == ${EARTH_SLURM_PARTITION:?Set EARTH_SLURM_PARTITION} ]] || exit 2
: "${OLMO_MODEL:?exported package required}" "${OLMO_OUTPUT:?output directory required}"
OLMO_DATA=${OLMO_EVAL_DATA:-${OLMO_DATA:-${EARTH_DATA_ROOT:?Set EARTH_DATA_ROOT}/olmoearth/eval}}
OLMO_IMAGE=${OLMO_IMAGE:-${EARTH_RUNTIME_ROOT:?Set EARTH_RUNTIME_ROOT}/images/olmoearth-verifier-v7.sif}
mkdir -p "$OLMO_OUTPUT"
OLMO_TMP=$(mktemp -d "$OLMO_OUTPUT/.container-tmp.XXXXXX")
trap 'rm -rf -- "$OLMO_TMP"' EXIT
OLMO_COMMAND=(bash /tests/test.sh)
OLMO_DATA_BIND="$OLMO_DATA:/data/eval:ro"
if [[ ${OLMO_VERIFIER_PHASE:-full} == inference ]]; then
  [[ -z ${OLMO_INFERENCE_TASK:-} ]] || exit 2
  OLMO_COMMAND=(python /tests/grade.py --model /workspace/submission
    --data /data/eval --output /logs/verifier --probes /evaluation/probes.py
    --inference-only)
  OLMO_DATA_BIND="$OLMO_DATA/awf/inputs.h5:/data/eval/awf/inputs.h5:ro,$OLMO_DATA/lfmc/inputs.h5:/data/eval/lfmc/inputs.h5:ro"
elif [[ ${OLMO_VERIFIER_PHASE:-full} != full ]]; then
  exit 2
fi
if [[ -n ${OLMO_INFERENCE_TASK:-} ]]; then
  [[ $OLMO_INFERENCE_TASK == awf || $OLMO_INFERENCE_TASK == lfmc ]] || exit 2
  OLMO_COMMAND=(python /tests/replay.py --model /workspace/submission
    --input "/data/eval/$OLMO_INFERENCE_TASK/inputs.h5" --output /logs/verifier)
fi
apptainer exec --cleanenv --nv --containall --no-mount home,cwd,bind-paths \
  --pwd /tmp --bind "/dev/full:/dev/full,$OLMO_TMP:/tmp" \
  --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES},SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION},SLURM_JOB_ID=${SLURM_JOB_ID},OMP_NUM_THREADS=4,OPENBLAS_NUM_THREADS=4" \
  --bind "$OLMO_MODEL:/workspace/submission:ro,$OLMO_DATA_BIND,$OLMO_OUTPUT:/logs/verifier" \
  "$OLMO_IMAGE" "${OLMO_COMMAND[@]}"
