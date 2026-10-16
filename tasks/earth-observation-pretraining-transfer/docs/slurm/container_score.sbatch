#!/bin/bash
#SBATCH --job-name=olmoearth-container-score
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=earth-%j.log
set -euo pipefail
[[ ${SLURM_JOB_PARTITION:-} == ${EARTH_SLURM_PARTITION:?Set EARTH_SLURM_PARTITION} ]] || exit 2
: "${OLMO_MODEL:?exported package required}" "${OLMO_OUTPUT:?completed independent replay required}"
[[ -f "$OLMO_OUTPUT/inference-manifest.json" ]] || exit 2
OLMO_DATA=${OLMO_EVAL_DATA:-${EARTH_DATA_ROOT:?Set EARTH_DATA_ROOT}/olmoearth/eval}
OLMO_IMAGE=${OLMO_IMAGE:-${EARTH_RUNTIME_ROOT:?Set EARTH_RUNTIME_ROOT}/images/olmoearth-verifier-v7.sif}
OLMO_TMP=$(mktemp -d "$OLMO_OUTPUT/.score-tmp.XXXXXX")
trap 'rm -rf -- "$OLMO_TMP"' EXIT
apptainer exec --cleanenv --containall --no-mount home,cwd,bind-paths \
  --pwd /tmp --bind "$OLMO_TMP:/tmp" \
  --env "CUDA_VISIBLE_DEVICES=,OMP_NUM_THREADS=4,OPENBLAS_NUM_THREADS=4" \
  --bind "$OLMO_MODEL:/workspace/submission:ro,$OLMO_DATA:/data/eval:ro,$OLMO_OUTPUT:/logs/verifier" \
  "$OLMO_IMAGE" python /tests/grade.py --model /workspace/submission \
  --data /data/eval --output /logs/verifier --probes /evaluation/probes.py --score-existing
