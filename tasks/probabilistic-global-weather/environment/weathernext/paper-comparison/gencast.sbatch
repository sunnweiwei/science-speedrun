#!/bin/bash
#SBATCH --job-name=paper-gencast
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=768G
#SBATCH --gres=gpu:8
#SBATCH --time=36:00:00
set -euo pipefail
root=${PAPER_ROOT:?}
base=${PAPER_WORKER_BASE:?}
source=${PAPER_SOURCE:-$root/source}
devices=${PAPER_GPUS:-8}
mkdir -p "$root/predictions/gencast" "$root/logs"
pids=()
cleanup() { for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT
for ((local_worker=0; local_worker<devices; local_worker++)); do
    worker=$((base + local_worker))
    apptainer exec --nv --containall --cleanenv \
      --env CUDA_VISIBLE_DEVICES="$local_worker" \
      --env XLA_PYTHON_CLIENT_PREALLOCATE=true --env XLA_PYTHON_CLIENT_MEM_FRACTION=0.80 --env OPENBLAS_NUM_THREADS=2 \
      --bind "$source:/source:ro" \
      --bind "$root/gencast-operational-lt2022:/assets:ro" \
      --bind "$root/data/inputs:/inputs:ro" \
      --bind "$root/predictions/gencast:/output" \
      /usr/project/xtmp/sc829/weather-track/images-final/environment.sif \
      /opt/weather-python/bin/python /source/paper-comparison/predict_gencast.py \
      --assets /assets --inputs /inputs --output /output --worker "$worker" --workers 1024 \
      > "$root/logs/gencast-worker-$worker.log" 2>&1 &
    pid=$!
    pids+=("$pid")
    # Gate only first compilation. Existing GPU workers continue forecasting.
    while ! rg -q '"lead": 12' "$root/logs/gencast-worker-$worker.log"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" || exit 1
            break
        fi
        sleep 10
    done
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
