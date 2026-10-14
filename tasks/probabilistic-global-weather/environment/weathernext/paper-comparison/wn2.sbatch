#!/bin/bash
#SBATCH --job-name=paper-wn2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --gres=gpu:a6000:8
#SBATCH --time=24:00:00
set -euo pipefail
root=${PAPER_ROOT:?}
source=${PAPER_SOURCE:?}
package=${PAPER_PACKAGE:?}
seed=${PAPER_TRAINING_SEED:?}
base=${PAPER_WORKER_BASE:-0}
devices=${PAPER_GPUS:-8}
output="$root/predictions/wn2-seed-$seed"
mkdir -p "$output" "$root/logs"
pids=()
cleanup() { for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT
for ((local_worker=0; local_worker<devices; local_worker++)); do
    worker=$((base + local_worker))
    log="$root/logs/wn2-seed-$seed-worker-$worker.log"
    apptainer exec --nv --containall --cleanenv \
      --env CUDA_VISIBLE_DEVICES="$local_worker" --env PYTHONHASHSEED=0 \
      --env XLA_PYTHON_CLIENT_PREALLOCATE=false --env OPENBLAS_NUM_THREADS=2 \
      --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      --env "XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true --xla_gpu_dump_autotune_results_to=/output/.autotune-worker-$worker.pb" \
      --env "JAX_COMPILATION_CACHE_DIR=/output/.jax-cache-worker-$worker" \
      --env JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 --env JAX_RAISE_PERSISTENT_CACHE_ERRORS=true \
      --bind "$source:/source:ro" --bind "$package:/package:ro" \
      --bind "$root/data/inputs:/inputs:ro" --bind "$output:/output" \
      /usr/project/xtmp/sc829/weather-track/images-final/environment.sif \
      /opt/weather-python/bin/python /source/paper-comparison/predict_wn2.py \
      --package /package --inputs /inputs --output /output --worker "$worker" --workers 1024 \
      > "$log" 2>&1 &
    pid=$!;pids+=("$pid")
    # One real forward gates capacity; subsequent workers compile concurrently.
    # Seed-0 A6000 evidence: ~34 GiB device usage, ~15 GiB observed host RSS,
    # versus 512 GiB per node. GenCast has a separate, much larger host peak.
    if (( local_worker == 0 )); then
        while ! rg -q '"lead": 6' "$log"; do
            if ! kill -0 "$pid" 2>/dev/null; then wait "$pid" || exit 1; break; fi
            sleep 10
        done
    fi
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
