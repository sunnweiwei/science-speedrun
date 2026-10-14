#!/usr/bin/env bash
#SBATCH --job-name=wn2-four-gpu
#SBATCH --partition=compsci-gpu
#SBATCH --account=users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --gres=gpu:rtx_pro_6000:4
#SBATCH --time=01:30:00
set -euo pipefail
TASK_ROOT=${WEATHER_TASK_ROOT:-${SLURM_SUBMIT_DIR}/probabilistic-global-weather/environment/weathernext}
WN2_ROOT=${WEATHER_WN2_ROOT:-/usr/project/xtmp/sc829/weather-track/wn2-pilot}
WN2_IMAGE=${WEATHER_IMAGE:-/usr/project/xtmp/sc829/weather-track/images-final/environment.sif}
WN2_RUN=${WN2_ROOT}/runs/${SLURM_JOB_ID}
mkdir -p "${WN2_RUN}"
cp "${TASK_ROOT}/slurm/wn2_validation.py" "${TASK_ROOT}/slurm/wn2_distributed.py" "${WN2_RUN}/"
sha256sum "${WN2_IMAGE}" > "${WN2_RUN}/image.sha256"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "${WN2_RUN}/hardware.csv"
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv --loop=2 > "${WN2_RUN}/gpu-memory.csv" &
WN2_MONITOR=$!
trap 'kill "${WN2_MONITOR}" 2>/dev/null || true' EXIT
for WN2_PHASE in initial resumed; do
  WN2_EXTRA=()
  WN2_XLA='--xla_gpu_exclude_nondeterministic_ops=true --xla_gpu_dump_autotune_results_to=/run/autotune.pb'
  if [[ "${WN2_PHASE}" == resumed ]]; then
    WN2_EXTRA=(--resume /run/initial/state-1.npz)
    WN2_XLA='--xla_gpu_exclude_nondeterministic_ops=true --xla_gpu_load_autotune_results_from=/run/autotune.pb --xla_gpu_require_complete_aot_autotune_results=true'
  fi
  apptainer exec --nv --containall --cleanenv \
    --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" --env PYTHONUNBUFFERED=1 \
    --env OMP_NUM_THREADS=8 --env OPENBLAS_NUM_THREADS=8 \
    --env XLA_PYTHON_CLIENT_PREALLOCATE=false --env XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    --env "XLA_FLAGS=${WN2_XLA}" \
    --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    --env JAX_COMPILATION_CACHE_DIR=/run/jax-cache \
    --env JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 \
    --env JAX_ENABLE_COMPILATION_CACHE=false \
    --env 'JAX_DEBUG_LOG_MODULES=jax._src.compiler,jax._src.compilation_cache' \
    --bind "${WN2_ROOT}/assets:/assets:ro" --bind "${WN2_RUN}:/run" \
    "${WN2_IMAGE}" /opt/weather-python/bin/python /run/wn2_validation.py \
    --assets /assets --output "/run/${WN2_PHASE}" --mode random-train \
    --precision bfloat16 --distributed --accumulation 16 "${WN2_EXTRA[@]}" \
    > "${WN2_RUN}/${WN2_PHASE}.log" 2>&1
done
apptainer exec --containall --cleanenv --bind "${WN2_RUN}:/run" "${WN2_IMAGE}" \
  /opt/weather-python/bin/python - <<'PY'
import json
import numpy as np
with np.load('/run/initial/state-2.npz',allow_pickle=False) as a, np.load('/run/resumed/state-2.npz',allow_pickle=False) as b:
    assert set(a.files)==set(b.files)
    for name in a.files: np.testing.assert_array_equal(a[name],b[name],err_msg=name)
    count=len(a.files)-1
with open('/run/replay.json','w') as f:
    json.dump(dict(status='passed',exact_state_leaves=count,scope='Fresh-process update-two continuation; one public fixture, no research data ordering'),f,indent=2)
print('WN2_FOUR_GPU_ACCUMULATION_EXACT_RESUME_OK')
PY
