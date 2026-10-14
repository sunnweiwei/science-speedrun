#!/usr/bin/env bash
#SBATCH --job-name=wn2-cudnn-cache-fix
#SBATCH --partition=compsci-gpu
#SBATCH --account=users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=00:10:00
set -euo pipefail
WN2_ROOT=/usr/project/xtmp/sc829/weather-track/wn2-pilot
WN2_RUN=${WN2_ROOT}/cudnn-oracles/${SLURM_JOB_ID}
mkdir -p "${WN2_RUN}/code"
cp "${SLURM_SUBMIT_DIR}/probabilistic-global-weather/slurm/wn2_query_chunks.py" \
  "${SLURM_SUBMIT_DIR}/probabilistic-global-weather/slurm/wn2_boolean_masks.py" \
  "${SLURM_SUBMIT_DIR}/probabilistic-global-weather/slurm/check_wn2_query_chunks.py" "${WN2_RUN}/code/"
cp "${SLURM_SUBMIT_DIR}/probabilistic-global-weather/slurm/wn2_cudnn_attention.py" \
  "${SLURM_SUBMIT_DIR}/probabilistic-global-weather/slurm/check_wn2_cudnn_attention.py" "${WN2_RUN}/code/"
WN2_EXTRA=()
if [[ ${WEATHER_NATIVE_COARSE:-0} == 1 ]]; then WN2_EXTRA=(--native-coarse); fi
if [[ ${WEATHER_NATIVE_FINE:-0} == 1 ]]; then WN2_EXTRA=(--native-fine); fi
if [[ ${WEATHER_PMAP_REMAT:-0} == 1 ]]; then WN2_EXTRA+=(--pmap-remat); fi
if [[ ${WEATHER_SCAN_ENSEMBLE:-0} == 1 ]]; then WN2_EXTRA+=(--scan-ensemble); fi
run_case() {
apptainer exec --nv --containall --cleanenv \
  --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" --env PYTHONHASHSEED=0 \
  --env OMP_NUM_THREADS=2 --env OPENBLAS_NUM_THREADS=2 \
  --env XLA_PYTHON_CLIENT_PREALLOCATE=false --env XLA_PYTHON_CLIENT_ALLOCATOR=platform \
  --env "XLA_FLAGS=${WEATHER_CUDNN_XLA_FLAGS:---xla_gpu_exclude_nondeterministic_ops=true}" \
  --env CUBLAS_WORKSPACE_CONFIG=:4096:8 --env JAX_ENABLE_COMPILATION_CACHE=false \
  --bind "${WN2_RUN}:/run" \
  /usr/project/xtmp/sc829/weather-track/images-final/environment.sif \
  /opt/weather-python/bin/python /run/code/check_wn2_cudnn_attention.py \
  --output "/run/${1}.json" --layers 2 "${WN2_EXTRA[@]}" "${@:2}" > "${WN2_RUN}/${1}.log" 2>&1
}

if run_case unfixed; then
  echo 'Expected the unfixed repeated-layer compiler bug to reproduce' >&2
  exit 1
fi
python3 - "${WN2_RUN}/unfixed.json" <<'PY_CHECK'
import json,sys
r=json.load(open(sys.argv[1]))
assert r['status']=='failed' and 'raw_hash_map' in r.get('error',''),r
PY_CHECK
run_case fixed --deterministic-cache-fix
python3 - "${WN2_RUN}" <<'PY_CHECK'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]);fixed=json.loads((r/'fixed.json').read_text())
assert fixed['status']=='measured_not_accepted'
assert all(c['repeat_exact'] for c in fixed['cases'])
(r/'regression.json').write_text(json.dumps(dict(status='passed',unfixed_raw_hash_map_reproduced=True,fixed_two_layer_gpu_cases=len(fixed['cases']),determinism_required=True,scope='Repeated-layer cuDNN compiler regression, not full-model equivalence'),indent=2)+'\n')
PY_CHECK
