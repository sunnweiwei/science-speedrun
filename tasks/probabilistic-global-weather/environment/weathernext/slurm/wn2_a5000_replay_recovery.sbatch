#!/usr/bin/env bash
#SBATCH --job-name=wn2-a5000-replay-recovery
#SBATCH --partition=compsci-gpu
#SBATCH --account=users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:a5000:4
#SBATCH --time=01:30:00
set -euo pipefail
WN2_ROOT=/usr/project/xtmp/sc829/weather-track/wn2-pilot
WN2_RUN=${WN2_ROOT}/a5000-coarse-runs/12540690
# Recovery is specific to the immutable snapshot and saved first update of
# terminal job 12540690. It does not count as a successful original Slurm job.
[[ $(sacct -X -j 12540690 --format=State -n | xargs) == TIMEOUT ]]
[[ -f ${WN2_RUN}/initial/state-1.npz ]]
[[ ! -e ${WN2_RUN}/resumed-timeout-12540690 ]]
mv "${WN2_RUN}/resumed" "${WN2_RUN}/resumed-timeout-12540690"
mv "${WN2_RUN}/resumed.log" "${WN2_RUN}/resumed-timeout-12540690.log"
if [[ -d ${WN2_RUN}/xla-dump ]]; then
  mv "${WN2_RUN}/xla-dump" "${WN2_RUN}/resumed-timeout-12540690/xla-dump"
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "${WN2_RUN}/recovery-${SLURM_JOB_ID}-hardware.csv"
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv --loop=5 > "${WN2_RUN}/recovery-${SLURM_JOB_ID}-gpu-memory.csv" &
WN2_MONITOR=$!
trap 'kill "${WN2_MONITOR}" 2>/dev/null || true' EXIT
WN2_PHASES=(resumed cached)
WN2_DEVICE_ARGS=()
export WEATHER_PROBE_CACHE=true WEATHER_REPLAY_DIAGNOSTIC=1 WEATHER_STAGE_INDEX=0
for WN2_PHASE in "${WN2_PHASES[@]}"; do
WN2_EXTRA=()
WN2_TUNING='--xla_gpu_dump_autotune_results_to=/run/autotune.pb'
if [[ ${WN2_PHASE} == resumed || ${WN2_PHASE} == cached ]]; then
  WN2_EXTRA=(--resume /run/initial/state-1.npz)
  WN2_TUNING='--xla_gpu_load_autotune_results_from=/run/autotune.pb --xla_gpu_require_complete_aot_autotune_results=true'
fi
apptainer exec --nv --containall --cleanenv \
  --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" --env PYTHONUNBUFFERED=1 \
  --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
  --env XLA_PYTHON_CLIENT_PREALLOCATE=false --env XLA_PYTHON_CLIENT_ALLOCATOR=platform \
  --env "XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true ${WN2_TUNING} --xla_dump_to=/run/xla-dump --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*fn_passed_to_pmap.*" \
  --env CUBLAS_WORKSPACE_CONFIG=:4096:8 --env "JAX_ENABLE_COMPILATION_CACHE=${WEATHER_PROBE_CACHE:-false}" \
  --env JAX_COMPILATION_CACHE_DIR=/run/jax-cache --env JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 \
  --env JAX_RAISE_PERSISTENT_CACHE_ERRORS=true \
  --env 'JAX_DEBUG_LOG_MODULES=jax._src.compiler,jax._src.compilation_cache' \
  --bind "${WN2_ROOT}/training-2017-2020:/data:ro" --bind "${WN2_RUN}:/run" \
  /usr/project/xtmp/sc829/weather-track/images-final/environment.sif \
  /opt/weather-python/bin/python /run/code/train_stage.py \
  --data /data --statistics /data/statistics --protocol /run/code/training-protocol.json \
  --output "/run/${WN2_PHASE}" --profile --stage-index "${WEATHER_STAGE_INDEX:-0}" "${WN2_DEVICE_ARGS[@]}" "${WN2_EXTRA[@]}" \
  > "${WN2_RUN}/${WN2_PHASE}.log" 2>&1
if [[ -d ${WN2_RUN}/xla-dump ]]; then
  mv "${WN2_RUN}/xla-dump" "${WN2_RUN}/${WN2_PHASE}/xla-dump"
fi
done
if [[ ${WEATHER_REPLAY_DIAGNOSTIC:-0} == 1 ]]; then
"${WN2_ROOT}/eval-py312/bin/python" - "${WN2_RUN}" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np
r=Path(sys.argv[1]); differences=[]
with np.load(r/'initial/state.npz',allow_pickle=False) as a,np.load(r/'resumed/state.npz',allow_pickle=False) as b:
    ma=json.loads(str(a['metadata']));mb=json.loads(str(b['metadata']))
    for key in ma.keys() | mb.keys():
        if key!='elapsed_stage_seconds':
            assert ma.get(key)==mb.get(key),key
    assert set(a.files)==set(b.files)
    for key in a.files:
        if key=='metadata':continue
        x=a[key];y=b[key]
        assert x.shape==y.shape and x.dtype==y.dtype
        if x.tobytes()!=y.tobytes():
            differences.append(dict(leaf=key,max_abs=float(np.max(np.abs(x-y))),changed=int(np.count_nonzero(x!=y))))
report=dict(status='failed' if differences else 'passed',changed_leaves=len(differences),largest=sorted(differences,key=lambda x:x['max_abs'],reverse=True)[:20],stage_index=ma['stage_index'],devices=ma['devices'],effective_batch=ma['effective_batch'],runtime=ma['runtime'],scope='Native train_stage fresh-process continuation; verifies only the recorded stage, device count and batch')
if (r/'cached/state.npz').exists():
    import re
    with np.load(r/'resumed/state.npz',allow_pickle=False) as a,np.load(r/'cached/state.npz',allow_pickle=False) as b:
        assert set(a.files)==set(b.files)
        for key in a.files:
            if key=='metadata':
                ma=json.loads(str(a[key]));mb=json.loads(str(b[key]))
                ma.pop('elapsed_stage_seconds');mb.pop('elapsed_stage_seconds')
                assert ma==mb
            else:
                assert a[key].shape==b[key].shape and a[key].dtype==b[key].dtype
                assert a[key].tobytes()==b[key].tobytes(),key
    assert re.search(r"Persistent compilation cache hit for 'pmap_fn_passed_to_pmap'",(r/'cached.log').read_text(),re.IGNORECASE),'Third process did not hit gradient cache'
    report['third_process_cache_hit_and_state_exact']=True
(r/'exact-replay.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
sys.exit(bool(differences))
PY
fi
