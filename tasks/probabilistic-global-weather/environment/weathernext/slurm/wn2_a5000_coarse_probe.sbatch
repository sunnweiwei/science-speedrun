#!/usr/bin/env bash
# Diagnostic only, never a production profile. Defaults to coarse A5000.
# Optional stage/GPU/mask/cache overrides support bounded native experiments;
# all modified Python sources are isolated in the run directory.
#SBATCH --job-name=wn2-a5000-coarse
#SBATCH --partition=compsci-gpu
#SBATCH --account=users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=00:30:00
set -euo pipefail
TASK_ROOT=${SLURM_SUBMIT_DIR}/probabilistic-global-weather
WN2_ROOT=/usr/project/xtmp/sc829/weather-track/wn2-pilot
WN2_RUN=${WN2_ROOT}/a5000-coarse-runs/${SLURM_JOB_ID}
WN2_DEVICE_ARGS=()
case ${WEATHER_PROBE_DEVICES:-1} in
  1) WN2_DEVICE_ARGS=(--debug-single-device --debug-updates 2) ;;
  4) ;; # Canonical four-device profile executes two batch-64 updates.
  *) echo 'WEATHER_PROBE_DEVICES must be 1 or 4' >&2; exit 2 ;;
esac
mkdir -p "${WN2_RUN}/code"
sha256sum /usr/project/xtmp/sc829/weather-track/images-final/environment.sif > "${WN2_RUN}/image.sha256"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "${WN2_RUN}/hardware.csv"
cp "${TASK_ROOT}"/wn2/*.py "${TASK_ROOT}/wn2/training-protocol.json" \
  "${TASK_ROOT}/wn2/protocol.json" "${TASK_ROOT}/wn2/precipitation-protocol.json" "${WN2_RUN}/code/"
cp "${TASK_ROOT}/slurm/wn2_validation.py" "${TASK_ROOT}/slurm/wn2_distributed.py" "${WN2_RUN}/code/"
# Production training loads a measured profile's autotuning results from its
# first update. Optionally test that same policy for all replay processes.
if [[ -n ${WEATHER_INITIAL_PROFILE_RUN:-} ]]; then
  python3 - "${WEATHER_INITIAL_PROFILE_RUN}" "${WN2_RUN}" "${WEATHER_STAGE_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}" "${WEATHER_PROBE_DEVICES:-1}" <<'PY_PROFILE'
import hashlib,json,shutil,sys
from pathlib import Path
profile,run=map(Path,sys.argv[1:3]);sys.path.insert(0,str(run/'code'))
from plan_schedule import STAGES
report=json.loads((profile/'result/report.json').read_text())
assert report['status']=='passed' and report['profile']
assert report['stage']==STAGES[int(sys.argv[3])]
assert report['devices']==int(sys.argv[4])
for name,digest in report['source_sha256'].items():
    source=run/'code'/name
    assert source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest()==digest,name
assert (profile/'autotune.pb').stat().st_size>0
shutil.copyfile(profile/'autotune.pb',run/'autotune.pb')
(run/'initial-profile-provenance.json').write_text(json.dumps({'profile_run':str(profile),'report_sha256':hashlib.sha256((profile/'result/report.json').read_bytes()).hexdigest(),'policy':'All phases load complete profile autotuning, matching production train.py'},indent=2)+'\n')
PY_PROFILE
fi
if [[ ${WEATHER_DENSE_REMAT:-1} == 0 ]]; then
  python3 - "${WN2_RUN}/code/wn2_validation.py" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]);s=p.read_text();marker="child['remat'] = True"
assert s.count(marker)==1
p.write_text(s.replace(marker,"child['remat'] = False"))
PY
fi
python3 - "${WN2_RUN}/code/train_stage.py" "${WEATHER_REPLAY_DIAGNOSTIC:-0}" "${WEATHER_PROBE_GPU_KIND:-A5000}" "${WEATHER_GRADIENT_AUDIT:-0}" <<'PY'
import sys
from pathlib import Path
p=Path(sys.argv[1]); text=p.read_text()
old="if any('RTX PRO 6000' not in name for name in runtime['device_kinds']):"
assert sys.argv[3] in ('A5000','A6000','A100','RTX PRO 6000')
new="if any("+repr(sys.argv[3])+" not in name for name in runtime['device_kinds']):"
assert text.count(old)==1
text=text.replace(old,new)
if sys.argv[2]=='1':
    # Preserve update one for a fresh-process continuation comparison.
    marker="        record['seconds']=time.monotonic()-start"
    assert text.count(marker)==1
    text=text.replace(marker,"        if global_updates == 1:\n            import shutil\n            shutil.copyfile(args.output/'state.npz',args.output/'state-1.npz')\n"+marker)
    marker='if len(durations)!=updates:'
    assert text.count(marker)==1
    text=text.replace(marker,'if len(durations)!=updates-start_update:')
if sys.argv[4]=='1':
    marker='        optimizer_start=time.monotonic()'
    assert text.count(marker)==1
    audit="""        import hashlib
        gradient_audit=[]
        for leaf_index, leaf in enumerate(jax.tree.leaves(total)):
            replicas=[]
            for shard in leaf.addressable_shards:
                host=np.asarray(shard.data)
                replicas.append(dict(device=str(shard.device), sha256=hashlib.sha256(host.tobytes()).hexdigest(), shape=list(host.shape), dtype=str(host.dtype)))
            gradient_audit.append(dict(leaf=leaf_index, replicas=replicas))
        (args.output/f'gradient-before-mean-{index+1}.json').write_text(json.dumps(gradient_audit))
"""
    text=text.replace(marker,audit+marker)
p.write_text(text)
PY
if [[ ${WEATHER_BOOLEAN_MASK:-0} == 1 ]]; then
  cp "${TASK_ROOT}/slurm/wn2_boolean_masks.py" "${WN2_RUN}/code/"
  cat >> "${WN2_RUN}/code/native_model.py" <<'PY'

# Isolated storage-only mask experiment; canonical training source unchanged.
from wn2_boolean_masks import enable as enable_boolean_attention_masks
enable_boolean_attention_masks()
PY
fi
if [[ ${WEATHER_PACKED_MASK:-0} == 1 ]]; then
  cp "${TASK_ROOT}/slurm/wn2_packed_masks.py" "${WN2_RUN}/code/"
  cat >> "${WN2_RUN}/code/native_model.py" <<'PY'

# Isolated packed-mask experiment; no canonical model mutation.
from wn2_packed_masks import enable as enable_packed_attention_masks
enable_packed_attention_masks()
PY
fi
if [[ ${WEATHER_CUDNN_ATTENTION:-0} == 1 ]]; then
  cp "${TASK_ROOT}/slurm/wn2_cudnn_attention.py" "${WN2_RUN}/code/"
  cat >> "${WN2_RUN}/code/native_model.py" <<'PY'

# Isolated arithmetic backend experiment; requires independent acceptance.
from weathernext.utils import sparse_transformer as cudnn_native_transformer
from wn2_cudnn_attention import attention as experimental_cudnn_attention
cudnn_native_transformer.triblockdiag_mha = experimental_cudnn_attention
PY
  if [[ ${WEATHER_CUDNN_CACHE_FIX:-0} == 1 ]]; then
    cat >> "${WN2_RUN}/code/native_model.py" <<'PY'
from wn2_cudnn_attention import enable_deterministic_cache_fix
enable_deterministic_cache_fix()
PY
  fi
fi
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv --loop=5 > "${WN2_RUN}/gpu-memory.csv" &
WN2_MONITOR=$!
trap 'kill "${WN2_MONITOR}" 2>/dev/null || true' EXIT
WN2_PHASES=(result)
if [[ ${WEATHER_REPLAY_DIAGNOSTIC:-0} == 1 ]]; then WN2_PHASES=(initial resumed); fi
if [[ ${WEATHER_VERIFY_CACHE_REUSE:-0} == 1 ]]; then
  [[ ${WEATHER_REPLAY_DIAGNOSTIC:-0} == 1 && ${WEATHER_PROBE_CACHE:-false} == true ]] || exit 2
  WN2_PHASES=(initial resumed cached)
fi
for WN2_PHASE in "${WN2_PHASES[@]}"; do
WN2_EXTRA=()
WN2_TUNING='--xla_gpu_dump_autotune_results_to=/run/autotune.pb'
if [[ -n ${WEATHER_INITIAL_PROFILE_RUN:-} ]]; then
  WN2_TUNING='--xla_gpu_load_autotune_results_from=/run/autotune.pb --xla_gpu_require_complete_aot_autotune_results=true'
fi
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
  --output "/run/${WN2_PHASE}" --profile --stage-index "${WEATHER_STAGE_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}" "${WN2_DEVICE_ARGS[@]}" "${WN2_EXTRA[@]}" \
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
if [[ ${WEATHER_PACKAGE_DIAGNOSTIC:-0} == 1 ]]; then
  cp "${TASK_ROOT}/wn2/LICENSE.weathernext-code" "${WN2_RUN}/code/"
  "${WN2_ROOT}/eval-py312/bin/python" "${WN2_RUN}/code/package_model.py" \
    --stage "${WN2_RUN}/${WN2_PHASES[0]}" --statistics "${WN2_ROOT}/training-2017-2020/statistics" \
    --output "${WN2_RUN}/package" --allow-profile > "${WN2_RUN}/package.log" 2>&1
fi
