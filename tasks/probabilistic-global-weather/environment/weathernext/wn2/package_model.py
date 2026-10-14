#!/usr/bin/env python3
"""Assemble a portable WN2 inference artifact from a completed native stage."""
import argparse
import json
from pathlib import Path
import shutil

from training_data import sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',type=Path,required=True);parser.add_argument('--statistics',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--allow-profile',action='store_true')
    parser.add_argument('--compilation-cache',type=Path)
    parser.add_argument('--autotune-results',type=Path)
    args=parser.parse_args()
    report=json.loads((args.stage/'report.json').read_text())
    if report['status']!='passed':raise ValueError('Stage did not complete')
    if report.get('profile') and not args.allow_profile:raise ValueError('Execution profile is not a completed curriculum')
    if not report.get('profile') and report['stage']['name']!='hres_025deg_ar8':
        raise ValueError('The final native curriculum stage has not completed')
    if sha256(args.stage/'weights.npz')!=report['weights_sha256']:raise ValueError('Changed stage weights')
    if sha256(args.statistics/'manifest.json')!=report['statistics_manifest_sha256']:raise ValueError('Changed statistics')
    # The documented standalone and Compose commands bind-mount an empty host
    # directory at the package path, so accept an empty existing directory.
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Refusing to overwrite package')
    args.output.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).resolve().parent
    license_path=source/'LICENSE.weathernext-code'
    shutil.copyfile(license_path,args.output/'LICENSE.weathernext-code')
    for name in ('predict.py','inference_state.py','forecast_store.py','native_model.py','plan_schedule.py','training_data.py'):
        shutil.copyfile(source/name,args.output/name)
    helper=source/'wn2_validation.py'
    if not helper.exists():helper=source.parent/'slurm/wn2_validation.py'
    shutil.copyfile(helper,args.output/'wn2_validation.py')
    for name in ('native_model.py','training_data.py','wn2_validation.py'):
        expected=report.get('source_sha256',{}).get(name)
        if expected and sha256(args.output/name)!=expected:
            raise ValueError(f'Inference model implementation differs from training: {name}')
    shutil.copyfile(args.stage/'weights.npz',args.output/'weights.npz')
    shutil.copyfile(args.stage/'report.json',args.output/'training-report.json')
    stats=json.loads((args.statistics/'manifest.json').read_text())
    for name,digest in stats['files'].items():
        if sha256(args.statistics/name)!=digest:raise ValueError('Statistics integrity failure')
        path=args.output/'statistics'/name;path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(args.statistics/name,path)
    shutil.copyfile(args.statistics/'manifest.json',args.output/'statistics/manifest.json')
    if args.compilation_cache:
        if any(p.is_symlink() for p in args.compilation_cache.rglob('*')):
            raise ValueError('Compilation cache must contain only ordinary files')
        shutil.copytree(args.compilation_cache,args.output/'compilation-cache')
    if args.autotune_results:
        if args.autotune_results.is_symlink() or not args.autotune_results.is_file() or args.autotune_results.stat().st_size==0:
            raise ValueError('Inference autotuning results must be a nonempty ordinary file')
        shutil.copyfile(args.autotune_results,args.output/'inference-autotune.pb')
    variables=json.loads((source/'protocol.json').read_text())['variables']
    variables.update(json.loads((source/'precipitation-protocol.json').read_text())['variables'])
    launcher=args.output/'infer.sh'
    launcher.write_text('''#!/usr/bin/env bash
set -euo pipefail
WN2_PACKAGE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$2/.jax-cache"
if [[ -d "${WN2_PACKAGE}/compilation-cache" ]]; then
  cp -a "${WN2_PACKAGE}/compilation-cache/." "$2/.jax-cache/"
fi
export XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true
if [[ -f "${WN2_PACKAGE}/inference-autotune.pb" ]]; then
  export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_load_autotune_results_from=${WN2_PACKAGE}/inference-autotune.pb --xla_gpu_require_complete_aot_autotune_results=true"
else
  export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_dump_autotune_results_to=$2/inference-autotune.pb"
fi
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export JAX_COMPILATION_CACHE_DIR="$2/.jax-cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_RAISE_PERSISTENT_CACHE_ERRORS=true
exec "${WEATHER_MODEL_PYTHON:-/opt/weather-python/bin/python}" "${WN2_PACKAGE}/predict.py" --package "${WN2_PACKAGE}" --input "$1" --output "$2"
''');launcher.chmod(0o755)
    files={str(p.relative_to(args.output)):sha256(p) for p in args.output.rglob('*') if p.is_file()}
    manifest=dict(schema='wn2-starter-package/v1',inference_schemas=['wn2-inference/v2','wn2-inference/v3'],source_revision='89c4b2a77a1c57b328b909c575550fd2e5aadc9c',
                  parameter_count=183785575,profile_artifact=bool(report.get('profile')),
                  variables=variables,files=files,
                  compilation_cache='Optional immutable JAX cache copied to private writable output; compiler/runtime-specific, safe cache misses recompile; contains no forecast or training examples',
                  runtime='Pinned task WeatherNext Python 3.12/JAX 0.6.2 image; no external pretrained weights or training files needed')
    (args.output/'package.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
