"""Prepare the single eight-date evaluation resource bundle on CPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

TASK = Path(__file__).resolve().parents[2]
PAPER = TASK / 'paper-comparison'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(root):
    manifest = json.loads((root / 'resources.json').read_text())
    if manifest.get('status') != 'passed' or not manifest.get('files'):
        raise ValueError('Incomplete evaluation resource inventory')
    if manifest['weather_protocol_sha256'] != digest(PAPER / 'weather-protocol.json'):
        raise ValueError('Evaluation resource protocol changed')
    for name, expected in manifest['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != expected:
            raise ValueError(f'Evaluation resource changed: {name}')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--eval-python', default=sys.executable)
    parser.add_argument('--model-python', default='/opt/weather-python/bin/python')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    root = args.root / 'evaluation-resources'
    if args.verify_only or (root / 'resources.json').exists():
        verify(root)
        print(json.dumps(dict(status='passed', mode='verified')))
        return
    root.mkdir(parents=True, exist_ok=True)
    logs = args.root / '.materialize/evaluation-logs'
    logs.mkdir(parents=True, exist_ok=True)

    def run(script, *parameters, model=False, label=None):
        command = [args.model_python if model else args.eval_python,
                   str(PAPER / script), *map(str, parameters)]
        name = label or Path(script).stem
        print(json.dumps(dict(step=name, status='running')), flush=True)
        with (logs / f'{name}.log').open('a') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                           env=os.environ | {'JAX_PLATFORMS': 'cpu', 'OPENBLAS_NUM_THREADS': '2'})

    run('prepare.py', '--output', root / 'data', '--workers', args.workers)
    run('prepare_rev.py', '--output', root / 'rev-climatology', '--workers', args.workers)
    # The historical precipitation cache is preparation scratch, never mounted in the verifier.
    rain = args.root / '.materialize/seeps-rain'
    for year in range(1980, 2016):
        run('stage_seeps_rain.py', '--output', rain, '--year', year,
            '--workers', args.workers, label=f'rain-{year}')
    for month in range(1, 13):
        run('prepare_seeps.py', '--rain', rain, '--output', root / 'seeps/climatology',
            '--month', month, '--workers', args.workers, label=f'seeps-{month}')
    run('spherical_pool.py', '--output', root / 'pooling', model=True)
    run('prepare_tracker.py', '--data', root / 'data', '--output', root / 'tracker/truth', '--workers', args.workers)
    run('prepare_named_cyclones.py', '--ibtracs', args.root / 'IBTrACS.since1980.v04r01.nc',
        '--output', root / 'tracker/named-truth.json')
    for case in range(8):
        run('prepare_truth_spectra.py', '--data', root / 'data', '--case', case,
            '--output', root / 'spectra/truth' / f'case-{case:03d}.json', model=True,
            label=f'spectra-{case}')
    shutil.copyfile(args.root / 'training-2017-2020/statistics/hres_025deg_6h/std.nc',
                    root / 'training-scales.nc')
    manifest = dict(status='passed', weather_protocol_sha256=digest(PAPER / 'weather-protocol.json'),
                    files={str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*')) if p.is_file()})
    temporary = root / 'resources.tmp'
    temporary.write_text(json.dumps(manifest, indent=2) + '\n')
    temporary.replace(root / 'resources.json')
    print(json.dumps(dict(status='passed', mode='prepared')))


if __name__ == '__main__':
    main()
