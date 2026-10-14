"""Independent CPU score tasks sharing one completed submission replay."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'benchmark', 'output', 'package'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--endpoint', choices=['core', 'pooling', 'spectra', 'track', 'cyclones', 'core-summary', 'finalize'], required=True)
    ap.add_argument('--index', type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, str(a.source / 'wn2'))
    from training_data import sha256
    from verify_track import primary_reward
    import xarray as xr
    protocol = a.source / 'wn2/protocol.json'
    p = json.loads(protocol.read_text())
    identity = sha256(protocol)
    forecast = json.loads((a.output / 'forecast-status.json').read_text())
    if (forecast.get('status') != 'passed' or forecast['protocol_sha256'] != identity
            or forecast['package_sha256'] != sha256(a.package / 'package.json')):
        raise ValueError('Missing or changed complete independent forecast')
    for i in range(len(p['evaluation_initializations'])):
        receipt = a.output / f'replay-{i:03d}/replay.json'
        if sha256(receipt) != forecast['replay_receipts'][str(i)]:
            raise ValueError('Changed independent replay receipt')
    paper = a.source / 'paper-comparison'
    predictions = a.output / 'predictions'
    env = os.environ | {'CUDA_VISIBLE_DEVICES': '', 'JAX_PLATFORMS': 'cpu', 'OPENBLAS_NUM_THREADS': '2'}
    def execute(script, *args, model=False):
        python = os.environ.get('WEATHER_MODEL_PYTHON', '/opt/weather-python/bin/python') if model else sys.executable
        subprocess.run([python, str(paper / script), *map(str, args)], env=env, check=True)
    if a.endpoint in ('core-summary', 'finalize'):
        endpoints = ['scores'] if a.endpoint == 'core-summary' else ['scores', 'pooling-scores', 'spectra', 'cyclone-scores']
        reports = []
        receipts = {}
        for endpoint in endpoints:
            for i in range(len(p['evaluation_initializations'])):
                path = a.output / endpoint / f'case-{i:03d}.json'
                report = json.loads(path.read_text())
                if report.get('status') != 'passed' or report.get('protocol_sha256') != identity:
                    raise ValueError('Incomplete endpoint suite')
                receipts[str(path.relative_to(a.output))] = sha256(path)
                if endpoint == 'scores':
                    reports.append(report)
        with xr.open_dataset(a.benchmark / 'training-scales.nc') as scales:
            metrics = primary_reward(reports, p, scales)
        if a.endpoint == 'core-summary':
            result = dict(status='core_scores_complete', cases=8, members=56, metrics=metrics,
                          scope='All eight native core scores; additional endpoints still required for final acceptance')
            (a.output / 'core-result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        else:
            (a.output / 'reward.json').write_text(json.dumps(metrics, indent=2, allow_nan=False) + '\n')
            result = dict(status='passed', protocol_sha256=identity, cases=8, members=56,
                          forecast_days=15, independent_model_replay=True, score_receipts_sha256=receipts,
                          training_provenance_verified=False,
                          scope='Unified independent inference and all metric endpoints; CPU families scheduled independently')
            (a.output / 'status.json').write_text(json.dumps(result, indent=2) + '\n')
        return
    index = a.index // p['members'] if a.endpoint == 'track' else a.index
    if not 0 <= index < len(p['evaluation_initializations']):
        raise ValueError('Case index outside frozen protocol')
    case = f'case-{index:03d}'
    if a.endpoint in ('core', 'pooling'):
        folder = a.output / ('scores' if a.endpoint == 'core' else 'pooling-scores')
        folder.mkdir(exist_ok=True)
        args = ['--predictions', predictions, '--data', a.benchmark / 'data', '--case', index, '--output', folder / f'{case}.json']
        if a.endpoint == 'core':
            args += ['--rev-climatology', a.benchmark / 'rev-climatology', '--seeps-climatology', a.benchmark / 'seeps/climatology']
        else:
            args += ['--pooling', a.benchmark / 'pooling', '--pooling-only']
        execute('score_marginals.py', *args)
    elif a.endpoint == 'spectra':
        (a.output / 'spectra').mkdir(exist_ok=True)
        execute('score_spectra.py', '--predictions', predictions, '--truth', a.benchmark / 'spectra/truth' / f'{case}.json',
                '--case', index, '--output', a.output / 'spectra' / f'{case}.json', model=True)
    elif a.endpoint == 'track':
        member = a.index % p['members']
        dest = a.output / 'tracks' / f'{case}-member-{member:03d}'
        execute('export_tracker_forecast.py', '--predictions', predictions, '--history', a.benchmark / 'tracker/truth',
                '--case', index, '--member', member, '--output', dest / 'input.nc')
        execute('run_tracker.py', '--bin', os.environ.get('WEATHER_TRACKER_BIN', '/opt/tempestextremes/bin'),
                '--input', dest / 'input.nc', '--initialization', p['evaluation_initializations'][index], '--output', dest / 'tracking')
    elif a.endpoint == 'cyclones':
        truth = a.output / 'tracker-truth'
        execute('run_tracker.py', '--bin', os.environ.get('WEATHER_TRACKER_BIN', '/opt/tempestextremes/bin'),
                '--input', a.benchmark / 'tracker/truth' / f'{case}.nc',
                '--initialization', p['evaluation_initializations'][index], '--output', truth / f'observed-{case}')
        (a.output / 'cyclone-scores').mkdir(exist_ok=True)
        execute('score_cyclones.py', '--predictions', predictions, '--tracks', a.output / 'tracks',
                '--truth-root', truth, '--case', index, '--output', a.output / 'cyclone-scores' / f'{case}.json')


if __name__ == '__main__':
    main()
