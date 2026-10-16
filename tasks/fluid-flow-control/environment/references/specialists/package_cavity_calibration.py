"""Freeze measured baseline inputs into a portable data bundle, not paper anchors."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from netCDF4 import Dataset

from cavity_calibrate import calibrate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurement', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--restart', type=Path, required=True)
    parser.add_argument('--restart-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Never overwrite an existing reference-data bundle')
    target = json.loads(args.calibration.read_text())
    restart = json.loads(args.restart_manifest.read_text())
    if sha(args.restart) != restart['restart_sha256']:
        raise ValueError('Restart changed since native header audit')
    with Dataset(args.restart) as state:
        if (float(state.ReynoldsNumber) != restart['reynolds']
                or float(state.MachNumber) != restart['mach']
                or int(state.globalTimeStep) != restart['global_time_step']):
            raise ValueError('Native restart header does not match its manifest')
    if target['binding']['reynolds'] != restart['reynolds']:
        raise ValueError('Calibration and restart Reynolds differ')
    with tempfile.TemporaryDirectory(prefix='cavity-data-validation-') as temporary:
        reproduced = calibrate(args.measurement, Path(temporary) / 'calibration.json',
                               target['measurement']['organizer_mean_drift_limit'])
    if reproduced != target:
        raise ValueError('Frozen calibration does not reproduce from its measurements')
    inputs = {'calibration.json': args.calibration, 'restart-manifest.json': args.restart_manifest,
              'restart.Netcdf': args.restart}
    for name in ('protocol.json', 'diagnostic.json', 'raw-measurements.npz', 'source.py',
                 'cavity_author.py', 'cavity_reward.py'):
        inputs['measurement/' + name] = args.measurement / name
    expected = {name: sha(source) for name, source in inputs.items()}
    args.output.mkdir(parents=True, exist_ok=False)
    for name, source in inputs.items():
        destination = args.output / name
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, destination)
        if sha(destination) != expected[name] or sha(source) != expected[name]:
            raise ValueError('Reference input changed during copying')
    record = dict(status='portable-reconstructed-calibration-not-paper-certified',
        environment='Cavity_2D_Re4200', benchmark_validated=False,
        files_sha256=expected, generator_sha256=sha(__file__),
        binding=target['binding'], measurement=target['measurement'],
        restart={k: restart[k] for k in ('restart_sha256', 'reynolds', 'mach', 'global_time_step')},
        limitations=['Not original paper targets or initial condition',
                    'Historical absolute source paths are provenance, not runtime dependencies',
                    'Parent mesh and derived configuration remain in the pinned data root'])
    (args.output / 'manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
