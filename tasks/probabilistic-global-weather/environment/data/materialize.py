#!/usr/bin/env python3
"""Operator-only online WN2 preparation. No GPU or Slurm allocation required."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

TASK = Path(__file__).resolve().parents[2]
CODE = TASK / 'environment' / 'weathernext' / 'wn2'
sys.path.insert(0, str(CODE))
from source_metadata import source_record, validate_source
LOCK = Path(__file__).with_name('sources.lock.json')
SPLITS = {
    'development': ('development-protocol.json', 'development-precipitation-protocol.json',
                    'development-inputs-2021', 'development-weather-2021', 'development-precipitation-2021'),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_files(root, files):
    require(bool(files), f'Empty file inventory: {root}')
    for name, expected in files.items():
        path = (root / name).resolve()
        require(path.is_relative_to(root.resolve()), f'Unsafe inventory path: {name}')
        require(digest(path) == expected, f'Corrupt file: {path}')


def check_sources(record, lock):
    require(set(record) == set(lock['sources']), 'Weather source set changed')
    for name, spec in lock['sources'].items():
        validate_source(record[name], spec)


def check_weather(directory, protocol_path, index, lock):
    p = read(protocol_path)
    m = read(directory / 'weather.json')
    require(m['status'] == 'passed' and m['case_index'] == index, 'Incomplete training case')
    require(m['protocol_sha256'] in {digest(protocol_path), p.get('acquisition_protocol_sha256')}, 'Training protocol changed')
    expected = {(datetime.fromisoformat(p['initializations'][index]) + timedelta(hours=h)).isoformat()
                for h in p['offset_hours']}
    require({x.split('.')[0] for x in m['times']} == expected, 'Training timestamps changed')
    names = {f'{source}-{i:02d}.nc' for source in lock['sources'] for i in range(len(p['offset_hours']))}
    require(set(m['files']) == names, 'Incomplete training file inventory')
    check_sources(m['sources'], lock)
    checked_files(directory, m['files'])


def check_labels(directory, protocol, index, resolution, lock):
    m = read(directory / 'manifest.json')
    require(m['status'] == 'passed' and m['resolution'] == float(resolution), 'Incomplete cyclone labels')
    require(m['ibtracs_sha256'] == lock['ibtracs']['sha256'], 'IBTrACS source changed')
    require(m['source_commit'] == '89c4b2a77a1c57b328b909c575550fd2e5aadc9c', 'Cyclone code revision changed')
    require(m['calibration_end'] == protocol['calibration_end'] and m['calibration_start'] == '1980-01-01', 'Cyclone calibration changed')
    expected = {(datetime.fromisoformat(protocol['initializations'][index]) + timedelta(hours=h)).isoformat()
                for h in protocol['offset_hours'] if h > 0}
    require(len(m['frames']) == len(expected) and {f['time'] for f in m['frames']} == expected, 'Incomplete cyclone timestamps')
    checked_files(directory, {f['file']: f['sha256'] for f in m['frames']})


def check_inputs(directory, protocol_path, index, lock):
    m = read(directory / f'inputs-{index:03d}.json')
    require(m['status'] == 'passed' and m['case_index'] == index and m['protocol_sha256'] == digest(protocol_path), 'Input protocol changed')
    require(m['file'] == f'inputs-{index:03d}.nc', 'Wrong input filename')
    for key, spec in lock['sources'].items():
        validate_source(m['sources'][key], spec)
    checked_files(directory, {m['file']: m['sha256']})


def check_truth(directory, protocol_path, index, lock):
    require(digest(directory / 'protocol.json') == digest(protocol_path), 'Truth protocol changed')
    p, m = read(protocol_path), read(directory / 'cases' / f'{index:03d}.json')
    require(m['status'] == 'passed' and m['case_index'] == index and m['protocol_sha256'] == digest(protocol_path), 'Incomplete truth case')
    source = lock['sources']['hres_fc0' if p['truth_source'] == 'HRES-fc0' else 'era5']
    require(m['source'] == source['url'] and m['metadata_sha256'] == source['metadata_sha256'], 'Truth source changed')
    expected = set()
    for name, spec in p['variables'].items():
        for lead in range(len(p['lead_hours'])):
            if spec['pressure_levels']:
                expected.update(f'{name}/{index}.{lead}.{level}.0.0' for level in range(len(p['pressure_levels_hpa'])))
            else:
                expected.add(f'{name}/{index}.{lead}.0.0')
    require(set(m['chunks']) == expected, 'Incomplete truth chunk inventory')
    checked_files(directory / 'truth.zarr', m['chunks'])


def check_truth_layout(directory, protocol_path):
    p = read(protocol_path)
    zarr = directory / 'truth.zarr'
    metadata = read(zarr / '.zmetadata')['metadata']
    for name, spec in p['variables'].items():
        dims = ['case', 'lead'] + (['level'] if spec['pressure_levels'] else []) + ['lat', 'lon']
        shape = [len(p['evaluation_initializations']), len(p['lead_hours'])]
        shape += ([len(p['pressure_levels_hpa'])] if spec['pressure_levels'] else []) + p['grid']
        array, attrs = read(zarr / name / '.zarray'), read(zarr / name / '.zattrs')
        require(array['shape'] == shape and array['chunks'] == [1] * (len(shape) - 2) + p['grid']
                and array['dtype'] == '<f4', f'Wrong truth array layout: {name}')
        require(attrs['_ARRAY_DIMENSIONS'] == dims and attrs['units'] == spec['units'], f'Wrong truth dimensions/units: {name}')
        require(metadata[f'{name}/.zarray'] == array and metadata[f'{name}/.zattrs'] == attrs,
                f'Inconsistent consolidated truth metadata: {name}')


def check_stats_case(root, protocol_path, index):
    directory = root / f'case-{index:03d}'
    m = read(root / 'statistics' / f'case-{index:03d}.json')
    require(m['status'] == 'passed' and m['protocol_sha256'] == digest(protocol_path), 'Statistics protocol changed')
    e = m['evidence']
    require(e['case_index'] == index and e['weather_manifest_sha256'] == digest(directory / 'weather.json'), 'Statistics weather evidence changed')
    require(e['cyclone_manifests'] == {s: digest(directory / s / 'manifest.json') for s in ('cyclones-1', 'cyclones-.25')}, 'Statistics cyclone evidence changed')
    require(set(m['groups']) == {'era5_1deg_12h', 'era5_1deg_6h', 'era5_025deg_6h', 'hres_025deg_6h'}, 'Missing statistics groups')


def check_stats_final(root, protocol_path):
    m = read(root / 'manifest.json')
    require(m['status'] == 'passed' and m['protocol_sha256'] == digest(protocol_path), 'Incomplete statistics manifest')
    expected = {f'case-{i:03d}.json' for i in range(len(read(protocol_path)['initializations']))}
    require(set(m['case_manifests']) == expected, 'Missing statistics cases')
    checked_files(root, m['case_manifests'])
    expected_files = {f'{g}/{n}.nc' for g in ('era5_1deg_12h', 'era5_1deg_6h', 'era5_025deg_6h', 'hres_025deg_6h') for n in ('mean', 'std', 'minimum', 'diff_std')}
    require(set(m['files']) == expected_files, 'Missing normalizers')
    checked_files(root, m['files'])


class Runner:
    def __init__(self, args):
        self.args = args
        self.root = args.target.resolve()
        self.lock = read(LOCK)
        self.state = self.root / '.materialize'
        self.env = dict(os.environ, JAX_PLATFORMS='cpu', OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2',
                        PYTHONPATH=args.wbx_source, PYTHONUNBUFFERED='1')

    def command(self, model, script, *arguments):
        python = self.args.model_python if model else self.args.eval_python
        command = [python, str(TASK / script), *map(str, arguments)]
        if self.args.image:
            command = ['apptainer', 'exec', '--containall', '--cleanenv',
                       '--bind', f'{TASK}:{TASK}:ro', '--bind', f'{self.root}:{self.root}',
                       '--env', 'JAX_PLATFORMS=cpu,OPENBLAS_NUM_THREADS=2,OMP_NUM_THREADS=2,PYTHONUNBUFFERED=1',
                       '--env', f'PYTHONPATH={self.args.wbx_source}', str(self.args.image.resolve()), *command]
        return command

    def run(self, name, command, reset=None):
        log = self.state / 'logs' / f'{name}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(self.args.retries + 1):
            if reset is not None and reset.exists():
                shutil.rmtree(reset)
            with log.open('a') as stream:
                stream.write(f'\nAttempt {attempt + 1}: {json.dumps(command)}\n')
                stream.flush()
                try:
                    with subprocess.Popen(command, env=self.env, stdout=stream, stderr=subprocess.STDOUT,
                                          start_new_session=True) as process:
                        try:
                            code = process.wait(timeout=self.args.job_timeout)
                        except BaseException:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                            raise
                        if code:
                            raise subprocess.CalledProcessError(code, command)
                    return
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    if attempt == self.args.retries:
                        raise RuntimeError(f'{name} failed; see {log}') from None
            time.sleep(min(2 ** attempt * 5, 30))

    def job(self, name, marker, check, command):
        # Never overwrite an existing completed record that fails validation.
        # An interrupted case without its completion record can be regenerated.
        if marker.exists():
            check()
            return name, 'verified_existing'
        require(not self.args.verify_only, f'Missing completed resource: {marker}')
        self.run(name, command)
        check()
        return name, 'prepared'

    def pool(self, jobs, workers):
        failures = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fn) for fn in jobs]
            for future in as_completed(futures):
                try:
                    name, status = future.result()
                    print(json.dumps(dict(job=name, status=status)), flush=True)
                except Exception as exc:
                    failures.append(str(exc))
        require(not failures, '\n'.join(failures))

    def sources(self):
        for name, spec in self.lock['sources'].items():
            url = spec['url'].replace('gs://', 'https://storage.googleapis.com/') + '/.zmetadata'
            with urllib.request.urlopen(url, timeout=60) as response:
                record = source_record(spec['url'], response.read())
            validate_source(record, spec)
        if self.args.scope == 'evaluation':
            return
        path = self.root / 'IBTrACS.since1980.v04r01.nc'
        if not path.exists():
            if self.args.ibtracs:
                shutil.copyfile(self.args.ibtracs, path.with_suffix('.part'))
            else:
                with urllib.request.urlopen(self.args.ibtracs_url or self.lock['ibtracs']['url'], timeout=60) as src, path.with_suffix('.part').open('wb') as dst:
                    shutil.copyfileobj(src, dst)
            require(digest(path.with_suffix('.part')) == self.lock['ibtracs']['sha256'],
                    'Upstream IBTrACS differs from the frozen corpus; provide --ibtracs with the matching snapshot')
            path.with_suffix('.part').replace(path)
        require(digest(path) == self.lock['ibtracs']['sha256'], 'IBTrACS checksum mismatch')

    def execute(self):
        training = self.args.scope in ('all', 'development')
        splits = ['development'] if training else []
        if not self.args.verify_only:
            # Fail before downloading if the declared CPU runtimes are unavailable.
            self.run('eval-runtime', self.command(False, 'environment/data/runtime_check.py', 'eval'))
            if training:
                self.run('model-runtime', self.command(True, 'environment/data/runtime_check.py', 'model'))
            for attempt in range(self.args.retries + 1):
                try:
                    self.sources()
                    break
                except (OSError, TimeoutError):
                    if attempt == self.args.retries:
                        raise
                    time.sleep(min(2 ** attempt * 5, 30))
        jobs = []
        protocol_path = CODE / 'training-protocol.json'
        p = read(protocol_path)
        train = self.root / 'training-2017-2020'
        if training:
            for i in range(len(p['initializations'])):
                d = train / f'case-{i:03d}'
                jobs.append(lambda i=i, d=d: self.job(f'training-{i:03d}', d / 'weather.json',
                    lambda: check_weather(d, protocol_path, i, self.lock),
                    self.command(False, 'wn2/materialize_training.py', '--protocol', protocol_path, '--output', train, '--case-index', i)))
        truth_stores = []
        for split in splits:
            weather, precip, inputs, weather_dir, precip_dir = SPLITS[split]
            for i in range(len(read(CODE / weather)['evaluation_initializations'])):
                d, pp = self.root / inputs, CODE / weather
                jobs.append(lambda i=i, d=d, pp=pp, split=split: self.job(f'{split}-inputs-{i:03d}', d / f'inputs-{i:03d}.json',
                    lambda: check_inputs(d, pp, i, self.lock),
                    self.command(False, 'wn2/materialize_inputs.py', '--protocol', pp, '--output', d, '--case-index', i)))
            for filename, directory in ((weather, weather_dir), (precip, precip_dir)):
                pp, d = CODE / filename, self.root / directory
                if not d.exists():
                    require(not self.args.verify_only, f'Missing truth store: {d}')
                    # Initialize in a temporary directory, then atomically publish.
                    temp = self.state / f'initialize-{directory}'
                    if temp.exists():
                        shutil.rmtree(temp)
                    self.run(f'initialize-{directory}', self.command(False, 'wn2/materialize_truth.py', '--initialize', '--protocol', pp, '--output', temp), reset=temp)
                    temp.replace(d)
                require(digest(d / 'protocol.json') == digest(pp), 'Existing truth protocol changed')
                check_truth_layout(d, pp)
                truth_stores.append((d, pp))
                for i in range(len(read(pp)['evaluation_initializations'])):
                    jobs.append(lambda i=i, d=d, pp=pp: self.job(f'{d.name}-{i:03d}', d / 'cases' / f'{i:03d}.json',
                        lambda: check_truth(d, pp, i, self.lock),
                        self.command(False, 'wn2/materialize_truth.py', '--output', d, '--case-index', i)))
        self.pool(jobs, self.args.workers)
        if training:
            jobs = []
            for i, stamp in enumerate(p['initializations']):
                times = [(datetime.fromisoformat(stamp) + timedelta(hours=h)).isoformat() for h in p['offset_hours'] if h > 0]
                for resolution in ('1', '.25'):
                    d = train / f'case-{i:03d}' / f'cyclones-{resolution}'
                    jobs.append(lambda i=i, d=d, resolution=resolution, times=times: self.job(f'labels-{i:03d}-{resolution}', d / 'manifest.json',
                        lambda: check_labels(d, p, i, resolution, self.lock),
                        self.command(True, 'slurm/prepare_wn2_cyclones.py', '--ibtracs', self.root / 'IBTrACS.since1980.v04r01.nc',
                                     '--output', d, '--resolution', resolution, '--training-end', p['calibration_end'], '--times', *times)))
            self.pool(jobs, self.args.compute_workers)
            jobs = []
            for i in range(len(p['initializations'])):
                jobs.append(lambda i=i: self.job(f'statistics-{i:03d}', train / 'statistics' / f'case-{i:03d}.json',
                    lambda: check_stats_case(train, protocol_path, i),
                    self.command(True, 'wn2/training_statistics.py', '--root', train, '--protocol', protocol_path,
                                 '--output', train / 'statistics', '--case-index', i)))
            self.pool(jobs, self.args.compute_workers)
            self.job('statistics-finalize', train / 'statistics' / 'manifest.json',
                     lambda: check_stats_final(train / 'statistics', protocol_path),
                     self.command(True, 'wn2/training_statistics.py', '--root', train, '--protocol', protocol_path,
                                  '--output', train / 'statistics', '--finalize'))
        for d, pp in truth_stores:
            status = read(d / 'status.json')
            if status['status'] != 'passed':
                require(not self.args.verify_only, f'Truth not finalized: {d}')
                self.run(f'finalize-{d.name}', self.command(False, 'wn2/materialize_truth.py', '--output', d, '--finalize'))
                status = read(d / 'status.json')
            require(status['status'] == 'passed' and status['protocol_sha256'] == digest(pp), 'Invalid truth finalization')
            expected = {f'cases/{i:03d}.json' for i in range(len(read(pp)['evaluation_initializations']))}
            records = {r['file']: r['sha256'] for r in status['case_manifests']}
            require(set(records) == expected, 'Incomplete final truth inventory')
            checked_files(d, records)
        if self.args.scope in ('all', 'evaluation'):
            check_stats_final(train / 'statistics', protocol_path)
            arguments = ['--root', self.root, '--eval-python', self.args.eval_python,
                         '--model-python', self.args.model_python, '--workers', self.args.workers]
            if self.args.verify_only: arguments.append('--verify-only')
            self.run('unified-evaluation-resources', self.command(False,
                     'environment/data/prepare_evaluation.py', *arguments))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target', type=Path)
    parser.add_argument('--scope', choices=['all', 'development', 'evaluation'], default='all')
    parser.add_argument('--workers', type=int, default=4, help='Concurrent network case workers (budget about 16 GiB RAM each)')
    parser.add_argument('--compute-workers', type=int, default=1, help='CPU label/statistics workers (budget about 40 GiB RAM each)')
    parser.add_argument('--retries', type=int, default=2, help='Retries after initial failed attempt')
    parser.add_argument('--job-timeout', type=int, default=10800, help='Seconds per subprocess attempt')
    parser.add_argument('--image', type=Path, help='Pinned WN2 Apptainer image; runs CPU-only with online networking')
    parser.add_argument('--eval-python', default=os.environ.get('WEATHER_EVAL_PYTHON', '/opt/wn2-eval/bin/python'))
    parser.add_argument('--model-python', default=os.environ.get('WEATHER_MODEL_PYTHON', '/opt/weather-python/bin/python'))
    parser.add_argument('--wbx-source', default=os.environ.get('WEATHER_WBX_SOURCE', '/opt/weatherbenchX'))
    parser.add_argument('--ibtracs', type=Path, help='Local copy matching the pinned NOAA checksum')
    parser.add_argument('--ibtracs-url', help='Snapshot mirror URL; the same frozen SHA-256 is required')
    parser.add_argument('--verify-only', action='store_true', help='Offline: rehash all selected data without running preparation')
    args = parser.parse_args(argv)
    require(args.workers > 0 and args.compute_workers > 0 and args.retries >= 0 and args.job_timeout > 0, 'Invalid worker/retry/timeout count')
    require(CODE.is_dir(), 'Run this operator helper from the repository checkout, outside agent containers')
    args.target.mkdir(parents=True, exist_ok=True)
    runner = Runner(args)
    runner.state.mkdir(exist_ok=True)
    with (runner.state / 'lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another materializer is using this target') from None
        status_path = runner.state / f'{args.scope}-status.json'
        status = dict(status='running', scope=args.scope, verify_only=args.verify_only,
                      started_unix=time.time(), sources_lock_sha256=digest(LOCK),
                      code_sha256={str(path.relative_to(TASK)): digest(path) for path in
                          [Path(__file__), Path(__file__).with_name('runtime_check.py'),
                           TASK / 'slurm/prepare_wn2_cyclones.py', CODE / 'materialize_inputs.py',
                           CODE / 'materialize_training.py', CODE / 'materialize_truth.py',
                           CODE / 'training_statistics.py', CODE / 'training_data.py']},
                      protocols={name: digest(CODE / name) for name in ('training-protocol.json', 'protocol.json',
                          'precipitation-protocol.json', 'development-protocol.json', 'development-precipitation-protocol.json')})
        write(status_path, status)
        try:
            runner.execute()
        except BaseException as exc:
            status.update(status='failed', error=str(exc), ended_unix=time.time())
            write(status_path, status)
            raise
        status.update(status='passed', ended_unix=time.time())
        write(status_path, status)
        print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
