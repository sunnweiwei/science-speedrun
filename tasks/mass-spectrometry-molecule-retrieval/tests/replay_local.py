#!/usr/bin/env python3
"""Operator-side input-only replay on a Linux host without a Docker daemon.

Uses a fresh rootless user/PID/network/mount namespace. This validates package
isolation and portability, but does not certify the production Docker image or
the H100 runtime limit. Private labels, training files and host home are absent.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from common import validate_input, write_json
from main_verifier import execute_batches, seal_output, validate_submission


def invoke_isolated(args, python, base, input_directory, current, remaining):
    output = current / 'untrusted-output'
    output.mkdir()
    # Only the chosen venv and its actual CPython installation are visible.
    # In particular, never bind /usr: a cluster may store private data below it.
    command = ['bwrap', '--unshare-all', '--die-with-parent', '--new-session',
               '--uid', '0', '--gid', '0', '--cap-drop', 'ALL', '--clearenv']
    for path in ['/usr/bin', '/usr/lib', '/usr/lib64', '/usr/share', '/bin', '/lib', '/lib64']:
        if Path(path).exists():
            command += ['--ro-bind', path, path]
    for path in ['/etc/ld.so.cache', '/etc/ld.so.conf', '/etc/ld.so.conf.d']:
        if Path(path).exists():
            command += ['--ro-bind', path, path]
    command += ['--ro-bind', str(base), str(base),
                '--ro-bind', str(args.runtime), str(args.runtime),
                '--ro-bind', str(args.submission), '/submission',
                '--ro-bind', str(input_directory), '/input',
                '--bind', str(output), '/output', '--tmpfs', '/tmp',
                '--dir', '/home/agent', '--proc', '/proc', '--dev', '/dev']
    if args.gpu:
        if Path('/sys').exists():
            command += ['--ro-bind', '/sys', '/sys']
        for path in sorted(Path('/dev').glob('nvidia*')):
            command += ['--dev-bind', str(path), str(path)]
    for key, value in {
        'PATH': f'{args.runtime}/bin:/usr/bin:/bin', 'HOME': '/home/agent',
        'MASSSPECGYM_GLACIER_PYTHON': str(python), 'OMP_NUM_THREADS': str(args.cpus),
        'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1',
        'PIP_NO_INDEX': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', '0') if args.gpu else '',
    }.items():
        command += ['--setenv', key, value]
    command += ['--chdir', '/submission', '/bin/bash', '/submission/infer.sh', '/input', '/output']
    with (current / 'inference.log').open('wb') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    if code:
        raise subprocess.CalledProcessError(code, command)
    return output


def replay(args):
    input_manifest, queries, _ = validate_input(args.input)
    manifest = validate_submission(args.submission)
    args.output.mkdir(parents=True, exist_ok=False)
    python = args.runtime / 'bin/python'
    base = Path(subprocess.check_output(
        [str(python), '-c', 'import sys; print(sys.base_prefix)'], text=True).strip()).resolve()
    started = time.monotonic()
    def invoke(input_directory, current, remaining):
        return invoke_isolated(args, python, base, input_directory, current, remaining)
    output = execute_batches(args.input, input_manifest, queries, args.output, args.timeout, invoke)
    seal_output(output, args.output / 'sealed/run', args.input / 'manifest.json', args.output / 'batches.json')
    # Preserve the aggregate local log used by earlier operator probes.
    with (args.output / 'inference.log').open('wb') as target:
        for source in sorted((args.output / 'logs').glob('*.log')):
            with source.open('rb') as stream:
                shutil.copyfileobj(stream, target)
    report = {'isolation': 'rootless user/PID/network/mount namespaces; all capabilities dropped',
              'host_uid': os.getuid(), 'namespace_uid': 0, 'elapsed_seconds': time.monotonic() - started,
              'checkpoint_sha256': manifest['checkpoint']['sha256'], 'gpu_enabled': args.gpu,
              'production_docker_acceptance': False, 'h100_budget_acceptance': False}
    write_json(args.output / 'replay.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['submission', 'input', 'output', 'runtime']:
        parser.add_argument('--' + name, type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument('--timeout', type=int, default=18000)
    parser.add_argument('--cpus', type=int, default=4)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    if not shutil.which('bwrap') or args.timeout < 1 or args.cpus < 1:
        parser.error('bwrap and positive limits are required')
    print(json.dumps(replay(args), indent=2))
