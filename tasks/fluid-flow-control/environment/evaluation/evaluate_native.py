"""Replay one submission on selected real native cases, without score anchors.

Uses frozen organizer execution profiles. The Harbor entrypoint exports the same
per-case physical metrics; there is no mixed-unit aggregate or paper-score anchor.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from case_spec import digest, estimated_seconds, select_cases, verify_resources
from evaluate import package_hash, write_json
from protocol import InfrastructureError, PolicyError
from metrics import aggregate as aggregate_physical


def implementation_hash():
    here = Path(__file__).resolve().parent
    files = [p for folder in (here, here.parent / 'data') for p in folder.iterdir()
             if p.suffix in ('.py', '.sh')]
    return digest({str(p.relative_to(here.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in files})


def stop_policy(output):
    """Clean up the separate policy session if its MPI parent was forcibly killed."""
    receipt = output / 'policy-process.json'
    if not receipt.exists():
        return
    record = json.loads(receipt.read_text())
    pid = record['pid']
    try:
        ticks = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
        # Do not signal a reused PID, or a group other than this policy's session.
        if ticks == record['start_ticks'] and os.getpgid(pid) == pid:
            os.killpg(pid, signal.SIGKILL)
    except (FileNotFoundError, ProcessLookupError):
        pass


def run_case(case, data, package, output, env=None):
    output.mkdir()
    here = Path(__file__).resolve().parent
    command = ['bash', str(here / 'native_runtime.sh'), case['backend'], 'python',
        str(here / 'run_native.py'), '--case', case['id'], '--data', str(data),
        '--submission', str(package), '--output', str(output)]
    start = time.monotonic()
    with (output / 'native.log').open('wb') as log:
        process = subprocess.Popen(command, cwd=output, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                   env=env)
        try:
            code = process.wait(timeout=case['timeout_sec'])
        except subprocess.TimeoutExpired as error:
            raise InfrastructureError(f'Native case timed out: {case["id"]}') from error
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            stop_policy(output)
    if code or not (output / 'result.json').is_file():
        detail = f'exit code {code}'
        for filename in ('worker-error.json', 'launcher-error.json'):
            path = output / filename
            if path.exists():
                detail = json.loads(path.read_text()).get('error', detail)
                break
        raise InfrastructureError(f'Native execution failed for {case["id"]}: {detail}; see {output / "native.log"}')
    result = json.loads((output / 'result.json').read_text())
    result['wall_seconds'] = time.monotonic() - start
    return result


def visible_gpus():
    """CUDA device UUIDs this process may use (enumerated in a fresh child, as run_native.py does)."""
    helper = Path(__file__).with_name('maia_gpu_rank.py')
    try:
        return json.loads(subprocess.check_output([sys.executable, str(helper), '--list-gpus'], text=True))
    except (OSError, subprocess.CalledProcessError, ValueError):
        return []


# One policy account per case that can run at once (created in environment/Dockerfile).
POLICY_UIDS = tuple(range(10001, 10017))
# Cases replayed side by side share the host's CPUs and memory bandwidth: on 8xH100 they ran
# 1.1-2.2x slower than alone, so a per-case limit sized for a solo run (about 2.5x its estimate)
# would turn a slow but healthy simulation into an infrastructure error that voids the reward.
SIDE_BY_SIDE_TIMEOUT_FACTOR = 2


def replay_cases(cases, replay, devices):
    """Run cases side by side, each on its own `solver_processes` GPUs, longest first.

    Each running case also gets its own policy account (HYDROGYM_POLICY_UID): closing a policy kills
    every process of its account, which must not reach another case's policy.
    With fewer GPUs than a case needs (or none visible) the cases run one at a time, as before."""
    if not devices or len(devices) < max(c['solver_processes'] for c in cases):
        return {case['id']: replay(case, None) for case in cases}
    free, lock = list(devices), threading.Condition()
    uids = list(POLICY_UIDS[:len(devices)])

    def run(case):
        need = case['solver_processes']
        with lock:
            lock.wait_for(lambda: len(free) >= need and uids)
            mine = [free.pop(0) for _ in range(need)]
            uid = uids.pop(0)
        try:
            slow = dict(case, timeout_sec=case['timeout_sec'] * SIDE_BY_SIDE_TIMEOUT_FACTOR)
            return case['id'], replay(slow, dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(mine),
                                                 HYDROGYM_POLICY_UID=str(uid)))
        finally:
            with lock:
                free.extend(mine)
                uids.append(uid)
                lock.notify_all()

    order = sorted(cases, key=lambda c: -estimated_seconds(c['id']) * c['solver_processes'])
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        return dict(pool.map(run, order))


def evaluate_native(data, submission, output, names, runner=run_case):
    data, submission, output = (Path(p).resolve() for p in (data, submission, output))
    if (output.is_relative_to(data) or data.is_relative_to(output) or
            output.is_relative_to(submission) or submission.is_relative_to(output)):
        raise InfrastructureError('Output must be separate from both source data and submission')
    if output.exists() and any(output.iterdir()):
        raise InfrastructureError('Evaluation output must be empty')
    output.mkdir(parents=True, exist_ok=True)
    try:
        spec, cases = select_cases(names)
        verify_resources(data, cases)
        identity = {'evaluation_mode': 'native-profile-replay', 'benchmark_validated': False,
            'metric_version': spec['metric_version'], 'metrics_sha256': spec['metrics_sha256'],
            'evaluator_sha256': implementation_hash(),
            'data_release_id': spec['data_release_id'], 'data_release_sha256': spec['data_release_sha256'],
            'profile_version': spec['profile_version'], 'profiles_sha256': digest(spec),
            'selected_case_ids': [c['id'] for c in cases],
            'all_frozen_cases_selected': len(cases) == len(spec['cases'])}
        write_json(output / 'profiles.json', spec)
        results = {}
        try:
            model_hash = package_hash(submission)
        except PolicyError as error:
            model_hash = None
            results = {c['id']: {'status': 'policy_error', 'error': str(error)} for c in cases}
        if model_hash is not None:
            with tempfile.TemporaryDirectory(prefix='.native-snapshot-', dir=output) as tmp:
                package = Path(tmp) / 'submission'
                shutil.copytree(submission, package, symlinks=True)
                # Readable (never writable) by the unprivileged policy account.
                os.chmod(tmp, 0o755)
                for path in [package, *package.rglob('*')]:
                    if not path.is_symlink():
                        path.chmod(0o555 if path.is_dir() or os.access(path, os.X_OK) else 0o444)
                if package_hash(package) != model_hash or package_hash(submission) != model_hash:
                    raise InfrastructureError('Submission changed while snapshotting')
                write_json(output / 'identity.json', dict(identity, submission_sha256=model_hash))
                progress = threading.Lock()

                def replay(case, env):
                    try:
                        row = (runner(case, data, package, output / case['id'], env) if env is not None
                               else runner(case, data, package, output / case['id']))
                        if row.get('submission_sha256') != model_hash:
                            raise InfrastructureError('Native worker replayed a different model package')
                        if row.get('status') == 'ok':
                            episodes = row.get('episodes', [])
                            if [e.get('seed') for e in episodes] != case['seeds'] or any(
                                    e.get('steps') != case['horizon'] or
                                    not (e.get('terminated') is True or e.get('truncated') is True)
                                    for e in episodes):
                                raise InfrastructureError('Wrong/missing complete native episode')
                            returns = [e.get('return') for e in episodes]
                            if not all(type(v) in (float, int) and math.isfinite(v) for v in returns):
                                raise InfrastructureError('Nonfinite/missing native returns')
                            row['mean_return'] = sum(returns) / len(returns)
                            if not math.isfinite(row['mean_return']):
                                raise InfrastructureError('Nonfinite mean native return')
                            row['primary_metric'] = aggregate_physical(case['primary_metric'], episodes)
                        elif row.get('status') != 'policy_error':
                            raise InfrastructureError('Native worker did not return an explicit outcome')
                    except Exception as error:
                        row = {'status': 'infrastructure_error', 'error': str(error)}
                    with progress:
                        results[case['id']] = row
                        write_json(output / 'partial-results.json', dict(identity, submission_sha256=model_hash, cases=results))
                    return row

                replay_cases(cases, replay, visible_gpus())
                results = {c['id']: results[c['id']] for c in cases if c['id'] in results}
                if package_hash(package) != model_hash:
                    raise InfrastructureError('Submission snapshot mutated during evaluation')
        verify_resources(data, cases)
        if digest(select_cases(names)[0]) != identity['profiles_sha256']:
            raise InfrastructureError('Executable profiles changed during replay')
        if implementation_hash() != identity['evaluator_sha256']:
            raise InfrastructureError('Evaluator implementation changed during replay')
        ok = sum(row['status'] == 'ok' for row in results.values())
        result = dict(identity, submission_sha256=model_hash,
            status='ok' if ok == len(cases) else 'completed_with_errors',
            successful_cases=ok, requested_cases=len(cases), cases=results)
        write_json(output / 'results.json', result)
        return result
    except Exception as error:
        write_json(output / 'error.json', {'status': 'infrastructure_error', 'error': str(error)})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('/data'))
    p.add_argument('--submission', type=Path, default=Path('/workspace/submission'))
    p.add_argument('--output', type=Path)
    p.add_argument('--case', action='append', required=True, help='Frozen case ID; repeat, or use all')
    p.add_argument('--preflight', action='store_true', help='Check source/profile bindings without running CFD')
    args = p.parse_args()
    if args.preflight:
        spec, cases = select_cases(args.case)
        verify_resources(args.data.resolve(), cases)
        print(json.dumps({'status': 'source-profile-preflight-passed', 'benchmark_validated': False,
            'profiles_sha256': digest(spec), 'cases': cases}, indent=2))
        return
    if args.output is None:
        p.error('--output is required for replay')
    result = evaluate_native(args.data, args.submission, args.output, args.case)
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, indent=2))
    if result['status'] != 'ok':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
