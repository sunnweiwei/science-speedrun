"""Private native solver launcher; invoked by evaluate_native.py, not by a model."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import traceback

from case_spec import local_only, select_cases, verify_resources
from evaluate import package_hash, write_json
from protocol import Policy, PolicyError, replay_episode


def worker(case, data, submission, output):
    from native_envs import make_env, ReplayAdapter
    env = make_env(case, data)
    adapter = ReplayAdapter(env, case, output)
    policy = None
    can_close = True
    before = package_hash(submission)
    try:
        policy = Policy(submission, output / 'policy.log', timeout=120)
        pid = policy.process.pid
        try:
            ticks = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
        except FileNotFoundError:
            # An immediately exiting candidate is a protocol/policy failure,
            # not missing solver infrastructure. No live session to reap.
            pass
        else:
            write_json(output / 'policy-process.json', {'pid': pid, 'start_ticks': ticks})
        episodes = [replay_episode(adapter, policy, case, seed) for seed in case['seeds']]
        policy.close()
        policy = None
        result = {'status': 'ok', 'episodes': episodes, 'submission_sha256': before}
    except PolicyError as error:
        result = {'status': 'policy_error', 'error': str(error), 'submission_sha256': before}
    except BaseException:
        # A failing MPI collective must be aborted by main(), not followed by
        # another collective close that can wait forever for a broken solver.
        can_close = case['backend'] != 'maia'
        raise
    finally:
        if policy is not None:
            policy.abort()
        if can_close:
            env.close()
    if package_hash(submission) != before:
        raise RuntimeError('Model package mutated during native replay')
    write_json(output / 'result.json', result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--submission', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--worker', action='store_true')
    args = p.parse_args()
    args.data, args.output, args.submission = (p.resolve() for p in (args.data, args.output, args.submission))
    spec, cases = select_cases([args.case])
    case = cases[0]
    verify_resources(args.data, cases)
    local_only(args.data, case, args.output)
    try:
        if not args.worker:
            # Enumerate in a fresh child: CUDA visibility is fixed at first init.
            import json
            helper = Path(__file__).with_name('maia_gpu_rank.py')
            devices = json.loads(subprocess.check_output([sys.executable, str(helper), '--list-gpus'], text=True))
            required = case['solver_processes']
            if len(devices) < required:
                raise RuntimeError(f'{case["id"]} requires {required} distinct GPUs; allocated {len(devices)}')
            selected = devices[:required]
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(selected)
            write_json(args.output / 'gpu-allocation.json', {'cuda_device_uuids': selected})
        if case['backend'] != 'maia' or args.worker:
            worker(case, args.data, args.submission, args.output)
            return
        from hydrogym.maia.workspace import prepare_maia_workspace
        work, props = prepare_maia_workspace(case['environment'],
            work_dir=str(args.output / 'cfd'), use_clean_cache=False,
            local_fallback_dir=str(args.data / 'environments'))
        solver = [sys.executable, str(Path(__file__).with_name('maia_gpu_rank.py'))]
        command = ['mpirun', '--bind-to', 'none', '-np', '1', sys.executable, str(Path(__file__).resolve()),
            '--worker', '--case', case['id'], '--data', str(args.data), '--output', str(args.output),
            '--submission', str(args.submission), ':', '-np', str(case['solver_processes']),
            *solver, props, '--silent']
        environment = dict(os.environ, OMPI_ALLOW_RUN_AS_ROOT='1', OMPI_ALLOW_RUN_AS_ROOT_CONFIRM='1',
                           OMPI_MCA_coll_hcoll_enable='0')
        subprocess.run(command, cwd=work, env=environment, check=True)
    except BaseException as error:
        write_json(args.output / ('worker-error.json' if args.worker else 'launcher-error.json'),
            {'status': 'infrastructure_error', 'error': str(error), 'traceback': traceback.format_exc()})
        if args.worker:
            # A broken MPI environment cannot safely perform collective env.close.
            from mpi4py import MPI
            MPI.COMM_WORLD.Abort(1)
        raise


if __name__ == '__main__':
    main()
