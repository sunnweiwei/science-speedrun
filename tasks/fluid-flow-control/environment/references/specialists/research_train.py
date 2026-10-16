"""Public-source learners on unchanged Research v2 native profiles.

Specialist comparison baseline, NOT paper-identical physics or a shared submission.
Run through native_runtime.sh in the pinned offline image. See RESEARCH_BASELINES.md.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

TASK = Path(__file__).resolve().parents[3]  # environment/references/specialists -> task
EVALUATION = Path(os.environ.get('HYDROGYM_EVALUATOR_CODE', str(TASK / 'environment/evaluation')))
sys.path.insert(0, str(EVALUATION))
from case_spec import local_only, select_cases, verify_resources
from evaluate import write_json


class TrainingAdapter:
    """Only action coordinates and learner time-limit semantics are adapted."""
    def __init__(self, env, case, output):
        import gymnasium as gym
        import numpy as np
        from native_envs import ReplayAdapter
        self.env, self.case, self.output = env, case, output
        (output / 'native').mkdir()
        self.checked = ReplayAdapter(env, case, output / 'native')
        self.observation_space = env.observation_space
        self.action_space = gym.spaces.Box(-1, 1, env.action_space.shape, dtype=np.float32)
        self.phase, self.training_step, self.episode = 'zero_control', 0, 0
        self.log = (output / 'episodes.jsonl').open('a', buffering=1)

    def reset(self, seed=None):
        import numpy as np
        self.total, self.count, self.rows = 0., 0, []
        self.episode += 1
        obs, info = self.checked.reset(seed=123 if self.phase != 'training' else seed)
        if not np.isfinite(obs).all():
            raise ValueError('Nonfinite reset observation')
        return np.asarray(obs, dtype=np.float32), info

    def step_native(self, action):
        import numpy as np
        try:
            obs, reward, done, truncated, info = self.checked.step(action)
        except Exception as error:
            write_json(self.output / 'transition-error.json', {'phase': self.phase,
                'training_step': self.training_step, 'episode': self.episode,
                'attempted_control': self.count + 1, 'native_action': np.asarray(action).tolist(),
                'error': str(error), 'previous_transitions': self.rows[-5:]})
            raise
        self.total += reward
        self.count += 1
        if self.count == 1 or self.count % 10 == 0 or done or truncated:
            write_json(self.output / 'training-state.json', {'phase': self.phase,
                'training_step': self.training_step, 'episode': self.episode,
                'control': self.count, 'horizon': self.case['horizon']})
        self.rows.append({'control': self.count, 'reward': reward,
                          'action': np.asarray(action).tolist(), 'info': info})
        if done or truncated:
            self.log.write(json.dumps({'phase': self.phase, 'training_step': self.training_step,
                'episode': self.episode, 'return': self.total, 'controls': self.count,
                'trajectory': self.rows}, allow_nan=False) + '\n')
        # The frozen native environment stops at its time horizon, not a failure.
        # Keep all controls/rewards; bootstrap the critic at that time limit.
        return np.asarray(obs, dtype=np.float32), reward, False, bool(done or truncated), info

    def step(self, action):
        from research_predict import native_action
        return self.step_native(native_action(action, self.env.action_space.low, self.env.action_space.high))


def worker(args, spec, case):
    import numpy as np
    import torch
    from native_envs import make_env
    from public_learner import run
    from export_policy import export
    native = make_env(case, args.data)
    env = TrainingAdapter(native, case, args.output)
    provenance = dict(environment=case['id'], protocol='research-v2-specialist/v1',
        data_release_id=spec['data_release_id'], data_release_sha256=spec['data_release_sha256'],
        runtime_profile=case['runtime_profile'], runtime_profile_sha256=case['runtime_profile_sha256'],
        horizon=case['horizon'], time_limit_bootstrap=True, observation_normalization='none',
        native_action_low=native.action_space.low.tolist(), native_action_high=native.action_space.high.tolist(),
        evaluation_seed=123, paper_identical=False)
    write_json(args.output / 'protocol.json', provenance)

    def finish_package(package, metadata):
        metadata.update(native_action_low=provenance['native_action_low'],
                        native_action_high=provenance['native_action_high'])
        write_json(package / 'config.json', metadata)

    def publish(checkpoint):
        package = args.output / ('submission-' + checkpoint.stem.split('-')[-1])
        temporary = package.with_name('.' + package.name + '.tmp')
        # Author factory initialization for export must not perturb learner RNG.
        state = np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]):
                metadata = export(checkpoint, temporary, args.author_source,
                                  Path(__file__).with_name('research_predict.py'))
            finish_package(temporary, metadata)
            os.rename(temporary, package)
        finally:
            np.random.set_state(state)

    can_close = True
    try:
        zero = args.output / 'zero-submission'
        zero.mkdir()
        shutil.copyfile(Path(__file__).with_name('research_predict.py'), zero / 'predict.py')
        (zero / 'infer.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec python "$(dirname -- "$0")/predict.py"\n')
        (zero / 'README.md').write_text('Native zero control; organizer comparison, not a learned model.\n')
        finish_package(zero, dict(zero_control=True, observation_size=case['observation_size'],
                                 action_size=case['action_size'], physics=provenance))
        env.reset(seed=123)
        for _ in range(case['horizon']):
            env.step_native(np.zeros(native.action_space.shape, dtype=native.action_space.dtype))
        write_json(args.output / 'zero.json', {'return': env.total, 'controls': env.count})
        run(env, args, provenance, checkpoint_callback=publish)
        # Preserve a development-selected best separately, even if it is step 0.
        with torch.random.fork_rng(devices=[]):
            meta = export(args.output / 'best.pt', args.output / '.submission-best.tmp', args.author_source,
                          Path(__file__).with_name('research_predict.py'))
        finish_package(args.output / '.submission-best.tmp', meta)
        os.rename(args.output / '.submission-best.tmp', args.output / 'submission-best')
    except BaseException:
        can_close = case['backend'] != 'maia'
        raise
    finally:
        env.log.close()
        if can_close:
            native.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', required=True)
    p.add_argument('--data', type=Path, default=Path('/data'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--author-source', type=Path,
                   default=Path(os.environ.get('HYDROGYM_AUTHOR_SOURCE', '/author-source')))
    p.add_argument('--algo', choices=['PPO', 'TD3', 'DDPG'], default='PPO')
    p.add_argument('--recipe', default='public')
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--eval-every', type=int, default=4000)
    p.add_argument('--save-every', type=int, default=400)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--worker', action='store_true')
    args = p.parse_args()
    if args.steps <= 0 or args.steps % 400 or args.eval_every <= 0 or args.eval_every % 400:
        p.error('Positive multiples of 400 required')
    args.mode = 'train'
    args.data, args.output = args.data.resolve(), args.output.resolve()
    if not args.worker:
        args.output.mkdir(parents=True, exist_ok=False)
    spec, cases = select_cases([args.case])
    case = cases[0]
    verify_resources(args.data, cases)
    local_only(args.data, case, args.output)
    start = time.monotonic()
    try:
        if not args.worker:
            helper = EVALUATION / 'maia_gpu_rank.py'
            devices = json.loads(subprocess.check_output([sys.executable, str(helper), '--list-gpus'], text=True))
            if len(devices) != case['solver_processes'] or len(set(devices)) != len(devices):
                raise ValueError('Need exactly the frozen number of distinct solver GPUs')
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(devices)
            write_json(args.output / 'launch.json', {'arguments': vars(args) | {
                'data': str(args.data), 'output': str(args.output), 'author_source': str(args.author_source)},
                'gpu_uuids': devices, 'case': case})
        if case['backend'] != 'maia' or args.worker:
            worker(args, spec, case)
        else:
            from hydrogym.maia.workspace import prepare_maia_workspace
            work, props = prepare_maia_workspace(case['id'], work_dir=str(args.output / 'cfd'),
                use_clean_cache=False, local_fallback_dir=str(args.data / 'environments'))
            solver = [sys.executable, str(EVALUATION / 'maia_gpu_rank.py')]
            cmd = ['mpirun', '--bind-to', 'none', '-np', '1', sys.executable, str(Path(__file__).resolve()),
                *sys.argv[1:], '--worker', ':', '-np', str(case['solver_processes']), *solver, props, '--silent']
            subprocess.run(cmd, cwd=work, env=dict(os.environ, OMPI_ALLOW_RUN_AS_ROOT='1',
                OMPI_ALLOW_RUN_AS_ROOT_CONFIRM='1', OMPI_MCA_coll_hcoll_enable='0'), check=True)
        if not args.worker:
            write_json(args.output / 'training-finished.json', {'wall_seconds': time.monotonic() - start,
                'status': 'trained-and-exported-awaiting-independent-evaluation'})
    except BaseException as error:
        write_json(args.output / ('worker-error.json' if args.worker else 'error.json'),
                   {'error': str(error), 'traceback': traceback.format_exc()})
        if args.worker:
            from mpi4py import MPI
            MPI.COMM_WORLD.Abort(1)
        raise


if __name__ == '__main__':
    main()
