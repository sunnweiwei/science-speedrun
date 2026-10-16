"""Train/resume one shared PPO on the frozen native profiles (all backends).

Each invocation is one finite experiment inside continuous AutoResearch.
Reuse --resume across tasks: never build a bank of specialist checkpoints.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

EVALUATION = Path(os.environ.get('HYDROGYM_EVALUATOR_CODE',
    str(Path(__file__).resolve().parents[2]/'evaluation')))
sys.path.insert(0, str(EVALUATION))
from case_spec import local_only, select_cases, verify_resources, digest
from evaluate import write_json


def worker(args, spec, case):
    import gymnasium as gym
    from native_envs import make_env, ReplayAdapter
    from stable_baselines3.common.utils import set_random_seed
    from train import train
    set_random_seed(args.seed)
    native = make_env(case, args.data)
    checked = ReplayAdapter(native, case, args.output)

    class TrainingEnv(gym.Env):
        observation_space = native.observation_space
        action_space = native.action_space
        broken = False

        def reset(self, seed=None, options=None):
            return checked.reset(args.seed if seed is None else seed)

        def step(self, action):
            try:
                obs, reward, done, truncated, info = checked.step(action)
            except BaseException:
                self.broken = True
                raise
            # Finite training rollouts bootstrap the critic. CFD horizon,
            # actions, observations and native reward are not modified.
            return obs, reward, False, bool(done or truncated), info

        def close(self):
            if not self.broken or case['backend'] != 'maia':
                native.close()

    training_spec = dict(spec, cases=[dict(c, episode_end='time_limit') for c in spec['cases']])
    training_case = next(c for c in training_spec['cases'] if c['id']==case['id'])
    train(TrainingEnv(), training_spec, training_case, args.steps, args.resume,
          args.submission, publish_every=args.publish_every)
    write_json(args.output/'completed.json', {
        'status': 'shared-policy-trained-and-published', 'case': case['id'],
        'requested_steps': args.steps, 'profiles_sha256': digest(spec),
        'resume': str(args.resume), 'submission': str(args.submission),
        'scope': 'One phase of shared training, not specialist or SOTA certification'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', required=True)
    p.add_argument('--data', type=Path, default=Path('/data'))
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--publish-every', type=int, default=2000)
    p.add_argument('--resume', type=Path, default=Path('/workspace/research/shared-ppo.zip'))
    p.add_argument('--submission', type=Path, default=Path('/workspace/submission'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--worker', action='store_true')
    args = p.parse_args()
    if args.steps <= 0 or args.steps % 200 or args.publish_every <= 0 or args.publish_every % 200:
        p.error('Training/publication intervals must be positive multiples of PPO rollout length 200')
    for key in ('data','output','resume','submission'):
        setattr(args,key,getattr(args,key).resolve())
    if args.output.is_relative_to(args.data) or args.submission.is_relative_to(args.data):
        p.error('Keep output and submission outside read-only data')
    if not args.worker:
        args.output.mkdir(parents=True,exist_ok=False)
    spec, cases = select_cases([args.case]);case=cases[0]
    verify_resources(args.data,cases)
    local_only(args.data,case,args.output)
    try:
        if not args.worker:
            helper=EVALUATION/'maia_gpu_rank.py'
            devices=json.loads(subprocess.check_output([sys.executable,str(helper),'--list-gpus'],text=True))
            if len(devices) < case['solver_processes']:
                raise RuntimeError('Insufficient distinct GPUs for frozen solver profile')
            devices=devices[:case['solver_processes']]
            os.environ['CUDA_VISIBLE_DEVICES']=','.join(devices)
            write_json(args.output/'launch.json',dict(case=case['id'],cuda_device_uuids=devices,
                profiles_sha256=digest(spec),seed=args.seed))
        if case['backend']!='maia' or args.worker:
            worker(args,spec,case)
            return
        from hydrogym.maia.workspace import prepare_maia_workspace
        work,props=prepare_maia_workspace(case['id'],work_dir=str(args.output/'cfd'),
            use_clean_cache=False,local_fallback_dir=str(args.data/'environments'))
        cmd=['mpirun','--bind-to','none','-np','1',sys.executable,str(Path(__file__).resolve()),
             *sys.argv[1:],'--worker',':','-np',str(case['solver_processes']),sys.executable,
             str(EVALUATION/'maia_gpu_rank.py'),props,'--silent']
        env=dict(os.environ,OMPI_ALLOW_RUN_AS_ROOT='1',OMPI_ALLOW_RUN_AS_ROOT_CONFIRM='1',
                 OMPI_MCA_coll_hcoll_enable='0')
        subprocess.run(cmd,cwd=work,env=env,check=True)
    except BaseException as error:
        write_json(args.output/'error.json',{'status':'training_error','error':str(error)})
        if args.worker and case['backend']=='maia':
            from mpi4py import MPI
            MPI.COMM_WORLD.Abort(1)
        raise


if __name__=='__main__':main()
