"""Build a narrow-mount Apptainer invocation for packaged Python inference.

The host orchestrator validates the original Slurm ledger before GPU execution.
This helper does not submit jobs, grant GPUs, or mount the budget/GT directory.
"""
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import device_guard, external


def invocation(command, image, readonly, output, *, gpu=None):
    image=external(image); output=external(output)
    if not image.is_file() or not output.is_dir():
        raise ValueError('Existing image and dedicated output directory required')
    inputs=sorted({external(p) for p in readonly})
    if not inputs or any(not p.exists() for p in inputs):
        raise ValueError('Existing explicitly mounted inputs required')
    # Do not let a writable output expose/modify any supposedly read-only input.
    if any(p==output or p in output.parents or output in p.parents for p in inputs):
        raise ValueError('Input and output mounts must not overlap')
    if len(command)<2 or not Path(command[1]).is_absolute():
        raise ValueError('Explicit Python script path required')
    script=Path(command[1]).resolve()
    if not any(script==p or (p.is_dir() and p in script.parents) for p in inputs):
        raise ValueError('Inference script must be in read-only inputs')
    paths=[image,output,*inputs]
    if any(any(c in str(p) for c in ',:\n\r') for p in paths):
        raise ValueError('Mount paths cannot contain Apptainer delimiters')
    if any(str(p) in ('/','/usr','/scratch','/tmp','/var','/opt')
           or p == Path(os.environ.get('PATHFINDER_SCRATCH_ROOT', '/scratch')).resolve()
           for p in inputs):
        raise ValueError('Mount explicit assets, not system or scratch roots')
    cache=output/'.runtime-cache'
    cache.mkdir(exist_ok=True)
    env=dict(os.environ)
    # Host bind/environment overrides must not widen the reviewed mount list.
    for name in list(env):
        if name.startswith(('APPTAINER','SINGULARITY')):
            del env[name]
    env.update(APPTAINER_CACHEDIR=str(cache),APPTAINER_TMPDIR=str(cache),TMPDIR=str(cache))
    inside=dict(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        TMPDIR=str(cache),XDG_CACHE_HOME=str(cache),MPLCONFIGDIR=str(cache/'matplotlib'))
    args=['apptainer','exec','--containall','--cleanenv','--no-home',
          '--no-mount','cwd,hostfs,bind-paths','--net','--network','none','--pwd','/tmp']
    if gpu is not None:
        device_guard('cuda')
        if not re.fullmatch(r'(?:\d+|GPU-[A-Za-z0-9-]+|MIG-[A-Za-z0-9/-]+)',gpu):
            raise ValueError('Exactly one allocated GPU mask required')
        if gpu not in os.environ.get('CUDA_VISIBLE_DEVICES','').split(','):
            raise ValueError('GPU mask is not in current allocation visibility')
        args.append('--nv')
        inside.update(CUDA_VISIBLE_DEVICES=gpu,SLURM_JOB_ID=os.environ['SLURM_JOB_ID'],
                      SLURM_JOB_PARTITION=os.environ['SLURM_JOB_PARTITION'],
                      PATHFINDER_GPU_PARTITION=os.environ.get('PATHFINDER_GPU_PARTITION', 'gpu'))
    for name,value in inside.items():
        if any(c in value for c in ',\n\r'):
            raise ValueError('Invalid explicit container environment')
        args.extend(['--env',name+'='+value])
    for path in inputs:
        args.extend(['--bind',f'{path}:{path}:ro'])
    args.extend(['--bind',f'{output}:{output}:rw',str(image),'python3',*command[1:]])
    return args,env
