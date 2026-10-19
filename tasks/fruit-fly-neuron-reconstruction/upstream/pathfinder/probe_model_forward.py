"""Full-size deterministic-input GPU runtime probe, not reconstruction quality."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import torch

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--package',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
p.add_argument('--host-netns');p.add_argument('--forbidden-root',type=Path)
p.add_argument('--forbidden-home',type=Path)
a=p.parse_args()
if os.environ.get('SLURM_JOB_PARTITION')!=os.environ.get('PATHFINDER_GPU_PARTITION', 'gpu') or not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('Allocated gpu execution required')
if a.host_netns:
    if (a.forbidden_root is None or a.forbidden_home is None
            or os.readlink('/proc/self/ns/net')==a.host_netns
            or a.forbidden_root.exists() or a.forbidden_home.exists()):
        raise RuntimeError('Container isolation check failed')
    for target in ('/package','/audit'):
        rows=[line.split() for line in Path('/proc/mounts').read_text().splitlines() if line.split()[1]==target]
        if not rows or any('ro' not in row[3].split(',') for row in rows):raise RuntimeError('Read-only mount missing')
sys.path.insert(0,str(a.package/'runtime/pathfinder'))
from model_package import verify,construct
from shape_model import FPS
from common import sha256,write_json
check=verify(a.package,load_extension=True)
torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
rng=torch.Generator(device='cpu').manual_seed(42);outputs={};rows={}
for stage in ('ffn','sense','shape'):
    ckpt=torch.load(a.package/f'weights/{stage}.pt',map_location='cpu',weights_only=True)
    model=construct(stage,ckpt['architecture']);model.load_state_dict(ckpt['model'],strict=True)
    if stage=='shape':model.fps=FPS(a.package)
    model=model.cuda().eval();begun=time.monotonic()
    with torch.inference_mode():
        if stage=='shape':
            points=torch.rand((1,8192,3),generator=rng)*2-1
            points=points/points.norm(dim=-1,keepdim=True).clamp_min(1e-6)*torch.rand((1,8192,1),generator=rng).pow(1/3)
            prediction=model(points.cuda())
        else:
            n=33 if stage=='ffn' else 128
            raw=torch.rand((1,1,n,n,n),generator=rng)
            prompt=torch.full_like(raw,-2.94443898 if stage=='ffn' else .5)
            prompt[:,:,n//2,n//2,n//2]=2.94443898 if stage=='ffn' else .95
            prediction=model(raw.cuda(),prompt.cuda())
    array=prediction.float().cpu().numpy()
    if not np.isfinite(array).all():raise ValueError('Nonfinite '+stage+' output')
    outputs[stage]=array;rows[stage]=dict(shape=list(array.shape),seconds=time.monotonic()-begun)
    del model,ckpt,prediction;torch.cuda.empty_cache()
np.savez(a.output,**outputs)
write_json(a.output.with_suffix('.json'),dict(package_check=check,outputs=rows,outputs_sha256=sha256(a.output),
    torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
    synthetic_input=True,reconstruction_quality_measured=False,container_isolation_checked=bool(a.host_netns)))
