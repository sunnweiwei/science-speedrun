"""Four allocated GPUs reconstruct an explicit continuous EM ROI in disjoint tiles."""
import argparse,concurrent.futures,json,os,subprocess,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import device_guard,external,write_json,sha256


def validate_geometry(manifest,raw,valid,roi=None):
    if (manifest.get('schema')!='pathfinder-continuous-image/v1' or manifest.get('axes')!='zyx'
        or manifest.get('coordinate_convention')!='voxel_outer_corner' or manifest.get('ground_truth_included') is not False):
        raise ValueError('Image-only explicit physical ZYX manifest required')
    if manifest.get('voxel_size_nm')!=[16]*3:raise ValueError('Current FFN recipe requires16nm grid')
    if raw.ndim!=3 or raw.dtype!=np.uint8 or manifest.get('shape')!=list(raw.shape):raise ValueError('Raw geometry mismatch')
    if valid.dtype!=np.bool_ or valid.shape!=(raw.shape[0],):raise ValueError('Explicit Z observation mask required')
    origin=np.asarray(manifest.get('origin_nm'),dtype=float)
    if origin.shape!=(3,) or not np.isfinite(origin).all():raise ValueError('Finite physical origin required')
    roi=[v for n in raw.shape for v in (16,n-16)] if roi is None else list(roi)
    if len(roi)!=6 or any(type(v)!=int for v in roi):raise ValueError('Six integer ROI bounds required')
    start=np.asarray(roi[::2]);stop=np.asarray(roi[1::2])
    if (start<16).any() or (stop>np.asarray(raw.shape)-16).any() or (stop<=start).any():
        raise ValueError('ROI requires real16voxel halo on all sides')
    return roi


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=36000);p.add_argument('--image-manifest',type=Path,required=True)
    p.add_argument('--model-package',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--container-image',type=Path,help='Optional isolated GPU stages; requires explicit model package')
    p.add_argument('--roi',type=int,nargs=6);p.add_argument('--plan-only',action='store_true')
    p.add_argument('--resume-from',type=Path,help='Prior finished four-shard FFN output; reuse complete tiles')
    a=p.parse_args();root=external(a.root)
    if a.seconds<=0:raise ValueError('Positive stage time required')
    image=a.image_manifest;source=image.parent
    manifest=json.loads(image.read_text());out=external(a.output)
    if root not in out.parents:raise ValueError('Output must remain under original budget root')
    if sha256(source/'raw.npy')!=manifest['files']['raw.npy']:raise ValueError('Continuous raw integrity mismatch')
    if sha256(source/'image_valid_z.npy')!=manifest['files']['image_valid_z.npy']:raise ValueError('Observation mask integrity mismatch')
    raw=np.load(source/'raw.npy',mmap_mode='r',allow_pickle=False)
    valid=np.load(source/'image_valid_z.npy',mmap_mode='r',allow_pickle=False)
    roi=validate_geometry(manifest,raw,valid,a.roi)
    from model_package import verify
    package=external(a.model_package);package_check=verify(package)
    code=package/'runtime/pathfinder';checkpoint=package/'weights/ffn.pt'
    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    plan=dict(image_manifest_sha256=sha256(image),image_manifest=str(image),
      image_observation=manifest['image_valid'],roi=roi,batch=16,tile_size=64,
      checkpoint_sha256=sha256(checkpoint),package_check=package_check,
      gpus=4,seconds_per_shard=a.seconds,ground_truth_used=False,scope='continuous developmental adaptation',
      geometric_context_does_not_prove_all_image_voxels_observed=True)
    readonly=None
    if a.resume_from:
        prior=external(a.resume_from)
        if root not in prior.parents or a.container_image:
            raise ValueError('Resume currently requires native execution and source under budget root')
        previous=json.loads((prior/'plan.json').read_text())
        for key in ('image_manifest_sha256','roi','batch','tile_size','checkpoint_sha256','gpus'):
            if previous[key]!=plan[key]:raise ValueError('Incompatible prior FFN plan: '+key)
        previous_run=json.loads((prior/'receipt.json').read_text())
        if sorted((r['shard'],r['exit_code']) for r in previous_run['shards'])!=[(i,0) for i in range(4)]:
            raise ValueError('Prior FFN shard processes must have finished successfully')
        plan['resume']=dict(directory=str(prior),plan_sha256=sha256(prior/'plan.json'),
            receipt_sha256=sha256(prior/'receipt.json'),
            shards={str(i):sha256(prior/f'shard-{i}/receipt.json') for i in range(4)})
    if a.container_image:
        if not a.model_package or not a.container_image.is_file():raise ValueError('Explicit package and existing container required')
        readonly=[a.model_package.resolve(),(source/'raw.npy').resolve()]
        plan['isolation']=dict(image=str(a.container_image.resolve()),image_sha256=sha256(a.container_image),
            readonly=[str(p) for p in readonly],writable=str(out),network='none',host_home=False,
            scope='GPU model stages only; trusted host orchestration remains outside container')
    if a.plan_only:print(json.dumps(plan,indent=2));return
    device_guard('cuda');masks=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if len(masks)!=4 or not all(masks) or len(set(masks))!=4:raise ValueError('Exactly four allocated GPUs required')
    ledger=json.loads((root/'budget.json').read_text())
    reserved=[j for j in ledger['jobs'] if j['job_id']==os.environ['SLURM_JOB_ID']]
    if len(reserved)!=1 or reserved[0]['gpus']!=4 or 'actual_seconds' in reserved[0]:raise ValueError('Active four-GPU ledger reservation required')
    if a.seconds+300>reserved[0]['reserved_seconds']:raise ValueError('Reserve at least5min beyond inference loops')
    out.mkdir(parents=True,exist_ok=False);write_json(out/'plan.json',plan)
    def one(i):
        command=[sys.executable,str(code/'batched_ffn.py'),'--raw',str(source/'raw.npy'),
          '--checkpoint',str(checkpoint),'--output',str(out/f'shard-{i}'),
          '--roi',*map(str,roi),'--batch','16','--tile-size','64',
          '--seconds',str(a.seconds),'--shards','4','--shard-index',str(i),
          '--origin-nm',*map(str,manifest['origin_nm'])]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=masks[i])
        if a.resume_from:command.extend(['--resume-from',str(prior/f'shard-{i}/receipt.json')])
        if a.container_image:
            from isolate_inference import invocation
            command,env=invocation(command,a.container_image,readonly,out,gpu=masks[i])
        with (out/f'shard-{i}.log').open('w') as f:
            result=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT)
        return dict(shard=i,exit_code=result.returncode)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(one,range(4)))
    write_json(out/'receipt.json',dict(shards=rows,slurm_job_id=os.environ['SLURM_JOB_ID'],ground_truth_used=False))
    if any(r['exit_code'] for r in rows):raise RuntimeError('Shard failure; inspect log files')

if __name__=='__main__':main()
