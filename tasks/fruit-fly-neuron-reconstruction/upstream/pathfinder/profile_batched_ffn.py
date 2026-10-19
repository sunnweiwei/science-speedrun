"""Four-GPU FFN batching comparison on explicit shared inputs, without labels."""
import concurrent.futures,json,os,subprocess,sys
from pathlib import Path
import argparse
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import device_guard,external,write_json,sha256
from fetch_axonem import reserve


def compare_completed(reference,candidate):
    def completed(receipt):
        return {r['index']:r for r in receipt['tile_results'] if not r['budget_exhausted']}
    left=completed(reference);right=completed(candidate);common=sorted(left.keys()&right.keys())
    voxels=changed=0
    for index in common:
        a,b=left[index],right[index]
        if any(a[k]!=b[k] for k in ('start','stop')):raise ValueError('Profile geometry differs')
        arrays=[]
        for row in (a,b):
            if sha256(row['path'])!=row['sha256']:raise ValueError('Profile tile bytes differ')
            arrays.append(np.load(row['path'],mmap_mode='r',allow_pickle=False))
        if arrays[0].shape!=arrays[1].shape:raise ValueError('Profile array shapes differ')
        voxels+=arrays[0].size;changed+=int(np.count_nonzero(arrays[0]!=arrays[1]))
    return dict(compared_complete_tiles=len(common),compared_voxels=voxels,changed_voxels=changed,
        exact_on_compared_tiles=bool(common) and changed==0,
        scope='Exact raster agreement only on jointly complete tiles; no ground truth or quality claim')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','image-manifest','checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--roi',type=int,nargs=6,required=True)
    p.add_argument('--batches',type=int,nargs=4,default=[16,32,64,128])
    p.add_argument('--seconds',type=int,default=900);p.add_argument('--max-tiles',type=int,default=128)
    a=p.parse_args();device_guard('cuda');root=external(a.root);out=external(a.output)
    if root not in out.parents or out.exists():raise ValueError('New output under original budget root required')
    if min(a.batches)<1 or len(set(a.batches))!=4 or a.seconds<1 or a.max_tiles<1:raise ValueError('Invalid profile limits')
    masks=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if len(masks)!=4 or len(set(masks))!=4 or not all(masks):raise ValueError('Exactly four allocated GPUs required')
    ledger=json.loads((root/'budget.json').read_text());job=next(j for j in ledger['jobs'] if j['job_id']==os.environ['SLURM_JOB_ID'])
    if job['gpus']!=4 or 'actual_seconds' in job or job['reserved_seconds']<a.seconds+300:raise ValueError('Bounded active ledger reservation required')
    manifest=json.loads(a.image_manifest.read_text());source=a.image_manifest.parent
    if sha256(source/'raw.npy')!=manifest['files']['raw.npy']:raise ValueError('Raw image changed')
    if sha256(source/'image_valid_z.npy')!=manifest['files']['image_valid_z.npy']:raise ValueError('Image validity changed')
    from run_continuous_ffn import validate_geometry
    validate_geometry(manifest,np.load(source/'raw.npy',mmap_mode='r'),np.load(source/'image_valid_z.npy',mmap_mode='r'),a.roi)
    reserve(root,4*a.max_tiles*64**3*4+10_000_000);out.mkdir()
    code=Path(__file__).parent
    def one(i,batch):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=masks[i]);dest=out/f'batch-{batch}'
        command=[sys.executable,str(code/'batched_ffn.py'),'--raw',str(source/'raw.npy'),
          '--checkpoint',str(a.checkpoint),'--output',str(dest),'--roi',*map(str,a.roi),
          '--batch',str(batch),'--tile-size','64','--seconds',str(a.seconds),'--max-tiles',str(a.max_tiles),
          '--origin-nm',*map(str,manifest['origin_nm'])]
        with (out/f'batch-{batch}.log').open('w') as f:result=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:return dict(batch=batch,exit_code=result.returncode)
        receipt=dest/'receipt.json';r=json.loads(receipt.read_text())
        return dict(batch=batch,exit_code=0,receipt=str(receipt),receipt_sha256=sha256(receipt),
          seconds=r['seconds'],model_windows=r['model_windows'],tiles=len(r['tile_results']),
          complete_tiles=sum(not t['budget_exhausted'] for t in r['tile_results']),
          time_exhausted=r['time_exhausted'],windows_per_second=r['model_windows']/r['seconds'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(lambda t:one(*t),enumerate(a.batches)))
    comparisons=[]
    if results[0]['exit_code']==0:
        reference=json.loads(Path(results[0]['receipt']).read_text())
        for row in results[1:]:
            if row['exit_code']==0:comparisons.append(dict(reference_batch=a.batches[0],candidate_batch=row['batch'],
                **compare_completed(reference,json.loads(Path(row['receipt']).read_text()))))
    write_json(out/'summary.json',dict(results=results,comparisons=comparisons,identical_tile_grid=True,
      image_manifest_sha256=sha256(a.image_manifest),checkpoint_sha256=sha256(a.checkpoint),roi=a.roi,
      ground_truth_used=False,inference_only=True,slurm_job_id=os.environ['SLURM_JOB_ID'],
      automatic_selection=False,scope='Inspect throughput, completeness and numerical differences before choosing a batch'))
    if any(row['exit_code'] for row in results):raise RuntimeError('Some profile variants failed; summary preserves successful variants')

if __name__=='__main__':main()
