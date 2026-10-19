"""Bounded batch 1/2/4 measurement on one unchanged full FFN window.

Partial diagnostic trajectories are never usable reconstruction windows.
"""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
import numpy as np
from batched_ffn import schedule
from seed_cache import get_seeds
from common import device_guard,sha256,write_json


def digest_array(array):
    digest=hashlib.sha256()
    for plane in array:digest.update(np.ascontiguousarray(plane).tobytes())
    return digest.hexdigest()


def run(plan_path):
    device_guard('cuda');plan=json.loads(plan_path.read_text());root=Path(plan['root']);out=Path(plan['output'])
    if root not in out.parents or out.exists():raise ValueError('Fresh diagnostic output required')
    for name,digest in plan['pins'].items():
        if sha256(name)!=digest:raise ValueError('Frozen benchmark input changed')
    ledger=json.loads((root/'budget.json').read_text());job=next(j for j in ledger['jobs'] if j['job_id']==os.environ['SLURM_JOB_ID'])
    if 'actual_seconds' in job or job['gpus']!=1 or job['reserved_seconds']!=600:raise ValueError('Registered ten-minute one-GPU job required')
    if len(os.environ.get('CUDA_VISIBLE_DEVICES','').split(','))!=1:raise ValueError('One allocated GPU required')
    raw=np.load(plan['raw'],mmap_mode='r');w=plan['window'];context=raw[tuple(slice(a-16,b+16) for a,b in zip(w['start'],w['stop']))]
    seeds,cache=get_seeds(context,root,Path(plan['seed_cache']),dict(raw_sha256=plan['pins'][plan['raw']],start=w['start'],stop=w['stop']))
    if not cache['hit']:raise ValueError('Benchmark requires existing verified seeds')
    import torch
    from ffn_model import FFN15
    torch.set_num_threads(2);model=FFN15(activation_checkpointing=False).cuda()
    model.load_state_dict(torch.load(plan['checkpoint'],map_location='cpu',weights_only=True)['model'],strict=True);model.eval()
    if not torch.cuda.is_bf16_supported():raise ValueError('Frozen BF16 policy requires supported GPU')
    out.mkdir();rows=[]
    with (out/'gpu-utilization.csv').open('x') as utilization:
        sampler=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,name,uuid,utilization.gpu,utilization.memory,memory.used,power.draw','--format=csv','-l','1'],stdout=utilization,stderr=subprocess.STDOUT)
        try:
            # Reverse order checks warm-up bias without increasing the window/call budget.
            for batch in (1,2,4,4,2,1):
                outputs=[];prediction_seconds=0.;trajectory=hashlib.sha256()
                def predict(images,canvases):
                    nonlocal prediction_seconds
                    begun=time.monotonic()
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                        values=model(torch.from_numpy(images[:,None]).cuda(),torch.from_numpy(canvases[:,None]).cuda())[:,0].float().cpu().numpy()
                    prediction_seconds+=time.monotonic()-begun
                    trajectory.update(values[0].tobytes())
                    return values
                def save(tile,seg,stats):outputs.append(dict(index=tile['index'],segmentation_sha256=digest_array(seg),statistics=stats))
                started=time.time();torch.cuda.reset_peak_memory_stats()
                report=schedule(raw,[dict(w,index=i) for i in range(batch)],predict,batch=batch,seconds=65,max_calls=1024,
                    on_result=save,seed_provider=lambda context,start,stop:seeds,seed_order='ascending')
                row=dict(batch=batch,started_unix=started,ended_unix=time.time(),execution=report,
                    predictor_seconds=prediction_seconds,model_windows_per_second=report['model_windows']/report['seconds'],
                    first_trajectory_sha256=trajectory.hexdigest(),outputs=outputs,gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated())
                rows.append(row);write_json(out/'progress.json',rows)
        finally:
            sampler.terminate()
            try:sampler.wait(timeout=5)
            except subprocess.TimeoutExpired:sampler.kill();sampler.wait()
    for name,digest in plan['pins'].items():
        if sha256(name)!=digest:raise ValueError('Benchmark source/input changed')
    write_json(out/'result.json',dict(complete=True,job_id=job['job_id'],plan_sha256=sha256(plan_path),seed_cache=cache,
        full_reconstruction_complete=False,rows=rows,ground_truth_used=False,
        scope='Six finite partial diagnostic trajectories; identical full window/checkpoint/seeds. No reconstruction output accepted. Calls/s include growth overhead; utilization samples retained.'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);run(p.parse_args().plan)
