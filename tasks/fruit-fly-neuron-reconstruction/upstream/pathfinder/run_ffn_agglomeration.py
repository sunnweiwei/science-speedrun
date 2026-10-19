"""Slurm GPU execution of independent two-seed FFN agglomeration candidates.

Decision points must be built from the base prediction only. Model inputs use
real image context. Candidate CPU workers feed one batched GPU predictor per
rank, preserving independent seed canvases and avoiding per-candidate GPUs.
"""
import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import json
from pathlib import Path
import queue
import sys
import time
import numpy as np
from ffn_resegmentation import resegment_pair
from dataset_contract import sha256
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import device_guard, external, write_json
from execution_resources import device_guard


def execute(image, base, decisions, predict_batch, image_offset, *, workers=8, seconds=3600, result_sink=None):
    requests = queue.Queue(); started = time.monotonic()
    offset = np.asarray(image_offset, dtype=int)
    def predict(raw, seed):
        if time.monotonic()-started >= seconds:
            raise TimeoutError('Allocation stage time limit')
        future = Future(); requests.put((raw[0],seed[0],future)); return future.result()[None]
    def one(row):
        center = np.asarray(row['center_zyx'],dtype=int)
        lo, hi = center-50, center+51
        if (lo<0).any() or (hi>np.asarray(base.shape)).any():
            return dict(**row,complete=False,accepted=False,reason='insufficient_base_context')
        raw_lo,raw_hi = lo+offset,hi+offset
        if (raw_lo<0).any() or (raw_hi>np.asarray(image.shape)).any():
            raise ValueError('Base/image coordinate mismatch')
        sl = tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))
        rsl = tuple(slice(int(a),int(b)) for a,b in zip(raw_lo,raw_hi))
        try:
            result = resegment_pair(np.asarray(image[rsl]),np.asarray(base[sl]),row['pair'],predict)
            return dict(center_zyx=row['center_zyx'],**result)
        except TimeoutError:
            return dict(**row,complete=False,accepted=False,reason='time_limit')
    returned=completed=accepted=0
    rows = iter(decisions); active = []; results = []; exhausted = False; batches=windows=0; fatal=None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while True:
            while len(active)<workers and not exhausted and fatal is None and time.monotonic()-started<seconds:
                try: row = next(rows)
                except StopIteration: exhausted=True; break
                active.append(pool.submit(one,row))
            done = [f for f in active if f.done()]
            for f in done:
                try:
                    value=f.result()
                    if result_sink is None:results.append(value)
                    else:result_sink(value)
                    returned+=1;completed+=bool(value['complete']);accepted+=bool(value['accepted'])
                except Exception as exc: fatal=exc
                active.remove(f)
            if not active and (exhausted or fatal is not None or time.monotonic()-started>=seconds):
                break
            batch=[]
            try: batch.append(requests.get(timeout=.01))
            except queue.Empty: continue
            # Short rendezvous lets independent CPU seed workers fill the GPU batch.
            end = time.monotonic()+.002
            while len(batch)<workers:
                try: batch.append(requests.get(timeout=max(0,end-time.monotonic())))
                except queue.Empty: break
            try:
                if fatal is not None:
                    raise RuntimeError('Model/candidate failure') from fatal
                if time.monotonic()-started>=seconds:
                    raise TimeoutError('Allocation stage time limit')
                outputs=np.asarray(predict_batch(np.stack([r[0] for r in batch]),np.stack([r[1] for r in batch])))
                if outputs.shape != (len(batch),33,33,33) or not np.isfinite(outputs).all():
                    raise ValueError('Invalid model batch')
                for (_,_,future),prediction in zip(batch,outputs): future.set_result(prediction)
                batches+=1; windows+=len(batch)
            except Exception as exc:
                for _,_,future in batch: future.set_exception(exc)
    if fatal is not None:
        raise RuntimeError('Agglomeration execution failed') from fatal
    return dict(results=results,completed_candidates=completed,
                requested_candidates=len(decisions),accepted_pairs=accepted,
                gpu_batches=batches,model_windows=windows,seconds=time.monotonic()-started,
                complete=returned==len(decisions) and completed==returned)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('raw','base','image_metadata','base_metadata','decisions','checkpoint','output'):
        parser.add_argument('--'+key.replace('_','-'),required=True,type=Path)
    parser.add_argument('--shards',type=int,default=4);parser.add_argument('--shard-index',type=int,required=True)
    parser.add_argument('--batch',type=int,default=8);parser.add_argument('--seconds',type=int,required=True)
    parser.add_argument('--resume-from',type=Path,help='Finished prior shard receipt; reuse only fully evaluated candidates')
    a=parser.parse_args();device_guard('cuda')
    if not 0<=a.shard_index<a.shards or min(a.batch,a.seconds)<=0:raise ValueError('Invalid resource arguments')
    output=external(a.output)
    if output.exists():raise ValueError('Output exists')
    import torch
    from ffn_model import FFN15
    torch.set_num_threads(2)
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model=FFN15(activation_checkpointing=False).cuda().eval();model.load_state_dict(checkpoint['model'],strict=True)
    def predictor(raw,seed):
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
            value=model(torch.from_numpy(raw[:,None]).cuda(),torch.from_numpy(seed[:,None]).cuda())
        return value[:,0].float().cpu().numpy()
    image=np.load(a.raw,mmap_mode='r');base=np.load(a.base,mmap_mode='r')
    im=json.loads(a.image_metadata.read_text());bm=json.loads(a.base_metadata.read_text())
    if im['voxel_size_nm'] != bm['voxel_size_nm'] or im['voxel_size_nm'] != [16]*3:
        raise ValueError('Require matching 16nm model grids')
    delta=np.asarray(bm['origin_nm'])-np.asarray(im['origin_nm'])
    if (delta%16).any():raise ValueError('Grid origins are misaligned')
    plan=json.loads(a.decisions.read_text())
    if plan['base_sha256'] != sha256(a.base) or plan['ground_truth_used'] is not False:
        raise ValueError('Decision source mismatch')
    from decision_table import load_rows
    rows=load_rows(plan,a.decisions)[a.shard_index::a.shards]
    signature=dict(checkpoint_sha256=sha256(a.checkpoint),decisions_sha256=sha256(a.decisions),
        raw_sha256=sha256(a.raw),base_sha256=plan['base_sha256'])
    retained=[];pending=rows;reused=accepted_reused=0
    if 'decision_table' in plan:
        from score_stream import ScoreWriter,load_shard
        from resume_agglomeration import resume_to_sink
        writer=ScoreWriter(output.with_suffix('.jsonl'))
        try:
            if a.resume_from:
                counts,pending=resume_to_sink(load_shard(a.resume_from),rows,signature,a.shard_index,a.shards,writer.append)
                reused=counts['count'];accepted_reused=counts['accepted']
            result=execute(image,base,pending,predictor,delta//16,workers=a.batch,seconds=a.seconds,result_sink=writer.append)
            result.pop('results');result['result_stream']=writer.finish()
        finally:writer.close()
    else:
        if a.resume_from:
            from resume_agglomeration import resume_rows
            from score_stream import load_shard
            retained,pending=resume_rows(load_shard(a.resume_from),rows,signature,a.shard_index,a.shards)
            reused=len(retained);accepted_reused=sum(r['accepted'] for r in retained)
        result=execute(image,base,pending,predictor,delta//16,workers=a.batch,seconds=a.seconds)
        result['results']=retained+result['results']
    if a.resume_from:
        result.update(resume=dict(source=str(a.resume_from.resolve()),source_sha256=sha256(a.resume_from),
            reused_complete_candidates=reused,new_requested_candidates=len(pending),
            new_completed_candidates=result['completed_candidates'],cost_scope='seconds, gpu_batches and model_windows describe this continuation only'))
        result.update(completed_candidates=reused+result['completed_candidates'],
            requested_candidates=len(rows),accepted_pairs=accepted_reused+result['accepted_pairs'])
        result['complete']=result['completed_candidates']==len(rows)
    result.update(schema='ffn-agglomeration-shard/v1',shard_index=a.shard_index,shards=a.shards,
        **signature,ground_truth_used=False,author_implementation_bit_parity=False)
    write_json(output,result)


if __name__=='__main__':main()
