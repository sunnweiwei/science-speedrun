"""Timed DDP SHAPE adaptation on declared full-context surface datasets."""
import argparse,json,os,time,sys
from pathlib import Path
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from shape_data import ShapeSamples
from shape_model import FPS,ShapeClassifier
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import external,device_guard,write_json,sha256,source_provenance
from execution_resources import device_guard


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','manifest','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seconds',type=int,required=True);p.add_argument('--steps',type=int,default=1000000)
    p.add_argument('--batch',type=int,default=2);p.add_argument('--eval-every',type=int,default=100)
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();device_guard('cuda')
    if min(a.seconds,a.steps,a.batch,a.eval_every)<=0:raise ValueError('Positive limits required')
    rank=int(os.getenv('RANK',0));world=int(os.getenv('WORLD_SIZE',1));local=int(os.getenv('LOCAL_RANK',0))
    torch.cuda.set_device(local);torch.set_num_threads(2);torch.manual_seed(a.seed)
    if world>1:dist.init_process_group('nccl')
    device=torch.device('cuda',local);out=external(a.output)
    if rank==0:out.mkdir(parents=True,exist_ok=False)
    if world>1:dist.barrier()
    train=ShapeSamples(a.manifest,'train',a.steps*a.batch*world,a.seed)
    dev=ShapeSamples(a.manifest,'validation',64,10000000)
    loader=DataLoader(train,batch_size=a.batch,sampler=range(rank,len(train),world),num_workers=0,drop_last=True)
    model=ShapeClassifier(FPS(a.root)).to(device)
    wrapped=DDP(model,device_ids=[local]) if world>1 else model
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
    architecture=dict(stage='shape',points=8192,radius_um=10,axes='zyx',widths=[64,128,256,512,1024,2048],neighbors=16)
    receipt=dict(architecture=architecture,manifest_sha256=sha256(a.manifest),paper_comparable=False,
        adaptation=train.manifest.get('adaptation', 'few-shot real preview surfaces; synthetic translated distinct-instance false joins'),
        eligible_train_ids=[r['id'] for r in train.rows],eligible_validation_ids=[r['id'] for r in dev.rows],
        world_size=world,batch_per_gpu=a.batch,seed=a.seed,time_limit_seconds=a.seconds,
        slurm_job_id=os.getenv('SLURM_JOB_ID'),source=source_provenance(),lr=a.lr,
        validation_samples=64,parameter_count=sum(p.numel() for p in model.parameters()))
    if rank==0:write_json(out/'run.json',receipt)
    start=time.monotonic();best=float('inf');bf16=torch.cuda.is_bf16_supported()
    def save(name,step,score):
        if rank==0:
            tmp=out/(name+'.tmp');torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                architecture=architecture,step=step,validation_bce=score,provenance=receipt),tmp);tmp.replace(out/name)
    for step,(points,target) in enumerate(loader,1):
        model.train();optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf16):
            logits=wrapped(points.to(device));loss=torch.nn.functional.binary_cross_entropy_with_logits(logits.float(),target.to(device))
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
        stop=torch.tensor(int(time.monotonic()-start>=a.seconds or step>=a.steps),device=device)
        if world>1:dist.all_reduce(stop,op=dist.ReduceOp.MAX)
        if step==1 or step%a.eval_every==0 or stop.item():
            if world>1:
                # Validation uses unwrapped model: synchronize running BN statistics.
                for buffer in model.buffers():dist.broadcast(buffer,src=0)
            model.eval();stats=torch.zeros(5,device=device)
            with torch.inference_mode():
                for i in range(rank,len(dev),world):
                    x,y=dev[i];pred=model(torch.as_tensor(x,device=device)[None]).float()[0]
                    stats[0]+=torch.nn.functional.binary_cross_entropy_with_logits(pred,torch.tensor(y,device=device))
                    stats[1]+=((pred>=0)==bool(y));stats[2]+=1;stats[3]+=((pred>=0)&bool(y));stats[4]+=((pred<0)&(not bool(y)))
            if world>1:dist.all_reduce(stats)
            bce=float(stats[0]/stats[2]);row=dict(step=step,seconds=time.monotonic()-start,train_loss=float(loss.detach()),
                validation_bce=bce,validation_accuracy=float(stats[1]/stats[2]),
                validation_positive_recall=float(stats[3]/32),validation_negative_recall=float(stats[4]/32),
                paper_comparable=False)
            if rank==0:
                with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
            if bce<best:best=bce;save('best.pt',step,bce)
            save('last.pt',step,bce)
        # Validation/checkpoint duration counts toward the stage limit too.
        stop=torch.tensor(int(time.monotonic()-start>=a.seconds or step>=a.steps),device=device)
        if world>1:dist.all_reduce(stop,op=dist.ReduceOp.MAX)
        if stop.item():
            save('last.pt',step,bce);break
    if world>1:dist.destroy_process_group()

if __name__=='__main__':main()
