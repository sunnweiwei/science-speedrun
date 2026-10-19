"""Numerical/full-size SHAPE throughput probe; synthetic inputs, no quality claim."""
import argparse,json,os,time
from pathlib import Path
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from shape_model import ShapeClassifier,FPS
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import device_guard,write_json,external


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args();root=external(a.root)
    device_guard('cuda');rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local);dist.init_process_group('nccl');torch.set_num_threads(2);torch.manual_seed(11+rank)
    model=ShapeClassifier(FPS(root)).cuda();wrapped=DDP(model,device_ids=[local]);opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    points=torch.rand(2,8192,3,device='cuda')*2-1;target=torch.tensor([0.,1.],device='cuda')
    start=time.monotonic()
    for step in range(8):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=wrapped(points);loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,target)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step()
    torch.cuda.synchronize()
    if rank==0:write_json(root/'shape-profile.json',dict(synthetic=True,quality_claim=False,steps=8,world_size=dist.get_world_size(),
        points=8192,neighbors=16,stages=6,params=sum(p.numel() for p in model.parameters()),
        seconds=time.monotonic()-start,max_memory_gb=torch.cuda.max_memory_allocated()/1e9,loss=loss.item()))
    dist.destroy_process_group()

if __name__=='__main__':main()
