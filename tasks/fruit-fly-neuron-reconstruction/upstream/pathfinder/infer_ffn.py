"""Bounded seed-and-grow FFN reconstruction from EM alone; no GT inputs.

Single-checkpoint/single-order baseline. Does not implement paper consensus or
semantic segmentation. Exhausted limits are reported, never hidden as completion.
"""
import argparse
from ffn_movement import MOVE_THRESHOLD,FaceMaxMovement,disconnected_bias
import json,time
from pathlib import Path
import numpy as np
from ffn_seeds import policy_peaks,seed_is_clear
import torch
from ffn_model import FFN15
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import external,device_guard,sha256,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--roi',type=int,nargs=6,required=True,metavar=('Z0','Z1','Y0','Y1','X0','X1'))
    p.add_argument('--seconds',type=int,default=1800);p.add_argument('--max-calls',type=int,default=100000)
    p.add_argument('--voxel-size-nm',type=float,nargs=3,default=[16,16,16])
    p.add_argument('--origin-nm',type=float,nargs=3,default=[0,0,0],help='Outer corner of raw voxel zero, ZYX nm')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda');a=p.parse_args();device_guard(a.device)
    spacing=np.asarray(a.voxel_size_nm);origin=np.asarray(a.origin_nm)
    if not np.isfinite(spacing).all() or (spacing<=0).any() or not np.isfinite(origin).all():
        raise ValueError('Finite positive voxel spacing and finite origin required')
    out=external(a.output);out.mkdir(parents=True,exist_ok=False)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if ck['architecture']['stage']!='ffn':raise ValueError('FFN checkpoint required')
    if not np.array_equal(spacing,ck['architecture'].get('voxel_size_nm',[16,16,16])):
        raise ValueError('Inference grid must match FFN checkpoint resolution')
    model=FFN15(activation_checkpointing=False).to(a.device);model.load_state_dict(ck['model']);model.eval()
    image=np.load(a.raw,mmap_mode='r');start=np.asarray(a.roi[::2]);stop=np.asarray(a.roi[1::2]);shape=stop-start
    if image.ndim!=3 or image.dtype!=np.uint8 or (start<16).any() or (stop>np.array(image.shape)-16).any() or (shape<=0).any():
        raise ValueError('ROI must have at least 16 voxels of real EM context')
    context=np.asarray(image[tuple(slice(int(a)-16,int(b)+16) for a,b in zip(start,stop))])
    seeds=policy_peaks(context)-16
    segmentation=np.zeros(tuple(shape),np.uint32);clock=time.monotonic();calls=0;objects=0;exhausted=False
    torch.set_num_threads(2)
    with torch.inference_mode():
        for seed in seeds:
            if not seed_is_clear(segmentation,seed):continue
            if time.monotonic()-clock>=a.seconds or calls>=a.max_calls:exhausted=True;break
            # Canvas includes a context halo; predictions are only assigned inside ROI.
            canvas=np.full(tuple(shape+32),-2.944439,np.float32);center=seed+16;canvas[tuple(center)]=2.944439
            initialized=np.zeros(canvas.shape,bool);initialized[tuple(center)]=True
            movement=FaceMaxMovement(center,canvas.shape)
            while movement.queue:
                c=movement.next(canvas,blocked=lambda point: segmentation[tuple(x-16 for x in point)]>0)
                if c is None:break
                if time.monotonic()-clock>=a.seconds or calls>=a.max_calls:exhausted=True;break
                local=tuple(slice(int(x-16),int(x+17)) for x in c)
                global_c=start+c-16;global_sl=tuple(slice(int(x-16),int(x+17)) for x in global_c)
                x=torch.from_numpy(np.array(image[global_sl],np.float32)/255)[None,None].to(a.device)
                q=torch.from_numpy(canvas[local].copy())[None,None].to(a.device)
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=a.device=='cuda'):
                    predicted=model(x,q)[0,0].float().cpu().numpy()
                predicted=disconnected_bias(canvas[local],predicted,initialized[local])
                canvas[local]=predicted;initialized[local]=True;calls+=1
                movement.update(predicted,c)
            interior=canvas[16:-16,16:-16,16:-16]
            selected=(interior>=2.944439)&(segmentation==0)&(canvas[tuple(center)]>=MOVE_THRESHOLD)
            if selected.sum()>=10:
                objects+=1;segmentation[selected]=objects
            if exhausted:break
    np.save(out/'segmentation.npy',segmentation,allow_pickle=False)
    write_json(out/'receipt.json',dict(checkpoint_sha256=sha256(a.checkpoint),raw_sha256=sha256(a.raw),roi=a.roi,
        seconds=time.monotonic()-clock,calls=calls,objects=objects,seed_count=len(seeds),
        budget_exhausted=exhausted,omitted_voxel_fraction=float((segmentation==0).mean()),
        ground_truth_used=False,consensus=False,paper_comparable=False,
        metadata=dict(axes='zyx',voxel_size_nm=spacing.tolist(),raw_origin_nm=origin.tolist(),
            origin_nm=(origin+start*spacing).tolist(),shape=shape.tolist(),coordinate_convention='voxel_outer_corner')))

if __name__=='__main__':main()
