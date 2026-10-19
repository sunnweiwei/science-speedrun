"""Independent PointNeXt-family SHAPE classifier, full 8192-point input.

Six downsampling stages, k=16. Width/depth beyond published details are explicit
choices, not claimed author settings: base32, channels64..2048, two-layer residual
set abstraction. No radius/point-count reductions. Reference implementation:
PointNeXt/OpenPoints db31e0d94ede35bc672ef06d1b6e4073a9c269d5 (MIT).
"""
import importlib.util
from pathlib import Path
import torch
from torch import nn


class FPS:
    def __init__(self,root):
        path=Path(root)/'point-ops/pathfinder_fps.so'
        spec=importlib.util.spec_from_file_location('pathfinder_fps',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);self.module=module
    @torch.no_grad()
    def __call__(self,p,n):
        p=p.float().contiguous();b,count,_=p.shape
        indices=torch.zeros((b,n),dtype=torch.int32,device=p.device)
        temp=torch.full((b,count),1e10,dtype=torch.float32,device=p.device)
        self.module.fps(b,count,n,p,temp,indices)
        return indices.long()


def index(x,indices):
    batch=torch.arange(x.shape[0],device=x.device).view(-1,*([1]*(indices.ndim-1)))
    return x[batch,indices]


class Abstraction(nn.Module):
    def __init__(self,incoming,outgoing):
        super().__init__()
        self.mlp=nn.Sequential(nn.Conv2d(incoming+3,outgoing//2,1,bias=False),nn.BatchNorm2d(outgoing//2),nn.ReLU(),
            nn.Conv2d(outgoing//2,outgoing,1,bias=False),nn.BatchNorm2d(outgoing))
        self.skip=nn.Conv1d(incoming,outgoing,1,bias=False)
    def forward(self,p,f,fps):
        selected=fps(p,p.shape[1]//2);q=index(p,selected)
        # Distances are geometric, no autograd. Chunking caps the largest KNN
        # matrix while preserving exact nearest-neighbor selection.
        with torch.no_grad():
            neighbors=torch.cat([torch.cdist(chunk.float(),p.float()).topk(16,largest=False).indices
                                 for chunk in q.split(512,dim=1)],dim=1)
        dp=index(p,neighbors)-q.unsqueeze(2);features=index(f,neighbors)
        grouped=torch.cat((dp,features),dim=-1).permute(0,3,1,2).contiguous()
        pooled=self.mlp(grouped).amax(-1)
        residual=self.skip(index(f,selected).transpose(1,2))
        return q,torch.relu(pooled+residual).transpose(1,2)


class ShapeClassifier(nn.Module):
    def __init__(self,fps):
        super().__init__();self.fps=fps
        self.embed=nn.Linear(3,32)
        self.stages=nn.ModuleList([Abstraction(a,b) for a,b in zip([32,64,128,256,512,1024],[64,128,256,512,1024,2048])])
        self.head=nn.Sequential(nn.Linear(2048,512),nn.LayerNorm(512),nn.ReLU(),nn.Dropout(.5),
                                nn.Linear(512,256),nn.LayerNorm(256),nn.ReLU(),nn.Linear(256,1))
    def forward(self,points):
        if points.shape[1:]!=(8192,3):raise ValueError('SHAPE requires B,8192,3 normalized physical coordinates')
        p=points;f=self.embed(p)
        for stage in self.stages:p,f=stage(p,f,self.fps)
        return self.head(f.amax(1)).squeeze(-1)
