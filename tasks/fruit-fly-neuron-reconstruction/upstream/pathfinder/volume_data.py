"""Memory-mapped, explicitly masked real EM crops. No GPU work in dataset workers."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from sense import prompt_mask


class Crops(Dataset):
    def __init__(self,manifest,split,shape,length,seed=42):
        self.manifest=json.loads(Path(manifest).read_text());self.shape=shape;self.length=length;self.seed=seed
        self.records=self.manifest['splits'][split];self.volumes=[]
        for row in self.records:
            d=Path(manifest).resolve().parent/row['folder'];raw=np.load(d/'raw.npy',mmap_mode='r');labels=np.load(d/'labels.npy',mmap_mode='r')
            valid=np.load(d/'valid.npy',mmap_mode='r');centers=np.load(d/'centers.npy')
            keep=((centers>=shape//2)&(centers<=np.array(raw.shape)-(shape-shape//2))).all(axis=1)
            centers=centers[keep]
            if not len(centers):raise ValueError('No crop center with requested full FOV')
            ids=np.asarray(labels[tuple(centers.T)])
            groups=[centers[ids==sid] for sid in np.unique(ids) if sid>0]
            self.volumes.append((raw,labels,valid,groups))
    def __len__(self):return self.length
    def __getitem__(self,index):
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,int(index)]))
        raw,labels,valid,groups=self.volumes[int(rng.integers(len(self.volumes)))]
        centers=groups[int(rng.integers(len(groups)))];center=centers[int(rng.integers(len(centers)))]
        sl=tuple(slice(int(c-self.shape//2),int(c-self.shape//2+self.shape)) for c in center)
        x=np.array(raw[sl],dtype=np.float32)/255;gt=np.asarray(labels[sl]);mask=np.array(valid[sl],dtype=np.float32)
        local=(self.shape//2,)*3;sid=int(gt[local])
        if sid==0 or mask[local]!=1:raise ValueError('Center must be labeled')
        target=np.asarray(gt==sid,dtype=np.float32)
        prompt=prompt_mask(gt,sid,16);prompt[mask==0]=.5
        return tuple(torch.from_numpy(a[None]) for a in [x,prompt,target,mask])
