import json
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
@lru_cache(maxsize=8)
def array(path):return np.load(path,mmap_mode='r',allow_pickle=False)
class Dataset(torch.utils.data.Dataset):
    def __init__(self,root,limit=None,cache=False):
        self.root=Path(root)
        self.rows=json.loads((self.root/'manifest.json').read_text())['samples']
        if limit:self.rows=self.rows[:limit]
        self.cached=None
        if cache and self.rows:
            from collections import defaultdict
            groups=defaultdict(list)
            for i,r in enumerate(self.rows):groups[r['path']].append((i,r['index']))
            for name,indices in groups.items():
                path=(self.root/name).resolve()
                if not path.is_relative_to(self.root.resolve()):raise ValueError('input outside root')
                source=np.load(path,mmap_mode='r',allow_pickle=False)
                if self.cached is None:self.cached=torch.empty((len(self.rows),*source.shape[1:]),dtype=torch.float32)
                dst,src=zip(*indices)
                self.cached[list(dst)]=torch.as_tensor(source[list(src)]/100,dtype=torch.float32)
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        if self.cached is not None:return self.cached[i],r.get('label',-1),r['id']
        p=(self.root/r['path']).resolve()
        if not p.is_relative_to(self.root.resolve()):raise ValueError('input outside root')
        # Division precedes float32 conversion, matching official FACED/TUEV collate.
        x=np.asarray(array(str(p))[r['index']])/100
        return torch.as_tensor(x,dtype=torch.float32),r.get('label',-1),r['id']
