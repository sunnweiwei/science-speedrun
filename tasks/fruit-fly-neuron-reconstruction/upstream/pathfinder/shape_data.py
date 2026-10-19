"""Shared physical surface sampling for few-shot SHAPE adaptation and inference.

8192 area-weighted surface points inside a 10 um sphere, centered and /10.
Synthetic negatives join two distinct instances by rigid translation, not paper
corruptions. Preview train/dev/test identity splits must never be mixed.
"""
import json
from pathlib import Path
import numpy as np

POINTS = 8192
RADIUS_UM = 10.0


def sample_surface(vertices, faces, center, rng, count=POINTS):
    vertices=np.asarray(vertices,dtype=np.float64);center=np.asarray(center,dtype=np.float64)
    triangles=vertices[np.asarray(faces,dtype=np.int64)]
    # Sphere intersects triangle AABB: retain a conservative superset, then
    # reject sampled points outside the sphere. This preserves area weighting.
    delta=np.maximum(np.maximum(triangles.min(1)-center,center-triangles.max(1)),0)
    triangles=triangles[(delta**2).sum(1)<=RADIUS_UM**2]
    area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)/2
    good=area>0;triangles=triangles[good];area=area[good]
    if not len(area):raise ValueError('No positive-area surface in SHAPE context')
    probability=area/area.sum();chunks=[];have=0
    for _ in range(100):
        n=max(2*(count-have),1024)
        t=triangles[rng.choice(len(triangles),n,p=probability)]
        uv=rng.random((n,2));u=np.sqrt(uv[:,0]);v=uv[:,1]
        points=(1-u[:,None])*t[:,0]+(u*(1-v))[:,None]*t[:,1]+(u*v)[:,None]*t[:,2]
        points=(points-center)/RADIUS_UM
        points=points[(points**2).sum(1)<=1+1e-10]
        chunks.append(points);have+=len(points)
        if have>=count:return np.concatenate(chunks)[:count].astype(np.float32)
    raise ValueError('Insufficient surface intersection after bounded rejection sampling')


class ShapeSamples:
    def __init__(self,manifest,split,length,seed=42,verify=True):
        self.path=Path(manifest);self.manifest=json.loads(self.path.read_text())
        if self.manifest['radius_um']!=RADIUS_UM or self.manifest['points_per_model_input']!=POINTS:
            raise ValueError('Full SHAPE radius and point count required')
        all_rows=self.manifest['rows'];ids=[r['id'] for r in all_rows]
        if len(ids)!=len(set(ids)):raise ValueError('Instance identity appears in multiple rows/splits')
        self.rows=[r for r in all_rows if r['split']==split and r['full_context_centers']>0]
        if len(self.rows)<2:raise ValueError('Need at least two full-context instances per split')
        self.meshes=[]
        for row in self.rows:
            path=Path(row['path']);path=path if path.is_absolute() else self.path.parent/path
            if verify:
                import hashlib
                if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Mesh hash mismatch')
            with np.load(path,allow_pickle=False) as f:
                self.meshes.append(tuple(f[k].copy() for k in ['vertices_um','faces','centers']))
        self.length=length;self.seed=seed
    def __len__(self):return self.length
    def __getitem__(self,index):
        rng=np.random.default_rng(self.seed+int(index));first=int(rng.integers(len(self.meshes)))
        v,f,c=self.meshes[first];center=v[rng.choice(c)].copy()
        # Balanced fixed blocks, shuffled so DDP rank strides do not isolate classes.
        label=int(np.random.default_rng(self.seed+int(index)//64).permutation(64)[int(index)%64]%2)
        if not label:
            second=int(rng.choice([i for i in range(len(self.meshes)) if i!=first]))
            other,of,oc=self.meshes[second];anchor=other[rng.choice(oc)]
            # Rigid false join at the first instance surface, with submicron jitter.
            shifted=other-anchor+center+rng.normal(0,.15,3)
            f=np.concatenate([f,of+len(v)]);v=np.concatenate([v,shifted])
        points=sample_surface(v,f,center,rng)
        # Isotropic rotations preserve physical geometry and context radius.
        q,_=np.linalg.qr(rng.normal(size=(3,3)));q[:,0]*=np.linalg.det(q)
        points=(points@q).astype(np.float32)
        return points,np.float32(label)
