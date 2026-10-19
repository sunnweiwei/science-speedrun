"""Bounded indexed meshes and explicit reconstruction/observation coverage."""
from collections import OrderedDict
import numpy as np
from skimage.measure import marching_cubes


class MeshCache:
    """Bound retained mesh bytes; live motif and extraction buffers are separate.

    Eviction only causes deterministic remeshing. It never changes a mesh,
    truncates a motif, samples fewer points or changes random seeds.
    """
    def __init__(self,max_bytes):
        if type(max_bytes) is not int or max_bytes<0:raise ValueError('Nonnegative mesh cache byte limit required')
        self.max_bytes=max_bytes;self.items=OrderedDict();self.bytes=0
        self.hits=0;self.misses=0;self.peak_bytes=0;self.largest_mesh_bytes=0

    def get(self,sid,loader):
        if sid in self.items:
            self.hits+=1;value,size=self.items.pop(sid);self.items[sid]=(value,size);return value
        self.misses+=1;value=loader(sid);size=sum(a.nbytes for a in value)
        self.largest_mesh_bytes=max(self.largest_mesh_bytes,size)
        if size<=self.max_bytes:
            while self.bytes+size>self.max_bytes:
                _,(_,removed)=self.items.popitem(last=False);self.bytes-=removed
            self.items[sid]=(value,size);self.bytes+=size;self.peak_bytes=max(self.peak_bytes,self.bytes)
        return value

    def statistics(self):
        return dict(limit_bytes=self.max_bytes,retained_bytes=self.bytes,peak_retained_bytes=self.peak_bytes,
            largest_mesh_bytes=self.largest_mesh_bytes,hits=self.hits,misses=self.misses,
            scope='Retained numpy mesh buffers only; live motif, extraction, sampling, model and array pages excluded')


def indexed_mesh(seg,sid,spacing,row):
    bounds=np.asarray(row['bbox_zyx']);shape=np.asarray(seg.shape)
    if (row['id']!=sid or bounds.shape!=(3,2) or bounds.dtype.kind not in 'iu'
        or (bounds[:,0]<0).any() or (bounds[:,1]>shape).any() or (bounds[:,1]<=bounds[:,0]).any()):
        raise ValueError('Invalid instance bounding box')
    lo,hi=bounds[:,0],bounds[:,1];mask=seg[tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))]==sid
    count=int(np.count_nonzero(mask))
    if count!=row['size'] or not count:raise ValueError('Instance index count disagrees with predicted labels')
    # Integer axis marginals give the same centroid without an N x 3 int64
    # coordinate array. The full mask and marching-cubes buffers still count
    # toward the job's memory requirement; this is not a streaming mesher.
    centroid=np.array([np.dot(np.arange(mask.shape[axis],dtype=np.int64),
        mask.sum(axis=tuple(i for i in range(3) if i!=axis),dtype=np.int64)) for axis in range(3)])/count+lo
    v,f,_,_=marching_cubes(np.pad(mask,1),.5,spacing=tuple(spacing))
    v+=(lo-.5)*spacing
    return v,f,centroid


def context_coverage(center,spacing,metadata,coverage,radius=10.):
    """center is ROI-local voxel index, spacing um; missing source planes retained."""
    c=(np.asarray(center)+.5)*spacing
    offset=(np.asarray(metadata['origin_nm'])-metadata['raw_origin_nm'])/(np.asarray(spacing)*1000)
    complete=True
    for tile in coverage['tiles']:
        if tile['status']=='complete':continue
        lo=(np.asarray(tile['start'])-offset)*spacing;hi=(np.asarray(tile['stop'])-offset)*spacing
        distance=np.maximum(np.maximum(lo-c,c-hi),0)
        if np.dot(distance,distance)<radius**2:complete=False;break
    image=coverage.get('image_observation')
    fraction=None
    if image is not None:
        count=metadata['shape'][0];z=(np.arange(count)+.5)*spacing[0]
        weight=np.maximum(0.,radius**2-(z-c[0])**2)
        unknown=np.asarray(image['unknown_output_z'],dtype=int)
        if (unknown<0).any() or (unknown>=count).any():raise ValueError('Image validity indices outside prediction volume')
        fraction=float(1-weight[unknown].sum()/weight.sum()) if weight.sum() else None
    return dict(prediction_context_complete=complete,image_observed_sphere_fraction=fraction,
        image_coverage_scope='detected full-plane filler only; other defects not excluded')
