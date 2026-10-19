"""Stream predicted instance sizes, bounding boxes and representative centers.

No GT input. Bounding boxes allow large-volume consumers to avoid rescanning the
entire array for every instance. Coordinates are local ZYX voxel indices.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import external,sha256,write_json


def build_index(seg,depth=16,max_instances=None):
    if seg.ndim!=3 or seg.dtype.kind!='u' or depth<1:raise ValueError('Unsigned ZYX segmentation and positive slab depth required')
    if max_instances is not None and (type(max_instances) is not int or max_instances<1):raise ValueError('Positive instance limit required')
    result={}
    for z in range(0,seg.shape[0],depth):
        slab=np.asarray(seg[z:z+depth]);coords=np.nonzero(slab)
        if not len(coords[0]):continue
        ids,inv,counts=np.unique(slab[coords],return_inverse=True,return_counts=True)
        if max_instances is not None and len(ids)>max_instances:raise ValueError('Instance limit exceeded; index is incomplete')
        lo=np.full((len(ids),3),np.iinfo(np.int64).max,np.int64);hi=np.zeros((len(ids),3),np.int64)
        sums=np.zeros((len(ids),3),np.float64)
        for axis,c in enumerate(coords):
            values=c+(z if axis==0 else 0)
            np.minimum.at(lo[:,axis],inv,values);np.maximum.at(hi[:,axis],inv,values)
            sums[:,axis]=np.bincount(inv,weights=values,minlength=len(ids))
        for i,sid in enumerate(ids):
            key=int(sid)
            if key not in result:
                if max_instances is not None and len(result)>=max_instances:raise ValueError('Instance limit exceeded; index is incomplete')
                result[key]=[int(counts[i]),lo[i].copy(),hi[i].copy()+1,sums[i].copy()]
            else:
                row=result[key];row[0]+=int(counts[i]);row[1]=np.minimum(row[1],lo[i]);row[2]=np.maximum(row[2],hi[i]+1);row[3]+=sums[i]
        del slab,coords,ids,inv,counts,lo,hi,sums,values,c
    records=[]
    for sid,(count,lo,hi,total) in sorted(result.items()):
        # Search only the object's box; choose an actual labeled voxel, not an
        # arbitrary centroid that might lie outside a nonconvex instance.
        # Bound coordinate storage even when one stitched object spans the full
        # volume. Ascending Z and strict improvement preserve the original
        # full-box argmin's lexicographic tie choice exactly.
        center=None;best=float('inf')
        for z in range(int(lo[0]),int(hi[0]),depth):
            base=np.array([z,lo[1],lo[2]])
            box=seg[z:min(z+depth,int(hi[0])),int(lo[1]):int(hi[1]),int(lo[2]):int(hi[2])]
            xyz=np.argwhere(box==sid)+base
            if not len(xyz):continue
            distances=((xyz-total/count)**2).sum(1);candidate=int(np.argmin(distances))
            if distances[candidate]<best:best=float(distances[candidate]);center=xyz[candidate].copy()
        records.append(dict(id=sid,size=count,bbox_zyx=[[int(a),int(b)] for a,b in zip(lo,hi)],center_zyx=center.tolist()))
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--segmentation',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--slab-depth',default=16,type=int);a=p.parse_args()
    seg=np.load(a.segmentation,mmap_mode='r',allow_pickle=False)
    result=dict(schema='predicted-instance-index/v1',shape=list(seg.shape),axes='zyx',records=build_index(seg,a.slab_depth),
        segmentation_sha256=sha256(a.segmentation),ground_truth_used=False)
    write_json(external(a.output),result)

if __name__=='__main__':main()
