"""Exact two-pass instance statistics with bounded dense label addressing.

No labels are filtered; numeric table records match the legacy index schema.
"""
import numpy as np
DTYPE=np.dtype([('size','<u8'),('lo','<i4',(3,)),('hi','<i4',(3,)),('center','<u8')])


def build_table(seg,depth=8,*,max_label,max_instances):
    if seg.ndim!=3 or seg.dtype.kind!='u' or type(depth) is not int or depth<1:
        raise ValueError('Unsigned ZYX volume and positive slab depth required')
    if any(type(x) is not int or x<1 for x in (max_label,max_instances)):
        raise ValueError('Explicit positive label and instance bounds required')
    if max(seg.shape)>np.iinfo(np.int32).max or seg.size*max(seg.shape)>2**53:
        raise ValueError('Coordinate sums must remain exactly representable')
    highest=int(seg.max(initial=0))
    if highest>max_label:raise ValueError('Dense label bound exceeded')
    n=highest+1;table=np.zeros(n,dtype=DTYPE)
    table['lo'][:]=np.iinfo(np.int32).max
    totals=np.zeros((n,3),np.float64)
    for z in range(0,len(seg),depth):
        slab=np.asarray(seg[z:z+depth]);coords=np.nonzero(slab)
        ids=slab[coords].astype(np.intp,copy=False)
        np.add.at(table['size'],ids,1)
        for axis,c in enumerate(coords):
            v=c+(z if axis==0 else 0)
            np.minimum.at(table['lo'][:,axis],ids,v)
            np.maximum.at(table['hi'][:,axis],ids,v+1)
            np.add.at(totals[:,axis],ids,v)
    active=table['size']>0
    if np.count_nonzero(active)>max_instances:raise ValueError('Instance bound exceeded; table incomplete')
    np.divide(totals,table['size'][:,None],out=totals,where=active[:,None])
    best=np.full(n,np.inf);table['center'][:]=np.iinfo(np.uint64).max
    plane=seg.shape[1]*seg.shape[2]
    for z in range(0,len(seg),depth):
        slab=np.asarray(seg[z:z+depth]);coords=np.nonzero(slab)
        ids=slab[coords].astype(np.intp,copy=False)
        if not ids.size:continue
        xyz=np.column_stack(coords);xyz[:,0]+=z
        distance=((xyz-totals[ids])**2).sum(axis=1)
        unique=np.unique(ids);previous=best[unique].copy()
        np.minimum.at(best,ids,distance)
        changed=unique[best[unique]<previous]
        table['center'][changed]=np.iinfo(np.uint64).max
        winners=distance==best[ids]
        flat=xyz[:,0]*plane+xyz[:,1]*seg.shape[2]+xyz[:,2]
        np.minimum.at(table['center'],ids[winners],flat[winners].astype(np.uint64))
    table['lo'][~active]=0;table['hi'][~active]=0;table['center'][~active]=0
    return table


def records(table,shape):
    """Yield ascending-ID legacy records without materializing a list."""
    for sid in np.flatnonzero(table['size']):
        row=table[sid];center=np.unravel_index(int(row['center']),shape)
        yield dict(id=int(sid),size=int(row['size']),
            bbox_zyx=[[int(a),int(b)] for a,b in zip(row['lo'],row['hi'])],
            center_zyx=list(map(int,center)))
