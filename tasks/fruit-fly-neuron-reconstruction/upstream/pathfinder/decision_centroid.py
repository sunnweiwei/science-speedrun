"""Stream the candidate table after a full official watershed expansion.

The seven neighborhood offsets, mean EDT distance, and nearest-centroid rule
follow google-ffn/ffn/utils/decision_point.py (Copyright 2020-2023 Google Inc.,
Apache License 2.0). Watershed expansion calls the pinned Google implementation.

Each neighborhood anchor belongs to exactly one chunk. Equal-distance points
from different offsets remain repeated, as in the author's concatenated table.
Two passes avoid retaining all candidate coordinates. Expansion itself is
global: a finite EDT threshold does not justify independent local watershed
runs, which failed the official oracle. Global expansion needs a separately
declared large memory allocation; this module never silently substitutes a
local approximation.
"""
import itertools
import numpy as np

OFFSETS=tuple(p for p in itertools.product((0,-1),repeat=3) if p!=(0,0,0))


def groups(expanded_volume,distance_volume,roi,*,tile,chunk_rank=0,chunk_shards=1):
    lo=np.asarray(roi[::2]);hi=np.asarray(roi[1::2])
    if type(chunk_shards) is not int or type(chunk_rank) is not int or not 0<=chunk_rank<chunk_shards:
        raise ValueError('Invalid chunk ownership')
    for ordinal,start in enumerate(itertools.product(*(range(a,b,tile) for a,b in zip(lo,hi)))):
        if ordinal%chunk_shards!=chunk_rank:continue
        start=np.array(start);stop=np.minimum(start+tile,hi);scan_stop=np.minimum(stop+1,hi)
        crop=tuple(slice(int(a),int(b)) for a,b in zip(start,scan_stop))
        expanded=expanded_volume[crop];edt=distance_volume[crop]
        for offset_index,offset in enumerate(OFFSETS):
            first=tuple(slice(None) if o==0 else slice(0,-1) for o in offset)
            second=tuple(slice(None) if o==0 else slice(1,None) for o in offset)
            a=expanded[first];b=expanded[second]
            touching=(a>0)&(b>0)&(a!=b)
            positions=np.array(np.where(touching)).T
            if not len(positions):continue
            # +1 scan halo supplies neighbor values, never additional anchors.
            owned=np.all(positions<stop-start,axis=1)
            distance=((edt[first][touching]+edt[second][touching])/2)[owned]
            left=a[touching][owned];right=b[touching][owned]
            positions=positions[owned]+start-lo
            if not len(positions):continue
            small=np.minimum(left,right);large=np.maximum(left,right)
            order=np.lexsort((large,small));small=small[order];large=large[order]
            positions=positions[order];distance=distance[order]
            edges=np.r_[0,np.flatnonzero((small[1:]!=small[:-1])|(large[1:]!=large[:-1]))+1,len(small)]
            for beg,end in zip(edges[:-1],edges[1:]):
                distances=distance[beg:end];minimum=float(distances.min())
                yield ((int(small[beg]),int(large[beg])),minimum,
                       positions[beg:end][distances==minimum],offset_index)


def reduce_centroid(group_factory,*,max_pairs):
    """Retain every global-minimum vote; match author's offset/ZYX tie order."""
    stats={}
    for pair,distance,points,offset in group_factory():
        count=len(points);total=points.sum(axis=0,dtype=np.int64)
        old=stats.get(pair)
        if old is None:
            if len(stats)>=max_pairs:raise ValueError('Decision pair limit exceeded; no partial result accepted')
            stats[pair]=[distance,count,total]
        elif distance<old[0]:stats[pair]=[distance,count,total]
        elif distance==old[0]:old[1]+=count;old[2]+=total
    selected={}
    for pair,distance,points,offset in group_factory():
        minimum,count,total=stats[pair]
        if distance!=minimum:continue
        # Author calculates XYZ means relative to the complete selected ROI.
        mean=total[::-1]/count
        squared=np.sum(np.square(points[:,::-1]-mean),axis=1)
        index=int(np.argmin(squared));point=tuple(map(int,points[index]))
        key=(float(squared[index]),offset,*point)
        if pair not in selected or key<selected[pair][0]:selected[pair]=(key,point)
    if set(selected)!=set(stats):raise ValueError('Decision groups changed between passes')
    return [(pair,stats[pair][0],selected[pair][1]) for pair in sorted(stats)]


def memory_preflight(shape,roi,*,memory_gib,tile=128,max_pairs=1_000_000):
    """Check full-volume planning RAM before GPU inference or watershed starts."""
    import math
    if (len(shape)!=3 or any(type(v) is not int or v<1 for v in shape)
        or len(roi)!=6 or any(type(v) is not int for v in roi)
        or type(tile) is not int or tile<1 or type(max_pairs) is not int or max_pairs<1
        or not np.isfinite(memory_gib) or memory_gib<=0):
        raise ValueError('Explicit positive bounded decision geometry required')
    lo=np.asarray(roi[::2]);hi=np.asarray(roi[1::2])
    if (lo<50).any() or (hi>np.array(shape)-50).any() or (hi<=lo).any():
        raise ValueError('Decision ROI requires 50-voxel resegmentation context')
    if 7*math.prod(map(int,hi-lo))*int(max(hi-lo))>=2**53:
        raise ValueError('Decision ROI exceeds exact centroid coordinate-sum bound')
    estimated_bytes=128*math.prod(shape)+512*min(tile+1,int(max(hi-lo)))**3+1024*max_pairs+2**30
    if estimated_bytes>memory_gib*2**30:
        raise ValueError('Global official watershed memory estimate exceeds declared allocation')
    return estimated_bytes


def generate_global(base,roi,expand,*,memory_gib,tile=128,max_distance_nm=64,max_pairs=1_000_000):
    if (base.ndim!=3 or base.dtype.kind!='u' or type(tile) is not int or tile<1
        or not np.isfinite(max_distance_nm) or max_distance_nm<=0
        or type(max_pairs) is not int or max_pairs<1
        or not np.isfinite(memory_gib) or memory_gib<=0
        or len(roi)!=6 or any(type(v) is not int for v in roi)):
        raise ValueError('Unsigned volume and explicit positive bounded decision geometry required')
    lo=np.asarray(roi[::2]);hi=np.asarray(roi[1::2])
    if (lo<50).any() or (hi>np.array(base.shape)-50).any() or (hi<=lo).any():
        raise ValueError('Decision ROI requires 50-voxel resegmentation context')
    # Coordinate sums must be exact int64 and exact float64 before division.
    if 7*int(np.prod(hi-lo))*int(max(hi-lo))>=2**53:
        raise ValueError('Decision ROI exceeds exact centroid coordinate-sum bound')
    # A conservative planning estimate, not a measured library peak. Include
    # whole-volume relabeling, EDT/watershed workspaces and candidate chunk RAM.
    estimated_bytes=memory_preflight(list(base.shape),roi,memory_gib=memory_gib,tile=tile,max_pairs=max_pairs)
    expanded,distance=expand(base,[16]*3,max_distance=max_distance_nm)
    if expanded.shape!=base.shape or distance.shape!=base.shape:
        raise ValueError('Official expansion geometry differs from base')
    rows=reduce_centroid(lambda:groups(expanded,distance,roi,tile=tile),max_pairs=max_pairs)
    return dict(decisions=[dict(pair=list(pair),distance_nm=distance,center_zyx=(np.array(point)+lo).tolist())
                           for pair,distance,point in rows],
                chunks=int(np.prod((hi-lo+tile-1)//tile)),roi_zyx=list(roi),tile=tile,
                max_distance_nm=max_distance_nm,estimated_memory_bytes=int(estimated_bytes),declared_memory_gib=memory_gib,
                max_pairs=max_pairs,passes=2,
                cross_chunk_tie_rule='All global-minimum points, nearest global XYZ centroid; first official offset then ZYX anchor',
                expansion_scope='Pinned official watershed on the entire input volume; only candidate-table reduction is chunked')
