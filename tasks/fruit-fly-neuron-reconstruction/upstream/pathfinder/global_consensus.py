"""Bounded global intersection for uint32 stitched FFN segmentations.

Partition-equivalent to Google's split_segmentation_by_intersection, with
compact positive IDs rather than preserving the largest fragment's old ID.
Counts are global, including intersections spanning slab/window boundaries.
This kernel alone does not validate inference provenance or six-run coverage.
"""
import numpy as np


def intersect_inplace(a,b,min_size,slab_depth=8,max_pairs=1_000_000,memory_gib=2):
    """Two-pass intersection; reject an excessive pair table before changing a.

    Both arrays must be C-contiguous uint32 ZYX volumes (memmaps supported).
    The background rule is asymmetric: (A>0,B=0) may survive, (A=0,B) cannot.
    Call sequentially in the official input order for multi-run consensus.
    Numeric IDs change; zero masks and equivalence classes match the author
    routine because its largest-overlap rule chooses IDs, not membership.
    """
    if a.ndim!=3 or a.shape!=b.shape or not a.size or a.dtype!=np.uint32 or b.dtype!=np.uint32 or not a.flags.c_contiguous or not b.flags.c_contiguous:
        raise ValueError('Matching nonempty C-contiguous uint32 3D arrays required')
    if not a.flags.writeable:raise ValueError('Writable first array required')
    if np.shares_memory(a,b) or (isinstance(a,np.memmap) and isinstance(b,np.memmap) and a.filename==b.filename):
        raise ValueError('Independent input storage required for in-place scanning')
    if any(type(v) is not int for v in (min_size,slab_depth,max_pairs)) or min_size<0 or slab_depth<1 or max_pairs<1 or not np.isfinite(memory_gib) or memory_gib<=0:
        raise ValueError('Finite positive bounds and nonnegative minimum size required')
    # Conservative allowance for numpy sort temporaries and Python count table.
    estimated=128*min(slab_depth,a.shape[0])*a.shape[1]*a.shape[2]+512*max_pairs
    if estimated>memory_gib*2**30:raise ValueError('Declared memory bound too small')
    counts={}
    def packed(z):
        return np.asarray(a[z:z+slab_depth],dtype=np.uint64)|(np.asarray(b[z:z+slab_depth],dtype=np.uint64)<<32)
    for z in range(0,a.shape[0],slab_depth):
        keys,number=np.unique(packed(z),return_counts=True)
        for key,count in zip(keys,number):
            key=int(key)
            if key not in counts and len(counts)>=max_pairs:raise ValueError('Global pair table exceeds declared bound; input unchanged')
            counts[key]=counts.get(key,0)+int(count)
    keys=np.fromiter(sorted(counts),dtype=np.uint64,count=len(counts))
    labels=np.zeros(len(keys),np.uint32);kept=0
    for index,key in enumerate(keys):
        if (int(key)&0xFFFFFFFF)!=0 and counts[int(key)]>=min_size:
            kept+=1
            if kept>np.iinfo(np.uint32).max:raise ValueError('Compact label overflow; input unchanged')
            labels[index]=kept
    pair_count=len(counts);del counts
    for z in range(0,a.shape[0],slab_depth):
        block=packed(z)
        a[z:z+slab_depth]=labels[np.searchsorted(keys,block)]
    return dict(global_pairs=pair_count,positive_components=kept,slab_depth=slab_depth,
                conservative_memory_bytes=estimated,partition_equivalent=True,
                numeric_ids_preserved=False,min_size=min_size)
