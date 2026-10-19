"""Exact SciPy fragment EDT with redundant exterior background removed.

Keep the full foreground bounding box plus one existing background voxel on
all available sides. Every omitted voxel is background; a nearest background
site outside this box is no closer than the retained intervening background.
Original array boundaries are never padded with invented background. Full-box
and all-foreground cases use the original SciPy call unchanged.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt


def fragment_distance(mask):
    if mask.ndim!=3 or mask.dtype!=np.bool_:raise ValueError('3D boolean mask required')
    bounds=[]
    for axis in range(3):
        occupied=np.flatnonzero(mask.any(axis=tuple(i for i in range(3) if i!=axis)))
        if not len(occupied):return np.zeros(mask.shape,np.float64)
        bounds.append(slice(max(0,int(occupied[0])-1),min(mask.shape[axis],int(occupied[-1])+2)))
    if all(s.start==0 and s.stop==n for s,n in zip(bounds,mask.shape)):
        return distance_transform_edt(mask)
    result=np.zeros(mask.shape,np.float64);crop=tuple(bounds)
    result[crop]=distance_transform_edt(mask[crop])
    return result
