"""Prediction-only overlap matching for an independent FFN stitching adapter.

The 2018 FFN author manuscript specifies mutual maximum overlap on a shared
mid-overlap plane. It does not specify tie/background details; this adapter
rejects tied maxima and lets background compete, then excludes zero-ID pairs.
This module does not crop/relabel connected components or assemble a volume.
"""
import numpy as np


def mutual_maximum_pairs(a, b):
    """Return unique mutual maxima; label namespaces remain separate A and B."""
    a=np.asarray(a);b=np.asarray(b)
    if a.ndim!=2 or a.shape!=b.shape or a.size==0 or a.dtype.kind!='u' or b.dtype.kind!='u':
        raise ValueError('Matching nonempty unsigned label planes required')
    pairs,counts=np.unique(np.stack((a.ravel().astype(np.uint64),b.ravel().astype(np.uint64)),axis=1),axis=0,return_counts=True)
    def winners(column):
        groups={}
        for pair,count in zip(pairs,counts):
            label=int(pair[column]);other=int(pair[1-column]);count=int(count)
            previous=groups.get(label)
            if previous is None or count>previous[0]:groups[label]=(count,other,False)
            elif count==previous[0]:groups[label]=(count,previous[1],True)
        return {label:other for label,(_,other,tied) in groups.items() if not tied}
    forward=winners(0);backward=winners(1)
    matches=[]
    for (left,right),count in zip(pairs,counts):
        left,right=int(left),int(right)
        if left and right and forward.get(left)==right and backward.get(right)==left:
            matches.append(dict(a=left,b=right,overlap_voxels=int(count)))
    return dict(pairs=matches,policy='Unique mutual maximum; background competes but never merges; ties rejected',
                implementation='Independent implementation of paper matching criterion with explicit local tie/background choices',
                paper_exact=False,ground_truth_used=False)


def overlap_midplane(a,b,origin_a,origin_b,axis):
    """Sample identical voxel coordinates in two axis-neighbor predictions.

    Origins are ZYX voxel indices on the SAME grid, not nanometres. For even
    overlap width, the lower of the two central voxel planes is selected.
    Transverse support is their shared extent; core/component mapping is separate.
    """
    a=np.asarray(a);b=np.asarray(b);oa=np.asarray(origin_a);ob=np.asarray(origin_b)
    if (type(axis) is not int or axis not in (0,1,2) or a.ndim!=3 or b.ndim!=3 or
            a.dtype.kind!='u' or b.dtype.kind!='u' or not a.size or not b.size or
            oa.shape!=(3,) or ob.shape!=(3,) or oa.dtype.kind not in 'iu' or ob.dtype.kind not in 'iu'):
        raise ValueError('Unsigned 3D arrays, integer ZYX origins and axis required')
    # Python integers avoid overflow when comparing unsigned coordinates.
    oa=np.asarray([int(x) for x in oa],dtype=object);ob=np.asarray([int(x) for x in ob],dtype=object)
    if oa[axis]==ob[axis] or any(oa[i]!=ob[i] for i in range(3) if i!=axis):
        raise ValueError('Axis-neighbor origins required')
    lo=np.maximum(oa,ob);hi=np.minimum(oa+np.asarray(a.shape),ob+np.asarray(b.shape))
    if any(hi<=lo):raise ValueError('Predictions do not overlap')
    coordinate=int((lo[axis]+hi[axis]-1)//2)
    def section(origin):
        result=[slice(int(x-y),int(z-y)) for x,y,z in zip(lo,origin,hi)]
        result[axis]=coordinate-int(origin[axis]);return tuple(result)
    return a[section(oa)],b[section(ob)],dict(axis=axis,coordinate_voxels=coordinate,
        overlap_origin_voxels=[int(x) for x in lo],overlap_shape=[int(x) for x in hi-lo],
        even_width_policy='Lower central voxel plane')


def component_interface_plane(prediction,components,prediction_origin,core_origin,axis,coordinate,transverse_bounds):
    """Map a shared plane to cropped-core components without rejoining them.

    The plane may be in the core or one voxel outside its face. Outside voxels
    map only through a face-adjacent core voxel with the SAME original label.
    Thus two disconnected core components never regain a shared original ID.
    This explicit boundary convention needs real-volume validation.
    """
    prediction=np.asarray(prediction);components=np.asarray(components)
    po=np.asarray(prediction_origin);co=np.asarray(core_origin);bounds=np.asarray(transverse_bounds)
    if (prediction.ndim!=3 or components.ndim!=3 or prediction.dtype.kind!='u' or
            components.dtype.kind!='u' or not prediction.size or not components.size or
            po.shape!=(3,) or co.shape!=(3,) or bounds.shape!=(3,2) or
            po.dtype.kind not in 'iu' or co.dtype.kind not in 'iu' or bounds.dtype.kind not in 'iu' or
            type(axis) is not int or axis not in (0,1,2) or type(coordinate) is not int):
        raise ValueError('Unsigned volumes and integer grid coordinates required')
    po=[int(x) for x in po];co=[int(x) for x in co]
    stop=[x+n for x,n in zip(co,components.shape)]
    if any(x<y or z>y+n for x,y,z,n in zip(co,po,stop,prediction.shape)):
        raise ValueError('Core leaves prediction')
    if not po[axis]<=coordinate<po[axis]+prediction.shape[axis]:raise ValueError('Plane leaves prediction')
    if not co[axis]-1<=coordinate<=stop[axis]:raise ValueError('Plane must touch core')
    nearest=min(max(coordinate,co[axis]),stop[axis]-1)
    slices=[]
    for i in range(3):
        low,high=map(int,bounds[i])
        if i!=axis and not co[i]<=low<high<=stop[i]:raise ValueError('Transverse plane leaves core')
        slices.append(slice(low,high))
    def section(origin,at):
        return tuple(at-origin[i] if i==axis else slice(slices[i].start-origin[i],slices[i].stop-origin[i]) for i in range(3))
    mapped=components[section(co,nearest)].copy()
    outside=prediction[section(po,coordinate)];inside=prediction[section(po,nearest)]
    if coordinate!=nearest:mapped[(outside==0)|(outside!=inside)]=0
    return mapped
