"""Google PolicyPeaks seeding on real image context, with explicit voxel spacing.

Adapted from google/ffn (Apache-2.0), pinned b8df2d96. The caller supplies the
image halo and excludes its border via margin, matching BaseSeedPolicy.
"""
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
import edt


def policy_peaks(image, margin=16, voxel_size=(16.,16.,16.), exclusion=None):
    if image.ndim!=3 or margin<0 or any(n<=2*margin for n in image.shape):
        raise ValueError('Three-dimensional image with real seed context required')
    if len(voxel_size)!=3 or any(not np.isfinite(v) or v<=0 for v in voxel_size):
        raise ValueError('Positive ZYX voxel spacing required')
    if exclusion is not None and (exclusion.shape!=image.shape or exclusion.dtype!=np.bool_):
        raise ValueError('Aligned boolean exclusion mask required')
    edges=ndimage.generic_gradient_magnitude(image.astype(np.float32),ndimage.sobel)
    adaptive=np.zeros(edges.shape,np.float32)
    ndimage.gaussian_filter(edges,49./6.,output=adaptive,mode='reflect')
    boundary=edges>adaptive
    if exclusion is not None:boundary[exclusion]=True
    if boundary.all():return np.empty((0,3),dtype=np.int64)
    distances=edt.edt(1-boundary,anisotropy=voxel_size).astype(np.float32)
    if exclusion is not None:distances[exclusion]=-1
    distances[~np.isfinite(distances)]=-1
    noise=np.random.RandomState(42).rand(*distances.shape)*1e-4
    points=peak_local_max(distances+noise,min_distance=3,threshold_abs=0,threshold_rel=0)
    points=np.asarray(sorted(map(tuple,points)),dtype=np.int64).reshape(-1,3)
    return points[((points-margin>=0)&(points+margin<image.shape)).all(axis=1)]


def seed_is_clear(segmentation,seed,boundary_distance=1):
    """Apply Canvas.segment_all's existing-segment exclusion around a seed."""
    if boundary_distance<0:raise ValueError('Nonnegative boundary distance required')
    region=tuple(slice(max(0,int(p)-boundary_distance),min(n,int(p)+boundary_distance+1))
                 for p,n in zip(seed,segmentation.shape))
    return not np.any(segmentation[region]>0)
