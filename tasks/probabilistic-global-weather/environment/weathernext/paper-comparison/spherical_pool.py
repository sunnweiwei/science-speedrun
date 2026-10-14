"""Spherical neighborhood pooling following GenCast Appendix E.6.

Build geometry using the pinned WeatherNext icosahedral mesh; apply on CPU.
Pool radii are mean mesh-edge geodesic lengths. Scored centers carry spherical
Voronoi areas. Pressure-level panels use a 1-degree grid and omit k=7.
"""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree, SphericalVoronoi
from scipy.sparse import csr_matrix, save_npz, load_npz


def grid_geometry(latitude, longitude):
    latitude = np.asarray(latitude, dtype=np.float64)
    longitude = np.asarray(longitude, dtype=np.float64)
    lat, lon = np.meshgrid(np.deg2rad(latitude), np.deg2rad(longitude), indexing='ij')
    points = np.stack([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)], axis=-1).reshape(-1,3)
    edges = np.concatenate(([-90.], (latitude[:-1]+latitude[1:])/2, [90.]))
    weights = np.repeat(np.diff(np.sin(np.deg2rad(edges))) * 2*np.pi/len(longitude), len(longitude))
    return points, weights


def geometry(vertices, faces, latitude, longitude):
    vertices = np.asarray(vertices, dtype=np.float64)
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    edges = np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    edges = np.unique(np.sort(edges,axis=1),axis=0)
    radius = np.arccos(np.clip(np.sum(vertices[edges[:,0]]*vertices[edges[:,1]],axis=1),-1,1)).mean()
    points, weights = grid_geometry(latitude, longitude)
    neighborhoods = cKDTree(points).query_ball_point(vertices, 2*np.sin(radius/2), workers=4)
    lengths = np.asarray([len(v) for v in neighborhoods])
    if not np.all(lengths): raise ValueError('Pooling region contains no grid cells')
    indices = np.concatenate(neighborhoods)
    indptr = np.concatenate(([0],np.cumsum(lengths)))
    matrix = csr_matrix((weights[indices],indices,indptr),shape=(len(vertices),len(points)))
    areas = SphericalVoronoi(vertices, radius=1, center=[0,0,0]).calculate_areas()
    np.testing.assert_allclose(areas.sum(),4*np.pi,rtol=1e-6)
    return matrix, areas, radius


def apply_pool(prediction, truth, matrix):
    """Shared truth-only mask; returns mean/max predictions and truth per center."""
    prediction = np.asarray(prediction).reshape(len(prediction),-1)
    truth = np.asarray(truth).reshape(-1)
    if not np.isfinite(prediction).all() or np.isinf(truth).any(): raise ValueError('Invalid panel')
    valid = np.isfinite(truth)
    masked = matrix.multiply(valid[None,:]).tocsr()
    totals = np.asarray(masked.sum(axis=1)).ravel()
    if np.any(totals <= 0): raise ValueError('Empty truth pooling region')
    average_p = (masked @ prediction.T).T / totals
    average_t = (masked @ np.nan_to_num(truth)) / totals
    maximum_p = np.empty((len(prediction),matrix.shape[0]))
    maximum_t = np.empty(matrix.shape[0])
    for center in range(matrix.shape[0]):
        indices = matrix.indices[matrix.indptr[center]:matrix.indptr[center+1]]
        indices = indices[valid[indices]]
        maximum_p[:,center] = prediction[:,indices].max(axis=1)
        maximum_t[center] = truth[indices].max()
    return {'average':(average_p,average_t), 'maximum':(maximum_p,maximum_t)}


def empirical_scores(prediction, truth, weights):
    """Biased empirical CRPS and ensemble-mean MSE with explicit area weights."""
    prediction = np.asarray(prediction,dtype=np.float64)
    truth = np.asarray(truth,dtype=np.float64)
    weights = np.asarray(weights,dtype=np.float64)
    if prediction.ndim != 2 or prediction.shape[1:] != truth.shape or truth.shape != weights.shape:
        raise ValueError('Wrong panel dimensions')
    if not all(np.isfinite(x).all() for x in (prediction,truth,weights)) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError('Invalid score values/weights')
    n = len(prediction)
    if n < 2: raise ValueError('Ensemble requires two members')
    ordered = np.sort(prediction,axis=0)
    pair_term = np.sum((2*np.arange(1,n+1)-n-1)[:,None]*ordered,axis=0)/(n*n)
    crps = np.abs(prediction-truth).mean(axis=0)-pair_term
    return dict(crps=float(np.average(crps,weights=weights)),
        mse=float(np.average((prediction.mean(axis=0)-truth)**2,weights=weights)),
        ensemble_variance_ddof0=float(np.average(prediction.var(axis=0,ddof=0),weights=weights)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',required=True,type=Path)
    args = ap.parse_args()
    from weathernext.utils import icosahedral_mesh
    meshes = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(splits=7)
    args.output.mkdir(parents=True,exist_ok=True)
    for stride in (1,4):
        for k in range(2,8 if stride==1 else 7):
            stem = args.output/f'stride-{stride}-mesh-{k}'
            matrix, areas, radius = geometry(meshes[k].vertices,meshes[k].faces,
                np.linspace(-90,90,721)[::stride],np.arange(1440)[::stride]/4)
            save_npz(str(stem)+'.npz',matrix)
            np.savez(str(stem)+'-metadata.npz',area=areas,radius_radians=radius,
                diameter_km=2*radius*6371.,stride=stride,mesh_refinement=k)
            print(str(stem),2*radius*6371.,flush=True)


if __name__ == '__main__': main()
