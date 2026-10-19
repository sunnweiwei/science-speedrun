"""Score native-grid predictions at the same frozen physical reference nodes.

No interpolation of labels or reference paths. Both models use the same pinned
funlib ERL and zero boundary margin. This is separate from native NISB scoring.
"""
import argparse
import json
from pathlib import Path
import sys

import networkx as nx
import numpy as np

from evaluate_erl import evaluate
from topology import counts, sample
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import external, sha256, write_json

EVALUATOR_SHA = '48f03d7c28b77853a431c0f163e9118befd057777a332be13cf500f72feb7f3c'


def grid(metadata, array_shape=None):
    axes=metadata.get('axes')
    if axes not in ('xyz','zyx') or metadata.get('coordinate_convention')!='voxel_outer_corner':
        raise ValueError('Explicit XYZ or ZYX outer-corner grid required')
    origin=np.asarray(metadata.get('origin_nm'),dtype=float)
    spacing=np.asarray(metadata.get('voxel_size_nm'),dtype=float)
    shape=np.asarray(metadata.get('shape'))
    if (origin.shape!=(3,) or spacing.shape!=(3,) or shape.shape!=(3,) or
        not np.isfinite(origin).all() or not np.isfinite(spacing).all() or
        (spacing<=0).any() or shape.dtype.kind not in 'iu' or (shape<=0).any()):
        raise ValueError('Finite origin, positive spacing and integer shape required')
    if array_shape is not None and tuple(shape)!=tuple(array_shape):
        raise ValueError('Prediction array and grid shape differ')
    order=[axes.index(axis) for axis in 'zyx']
    return origin[order],spacing[order],shape[order],order


def reference_bounds(metadata):
    origin,spacing,shape,_=grid(metadata)
    fields=('scored_core_origin_nm','scored_core_shape')
    if any(k in metadata for k in fields):
        if not all(k in metadata for k in fields) or metadata['axes']!='zyx':
            raise ValueError('Explicit complete ZYX scored-core geometry required')
        core=dict(axes='zyx',coordinate_convention='voxel_outer_corner',
            origin_nm=metadata[fields[0]],shape=metadata[fields[1]],voxel_size_nm=spacing.tolist())
        low,_,core_shape,_=grid(core);high=low+spacing*core_shape
        if (low<origin).any() or (high>origin+spacing*shape).any():
            raise ValueError('Scored core leaves reference extent')
        return low,high
    return origin,origin+spacing*shape


def assignments(segmentation,prediction_metadata,reference_metadata,coordinates):
    if segmentation.ndim!=3 or segmentation.dtype.kind!='u':
        raise ValueError('Unsigned 3D instance labels required; zero is omission')
    origin,spacing,shape,order=grid(prediction_metadata,segmentation.shape)
    low,high=reference_bounds(reference_metadata)
    if (origin>low).any() or (origin+spacing*shape<high).any():
        raise ValueError('Prediction must cover the entire frozen scoring region')
    coordinates=np.asarray(coordinates,dtype=float)
    if (coordinates.ndim!=2 or coordinates.shape[1]!=3 or not len(coordinates) or
        not np.isfinite(coordinates).all() or (coordinates<low).any() or (coordinates>=high).any()):
        raise ValueError('All frozen reference nodes must lie inside the scoring region')
    return sample(segmentation.transpose(order),coordinates,origin,spacing)


def reference_graph(truth):
    if truth.get('axes')!='zyx' or truth.get('coordinate_units')!='nm':
        raise ValueError('Physical ZYX reference required')
    graph=nx.Graph()
    for node in truth['nodes']:
        n=node['id'];s=node['skeleton_id'];p=np.asarray(node['position_nm'],dtype=float)
        if (type(n)!=int or not 0<=n<=np.iinfo(np.uint64).max or n in graph or
            type(s)!=int or s<=0 or p.shape!=(3,) or not np.isfinite(p).all()):
            raise ValueError('Invalid or duplicate reference node')
        graph.add_node(n,skeleton_id=s,z=p[0],y=p[1],x=p[2])
    for u,v in truth['edges']:
        if type(u)!=int or type(v)!=int or u not in graph or v not in graph or u==v or graph.has_edge(u,v) or graph.nodes[u]['skeleton_id']!=graph.nodes[v]['skeleton_id']:
            raise ValueError('Invalid reference edge')
        graph.add_edge(u,v)
    return graph


def score(suite,prediction,output):
    suite=Path(suite);prediction=Path(prediction);output=external(output)
    frozen=json.loads((suite/'suite.json').read_text())
    if (frozen.get('schema')!='pathfinder-evaluation-suite/v1' or
        set(frozen['files'])!={'metadata.json','skeletons.json','run_length.py'} or
        frozen.get('boundary_margin_voxels')!=[0,0,0]):
        raise ValueError('Frozen zero-margin reference suite required')
    for name,digest in frozen['files'].items():
        if sha256(suite/name)!=digest:raise ValueError('Reference integrity mismatch: '+name)
    if sha256(suite/'run_length.py')!=EVALUATOR_SHA:raise ValueError('Unrecognized ERL implementation')
    truth=json.loads((suite/'skeletons.json').read_text())
    meta=json.loads((suite/'metadata.json').read_text())
    pred_meta=json.loads((prediction/'metadata.json').read_text())
    seg=np.load(prediction/'segmentation.npy',mmap_mode='r',allow_pickle=False)
    graph=reference_graph(truth);nodes=truth['nodes']
    labels=assignments(seg,pred_meta,meta,[n['position_nm'] for n in nodes])
    result=evaluate(graph,{n['id']:int(s) for n,s in zip(nodes,labels)},suite/'run_length.py')
    diagnostic=counts([n['skeleton_id'] for n in nodes],labels,set(map(int,labels))-{0})
    for key in ('split_errors','merge_errors','omitted_nodes','omitted_node_fraction'):
        result[key]=diagnostic[key]
    result.update(core_evaluator_adapter_sha256=result['adapter_sha256'],adapter_sha256=sha256(Path(__file__)),
        inputs=dict(skeletons=sha256(suite/'skeletons.json'),metadata=sha256(suite/'metadata.json'),
            segmentation=sha256(prediction/'segmentation.npy')),
        topology_sha256=sha256(Path(__file__).with_name('topology.py')),
        suite_sha256=sha256(suite/'suite.json'),prediction_grid_sha256=sha256(prediction/'metadata.json'),
        prediction_grid=pred_meta,boundary_margin_voxels=[0,0,0],sampled_nodes=len(nodes),
        sampling='Native-grid floor at identical physical reference positions; no label interpolation',
        scope='Same frozen physical scoring region and skeletons; not native NISB or PATHFINDER paper parity')
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'node-assignments.npz',node_id=np.asarray([n['id'] for n in nodes],dtype=np.uint64),prediction_id=labels)
    result['node_assignments_sha256']=sha256(output/'node-assignments.npz')
    write_json(output/'metrics.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('suite','prediction','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();print(json.dumps(score(a.suite,a.prediction,a.output),indent=2))

if __name__=='__main__':main()
