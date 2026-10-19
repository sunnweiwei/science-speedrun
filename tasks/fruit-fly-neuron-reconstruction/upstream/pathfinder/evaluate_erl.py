"""Local funlib ERL/nERL from fixed input skeletons; not PATHFINDER parity.

Uses pinned primary implementation; node-support topology counts are separate.
"""
import argparse,importlib.util,json
from pathlib import Path
import networkx as nx
import numpy as np
from topology import sample,counts,interior_segments
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import external,sha256,write_json


def compact_ids(graph,lookup):
    """Preserve equality/zero while avoiding upstream mixed-int float coercion."""
    def integer(value,minimum):
        return isinstance(value,(int,np.integer)) and not isinstance(value,(bool,np.bool_)) and value>=minimum
    if set(lookup)!=set(graph):raise ValueError('Exactly one prediction label per skeleton node required')
    if not all(integer(v,0) for v in lookup.values()):raise ValueError('Nonnegative integer prediction IDs required')
    if not all(integer(d['skeleton_id'],1) for _,d in graph.nodes(data=True)):raise ValueError('Positive integer skeleton IDs required')
    pred={v:i+1 for i,v in enumerate(sorted({int(v) for v in lookup.values() if v!=0}))};pred[0]=0
    truth={v:i+1 for i,v in enumerate(sorted({int(d['skeleton_id']) for _,d in graph.nodes(data=True)}))}
    mapped=graph.copy()
    for _,data in mapped.nodes(data=True):data['skeleton_id']=truth[int(data['skeleton_id'])]
    return mapped,{n:pred[int(v)] for n,v in lookup.items()}


def evaluate(graph,lookup,source):
    graph,lookup=compact_ids(graph,lookup)
    spec=importlib.util.spec_from_file_location('pinned_run_length',source)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    lengths=mod.get_skeleton_lengths(graph,['z','y','x'],'skeleton_id',store_edge_length='length')
    if not lengths or sum(lengths.values())<=0:raise ValueError('Positive GT path length required')
    erl=mod.expected_run_length(graph,'skeleton_id','length',lookup,skeleton_lengths=lengths)
    maxerl=mod.expected_run_length(graph,'skeleton_id','length',{n:v['skeleton_id'] for n,v in graph.nodes(data=True)},skeleton_lengths=lengths)
    if maxerl<=0:raise ValueError('Invalid ERL normalizer')
    return dict(erl_nm=float(erl),maximum_erl_nm=float(maxerl),nerl=float(erl/maxerl),total_path_length_nm=float(sum(lengths.values())),
                evaluator_sha256=sha256(source),protocol='local-funlib-erl/v1',paper_protocol_verified=False,
                id_handling='Lossless independent positive-ID compaction; prediction0 remains omission',
                adapter_sha256=sha256(Path(__file__)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['skeletons','segmentation','metadata','evaluator','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prediction-contract',type=Path,help='Validate sealed prediction against the independently supplied metadata')
    p.add_argument('--boundary-margin-voxels',type=int,nargs=3,required=True);a=p.parse_args()
    truth=json.loads(a.skeletons.read_text());meta=json.loads(a.metadata.read_text())
    contract_check=None
    if a.prediction_contract:
        from prediction_contract import validate
        if a.segmentation.resolve()!=(a.prediction_contract/'segmentation.npy').resolve():raise ValueError('Scored volume differs from prediction contract')
        contract_check=validate(a.prediction_contract,meta)
    if truth['axes']!='zyx' or truth['coordinate_units']!='nm' or meta['axes']!='zyx':raise ValueError('Explicit physical ZYX input required')
    graph=nx.Graph()
    for n in truth['nodes']:
        if n['id'] in graph or n['skeleton_id']<=0:raise ValueError('Duplicate node or invalid skeleton ID')
        z,y,x=n['position_nm'];graph.add_node(n['id'],skeleton_id=n['skeleton_id'],z=z,y=y,x=x)
    for u,v in truth['edges']:
        if u not in graph or v not in graph or u==v or graph.has_edge(u,v) or graph.nodes[u]['skeleton_id']!=graph.nodes[v]['skeleton_id']:
            raise ValueError('Invalid skeleton edge')
        graph.add_edge(u,v)
    seg=np.load(a.segmentation,mmap_mode='r',allow_pickle=False);nodes=truth['nodes']
    if 'shape' in meta and list(seg.shape)!=meta['shape']:
        raise ValueError('Prediction extent must match frozen evaluation ROI metadata')
    ids=sample(seg,[n['position_nm'] for n in nodes],meta['origin_nm'],meta['voxel_size_nm'])
    result=evaluate(graph,{n['id']:int(s) for n,s in zip(nodes,ids)},a.evaluator)
    diagnostic=counts([n['skeleton_id'] for n in nodes],ids,interior_segments(seg,a.boundary_margin_voxels))
    for k in ['split_errors','merge_errors','omitted_nodes','omitted_node_fraction']:result[k]=diagnostic[k]
    result['inputs']={n:sha256(getattr(a,n)) for n in ['skeletons','segmentation','metadata']}
    result['boundary_margin_voxels']=a.boundary_margin_voxels
    result['skeleton_protocol']=truth.get('protocol','externally supplied skeletons')
    result['sampling']='floor of physical coordinate relative to voxel outer corner'
    result['scope']='ROI-local paths; no paper endpoint erosion or resampling parity'
    if contract_check is not None:result['prediction_contract']=contract_check
    write_json(external(a.output),result)

if __name__=='__main__':main()
