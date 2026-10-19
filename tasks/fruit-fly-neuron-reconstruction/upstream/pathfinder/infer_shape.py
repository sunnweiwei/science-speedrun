"""Score predicted-segmentation motifs with SHAPE; no ground-truth input.

Incomplete 10 um contexts are refused by default. --allow-partial-context is an
explicit engineering diagnostic and never a paper-comparable shape evaluation.
"""
import argparse,json,sys,itertools,time,heapq
from pathlib import Path
import numpy as np
from index_storage import load_index,TableRecords,TableMapping,validate_table
from skimage.measure import marching_cubes
from shape_data import sample_surface,RADIUS_UM
from shape_context import indexed_mesh,context_coverage,MeshCache
from inference_progress import Progress
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import external,device_guard,write_json,sha256
from execution_resources import device_guard


def canonical_edges(edges):
    """Collapse reciprocal/duplicate candidates with maximum SENSE evidence."""
    unique={}
    for row in edges:
        a,b=sorted((int(row['source']),int(row['target'])))
        sense=float(row['sense'])
        if a<=0 or b<=0 or a==b or not np.isfinite(sense) or not 0<=sense<=1:
            raise ValueError('Distinct positive IDs and finite SENSE probability required')
        if (a,b) not in unique or sense>unique[a,b]['sense']:
            unique[a,b]=dict(row,source=a,target=b,sense=sense)
    return [unique[key] for key in sorted(unique)]


def connected_motifs(edges,max_nodes=4,limit=256):
    if max_nodes<2 or max_nodes>4 or limit<1:raise ValueError('Require 2..4 nodes and positive motif limit')
    neighbors={};pairs=set()
    for row in edges:
        a,b=int(row['source']),int(row['target'])
        if a<=0 or b<=0 or a==b:raise ValueError('Expected distinct positive segment IDs')
        neighbors.setdefault(a,set()).add(b);neighbors.setdefault(b,set()).add(a);pairs.add(tuple(sorted((a,b))))
    result=[];frontier=sorted(pairs)
    for size in range(2,max_nodes+1):
        room=limit-len(result)
        if len(frontier)>room:return result+frontier[:room],True
        result.extend(frontier)
        if size==max_nodes:return result,False
        if len(result)==limit:
            more=any(set().union(*(neighbors[i] for i in nodes))-set(nodes) for nodes in frontier)
            return result,bool(more)
        # Keep only the lexicographically earliest remaining+1 motifs. The extra
        # item detects truncation without materializing a dense combinatorial
        # frontier that will never be scored under this global cap.
        keep=limit-len(result)+1;heap=[];kept=set();cutoff=None
        for nodes in frontier:
            for n in set().union(*(neighbors[i] for i in nodes))-set(nodes):
                candidate=tuple(sorted((*nodes,n)))
                if cutoff is not None and candidate>=cutoff:continue
                if candidate in kept:continue
                reverse=tuple(-v for v in candidate)
                if len(heap)<keep:
                    heapq.heappush(heap,reverse);kept.add(candidate)
                    if len(heap)==keep:cutoff=tuple(-v for v in heap[0])
                elif reverse>heap[0]:
                    removed=heapq.heapreplace(heap,reverse)
                    kept.remove(tuple(-v for v in removed));kept.add(candidate)
                    cutoff=tuple(-v for v in heap[0])
        frontier=sorted(kept)
    return result,False


def complete_context(center_zyx,shape,spacing_um):
    center=(np.asarray(center_zyx,dtype=float)+.5)*spacing_um
    extent=np.asarray(shape)*spacing_um
    return bool(((center>=RADIUS_UM)&(center<=extent-RADIUS_UM)).all())


def mesh_segment(seg,sid,spacing):
    positions=np.where(seg==sid)
    if not len(positions[0]):raise ValueError(f'Missing predicted segment {sid}')
    lo=np.array([x.min() for x in positions]);hi=np.array([x.max()+1 for x in positions])
    mask=seg[tuple(slice(a,b) for a,b in zip(lo,hi))]==sid
    v,f,_,_=marching_cubes(np.pad(mask,1),.5,spacing=tuple(spacing))
    # Input index0 denotes a voxel center; physical boundary origin is0.
    v+=(lo-.5)*spacing
    return v,f,np.array([x.mean() for x in positions])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','checkpoint','segments','edges','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--voxel-size-nm',type=float,nargs=3,required=True)
    p.add_argument('--origin-nm',type=float,nargs=3,default=[0,0,0])
    p.add_argument('--max-motifs',type=int,default=256);p.add_argument('--max-nodes',type=int,default=4)
    p.add_argument('--allow-partial-context',action='store_true');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--dry-run',action='store_true',help='Validate contexts without model loading or GPU work')
    p.add_argument('--instance-index',type=Path);p.add_argument('--coverage',type=Path)
    p.add_argument('--metadata',type=Path);p.add_argument('--seconds',type=float,default=1800)
    p.add_argument('--shards',type=int,default=1);p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--mesh-cache-bytes',type=int,default=512*2**20,
        help='Retained CPU mesh cache limit; excludes current motif and extraction/sampling buffers. Zero disables retention.')
    a=p.parse_args();out=external(a.output);spacing=np.asarray(a.voxel_size_nm)/1000
    cache=MeshCache(a.mesh_cache_bytes)
    if a.seconds<=0 or a.shards<1 or not 0<=a.shard_index<a.shards:raise ValueError('Invalid time/shard limits')
    if not np.isfinite(spacing).all() or (spacing<=0).any():raise ValueError('Positive finite ZYX voxel sizes required')
    seg=np.load(a.segments,mmap_mode='r',allow_pickle=False)
    if seg.ndim!=3 or seg.dtype.kind not in 'ui':raise ValueError('Integer ZYX predicted segmentation required')
    progress=Progress(out.with_suffix('.progress.jsonl'),'shape',a.shard_index)
    progress.emit(force=True,status='validating_inputs')
    segmentation_hash=sha256(a.segments);instance_rows=None;coverage=None;meta=None
    if a.instance_index:
        idx=load_index(a.instance_index)
        if idx['segmentation_sha256']!=segmentation_hash or idx['shape']!=list(seg.shape) or idx.get('ground_truth_used') is not False:raise ValueError('Instance index provenance mismatch')
        instance_rows=TableMapping(idx['records']) if isinstance(idx['records'],TableRecords) else {r['id']:r for r in idx['records']}
        if isinstance(idx['records'],TableRecords):validate_table(idx['records'],seg)
    if a.coverage:
        if not a.metadata:raise ValueError('Coverage requires physical metadata')
        coverage=json.loads(a.coverage.read_text());meta=json.loads(a.metadata.read_text())
        if meta['shape']!=list(seg.shape) or meta['voxel_size_nm']!=a.voxel_size_nm or meta['origin_nm']!=a.origin_nm:raise ValueError('Coverage geometry mismatch')
    begun=time.monotonic();exhausted=False
    obj=json.loads(a.edges.read_text());edges=canonical_edges(obj if isinstance(obj,list) else obj['edges'])
    motifs,truncated=connected_motifs(edges,a.max_nodes,a.max_motifs);rows=[];model=None
    edge_centers={tuple(sorted((int(e['source']),int(e['target'])))):e.get('center_zyx') for e in edges}
    scored_count=0;assigned=len(range(a.shard_index,len(motifs),a.shards))
    def record(row):
        nonlocal scored_count
        rows.append(row);scored_count+=row['shape'] is not None
        progress.emit(status='running',processed=len(rows),total=assigned,scored=scored_count)
    progress.emit(force=True,status='running',processed=0,total=assigned,scored=0)
    for index,nodes in enumerate(motifs):
        if index%a.shards!=a.shard_index:continue
        if time.monotonic()-begun>=a.seconds:exhausted=True;break
        # Release the previous motif before extraction. Only its active 2..4
        # meshes and the explicitly bounded cache can retain mesh arrays.
        meshes={}
        for sid in nodes:
            meshes[sid]=cache.get(sid,lambda key:indexed_mesh(seg,key,spacing,instance_rows[key]) if instance_rows is not None else mesh_segment(seg,key,spacing))
        center=edge_centers.get(nodes)
        if center is None:center=np.mean([meshes[n][2] for n in nodes],axis=0)
        center=np.asarray(center,dtype=float)
        if center.shape!=(3,) or not np.isfinite(center).all() or (center<0).any() or (center>=np.array(seg.shape)).any():
            raise ValueError('Invalid ROI-relative center_zyx')
        complete=complete_context(center,seg.shape,spacing)
        row=dict(nodes=list(nodes),shape=None,center_zyx=center.tolist(),context_complete=complete)
        if coverage is not None:
            row.update(context_coverage(center,spacing,meta,coverage))
            if not row['prediction_context_complete']:row['status']='incomplete_prediction_context';record(row);continue
        if not complete and not a.allow_partial_context:row['status']='insufficient_context';record(row);continue
        if a.dry_run:row['status']='context_check_only';record(row);continue
        vertices=[];faces=[];offset=0
        for n in nodes:
            v,f,_=meshes[n];vertices.append(v);faces.append(f+offset);offset+=len(v)
        points=sample_surface(np.concatenate(vertices),np.concatenate(faces),(center+.5)*spacing,np.random.default_rng(a.seed+index))
        del vertices,faces,v,f
        meshes.clear()
        if model is None:
            device_guard('cuda')
            import torch
            from shape_model import FPS,ShapeClassifier
            torch.set_num_threads(2)
            checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
            architecture=checkpoint['architecture']
            if architecture.get('stage')!='shape' or architecture.get('points')!=8192 or architecture.get('radius_um')!=10:
                raise ValueError('Full8192-point/10um SHAPE checkpoint required')
            model=ShapeClassifier(FPS(a.root)).cuda();model.load_state_dict(checkpoint['model'],strict=True);model.eval()
        with torch.inference_mode():row['shape']=float(model(torch.as_tensor(points,device='cuda')[None]).sigmoid()[0])
        row['status']='scored' if complete else 'partial_context_diagnostic';record(row)
    progress.emit(force=True,status='limit_reached' if exhausted else 'complete',processed=len(rows),total=assigned,scored=scored_count)
    pair_scores={tuple(r['nodes']):r for r in rows if len(r['nodes'])==2}
    scored_edges=[]
    for e in edges:
        r=pair_scores.get(tuple(sorted((int(e['source']),int(e['target'])))))
        if r is not None and r['shape'] is not None:
            scored_edges.append(dict(e,shape=r['shape'],context_complete=r['context_complete']))
    write_json(out,dict(edges=scored_edges,motifs=rows,segments_sha256=segmentation_hash,edges_sha256=sha256(a.edges),
        checkpoint_sha256=sha256(a.checkpoint),voxel_size_nm=a.voxel_size_nm,origin_nm=a.origin_nm,
        axes='zyx',radius_um=10,points=8192,motif_limit=a.max_motifs,truncated=truncated,
        full_context_scored=sum(r['shape'] is not None and r['context_complete'] for r in rows),
        allow_partial_context=a.allow_partial_context,paper_comparable=False,ground_truth_used=False,
        budget_exhausted=exhausted,seconds=time.monotonic()-begun,shards=a.shards,shard_index=a.shard_index,
        total_enumerated_motifs=len(motifs),max_nodes=a.max_nodes,seed=a.seed,
        index_sha256=sha256(a.instance_index) if a.instance_index else None,
        coverage_sha256=sha256(a.coverage) if a.coverage else None,mesh_cache=cache.statistics()))

if __name__=='__main__':main()
