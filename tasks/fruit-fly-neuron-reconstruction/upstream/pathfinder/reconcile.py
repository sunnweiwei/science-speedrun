"""Budgeted whole-graph reconciliation of model-scored motifs.

Independent maximum retained-edge-score objective, not the author's optimizer.
Exact for components with <=16 eligible edges; larger components use the existing
conservative greedy filter. Missing motif scores forbid assembling that motif.
Components over 32 nodes use bounded lazy constraint checks instead of refusing
every merge. Check-budget exhaustion always rejects the proposed join.
"""
import argparse
import itertools
import json
import math
from pathlib import Path
from search import filter_graph
from lazy_search import filter_large_component
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from index_storage import load_index,TableRecords,NumericNodes
from common import external,sha256,write_json


def connected(nodes, edges):
    nodes=set(nodes)
    if not nodes:return False
    seen={min(nodes)}
    while True:
        more={v for u,v in edges if u in seen and v in nodes}|{u for u,v in edges if v in seen and u in nodes}
        if more<=seen:return seen==nodes
        seen |= more


def reconcile(nodes,edges,motifs,threshold=.2,max_exact_edges=16,include_singletons=True):
    if isinstance(nodes,NumericNodes):node_set=nodes
    else:
        nodes=list(nodes);node_set=set(nodes)
        if len(nodes)!=len(node_set) or any(type(n) is not int or n<=0 for n in nodes):
            raise ValueError('Unique positive integer node IDs required')
    if not 0<=threshold<=1 or not 0<=max_exact_edges<=20:raise ValueError('Invalid search configuration')
    scores={}
    for motif,score in motifs:
        motif=tuple(sorted(set(motif)))
        if not 2<=len(motif)<=4 or not all(n in node_set for n in motif) or not math.isfinite(score) or not 0<=score<=1:
            raise ValueError('Invalid motif or probability')
        if motif in scores:raise ValueError('Duplicate motif scores')
        scores[motif]=score
    seen=set();eligible=[];adjacency={}
    for e in edges:
        u,v=e['source'],e['target'];key=tuple(sorted((u,v)))
        if u not in node_set or v not in node_set or u==v or key in seen:raise ValueError('Unknown, self or duplicate edge')
        if any(not math.isfinite(e[k]) or not 0<=e[k]<=1 for k in ('sense','shape')):raise ValueError('Invalid edge probability')
        seen.add(key)
        if e['shape']<.5:continue
        index=len(eligible);eligible.append(e)
        adjacency.setdefault(u,[]).append((v,index));adjacency.setdefault(v,[]).append((u,index))
    # Visit only nodes incident on eligible edges. Isolates never enter graph
    # search and no component rescans the full edge list.
    visited=set();components=[];component_owner={}
    for root in adjacency:
        if root in visited:continue
        stack=[root];visited.add(root);group=[];edge_indices=set()
        while stack:
            u=stack.pop();group.append(u);component_owner[u]=len(components)
            for v,index in adjacency[u]:
                edge_indices.add(index)
                if v not in visited:visited.add(v);stack.append(v)
        components.append((sorted(group),[eligible[i] for i in sorted(edge_indices)]))
    forbidden_by_component={}
    for motif,score in scores.items():
        owner=component_owner.get(motif[0])
        if score<threshold and owner is not None and all(component_owner.get(n)==owner for n in motif):
            forbidden_by_component.setdefault(owner,[]).append(set(motif))
    retained=[];methods=[];missing=0
    for component_index,(group,local) in enumerate(components):
        localpairs=[(e['source'],e['target']) for e in local]
        forbidden=forbidden_by_component.get(component_index,[]).copy()
        if len(group)>32:
            result=filter_large_component(group,local,scores,threshold)
            retained.extend(result['retained'])
            missing+=result['missing_motif_scores']
            methods.append(dict(nodes=len(group),method='lazy_conservative_greedy',
                **{k:v for k,v in result.items() if k not in ('retained','components')}))
            continue
        for size in range(2,min(4,len(group))+1):
            for motif in itertools.combinations(group,size):
                if not connected(motif,localpairs):continue
                score=scores.get(motif)
                if score is None:missing+=1
                if score is None or score<threshold:forbidden.append(set(motif))
        if len(local)>max_exact_edges:
            result=filter_graph(group,local,[(tuple(m),0.) for m in forbidden],1.,priority='sense')
            retained.extend(result['retained']);methods.append(dict(nodes=len(group),method='greedy'));continue
        ordered=sorted(local,key=lambda e:-e['sense']);best=(-1.,[])
        suffix=[0.]*(len(ordered)+1)
        for i in range(len(ordered)-1,-1,-1):suffix[i]=suffix[i+1]+ordered[i]['sense']
        def visit(i,groups,value,chosen):
            nonlocal best
            if value+suffix[i]<=best[0]:return
            if i==len(ordered):best=(value,chosen);return
            e=ordered[i];a=next(g for g in groups if e['source'] in g);b=next(g for g in groups if e['target'] in g)
            joined=a|b
            if not any(m<=joined for m in forbidden):
                new=[g for g in groups if g is not a and g is not b]+[joined]
                visit(i+1,new,value+e['sense'],chosen+[e])
            visit(i+1,groups,value,chosen)
        visit(0,[{n} for n in group],0.,[])
        retained.extend(best[1]);methods.append(dict(nodes=len(group),method='exact_retained_sense_sum'))
    # Union-find over retained-edge endpoints only; sparse/high IDs are safe.
    parent={}
    def find(n):
        parent.setdefault(n,n)
        while parent[n]!=n:
            parent[n]=parent[parent[n]];n=parent[n]
        return n
    for e in retained:
        a=find(e['source']);b=find(e['target'])
        if a!=b:parent[max(a,b)]=min(a,b)
    merged_groups={}
    for n in parent:merged_groups.setdefault(find(n),[]).append(n)
    groups=[sorted(g) for g in merged_groups.values()]
    if include_singletons:groups.extend([n] for n in nodes if n not in parent)
    return dict(components=sorted(groups),retained=retained,methods=methods,
                singleton_count=len(nodes)-len(parent),components_include_singletons=include_singletons,
                missing_motif_scores=missing,
                missing_motif_scores_scope='exhaustive_small_components_first_missing_per_rejected_large_join',
                author_parity=False,paper_comparable=False)


def streaming_nodes(seg,slab_depth=4):
    import numpy as np
    if seg.ndim!=3 or seg.dtype.kind!='u' or slab_depth<1:raise ValueError('Unsigned ZYX segmentation required')
    nodes=set()
    for z in range(0,len(seg),slab_depth):nodes.update(int(n) for n in np.unique(seg[z:z+slab_depth]) if n>0)
    return sorted(nodes)


def reconcile_pairs(nodes, edges, motifs, threshold=.8):
    """Deterministic disjoint-pair ablation; no unscored transitive joins.

    Each original instance can participate in at most one retained merge.
    This explicitly omits higher-order motif consistency and graph search.
    """
    if not math.isfinite(threshold) or not 0<=threshold<=1:
        raise ValueError('Invalid pair threshold')
    node_set=nodes if isinstance(nodes,NumericNodes) else set(nodes)
    if not isinstance(nodes,NumericNodes) and (len(node_set)!=len(nodes) or
        any(type(n) is not int or n<=0 for n in nodes)):
        raise ValueError('Unique positive node IDs required')
    scores={}
    for motif,score in motifs:
        pair=tuple(sorted(motif))
        if (len(pair)!=2 or pair[0]==pair[1] or any(n not in node_set for n in pair)
            or pair in scores or not math.isfinite(score) or not 0<=score<=1):
            raise ValueError('Unique scored pairs required')
        scores[pair]=score
    seen=set();eligible=[]
    for e in edges:
        pair=tuple(sorted((e['source'],e['target'])))
        if (pair[0]==pair[1] or any(n not in node_set for n in pair) or pair in seen
            or any(not math.isfinite(e[k]) or not 0<=e[k]<=1 for k in ('sense','shape'))
            or pair not in scores or scores[pair]!=e['shape']):
            raise ValueError('Unique edges with matching pair scores required')
        seen.add(pair)
        if e['shape']>=threshold:eligible.append((pair,e))
    used=set();groups=[];retained=[]
    for pair,e in sorted(eligible,key=lambda item:(-item[1]['shape'],-item[1]['sense'],item[0])):
        if used.intersection(pair):continue
        used.update(pair);groups.append(list(pair));retained.append(e)
    return dict(components=sorted(groups),retained=retained,
        methods=[dict(method='disjoint_pair_greedy',eligible_edges=len(eligible))],
        singleton_count=len(nodes)-len(used),components_include_singletons=False,
        missing_motif_scores=0,missing_motif_scores_scope='scored_pairs_only',
        higher_order_motifs_checked=False,transitive_merges_allowed=False,
        author_parity=False,paper_comparable=False)


def indexed_nodes(index,shape,segmentation_sha256):
    if (index.get('schema')!='predicted-instance-index/v1' or index.get('axes')!='zyx' or
        index.get('ground_truth_used') is not False or index.get('shape')!=list(shape) or
        index.get('segmentation_sha256')!=segmentation_sha256):raise ValueError('Instance index provenance mismatch')
    if isinstance(index['records'],TableRecords):return NumericNodes(index['records'].ids)
    nodes=[r['id'] for r in index['records']]
    if len(nodes)!=len(set(nodes)) or any(type(n) is not int or n<=0 for n in nodes):raise ValueError('Invalid indexed IDs')
    return nodes


def remap_slabs(seg,destination,components,slab_depth=4):
    """Sparse binary-search remapping, bounded by one slab; zero is preserved."""
    import numpy as np
    if seg.ndim!=3 or seg.dtype.kind!='u' or slab_depth<1 or destination.shape!=seg.shape or destination.dtype!=seg.dtype:
        raise ValueError('Matching unsigned ZYX arrays and positive slab depth required')
    mapping={};seen=set()
    for group in components:
        if len(group)<2:continue
        target=min(group)
        for n in group:
            if n<=0 or n>np.iinfo(seg.dtype).max:raise ValueError('Invalid merge ID')
            if n in seen:raise ValueError('Overlapping merge components')
            seen.add(n)
            if n!=target:mapping[n]=target
    keys=np.array(sorted(mapping),dtype=seg.dtype);values=np.array([mapping[int(k)] for k in keys],dtype=seg.dtype)
    for z in range(0,len(seg),slab_depth):
        slab=np.array(seg[z:z+slab_depth],copy=True)
        if len(keys):
            indices=np.searchsorted(keys,slab);np.minimum(indices,len(keys)-1,out=indices)
            matching=keys[indices]==slab;slab[matching]=values[indices[matching]]
        destination[z:z+slab_depth]=slab
    if hasattr(destination,'flush'):destination.flush()
    return len(mapping)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scores',type=Path,required=True);p.add_argument('--segmentation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--threshold',type=float,default=.2)
    p.add_argument('--instance-index',type=Path);p.add_argument('--slab-depth',type=int,default=4)
    p.add_argument('--root',type=Path,help='Budget root for output capacity reservation')
    p.add_argument('--mode',choices=['motif','disjoint-pairs'],default='motif')
    a=p.parse_args()
    import numpy as np
    scores=json.loads(a.scores.read_text());seg=np.load(a.segmentation,mmap_mode='r',allow_pickle=False)
    if seg.ndim!=3 or seg.dtype.kind!='u' or a.slab_depth<1:raise ValueError('Unsigned ZYX segmentation and positive slab depth required')
    seg_hash=sha256(a.segmentation)
    if scores.get('segments_sha256',seg_hash)!=seg_hash:raise ValueError('SHAPE scores refer to a different segmentation')
    nodes=indexed_nodes(load_index(a.instance_index),seg.shape,seg_hash) if a.instance_index else streaming_nodes(seg,a.slab_depth)
    motifs=[(r['nodes'],r['shape']) for r in scores['motifs'] if r['shape'] is not None]
    if a.mode=='disjoint-pairs':
        if scores.get('max_nodes')!=2 or scores.get('truncated') is not False or scores.get('unprocessed_motifs'):
            raise ValueError('Complete untruncated pair enumeration required')
        result=reconcile_pairs(nodes,scores['edges'],motifs,a.threshold)
    else:result=reconcile(nodes,scores['edges'],motifs,a.threshold,include_singletons=False)
    result['reconciliation_mode']=a.mode
    out=external(a.output)
    if a.root:
        from fetch_axonem import reserve
        root=external(a.root)
        if root not in out.parents:raise ValueError('Output must be under budget root')
        reserve(root,int(seg.nbytes)+1048576)
    out.mkdir(parents=True,exist_ok=False)
    merged=np.lib.format.open_memmap(out/'segmentation.npy',mode='w+',dtype=seg.dtype,shape=seg.shape)
    changed_ids=remap_slabs(seg,merged,result['components'],a.slab_depth)
    result.update(inputs=dict(scores_sha256=sha256(a.scores),segmentation_sha256=seg_hash,
        instance_index_sha256=sha256(a.instance_index) if a.instance_index else None),
        ground_truth_used=False,node_count=len(nodes),remapped_ids=changed_ids,slab_depth=a.slab_depth,
        segmentation_sha256=sha256(out/'segmentation.npy'),
        untouched_instances_preserved=True)
    write_json(out/'receipt.json',result)


if __name__=='__main__':main()
