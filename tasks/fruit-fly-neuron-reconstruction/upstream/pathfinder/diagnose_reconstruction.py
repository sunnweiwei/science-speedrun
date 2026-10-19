"""Evaluation-only paired skeleton diagnostics; sparse labels leave false merges hidden."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
from topology import sample
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import external,sha256,write_json


def diagnose(truth,before,after,processed_ids=(),scored_ids=(),merged_ids=()):
    nodes=truth['nodes'];before=np.asarray(before);after=np.asarray(after)
    if before.shape!=(len(nodes),) or after.shape!=before.shape or before.dtype.kind not in 'iu' or after.dtype.kind not in 'iu':
        raise ValueError('Aligned integer node assignments required')
    if (before<0).any() or (after<0).any():raise ValueError('Nonnegative prediction labels required')
    if not np.array_equal(before==0,after==0):raise ValueError('Merge-only reconstruction changed omission mask')
    remap=defaultdict(set);lookup={n['id']:i for i,n in enumerate(nodes)}
    if len(lookup)!=len(nodes):raise ValueError('Duplicate skeleton node ID')
    for a,b in zip(before,after):remap[int(a)].add(int(b))
    if any(len(v)>1 for v in remap.values()):raise ValueError('Merge-only reconstruction split a sampled instance')
    gt=np.asarray([n['skeleton_id'] for n in nodes]);positions=np.asarray([n['position_nm'] for n in nodes],float)
    if (gt<=0).any() or positions.shape!=(len(nodes),3) or not np.isfinite(positions).all():raise ValueError('Invalid skeleton geometry')
    support=[]
    for labels in (before,after):
        groups=defaultdict(set)
        for g,p in zip(gt,labels):
            if p:groups[int(p)].add(int(g))
        support.append({p:sorted(g) for p,g in groups.items() if len(g)>1})
    per={int(g):dict(skeleton_id=int(g),nodes=int((gt==g).sum()),
        omitted_nodes=int(((gt==g)&(before==0)).sum()),
        before_segments=len(set(map(int,before[gt==g]))-{0}),after_segments=len(set(map(int,after[gt==g]))-{0}),
        path_nm=0.,before_adjacent_connected_nm=0.,after_adjacent_connected_nm=0.,
        restored_edges=0,restored_nm=0.,restored_in_visible_mixed_segment_nm=0.) for g in np.unique(gt)}
    restored=[];seen=set()
    for u,v in truth['edges']:
        if u not in lookup or v not in lookup or u==v or tuple(sorted((u,v))) in seen:raise ValueError('Invalid or duplicate edge')
        seen.add(tuple(sorted((u,v))));i,j=lookup[u],lookup[v]
        if gt[i]!=gt[j]:raise ValueError('GT edge crosses instance identities')
        length=float(np.linalg.norm(positions[i]-positions[j]));row=per[int(gt[i])];row['path_nm']+=length
        a=before[i]>0 and before[i]==before[j];b=after[i]>0 and after[i]==after[j]
        if a:row['before_adjacent_connected_nm']+=length
        if b:row['after_adjacent_connected_nm']+=length
        if a and not b:raise ValueError('Merge-only reconstruction broke a previously connected edge')
        if b and not a:
            mixed=int(after[i]) in support[1]
            row['restored_edges']+=1;row['restored_nm']+=length
            if mixed:row['restored_in_visible_mixed_segment_nm']+=length
            restored.append(dict(nodes=[u,v],skeleton_id=int(gt[i]),length_nm=length,
                before_segments=[int(before[i]),int(before[j])],after_segment=int(after[i]),visible_mixed_segment=mixed))
    for row in per.values():
        selected=gt==row['skeleton_id']
        for name,ids in [('sense_processed',processed_ids),('shape_scored',scored_ids),('in_merged_component',merged_ids)]:
            row[name+'_nodes']=int(np.isin(before[selected],list(ids)).sum())
    return dict(schema='pathfinder-paired-diagnostics/v1',per_skeleton=list(per.values()),restored_edges=restored,
        visible_mixed_segments_before=support[0],visible_mixed_segments_after=support[1],
        sampled_nodes=len(nodes),changed_node_labels=int((before!=after).sum()),
        scope='Evaluation only; adjacent-edge continuity is not nERL and can increase during erroneous merges.',
        mixed_segment_policy='Any sampled nodes from two frozen GT instances; separate from10-node merge diagnostic.',
        sparse_gt_limitation='Merges into unlabelled or excluded neurons are not detected.',paper_comparable=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('skeletons','before','after','metadata','candidates','scores','search','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();read=lambda path:json.loads(path.read_text())
    truth=read(a.skeletons);meta=read(a.metadata);candidates=read(a.candidates);scores=read(a.scores);search=read(a.search)
    if truth['axes']!='zyx' or truth['coordinate_units']!='nm' or meta['axes']!='zyx':raise ValueError('Physical ZYX inputs required')
    labels=[]
    for path in (a.before,a.after):
        array=np.load(path,mmap_mode='r',allow_pickle=False)
        if list(array.shape)!=meta['shape']:raise ValueError('Prediction ROI mismatch')
        labels.append(sample(array,[n['position_nm'] for n in truth['nodes']],meta['origin_nm'],meta['voxel_size_nm']))
    result=diagnose(truth,*labels,
        processed_ids=[r['instance_id'] for r in candidates['coverage']['processed']],
        scored_ids={n for r in scores['motifs'] if r['shape'] is not None for n in r['nodes']},
        merged_ids={n for group in search['components'] if len(group)>1 for n in group})
    result['inputs']={key:sha256(getattr(a,key)) for key in ('skeletons','before','after','metadata','candidates','scores','search')}
    if result['inputs']['before']!=scores['segments_sha256']:raise ValueError('Scores do not match input segmentation')
    if result['inputs']['after']!=search['segmentation_sha256']:raise ValueError('Search does not match final segmentation')
    write_json(external(a.output),result)

if __name__=='__main__':main()
