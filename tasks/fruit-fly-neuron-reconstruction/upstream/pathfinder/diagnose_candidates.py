"""Post-hoc GT audit of frozen development predictions; never changes inference.

Dominant-instance agreement is a diagnostic proxy, not biological merge truth.
Label semantics (especially ID1) require an independent author-source audit.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import numpy as np
from topology import sample
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import external, sha256, write_json


def overlap_audit(prediction, truth, valid, purity=.95, min_known=20):
    if prediction.shape != truth.shape or valid.shape != truth.shape:
        raise ValueError('Aligned prediction, GT and validity required')
    known = valid & (truth > 0)
    records = {}
    for sid in np.unique(prediction):
        if not sid:
            continue
        mask = prediction == sid
        values, sizes = np.unique(truth[mask & known], return_counts=True)
        count = int(sizes.sum()); total = int(mask.sum())
        dominant = int(values[sizes.argmax()]) if count else None
        share = float(sizes.max()/count) if count else 0.
        records[int(sid)] = dict(size=total, known_voxels=count, known_fraction=count/total,
            dominant_gt=dominant, purity=share,
            unambiguous=count>=min_known and share>=purity and count/total>=purity,
            gt_overlap={str(int(v)):int(n) for v,n in zip(values,sizes)})
    return records, dict(known_gt_voxels=int(known.sum()),
        omitted_known_voxels=int(((prediction==0)&known).sum()),
        omitted_known_fraction=float(((prediction==0)&known).sum()/max(1,known.sum())),
        gt_label1_voxels=int(((truth==1)&known).sum()),
        gt_label1_fraction=float(((truth==1)&known).sum()/max(1,known.sum())))


def candidate_audit(prediction, records, edges, covered_ids, start, raw_shape,
                    min_size=500, overlap_threshold=.5):
    """Upper-bound eligibility from the exact one-center inference geometry.

    Reachable means enough target voxels are visible to possibly clear the
    overlap gate; it says nothing about connectivity or attainable model output.
    """
    eligible_true, eligible_false = set(), set()
    for source in covered_ids:
        if not records[source]['unambiguous']:
            continue
        locations = np.argwhere(prediction==source)
        global_locations = locations + start
        locations = locations[((global_locations>=64)&(global_locations<=raw_shape-64)).all(axis=1)]
        if not len(locations):
            continue
        center = locations[np.argmin(((locations-locations.mean(axis=0))**2).sum(axis=1))]
        lo=np.maximum(center-64,0);hi=np.minimum(center+64,prediction.shape)
        visible=Counter(map(int,prediction[tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))].ravel()))
        for target, row in records.items():
            if target==source or not row['unambiguous'] or row['size']<min_size:
                continue
            if visible[target]/row['size'] < overlap_threshold:
                continue
            destination=eligible_true if records[source]['dominant_gt']==row['dominant_gt'] else eligible_false
            destination.add((source,target))
    judged=[]
    for edge in edges:
        source,target=int(edge['source']),int(edge['target'])
        if source not in records or target not in records:
            raise ValueError('Candidate references an unknown predicted segment')
        if not records[source]['unambiguous'] or not records[target]['unambiguous']:
            status='ambiguous'
        else:
            status='same_gt' if records[source]['dominant_gt']==records[target]['dominant_gt'] else 'different_gt'
        judged.append(dict(source=source,target=target,sense=edge['sense'],status=status,
                           source_gt=records[source]['dominant_gt'],target_gt=records[target]['dominant_gt']))
    counts=Counter(r['status'] for r in judged)
    proposed_true={(r['source'],r['target']) for r in judged if r['status']=='same_gt'}
    return dict(directed_edges=judged, counts=dict(counts),
        same_gt_precision_excluding_ambiguous=counts['same_gt']/(counts['same_gt']+counts['different_gt']) if counts['same_gt']+counts['different_gt'] else None,
        eligible_same_gt_directed_pairs=len(eligible_true), eligible_different_gt_directed_pairs=len(eligible_false),
        proposed_eligible_same_gt_pairs=len(proposed_true&eligible_true),
        same_gt_proposal_fraction_within_geometry=len(proposed_true&eligible_true)/len(eligible_true) if eligible_true else None,
        complete_recall_claim=False,
        scope='Purity-filtered same-instance proxy among size/FOV/processed-center-eligible ordered pairs; no full connectivity recall')


def skeleton_audit(truth, prediction, metadata):
    nodes=truth['nodes']
    sampled=sample(prediction,[n['position_nm'] for n in nodes],metadata['origin_nm'],metadata['voxel_size_nm'])
    assignments={n['id']:int(p) for n,p in zip(nodes,sampled)}
    positions={n['id']:np.asarray(n['position_nm']) for n in nodes}
    support=defaultdict(Counter)
    for n,p in zip(nodes,sampled):
        if p: support[int(p)][int(n['skeleton_id'])]+=1
    merging={p for p,counts in support.items() if len(counts)>1}
    counts=Counter();lengths=Counter();gt_lengths=Counter()
    by_id={n['id']:n for n in nodes}
    for u,v in truth['edges']:
        a,b=assignments[u],assignments[v]
        kind='omitted' if not a or not b else 'split' if a!=b else 'merge_contaminated' if a in merging else 'correct'
        distance=float(np.linalg.norm(positions[u]-positions[v]))
        counts[kind]+=1;lengths[kind]+=distance;gt_lengths[int(by_id[u]['skeleton_id'])]+=distance
    return dict(edge_counts=dict(counts),edge_length_nm=dict(lengths),
        erl_merging_segment_support={str(p):dict(support[p]) for p in sorted(merging)},
        gt_label1_path_length_nm=gt_lengths[1], total_path_length_nm=sum(gt_lengths.values()),
        caveat='funlib ERL treats even one node from another GT ID as merge contamination; reported merge_errors requires >=10 nodes per GT ID. ERL squares accumulated correct lengths, so few cuts can sharply reduce score.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--run',default='dev-probe-v1')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    manifest_path=a.root/'prepared/manifest.json';manifest=json.loads(manifest_path.read_text())
    block=next(row for row in manifest['splits']['validation'] if row['name']=='700-3584-0')
    folder=Path(block['folder']);gt=np.load(folder/'labels.npy',mmap_mode='r');valid=np.load(folder/'valid.npy',mmap_mode='r')
    for name in ('labels.npy','valid.npy'):
        if sha256(folder/name)!=block['files'][name]:
            raise ValueError('GT hash differs from frozen manifest')
    raw_shape=np.load(folder/'raw.npy',mmap_mode='r').shape
    results=[]
    for i in range(4):
        d=a.root/a.run/str(i)
        ffn=json.loads((d/'ffn/receipt.json').read_text());sense=json.loads((d/'sense/receipt.json').read_text())
        if ffn['ground_truth_used'] is not False or sense['ground_truth_used'] is not False:
            raise ValueError('Require frozen GT-free inference receipts')
        roi=np.asarray(ffn['roi']).reshape(3,2);slices=tuple(slice(int(lo),int(hi)) for lo,hi in roi)
        expected_origin=np.asarray(block['origin_nm'])+roi[:,0]*np.asarray(manifest['voxel_size_nm'])
        if not np.array_equal(expected_origin,ffn['metadata']['origin_nm']):
            raise ValueError('Prediction and GT physical origins disagree')
        if sense['segmentation_sha256']!=sha256(d/'ffn/segmentation.npy'):
            raise ValueError('Candidate provenance does not match FFN prediction')
        prediction=np.load(d/'ffn/segmentation.npy');records,overlap=overlap_audit(prediction,gt[slices],valid[slices])
        edges=json.loads((d/'sense/edges.json').read_text())['edges']
        candidate=candidate_audit(prediction,records,edges,sense['covered_instance_ids'],roi[:,0],np.asarray(raw_shape))
        skeleton_path=a.root/f'dev-skeletons-probe-{i}/skeletons.json'
        skeleton=skeleton_audit(json.loads(skeleton_path.read_text()),prediction,ffn['metadata'])
        metrics=json.loads((d/'ffn-metrics.json').read_text())
        results.append(dict(roi=i,roi_zyx=ffn['roi'],nerl=metrics['nerl'],split_errors=metrics['split_errors'],
            merge_errors=metrics['merge_errors'],overlap=overlap,segments=records,candidates=candidate,skeleton=skeleton,
            reconstruction_unchanged=bool(np.array_equal(prediction,np.load(d/'reconstruction/segmentation.npy'))),
            inputs={str(path.relative_to(a.root)):sha256(path) for path in [d/'ffn/segmentation.npy',d/'ffn/receipt.json',d/'sense/edges.json',d/'sense/receipt.json',skeleton_path]}))
    result=dict(protocol='local-posthoc-candidate-diagnostics/v1', ground_truth_used_for='evaluation lookup only; inference outputs never modified',
        label_semantics='Frozen positive-ID convention; ID1/nonaxon semantics not independently verified. Not an axon-only scientific claim.',
        purity_threshold=.95,min_known_voxels=20,known_fraction_threshold=.95,target_min_size=500,
        candidate_overlap_threshold=.5,selection='all four previously frozen dev ROIs; no test access',
        manifest_sha256=sha256(manifest_path),gt_sha256=sha256(folder/'labels.npy'),valid_sha256=sha256(folder/'valid.npy'),
        diagnostic_code_sha256=sha256(Path(__file__)),paper_comparable=False,results=results)
    write_json(external(a.output),result)
    for r in results:
        print(json.dumps({k:r[k] for k in ['roi','nerl','split_errors','merge_errors','overlap','candidates','skeleton','reconstruction_unchanged']}))


if __name__=='__main__':main()
