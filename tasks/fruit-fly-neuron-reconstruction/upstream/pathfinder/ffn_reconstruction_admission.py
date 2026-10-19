"""Fail-closed quality/cost admission for a replacement of FFN members 2/3.

Consumes scored, completed fixed-validation reconstructions. Hash/array
verification belongs to the scorer; its pinned reports are recorded by caller.
Passing seed growth alone never authorizes a full test reconstruction.
"""
import math


def admit_reconstruction(candidate, receipts, reference, reference_receipts):
    if candidate.get('test_used') is not False or reference.get('test_used') is not False:
        raise ValueError('Only fixed validation scores may select weights')
    if len(receipts)!=6 or len(reference_receipts)!=6:
        raise ValueError('Six completed member receipts required')
    expected={f'variant-{i}' for i in range(6)}|{'consensus'}
    def scores(report):
        result={r['name']:r['metrics'] for r in report['results']}
        if set(result)!=expected:raise ValueError('Missing or extra validation score')
        if any(not math.isfinite(v) for row in result.values() for v in row.values()):
            raise ValueError('Nonfinite validation score')
        return result
    c,r=scores(candidate),scores(reference)
    for a,b in zip(receipts,reference_receipts):
        if (a.get('all_shard_tiles_complete') is not True or
            a.get('ground_truth_used') is not False or
            any(t['budget_exhausted'] for t in a['tile_results'])):
            raise ValueError('Incomplete or label-fed validation inference')
        for key in ('raw_sha256','roi','metadata','tile_size','seed_order'):
            if a[key]!=b[key]:raise ValueError('Fixed validation geometry/policy differs: '+key)
    for i in (0,1,4,5):
        if receipts[i]['checkpoint_sha256']!=reference_receipts[i]['checkpoint_sha256']:
            raise ValueError('Unexpected change to retained member')
    hashes=[a['checkpoint_sha256'] for a in receipts]
    if len(set(hashes))!=3 or hashes[2]!=hashes[3] or hashes[2]==reference_receipts[2]['checkpoint_sha256']:
        raise ValueError('Three distinct checkpoints and actual replacement required')
    reasons=[]
    minimum_nerl=min(r[f'variant-{i}']['nerl'] for i in (0,1))
    maximum_omission=max(r[f'variant-{i}']['omitted_node_fraction'] for i in (0,1))+.02
    for i,baseline in [(2,0),(3,1)]:
        if c[f'variant-{i}']['nerl']<minimum_nerl:reasons.append(f'member-{i} nERL below first-weight reference')
        if c[f'variant-{i}']['omitted_node_fraction']>maximum_omission:reasons.append(f'member-{i} omission exceeds reference')
        if receipts[i]['model_windows']>1.2*reference_receipts[baseline]['model_windows']:
            reasons.append(f'member-{i} total model calls exceed reference*1.2')
    if c['consensus']['nerl']<1.1*r['consensus']['nerl']:
        reasons.append('consensus nERL improvement below 10%')
    return dict(passed=not reasons,reasons=reasons,minimum_replacement_nerl=minimum_nerl,
                maximum_replacement_omission=maximum_omission,
                required_consensus_nerl=1.1*r['consensus']['nerl'],
                scope='Fixed validation reconstruction, not test selection or full-scale throughput proof')
