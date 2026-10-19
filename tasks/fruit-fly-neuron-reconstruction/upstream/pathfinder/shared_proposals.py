"""Use complete SENSE proposals for an explicit shared-proposal FFN ablation.

This omits independent watershed decision generation. It is not the paper's
traditional agglomeration baseline. No evaluation labels enter this stage.
"""
import argparse
import json
import math
from pathlib import Path
import sys
from infer_shape import canonical_edges
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import sha256, write_json


def build(candidates, base_sha256, shape, segmentation=None):
    if (candidates.get('ground_truth_used') is not False
        or candidates.get('provenance', {}).get('segmentation_sha256') != base_sha256):
        raise ValueError('Prediction-only candidates from this base required')
    coverage = candidates['coverage']
    if (coverage['unprocessed_instance_ids']
        or coverage['processed_count'] != coverage['eligible_count']):
        raise ValueError('Complete eligible SENSE coverage required')
    rows = []
    for edge in canonical_edges(candidates['edges']):
        center = edge.get('center_zyx')
        if (not isinstance(center, list) or len(center) != 3
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in center)
            or any(v < 50 or v >= extent-50 for v, extent in zip(center, shape))):
            raise ValueError('Full FFN crop around a finite prediction center required')
        point=[int(v) for v in center]
        if segmentation is not None:
            import numpy as np
            seed=edge.get('evidence_seed_id');pair=[edge['source'],edge['target']]
            if seed not in pair or any(v!=int(v) for v in center) or int(segmentation[tuple(point)])!=seed:
                raise ValueError('SENSE representative must be an actual evidence-seed voxel')
            target=pair[1] if seed==pair[0] else pair[0]
            lo=np.asarray(point)-64;hi=np.asarray(point)+64
            if (lo<0).any() or (hi>np.asarray(shape)).any():raise ValueError('Complete original SENSE crop required')
            crop=segmentation[tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))]
            positions=np.argwhere(crop==target)
            if not len(positions):raise ValueError('Proposed target absent from original SENSE crop')
            # A nearest actual target voxel is at most64 away on each axis.
            # Its midpoint with the actual seed places both in the FFN69³
            # analysis field and gives each a context-safe seed in101³.
            nearest=positions[np.argmin(np.sum((positions-64)**2,axis=1))]+lo
            point=((np.asarray(point)+nearest)//2).tolist()
        rows.append(dict(pair=[edge['source'], edge['target']],center_zyx=point))
    return dict(schema='shared-sense-ffn-decision-points/v1',
        base_sha256=base_sha256, decisions=rows, ground_truth_used=False,
        paper_comparable=False, independent_watershed_decisions=False,
        center_rule='Midpoint of SENSE representative and nearest actual target voxel in original128-cube; stable ZYX ties',
        scope='Shared SENSE proposals; candidate-controlled FFN/SHAPE ablation')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('edges', 'base', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    import numpy as np
    if a.output.exists():
        raise FileExistsError(a.output)
    base = np.load(a.base, mmap_mode='r', allow_pickle=False)
    if base.ndim != 3 or base.dtype.kind != 'u':
        raise ValueError('Unsigned ZYX base required')
    result = build(json.loads(a.edges.read_text()), sha256(a.base), base.shape, base)
    result['candidates_sha256'] = sha256(a.edges)
    write_json(a.output, result)


if __name__ == '__main__':
    main()
