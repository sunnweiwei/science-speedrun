"""Evaluation-only edge continuity by inference tile boundary; not a causal ablation."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from topology import sample
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import external, sha256, write_json


def diagnose(truth, labels, tile_origin_nm, tile_extent_nm):
    nodes = truth['nodes']
    positions = np.asarray([n['position_nm'] for n in nodes], dtype=float)
    labels = np.asarray(labels)
    origin = np.asarray(tile_origin_nm, dtype=float)
    extent = np.asarray(tile_extent_nm, dtype=float)
    if (positions.shape != (len(nodes), 3) or not len(nodes) or
            not np.isfinite(positions).all() or origin.shape != (3,) or
            extent.shape != (3,) or not np.isfinite(origin).all() or
            not np.isfinite(extent).all() or (extent <= 0).any()):
        raise ValueError('Nonempty finite physical geometry and positive tile extent required')
    if labels.shape != (len(nodes),) or labels.dtype.kind not in 'ui' or (labels < 0).any():
        raise ValueError('Aligned nonnegative integer node assignments required')
    lookup = {n['id']: i for i, n in enumerate(nodes)}
    if len(lookup) != len(nodes):
        raise ValueError('Duplicate skeleton node ID')
    cells = np.floor((positions - origin) / extent).astype(np.int64)
    groups = {name: {state: dict(edges=0, length_nm=0.) for state in
              ('connected', 'different_nonzero_labels', 'omitted_endpoint')}
              for name in ('cross_tile', 'within_tile')}
    seen = set()
    for u, v in truth['edges']:
        key = frozenset((u, v))
        if u not in lookup or v not in lookup or u == v or key in seen:
            raise ValueError('Invalid or duplicate skeleton edge')
        seen.add(key)
        i, j = lookup[u], lookup[v]
        if nodes[i]['skeleton_id'] != nodes[j]['skeleton_id']:
            raise ValueError('Skeleton edge crosses ground-truth identities')
        group = 'cross_tile' if np.any(cells[i] != cells[j]) else 'within_tile'
        state = ('omitted_endpoint' if labels[i] == 0 or labels[j] == 0 else
                 'connected' if labels[i] == labels[j] else 'different_nonzero_labels')
        row = groups[group][state]
        row['edges'] += 1
        row['length_nm'] += float(np.linalg.norm(positions[i] - positions[j]))
    for row in groups.values():
        assigned = row['connected']['edges'] + row['different_nonzero_labels']['edges']
        row['assigned_edges'] = assigned
        row['split_fraction_among_assigned_edges'] = (
            row['different_nonzero_labels']['edges'] / assigned if assigned else None)
    return dict(schema='tile-boundary-diagnostic/v1', groups=groups,
                tile_origin_nm=origin.tolist(), tile_extent_nm=extent.tolist(),
                nodes=len(nodes), edges=len(seen),
                scope='Evaluation only. Edge continuity is not nERL. Boundary association does not '
                      'establish the causal gain from stitching; connected edges may be false merges.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('skeletons', 'segmentation', 'metadata', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--tile-origin-nm', nargs=3, type=float, required=True)
    p.add_argument('--tile-extent-nm', nargs=3, type=float, required=True)
    a = p.parse_args()
    truth = json.loads(a.skeletons.read_text())
    meta = json.loads(a.metadata.read_text())
    if truth.get('axes') != 'zyx' or truth.get('coordinate_units') != 'nm' or meta.get('axes') != 'zyx':
        raise ValueError('Explicit physical ZYX coordinates required')
    segmentation = np.load(a.segmentation, mmap_mode='r', allow_pickle=False)
    if list(segmentation.shape) != meta['shape']:
        raise ValueError('Prediction shape differs from metadata')
    labels = sample(segmentation, [n['position_nm'] for n in truth['nodes']],
                    meta['origin_nm'], meta['voxel_size_nm'])
    result = diagnose(truth, labels, a.tile_origin_nm, a.tile_extent_nm)
    result['inputs'] = {name: dict(path=str(getattr(a, name).resolve()),
                                  sha256=sha256(getattr(a, name)))
                        for name in ('skeletons', 'segmentation', 'metadata')}
    result['script_sha256'] = sha256(Path(__file__))
    write_json(external(a.output), result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
