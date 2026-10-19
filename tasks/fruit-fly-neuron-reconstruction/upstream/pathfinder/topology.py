"""Local skeleton split/merge diagnostics with PATHFINDER's node thresholds.

This is not the unpublished author evaluator. nERL and paper-score parity are
intentionally not reported. Requires GT skeletons and explicit boundary policy.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import external, sha256, write_json


def counts(skeleton_ids, segment_ids, interior_ids, split_min_nodes=5, merge_min_nodes=10):
    if len(skeleton_ids) != len(segment_ids) or not len(skeleton_ids):
        raise ValueError('Nonempty aligned node assignments required')
    def positive(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and value > 0
    if not all(positive(x) for x in skeleton_ids):
        raise ValueError('Ground-truth skeleton IDs must be positive integers')
    if not all(positive(x) or (isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_)) and x == 0) for x in segment_ids):
        raise ValueError('Predicted IDs must be nonnegative integers; zero denotes omission')
    if not all(positive(x) for x in interior_ids):
        raise ValueError('Interior IDs must be positive integers')
    if not all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in [split_min_nodes, merge_min_nodes]):
        raise ValueError('Positive integer support thresholds required')
    pairs = Counter((int(g), int(p)) for g, p in zip(skeleton_ids, segment_ids) if p != 0)
    split_groups = defaultdict(set); merge_groups = defaultdict(set)
    for (gt, pred), n in pairs.items():
        if n >= split_min_nodes and pred in interior_ids: split_groups[gt].add(pred)
        if n >= merge_min_nodes: merge_groups[pred].add(gt)
    missing = sum(p == 0 for p in segment_ids)
    return dict(split_errors=sum(max(0, len(v)-1) for v in split_groups.values()),
        merge_errors=sum(max(0, len(v)-1) for v in merge_groups.values()),
        skeletons=len(set(skeleton_ids)), nodes=len(skeleton_ids), omitted_nodes=int(missing),
        omitted_node_fraction=float(missing/len(skeleton_ids)),
        split_min_nodes=split_min_nodes, merge_min_nodes=merge_min_nodes,
        nerl=None, paper_protocol_verified=False, author_model_replayed=False,
        scope='Local node-count diagnostic only; zero split/merge counts do not imply complete reconstruction')


def sample(segmentation, coordinates, origin, voxel_size):
    coordinates = np.asarray(coordinates, dtype=float)
    origin = np.asarray(origin, dtype=float); voxel_size = np.asarray(voxel_size, dtype=float)
    if (segmentation.ndim != 3 or segmentation.dtype.kind not in 'ui' or
        coordinates.ndim != 2 or coordinates.shape[1] != 3 or origin.shape != (3,) or voxel_size.shape != (3,) or
        not np.isfinite(coordinates).all() or not np.isfinite(origin).all() or
        not np.isfinite(voxel_size).all() or (voxel_size <= 0).any()):
        raise ValueError('Require integer 3D labels and finite physical ZYX coordinates')
    index = np.floor((coordinates-origin)/voxel_size).astype(np.int64)
    if (index < 0).any() or (index >= segmentation.shape).any():
        raise ValueError('GT nodes outside prediction: align domains explicitly, never silently drop nodes')
    return segmentation[tuple(index.T)]


def interior_segments(segmentation, margin):
    margin = np.asarray(margin)
    if margin.shape != (3,) or margin.dtype.kind not in 'ui' or (margin < 0).any() or (2*margin >= segmentation.shape).any():
        raise ValueError('Explicit integer ZYX voxel margin must leave a nonempty interior')
    # Full segmentation extent matters, not just the sampled skeleton nodes.
    ids = set()
    for z in range(int(margin[0]), int(segmentation.shape[0]-margin[0])):
        slab = segmentation[z, int(margin[1]):int(segmentation.shape[1]-margin[1]), int(margin[2]):int(segmentation.shape[2]-margin[2])]
        ids.update(map(int, np.unique(slab)))
    ids.discard(0)
    return ids


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['segmentation', 'metadata', 'skeletons', 'output']: p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--boundary-margin-voxels', nargs=3, type=int, required=True, metavar=('Z','Y','X'))
    a = p.parse_args(); out = external(a.output)
    metadata = json.loads(a.metadata.read_text()); truth = json.loads(a.skeletons.read_text())
    if metadata.get('axes') != 'zyx' or truth.get('axes') != 'zyx' or truth.get('coordinate_units') != 'nm':
        raise ValueError('Explicit ZYX axes and skeleton units nm required')
    nodes = truth['nodes']
    if len({n['id'] for n in nodes}) != len(nodes): raise ValueError('Duplicate skeleton node IDs')
    seg = np.load(a.segmentation, mmap_mode='r', allow_pickle=False)
    labels = sample(seg, [n['position_nm'] for n in nodes], metadata['origin_nm'], metadata['voxel_size_nm'])
    result = counts([n['skeleton_id'] for n in nodes], labels, interior_segments(seg, a.boundary_margin_voxels))
    result.update(protocol='local-pathfinder-style-node-counts/v1', boundary_margin_voxels=a.boundary_margin_voxels,
        sampling='floor', inputs={name:dict(path=str(getattr(a,name).resolve()), sha256=sha256(getattr(a,name))) for name in ['segmentation','metadata','skeletons']})
    write_json(out, result)
    print(json.dumps(result, indent=2))

if __name__ == '__main__': main()
