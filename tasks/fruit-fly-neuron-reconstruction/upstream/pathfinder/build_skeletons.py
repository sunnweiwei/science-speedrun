"""Freeze local GT skeletons for a bounded ROI, independently of predictions.

This is an explicitly local Kimimaro/TEASAR protocol, not the paper evaluator.
Zero/invalid annotation is unknown, never foreground or a negative example.
ROI boundaries truncate paths; no endpoint erosion or paper resampling occurs.
"""
import argparse
from contextlib import contextmanager
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

import numpy as np

from dataset_contract import validate_manifest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import external, sha256, write_json


TEASAR = dict(scale=1.5, const=300, pdrf_scale=100000, pdrf_exponent=4,
              soma_acceptance_threshold=3500, soma_detection_threshold=750,
              soma_invalidation_const=300, soma_invalidation_scale=2)


@contextmanager
def compatible_osteoid_merge():
    """Work around osteoid 0.7.3's uint32 += int64 NumPy 1.x failure.

    Promote only the merge offset arithmetic, restore inputs/method afterward,
    and retain original uint32 result topology. No environment files are edited.
    """
    from osteoid import Skeleton
    descriptor = Skeleton.__dict__['simple_merge']
    original = Skeleton.simple_merge

    def merge(cls, skeletons):
        skeletons = list(skeletons)
        old_edges = [s.edges for s in skeletons]
        try:
            for s in skeletons:
                s.edges = s.edges.astype(np.int64)
            result = original(skeletons)
            if result.edges.size and result.edges.max() > np.iinfo(np.uint32).max:
                raise ValueError('Skeleton exceeds uint32 index range')
            result.edges = result.edges.astype(np.uint32)
            return result
        finally:
            for s, edges in zip(skeletons, old_edges):
                s.edges = edges

    if version('osteoid') == '0.7.3':
        Skeleton.simple_merge = classmethod(merge)
    try:
        yield
    finally:
        Skeleton.simple_merge = descriptor


def convert_skeletons(skeletons, labels, origin_nm, voxel_size_nm):
    """Kimimaro coordinates follow array axes and locate voxel indices, not centers."""
    spacing = np.asarray(voxel_size_nm, dtype=float)
    origin = np.asarray(origin_nm, dtype=float)
    nodes, edges = [], []
    covered = set()
    for label, skeleton in sorted(skeletons.items()):
        label = int(label)
        vertices = np.asarray(skeleton.vertices, dtype=float)
        if vertices.size == 0:
            continue
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
            raise ValueError('Invalid skeleton vertices')
        indices = np.rint(vertices / spacing).astype(np.int64)
        if (not np.allclose(vertices / spacing, indices, atol=1e-4) or
            (indices < 0).any() or (indices >= labels.shape).any() or
            not (labels[tuple(indices.T)] == label).all()):
            raise ValueError('Skeleton vertices leave their valid GT instance')
        offset = len(nodes)
        physical = origin + vertices + spacing / 2
        nodes.extend(dict(id=offset+i, skeleton_id=label, position_nm=p.tolist())
                     for i, p in enumerate(physical))
        for u, v in np.asarray(skeleton.edges).reshape(-1, 2):
            u, v = int(u), int(v)
            if u == v or min(u, v) < 0 or max(u, v) >= len(vertices):
                raise ValueError('Invalid skeleton edge')
            # TEASAR emits adjacent voxel paths. Reject unexpected jumps rather
            # than evaluating a shortcut through unknown annotation.
            if np.max(np.abs(indices[u] - indices[v])) > 1:
                raise ValueError('Nonadjacent skeleton edge needs an explicit resampling protocol')
            edges.append([offset+u, offset+v])
        covered.add(label)
    return dict(axes='zyx', coordinate_units='nm', nodes=nodes, edges=edges,
                protocol='local-kimimaro-teasar/v1', paper_protocol_verified=False), covered


def build(manifest_path, block, split, roi, output, max_voxels=10_000_000):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest, base_dir=manifest_path.parent)
    rows = [r for r in manifest['splits'][split] if r['name'] == block]
    if len(rows) != 1:
        raise ValueError('Choose one block from the requested frozen split')
    row = rows[0]
    bounds = np.asarray(row['dense_core_zyx'] if roi is None else np.asarray(roi).reshape(3, 2))
    core = np.asarray(row['dense_core_zyx'])
    if (bounds.dtype.kind not in 'ui' or bounds.shape != (3, 2) or
        (bounds[:, 0] < core[:, 0]).any() or (bounds[:, 1] > core[:, 1]).any() or
        (bounds[:, 1] <= bounds[:, 0]).any()):
        raise ValueError('ROI must be wholly within the fixed annotation core')
    shape = bounds[:, 1] - bounds[:, 0]
    if np.prod(shape) > max_voxels or max_voxels <= 0:
        raise ValueError('ROI exceeds CPU skeletonization voxel budget')
    folder = (manifest_path.parent / row['folder']).resolve()
    hashes = {}
    for name in ('labels.npy', 'valid.npy'):
        hashes[name] = sha256(folder/name)
        if hashes[name] != row['files'][name]:
            raise ValueError('GT input hash mismatch')
    slices = tuple(slice(int(lo), int(hi)) for lo, hi in bounds)
    labels = np.array(np.load(folder/'labels.npy', mmap_mode='r', allow_pickle=False)[slices])
    valid = np.asarray(np.load(folder/'valid.npy', mmap_mode='r', allow_pickle=False)[slices])
    if labels.dtype.kind not in 'ui' or valid.dtype != np.bool_:
        raise ValueError('Require integer instances and boolean annotation mask')
    labels[~valid] = 0
    spacing = np.asarray(manifest['voxel_size_nm'], dtype=float)
    origin = np.asarray(row['origin_nm']) + bounds[:, 0]*spacing
    import kimimaro
    started = time.monotonic()
    with compatible_osteoid_merge():
        skeletons = kimimaro.skeletonize(labels, teasar_params=dict(TEASAR), anisotropy=spacing,
            dust_threshold=0, progress=False, fix_branching=True, fix_borders=True,
            parallel=1, fill_holes=False, fix_avocados=False)
    truth, covered = convert_skeletons(skeletons, labels, origin, spacing)
    if not truth['edges']:
        raise ValueError('ROI has no positive skeleton path length')
    expected = set(map(int, np.unique(labels))) - {0}
    missing = sorted(expected-covered)
    # Small/singleton components may produce no paths. Preserve and report all
    # covered nodes, and disclose any entirely unrepresented labels explicitly.
    metadata = dict(axes='zyx', coordinate_convention='voxel_outer_corner',
        origin_nm=origin.tolist(), voxel_size_nm=spacing.tolist(), shape=shape.tolist(),
        roi_zyx=bounds.reshape(-1).tolist(), block=block, split=split,
        manifest_sha256=sha256(manifest_path), paper_comparable=False)
    receipt = dict(protocol=truth['protocol'], manifest_sha256=sha256(manifest_path),
        input_sha256=hashes, block=block, split=split, roi_zyx=metadata['roi_zyx'],
        kimimaro_version=version('kimimaro'), teasar_params=TEASAR, dust_threshold=0,
        osteoid_version=version('osteoid'), osteoid_merge_int64_compat=version('osteoid')=='0.7.3',
        fix_branching=True, fix_borders=True, fill_holes=False, parallel=1,
        voxel_center_offset=True, node_spacing='native skeleton voxel path; not paper resampling',
        boundary_policy='retain ROI-truncated endpoints; no endpoint erosion',
        unknown_policy='invalid or zero GT excluded before skeletonization; no GT provided to inference',
        valid_foreground_voxels=int((labels>0).sum()), unknown_voxels=int((labels==0).sum()),
        gt_instances=len(expected), represented_instances=len(covered),
        missing_instance_ids=missing, nodes=len(truth['nodes']), edges=len(truth['edges']),
        elapsed_seconds=time.monotonic()-started, paper_protocol_verified=False,
        builder_sha256=sha256(Path(__file__)))
    out = external(output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'skeletons.json', truth)
    write_json(out/'metadata.json', metadata)
    receipt['skeletons_sha256'] = sha256(out/'skeletons.json')
    receipt['metadata_sha256'] = sha256(out/'metadata.json')
    write_json(out/'receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--block', required=True)
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--roi', type=int, nargs=6)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-voxels', type=int, default=10_000_000)
    args = parser.parse_args()
    print(json.dumps(build(args.manifest, args.block, args.split, args.roi,
                           args.output, args.max_voxels), indent=2))


if __name__ == '__main__':
    main()
