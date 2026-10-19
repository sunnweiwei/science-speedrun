"""Prepare spatially held-out MaleCNS FFN/SENSE crops from frozen public cores.

Held-out public IDs are unknown, not negative supervision, in training. The
published 16nm core labels remain intact on disk. No model output is inspected.
"""
import os
import argparse
import json
from pathlib import Path
import numpy as np
from dataset_contract import sha256, validate_manifest
from fetch_axonem import reserve


def supervision(labels, excluded):
    valid = labels > 0
    if excluded:
        valid &= ~np.isin(labels, np.asarray(sorted(excluded), dtype=labels.dtype))
    return valid


def sample_centers(labels, valid, margin=64, stride=8):
    if labels.shape != valid.shape or min(labels.shape) <= 2*margin:
        raise ValueError('Need matching arrays and full 128-cube centers')
    axes = [np.arange(margin, n-margin, stride) for n in labels.shape]
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    grid = grid[valid[tuple(grid.T)]]
    if not len(grid):
        raise ValueError('No eligible training centers')
    return grid.astype(np.int32)


def prepare(root, plan_path, audit_path, output):
    root, output = root.resolve(), output.resolve()
    if not str(root).startswith(os.environ.get('PATHFINDER_SCRATCH_ROOT', '/scratch').rstrip('/') + '/') or root not in output.parents:
        raise ValueError('Use existing Xtemp budget root')
    plan = json.loads(plan_path.read_text()); audit = json.loads(audit_path.read_text())
    if audit['plan_sha256'] != sha256(plan_path):
        raise ValueError('Context audit belongs to a different split')
    if output.exists():
        raise ValueError('Refusing to overwrite training data')
    reserve(root, 5_000_000_000)
    output.mkdir(parents=True)
    manifest = dict(schema='pathfinder-dataset/v1', dataset_id='malecns-domain-v1', axes='zyx',
        coordinate_convention='voxel_outer_corner', voxel_size_nm=[16]*3,
        native_voxel_size_nm=[8]*3, interpolation='Official 16nm image pyramid and published 16nm labels; no local interpolation',
        source_receipt_sha256=sha256(audit_path), split_kind='disjoint spatial context boxes; shared observed public IDs excluded from train',
        paper_comparable=False, splits={}, adapter_sha256=sha256(__file__),
        split_plan_sha256=sha256(plan_path), exclusion_scope=audit['scope'])
    seen = {}; boxes = []
    for row in plan['regions']:
        split = row['split']; source = root/row['prepared_relative_path']
        if sha256(source/'image/manifest.json') != row['image_manifest_sha256']:
            raise ValueError('Image manifest changed')
        if sha256(source/'reference/labels.npy') != row['core_labels_sha256']:
            raise ValueError('Reference changed')
        image = json.loads((source/'image/manifest.json').read_text())
        metadata = json.loads((source/'suite/metadata.json').read_text())
        origin = np.array(metadata['scored_core_origin_nm'], dtype=np.int64)
        delta = origin-np.array(image['origin_nm'])
        if (delta % 16).any() or image['voxel_size_nm'] != [16]*3:
            raise ValueError('Core and image grids are not aligned')
        lo = np.array(image['origin_nm']); hi = lo+np.array(image['shape'])*16
        for other_lo, other_hi in boxes:
            if ((np.maximum(lo,other_lo) < np.minimum(hi,other_hi))).all():
                raise ValueError('Full image contexts overlap across splits')
        boxes.append((lo,hi))
        if sha256(source/'image/raw.npy') != image['files']['raw.npy']:
            raise ValueError('Raw image changed')
        start = delta//16
        raw = np.load(source/'image/raw.npy', mmap_mode='r')
        gt = np.load(source/'reference/labels.npy', mmap_mode='r')
        if gt.shape != (512,)*3:
            raise ValueError('Frozen 512-cube core required')
        slices = tuple(slice(int(a), int(a)+n) for a,n in zip(start,gt.shape))
        crop = np.array(raw[slices])
        excluded = set(audit['excluded_training_ids']) if split == 'train' else set()
        valid = supervision(gt, excluded)
        centers = sample_centers(gt, valid)
        seen[split] = set(map(int, np.unique(gt[valid])))
        folder = output/split; folder.mkdir()
        for name, array in [('raw',crop), ('labels',gt), ('valid',valid), ('centers',centers)]:
            np.save(folder/(name+'.npy'), array)
        record = dict(name='malecns-'+split, folder=split, volume_id='malecns-v1.0',
            shape=list(gt.shape), origin_nm=origin.tolist(), dense_core_zyx=[[0,512]]*3,
            raw_source_sha256=image['files']['raw.npy'], labels_source_sha256=row['core_labels_sha256'],
            files={f.name:sha256(f) for f in folder.glob('*.npy')},
            center_count=len(centers), eligible_positive_ids=len(seen[split]),
            ignored_voxel_fraction=float(1-valid.mean()))
        manifest['splits'][split] = [record]
        print(json.dumps(dict(split=split, centers=len(centers), instances=len(seen[split]))), flush=True)
    if seen['train'] & (seen['validation'] | seen['test']):
        raise ValueError('Training supervision contains observed held-out IDs')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    receipt = validate_manifest(manifest, base_dir=output, verify_hashes=True)
    receipt.update(manifest_sha256=sha256(output/'manifest.json'),
        supervised_train_vs_heldout_public_ids_disjoint=True,
        biological_neuron_disjointness_verified=False,
        contexts_physically_disjoint=True, test_used_for_model_selection=False)
    (output/'preparation.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','plan','audit','output'):
        parser.add_argument('--'+name, required=True, type=Path)
    a = parser.parse_args(); prepare(a.root, a.plan, a.audit, a.output)
