"""Audit downloaded FFN examples and resolve NeuroProof supervoxel mappings.

CPU only. Does not invent physical spacing, train/test separation, or a score.
Outputs are adapter-owned new directories; source files remain unchanged.
"""
import os
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from dataset_contract import sha256


def map_bodies(stack, transforms):
    if stack.dtype.kind not in 'iu' or transforms.dtype.kind not in 'iu':
        raise ValueError('Integer IDs required')
    if stack.min() < 0 or transforms.min() < 0:
        raise ValueError('Negative IDs are invalid')
    if transforms.ndim != 2 or transforms.shape[1] != 2:
        raise ValueError('Expected source/body pairs')
    order = np.argsort(transforms[:, 0])
    source, target = transforms[order].T
    if len(np.unique(source)) != len(source):
        raise ValueError('Duplicate mapping keys')
    ids, inverse = np.unique(stack, return_inverse=True)
    index = np.searchsorted(source, ids)
    if np.any(index >= len(source)) or not np.array_equal(source[index], ids):
        raise ValueError('Unmapped supervoxels')
    if 0 in source and target[np.searchsorted(source, 0)] != 0:
        raise ValueError('Background mapped to foreground')
    return target[index][inverse].reshape(stack.shape).astype(np.uint64)


def inspect(root):
    root = Path(root).resolve()
    # Deliberately keep data off home, including symlink-resolved locations.
    if not str(root).startswith(os.environ.get('PATHFINDER_SCRATCH_ROOT', '/scratch').rstrip('/') + '/'):
        raise ValueError('Use Xtemp for data and outputs')
    download = json.loads((root / 'downloads.json').read_text())
    hashes = {r['path']: r['sha256'] for r in download}
    output = root / 'resolved-labels'
    if output.exists():
        raise ValueError('Refusing to overwrite an existing audit')
    for rel, expected in hashes.items():
        if sha256(root / rel) != expected:
            raise ValueError('Download hash mismatch: ' + rel)
    output.mkdir()
    records = []
    for folder in ('validation_sample', 'training_sample2'):
        with h5py.File(root / folder / 'groundtruth.h5') as f:
            stack, transforms = f['stack'][:], f['transforms'][:]
        with h5py.File(root / folder / 'grayscale_maps.h5') as f:
            if f['raw'].shape != stack.shape or f['raw'].dtype != np.uint8:
                raise ValueError('Image/GT shape or dtype mismatch')
        mapped = map_bodies(stack, transforms)
        dest = output / (folder + '.npy')
        np.save(dest, mapped)
        records.append(dict(sample=folder, shape_zyx=list(stack.shape),
            source_supervoxels=int(len(np.unique(stack))),
            positive_bodies=int(len(np.unique(mapped[mapped > 0]))),
            nonidentity_mapping_rows=int(np.sum(transforms[:, 0] != transforms[:, 1])),
            labels_path=str(dest.relative_to(root)), labels_sha256=sha256(dest),
            voxel_size_nm=None, physical_origin_nm=None,
            physical_metadata_status='not encoded in HDF5; source verification required'))
    receipt = dict(schema='ffn-example-data-audit/v1', records=records,
        download_receipt_sha256=sha256(root/'downloads.json'),
        adapter_sha256=sha256(__file__), paper_comparable=False,
        split_frozen=False, inference_executed=False)
    (output/'audit.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    inspect(parser.parse_args().root)
