"""Validate the portable EM dataset boundary without importing training/GPU code.

Adapters emit pathfinder-dataset/v1; existing axonem-spatial16/v1 is accepted.
Axes are ZYX, lengths nm, origin_nm is the outer corner of voxel zero, and
voxel centers are origin_nm + (index + 0.5) * voxel_size_nm. dense_core_zyx
uses half-open voxel bounds. Context halos may overlap; supervised cores may
not overlap across splits in the same volume_id (default: dataset_id).

This validates an input contract, NOT official-paper evaluation equivalence.
Use --check-files for mmap/header/center checks and --verify-hashes for full
streamed integrity checking. Neither mode loads complete volumes into RAM.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


class ContractError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ContractError(message)


def triple(value, name, positive=False, integer=False):
    require(isinstance(value, (list, tuple)) and len(value) == 3, f'{name}: expected ZYX triple')
    for item in value:
        require(type(item) in (int, float) and math.isfinite(item), f'{name}: non-finite/non-numeric value')
        require(not positive or item > 0, f'{name}: must be positive')
        require(not integer or type(item) is int, f'{name}: must be integers')
    return value


def digest(value, name):
    require(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value), f'{name}: invalid SHA256')


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def validate_manifest(manifest, *, base_dir='.', check_files=False, verify_hashes=False):
    """Return a small validation receipt, or raise ContractError on violations.

    Relative folders resolve against the manifest directory. Generic adapters
    must declare coordinate_convention='voxel_outer_corner'; legacy AxonEM
    already uses that convention. Split identity is spatial, not neuron identity.
    """
    require(isinstance(manifest, dict), 'manifest must be an object')
    schema = manifest.get('schema')
    require(schema in ('axonem-spatial16/v1', 'pathfinder-dataset/v1'), 'unsupported dataset schema')
    require(isinstance(manifest.get('dataset_id'), str) and manifest['dataset_id'].strip(), 'dataset_id required')
    require(manifest.get('axes') == 'zyx', 'axes must be zyx')
    require(manifest.get('coordinate_convention', 'voxel_outer_corner' if schema == 'axonem-spatial16/v1' else None)
            == 'voxel_outer_corner', 'coordinate_convention must be voxel_outer_corner')
    spacing = triple(manifest.get('voxel_size_nm'), 'voxel_size_nm', positive=True)
    native = triple(manifest.get('native_voxel_size_nm'), 'native_voxel_size_nm', positive=True)
    if spacing != native:
        require(isinstance(manifest.get('interpolation'), str) and manifest['interpolation'].strip(),
                'resampled data must describe interpolation; grid spacing is not native resolution')
    digest(manifest.get('source_receipt_sha256'), 'source_receipt_sha256')
    require(isinstance(manifest.get('split_kind'), str) and manifest['split_kind'].strip(), 'split_kind required')
    # Author-data identity alone never establishes parity. An independent protocol
    # audit is required outside this data contract before reporting a paper gap.
    require(manifest.get('paper_comparable') is False,
            'dataset contract cannot certify paper parity; paper_comparable must be false')
    splits = manifest.get('splits')
    expected = {'train', 'validation'} if manifest.get('visibility') == 'development' else {'train', 'validation', 'test'}
    require(isinstance(splits, dict) and set(splits) == expected,
            'require exactly the declared development or full split set')
    names, folders, cores = set(), set(), []
    for split, rows in splits.items():
        require(isinstance(rows, list) and rows, f'{split}: nonempty record list required')
        for row in rows:
            require(isinstance(row, dict), f'{split}: record must be an object')
            name = row.get('name')
            require(isinstance(name, str) and name.strip() and name not in names, 'record names must be unique')
            names.add(name)
            require(isinstance(row.get('folder'), str) and row['folder'].strip(), f'{name}: folder required')
            folder = (Path(base_dir) / row['folder']).resolve()
            require(folder not in folders, f'{name}: folder reused across records/splits')
            folders.add(folder)
            shape = triple(row.get('shape'), f'{name}.shape', positive=True, integer=True)
            origin = triple(row.get('origin_nm'), f'{name}.origin_nm')
            bounds = row.get('dense_core_zyx')
            require(isinstance(bounds, list) and len(bounds) == 3, f'{name}: dense core required')
            for axis, bound in enumerate(bounds):
                require(isinstance(bound, list) and len(bound) == 2 and all(type(v) is int for v in bound)
                        and 0 <= bound[0] < bound[1] <= shape[axis], f'{name}: invalid dense core bounds')
            physical = [(o + b[0] * s, o + b[1] * s) for o, b, s in zip(origin, bounds, spacing)]
            frame = row.get('volume_id', manifest['dataset_id'])
            require(isinstance(frame, str) and frame.strip(), f'{name}: volume_id must be nonempty')
            for other_split, other_frame, other_name, other in cores:
                if other_split != split and other_frame == frame:
                    require(not all(max(a[0], b[0]) < min(a[1], b[1]) for a, b in zip(physical, other)),
                            f'supervised core leakage between {other_name} and {name}')
            cores.append((split, frame, name, physical))
            for key in ('raw_source_sha256', 'labels_source_sha256'):
                digest(row.get(key), f'{name}.{key}')
            files = row.get('files')
            require(isinstance(files, dict) and {'raw.npy', 'labels.npy', 'valid.npy', 'centers.npy'} <= set(files),
                    f'{name}: required array hashes missing')
            for filename, expected in files.items():
                require(Path(filename).name == filename and filename not in ('.', '..'), 'file keys must be plain filenames')
                digest(expected, f'{name}.{filename}')
                if verify_hashes:
                    require(sha256(folder / filename) == expected, f'{name}.{filename}: SHA256 mismatch')
            if check_files or verify_hashes:
                _check_arrays(folder, shape, bounds, name)
    return {'contract_valid': True, 'dataset_id': manifest['dataset_id'],
            'counts': {s: len(v) for s, v in splits.items()},
            'arrays_checked': check_files or verify_hashes, 'hashes_verified': verify_hashes,
            'paper_protocol_verified': False, 'neuron_disjointness_verified': False}


def _check_arrays(folder, shape, bounds, name):
    import numpy as np
    arrays = {key: np.load(folder / (key + '.npy'), mmap_mode='r', allow_pickle=False)
              for key in ('raw', 'labels', 'valid', 'centers')}
    for key in ('raw', 'labels', 'valid'):
        require(arrays[key].shape == tuple(shape), f'{name}.{key}: array shape mismatch')
    require(arrays['raw'].dtype == np.uint8, f'{name}: raw must be uint8')
    require(arrays['labels'].dtype.kind == 'u', f'{name}: labels must be unsigned integers')
    require(arrays['valid'].dtype == np.bool_, f'{name}: valid mask must be boolean')
    centers = arrays['centers']
    require(centers.ndim == 2 and centers.shape[1] == 3 and len(centers) > 0 and centers.dtype.kind in 'iu',
            f'{name}: centers must be nonempty integer N x 3')
    for offset in range(0, len(centers), 65536):
        batch = centers[offset:offset + 65536]
        require(bool(((batch >= 0) & (batch < np.asarray(shape))).all()), f'{name}: center outside volume')
        require(bool(((batch >= np.asarray(bounds)[:, 0]) & (batch < np.asarray(bounds)[:, 1])).all()),
                f'{name}: center outside supervised core')
        index = tuple(batch.T)
        require(bool(arrays['valid'][index].all()) and bool((arrays['labels'][index] > 0).all()),
                f'{name}: center must have valid positive instance label')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--check-files', action='store_true')
    parser.add_argument('--verify-hashes', action='store_true')
    args = parser.parse_args()
    try:
        report = validate_manifest(json.loads(args.manifest.read_text()), base_dir=args.manifest.parent,
                                   check_files=args.check_files, verify_hashes=args.verify_hashes)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'dataset contract rejected: {exc}\n')
    report['manifest_sha256'] = sha256(args.manifest)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
