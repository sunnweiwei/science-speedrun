"""Fetch published 32nm GT geometry for a frozen spatial split, CPU only.

The 16nm image and 512-cube scoring GT are reused read-only. Coarse GT is for
SHAPE surfaces and exposure auditing, never substituted for evaluation labels.
"""
import os
import argparse
import json
from pathlib import Path
import numpy as np
from dataset_contract import sha256
from fetch_malecns import pinned_metadata
from fetch_axonem import reserve


def prepare(plan_path, root, output):
    import tensorstore as ts
    root, output = root.resolve(), output.resolve()
    if not str(root).startswith(os.environ.get('PATHFINDER_SCRATCH_ROOT', '/scratch').rstrip('/') + '/') or root not in output.parents:
        raise ValueError('Output must be under Xtemp budget root')
    plan = json.loads(plan_path.read_text())
    if plan['schema'] != 'malecns-spatial-comparison/v1':
        raise ValueError('Unexpected plan')
    if output.exists():
        raise ValueError('Output already exists')
    reserve(root, 20_000_000_000)
    output.mkdir(parents=True)
    for entry in plan['metadata'].values():
        pinned_metadata(entry)
    labels = ts.open(dict(driver='neuroglancer_precomputed',
        kvstore=dict(driver='http', base_url=plan['segmentation_url']),
        scale_metadata={'resolution': [32, 32, 32]}, fill_missing_data_reads=False),
        context=ts.Context({'cache_pool': {'total_bytes_limit': 268435456}})).result()
    records = []
    sets = {}
    for row in plan['regions']:
        existing = root / row['prepared_relative_path']
        image_manifest = existing/'image/manifest.json'
        if sha256(image_manifest) != row['image_manifest_sha256']:
            raise ValueError('Image manifest changed')
        image = json.loads(image_manifest.read_text())
        origin = np.array(image['origin_nm'], dtype=np.int64)
        end = origin + np.array(image['shape']) * 16
        # Inner grid only: every stored cell has actual image context.
        start32 = (origin + 31) // 32
        end32 = end // 32
        shape = end32 - start32
        dest = output/row['split']; dest.mkdir()
        array = np.lib.format.open_memmap(dest/'labels32.npy', mode='w+',
            dtype=np.uint64, shape=tuple(shape))
        ids = set()
        z, y, x = map(int, start32)
        dz, dy, dx = map(int, shape)
        for lo in range(0, dz, 32):
            hi = min(dz, lo+32)
            block = labels[x:x+dx, y:y+dy, z+lo:z+hi, 0].read().result().transpose(2,1,0)
            array[lo:hi] = block
            ids.update(map(int, np.unique(block)))
            array.flush()
            print(json.dumps(dict(split=row['split'], completed_z=hi, total_z=dz)), flush=True)
        del array
        core = existing/'reference/labels.npy'
        if sha256(core) != row['core_labels_sha256']:
            raise ValueError('Core labels changed')
        ids.update(map(int, np.unique(np.load(core, mmap_mode='r'))))
        ids.discard(0); sets[row['split']] = ids
        records.append(dict(split=row['split'], path=str((dest/'labels32.npy').relative_to(output)),
            shape_zyx=shape.tolist(), origin_nm=(start32*32).tolist(), voxel_size_nm=[32]*3,
            sha256=sha256(dest/'labels32.npy'), observed_ids=sorted(ids)))
    shared = sets['train'] & (sets['validation'] | sets['test'])
    audit = dict(schema='malecns-context-audit/v1', plan_sha256=sha256(plan_path),
        adapter_sha256=sha256(__file__), records=records, excluded_training_ids=sorted(shared),
        eligible_training_ids=sorted(sets['train']-shared),
        validation_test_shared_ids=sorted(sets['validation'] & sets['test']),
        scope='Published IDs observed in 32nm contexts plus exact 16nm scored cores; not a complete biological neuron identity audit',
        training_not_started=True, reference_grid_unchanged=True)
    (output/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan','root','output'):
        parser.add_argument('--'+name, required=True, type=Path)
    a = parser.parse_args(); prepare(a.plan, a.root, a.output)
