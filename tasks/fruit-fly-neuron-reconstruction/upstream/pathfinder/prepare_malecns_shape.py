"""Build full-context SHAPE training surfaces from frozen MaleCNS geometry.

Uses published 32nm labels, 10um radius and 8192-point model inputs. Selects
identities before model training; validation/test IDs cannot enter training.
Initial negatives use the existing translated-instance recipe, not paper hard
negatives. No inference results or test-quality scores are read.
"""
import os
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import find_objects
from skimage.measure import marching_cubes
from dataset_contract import sha256
from fetch_axonem import reserve


def prepare(root, audit_path, output):
    root, output = root.resolve(), output.resolve()
    if not str(root).startswith(os.environ.get('PATHFINDER_SCRATCH_ROOT', '/scratch').rstrip('/') + '/') or root not in output.parents:
        raise ValueError('Use existing Xtemp budget root')
    if output.exists():
        raise ValueError('Refusing to overwrite meshes')
    audit = json.loads(audit_path.read_text())
    reserve(root, 8_000_000_000)
    output.mkdir(parents=True)
    rows = []; selections = {}
    for record in audit['records']:
        split = record['split']
        if split == 'test':
            continue
        path = audit_path.parent/record['path']
        if sha256(path) != record['sha256']:
            raise ValueError('Context label hash mismatch')
        seg = np.load(path, mmap_mode='r')
        if record['voxel_size_nm'] != [32]*3:
            raise ValueError('Expected declared 32nm source geometry')
        # >10um margin also protects the half-voxel marching-cubes boundary.
        margin = 314
        if min(seg.shape) <= 2*margin:
            raise ValueError('No complete 10um context')
        inner = np.asarray(seg[tuple(slice(margin,n-margin) for n in seg.shape)])
        ids, counts = np.unique(inner, return_counts=True)
        eligible = (ids > 0) & (counts >= 64)
        if split == 'train':
            eligible &= ~np.isin(ids, np.asarray(audit['excluded_training_ids'], dtype=ids.dtype))
        candidates = ids[eligible]
        rng = np.random.default_rng(20260928 if split == 'train' else 20260929)
        limit = 256 if split == 'train' else 128
        chosen = np.sort(rng.choice(candidates, min(limit,len(candidates)), replace=False))
        if len(chosen) < 2:
            raise ValueError('Too few full-context identities')
        selections[split] = dict(candidate_ids=len(candidates), selected_ids=list(map(int,chosen)))
        # Compact only chosen IDs for one streaming bbox pass; remove temp after use.
        temp = output/(split+'-index.npy')
        compact = np.lib.format.open_memmap(temp, mode='w+', dtype=np.uint16, shape=seg.shape)
        for z in range(0,seg.shape[0],16):
            block = np.asarray(seg[z:z+16]); idx = np.searchsorted(chosen,block)
            safe = np.minimum(idx,len(chosen)-1)
            compact[z:z+16] = np.where((idx<len(chosen)) & (chosen[safe]==block),safe+1,0)
        compact.flush()
        boxes = find_objects(compact, max_label=len(chosen))
        for sid, box in zip(chosen, boxes):
            if box is None:
                raise ValueError('Selected ID vanished')
            lo = np.array([s.start for s in box])
            mask = np.asarray(seg[box] == sid)
            vertices, faces, _, _ = marching_cubes(np.pad(mask,1), level=.5, spacing=(.032,)*3)
            vertices += (lo-1)*.032
            extent = np.array(seg.shape)*.032
            centers = np.where(((vertices >= 10) & (vertices <= extent-10)).all(axis=1))[0]
            if not len(centers):
                raise ValueError('Selected identity lacks full-context surface centers')
            if len(centers) > 128:
                centers = rng.choice(centers,128,replace=False)
            dest = output/(split+'-'+str(int(sid))+'.npz')
            np.savez_compressed(dest, vertices_um=vertices.astype(np.float32),
                                faces=faces.astype(np.int32), centers=centers)
            rows.append(dict(id=int(sid), split=split, path=dest.name, sha256=sha256(dest),
                vertices=len(vertices), faces=len(faces), full_context_centers=len(centers)))
            print(json.dumps(dict(split=split, id=int(sid), vertices=len(vertices),
                                  meshes_completed=sum(r['split']==split for r in rows))),flush=True)
        del compact; temp.unlink()
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Identity crosses shape train/validation split')
    manifest = dict(source_sha256=sha256(audit_path), radius_um=10, points_per_model_input=8192,
        axes='zyx', coordinate_units='um', rows=rows, selections=selections,
        voxel_size_nm=[32]*3, paper_comparable=False, sparse_preview_only=False,
        adaptation='Published MaleCNS full-context surfaces; synthetic translated distinct-instance false joins; not paper SENSE hard negatives',
        eligible_instances={s:sum(r['split']==s for r in rows) for s in ('train','validation')},
        adapter_sha256=sha256(__file__), training_not_started=True)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','audit','output'):
        parser.add_argument('--'+name, required=True, type=Path)
    a = parser.parse_args(); prepare(a.root,a.audit,a.output)
