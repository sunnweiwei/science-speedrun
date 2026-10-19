"""Fetch one frozen MaleCNS context volume and build independent local GT.

Requires tensorstore==0.1.76. Never downloads the whole brain. Public source
metadata is pinned; missing chunks raise errors instead of becoming background.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import urllib.request

import numpy as np

from fetch_axonem import reserve
from import_continuous import import_image
from build_skeletons import TEASAR, compatible_osteoid_merge, convert_skeletons
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import external, sha256, write_json

CODE = Path(__file__).resolve().parent


def region_plan(plan, split=None):
    """Resolve a fetch request from the frozen spatial plan without old pilot settings."""
    if plan.get('schema') == 'pathfinder-malecns-evaluation/v1':
        if split is not None:
            raise ValueError('A split requires the frozen spatial comparison plan')
        return dict(plan)
    if plan.get('schema') != 'malecns-spatial-comparison/v1' or split not in ('train', 'validation', 'test'):
        raise ValueError('Spatial comparison plan requires an explicit train/validation/test split')
    rows = [row for row in plan['regions'] if row['split'] == split]
    if len(rows) != 1:
        raise ValueError('Exactly one frozen region per split required')
    row = rows[0]
    url = plan['metadata']['em_s1']['url']
    if not url.endswith('/attributes.json'):
        raise ValueError('Pinned N5 attributes URL required')
    return dict(schema='pathfinder-malecns-evaluation/v1', dataset='MaleCNS v1.0',
                region_id=row['region_id'], center_xyz_16nm=row['center_xyz_16nm'],
                core_size_voxels=512, em_url=url[:-len('attributes.json')],
                segmentation_url=plan['segmentation_url'], metadata=plan['metadata'],
                expected_image_manifest_sha256=row['image_manifest_sha256'],
                expected_core_labels_sha256=row['core_labels_sha256'],
                split=split, paper_comparable=False)


def geometry(plan):
    center = np.asarray(plan['center_xyz_16nm'])
    core = plan['core_size_voxels']
    if center.shape != (3,) or center.dtype.kind not in 'iu' or type(core) != int or core not in (256, 512):
        raise ValueError('Frozen integer XYZ center and 256/512-voxel core required')
    # Each scored voxel has >=10um inside the FFN prediction extent, which
    # itself has 16 input voxels of receptive-field padding on each side.
    halo = 625 + 16
    size = core + 2 * halo
    start = center - size // 2
    if (start < 0).any():
        raise ValueError('Negative source coordinate')
    return start, size, halo


def pinned_metadata(entry):
    with urllib.request.urlopen(entry['url'], timeout=60) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != entry['sha256']:
        raise ValueError('Public metadata changed: ' + entry['url'])
    return json.loads(content)


def fetch(plan_path, root, output, evaluator, split=None):
    import tensorstore as ts
    import kimimaro
    from importlib.metadata import version
    root, output = external(root), external(output)
    if root not in output.parents:
        raise ValueError('Output must be inside the original budget root')
    plan = region_plan(json.loads(plan_path.read_text()), split)
    start, size, halo = geometry(plan)
    core = plan['core_size_voxels']
    for entry in plan['metadata'].values():
        pinned_metadata(entry)
    # Includes temporary image staging and a conservative skeleton JSON reserve.
    reserve(root, 2 * size**3 + core**3 * 8 + 2_000_000_000)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'frozen-plan.json', plan)
    image = ts.open(dict(driver='n5', kvstore=dict(driver='http',
        base_url=plan['em_url']), fill_missing_data_reads=False),
        context=ts.Context({'cache_pool': {'total_bytes_limit': 268435456}})).result()
    labels = ts.open(dict(driver='neuroglancer_precomputed', kvstore=dict(driver='http',
        base_url=plan['segmentation_url']), scale_metadata={'resolution':[16,16,16]},
        fill_missing_data_reads=False),
        context=ts.Context({'cache_pool': {'total_bytes_limit': 268435456}})).result()
    if image.dtype.numpy_dtype != np.dtype('uint8') or labels.dtype.numpy_dtype != np.dtype('uint64'):
        raise ValueError('Unexpected source dtype')
    if (start + size > np.asarray(image.domain.exclusive_max)).any():
        raise ValueError('Context exceeds EM bounds')
    stage = output/'staging'; stage.mkdir()
    raw = np.lib.format.open_memmap(stage/'raw.npy', mode='w+', dtype=np.uint8, shape=(size,)*3)
    x, y, z = map(int, start)
    for offset in range(0, size, 64):
        end = min(offset+64, size)
        chunk = image[x:x+size, y:y+size, z+offset:z+end].read().result()
        raw[offset:end] = chunk.transpose(2,1,0)
        raw.flush()
        print(json.dumps(dict(phase='em', completed_z=end, total_z=size)), flush=True)
    del raw
    np.save(stage/'image_valid_z.npy', np.ones(size, dtype=bool))
    source = dict(schema='pathfinder-continuous-source/v1', dataset_id=plan.get('region_id', 'malecns-v1.0-scene-center'),
        axes='zyx', coordinate_convention='voxel_outer_corner', shape=[size]*3,
        origin_nm=(start[::-1]*16).tolist(), voxel_size_nm=[16]*3, native_voxel_size_nm=[8]*3,
        source='https://male-cns.janelia.org/download/', source_revision='MaleCNS v1.0; pinned public metadata',
        conversion='Official lossless N5 s1 pyramid at 16nm; XYZ transposed to ZYX; no intensity transform',
        image_valid_scope='All requested N5 chunks exist and decoded; biological/image defects are not exhaustively detected',
        ground_truth_included=False)
    for name in ('raw','image_valid_z'):
        source[name] = dict(path=name+'.npy', sha256=sha256(stage/(name+'.npy')))
    write_json(stage/'source.json', source)
    import_image(stage/'source.json', root, output/'image')
    if ('expected_image_manifest_sha256' in plan and
            sha256(output/'image/manifest.json') != plan['expected_image_manifest_sha256']):
        raise ValueError('Downloaded image differs from frozen spatial split')
    # Delete only adapter-owned duplicate staging arrays after verified copying.
    (stage/'raw.npy').unlink(); (stage/'image_valid_z.npy').unlink()
    core_start = start + halo
    cx, cy, cz = map(int, core_start)
    gt = labels[cx:cx+core,cy:cy+core,cz:cz+core,0].read().result().transpose(2,1,0).copy()
    reference = output/'reference'; reference.mkdir()
    np.save(reference/'labels.npy', gt)
    if ('expected_core_labels_sha256' in plan and
            sha256(reference/'labels.npy') != plan['expected_core_labels_sha256']):
        raise ValueError('Downloaded labels differ from frozen spatial split')
    print(json.dumps(dict(phase='reference', positive_voxels=int((gt>0).sum()),
        instances=int(len(np.unique(gt[gt>0]))))), flush=True)
    with compatible_osteoid_merge():
        skeletons = kimimaro.skeletonize(gt, teasar_params=dict(TEASAR), anisotropy=[16]*3,
            dust_threshold=0, progress=False, fix_branching=True, fix_borders=True,
            parallel=1, fill_holes=False, fix_avocados=False)
    truth, covered = convert_skeletons(skeletons, gt, core_start[::-1]*16, [16]*3)
    if not truth['edges']:
        raise ValueError('Preselected core has no evaluable reference path')
    suite = output/'suite'; suite.mkdir()
    write_json(suite/'skeletons.json', truth)
    write_json(suite/'metadata.json', dict(axes='zyx', coordinate_convention='voxel_outer_corner',
        shape=[size-32]*3, origin_nm=((start[::-1]+16)*16).tolist(), voxel_size_nm=[16]*3,
        scored_core_origin_nm=(core_start[::-1]*16).tolist(), scored_core_shape=[core]*3,
        paper_comparable=False))
    if sha256(evaluator) != '48f03d7c28b77853a431c0f163e9118befd057777a332be13cf500f72feb7f3c':
        raise ValueError('Unexpected evaluator')
    shutil.copyfile(evaluator, suite/'run_length.py')
    write_json(suite/'suite.json', dict(schema='pathfinder-evaluation-suite/v1', primary_metric='nerl',
        files={name:sha256(suite/name) for name in ('skeletons.json','metadata.json','run_length.py')},
        verifier_code={name:sha256(CODE/name) for name in
            ('verify_prediction.py','evaluate_erl.py','prediction_contract.py','topology.py')},
        common_sha256=sha256(CODE.parent/'common.py'), boundary_margin_voxels=[0,0,0],
        scope=f'MaleCNS v1.0 preselected {core*0.016:g}um core; local TEASAR paths from published segmentation; training and transfer protocol are defined by the experiment, not this reference suite',
        paper_comparable=False))
    expected = set(map(int, np.unique(gt))) - {0}
    write_json(output/'receipt.json', dict(plan_sha256=sha256(plan_path), split=split, adapter_sha256=sha256(Path(__file__)),
        labels_sha256=sha256(reference/'labels.npy'), suite_sha256=sha256(suite/'suite.json'),
        kimimaro_version=version('kimimaro'), tensorstore_version=version('tensorstore'),
        teasar_params=TEASAR, ground_truth='Published proofread segmentation; not error-free biological truth',
        unknown_policy='Zero excluded; retain all positive IDs without prediction-dependent filtering',
        boundary_policy='Core-truncated paths retained; no paper endpoint erosion',
        gt_instances=len(expected), represented_instances=len(covered), missing_instance_ids=sorted(expected-covered),
        nodes=len(truth['nodes']), edges=len(truth['edges']), paper_comparable=False))
    print('MaleCNS image and independent evaluation suite ready', flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','root','output','evaluator'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--split', choices=('train','validation','test'),
                   help='Required with malecns-spatial-comparison.json; verifies frozen image and label hashes')
    a=p.parse_args(); fetch(a.plan,a.root,a.output,a.evaluator,a.split)

if __name__=='__main__':
    main()
