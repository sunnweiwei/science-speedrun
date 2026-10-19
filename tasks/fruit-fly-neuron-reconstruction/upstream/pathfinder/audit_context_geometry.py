"""Geometry-only audit of the full10um SENSE representative-center domain.

No skeletons or predictions are read. Grid-volume fractions are not instance
coverage fractions and cannot predict final reconstruction quality.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import external,sha256,write_json
from run_continuous_ffn import validate_geometry
from infer_continuous_sense import select_centers
from dataset_contract import validate_manifest


def audit(plan_path,image_path,training_path):
    read=lambda p:json.loads(p.read_text())
    plan=read(plan_path);image=read(image_path);training=read(training_path)
    if sha256(image_path)!=plan['image_manifest_sha256']:raise ValueError('FFN plan image mismatch')
    validate_manifest(training,base_dir=training_path.parent)
    raw=np.load(image_path.parent/'raw.npy',mmap_mode='r',allow_pickle=False)
    valid=np.load(image_path.parent/'image_valid_z.npy',mmap_mode='r',allow_pickle=False)
    validate_geometry(image,raw,valid,plan['roi'])
    roi=np.array(plan['roi']).reshape(3,2);spacing=np.array(image['voxel_size_nm'])
    shape=roi[:,1]-roi[:,0];origin=np.array(image['origin_nm'])+roi[:,0]*spacing
    lo=np.ceil(10000/spacing-.5).astype(int)
    hi=np.floor(shape-10000/spacing-.5).astype(int)+1
    counts=np.maximum(0,hi-lo);lower=origin+lo*spacing;upper=origin+hi*spacing
    if (counts>0).all():
        corners=[dict(id=i+1,center_zyx=[int(hi[a]-1 if i&(1<<a) else lo[a]) for a in range(3)]) for i in range(8)]
        checks=corners+[dict(id=9,center_zyx=(lo-1).tolist()),dict(id=10,center_zyx=hi.tolist())]
        _,skipped,_,eligible=select_centers(checks,shape,0,1)
        if eligible!=list(range(1,9)) or skipped!=[9,10]:raise ValueError('Geometry differs from actual center selector')
    containing=[]
    for row in training['splits']['train']:
        start=np.array(row['origin_nm']);stop=start+np.array(row['shape'])*training['voxel_size_nm']
        if (counts>0).all() and (start<=lower).all() and (stop>=upper).all():containing.append(row['name'])
    return dict(schema='pathfinder-geometric-center-coverage/v1',
        inputs={str(f.resolve()):sha256(f) for f in (plan_path,image_path,training_path)},
        audit_code_sha256=sha256(Path(__file__)),selector_sha256=sha256(Path(__file__).with_name('infer_continuous_sense.py')),
        radius_nm=10000,roi_shape=shape.tolist(),roi_origin_nm=origin.tolist(),roi_extent_nm=(shape*spacing).tolist(),
        eligible_center_index_bounds=np.stack([lo,hi],axis=1).tolist(),
        eligible_center_cell_bounds_nm=np.stack([lower,upper],axis=1).tolist(),
        eligible_center_grid_shape=counts.tolist(),eligible_center_grid_voxels=int(np.prod(counts)),
        eligible_center_volume_fraction=float(np.prod(counts)/np.prod(shape)),
        training_raw_blocks_containing_all_eligible_centers=containing,
        coordinate_frame_assumption='Caller establishes shared physical frame; numeric coordinates alone cannot establish dataset identity',
        actual_instance_center_coverage_known=False,
        scope='Geometric maximum of possible center positions, not instance coverage or final reconstruction coverage',
        training_interpretation='Containing training raw extents imply potential spatial exposure, not actual minibatch sampling',
        ground_truth_skeletons_read=False,predictions_read=False,paper_comparable=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('ffn-plan','image-manifest','training-manifest','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();write_json(external(a.output),audit(a.ffn_plan,a.image_manifest,a.training_manifest))


if __name__=='__main__':main()
