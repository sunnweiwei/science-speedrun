"""Freeze sparse validation or test skeletons in an explicit prediction physical ROI.

Only members of a pre-existing SHAPE split are selected. Unlabelled tissue and other
splits are unknown, not negative. Merges with unlabelled objects are undetectable;
this is not a dense benchmark or a paper-comparable metric.
"""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time
import numpy as np
from build_skeletons import TEASAR, compatible_osteoid_merge, convert_skeletons
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import external,sha256,write_json


def center_bounds(lower, upper, spacing, source_shape):
    """Native voxel centers inside half-open physical bounds; source origin=0."""
    lower,upper,spacing=map(lambda a:np.asarray(a,dtype=float),(lower,upper,spacing))
    if (lower.shape!=(3,) or upper.shape!=(3,) or spacing.shape!=(3,) or
        not np.isfinite([lower,upper,spacing]).all() or (spacing<=0).any() or
        (lower<0).any() or (upper<=lower).any() or (upper>np.asarray(source_shape)*spacing).any()):
        raise ValueError('Require finite nonempty physical ROI within sparse source')
    start=np.ceil(lower/spacing-.5).astype(np.int64)
    stop=np.ceil(upper/spacing-.5).astype(np.int64)
    if (stop<=start).any():raise ValueError('ROI contains no native voxel centers')
    return start,stop


def build(root, output, lower, upper, prediction_spacing=(16,16,16),max_voxels=50_000_000,split="validation"):
    if split not in ("validation","test"):raise ValueError("Only frozen validation/test splits may be evaluated")
    root=Path(root);manifest_path=root/'shape-meshes/manifest.json'
    manifest=json.loads(manifest_path.read_text());source=root/'data/sparse-axon-preview.npy'
    source_hash=sha256(source)
    if source_hash!=manifest['source_sha256']:raise ValueError('Sparse data hash differs from frozen SHAPE split source')
    if manifest.get('axes')!='zyx':raise ValueError('Require ZYX SHAPE manifest')
    selected=sorted(int(row['id']) for row in manifest['rows'] if row['split']==split)
    if not selected or len(selected)!=len(set(selected)):raise ValueError('Unique frozen evaluation IDs required')
    original=np.load(source,mmap_mode='r',allow_pickle=False)
    spacing=np.asarray(manifest['voxel_size_nm'],dtype=float)
    start,stop=center_bounds(lower,upper,spacing,original.shape)
    if np.prod(stop-start)>max_voxels:raise ValueError('Native crop exceeds CPU voxel budget')
    lower=np.asarray(lower,dtype=float);upper=np.asarray(upper,dtype=float);predspacing=np.asarray(prediction_spacing,dtype=float)
    shape=(upper-lower)/predspacing
    if predspacing.shape!=(3,) or not np.isfinite(predspacing).all() or (predspacing<=0).any() or not np.allclose(shape,np.rint(shape)):
        raise ValueError('Prediction physical extent must be an integer positive voxel grid')
    labels=np.array(original[tuple(slice(int(a),int(b)) for a,b in zip(start,stop))])
    labels[~np.isin(labels,selected)]=0
    import kimimaro
    clock=time.monotonic()
    with compatible_osteoid_merge():
        skeletons=kimimaro.skeletonize(labels,teasar_params=dict(TEASAR),anisotropy=spacing,dust_threshold=0,
            progress=False,fix_branching=True,fix_borders=True,parallel=1,fill_holes=False,fix_avocados=False)
    truth,represented=convert_skeletons(skeletons,labels,start*spacing,spacing)
    if not truth['edges']:raise ValueError('No selected development skeleton path in this ROI')
    positions=np.asarray([n['position_nm'] for n in truth['nodes']])
    if (positions<lower).any() or (positions>=upper).any():raise ValueError('Skeleton node outside prediction ROI')
    truth.update(protocol='local-sparse-development-teasar/v1' if split=='validation' else 'local-sparse-heldout-shape-teasar/v1',
                 split=split,scope=f'pre-existing SHAPE {split} IDs only; all other tissue unknown')
    metadata=dict(axes='zyx',coordinate_convention='voxel_outer_corner',origin_nm=lower.tolist(),
        voxel_size_nm=predspacing.tolist(),shape=np.rint(shape).astype(int).tolist(),
        physical_roi_nm=np.stack([lower,upper],axis=1).tolist(),paper_comparable=False)
    present=set(map(int,np.unique(labels)))-{0}
    receipt=dict(protocol=truth['protocol'],source_sha256=source_hash,split_manifest_sha256=sha256(manifest_path),
        selected_split=split,selected_ids=selected,selected_ids_in_roi=sorted(present),represented_ids=sorted(represented),
        selected_ids_outside_roi=sorted(set(selected)-present),present_ids_without_skeleton=sorted(present-represented),
        native_voxel_size_nm=spacing.tolist(),native_source_origin_nm=[0,0,0],
        native_crop_bounds_zyx=np.stack([start,stop],axis=1).tolist(),
        native_crop_origin_nm=(start*spacing).tolist(),prediction_metadata=metadata,
        selected_foreground_voxels=int((labels>0).sum()),nodes=len(truth['nodes']),edges=len(truth['edges']),
        kimimaro_version=version('kimimaro'),osteoid_version=version('osteoid'),teasar_params=TEASAR,
        dust_threshold=0,fix_borders=True,fix_branching=True,fill_holes=False,parallel=1,
        boundary_policy='retain ROI-truncated paths at included native centers; no erosion or interpolation of skeletons',
        unknown_policy='zero and all IDs outside selected split excluded; merges with those objects cannot be detected',
        cross_stage_neuron_disjointness_verified=False,ground_truth_for='evaluation only, not provided to inference',
        source_alignment_assumption='sparse core outer origin [0,0,0] nm from releaseviewer metadata',
        paper_comparable=False,elapsed_seconds=time.monotonic()-clock,builder_sha256=sha256(Path(__file__)))
    if split=='validation':receipt['selected_validation_ids']=selected  # historical receipt compatibility
    output=external(output);output.mkdir(parents=True,exist_ok=False)
    write_json(output/'skeletons.json',truth);write_json(output/'metadata.json',metadata)
    receipt['skeletons_sha256']=sha256(output/'skeletons.json');receipt['metadata_sha256']=sha256(output/'metadata.json')
    write_json(output/'receipt.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--physical-roi-nm',type=float,nargs=6,default=[3256,26744,4640,28128,4640,28128],
        help='ZYX half-open lower/upper nm, default 24um volume trimmed by 16 prediction voxels each side')
    parser.add_argument('--prediction-voxel-size-nm',type=float,nargs=3,default=[16]*3)
    parser.add_argument('--split',choices=['validation','test'],default='validation')
    args=parser.parse_args();roi=np.asarray(args.physical_roi_nm).reshape(3,2)
    print(json.dumps(build(args.root,args.output,roi[:,0],roi[:,1],args.prediction_voxel_size_nm,split=args.split),indent=2))


if __name__=='__main__':main()
