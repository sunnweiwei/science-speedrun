"""Evaluation-only audit of potential FFN/SENSE training-region exposure.

Samples frozen skeleton nodes, not every point along edges. A node inside an
eligible training region need not have been sampled in the realized training
run. Absence of spatial overlap does not prove neuron-identity disjointness.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import external,sha256,write_json
from dataset_contract import validate_manifest


def node_exposure(positions,origin,spacing,valid):
    positions=np.asarray(positions,dtype=float);origin=np.asarray(origin,dtype=float)
    spacing=np.asarray(spacing,dtype=float)
    if (positions.ndim!=2 or positions.shape[1]!=3 or not np.isfinite(positions).all()
        or origin.shape!=(3,) or spacing.shape!=(3,) or not np.isfinite(origin).all()
        or not np.isfinite(spacing).all() or (spacing<=0).any()
        or valid.ndim!=3 or valid.dtype!=np.bool_):
        raise ValueError('Finite ZYX geometry and 3D boolean training mask required')
    relative=(positions-origin)/spacing
    inside=((relative>=0)&(relative<valid.shape)).all(axis=1)
    exposed=np.zeros(len(positions),dtype=bool)
    indices=np.floor(relative[inside]).astype(np.int64)
    exposed[np.flatnonzero(inside)]=valid[tuple(indices.T)]
    return inside,exposed


def audit(manifest_path,skeletons_path,training_runs):
    manifest_path=Path(manifest_path);skeletons_path=Path(skeletons_path)
    manifest=json.loads(manifest_path.read_text());validate_manifest(manifest,base_dir=manifest_path.parent)
    manifest_hash=sha256(manifest_path)
    runs={}
    for path in training_runs:
        path=Path(path);run=json.loads(path.read_text())
        if run['manifest_sha256']!=manifest_hash:raise ValueError('Training run refers to different manifest')
        stage=run['architecture']['stage']
        if stage not in ('ffn','sense') or stage in runs:raise ValueError('Distinct FFN/SENSE training run records required')
        runs[stage]=dict(path=str(path.resolve()),sha256=sha256(path))
    if set(runs)!={'ffn','sense'}:raise ValueError('Both trained-stage run records required')
    truth=json.loads(skeletons_path.read_text())
    if truth.get('axes')!='zyx' or truth.get('coordinate_units')!='nm':raise ValueError('Physical ZYX skeletons required')
    nodes=truth['nodes'];ids=[n['id'] for n in nodes]
    if not nodes or len(ids)!=len(set(ids)):raise ValueError('Nonempty unique skeleton node IDs required')
    positions=np.asarray([n['position_nm'] for n in nodes],dtype=float)
    raw=np.zeros(len(nodes),bool);supervised=raw.copy();blocks=[]
    for row in manifest['splits']['train']:
        folder=Path(row['folder'])
        if not folder.is_absolute():folder=manifest_path.parent/folder
        mask_path=folder/'valid.npy'
        if sha256(mask_path)!=row['files']['valid.npy']:raise ValueError('Training mask integrity mismatch')
        valid=np.load(mask_path,mmap_mode='r',allow_pickle=False)
        if list(valid.shape)!=row['shape']:raise ValueError('Training mask shape mismatch')
        inside,exposed=node_exposure(positions,row['origin_nm'],manifest['voxel_size_nm'],valid)
        raw|=inside;supervised|=exposed
        blocks.append(dict(name=row['name'],raw_extent_nodes=int(inside.sum()),
            eligible_supervised_nodes=int(exposed.sum()),valid_sha256=sha256(mask_path)))
    per_skeleton=[]
    for sid in sorted({n['skeleton_id'] for n in nodes}):
        selected=np.array([n['skeleton_id']==sid for n in nodes])
        per_skeleton.append(dict(skeleton_id=sid,nodes=int(selected.sum()),
            raw_extent_nodes=int((raw&selected).sum()),eligible_supervised_nodes=int((supervised&selected).sum())))
    return dict(schema='pathfinder-training-region-overlap/v1',
        manifest_sha256=manifest_hash,skeletons_sha256=sha256(skeletons_path),training_runs=runs,
        audit_code_sha256=sha256(Path(__file__)),nodes=len(nodes),
        raw_extent_nodes=int(raw.sum()),eligible_supervised_nodes=int(supervised.sum()),
        raw_extent_fraction=float(raw.mean()),eligible_supervised_fraction=float(supervised.mean()),
        raw_extent_node_ids=[n['id'] for n,flag in zip(nodes,raw) if flag],
        eligible_supervised_node_ids=[n['id'] for n,flag in zip(nodes,supervised) if flag],
        blocks=blocks,per_skeleton=per_skeleton,
        scope='Potential training spatial exposure at skeleton nodes only; not actual minibatch exposure or whole-edge overlap',
        neuron_disjointness_proven=False,paper_comparable=False,
        splits_modified=False,predictions_read=False,
        interpretation='SHAPE member holdout does not establish FFN/SENSE spatial or neuron holdout; preserve original evaluation and report this audit alongside it')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','skeletons','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--training-run',type=Path,action='append',required=True)
    a=p.parse_args();write_json(external(a.output),audit(a.manifest,a.skeletons,a.training_run))


if __name__=='__main__':main()
