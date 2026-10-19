"""Import externally prepared ZYX uint8 EM and explicit plane validity, without GT.

This backend does not resample, infer resolution, normalize intensities, or label
all pixels observed by default. Source-specific conversion must be documented
upstream. Existing trained models require16nm; importing another grid does not
make those checkpoints compatible.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from fetch_axonem import reserve
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import external,sha256,write_json


def validate(config,raw,valid):
    if config.get('schema')!='pathfinder-continuous-source/v1':raise ValueError('Unsupported source schema')
    if config.get('axes')!='zyx' or config.get('coordinate_convention')!='voxel_outer_corner':
        raise ValueError('Explicit ZYX outer-corner coordinates required')
    if raw.ndim!=3 or raw.dtype!=np.uint8 or any(n<1 for n in raw.shape):raise ValueError('Nonempty3D uint8 EM required')
    if valid.dtype!=np.bool_ or valid.shape!=(raw.shape[0],):raise ValueError('Explicit boolean validity per Z plane required')
    if config.get('shape')!=list(raw.shape):raise ValueError('Source shape mismatch')
    for name in ('origin_nm','voxel_size_nm','native_voxel_size_nm'):
        value=np.asarray(config.get(name),dtype=float)
        if value.shape!=(3,) or not np.isfinite(value).all() or (name!='origin_nm' and (value<=0).any()):
            raise ValueError('Invalid physical coordinates: '+name)
    for name in ('dataset_id','source','source_revision','conversion','image_valid_scope'):
        if not isinstance(config.get(name),str) or not config[name].strip():raise ValueError('Missing provenance: '+name)
    if config.get('ground_truth_included') is not False:raise ValueError('Image-only adapter required')


def import_image(config_path,root,out):
    root=external(root);out=external(out);config_path=Path(config_path).resolve()
    if root not in out.parents:raise ValueError('Output must remain under budget root')
    config=json.loads(config_path.read_text());paths={}
    for name in ('raw','image_valid_z'):
        entry=config[name];path=(config_path.parent/entry['path']).resolve()
        if sha256(path)!=entry['sha256']:raise ValueError('Source hash mismatch: '+name)
        paths[name]=path
    raw=np.load(paths['raw'],mmap_mode='r',allow_pickle=False)
    valid=np.load(paths['image_valid_z'],mmap_mode='r',allow_pickle=False)
    validate(config,raw,valid)
    root.mkdir(parents=True,exist_ok=True)
    reserve(root,sum(p.stat().st_size for p in paths.values())+1048576)
    out.mkdir(parents=True,exist_ok=False)
    files={}
    for name,path in paths.items():
        target=out/(name+'.npy')
        with path.open('rb') as src,target.open('xb') as dst:shutil.copyfileobj(src,dst,8<<20)
        digest=sha256(target)
        if digest!=config[name]['sha256']:raise ValueError('Source changed during import: '+name)
        files[target.name]=digest
    manifest={k:config[k] for k in ('dataset_id','axes','coordinate_convention','shape','origin_nm','voxel_size_nm',
                                   'native_voxel_size_nm','source','source_revision')}
    manifest.update(schema='pathfinder-continuous-image/v1',files=files,source_config_sha256=sha256(config_path),
        conversion=config['conversion'],adapter='explicit preprocessed NPY import; no interpolation or intensity conversion',
        ground_truth_included=False,paper_comparable=False,native_resolution_restored=False,
        image_valid=dict(layout='z vector broadcasts over y,x',scope=config['image_valid_scope'],
            zero_means='unavailable image plane per supplied source mask; not GT/background',
            valid_planes=int(valid.sum()),invalid_output_planes=np.flatnonzero(~valid).tolist(),
            all_source_defects_detected=False,filler_values_preserved=True),
        geometric_full20um_sphere_possible=bool((np.asarray(raw.shape)*config['voxel_size_nm']>=20000).all()),
        full_context_image_validity_verified=False,
        current_16nm_checkpoint_grid_compatible=config['voxel_size_nm']==[16,16,16])
    write_json(out/'manifest.json',manifest)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','root','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();result=import_image(a.source,a.root,a.output)
    print(json.dumps({k:result[k] for k in ('dataset_id','shape','voxel_size_nm','current_16nm_checkpoint_grid_compatible')}))

if __name__=='__main__':main()
