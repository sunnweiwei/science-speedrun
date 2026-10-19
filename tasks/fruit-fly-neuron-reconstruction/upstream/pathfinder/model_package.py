"""Export full-model assets with an optional three-checkpoint FFN ensemble.

This verifies model assets, not a completed inference or task submission.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import shutil
import sys
import torch
from ffn_model import FFN15
from sense import Sense
from shape_model import FPS,ShapeClassifier
from fetch_axonem import reserve
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import external,sha256,write_json,source_provenance


def construct(stage,architecture):
    if architecture.get('stage')!=stage:raise ValueError('Wrong checkpoint stage')
    if stage=='ffn':
        if (architecture['width'],architecture['residual_blocks'],architecture['input_shape'],architecture['voxel_size_nm'])!=(32,20,[33]*3,[16]*3):
            raise ValueError('Full FFN architecture required')
        return FFN15(activation_checkpointing=False,raw_normalization=architecture.get('raw_normalization'))
    if stage=='sense':
        if (architecture['width'],architecture['factors'],architecture['input_shape'],architecture['voxel_size_nm'])!=(8,[1,2,4,8,8],[128]*3,[16]*3):
            raise ValueError('Full SENSE architecture required')
        return Sense(8,(1,2,4,8,8))
    if stage=='shape':
        if (architecture['points'],architecture['radius_um'],architecture['widths'],architecture['neighbors'])!=(8192,10,[64,128,256,512,1024,2048],16):
            raise ValueError('Full SHAPE architecture required')
        return ShapeClassifier(None)  # strict state loading has no CUDA execution
    raise ValueError('Unknown stage')


def configured_threshold(manifest):
    value=manifest.get('inference',{}).get('search_threshold',.2)
    if type(value) not in (int,float) or not 0<=value<=1:
        raise ValueError('Finite search threshold in [0,1] required')
    return float(value)


ENSEMBLE_PATHS=('weights/ffn.pt','weights/ffn-ensemble-1.pt','weights/ffn-ensemble-2.pt')
SELECTION_RULES=('validation_bce','explicit_unverified','production_seed_growth')


def ffn_inference_architecture(architecture):
    # Unroll count is a training recipe, not an FFN15 inference parameter.
    # Full architecture metadata remains pinned in each checkpoint identity.
    result=dict(architecture)
    result.pop('ffn_recurrent_steps',None)
    return result


def ffn_identity(checkpoint):
    """Hash the actual full-model state after strict loading, not file metadata.

    Loading can cast checkpoint dtypes. Two serialized states that become the
    same model must not count as distinct ensemble members.
    """
    model=construct('ffn',checkpoint['architecture']);model.load_state_dict(checkpoint['model'],strict=True)
    digest=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        if not torch.isfinite(tensor).all():raise ValueError('Finite FFN model tensors required')
        digest.update(json.dumps([name,str(tensor.dtype),list(tensor.shape)],separators=(',',':')).encode()+b'\0')
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return dict(tensor_sha256=digest.hexdigest(),architecture=checkpoint['architecture'],step=checkpoint['step'],
        training_manifest_sha256=checkpoint.get('manifest_sha256',checkpoint.get('provenance',{}).get('manifest_sha256')))


def ordered_ffn_runs():
    return [dict(checkpoint_index=i,seed_order=order) for i in range(3) for order in ('ascending','descending')]


def verify_ffn_ensemble(package,manifest):
    ensemble=manifest.get('ffn_ensemble')
    if (not isinstance(ensemble,dict) or ensemble.get('schema')!='three-checkpoint-ffn/v1'
        or ensemble.get('selection_rule') not in SELECTION_RULES or ensemble.get('selection_verified') is not False
        or ensemble.get('ordered_runs')!=ordered_ffn_runs()):raise ValueError('Explicit unverified selection and ordered six-run FFN contract required')
    rows=ensemble.get('checkpoints')
    if not isinstance(rows,list) or len(rows)!=3 or [r.get('path') for r in rows]!=list(ENSEMBLE_PATHS):raise ValueError('Three canonical FFN checkpoint assets required')
    identities=[];runs=[];package=Path(package).resolve()
    for i,row in enumerate(rows):
        if not isinstance(row.get('source_sha256'),str) or not re.fullmatch('[0-9a-f]{64}',row['source_sha256']):raise ValueError('Checkpoint source hash required')
        path=package/row['path'];expected=manifest['files'].get(row['path'])
        if path.is_symlink() or package not in path.resolve().parents or not expected or sha256(path)!=expected:raise ValueError('Unpinned/changed ensemble asset')
        identity=ffn_identity(torch.load(path,map_location='cpu',weights_only=True))
        if any(row.get(key)!=value for key,value in identity.items()):raise ValueError('FFN ensemble identity differs from checkpoint')
        if ffn_inference_architecture(identity['architecture'])!=ffn_inference_architecture(manifest['models']['ffn']['architecture']):raise ValueError('Mixed FFN model architectures')
        identities.append(identity)
        runs.extend(dict(checkpoint_index=i,checkpoint_path=str(path),checkpoint_sha256=expected,seed_order=order)
                    for order in ('ascending','descending'))
    if len({r['tensor_sha256'] for r in identities})!=3:raise ValueError('Three distinct loaded FFN model states required')
    if len({r['training_manifest_sha256'] for r in identities})!=1:raise ValueError('Ensemble training provenance differs')
    training=identities[0]['training_manifest_sha256']
    if not isinstance(training,str) or not re.fullmatch('[0-9a-f]{64}',training):raise ValueError('Pinned training manifest required for ensemble')
    return dict(checkpoints=3,variants=6,loaded_model_states_distinct=True,selection_rule=ensemble['selection_rule'],
        selection_verified=False,ordered_runs=runs,
        scope='Full-model state identity and deterministic run order; does not prove paper checkpoint selection or six completed predictions')


def verify(package,load_extension=False,require_ensemble=False):
    package=Path(package).resolve();manifest=json.loads((package/'package.json').read_text())
    if manifest['schema']!='pathfinder-model-package/v1':raise ValueError('Unknown package schema')
    for name,digest in manifest['files'].items():
        path=(package/name).resolve()
        if package not in path.parents or (package/name).is_symlink() or not path.is_file() or sha256(path)!=digest:
            raise ValueError('Package integrity mismatch: '+name)
    threshold=configured_threshold(manifest)
    counts={}
    for stage in ('ffn','sense','shape'):
        path=package/f'weights/{stage}.pt'
        if str(path.relative_to(package)) not in manifest['files']:raise ValueError('Unpinned model file')
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        if checkpoint['architecture']!=manifest['models'][stage]['architecture']:raise ValueError('Model config mismatch')
        model=construct(stage,checkpoint['architecture']);model.load_state_dict(checkpoint['model'],strict=True)
        counts[stage]=sum(p.numel() for p in model.parameters())
        del checkpoint,model
    if require_ensemble and 'ffn_ensemble' not in manifest:raise ValueError('Three-checkpoint FFN ensemble required')
    ensemble=verify_ffn_ensemble(package,manifest) if 'ffn_ensemble' in manifest else None
    if load_extension:FPS(package)
    result=dict(schema='pathfinder-package-check/v1',package_manifest_sha256=sha256(package/'package.json'),
        search_threshold=threshold,parameters=counts,extension_loaded=load_extension,gpu_forward_verified=False,
        task_replay_verified=False,paper_comparable=False)
    if ensemble is not None:result['ffn_ensemble']=ensemble
    return result


def stage(root,source,out,*,require_ensemble=True):
    """Copy only verified model assets into the charged inference workspace.

    Each copied file has a fixed byte cap and hash. No hardlinks or native code
    execution are needed; predictions and unrelated submission files are ignored.
    """
    root=external(root);out=external(out);source=Path(source).resolve()
    if not (root/'budget.json').is_file():raise ValueError('Existing original budget ledger required')
    if root not in out.parents or out.exists() or source==out or source in out.parents or out in source.parents:
        raise ValueError('New independent model destination inside original root required')
    if type(require_ensemble) is not bool:raise ValueError('Explicit staging protocol required')
    check=verify(source,require_ensemble=require_ensemble)
    if not require_ensemble and 'ffn_ensemble' in check:raise ValueError('Explicit single-checkpoint package required')
    manifest_path=source/'package.json';manifest=json.loads(manifest_path.read_text())
    if sha256(manifest_path)!=check['package_manifest_sha256']:raise ValueError('Submission manifest changed during staging')
    pins=dict(manifest['files'])
    if 'package.json' in pins:raise ValueError('Manifest cannot include itself as an asset')
    fixed=set(ENSEMBLE_PATHS)|{'weights/sense.pt','weights/shape.pt','runtime/common.py',
        'runtime/pathfinder/paper_reference.json','point-ops/pathfinder_fps.so',
        'vendor/pointnext/LICENSE','vendor/pointnext/openpoints/LICENSE'}
    fixed.update('vendor/pointnext/openpoints/cpp/pointnet2_batch/src/'+name for name in
                 ('sampling.cpp','sampling_gpu.cu','sampling_gpu.h','cuda_utils.h'))
    for name in pins:
        path=Path(name)
        if name not in fixed and not (path.parent==Path('runtime/pathfinder') and path.suffix=='.py'):
            raise ValueError('Unrecognized model asset; predictions are not submission assets')
    pins['package.json']=check['package_manifest_sha256']
    for name in pins:
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or p.as_posix()!=name:raise ValueError('Canonical relative model asset paths required')
    sizes={name:(source/name).stat().st_size for name in pins}
    reserve(root,sum(sizes.values())+1048576)
    out.mkdir(parents=True,exist_ok=False)
    for name,digest in pins.items():
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True);written=0;hashed=hashlib.sha256()
        with (source/name).open('rb') as reader,target.open('xb') as writer:
            while chunk:=reader.read(2**20):
                written+=len(chunk)
                if written>sizes[name]:raise ValueError('Model asset grew beyond reserved size')
                writer.write(chunk);hashed.update(chunk)
        if written!=sizes[name] or hashed.hexdigest()!=digest:raise ValueError('Model asset changed during copy')
    staged=verify(out,require_ensemble=require_ensemble)
    if staged['package_manifest_sha256']!=check['package_manifest_sha256']:raise ValueError('Staged manifest differs')
    for name,digest in pins.items():
        if sha256(source/name)!=digest:raise ValueError('Submission changed during staging')
    return dict(schema='pathfinder-model-staging/v1',source=str(source),destination=str(out),
                package_manifest_sha256=check['package_manifest_sha256'],copied_bytes=sum(sizes.values()),
                copied_files=len(pins),check=staged,source_preserved=True,
                scope='Verified byte-identical model assets copied; unrelated submission files omitted. No GPU or native-extension execution.')


def export(root,out,checkpoints,search_threshold=.2,ffn_ensemble=None,ffn_selection_rule=None):
    root=external(root);out=external(out);code=Path(__file__).resolve().parent
    if root not in out.parents:raise ValueError('Package must remain under budget root')
    search_threshold=configured_threshold({'inference':{'search_threshold':search_threshold}})
    sources={}
    for p in code.glob('*.py'):
        if not p.name.startswith('test_'):sources['runtime/pathfinder/'+p.name]=p
    sources['runtime/pathfinder/paper_reference.json']=code/'paper_reference.json'
    sources['runtime/common.py']=code.parent/'common.py'
    native=root/'vendor/pointnext';prefix='vendor/pointnext/openpoints/cpp/pointnet2_batch/src/'
    for name in ('sampling.cpp','sampling_gpu.cu','sampling_gpu.h','cuda_utils.h'):
        sources[prefix+name]=root/(prefix+name)
    sources['vendor/pointnext/LICENSE']=native/'LICENSE'
    sources['vendor/pointnext/openpoints/LICENSE']=native/'openpoints/LICENSE'
    sources['point-ops/pathfinder_fps.so']=root/'point-ops/pathfinder_fps.so'
    for path in sources.values():
        if not path.is_file():raise FileNotFoundError(path)
    if set(checkpoints)!={'ffn','sense','shape'}:
        raise ValueError('Explicit FFN, SENSE and SHAPE checkpoints required')
    checkpoints={stage:external(path) for stage,path in checkpoints.items()}
    if any(root not in path.parents for path in checkpoints.values()):
        raise ValueError('Checkpoint must be under the original budget root')
    if (ffn_ensemble is None)!=(ffn_selection_rule is None):raise ValueError('Provide both three FFN checkpoints and a declared selection rule')
    ensemble_paths=[];ensemble=None
    if ffn_ensemble is not None:
        if len(ffn_ensemble)!=3 or ffn_selection_rule not in SELECTION_RULES:raise ValueError('Three FFN checkpoints and explicit selection rule required')
        ensemble_paths=[external(p) for p in ffn_ensemble]
        if any(root not in p.parents for p in ensemble_paths):raise ValueError('Ensemble checkpoints must stay inside original budget root')
    pins={str(p):sha256(p) for p in [*sources.values(),*checkpoints.values(),*ensemble_paths]}
    if ensemble_paths:
        identities=[ffn_identity(torch.load(p,map_location='cpu',weights_only=True)) for p in ensemble_paths]
        primary=ffn_identity(torch.load(checkpoints['ffn'],map_location='cpu',weights_only=True))
        if identities[0]!=primary:raise ValueError('First ensemble model must equal the primary FFN checkpoint')
        if len({r['tensor_sha256'] for r in identities})!=3:raise ValueError('Three distinct loaded FFN model states required')
        if any(ffn_inference_architecture(r['architecture'])!=ffn_inference_architecture(primary['architecture']) or r['training_manifest_sha256']!=primary['training_manifest_sha256'] for r in identities):raise ValueError('Ensemble architecture/training provenance differs')
        training=primary['training_manifest_sha256']
        if not isinstance(training,str) or not re.fullmatch('[0-9a-f]{64}',training):raise ValueError('Pinned training manifest required for ensemble')
        ensemble=dict(schema='three-checkpoint-ffn/v1',selection_rule=ffn_selection_rule,selection_verified=False,
            checkpoints=[dict(path=name,source_sha256=pins[str(p)],**identity) for name,p,identity in zip(ENSEMBLE_PATHS,ensemble_paths,identities)],
            ordered_runs=ordered_ffn_runs())
    reserve(root,sum(p.stat().st_size for p in [*sources.values(),*checkpoints.values(),*ensemble_paths[1:]])+1048576)
    out.mkdir(parents=True,exist_ok=False);models={}
    for stage,path in checkpoints.items():
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        model=construct(stage,checkpoint['architecture']);model.load_state_dict(checkpoint['model'],strict=True)
        models[stage]=dict(source_sha256=sha256(path),architecture=checkpoint['architecture'],
            step=checkpoint['step'],validation_bce=checkpoint['validation_bce'],parameters=sum(p.numel() for p in model.parameters()),
            training_manifest_sha256=checkpoint.get('manifest_sha256', checkpoint.get('provenance',{}).get('manifest_sha256')))
        # Optimizer state is irrelevant to frozen inference; original hash remains pinned above.
        target=out/f'weights/{stage}.pt';target.parent.mkdir(exist_ok=True)
        torch.save({k:v for k,v in checkpoint.items() if k!='optimizer'},target)
        del checkpoint,model
    for name,path in zip(ENSEMBLE_PATHS[1:],ensemble_paths[1:]):
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        torch.save({k:v for k,v in checkpoint.items() if k!='optimizer'},out/name)
        del checkpoint
    for name,path in sources.items():
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    versions={name:importlib.metadata.version(name) for name in ('torch','numpy','scipy','scikit-image','h5py','networkx')}
    manifest=dict(schema='pathfinder-model-package/v1',models=models,source=source_provenance(),
        inference=dict(search_threshold=search_threshold),
        files={str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob('*')) if p.is_file()},
        runtime=dict(python=platform.python_version(),platform=platform.platform(),packages=versions,
            torch_cuda=torch.version.cuda,compiled_cuda_arch='8.6',
            binary_portability='Same Python/PyTorch/CUDA ABI only; rebuild bundled FPS source for another compatible runtime'),
        paper_comparable=False,task_submission_verified=False,ground_truth_included=False)
    if ensemble is not None:manifest['ffn_ensemble']=ensemble
    for path,digest in pins.items():
        if sha256(path)!=digest:raise ValueError('Model/export source changed during packaging')
    write_json(out/'package.json',manifest)
    return verify(out,load_extension=True,require_ensemble=ensemble is not None)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['export','verify'],required=True)
    p.add_argument('--root',type=Path);p.add_argument('--package',type=Path,required=True)
    p.add_argument('--search-threshold',type=float,default=.2)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--load-extension',action='store_true')
    p.add_argument('--ffn-ensemble-checkpoints',type=Path,nargs=3,help='Ordered FFN checkpoints; first must match --ffn-checkpoint')
    p.add_argument('--ffn-selection-rule',choices=SELECTION_RULES,help='Declaration only; no paper selection proof is implied')
    p.add_argument('--require-ffn-ensemble',action='store_true',help='Reject legacy single-FFN packages during verification')
    for stage in ('ffn','sense','shape'):
        p.add_argument('--'+stage+'-checkpoint',type=Path)
    a=p.parse_args()
    if a.mode=='export':
        if not a.root:raise ValueError('--root required')
        checkpoints={stage:getattr(a,stage+'_checkpoint') for stage in ('ffn','sense','shape')}
        if any(path is None for path in checkpoints.values()):
            raise ValueError('Export requires --ffn-checkpoint, --sense-checkpoint and --shape-checkpoint; no historical defaults')
        if a.require_ffn_ensemble and a.ffn_ensemble_checkpoints is None:raise ValueError('Ensemble assets required')
        result=export(a.root,a.package,checkpoints,a.search_threshold,a.ffn_ensemble_checkpoints,a.ffn_selection_rule)
    else:
        if a.ffn_ensemble_checkpoints is not None or a.ffn_selection_rule is not None:raise ValueError('Ensemble export arguments are not verification options')
        result=verify(a.package,a.load_extension,a.require_ffn_ensemble)
    write_json(external(a.report),result);print(json.dumps(result))

if __name__=='__main__':main()
