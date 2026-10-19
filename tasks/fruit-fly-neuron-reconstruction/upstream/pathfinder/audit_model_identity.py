"""Compare original trained state bytes and selected frozen inference modules.

This is a provenance check, not GPU parity or full model replay. Optimizer removal
and checkpoint serialization can change file hashes without changing weights.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
from model_package import verify

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import external,sha256,write_json


def state_sha(state):
    digest=hashlib.sha256()
    for key,value in sorted(state.items()):
        digest.update(json.dumps([key,str(value.dtype),list(value.shape)],separators=(',',':')).encode()+b'\0')
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def audit(root,package,ffn_code,post_package):
    check=verify(package);verify(post_package)
    manifest=json.loads((package/'package.json').read_text());rows=[]
    for stage in ('ffn','sense','shape'):
        source=root/f'formal-{stage}/best.pt';exported=package/f'weights/{stage}.pt'
        source_hash=sha256(source)
        if source_hash!=manifest['models'][stage]['source_sha256']:raise ValueError('Export refers to another original checkpoint')
        original=torch.load(source,map_location='cpu',weights_only=True)
        packed=torch.load(exported,map_location='cpu',weights_only=True)
        old_hash=state_sha(original['model']);new_hash=state_sha(packed['model'])
        if original['architecture']!=packed['architecture'] or old_hash!=new_hash:raise ValueError('Export changed model state')
        post=torch.load(post_package/f'weights/{stage}.pt',map_location='cpu',weights_only=True)
        if state_sha(post['model'])!=old_hash:raise ValueError('Postprocess package has different model state')
        rows.append(dict(stage=stage,original_checkpoint_sha256=source_hash,
            exported_checkpoint_sha256=sha256(exported),model_state_sha256=old_hash,
            tensors=len(original['model']),state_elements=sum(v.numel() for v in original['model'].values()),
            model_parameters=check['parameters'][stage],state_byte_identical=True))
    core={}
    selections={'ffn_source':(ffn_code,['batched_ffn.py','ffn_model.py']),
        'postprocess_source':(post_package/'runtime/pathfinder',
            ['infer_continuous_sense.py','sense.py','infer_shape.py','shape_model.py'])}
    for role,(folder,names) in selections.items():
        for name in names:
            source=folder/name;target=package/'runtime/pathfinder'/name
            core[role+'/'+name]=dict(source=str(source),source_sha256=sha256(source),
                package_sha256=sha256(target),identical=sha256(source)==sha256(target))
    return dict(schema='pathfinder-model-identity-audit/v1',package=str(package),
        package_manifest_sha256=check['package_manifest_sha256'],postprocess_package=str(post_package),
        postprocess_manifest_sha256=sha256(post_package/'package.json'),models=rows,selected_core_code=core,
        audit_code_sha256=sha256(Path(__file__)),
        state_digest_algorithm='SHA256 over sorted keys: compact JSON [key,dtype,shape], NUL, contiguous CPU tensor uint8 bytes',
        scope='Exact state bytes and selected inference modules; not transitive dependency equivalence, GPU parity, or native replay',
        gpu_execution_performed=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','package','ffn-code','post-package','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();write_json(external(a.output),audit(a.root,a.package,a.ffn_code,a.post_package))


if __name__=='__main__':main()
