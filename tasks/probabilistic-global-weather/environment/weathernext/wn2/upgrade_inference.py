#!/usr/bin/env python3
"""Repackage the declared starter's existing weights for v3; never train."""
import argparse
import json
from pathlib import Path
import shutil
from training_data import sha256


def upgrade(source, output):
    source, output = Path(source), Path(output)
    manifest = json.loads((source/'package.json').read_text())
    if manifest.get('schema') != 'wn2-starter-package/v1':
        raise ValueError('Only the declared starter can be upgraded automatically')
    if output.exists() or output.resolve().is_relative_to(source.resolve()):
        raise ValueError('Use a new output directory; original package stays intact')
    if any(p.is_symlink() for p in source.rglob('*')):
        raise ValueError('Package symlinks forbidden')
    for name, digest in manifest['files'].items():
        path=(source/name).resolve()
        if not path.is_relative_to(source.resolve()) or sha256(path)!=digest:
            raise ValueError('Package inventory mismatch')
    here=Path(__file__).resolve().parent
    # Only the stock model implementation is eligible for this entry-point update.
    for name in ('native_model.py','forecast_store.py'):
        if sha256(source/name)!=sha256(here/name):
            raise ValueError(f'Custom {name}: implement v3 for this package explicitly')
    shutil.copytree(source,output)
    for name in ('predict.py','inference_state.py'):
        shutil.copyfile(here/name,output/name)
    manifest['inference_schemas']=['wn2-inference/v2','wn2-inference/v3']
    manifest['upgrade']=dict(source_package_sha256=sha256(source/'package.json'),
                            weights_sha256=sha256(source/'weights.npz'),training_rerun=False)
    manifest['files']={str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob('*'))
                       if p.is_file() and p.name!='package.json'}
    (output/'package.json').write_text(json.dumps(manifest,indent=2)+'\n')
    assert sha256(source/'weights.npz')==sha256(output/'weights.npz')
    return manifest['upgrade']


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
    a=ap.parse_args();print(json.dumps(upgrade(a.source,a.output)))
