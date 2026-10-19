"""Maintainer: package selected prepared data only, never an experiment directory.

The output is relocatable and has separate development/private inventories.
No predictions, trained weights, logs, or historical ledger are exported.
"""
import argparse
import json
from pathlib import Path
import shutil
from materialize import digest,relative


def export(training,shape,validation,image,suite,runtime,output):
    output.mkdir(parents=True,exist_ok=False)
    def copy(source,dest):
        if source.is_symlink() or not source.is_file():raise ValueError('Regular source resource required')
        relative(str(dest))
        dest=output/dest;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    def write(dest,data):
        target=output/dest;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(data,indent=2)+'\n')
    data=json.loads((training/'manifest.json').read_text())
    data['splits']={key:data['splits'][key] for key in ('train','validation')}
    data['visibility']='development'
    for rows in data['splits'].values():
        for row in rows:
            for name,sha in row['files'].items():
                src=training/row['folder']/name
                if digest(src)!=sha:raise ValueError('Training source hash mismatch')
                copy(src,Path('development/training/volume')/row['folder']/name)
    write('development/training/volume/manifest.json',data)
    data=json.loads((shape/'manifest.json').read_text())
    data['rows']=[row for row in data['rows'] if row['split'] in ('train','validation')]
    if 'selections' in data:data['selections']={k:v for k,v in data['selections'].items() if k in ('train','validation')}
    for row in data['rows']:
        src=shape/row['path']
        if digest(src)!=row['sha256']:raise ValueError('Shape source hash mismatch')
        copy(src,Path('development/training/shape')/row['path'])
    write('development/training/shape/manifest.json',data)
    for src,prefix in [(validation/'image','development/validation/image'),(image,'private/image')]:
        for name in ('manifest.json','raw.npy','image_valid_z.npy'):copy(src/name,Path(prefix)/name)
    for src,prefix in [(validation/'suite','development/validation/suite'),(suite,'private/suite')]:
        for name in ('suite.json','metadata.json','skeletons.json','run_length.py'):copy(src/name,Path(prefix)/name)
    # Runtime must be curated beforehand: clean pinned FFN/connectomics source,
    # trusted FPS binary, source licenses and provenance only.
    for src in runtime.rglob('*'):
        if src.is_symlink():raise ValueError('Runtime symlinks are not portable resources')
        if src.is_file():copy(src,Path('development/runtime')/src.relative_to(runtime))
    base=output/'development/runtime'
    files={str(p.relative_to(base)):digest(p) for p in sorted(base.rglob('*')) if p.is_file() and p.name!='runtime-manifest.json'}
    write('development/runtime/runtime-manifest.json',dict(schema='pathfinder-trusted-runtime/v1',files=files))
    files={str(p.relative_to(output)):dict(sha256=digest(p),bytes=p.stat().st_size) for p in sorted(output.rglob('*')) if p.is_file()}
    write('release.json',dict(schema='pathfinder-resources/v1',files=files,
        source='MaleCNS v1.0 prepared spatial splits; maintainer-generated resources',
        predictions_included=False,model_weights_included=False))
    return dict(manifest_sha256=digest(output/'release.json'),files=len(files),bytes=sum(v['bytes'] for v in files.values()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('training','shape','validation','image','suite','runtime','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();print(json.dumps(export(**vars(a))))
