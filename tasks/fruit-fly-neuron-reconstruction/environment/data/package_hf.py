"""Package an existing verified prepared release as HF content-addressed blobs."""
import argparse
import json
from pathlib import Path
import shutil
from materialize import digest, relative


def pack(root,output):
    root=root.resolve();output=output.resolve()
    if output==root or root in output.parents or output in root.parents:
        raise ValueError('Source and staging must be disjoint')
    if output.exists():raise ValueError('Use a new staging directory')
    m=json.loads((root/'release.json').read_text())
    if m.get('schema')!='pathfinder-resources/v1':raise ValueError('Unknown resource schema')
    (output/'blobs').mkdir(parents=True)
    for name,spec in m['files'].items():
        p=root/relative(name)
        if p.is_symlink() or root not in p.resolve().parents:raise ValueError('Unsafe source')
        if p.stat().st_size!=spec['bytes'] or digest(p)!=spec['sha256']:
            raise ValueError('Corrupt source: '+name)
        dest=output/'blobs'/spec['sha256']
        if not dest.exists():shutil.copyfile(p,dest)
    shutil.copyfile(root/'release.json',output/'release.json')
    shutil.copyfile(Path(__file__).with_name('DATASET_CARD.md'),output/'README.md')
    for name in ('ffn/LICENSE','connectomics/LICENSE','vendor/pointnext/LICENSE','vendor/pointnext/openpoints/LICENSE'):
        src=root/'development/runtime'/name
        if src.is_file():
            dest=output/'licenses'/name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(src,dest)
    plans=Path(__file__).resolve().parents[2]/'upstream/pathfinder'
    for name in ('malecns-multiregion.json','malecns-spatial-comparison.json','PAPER_PROTOCOLS.md'):
        dest=output/'provenance'/name;dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(plans/name,dest)
    return dict(files=len(m['files']),manifest_sha256=digest(output/'release.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();print(json.dumps(pack(a.root,a.output)))
