"""Materialize a pinned, resource-only release into disjoint visible/private roots."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import urllib.request


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


def relative(name):
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or p.as_posix()!=name:
        raise ValueError('Canonical relative resource path required')
    return p


def materialize(manifest,expected,source,data,private,verify_only=False,fetch=None):
    if digest(manifest)!=expected:raise ValueError('Release manifest hash mismatch')
    release=json.loads(manifest.read_text())
    if release.get('schema')!='pathfinder-resources/v1':raise ValueError('Unknown release schema')
    data=data.resolve();private=private.resolve()
    if data==private or data in private.parents or private in data.parents:
        raise ValueError('Development and private resources must be disjoint')
    expected_paths={'development':set(),'private':set()}
    for name in release['files']:
        path=relative(name)
        if len(path.parts)<3 or path.parts[0] not in expected_paths:
            raise ValueError('Explicit resource visibility required')
        expected_paths[path.parts[0]].add('/'.join(path.parts[1:]))
    for kind,root in (('development',data),('private',private)):
        if root.exists():
            for path in root.rglob('*'):
                relative_path=path.relative_to(root).as_posix()
                if path.is_symlink():raise ValueError('Symlink in resource destination')
                if path.is_file() and relative_path not in expected_paths[kind]:
                    if not (relative_path.endswith('.partial') and relative_path[:-8] in expected_paths[kind]):
                        raise ValueError('Unlisted existing resource: '+relative_path)
    for name,entry in release['files'].items():
        path=relative(name)
        if path.parts[0] not in ('development','private'):raise ValueError('Explicit resource visibility required')
        if len(path.parts)<3 or (path.parts[0]=='development' and path.parts[1] not in ('training','validation','runtime')) or (path.parts[0]=='private' and path.parts[1] not in ('image','suite')):
            raise ValueError('Unexpected resource directory')
        root=data if path.parts[0]=='development' else private
        target=root/Path(*path.parts[1:])
        if target.is_symlink() or root not in target.resolve().parents:raise ValueError('Resource escapes destination')
        if target.is_file() and target.stat().st_size==entry['bytes'] and digest(target)==entry['sha256']:continue
        if verify_only:raise ValueError('Missing or corrupt resource: '+name)
        target.parent.mkdir(parents=True,exist_ok=True);partial=target.with_name(target.name+'.partial')
        if partial.is_symlink():raise ValueError('Symlink partial file')
        if fetch is not None:
            reader=fetch(entry['sha256']).open('rb')
        elif source.startswith(('https://','http://')):
            reader=urllib.request.urlopen(source.rstrip('/')+'/'+name,timeout=120)
        else:
            source_root=Path(source).resolve();src=source_root/Path(name)
            if src.is_symlink() or source_root not in src.resolve().parents:raise ValueError('Resource escapes source')
            reader=src.open('rb')
        try:
            with reader,partial.open('wb') as writer:
                count=0
                while chunk:=reader.read(8<<20):
                    count+=len(chunk)
                    if count>entry['bytes']:raise ValueError('Resource exceeds declared size')
                    writer.write(chunk)
            if partial.stat().st_size!=entry['bytes'] or digest(partial)!=entry['sha256']:
                raise ValueError('Resource integrity mismatch: '+name)
            partial.replace(target)
        except BaseException:
            partial.unlink(missing_ok=True);raise
    return dict(complete=True,manifest_sha256=expected,files=len(release['files']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--source',required=True)
    for n in ('data','private'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--verify-only',action='store_true')
    a=p.parse_args();print(json.dumps(materialize(a.manifest,a.sha256,a.source,a.data,a.private,a.verify_only)))
