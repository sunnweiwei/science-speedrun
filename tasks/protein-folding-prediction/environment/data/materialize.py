#!/usr/bin/env python3
"""Download pinned data shards, verify bytes, and restore the unchanged /data layout."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

HERE = Path(__file__).resolve().parent
ROOTS = {'manifests', 'objects', 'features', 'foldbench', 'models'}
CANONICAL = {
    'manifests/foldingtrain-100k-v1/foldingtrain_100k.jsonl':
        'c2b7e935b130fc5f423916beb2577f7e06afb40943fde7c1d55192604e17a229',
    'foldbench/suite/suite_manifest.json':
        'aeb4883858133891d9dbdc7ed8dfbbbb0bc6d7266a349a64fd6fe7c31494420f',
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checked_name(name):
    if not isinstance(name, str):
        raise ValueError('Invalid data path')
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or str(p)!=name or not p.parts:
        raise ValueError(f'Unsafe data path: {name}')
    return p


def target_path(root, name):
    p = checked_name(name)
    if p.parts[0] not in ROOTS:
        raise ValueError(f'Not an allowed data subtree: {name}')
    target = root.joinpath(*p.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Escaping data path: {name}')
    cursor = target
    while cursor != root:
        if cursor.is_symlink():
            raise ValueError(f'Symlink in data path: {name}')
        cursor = cursor.parent
    return target


def verify_file(path, record):
    if (not path.is_file() or path.is_symlink() or path.stat().st_size!=record['size']
            or sha256(path)!=record['sha256']):
        raise ValueError(f'File checksum/size mismatch: {path}')


def load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if manifest.get('format')!='foldbench-data-archives/v1':
        raise ValueError('Unknown archive manifest')
    seen = set()
    for record in [manifest['inventory'], *manifest['archives']]:
        name = record['path']; checked_name(name)
        if (name in seen or type(record.get('size')) is not int or record['size']<0
                or not re.fullmatch('[0-9a-f]{64}',record.get('sha256',''))):
            raise ValueError('Invalid/duplicate archive record')
        seen.add(name)
    if manifest['inventory']['path']!='payload-files.jsonl':
        raise ValueError('Unexpected inventory path')
    for record in manifest['archives']:
        if (not re.fullmatch(r'archives/[a-z]+-[0-9]{4}\.tar',record['path'])
                or type(record.get('files')) is not int or record['files']<=0
                or type(record.get('payload_bytes')) is not int or record['payload_bytes']<0):
            raise ValueError('Invalid data shard')
    if (sum(r['files'] for r in manifest['archives'])!=manifest['payload_files']
            or sum(r['payload_bytes'] for r in manifest['archives'])!=manifest['payload_bytes']):
        raise ValueError('Archive totals differ from release')
    return manifest


def load_inventory(path, manifest):
    verify_file(path,manifest['inventory'])
    rows = []; seen = set()
    with path.open() as stream:
        for line in stream:
            row = json.loads(line); name = row['path']
            if (checked_name(name).parts[0] not in ROOTS or name in seen
                    or type(row.get('size')) is not int or row['size']<0
                    or not re.fullmatch('[0-9a-f]{64}',row.get('sha256',''))):
                raise ValueError('Invalid/duplicate payload record')
            seen.add(name); rows.append(row)
    if len(rows)!=manifest['payload_files'] or sum(r['size'] for r in rows)!=manifest['payload_bytes']:
        raise ValueError('Payload inventory totals differ from release')
    return rows


def unpack(archive, destination, records):
    expected = {r['path']:r for r in records}; seen = set()
    # An interrupted write may leave a disposable temporary file, never an
    # incomplete payload or an extra file inside an agent-visible subtree.
    temporary_dir = destination/'.foldbench-download'/'tmp'
    if temporary_dir.is_symlink() or temporary_dir.parent.is_symlink():
        raise ValueError('Unpack cache must not be a symlink')
    temporary_dir.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive, 'r:') as source:
        for member in source:
            if (not member.isfile() or member.name not in expected or member.name in seen
                    or member.size!=expected[member.name]['size']):
                raise ValueError(f'Unexpected/duplicate/special archive entry: {member.name}')
            seen.add(member.name); record = expected[member.name]
            target = target_path(destination,member.name)
            if target.exists():
                verify_file(target,record)
                continue
            target.parent.mkdir(parents=True,exist_ok=True)
            fd, name = tempfile.mkstemp(prefix='.foldbench-unpack-',dir=temporary_dir)
            temporary = Path(name)
            try:
                digest = hashlib.sha256()
                with os.fdopen(fd,'wb') as out, source.extractfile(member) as content:
                    os.fchmod(out.fileno(),0o644)
                    for block in iter(lambda: content.read(4*1024*1024),b''):
                        digest.update(block);out.write(block)
                if temporary.stat().st_size!=record['size'] or digest.hexdigest()!=record['sha256']:
                    raise ValueError(f'Payload checksum mismatch: {member.name}')
                os.link(temporary,target)  # Atomic create; never overwrite.
            finally:
                temporary.unlink(missing_ok=True)
    if seen!=set(expected):
        raise ValueError('Archive is missing expected payloads')


def materialize(root, manifest, repo, revision, *, verify_only=False, keep_archives=False):
    root = Path(root).resolve()
    if not verify_only:
        root.mkdir(parents=True,exist_ok=True)
    cache = root/'.foldbench-download'
    if cache.is_symlink():
        raise ValueError('Download cache must not be a symlink')
    if not verify_only:
        cache.mkdir(exist_ok=True)
    def fetch(record):
        path = cache/record['path']
        if path.is_symlink() or path.parent.is_symlink() or not path.resolve().is_relative_to(cache):
            raise ValueError('Download path must remain inside the cache')
        if not verify_only:
            env = dict(os.environ)
            env.setdefault('HF_XET_CACHE',str(cache/'xet'))
            subprocess.run(['hf','download',repo,record['path'],'--type','dataset',
                '--revision',revision,'--local-dir',str(cache)],check=True,env=env)
        verify_file(path,record)
        return path
    rows = load_inventory(fetch(manifest['inventory']),manifest)
    offset = 0
    for archive in manifest['archives']:
        subset = rows[offset:offset+archive['files']]; offset+=archive['files']
        if sum(r['size'] for r in subset)!=archive['payload_bytes']:
            raise ValueError('Shard inventory ordering/size mismatch')
        missing = False
        for row in subset:
            target = target_path(root,row['path'])
            if target.exists():
                verify_file(target,row)
            else:
                missing = True
        if missing:
            if verify_only:
                raise ValueError(f'Missing files in shard {archive["path"]}')
            path = fetch(archive)
            unpack(path,root,subset)
            if not keep_archives:
                path.unlink()  # Only this verified disposable downloaded shard.
        print(json.dumps({'verified_payloads':offset,'total':len(rows),'shard':archive['path']}),flush=True)
    expected = {r['path'] for r in rows}; actual = set()
    for folder in ROOTS:
        for path in (root/folder).rglob('*'):
            if path.is_symlink():
                raise ValueError('Symlink in materialized release')
            if path.is_file():
                actual.add(str(path.relative_to(root)))
    if actual!=expected:
        raise ValueError('Materialized file membership differs from release; unrelated files are not removed')
    return {'repo_id':repo,'revision':revision,'payload_files':len(rows),
        'payload_bytes':manifest['payload_bytes'],'inventory_sha256':manifest['inventory']['sha256']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('destination',type=Path)
    p.add_argument('--verify-only',action='store_true')
    p.add_argument('--keep-archives',action='store_true')
    args=p.parse_args()
    config=json.loads((HERE/'download-source.json').read_text())
    repo=os.environ.get('FOLDBENCH_HF_REPO',config['repo_id'])
    revision=os.environ.get('FOLDBENCH_HF_REVISION',config['revision'])
    if repo.startswith('__') or revision.startswith('__'):
        p.error('Data are not published yet: configure the HF repository and commit placeholders.')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',repo) or not re.fullmatch('[a-fA-F0-9]{40}',revision):
        p.error('Expected an owner/repository and a full 40-character immutable commit SHA.')
    if not args.verify_only and not shutil.which('hf'):
        p.error('Install the HF CLI first (hf download).')
    manifest_path=HERE/'release-manifest.json'
    if sha256(manifest_path)!=config['manifest_sha256']:
        raise ValueError('Archive manifest changed from the pinned publication')
    manifest=load_manifest(manifest_path)
    receipt=materialize(args.destination,manifest,repo,revision,
        verify_only=args.verify_only,keep_archives=args.keep_archives)
    for relative,digest in CANONICAL.items():
        if sha256(args.destination/relative)!=digest:
            raise ValueError(f'Canonical scientific manifest changed: {relative}')
    subprocess.run([sys.executable,str(HERE/'validate_release.py'),str(args.destination)],check=True)
    receipt.update(release_id=manifest['release_id'],manifest_sha256=config['manifest_sha256'])
    if not args.verify_only:
        receipt_path = args.destination/'materialized.json'
        if receipt_path.is_symlink():
            raise ValueError('Materialization receipt must not be a symlink')
        receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    print('Pinned FoldBench data verified offline.' if args.verify_only else
          'Pinned FoldBench data downloaded, restored and verified.')


if __name__=='__main__':
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode)
