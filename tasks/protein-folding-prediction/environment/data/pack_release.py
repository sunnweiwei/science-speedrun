#!/usr/bin/env python3
"""Package an already-audited release into byte-preserving, bounded tar shards."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import tarfile


class HashedIO:
    def __init__(self, stream):
        self.stream, self.digest = stream, hashlib.sha256()

    def read(self, size=-1):
        block = self.stream.read(size)
        self.digest.update(block)
        return block

    def write(self, block):
        size = self.stream.write(block)
        self.digest.update(block[:size])
        return size

    def tell(self):
        return self.stream.tell()


def pack(source, output, shard_bytes=1_000_000_000):
    if shard_bytes <= 0:
        raise ValueError('Shard size must be positive')
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Source and upload staging must be disjoint')
    output.mkdir(parents=True, exist_ok=True)
    archives_dir = output/'archives'
    archives_dir.mkdir(exist_ok=False)
    allowed = {'manifests','objects','features','foldbench','models'}
    groups = defaultdict(list)
    for path in sorted(source.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'Symlink in release: {path}')
        if path.is_dir():
            continue
        relative = path.relative_to(source)
        if not path.is_file() or relative.parts[0] not in allowed:
            raise ValueError(f'Unexpected release entry: {relative}')
        if relative.parts[0]=='models' and not str(relative).startswith('models/esmfold2-fast-b28d8ace/'):
            raise ValueError('Only the approved reference checkpoint is allowed')
        groups[relative.parts[0]].append((path, str(relative), path.stat().st_size))
    if set(groups) != allowed:
        raise ValueError('Missing release component')
    manifest = {'format':'foldbench-data-archives/v1','release_id':'foldbench-100k-esmc-int8-v1',
        'payload_files':sum(len(rows) for rows in groups.values()),
        'payload_bytes':sum(row[2] for rows in groups.values() for row in rows),
        'groups':{name:{'files':len(rows),'bytes':sum(r[2] for r in rows)} for name,rows in groups.items()},
        'archives':[]}
    inventory_path = output/'payload-files.jsonl'
    with inventory_path.open('xb') as inventory:
        inventory_hash = HashedIO(inventory)
        for group, rows in sorted(groups.items()):
            batches, batch, size = [], [], 0
            for row in rows:
                if batch and size + row[2] > shard_bytes:
                    batches.append(batch); batch, size = [], 0
                batch.append(row); size += row[2]
            if batch:
                batches.append(batch)
            for number, batch in enumerate(batches):
                target = archives_dir/f'{group}-{number:04d}.tar'
                with target.open('xb') as raw:
                    writer = HashedIO(raw)
                    with tarfile.open(fileobj=writer, mode='w', format=tarfile.PAX_FORMAT,
                                      copybufsize=1024*1024) as archive:
                        for path, relative, size in batch:
                            info = tarfile.TarInfo(relative)
                            info.size, info.mode, info.mtime = size, 0o644, 0
                            with path.open('rb') as stream:
                                reader = HashedIO(stream)
                                archive.addfile(info, reader)
                                if stream.tell()!=size or path.stat().st_size!=size:
                                    raise ValueError(f'Source changed while packing: {relative}')
                            row = {'path':relative,'size':size,'sha256':reader.digest.hexdigest()}
                            inventory_hash.write((json.dumps(row,separators=(',',':'))+'\n').encode())
                item = {'path':str(target.relative_to(output)),'size':target.stat().st_size,
                    'sha256':writer.digest.hexdigest(),'files':len(batch),
                    'payload_bytes':sum(row[2] for row in batch)}
                manifest['archives'].append(item)
                print(json.dumps(item), flush=True)
    manifest['inventory'] = {'path':inventory_path.name,'size':inventory_path.stat().st_size,
        'sha256':inventory_hash.digest.hexdigest()}
    (output/'release-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'complete':True,'archives':len(manifest['archives']),
        'files':manifest['payload_files'],'bytes':manifest['payload_bytes']}), flush=True)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();pack(args.source,args.output)
