"""Operator-only: archive the allowlisted prepared resources without changing bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def components():
    yield 'development', 'IBTrACS.since1980.v04r01.nc'
    yield 'development', 'training-2017-2020/statistics'
    for i in range(32):
        yield 'development', f'training-2017-2020/case-{i:03d}'
    for name in ('inputs', 'weather', 'precipitation'):
        yield 'development', f'development-{name}-2021'
    yield 'evaluation', 'evaluation-resources'


class HashedReader:
    def __init__(self, stream):
        self.stream = stream
        self.hash = hashlib.sha256()

    def read(self, size=-1):
        chunk = self.stream.read(size)
        self.hash.update(chunk)
        return chunk


def pack(root, output, max_bytes=8 * 1024**3):
    root, output = root.resolve(), output.resolve()
    if output.is_relative_to(root):
        raise ValueError('Release staging must be outside the input root')
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'release.json').exists():
        raise ValueError('Release manifest already exists; use a new staging directory')
    manifest = {'schema': 'wn2-hf-release/v1', 'files': {}, 'shards': []}
    for scope, component in components():
        source = root / component
        if not source.exists() or source.is_symlink():
            raise ValueError(f'Missing component: {component}')
        paths = [source] if source.is_file() else sorted(source.rglob('*'))
        groups, group, size = [], [], 0
        for path in paths:
            if path.is_symlink():
                raise ValueError(f'Symlink in release: {path}')
            if path.is_dir():
                continue
            if not path.is_file():
                raise ValueError(f'Not a regular file: {path}')
            n = path.stat().st_size
            if n > max_bytes:
                raise ValueError(f'File exceeds shard limit: {path}')
            if group and size + n > max_bytes:
                groups.append(group)
                group, size = [], 0
            group.append(path)
            size += n
        if group:
            groups.append(group)
        for index, group in enumerate(groups):
            name = f'shards/{scope}/{component.replace("/", "--")}-{index:03d}.tar'
            dest = output / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            temporary = dest.with_suffix('.part')
            members = []
            with tarfile.open(temporary, 'w', format=tarfile.PAX_FORMAT) as archive:
                for path in group:
                    relative = path.relative_to(root).as_posix()
                    before = path.stat()
                    info = tarfile.TarInfo(relative)
                    info.size, info.mode, info.mtime = before.st_size, 0o644, 0
                    with path.open('rb') as f:
                        reader = HashedReader(f)
                        archive.addfile(info, reader)
                    after = path.stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        raise ValueError(f'Source changed while packaging: {relative}')
                    manifest['files'][relative] = {
                        'bytes': info.size, 'sha256': reader.hash.hexdigest(), 'shard': name}
                    members.append(relative)
            temporary.replace(dest)
            shard = {'path': name, 'scope': scope, 'bytes': dest.stat().st_size,
                     'sha256': digest(dest), 'members': members}
            manifest['shards'].append(shard)
            print(json.dumps({'packed': name, 'bytes': shard['bytes']}), flush=True)
    manifest['total_bytes'] = sum(f['bytes'] for f in manifest['files'].values())
    manifest['total_files'] = len(manifest['files'])
    p = output / 'release.json'
    p.with_suffix('.tmp').write_text(json.dumps(manifest, indent=2) + '\n')
    p.with_suffix('.tmp').replace(p)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    pack(args.root, args.output)
