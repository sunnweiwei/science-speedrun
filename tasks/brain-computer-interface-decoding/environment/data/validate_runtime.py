"""Fail closed on incomplete or modified prepared BCI assets."""
import argparse
import hashlib
import json
from pathlib import Path

EXPECTED = {
    'tuev': (68445, 15487, 29421),
    'ad65': (3774, 1343, 1376),
    'openbmi': (12800, 4400, 4400),
    'longitudinal': (28523, 8418, 10028),
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_split(root, track, split, count):
    base = root / ('private' if split == 'test' else 'development') / track
    base = base / ('public' if split == 'test' else split)
    manifest = json.loads((base / 'manifest.json').read_text())
    checksums = json.loads((base / 'checksums.json').read_text())
    rows = manifest['samples']
    if manifest.get('schema') != 'bci-input-v1' or manifest.get('track') != track or len(rows) != count:
        raise ValueError(f'{track}/{split}: manifest identity or count mismatch')
    ids = [row['id'] for row in rows]
    if len(set(ids)) != count:
        raise ValueError(f'{track}/{split}: duplicate IDs')
    indexed = {row['path'] for row in rows}
    if indexed != {item['path'] for item in checksums}:
        raise ValueError(f'{track}/{split}: shard inventory mismatch')
    for item in checksums:
        path = (base / item['path']).resolve()
        if not path.is_relative_to(base.resolve()) or path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
            raise ValueError(f'{track}/{split}: shard checksum mismatch')
    if split == 'test':
        if manifest.get('storage_layout') != 'global-random-v2':
            raise ValueError(f'{track}: outdated private storage layout; repack before evaluation')
        truth = [json.loads(line) for line in (root / 'private' / track / 'truth.jsonl').read_text().splitlines()]
        if {row['id'] for row in truth} != set(ids) or len(truth) != count:
            raise ValueError(f'{track}: private truth identity mismatch')
        public_by_id = {row['id']: row for row in rows}
        if any(any(row[key] != public_by_id[row['id']][key] for key in ('path', 'index')) for row in truth):
            raise ValueError(f'{track}: public/private position mismatch')
        if len({(row['path'], row['index']) for row in rows}) != count:
            raise ValueError(f'{track}: duplicate sample positions')
        if any(set(row) != {'id', 'path', 'index'} for row in rows):
            raise ValueError(f'{track}: public manifest leaks metadata')
    return {'samples': count, 'manifest_sha256': digest(base / 'manifest.json')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    result = {}
    for track, counts in EXPECTED.items():
        result[track] = {split: validate_split(args.root, track, split, count)
                         for split, count in zip(('train', 'val', 'test'), counts)}
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
