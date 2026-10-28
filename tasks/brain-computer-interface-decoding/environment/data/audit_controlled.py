"""Write an authorized source inventory outside Git for maintainer replay."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('track', choices=('faced', 'tuev'))
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    if args.track == 'faced':
        files = sorted(p for p in source.rglob('*') if p.is_file() and p.name in ('data.mdb', 'lock.mdb'))
        if not any(p.name == 'data.mdb' for p in files):
            raise ValueError('FACED LMDB data.mdb missing')
    else:
        files = sorted(p for folder in ('processed_train', 'processed_eval', 'processed_test')
                       for p in (source / folder).glob('*.pkl'))
        from materialize import validate_tuev_layout
        validate_tuev_layout(tuple(len(list((source / folder).glob('*.pkl')))
                                   for folder in ('processed_train', 'processed_eval', 'processed_test')))
    entries = [{'path': str(p.relative_to(source)), 'bytes': p.stat().st_size,
                'sha256': digest(p)} for p in files]
    identity = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({'schema': 'bci-controlled-source-inventory-v1',
                          'track': args.track, 'file_count': len(entries),
                          'inventory_sha256': identity, 'files': entries}, indent=2) + '\n'
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        stream.write(payload)
    print(json.dumps({'track': args.track, 'file_count': len(entries),
                      'inventory_sha256': identity}))


if __name__ == '__main__':
    main()
