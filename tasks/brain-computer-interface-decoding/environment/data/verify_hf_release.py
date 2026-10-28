"""Verify a downloaded BCI HF release using its pinned SHA-256 manifest."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'release-manifest.json').read_text())
    if manifest.get('schema') != 'bci-hf-release-v1':
        raise ValueError('unsupported release manifest')
    seen = set()
    total = 0
    for row in manifest['files']:
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts or str(relative) in seen:
            raise ValueError('unsafe or duplicate release path')
        seen.add(str(relative))
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError('release path escapes download directory')
        if not re.fullmatch('[0-9a-f]{64}', row['sha256']):
            raise ValueError('invalid SHA-256 digest')
        if not path.is_file() or path.stat().st_size != row['bytes']:
            raise ValueError(f'missing file or incorrect size: {relative}')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != row['sha256']:
            raise ValueError(f'checksum mismatch: {relative}')
        total += row['bytes']
    return {'status': 'verified', 'files': len(seen), 'bytes': total}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('download_directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.download_directory)))
