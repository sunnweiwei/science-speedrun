#!/usr/bin/env python3
"""Fail before an agent run if the promised research resources are incomplete."""

import argparse
import hashlib
import json
from pathlib import Path


def check(root, *, hashes=False):
    manifests = {}
    for relative in ['manifest.json', 'synthetic_augmentation/manifest.json', 'glacier/manifest.json']:
        path = root / relative
        manifest = json.loads(path.read_text())
        manifests[relative] = manifest
        for name, entry in manifest['files'].items():
            expected = entry['sha256'] if isinstance(entry, dict) else entry
            file = path.parent / name
            if not file.resolve().is_relative_to(root.resolve()):
                raise ValueError('resource path escapes development data')
            if not file.is_file() or (isinstance(entry, dict) and file.stat().st_size != entry['bytes']):
                raise ValueError(f'resource file missing or wrong size: {file}')
            if hashes:
                digest = hashlib.sha256()
                with file.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(block)
                if digest.hexdigest() != expected:
                    raise ValueError(f'resource hash mismatch: {file}')
    if manifests['synthetic_augmentation/manifest.json'].get('rows') != 2333485:
        raise ValueError('incomplete held-out-clean synthetic corpus')
    native = manifests['glacier/manifest.json']
    preparation = json.loads((root / 'glacier/preparation.json').read_text())
    if native.get('schema') != 'massspecgym-glacier-training/v1' or native['config'] != preparation:
        raise ValueError('invalid or inconsistent native preparation manifest')
    if preparation['limits'] != [None, None]:
        raise ValueError('subset MAGMa preparation is for smoke tests, not the full task')
    if len(preparation.get('native_source_sha256', '')) != 64:
        raise ValueError('native preparation source hash is missing')
    for split, expected in [('train', 194119), ('validation', 19429)]:
        if native['counts'][split]['available'] != expected:
            raise ValueError(f'incomplete GLACIER input fold: {split}')
        if native['counts'][split]['selected'] != native['counts'][split]['eligible']:
            raise ValueError(f'incomplete GLACIER condition selection: {split}')
        if not 0 < native['counts'][split]['annotated'] <= native['counts'][split]['selected']:
            raise ValueError(f'invalid GLACIER annotation coverage: {split}')
        if preparation['source_files'][split] != manifests['manifest.json']['files'][f'{split}.jsonl']:
            raise ValueError('fragment supervision belongs to a different development release')
        if not (root / f'{split}.jsonl').is_file():
            raise FileNotFoundError(split)
    return {'ready': True, 'verified_hashes': hashes, 'glacier_counts': native['counts']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/workspace/data'))
    parser.add_argument('--hashes', action='store_true')
    args = parser.parse_args()
    print(json.dumps(check(args.data_root, hashes=args.hashes), indent=2))
