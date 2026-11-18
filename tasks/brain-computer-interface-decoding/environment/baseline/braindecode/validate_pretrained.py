"""Validate staged allowlisted weights and reject disallowed checkpoints before agent exposure."""
import argparse
import hashlib
import json
from pathlib import Path


def validate_directory(directory: Path) -> dict:
    directory = Path(directory)
    lock = json.loads(Path(__file__).with_name('pretrained.lock.json').read_text())
    disallowed_hashes = {
        info['sha256']: family for family, info in lock.get('disallowed_models', {}).items()
    }
    for family in lock.get('disallowed_models', {}):
        forbidden = directory / f'bci-{family}.safetensors'
        if forbidden.exists():
            raise ValueError(f'{family}: disallowed pretrained checkpoint must not be staged ({forbidden})')
    for candidate in sorted(directory.glob('*.safetensors')):
        h = hashlib.sha256()
        with candidate.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
        digest = h.hexdigest()
        if digest in disallowed_hashes:
            raise ValueError(f'{disallowed_hashes[digest]}: disallowed pretrained checkpoint hash staged ({candidate})')
    verified = {}
    for family, info in lock['models'].items():
        path = directory / f'bci-{family}.safetensors'
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
        if h.hexdigest() != info['sha256']:
            raise ValueError(f'{family}: checkpoint checksum mismatch')
        verified[family] = info['sha256']
    expected_files = {f'bci-{family}.safetensors' for family in lock['models']}
    actual_files = {p.name for p in directory.iterdir() if p.is_file()}
    unexpected = sorted(actual_files - expected_files)
    if unexpected:
        raise ValueError(f'unexpected files in pretrained directory: {unexpected}')
    return verified


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    verified = validate_directory(args.directory)
    print(f'verified {len(verified)} allowlisted checkpoints: {", ".join(sorted(verified))}')


if __name__ == '__main__':
    main()
