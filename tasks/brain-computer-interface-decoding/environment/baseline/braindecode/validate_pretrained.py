"""Validate staged allowlisted weights before exposing them to an agent."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    lock = json.loads(Path(__file__).with_name('pretrained.lock.json').read_text())
    for family, info in lock['models'].items():
        path = args.directory / f'bci-{family}.safetensors'
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
        if h.hexdigest() != info['sha256']:
            raise ValueError(f'{family}: checkpoint checksum mismatch')
    print('all three allowlisted checkpoints verified')


if __name__ == '__main__':
    main()
