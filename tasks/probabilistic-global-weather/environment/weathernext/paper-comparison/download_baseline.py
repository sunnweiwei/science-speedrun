#!/usr/bin/env python3
"""Fetch generation-pinned public GenCast assets and verify every byte."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8 << 20): h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    lock = json.loads((Path(__file__).parent/'baseline-assets.lock.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for entry in lock['files']:
        destination = args.output/entry['file']
        if destination.exists():
            if destination.stat().st_size != entry['bytes'] or digest(destination) != entry['sha256']:
                raise ValueError(f'Existing asset differs: {destination}')
            continue
        temporary = destination.with_suffix(destination.suffix+'.part')
        with urllib.request.urlopen(entry['url'], timeout=120) as response, temporary.open('wb') as stream:
            while block := response.read(8 << 20): stream.write(block)
        if temporary.stat().st_size != entry['bytes'] or digest(temporary) != entry['sha256']:
            raise ValueError(f'Download failed checksum: {destination}')
        temporary.replace(destination)
        print(json.dumps(dict(file=entry['file'], status='verified')), flush=True)


if __name__ == '__main__': main()
