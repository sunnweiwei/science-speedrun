#!/usr/bin/env python3
"""Exercise absent private paths, offline sockets and read-only replay mounts."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from common import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True, help='Small public validation view')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    secret = args.output / 'organizer-private-canary'
    secret.write_text('not visible to the submission\n')
    package = args.output / 'submission'
    package.mkdir()
    (package / 'weights.pt').write_bytes(b'isolation probe only')
    (package / 'config.json').write_text('{}')
    (package / 'README.md').write_text('Kernel boundary probe, not a trained benchmark model.\n')
    write_json(package / 'model_manifest.json', {
        'schema': 'massspecgym-retrieval-model/v1',
        'checkpoint': {'path': 'weights.pt', 'sha256': hashlib.sha256((package / 'weights.pt').read_bytes()).hexdigest(), 'format': 'pytorch'},
        'model_config': 'config.json', 'training_provenance': {'data': 'isolation probe', 'pretrained_assets': []},
    })
    (package / 'infer.sh').write_text('#!/bin/bash\nexec python3 /submission/probe.py "$@"\n')
    probe = '''import json, os, pathlib, socket, sys
input_root, output_root = map(pathlib.Path, sys.argv[1:])
assert not pathlib.Path(PRIVATE_CANARY).exists(), 'private path exposed'
assert not pathlib.Path('/workspace/data/train.jsonl').exists(), 'training data exposed'
assert not pathlib.Path('/private').exists(), 'private mount exposed'
assert not pathlib.Path('/run').exists(), 'host runtime sockets exposed'
for root in [pathlib.Path('/submission'), input_root]:
    try:
        (root / 'write-probe').write_text('bad')
    except OSError:
        pass
    else:
        raise AssertionError('read-only mount is writable')
sock = socket.socket()
sock.settimeout(0.5)
try:
    sock.connect(('198.51.100.1', 443))
except OSError:
    pass
else:
    raise AssertionError('external network route available')
finally:
    sock.close()
assert os.getuid() == 0  # mapped root only, with all capabilities dropped
status = pathlib.Path('/proc/self/status').read_text()
assert 'CapEff:\\t0000000000000000' in status
pools = {r['candidate_pool_id']: [c['candidate_id'] for c in r['candidates']] for r in map(json.loads, (input_root/'candidate_pools.jsonl').read_text().splitlines())}
with (output_root/'rankings.jsonl').open('w') as f:
    for row in map(json.loads, (input_root/'queries.jsonl').read_text().splitlines()):
        f.write(json.dumps({'query_id': row['query_id'], 'ranking': pools[row['candidate_pool_id']]})+'\\n')
print('private paths absent; mounts read-only; external route unavailable; effective capabilities empty')
'''
    (package / 'probe.py').write_text('PRIVATE_CANARY = ' + repr(str(secret)) + '\n' + probe)
    subprocess.run([sys.executable, str(Path(__file__).with_name('replay_local.py')),
                    '--runtime', str(args.runtime), '--input', str(args.input),
                    '--submission', str(package), '--output', str(args.output / 'replay'), '--timeout', '30'], check=True)
    print((args.output / 'replay/inference.log').read_text())


if __name__ == '__main__':
    main()
