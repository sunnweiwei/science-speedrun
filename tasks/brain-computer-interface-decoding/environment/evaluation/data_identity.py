"""Bind a prepared split to both metadata and its complete shard hash inventory."""
import hashlib
from pathlib import Path


def split_identity(directory, truth=None):
    directory = Path(directory)
    parts = [directory / 'manifest.json', directory / 'checksums.json']
    if truth is not None:
        parts.append(Path(truth))
    digest = hashlib.sha256(b'bci-split-identity-v2\0')
    for path in parts:
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, 'big'))
        digest.update(payload)
    return digest.hexdigest()
