"""Publish a complete directory atomically without exposing a partial model."""

import json
import hashlib
import os
from pathlib import Path
import shutil
import tempfile


def atomic_text(path, content):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(content)
    temporary.chmod(0o644)
    os.replace(temporary, path)


def publish_directory(destination, build):
    destination = Path(destination).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError('submission destination must not be a symlink')
    destination.mkdir(exist_ok=True)
    if destination.stat().st_uid == os.getuid():
        destination.chmod(destination.stat().st_mode | 0o555)
    versions = destination / '.versions'
    versions.mkdir(exist_ok=True)
    versions.chmod(0o755)
    stage = Path(tempfile.mkdtemp(prefix='generation-', dir=versions))
    try:
        build(stage)
        for name in ['infer.sh', 'model_manifest.json', 'README.md']:
            if not (stage / name).is_file():
                raise ValueError(f'incomplete publication: {name}')
        for path in stage.rglob('*'):
            if path.is_symlink() and not path.resolve().is_relative_to(stage.resolve()):
                raise ValueError('publication contains an escaping symlink')
        # mkdtemp creates mode 0700. The relocated verifier runs as a different
        # UID, so every directory must be traversable and every asset readable.
        for path in [stage, *stage.rglob('*')]:
            if not path.is_symlink():
                path.chmod(0o755 if path.is_dir() or path.stat().st_mode & 0o111 else 0o644)
        manifest = json.loads((stage / 'model_manifest.json').read_text())
        for relative in [manifest['checkpoint']['path'], manifest['model_config']]:
            path = stage / relative
            if Path(relative).is_absolute() or not path.resolve().is_relative_to(stage.resolve()) or not path.is_file():
                raise ValueError('publication manifest references a missing or external asset')
        checkpoint = stage / manifest['checkpoint']['path']
        digest = hashlib.sha256()
        with checkpoint.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != manifest['checkpoint']['sha256']:
            raise ValueError('publication checkpoint checksum mismatch')
        prefix = stage.relative_to(destination)
        manifest['checkpoint']['path'] = str(prefix / manifest['checkpoint']['path'])
        manifest['model_config'] = str(prefix / manifest['model_config'])
        # Use a single atomic manifest replacement: rename-exchange of directories
        # is not supported by some cluster filesystems. Referenced generations
        # remain immutable and inside the package, including during live snapshots.
        if (destination / 'infer.sh').exists() and not (destination / 'dispatch.py').exists():
            shutil.copyfile(destination / 'infer.sh', destination / 'legacy-infer.sh')
        dispatch = '''import json, os, pathlib, sys
root = pathlib.Path(__file__).resolve().parent
manifest = json.loads((root / 'model_manifest.json').read_text())
config = (root / manifest['model_config']).resolve()
if not config.is_relative_to(root):
    raise ValueError('manifest path escapes package')
generation = config.parent
script = generation / ('legacy-infer.sh' if generation == root else 'infer.sh')
os.execv('/bin/bash', ['bash', str(script), *sys.argv[1:]])
'''
        atomic_text(destination / 'dispatch.py', dispatch)
        atomic_text(destination / 'infer.sh', '#!/usr/bin/env bash\nset -euo pipefail\n'
                    'root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"\n'
                    'exec python3 "$root/dispatch.py" "$@"\n')
        (destination / 'infer.sh').chmod(0o755)
        atomic_text(destination / 'README.md', (stage / 'README.md').read_text())
        atomic_text(destination / 'model_manifest.json', json.dumps(manifest, indent=2) + '\n')
        stage = None
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def snapshot_package(source, destination):
    """Copy a frozen publication while omitting its unused old generations."""
    source, destination = Path(source), Path(destination)
    manifest = json.loads((source / 'model_manifest.json').read_text())
    parts = Path(manifest['model_config']).parts
    selected = parts[1] if len(parts) >= 3 and parts[0] == '.versions' and (source / 'dispatch.py').is_file() else None
    def ignore(directory, names):
        excluded = {name for name in names if name == '__pycache__' or name.endswith('.tmp')}
        if selected and Path(directory) == source / '.versions':
            excluded.update(name for name in names if name != selected)
        return excluded
    shutil.copytree(source, destination, ignore=ignore)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Snapshot a frozen package without unused publication generations')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output == source or output.is_relative_to(source):
        parser.error('snapshot output must be outside the source package')
    snapshot_package(source, output)
