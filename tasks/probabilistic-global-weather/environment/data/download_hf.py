"""Download a pinned prepared WN2 release on the operator host, then verify it."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_path(root, name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or p.as_posix() != name:
        raise ValueError(f'Unsafe release path: {name}')
    dest = root / name
    if dest.is_symlink() or not dest.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Unsafe destination: {name}')
    for parent in dest.parents:
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError(f'Symlink destination: {name}')
    return dest


def validate_manifest(manifest):
    if manifest.get('schema') != 'wn2-hf-release/v1' or not manifest.get('shards'):
        raise ValueError('Invalid release manifest')
    seen, shard_names = set(), set()
    for shard in manifest['shards']:
        name = shard['path']
        scope = shard['scope']
        if scope not in ('development', 'evaluation') or not name.startswith(f'shards/{scope}/'):
            raise ValueError('Invalid shard scope/path')
        safe_path(Path('/release'), name)
        if name in shard_names or not re.fullmatch('[0-9a-f]{64}', shard['sha256']):
            raise ValueError('Duplicate shard or invalid hash')
        shard_names.add(name)
        if not isinstance(shard['bytes'], int) or not 0 < shard['bytes'] <= 9 * 1024**3:
            raise ValueError('Invalid shard size')
        if not shard['members']:
            raise ValueError('Empty shard')
        for member in shard['members']:
            safe_path(Path('/release'), member)
            if member in seen:
                raise ValueError('Duplicate member')
            seen.add(member)
            if scope == 'evaluation':
                allowed = member.startswith('evaluation-resources/')
            else:
                allowed = (member == 'IBTrACS.since1980.v04r01.nc' or
                           bool(re.match(r'training-2017-2020/(case-\d{3}|statistics)/', member)) or
                           bool(re.match(r'development-(inputs|weather|precipitation)-2021/', member)))
            if not allowed:
                raise ValueError('Resource outside scope allowlist')
            spec = manifest['files'][member]
            if (spec['shard'] != name or not isinstance(spec['bytes'], int) or spec['bytes'] < 0
                    or not re.fullmatch('[0-9a-f]{64}', spec['sha256'])):
                raise ValueError('Invalid member record')
    if seen != set(manifest['files']) or len(seen) != manifest['total_files']:
        raise ValueError('Incomplete member inventory')
    if sum(f['bytes'] for f in manifest['files'].values()) != manifest['total_bytes']:
        raise ValueError('Wrong byte count')
    for name, spec in manifest.get('metadata', {}).items():
        safe_path(Path('/release'), name)
        if not (name in ('README.md', 'LICENSES.md') or name.startswith(('licenses/', 'provenance/'))):
            raise ValueError('Unexpected release metadata path')
        if (not isinstance(spec['bytes'], int) or not 0 <= spec['bytes'] <= 16 * 1024**2
                or not re.fullmatch('[0-9a-f]{64}', spec['sha256'])):
            raise ValueError('Invalid metadata record')


def matches(path, spec):
    return path.is_file() and path.stat().st_size == spec['bytes'] and digest(path) == spec['sha256']


def extract_shard(archive, root, shard, files):
    if not matches(archive, shard):
        raise ValueError(f'Corrupt archive: {shard["path"]}')
    expected, seen = set(shard['members']), set()
    with tarfile.open(archive, 'r:') as tar:
        for member in tar:
            name = member.name
            if not member.isfile() or member.issparse() or name not in expected or name in seen:
                raise ValueError(f'Unexpected archive member: {name}')
            seen.add(name)
            spec = files[name]
            if member.size != spec['bytes']:
                raise ValueError(f'Wrong member size: {name}')
            dest = safe_path(root, name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            temporary = safe_path(root, name + '.hf-part')
            h = hashlib.sha256()
            with tar.extractfile(member) as source, temporary.open('wb') as target:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                    h.update(block)
                    target.write(block)
            if h.hexdigest() != spec['sha256']:
                temporary.unlink()
                raise ValueError(f'Corrupt extracted member: {name}')
            temporary.chmod(0o644)
            temporary.replace(dest)
    if seen != expected:
        raise ValueError('Missing archive members')


def run(args):
    source = json.loads(args.source.read_text())
    if not re.fullmatch('[0-9a-f]{40}', source['revision']):
        raise ValueError('Expected an immutable full HF commit SHA')
    root = args.target.resolve()
    root.mkdir(parents=True, exist_ok=True)
    state = safe_path(root, '.hf-release')
    state.mkdir(exist_ok=True)
    with safe_path(state, 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cache = safe_path(state, 'download')
        def fetch(name):
            from huggingface_hub import hf_hub_download
            return Path(hf_hub_download(repo_id=source['repo_id'], repo_type='dataset',
                                       revision=source['revision'], filename=name,
                                       local_dir=cache, token=False))
        manifest_path = safe_path(state, 'release.json') if args.verify_only else fetch('release.json')
        if digest(manifest_path) != source['manifest_sha256']:
            raise ValueError('Release manifest hash mismatch')
        manifest = json.loads(manifest_path.read_text())
        validate_manifest(manifest)
        if not args.verify_only:
            safe_path(state, 'release.json').write_bytes(manifest_path.read_bytes())
        ready = safe_path(state, f'{args.scope}-ready.json')
        ready.unlink(missing_ok=True)
        for name, spec in manifest.get('metadata', {}).items():
            dest = safe_path(state, 'metadata/' + name)
            if not matches(dest, spec):
                if args.verify_only:
                    raise ValueError(f'Missing/corrupt release metadata: {name}')
                downloaded = fetch(name)
                if not matches(downloaded, spec):
                    raise ValueError(f'Corrupt release metadata: {name}')
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(downloaded.read_bytes())
        selected = [s for s in manifest['shards'] if args.scope == 'all' or s['scope'] == args.scope]
        for shard in selected:
            complete = all(matches(safe_path(root, n), manifest['files'][n]) for n in shard['members'])
            if not complete:
                if args.verify_only:
                    raise ValueError(f'Missing/corrupt resources: {shard["path"]}')
                archive = fetch(shard['path'])
                try:
                    extract_shard(archive, root, shard, manifest['files'])
                except (ValueError, tarfile.TarError):
                    # Do not leave a corrupt local archive poisoning subsequent retries.
                    archive.unlink(missing_ok=True)
                    raise
                if not args.keep_archives:
                    archive.unlink()
            print(json.dumps({'verified': shard['path']}), flush=True)
        receipt = {'status': 'passed', 'repo_id': source['repo_id'], 'revision': source['revision'],
                   'manifest_sha256': source['manifest_sha256'], 'scope': args.scope,
                   'shards': len(selected)}
        tmp = safe_path(state, ready.name + '.tmp')
        tmp.write_text(json.dumps(receipt, indent=2) + '\n')
        tmp.replace(ready)
        print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target', type=Path)
    parser.add_argument('--source', type=Path, default=Path(__file__).with_name('download-source.json'))
    parser.add_argument('--scope', choices=('all', 'development', 'evaluation'), default='all')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--keep-archives', action='store_true')
    run(parser.parse_args())
