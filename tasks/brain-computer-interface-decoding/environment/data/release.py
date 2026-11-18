"""Install or build the prepared data releases (controller side, never in an agent container).

    python release.py install --root ROOT                (HF_TOKEN with FACED/TUEV access)
    python release.py manifest --root RELEASE            write RELEASE/release-manifest.json

`install` assembles ROOT/runtime/development (TASK_DATA_DIR) and ROOT/runtime/private
(TASK_PRIVATE_DIR) from three pinned sources and verifies every file:

- AD65, OpenBMI and Longitudinal: the public prepared release in download-source.json, built
  by materialize.py from the pinned public sources (CC0 / GigaDB);
- FACED and TUEV: the runtime trees of the controlled-access release in download-source.json.
  Its access requires the providers' terms (Synapse, TUH) and approval by the repository owner;
- pretrained/: the three initialization checkpoints pinned in
  ../baseline/braindecode/pretrained.lock.json, from their public Hugging Face repositories.
"""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCK = HERE.parent / 'baseline' / 'braindecode' / 'pretrained.lock.json'
ACTIVE_TRACKS = frozenset({'tuev', 'ad65', 'openbmi', 'longitudinal'})


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def manifest(root):
    root = Path(root)
    rows = [{'path': str(p.relative_to(root)), 'bytes': p.stat().st_size, 'sha256': sha256(p)}
            for p in sorted(root.rglob('*'))
            if p.is_file() and p.name != 'release-manifest.json' and '.cache' not in p.parts]
    (root / 'release-manifest.json').write_text(
        json.dumps({'schema': 'bci-hf-release-v1', 'files': rows}, indent=1) + '\n')
    return len(rows)


def subset(download, prefix):
    """Verify the files under PREFIX of a release that was downloaded partially."""
    rows = [r for r in json.loads((download / 'release-manifest.json').read_text())['files']
            if r['path'].startswith(prefix)]
    if not rows:
        raise ValueError(f'no {prefix} files in the release manifest')
    for r in rows:
        path = download / r['path']
        if '..' in Path(r['path']).parts or not path.is_file() or path.stat().st_size != r['bytes']:
            raise ValueError(f'missing file or incorrect size: {r["path"]}')
        if sha256(path) != r['sha256']:
            raise ValueError(f'checksum mismatch: {r["path"]}')
    listed = {r['path'] for r in rows}
    extra = [p for p in (download / prefix).rglob('*') if p.is_file()
             and str(p.relative_to(download)) not in listed]
    if extra:
        raise ValueError(f'unlisted files under {prefix}: {extra[:3]}')
    return {'status': 'verified', 'files': len(rows), 'bytes': sum(r['bytes'] for r in rows)}


def move_tree(source, target):
    if target.exists():
        raise FileExistsError(f'{target} exists; install into a new root')
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, target)


def install(root):
    from huggingface_hub import hf_hub_download, snapshot_download
    root = Path(root).resolve()
    pins = json.loads((HERE / 'download-source.json').read_text())
    work = root / '.download'
    runtime = root / 'runtime'
    for name in ('public', 'controlled'):
        pin = pins[name]
        download = work / name
        snapshot_download(pin['repo_id'], repo_type='dataset', revision=pin['revision'],
                          local_dir=download, allow_patterns=['release-manifest.json', 'runtime/**'])
        if sha256(download / 'release-manifest.json') != pin['manifest_sha256']:
            raise ValueError(f'{name}: release manifest differs from the pin')
        print(name, json.dumps(subset(download, 'runtime/')), flush=True)
        for split in ('development', 'private'):
            for track in pin['tracks']:
                if track in ACTIVE_TRACKS:
                    move_tree(download / 'runtime' / split / track, runtime / split / track)
    pretrained = runtime / 'development' / 'pretrained'
    pretrained.mkdir(parents=True, exist_ok=True)
    for family, entry in json.loads(LOCK.read_text())['models'].items():
        path = hf_hub_download(entry['repo'], entry['file'], revision=entry['revision'],
                               local_dir=work / 'pretrained' / family)
        if sha256(path) != entry['sha256']:
            raise ValueError(f'{family}: pretrained checkpoint differs from the lock')
        shutil.copyfile(path, pretrained / f'bci-{family}.safetensors')
    (runtime / 'private').chmod(0o700)
    shutil.rmtree(work)
    print(f'TASK_DATA_DIR={runtime / "development"}\nTASK_PRIVATE_DIR={runtime / "private"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('command', choices=['install', 'manifest'])
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'install':
        install(args.root)
    else:
        print(manifest(args.root), 'files')


if __name__ == '__main__':
    main()
