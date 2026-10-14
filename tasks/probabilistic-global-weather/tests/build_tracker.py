"""Build the pinned TempestExtremes CPU tracker in the evaluator image."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile


def build(lock_path, output):
    lock = json.loads(lock_path.read_text())
    with tempfile.TemporaryDirectory(prefix='weather-tracker-') as tmp:
        work = Path(tmp)
        archive = work / 'source.zip'
        with urllib.request.urlopen(lock['archive_url'], timeout=120) as source, archive.open('wb') as target:
            shutil.copyfileobj(source, target)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != lock['sha256']:
            raise ValueError('Tracker source differs from pinned release')
        with zipfile.ZipFile(archive) as source:
            for entry in source.infolist():
                if not (work / entry.filename).resolve().is_relative_to(work):
                    raise ValueError('Invalid tracker archive path')
            source.extractall(work)
        source = next(work.glob('ClimateGlobalChange-*'))
        options = ['PARALLEL=NONE', 'NETCDF_HOME=/usr']
        for target in ['src/netcdf-cxx-4.2', 'src/base', 'src/nodes']:
            subprocess.run(['make', '-j4', *options, target], cwd=source, check=True)
        output.mkdir(parents=True, exist_ok=True)
        for name in ['DetectNodes', 'StitchNodes']:
            shutil.copy2(source / 'bin' / name, output / name)
        shutil.copy2(lock_path, output / 'source.lock.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.lock, args.output)
