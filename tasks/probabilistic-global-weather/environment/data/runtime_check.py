"""Check preparation runtimes before acquiring large data; operator-only."""
import importlib
from importlib.metadata import version
from pathlib import Path
import subprocess
import sys

kind = sys.argv[1]
pins = ({'numpy': '2.2.6', 'xarray': '2025.7.1', 'zarr': '2.18.7'} if kind == 'eval'
        else {'numpy': '1.26.4', 'xarray': '2026.1.0', 'jax': '0.6.2'})
for name, expected in pins.items():
    if version(name) != expected:
        raise RuntimeError(f'{name}: expected {expected}, found {version(name)}')
for name in ('gcsfs', 'netCDF4', 'numpy', 'xarray'):
    importlib.import_module(name)
if kind == 'eval':
    import weatherbenchX
    from weatherbenchX.metrics import probabilistic
    module, expected = weatherbenchX, '88af2fde7a78f13b77496a2abb8ca88905fa5e95'
else:
    import weathernext
    from weathernext.cyclones import ibtracs_processing_stages
    from weathernext.utils import data_utils
    module, expected = weathernext, '89c4b2a77a1c57b328b909c575550fd2e5aadc9c'
root = Path(module.__file__).resolve().parent
actual = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
if actual != expected:
    raise RuntimeError(f'{kind}: wrong upstream source revision {actual}')
if subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
    raise RuntimeError(f'{kind}: modified upstream source checkout')
print(f'{kind} data runtime verified (CPU only)')
