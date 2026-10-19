"""Resource checks for native Slurm and the image-owned offline verifier.

The container marker is installed in the verifier image, never supplied by a
model package. Native execution retains the original Slurm/ledger checks.
"""
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import device_guard as native_device_guard

MARKER = Path('/opt/pathfinder/container-verifier.json')


def container_execution():
    if not MARKER.is_file():
        return False
    if not Path('/.dockerenv').exists() and not Path('/run/.containerenv').exists():
        raise RuntimeError('Verifier container marker outside a container')
    if MARKER.stat().st_uid != 0 or MARKER.stat().st_mode & 0o022:
        raise RuntimeError('Verifier marker must be root-owned and not writable by others')
    return json.loads(MARKER.read_text()) == {'protocol': 'single-checkpoint-fast/v1'}


def valid_root(root):
    root = Path(root).resolve()
    if container_execution():
        return root == Path('/workspace/research/run')
    return str(root).startswith(os.environ.get('PATHFINDER_SCRATCH_ROOT', '/scratch').rstrip('/') + '/')


def device_guard(device):
    if not device.startswith('cuda') or not container_execution():
        return native_device_guard(device)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Verifier has no allocated CUDA device')


def allocation(root, seconds, gpus=None):
    device_guard('cuda')
    if container_execution():
        if not valid_root(root):
            raise ValueError('Container work must use its isolated scratch root')
        if gpus is not None:
            import torch
            if torch.cuda.device_count() != gpus:
                raise RuntimeError(f'Verifier requires {gpus} exposed GPUs')
        return None
    jobs = json.loads((Path(root) / 'budget.json').read_text())['jobs']
    matches = [j for j in jobs if j['job_id'] == os.environ['SLURM_JOB_ID']]
    if len(matches) != 1 or 'actual_seconds' in matches[0]:
        raise ValueError('Registered active original allocation required')
    job = matches[0]
    if seconds > job['reserved_seconds'] or (gpus is not None and job['gpus'] != gpus):
        raise ValueError('Stage exceeds its registered allocation')
    return job['job_id']
