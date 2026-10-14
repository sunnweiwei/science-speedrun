"""Lossless continuation for the starter; no forecast history or scoring data."""
import json
from pathlib import Path
import numpy as np
import xarray as xr
from training_data import sha256


def resume_lead(request):
    if request['schema'] not in ('wn2-inference/v2', 'wn2-inference/v3'):
        raise ValueError('Unknown request schema')
    if request['schema'] == 'wn2-inference/v2':
        if 'resume' in request:
            raise ValueError('Continuation requires v3')
        return 0
    leads = request['lead_hours']
    start = request.get('resume', {}).get('lead_hours', 0)
    if type(start) is not int or start < 0 or start % 12 or leads != list(range(start+12, leads[-1]+1, 12)):
        raise ValueError('Continuation must request consecutive new 12-hour endpoints')
    if len(request['cases']) != 1:
        raise ValueError('v3 continuation supports one case per invocation')
    return start


def state_identity(init, seed, member, lead, package_digest):
    return dict(initialization=str(np.datetime64(init, 'ns')), seed=seed,
                member=member, lead_hours=lead, package_sha256=package_digest)


def save_state(directory, current, identity):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    # Zarr's NumPy-style astype(order=...) is not supported by JAX duck arrays.
    # Transfer every variable explicitly, preserving dtype, values and ordering.
    order=list(current.data_vars)
    current=xr.Dataset(
        {name:(var.dims,np.asarray(var.values),dict(var.attrs)) for name,var in current.data_vars.items()},
        coords={name:(var.dims,np.asarray(var.values),dict(var.attrs)) for name,var in current.coords.items()},
        attrs=dict(current.attrs))
    current.to_zarr(directory/'context.zarr', mode='w-', consolidated=True)
    files = {str(p.relative_to(directory)): sha256(p) for p in sorted(directory.rglob('*')) if p.is_file()}
    (directory/'state.json').write_text(json.dumps(dict(status='passed', identity=identity, files=files, variable_order=order))+'\n')


def load_state(directory, identity):
    directory = Path(directory)
    receipt = json.loads((directory/'state.json').read_text())
    if receipt.get('status') != 'passed' or receipt['identity'] != identity:
        raise ValueError('Continuation state identity mismatch')
    for name, digest in receipt['files'].items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha256(path) != digest:
            raise ValueError('Corrupt continuation state')
    with xr.open_zarr(directory/'context.zarr', chunks=None, consolidated=True, decode_timedelta=True) as ds:
        return ds[receipt.get('variable_order',list(ds.data_vars))].load()
