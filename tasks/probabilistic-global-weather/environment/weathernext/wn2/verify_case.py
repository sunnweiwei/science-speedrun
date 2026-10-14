#!/usr/bin/env python3
"""Run one fixed paper-subset initialization through the isolated submission entry."""
import argparse
import json
from pathlib import Path
import shutil
import time
import uuid
import numpy as np
import xarray as xr
from namespace_inference import run
from training_data import sha256
from zarr_safety import audit_zarr


def replay(package, inputs, protocol_path, index, work, predictions, timeout, lead_hours=None, state=None):
    begin=time.monotonic()
    protocol=json.loads(protocol_path.read_text())
    if not 0<=index<len(protocol['evaluation_initializations']):raise ValueError('Invalid case')
    work.mkdir(parents=True,exist_ok=False)
    package=package.resolve()
    if any(p.is_symlink() for p in package.rglob('*')):raise ValueError('Submission symlinks forbidden')
    files=[p for p in package.rglob('*') if p.is_file()]
    if len(files)>10000 or sum(p.stat().st_size for p in files)>8*1024**3:raise ValueError('Submission exceeds limits')
    model=work/'relocated-model';shutil.copytree(package,model)
    public=work/'public';public.mkdir()
    source=inputs/f'inputs-{index:03d}.nc'
    record=json.loads(source.with_suffix('.json').read_text())
    if record['sha256']!=sha256(source) or record['protocol_sha256']!=sha256(protocol_path):
        raise ValueError('Input provenance mismatch')
    leads=protocol['lead_hours'] if lead_hours is None else lead_hours
    if not leads or leads!=sorted(set(leads)) or any(h not in protocol['lead_hours'] for h in leads):
        raise ValueError('Invalid forecast window')
    if lead_hours is not None:
        if leads!=list(range(leads[0],leads[-1]+1,12)) or (leads[0]>12)!=(state is not None):
            raise ValueError('Nonconsecutive window or missing continuation')
        if 'wn2-inference/v3' not in json.loads((model/'package.json').read_text()).get('inference_schemas',[]):
            raise ValueError('Disk-bounded evaluation requires v3 continuation; repackage existing weights without training')
    date=protocol['evaluation_initializations'][index]
    if np.datetime64(record['initialization'])!=np.datetime64(date):raise ValueError('Wrong input date')
    with xr.open_dataset(source) as ds:
        ds.sel(time=np.asarray([-6,0],dtype='timedelta64[h]')).load().to_netcdf(public/'context.nc')
    request_id='case_'+uuid.uuid4().hex
    seed=20220910
    request=dict(schema='wn2-inference/v2' if lead_hours is None else 'wn2-inference/v3',cases=[dict(id=request_id,file='context.nc')],
                 lead_hours=leads,members=protocol['members'],seed=seed)
    if state is not None:
        if state.is_symlink() or any(p.is_symlink() for p in state.rglob('*')):
            raise ValueError('Continuation symlinks forbidden')
        shutil.move(str(state),str(public/'resume'))
        request['resume']=dict(directory='resume',lead_hours=leads[0]-12)
    (public/'request.json').write_text(json.dumps(request,indent=2)+'\n')
    output=run(model,public,work,timeout)
    expected={f'{request_id}-member-{member:03d}.zarr' for member in range(protocol['members'])}
    if {p.name for p in output.glob('*.zarr')}!=expected:raise ValueError('Missing or unexpected member stores')
    predictions.mkdir(parents=True,exist_ok=True)
    package_digest=sha256(model/'package.json')
    for member in range(protocol['members']):
        forecast=output/f'{request_id}-member-{member:03d}.zarr'
        audit_zarr(forecast)
        with xr.open_zarr(forecast,consolidated=True,chunks=None,decode_timedelta=True) as ds:
            if lead_hours is not None and (set(ds.data_vars)!=set(protocol['variables'])|{'total_precipitation_12hr'} or any(ds[n].dtype!=np.dtype('float32') for n in ds.data_vars)):
                raise ValueError('v3 windows require exactly the declared float32 forecast fields')
            np.testing.assert_array_equal(ds.member,[member])
            np.testing.assert_array_equal(ds.case,[np.datetime64(date,'ns')])
            np.testing.assert_array_equal(ds.lead,np.array(leads,dtype='timedelta64[h]'))
        name=f'case-{index:03d}-member-{member:03d}'
        destination=predictions/f'{name}.zarr'
        if destination.exists():raise ValueError('Refusing to overwrite forecasts')
        shutil.move(str(forecast),str(destination))
        receipt=dict(status='passed',protocol_sha256=sha256(protocol_path),
            context_sha256=sha256(source),package_sha256=package_digest,
            runner_sha256=sha256(Path(__file__)),seed=seed,member=member,lead_hours=leads,
            initialization=str(np.datetime64(date,'ns')),
            files={str(p.relative_to(destination)):sha256(p) for p in sorted(destination.rglob('*')) if p.is_file()})
        (predictions/f'{name}.complete.json').write_text(json.dumps(receipt)+'\n')
    result=dict(status='passed',case_index=index,protocol_sha256=sha256(protocol_path),
        seconds=time.monotonic()-begin,members=protocol['members'],leads=leads,
        relocated_files={str(p.relative_to(model)):sha256(p) for p in model.rglob('*') if p.is_file()},
        boundary='Offline PID/user/mount/network namespace; only runtime, submission, context, GPU and output')
    (work/'replay.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('package','inputs','protocol','output','predictions'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--case-index',type=int,required=True)
    ap.add_argument('--inference-timeout',type=int,default=21600)
    ap.add_argument('--lead-hours',type=int,nargs='+');ap.add_argument('--state',type=Path)
    a=ap.parse_args();replay(a.package,a.inputs,a.protocol,a.case_index,a.output,a.predictions,a.inference_timeout,a.lead_hours,a.state)

if __name__=='__main__':main()
