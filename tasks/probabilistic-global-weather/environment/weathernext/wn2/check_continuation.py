#!/usr/bin/env python3
"""Compare an existing trained v2 package with v3 continuation, without training."""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
import xarray as xr
from namespace_inference import run
from training_data import sha256
from disk_budget import tree_bytes


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('original','upgraded','context','output'):ap.add_argument('--'+name,required=True,type=Path)
    ap.add_argument('--members',type=int,default=2)
    ap.add_argument('--leads',type=int,nargs='+',default=[12,24])
    ap.add_argument('--window-leads',type=int,default=1)
    ap.add_argument('--timeout-seconds',type=int,default=5400)
    a=ap.parse_args()
    if a.leads!=list(range(12,a.leads[-1]+1,12)) or a.window_leads<1:ap.error('Consecutive 12h leads and positive window required')
    if sha256(a.original/'weights.npz')!=sha256(a.upgraded/'weights.npz'):raise ValueError('Weights changed')
    a.output.mkdir(parents=True,exist_ok=False);begin=time.monotonic();windows=[];previous=None
    requests=[('original',a.original,a.leads,'wn2-inference/v2')]
    requests += [(f'window-{i:03d}',a.upgraded,a.leads[i:i+a.window_leads],'wn2-inference/v3') for i in range(0,len(a.leads),a.window_leads)]
    for label,package,leads,schema in requests:
        dest=a.output/label;public=dest/'public';public.mkdir(parents=True)
        with xr.open_dataset(a.context,decode_timedelta=True) as ds:
            ds.sel(time=np.array([-6,0],dtype='timedelta64[h]')).load().to_netcdf(public/'context.nc')
        request=dict(schema=schema,cases=[dict(id='renamed',file='context.nc')],members=a.members,lead_hours=leads,seed=20220910)
        if schema=='wn2-inference/v3' and previous is not None:
            shutil.move(str(previous),str(public/'resume'))
            request['resume']=dict(directory='resume',lead_hours=leads[0]-12)
        (public/'request.json').write_text(json.dumps(request)+'\n')
        output=run(package,public,dest,a.timeout_seconds)
        if schema=='wn2-inference/v3':
            previous=output/'state';windows.append((output,leads))
        print(json.dumps(dict(completed=label,seconds=time.monotonic()-begin)),flush=True)
    checks=[]
    for member in range(a.members):
        stem=f'renamed-member-{member:03d}.zarr'
        with xr.open_zarr(a.output/'original/untrusted-output'/stem,chunks=None,decode_timedelta=True) as original:
            for output,leads in windows:
                with xr.open_zarr(output/stem,chunks=None,decode_timedelta=True) as actual:
                    for name in original.data_vars:
                        expected=original[name].sel(lead=np.array(leads,dtype='timedelta64[h]')).values
                        observed=actual[name].values
                        if expected.dtype!=observed.dtype or expected.tobytes()!=observed.tobytes():
                            raise AssertionError((member,leads,name))
                        checks.append(dict(member=member,lead_hours=leads,variable=name,bitwise_equal=True))
    receipt=dict(status='passed',weights_sha256=sha256(a.original/'weights.npz'),training_rerun=False,
        members=a.members,lead_hours=a.leads,comparisons=checks,seconds=time.monotonic()-begin,
        validation_output_bytes=tree_bytes(a.output),scope='Original trained package versus initial and resumed v3 namespace runs; not full benchmark')
    (a.output/'result.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
