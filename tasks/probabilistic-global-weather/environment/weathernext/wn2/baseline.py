#!/usr/bin/env python3
"""Portable persistence/dry-precipitation and seasonal-mean references."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import xarray as xr
from scipy.special import ndtri

from forecast_store import initialize
from training_data import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--package',type=Path,required=True);p.add_argument('--mode',choices=['persistence','climatology','seasonal-gaussian'],required=True)
    args=p.parse_args();request=json.loads((args.input/'request.json').read_text());begin=time.monotonic()
    variables=json.loads((args.package/'variables.json').read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    for case in request['cases']:
        with xr.open_dataset(args.input/case['file']) as ds:
            context=ds.isel(batch=0,time=-1,drop=True).load()
            init=ds.datetime.values[0,-1]
        store=initialize(args.output/f"{case['id']}.zarr",init,request['lead_hours'],request['members'],
                         context.level.values,variables,context.lat.values,context.lon.values)
        for lead_index,lead in enumerate(request['lead_hours']):
            if args.mode in ('climatology','seasonal-gaussian'):
                stamp=str(init+np.timedelta64(lead,'h'));quarter=(int(stamp[5:7])-1)//3;hour=int(stamp[11:13])
                path=args.package/'climatology'/f'group-{quarter*4+hour//6:02d}.nc'
                record=json.loads(path.with_suffix('.json').read_text())
                if sha256(path)!=record['sha256']:raise ValueError('Changed climatology')
                values=xr.load_dataset(path)
                if args.mode=='seasonal-gaussian':
                    std_path=path.parent/record['std_file']
                    if sha256(std_path)!=record['std_sha256']:raise ValueError('Changed seasonal variance')
                    deviations=xr.load_dataset(std_path)
            else:
                values=context.copy(deep=False)
                values['total_precipitation_6hr']=xr.zeros_like(context['2m_temperature'])
            for name,spec in variables.items():
                dims=(['level'] if spec['pressure_levels'] else [])+['lat','lon']
                field=values[name].transpose(*dims).values
                if not np.isfinite(field).all():raise ValueError('Nonfinite baseline output')
                for member in range(request['members']):
                    forecast=field
                    if args.mode=='seasonal-gaussian':
                        quantile=ndtri((member+.5)/request['members'])
                        forecast=field+quantile*deviations[name].transpose(*dims).values
                        if name in ('total_precipitation_6hr','specific_humidity'):forecast=np.maximum(forecast,0.)
                    store[name][0,lead_index,member]=forecast
            print(json.dumps(dict(case_id=case['id'],lead_hours=lead,seconds=time.monotonic()-begin)),flush=True)
        (args.output/f"{case['id']}.complete.json").write_text(json.dumps(dict(status='passed',baseline=args.mode))+'\n')


if __name__=='__main__':main()
