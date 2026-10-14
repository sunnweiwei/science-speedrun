#!/usr/bin/env python3
"""Prepare 10-day HRES history and 15-day verification trajectories for TE 2.1."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'wn2'))
import numpy as np
import xarray as xr
from audit_data import SOURCES
from training_data import sha256

FIELDS=['mean_sea_level_pressure','10m_u_component_of_wind','10m_v_component_of_wind','geopotential']


def fields(ds):
    result=xr.Dataset(dict(msl=ds.mean_sea_level_pressure,
        thickness=ds.geopotential.sel(level=300,drop=True)-ds.geopotential.sel(level=500,drop=True),
        wind=np.hypot(ds['10m_u_component_of_wind'],ds['10m_v_component_of_wind'])))
    for name,units in [('msl','Pa'),('thickness','m2 s-2'),('wind','m s-1')]:result[name].attrs['units']=units
    return result


def prepare(job):
    data,output,protocol,index=job
    p=json.loads(protocol.read_text());init=np.datetime64(p['evaluation_initializations'][index],'ns')
    path=output/f'case-{index:03d}.nc';receipt=path.with_suffix('.json')
    if receipt.exists():
        record=json.loads(receipt.read_text())
        if record['protocol_sha256']!=sha256(protocol) or record['sha256']!=sha256(path):raise ValueError('Changed tracker truth')
        return
    times=init+np.arange(-240,1,12).astype('timedelta64[h]')
    with xr.open_zarr(SOURCES['hres_fc0'],consolidated=True,chunks=None,storage_options={'token':'anon'}) as raw:
        history=raw[FIELDS].rename(latitude='lat',longitude='lon').sortby('lat').sel(time=times,level=[300,500]).load()
    with xr.open_zarr(data/'weather/truth.zarr',consolidated=True,chunks=None,decode_timedelta=True) as raw:
        future=raw[FIELDS].isel(case=index,drop=True).sel(level=[300,500]).load()
    future=future.drop_vars('valid_time').rename(lead='time')
    future=future.assign_coords(time=init+future.time.values)
    values=xr.concat([fields(history),fields(future)],dim='time',join='exact')
    with xr.open_dataset(data/'inputs'/f'inputs-{index:03d}.nc') as context:
        values['zs']=(context.geopotential_at_surface.load()/9.80665).astype('float32')
    values.zs.attrs['units']='m'
    for field in values:
        if not np.isfinite(values[field]).all():raise ValueError(f'Nonfinite tracker truth: {field}')
    expected=init+np.arange(-240,361,12).astype('timedelta64[h]')
    np.testing.assert_array_equal(values.time,expected)
    values=values.astype('float32')
    values.lat.attrs['units']='degrees_north';values.lon.attrs['units']='degrees_east'
    temporary=path.with_suffix('.part')
    values.to_netcdf(temporary,format='NETCDF3_64BIT')
    temporary.replace(path)
    record=dict(status='passed',protocol_sha256=sha256(protocol),sha256=sha256(path),case=index,
        initialization=str(init),history_hours=240,forecast_hours=360,cadence_hours=12,
        truth_source=SOURCES['hres_fc0'],static_source='same ERA5 orography as shared model contexts',
        weather_case_receipt_sha256=sha256(data/'weather/cases'/f'{index:03d}.json'))
    temp=receipt.with_suffix('.tmp');temp.write_text(json.dumps(record)+'\n');temp.replace(receipt)
    print(json.dumps(dict(case=index,status='prepared',file=str(path))),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    protocol=Path(__file__).parent/'weather-protocol.json'
    if json.loads((args.data/'ready.json').read_text())['weather_protocol_sha256']!=sha256(protocol):raise ValueError('Data protocol mismatch')
    jobs=[(args.data,args.output,protocol,i) for i in range(len(json.loads(protocol.read_text())['evaluation_initializations']))]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:list(pool.map(prepare,jobs))
    (args.output/'ready.json').write_text(json.dumps(dict(status='passed',cases=len(jobs),
        records={f.name:sha256(f) for f in sorted(args.output.glob('case-*.json'))}))+'\n')


if __name__=='__main__':main()
