#!/usr/bin/env python3
"""Prepare paper-period empirical extreme thresholds, grouped by month and hour.

Uses 1993--2016 ERA5, independently of the 2022 evaluation observations.
Quantile interpolation is NumPy's explicit linear convention; no Gaussian fit.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'wn2'))
import numpy as np
import xarray as xr
from audit_data import SOURCES
from training_data import sha256

QUANTILES=[0.0001,0.001,0.01,0.99,0.999,0.9999]
FIELDS=['2m_temperature','10m_wind_speed','mean_sea_level_pressure']


def prepare(job):
    output,month,hour=job
    stem=f'month-{month:02d}-hour-{hour:02d}'
    path=output/f'{stem}.nc';record=output/f'{stem}.json'
    spec=dict(years=[1993,2016],month=month,hour=hour,quantiles=QUANTILES,
              fields=FIELDS,method='linear',source=SOURCES['era5'])
    if record.exists():
        old=json.loads(record.read_text())
        if old.get('spec')!=spec or old.get('sha256')!=sha256(path): raise ValueError('Changed/corrupt REV thresholds')
        return
    with xr.open_zarr(SOURCES['era5'],consolidated=True,chunks=None,storage_options={'token':'anon'}) as raw:
        raw=raw.rename(latitude='lat',longitude='lon').sortby('lat').sel(time=slice('1993-01-01','2016-12-31T23:59:59'))
        timestamps=raw.time.values
        selected=(raw.time.dt.month.values==month)&(raw.time.dt.hour.values==hour)
        times=timestamps[selected]
        if len(times)<24*28: raise ValueError('Incomplete climatology time coverage')
        results={}
        for field in FIELDS:
            values=raw[field].sel(time=times).load().transpose('time','lat','lon')
            array=values.values
            if not np.isfinite(array).all(): raise ValueError(f'Nonfinite climatology: {field}')
            # Bound quantile scratch memory; all temporal samples stay in each block.
            threshold=np.empty((len(QUANTILES),len(values.lat),len(values.lon)),dtype=np.float32)
            for start in range(0,len(values.lat),16):
                threshold[:,start:start+16]=np.quantile(array[:,start:start+16],QUANTILES,axis=0,method='linear')
            results[field]=(('quantile','lat','lon'),threshold,dict(units=values.attrs.get('units','')))
            print(json.dumps(dict(month=month,hour=hour,field=field,samples=len(times),status='thresholds_ready')),flush=True)
        dataset=xr.Dataset(results,coords=dict(quantile=QUANTILES,lat=raw.lat,lon=raw.lon))
        dataset.attrs.update(climatology_start='1993-01-01',climatology_end='2016-12-31',
            month=month,hour=hour,quantile_method='linear',samples=len(times),source=SOURCES['era5'])
        temporary=path.with_suffix('.part')
        dataset.to_netcdf(temporary,encoding={n:dict(zlib=True,complevel=1) for n in FIELDS})
        temporary.replace(path)
        receipt=record.with_suffix('.tmp')
        receipt.write_text(json.dumps(dict(status='passed',spec=spec,sha256=sha256(path),samples=len(times),
            timestamps_sha256=hashlib.sha256(times.astype('datetime64[ns]').astype('<i8').tobytes()).hexdigest()))+'\n')
        receipt.replace(record)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    metadata_url=SOURCES['era5'].replace('gs://','https://storage.googleapis.com/')+'/.zmetadata'
    with urllib.request.urlopen(metadata_url,timeout=60) as response:
        digest=hashlib.sha256(response.read()).hexdigest()
    manifest=args.output/'source.json'
    spec=dict(source=SOURCES['era5'],metadata_sha256=digest,years=[1993,2016],fields=FIELDS,quantiles=QUANTILES,method='linear')
    if manifest.exists() and json.loads(manifest.read_text())!=spec: raise ValueError('Climatology source changed')
    manifest.write_text(json.dumps(spec,indent=2)+'\n')
    jobs=[(args.output,m,h) for m in range(1,13) for h in (0,6,12,18)]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        list(pool.map(prepare,jobs))
    records={p.name:sha256(p) for p in sorted(args.output.glob('month-*.json'))}
    if len(records)!=48: raise ValueError('Missing month/hour groups')
    (args.output/'ready.json').write_text(json.dumps(dict(status='passed',source_sha256=sha256(manifest),records=records))+'\n')


if __name__=='__main__':main()
