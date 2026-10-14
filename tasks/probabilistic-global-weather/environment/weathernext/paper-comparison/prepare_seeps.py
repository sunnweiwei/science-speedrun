#!/usr/bin/env python3
"""Build one paper-period monthly SEEPS climatology from verified cached rain."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import xarray as xr


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def accumulate(raw,hours):
    """Input starts 18 hours before the first desired endpoint."""
    if hours not in (12,24) or len(raw)<4:raise ValueError('Invalid accumulation input')
    values=np.asarray(raw,dtype=np.float64)
    return sum(values[3-offset:len(values)-offset] for offset in range(hours//6))


def thresholds(samples,dry_mm):
    if not np.isfinite(samples).all():raise ValueError('Incomplete climatology samples')
    dry=samples<dry_mm/1000
    p1=dry.mean(axis=0)
    # Sort contiguous sample rows in C, rather than a Python loop per grid point.
    values=np.ascontiguousarray(np.where(dry,np.inf,samples).reshape(len(samples),-1).T)
    values.sort(axis=-1)
    counts=(~dry).sum(axis=0).ravel()
    position=np.maximum(counts-1,0)*(2/3)
    lower=np.floor(position).astype(int);upper=np.ceil(position).astype(int)
    lo=np.take_along_axis(values,lower[:,None],axis=1)[:,0]
    hi=np.take_along_axis(values,upper[:,None],axis=1)[:,0]
    with np.errstate(invalid='ignore'):
        wet=np.where(counts>0,lo+(hi-lo)*(position-lower),np.nan).reshape(p1.shape)
    if np.any(~np.isfinite(wet)&(p1!=1)):raise ValueError('Unexpected missing wet threshold')
    return p1,wet


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rain',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--month',required=True,type=int);ap.add_argument('--workers',type=int,default=8)
    args=ap.parse_args()
    if not 1<=args.month<=12:raise ValueError('Invalid month')
    here=Path(__file__).resolve().parent;protocol=here/'seeps-protocol.json';spec=json.loads(protocol.read_text())
    if spec['years']!=[1980,2015]:raise ValueError('Wrong paper climatology period')
    destination=args.output/f'month-{args.month:02d}.nc';receipt=destination.with_suffix('.json')
    identity=dict(protocol_sha256=digest(protocol),source_lock_sha256=digest(here/'seeps-climatology-source.lock.json'),
        runner_sha256=digest(Path(__file__)),month=args.month)
    if receipt.exists():
        record=json.loads(receipt.read_text())
        if record['status']!='passed' or record['identity']!=identity or record['sha256']!=digest(destination):
            raise ValueError('Changed completed SEEPS climatology')
        return
    started=time.monotonic();blocks=[];records={};metadata_hash=None
    for year in range(1980,2016):
        folder=args.rain/f'year-{year}.zarr';source_receipt=args.rain/f'year-{year}.json'
        r=json.loads(source_receipt.read_text());source_identity=r['identity']
        if r['status']!='passed' or source_identity['year']!=year:raise ValueError('Unverified rain year')
        if metadata_hash is not None and metadata_hash!=source_identity['source_metadata_sha256']:
            raise ValueError('Changed ERA5 source metadata between years')
        metadata_hash=source_identity['source_metadata_sha256']
        records[source_receipt.name]=digest(source_receipt)
        for name,value in r['files'].items():
            if not name.startswith('total_precipitation_6hr/') or Path(name).name.startswith('.'):
                if digest(folder/name)!=value:raise ValueError('Corrupt cache metadata/coordinates')
        start=np.datetime64(f'{year}-{args.month:02d}-01','ns')
        stop=(start.astype('datetime64[M]')+np.timedelta64(1,'M')).astype('datetime64[ns]')
        times=np.arange(start-np.timedelta64(18,'h'),stop,np.timedelta64(6,'h'))
        with xr.open_zarr(folder,consolidated=True,chunks=None) as ds:
            ds=ds.sortby('latitude')
            np.testing.assert_array_equal(ds.latitude,np.linspace(-90,90,721))
            np.testing.assert_array_equal(ds.longitude,np.arange(1440)/4)
            indices=np.searchsorted(ds.time.values,times)
            np.testing.assert_array_equal(ds.time.values[indices],times)
            for index in indices:
                name=f'total_precipitation_6hr/{index}.0.0'
                if digest(folder/name)!=r['files'][name]:raise ValueError('Corrupt rain chunk')
            rain=ds.total_precipitation_6hr
            if rain.attrs.get('units')!='m':raise ValueError('Wrong rain units')
            values=rain.isel(time=indices).values
            if not np.isfinite(values).all():raise ValueError('Incomplete rain')
            blocks.append(values)
        print(json.dumps(dict(month=args.month,loaded_year=year,elapsed_seconds=time.monotonic()-started)),flush=True)
    shape=(2,2,721,1440);dry_fraction=np.empty(shape);wet_threshold=np.empty(shape)
    def tile(start):
        stop=min(start+18,721);result=[]
        for h,hours in enumerate((12,24)):
            samples=np.concatenate([accumulate(b[:,start:stop],hours) for b in blocks],axis=0)
            for d,dry in enumerate((.1,.25)):
                p1,wet=thresholds(samples,dry);result.append((h,d,p1,wet))
        return start,stop,result
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(tile,start) for start in range(0,721,18)]):
            start,stop,result=future.result()
            for h,d,p1,wet in result:dry_fraction[h,d,start:stop]=p1;wet_threshold[h,d,start:stop]=wet
            print(json.dumps(dict(month=args.month,latitude_start=start,elapsed_seconds=time.monotonic()-started)),flush=True)
    ds=xr.Dataset(dict(dry_fraction=(('accumulation_hours','dry_threshold_mm','lat','lon'),dry_fraction),
        wet_threshold=(('accumulation_hours','dry_threshold_mm','lat','lon'),wet_threshold)),
        coords=dict(accumulation_hours=[12,24],dry_threshold_mm=[.1,.25],lat=np.linspace(-90,90,721),lon=np.arange(1440)/4))
    ds.wet_threshold.attrs['units']='m';ds.dry_fraction.attrs['units']='1'
    args.output.mkdir(parents=True,exist_ok=True)
    temp=destination.with_suffix('.part');ds.to_netcdf(temp);temp.replace(destination)
    record=dict(status='passed',identity=identity,sha256=digest(destination),rain_year_receipts_sha256=records,
        source_metadata_sha256=metadata_hash,samples_per_grid_point=sum(len(b)-3 for b in blocks),
        wall_seconds=time.monotonic()-started,all_dry_grid_points=int((dry_fraction==1).sum()))
    temp=receipt.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(receipt)


if __name__=='__main__':main()
