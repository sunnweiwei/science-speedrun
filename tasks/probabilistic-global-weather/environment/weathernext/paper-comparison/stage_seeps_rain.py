#!/usr/bin/env python3
"""Cache one complete ERA5 precipitation year plus 18-hour accumulation context.

Copy original compressed global chunks once, avoiding a repeated remote read
for every spatial climatology tile. Completion inventories bind all local files.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

import fsspec
import numcodecs
import numpy as np
import xarray as xr
import zarr
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'wn2'))
from audit_data import SOURCES


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',required=True,type=Path);ap.add_argument('--year',required=True,type=int)
    ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    if not 1980<=args.year<=2015:raise ValueError('Year outside paper SEEPS climatology')
    source=SOURCES['era5'];variable='total_precipitation_6hr'
    with fsspec.open(source+'/.zmetadata',token='anon') as stream:metadata_bytes=stream.read()
    metadata=json.loads(metadata_bytes)['metadata'];layout=metadata[variable+'/.zarray']
    if layout['chunks']!=[1,721,1440] or layout['dtype']!='<f4':raise ValueError('Changed upstream chunk layout')
    with xr.open_zarr(source,consolidated=True,chunks=None,storage_options={'token':'anon'}) as ds:
        times=ds.time.values;start=np.datetime64(f'{args.year}-01-01','ns')-np.timedelta64(18,'h')
        stop=np.datetime64(f'{args.year+1}-01-01','ns')
        indices=np.flatnonzero((times>=start)&(times<stop))
        selected=times[indices]
        np.testing.assert_array_equal(selected,np.arange(start,stop,np.timedelta64(6,'h')))
        coords=xr.Dataset(coords=dict(time=selected,latitude=ds.latitude.values,longitude=ds.longitude.values))
    directory=args.output/f'year-{args.year}.zarr';receipt=args.output/f'year-{args.year}.json'
    identity=dict(source=source,source_metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),year=args.year,
        start=str(start),stop_exclusive=str(stop),source_indices=[int(indices[0]),int(indices[-1])],
        runner_sha256=digest(Path(__file__)))
    if receipt.exists():
        previous=json.loads(receipt.read_text())
        if previous['status']!='passed' or previous['identity']!=identity:raise ValueError('Changed completed rain cache')
        for name,expected in previous['files'].items():
            if digest(directory/name)!=expected:raise ValueError('Corrupt completed rain cache')
        return
    args.output.mkdir(parents=True,exist_ok=True)
    partial=args.output/f'year-{args.year}.partial.json'
    if partial.exists():
        previous=json.loads(partial.read_text())
        if previous['identity']!=identity:raise ValueError('Changed partial rain cache')
        inventory=previous['chunks']
    else:
        if directory.exists():raise ValueError('Unverified cache directory exists')
        coords.to_zarr(directory,mode='w',consolidated=False)
        group=zarr.open_group(str(directory),mode='a')
        array=group.create_dataset(variable,shape=(len(indices),721,1440),chunks=tuple(layout['chunks']),
            dtype=layout['dtype'],compressor=numcodecs.get_codec(layout['compressor']),fill_value=np.nan)
        array.attrs.update(metadata[variable+'/.zattrs'])
        inventory={}
    def save_progress():
        temp=partial.with_suffix('.tmp');temp.write_text(json.dumps(dict(identity=identity,chunks=inventory))+'\n');temp.replace(partial)
    save_progress()
    url=source.replace('gs://','https://storage.googleapis.com/',1)
    def fetch(pair):
        local,remote=pair;name=f'{variable}/{local}.0.0';path=directory/name
        if name in inventory:
            if digest(path)!=inventory[name]:raise ValueError('Corrupt partial cache chunk')
            return name,inventory[name]
        for attempt in range(3):
            try:
                with urllib.request.urlopen(f'{url}/{variable}/{remote}.0.0',timeout=120) as stream:content=stream.read()
                break
            except (OSError,TimeoutError):
                if attempt==2:raise
                time.sleep(2**attempt)
        decoded=numcodecs.get_codec(layout['compressor']).decode(content)
        values=np.frombuffer(decoded,dtype='<f4')
        if values.size!=721*1440 or not np.isfinite(values).all():raise ValueError(f'Invalid upstream rain at {selected[local]}')
        temp=path.with_suffix('.part');temp.write_bytes(content);temp.replace(path)
        return name,hashlib.sha256(content).hexdigest()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(fetch,(i,int(index))) for i,index in enumerate(indices)]
        for count,future in enumerate(as_completed(futures),1):
            name,value=future.result();inventory[name]=value
            if count==1 or count%64==0:
                save_progress();print(json.dumps(dict(year=args.year,chunks=count,total=len(indices))),flush=True)
    zarr.consolidate_metadata(str(directory))
    files={str(path.relative_to(directory)):digest(path) for path in directory.rglob('*') if path.is_file()}
    temp=receipt.with_suffix('.tmp');temp.write_text(json.dumps(dict(status='passed',identity=identity,files=files))+'\n');temp.replace(receipt)
    partial.unlink()
    print(json.dumps(dict(year=args.year,status='passed',chunks=len(indices))),flush=True)


if __name__=='__main__':main()
