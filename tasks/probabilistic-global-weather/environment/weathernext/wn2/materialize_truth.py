#!/usr/bin/env python3
"""Materialize exact HRES-fc0 truth into disjoint case shards of one Zarr store."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

import numcodecs
import numpy as np
import xarray as xr
import zarr

from audit_data import SOURCES
from evaluate import sha256, cell_weights


def write_json(path, data):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def initialize(output, protocol_path):
    protocol=json.loads(protocol_path.read_text())
    if output.exists():raise ValueError('Refusing to overwrite an existing truth release')
    output.mkdir(parents=True)
    cases=np.asarray(protocol['evaluation_initializations'],dtype='datetime64[ns]')
    leads=np.asarray(protocol['lead_hours'],dtype='timedelta64[h]').astype('timedelta64[ns]')
    coords=dict(case=cases,lead=leads,level=protocol['pressure_levels_hpa'],lat=np.linspace(-90,90,protocol['grid'][0]),
                lon=np.arange(protocol['grid'][1])*360/protocol['grid'][1],valid_time=(('case','lead'),cases[:,None]+leads[None,:]))
    xr.Dataset(coords=coords,attrs={'truth_source':protocol['truth_source'],'protocol_sha256':sha256(protocol_path)}).to_zarr(output/'truth.zarr',mode='w',consolidated=False)
    group=zarr.open_group(str(output/'truth.zarr'),mode='a')
    for name,spec in protocol['variables'].items():
        dims=['case','lead']+(['level'] if spec['pressure_levels'] else [])+['lat','lon']
        shape=[len(cases),len(leads)]+([len(protocol['pressure_levels_hpa'])] if spec['pressure_levels'] else [])+protocol['grid']
        arr=group.create_dataset(name,shape=shape,chunks=[1]*(len(shape)-2)+protocol['grid'],dtype='float32',fill_value=np.nan,
                                 compressor=numcodecs.Blosc(cname='lz4',clevel=1,shuffle=numcodecs.Blosc.SHUFFLE))
        arr.attrs.update(_ARRAY_DIMENSIONS=dims,units=spec['units'])
    zarr.consolidate_metadata(str(output/'truth.zarr'))
    (output/'cases').mkdir()
    (output/'protocol.json').write_bytes(protocol_path.read_bytes())
    write_json(output/'status.json',dict(status='initialized_not_materialized',protocol_sha256=sha256(protocol_path)))


def materialize(output, case_index):
    protocol=json.loads((output/'protocol.json').read_text())
    stamp=protocol['evaluation_initializations'][case_index]
    record_path=output/'cases'/f'{case_index:03d}.json'
    if record_path.exists():
        previous=json.loads(record_path.read_text())
        if previous['protocol_sha256']!=sha256(output/'protocol.json'):raise ValueError('Changed protocol')
        for path,digest in previous['chunks'].items():
            if sha256(output/'truth.zarr'/path)!=digest:raise ValueError(f'Corrupt existing chunk: {path}')
        return
    source_keys={'HRES-fc0':'hres_fc0','ERA5-precipitation-6h':'era5'}
    url=SOURCES[source_keys[protocol['truth_source']]]
    metadata=urllib.request.urlopen(url.replace('gs://','https://storage.googleapis.com/')+'/.zmetadata',timeout=60).read()
    begin=time.monotonic()
    coverage_records=[]
    with xr.open_zarr(url,consolidated=True,chunks=None,storage_options={'token':'anon'}) as source:
        source=source.rename(latitude='lat',longitude='lon').sortby('lat')
        np.testing.assert_array_equal(source.lat.values,np.linspace(-90,90,protocol['grid'][0]))
        np.testing.assert_array_equal(source.lon.values,np.arange(protocol['grid'][1])*360/protocol['grid'][1])
        source=source.sel(level=protocol['pressure_levels_hpa'])
        group=zarr.open_group(str(output/'truth.zarr'),mode='r+')
        for lead_index,lead in enumerate(protocol['lead_hours']):
            valid=np.datetime64(stamp,'ns')+np.timedelta64(lead,'h')
            for name,spec in protocol['variables'].items():
                if source[name].attrs.get('units','').replace('**','')!=spec['units']:
                    raise ValueError(f'Unexpected source units for {name}')
                dims=(['level'] if spec['pressure_levels'] else [])+['lat','lon']
                values=np.asarray(source[name].sel(time=valid).transpose(*dims).values,dtype=np.float32)
                weights=cell_weights(source.lat).values[:,None]
                coverage=(np.isfinite(values)*weights).sum(axis=(-2,-1))/(weights.sum()*len(source.lon))
                if np.isinf(values).any() or np.min(coverage)+1e-12<protocol.get('minimum_valid_area_fraction',1.):
                    raise ValueError(f'Insufficient valid HRES truth {valid}/{name}')
                coverage_records.append(dict(lead_hours=lead,variable=name,minimum_valid_area_fraction=float(np.min(coverage)),missing_cells=int(np.isnan(values).sum())))
                group[name][case_index,lead_index]=values
            print(json.dumps(dict(case_index=case_index,lead_hours=lead,elapsed_seconds=time.monotonic()-begin)),flush=True)
    chunks={}
    for name in protocol['variables']:
        for path in (output/'truth.zarr'/name).glob(f'{case_index}.*'):
            chunks[str(path.relative_to(output/'truth.zarr'))]=sha256(path)
    expected=len(protocol['lead_hours'])*sum(len(protocol['pressure_levels_hpa']) if s['pressure_levels'] else 1 for s in protocol['variables'].values())
    if len(chunks)!=expected:raise ValueError(f'Chunk coverage mismatch: {len(chunks)} vs {expected}')
    write_json(record_path,dict(status='passed',case_index=case_index,initialization=stamp,
                  runner_sha256=sha256(Path(__file__)),
                  protocol_sha256=sha256(output/'protocol.json'),source=url,metadata_sha256=hashlib.sha256(metadata).hexdigest(),
                  seconds=time.monotonic()-begin,coverage=coverage_records,chunks=chunks))


def finalize(output):
    protocol=json.loads((output/'protocol.json').read_text())
    metadata_hashes=set()
    manifests=[]
    for i in range(len(protocol['evaluation_initializations'])):
        path=output/'cases'/f'{i:03d}.json'
        record=json.loads(path.read_text())
        if record['status']!='passed' or record['protocol_sha256']!=sha256(output/'protocol.json'):
            raise ValueError('Incomplete/mismatched case manifest')
        for chunk,digest in record['chunks'].items():
            if sha256(output/'truth.zarr'/chunk)!=digest:raise ValueError(f'Corrupt truth: {chunk}')
        manifests.append(dict(file=str(path.relative_to(output)),sha256=sha256(path)))
        metadata_hashes.add(record['metadata_sha256'])
    if len(metadata_hashes)!=1:raise ValueError('Source metadata changed during materialization')
    write_json(output/'status.json',dict(status='passed',scope='Weather truth materialized; no model inputs, trained model or evaluation scores',truth_source=protocol['truth_source'],
               protocol_sha256=sha256(output/'protocol.json'),case_manifests=manifests))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--protocol',type=Path)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--initialize',action='store_true')
    action.add_argument('--case-index',type=int)
    action.add_argument('--finalize',action='store_true')
    args=parser.parse_args()
    if args.initialize:
        if args.protocol is None:parser.error('--initialize requires --protocol')
        initialize(args.output,args.protocol)
    elif args.finalize:finalize(args.output)
    else:
        count=len(json.loads((args.output/'protocol.json').read_text())['evaluation_initializations'])
        if not 0<=args.case_index<count:parser.error('case index out of range')
        materialize(args.output,args.case_index)


if __name__=='__main__':main()
