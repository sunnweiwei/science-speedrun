#!/usr/bin/env python3
"""Prepare two real native WN2 context frames per fixed evaluation case."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import xarray as xr

from audit_data import SOURCES
from source_metadata import source_record
from evaluate import sha256, cell_weights
from materialize_truth import write_json


def fill_sparse_context(values, latitude, minimum_valid_area_fraction=.9999):
    """Interpolate isolated input gaps using only same-time/level neighbors."""
    values=np.asarray(values)
    if np.isinf(values).any():raise ValueError('Infinite context')
    valid=np.isfinite(values)
    weights=cell_weights(latitude).values[:,None]
    coverage=(valid*weights).sum(axis=(-2,-1))/(weights.sum()*values.shape[-1])
    if np.min(coverage)+1e-12<minimum_valid_area_fraction:raise ValueError('Insufficient context coverage')
    out=values.copy()
    left,right=np.roll(values,1,-1),np.roll(values,-1,-1)
    repair=~valid & np.isfinite(left) & np.isfinite(right)
    out[repair]=(left[repair]*.5+right[repair]*.5)
    north,south=np.roll(values,1,-2),np.roll(values,-1,-2)
    repair=~np.isfinite(out) & np.isfinite(north) & np.isfinite(south)
    repair[...,0,:]=False;repair[...,-1,:]=False
    out[repair]=(north[repair]*.5+south[repair]*.5)
    if not np.isfinite(out).all():raise ValueError('Only isolated context gaps can be interpolated')
    return out,np.argwhere(~valid).tolist()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--case-index',required=True,type=int);args=p.parse_args()
    protocol=json.loads(args.protocol.read_text())
    if not 0<=args.case_index<len(protocol['evaluation_initializations']):p.error('Case index out of bounds')
    init=np.datetime64(protocol['evaluation_initializations'][args.case_index],'ns')
    times=init+np.asarray([-6,0],dtype='timedelta64[h]')
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/f'inputs-{args.case_index:03d}.nc';record_path=path.with_suffix('.json')
    if record_path.exists():
        old=json.loads(record_path.read_text())
        if old['protocol_sha256']!=sha256(args.protocol) or old['sha256']!=sha256(path):raise ValueError('Changed/corrupt existing inputs')
        return
    provenance={};pieces=[];repairs={}
    selections={
        'hres_fc0':list(protocol['variables']),
        'era5':['sea_surface_temperature','geopotential_at_surface','land_sea_mask'],
        'arco_era5':['100m_u_component_of_wind','100m_v_component_of_wind'],
    }
    for source_name,names in selections.items():
        url=SOURCES[source_name]
        meta=urllib.request.urlopen(url.replace('gs://','https://storage.googleapis.com/')+'/.zmetadata',timeout=60).read()
        provenance[source_name]=source_record(url, meta) | dict(variables=names)
        with xr.open_zarr(url,consolidated=True,chunks=None,storage_options={'token':'anon'}) as raw:
            raw=raw.rename(latitude='lat',longitude='lon').sortby('lat')
            np.testing.assert_array_equal(raw.lat.values,np.linspace(-90,90,protocol['grid'][0]))
            np.testing.assert_array_equal(raw.lon.values,np.arange(protocol['grid'][1])*360/protocol['grid'][1])
            selected=raw[names].sel(time=times)
            if 'level' in selected.dims:selected=selected.sel(level=protocol['pressure_levels_hpa'])
            # Load only these two timestamps, never the archive-wide dataset.
            selected=selected.load().astype('float32')
            for name in names:
                if name=='sea_surface_temperature':
                    if np.isinf(selected[name]).any():raise ValueError('Infinite SST')
                elif not np.isfinite(selected[name]).all():
                    leading=[d for d in selected[name].dims if d not in ('lat','lon')]
                    array=selected[name].transpose(*leading,'lat','lon')
                    filled,indices=fill_sparse_context(array.values,selected.lat)
                    selected[name]=array.copy(data=filled)
                    repairs[name]=dict(dimensions=list(array.dims),indices=indices)
            pieces.append(selected)
    raw=xr.merge(pieces,join='exact',compat='equals')
    static=raw[['geopotential_at_surface','land_sea_mask']]
    temporal=raw.drop_vars(list(static)).expand_dims(batch=[0])
    ds=xr.merge([temporal,static],join='exact').assign_coords(
        datetime=(('batch','time'),times[None,:]),time=times-init)
    ds.attrs.update(initialization=str(init),profile='HRES core with retrospective ERA5 SST/100m winds/static fields',
                    target_values_included='false')
    temp=path.with_suffix('.nc.part')
    ds.to_netcdf(temp,encoding={n:dict(zlib=True,complevel=1) for n in ds.data_vars});temp.replace(path)
    write_json(record_path,dict(status='passed',case_index=args.case_index,initialization=str(init),
                 context_times=[str(t) for t in times],file=path.name,sha256=sha256(path),
                 protocol_sha256=sha256(args.protocol),runner_sha256=sha256(Path(__file__)),sources=provenance,
                 sst_missing_values=int(np.isnan(ds.sea_surface_temperature.values).sum()),
                 isolated_input_repairs=repairs,
                 repair_rule='Same-time/level longitude-neighbor mean, latitude fallback without polar wrap; >=99.99% valid area required; truth is never imputed',
                 scope='Native context fields; calendar forcings are derived by pinned WeatherNext at inference. No future weather targets.'))


if __name__=='__main__':main()
