#!/usr/bin/env python3
"""Training-only, area-weighted normalization sufficient statistics."""
import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr
from weathernext.utils import data_utils

from training_data import load_window, sha256, verify_case

GROUPS = {'era5_1deg_12h': ('era5',1.,12), 'era5_1deg_6h': ('era5',1.,6),
          'era5_025deg_6h': ('era5',.25,6), 'hres_025deg_6h': ('hres_fc0',.25,6)}


def moments(field):
    leading = ['level'] if 'level' in field.dims else []
    field = field.transpose(*leading,*[d for d in field.dims if d not in leading])
    values=np.asarray(field.values,dtype=np.float64)
    weights=np.ones([1]*values.ndim)
    if 'lat' in field.dims:
        latitude=field.lat.values
        edges=np.r_[-90.,(latitude[:-1]+latitude[1:])/2,90.]
        area=np.diff(np.sin(np.deg2rad(edges)))
        shape=[1]*values.ndim;shape[field.get_axis_num('lat')]=len(area)
        weights=area.reshape(shape)
    valid=np.isfinite(values)
    weights=np.broadcast_to(weights,values.shape)*valid
    filled=np.where(valid,values,0.)
    axes=tuple(range(len(leading),values.ndim))
    return dict(weight=weights.sum(axis=axes).tolist(),sum=(filled*weights).sum(axis=axes).tolist(),
                squared_sum=(filled*filled*weights).sum(axis=axes).tolist(),
                minimum=np.where(valid,values,1e300).min(axis=axes).tolist(),dims=leading)


def collect(root, protocol_path, index, output):
    evidence=verify_case(root,protocol_path,index)
    protocol=json.loads(protocol_path.read_text())
    groups={}
    for name,(source,resolution,cadence) in GROUPS.items():
        ds=load_window(root,protocol,index,source,resolution,cadence,48//cadence)
        data_utils.add_derived_vars(ds)
        groups[name]={'values':{n:moments(v) for n,v in ds.data_vars.items()},
                      'differences':{n:moments(v.diff('time')) for n,v in ds.data_vars.items() if 'time' in v.dims}}
        del ds
        print(json.dumps(dict(case_index=index,group=name,status='passed')),flush=True)
    output.mkdir(parents=True,exist_ok=True)
    path=output/f'case-{index:03d}.json';tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(status='passed',protocol_sha256=sha256(protocol_path),
        runner_sha256=sha256(Path(__file__)),loader_sha256=sha256(Path(__file__).with_name('training_data.py')),
        evidence=evidence,groups=groups),indent=2,allow_nan=False)+'\n');tmp.replace(path)


def merge(parts):
    result={}
    for name in parts[0]:
        records=[p[name] for p in parts]
        if any(r['dims']!=records[0]['dims'] for r in records):raise ValueError('Statistic dimensions changed')
        weight=sum(np.asarray(r['weight']) for r in records)
        if np.any(weight<=0):raise ValueError(f'No training observations for {name}')
        mean=sum(np.asarray(r['sum']) for r in records)/weight
        variance=np.maximum(sum(np.asarray(r['squared_sum']) for r in records)/weight-mean**2,0)
        # A documented scale floor handles constant training features only.
        floor=np.maximum(np.abs(mean)*1e-6,1e-12)
        result[name]=dict(mean=mean,std=np.maximum(np.sqrt(variance),floor),
                          minimum=np.minimum.reduce([np.asarray(r['minimum']) for r in records]),
                          dims=records[0]['dims'],floored=(np.sqrt(variance)<floor).tolist())
    return result


def finalize(protocol_path, output):
    protocol=json.loads(protocol_path.read_text())
    paths=[output/f'case-{i:03d}.json' for i in range(len(protocol['initializations']))]
    records=[json.loads(p.read_text()) for p in paths]
    if any(r['status']!='passed' or r['protocol_sha256']!=sha256(protocol_path) for r in records):
        raise ValueError('Incomplete training statistics')
    for field in ('runner_sha256','loader_sha256'):
        if len({r[field] for r in records})!=1:raise ValueError('Statistics source changed during collection')
    files={};floors={}
    for group in GROUPS:
        values=merge([r['groups'][group]['values'] for r in records])
        # Only variables that can enter residual prediction need diff scales.
        differences=[{n:v for n,v in r['groups'][group]['differences'].items()
                      if not n.startswith('cyclone_') and n!='total_precipitation_6hr'} for r in records]
        diffs=merge(differences)
        floors[group]={n:v['floored'] for n,v in values.items() if np.any(v['floored'])}
        directory=output/group;directory.mkdir(exist_ok=True)
        for filename,source,key in [('mean.nc',values,'mean'),('std.nc',values,'std'),
                                    ('minimum.nc',values,'minimum'),('diff_std.nc',diffs,'std')]:
            ds=xr.Dataset({n:xr.DataArray(v[key],dims=v['dims']) for n,v in source.items()})
            ds=ds.assign_coords(level=protocol['pressure_levels_hpa'])
            path=directory/filename;ds.to_netcdf(path);files[str(path.relative_to(output))]=sha256(path)
    manifest=dict(status='passed',protocol_sha256=sha256(protocol_path),
                  case_manifests={p.name:sha256(p) for p in paths},files=files,scale_floors=floors,
                  definition='Spherical-area-weighted population moments over frozen training windows only; valid observation masks; residual differences at each stage cadence; scale floor max(abs(mean)*1e-6,1e-12)')
    tmp=output/'manifest.json.tmp';tmp.write_text(json.dumps(manifest,indent=2)+'\n');tmp.replace(output/'manifest.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--case-index',type=int);action.add_argument('--finalize',action='store_true')
    args=parser.parse_args()
    if args.finalize:finalize(args.protocol,args.output)
    else:collect(args.root,args.protocol,args.case_index,args.output)


if __name__=='__main__':main()
