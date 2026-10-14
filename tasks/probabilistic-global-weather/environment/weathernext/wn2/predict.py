#!/usr/bin/env python3
"""Portable native WN2 autoregressive inference from context-only requests."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

import jax
import numpy as np
import xarray as xr
from weathernext.utils import checkpoint, data_utils
from weathernext.weathernext2 import fgn

from forecast_store import initialize
from inference_state import resume_lead, save_state, load_state, state_identity
from native_model import stage_config, transforms
from plan_schedule import STAGES
from training_data import sha256


def context(path,task):
    ds=xr.load_dataset(path)
    if ds.sizes.get('batch')!=1 or ds.sizes.get('time')!=2:raise ValueError('One case and two context frames required')
    np.testing.assert_array_equal(ds.time.values,np.asarray([-6,0],dtype='timedelta64[h]'))
    np.testing.assert_array_equal(ds.lat.values,np.linspace(-90,90,721))
    np.testing.assert_array_equal(ds.lon.values,np.arange(1440)/4)
    np.testing.assert_array_equal(ds.level.values,task.pressure_levels)
    init=ds.datetime.values[0,-1].astype('datetime64[ns]')
    np.testing.assert_array_equal(ds.datetime.values,init+np.asarray([[-6,0]],dtype='timedelta64[h]'))
    data_utils.add_derived_vars(ds)
    ds=ds[list(task.input_variables)].drop_vars('datetime',errors='ignore')
    for name in ds.data_vars:
        values=ds[name].values
        if np.isinf(values).any() or (name!='sea_surface_temperature' and not np.isfinite(values).all()):
            raise ValueError(f'Invalid input field {name}')
    return init,ds


def forcing(inputs,init,lead,task):
    ds=xr.Dataset(coords=dict(batch=[0],time=np.asarray([6],dtype='timedelta64[h]'),
        lat=inputs.lat,lon=inputs.lon,datetime=(('batch','time'),np.asarray([[init+np.timedelta64(lead,'h')]]))))
    data_utils.add_derived_vars(ds)
    return ds[list(task.forcing_variables)].drop_vars('datetime',errors='ignore')


def template(inputs,task):
    variables={}
    for name in task.target_variables:
        dims=['batch','time']+(['level'] if name in inputs and 'level' in inputs[name].dims else [])+['lat','lon']
        sizes=dict(batch=1,time=1,level=len(task.pressure_levels),lat=721,lon=1440)
        variables[name]=(dims,np.zeros([sizes[d] for d in dims],dtype=np.float32))
    return xr.Dataset(variables,coords=dict(batch=[0],time=np.asarray([6],dtype='timedelta64[h]'),
        level=list(task.pressure_levels),lat=inputs.lat,lon=inputs.lon))


def next_context(inputs,prediction,forcings,task):
    result={}
    combined=xr.merge([prediction,forcings],compat='equals')
    for name in task.input_variables:
        if 'time' not in inputs[name].dims:
            result[name]=inputs[name]
        else:
            array=xr.concat([inputs[name].isel(time=[-1]),combined[name]],dim='time')
            result[name]=array.assign_coords(time=np.asarray([-6,0],dtype='timedelta64[h]'))
    return xr.Dataset(result)


def main():
    from wn2_validation import ensure_deterministic_process
    ensure_deterministic_process()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();begin=time.monotonic()
    manifest=json.loads((args.package/'package.json').read_text())
    revision=subprocess.check_output(['git','-C',str(Path(fgn.__file__).resolve().parents[2]),
                                      'rev-parse','HEAD'],text=True).strip()
    if revision!=manifest['source_revision']:raise ValueError('Unexpected native model runtime revision')
    for name,digest in manifest['files'].items():
        if sha256(args.package/name)!=digest:raise ValueError(f'Package checksum mismatch: {name}')
    request=json.loads((args.input/'request.json').read_text())
    start_lead=resume_lead(request)
    ids=[c['id'] for c in request['cases']]
    if len(set(ids))!=len(ids) or not ids or any(not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',i) for i in ids):
        raise ValueError('Invalid or duplicate input IDs')
    leads=request['lead_hours'];members=request['members']
    if not leads or leads!=sorted(set(leads)) or any(type(h)!=int or h<12 or h>360 or h%12 for h in leads):
        raise ValueError('Invalid lead hours')
    if type(members)!=int or not 1<=members<=56:raise ValueError('Expected one to 56 members')
    seed=request.get('seed',0)
    if type(seed)!=int or not 0<=seed<2**32:raise ValueError('Invalid inference seed')
    config=stage_config(STAGES[-1],args.package/'statistics')
    forward,_=transforms(config)
    with (args.package/'weights.npz').open('rb') as stream:
        params=checkpoint.load(stream,fgn.CheckPoint).params
    if sum(a.size for a in jax.tree.leaves(params))!=183785575:raise ValueError('Wrong WN2 parameter count')
    params=jax.device_put(params)
    apply=jax.jit(forward.apply)
    args.output.mkdir(parents=True,exist_ok=True)
    for case in request['cases']:
        input_path=(args.input/case['file']).resolve()
        if not input_path.is_relative_to(args.input.resolve()):raise ValueError('Context path escapes input directory')
        init,inputs=context(input_path,config.task)
        target=template(inputs,config.task)
        # Seed on timestamp, not request order or visible benchmark IDs.
        case_seed=int.from_bytes(hashlib.sha256(str(init).encode()).digest()[:4],'little')
        for member in range(members):
            store=initialize(args.output/f"{case['id']}-member-{member:03d}.zarr",init,leads,1,
                list(config.task.pressure_levels),manifest['variables'],inputs.lat.values,inputs.lon.values)
            store['member'][:]=[member]
            current=inputs
            if start_lead:
                relative=Path(request['resume']['directory'])
                state_path=(args.input/relative/f'member-{member:03d}').resolve()
                if not state_path.is_relative_to(args.input.resolve()):raise ValueError('State path escapes input directory')
                current=load_state(state_path,state_identity(init,seed,member,start_lead,sha256(args.package/'package.json')))
            previous_rain=None
            for lead in range(start_lead+6,max(leads)+1,6):
                forcings=forcing(current,init,lead,config.task)
                key=jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(seed),case_seed),member),lead)
                prediction=apply(params,key,current,target,forcings)
                jax.block_until_ready(prediction)
                for name in config.task.target_variables:
                    values=np.asarray(prediction[name].values)
                    if np.isinf(values).any() or (name!='sea_surface_temperature' and not np.isfinite(values).all()):
                        raise ValueError(f'Nonfinite native forecast: {name}, member={member}, lead={lead}')
                current_rain=np.asarray(prediction.total_precipitation_6hr.isel(batch=0,time=0).transpose('lat','lon').values)
                if lead in leads:
                    if previous_rain is None:raise ValueError('Missing first precipitation interval')
                    for name,spec in manifest['variables'].items():
                        dims=(['level'] if spec['pressure_levels'] else [])+['lat','lon']
                        store[name][0,leads.index(lead),0]=(previous_rain+current_rain if name=='total_precipitation_12hr'
                            else prediction[name].isel(batch=0,time=0).transpose(*dims).values)
                previous_rain=current_rain
                current=next_context(current,prediction,forcings,config.task)
                print(json.dumps(dict(case_id=case['id'],member=member,lead_hours=lead,elapsed_seconds=time.monotonic()-begin)),flush=True)
            if request['schema']=='wn2-inference/v3' and max(leads)<360:
                save_state(args.output/'state'/f'member-{member:03d}',current,
                    state_identity(init,seed,member,max(leads),sha256(args.package/'package.json')))
        (args.output/f"{case['id']}.complete.json").write_text(json.dumps(dict(status='passed',initialization=str(init),
            request_sha256=sha256(args.input/'request.json'),package_sha256=sha256(args.package/'package.json')))+'\n')


if __name__=='__main__':main()
