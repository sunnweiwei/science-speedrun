#!/usr/bin/env python3
"""Repeat identical full-native calls in one process without executable caching."""
import hashlib
import json
from pathlib import Path
import time

import jax
import numpy as np
from weathernext.utils import checkpoint
from weathernext.weathernext2 import fgn

from native_model import stage_config,transforms
from plan_schedule import STAGES
from predict import context,forcing,template

begin=time.monotonic()
config=stage_config(STAGES[-1],Path('/model/statistics'))
forward,_=transforms(config)
with open('/model/weights.npz','rb') as stream:
    params=jax.device_put(checkpoint.load(stream,fgn.CheckPoint).params)
init,inputs=context(Path('/data/inputs-000.nc'),config.task)
target=template(inputs,config.task);forcings=forcing(inputs,init,6,config.task)
case_seed=int.from_bytes(hashlib.sha256(str(init).encode()).digest()[:4],'little')
key=jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(0),case_seed),0),6)
apply=jax.jit(forward.apply);reference=None;reports=[]
for repeat in range(3):
    output=apply(params,key,inputs,target,forcings);jax.block_until_ready(output)
    arrays={name:np.asarray(output[name].values) for name in output.data_vars}
    if reference is None:reference=arrays
    differences={}
    for name,a in arrays.items():
        b=reference[name];changed=~((a==b)|(np.isnan(a)&np.isnan(b)))
        if changed.any():differences[name]=dict(count=int(changed.sum()),max_abs=float(np.nanmax(abs(a-b))))
    record=dict(repeat=repeat,seconds=time.monotonic()-begin,differences=differences,
                first_wind=float(arrays['10m_u_component_of_wind'].ravel()[0]))
    reports.append(record);print(json.dumps(record),flush=True)
Path('/run/repeats.json').write_text(json.dumps(dict(status='passed' if all(not r['differences'] for r in reports) else 'failed',
    scope='Same process, identical native inputs/parameters/key, three forward calls, executable cache disabled',repeats=reports),indent=2)+'\n')
if any(r['differences'] for r in reports):raise ValueError('Native forward changed within one process')
