"""Diagnostic only: paired pre-HRES/final forecasts on four fixed 2021 dates."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ap=argparse.ArgumentParser(description=__doc__)
for name in ('package','stages','inputs','output','plan'):
    ap.add_argument('--'+name,type=Path,required=True)
a=ap.parse_args()
sys.path.insert(0,str(a.package))
from wn2_validation import ensure_deterministic_process
ensure_deterministic_process()
import jax
import numpy as np
from weathernext.utils import checkpoint
from weathernext.weathernext2 import fgn
from native_model import stage_config,transforms
from plan_schedule import STAGES
from predict import context,template,forcing,next_context
from training_data import sha256

plan=json.loads(a.plan.read_text())
assert plan['members']==8 and plan['lead_hours']==[6,24]
config=stage_config(STAGES[-1],a.package/'statistics')
forward,_=transforms(config)
apply=jax.jit(forward.apply)
params={}
for label,stage in [('before_hres','stage-02'),('final','stage-10')]:
    path=a.stages/stage/'weights.npz'
    report=json.loads((a.stages/stage/'report.json').read_text())
    assert report['status']=='passed' and not report['profile']
    assert sha256(path)==report['weights_sha256']==plan['checkpoints'][label]['weights_sha256']
    assert report['statistics_manifest_sha256']==sha256(a.package/'statistics/manifest.json')
    with path.open('rb') as stream: p=checkpoint.load(stream,fgn.CheckPoint).params
    assert sum(v.size for v in jax.tree.leaves(p))==183785575
    params[label]=jax.device_put(p)
assert jax.tree.structure(params['before_hres'])==jax.tree.structure(params['final'])
assert [(x.shape,x.dtype) for x in jax.tree.leaves(params['before_hres'])]==[(x.shape,x.dtype) for x in jax.tree.leaves(params['final'])]
a.output.mkdir(parents=True,exist_ok=True)
begin=time.monotonic()
fields=['2m_temperature','mean_sea_level_pressure','geopotential',
        '10m_u_component_of_wind','10m_v_component_of_wind','total_precipitation_6hr']
for case in plan['cases']:
    path=a.inputs/case['file']
    assert sha256(path)==case['sha256']
    init,inputs=context(path,config.task)
    assert str(init)==case['initialization'] and str(init).startswith('2021-')
    target=template(inputs,config.task)
    case_seed=int.from_bytes(hashlib.sha256(str(init).encode()).digest()[:4],'little')
    for label in params:
        stem=a.output/f"case-{case['case_index']:03d}-{label}"
        receipt=stem.with_suffix('.json')
        if receipt.exists():
            r=json.loads(receipt.read_text())
            assert r['plan_sha256']==sha256(a.plan) and r['forecast_sha256']==sha256(stem.with_suffix('.npz'))
            continue
        values={(name,lead):[] for name in fields for lead in plan['lead_hours']}
        for member in range(plan['members']):
            current=inputs
            for lead in range(6,25,6):
                forcings=forcing(current,init,lead,config.task)
                key=jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(plan['inference_seed']),case_seed),member),lead)
                prediction=apply(params[label],key,current,target,forcings)
                jax.block_until_ready(prediction)
                for name in fields:
                    field=prediction[name].isel(batch=0,time=0)
                    if name=='geopotential': field=field.sel(level=500)
                    array=np.asarray(field.transpose('lat','lon').values)
                    assert np.isfinite(array).all(), (label,case['case_index'],member,lead,name)
                    if lead in plan['lead_hours']: values[name,lead].append(array)
                current=next_context(current,prediction,forcings,config.task)
                print(json.dumps(dict(checkpoint=label,case=case['case_index'],member=member,lead=lead,elapsed_seconds=time.monotonic()-begin)),flush=True)
        file=stem.with_suffix('.npz')
        np.savez(file,lat=inputs.lat.values,lon=inputs.lon.values,
                 **{f'{name}__{lead}':np.stack(arrays) for (name,lead),arrays in values.items()})
        r=dict(status='passed',checkpoint=label,case=case['case_index'],initialization=str(init),
            members=plan['members'],lead_hours=plan['lead_hours'],plan_sha256=sha256(a.plan),
            weights_sha256=plan['checkpoints'][label]['weights_sha256'],input_sha256=sha256(path),
            forecast_sha256=sha256(file),runner_sha256=sha256(Path(__file__)),jax_version=jax.__version__,
            device_kinds=[d.device_kind for d in jax.devices()],scope='Development checkpoint diagnostic; not a submitted final-model evaluation')
        temp=receipt.with_suffix('.tmp');temp.write_text(json.dumps(r,indent=2)+'\n');temp.replace(receipt)
