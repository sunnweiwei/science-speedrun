"""Score completed checkpoint pairs; never mix unpaired dates."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
import xarray as xr
sys.path.insert(0,str(Path(__file__).resolve().parent))
from evaluate import panel,verify_metric_source
ap=argparse.ArgumentParser(description=__doc__)
for name in ('predictions','weather','rain','plan','output'):ap.add_argument('--'+name,type=Path,required=True)
a=ap.parse_args()
hash_file=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads(a.plan.read_text());revision=verify_metric_source()
weather=xr.open_zarr(a.weather/'truth.zarr',chunks=None,consolidated=True,decode_timedelta=True)
rain=xr.open_zarr(a.rain/'truth.zarr',chunks=None,consolidated=True,decode_timedelta=True)
rows=[];receipts=[];truth_hash=hashlib.sha256();dates=[]
fields={'2m_temperature':'K','mean_sea_level_pressure':'Pa','geopotential':'m2 s-2',
        '10m_wind_speed':'m s-1','total_precipitation_6hr':'m'}
for case in plan['cases']:
    stems={label:a.predictions/f"case-{case['case_index']:03d}-{label}" for label in ('before_hres','final')}
    if not all(p.with_suffix('.json').exists() for p in stems.values()):continue
    data={}
    for label,p in stems.items():
        rec=json.loads(p.with_suffix('.json').read_text())
        assert rec['status']=='passed' and rec['checkpoint']==label and rec['plan_sha256']==hash_file(a.plan)
        assert rec['forecast_sha256']==hash_file(p.with_suffix('.npz'))
        assert rec['weights_sha256']==plan['checkpoints'][label]['weights_sha256']
        data[label]=np.load(p.with_suffix('.npz'))
        receipts.append(dict(file=p.with_suffix('.json').name,sha256=hash_file(p.with_suffix('.json'))))
    dates.append(case['initialization'])
    for lead in plan['lead_hours']:
        for field,unit in fields.items():
            selection={'case':np.datetime64(case['initialization']),'lead':np.timedelta64(lead,'h')}
            if field=='geopotential':selection['level']=500
            target_source=rain if field=='total_precipitation_6hr' else weather
            if field=='10m_wind_speed':
                target=np.hypot(weather['10m_u_component_of_wind'].sel(selection),weather['10m_v_component_of_wind'].sel(selection))
            else:target=target_source[field].sel(selection)
            target=target.transpose('lat','lon').load()
            truth_hash.update(json.dumps([case['initialization'],lead,field]).encode())
            truth_hash.update(np.asarray(target.values,dtype='<f8').tobytes())
            metrics={}
            for label,arrays in data.items():
                if field=='10m_wind_speed':
                    v=np.hypot(arrays[f'10m_u_component_of_wind__{lead}'],arrays[f'10m_v_component_of_wind__{lead}'])
                else:v=arrays[f'{field}__{lead}']
                pred=xr.DataArray(v,dims=('member','lat','lon'),coords={'member':range(plan['members']),'lat':arrays['lat'],'lon':arrays['lon']})
                metrics[label]=panel(pred,target,minimum_valid_area_fraction=.9999)
            assert metrics['before_hres']['valid_area_fraction']==metrics['final']['valid_area_fraction']
            rows.append(dict(case=case['initialization'],variable=field,level_hpa=500 if field=='geopotential' else None,
                lead_hours=lead,units=unit,**metrics))
    for d in data.values():d.close()
summary=[]
for field,unit in fields.items():
    for lead in plan['lead_hours']:
        group=[r for r in rows if r['variable']==field and r['lead_hours']==lead]
        if not group:continue
        entry=dict(variable=field,lead_hours=lead,units=unit,cases=len(group))
        for label in ('before_hres','final'):
            entry[label+'_crps']=math.fsum(r[label]['crps'] for r in group)/len(group)
            entry[label+'_rmse']=math.sqrt(math.fsum(r[label]['mse'] for r in group)/len(group))
        for metric in ('crps','rmse'):
            b=entry['before_hres_'+metric];f=entry['final_'+metric]
            entry[metric+'_improvement_percent']=100*(1-f/b) if b>0 else None
            entry[metric+'_improved_dates']=sum(r['final'][metric if metric=='crps' else 'mse']<r['before_hres'][metric if metric=='crps' else 'mse'] for r in group)
        summary.append(entry)
result=dict(status='complete' if len(dates)==len(plan['cases']) else 'partial',
    scope='Paired 2021 development diagnostic: updates 544 versus 562; no full-training, test-set or significance claim',
    plan_sha256=hash_file(a.plan),evaluator_sha256=hash_file(Path(__file__).parent/'evaluate.py'),
    scorer_sha256=hash_file(Path(__file__)),weatherbenchx_revision=revision,truth_values_sha256=truth_hash.hexdigest(),
    completed_dates=dates,requested_dates=[c['initialization'] for c in plan['cases']],members=plan['members'],
    forecast_receipts=receipts,summary=summary,per_case=rows)
a.output.parent.mkdir(parents=True,exist_ok=True)
temp=a.output.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temp.replace(a.output)
print(json.dumps(dict(status=result['status'],paired_dates=len(dates),summary=summary)),flush=True)
