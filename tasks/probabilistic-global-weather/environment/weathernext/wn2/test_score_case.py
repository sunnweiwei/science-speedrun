import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import xarray as xr


def test_surface_only_shard_retains_level_coordinate_and_selects_truth_leads(tmp_path):
    case=np.array(['2020-01-01'],dtype='datetime64[ns]')
    leads=np.array([6,12],dtype='timedelta64[h]')
    coords=dict(case=case,lead=leads,level=[500,850],lat=[-90.,0.,90.],lon=[0.,90.,180.,270.],
                valid_time=(('case','lead'),case[:,None]+leads[None,:]))
    truth=xr.Dataset({'p':(('case','lead','lat','lon'),np.ones((1,2,3,4),dtype=np.float32))},coords=coords,attrs=dict(truth_source='test'))
    truth.p.attrs['units']='m'
    prediction=truth.isel(lead=[0]).expand_dims(member=[0,1])
    prediction['extra']=xr.zeros_like(prediction.p)
    prediction=prediction.assign_coords(valid_time=(('case','lead'),case[:,None]+leads[None,:1]))
    truth.to_zarr(tmp_path/'truth.zarr',consolidated=True)
    prediction.to_zarr(tmp_path/'prediction.zarr',consolidated=True)
    protocol=dict(status='test',evaluation_initializations=['2020-01-01'],lead_hours=[6],members=2,
                  pressure_levels_hpa=[500,850],grid=[3,4],variables={'p':dict(pressure_levels=False,units='m')},
                  truth_source='test',pool_cells=[1])
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    subprocess.run([sys.executable,str(Path(__file__).with_name('score_case.py')),'--prediction',str(tmp_path/'prediction.zarr'),
        '--truth',str(tmp_path/'truth.zarr'),'--protocol',str(tmp_path/'protocol.json'),'--case-index','0',
        '--output',str(tmp_path/'scores.json')],check=True)
    report=json.loads((tmp_path/'scores.json').read_text())
    assert report['status']=='passed' and report['aggregate'][0]['crps']==0
