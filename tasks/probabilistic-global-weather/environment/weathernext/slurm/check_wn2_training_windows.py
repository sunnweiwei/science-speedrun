"""Audit real 12h/6h contexts and native autoregressive targets on CPU."""
import dataclasses
import json
from pathlib import Path
import sys

import numpy as np
from weathernext.utils import data_utils,fiddle_config_io
from training_data import load_window,verify_case

root,protocol_path=map(Path,sys.argv[1:3])
protocol=json.loads(protocol_path.read_text())
verify_case(root,protocol_path,0)
task=fiddle_config_io.get_fiddle_config_by_name('weathernext2/configs/WeatherNext2').task
for source,resolution,cadence,ar in [('era5',1.,12,1),('hres_fc0',.25,6,8)]:
    dataset=load_window(root,protocol,0,source,resolution,cadence,ar)
    selected=dataclasses.replace(task,input_duration=f'{2*cadence}h')
    inputs,targets,forcings=data_utils.extract_inputs_targets_forcings(dataset,
        target_lead_times=slice(f'{cadence}h',f'{cadence*ar}h'),**dataclasses.asdict(selected))
    np.testing.assert_array_equal(inputs.time.values,np.asarray([-cadence,0],dtype='timedelta64[h]'))
    np.testing.assert_array_equal(targets.time.values,np.arange(1,ar+1)*np.timedelta64(cadence,'h'))
    assert set(targets)==set(task.target_variables) and set(inputs)==set(task.input_variables)
    assert targets.sizes['lat']==int(180/resolution)+1
    assert 'datetime' not in inputs.coords and 'total_precipitation_6hr' not in inputs
    print(json.dumps(dict(source=source,resolution=resolution,cadence=cadence,ar=ar,
                          input_frames=inputs.sizes['time'],target_fields=len(targets.data_vars))),flush=True)
    del dataset,inputs,targets,forcings
print('WN2_REAL_NATIVE_TRAINING_WINDOWS_OK')
