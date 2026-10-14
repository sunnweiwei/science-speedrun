"""Native stage definitions and training-statistic replacement scope oracle."""
import json
from pathlib import Path
import tempfile

import numpy as np
import xarray as xr
from weathernext.utils import fiddle_config_io

from native_model import stage_config
from plan_schedule import STAGES
from training_data import sha256

original=fiddle_config_io.get_fiddle_config_by_name('weathernext2/configs/WeatherNext2')
native_maps=original.predictor_wrappers[0]['kwargs']
names=set(original.task.input_variables)|set(original.task.target_variables)
pressure={'temperature','geopotential','u_component_of_wind','v_component_of_wind','vertical_velocity','specific_humidity'}
with tempfile.TemporaryDirectory() as temporary:
    root=Path(temporary);files={}
    for group in ('era5_1deg_12h','hres_025deg_6h'):
        (root/group).mkdir()
        for filename,value in [('mean.nc',17.),('std.nc',2.),('diff_std.nc',3.),('minimum.nc',4.)]:
            ds=xr.Dataset({n:xr.DataArray(np.full(13,value),dims='level') if n in pressure else xr.DataArray(value) for n in names})
            ds=ds.assign_coords(level=list(original.task.pressure_levels))
            path=root/group/filename;ds.to_netcdf(path);files[str(path.relative_to(root))]=sha256(path)
    (root/'manifest.json').write_text(json.dumps(dict(status='passed',files=files)))
    for stage in (STAGES[0],STAGES[-1]):
        config=stage_config(stage,root)
        noisy=config.predictor_kwargs['noisy_function_kwargs']
        assert noisy['mesh_num_splits']==stage['mesh_splits']
        assert noisy['points_to_mesh_model_ctor'].keywords['deep_gnn_kwargs']['aggregate_normalization']==stage['encoder_aggregate_normalization']
        assert config.task.input_duration==f"{2*stage['cadence_hours']}h"
        assert config.task.target_variables==original.task.target_variables
        transformer=noisy['mesh_model_ctor'].keywords['transformer_kwargs']
        assert (transformer['num_layers'],transformer['d_model'],transformer['num_heads'])==(24,768,6)
        maps=config.predictor_wrappers[0]['kwargs']
        for name,value in [('mean_by_level',17.),('stddev_by_level',2.),('diffs_stddev_by_level',3.)]:
            assert set(maps[name])==names&set(native_maps[name])
            for array in maps[name].data_vars.values():np.testing.assert_array_equal(array.values,np.full(array.shape,value))
        assert 'cyclone_exists_gaussian_unit_mode' not in maps['stddev_by_level']
print('WN2_NATIVE_STAGE_NORMALIZATION_SCOPE_OK')
