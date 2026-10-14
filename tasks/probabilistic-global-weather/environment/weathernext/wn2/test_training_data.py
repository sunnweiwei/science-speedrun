import json
import numpy as np
import pytest
import xarray as xr

from materialize_training import window_times
from training_data import load_window


def test_training_window_cannot_cross_split():
    protocol=dict(initializations=['2020-12-31T00:00:00'],offset_hours=[-12,0,48],
                  training_start='2017-01-01',training_end='2020-12-31T23:59:59')
    with pytest.raises(ValueError,match='split boundary'):window_times(protocol,0)
    protocol['initializations']=['2020-07-05T00:00:00']
    assert len(window_times(protocol,0))==3


def test_native_window_selects_context_targets_and_source(tmp_path):
    protocol=dict(initializations=['2020-07-05T00:00:00'],offset_hours=[-12,-6,0,6,12],
                  weather_variables=['temperature'])
    root=tmp_path/'case-000';root.mkdir()
    labels=root/'cyclones-.25';labels.mkdir();records=[]
    for i,h in enumerate(protocol['offset_hours']):
        stamp=np.datetime64(protocol['initializations'][0])+np.timedelta64(h,'h')
        coords=dict(lat=[-90.,0.,90.],lon=[0.,180.],time=stamp)
        def field(value):return xr.DataArray(np.full((3,2),value,dtype=np.float32),dims=['lat','lon'],coords={k:v for k,v in coords.items() if k!='time'})
        xr.Dataset(dict(temperature=field(i),total_precipitation_6hr=field(i),sea_surface_temperature=field(280),
                        geopotential_at_surface=field(10),land_sea_mask=field(0)),coords=coords).to_netcdf(root/f'era5-{i:02d}.nc')
        xr.Dataset(dict(temperature=field(100+i)),coords=coords).to_netcdf(root/f'hres_fc0-{i:02d}.nc')
        xr.Dataset({'100m_u_component_of_wind':field(5)},coords=coords).to_netcdf(root/f'arco_era5-{i:02d}.nc')
        if h>0:
            name=f'label-{i}.nc';xr.Dataset(dict(cyclone_exists_gaussian_unit_mode=field(0))).expand_dims(time=[stamp]).to_netcdf(labels/name)
            records.append(dict(time=str(stamp),file=name))
    (labels/'manifest.json').write_text(json.dumps(dict(frames=records)))
    ds=load_window(tmp_path,protocol,0,'hres_fc0',.25,6,2)
    np.testing.assert_array_equal(ds.time.values,np.array([-6,0,6,12],dtype='timedelta64[h]'))
    assert ds.temperature.isel(time=0).mean()==101
    assert ds.geopotential_at_surface.dims==('lat','lon')
    assert ds.cyclone_exists_gaussian_unit_mode.isel(time=slice(0,2)).isnull().all()
    assert ds.cyclone_exists_gaussian_unit_mode.isel(time=slice(2,None)).notnull().all()
    twelve=load_window(tmp_path,protocol,0,'era5',.25,12,1)
    assert twelve.total_precipitation_6hr.isel(time=-1).mean()==7
    assert twelve.total_precipitation_6hr.isel(time=slice(0,2)).isnull().all()
