"""Context-only, deterministic references for the current forecast contract."""
import numpy as np
import xarray as xr

NAME = 'persistence-dry'


def forecast(context, protocol, precipitation, date):
    """Hold instantaneous fields fixed; predict zero interval precipitation.

    Expand lazily: 56 identical members represent one point mass, not 56
    independent samples. No future truth or fitted statistics are used.
    """
    if 'time' not in context or np.any(context.time.values > np.timedelta64(0, 'h')):
        raise ValueError('Baseline context must not contain future frames')
    current = context.sel(time=np.timedelta64(0, 'h'))
    if not np.all(current.datetime.values == np.datetime64(date)):
        raise ValueError('Baseline initialization mismatch')
    current = current.squeeze(drop=True)
    fields = {}
    for name, spec in {**protocol['variables'], **precipitation['variables']}.items():
        value = xr.zeros_like(current['2m_temperature']) if name.startswith('total_precipitation_') else current[name]
        dims = (['level'] if spec['pressure_levels'] else []) + ['lat', 'lon']
        value = value.transpose(*dims)
        if not np.isfinite(value.values).all():
            raise ValueError(f'Nonfinite initial field: {name}')
        fields[name] = xr.DataArray(value.values, dims=dims, coords={d:value[d] for d in dims}, attrs={'units':spec['units']})
    ds = xr.Dataset(fields).chunk({'lat':180, 'lon':360})
    leads = np.asarray(protocol['lead_hours'], dtype='timedelta64[h]').astype('timedelta64[ns]')
    dates = np.asarray([date], dtype='datetime64[ns]')
    ds = ds.expand_dims(case=dates, lead=leads, member=np.arange(protocol['members']))
    return ds.assign_coords(valid_time=(('case','lead'), dates[:,None]+leads[None,:]))
