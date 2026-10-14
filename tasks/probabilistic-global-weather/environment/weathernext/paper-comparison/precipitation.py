"""Common precipitation accumulations, preserving each member's trajectory."""
import numpy as np
import xarray as xr


def accumulate_24h(rain, allow_partial=False):
    """Sum adjacent 12-hour intervals; never fill unavailable initial history."""
    leads = np.asarray(rain.lead.values).astype('timedelta64[h]')
    start=int(leads[0].astype('int64')) if len(leads) and allow_partial else 12
    if (len(leads)<(1 if allow_partial else 2) or start<12 or start%12
            or not np.array_equal(leads,np.arange(start,start+12*len(leads),12).astype('timedelta64[h]'))):
        raise ValueError('Expected consecutive 12-hour interval endpoints starting at hour 12')
    result = rain.isel(lead=slice(0,0)) if len(leads)==1 else (rain + rain.shift(lead=1)).isel(lead=slice(1,None))
    result.attrs = dict(rain.attrs, accumulation_hours=24)
    return result


def prepend_interval(current, previous):
    """Join trusted rain across windows; valid_time is redundant with case/lead."""
    current=current.drop_vars('valid_time',errors='ignore')
    previous=previous.drop_vars('valid_time',errors='ignore')
    if previous.sizes['lead']!=1 or previous.lead.values[0]+np.timedelta64(12,'h')!=current.lead.values[0]:
        raise ValueError('Missing adjacent precipitation interval')
    return xr.concat([previous,current],dim='lead',join='exact')
