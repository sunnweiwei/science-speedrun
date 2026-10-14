#!/usr/bin/env python3
"""Changing post-cutoff winds must not change the native calibration model."""
import argparse
import numpy as np
import xarray as xr
from weathernext.cyclones import ibtracs_processing_utils as utils


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ibtracs',required=True);args=p.parse_args()
    raw=xr.open_dataset(args.ibtracs).load()
    raw['time']=raw.time.dt.round('1s')
    raw=raw.assign_coords(start_time=raw.time.min('date_time',skipna=True),end_time=raw.time.max('date_time',skipna=True))
    cutoff='2020-12-31T23:59:59'
    def coefficients(data):
        model=utils.create_wind_speed_calibration_and_aggregation_model(data,'1980-01-01',cutoff,True)
        return {k:np.array([v.keywords['coefficient'],v.keywords['intercept']]) for k,v in model.keywords['models'].items()}
    before=coefficients(raw)
    post=raw.time>np.datetime64(cutoff)
    assert int(post.sum())>0
    for name in utils.IBTRACS_WIND_VARIABLES:
        raw[name]=raw[name].where(~post,other=1e7)
    after=coefficients(raw)
    assert before.keys()==after.keys()
    for name in before:np.testing.assert_array_equal(before[name],after[name])
    print('WN2_CYCLONE_POST_CUTOFF_CALIBRATION_INVARIANCE_OK')


if __name__=='__main__':main()
