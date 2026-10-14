#!/usr/bin/env python3
"""Materialize native cyclone targets with training-only wind calibration."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import xarray as xr
from weathernext.cyclones import ibtracs_processing_stages as stages
from weathernext.cyclones import ibtracs_processing_utils as utils
from weathernext.utils import fiddle_config_io


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ibtracs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--times',nargs='+',required=True)
    parser.add_argument('--resolution',type=float,choices=[1.,.25],default=.25)
    parser.add_argument('--calibration-start',default='1980-01-01')
    parser.add_argument('--training-end',default='2020-12-31T23:59:59')
    args=parser.parse_args()
    if np.datetime64(args.calibration_start)>=np.datetime64(args.training_end):parser.error('Calibration interval is empty')
    times=np.asarray(args.times,dtype='datetime64[s]')
    if len(np.unique(times))!=len(times) or np.any(times.astype('int64')%(6*3600)):
        parser.error('Times must be unique exact six-hour synoptic timestamps')
    args.output.mkdir(parents=True,exist_ok=True)
    revision=subprocess.check_output(['git','-C',str(Path(stages.__file__).resolve().parents[2]),'rev-parse','HEAD'],text=True).strip()
    if revision!='89c4b2a77a1c57b328b909c575550fd2e5aadc9c':raise ValueError('Wrong WeatherNext revision')
    names=[n for n in fiddle_config_io.get_fiddle_config_by_name('weathernext2/configs/WeatherNext2').task.target_variables if n.startswith('cyclone_')]
    raw=xr.open_dataset(args.ibtracs).load()
    raw['time']=raw.time.dt.round('1s')
    raw=raw.assign_coords(start_time=raw.time.min('date_time',skipna=True),end_time=raw.time.max('date_time',skipna=True))
    raw['lon']=((raw.lon+180)%360)-180
    calibration=utils.create_wind_speed_calibration_and_aggregation_model(
        raw,args.calibration_start,args.training_end,wind_calibration_skip_if_no_overlap=True)
    report=dict(schema='wn2-cyclone-targets/v1',status='started',source_commit=revision,
                ibtracs_sha256=sha256(args.ibtracs),runner_sha256=sha256(Path(__file__)),
                calibration_start=args.calibration_start,calibration_end=args.training_end,
                calibration_storms_must_be_wholly_inside_training_interval=True,
                calibration_models={name:fn.keywords for name,fn in calibration.keywords['models'].items()},
                source='NOAA IBTrACS v04r01 since1980, retrospective best tracks',
                resolution=args.resolution,variables=names,frames=[],
                settings=dict(existence_gaussian_and_disc_radius_km=120.,scalar_variable_disc_radius_km=240.,
                              quadrant_nan_mask_disc_radius_km=800.,quadrant_num_subgrid=8,core_cyclone_variables_only=True))
    for i,stamp in enumerate(times):
        start=time.monotonic()
        # A wholly absent record cannot distinguish an empty basin from absent
        # archive coverage. Refuse dates beyond the archive's observed span.
        if stamp<raw.start_time.min().values or stamp>raw.end_time.max().values:raise ValueError('Date outside IBTrACS observed coverage')
        sliced=stages.load_time_slice_from_ibtracs(raw,stamp,i)
        gridded=stages.convert_ibtracs_to_gridded_data(sliced,resolution=args.resolution,
                    wind_calibration_model=calibration,**report['settings']).grid_data
        ds=gridded.single_level_temporal.to_dataset(dim='single_level_temporal_variable')[names]
        ds=ds.rename(latitude='lat',longitude='lon').astype('float32')
        if set(ds.data_vars)!=set(names):raise ValueError('Native cyclone target schema mismatch')
        counts={name:dict(finite=int(np.isfinite(ds[name].values).sum()),nan=int(np.isnan(ds[name].values).sum())) for name in names}
        # NaNs in disc labels encode missing observations and are retained for
        # the native CRPS mask. Infinity is never a valid mask.
        if any(np.isinf(ds[name].values).any() for name in names):raise ValueError('Infinite label')
        if not np.isfinite(ds.cyclone_exists_gaussian_unit_mode.values).all():raise ValueError('Invalid existence field')
        path=args.output/f'cyclones-{str(stamp).replace(":", "")}.nc'
        temp=path.with_suffix('.nc.part')
        ds.to_netcdf(temp,encoding={name:dict(zlib=True,complevel=1) for name in names})
        temp.replace(path)
        frame=dict(time=str(stamp),file=path.name,sha256=sha256(path),seconds=time.monotonic()-start,counts=counts)
        report['frames'].append(frame)
        print(json.dumps(frame),flush=True)
    report['status']='passed'
    report['scope']='Native label materialization only; no meteorological corpus, normalization or scientific evaluation acceptance'
    temp=args.output/'manifest.json.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(args.output/'manifest.json')


if __name__=='__main__':main()
