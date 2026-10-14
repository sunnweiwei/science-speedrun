"""Native training-window loading shared by statistics, profiles and training."""
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(8*1024*1024):
            h.update(chunk)
    return h.hexdigest()


def read_frame(path, stride):
    with xr.open_dataset(path) as frame:
        return frame.isel(lat=slice(None,None,stride), lon=slice(None,None,stride)).load()


def load_window(root, protocol, index, source, resolution, cadence_hours=6, ar_steps=8):
    if source not in ('era5','hres_fc0') or resolution not in (1.,.25):
        raise ValueError('Unknown training source/resolution')
    directory = Path(root)/f'case-{index:03d}'
    stride = 4 if resolution == 1. else 1
    offsets = protocol['offset_hours']
    chosen = [i for i,h in enumerate(offsets) if -cadence_hours <= h <= cadence_hours*ar_steps and h % cadence_hours == 0]
    if len(chosen) != ar_steps+2:
        raise ValueError('Window does not cover requested cadence/rollout')
    frames = []
    for i in chosen:
        era = read_frame(directory/f'era5-{i:02d}.nc',stride)
        if cadence_hours == 12 and 'total_precipitation_6hr' in era:
            if offsets[i] > 0:
                previous = read_frame(directory/f'era5-{i-1:02d}.nc',stride)
                era['total_precipitation_6hr'] = (era.total_precipitation_6hr +
                    previous.total_precipitation_6hr.drop_vars('time'))
            else:
                # Precipitation is output-only; context accumulations are not
                # needed and the acquisition window lacks an earlier interval.
                era['total_precipitation_6hr'] = xr.full_like(era.total_precipitation_6hr,np.nan)
        if source == 'hres_fc0':
            era = xr.merge([era.drop_vars(protocol['weather_variables']),
                            read_frame(directory/f'hres_fc0-{i:02d}.nc',stride)],join='exact')
        frame = xr.merge([era,read_frame(directory/f'arco_era5-{i:02d}.nc',stride)],join='exact')
        frames.append(frame.expand_dims('time'))
    weather = xr.concat(frames,dim='time',data_vars='minimal',coords='minimal',compat='equals')
    # Static fields must remain independent of time and batch.
    static = weather[['geopotential_at_surface','land_sea_mask']].isel(time=0,drop=True)
    weather = weather.drop_vars(list(static))
    cyclone_dir = directory/('cyclones-1' if resolution == 1. else 'cyclones-.25')
    label_manifest = json.loads((cyclone_dir/'manifest.json').read_text())
    available = {np.datetime64(f['time'],'ns'):f for f in label_manifest['frames']}
    labels=[]
    for stamp in weather.time.values[2:]:
        frame=available[np.datetime64(stamp,'ns')]
        with xr.open_dataset(cyclone_dir/frame['file']) as ds:
            ds=ds.load()
        if 'time' in ds.dims:
            ds=ds.squeeze('time',drop=True)
        labels.append(ds.expand_dims(time=[stamp]))
    cyclones=xr.concat(labels,dim='time').reindex(time=weather.time)
    temporal=xr.merge([weather,cyclones],join='exact').expand_dims(batch=[0])
    init=np.datetime64(protocol['initializations'][index],'ns')
    stamps=temporal.time.values
    ds=xr.merge([temporal,static],join='exact').assign_coords(
        datetime=(('batch','time'),stamps[None,:]),time=stamps-init)
    return ds


def verify_case(root, protocol_path, index):
    """Verify all data before a training process may consume a window."""
    directory=Path(root)/f'case-{index:03d}'
    weather=json.loads((directory/'weather.json').read_text())
    protocol=json.loads(Path(protocol_path).read_text())
    approved={sha256(protocol_path),protocol.get('acquisition_protocol_sha256')}
    if weather['status']!='passed' or weather['protocol_sha256'] not in approved:
        raise ValueError('Incomplete or mismatched training weather')
    expected_times=[str(np.datetime64(protocol['initializations'][index],'ns')+np.timedelta64(h,'h'))
                    for h in protocol['offset_hours']]
    if weather['times']!=expected_times:raise ValueError('Acquisition timestamps differ from training protocol')
    for name,digest in weather['files'].items():
        if sha256(directory/name)!=digest:raise ValueError(f'Corrupt training file: {name}')
    for subdir in ('cyclones-1','cyclones-.25'):
        labels=json.loads((directory/subdir/'manifest.json').read_text())
        if labels['status']!='passed' or labels['calibration_end']!=protocol['calibration_end']:
            raise ValueError('Incomplete or noncausal cyclone labels')
        expected={str(np.datetime64(protocol['initializations'][index])+np.timedelta64(h,'h')) for h in protocol['offset_hours'] if h>0}
        if {f['time'] for f in labels['frames']}!=expected:raise ValueError('Cyclone timestamp coverage mismatch')
        for frame in labels['frames']:
            if sha256(directory/subdir/frame['file'])!=frame['sha256']:
                raise ValueError('Corrupt cyclone label')
    return dict(case_index=index,weather_manifest_sha256=sha256(directory/'weather.json'),
                cyclone_manifests={d:sha256(directory/d/'manifest.json') for d in ('cyclones-1','cyclones-.25')})
