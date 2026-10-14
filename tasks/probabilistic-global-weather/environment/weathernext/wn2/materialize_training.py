#!/usr/bin/env python3
"""Materialize split-contained native weather windows for the WN2 curriculum.

Cyclone labels are generated separately by the pinned native model runtime.
The release is incomplete until those labels and training-only statistics pass.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

import numpy as np
import xarray as xr

from audit_data import SOURCES
from source_metadata import source_record
from evaluate import sha256
from materialize_inputs import fill_sparse_context
from materialize_truth import write_json


def window_times(protocol, index):
    stamp = np.datetime64(protocol['initializations'][index], 'ns')
    times = stamp + np.asarray(protocol['offset_hours'], dtype='timedelta64[h]')
    if (times.min() < np.datetime64(protocol['training_start']) or
            times.max() > np.datetime64(protocol['training_end'])):
        raise ValueError('Training window crosses split boundary')
    if len(np.unique(times)) != len(times) or np.any(np.diff(times) <= np.timedelta64(0, 'h')):
        raise ValueError('Training timestamps must increase strictly')
    return times


def materialize(protocol_path, output, index):
    protocol = json.loads(protocol_path.read_text())
    times = window_times(protocol, index)
    directory = output / f'case-{index:03d}'
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / 'weather.json'
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous['protocol_sha256'] not in {sha256(protocol_path),protocol.get('acquisition_protocol_sha256')}:
            raise ValueError('Changed training protocol')
        for name, digest in previous['files'].items():
            if sha256(directory/name) != digest:
                raise ValueError('Corrupt existing training frame')
        return
    selections = {
        'era5': protocol['weather_variables'] + ['sea_surface_temperature',
                 'total_precipitation_6hr', 'geopotential_at_surface', 'land_sea_mask'],
        'hres_fc0': protocol['weather_variables'],
        'arco_era5': ['100m_u_component_of_wind', '100m_v_component_of_wind'],
    }
    record = dict(status='started', protocol_sha256=sha256(protocol_path),
                  runner_sha256=sha256(Path(__file__)), case_index=index,
                  initialization=protocol['initializations'][index],
                  times=[str(t) for t in times], sources={}, files={}, repairs={})
    begin = time.monotonic()
    for source_name, names in selections.items():
        url = SOURCES[source_name]
        metadata = urllib.request.urlopen(url.replace('gs://', 'https://storage.googleapis.com/') + '/.zmetadata', timeout=60).read()
        record['sources'][source_name] = source_record(url, metadata)
        with xr.open_zarr(url, consolidated=True, chunks=None, storage_options={'token':'anon'}) as raw:
            raw = raw.rename(latitude='lat', longitude='lon').sortby('lat')
            np.testing.assert_array_equal(raw.lat.values, np.linspace(-90, 90, 721))
            np.testing.assert_array_equal(raw.lon.values, np.arange(1440)/4)
            if 'level' in raw.dims:
                raw = raw.sel(level=protocol['pressure_levels_hpa'])
            for frame_index, stamp in enumerate(times):
                frame = raw[names].sel(time=stamp).load().astype('float32')
                for name in names:
                    if np.isinf(frame[name]).any():
                        raise ValueError(f'Infinite training field: {name}')
                    if name == 'sea_surface_temperature':
                        if float(np.isfinite(frame[name]).mean()) < .5:
                            raise ValueError('Missing SST frame')
                    elif not np.isfinite(frame[name]).all():
                        dims = [d for d in frame[name].dims if d not in ('lat','lon')]+['lat','lon']
                        field = frame[name].transpose(*dims)
                        values, locations = fill_sparse_context(field.values, frame.lat)
                        frame[name] = field.copy(data=values)
                        record['repairs'][f'{source_name}/{frame_index}/{name}'] = dict(dimensions=dims, indices=locations)
                filename = f'{source_name}-{frame_index:02d}.nc'
                path = directory/filename
                temporary = path.with_suffix('.nc.part')
                frame.to_netcdf(temporary, encoding={n:dict(zlib=True, complevel=1) for n in frame.data_vars})
                temporary.replace(path)
                record['files'][filename] = sha256(path)
                print(json.dumps(dict(case_index=index, source=source_name, frame=frame_index,
                                      seconds=time.monotonic()-begin)), flush=True)
    record.update(status='passed', seconds=time.monotonic()-begin,
                  scope='Weather windows only; cyclone labels and normalization require separate verification')
    write_json(manifest, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--case-index', required=True, type=int)
    args = parser.parse_args()
    count = len(json.loads(args.protocol.read_text())['initializations'])
    if not 0 <= args.case_index < count:
        parser.error('Case index out of bounds')
    materialize(args.protocol, args.output, args.case_index)


if __name__ == '__main__':
    main()
