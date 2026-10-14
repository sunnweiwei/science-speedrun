#!/usr/bin/env python3
"""Online CPU preparation of identical contexts and truth for the paired subset."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import json
from pathlib import Path
import sys
import urllib.request
import hashlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'wn2'))
import numpy as np
import xarray as xr
import zarr
from audit_data import SOURCES
from source_metadata import source_record
from materialize_inputs import fill_sparse_context
from materialize_truth import initialize, materialize, finalize, write_json
from training_data import sha256


def prepare_context(root, protocol, index):
    p = json.loads(protocol.read_text())
    directory = root / 'inputs'
    directory.mkdir(exist_ok=True)
    path = directory / f'inputs-{index:03d}.nc'
    record = path.with_suffix('.json')
    if record.exists():
        old = json.loads(record.read_text())
        if old['protocol_sha256'] != sha256(protocol) or old['sha256'] != sha256(path):
            raise ValueError('Changed/corrupt paired context')
        return
    init = np.datetime64(p['evaluation_initializations'][index], 'ns')
    times = init + np.array([-12, -6, 0], dtype='timedelta64[h]')
    pieces, provenance = [], {}
    sources = {'hres_fc0': list(p['variables']),
               'era5': ['sea_surface_temperature', 'geopotential_at_surface', 'land_sea_mask'],
               'arco_era5': ['100m_u_component_of_wind', '100m_v_component_of_wind']}
    for name, fields in sources.items():
        url = SOURCES[name]
        with urllib.request.urlopen(url.replace('gs://', 'https://storage.googleapis.com/') + '/.zmetadata', timeout=60) as response:
            metadata = response.read()
        provenance[name] = source_record(url, metadata)
        with xr.open_zarr(url, consolidated=True, chunks=None, storage_options={'token': 'anon'}) as raw:
            raw = raw.rename(latitude='lat', longitude='lon').sortby('lat')
            selected = raw[fields].sel(time=times)
            if 'level' in selected.dims:
                selected = selected.sel(level=p['pressure_levels_hpa'])
            selected = selected.load().astype('float32')
            for field in fields:
                if field == 'sea_surface_temperature':
                    if np.isinf(selected[field]).any(): raise ValueError('Infinite SST')
                elif not np.isfinite(selected[field]).all():
                    dims = [d for d in selected[field].dims if d not in ('lat','lon')] + ['lat','lon']
                    array = selected[field].transpose(*dims)
                    values, _ = fill_sparse_context(array.values, selected.lat)
                    selected[field] = array.copy(data=values)
            pieces.append(selected)
    raw = xr.merge(pieces, join='exact', compat='equals')
    static = raw[['geopotential_at_surface', 'land_sea_mask']]
    result = xr.merge([raw.drop_vars(list(static)).expand_dims(batch=[0]), static], join='exact')
    result = result.assign_coords(datetime=(('batch','time'), times[None,:]), time=times-init)
    temp = path.with_suffix('.part')
    result.to_netcdf(temp, encoding={n: dict(zlib=True, complevel=1) for n in result.data_vars})
    temp.replace(path)
    write_json(record, dict(status='passed', protocol_sha256=sha256(protocol), sha256=sha256(path),
                           initialization=str(init), sources=provenance,
                           context_hours=[-12,-6,0],
                           limitation='HRES public archive lacks SST; both models receive the same ERA5 SST ancillary source'))


def precip(root, protocol, index):
    p = json.loads(protocol.read_text())
    record = root / 'cases' / f'{index:03d}.json'
    if record.exists():
        old = json.loads(record.read_text())
        if old['protocol_sha256'] != sha256(protocol): raise ValueError('Precipitation protocol changed')
        for name, value in old['chunks'].items():
            if sha256(root / 'truth.zarr' / name) != value: raise ValueError('Corrupt precipitation')
        return
    init = np.datetime64(p['evaluation_initializations'][index], 'ns')
    store = zarr.open_group(str(root / 'truth.zarr'), mode='a')
    with xr.open_zarr(SOURCES['era5'], consolidated=True, chunks=None, storage_options={'token':'anon'}) as raw:
        raw = raw.rename(latitude='lat', longitude='lon').sortby('lat')
        for lead_index, lead in enumerate(p['lead_hours']):
            stamps = init + np.array([lead-6, lead], dtype='timedelta64[h]')
            values = raw.total_precipitation_6hr.sel(time=stamps).load().sum('time', skipna=False).values.astype('float32')
            if not np.isfinite(values).all(): raise ValueError('Incomplete 12-hour precipitation truth')
            store['total_precipitation_12hr'][index, lead_index] = values
    chunks = {str(f.relative_to(root/'truth.zarr')): sha256(f)
              for f in (root/'truth.zarr/total_precipitation_12hr').glob(f'{index}.*')}
    if len(chunks) != len(p['lead_hours']): raise ValueError('Missing precipitation chunks')
    write_json(record, dict(status='passed', protocol_sha256=sha256(protocol), case_index=index,
                           chunks=chunks, source=SOURCES['era5'], accumulation_hours=12))


def prepare_case(job):
    root, weather, precipitation, i = job
    prepare_context(root, weather, i)
    materialize(root/'weather', i)
    precip(root/'precipitation', precipitation, i)
    print(json.dumps(dict(case=i, status='prepared')), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parent
    weather, precipitation = source/'weather-protocol.json', source/'precipitation-protocol.json'
    for name, protocol in [('weather',weather), ('precipitation',precipitation)]:
        d = root/name
        if not d.exists(): initialize(d, protocol)
        if sha256(d/'protocol.json') != sha256(protocol): raise ValueError('Prepared protocol differs')
    jobs = [(root, weather, precipitation, i) for i in range(len(json.loads(weather.read_text())['evaluation_initializations']))]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        list(pool.map(prepare_case, jobs))
    finalize(root/'weather')
    p = json.loads(precipitation.read_text())
    records = [root/'precipitation/cases'/f'{i:03d}.json' for i in range(len(p['evaluation_initializations']))]
    write_json(root/'precipitation/status.json', dict(status='passed', protocol_sha256=sha256(precipitation),
               case_manifests={f.name:sha256(f) for f in records}))
    write_json(root/'ready.json', dict(status='passed', weather_protocol_sha256=sha256(weather),
               precipitation_protocol_sha256=sha256(precipitation), cases=len(records)))


if __name__ == '__main__': main()
