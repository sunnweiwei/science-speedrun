"""Visible 2021 development scores using the same v2 submission contract."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import uuid

import numpy as np
import xarray as xr
from evaluate import evaluate, summarize, verify_metric_source
from merge_scores import reward
from namespace_inference import run
from training_data import sha256
from zarr_safety import audit_zarr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['package', 'data', 'scales', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--inference-timeout', type=int, default=7200)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    protocol_path = here / 'development-protocol.json'
    weather = json.loads(protocol_path.read_text())
    weather['lead_hours'] = list(range(12, 241, 12))
    weather['pool_cells'] = [1]
    rain = copy.deepcopy(weather)
    rain.update(truth_source='ERA5', variables={'total_precipitation_12hr': {'units': 'm', 'pressure_levels': False}})
    revision = verify_metric_source()
    rows = []
    rain_rows = []
    with xr.open_zarr(args.data / 'weather/truth.zarr', consolidated=True, chunks={}) as truth, \
            xr.open_zarr(args.data / 'precipitation/truth.zarr', consolidated=True, chunks={}) as rain_truth:
        for index, date in enumerate(weather['evaluation_initializations']):
            work = args.output / f'case-{index:03d}'
            work.mkdir()
            model = work / 'model'
            if any(p.is_symlink() for p in args.package.rglob('*')):
                raise ValueError('Submission symlinks forbidden')
            shutil.copytree(args.package, model)
            public = work / 'public'
            public.mkdir()
            context = args.data / 'inputs' / f'inputs-{index:03d}.nc'
            record = json.loads(context.with_suffix('.json').read_text())
            if record['sha256'] != sha256(context) or record['protocol_sha256'] != sha256(protocol_path):
                raise ValueError('Development input changed')
            shutil.copyfile(context, public / 'context.nc')
            case_id = 'development_' + uuid.uuid4().hex
            request = dict(schema='wn2-inference/v2', cases=[dict(id=case_id, file='context.nc')],
                           members=8, lead_hours=weather['lead_hours'], seed=20210910)
            (public / 'request.json').write_text(json.dumps(request))
            output = run(model.resolve(), public.resolve(), work.resolve(), args.inference_timeout)
            shards = []
            try:
                for member in range(8):
                    path = output / f'{case_id}-member-{member:03d}.zarr'
                    audit_zarr(path)
                    shards.append(xr.open_zarr(path, consolidated=True, chunks={}, decode_timedelta=True))
                forecast = xr.concat(shards, dim='member', join='exact', data_vars='minimal', coords='minimal', compat='equals')
                subset = weather | dict(evaluation_initializations=[date])
                leads = np.array(weather['lead_hours'], dtype='timedelta64[h]')
                observed = truth.isel(case=[index]).sel(lead=leads)
                rows.extend(evaluate(forecast[list(weather['variables'])], observed, subset, hashlib.sha256(), hashlib.sha256()))
                first = rain_truth.total_precipitation_6hr.isel(case=[index]).sel(lead=leads - np.timedelta64(6, 'h'))
                second = rain_truth.total_precipitation_6hr.isel(case=[index]).sel(lead=leads)
                accumulated = (first.assign_coords(lead=leads) + second).rename('total_precipitation_12hr').to_dataset()
                accumulated['total_precipitation_12hr'].attrs['units'] = 'm'
                accumulated = accumulated.assign_coords(level=weather['pressure_levels_hpa'], valid_time=observed.valid_time)
                accumulated.attrs['truth_source'] = 'ERA5'
                rain_rows.extend(evaluate(forecast[['total_precipitation_12hr']].assign_coords(level=forecast.level),
                    accumulated, rain | dict(evaluation_initializations=[date]), hashlib.sha256(), hashlib.sha256()))
            finally:
                for shard in shards:
                    shard.close()
    with xr.open_dataset(args.scales) as scales:
        metric = reward({'aggregate': summarize(rows)}, scales)
    result = dict(status='passed', scope='Four visible 2021 development dates; not test acceptance',
                  weatherbenchx_revision=revision, weather=summarize(rows), precipitation=summarize(rain_rows), reward=metric)
    (args.output / 'scores.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
