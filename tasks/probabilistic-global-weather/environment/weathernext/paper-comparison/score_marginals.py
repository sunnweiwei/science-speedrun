#!/usr/bin/env python3
"""Score all members of one completed case; preserve per-case sufficient statistics.

Includes enabled derived, precipitation, REV and pooling panels; the full suite
also requires the separate cyclone and spectral comparisons.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'wn2'))
import numpy as np
import xarray as xr
from evaluate import panel, validate_contract, verify_metric_source, cell_weights
from training_data import sha256
from spherical_pool import apply_pool, empirical_scores
from scipy.sparse import load_npz
from precipitation import accumulate_24h, prepend_interval
from window_scores import checked_leads, canonical_truth_sha256
from rev import Thresholds, panel_rows
from seeps import Climatology as SEEPSClimatology, score as seeps_score


def derived(ds):
    return xr.Dataset({
        'wind_speed': np.hypot(ds.u_component_of_wind, ds.v_component_of_wind),
        '10m_wind_speed': np.hypot(ds['10m_u_component_of_wind'], ds['10m_v_component_of_wind']),
        'geopotential_300_minus_500': ds.geopotential.sel(level=300, drop=True)-ds.geopotential.sel(level=500, drop=True),
    })


def main():
    # Ensemble concatenation must remain lazy; chunks=None would materialize
    # every member and lead. Check the required backend before inventory I/O.
    import dask.array
    import dask
    dask.config.set(scheduler='threads', num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--predictions', type=Path)
    group.add_argument('--baseline', choices=['persistence-dry'])
    ap.add_argument('--reference-scorer', type=Path, help='Archived model scorer for an existing run; baseline mode only')
    ap.add_argument('--data', required=True, type=Path)
    ap.add_argument('--case', required=True, type=int)
    ap.add_argument('--rev-climatology',type=Path)
    ap.add_argument('--seeps-climatology',type=Path)
    ap.add_argument('--pooling', type=Path, help='Verified spherical geometry directory; omit for marginal-only scoring')
    ap.add_argument('--pooling-only', action='store_true', help='Score only spherical pooling panels')
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--lead-hours',type=int,nargs='+')
    ap.add_argument('--previous-rain',type=Path,help='Trusted preceding 12h interval for window boundary')
    ap.add_argument('--score-leads',type=int,nargs='+',help='Consecutive part of --lead-hours to score; the '
                    'verifier splits a window across processes this way. The whole window is still loaded, '
                    'so 24-hour precipitation at a part boundary is unchanged')
    args = ap.parse_args()
    if args.pooling_only and (not args.pooling or args.rev_climatology or args.seeps_climatology):
        ap.error('--pooling-only requires --pooling and excludes native-grid climatology scores')
    here = Path(__file__).resolve().parent
    p = json.loads((here/'weather-protocol.json').read_text())
    leads=checked_leads(args.lead_hours,p['lead_hours'])
    score_leads=leads if args.score_leads is None else checked_leads(args.score_leads,leads)
    revision = verify_metric_source()
    index = args.case
    date = p['evaluation_initializations'][index]
    records = []
    shards = []
    lineage = None
    for member in range(p['members']) if not args.baseline else ():
        name = f'case-{index:03d}-member-{member:03d}'
        record_path = args.predictions/f'{name}.complete.json'
        record = json.loads(record_path.read_text())
        if record['status'] != 'passed' or record['protocol_sha256'] != sha256(here/'weather-protocol.json'):
            raise ValueError('Unverified prediction protocol')
        if record['member'] != member or np.datetime64(record['initialization']) != np.datetime64(date):
            raise ValueError('Mismatched member/case identity')
        identity = {k: record.get(k) for k in ('assets_lock_sha256','package_sha256','runner_sha256','noise_grid_cache_sha256','seed','context_sha256')}
        if lineage is not None and identity != lineage: raise ValueError('Mixed model/RNG/context lineage')
        lineage = identity
        directory = args.predictions/f'{name}.zarr'
        for name, digest in record['files'].items():
            path = (directory/name).resolve()
            if not path.is_relative_to(directory.resolve()) or sha256(path) != digest:
                raise ValueError('Corrupt forecast inventory')
        records.append(sha256(record_path))
        shard = xr.open_zarr(directory, consolidated=True, chunks={}, decode_timedelta=True)
        np.testing.assert_array_equal(shard.member, [member])
        shards.append(shard)
    if args.reference_scorer and not args.baseline:
        ap.error('--reference-scorer is only valid for simple baselines')
    if args.baseline:
        from simple_baselines import forecast as baseline_forecast
        context_path = args.data/'inputs'/f'inputs-{index:03d}.nc'
        with xr.open_dataset(context_path, decode_timedelta=True) as context:
            forecast = baseline_forecast(context, p, json.loads((here/'precipitation-protocol.json').read_text()), date)
        lineage = dict(baseline=args.baseline, runner_sha256=sha256(here/'simple_baselines.py'), context_sha256=sha256(context_path))
    else:
        forecast = xr.concat(shards, dim='member', join='exact', data_vars='minimal', coords='minimal', compat='equals')
    if args.baseline and args.lead_hours is not None:
        forecast=forecast.sel(lead=np.array(leads,dtype='timedelta64[h]'))
    truth = xr.open_zarr(args.data/'weather/truth.zarr', consolidated=True, chunks={}, decode_timedelta=True).isel(case=[index])
    one = copy.deepcopy(p); one['evaluation_initializations'] = [date]
    if args.lead_hours is not None:
        one['lead_hours']=leads
        truth=truth.sel(lead=np.array(leads,dtype='timedelta64[h]'))
    validate_contract(forecast[list(p['variables'])], truth, one)
    precipitation = json.loads((here/'precipitation-protocol.json').read_text())
    precipitation['evaluation_initializations'] = [date]
    rain_truth = xr.open_zarr(args.data/'precipitation/truth.zarr', consolidated=True, chunks={}, decode_timedelta=True).isel(case=[index])
    # Rain has no pressure levels; retain weather's level coordinate for strict common-grid checks.
    precipitation['pressure_levels_hpa'] = p['pressure_levels_hpa']
    rain_truth = rain_truth.assign_coords(level=p['pressure_levels_hpa'])
    rain_prediction = forecast[list(precipitation['variables'])].assign_coords(level=p['pressure_levels_hpa'])
    if args.lead_hours is not None:
        precipitation['lead_hours']=leads
        rain_truth=rain_truth.sel(lead=np.array(leads,dtype='timedelta64[h]'))
    validate_contract(rain_prediction, rain_truth, precipitation)
    thresholds=Thresholds(args.rev_climatology) if args.rev_climatology else None
    rain_climatology=SEEPSClimatology(args.seeps_climatology) if args.seeps_climatology else None
    seeps_rows=[]
    rev_rows=[]
    rows = []
    pooled_rows = []
    geometries = {}
    geometry_hashes = {}
    if args.pooling:
        for stride in (1,4):
            for k in range(2,8 if stride==1 else 7):
                stem = args.pooling/f'stride-{stride}-mesh-{k}'
                matrix_path = Path(str(stem)+'.npz')
                metadata_path = Path(str(stem)+'-metadata.npz')
                meta = np.load(metadata_path)
                if int(meta['stride']) != stride or int(meta['mesh_refinement']) != k:
                    raise ValueError('Mismatched pooling geometry')
                geometries[stride,k] = (load_npz(matrix_path), meta['area'], float(meta['diameter_km']))
                for path in (matrix_path,metadata_path): geometry_hashes[path.name] = sha256(path)
    truth_lead_hashes = {}
    all_prediction = xr.merge([forecast, derived(forecast)], compat='equals')
    all_truth = xr.merge([truth, rain_truth.drop_vars('level'), derived(truth)], compat='equals')
    rain_for_sum=forecast.total_precipitation_12hr
    truth_for_sum=rain_truth.total_precipitation_12hr
    if leads[0]>12:
        if args.previous_rain is None:raise ValueError('Missing precipitation boundary state')
        with xr.open_zarr(args.previous_rain,consolidated=True,chunks=None,decode_timedelta=True) as previous:
            np.testing.assert_array_equal(previous.lead,np.array([leads[0]-12],dtype='timedelta64[h]'))
            for coord in ('case','member','lat','lon'):
                np.testing.assert_array_equal(previous[coord],forecast[coord])
            rain_for_sum=prepend_interval(rain_for_sum,previous.total_precipitation_12hr.load())
        with xr.open_zarr(args.data/'precipitation/truth.zarr',consolidated=True,chunks=None,decode_timedelta=True) as raw:
            prefix=raw.total_precipitation_12hr.isel(case=[index]).sel(lead=np.array([leads[0]-12],dtype='timedelta64[h]')).load()
        truth_for_sum=prepend_interval(truth_for_sum,prefix)
    all_prediction['total_precipitation_24hr'] = accumulate_24h(rain_for_sum,allow_partial=args.lead_hours is not None)
    all_truth['total_precipitation_24hr'] = accumulate_24h(truth_for_sum,allow_partial=args.lead_hours is not None)
    variables = {**p['variables'], **precipitation['variables'],
        'total_precipitation_24hr': {'pressure_levels':False},
        'wind_speed': {'pressure_levels':True}, '10m_wind_speed': {'pressure_levels':False}, 'geopotential_300_minus_500':{'pressure_levels':False}}
    pooled_fields = {'2m_temperature','mean_sea_level_pressure','total_precipitation_12hr',
                     '10m_wind_speed','temperature','geopotential','specific_humidity','wind_speed'}
    for lead in score_leads:
        truth_digest=hashlib.sha256()
        for name, spec in variables.items():
            if args.pooling_only and name not in pooled_fields: continue
            if name=='total_precipitation_24hr' and lead<24: continue
            for level in p['pressure_levels_hpa'] if spec['pressure_levels'] else [None]:
                selection = dict(case=np.datetime64(date), lead=np.timedelta64(lead,'h'))
                if level is not None: selection['level'] = level
                prediction = all_prediction[name].sel(selection)
                # Any number >=2 of identical members has exactly the same
                # empirical CRPS, mean, zero spread, pooling and event probabilities.
                if args.baseline: prediction = prediction.isel(member=slice(0,2))
                prediction = prediction.load()
                target = all_truth[name].sel(selection).load()
                truth_digest.update(json.dumps([date,lead,name,level]).encode())
                truth_digest.update(np.asarray(target.transpose('lat','lon'), dtype='<f8').tobytes())
                if not args.pooling_only:
                    stats = panel(prediction, target, minimum_valid_area_fraction=p['minimum_valid_area_fraction'])
                    # Keep both definitions explicit; do not silently call a ddof=1 spread biased.
                    stats['ensemble_variance_ddof1'] = stats.pop('ensemble_variance')
                    stats['ensemble_variance_ddof0'] = stats['ensemble_variance_ddof1']*(p['members']-1)/p['members']
                    rows.append(dict(case=date, variable=name, level_hpa=level, lead_hours=lead, **stats))
                if rain_climatology is not None and name in ('total_precipitation_12hr','total_precipitation_24hr'):
                    hours=12 if name=='total_precipitation_12hr' else 24
                    area=cell_weights(target.lat).broadcast_like(target).transpose('lat','lon').values
                    for dry in (.1,.25):
                        clim=rain_climatology.get(np.datetime64(date)+np.timedelta64(lead,'h'),hours,dry)
                        xr.align(target,clim.dry_fraction,clim.wet_threshold,join='exact',copy=False)
                        result=seeps_score(prediction.mean('member').transpose('lat','lon').values,
                            target.transpose('lat','lon').values,clim.wet_threshold.transpose('lat','lon').values,
                            clim.dry_fraction.transpose('lat','lon').values,area,dry)
                        seeps_rows.append(dict(case=date,variable=name,lead_hours=lead,accumulation_hours=hours,**result))
                if thresholds is not None and level is None:
                    area=cell_weights(target.lat).broadcast_like(target).transpose('lat','lon').values
                    rev_rows.extend(panel_rows(prediction,target,area,thresholds,date,lead,name))
                if args.pooling and name in pooled_fields:
                    stride = 4 if level is not None else 1
                    values = prediction.transpose('member','lat','lon').values[:,::stride,::stride]
                    observed = target.transpose('lat','lon').values[::stride,::stride]
                    for k in range(2,8 if stride==1 else 7):
                        matrix, area, diameter = geometries[stride,k]
                        for method,(pooled_prediction,pooled_truth) in apply_pool(values,observed,matrix).items():
                            pooled_rows.append(dict(case=date,variable=name,level_hpa=level,lead_hours=lead,
                                pooling=method,mesh_refinement=k,diameter_km=diameter,
                                **empirical_scores(pooled_prediction,pooled_truth,area)))
        truth_lead_hashes[str(lead)]=truth_digest.hexdigest()
        print(json.dumps(dict(case=index, lead=lead, status='scored')), flush=True)
    report = dict(status='passed', scope='marginal, derived and enabled precipitation/REV/pooling panels; not the full paper suite',
        baseline=args.baseline, baseline_reference_scorer_sha256=sha256(args.reference_scorer) if args.reference_scorer else None,
        protocol_sha256=sha256(here/'weather-protocol.json'),
        precipitation_protocol_sha256=sha256(here/'precipitation-protocol.json'),
        evaluator_sha256=sha256(Path(__file__)), panel_evaluator_sha256=sha256(here.parent/'wn2/evaluate.py'),
        weatherbenchx_revision=revision, estimator='empirical CRPS fair=False; ensemble-mean MSE',
        truth_values_sha256=canonical_truth_sha256(truth_lead_hashes), truth_lead_sha256=truth_lead_hashes,
        truth_digest_definition='SHA256 of sorted compact JSON mapping lead hours to per-lead ordered truth digests',
        prediction_manifests_sha256=records,
        rev_per_case=rev_rows, rev_evaluator_sha256=sha256(here/'rev.py'),
        rev_climatology_source_sha256=thresholds.source_sha256 if thresholds else None,
        rev_threshold_receipts=thresholds.receipts if thresholds else {},
        seeps_per_case=seeps_rows,seeps_evaluator_sha256=sha256(here/'seeps.py'),
        seeps_protocol_sha256=rain_climatology.protocol_sha256 if rain_climatology else None,
        seeps_climatology_source_sha256=rain_climatology.source_sha256 if rain_climatology else None,
        seeps_climatology_receipts=rain_climatology.receipts if rain_climatology else {},
        lineage=lineage, members=p['members'], lead_hours=score_leads, per_case=rows, pooled_per_case=pooled_rows,
        precipitation_evaluator_sha256=sha256(here/'precipitation.py'),
        pooling_geometry_sha256=geometry_hashes, pooling_evaluator_sha256=sha256(here/'spherical_pool.py'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    temporary.replace(args.output)
    for ds in shards+[truth,rain_truth]: ds.close()


if __name__ == '__main__': main()
