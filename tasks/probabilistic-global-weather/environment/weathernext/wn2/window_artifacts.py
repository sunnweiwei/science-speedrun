"""Retain only cross-window precipitation and cyclone sufficient fields."""
import json
from pathlib import Path
import sys
import numpy as np
import xarray as xr
from training_data import sha256
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'paper-comparison'))
from prepare_tracker import FIELDS, fields


def capture(predictions, buffer, index, leads, protocol):
    buffer.mkdir(parents=True,exist_ok=True)
    rains=[]
    for member in range(protocol['members']):
        stem=f'case-{index:03d}-member-{member:03d}'
        manifest=predictions/f'{stem}.complete.json'
        record=json.loads(manifest.read_text())
        with xr.open_zarr(predictions/f'{stem}.zarr',chunks=None,consolidated=True,decode_timedelta=True) as ds:
            np.testing.assert_array_equal(ds.lead,np.array(leads,dtype='timedelta64[h]'))
            # Keep the float32 transform identical to export_tracker_forecast.
            forecast=ds[FIELDS].isel(case=0,member=0,drop=True).sel(level=[300,500]).load()
            values=fields(forecast).drop_vars('valid_time',errors='ignore').rename(lead='time')
            init=np.datetime64(protocol['evaluation_initializations'][index],'ns')
            values=values.assign_coords(time=init+values.time.values)
            # A one-lead initial window otherwise infers days; appending 12h
            # values to that integer encoding silently corrupts timestamps.
            values.time.encoding=dict(units='hours since 1970-01-01 00:00:00',calendar='proleptic_gregorian',dtype='int64')
            if any(not np.isfinite(values[n]).all() for n in values):raise ValueError('Invalid tracker projection')
            path=buffer/f'{stem}.zarr'
            if path.exists():
                with xr.open_zarr(path,chunks=None,consolidated=True) as previous:
                    if previous.time.values[-1]+np.timedelta64(12,'h')!=values.time.values[0]:
                        raise ValueError('Missing tracker window')
                values.to_zarr(path,mode='a',append_dim='time',consolidated=True)
            else:
                if leads[0]!=12:raise ValueError('Missing initial tracker window')
                values.to_zarr(path,mode='w-',consolidated=True)
            with xr.open_zarr(path,chunks=None,consolidated=True) as stored:
                expected=init+np.array([h for h in protocol['lead_hours'] if h<=leads[-1]],dtype='timedelta64[h]')
                np.testing.assert_array_equal(stored.time,expected)
            rain=ds[['total_precipitation_12hr']].isel(lead=[-1]).load().drop_vars('valid_time',errors='ignore')
            rains.append(rain)
        receipt=buffer/f'{stem}.json'
        prior=json.loads(receipt.read_text()) if receipt.exists() else dict(identity=record,windows=[])
        identity_keys=('protocol_sha256','package_sha256','runner_sha256','seed','member','initialization','context_sha256')
        if any(prior['identity'][k]!=record[k] for k in identity_keys):raise ValueError('Changed window lineage')
        prior['windows'].append(dict(lead_hours=leads,manifest_sha256=sha256(manifest)))
        receipt.write_text(json.dumps(prior)+'\n')
    rain=xr.concat(rains,dim='member',join='exact',data_vars='minimal',coords='minimal',compat='equals')
    # Scoring already consumed the preceding interval before this replacement.
    import shutil
    rain_path=buffer/'previous-rain.zarr'
    if rain_path.exists():shutil.rmtree(rain_path)
    rain.to_zarr(rain_path,mode='w-',consolidated=True)


def export_member(buffer, predictions, history_root, directory, index, member, protocol, protocol_hash):
    stem=f'case-{index:03d}-member-{member:03d}'
    record=json.loads((buffer/f'{stem}.json').read_text())
    if [h for window in record['windows'] for h in window['lead_hours']]!=protocol['lead_hours']:
        raise ValueError('Incomplete tracker trajectory')
    combined=dict(record['identity'],lead_hours=protocol['lead_hours'],files={},
                  window_manifests=record['windows'],storage='scored-and-released windows')
    predictions.mkdir(parents=True,exist_ok=True)
    manifest=predictions/f'{stem}.complete.json';manifest.write_text(json.dumps(combined)+'\n')
    history=history_root/f'case-{index:03d}.nc';history_receipt=history.with_suffix('.json')
    observed=json.loads(history_receipt.read_text())
    init=np.datetime64(protocol['evaluation_initializations'][index],'ns')
    if (observed['status']!='passed' or observed['protocol_sha256']!=protocol_hash
            or observed['sha256']!=sha256(history) or np.datetime64(observed['initialization'])!=init):
        raise ValueError('Invalid tracker history')
    with xr.open_dataset(history) as ds:
        prefix=ds[['msl','thickness','wind']].sel(time=slice(None,init)).load()
        zs=ds.zs.load()
    with xr.open_zarr(buffer/f'{stem}.zarr',chunks=None,consolidated=True) as ds:
        values=xr.concat([prefix,ds.load()],dim='time',join='exact')
    values['zs']=zs
    np.testing.assert_array_equal(values.time,init+np.arange(-240,361,12).astype('timedelta64[h]'))
    if any(not np.isfinite(values[n]).all() for n in values):raise ValueError('Invalid tracker input')
    values=values.astype('float32');values.lat.attrs['units']='degrees_north';values.lon.attrs['units']='degrees_east'
    directory.mkdir(parents=True,exist_ok=True)
    output=directory/'input.nc';values.to_netcdf(output,format='NETCDF3_64BIT')
    identity=dict(protocol_sha256=protocol_hash,prediction_manifest_sha256=sha256(manifest),
        history_receipt_sha256=sha256(history_receipt),exporter_sha256=sha256(Path(__file__)),
        field_transform_sha256=sha256(Path(__file__).resolve().parents[1]/'paper-comparison/prepare_tracker.py'),
        case=index,member=member,initialization=str(init))
    (directory/'input.json').write_text(json.dumps(dict(status='passed',identity=identity,sha256=sha256(output)))+'\n')
