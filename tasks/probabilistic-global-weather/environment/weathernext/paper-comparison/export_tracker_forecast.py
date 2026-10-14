#!/usr/bin/env python3
"""Export a verified member plus shared HRES history for the frozen TE workflow."""
import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr
from prepare_tracker import FIELDS, fields, sha256


def export(args):
    here=Path(__file__).resolve().parent
    protocol=here/'weather-protocol.json'
    p=json.loads(protocol.read_text())
    if not 0<=args.member<p['members']:raise ValueError('Member outside protocol')
    if not 0<=args.case<len(p['evaluation_initializations']):raise ValueError('Case outside protocol')
    init=np.datetime64(p['evaluation_initializations'][args.case],'ns')
    stem=f'case-{args.case:03d}-member-{args.member:03d}'
    manifest=args.predictions/f'{stem}.complete.json'
    record=json.loads(manifest.read_text())
    if (record['status']!='passed' or record['protocol_sha256']!=sha256(protocol)
            or record['member']!=args.member or np.datetime64(record['initialization'])!=init):
        raise ValueError('Incomplete or mismatched prediction')
    history=args.history/f'case-{args.case:03d}.nc'
    history_receipt=history.with_suffix('.json')
    observed=json.loads(history_receipt.read_text())
    if (observed['status']!='passed' or observed['protocol_sha256']!=sha256(protocol)
            or observed['sha256']!=sha256(history) or np.datetime64(observed['initialization'])!=init):
        raise ValueError('Unverified shared history')
    identity=dict(protocol_sha256=sha256(protocol),prediction_manifest_sha256=sha256(manifest),
        history_receipt_sha256=sha256(history_receipt),exporter_sha256=sha256(Path(__file__)),
        field_transform_sha256=sha256(here/'prepare_tracker.py'),case=args.case,member=args.member,
        initialization=str(init))
    receipt=args.output.with_suffix('.json')
    if receipt.exists():
        previous=json.loads(receipt.read_text())
        if previous['identity']!=identity or previous['sha256']!=sha256(args.output):
            raise ValueError('Changed tracker export')
        return
    directory=args.predictions/f'{stem}.zarr'
    for name,digest in record['files'].items():
        file=(directory/name).resolve()
        if not file.is_relative_to(directory.resolve()) or sha256(file)!=digest:
            raise ValueError('Corrupt forecast inventory')
    with xr.open_zarr(directory,consolidated=True,chunks=None,decode_timedelta=True) as ds:
        np.testing.assert_array_equal(ds.member,[args.member])
        np.testing.assert_array_equal(ds.case,[init])
        np.testing.assert_array_equal(ds.lead,np.array(p['lead_hours'],dtype='timedelta64[h]'))
        forecast=ds[FIELDS].isel(case=0,member=0,drop=True).sel(level=[300,500]).load()
    forecast=forecast.drop_vars('valid_time',errors='ignore').rename(lead='time')
    forecast=forecast.assign_coords(time=init+forecast.time.values)
    with xr.open_dataset(history) as ds:
        prefix=ds[['msl','thickness','wind']].sel(time=slice(None,init)).load()
        zs=ds.zs.load()
    values=xr.concat([prefix,fields(forecast)],dim='time',join='exact')
    values['zs']=zs
    np.testing.assert_array_equal(values.time,init+np.arange(-240,361,12).astype('timedelta64[h]'))
    np.testing.assert_array_equal(values.lat,np.linspace(-90,90,721))
    np.testing.assert_array_equal(values.lon,np.arange(1440)/4)
    for name in values:
        if not np.isfinite(values[name]).all():raise ValueError(f'Nonfinite tracker field {name}')
    values=values.astype('float32')
    values.lat.attrs['units']='degrees_north';values.lon.attrs['units']='degrees_east'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix('.part')
    values.to_netcdf(temporary,format='NETCDF3_64BIT');temporary.replace(args.output)
    temporary=receipt.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(status='passed',identity=identity,sha256=sha256(args.output)))+'\n')
    temporary.replace(receipt)
    print(json.dumps(dict(status='passed',output=str(args.output))),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--predictions',required=True,type=Path)
    ap.add_argument('--history',required=True,type=Path)
    ap.add_argument('--case',required=True,type=int);ap.add_argument('--member',required=True,type=int)
    ap.add_argument('--output',required=True,type=Path)
    export(ap.parse_args())


if __name__=='__main__':main()
