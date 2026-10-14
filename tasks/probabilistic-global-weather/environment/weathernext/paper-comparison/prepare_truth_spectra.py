#!/usr/bin/env python3
"""Compute the full-resolution HRES spectral reference for one frozen case."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import xarray as xr
from spherical_spectra import Spectrum


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',required=True,type=Path);ap.add_argument('--case',required=True,type=int)
    ap.add_argument('--output',required=True,type=Path);args=ap.parse_args()
    here=Path(__file__).resolve().parent;protocol=here/'weather-protocol.json'
    p=json.loads(protocol.read_text())
    if not 0<=args.case<len(p['evaluation_initializations']):raise ValueError('Case outside protocol')
    ready=args.data/'ready.json'
    if json.loads(ready.read_text())['weather_protocol_sha256']!=digest(protocol):raise ValueError('Wrong data protocol')
    spec=json.loads((here/'spectra-protocol.json').read_text())
    identity=dict(protocol_sha256=digest(protocol),spectra_protocol_sha256=digest(here/'spectra-protocol.json'),
        data_ready_sha256=digest(ready),truth_case_receipt_sha256=digest(args.data/'weather/cases'/f'{args.case:03d}.json'),
        runner_sha256=digest(Path(__file__)),spectrum_evaluator_sha256=digest(here/'spherical_spectra.py'))
    if args.output.exists():
        previous=json.loads(args.output.read_text())
        if previous['status']!='passed' or previous['identity']!=identity:raise ValueError('Changed completed truth spectra')
        return
    started=time.monotonic();rows=[];truth_digest=hashlib.sha256()
    with xr.open_zarr(args.data/'weather/truth.zarr',consolidated=True,chunks=None,decode_timedelta=True) as ds:
        ds=ds.isel(case=args.case)
        if np.datetime64(ds.case.values)!=np.datetime64(p['evaluation_initializations'][args.case]):raise ValueError('Wrong truth case coordinate')
        if ds.sizes['lat']!=spec['latitude_nodes'] or ds.sizes['lon']!=spec['longitude_nodes']:raise ValueError('Wrong truth grid')
        spectrum=Spectrum(ds.lat.values,ds.lon.values)
        print(json.dumps(dict(event='basis_ready',elapsed_seconds=time.monotonic()-started)),flush=True)
        for lead in p['lead_hours']:
            for name,levels in spec['variables'].items():
                for level in levels if levels else [None]:
                    selection=dict(lead=np.timedelta64(lead,'h'))
                    if level is not None:selection['level']=level
                    field=ds[name].sel(selection).transpose('lat','lon').values
                    truth_digest.update(json.dumps([lead,name,level]).encode())
                    truth_digest.update(np.asarray(field,dtype='<f8').tobytes())
                    missing=int((~np.isfinite(field)).sum())
                    if np.isinf(field).any():raise ValueError('Infinite truth field')
                    if missing:
                        rows.append(dict(lead_hours=lead,variable=name,level_hpa=level,power=None,
                            status='missing_truth',missing_grid_points=missing))
                        continue
                    power=spectrum.power(field)
                    if not np.isfinite(power).all() or np.any(power<0):raise ValueError('Invalid spectral power')
                    rows.append(dict(lead_hours=lead,variable=name,level_hpa=level,power=power.tolist(),status='passed',missing_grid_points=0))
            print(json.dumps(dict(lead_hours=lead,elapsed_seconds=time.monotonic()-started)),flush=True)
    report=dict(status='passed',case=p['evaluation_initializations'][args.case],identity=identity,
        transform=spectrum.provenance(),total_wavenumber=list(range(spectrum.grid.total_wavenumbers)),
        truth_values_sha256=truth_digest.hexdigest(),rows=rows,wall_seconds=time.monotonic()-started,
        scope='HRES reference spectra only, not model evaluation')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temp=args.output.with_suffix('.tmp');temp.write_text(json.dumps(report,allow_nan=False)+'\n');temp.replace(args.output)


if __name__=='__main__':main()
