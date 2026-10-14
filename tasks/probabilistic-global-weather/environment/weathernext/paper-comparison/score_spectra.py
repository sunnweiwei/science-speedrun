#!/usr/bin/env python3
"""Compute member-mean and ensemble-mean-field spectra for a complete case."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import xarray as xr
from spherical_spectra import Spectrum
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'wn2'))
from window_scores import checked_leads


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--predictions',required=True,type=Path);ap.add_argument('--truth',required=True,type=Path)
    ap.add_argument('--case',required=True,type=int);ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--lead-hours',type=int,nargs='+')
    ap.add_argument('--score-leads',type=int,nargs='+',help='Consecutive part of --lead-hours to score')
    args=ap.parse_args();here=Path(__file__).resolve().parent
    protocol=here/'weather-protocol.json';p=json.loads(protocol.read_text());spec_path=here/'spectra-protocol.json'
    spec=json.loads(spec_path.read_text())
    if not 0<=args.case<len(p['evaluation_initializations']):raise ValueError('Case outside protocol')
    leads=checked_leads(args.lead_hours,p['lead_hours'])
    score_leads=leads if args.score_leads is None else checked_leads(args.score_leads,leads)
    date=p['evaluation_initializations'][args.case];truth=json.loads(args.truth.read_text())
    if (truth['status']!='passed' or truth['case']!=date or truth['identity']['protocol_sha256']!=digest(protocol)
            or truth['identity']['spectra_protocol_sha256']!=digest(spec_path)):
        raise ValueError('Wrong spectral reference')
    started=time.monotonic();shards=[];manifests=[];lineage=None
    for member in range(p['members']):
        stem=f'case-{args.case:03d}-member-{member:03d}';manifest=args.predictions/f'{stem}.complete.json'
        r=json.loads(manifest.read_text())
        if r['status']!='passed' or r['protocol_sha256']!=digest(protocol) or r['member']!=member or np.datetime64(r['initialization'])!=np.datetime64(date):
            raise ValueError('Invalid forecast identity')
        identity={k:r.get(k) for k in ('assets_lock_sha256','package_sha256','runner_sha256','noise_grid_cache_sha256','seed','context_sha256')}
        if lineage is not None and identity!=lineage:raise ValueError('Mixed forecast lineage')
        lineage=identity;directory=args.predictions/f'{stem}.zarr'
        for name,expected in r['files'].items():
            file=(directory/name).resolve()
            if not file.is_relative_to(directory.resolve()) or digest(file)!=expected:raise ValueError('Corrupt forecast inventory')
        ds=xr.open_zarr(directory,consolidated=True,chunks=None,decode_timedelta=True)
        np.testing.assert_array_equal(ds.case,[np.datetime64(date,'ns')]);np.testing.assert_array_equal(ds.member,[member])
        np.testing.assert_array_equal(ds.lead,np.array(leads,dtype='timedelta64[h]'))
        np.testing.assert_array_equal(ds.lat,np.linspace(-90,90,spec['latitude_nodes']))
        np.testing.assert_array_equal(ds.lon,np.arange(spec['longitude_nodes'])*360/spec['longitude_nodes'])
        shards.append(ds.isel(case=0,member=0,drop=True));manifests.append(digest(manifest))
    spectrum=Spectrum(shards[0].lat.values,shards[0].lon.values)
    if spectrum.provenance()!=truth['transform']:raise ValueError('Reference uses different transform implementation')
    truth_rows={(r['lead_hours'],r['variable'],r['level_hpa']):r for r in truth['rows']}
    expected_rows={(lead,name,level) for lead in p['lead_hours'] for name,levels in spec['variables'].items() for level in (levels or [None])}
    if len(truth_rows)!=len(truth['rows']) or truth_rows.keys()!=expected_rows:raise ValueError('Incomplete reference panels')
    rows=[]
    for lead in score_leads:
        for name,levels in spec['variables'].items():
            for level in levels or [None]:
                coefficient_sum=np.zeros(spectrum.mask.shape,dtype=np.float64)
                power_sum=np.zeros(spectrum.grid.total_wavenumbers,dtype=np.float64)
                for ds in shards:
                    selection=dict(lead=np.timedelta64(lead,'h'))
                    if level is not None:selection['level']=level
                    coefficients=spectrum.coefficients(ds[name].sel(selection).transpose('lat','lon').values)
                    coefficient_sum+=coefficients
                    power_sum+=np.square(coefficients).sum(axis=0)/(4*np.pi)
                mean_power=power_sum/p['members']
                mean_field_power=np.square(coefficient_sum/p['members']).sum(axis=0)/(4*np.pi)
                if not np.isfinite(mean_power).all() or not np.isfinite(mean_field_power).all():raise ValueError('Nonfinite model spectra')
                rows.append(dict(lead_hours=lead,variable=name,level_hpa=level,
                    member_mean_power=mean_power.tolist(),ensemble_mean_field_power=mean_field_power.tolist(),
                    truth_power=truth_rows[lead,name,level]['power'],
                    truth_status=truth_rows[lead,name,level]['status'],
                    missing_truth_grid_points=truth_rows[lead,name,level]['missing_grid_points']))
        print(json.dumps(dict(lead_hours=lead,elapsed_seconds=time.monotonic()-started)),flush=True)
    report=dict(status='passed',case=date,members=p['members'],lead_hours=score_leads,protocol_sha256=digest(protocol),
        spectra_protocol_sha256=digest(spec_path),evaluator_sha256=digest(Path(__file__)),transform=spectrum.provenance(),
        truth_receipt_sha256=digest(args.truth),truth_values_sha256=truth['truth_values_sha256'],
        lineage=lineage,prediction_manifests_sha256=manifests,total_wavenumber=truth['total_wavenumber'],rows=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temp=args.output.with_suffix('.tmp');temp.write_text(json.dumps(report,allow_nan=False)+'\n');temp.replace(args.output)
    for ds in shards:ds.close()


if __name__=='__main__':main()
