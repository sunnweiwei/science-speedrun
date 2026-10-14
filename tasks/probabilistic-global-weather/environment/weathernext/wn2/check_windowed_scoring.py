#!/usr/bin/env python3
"""Read-only historical native-grid equivalence check for window scorers.

Hard links share immutable forecast chunks; generated metadata is separate.
No original prediction, weight, receipt or training artifact is modified.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import xarray as xr
import zarr
from training_data import sha256
from window_artifacts import capture


def slice_store(source, destination, first, stop):
    old=zarr.open_group(str(source),mode='r');new=zarr.open_group(str(destination),mode='w-')
    new.attrs.update(dict(old.attrs))
    for name,array in old.arrays():
        dims=array.attrs['_ARRAY_DIMENSIONS'];shape=list(array.shape)
        if 'lead' in dims:shape[dims.index('lead')]=stop-first
        target=new.create_dataset(name,shape=shape,chunks=array.chunks,dtype=array.dtype,
            compressor=array.compressor,fill_value=array.fill_value,filters=array.filters,order=array.order)
        target.attrs.update(dict(array.attrs))
        if 'lat' not in dims or 'lon' not in dims:
            selection=[slice(None)]*array.ndim
            if 'lead' in dims:selection[dims.index('lead')]=slice(first,stop)
            target[:]=array[tuple(selection)]
            continue
        lead_axis=dims.index('lead')
        if array.chunks[lead_axis]!=1:raise ValueError('Historical check requires one lead per chunk')
        for file in (source/name).iterdir():
            if file.name.startswith('.'):continue
            indices=list(map(int,file.name.split('.')))
            if first<=indices[lead_axis]<stop:
                indices[lead_axis]-=first
                os.link(file,destination/name/'.'.join(map(str,indices)))
    zarr.consolidate_metadata(str(destination))


def equivalent(actual, expected):
    if isinstance(expected,dict):
        if actual.keys()!=expected.keys():raise AssertionError((actual.keys(),expected.keys()))
        for key in expected:equivalent(actual[key],expected[key])
    elif isinstance(expected,list):
        assert len(actual)==len(expected)
        for a,b in zip(actual,expected):equivalent(a,b)
    elif isinstance(expected,float):np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12,equal_nan=True)
    else:assert actual==expected,(actual,expected)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--historical',required=True,type=Path);ap.add_argument('--benchmark',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path);ap.add_argument('--spectra-image',type=Path)
    ap.add_argument('--case',type=int,default=0);ap.add_argument('--leads',type=int,nargs='+',default=[12,24])
    ap.add_argument('--resume',action='store_true',help='Reuse historical slices, rerun scoring')
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=a.resume)
    here=Path(__file__).resolve().parent;paper=here.parent/'paper-comparison'
    p=json.loads((here/'protocol.json').read_text());buffer=a.output/'buffer';results=[]
    for lead in a.leads:
        first=p['lead_hours'].index(lead);work=a.output/f'lead-{lead:03d}';pred=work/'predictions';pred.mkdir(parents=True,exist_ok=a.resume)
        for member in range(p['members']):
            stem=f'case-{a.case:03d}-member-{member:03d}'
            if a.resume and (pred/f'{stem}.complete.json').exists():continue
            slice_store(a.historical/'predictions'/f'{stem}.zarr',pred/f'{stem}.zarr',first,first+1)
            record=json.loads((a.historical/'predictions'/f'{stem}.complete.json').read_text())
            record.update(lead_hours=[lead],files={str(f.relative_to(pred/f'{stem}.zarr')):sha256(f) for f in sorted((pred/f'{stem}.zarr').rglob('*')) if f.is_file()})
            (pred/f'{stem}.complete.json').write_text(json.dumps(record)+'\n')
        common=[sys.executable,str(paper/'score_marginals.py'),'--predictions',str(pred),'--data',str(a.benchmark/'data'),
                '--case',str(a.case),'--lead-hours',str(lead)]
        if lead>12:common+=['--previous-rain',str(buffer/'previous-rain.zarr')]
        commands={'scores':common+['--rev-climatology',str(a.benchmark/'rev-climatology'),'--seeps-climatology',str(a.benchmark/'seeps/climatology')],
                  'pooling-scores':common+['--pooling',str(a.benchmark/'pooling'),'--pooling-only']}
        if a.spectra_image:
            commands['spectra']=['apptainer','exec','--bind','/usr/project:/usr/project',str(a.spectra_image),'/opt/weather-python/bin/python',
                str(paper/'score_spectra.py'),'--predictions',str(pred),'--truth',str(a.benchmark/'spectra/truth'/f'case-{a.case:03d}.json'),
                '--case',str(a.case),'--lead-hours',str(lead)]
        for endpoint,command in commands.items():
            path=work/f'{endpoint}.json'
            with (work/f'{endpoint}.log').open('w') as log:subprocess.run(command+['--output',str(path)],stdout=log,stderr=subprocess.STDOUT,check=True)
            actual=json.loads(path.read_text());old=json.loads((a.historical/endpoint/f'case-{a.case:03d}.json').read_text())
            for name in ('per_case','pooled_per_case','rev_per_case','seeps_per_case','rows'):
                if name in old:
                    expected=[row for row in old[name] if row['lead_hours']==lead]
                    equivalent(actual[name],expected)
                    results.append(dict(endpoint=endpoint,lead=lead,rows=name,count=len(expected),matched=True))
            print(json.dumps(results[-1]),flush=True)
        capture(pred,buffer,a.case,[lead],p)
        import shutil
        shutil.rmtree(pred)
    (a.output/'result.json').write_text(json.dumps(dict(status='passed',case=a.case,members=p['members'],leads=a.leads,
        native_grid=p['grid'],training_rerun=False,comparisons=results,scope='Historical arrays re-exposed as windows; actual scorers; no model inference'),indent=2)+'\n')


if __name__=='__main__':main()
