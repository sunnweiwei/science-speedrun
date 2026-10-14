#!/usr/bin/env python3
"""Check full-horizon tracker projection against immutable historical forecasts."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import xarray as xr
from check_windowed_scoring import slice_store
from window_artifacts import capture, export_member
from training_data import sha256


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--historical',type=Path,required=True);ap.add_argument('--benchmark',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--image',type=Path,required=True)
    ap.add_argument('--members',type=int,default=2)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    here=Path(__file__).resolve().parent;protocol_path=here/'protocol.json';p=json.loads(protocol_path.read_text())
    p['members']=a.members;buffer=a.output/'buffer';case=0
    for first,stop in [(0,1),(1,8),(8,30)]:
        pred=a.output/f'window-{first:03d}';pred.mkdir()
        for member in range(p['members']):
            stem=f'case-000-member-{member:03d}'
            slice_store(a.historical/'predictions'/f'{stem}.zarr',pred/f'{stem}.zarr',first,stop)
            record=json.loads((a.historical/'predictions'/f'{stem}.complete.json').read_text())
            record['lead_hours']=p['lead_hours'][first:stop]
            (pred/f'{stem}.complete.json').write_text(json.dumps(record)+'\n')
        capture(pred,buffer,case,p['lead_hours'][first:stop],p)
        shutil.rmtree(pred)
    results=[]
    for member in range(p['members']):
        stem=f'case-000-member-{member:03d}';dest=a.output/'tracks'/stem
        export_member(buffer,a.output/'predictions',a.benchmark/'tracker/truth',dest,case,member,p,sha256(protocol_path))
        original=a.historical/'tracks'/stem
        with xr.open_dataset(original/'input.nc') as expected,xr.open_dataset(dest/'input.nc') as actual:
            xr.testing.assert_identical(actual,expected)
            for name in expected.data_vars:
                assert expected[name].values.tobytes()==actual[name].values.tobytes()
        command=['apptainer','exec','--bind','/usr/project:/usr/project',str(a.image),'/opt/wn2-eval/bin/python',
            str(here.parent/'paper-comparison/run_tracker.py'),'--bin','/opt/tempestextremes/bin',
            '--input',str(dest/'input.nc'),'--initialization',p['evaluation_initializations'][0],
            '--output',str(dest/'tracking')]
        with (dest/'execute.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        for name in ('candidates.txt','tracks.csv'):
            assert sha256(dest/'tracking'/name)==sha256(original/'tracking'/name),name
        results.append(dict(member=member,input_arrays_bitwise_equal=True,detected_candidates_equal=True,tracks_equal=True))
        print(json.dumps(results[-1]),flush=True)
        (dest/'input.nc').unlink();shutil.rmtree(buffer/f'{stem}.zarr')
    (a.output/'result.json').write_text(json.dumps(dict(status='passed',members=p['members'],lead_hours=p['lead_hours'],
        native_grid=p['grid'],training_rerun=False,windows=[1,7,22],comparisons=results),indent=2)+'\n')


if __name__=='__main__':main()
