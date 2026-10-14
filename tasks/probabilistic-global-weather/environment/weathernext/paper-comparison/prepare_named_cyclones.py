#!/usr/bin/env python3
"""Freeze exact-time main named IBTrACS tracks for position verification."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import xarray as xr
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'wn2'))
from training_data import sha256

IBTRACS_SHA256='5584637db168ddd65e115114888ae2bea9587d0439954c2bde776703525005b5'


def canonical_times(iso_times,numeric_times):
    # Numeric CF days have sub-millisecond floating-point decoding noise. The
    # source's ISO strings preserve the intended second, without interpolation.
    result=np.full(len(iso_times),np.datetime64('NaT','ns'))
    for i,value in enumerate(iso_times):
        text=value.decode().strip()
        if not text:continue
        result[i]=np.datetime64(text.replace(' ','T'),'ns')
        if np.isnat(numeric_times[i]) or abs(result[i]-numeric_times[i])>np.timedelta64(1,'ms'):
            raise ValueError('Inconsistent IBTrACS time representations')
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ibtracs',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args()
    if sha256(args.ibtracs)!=IBTRACS_SHA256:raise ValueError('IBTrACS differs from frozen corpus')
    protocol=Path(__file__).parent/'weather-protocol.json'
    p=json.loads(protocol.read_text())
    with xr.open_dataset(args.ibtracs) as raw:
        # Southern-hemisphere seasons can straddle calendar years.
        data=raw[['sid','name','track_type','time','iso_time','lat','lon']].isel(storm=np.flatnonzero((raw.season>=2021)&(raw.season<=2023))).load()
    exact_times=[canonical_times(data.iso_time.values[i],data.time.values[i]) for i in range(data.sizes['storm'])]
    cases=[]
    for date in p['evaluation_initializations']:
        init=np.datetime64(date,'ns');expected=init+np.arange(0,217,12).astype('timedelta64[h]')
        storms={}
        for i in range(data.sizes['storm']):
            name=data.name.values[i].decode().strip();kind=data.track_type.values[i].decode().strip()
            if kind!='main' or name.upper() in {'','UNNAMED','NOT_NAMED','NONAME'}:continue
            times=exact_times[i]
            selected=np.flatnonzero(np.isin(times,expected))
            points={}
            for j in selected:
                lat=float(data.lat.values[i,j]);lon=float(data.lon.values[i,j])
                if not np.isfinite([lat,lon]).all():continue
                if not -90<=lat<=90:raise ValueError('Invalid IBTrACS latitude')
                stamp=str(times[j].astype('datetime64[ns]'))
                if stamp in points:raise ValueError('Duplicate IBTrACS time')
                points[stamp]=[lat,lon%360]
            if str(init) not in points:continue
            sid=data.sid.values[i].decode().strip()
            if sid in storms:raise ValueError('Duplicate main track')
            storms[sid]=dict(name=name,points=points)
        cases.append(dict(initialization=str(init),storms=storms))
    report=dict(status='passed',ibtracs_sha256=IBTRACS_SHA256,protocol_sha256=sha256(protocol),
        preparer_sha256=sha256(Path(__file__)),selection='main named tracks present at initialization; exact source ISO timestamps, cross-checked against numeric CF times within 1 ms; no interpolation',cases=cases)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temp=args.output.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(args.output)
    print(json.dumps(dict(status='passed',storms_per_case=[len(c['storms']) for c in cases])),flush=True)


if __name__=='__main__':main()
