#!/usr/bin/env python3
"""Validate the complete paper-period monthly climatology before scoring."""
import argparse
import calendar
import hashlib
import json
from pathlib import Path

import numpy as np
from seeps import Climatology


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--climatology',required=True,type=Path)
    args=ap.parse_args();clim=Climatology(args.climatology);rows=[]
    for month in range(1,13):
        record=json.loads((args.climatology/f'month-{month:02d}.json').read_text())
        samples=sum(calendar.monthrange(year,month)[1]*4 for year in range(1980,2016))
        if record['samples_per_grid_point']!=samples:raise ValueError('Wrong monthly climatology sample count')
        for hours in (12,24):
            earlier=None
            for dry in (.1,.25):
                ds=clim.get(f'2022-{month:02d}-01',hours,dry)
                p1=ds.dry_fraction.values;wet=ds.wet_threshold.values;valid=(p1>=.1)&(p1<=.85)
                if not np.isfinite(wet[valid]).all() or np.any(wet[valid]<dry/1000):raise ValueError('Invalid eligible wet threshold')
                if earlier is not None:
                    previous_p1,previous_wet=earlier
                    if np.any(p1<previous_p1):raise ValueError('Nonmonotonic dry fraction')
                    paired=np.isfinite(wet)&np.isfinite(previous_wet)
                    if np.any(wet[paired]+1e-14<previous_wet[paired]):raise ValueError('Nonmonotonic conditional quantile')
                earlier=(p1,wet)
                rows.append(dict(month=month,accumulation_hours=hours,dry_threshold_mm=dry,
                    samples_per_grid_point=samples,valid_grid_points=int(valid.sum())))
    report=dict(status='passed',protocol_sha256=clim.protocol_sha256,source_lock_sha256=clim.source_sha256,
        month_receipts_sha256=clim.receipts,verification_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),combinations=rows)
    path=args.climatology/'ready.json';temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(path)
    print(json.dumps(dict(status='passed',months=12,combinations=len(rows))),flush=True)


if __name__=='__main__':main()
