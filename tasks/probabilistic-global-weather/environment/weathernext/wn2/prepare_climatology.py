#!/usr/bin/env python3
"""Training-only seasonal/hour mean baseline on the native grid."""
import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from training_data import read_frame,sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('root','protocol','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--group',type=int,required=True,choices=range(16))
    args=parser.parse_args();protocol=json.loads(args.protocol.read_text())
    quarter,hour=divmod(args.group,4);hour*=6
    selected=[i for i,t in enumerate(protocol['initializations']) if (int(t[5:7])-1)//3==quarter]
    total=None;squared=None;count=None;evidence=[]
    for index in selected:
        init=np.datetime64(protocol['initializations'][index])
        offsets=[h for h in protocol['offset_hours'] if h>=0 and int(str(init+np.timedelta64(h,'h'))[11:13])==hour]
        # One earliest matching synoptic hour per date gives all fixed dates
        # equal weight without treating neighboring frames as independent days.
        offset=min(offsets);frame=protocol['offset_hours'].index(offset)
        directory=args.root/f'case-{index:03d}'
        path=directory/f'hres_fc0-{frame:02d}.nc'
        ds=read_frame(path,1).drop_vars('time')
        precip_path=directory/f'era5-{frame:02d}.nc'
        with xr.open_dataset(precip_path) as era:
            ds['total_precipitation_6hr']=era.total_precipitation_6hr.drop_vars('time').load()
        valid=np.isfinite(ds);values=ds.where(valid,0).astype('float64')
        total=values if total is None else total+values
        squared=values*values if squared is None else squared+values*values
        count=valid.astype('int32') if count is None else count+valid
        evidence.append(dict(case_index=index,frame=frame,weather_sha256=sha256(path),precipitation_source_sha256=sha256(precip_path)))
    if any((v==0).any() for v in count.data_vars.values()):raise ValueError('Climatology contains an unobserved cell')
    result=(total/count).astype('float32')
    if any((v<2).any() for v in count.data_vars.values()):raise ValueError('Too few seasonal observations for variance')
    std=np.sqrt(((squared-total*total/count)/(count-1)).clip(min=0)).astype('float32')
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/f'group-{args.group:02d}.nc'
    result.to_netcdf(path,encoding={n:dict(zlib=True,complevel=1) for n in result.data_vars})
    std_path=args.output/f'group-{args.group:02d}-std.nc'
    std.to_netcdf(std_path,encoding={n:dict(zlib=True,complevel=1) for n in std.data_vars})
    record=dict(status='passed',quarter=quarter+1,hour=hour,cases=len(selected),source_cases=evidence,
                protocol_sha256=sha256(args.protocol),runner_sha256=sha256(Path(__file__)),file=path.name,sha256=sha256(path),
                std_file=std_path.name,std_sha256=sha256(std_path),variance_definition='Pointwise sample variance over equally weighted training dates, ddof=1')
    path.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
