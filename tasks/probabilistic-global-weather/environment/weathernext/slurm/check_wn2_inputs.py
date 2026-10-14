#!/usr/bin/env python3
"""Hash and validate prepared context files against the pinned native WN2 task."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import xarray as xr
from weathernext.utils import data_utils, fiddle_config_io


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    protocol=json.loads(args.protocol.read_text());protocol_hash=sha256(args.protocol)
    revision=subprocess.check_output(['git','-C',str(Path(fiddle_config_io.__file__).resolve().parents[2]),'rev-parse','HEAD'],text=True).strip()
    if revision!='89c4b2a77a1c57b328b909c575550fd2e5aadc9c':raise ValueError('Wrong native source revision')
    task=fiddle_config_io.get_fiddle_config_by_name('weathernext2/configs/WeatherNext2').task
    records=[]
    for index,stamp in enumerate(protocol['evaluation_initializations']):
        record=json.loads((args.inputs/f'inputs-{index:03d}.json').read_text())
        path=args.inputs/f'inputs-{index:03d}.nc'
        if record['status']!='passed' or record['protocol_sha256']!=protocol_hash or record['sha256']!=sha256(path):raise ValueError('Input record/hash mismatch')
        with xr.open_dataset(path) as ds:
            np.testing.assert_array_equal(ds.datetime.values,np.datetime64(stamp,'ns')+np.asarray([[-6,0]],dtype='timedelta64[h]'))
            np.testing.assert_array_equal(ds.time.values,np.asarray([-6,0],dtype='timedelta64[h]'))
            np.testing.assert_array_equal(ds.level.values,task.pressure_levels)
            np.testing.assert_array_equal(ds.lat.values,np.linspace(-90,90,721))
            np.testing.assert_array_equal(ds.lon.values,np.arange(1440)/4)
            if ds.sizes['batch']!=1:raise ValueError('Wrong batch size')
            data_utils.add_derived_vars(ds)
            missing=set(task.input_variables)-set(ds.data_vars)
            if missing:raise ValueError(f'Missing native inputs: {missing}')
            if ds.sizes['time']!=2:raise ValueError('Future weather included in inputs')
        records.append(dict(case_index=index,initialization=stamp,file=path.name,sha256=record['sha256'],isolated_input_repairs=record.get('isolated_input_repairs',{})))
    report=dict(status='passed',source_commit=revision,protocol_sha256=protocol_hash,cases=records,
                native_input_variables=list(task.input_variables),scope='Native context schema, time coverage and hashes; not model forecast replay')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('WN2_NATIVE_EVALUATION_INPUTS_OK')


if __name__=='__main__':main()
