#!/usr/bin/env python3
"""Paper subset inference from a verified final WN2 package, native 6-hour steps."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')


def claimed_members(output,case,members):
    """Allow later GPU allocations to join the same member queue."""
    for member in range(members):
        name=f'case-{case:03d}-member-{member:03d}'
        with (output/f'{name}.lock').open('w') as claim:
            try:fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if (output/f'{name}.complete.json').exists():continue
            yield member


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--package', required=True, type=Path)
    ap.add_argument('--inputs', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--worker', required=True, type=int)
    ap.add_argument('--workers', required=True, type=int)
    ap.add_argument('--seed', type=int, default=20220910)
    args = ap.parse_args()
    if not 0 <= args.worker < args.workers: raise ValueError('Invalid worker shard')
    # Import the packaged implementation that was matched to training, not a live checkout.
    sys.path.insert(0,str(args.package.resolve()))
    from wn2_validation import ensure_deterministic_process
    ensure_deterministic_process()
    import jax
    import numpy as np
    import xarray as xr
    from weathernext.utils import checkpoint
    from weathernext.weathernext2 import fgn
    from native_model import stage_config, transforms
    from plan_schedule import STAGES
    from training_data import sha256
    from forecast_store import initialize
    from predict import context, template, forcing, next_context

    here = Path(__file__).resolve().parent
    protocol = here/'weather-protocol.json'
    p = json.loads(protocol.read_text())
    rain = json.loads((here/'precipitation-protocol.json').read_text())
    manifest = json.loads((args.package/'package.json').read_text())
    if manifest['profile_artifact']: raise ValueError('Profiling artifact is not a final trained model')
    for name,digest in manifest['files'].items():
        path = (args.package/name).resolve()
        if not path.is_relative_to(args.package.resolve()) or sha256(path)!=digest:
            raise ValueError(f'Package integrity failure: {name}')
    report = json.loads((args.package/'training-report.json').read_text())
    if report['status']!='passed' or report['stage']['name']!='hres_025deg_ar8' or report.get('profile'):
        raise ValueError('Final curriculum has not completed')
    revision = subprocess.check_output(['git','-C',str(Path(fgn.__file__).resolve().parents[2]),'rev-parse','HEAD'],text=True).strip()
    if revision != manifest['source_revision']: raise ValueError('Runtime source mismatch')
    config = stage_config(STAGES[-1],args.package/'statistics')
    forward,_ = transforms(config)
    with (args.package/'weights.npz').open('rb') as stream:
        params = checkpoint.load(stream,fgn.CheckPoint).params
    if sum(x.size for x in jax.tree.leaves(params))!=183785575: raise ValueError('Wrong WN2 capacity')
    params = jax.device_put(params)
    apply = jax.jit(forward.apply)
    variables = {**p['variables'],**rain['variables']}
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/f'worker-{args.worker:03d}.lock').open('w') as worker_lock:
        fcntl.flock(worker_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for case,date in enumerate(p['evaluation_initializations']):
            path = args.inputs/f'inputs-{case:03d}.nc'
            record = json.loads(path.with_suffix('.json').read_text())
            context_hash = sha256(path)
            if record['protocol_sha256']!=sha256(protocol) or record['sha256']!=context_hash: raise ValueError('Unverified input')
            # Existing native input validator requires exactly its two context frames.
            local = args.output/f'.context-{args.worker:03d}.nc'
            with xr.open_dataset(path) as ds:
                ds.sel(time=np.asarray([-6,0],dtype='timedelta64[h]')).load().to_netcdf(local)
            init,inputs = context(local,config.task)
            local.unlink()
            if init != np.datetime64(date,'ns'): raise ValueError('Wrong initialization')
            target = template(inputs,config.task)
            case_seed = int.from_bytes(hashlib.sha256(str(init).encode()).digest()[:4],'little')
            expected=dict(protocol_sha256=sha256(protocol),context_sha256=context_hash,
                package_sha256=sha256(args.package/'package.json'),runner_sha256=sha256(Path(__file__)),
                seed=args.seed,initialization=str(init))
            for complete in args.output.glob(f'case-{case:03d}-member-*.complete.json'):
                previous=json.loads(complete.read_text())
                member=previous.get('member',-1)
                if (previous.get('status')!='passed' or not 0<=member<p['members']
                        or complete.name!=f'case-{case:03d}-member-{member:03d}.complete.json'
                        or any(previous.get(k)!=v for k,v in expected.items())):
                    raise ValueError('Existing forecast belongs to a different model/protocol/RNG/context')
            for member in claimed_members(args.output,case,p['members']):
                name = f'case-{case:03d}-member-{member:03d}'
                destination = args.output/f'{name}.zarr'
                done = args.output/f'{name}.complete.json'
                provenance = dict(protocol_sha256=sha256(protocol),context_sha256=context_hash,
                    package_sha256=sha256(args.package/'package.json'),runner_sha256=sha256(Path(__file__)),
                    runtime=dict(jax_version=jax.__version__,device_kinds=[d.device_kind for d in jax.devices()],
                                 deterministic_gpu_ops=True),
                    seed=args.seed,member=member,initialization=str(init))
                if done.exists():
                    previous = json.loads(done.read_text())
                    if any(previous.get(k)!=v for k,v in provenance.items()): raise ValueError('Changed completed forecast')
                    for file,digest in previous['files'].items():
                        if sha256(destination/file)!=digest: raise ValueError('Corrupt forecast')
                    continue
                if destination.exists(): shutil.rmtree(destination)
                store = initialize(destination,init,p['lead_hours'],1,p['pressure_levels_hpa'],variables,inputs.lat.values,inputs.lon.values)
                store['member'][:]=[member]
                current=inputs
                previous_rain=None
                begin=time.monotonic()
                for lead in range(6,max(p['lead_hours'])+1,6):
                    forcings=forcing(current,init,lead,config.task)
                    key=jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(args.seed),case_seed),member),lead)
                    prediction=apply(params,key,current,target,forcings)
                    jax.block_until_ready(prediction)
                    for field in config.task.target_variables:
                        values=np.asarray(prediction[field].values)
                        if np.isinf(values).any() or (field!='sea_surface_temperature' and not np.isfinite(values).all()):
                            raise ValueError(f'Nonfinite {field}')
                    current_rain=np.asarray(prediction.total_precipitation_6hr.isel(batch=0,time=0).transpose('lat','lon').values)
                    if lead in p['lead_hours']:
                        if previous_rain is None: raise ValueError('Missing first precipitation interval')
                        for field,spec in variables.items():
                            dims=(['level'] if spec['pressure_levels'] else [])+['lat','lon']
                            values = previous_rain+current_rain if field=='total_precipitation_12hr' else prediction[field].isel(batch=0,time=0).transpose(*dims).values
                            store[field][0,p['lead_hours'].index(lead),0]=values
                    previous_rain=current_rain
                    current=next_context(current,prediction,forcings,config.task)
                    print(json.dumps(dict(case=case,member=member,lead=lead,elapsed_seconds=time.monotonic()-begin)),flush=True)
                inventory={str(f.relative_to(destination)):sha256(f) for f in sorted(destination.rglob('*')) if f.is_file()}
                temporary=done.with_suffix('.tmp')
                temporary.write_text(json.dumps(dict(status='passed',**provenance,files=inventory))+'\n')
                temporary.replace(done)


if __name__=='__main__': main()
