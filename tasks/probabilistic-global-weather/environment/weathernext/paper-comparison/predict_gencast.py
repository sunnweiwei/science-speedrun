#!/usr/bin/env python3
"""Operator-only full Operational GenCast inference, sharded by ensemble member.

Wrapping and autoregression follow the official WeatherNext GenCast demo.
Inputs are context-only; this process never opens verification truth.
"""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import shutil
import fcntl

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'wn2'))
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from weathernext.utils import checkpoint, data_utils, nan_cleaning, normalization, rollout
from weathernext.weathernext1_gen import gencast
from forecast_store import initialize
from training_data import sha256
from noise_grid_cache import install as install_noise_grid_cache


def templates(context, init, task, leads):
    times = np.concatenate((np.array([-12, 0], dtype='timedelta64[h]'), np.asarray(leads, dtype='timedelta64[h]')))
    context = context.sel(time=np.array([-12, 0], dtype='timedelta64[h]'))
    np.testing.assert_array_equal(context.lat, np.linspace(-90, 90, 721))
    np.testing.assert_array_equal(context.lon, np.arange(1440) / 4)
    np.testing.assert_array_equal(context.level, task.pressure_levels)
    dataset = xr.Dataset(coords=dict(batch=[0], time=times, lat=context.lat, lon=context.lon, level=context.level))
    for name in task.target_variables:
        source = context['2m_temperature'] if name == 'total_precipitation_12hr' else context[name]
        leading = ('level',) if 'level' in source.dims else ()
        source = source.transpose('batch', 'time', *leading, 'lat', 'lon')
        values = np.zeros((1, len(times), *source.shape[2:]), dtype=np.float32)
        if name != 'total_precipitation_12hr':
            values[:, :2] = source.values
        dataset[name] = (('batch', 'time', *leading, 'lat', 'lon'), values)
    for name in ('geopotential_at_surface', 'land_sea_mask'):
        dataset[name] = context[name].transpose('lat', 'lon')
    dataset = dataset.assign_coords(datetime=(('batch', 'time'), (init + times)[None, :]))
    return data_utils.extract_inputs_targets_forcings(dataset,
        target_lead_times=slice('12h', f'{max(leads)}h'), **dataclasses.asdict(task))


def claimed_members(output, case, members):
    """Shared queue permits additional suitable GPU workers to join later."""
    for member in range(members):
        name=f'case-{case:03d}-member-{member:03d}'
        with (output/f'{name}.lock').open('w') as claim:
            try: fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError: continue
            if (output/f'{name}.complete.json').exists(): continue
            yield member


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--assets', required=True, type=Path)
    ap.add_argument('--inputs', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--worker', required=True, type=int)
    ap.add_argument('--workers', required=True, type=int)
    ap.add_argument('--seed', type=int, default=20220910)
    args = ap.parse_args()
    if not 0 <= args.worker < args.workers: raise ValueError('Invalid worker shard')
    here = Path(__file__).resolve().parent
    protocol = here / 'weather-protocol.json'
    p = json.loads(protocol.read_text())
    precip = json.loads((here / 'precipitation-protocol.json').read_text())
    variables = {**p['variables'], **precip['variables']}
    lock = json.loads((here / 'baseline-assets.lock.json').read_text())
    # The lock describes exactly the public full-resolution Operational model.
    files = lock['files']
    for entry in files:
        if sha256(args.assets / entry['file']) != entry['sha256']:
            raise ValueError(f"Baseline asset checksum mismatch: {entry['file']}")
    with (args.assets / 'GenCast 0p25deg Operational <2022.npz').open('rb') as stream:
        ckpt = checkpoint.load(stream, gencast.CheckPoint)
    if ckpt.sampler_config.num_noise_levels != 20: raise ValueError('Expected the published full sampler')
    config = ckpt.denoiser_architecture_config
    config.sparse_transformer_config.attention_type = 'triblockdiag_mha'
    config.sparse_transformer_config.mask_type = 'full'
    stats = {name: xr.load_dataset(args.assets / f'{name}.nc') for name in (
        'diffs_stddev_by_level', 'mean_by_level', 'stddev_by_level', 'min_by_level')}

    @hk.transform_with_state
    def forward(inputs, targets_template, forcings):
        model = gencast.GenCast(task_config=ckpt.task_config, denoiser_architecture_config=config,
            sampler_config=ckpt.sampler_config, noise_config=ckpt.noise_config,
            noise_encoder_config=ckpt.noise_encoder_config)
        model = normalization.InputsAndResiduals(model, **{n: stats[n] for n in (
            'diffs_stddev_by_level', 'mean_by_level', 'stddev_by_level')})
        model = nan_cleaning.NaNCleaner(model, var_to_clean='sea_surface_temperature',
            fill_value=stats['min_by_level'], reintroduce_nans=True)
        return model(inputs, targets_template=targets_template, forcings=forcings)

    install_noise_grid_cache()
    params = jax.tree.map(jnp.asarray, ckpt.params)
    apply = jax.jit(forward.apply)
    def predictor_fn(rng, inputs, targets_template, forcings):
        return apply(params, {}, rng, inputs, targets_template, forcings)[0]

    args.output.mkdir(parents=True, exist_ok=True)
    # A duplicate worker must never overwrite a live or completed trajectory.
    with (args.output / f'worker-{args.worker:03d}.lock').open('w') as worker_lock:
        fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for index, date in enumerate(p['evaluation_initializations']):
            context_path = args.inputs / f'inputs-{index:03d}.nc'
            record = json.loads(context_path.with_suffix('.json').read_text())
            context_hash = sha256(context_path)
            if record['protocol_sha256'] != sha256(protocol) or record['sha256'] != context_hash:
                raise ValueError('Unverified context')
            init = np.datetime64(date, 'ns')
            context = xr.load_dataset(context_path)
            np.testing.assert_array_equal(context.datetime.values,
                init + np.asarray([[-12, -6, 0]], dtype='timedelta64[h]'))
            inputs, targets, forcings = templates(context, init, ckpt.task_config, p['lead_hours'])
            date_key = int.from_bytes(hashlib.sha256(str(init).encode()).digest()[:4], 'little')
            key = jax.random.fold_in(jax.random.PRNGKey(args.seed), date_key)
            for member in claimed_members(args.output, index, p['members']):
                name = f'case-{index:03d}-member-{member:03d}'
                destination = args.output / f'{name}.zarr'
                done = args.output / f'{name}.complete.json'
                provenance = dict(protocol_sha256=sha256(protocol), context_sha256=context_hash,
                    assets_lock_sha256=sha256(here/'baseline-assets.lock.json'),
                    runner_sha256=sha256(Path(__file__)), noise_grid_cache_sha256=sha256(here/'noise_grid_cache.py'), seed=args.seed, member=member, initialization=str(init))
                if done.exists():
                    previous = json.loads(done.read_text())
                    if any(previous.get(k) != v for k, v in provenance.items()): raise ValueError('Changed completed forecast')
                    for file, digest in previous['files'].items():
                        if sha256(destination/file) != digest: raise ValueError('Corrupt completed forecast')
                    continue
                if destination.exists(): shutil.rmtree(destination)  # Only this worker's incomplete trajectory.
                store = initialize(destination, init, p['lead_hours'], 1, p['pressure_levels_hpa'],
                    variables, context.lat.values, context.lon.values)
                store['member'][:] = [member]
                begin = time.monotonic()
                count = 0
                for lead_index, prediction in enumerate(rollout.chunked_prediction_generator(
                        predictor_fn=predictor_fn, rng=jax.random.fold_in(key, member),
                        inputs=inputs, targets_template=targets, forcings=forcings, num_steps_per_chunk=1)):
                    for name, spec in variables.items():
                        dims = (['level'] if spec['pressure_levels'] else []) + ['lat', 'lon']
                        values = np.asarray(prediction[name].isel(batch=0, time=0).transpose(*dims).values)
                        if not np.isfinite(values).all(): raise ValueError(f'Nonfinite {name}')
                        store[name][0, lead_index, 0] = values
                    count += 1
                    print(json.dumps(dict(case=index, member=member, lead=p['lead_hours'][lead_index],
                        elapsed_seconds=time.monotonic()-begin)), flush=True)
                if count != len(p['lead_hours']): raise ValueError('Incomplete trajectory')
                inventory = {str(f.relative_to(destination)): sha256(f) for f in sorted(destination.rglob('*')) if f.is_file()}
                temporary = done.with_suffix('.tmp')
                temporary.write_text(json.dumps(dict(status='passed', **provenance, files=inventory))+'\n')
                temporary.replace(done)
            del inputs, targets, forcings, context


if __name__ == '__main__': main()
