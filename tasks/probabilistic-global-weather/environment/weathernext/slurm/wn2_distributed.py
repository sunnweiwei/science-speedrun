"""Four-device accumulation and exact optimizer-state replay execution gate."""
import hashlib
import json
import re
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import xarray_jax
from weathernext.utils import rollout


def make_step(loss_apply, optimizer, accumulation, devices):
    def step(params, state, key, inputs, targets, forcings):
        zeros = jax.tree.map(jnp.zeros_like, params)
        def body(index, carry):
            total, loss_sum, finite = carry
            (value, _), grad = jax.value_and_grad(loss_apply, has_aux=True)(
                params, jax.random.fold_in(key, index), inputs, targets, forcings)
            valid = jnp.isfinite(value) & jnp.all(jnp.stack([
                jnp.all(jnp.isfinite(v)) for v in jax.tree.leaves(grad)]))
            return jax.tree.map(jnp.add, total, grad), loss_sum + value, finite & valid
        grad, value, finite = jax.lax.fori_loop(0, accumulation, body, (zeros, jnp.float32(0), jnp.bool_(True)))
        grad = jax.tree.map(lambda g: jax.lax.pmean(g / accumulation, "device"), grad)
        value = jax.lax.pmean(value / accumulation, "device")
        updates, state = optimizer.update(grad, state, params)
        updated = optax.apply_updates(params, updates)
        valid = jnp.all(jnp.stack([jnp.all(jnp.isfinite(v)) for v in jax.tree.leaves((updated, state))]))
        changed = jnp.any(jnp.stack([jnp.any(a != b) for a,b in zip(jax.tree.leaves(params),jax.tree.leaves(updated))]))
        finite = jax.lax.pmin((finite & valid).astype(jnp.int32), "device").astype(bool)
        return updated, state, value, optax.global_norm(grad), finite, changed
    return xarray_jax.pmap(step, dim="device", devices=devices)


def replicate_from_host(tree, devices):
    """Populate pmap replicas with independent host-to-device transfers.

    The cluster's peer replication path returned zero data on a secondary GPU
    even for [1, 0]. Distinct host buffers avoid that observed copy path.
    """
    def put_leaf(value):
        host = np.asarray(jax.device_get(value))
        return jax.device_put_sharded([np.array(host, copy=True) for _ in devices], devices)
    return jax.tree.map(put_leaf, tree)


def host_replica(tree, verify_replicas=False):
    # A checkpoint stores one replica; reject divergent replicas before losing
    # their state. Compare one leaf/shard at a time to bound host memory.
    def copy_leaf(array):
        shards = array.addressable_shards
        first = np.asarray(shards[0].data)[0]
        if verify_replicas:
            for shard in shards[1:]:
                other = np.asarray(shard.data)[0]
                if (first.shape != other.shape or first.dtype != other.dtype or
                        first.tobytes() != other.tobytes()):
                    raise ValueError('Checkpoint replicas differ in exact bits')
        return first
    return jax.tree.map(copy_leaf, tree)


def save_state(path, params, state, metadata):
    entries, _ = jax.tree_util.tree_flatten_with_path((host_replica(params, verify_replicas=True),
                                                       host_replica(state, verify_replicas=True)))
    paths = [jax.tree_util.keystr(key) for key, _ in entries]
    leaves = [value for _, value in entries]
    metadata = metadata | {'state_leaf_paths': paths}
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **{f"leaf_{i:04d}": a for i,a in enumerate(leaves)},
                 metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
    temporary.replace(path)


def load_state(path, template, expected, cadence_transition=None):
    entries, structure = jax.tree_util.tree_flatten_with_path(template)
    paths = [jax.tree_util.keystr(key) for key, _ in entries]
    leaves = [value for _, value in entries]
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        for k,v in expected.items():
            if metadata.get(k) != v: raise ValueError(f"Resume mismatch: {k}")
        saved_paths = metadata.get('state_leaf_paths')
        if not isinstance(saved_paths, list) or len(saved_paths) != len(leaves):
            raise ValueError('Checkpoint lacks explicit parameter/optimizer leaf identities')
        if cadence_transition is not None:
            if tuple(cadence_transition) != (12, 6) or metadata.get('cadence_hours') != 12:
                raise ValueError('Only the declared 12h-to-6h curriculum transfer is supported')
            def rename(match):
                seconds = int(match.group(0))
                if seconds not in (-43200, 0, 43200):
                    raise ValueError('Unexpected parameter time coordinate')
                return str(seconds // 2)
            saved_paths = [re.sub(r'(?<=_time=)-?\d+', rename, key) for key in saved_paths]
        if len(set(saved_paths)) != len(saved_paths) or set(saved_paths) != set(paths):
            raise ValueError('Checkpoint parameter/optimizer identities do not match the native stage')
        source_indices = {key: i for i, key in enumerate(saved_paths)}
        if set(data.files) != {"metadata"} | {f"leaf_{i:04d}" for i in range(len(leaves))}:
            raise ValueError("Resume state leaf set mismatch")
        restored = []
        for i, (key, ref) in enumerate(zip(paths, leaves)):
            value = data[f"leaf_{source_indices[key]:04d}"]
            if value.shape != ref.shape or value.dtype != ref.dtype or not np.isfinite(value).all():
                raise ValueError(f"Invalid state leaf {i}")
            restored.append(value)
    return jax.tree.unflatten(structure, restored), metadata


def run_probe(params, loss, inputs, targets, forcings, output, save, accumulation=16, resume=None):
    devices = jax.local_devices()
    if len(devices) != 4: raise ValueError("Distributed probe requires exactly four allocated GPUs")
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1e-5,b2=.95,weight_decay=.1))
    state = optimizer.init(params)
    spec = dict(schema="wn2-distributed-probe/v1", devices=4, accumulation=accumulation,
                seed=0, effective_batch=4*accumulation, unique_weather_windows=1,
                ensemble_samples_per_window=2,
                helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                runner_sha256=hashlib.sha256(Path(__file__).with_name("wn2_validation.py").read_bytes()).hexdigest())
    start_step = 0
    if resume:
        (params,state), metadata = load_state(resume,(params,state),spec)
        start_step = metadata["completed_updates"]
        if start_step != 1: raise ValueError("Replay probe must resume exactly update one")
    params,state = replicate_from_host((params,state),devices)
    i,t,f = [rollout.replicate_dataset(v,"device",True,devices=devices) for v in (inputs,targets,forcings)]
    step = make_step(loss.apply,optimizer,accumulation,devices)
    save(distributed_spec=spec, phase="distributed_compile_and_first_update", resumed_from=str(resume) if resume else None)
    for index in range(start_step,2):
        keys = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(0),index),4)
        begin = time.monotonic()
        params,state,value,norm,finite,changed = step(params,state,keys,i,t,f)
        jax.block_until_ready((params,state,value,norm,finite,changed))
        seconds = time.monotonic()-begin
        if not np.asarray(finite).all() or not np.asarray(changed).all() or not (np.asarray(norm)>0).all():
            raise ValueError("Distributed finite/nonzero/update gate failed")
        if not np.equal(np.asarray(value), np.asarray(value)[0]).all(): raise ValueError("Replica loss mismatch")
        save(phase="distributed_update_complete", completed_updates=index+1, update_seconds=seconds,
             includes_compile=index==start_step, loss=float(np.asarray(value)[0]), gradient_norm=float(np.asarray(norm)[0]))
        save_state(output/f"state-{index+1}.npz",params,state,spec|dict(completed_updates=index+1,
                   microbatch_draws=(index+1)*4*accumulation))
