#!/usr/bin/env python3
"""Four virtual CPU devices: gradient averaging oracle and state-file replay."""
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import xarray as xr
import xarray_jax
from weathernext.utils import rollout
from wn2_distributed import make_step, save_state, load_state, host_replica


def main():
    devices=jax.local_devices()
    assert len(devices)==4 and all(d.platform=='cpu' for d in devices)
    params={"w":jnp.asarray([.2,.3],dtype=jnp.float32)}
    optimizer=optax.adamw(1e-3)
    state=optimizer.init(params)
    data=xr.Dataset({"x":xarray_jax.DataArray(jnp.asarray([1.,2.]),dims='feature')})
    def loss(p,key,i,t,f):
        x=xarray_jax.unwrap_data(i.x,require_jax=True)
        return jnp.mean((p['w']*x-jax.random.normal(key,(2,)))**2), {}
    accumulate=16
    step=make_step(loss,optimizer,accumulate,devices)
    mapped=rollout.replicate_dataset(data,'device',True,devices=devices)
    pp,ss=jax.device_put_replicated((params,state),devices)
    keys=jax.random.split(jax.random.PRNGKey(0),4)
    updated,updated_state,*_=step(pp,ss,keys,mapped,mapped,mapped)
    grads=[jax.grad(lambda p: loss(p,jax.random.fold_in(k,n),data,data,data)[0])(params)
           for k in keys for n in range(accumulate)]
    mean=jax.tree.map(lambda *xs:sum(xs)/len(xs),*grads)
    delta,expected_state=optimizer.update(mean,state,params)
    expected=optax.apply_updates(params,delta)
    for a,b in zip(jax.tree.leaves(host_replica(updated)),jax.tree.leaves(expected)):
        np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-7)
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'state.npz'
        save_state(path,updated,updated_state,{'schema':'test','completed_updates':1})
        restored,meta=load_state(path,(params,state),{'schema':'test'})
        assert meta['completed_updates']==1
        rp,rs=jax.device_put_replicated(restored,devices)
        result=step(updated,updated_state,keys,mapped,mapped,mapped)
        replay=step(rp,rs,keys,mapped,mapped,mapped)
        for a,b in zip(jax.tree.leaves(result),jax.tree.leaves(replay)):
            np.testing.assert_array_equal(a,b)
    print('WN2_FOUR_CPU_ACCUMULATION_ORACLE_STATE_REPLAY_OK')


if __name__=='__main__':main()
