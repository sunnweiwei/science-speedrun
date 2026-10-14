"""Four-CPU-device oracle for streamed accumulation and stage configuration."""
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import optax
import xarray as xr
import xarray_jax

from train_stage import draw_indices, make_functions, optimizer


devices=jax.local_devices()
assert len(devices)==4
params={'w':jnp.asarray([.25,-.5],dtype=jnp.float32)}
optim=optimizer();state=optim.init(params)
def loss(p,key,inputs,targets,forcings):
    del key,forcings
    x=xarray_jax.unwrap_data(inputs.x);y=xarray_jax.unwrap_data(targets.x)
    return jnp.mean((p['w']*x-y)**2),{}
accumulate,update=make_functions(SimpleNamespace(apply=loss),optim,devices)
replicated,replicated_state=jax.device_put_replicated((params,state),devices)
total=jax.tree.map(jnp.zeros_like,replicated)
direct=[]
for micro in range(16):
    x=np.arange(8,dtype=np.float32).reshape(4,2)/8+micro/16
    y=x*.3
    ds=xr.Dataset({'x':(('device','feature'),x)})
    target=xr.Dataset({'x':(('device','feature'),y)})
    total,value,finite=accumulate(replicated,total,jax.random.split(jax.random.PRNGKey(micro),4),ds,target,xr.Dataset())
    assert np.asarray(finite).all()
    direct.extend([2*(np.asarray(params['w'])*a-b)*a/2 for a,b in zip(x,y)])
actual,actual_state,norm,finite,changed=update(replicated,replicated_state,total,np.full(4,.01,dtype=np.float32))
grad={'w':jnp.asarray(np.mean(direct,axis=0))}
updates,expected_state=optim.update(grad,state,params)
expected=optax.apply_updates(params,jax.tree.map(lambda a:a*.01,updates))
for leaf,reference in zip(jax.tree.leaves(actual),jax.tree.leaves(expected)):
    np.testing.assert_allclose(np.asarray(leaf),np.broadcast_to(np.asarray(reference),np.asarray(leaf).shape),rtol=1e-6,atol=1e-7)
assert np.asarray(finite).all() and np.asarray(changed).all()
draws=draw_indices(0,2,3,32)
np.testing.assert_array_equal(draws,draw_indices(0,2,3,32))
np.testing.assert_array_equal(np.sort(draws.reshape(-1)),np.repeat(np.arange(32),2))
assert not np.array_equal(draws,draw_indices(0,2,4,32))
print('WN2_STREAMED_BATCH64_CPU_ORACLE_OK')
