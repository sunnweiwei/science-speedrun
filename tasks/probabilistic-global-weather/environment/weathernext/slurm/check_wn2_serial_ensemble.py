"""Native versus serial FGN predictions and full gradient, identical noise."""
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import xarray_jax
from weathernext.weathernext2 import fgn
from wn2_validation import enable_serial_ensemble

class Network:
    def __init__(self): pass
    def __call__(self, *, inputs, targets_template, forcings, is_training):
        w = hk.get_parameter('w', (3,), init=hk.initializers.Constant(0.7))
        return xarray_jax.Dataset({'x': (inputs.x.dims, jnp.tanh(xarray_jax.unwrap_data(inputs.x) * w + xarray_jax.unwrap_data(forcings.noise)))}, coords=inputs.coords)

def noise(inputs, forcings, default_dtype):
    forcings = forcings.copy()
    forcings['noise'] = xarray_jax.DataArray(jax.random.normal(hk.next_rng_key(), inputs.x.shape), dims=inputs.x.dims)
    return inputs, forcings

def run(inputs, targets, forcings):
    p=fgn.Predictor(Network, {}, noise, {})
    out=p(inputs, targets, forcings, is_training=True)
    a=xarray_jax.unwrap_data(out.x)
    return jnp.mean(a*a)+jnp.mean((a[0]-a[1])**2), a

inputs=xarray_jax.Dataset({'x': xarray_jax.DataArray(jnp.arange(12,dtype=jnp.float32).reshape(2,2,3)/8,dims=('sample','batch','feature'))})
forcing=xarray_jax.Dataset({'f':inputs.x*0})
key=jax.random.PRNGKey(18)
fn=hk.transform(run)
params=fn.init(key,inputs,inputs,forcing)
original=jax.jit(jax.value_and_grad(fn.apply,argnums=(0,2),has_aux=True))(params,key,inputs,inputs,forcing)
enable_serial_ensemble()
serial=jax.jit(jax.value_and_grad(hk.transform(run).apply,argnums=(0,2),has_aux=True))(params,key,inputs,inputs,forcing)
for a,b in zip(jax.tree_util.tree_leaves(original),jax.tree_util.tree_leaves(serial)):
    np.testing.assert_allclose(a,b,rtol=2e-6,atol=2e-6)
print('WN2_SERIAL_ENSEMBLE_SAME_NOISE_OUTPUT_AND_GRADIENT_OK',jax.default_backend())
