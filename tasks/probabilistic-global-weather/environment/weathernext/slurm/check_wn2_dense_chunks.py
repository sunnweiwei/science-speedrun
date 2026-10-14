"""Native DenseLayer versus chunked values and complete derivatives."""
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from weathernext.utils import dense
from wn2_dense_chunks import enable_chunked_dense

# Compare the mathematical row split without backend reduced-precision GEMMs.
jax.config.update("jax_default_matmul_precision", "highest")


def model(x, condition):
    layer=dense.DenseLayer(name='dense',output_size=7,hidden_size=11,num_hidden_layers=2,
        activation='swish',activate_final=False,activation_normalization='layer_norm',
        activation_normalization_kwargs=dict(use_norm_conditioning=True),remat=True)
    out=layer(x,condition)
    return jnp.sum(jnp.sin(out)),out

fn=hk.transform(model)
key=jax.random.PRNGKey(14)
x=jax.random.normal(key,(11,2,5));condition=jax.random.normal(key,(2,4))
params=fn.init(key,x,condition)
ref=jax.jit(jax.value_and_grad(fn.apply,argnums=(0,2,3),has_aux=True))(params,key,x,condition)
enable_chunked_dense(3)
actual=jax.jit(jax.value_and_grad(hk.transform(model).apply,argnums=(0,2,3),has_aux=True))(params,key,x,condition)
for a,b in zip(jax.tree.leaves(ref),jax.tree.leaves(actual)):
    np.testing.assert_allclose(a,b,rtol=5e-5,atol=5e-5)
new_params=hk.transform(model).init(key,x,condition)
assert jax.tree.structure(params)==jax.tree.structure(new_params)
for a,b in zip(jax.tree.leaves(params),jax.tree.leaves(new_params)):
    np.testing.assert_array_equal(a,b)
print('WN2_DENSE_CHUNKS_NATIVE_VALUES_PARAMETERS_AND_GRADIENTS_OK',jax.default_backend())
