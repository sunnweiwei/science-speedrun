#!/usr/bin/env python3
"""Independent NumPy value/adjoint oracle for bounded deterministic graph sums."""
import jax
import jax.numpy as jnp
import numpy as np
from weathernext.utils import gather_scatter_ops
from wn2_validation import enable_deterministic_graph_ops

enable_deterministic_graph_ops()
rng=np.random.default_rng(13)
cases=[np.array([],np.int32),np.array([-1,-1],np.int32),
       np.array([0,0,-1,2,2,2,7,8,-1],np.int32),
       np.repeat(np.arange(8,dtype=np.int32),[1,2,3,5,9,17,33,129])]
for indices in cases:
    for shuffle in (False,True):
        indices=indices.copy()
        if shuffle:rng.shuffle(indices)
        for dtype in (jnp.float32,jnp.bfloat16):
            x=jnp.asarray(rng.normal(size=(len(indices),2,5)),dtype=dtype)
            w=jnp.asarray(rng.normal(size=(8,2,5)),dtype=dtype)
            operation=gather_scatter_ops.create_static_segment_sum(indices,8)
            expected=np.zeros((8,2,5),np.float32)
            valid=(indices>=0)&(indices<8)
            np.add.at(expected,indices[valid],np.asarray(x,dtype=np.float32)[valid])
            expected=np.asarray(jnp.asarray(expected,dtype=dtype),dtype=np.float32)
            actual=np.asarray(jax.jit(operation)(x),dtype=np.float32)
            np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-5)
            gradient=jax.jit(jax.grad(lambda a:jnp.sum(operation(a).astype(jnp.float32)*w.astype(jnp.float32))))(x)
            reference=np.zeros(x.shape,np.float32)
            reference[valid]=np.asarray(w,dtype=np.float32)[indices[valid]]
            np.testing.assert_array_equal(np.asarray(gradient,dtype=np.float32),reference)
            for _ in range(3):
                np.testing.assert_array_equal(np.asarray(jax.jit(operation)(x),dtype=np.float32),actual)
print('WN2_DETERMINISTIC_GRAPH_SUM_NUMPY_AND_ADJOINT_ORACLE_OK',jax.default_backend())
