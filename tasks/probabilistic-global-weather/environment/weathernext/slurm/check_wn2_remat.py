#!/usr/bin/env python3
"""CPU equivalence check of the actual WeatherNext Transformer remat hook.

Uses a small graph; it does not establish full WN2 GPU execution or skill.
"""
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse
from weathernext.utils import sparse_transformer
from wn2_validation import enable_transformer_remat


def main():
    @hk.transform
    def model(x, conditioning):
        return sparse_transformer.Transformer(
            adj_mat=sparse.eye(8, format="csr", dtype=bool), attention_k_hop=1,
            attention_type="triblockdiag_mha", mask_type="full", num_heads=2,
            num_layers=2, d_model=12, ffw_hidden=24)(x, conditioning)

    x = jnp.arange(96, dtype=jnp.float32).reshape(1, 8, 12) / 100
    conditioning = jnp.ones((1, 4))
    key = jax.random.PRNGKey(0)
    params = model.init(key, x, conditioning)

    def objective(p):
        return jnp.square(model.apply(p, key, x, conditioning)).sum()

    expected = model.apply(params, key, x, conditioning)
    gradients = jax.grad(objective)(params)
    enable_transformer_remat()
    remat_params = model.init(key, x, conditioning)
    actual = model.apply(params, key, x, conditioning)
    remat_gradients = jax.grad(objective)(params)
    assert jax.tree_util.tree_structure(params) == jax.tree_util.tree_structure(remat_params)
    np.testing.assert_allclose(expected, actual, rtol=1e-6, atol=1e-6)
    for first, second in zip(jax.tree_util.tree_leaves(gradients), jax.tree_util.tree_leaves(remat_gradients)):
        np.testing.assert_allclose(first, second, rtol=1e-5, atol=1e-5)
    print("REMAT_PARAM_NAMES_FORWARD_GRADIENT_EQUIVALENCE_OK")


if __name__ == "__main__":
    main()
