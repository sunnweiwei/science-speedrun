#!/usr/bin/env python3
"""Exercise the upstream FGN ensemble loss and gradients under mixed precision.

Uses a tiny differentiable noisy function; no full-model or forecast-skill claim.
"""
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
import xarray_jax
from weathernext.utils import ensemble
from weathernext.weathernext2 import fgn
from wn2_validation import construct_predictor


class TinyNetwork:
    def __call__(self, *, inputs, targets_template, forcings=None, **kwargs):
        scale = hk.get_parameter("scale", (), dtype=inputs["2m_temperature"].dtype,
                                 init=hk.initializers.Constant(0.5))
        return inputs[["2m_temperature"]] * scale


def main():
    # Also confirm the runtime behavior behind the original failure.
    bf16 = xr.Dataset({"t": xarray_jax.DataArray(jnp.ones((2, 1), dtype=jnp.bfloat16),
                                               dims=("sample", "batch"))})
    assert not bf16.mean("sample").data_vars
    data = xr.Dataset({"2m_temperature": xarray_jax.DataArray(
        jnp.arange(18, dtype=jnp.float32).reshape(1, 1, 3, 6) / 4,
        dims=("batch", "time", "lat", "lon"),
        coords={"lat": [-90., 0., 90.], "lon": np.arange(6) * 60.})})
    targets = data * 0.25
    config = fgn.PredictorConfig(
        task=None, predictor_constructor=fgn.Predictor,
        predictor_kwargs={"noisy_function_constructor": TinyNetwork,
                          "noisy_function_kwargs": {},
                          "noise_generator_constructor": lambda i, f, **kw: (i, f),
                          "noise_generator_kwargs": {}},
        predictor_wrappers=[{"constructor": ensemble.WithSampleDim,
                             "kwargs": {"num_samples": 2}}])
    results = []
    for precision in ("float32", "bfloat16"):
        @hk.transform
        def loss(i, t):
            value, diagnostics = construct_predictor(config, precision).loss(i, t, forcings=i)
            assert set(diagnostics.data_vars) == {"2m_temperature"}
            return xarray_jax.unwrap_data(value.mean(), require_jax=True)

        key = jax.random.PRNGKey(0)
        params = loss.init(key, data, targets)
        value, grad = jax.jit(jax.value_and_grad(loss.apply))(params, key, data, targets)
        assert value.dtype == jnp.float32 and np.isfinite(value) and float(value) > 0
        assert all(np.isfinite(g).all() for g in jax.tree_util.tree_leaves(grad))
        assert any(np.any(g != 0) for g in jax.tree_util.tree_leaves(grad))
        results.append((np.asarray(value), jax.tree_util.tree_leaves(grad)))
    np.testing.assert_allclose(results[0][0], results[1][0], rtol=0.02)
    for a, b in zip(results[0][1], results[1][1]):
        np.testing.assert_allclose(a, b, rtol=0.02)
    print("WN2_NATIVE_CRPS_FLOAT32_REDUCTION_BF16_NETWORK_GRADIENT_OK")


if __name__ == "__main__":
    main()
