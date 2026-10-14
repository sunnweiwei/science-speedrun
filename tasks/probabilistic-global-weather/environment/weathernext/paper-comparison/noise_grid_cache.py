"""Reuse deterministic GenCast spherical bases through its supported grid argument.

No RNG draws, coefficients, truncation, sampler settings or arithmetic change.
Upstream sample() explicitly recommends sharing Grid to save RAM and compute.
"""
import functools
import numpy as np


def install():
    from weathernext.weathernext1_gen import samplers_utils
    original = samplers_utils.sample
    if getattr(original, '_shared_grid_cache', None) is not None:
        return original._shared_grid_cache
    cache = {}

    @functools.wraps(original)
    def sample(key, power_spectrum, template, grid=None):
        if grid is None:
            lat = np.asarray(template.lat.data)
            lon = np.asarray(template.lon.data)
            identity = (lat.dtype.str, lat.shape, lat.tobytes(), lon.dtype.str, lon.shape, lon.tobytes())
            if identity not in cache:
                cache[identity] = samplers_utils.Grid.for_nodal_data(template)
            grid = cache[identity]
        return original(key=key, power_spectrum=power_spectrum, template=template, grid=grid)

    sample._shared_grid_cache = cache
    samplers_utils.sample = sample
    return cache
