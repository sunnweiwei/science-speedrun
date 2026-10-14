"""Run in pinned model image: exact noise parity across variables and RNG keys."""
import json
import haiku as hk
import jax
import numpy as np
import xarray as xr
from weathernext.weathernext1_gen import samplers_utils
from noise_grid_cache import install

coords=dict(batch=[0],time=np.array([12],dtype='timedelta64[h]'),lat=np.linspace(-90,90,5),lon=np.arange(8)*45.,level=[500,850])
template=xr.Dataset({'surface':(('batch','time','lat','lon'),np.zeros((1,1,5,8),np.float32)),
                     'pressure':(('batch','time','level','lat','lon'),np.zeros((1,1,2,5,8),np.float32))},coords=coords)
original=hk.transform(lambda: samplers_utils.spherical_white_noise_like(template))
old_apply=jax.jit(original.apply)
expected=[old_apply({},jax.random.PRNGKey(seed)) for seed in (0,23)]
cache=install()
cached=hk.transform(lambda: samplers_utils.spherical_white_noise_like(template))
new_apply=jax.jit(cached.apply)
for seed,reference in zip((0,23),expected):
 actual=new_apply({},jax.random.PRNGKey(seed))
 for name in template.data_vars:
  np.testing.assert_array_equal(reference[name].values,actual[name].values)
assert len(cache)==1, len(cache)
assert not np.array_equal(expected[0].surface.values,expected[1].surface.values)
print(json.dumps(dict(status='passed',exact_noise_parity=True,variables=2,seeds=2,grid_instances=len(cache))),flush=True)
