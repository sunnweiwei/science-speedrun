"""Check masked spherical moments, merging and absent-label rejection."""
import numpy as np
import xarray as xr
from training_statistics import moments,merge

field=xr.DataArray([[1.,2.],[3.,4.],[5.,np.nan]],dims=('lat','lon'),coords={'lat':[-90.,0.,90.]})
record=moments(field)
weights=np.diff(np.sin(np.deg2rad([-90.,-45.,45.,90.])))[:,None]
valid=np.isfinite(field.values)
expected=np.nansum(field.values*weights)/np.sum(valid*weights)
result=merge([{'x':record},{'x':record}])['x']
np.testing.assert_allclose(result['mean'],expected)
variance=np.nansum((field.values-expected)**2*weights)/np.sum(valid*weights)
np.testing.assert_allclose(result['std'],np.sqrt(variance))
assert result['minimum']==1
try:merge([{'x':moments(xr.full_like(field,np.nan))}])
except ValueError as e:assert 'No training observations' in str(e)
else:raise AssertionError('Absent training labels were accepted')
print('WN2_MASKED_TRAINING_STATISTICS_ORACLE_OK')
