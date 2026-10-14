"""Independent analytic modes, Parseval and pinned native-transform checks."""
import json
import jax
import numpy as np
from spherical_spectra import Spectrum

jax.config.update('jax_enable_x64',True)
lat=np.linspace(-90,90,65);lon=np.arange(128)*360/128
spectrum=Spectrum(lat,lon)
phi=np.deg2rad(lat)[:,None];lam=np.deg2rad(lon)[None,:]
constant=np.full((65,128),3.)
dipole=np.broadcast_to(np.sin(phi),(65,128))
sectoral=np.cos(phi)**2*np.cos(2*lam)
expected=[(constant,0,9.),(dipole,1,1/3),(sectoral,2,4/15)]
for field,degree,power in expected:
    actual=spectrum.power(field)
    np.testing.assert_allclose(actual[degree],power,rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose(np.delete(actual,degree),0,atol=1e-20)
combined=constant+dipole+sectoral
native=np.asarray(spectrum.grid.to_modal(combined.T))
np.testing.assert_allclose(spectrum.coefficients(combined),native,rtol=1e-11,atol=1e-12)
np.testing.assert_allclose(spectrum.power(combined).sum(),9+1/3+4/15,rtol=1e-11)
np.testing.assert_allclose(spectrum.power(np.roll(combined,17,axis=1)),spectrum.power(combined),rtol=1e-11,atol=1e-12)
print(json.dumps(dict(status='passed',checks=['constant mode','zonal dipole','nonzonal degree 2','native transform parity','Parseval on analytic modes','longitude rotation invariance'],provenance=spectrum.provenance())),flush=True)
