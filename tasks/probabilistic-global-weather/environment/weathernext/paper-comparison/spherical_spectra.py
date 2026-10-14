"""Real spherical harmonic power using the pinned Dinosaur basis and quadrature.

Sum squared orthonormal coefficients over signed zonal orders and divide by
4*pi, giving mean-square power per total wavenumber. The constant mode is kept.
Use the actual equiangular-with-poles grid, without latitude regridding or a
planar Fourier approximation. NumPy contractions avoid compiling a multi-GB
constant into XLA; the basis, quadrature and transform are Dinosaur's.
"""
import hashlib
import importlib.metadata
from pathlib import Path

import numpy as np
from dinosaur import spherical_harmonic


class Spectrum:
    def __init__(self,lat,lon):
        if importlib.metadata.version('dinosaur-dycore')!='1.2.1':raise ValueError('Unpinned Dinosaur version')
        lat=np.asarray(lat);lon=np.asarray(lon)
        if len(lon)!=2*(len(lat)-1):raise ValueError('Expected regular global grid including poles')
        np.testing.assert_allclose(lat,np.linspace(-90,90,len(lat)),atol=1e-8,rtol=0)
        np.testing.assert_allclose(lon,np.arange(len(lon))*360/len(lon),atol=1e-8,rtol=0)
        self.shape=(len(lat),len(lon))
        self.grid=spherical_harmonic.Grid(longitude_wavenumbers=len(lon)//2,
            total_wavenumbers=len(lon)//2,longitude_nodes=len(lon),latitude_nodes=len(lat),
            latitude_spacing='equiangular_with_poles',spherical_harmonics_impl=spherical_harmonic.RealSphericalHarmonics)
        self.basis=self.grid.spherical_harmonics.basis
        self.mask=self.grid.spherical_harmonics.mask

    def coefficients(self,field):
        values=np.asarray(field,dtype=np.float64)
        if values.shape!=self.shape or not np.isfinite(values).all():raise ValueError('Invalid spectrum field')
        weighted=values.T*self.basis.w
        fourier=np.einsum('im,ij->mj',self.basis.f,weighted,optimize=True)
        result=np.einsum('mjl,mj->ml',self.basis.p,fourier,optimize=True)
        return np.where(self.mask,result,0.)

    def power(self,field):
        return np.square(self.coefficients(field)).sum(axis=0)/(4*np.pi)

    def provenance(self):
        source=Path(spherical_harmonic.__file__)
        return dict(dinosaur_version=importlib.metadata.version('dinosaur-dycore'),
            spherical_harmonic_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            basis_dependency_sha256={name:hashlib.sha256(Path(getattr(spherical_harmonic,name).__file__).read_bytes()).hexdigest()
                for name in ('fourier','associated_legendre')},
            evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            latitude_nodes=self.shape[0],longitude_nodes=self.shape[1],
            total_wavenumbers=self.grid.total_wavenumbers,
            normalization='sum_m coefficient(m,l)^2 / (4*pi); includes l=0',
            transform='Dinosaur RealSphericalHarmonics basis and equiangular_with_poles quadrature; float64 NumPy contractions')
