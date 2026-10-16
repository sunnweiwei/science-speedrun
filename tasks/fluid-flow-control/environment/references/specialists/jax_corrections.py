"""Explicit, opt-in numerical reconstruction corrections; not silent upstream edits."""

import hashlib
import inspect
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from hydrogym.jax.envs import kolmogorov


def two_thirds(field):
    # rFFT layout for an even physical grid. Keep |k| < N/3 symmetrically;
    # .at[].set() is functional in JAX and cannot be discarded as upstream does.
    nx, half_y = field.shape
    ny = 2 * (half_y - 1)
    kx = jnp.fft.fftfreq(nx) * nx
    ky = jnp.arange(half_y)
    mask = (jnp.abs(kx[:, None]) < nx / 3) & (ky[None, :] < ny / 3)
    return jnp.where(mask, field, jnp.zeros_like(field))


def control_curl(self, omega_hat, control_field=None):
    if control_field is None:
        return jnp.zeros_like(omega_hat)
    cfx, cfy = control_field
    # Same velocity-force -> vorticity curl as related author's original solver
    # and this native class's own background-forcing implementation.
    return 2j * jnp.pi * (self.kx * jnp.fft.rfftn(cfy) - self.ky * jnp.fft.rfftn(cfx))


def apply_corrections(mode):
    from jax_source import verify
    verify()
    if mode not in ("dealias", "dealias-curl"):
        raise ValueError("Unknown explicit solver correction")
    old_dealias = inspect.getsource(kolmogorov.dealiasing)
    old_curl = inspect.getsource(kolmogorov.PseudoSpectralNavierStokes2D.control_term)
    if "advection_term.at[kn : 2 * kn, :].set(0.0)" not in old_dealias:
        raise RuntimeError("Pinned no-op dealiasing implementation changed")
    if "return self.kx * cfy_hat - self.ky * cfx_hat" not in old_curl:
        raise RuntimeError("Pinned control-curl implementation changed")
    test = jnp.ones((64, 33), dtype=jnp.complex64)
    np.testing.assert_array_equal(np.asarray(kolmogorov.dealiasing(test)), np.asarray(test))
    filtered = np.asarray(jax.jit(two_thirds)(test))
    assert filtered[0, 0] == filtered[21, 21] == filtered[-21, 21] == 1
    assert filtered[22, 0] == filtered[-22, 0] == filtered[0, 22] == 0
    np.testing.assert_array_equal(filtered, np.asarray(two_thirds(jnp.asarray(filtered))))
    kolmogorov.dealiasing = two_thirds
    if mode == "dealias-curl":
        kolmogorov.PseudoSpectralNavierStokes2D.control_term = control_curl
    return {"mode": mode, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "native_dealiasing_was_identity": True,
            "dealiasing": "retain symmetric Fourier wavenumbers |kx|<Nx/3, ky<Ny/3",
            "control_curl": "restore 2*pi*i as in related-author source" if mode == "dealias-curl" else "unchanged native",
            "native_dealiasing_source_sha256": hashlib.sha256(old_dealias.encode()).hexdigest(),
            "native_control_source_sha256": hashlib.sha256(old_curl.encode()).hexdigest()}


if __name__ == "__main__":
    import json
    jax.config.update("jax_enable_x64", True)
    audit = apply_corrections("dealias-curl")
    flow = kolmogorov.FlowConfig(grid_size=(64, 64))
    equation = kolmogorov.PseudoSpectralNavierStokes2D(flow)
    y = jnp.arange(64) * (2 * jnp.pi / 64)
    forcing = jnp.broadcast_to(jnp.sin(4 * y)[None, :], (64, 64))
    result = equation.control_term(jnp.zeros((64, 33)), (forcing, jnp.zeros_like(forcing)))
    expected = jnp.broadcast_to(-4 * jnp.cos(4 * y)[None, :], (64, 64))
    error = float(jnp.max(jnp.abs(jnp.fft.irfftn(result) - expected)))
    np.testing.assert_allclose(np.asarray(jnp.fft.irfftn(result)), np.asarray(expected), atol=1e-10)
    print(json.dumps(dict(audit, analytic_control_curl_max_error=error)), flush=True)
