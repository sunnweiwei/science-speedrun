"""Rejected memory experiment: increased native eight-step GPU memory.

Retained only for the CPU/GPU numerical oracle; not used by the trainer.
"""
import haiku as hk
import jax.numpy as jnp

_CHUNKED_DENSE = False


def enable_chunked_dense(chunk_rows=131072):
    """Bound row-local MLP/normalization activations without dropping rows.

    Large graph node/edge axes are scanned in fixed blocks. DenseLayer's
    normalization is over features; the final padded rows are discarded.
    Parameters, real rows, norm conditioning and full derivatives are retained.
    """
    global _CHUNKED_DENSE
    if _CHUNKED_DENSE:return
    from weathernext.utils import dense
    original = dense.DenseLayer.__call__

    def call(self, inputs, norm_conditioning=None):
        count = inputs.shape[0]
        if count <= chunk_rows:
            return original(self, inputs, norm_conditioning)
        padding = (-count) % chunk_rows
        def blocks(value):
            padded = jnp.pad(value, [(0, padding)] + [(0, 0)] * (value.ndim-1))
            return padded.reshape((-1, chunk_rows, *value.shape[1:]))
        per_row = (norm_conditioning is not None and
                   norm_conditioning.ndim == inputs.ndim and
                   norm_conditioning.shape[0] == count)
        def step(carry, values):
            data, conditioning = values
            return carry, original(self, data,
                                   conditioning if per_row else norm_conditioning)
        _, result = hk.scan(hk.remat(step), (),
                            (blocks(inputs), blocks(norm_conditioning) if per_row else None))
        return result.reshape((-1, *result.shape[2:]))[:count]

    dense.DenseLayer.__call__ = call
    _CHUNKED_DENSE = True
