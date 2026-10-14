"""Experimental bit-packed native masks; unpack behind an optimization barrier.

Experimental until full-native execution and replay are verified. Adapted from
WeatherNext sparse_transformer.py, Copyright 2026 Google LLC, Apache 2.0;
see wn2/LICENSE.weathernext-code. Only mask storage/construction is changed.
"""
import jax
import jax.numpy as jnp
import numpy as np
from weathernext.utils import sparse_transformer as native


def mask_block_diags(mask, num_padding_nodes, block_size):
    if mask.dtype != np.bool_:
        raise ValueError('Expected native boolean adjacency power')
    mask = native._pad_mask(mask, (num_padding_nodes, num_padding_nodes)).tocsr()
    assert mask.shape[0] == mask.shape[1] and mask.shape[0] % block_size == 0
    count = mask.shape[0]//block_size
    def block(row, column):
        if row < 0 or column < 0 or row >= count or column >= count:
            return jnp.zeros((block_size, block_size), dtype=jnp.bool_)
        dense = mask[row*block_size:(row+1)*block_size,
                     column*block_size:(column+1)*block_size].toarray()
        packed = np.packbits(dense.reshape(-1), bitorder='little')
        packed = jax.lax.optimization_barrier(jnp.asarray(packed))
        return jnp.unpackbits(packed, count=block_size*block_size,
                              bitorder='little').reshape(block_size,block_size).astype(jnp.bool_)
    diagonals = [jnp.stack([block(i,i+offset) for i in range(count)]) for offset in (0,1,-1)]
    return jnp.expand_dims(jnp.stack(diagonals), (0,3))


def enable():
    native.mask_block_diags = mask_block_diags
