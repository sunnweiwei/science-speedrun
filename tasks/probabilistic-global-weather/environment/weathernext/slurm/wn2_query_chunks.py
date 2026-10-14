# Adapted from WeatherNext sparse_transformer.py, Copyright 2026 Google LLC.
# Licensed under the Apache License, Version 2.0; see wn2/LICENSE.weathernext-code.
# Modified to compute independent query rows in checkpointed chunks.
"""Experimental query-chunked native attention; not enabled in production.

Every query still attends to all keys in the same three native blocks. Only
independent query rows are chunked. Parameter names/projections and the native
softmax expression are retained. GPU value/gradient and performance evidence
is required before promotion.
"""
import haiku as hk
import jax
import jax.numpy as jnp
from weathernext.utils import sparse_transformer as native
from weathernext.utils import sparse_transformer_utils as utils


def attention(q_input, kv_input, mask, cfg, query_chunk=128):
    q = native.multihead_linear(q_input, 'q', cfg)
    k = native.multihead_linear(kv_input, 'k', cfg)
    v = native.multihead_linear(kv_input, 'v', cfg)
    k = jnp.pad(k, ((0, 0), (1, 1), (0, 0), (0, 0), (0, 0)))
    v = jnp.pad(v, ((0, 0), (1, 1), (0, 0), (0, 0), (0, 0)))
    width = q.shape[2]
    chunk = min(query_chunk, width)
    count = (width + chunk - 1)//chunk
    padding = count*chunk-width
    q = jnp.pad(q, ((0, 0), (0, 0), (0, padding), (0, 0), (0, 0)))
    mask_padding = [(0, 0)]*mask.ndim
    mask_padding[-2] = (0, padding)
    mask = jnp.pad(mask, mask_padding)

    def rows(index):
        queries = jax.lax.dynamic_slice_in_dim(q, index*chunk, chunk, axis=2)
        row_mask = jax.lax.dynamic_slice_in_dim(mask, index*chunk, chunk, axis=mask.ndim-2)
        logits = tuple(jnp.einsum('bnqhd,bnkhd->bnhqk', queries, keys)*cfg.key_size**-0.5
                       for keys in (k[:, 1:-1], k[:, 2:], k[:, :-2]))
        logits = tuple(jnp.where(row_mask[:, i], value, -1e30) for i, value in enumerate(logits))
        weights = utils.wrap_fn_for_upcast_downcast(logits, native.triblockdiag_softmax)
        outputs = tuple(jnp.einsum('bnhqk,bnkhd->bnqhd', weight, values)
                        for weight, values in zip(weights, (v[:, 1:-1], v[:, 2:], v[:, :-2])))
        return outputs[0] + outputs[1] + outputs[2]

    x = jax.lax.map(jax.checkpoint(rows), jnp.arange(count))
    x = x.transpose(1, 2, 0, 3, 4, 5).reshape(q.shape)[:, :, :width]
    x = x.reshape(x.shape[:-2] + (cfg.num_heads*cfg.value_size,))
    initializer = hk.initializers.VarianceScaling(cfg.attn_winit_final_mult/cfg.num_layers)
    return hk.Linear(cfg.d_model, name='mha_final', w_init=initializer)(x)
