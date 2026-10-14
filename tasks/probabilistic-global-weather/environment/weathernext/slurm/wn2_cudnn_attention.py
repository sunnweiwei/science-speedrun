# Adapted from WeatherNext sparse_transformer.py, Copyright 2026 Google LLC.
# Apache-2.0; see wn2/LICENSE.weathernext-code.
"""Diagnostic cuDNN backend; never enabled by canonical model construction."""
import haiku as hk
import jax
import jax.numpy as jnp
from weathernext.utils import sparse_transformer as native


def enable_deterministic_cache_fix():
    """Set the pinned XLA backward flag before its compiled-graph key is made.

    XLA 3d5ece64 sets this only on the first cache miss, leaving repeated
    same-shaped backward calls with a different key. Keep deterministic
    engines required; do not bypass the requirement to avoid the lookup bug.
    See https://github.com/openxla/xla/pull/46500 (upstream draft diagnosis).
    """
    import os
    from jax._src.cudnn import fused_attention_stablehlo as lowering
    if jax.__version__ != '0.6.2':
        raise ValueError('Revalidate this compatibility fix for a different JAX')
    flags=os.environ.get('XLA_FLAGS','').split()
    if '--xla_gpu_exclude_nondeterministic_ops=true' not in flags:
        raise ValueError('Deterministic GPU policy is required')
    original=lowering.create_dot_product_attention_backend_config_base
    if getattr(original,'_wn2_deterministic_cache_fix',False):return
    def config(batch,num_heads,seq_q,seq_kv,dtype,fmha_scale,mask_type,layout,is_bwd):
        result=original(batch,num_heads,seq_q,seq_kv,dtype,fmha_scale,mask_type,layout,is_bwd)
        if is_bwd:result['cudnn_fmha_backend_config']['force_deterministic']=True
        return result
    config._wn2_deterministic_cache_fix=True
    lowering.create_dot_product_attention_backend_config_base=config


def attention(q_input, kv_input, mask, cfg):
    q = native.multihead_linear(q_input, 'q', cfg)
    k = native.multihead_linear(kv_input, 'k', cfg)
    v = native.multihead_linear(kv_input, 'v', cfg)
    batch, blocks, width, heads, depth = q.shape
    if q.dtype != jnp.bfloat16:
        raise ValueError('This experiment requires BF16; no implicit dtype change')
    k = jnp.pad(k, ((0,0),(1,1),(0,0),(0,0),(0,0)))
    v = jnp.pad(v, ((0,0),(1,1),(0,0),(0,0),(0,0)))
    k = jnp.concatenate((k[:,1:-1],k[:,2:],k[:,:-2]),axis=2)
    v = jnp.concatenate((v[:,1:-1],v[:,2:],v[:,:-2]),axis=2)
    allowed = jnp.concatenate(tuple(mask[:,i] for i in range(3)),axis=-1).astype(bool)
    allowed = jnp.broadcast_to(allowed,(batch,blocks,1,width,3*width))
    # cuDNN backward with a mask requires even Q/K lengths. Added keys are
    # excluded; added queries are discarded before the output projection.
    pad = width % 2
    q = jnp.pad(q,((0,0),(0,0),(0,pad),(0,0),(0,0)))
    k = jnp.pad(k,((0,0),(0,0),(0,pad),(0,0),(0,0)))
    v = jnp.pad(v,((0,0),(0,0),(0,pad),(0,0),(0,0)))
    allowed = jnp.pad(allowed,((0,0),(0,0),(0,0),(0,pad),(0,pad)))
    if pad:
        allowed = allowed.at[:,:,:,width,0].set(True)
    x = jax.nn.dot_product_attention(
        q.reshape(batch*blocks,width+pad,heads,depth),
        k.reshape(batch*blocks,3*width+pad,heads,depth),
        v.reshape(batch*blocks,3*width+pad,heads,cfg.value_size),
        mask=allowed.reshape(batch*blocks,1,width+pad,3*width+pad),
        scale=cfg.key_size**-0.5,implementation='cudnn')
    x = x.reshape(batch,blocks,width+pad,heads*cfg.value_size)[:,:,:width]
    init = hk.initializers.VarianceScaling(cfg.attn_winit_final_mult/cfg.num_layers)
    return hk.Linear(cfg.d_model,name='mha_final',w_init=init)(x)
