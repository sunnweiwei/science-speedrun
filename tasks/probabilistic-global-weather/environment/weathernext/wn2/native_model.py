"""Full WN2 stage configuration with task-owned training-only normalization."""
import copy
import dataclasses
import functools
import json
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
import xarray_jax
from weathernext.utils import data_utils, fiddle_config_io, sparse_transformer, xarray_tree

from training_data import sha256
from wn2_validation import construct_predictor, enable_transformer_remat

_REMAT_ENABLED = False


# Mask construction adapted from WeatherNext, Copyright 2026 Google LLC.
# Apache-2.0; see LICENSE.weathernext-code. Values and topology are unchanged.
# Packed constants avoid the protobuf 2 GB executable serialization limit.
def _packed_mask_block_diags(mask, num_padding_nodes, block_size):
    if mask.dtype != np.bool_:
        raise ValueError('Expected native boolean adjacency power')
    mask = sparse_transformer._pad_mask(mask, (num_padding_nodes, num_padding_nodes)).tocsr()
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


sparse_transformer.mask_block_diags = _packed_mask_block_diags

# cuDNN attention adapted from WeatherNext (Apache-2.0). Same projections,
# neighborhood and parameter initialization; fused BF16 arithmetic differs.
# Native AR1/AR8 three-process state replay and development package verified.
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


def _cudnn_attention(q_input, kv_input, mask, cfg):
    q = sparse_transformer.multihead_linear(q_input, 'q', cfg)
    k = sparse_transformer.multihead_linear(kv_input, 'k', cfg)
    v = sparse_transformer.multihead_linear(kv_input, 'v', cfg)
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


enable_deterministic_cache_fix()
sparse_transformer.triblockdiag_mha = _cudnn_attention



def stage_config(stage, statistics):
    statistics=Path(statistics)
    manifest=json.loads((statistics/'manifest.json').read_text())
    if manifest['status']!='passed':raise ValueError('Training statistics incomplete')
    group=stage['name'] if stage['name'].startswith('era5') else 'hres_025deg_6h'
    config=copy.deepcopy(fiddle_config_io.get_fiddle_config_by_name('weathernext2/configs/WeatherNext2'))
    config.task=dataclasses.replace(config.task,input_duration=f"{2*stage['cadence_hours']}h")
    noisy=config.predictor_kwargs['noisy_function_kwargs']
    noisy['mesh_num_splits']=stage['mesh_splits']
    # Native rematerialization switches preserve the network and parameter
    # names while recomputing the large encoder/decoder activations in backward.
    noisy['remat_grid_to_mesh_gnn']=True
    noisy['remat_mesh_gnn']=True
    noisy['remat_mesh_to_grid_gnn']=True
    noisy['mesh_model_ctor'].keywords['transformer_kwargs']['attention_type']='triblockdiag_mha'
    points=noisy['points_to_mesh_model_ctor']
    kw=copy.deepcopy(points.keywords)
    kw['deep_gnn_kwargs']['aggregate_normalization']=stage['encoder_aggregate_normalization']
    noisy['points_to_mesh_model_ctor']=functools.partial(points.func,**kw)
    def load(name):
        relative=f'{group}/{name}'
        if sha256(statistics/relative)!=manifest['files'][relative]:raise ValueError('Statistics hash mismatch')
        return xr.load_dataset(statistics/relative)
    original=config.predictor_wrappers[0]['kwargs']
    def replace_values(key,filename):
        values=load(filename)
        # Native bounded/activated outputs such as cyclone existence are
        # intentionally absent from these maps. Preserve that normalization
        # scope while replacing every used numerical statistic.
        return values[[name for name in values if name in original[key]]]
    config.predictor_wrappers[0]['kwargs']=dict(mean_by_level=replace_values('mean_by_level','mean.nc'),
        stddev_by_level=replace_values('stddev_by_level','std.nc'),
        diffs_stddev_by_level=replace_values('diffs_stddev_by_level','diff_std.nc'))
    config.predictor_wrappers[1]['kwargs']['fill_value']=load('minimum.nc')
    assert config.predictor_wrappers[2]['kwargs']['gradient_checkpointing']
    return config


def transforms(config):
    global _REMAT_ENABLED
    if not _REMAT_ENABLED:
        enable_transformer_remat()
        _REMAT_ENABLED = True
    inference=dataclasses.replace(config,predictor_wrappers=config.predictor_wrappers[:-1])
    @hk.transform
    def forward(inputs,template,forcings):
        return construct_predictor(inference,'bfloat16')(inputs,targets_template=template,forcings=forcings)
    @hk.transform
    def loss(inputs,targets,forcings):
        value,diagnostics=construct_predictor(config,'bfloat16').loss(inputs,targets,forcings)
        return xarray_tree.map_structure(lambda a:xarray_jax.unwrap_data(a.mean(),require_jax=True),(value,diagnostics))
    return forward,loss


def extract(dataset,config,stage):
    return data_utils.extract_inputs_targets_forcings(dataset,
        target_lead_times=slice(f"{stage['cadence_hours']}h",f"{stage['cadence_hours']*stage['ar_steps']}h"),
        **dataclasses.asdict(config.task))
