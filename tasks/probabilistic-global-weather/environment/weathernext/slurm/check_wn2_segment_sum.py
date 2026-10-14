#!/usr/bin/env python3
"""Diagnose deterministic native GPU aggregation, including its first row."""
import json
import jax
import jax.numpy as jnp
import numpy as np
from weathernext.utils import gather_scatter_ops

assert jax.default_backend()=='gpu'
rows=4096
indices=np.repeat(np.arange(rows,dtype=np.int32),3)
for dtype in (jnp.float32,jnp.bfloat16):
    op=jax.jit(gather_scatter_ops.create_static_sorted_segment_sum(indices,rows))
    x=jnp.ones((len(indices),1,768),dtype=dtype)
    for repeat in range(20):
        result=np.asarray(op(x).block_until_ready(),dtype=np.float32)
        wrong=np.argwhere(result!=3.)
        print(json.dumps(dict(dtype=str(dtype),repeat=repeat,wrong=int(len(wrong)),
            first_row_min=float(result[0].min()),first_row_max=float(result[0].max()))),flush=True)
        if len(wrong):raise ValueError(f'Native segment sum differs from exact integer oracle: {wrong[:5]}')
print('WN2_NATIVE_GPU_SEGMENT_SUM_REPEAT_OK')
