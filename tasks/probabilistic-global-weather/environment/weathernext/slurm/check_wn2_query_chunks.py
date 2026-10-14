#!/usr/bin/env python3
"""Native attention value/adjoint comparison; BF16 errors are reported, not waived."""
import argparse
import json
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from weathernext.utils import casting, sparse_transformer as native
from wn2_query_chunks import attention


def error(a, b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    return dict(max_abs=float(np.max(np.abs(a-b))),
                relative_l2=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-30)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--require-gpu',action='store_true')
    parser.add_argument('--boolean-mask',action='store_true')
    parser.add_argument('--packed-mask',action='store_true')
    args=parser.parse_args()
    if args.packed_mask:args.boolean_mask=True
    if args.require_gpu and jax.default_backend()!='gpu':raise ValueError('GPU required')
    rows=[]
    failures=[]
    for dtype in (jnp.float32,jnp.bfloat16):
        for batch,width,chunk in [(1,8,4),(2,9,4),(1,33,8)]:
            cfg=native._ModelConfig(num_layers=2,d_model=24,num_heads=3,mask_block_size=width)
            rng=np.random.default_rng(14)
            mask=rng.random((1,3,3,1,width,width))>.3
            mask[0,0,:,0,np.arange(width),np.arange(width)]=True
            mask=jnp.asarray(mask)
            if args.boolean_mask:
                from scipy import sparse
                from wn2_boolean_masks import mask_block_diags
                if args.packed_mask:
                    from wn2_packed_masks import mask_block_diags
                nodes=3*width-2
                graph=rng.random((nodes,nodes))>.3
                np.fill_diagonal(graph,True)
                graph=sparse.csr_matrix(graph)
                old_mask=native.mask_block_diags(graph,2,width)
                new_mask=mask_block_diags(graph,2,width)
                np.testing.assert_array_equal(np.asarray(old_mask)!=0,np.asarray(new_mask))
            def model(x,chunked):
                with casting.bfloat16_variable_view(enabled=dtype==jnp.bfloat16):
                    if args.boolean_mask:
                        candidate_mask=mask_block_diags(graph,2,width) if args.packed_mask and chunked else new_mask
                        return native.triblockdiag_mha(x,x,candidate_mask if chunked else old_mask,cfg)
                    if chunked:return attention(x,x,mask,cfg,query_chunk=chunk)
                    return native.triblockdiag_mha(x,x,mask,cfg)
            ref=hk.transform(lambda x:model(x,False));new=hk.transform(lambda x:model(x,True))
            x=jnp.asarray(rng.normal(size=(batch,3,width,24)),dtype=dtype)
            key=jax.random.PRNGKey(0);p=ref.init(key,x);q=new.init(key,x)
            assert jax.tree.structure(p)==jax.tree.structure(q)
            for u,v in zip(jax.tree.leaves(p),jax.tree.leaves(q)):np.testing.assert_array_equal(u,v)
            def run(net):
                def objective(p,x):
                    y=net.apply(p,key,x)
                    return jnp.sum(jnp.sin(y.astype(jnp.float32))),y
                return jax.jit(jax.value_and_grad(objective,argnums=(0,1),has_aux=True))
            baseline=run(ref);candidate=run(new)
            a=baseline(p,x);b=candidate(p,x);repeat=candidate(p,x)
            for u,v in zip(jax.tree.leaves(b),jax.tree.leaves(repeat)):
                assert np.asarray(u).tobytes()==np.asarray(v).tobytes(),'In-process nondeterminism'
            errors=[error(u,v) for u,v in zip(jax.tree.leaves(a),jax.tree.leaves(b))]
            if args.boolean_mask:
                for u,v in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
                    if np.asarray(u).tobytes()!=np.asarray(v).tobytes():
                        failures.append(dict(dtype=str(dtype),width=width,reason='Mask storage changed numerical result'))
                        break
            if dtype==jnp.float32:
                try:
                    for u,v in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
                        np.testing.assert_allclose(u,v,rtol=2e-5,atol=2e-5)
                except AssertionError as error_message:
                    failures.append(dict(dtype=str(dtype),width=width,reason=str(error_message)))
            rows.append(dict(dtype=str(dtype),batch=batch,width=width,query_chunk=chunk,
                             forward=error(a[0][1],b[0][1]),
                             parameter_gradient_errors=[error(u,v) for u,v in zip(jax.tree.leaves(a[1][0]),jax.tree.leaves(b[1][0]))],
                             input_gradient=error(a[1][1],b[1][1]),
                             repeat_exact=True,parameters_initialized_exactly=True))
    report=dict(status='failed' if failures else 'boolean_mask_exact_passed' if args.boolean_mask else 'fp32_oracle_passed_bf16_measured',backend=jax.default_backend(),
                failures=failures,
                experiment='packed_mask' if args.packed_mask else 'boolean_mask' if args.boolean_mask else 'query_chunks',
                devices=[d.device_kind for d in jax.devices()],cases=rows,
                scope='Synthetic native attention only; BF16 measurements do not establish full-model equivalence, replay or production acceptance')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
