#!/usr/bin/env python3
"""Measure BF16 backend differences, without declaring model equivalence."""
import argparse
import json
import time
from pathlib import Path
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from weathernext.utils import casting, sparse_transformer as native
from wn2_cudnn_attention import attention
from check_wn2_query_chunks import error


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--native-coarse',action='store_true',help='Actual coarse block width/head count, synthetic valid mask')
    parser.add_argument('--native-fine',action='store_true',help='Actual fine block width/head count, synthetic valid mask')
    parser.add_argument('--pmap-remat',action='store_true',help='Isolate pmap and Haiku rematerialization compilation')
    parser.add_argument('--scan-ensemble',action='store_true',help='Add checkpointed two-member scan as in native network')
    parser.add_argument('--layers',type=int,default=1)
    parser.add_argument('--deterministic-cache-fix',action='store_true')
    args=parser.parse_args()
    if args.deterministic_cache_fix:
        from wn2_cudnn_attention import enable_deterministic_cache_fix
        enable_deterministic_cache_fix()
    assert jax.default_backend()=='gpu'
    report=dict(status='incomplete',layers=args.layers,deterministic_cache_fix=args.deterministic_cache_fix,
                scope='BF16 synthetic diagnostic only; no equivalence or production acceptance',cases=[])
    try:
        cases=[(1,4,10273,768,6)] if args.native_fine else [(1,2,5153,768,6)] if args.native_coarse else [(1,3,8,24,3),(2,3,9,24,3),(1,3,128,768,6)]
        for batch,blocks,width,dim,heads in cases:
            cfg=native._ModelConfig(num_layers=24,d_model=dim,num_heads=heads,mask_block_size=width)
            rng=np.random.default_rng(14)
            mask=rng.random((1,3,blocks,1,width,width))>.3
            mask[0,0,:,0,np.arange(width),np.arange(width)]=True
            mask=jnp.asarray(mask)
            def model(x,fused):
                with casting.bfloat16_variable_view(enabled=True):
                    for _ in range(args.layers):
                        x=(attention if fused else native.triblockdiag_mha)(x,x,mask,cfg)
                    return x
            def members(x,fused):
                if not args.scan_ensemble:return model(x,fused)
                shared=hk.to_module(lambda member:model(member,fused))(name='shared_attention')
                def step(carry,member):return carry,shared(member)
                return hk.scan(hk.remat(step),(),jnp.stack((x,x*jnp.asarray(.9,x.dtype))))[1]
            ref_fn=lambda x:members(x,False)
            new_fn=lambda x:members(x,True)
            ref=hk.transform(lambda x:hk.remat(ref_fn)(x)) if args.pmap_remat else hk.transform(ref_fn)
            new=hk.transform(lambda x:hk.remat(new_fn)(x)) if args.pmap_remat else hk.transform(new_fn)
            x=jnp.asarray(rng.normal(size=(batch,blocks,width,dim)),dtype=jnp.bfloat16)
            key=jax.random.PRNGKey(0)
            p=ref.init(key,x);q=new.init(key,x)
            assert jax.tree.structure(p)==jax.tree.structure(q)
            for u,v in zip(jax.tree.leaves(p),jax.tree.leaves(q)):
                np.testing.assert_array_equal(u,v)
            def compile(net):
                def objective(p,x):
                    y=net.apply(p,key,x)
                    return jnp.sum(jnp.sin(y.astype(jnp.float32))),y
                wrapper=jax.pmap if args.pmap_remat else jax.jit
                return wrapper(jax.value_and_grad(objective,argnums=(0,1),has_aux=True))
            if args.pmap_remat:
                assert len(jax.local_devices())==1
                p=jax.tree.map(lambda a:a[None],p);x=x[None]
            baseline=compile(ref).lower(p,x).compile();candidate=compile(new).lower(p,x).compile()
            memory=[]
            for fn in (baseline,candidate):
                stats=fn.memory_analysis()
                memory.append(None if stats is None else {name:getattr(stats,name) for name in
                    ('argument_size_in_bytes','output_size_in_bytes','alias_size_in_bytes','temp_size_in_bytes')})
            a=jax.block_until_ready(baseline(p,x));b=jax.block_until_ready(candidate(p,x))
            timings=[]
            for fn in (baseline,candidate):
                start=time.monotonic()
                for _ in range(5): c=jax.block_until_ready(fn(p,x))
                timings.append((time.monotonic()-start)/5)
            exact=all(np.asarray(u).tobytes()==np.asarray(v).tobytes() for u,v in zip(jax.tree.leaves(b),jax.tree.leaves(c)))
            report['cases'].append(dict(batch=batch,blocks=blocks,width=width,dim=dim,heads=heads,pmap_remat=args.pmap_remat,scan_ensemble=args.scan_ensemble,
                forward=error(a[0][1],b[0][1]),input_gradient=error(a[1][1],b[1][1]),
                parameter_gradient_errors=[error(u,v) for u,v in zip(jax.tree.leaves(a[1][0]),jax.tree.leaves(b[1][0]))],
                repeat_exact=exact,native_seconds=timings[0],cudnn_seconds=timings[1],
                native_compiler_memory=memory[0],cudnn_compiler_memory=memory[1]))
            assert exact,'Repeated GPU execution differs'
        report['status']='measured_not_accepted'
    except Exception as exc:
        report['status']='failed';report['error']=repr(exc)
        raise
    finally:
        args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
