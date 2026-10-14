"""Explicit cadence transfer preserves weights and nonzero Adam moments."""
from pathlib import Path
import re
import tempfile
import jax
import jax.numpy as jnp
import numpy as np
import optax
from wn2_distributed import save_state,load_state

params={'module':{f'w_time={seconds}':jnp.asarray([i+.2,i-.3],jnp.float32) for i,seconds in enumerate((-43200,0,43200))}}
optim=optax.adamw(1e-3,b2=.95)
grad=jax.tree.map(lambda a:a*.7+.1,params)
_,state=optim.update(grad,optim.init(params),params)
def transfer(value):
    if isinstance(value,dict):
        return {re.sub(r'(?<=_time=)-?\d+',lambda m:str(int(m[0])//2),k):transfer(v) for k,v in value.items()}
    if isinstance(value,tuple):
        children=[transfer(v) for v in value]
        return type(value)(*children) if hasattr(value,'_fields') else tuple(children)
    return value
expected=transfer((params,state));template=(expected[0],optim.init(expected[0]))
with tempfile.TemporaryDirectory() as tmp:
    path=Path(tmp)/'state.npz'
    pp,ss=jax.device_put_replicated((params,state),jax.local_devices())
    save_state(path,pp,ss,{'schema':'cadence-test','cadence_hours':12})
    restored,_=load_state(path,template,{'schema':'cadence-test'},cadence_transition=(12,6))
    assert jax.tree.structure(restored)==jax.tree.structure(expected)
    for a,b in zip(jax.tree.leaves(restored),jax.tree.leaves(expected)):
        np.testing.assert_array_equal(a,b)
    wrong={'module':{key.replace('w_','wrong_'):value for key,value in template[0]['module'].items()}}
    for bad_template,transition in ((template,None),((wrong,optim.init(wrong)),(12,6))):
        try:load_state(path,bad_template,{'schema':'cadence-test'},cadence_transition=transition)
        except ValueError:pass
        else:raise AssertionError('Unmapped or incorrect state identities accepted')
print('WN2_CADENCE_PARAMETER_AND_ADAM_TRANSFER_EXACT_OK')
