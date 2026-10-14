"""Record each native compilation-cache key component without overriding keys."""
import argparse
import functools
import hashlib
import inspect
import json
from pathlib import Path

import jax
import numpy as np
from jax._src import cache_key
from weathernext.utils import checkpoint
from weathernext.weathernext2 import fgn
from native_model import stage_config,transforms
from plan_schedule import STAGES
from predict import context,forcing,template

def install_capture(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    original=cache_key.get
    records=[]
    @functools.wraps(original)
    def capture(*args,**kwargs):
        bound=inspect.signature(original).bind(*args,**kwargs);bound.apply_defaults();v=bound.arguments
        module=v['module'];name=str(module.operation.attributes['sym_name'])
        key=original(*args,**kwargs)
        if 'apply_fn' in name:
            pieces={}
            options=v['compile_options']
            (out/f'options-{len(records)}.pb').write_bytes(options.SerializeAsString())
            (out/f'debug-options-{len(records)}.txt').write_text(str(options.executable_build_options.debug_options))
            class SerializedOptions:
                def update(self, value):
                    (out/f'normalized-options-{len(records)}.pb').write_bytes(value)
            cache_key._hash_serialized_compile_options(SerializedOptions(),options,strip_device_assignment=True)
            for label,fn in [
                ('computation',lambda h:cache_key._hash_computation(h,module,v['ignore_callbacks'])),
                ('backend',lambda h:cache_key._hash_platform(h,v['backend'])),
                ('flags',lambda h:cache_key._hash_xla_flags(h,cache_key.get_flag_prefixes())),
                ('options',lambda h:cache_key._hash_serialized_compile_options(h,v['compile_options'],strip_device_assignment=True)),
                ('accelerator',lambda h:cache_key._hash_accelerator_config(h,v['devices']))]:
                h=hashlib.sha256();fn(h);pieces[label]=h.hexdigest()
            (out/f'computation-{len(records)}.mlir').write_text(module.operation.get_asm(large_elements_limit=5,enable_debug_info=False))
            records.append(dict(key=key,pieces=pieces));(out/'cache-keys.json').write_text(json.dumps(records,indent=2)+'\n')
        return key
    cache_key.get=capture


def main():
    p=argparse.ArgumentParser();p.add_argument("--phase",required=True);a=p.parse_args()
    out=Path("/run")/a.phase;install_capture(out)
    config=stage_config(STAGES[-1],Path('/model/statistics'));forward,_=transforms(config)
    with open('/model/weights.npz','rb') as stream:params=jax.device_put(checkpoint.load(stream,fgn.CheckPoint).params)
    init,inputs=context(Path('/data/inputs-000.nc'),config.task)
    target=template(inputs,config.task);forces=forcing(inputs,init,6,config.task)
    key=jax.random.PRNGKey(0)
    result=jax.jit(forward.apply)(params,key,inputs,target,forces);jax.block_until_ready(result)
    values={name:hashlib.sha256(np.asarray(result[name]).tobytes()).hexdigest() for name in result}
    (out/'prediction-hashes.json').write_text(json.dumps(values,indent=2)+'\n')
    print(json.dumps(dict(phase=a.phase,first_wind=float(np.asarray(result['10m_u_component_of_wind']).ravel()[0]),keys=json.loads((out/'cache-keys.json').read_text()))),flush=True)


if __name__=="__main__":main()
