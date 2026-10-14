"""Exercise the real WN2 namespace with a hostile read and runtime identity."""
import argparse
import json
import os
from pathlib import Path
import sys
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'wn2'))
from namespace_inference import run

p=argparse.ArgumentParser()
p.add_argument('--runtime',type=Path)
p.add_argument('--gpu',action='store_true')
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
model=a.output/'model';model.mkdir()
public=a.output/'public';public.mkdir()
secret=a.output/'protected-truth.txt';secret.write_text(uuid.uuid4().hex)
script='''import json, os, pathlib, sys
import numpy, xarray
secret=pathlib.Path(SECRET_PATH)
assert not secret.exists(), 'Host-only truth crossed the boundary'
report=dict(status='passed',python=sys.prefix,numpy=numpy.__version__,xarray=xarray.__version__,
            host_truth_hidden=True,pid_namespace=os.stat('/proc/self/ns/pid').st_ino,
            network_namespace=os.stat('/proc/self/ns/net').st_ino)
assert report['python']=='/opt/weather-python'
assert report['numpy']=='1.26.4'
assert report['xarray']=='2026.1.0'
pathlib.Path(sys.argv[1],'boundary.json').write_text(json.dumps(report))
'''.replace('SECRET_PATH',repr(str(secret.resolve())))
if a.gpu:
 script=script.replace("pathlib.Path(sys.argv[1],'boundary.json')", "import jax, jax.numpy as jnp\nassert all(d.platform=='gpu' for d in jax.devices())\nx=jnp.ones((32,32)); assert float(jax.block_until_ready(x@x)[0,0])==32\nreport['gpu_devices']=[str(d) for d in jax.devices()]\npathlib.Path(sys.argv[1],'boundary.json')")
(model/'check.py').write_text(script)
(model/'infer.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec /opt/weather-python/bin/python check.py "$2"\n')
if a.runtime:os.environ['WEATHER_RUNTIME_ROOT']=str(a.runtime.resolve())
else:os.environ.pop('WEATHER_RUNTIME_ROOT',None)
if not a.gpu:os.environ['CUDA_VISIBLE_DEVICES']=''
try:
 result=run(model.resolve(),public.resolve(),a.output.resolve(),60)
 report=json.loads((result/'boundary.json').read_text())
 assert report['pid_namespace']!=os.stat('/proc/self/ns/pid').st_ino
 assert report['network_namespace']!=os.stat('/proc/self/ns/net').st_ino
 report['scope']='Actual Bubblewrap execution from separate CPU scoring Python; hidden host truth, separate PID/network namespaces, pinned model interpreter and NumPy/xarray'
 (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 print('WN2_MODEL_RUNTIME_AND_NAMESPACE_ISOLATION_OK')
finally:
 secret.unlink()
