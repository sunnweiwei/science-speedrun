"""Rebuild bundled FPS sources on a CPU Slurm node without invoking a GPU."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import external,sha256,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','package','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();root=external(a.root);package=external(a.package);out=external(a.output)
    if root not in out.parents:raise ValueError('Build outputs must remain under budget root')
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION')!=os.environ.get('PATHFINDER_CPU_PARTITION', 'cpu'):
        raise ValueError('Use a CPU cpu allocation for this build audit')
    manifest=json.loads((package/'package.json').read_text())
    selected={k:v for k,v in manifest['files'].items() if k.startswith('vendor/pointnext/')}
    builder='runtime/pathfinder/build_point_ops.py';selected[builder]=manifest['files'][builder]
    for name,digest in selected.items():
        if sha256(package/name)!=digest:raise ValueError('Bundled source integrity mismatch')
    from fetch_axonem import reserve
    reserve(root,100_000_000);out.mkdir(parents=True,exist_ok=False)
    for name in selected:
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(package/name,target)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',TORCH_CUDA_ARCH_LIST='8.6',MAX_JOBS='2',
        TMPDIR=str(root/'tmp'),TORCH_EXTENSIONS_DIR=str(out/'extension-cache'),PYTHONDONTWRITEBYTECODE='1')
    if env.get('CUDA_HOME'):env['PATH']=str(Path(env['CUDA_HOME'])/'bin')+os.pathsep+env.get('PATH','')
    compiler=shutil.which('nvcc',path=env['PATH'])
    if compiler is None:raise RuntimeError('No CUDA compiler accessible on this CPU node')
    compiler_version=subprocess.check_output([compiler,'--version'],env=env,text=True)
    started=time.monotonic()
    with (out/'build.log').open('w') as log:
        result=subprocess.run([sys.executable,str(out/builder),'--root',str(out)],env=env,stdout=log,stderr=subprocess.STDOUT)
    receipt=dict(schema='pathfinder-fps-build-audit/v1',package_manifest_sha256=sha256(package/'package.json'),
        source_hashes=selected,slurm_job_id=os.environ['SLURM_JOB_ID'],seconds=time.monotonic()-started,
        exit_code=result.returncode,cuda_visible_devices='',cuda_arch='8.6',gpu_execution=False,
        cuda_home=env.get('CUDA_HOME'),nvcc_version=compiler_version,
        build_log_sha256=sha256(out/'build.log'),kernel_numerical_parity_verified=False)
    if result.returncode==0:
        receipt['extension_sha256']=sha256(out/'point-ops/pathfinder_fps.so')
        # torch cpp_extension.load already loads the freshly built .so in its child.
        receipt['compiled_and_loaded']=True
    write_json(out/'receipt.json',receipt)
    if result.returncode:raise RuntimeError('Rebuild failed; inspect retained build.log')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
