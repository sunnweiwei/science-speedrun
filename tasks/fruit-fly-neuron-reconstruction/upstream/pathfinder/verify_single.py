"""Offline verifier for the selected protocol, with isolated prediction and scoring."""
import argparse
import json
import math
import pickle
import hashlib
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import sha256, write_json

CODE=Path(__file__).resolve().parent


class InvalidSubmission(ValueError):
    pass


def stage_weights(source, target):
    """Accept data-only checkpoints. Never import package runtime or native code."""
    source=Path(source).resolve();target=Path(target)
    try:
        manifest_path=source/'package.json'
        if manifest_path.stat().st_size>4*2**20:raise ValueError('Manifest exceeds 4MiB')
        manifest_bytes=manifest_path.read_bytes()
        manifest=json.loads(manifest_bytes)
        if manifest['schema']!='pathfinder-model-package/v1' or 'ffn_ensemble' in manifest:
            raise ValueError('One FFN, one SENSE and one SHAPE checkpoint required')
        if manifest.get('inference',{}).get('search_threshold')!=0.8:
            raise ValueError('Protocol requires SHAPE threshold 0.8')
        files={}
        for stage in ('ffn','sense','shape'):
            name=f'weights/{stage}.pt';path=source/name
            if path.is_symlink() or path.resolve()!=path or not path.is_file() or path.stat().st_size>2**30:
                raise ValueError('Regular in-package checkpoint of at most 1GiB required')
            if sha256(path)!=manifest['files'][name]:raise ValueError('Checkpoint hash mismatch')
            files[name]=manifest['files'][name]
        target.mkdir(parents=True)
        for name in files:
            out=target/name;out.parent.mkdir(exist_ok=True);shutil.copyfile(source/name,out)
            if sha256(out)!=files[name]:raise ValueError('Checkpoint changed during staging')
        staged=dict(schema=manifest['schema'],files=files,models=manifest['models'],
                    inference=dict(search_threshold=0.8))
        write_json(target/'package.json',staged)
        from model_package import verify
        try:
            verify(target)  # trusted classes, weights_only=True, no submitted imports
        except RuntimeError as error:
            if 'out of memory' in str(error).lower():raise
            raise InvalidSubmission(str(error)) from error
        return dict(submitted_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),weight_sha256=files,
                    submitted_code_executed=False)
    except (KeyError,ValueError,TypeError,json.JSONDecodeError,FileNotFoundError,pickle.UnpicklingError,EOFError) as error:
        raise InvalidSubmission(str(error)) from error


def visible_gpus():
    """Every GPU the verifier container exposes (the declared task resources), as CUDA indices."""
    try:
        listing=subprocess.run(['nvidia-smi','-L'],capture_output=True,text=True,check=True).stdout
        count=sum(line.startswith('GPU ') for line in listing.splitlines())
    except (OSError,subprocess.CalledProcessError):
        count=0
    return ','.join(map(str,range(count or 4)))


def namespace_command(root, runtime, limits):
    """Only trusted software, image/model data and scratch exist in this namespace."""
    cmd=['bwrap','--die-with-parent','--unshare-all','--new-session','--clearenv',
         '--ro-bind','/usr','/usr','--ro-bind','/opt/pathfinder','/opt/pathfinder',
         '--proc','/proc','--dev-bind','/dev','/dev','--tmpfs','/dev/shm','--tmpfs','/tmp',
         '--dir','/workspace','--dir','/workspace/research',
         '--bind',str(root),str(root),'--ro-bind',str(runtime),'/runtime',
         '--ro-bind',str(limits),'/limits.json','--ro-bind','/.dockerenv','/.dockerenv']
    for path in ('/bin','/sbin','/lib','/lib64'):
        if Path(path).exists():cmd+=['--ro-bind',path,path]
    # /etc/gitconfig marks the read-only runtime checkouts as safe; stitching checks their commits.
    for path in ('/etc/ld.so.cache','/etc/ld.so.conf','/etc/gitconfig'):
        if Path(path).exists():cmd+=['--ro-bind',path,path]
    for key,value in dict(PATH='/usr/local/bin:/usr/bin:/bin',HOME='/tmp',
            TMPDIR='/tmp',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',
            CUDA_VISIBLE_DEVICES=visible_gpus()).items():cmd+=['--setenv',key,value]
    cmd+=['--chdir',str(root),sys.executable,'-B',str(CODE/'single_pipeline.py'),
          '--root',str(root),'--runtime','/runtime','--limits','/limits.json']
    return cmd


def run_process(command, timeout, log):
    with log.open('x') as output, subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT,
                                                   start_new_session=True,stdin=subprocess.DEVNULL) as process:
        try:
            code=process.wait(timeout=timeout)
            if code:raise RuntimeError(f'Prediction process exited {code}; see {log}')
        finally:
            # Also kill orphan descendants if a worker/launcher failed or timed out.
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.wait()


def score(root,suite,logs):
    from evaluate_physical import score as physical_score
    from suite_validation import suite_pins
    from single_pipeline import command
    prediction=root/'prediction';geometry=json.loads((prediction/'submission/metadata.json').read_text())
    pins=suite_pins(suite,geometry)
    execution=json.loads((prediction/'execution.json').read_text())
    if execution.get('complete') is not True or execution['protocol']!='single-checkpoint-fast/v1':
        raise RuntimeError('Missing complete fresh model execution')
    if execution['prediction_sha256']!=sha256(prediction/'submission/segmentation.npy'):
        raise RuntimeError('Prediction changed after inference')
    subprocess.run(command('export_agglomeration_core.py',root=root,
        base=prediction/'stitched/segmentation.npy',base_metadata=prediction/'context/metadata.json',
        mapping=prediction/'baseline.json',suite=suite,output=prediction/'baseline-core'),check=True)
    methods={name:physical_score(suite,path,logs/name) for name,path in [
        ('ffn',prediction/'stitched'),('shared-proposal-ffn',prediction/'baseline-core'),
        ('disjoint-pair-shape',prediction/'submission')]}
    reward=methods['disjoint-pair-shape']['nerl']
    if not math.isfinite(reward) or not 0<=reward<=1:raise RuntimeError('Invalid metric')
    if suite_pins(suite,geometry)!=pins:raise RuntimeError('Private suite changed during scoring')
    write_json(logs/'comparison.json',dict(protocol='single-checkpoint-fast/v1',primary_metric='nerl',methods=methods,
        same_frozen_core=True,model_execution_verified=True,paper_comparable=False))
    write_json(logs/'reward.json',reward_metrics(reward,methods))


# Raw metrics copied into reward.json; Harbor reads it as a flat object of numbers.
METRICS=('nerl','erl_nm','maximum_erl_nm','split_errors','merge_errors','omitted_nodes','omitted_node_fraction','sampled_nodes')


def reward_metrics(reward,methods):
    """reward (disjoint-pair SHAPE nERL) next to every method's raw metrics, all numbers."""
    result=dict(reward=reward)
    for name,metrics in methods.items():
        for key in METRICS:result[name.replace('-','_')+'_'+key]=metrics[key]
    return result


def failure_result(logs,error):
    logs=Path(logs);reward=logs/'reward.json'
    invalid=isinstance(error,InvalidSubmission)
    if invalid:write_json(reward,dict(reward=0,invalid_submission=1))
    else:reward.unlink(missing_ok=True)
    write_json(logs/'status.json',dict(status='invalid_submission' if invalid else 'infrastructure_failure',error=repr(error)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('submission','image','runtime','suite','limits','root','logs'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--seconds',type=int,required=True)
    a=p.parse_args();a.logs.mkdir(parents=True,exist_ok=True)
    reward=a.logs/'reward.json'
    if reward.exists():raise FileExistsError('Fresh verifier log directory required')
    try:
        from execution_resources import container_execution
        if not container_execution():raise RuntimeError('Formal verifier requires the offline verifier image')
        if a.root!=Path('/workspace/research/run') or a.root.exists():
            raise RuntimeError('Fresh isolated verifier scratch required')
        if a.seconds<=0:raise ValueError('Positive whole-inference timeout required')
        if not shutil.which('bwrap'):raise RuntimeError('bubblewrap is required; no unsafe fallback')
        a.root.mkdir(parents=True)
        staged=stage_weights(a.submission,a.root/'model')
        # Operator-controlled runtime is a hash-pinned materialized resource.
        runtime=json.loads((a.runtime/'runtime-manifest.json').read_text())
        actual={str(path.relative_to(a.runtime)) for path in a.runtime.rglob('*') if path.is_file()}
        if actual != set(runtime['files'])|{'runtime-manifest.json'}:
            raise RuntimeError('Unlisted trusted runtime resource')
        for name,digest in runtime['files'].items():
            path=a.runtime/name
            if path.is_symlink() or a.runtime.resolve() not in path.resolve().parents or sha256(path)!=digest:
                raise RuntimeError('Trusted runtime resource changed')
        if 'point-ops/pathfinder_fps.so' not in runtime['files']:
            raise RuntimeError('Trusted FPS binary is missing')
        shutil.copytree(a.runtime/'point-ops',a.root/'point-ops')
        im=json.loads((a.image/'manifest.json').read_text());(a.root/'image').mkdir()
        for name in ('raw.npy','image_valid_z.npy'):
            if sha256(a.image/name)!=im['files'][name]:raise RuntimeError('Image input hash mismatch')
            shutil.copyfile(a.image/name,a.root/'image'/name)
        shutil.copyfile(a.image/'manifest.json',a.root/'image/manifest.json')
        write_json(a.logs/'model-staging.json',staged)
        started=time.monotonic()
        run_process(namespace_command(a.root,a.runtime,a.limits),a.seconds,a.logs/'inference.log')
        inference_seconds=time.monotonic()-started
        score(a.root,a.suite,a.logs)
        write_json(a.logs/'status.json',dict(status='passed',inference_seconds=inference_seconds,
                    scope='Fresh selected-protocol inference and private scoring'))
    except InvalidSubmission as error:
        failure_result(a.logs,error)
    except Exception as error:
        failure_result(a.logs,error)
        raise


if __name__=='__main__':main()
