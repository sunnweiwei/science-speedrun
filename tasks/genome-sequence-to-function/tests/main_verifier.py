#!/usr/bin/env python3
"""Run untrusted inference and seal bounded multi-family Zarr predictions."""
from __future__ import annotations
import argparse,json,os,pwd,shutil,signal,stat,subprocess,sys,time
from pathlib import Path
import numpy as np,pandas as pd,zarr
from common import sha256,tree_sha256,write_json
sys.path[:0]=["/evaluation",str(Path(__file__).resolve().parents[1]/"environment"/"evaluation")]  # image, then repository
from scoring import MAX_PREDICTION_BYTES,validate_predictions  # noqa: E402
from variant_scoring import validate_perturbations  # noqa: E402

PUBLIC_COLUMNS=("example_id","split","chromosome","input_start","input_end","input_pad_left","input_pad_right","target_start","target_end")

def validate_submission(root):
    root=root.resolve(); entry=root/"infer.sh"
    if not entry.is_file() or entry.is_symlink(): raise FileNotFoundError("submission/infer.sh required")
    for path in root.rglob("*"):
        if path.is_symlink() and not path.resolve().is_relative_to(root): raise ValueError(f"submission symlink escapes package: {path.relative_to(root)}")

def validate_input(root):
    manifest=root/"public/test.parquet"
    for path in (manifest,root/"public/junction_tracks.parquet",root/"public/perturbations.parquet",root/"reference/GRCh38.p13.genome.fa",root/"reference/GRCh38.p13.genome.fa.fai"):
        if not path.is_file(): raise FileNotFoundError(f"missing verifier input: {path}")
    if (root/"labels").exists() or (root/"scoring").exists() or (root/"targets").exists(): raise ValueError("untrusted input contains trusted data")
    rows=pd.read_parquet(manifest); ids=rows.example_id.astype(str).tolist()
    if tuple(rows.columns)!=PUBLIC_COLUMNS: raise ValueError("public manifest field whitelist mismatch")
    if not ids or len(ids)!=len(set(ids)): raise ValueError("invalid test IDs")
    return rows,ids

def _agent_processes(uid):
    result=[]
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit(): continue
        try:
            status=(entry/"status").read_text()
            real_uid=int(next(line for line in status.splitlines() if line.startswith("Uid:" )).split()[1])
            state=next(line for line in status.splitlines() if line.startswith("State:" )).split()[1]
        except (FileNotFoundError,PermissionError,StopIteration,ValueError): continue
        if real_uid==uid and state!="Z": result.append(int(entry.name))
    return result

def _stop_untrusted(uid,pgid):
    deadline=time.monotonic()+10
    while True:
        try: os.killpg(pgid,signal.SIGKILL)
        except ProcessLookupError: pass
        for pid in _agent_processes(uid):
            try: os.kill(pid,signal.SIGKILL)
            except ProcessLookupError: pass
        remaining=_agent_processes(uid)
        if not remaining: return
        if time.monotonic()>=deadline: raise RuntimeError(f"untrusted processes survived cleanup: {remaining}")
        time.sleep(0.05)

def run_submission(submission,input_root,work,timeout,entry_name="infer.sh",output_name="untrusted-output"):
    validate_submission(submission)
    entry=submission/entry_name
    if not entry.is_file() or entry.is_symlink(): raise FileNotFoundError(f"submission/{entry_name} required")
    agent=pwd.getpwnam("agent"); output=work/output_name; output.mkdir(); os.chown(output,agent.pw_uid,agent.pw_gid)
    env={"PATH":"/opt/conda/bin:/usr/local/bin:/usr/bin:/bin","HOME":agent.pw_dir,"PYTHONUNBUFFERED":"1","PIP_NO_INDEX":"1","CUDA_VISIBLE_DEVICES":os.environ.get("CUDA_VISIBLE_DEVICES","0")}
    def demote(): os.setgid(agent.pw_gid); os.setuid(agent.pw_uid)
    with (work/(entry_name.split(".")[0]+".log")).open("wb") as log:
        preflight='import os,torch; assert os.getuid()==10001; assert torch.cuda.is_available(), "CUDA unavailable in uid 10001 inference process"'
        process=subprocess.Popen(["bash","-c",'python -c "$1" && exec bash "$2" "$3" "$4"',"verifier",preflight,str(entry),str(input_root),str(output)],cwd=submission,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=demote)
        failure=None
        try: code=process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as error: failure=error; code=None
        finally:
            _stop_untrusted(agent.pw_uid,process.pid)
            if process.poll() is None: process.wait()
        if failure is not None: raise failure
    if code: raise subprocess.CalledProcessError(code,process.args)
    return output

def validate_output(root,rows,expected_ids):
    if {path.name for path in root.iterdir()}!={"example_ids.json","predictions.zarr"}: raise ValueError("prediction output contains unexpected top-level entries")
    ids_path=root/"example_ids.json"; prediction_path=root/"predictions.zarr"
    if not ids_path.is_file() or ids_path.is_symlink(): raise ValueError("example_ids.json must be a regular file")
    if not prediction_path.is_dir() or prediction_path.is_symlink(): raise ValueError("predictions.zarr must be a regular directory")
    if json.loads(ids_path.read_text())!=expected_ids: raise ValueError("example ID order mismatch")
    total=0
    for item in prediction_path.rglob("*"):
        if item.is_symlink(): raise ValueError("prediction tree contains symlink")
        if item.is_file(): total+=item.stat().st_size
    if total>MAX_PREDICTION_BYTES: raise ValueError(f"predictions exceed the {MAX_PREDICTION_BYTES//1024**3}-GiB v4 budget")
    validate_predictions(prediction_path,expected_ids)
    return prediction_path

def _lock_snapshot(root):
    for path in sorted(root.rglob("*"),reverse=True):
        if path.is_symlink(): continue
        os.chown(path,0,0,follow_symlinks=False)
        os.chmod(path,0o500 if path.is_dir() else 0o400,follow_symlinks=False)
    os.chown(root,0,0); os.chmod(root,0o500)

def _copy_snapshot(source,destination):
    for path in source.rglob("*"):
        mode=path.lstat().st_mode
        if path.is_symlink() or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError(f"prediction output contains non-regular entry: {path.relative_to(source)}")
    shutil.copytree(source,destination,symlinks=True)

def _atomic_text(path,value):
    temporary=path.with_name(path.name+f".tmp-{os.getpid()}")
    temporary.write_text(value); os.replace(temporary,path)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--submission",type=Path,required=True); p.add_argument("--input",type=Path,required=True); p.add_argument("--sealed",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--timeout",type=int,default=14400); p.add_argument("--no-wait",action="store_true"); p.add_argument("--grader-timeout",type=int,default=7200); a=p.parse_args()
    rows,ids=validate_input(a.input); a.output.mkdir(parents=True,exist_ok=True); work=a.output/"main"; work.mkdir()
    agent=pwd.getpwnam("agent"); os.chown(work,agent.pw_uid,agent.pw_gid); os.chmod(work,0o700)
    temporary=a.sealed.with_name(a.sealed.name+f".tmp-{os.getpid()}")
    try:
        begin=time.monotonic()
        source=run_submission(a.submission,a.input,work,a.timeout)
        # Variant and perturbation queries: both sequences of each edit, same package, same budget.
        infer_seconds=time.monotonic()-begin; remaining=max(1,int(a.timeout-infer_seconds))
        perturbation_source=run_submission(a.submission,a.input,work,remaining,"perturb.sh","untrusted-perturbations")
        print(json.dumps({"event":"inference_complete","infer_seconds":round(infer_seconds,1),"perturb_seconds":round(time.monotonic()-begin-infer_seconds,1)}),flush=True)
        _copy_snapshot(source,temporary)
        predictions=validate_output(temporary,rows,ids)
        perturbations=temporary/"perturbations.zarr"
        _copy_snapshot(perturbation_source/"perturbations.zarr",perturbations)
        validate_perturbations(perturbations,pd.read_parquet(a.input/"public/perturbations.parquet"))
        _lock_snapshot(temporary)
    except (FileNotFoundError,ValueError,KeyError,subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
        # The submission failed, not the verifier: it scores 0 and nothing is sealed.
        write_json(a.output/"report.json",{"schema":"genome-sequence-to-function-report/v4","split":"test","status":"invalid_submission","error":f"{type(error).__name__}: {error}"[:2000],"overall_mean_family_pearson":0.0})
        write_json(a.output/"reward.json",{"reward":0.0,"overall_mean_family_pearson":0.0})
        return
    snapshot={"schema":"genome-sequence-to-function-prediction-snapshot/v4","input_manifest_sha256":sha256(a.input/"public/test.parquet"),"prediction_tree_sha256":tree_sha256(predictions),
              "perturbation_manifest_sha256":sha256(a.input/"public/perturbations.parquet"),"perturbation_tree_sha256":tree_sha256(perturbations)}
    os.chmod(temporary,0o700); write_json(temporary/"snapshot.json",snapshot); os.chmod(temporary/"snapshot.json",0o400); os.chmod(temporary,0o500)
    os.replace(temporary,a.sealed)
    _atomic_text(a.sealed.parent/"READY",sha256(a.sealed/"snapshot.json")+"\n")
    if not a.no_wait:
        # The grader re-hashes the snapshot and scores every example before it publishes; allow it the
        # rest of the verifier budget rather than a fixed 10 minutes (24 GiB alone took 12.3 min).
        results=a.sealed.parent/"results"; deadline=time.monotonic()+a.grader_timeout
        while not (results/"reward.json").is_file():
            if time.monotonic()>=deadline: raise TimeoutError("grader did not publish reward")
            time.sleep(1)
        for name in ("report.json","reward.json"): shutil.copyfile(results/name,a.output/name)
if __name__=="__main__": main()
