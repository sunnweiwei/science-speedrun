"""Root controller; inference runs as an unprivileged UID without label access."""
import argparse,hashlib,json,os,shutil,signal,stat,subprocess,sys,tempfile,time
from pathlib import Path
local_evaluation=Path(__file__).resolve().parents[1]/'environment/evaluation'
sys.path.insert(0,str(local_evaluation if local_evaluation.exists() else Path('/opt/bci/evaluation')))
from bci_score import CLASSES,read_predictions,score
from data_identity import split_identity

def terminate_users():
    subprocess.run(['pkill','-KILL','-u','10001'],check=False,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
def package_hash(source):
    h=hashlib.sha256()
    for p in sorted(source.rglob('*')):
        if p.is_symlink():
            raise ValueError('package symlink is not allowed')
        if p.is_file():
            h.update(str(p.relative_to(source)).encode());h.update(b'\0')
            with p.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def evaluate(track,submission,private,limit,timeout):
    truth_path=private/track/'truth.jsonl';public=private/track/'public'
    rows=[json.loads(l) for l in truth_path.read_text().splitlines()]
    if limit:
        # Select complete subjects for grouped metrics, classes for event metrics.
        if track in ('ad65','openbmi','longitudinal'):
            selected=set()
            for label in sorted({r['label'] for r in rows}):
                selected.add(next(r['subject'] for r in rows if r['label']==label))
            rows=[r for r in rows if r['subject'] in selected]
        else:
            keep=[]
            for label in sorted({r['label'] for r in rows}):keep.extend([r for r in rows if r['label']==label][:limit])
            rows=keep
    if not rows:raise RuntimeError('empty verifier data')
    base=Path(tempfile.mkdtemp(prefix='bci-verify-',dir='/tmp'));os.chmod(base,0o755)
    inp=base/'input';out=base/'output';pkg=base/'package'
    inp.mkdir();out.mkdir();os.chown(out,10001,10001)
    manifest=json.loads((public/'manifest.json').read_text());ids={r['id'] for r in rows}
    manifest['samples']=[r for r in manifest['samples'] if r['id'] in ids]
    if any(set(r)!={'id','path','index'} for r in manifest['samples']):raise RuntimeError('labels in public manifest')
    for name in {r['path'] for r in manifest['samples']}:
        source=(public/name).resolve()
        if not source.is_relative_to(public.resolve()):raise RuntimeError('invalid public path')
        shutil.copyfile(source,inp/name)
    (inp/'manifest.json').write_text(json.dumps(manifest))
    started=time.monotonic()
    try:
        source=submission.resolve()
        if not source.is_relative_to(submission.resolve()):raise ValueError('package escapes submission root')
        for p in source.rglob('*'):
            if p.is_symlink():raise ValueError('package symlink is not allowed')
        shutil.copytree(source,pkg)
        for p in [pkg,*pkg.rglob('*')]:
            os.chmod(p,0o755 if p.is_dir() or p.suffix=='.sh' else 0o644)
        cmd=['setpriv','--reuid=10001','--regid=10001','--clear-groups','--bounding-set=-all','--no-new-privs','bash',str(pkg/'infer.sh'),str(inp),str(out)]
        env={'PATH':os.environ['PATH'],'HOME':str(out),'TMPDIR':str(out),'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}
        for key in ('LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','NVIDIA_DRIVER_CAPABILITIES'):
            if key in os.environ:env[key]=os.environ[key]
        with open(base/'inference.log','wb') as log:
            proc=subprocess.Popen(cmd,cwd=pkg,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait();raise ValueError('inference timeout')
        terminate_users()
        if code:raise ValueError('inference failed: '+(base/'inference.log').read_text(errors='replace')[-2000:])
        pred=out/'predictions.jsonl'
        fd=os.open(pred,os.O_RDONLY|os.O_NOFOLLOW)
        with os.fdopen(fd,'rb') as stream:
            info=os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size>1024**3:raise ValueError('invalid output file')
            trusted=base/'trusted-predictions.jsonl';trusted.write_bytes(stream.read())
        probs=read_predictions(trusted,rows,CLASSES[track])
        result=score(track,rows,probs);result['status']='ok';result['seconds']=time.monotonic()-started
        result['sample_count']=len(rows)
        result['subject_count']=len({r.get('subject') for r in rows})
        result['data_identity']=split_identity(public,truth_path)
        result['package_sha256']=package_hash(source)
        return result
    except (ValueError,KeyError,TypeError,FileNotFoundError,NotADirectoryError,PermissionError) as e:
        return {'reward':0.0,'status':'invalid_submission','error':str(e),'seconds':time.monotonic()-started}
    finally:
        terminate_users();shutil.rmtree(base)
def write_receipts(output,result):
    rewards={track:float(detail['reward']) for track,detail in result['tracks'].items()}
    (output/'reward.json').write_text(json.dumps(rewards,indent=2,allow_nan=False)+'\n')
    (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--submission',type=Path,default=Path('/workspace/submission'));ap.add_argument('--private',type=Path,default=Path('/private/data'));ap.add_argument('--output',type=Path,default=Path('/logs/verifier'));ap.add_argument('--tracks',nargs='+',choices=CLASSES,default=list(CLASSES));ap.add_argument('--smoke-limit',type=int,default=0);ap.add_argument('--timeout',type=int,default=7200);a=ap.parse_args()
    if os.getuid()!=0:raise RuntimeError('verifier controller must be root')
    for path in (Path('/private'),Path('/tests'),a.output):
        path.mkdir(parents=True,exist_ok=True);os.chmod(path,0o700)
    # Test the actual permission boundary before executing any untrusted model.
    for p in [a.private/t/'truth.jsonl' for t in a.tracks]:
        if not p.is_file():raise RuntimeError('missing infrastructure data '+str(p))
        r=subprocess.run(['setpriv','--reuid=10001','--regid=10001','--clear-groups','--bounding-set=-all','--no-new-privs','test','-r',str(p)])
        if r.returncode==0:raise RuntimeError('private labels accessible')
    try:root_hash=package_hash(a.submission.resolve())
    except ValueError:root_hash=None
    result={'schema':'bci-shared-v2','aggregate':None,'scope':'smoke' if a.smoke_limit else 'full','tracks':{},
            'package_sha256':root_hash}
    for track in a.tracks:
        result['tracks'][track]=evaluate(track,a.submission,a.private,a.smoke_limit,a.timeout)
    # Harbor reads reward.json as a flat map of finite numeric metrics. Keep
    # statuses and auxiliary metrics in report.json; there is no five-track
    # aggregate reward.
    write_receipts(a.output,result)
    print(json.dumps(result,indent=2,allow_nan=False))
if __name__=='__main__':main()
