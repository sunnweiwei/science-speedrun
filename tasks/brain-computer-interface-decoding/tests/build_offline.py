"""Build from a controller-prepared wheelhouse when Docker build DNS is unavailable."""
import argparse,hashlib,json,shutil,subprocess,time
from pathlib import Path
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    task=Path(__file__).resolve().parents[1];context=a.root/('docker-context-'+str(time.time_ns()))
    if context.exists():raise FileExistsError('Use a fresh build context: '+str(context))
    shutil.copytree(task,context,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git'))
    shutil.copytree(a.root/'wheels',context/'wheels')
    p=context/'tests/Dockerfile';s=p.read_text()
    old='RUN pip install --index-url https://pypi.org/simple --no-cache-dir -r /opt/bci/requirements.txt'
    new='COPY wheels /opt/bci/wheels\nRUN pip install --no-index --find-links /opt/bci/wheels --no-cache-dir -r /opt/bci/requirements.txt'
    if old not in s:raise ValueError('unexpected Dockerfile')
    p.write_text(s.replace(old,new))
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (context/'wheels').glob('*.whl')}
    (a.root/'wheel-checksums.json').write_text(json.dumps(hashes,indent=2)+'\n')
    subprocess.run(['docker','build','--network','host','-f','tests/Dockerfile','-t','bci-verifier:0.2','.'],cwd=context,check=True)
    p=context/'environment/Dockerfile'
    p.write_text(p.read_text().replace(old,new))
    subprocess.run(['docker','build','--network','host','-f','environment/Dockerfile','-t','bci-development:0.2','.'],cwd=context,check=True)
if __name__=='__main__':main()
