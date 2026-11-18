"""Synthetic container checks only; not evidence of real-data model quality."""
import argparse,hashlib,json,subprocess,tempfile
from pathlib import Path
import numpy as np
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--image',default='bci-verifier:0.2');a=ap.parse_args()
    with tempfile.TemporaryDirectory(prefix='bci-container-probe-') as td:
        root=Path(td);private=root/'private';public=private/'tuev/public';public.mkdir(parents=True)
        np.save(public/'shard.npy',np.zeros((2,16,5,200),dtype=np.float32))
        shard=public/'shard.npy'
        (public/'checksums.json').write_text(json.dumps([{'path':'shard.npy', 'bytes':shard.stat().st_size,
            'sha256':hashlib.sha256(shard.read_bytes()).hexdigest()}]))
        rows=[{'id':'a','label':0,'subject':'s1','session':'1'},{'id':'b','label':1,'subject':'s2','session':'1'}]
        (private/'tuev/truth.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        (public/'manifest.json').write_text(json.dumps({'schema':'bci-input-v1','track':'tuev','samples':[{'id':r['id'],'path':'shard.npy','index':i} for i,r in enumerate(rows)]}))
        package=root/'submission';package.mkdir(parents=True)
        # Use package-relative code so relocation is actually exercised.
        (package/'infer.sh').write_text('#!/bin/bash\nset -eu\npython "$(dirname "$0")/probe.py" "$1" "$2"\n')
        outputs=[]
        for mode in ('valid','missing','nan','duplicate','classes'):
            code="""import json,sys
from pathlib import Path
for private in ['/private/data/tuev/truth.jsonl','/tests/main_verifier.py']:
 try:
  open(private).read()
 except PermissionError: pass
 else: raise RuntimeError('isolation failed: '+private)
try:
 Path('/logs/verifier/forged').write_text('bad')
except PermissionError: pass
else: raise RuntimeError('reward writable')
rows=json.loads((Path(sys.argv[1])/'manifest.json').read_text())['samples']
pred=[]
for r in reversed(rows):
 p=[0.0]*6;p[0 if r['id']=='a' else 1]=1.0
 pred.append({'id':r['id'],'probabilities':p})
"""
            if mode=='missing':code+="pred=pred[:1]\n"
            if mode=='nan':code+="pred[0]['probabilities'][0]=float('nan')\n"
            if mode=='duplicate':code+="pred.append(pred[0])\n"
            if mode=='classes':code+="pred[0]['probabilities']=[1,0]\n"
            code+="Path(sys.argv[2],'predictions.jsonl').write_text(''.join(json.dumps(r)+'\\n' for r in pred))\n"
            (package/'probe.py').write_text(code)
            out=root/('output-'+mode);out.mkdir()
            cmd=['docker','run','--rm','--network','none','-v',str(private)+':/private/data:ro','-v',str(root/'submission')+':/workspace/submission:ro','-v',str(out)+':/logs/verifier',a.image,'bash','/tests/test.sh','--tracks','tuev']
            subprocess.run(cmd,check=True)
            result=json.loads((out/'report.json').read_text())['tracks']['tuev']
            reward=json.loads((out/'reward.json').read_text())
            track_only={k:v for k,v in reward.items() if k not in ('score','reward','signed_score')}
            assert track_only=={'tuev':result['reward']},reward
            assert result['reward']==(1.0 if mode=='valid' else 0.0),result
            assert result['status']==('ok' if mode=='valid' else 'invalid_submission'),result
            outputs.append({'case':mode,'status':result['status'],'reward':result['reward']})
        print(json.dumps({'scope':'synthetic container isolation and output-contract probes','cases':outputs},indent=2))
if __name__=='__main__':main()
