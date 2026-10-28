"""Negative control: read public positions only, never EEG arrays or labels."""
import json,sys
from pathlib import Path
LABELS=[0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8]
def position_label(path,index):
    return LABELS[((int(Path(path).stem.split('-')[-1])*128+index)%84)//3]
def main():
    inp,out=map(Path,sys.argv[1:3]);manifest=json.loads((inp/'manifest.json').read_text())
    if manifest['track']!='faced':raise ValueError('FACED position-only diagnostic')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'predictions.jsonl').open('w') as f:
        for r in manifest['samples']:
            p=[0.0]*9;p[position_label(r['path'],r['index'])]=1.0
            f.write(json.dumps({'id':r['id'],'probabilities':p})+'\n')
if __name__=='__main__':main()

