import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'environment/evaluation'))
from bci_score import CLASSES,score
track,truth,out=sys.argv[1:]
rows=[json.loads(l) for l in Path(truth).read_text().splitlines()]
n=len(rows);k=CLASSES[track]
constant=np.zeros((n,k));constant[:,0]=1
rng=np.random.default_rng(3407);random=np.eye(k)[rng.integers(k,size=n)]
Path(out).write_text(json.dumps({'scope':'full test labels, trivial controls only','constant':score(track,rows,constant),'random_seed_3407':score(track,rows,random)},indent=2,allow_nan=False)+'\n')
