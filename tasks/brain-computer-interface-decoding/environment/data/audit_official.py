"""Numerical parity of materialized real FACED/TUEV data with official loader semantics."""
import argparse,json,pickle
from pathlib import Path
import lmdb,numpy as np
def main():
    ap=argparse.ArgumentParser();ap.add_argument('track',choices=['faced','tuev']);ap.add_argument('source',type=Path);ap.add_argument('runtime',type=Path);a=ap.parse_args()
    db=lmdb.open(str(a.source),readonly=True,lock=False) if a.track=='faced' else None
    count=0
    for split,folder in [('train','processed_train'),('val','processed_eval'),('test','processed_test')]:
        root=a.runtime/('private' if split=='test' else 'development')/a.track/('public' if split=='test' else split)
        rows=[json.loads(l) for l in (root.parent/'truth.jsonl').read_text().splitlines()] if split=='test' else json.loads((root/'manifest.json').read_text())['samples']
        for i in np.linspace(0,len(rows)-1,min(10,len(rows)),dtype=int):
            r=rows[i]
            if db:
                with db.begin() as txn:pair=pickle.loads(txn.get(r['source'].encode()))
                expected=np.asarray(pair['sample'])/100;label=int(pair['label'])
            else:
                source=a.source/r['source'] if '/' in r['source'] else a.source/folder/r['source']
                pair=pickle.loads(source.read_bytes())
                expected=np.asarray(pair['signal']).reshape(16,5,200)/100;label=int(np.asarray(pair['label']).ravel()[0])-1
            actual=np.load(root/r['path'],mmap_mode='r',allow_pickle=False)[r['index']]/100
            np.testing.assert_array_equal(actual.astype(np.float32),expected.astype(np.float32))
            assert label==r['label'];count+=1
    if db:db.close()
    print(json.dumps({'track':a.track,'samples_checked':count,'loader_float32_parity':'exact'}))
if __name__=='__main__':main()
