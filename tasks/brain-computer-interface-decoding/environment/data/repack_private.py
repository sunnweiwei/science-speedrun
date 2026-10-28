"""Version private shards without exposing source order; keep EEG/label/ID values."""
import argparse,hashlib,json,secrets,shutil,tempfile
from functools import lru_cache
from pathlib import Path
import numpy as np

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def repack(source,output):
    source,output=Path(source),Path(output)
    if output.exists():raise FileExistsError(output)
    manifest=json.loads((source/'public/manifest.json').read_text())
    rows=[json.loads(x) for x in (source/'truth.jsonl').read_text().splitlines()]
    public={r['id']:r for r in manifest['samples']}
    if len(public)!=len(rows) or set(public)!={r['id'] for r in rows}:
        raise ValueError('truth/public membership mismatch')
    for r in rows:
        if any(r[k]!=public[r['id']][k] for k in ('path','index')):
            raise ValueError('truth/public position mismatch')
    output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+output.name+'-',dir=output.parent))
    stage.chmod(0o700)
    @lru_cache(maxsize=512)
    def load(name):
        p=(source/'public'/name).resolve()
        if not p.is_relative_to((source/'public').resolve()):raise ValueError('unsafe shard path')
        return np.load(p,mmap_mode='r',allow_pickle=False)
    try:
        dest=stage/'public';dest.mkdir()
        shuffled=list(rows);secrets.SystemRandom().shuffle(shuffled)
        new=[];checks=[];moved=0
        for start in range(0,len(shuffled),128):
            batch=shuffled[start:start+128];name=f'shard-{start//128:06d}.npy'
            arr=np.stack([load(r['path'])[r['index']] for r in batch])
            np.save(dest/name,arr,allow_pickle=False)
            checks.append(dict(path=name,bytes=(dest/name).stat().st_size,sha256=digest(dest/name)))
            for i,r in enumerate(batch):
                moved+=int((name,i)!=(r['path'],r['index']))
                new.append({**r,'path':name,'index':i})
        # Compare every output sample to its old source, not just aggregate hashes.
        for block in checks:
            a=np.load(dest/block['path'],mmap_mode='r',allow_pickle=False)
            for r in (x for x in new if x['path']==block['path']):
                old=public[r['id']]
                if not np.array_equal(a[r['index']],load(old['path'])[old['index']]):
                    raise ValueError('signal changed')
        manifest['storage_layout']='global-random-v2'
        manifest['samples']=sorted([{k:r[k] for k in ('id','path','index')} for r in new],key=lambda r:r['id'])
        (dest/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
        (dest/'checksums.json').write_text(json.dumps(checks,indent=2)+'\n')
        (stage/'truth.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in new))
        (stage/'truth.jsonl').chmod(0o600)
        stage.rename(output)
        return {'track':manifest['track'],'samples':len(new),'positions_changed':moved,
                'signals_labels_ids_preserved':True,'storage_layout':'global-random-v2'}
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True);raise
    finally:load.cache_clear()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();print(json.dumps(repack(a.source,a.output)))

