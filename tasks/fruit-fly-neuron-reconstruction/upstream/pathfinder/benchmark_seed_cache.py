"""Validate exact cached seeds against a separately retained real-window result."""
import argparse,json,time
from pathlib import Path
import numpy as np
from seed_cache import get_seeds
from common import sha256,write_json

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','manifest','reference','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--window',type=int,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh output required')
    a.output.mkdir();m=json.loads(a.manifest.read_text());w=next(w for w in m['windows'] if w['index']==a.window)
    rawpath=Path(m['image_manifest']).parent/'raw.npy'
    if sha256(rawpath)!=m['raw_sha256']:raise ValueError('Raw identity changed')
    raw=np.load(rawpath,mmap_mode='r');context=raw[tuple(slice(x-16,y+16) for x,y in zip(w['start'],w['stop']))]
    identity=dict(raw_sha256=m['raw_sha256'],start=w['start'],stop=w['stop'])
    start=time.monotonic();seeds,created=get_seeds(context,a.root,a.output/'cache',identity);miss=time.monotonic()-start
    start=time.monotonic();cached,hit=get_seeds(context,a.root,a.output/'cache',identity);hit_time=time.monotonic()-start
    reference=np.load(a.reference,allow_pickle=False)
    if not np.array_equal(seeds-16,reference) or not np.array_equal(cached,seeds):raise ValueError('Exact seed mismatch')
    if created['hit'] or not hit['hit']:raise ValueError('Expected actual miss then hit')
    write_json(a.output/'comparison.json',dict(complete=True,window=a.window,count=len(seeds),
        cache_miss_seconds=miss,cache_hit_including_context_hash_seconds=hit_time,
        reference_sha256=sha256(a.reference),manifest_sha256=sha256(a.manifest),created=created,hit=hit,
        ascending_equal=True,descending_equal=bool(np.array_equal((cached-16)[::-1],reference[::-1])),
        scope='Exact real-window preprocessing equivalence only; no end-to-end throughput claim'))
