"""Operator-only anonymous download of the immutable prepared resource release."""
import argparse
import fcntl
import json
from pathlib import Path
import re
from materialize import materialize


def run(args):
    source=json.loads(args.source.read_text())
    if not re.fullmatch(r'[0-9a-f]{40}',source['revision']):
        raise ValueError('Immutable full HF commit required')
    if not re.fullmatch(r'[0-9a-f]{64}',source['manifest_sha256']):
        raise ValueError('Manifest SHA-256 required')
    roots=[p.resolve() for p in (args.cache,args.data,args.private)]
    if any(a==b or a in b.parents or b in a.parents for i,a in enumerate(roots) for b in roots[i+1:]):
        raise ValueError('Cache, development and private roots must be disjoint')
    args.cache.mkdir(parents=True,exist_ok=True)
    with (args.cache/'materialize.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (args.cache/'receipt.json').unlink(missing_ok=True)
        def fetch(name):
            from huggingface_hub import hf_hub_download
            return Path(hf_hub_download(repo_id=source['repo_id'],repo_type='dataset',
                        revision=source['revision'],filename=name,cache_dir=args.cache/'hub',
                        token=False,local_files_only=args.verify_only))
        manifest=fetch('release.json')
        def blob(sha):
            if not re.fullmatch(r'[0-9a-f]{64}',sha):raise ValueError('Invalid blob SHA-256')
            print(json.dumps({'downloading':sha}),flush=True)
            return fetch('blobs/'+sha)
        result=materialize(manifest,source['manifest_sha256'],'',args.data,args.private,
                           args.verify_only,fetch=blob)
        result.update(repo_id=source['repo_id'],revision=source['revision'],anonymous=True)
        (args.cache/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
        return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path(__file__).with_name('download-source.json'))
    for name in ('data','private','cache'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--verify-only',action='store_true')
    run(p.parse_args())
