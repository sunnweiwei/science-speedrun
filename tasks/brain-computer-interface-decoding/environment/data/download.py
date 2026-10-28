"""Controller-side verified downloads. Never run inside an agent container."""
import argparse, base64, concurrent.futures, hashlib, json, os, re, time, urllib.request
from pathlib import Path

PINS = {'ad65': ('ds004504','63b2ec152bc3cd4487d131124f10f767ebe543bc'),
        'longitudinal': ('ds007176','5578877aa48b7094df79e11d6058c5b0c072d149')}
def get(url):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url,timeout=120) as r: return r.read()
        except Exception:
            if attempt==4: raise
            time.sleep(2**attempt)
def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()
def download(item,root):
    p=root/item['path']; p.parent.mkdir(parents=True,exist_ok=True)
    receipt=p.with_name(p.name+'.receipt.json')
    if p.exists() and receipt.exists():
        prior=json.loads(receipt.read_text())
        if prior['url']==item['url'] and prior.get('md5')==item.get('md5') and prior.get('sha256')==(item.get('sha256') or prior.get('sha256')) and p.stat().st_size==prior['bytes'] and digest(p)==prior['sha256']:
            return prior
    for attempt in range(5):
        try:
            tmp=p.with_name(p.name+'.part')
            offset=tmp.stat().st_size if tmp.exists() else 0
            req=urllib.request.Request(item['url'],headers={'Range':f'bytes={offset}-'} if offset else {})
            with urllib.request.urlopen(req,timeout=120) as r:
                resume=offset>0 and r.status==206
                if resume and not r.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                    raise ValueError('invalid resume range')
                h=hashlib.sha256(); n=0
                if resume:
                    with open(tmp,'rb') as prior:
                        for b in iter(lambda:prior.read(4*1024*1024),b''):h.update(b);n+=len(b)
                expected=n+int(r.headers['Content-Length']) if r.headers.get('Content-Length') else None
                with open(tmp,'ab' if resume else 'wb') as f:
                    for b in iter(lambda:r.read(256*1024),b''):f.write(b);h.update(b);n+=len(b)
                if expected is not None and n!=expected:raise ValueError('incomplete HTTP body')
            if item.get('size') is not None and n!=item['size']: raise ValueError('size mismatch '+str(p))
            if item.get('sha256') and h.hexdigest()!=item['sha256']: raise ValueError('checksum mismatch '+str(p))
            if item.get('md5'):
                md5=hashlib.md5()
                with open(tmp,'rb') as f:
                    for b in iter(lambda:f.read(4*1024*1024),b''):md5.update(b)
                if md5.hexdigest()!=item['md5']:raise ValueError('upstream MD5 mismatch '+str(p))
            if item.get('git_sha1'):
                gh=hashlib.sha1(b'blob '+str(n).encode()+b'\0'+tmp.read_bytes()).hexdigest()
                if gh!=item['git_sha1']: raise ValueError('git blob mismatch '+str(p))
            tmp.replace(p)
            result={**item,'bytes':n,'sha256':h.hexdigest()}
            receipt.write_text(json.dumps(result,indent=2)+'\n')
            print('VERIFIED',item['path'],n,flush=True)
            return result
        except Exception:
            if attempt==4: raise
            time.sleep(2**attempt)
def inventory(track,root):
    if track=='openbmi':
        pinned=Path(__file__).with_name('openbmi.lock.json')
        if pinned.exists():return json.loads(pinned.read_text())
        base='https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100542/'
        return [dict(path=f'session{s}/s{i}/sess{s:02d}_subj{i:02d}_EEG_MI.mat',
                     url=base+f'session{s}/s{i}/sess{s:02d}_subj{i:02d}_EEG_MI.mat')
                for i in range(1,55) for s in (1,2)]
    ds,sha=PINS[track]
    tree=json.loads(get(f'https://api.github.com/repos/OpenNeuroDatasets/{ds}/git/trees/{sha}?recursive=1'))
    root.mkdir(parents=True,exist_ok=True)
    (root/'source-tree.json').write_text(json.dumps(tree,indent=2))
    selected=[x for x in tree['tree'] if x['type']=='blob' and (
        x['path'] in ('participants.tsv','participants.json','dataset_description.json','README','README.md','CHANGES')
        or (x['path'].startswith('derivatives/sub-') if track=='ad65' else x['path'].startswith('sub-')))]
    def resolve(x):
        p=x['path']; raw=f'https://raw.githubusercontent.com/OpenNeuroDatasets/{ds}/{sha}/{p}'
        if x['mode']=='120000':
            target=get(raw).decode()
            m=re.search(r'SHA256E-s(\d+)--([a-f0-9]{64})',target)
            if not m: raise ValueError('unsupported annex key: '+target)
            return dict(path=p,url=f'https://s3.amazonaws.com/openneuro.org/{ds}/{p}',size=int(m[1]),sha256=m[2])
        return dict(path=p,url=raw,size=x['size'],git_sha1=x['sha'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        return list(pool.map(resolve,selected))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('track',choices=['ad65','openbmi','longitudinal']);ap.add_argument('destination',type=Path);ap.add_argument('--workers',type=int,default=4)
    a=ap.parse_args();a.destination.mkdir(parents=True,exist_ok=True)
    lock=a.destination/'download-manifest.json'
    items=json.loads(lock.read_text()) if lock.exists() else inventory(a.track,a.destination)
    if a.track=='openbmi' and any('md5' not in item for item in items):
        def pin(item):
            with urllib.request.urlopen(urllib.request.Request(item['url'],method='HEAD'),timeout=60) as r:
                encoded=r.headers.get('x-amz-meta-md5chksum')
                if not encoded:raise ValueError('official OpenBMI MD5 missing')
                return {**item,'size':int(r.headers['Content-Length']),'md5':base64.b64decode(encoded).hex(),'source_last_modified':r.headers.get('Last-Modified')}
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:items=list(pool.map(pin,items))
    lock.write_text(json.dumps(items,indent=2)+'\n')
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        receipts=list(pool.map(lambda item:download(item,a.destination),items))
    (a.destination/'download-receipt.json').write_text(json.dumps(receipts,indent=2)+'\n')
    (a.destination/'DOWNLOAD_COMPLETE').touch()
if __name__=='__main__':main()
