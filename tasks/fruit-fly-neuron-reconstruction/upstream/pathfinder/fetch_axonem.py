"""Bounded downloads of the author-linked AxonEM-M training archive and sparse preview.

Sparse preview is NOT full challenge GT. No authentication or full test GT access.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import ssl
import shutil
import time
import urllib.request
import zipfile
import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import external, sha256, write_json

TRAIN = dict(repo='pytc/AxonEM', revision='af20076475846ffa8a68d964a29fd4ca23eb66c2',
    name='EM30-M-train-9vol-pad-20-512-512.zip',bytes=1940800499,
    sha256='dae06b5dabe388ab7a0ff4e51548174f041a338d0d06bd665586aa7fdd43bac2')
SPARSE='https://rhoana.rc.fas.harvard.edu/ng/EM30-M/gt_axon_sparse'
LIMIT=200_000_000_000


def usage(root):
    return sum(p.stat().st_size for p in root.rglob('*') if p.is_file())


def storage_limit(root, ledger_limit=LIMIT):
    """Keep the portable default; development authorization is root-bound."""
    root=Path(root).resolve()
    policy=root/'development-storage-policy.json'
    if not policy.exists():return min(LIMIT,ledger_limit)
    data=json.loads(policy.read_text())
    if (data.get('schema')!=1 or data.get('scope')!='development-only'
            or data.get('root')!=str(root) or 'resident_limit_bytes' not in data
            or data['resident_limit_bytes'] is not None
            or data.get('starter_target_bytes')!=LIMIT or not data.get('user_authorization')):
        raise ValueError('Invalid root-bound development storage authorization')
    return float('inf')


def reserve(root, amount):
    if amount<0:raise ValueError('Negative storage reservation')
    ceiling=storage_limit(root)  # Validate root-bound authorization before any shortcut.
    if ceiling!=float('inf') and usage(root)+amount>ceiling:raise ValueError('Starter storage limit would be exceeded')
    if amount>shutil.disk_usage(root).free:raise ValueError('Insufficient filesystem free space')


def training(root):
    folder=root/'data';folder.mkdir(exist_ok=True)
    path=folder/TRAIN['name'];url=f"https://huggingface.co/datasets/{TRAIN['repo']}/resolve/{TRAIN['revision']}/{TRAIN['name']}"
    if not path.exists():
        reserve(root,TRAIN['bytes']);temp=path.with_suffix('.partial');done=0;last=time.monotonic()
        with urllib.request.urlopen(url,timeout=60) as src,temp.open('wb') as dst:
            while chunk:=src.read(8<<20):
                done+=len(chunk)
                if done>TRAIN['bytes']:raise ValueError('Download exceeds pinned byte count')
                dst.write(chunk)
                if time.monotonic()-last>30:print(json.dumps(dict(download_bytes=done)),flush=True);last=time.monotonic()
        if done!=TRAIN['bytes'] or sha256(temp)!=TRAIN['sha256']:raise ValueError('Archive integrity failure')
        temp.replace(path)
    if sha256(path)!=TRAIN['sha256']:raise ValueError('Archive hash mismatch')
    with zipfile.ZipFile(path) as z:
        reserve(root,sum(i.file_size for i in z.infolist()))
        for i in z.infolist():
            if folder.resolve() not in (folder/i.filename).resolve().parents:raise ValueError('Unsafe archive path')
        z.extractall(folder)
    write_json(root/'audit'/'training-download.json',dict(**TRAIN,url=url,source='https://axonem.grand-challenge.org/',
        license='MIT according to pinned author HF metadata',bytes_on_disk=usage(root)))
    print('Author training archive verified and extracted',flush=True)


def sparse(root):
    import compressed_segmentation as cseg
    # Public preview host certificate is expired. Explicit, host-scoped exception;
    # never used for authentication, and every received byte is hashed in receipt.
    ctx=ssl._create_unverified_context()
    def fetch(url):
        req=urllib.request.Request(url,headers={'Accept-Encoding':'gzip'})
        with urllib.request.urlopen(req,context=ctx,timeout=30) as f:
            b=f.read()
        return gzip.decompress(b) if b[:2]==b'\x1f\x8b' else b
    info_bytes=fetch(SPARSE+'/info');info=json.loads(info_bytes);s=info['scales'][0]
    assert s['key']=='64_64_80' and s['size']==[512,512,375] and info['data_type']=='uint32'
    reserve(root,512*512*375*4+10_000_000)
    folder=root/'data';folder.mkdir(exist_ok=True)
    path=folder/'sparse-axon-preview.npy'
    labels=np.lib.format.open_memmap(path,mode='w+',dtype=np.uint32,shape=(375,512,512))
    tasks=[]
    for z in range(0,375,64):
        for y in range(0,512,64):
            for x in range(0,512,64):tasks.append((x,y,z))
    def task(origin):
        x,y,z=origin;stop=(min(x+64,512),min(y+64,512),min(z+64,375))
        key=f'{x}-{stop[0]}_{y}-{stop[1]}_{z}-{stop[2]}'
        raw=fetch(SPARSE+'/'+s['key']+'/'+key)
        decoded=cseg.decompress(raw,tuple(b-a for a,b in zip(origin,stop)),dtype=np.uint32,block_size=(8,8,8),order='F')
        return origin,stop,decoded,key,hashlib.sha256(raw).hexdigest()
    receipts=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for origin,stop,decoded,key,digest in pool.map(task,tasks):
            x,y,z=origin;xx,yy,zz=stop;labels[z:zz,y:yy,x:xx]=decoded.transpose(2,1,0)
            receipts.append(dict(chunk=key,sha256=digest))
    labels.flush();ids,counts=np.unique(labels,return_counts=True)
    write_json(root/'audit'/'sparse-download.json',dict(url=SPARSE,info=info,
        info_sha256=hashlib.sha256(info_bytes).hexdigest(),labels_sha256=sha256(path),chunks=receipts,
        axis_order='zyx',voxel_size_nm=[80,64,64],origin_nm=[0,0,0],
        raw_padded_origin_shift_voxels_zyx=[20,512,512],
        source='Official challenge viewer; approximately 1% preview, not full GT',
        tls_certificate_verified=False,reason='Expired certificate on public preview host',
        ids={str(int(i)):int(c) for i,c in zip(ids,counts)}))
    print(json.dumps(dict(sparse_neurons=len(ids)-int(0 in ids),shape=list(labels.shape))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--kind',choices=['training','sparse'],required=True);a=p.parse_args()
    root=external(a.root);root.mkdir(exist_ok=True,parents=True);(root/'audit').mkdir(exist_ok=True)
    (training if a.kind=='training' else sparse)(root)

if __name__=='__main__':main()
