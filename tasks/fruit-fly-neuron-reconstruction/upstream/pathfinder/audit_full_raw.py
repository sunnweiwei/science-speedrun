"""Verify pinned full AxonEM-M archive and inspect real PNG geometry, CPU only."""
import argparse,json,time,zipfile,io
from pathlib import Path
import numpy as np
from PIL import Image
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import external,sha256,write_json
from fetch_axonem import reserve
SIZE=15207108024
HASH='93824734ec7c2c09df87f8b01e32a556446c437988618b27b5234df76659eccf'
PIN='63519706548316c1f00863cbea0ece40358a759a'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--wait-seconds',type=int,default=0);a=p.parse_args();root=external(a.root)
    final=root/'data/EM30-M-im-pad.zip';part=final.with_suffix('.zip.partial');start=time.monotonic()
    while not final.exists() and (not part.exists() or part.stat().st_size<SIZE):
        if time.monotonic()-start>=a.wait_seconds:raise ValueError('Download not complete')
        time.sleep(30)
    path=final if final.exists() else part
    if path.stat().st_size!=SIZE or sha256(path)!=HASH:raise ValueError('Pinned archive hash/size mismatch')
    if path!=final:path.replace(final)
    inspected=[]
    with zipfile.ZipFile(final) as archive:
        files=[f for f in archive.infolist() if f.filename.lower().endswith('.png')]
        reserve(root,0)
        if len(files)!=790:raise ValueError('Unexpected raw archive slice count')
        for f in files:
            if Path(f.filename).is_absolute() or '..' in Path(f.filename).parts:raise ValueError('Unsafe archive member')
        chosen={f.filename for f in files if f.file_size<100000}|{files[i].filename for i in [0,20,len(files)//2,len(files)-21,len(files)-1]}
        for name in sorted(chosen):
            with Image.open(io.BytesIO(archive.read(name))) as im:arr=np.asarray(im)
            if arr.shape!=(5120,5120) or arr.dtype!=np.uint8:raise ValueError('Unexpected PNG geometry/type')
            inspected.append(dict(name=name,shape=list(arr.shape),dtype=str(arr.dtype),minimum=int(arr.min()),maximum=int(arr.max()),nonzero_voxels=int(np.count_nonzero(arr))))
        result=dict(revision=PIN,archive_sha256=HASH,archive_bytes=SIZE,png_count=len(files),
            png_uncompressed_bytes=sum(f.file_size for f in files),sampled_slices=inspected,
            raw_download_complete=True,extracted=False,ground_truth_downloaded=False,
            source='https://axonem.grand-challenge.org/',
            url=f'https://huggingface.co/datasets/pytc/EM30/resolve/{PIN}/EM30-M-im-pad.zip',
            paper_comparable=False,geometry_verified_for_full_pipeline=False,
            note='PNG sample audit only; missing/blank slices and archive-viewer extent difference must be handled by adapter')
    write_json(root/'audit/full-raw-download.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
