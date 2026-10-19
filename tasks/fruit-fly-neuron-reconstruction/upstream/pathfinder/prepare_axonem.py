"""Freeze spatial AxonEM splits and make bounded 16 nm resampled volumes.

40 nm Z information is interpolated, not newly resolved. Only official dense
annotation cores contribute to training/evaluation. Split IDs are local to each
block; this is a spatial holdout, not a verified neuron-identity holdout.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import h5py
from scipy.ndimage import map_coordinates
from fetch_axonem import reserve,usage
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import external,sha256,write_json

SPLITS={'train':['0-3584-0','350-1792-1792','700-0-0','700-0-3584'],
        'validation':['700-3584-0'],'test':['700-3584-3584']}
EXCLUDED=['0-0-0','0-0-3584','0-3584-3584']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=external(a.root);out=root/'prepared';out.mkdir(exist_ok=True)
    manifest=dict(schema='axonem-spatial16/v1',dataset_id='AxonEM-M-author-training-blocks',
        axes='zyx',voxel_size_nm=[16,16,16],native_voxel_size_nm=[40,8,8],
        interpolation='raw linear; instance labels nearest; explicit voxel-center mapping',
        excluded_low_label_diversity_blocks=EXCLUDED,split_kind='spatial, not proven neuron-disjoint',
        paper_comparable=False,source_receipt_sha256=sha256(root/'audit/training-download.json'),splits={})
    # Small cores plus real EM context; no need to store the entire padded image.
    # 90 x 768 x 768 native crop -> 225 x 384 x 384 at 16 nm.
    shape=(225,384,384);reserve(root,int(np.prod(shape))*4*6+100_000_000)
    yy,xx=np.meshgrid((np.arange(384)+.5)*2-.5,(np.arange(384)+.5)*2-.5,indexing='ij')
    for split,names in SPLITS.items():
        manifest['splits'][split]=[]
        for name in names:
            dest=out/name;dest.mkdir(exist_ok=True)
            rawpath=root/'data'/('im_'+name+'_pad.h5');labelpath=root/'data'/('seg_'+name+'_pad.h5')
            with h5py.File(rawpath) as f:raw=f['main'][:,384:1152,384:1152]
            with h5py.File(labelpath) as f:labels=f['main'][:,384:1152,384:1152]
            images=np.lib.format.open_memmap(dest/'raw.npy',mode='w+',dtype=np.uint8,shape=shape)
            gt=np.lib.format.open_memmap(dest/'labels.npy',mode='w+',dtype=np.uint16,shape=shape)
            mask=np.lib.format.open_memmap(dest/'valid.npy',mode='w+',dtype=np.bool_,shape=shape)
            for z in range(shape[0]):
                zz=(z+.5)*16/40-.5
                coords=np.array([np.full(yy.shape,zz),yy,xx])
                images[z]=map_coordinates(raw,coords,order=1,mode='nearest',prefilter=False)
                gt[z]=map_coordinates(labels,coords,order=0,mode='nearest',prefilter=False)
                mask[z]=(zz+.5>=20)&(zz+.5<70)&(yy+.5>=128)&(yy+.5<640)&(xx+.5>=128)&(xx+.5<640)&(gt[z]>0)
            for arr in [images,gt,mask]:arr.flush()
            # A bounded deterministic candidate pool; foreground/neurite identity
            # is NOT inferred from test performance.
            sampled=np.asarray(gt)[::2,::2,::2];valid=np.asarray(mask)[::2,::2,::2]
            coords=np.argwhere(valid)*2
            rng=np.random.default_rng(20260925);rng.shuffle(coords);coords=coords[:20000]
            np.save(dest/'centers.npy',coords.astype(np.int16))
            native_start=np.array(list(map(int,name.split('-'))))
            # Archive padded origin is native start - [20,512,512].
            origin=(native_start+np.array([-20,-128,-128]))*np.array([40,8,8])
            record=dict(name=name,folder=str(dest),origin_nm=origin.tolist(),shape=list(shape),
                dense_core_zyx=[[50,175],[64,320],[64,320]],
                raw_source_sha256=sha256(rawpath),labels_source_sha256=sha256(labelpath),
                files={n:sha256(dest/n) for n in ['raw.npy','labels.npy','valid.npy','centers.npy']})
            manifest['splits'][split].append(record)
            write_json(out/'manifest.json',manifest)
            print(json.dumps(dict(prepared=name,split=split,bytes_on_disk=usage(root))),flush=True)
    write_json(out/'manifest.json',manifest)

if __name__=='__main__':main()
