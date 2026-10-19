"""Create physical surface meshes from public sparse AxonEM preview labels.

These are preview-derived shapes, not complete challenge GT or paper SHAPE data.
No sampling into a smaller model context: list only centers with full 10um radius.
"""
import argparse,json
from pathlib import Path
import numpy as np
from skimage.measure import marching_cubes
from fetch_axonem import reserve
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import external,sha256,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=external(a.root)
    path=root/'data/sparse-axon-preview.npy';seg=np.load(path,mmap_mode='r')
    receipt=json.loads((root/'audit/sparse-download.json').read_text())
    ids=sorted(int(i) for i in receipt['ids'] if int(i)>0)
    rng=np.random.default_rng(42);rng.shuffle(ids)
    split={sid:('train' if i<47 else 'validation' if i<57 else 'test') for i,sid in enumerate(ids)}
    out=root/'shape-meshes';out.mkdir(exist_ok=True);reserve(root,1_000_000_000)
    spacing=np.array([.08,.064,.064]);extent=np.array(seg.shape)*spacing
    rows=[]
    for sid in ids:
        coords=np.where(seg==sid);lo=np.array([x.min() for x in coords]);hi=np.array([x.max()+1 for x in coords])
        mask=np.asarray(seg[tuple(slice(a,b) for a,b in zip(lo,hi))]==sid)
        vertices,faces,_,_=marching_cubes(np.pad(mask,1),level=.5,spacing=tuple(spacing))
        vertices+=(lo-1)*spacing
        eligible=np.where(((vertices>=10)&(vertices<=extent-10)).all(axis=1))[0]
        if len(eligible)>256:eligible=rng.choice(eligible,256,replace=False)
        dest=out/f'{sid}.npz';np.savez_compressed(dest,vertices_um=vertices.astype(np.float32),faces=faces.astype(np.int32),centers=eligible)
        rows.append(dict(id=sid,split=split[sid],path=str(dest),sha256=sha256(dest),vertices=len(vertices),faces=len(faces),full_context_centers=len(eligible)))
    result=dict(source_sha256=sha256(path),voxel_size_nm=[80,64,64],radius_um=10,points_per_model_input=8192,
        axes='zyx',coordinate_units='um',splits='fixed seed42 shuffle of preview instance IDs; not cross-stage neuron-disjoint proof',
        rows=rows,paper_comparable=False,sparse_preview_only=True,
        eligible_instances={s:sum(r['split']==s and r['full_context_centers']>0 for r in rows) for s in ['train','validation','test']})
    write_json(out/'manifest.json',result);print(json.dumps(result['eligible_instances']),flush=True)

if __name__=='__main__':main()
