"""Static development slices and continuous EM/FFN/final previews; no web server."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import external


def colors(labels):
    # Each segmentation uses its own instance IDs; zero stays black.
    n=labels.astype(np.uint64)
    # Mix the entire ID before selecting RGB; adjacent tile IDs should not
    # collapse to the small correlated palette of a single modulo operation.
    mixed=n.copy();mixed^=mixed>>30;mixed*=np.uint64(0xbf58476d1ce4e5b9)
    mixed^=mixed>>27;mixed*=np.uint64(0x94d049bb133111eb);mixed^=mixed>>31
    out=np.stack([((mixed>>shift)&255)/255 for shift in (0,8,16)],-1)
    out[n==0]=0
    return out


def render_continuous(root, assembled, prediction, output):
    from matplotlib.patches import Rectangle
    meta=json.loads((assembled/'metadata.json').read_text())
    manifest=json.loads((root/'continuous24/manifest.json').read_text())
    if meta['axes']!='zyx' or meta['voxel_size_nm']!=[16]*3:
        raise ValueError('Current preview requires the explicit 16nm ZYX experiment')
    raw=np.load(root/'continuous24/raw.npy',mmap_mode='r',allow_pickle=False)
    before=np.load(assembled/'segmentation.npy',mmap_mode='r',allow_pickle=False)
    after=np.load(prediction/'segmentation.npy',mmap_mode='r',allow_pickle=False)
    if list(before.shape)!=meta['shape'] or after.shape!=before.shape:
        raise ValueError('Prediction extent mismatch')
    offset=(np.asarray(meta['origin_nm'])-manifest['origin_nm'])/16
    if not np.equal(offset,np.floor(offset)).all():raise ValueError('Nonintegral image alignment')
    offset=offset.astype(int);shape=np.asarray(before.shape)
    if (offset<0).any() or (offset+shape>raw.shape).any():raise ValueError('Prediction outside raw image')
    z=int(shape[0]//2);lo=625;hi=shape[1:]-625
    if (hi<=lo).any() or not lo<=z<shape[0]-625:raise ValueError('No full10um center region')
    plane=raw[z+offset[0],offset[1]:offset[1]+shape[1],offset[2]:offset[2]+shape[2]]
    crop=(z,slice(lo,int(hi[0])),slice(lo,int(hi[1])))
    fig,axes=plt.subplots(1,4,figsize=(12,3.6),constrained_layout=True)
    images=[plane,plane[lo:hi[0],lo:hi[1]],colors(before[crop]),colors(after[crop])]
    for ax,im,title in zip(axes,images,['Full ROI','Central EM','FFN','After merge']):
        ax.imshow(im,cmap='gray' if im.ndim==2 else None,interpolation='nearest')
        ax.set_title(title);ax.set_xticks([]);ax.set_yticks([])
    axes[0].add_patch(Rectangle((lo,lo),int(hi[1]-lo),int(hi[0]-lo),fill=False,edgecolor='#f6aa43',linewidth=1.5))
    axes[1].plot([12,74.5],[hi[0]-lo-15]*2,color='white',linewidth=3)
    axes[1].text(12,hi[0]-lo-21,'1 µm',color='white',fontsize=9)
    fig.suptitle('Central Z slice · orange box = full 10 µm center eligibility',fontsize=11)
    fig.text(.5,-.025,'Colors denote predicted IDs · selected center region, not whole-volume quality',ha='center',fontsize=9)
    output=external(output);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=180,bbox_inches='tight');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--continuous',action='store_true')
    p.add_argument('--assembled',type=Path);p.add_argument('--prediction',type=Path)
    a=p.parse_args()
    if a.continuous:
        render_continuous(a.root,a.assembled or a.root/'continuous-assembled-v1',
            a.prediction or a.root/'continuous-evaluation-v1/reconstruction',a.output)
        return
    manifest=json.loads((a.root/'prepared/manifest.json').read_text());row=manifest['splits']['validation'][0]
    folder=Path(row['folder']);raw=np.load(folder/'raw.npy',mmap_mode='r');gt=np.load(folder/'labels.npy',mmap_mode='r')
    valid=np.load(folder/'valid.npy',mmap_mode='r')
    fig,axes=plt.subplots(4,3,figsize=(8,10),constrained_layout=True)
    for i in range(4):
        d=a.root/'dev-probe-v1'/str(i);receipt=json.loads((d/'ffn/receipt.json').read_text())
        bounds=receipt['roi'];z=(bounds[0]+bounds[1])//2;sl=(z,slice(bounds[2],bounds[3]),slice(bounds[4],bounds[5]))
        pred=np.load(d/'ffn/segmentation.npy');label=np.where(valid[sl],gt[sl],0)
        images=[raw[sl],colors(label),colors(pred[pred.shape[0]//2])]
        for j,im in enumerate(images):
            axes[i,j].imshow(im,cmap='gray' if j==0 else None,interpolation='nearest');axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
            if i==0:axes[i,j].set_title(['EM','GT instances','FFN instances'][j])
        score=json.loads((d/'ffn-metrics.json').read_text())['nerl']
        axes[i,0].set_ylabel(f'ROI {i}\nnERL {score:.3f}')
    fig.suptitle('Development ROIs · central Z slice · colors denote IDs within each panel',fontsize=10)
    out=external(a.output);out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,dpi=160);plt.close(fig)

if __name__=='__main__':main()
