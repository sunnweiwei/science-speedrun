"""Replay an independent SENSE checkpoint on one explicitly centered crop."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sense import Sense, prompt_mask, candidates
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import external, device_guard, sha256, write_json


def checkpoint_config(checkpoint, voxel_size_nm):
    """Read both explicit-crop and bounded-volume trainer checkpoint schemas.

    The first volume trainer recorded geometry under ``architecture`` and used
    fixed ZYX crops with a radius-16 prompt. Keep that historical schema usable;
    newer checkpoints may explicitly record axes and prompt radius there.
    """
    if 'architecture' in checkpoint:
        architecture = checkpoint['architecture']
        if architecture.get('stage') != 'sense':
            raise ValueError('SENSE checkpoint required')
        axes = architecture.get('axes', 'zyx')
        resolution = architecture['voxel_size_nm']
        config = dict(width=architecture['width'], factors=architecture['factors'],
                      prompt_radius=architecture.get('prompt_radius', 16),
                      input_shape=architecture['input_shape'])
    else:
        axes = checkpoint['axes']; resolution = checkpoint['voxel_size_nm']
        config = dict(checkpoint['config'])
    supplied = np.asarray(voxel_size_nm, dtype=float)
    recorded = np.asarray(resolution, dtype=float)
    if (axes != 'zyx' or supplied.shape != (3,) or recorded.shape != (3,) or
        not np.isfinite(recorded).all() or (recorded <= 0).any() or
        not np.array_equal(supplied, recorded)):
        raise ValueError('Crop resolution must match training resolution')
    return config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--crop',type=Path,required=True)
    p.add_argument('--full-sizes',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--min-size',type=int,default=500)
    p.add_argument('--voxel-size-nm',type=float,nargs=3,required=True)
    a=p.parse_args();device_guard(a.device);torch.set_num_threads(2)
    out=external(a.output);out.mkdir(parents=True,exist_ok=False)
    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    config=checkpoint_config(ckpt,a.voxel_size_nm)
    model=Sense(config['width'],tuple(config['factors'])).to(a.device)
    model.load_state_dict(ckpt['model'],strict=True);model.eval()
    with np.load(a.crop,allow_pickle=False) as f:
        raw=f['raw'];seg=f['labels'];valid=f['valid']
    if (raw.shape != seg.shape or valid.shape != raw.shape or valid.dtype != np.bool_ or
        raw.ndim != 3 or seg.dtype.kind not in 'ui' or not np.isfinite(raw).all() or raw.min()<0 or raw.max()>1):
        raise ValueError('Invalid crop')
    if 'input_shape' in config and list(raw.shape) != list(config['input_shape']):
        raise ValueError('Crop field of view must match training field of view')
    center=tuple(np.asarray(seg.shape)//2)
    if not valid[center]:raise ValueError('Invalid crop center')
    seed=int(seg[center]);prompt=prompt_mask(seg,seed,config['prompt_radius']);prompt[~valid]=.5
    def tensor(x):return torch.as_tensor(x.astype(np.float32),device=a.device)[None,None]
    with torch.inference_mode():prob=model(tensor(raw),tensor(prompt)).sigmoid()[0,0].cpu().numpy()
    sizes={int(k):v for k,v in json.loads(a.full_sizes.read_text()).items()}
    # At inference labels must be the base segmentation, never evaluation GT.
    rows=candidates(prob,seg,sizes,seed,min_size=a.min_size)
    np.save(out/'probability.npy',prob,allow_pickle=False)
    write_json(out/'candidates.json',dict(candidates=rows,checkpoint_sha256=sha256(a.checkpoint),
        crop_sha256=sha256(a.crop),full_sizes_sha256=sha256(a.full_sizes),paper_comparable=False))

if __name__=='__main__':main()
