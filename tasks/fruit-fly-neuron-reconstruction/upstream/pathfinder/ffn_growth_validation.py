"""Validation-only FFN admission using the unchanged production grow routine.

Labels score completed predictions; they never enter the grow generator.
The reference and candidate use identical fixed crops, thresholds and caps.
This is a seed-growth gate; full-volume nERL/consensus admission is separate.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from scipy import ndimage

from batched_ffn import grow_tile
from volume_data import Crops
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import device_guard, sha256, write_json
from execution_resources import device_guard


def growth_validation(model, manifest, device, *, samples=16, shape=97,
                      max_calls=512, bf16=False):
    if samples < 1 or shape < 65 or shape % 2 != 1 or max_calls < 1:
        raise ValueError('Positive samples/caps and odd crop >=65 required')
    data = Crops(manifest, 'validation', shape, samples, seed=125)
    rows = []; begin = time.monotonic()
    model.eval()
    for index in range(samples):
        raw, _, truth, valid = data[index]
        image = np.rint(raw[0].numpy()*255).astype(np.uint8)
        def seeds(context, start, stop):
            return np.array([[shape//2]*3])
        gen = grow_tile(image, [16]*3, [shape-16]*3,
                        max_calls=max_calls, seed_provider=seeds)
        started = time.monotonic()
        try:
            request = next(gen)
            while True:
                x, canvas = request
                with torch.inference_mode(), torch.autocast(
                        'cuda', dtype=torch.bfloat16, enabled=bf16):
                    logits = model(torch.from_numpy(x[None,None]).to(device),
                                   torch.from_numpy(canvas[None,None]).to(device))
                request = gen.send(logits[0,0].float().cpu().numpy())
        except StopIteration as done:
            segmentation, execution = done.value
        positive = segmentation > 0
        target = truth[0,16:-16,16:-16,16:-16].numpy() > .5
        mask = valid[0,16:-16,16:-16,16:-16].numpy() > .5
        cc, count = ndimage.label(positive)
        sizes = np.bincount(cc.ravel())[1:]
        rows.append(dict(index=index, tp=int((positive&target&mask).sum()),
                         fp=int((positive&~target&mask).sum()),
                         fn=int((~positive&target&mask).sum()),
                         components=int(count), tiny_components=int((sizes<10).sum()),
                         positive_voxels=int(positive.sum()), **execution,
                         seconds=time.monotonic()-started))
    totals = {key:sum(row[key] for row in rows) for key in
              ('tp','fp','fn','calls','components','tiny_components','positive_voxels')}
    tp, fp, fn = [totals[k] for k in ('tp','fp','fn')]
    totals.update(dice=2*tp/max(1,2*tp+fp+fn), recall=tp/max(1,tp+fn),
                  precision=tp/max(1,tp+fp),
                  components_per_positive_voxel=totals['components']/max(1,totals['positive_voxels']),
                  calls_per_reference_voxel=totals['calls']/max(1,tp+fn),
                  calls_per_correct_voxel=totals['calls']/max(1,tp),
                  exhausted_samples=sum(row['budget_exhausted'] for row in rows))
    return dict(contract=dict(schema=2, split='validation', manifest_sha256=sha256(manifest),
                              seed=125, samples=samples, shape=shape, max_calls=max_calls,
                              bf16=bf16, movement='production_face_max', disconnectedness_bias=0,
                              assignment_threshold=.95, seed_policy='one_fixed_labeled_center'),
                totals=totals, rows=rows, seconds=time.monotonic()-begin,
                ground_truth_used_for_inference=False, full_volume_admission=False)


def admit(candidate, reference):
    """Reject regressions or incomplete/censored measurements, fail closed."""
    if candidate['contract'] != reference['contract']:
        raise ValueError('Validation reference contract differs')
    c, r = candidate['totals'], reference['totals']
    reasons = []
    for name, stats in [('reference',r),('candidate',c)]:
        if any(not np.isfinite(value) for value in stats.values()):
            raise ValueError(name+' has nonfinite validation measurements')
        if stats['exhausted_samples'] or stats['tp'] == 0:
            reasons.append(name+' incomplete or empty')
    for key, slack in [('dice',0.),('recall',0.),('precision',.01)]:
        if c[key] < r[key]-slack:
            reasons.append(key+' below reference')
    for key, factor in [('components_per_positive_voxel',1.),
                        ('calls_per_correct_voxel',1.2)]:
        if c[key] > r[key]*factor:
            reasons.append(key+' exceeds reference bound')
    return dict(passed=not reasons, reasons=reasons,
                scope='fixed validation seed growth; full consensus/nERL gate still required')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--reference',type=Path)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--samples',type=int,default=16)
    p.add_argument('--shape',type=int,default=97)
    p.add_argument('--max-calls',type=int,default=512)
    a=p.parse_args();device_guard(a.device)
    from ffn_model import FFN15
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    model=FFN15(activation_checkpointing=False).to(a.device)
    model.load_state_dict(ck['model'],strict=True)
    torch.set_num_threads(2 if a.device=='cuda' else 8)
    report=growth_validation(model,a.manifest,a.device,samples=a.samples,shape=a.shape,
                             max_calls=a.max_calls,
                             bf16=a.device=='cuda' and torch.cuda.is_bf16_supported())
    report.update(checkpoint_sha256=sha256(a.checkpoint),step=ck['step'])
    if a.reference:report['admission']=admit(report,json.loads(a.reference.read_text()))
    write_json(a.output,report)
    print(json.dumps(report['totals']),flush=True)
    if a.reference and not report['admission']['passed']:raise SystemExit(2)


if __name__=='__main__':main()
