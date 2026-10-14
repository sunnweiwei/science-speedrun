#!/usr/bin/env python3
"""Freeze one common curriculum shortening factor only from complete profiles."""
import argparse
import hashlib
import json
import math
from pathlib import Path

# FGN appendix A.2, Table A.2: https://arxiv.org/html/2506.10772v1#A2
STAGES = [
    dict(name='era5_1deg_12h',updates=400000,warmup=1000,peak_lr=8e-4,resolution=1.,cadence_hours=12,ar_steps=1,mesh_splits=5,encoder_aggregate_normalization=1.),
    dict(name='era5_1deg_6h',updates=100000,warmup=1000,peak_lr=8e-5,resolution=1.,cadence_hours=6,ar_steps=1,mesh_splits=5,encoder_aggregate_normalization=1.),
    dict(name='era5_025deg_6h',updates=32000,warmup=1000,peak_lr=8e-5,resolution=.25,cadence_hours=6,ar_steps=1,mesh_splits=6,encoder_aggregate_normalization=4.),
]+[dict(name=f'hres_025deg_ar{ar}',updates=8000 if ar==1 else 4000 if ar==2 else 1000,
        warmup=800 if ar==1 else 400 if ar==2 else 100,peak_lr=8e-5 if ar==1 else 8e-6 if ar==2 else 8e-7,
        resolution=.25,cadence_hours=6,ar_steps=ar,mesh_splits=6,encoder_aggregate_normalization=4.) for ar in range(1,9)]


def learning_rate(stage, update_index):
    """Midpoint-sample the compressed warmup/cosine, including one-update stages.

    Midpoints avoid a one-update stage consisting solely of a zero-LR warmup.
    Warmup duration is fractional when the schedule is severely shortened.
    """
    count=stage['short_updates']
    if not 0<=update_index<count:raise ValueError('Update outside stage')
    fraction=(update_index+.5)/count
    warm_fraction=stage['warmup']/stage['updates']
    if fraction<warm_fraction:return stage['peak_lr']*fraction/warm_fraction
    return stage['peak_lr']*.5*(1+math.cos(math.pi*(fraction-warm_fraction)/(1-warm_fraction)))


def plan(profiles, training_seconds=17*3600):
    timing_margin=1.10
    if set(profiles)!=set(s['name'] for s in STAGES):raise ValueError('Every curriculum stage needs a profile')
    for p in profiles.values():
        if p.get('status')!='passed' or p.get('devices')!=4 or p.get('effective_batch')!=64 or not p.get('includes_io_and_communication'):
            raise ValueError('Profiles must measure four-device batch-64 updates including I/O and communication')
        if not (math.isfinite(p['update_seconds']) and p['update_seconds']>0 and math.isfinite(p['overhead_seconds']) and p['overhead_seconds']>=0):
            raise ValueError('Invalid timing')
    def count(stage,factor):return math.floor(stage['updates']*factor+1e-9)
    def raw_cost(factor):return sum(count(s,factor)*profiles[s['name']]['update_seconds']+profiles[s['name']]['overhead_seconds'] for s in STAGES)
    def cost(factor):return timing_margin*raw_cost(factor)
    low=1/min(s['updates'] for s in STAGES)
    if cost(low)>training_seconds:raise ValueError('Even the smallest common-factor curriculum exceeds the budget; do not freeze a different schedule silently')
    high=1.
    for _ in range(60):
        middle=(low+high)/2
        if cost(middle)<=training_seconds:low=middle
        else:high=middle
    # Avoid a floating-point floor boundary inflating the final update count.
    factor=math.nextafter(low,0)
    result=[s|dict(short_updates=count(s,factor),warmup_equivalent_updates=count(s,factor)*s['warmup']/s['updates'],
                   estimated_update_seconds=timing_margin*profiles[s['name']]['update_seconds']) for s in STAGES]
    assert all(s['short_updates']>=1 for s in result) and cost(factor)<=training_seconds
    return dict(common_shortening_factor=factor,training_wall_seconds=cost(factor),measured_cost_seconds=raw_cost(factor),
                timing_margin=timing_margin,training_cap_seconds=training_seconds,
                devices=4,effective_batch=64,total_sample_windows=64*sum(s['short_updates'] for s in result),
                learning_rate_sampling='Normalized schedule sampled at update midpoints; fractional compressed warmup',stages=result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profiles',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    raw=args.profiles.read_bytes();result=plan(json.loads(raw))
    result.update(profile_sha256=hashlib.sha256(raw).hexdigest(),reference='FGN appendix A.2, Table A.2',
                  evaluation_reserve_seconds=6*3600,checkpoint_reserve_seconds=30*60,total_wall_cap_seconds=int(23.5*3600))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
