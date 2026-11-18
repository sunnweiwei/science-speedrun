#!/usr/bin/env python3
"""Compute the SSI v2 index from trusted native verifier outputs for a new evaluation.

Input directory layout: ROOT/<task slug>/reward.json, plus the task-specific report/grading
files. Pass --partial to inspect an unfinished run. The CLI never assigns a numeric 20-task
mean to an incomplete evaluation, and never executes the submitted policy.
"""
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean

TASKS = (
    'atomistic-materials','brain-computer-interface-decoding','crystal-structure-generation',
    'earth-observation-pretraining-transfer','fluid-flow-control','fruit-fly-neuron-reconstruction',
    'genome-sequence-to-function','mass-spectrometry-molecule-retrieval','molecular-pretraining-transfer',
    'natural-disease-progression','pde-surrogate','probabilistic-global-weather','protein-design',
    'protein-folding-prediction','protein-sequence-modeling','single-cell-perturbation-prediction',
    'surface-code-decoding','target-based-drug-screening','timeseries-forecasting','tokamak-magnetic-control',
)
EEG_CLASSES = {'tuev':6,'ad65':3,'openbmi':2,'longitudinal':2}
GROUP_WEIGHTS = {'qm9':6,'compas1d':4,'expansionrx':9,'asap':5}
DESIGN_TRACKS = ('unconditional','sm_binder','ppi_binder','enzyme')
MACE_CHECKPOINTS={
    'macemp0b3mediummodel':'2f2be696351ac9e94fbe01cdfb6f017679acdbd2db7645209ef55fec9826b012',
    'macempa0mediummodel':'75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638',
}
UMA_RELAXATION={'model':'uma-s-1','task':'omat','max_steps':200,'fmax_ev_per_angstrom':.02,
                'optimizer':'BFGSLineSearch','cell_filter':'FrechetCellFilter','sample_timeout_seconds':300}

def sha256_field(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def number(value):
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError('A finite numeric verifier measurement is required')
    return float(value)

def check(condition,message):
    if not condition: raise ValueError(message)

def task_score(task, root):
    used={}
    def read(*alternatives):
        for relative in alternatives:
            p=root/relative
            if p.is_file():
                content=p.read_bytes()
                used[relative]=sha256(content).hexdigest()
                return json.loads(content)
        raise FileNotFoundError(f'{root}: expected one of {alternatives}')
    rewards=read('reward.json')
    locator='reward.json/reward'
    if task=='brain-computer-interface-decoding':
        report=read('report.json');a=report['aggregate']
        check(report['scope']=='full' and set(a['tracks'])==EEG_CLASSES.keys(),'Expected the full four-track EEG v2 index')
        for track in (*EEG_CLASSES,'faced'):
            check(0<=number(rewards[track])<=1,f'Invalid EEG accuracy on {track}')
        value=mean((rewards[t]-1/k)/(1-1/k) for t,k in EEG_CLASSES.items())
        check(math.isclose(number(a['reward']),value,abs_tol=1e-10),'EEG report and track scores disagree')
        locator='report.json/aggregate.reward'
    elif task=='molecular-pretraining-transfer':
        errors={k:number(rewards[k+'_ma_rae']) for k in GROUP_WEIGHTS}
        check(min(errors.values())>=0,'Negative molecular error')
        value=1/(1+sum(GROUP_WEIGHTS[k]*e for k,e in errors.items())/sum(GROUP_WEIGHTS.values()))
        locator='SSI index over the four native group MA-RAEs'
    elif task=='protein-design':
        report=read('report.json')
        if rewards.get('invalid_submission')==1:
            check(rewards.get('reward')==0 and report.get('status')=='invalid_submission' and report.get('error'),
                  'Unverified invalid protein design submission')
            return dict(value=0.,rule=locator,receipt_sha256=used)
        check(report.get('ppi_inverse_folding')=='proteinmpnn-fixed-target/v2',
              'Expected target-conditioned PPI inverse folding')
        counts={'unconditional':20,'sm_binder':80,'ppi_binder':100,'enzyme':820}
        rates=[]
        for track in DESIGN_TRACKS:
            rate=number(rewards[track+'_pass_rate'])
            result=report['tracks'][track]
            check(result['designs_requested']==counts[track] and 0<=rate<=1 and rate==result['pass_rate'],
                  'Incomplete design track or mismatched native pass rate')
            rates.append(rate)
        value=mean(rates);locator='SSI mean of four native pass rates, including target-conditioned PPI'
    else:
        value=number(rewards['reward'])
        lower = -1 if task=='genome-sequence-to-function' else 0
        check(lower<=value<=1,'Overall reward outside the native task interval')
        if task=='single-cell-perturbation-prediction':
            rep=read('report.json')
            check(rep['schema']=='virtual-cell-zeroshot-report/v2' and rep['reward_version'].startswith('ssi-perturbation-weighted-error-skill/2.'),'Expected calibrated WMSE')
            contexts=rep['contexts'];check(set(contexts)=={'ctx_near','ctx_mid','ctx_far'},'Missing a Virtual Cell context')
            per=[number(x['v2']['reward']) for x in contexts.values()]
            check(math.isclose(value,mean(per),abs_tol=1e-10),'Invalid Virtual Cell aggregate')
        elif task=='mass-spectrometry-molecule-retrieval':
            rep=read('report.json')
            check(rep['evaluation_protocol']=='massspecgym-per-spectrum-batches/v2','Missing spectrum-isolation protocol')
            if rep.get('model_quality_measured'):
                check(rep['batch_count']==373 and len(rep['batch_record_sha256'])==64 and
                      rep.get('batch_plan_sha256')=='3ec000e7f11cfd915be9f29b3bbaa371adf43cbdde87f48fa4a622f3d497712b' and
                      rep.get('input_manifest_sha256')=='6d88d2767a1324269385bd0bd51b70e2e87c2aa10914ca27688c29f6b15f819d',
                      'Incomplete or different frozen per-spectrum evaluation')
                check(math.isclose(value,number(rep['metrics']['hit_rate_at_1']),abs_tol=1e-10),'Mismatch with raw Hit@1')
            else:check(value==0 and rep.get('status') in ('invalid_submission','invalid_output','timeout','inference_timeout','inference_failed','failed_inference'),'Invalid spectrum evaluation')
        elif task=='crystal-structure-generation':
            rep=read('report.json')
            check(rep['protocol']=='crystalgen-evaluation/v4-uma200-hull-aligned' and
                  rep['schema']=='crystalgen-report/v4-uma200-hull-aligned' and
                  rep['primary_geometry']=='uma200_relaxed' and rep['pre_scoring_relaxation'] is True and
                  sha256_field(rep['requests_sha256']), 'Expected the original crystal relaxation condition')
            if rep.get('evaluation_mode')=='sealed_replay_invalid_submission':
                check(rep.get('status')=='invalid_submission' and rep.get('error') and
                      rewards.get('invalid_submission')==1 and value==0, 'Unverified invalid crystal submission')
                return dict(value=0.,rule=locator,receipt_sha256=used)
            check(rep['evaluation_mode']=='sealed_replay_uma200_hull_aligned_exact_lemat' and
                  rep['matcher_version']=='mattergen-v1.0.3-or-ssi-dilute-0.25','Expected a sealed UMA-200 LeMat evaluation')
            check(rep['mace_checkpoint_sha256']==MACE_CHECKPOINTS and rep['mace_energy_model']=='medium-0b3' and
                  rep['mace_embedding_model']=='medium-mpa-0' and rep['hull_revision']=='70d505bb294c16658a9551e2b81ace69a28d9790',
                  'Crystal model and energy hull must match')
            check(sha256_field(rep['assets_tree_sha256']) and sha256_field(rep['sealed_snapshot_sha256']),
                  'Missing native asset or submission identity')
            relaxation=rep['relaxation']
            check(relaxation['requested']==2500 and all(relaxation['recipe'].get(k)==v for k,v in UMA_RELAXATION.items()),
                  'Incomplete or different crystal relaxation protocol')
            counts=rep['tracks']['unconditional']['counts']
            check(counts['expected']==2500 and 0<=counts['sun']+counts['metasun']<=counts['valid']<=2500,
                  'Incomplete crystal evaluation')
            check(math.isclose(value,(counts['sun']+counts['metasun'])/2500,abs_tol=1e-10),'Crystal count mismatch')
        elif task=='fluid-flow-control':
            if rewards.get('invalid_submission')==1:
                g=read('replay/grading-error.json','grading-error.json')
                check(value==0 and g['status']=='policy_error','Unverified invalid flow submission')
            else:
                g=read('replay/grading.json','grading.json')
                check(g['format']=='hydrogym-harbor-physical/v2','Expected v2 relative improvement')
                terms=g['signed_improvement']['per_case']
                check(len(terms)==26 and all(-1<=number(v)<=1 for v in terms.values()),'Incomplete fluid panel')
                check(math.isclose(value,max(0.,mean(terms.values())),abs_tol=1e-10),'Fluid term mismatch')
        elif task=='fruit-fly-neuron-reconstruction':
            if rewards.get('invalid_submission')==1:
                g=read('status.json');check(g['status']=='invalid_submission' and value==0,'Unverified invalid fly submission')
            else:
                native=number(rewards['disjoint_pair_shape_nerl'])
                check(value==native and ('raw_nerl' not in rewards or number(rewards['raw_nerl'])==native),
                      'Expected the native SHAPE nERL recorded as reward')
                value=native**.25
                locator='Historical SSI index scale: fourth root of native SHAPE nERL'
        elif task=='tokamak-magnetic-control':
            rep=read('results.json','replay/results.json');g=read('grading.json','replay/grading.json')
            check(rep['metric_version']=='tokamak-shape-metrics/v4' and rep['scope']=='full-suite','Expected native v4 full suite')
            check(math.isclose(number(g['aggregate_score']),value,abs_tol=1e-10),'Tokamak grading mismatch')
    return dict(value=value, rule=locator, receipt_sha256=used)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('receipts',type=Path)
    p.add_argument('--partial',action='store_true',help='Report pending tasks and withhold the incomplete overall')
    p.add_argument('--out',type=Path);args=p.parse_args();rows={};pending=[]
    for task in TASKS:
        root=args.receipts/task
        if args.partial and not (root/'reward.json').is_file():pending.append(task);continue
        try: rows[task]=task_score(task,root)
        except (KeyError,TypeError,FileNotFoundError,ValueError) as err:raise SystemExit(f'{task}: {err}')
    res=dict(schema='ssi-index/v2',tasks=rows,pending=pending,coverage=len(rows),task_count=len(TASKS),
             overall=mean(r['value'] for r in rows.values()) if not pending else None)
    content=json.dumps(res,indent=2,allow_nan=False)+'\n'
    if args.out:
        args.out.write_text(content)
    else: print(content,end='')

if __name__=='__main__':main()
