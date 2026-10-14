#!/usr/bin/env python3
"""Independent paper-subset replay and complete scoring from a single forecast set."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import threading
import numpy as np
import xarray as xr
from training_data import sha256
from merge_scores import reward


def primary_reward(reports,protocol,scales):
    """Require every native weather panel before computing the repo scalar."""
    expected={(date,name,level,lead) for date in protocol['evaluation_initializations']
        for name,spec in protocol['variables'].items()
        for level in (protocol['pressure_levels_hpa'] if spec['pressure_levels'] else [None])
        for lead in protocol['lead_hours']}
    rows={}
    for report in reports:
        if report.get('status')!='passed' or report.get('members')!=protocol['members']:
            raise ValueError('Unverified complete ensemble score')
        for row in report['per_case']:
            if row['variable'] not in protocol['variables']:continue
            key=tuple(row[k] for k in ('case','variable','level_hpa','lead_hours'))
            if key in rows or key not in expected:raise ValueError('Duplicate or unexpected primary panel')
            if not np.isfinite(row['crps']) or row['crps']<0:raise ValueError('Invalid CRPS')
            rows[key]=row
    if rows.keys()!=expected:raise ValueError('Missing primary panels')
    # Equal dates, then levels/leads within variable, then equal variables.
    return reward({'aggregate':[dict(row,pool_cells=1) for row in rows.values()]},scales)


ENDPOINTS = ('scores', 'pooling-scores', 'spectra', 'cyclone-scores')
# Fewest leads per window worth running cases side by side for; below it, restarts dominate.
MIN_CONCURRENT_WINDOW = 3


def case_reports(output, index, protocol_hash):
    reports = {}
    for endpoint in ENDPOINTS:
        path = output / endpoint / f'case-{index:03d}.json'
        report = json.loads(path.read_text())
        if report.get('status') != 'passed' or report.get('protocol_sha256') != protocol_hash:
            raise ValueError(f'Incomplete suite: {endpoint}/case-{index:03d}')
        reports[endpoint] = report
    return reports


def discard_tracker_input(directory):
    """Keep export/track receipts; the scorer reads those, not the large NetCDF."""
    exported = json.loads((directory / 'input.json').read_text())
    tracked = json.loads((directory / 'tracking/complete.json').read_text())
    if (exported.get('status') != 'passed' or tracked.get('status') != 'passed'
            or tracked.get('input_sha256') != exported.get('sha256')
            or not exported.get('sha256')):
        raise ValueError('Cannot discard an unverified tracker input')
    (directory / 'input.nc').unlink()


def discard_case_forecasts(output, index, members):
    """Remove only this completed case's large products, retaining audit receipts."""
    paths = [output / 'predictions' / f'case-{index:03d}-member-{member:03d}.zarr'
             for member in range(members)]
    replay = output / f'replay-{index:03d}'
    paths += [replay / name for name in ('relocated-model', 'untrusted-output')
              if (replay / name).exists() or (replay / name).is_symlink()]
    for path in paths:
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f'Expected an ordinary output directory: {path}')
    removed = []
    for path in paths:
        shutil.rmtree(path)
        removed.append(str(path.relative_to(output)))
    (replay / 'cleanup.json').write_text(json.dumps(dict(
        status='passed', removed=removed, after='all case score reports validated')) + '\n')


def finish_case(output, index, protocol, protocol_hash, scales, keep_predictions):
    checked = case_reports(output, index, protocol_hash)
    primary_reward([checked['scores']], dict(protocol, evaluation_initializations=[
        protocol['evaluation_initializations'][index]]), scales)
    if not keep_predictions:
        discard_case_forecasts(output, index, protocol['members'])


def bounded_case_pipeline(count, devices, max_inflight, score_workers, infer, score):
    """A slot covers inference, all scoring and cleanup; no completed-case backlog."""
    workers = min(count, len(devices), max_inflight)
    if workers < 1 or score_workers < 1:
        raise ValueError('Positive case/GPU/scoring concurrency required')
    stopped = threading.Event()
    cpu_slots = threading.BoundedSemaphore(score_workers)

    def worker(slot):
        try:
            for index in range(slot, count, workers):
                if stopped.is_set():
                    return
                infer(index, devices[slot])
                with cpu_slots:
                    score(index)
        except BaseException:
            stopped.set()
            raise

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(worker, slot) for slot in range(workers)]):
            future.result()


def available_cpus():
    """CPUs this process may use: its affinity, capped by a cgroup v2 quota (`docker --cpus`)."""
    count=len(os.sched_getaffinity(0))
    try:
        quota,period=Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if quota!='max':count=min(count,max(1,int(quota)//int(period)))
    except (OSError,ValueError):pass
    return count


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('package','benchmark','output'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--budget-seconds',type=int,default=43200)
    ap.add_argument('--phase',choices=['all','forecast','score'],default='all')
    ap.add_argument('--score-workers',type=int,default=0,
                    help='Concurrent CPU scoring processes; 0 means one per available CPU (legacy mode: 2)')
    ap.add_argument('--concurrent-cases',type=int,default=0,
                    help='Cases evaluated at once, one GPU each; 0 picks the most the GPUs and disk allowance fit')
    ap.add_argument('--score-chunk-leads',type=int,default=1,
                    help='Leads scored per process within a window; results do not depend on it')
    ap.add_argument('--max-inflight-cases',type=int,default=2,
                    help='Legacy case mode only: cases resident through inference/scoring/cleanup')
    ap.add_argument('--disk-allowance-gb',type=float,default=500,help='Total decimal GB: inputs, package, outputs and reserve')
    ap.add_argument('--disk-reserve-gb',type=float,default=25,help='Reserved for image/runtime and untracked workspace')
    ap.add_argument('--legacy-case-mode',action='store_true',help='Opt out of continuation; requires full-case storage')
    ap.add_argument('--keep-predictions',action='store_true',
                    help='Retain predictions and tracker inputs; requires full-dataset disk space')
    a=ap.parse_args()
    if a.score_workers==0:
        a.score_workers=2 if a.legacy_case_mode else available_cpus()
    if a.budget_seconds<=0 or a.score_workers<1 or a.max_inflight_cases<1 or a.concurrent_cases<0 or a.score_chunk_leads<1:
        ap.error('Positive budget and worker count required')
    if a.phase=='forecast' and not a.keep_predictions:
        ap.error('--phase forecast requires --keep-predictions and full-dataset disk space')
    if (a.phase!='all' or a.keep_predictions) and not a.legacy_case_mode:
        ap.error('Split phases and retained forecasts require explicit --legacy-case-mode')
    disk_budget=None
    begin=time.monotonic();a.output.mkdir(parents=True,exist_ok=a.phase=='score')
    here=Path(__file__).resolve().parent;paper=here.parent/'paper-comparison'
    protocol=here/'protocol.json';p=json.loads(protocol.read_text());root=a.benchmark.resolve()
    predictions=a.output/'predictions';predictions.mkdir(exist_ok=a.phase=='score')
    devices=[x for x in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if x]
    if a.phase!='score' and not devices:raise ValueError('Allocated inference GPUs required')
    model_python=os.environ.get('WEATHER_MODEL_PYTHON','/opt/weather-python/bin/python')
    tracker_bin=Path(os.environ.get('WEATHER_TRACKER_BIN','/opt/tempestextremes/bin'))
    stopped=threading.Event()
    def execute(command,label,env=None):
        remaining=a.budget_seconds-(time.monotonic()-begin)
        if remaining<=0:raise TimeoutError('Evaluation budget exhausted')
        if stopped.is_set():raise RuntimeError('Evaluation stopped after a failure elsewhere')
        with (a.output/(label+'.log')).open('wb') as log:
            proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                while True:
                    if stopped.is_set():raise RuntimeError('Evaluation stopped after a failure elsewhere')
                    if disk_budget is not None:disk_budget.check()
                    remaining=a.budget_seconds-(time.monotonic()-begin)
                    if remaining<=0:raise TimeoutError('Evaluation budget exhausted')
                    try:
                        code=proc.wait(timeout=min(2,remaining))
                        break
                    except subprocess.TimeoutExpired:pass
                if code:raise subprocess.CalledProcessError(code,command)
                if disk_budget is not None:disk_budget.check()
            finally:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                proc.wait()
    def score(index):
        case=f'case-{index:03d}';env=os.environ|dict(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='2')
        for endpoint in ('scores','pooling-scores','spectra','cyclone-scores'):(a.output/endpoint).mkdir(exist_ok=True)
        common=[sys.executable,str(paper/'score_marginals.py'),'--predictions',str(predictions),
            '--data',str(root/'data'),'--case',str(index)]
        execute(common+['--rev-climatology',str(root/'rev-climatology'),'--seeps-climatology',str(root/'seeps/climatology'),
            '--output',str(a.output/'scores'/f'{case}.json')],case+'-core',env)
        execute(common+['--pooling',str(root/'pooling'),'--pooling-only','--output',str(a.output/'pooling-scores'/f'{case}.json')],case+'-pooling',env)
        execute([model_python,str(paper/'score_spectra.py'),'--predictions',str(predictions),'--truth',str(root/'spectra/truth'/f'{case}.json'),
            '--case',str(index),'--output',str(a.output/'spectra'/f'{case}.json')],case+'-spectra',env)
        execute([sys.executable,str(paper/'run_tracker.py'),'--bin',str(tracker_bin),
            '--input',str(root/'tracker/truth'/f'{case}.nc'),
            '--initialization',p['evaluation_initializations'][index],
            '--output',str(tracker_truth/f'observed-{case}')],case+'-observed-tracking',env)
        tracks=a.output/'tracks'
        for member in range(p['members']):
            stem=f'{case}-member-{member:03d}';dest=tracks/stem
            execute([sys.executable,str(paper/'export_tracker_forecast.py'),'--predictions',str(predictions),
                '--history',str(root/'tracker/truth'),'--case',str(index),'--member',str(member),'--output',str(dest/'input.nc')],stem+'-export',env)
            execute([sys.executable,str(paper/'run_tracker.py'),'--bin',str(tracker_bin),'--input',str(dest/'input.nc'),
                '--initialization',p['evaluation_initializations'][index],'--output',str(dest/'tracking')],stem+'-tracking',env)
            if not a.keep_predictions:discard_tracker_input(dest)
        execute([sys.executable,str(paper/'score_cyclones.py'),'--predictions',str(predictions),'--tracks',str(tracks),
            '--truth-root',str(tracker_truth),'--case',str(index),'--output',str(a.output/'cyclone-scores'/f'{case}.json')],case+'-cyclones',env)
        finish_case(a.output,index,p,sha256(protocol),scales,a.keep_predictions)
    # Run the observed tracks with the same container binaries as the candidate.
    tracker_truth=a.output/'tracker-truth';tracker_truth.mkdir(exist_ok=True)
    if not (tracker_truth/'truth').exists():
        (tracker_truth/'truth').symlink_to(root/'tracker/truth',target_is_directory=True)
    shutil.copyfile(root/'tracker/named-truth.json',tracker_truth/'named-truth.json')
    def infer(index,device):
        execute([sys.executable,str(here/'verify_case.py'),'--package',str(a.package.resolve()),
            '--inputs',str(root/'data/inputs'),'--protocol',str(protocol),'--case-index',str(index),
            '--output',str(a.output/f'replay-{index:03d}'),'--predictions',str(predictions)],
            f'case-{index:03d}-inference',os.environ|dict(CUDA_VISIBLE_DEVICES=device))
    try:
        # Load once before starting threads; NetCDF backends are not thread-safe.
        with xr.open_dataset(root/'training-scales.nc') as source:
            scales=source.load()
        forecast_identity=dict(protocol_sha256=sha256(protocol),package_sha256=sha256(a.package/'package.json'))
        if a.phase=='score':
            previous=json.loads((a.output/'forecast-status.json').read_text())
            if previous.get('status')!='passed' or any(previous.get(k)!=v for k,v in forecast_identity.items()):
                raise ValueError('Missing or changed independent forecast phase')
            for index in range(len(p['evaluation_initializations'])):
                path=a.output/f'replay-{index:03d}'/'replay.json'
                if sha256(path)!=previous['replay_receipts'][str(index)]:raise ValueError('Changed replay receipt')
        if a.phase=='score':
            if previous.get('predictions_retained') is False:
                raise ValueError('Predictions were cleaned; a retained forecast run is required for rescoring')
            if not a.keep_predictions:
                previous['predictions_retained']=False
                (a.output/'forecast-status.json').write_text(json.dumps(previous)+'\n')
            with ThreadPoolExecutor(max_workers=a.score_workers) as cpu:
                futures=[cpu.submit(score,i) for i in range(len(p['evaluation_initializations']))]
                for result in as_completed(futures):result.result()
        elif a.phase=='all' and not a.legacy_case_mode:
            from disk_budget import plan, tree_bytes, DiskBudget
            from windowed_evaluation import evaluate_case
            if 'wn2-inference/v3' not in json.loads((a.package/'package.json').read_text()).get('inference_schemas',[]):
                raise ValueError('Repackage existing trained weights with upgrade_inference.py for v3 continuation; no training needed')
            resource_record=json.loads((root/'resources.json').read_text())
            resource_bytes=sum((root/name).stat().st_size for name in resource_record['files'])+(root/'resources.json').stat().st_size
            count=len(p['evaluation_initializations'])
            def storage_for(cases):
                return plan(p,a.disk_allowance_gb,resource_bytes,tree_bytes(a.package),a.disk_reserve_gb,cases)
            if a.concurrent_cases:
                storage=storage_for(min(a.concurrent_cases,len(devices),count))
            else:
                # More cases at once beats longer windows, as long as a window keeps a few leads
                # per model launch; the single-case plan is the fallback and fails if nothing fits.
                storage=storage_for(1)
                for cases in range(min(len(devices),count),1,-1):
                    try:candidate=storage_for(cases)
                    except ValueError:continue
                    if candidate['lead_window']>=MIN_CONCURRENT_WINDOW:
                        storage=candidate;break
            (a.output/'storage-plan.json').write_text(json.dumps(storage,indent=2)+'\n')
            disk_budget=DiskBudget(a.output,storage,interval=1.0);disk_budget.check()
            cpu_slots=threading.BoundedSemaphore(a.score_workers);reward_lock=threading.Lock();failures=[]
            def run_jobs(jobs):
                def run(job):
                    with cpu_slots:
                        if stopped.is_set():raise RuntimeError('Evaluation stopped after a failure elsewhere')
                        try:job()
                        except BaseException as exc:
                            failures.append(exc);stopped.set();raise
                with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
                    for future in as_completed([pool.submit(run,job) for job in jobs]):future.result()
            pending=iter(range(count));pending_lock=threading.Lock()
            def case_worker(device):
                try:
                    while not stopped.is_set():
                        with pending_lock:index=next(pending,None)
                        if index is None:return
                        evaluate_case(index,device,a.package,root,a.output,protocol,p,storage,execute,model_python,
                                      tracker_bin,scales,disk_budget,run_jobs=run_jobs,
                                      score_chunk=a.score_chunk_leads,reward_lock=reward_lock)
                except BaseException as exc:
                    failures.append(exc);stopped.set()
            with ThreadPoolExecutor(max_workers=storage['concurrent_cases']) as gpus:
                for slot in range(storage['concurrent_cases']):gpus.submit(case_worker,devices[slot])
            # The first failure is the cause; the others are the cases it stopped.
            if failures:raise failures[0]
        elif a.phase=='all':
            bounded_case_pipeline(len(p['evaluation_initializations']),devices,
                                  a.max_inflight_cases,a.score_workers,infer,score)
        else:
            def forecasts(slot):
                for index in range(slot,len(p['evaluation_initializations']),len(devices)):
                    infer(index,devices[slot])
            with ThreadPoolExecutor(max_workers=len(devices)) as gpu:
                for result in as_completed([gpu.submit(forecasts,i) for i in range(len(devices))]):result.result()
        receipt=dict(status='passed',**forecast_identity,predictions_retained=a.keep_predictions,
            replay_receipts={str(i):sha256(a.output/f'replay-{i:03d}'/'replay.json')
                             for i in range(len(p['evaluation_initializations']))})
        (a.output/'forecast-status.json').write_text(json.dumps(receipt)+'\n')
        if a.phase=='forecast':return
        reports=[];receipts={}
        for endpoint in ('scores','pooling-scores','spectra','cyclone-scores'):
            for index in range(len(p['evaluation_initializations'])):
                path=a.output/endpoint/f'case-{index:03d}.json';r=json.loads(path.read_text())
                if r.get('status')!='passed' or r.get('protocol_sha256')!=sha256(protocol):raise ValueError('Incomplete suite')
                receipts[str(path.relative_to(a.output))]=sha256(path)
                if endpoint=='scores':reports.append(r)
        with xr.open_dataset(root/'training-scales.nc') as scales:metrics=primary_reward(reports,p,scales)
        (a.output/'reward.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')
        status=dict(status='passed',protocol_sha256=sha256(protocol),cases=8,members=56,
            forecast_days=15,independent_model_replay=True,score_receipts_sha256=receipts,
            seconds=time.monotonic()-begin,training_provenance_verified=False,
            predictions_retained=a.keep_predictions,
            max_inflight_cases=(disk_budget.plan['concurrent_cases'] if disk_budget is not None else min(a.max_inflight_cases,len(devices),len(p['evaluation_initializations']))) if a.phase=='all' else None,
            storage_mode='continuation-windows' if disk_budget is not None else 'legacy-cases',
            measured_accounted_peak_bytes=disk_budget.peak if disk_budget else None,
            disk_allowance_bytes=disk_budget.plan['disk_allowance_bytes'] if disk_budget else None,
            scope='Unified paper-subset independent inference and all metric endpoints; training eligibility and baseline comparison audited separately')
        (a.output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    except Exception as exc:
        (a.output/'reward.json').write_text('{"reward":0.0}\n')
        (a.output/'status.json').write_text(json.dumps(dict(status='failed',error=repr(exc),measured_accounted_peak_bytes=disk_budget.peak if disk_budget else None))+'\n')
        raise

if __name__=='__main__':main()
