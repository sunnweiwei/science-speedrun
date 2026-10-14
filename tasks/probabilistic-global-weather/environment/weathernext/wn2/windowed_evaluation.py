"""Full-suite evaluation using single-pass, lossless continuation windows."""
import json
import os
from pathlib import Path
import shutil
import sys
import threading
from training_data import sha256
from disk_budget import tree_bytes
from window_scores import merge_windows, merge_lead_chunks
from window_artifacts import capture, export_member


# Exports write NetCDF, whose library is not thread-safe, so they take turns across all cases.
EXPORT_LOCK=threading.Lock()


def serial(jobs):
    for job in jobs:job()


def evaluate_case(index, device, package, root, output, protocol_path, protocol, storage, execute,
                  model_python, tracker_bin, scales, budget, run_jobs=serial, score_chunk=None,
                  reward_lock=None):
    """Replay one case window by window and score it.

    `run_jobs` receives a list of zero-argument callables that are independent of each other and
    may run them concurrently; `score_chunk` splits each window's scorers into processes that each
    score that many consecutive leads. Neither changes a single computed number: every panel is
    scored by the same code on the same arrays, only in more processes.
    """
    from verify_track import primary_reward, discard_tracker_input
    reward_lock=reward_lock or threading.Lock()
    here=Path(__file__).resolve().parent;paper=here.parent/'paper-comparison'
    case=f'case-{index:03d}';case_root=output/'windows'/case;case_root.mkdir(parents=True)
    buffer=case_root/'buffer';state=None;window_receipts=[]
    env=os.environ|dict(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='2')
    endpoint_paths={k:[] for k in ('scores','pooling-scores','spectra')}
    for number,leads in enumerate(storage['windows']):
        window=case_root/f'window-{number:03d}';window.mkdir()
        predictions=window/'predictions';replay=window/'replay'
        command=[sys.executable,str(here/'verify_case.py'),'--package',str(package.resolve()),
            '--inputs',str(root/'data/inputs'),'--protocol',str(protocol_path),'--case-index',str(index),
            '--output',str(replay),'--predictions',str(predictions),'--lead-hours',*map(str,leads)]
        if state is not None:command+=['--state',str(state)]
        execute(command,f'{case}-window-{number:03d}-inference',os.environ|dict(CUDA_VISIBLE_DEVICES=device))
        next_state=replay/'untrusted-output/state'
        if leads[-1]<protocol['lead_hours'][-1]:
            expected={f'member-{m:03d}' for m in range(protocol['members'])}
            if (not next_state.is_dir() or next_state.is_symlink()
                or {p.name for p in next_state.iterdir()}!=expected
                or any(p.is_symlink() for p in next_state.rglob('*'))
                or tree_bytes(next_state)>storage['state_limit_bytes']):
                raise ValueError('Missing, unsafe or oversized continuation state')
        common=[sys.executable,str(paper/'score_marginals.py'),'--predictions',str(predictions),
            '--data',str(root/'data'),'--case',str(index),'--lead-hours',*map(str,leads)]
        if number:common+=['--previous-rain',str(buffer/'previous-rain.zarr')]
        commands={
            'scores':common+['--rev-climatology',str(root/'rev-climatology'),'--seeps-climatology',str(root/'seeps/climatology')],
            'pooling-scores':common+['--pooling',str(root/'pooling'),'--pooling-only'],
            'spectra':[model_python,str(paper/'score_spectra.py'),'--predictions',str(predictions),
                '--truth',str(root/'spectra/truth'/f'{case}.json'),'--case',str(index),'--lead-hours',*map(str,leads)]}
        parts=[leads[i:i+score_chunk] for i in range(0,len(leads),score_chunk)] if score_chunk else [leads]
        jobs=[];chunk_paths={}
        for endpoint,command in commands.items():
            chunk_paths[endpoint]=[]
            for part in parts:
                if len(parts)==1:
                    path=window/f'{endpoint}.json';label=f'{case}-window-{number:03d}-{endpoint}';extra=[]
                else:
                    path=window/'lead-chunks'/f'{endpoint}-{part[0]:03d}.json'
                    label=f'{case}-window-{number:03d}-{endpoint}-{part[0]:03d}'
                    extra=['--score-leads',*map(str,part)]
                chunk_paths[endpoint].append(path)
                jobs.append(lambda c=command+extra+['--output',str(path)],l=label:execute(c,l,env))
        if len(parts)>1:(window/'lead-chunks').mkdir()
        run_jobs(jobs)
        for endpoint,paths in chunk_paths.items():
            path=window/f'{endpoint}.json'
            if len(parts)>1:
                staged=path.with_suffix('.tmp')
                staged.write_text(json.dumps(merge_lead_chunks(paths,leads),allow_nan=False)+'\n');staged.replace(path)
            report=json.loads(path.read_text())
            if report.get('status')!='passed' or report.get('protocol_sha256')!=sha256(protocol_path) or report.get('lead_hours')!=leads:
                raise ValueError('Invalid window score receipt')
            if endpoint=='scores':
                with reward_lock:
                    primary_reward([report],dict(protocol,evaluation_initializations=[protocol['evaluation_initializations'][index]],lead_hours=leads),scales)
            endpoint_paths[endpoint].append(path)
        capture(predictions,buffer,index,leads,protocol)
        budget.check()
        if leads[-1]<protocol['lead_hours'][-1]:
            state=case_root/'state'
            shutil.move(str(next_state),str(state))
        else:state=None
        # All consumers finished. Keep manifests and reports, remove only arrays
        # and per-invocation model/context/cache/state copies.
        for member in range(protocol['members']):
            shutil.rmtree(predictions/f'{case}-member-{member:03d}.zarr')
        for name in ('relocated-model','untrusted-output','public'):
            shutil.rmtree(replay/name)
        window_receipts.append(dict(lead_hours=leads,replay_sha256=sha256(replay/'replay.json'),
            reports={k:sha256(v[-1]) for k,v in endpoint_paths.items()}))
        (window/'cleanup.json').write_text(json.dumps(dict(status='passed',after='window scoring and cross-window capture'))+'\n')
        budget.check()
    for endpoint,paths in endpoint_paths.items():
        destination=output/endpoint;destination.mkdir(exist_ok=True)
        (destination/f'{case}.json').write_text(json.dumps(merge_windows(paths,protocol['lead_hours']),allow_nan=False)+'\n')
    truth=output/'tracker-truth'
    execute([sys.executable,str(paper/'run_tracker.py'),'--bin',str(tracker_bin),
        '--input',str(root/'tracker/truth'/f'{case}.nc'),'--initialization',protocol['evaluation_initializations'][index],
        '--output',str(truth/f'observed-{case}')],case+'-observed-tracking',env)
    # The trackers, which dominate, overlap; at most one exported input per running job exists.
    def track(member):
        stem=f'{case}-member-{member:03d}';dest=output/'tracks'/stem
        with EXPORT_LOCK:
            export_member(buffer,output/'predictions',root/'tracker/truth',dest,index,member,protocol,sha256(protocol_path))
        budget.check()
        execute([sys.executable,str(paper/'run_tracker.py'),'--bin',str(tracker_bin),'--input',str(dest/'input.nc'),
            '--initialization',protocol['evaluation_initializations'][index],'--output',str(dest/'tracking')],stem+'-tracking',env)
        discard_tracker_input(dest)
        shutil.rmtree(buffer/f'{stem}.zarr')
    run_jobs([lambda m=member:track(m) for member in range(protocol['members'])])
    destination=output/'cyclone-scores';destination.mkdir(exist_ok=True)
    execute([sys.executable,str(paper/'score_cyclones.py'),'--predictions',str(output/'predictions'),
        '--tracks',str(output/'tracks'),'--truth-root',str(truth),'--case',str(index),
        '--output',str(destination/f'{case}.json')],case+'-cyclones',env)
    from verify_track import case_reports
    case_reports(output,index,sha256(protocol_path))
    shutil.rmtree(buffer)
    replay=output/f'replay-{index:03d}';replay.mkdir()
    (replay/'replay.json').write_text(json.dumps(dict(status='passed',case_index=index,
        members=protocol['members'],leads=protocol['lead_hours'],windows=window_receipts,
        forecast_recomputation=False,protocol_sha256=sha256(protocol_path),package_sha256=sha256(package/'package.json')))+'\n')
    budget.check()
