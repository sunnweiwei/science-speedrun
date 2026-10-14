"""Read compact progress during the existing half-hour Slurm monitor sweep."""
import json
from pathlib import Path


def collect(active):
    errors=[]
    def read(path):
        if not path.exists():return None
        try:return json.loads(path.read_text())
        except (OSError,ValueError) as error:
            errors.append(dict(path=str(path),error=str(error)))
            return None  # A concurrent partial report write is not a failed job.
    progress={'training':[],'forecasts':{},'score_cases':{},'comparisons':{},'read_errors':errors}
    pair=active.get('paired_pipeline',{})
    if pair.get('root'):
        for seed in pair.get('seeds',[]):
            root=Path(pair['root'])/f'pipeline-{seed}/curriculum'
            run=read(root/'run.json') or {}
            completed=sum(stage.get('updates',0) for stage in run.get('stages',[]))
            latest=None
            for path in sorted(root.glob('stage-*/report.json'))[-2:]:
                report=read(path)
                if report is None:continue
                completed=max([completed]+[row['global_updates'] for row in report.get('updates',[])])
                latest=dict(stage=report.get('stage',{}).get('name'),phase=report.get('phase'),
                            status=report.get('status'),report=str(path))
            item=dict(seed=seed,run_status=run.get('status'),run_phase=run.get('phase'),
                      completed_updates=completed,latest_stage=latest)
            reward=read(root/'evaluation/reward.json')
            if reward is not None:item['evaluation_reward']=reward
            evaluation=root/'evaluation'
            item['evaluation_cases']={}
            for filename in ('replay.json','weather-scores.json','precipitation-scores.json'):
                records=[read(p) for p in evaluation.glob(f'case-*/{filename}')]
                item['evaluation_cases'][filename]=sum(bool(r and r.get('status')=='passed') for r in records)
            failures=[]
            for path in sorted(evaluation.glob('case-*/failure.json')):
                report=read(path)
                if report is not None:
                    failures.append(dict(path=str(path),status=report.get('status'),
                        failure_stage=report.get('failure_stage')))
            item['evaluation_case_failures']=failures
            progress['training'].append(item)
    paper=active.get('paper_comparison',{})
    if paper.get('run_record'):
        root=Path(paper['run_record']).parent
        for model in ('gencast','wn2-seed-0','wn2-seed-1'):
            cases={}
            for path in (root/'predictions'/model).glob('*.complete.json'):
                report=read(path)
                if report and report.get('status')=='passed':
                    case=report.get('initialization','unknown')
                    cases[case]=cases.get(case,0)+1
            progress['forecasts'][model]=dict(completed_trajectories=sum(cases.values()),by_initialization=cases)
            progress['score_cases'][model]={}
            for endpoint in ('scores','pooling-scores','spectra','cyclone-scores'):
                records=[read(p) for p in (root/endpoint/model).glob('case-*.json')]
                progress['score_cases'][model][endpoint]=sum(bool(r and r.get('status')=='passed') for r in records)
        for path in (root/'comparisons').glob('*.json'):
            report=read(path)
            if report:progress['comparisons'][path.name]=dict(status=report.get('status'),path=str(path))
    progress['scope']='Progress from producer receipts; not an independent forecast rehash or final acceptance. No chat notification.'
    return progress
