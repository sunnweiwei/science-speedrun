"""Export small, path-independent baseline evidence; never publish model/data bytes."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(campaign, output):
    report=read(campaign/'report.json');plan=read(campaign/'plan.json');states=read(campaign/'state.json')
    reference=read(campaign.parent/'paper-drag-comparison-20260910/comparison.json')
    refs={name:{k:v[k] for k in ('value','metric','approximate','source','aggregation','panel')}
          for name,v in reference['references'].items()}
    pinball=read(campaign.parent/'best-primary-39-20260910/report.json')['pinball3d_digitization']
    refs['pinball3d_ppo']={k:pinball[k] for k in ('value','metric','approximate','source','aggregation','panel') if k in pinball}
    rows=[]
    for row in report['rows']:
        item={k:row[k] for k in ('case','primary_metric','training_complete','training_budget',
            'zero','best','best_step','algorithm','accepted','all_planned_evaluations_complete')}
        if row['accepted']:
            job=next(j for j in plan['jobs'] if j['kind']=='eval' and j.get('package')==row['best_package'])
            zero=next(j for j in plan['jobs'] if j['kind']=='eval' and j['case']==row['case'] and j['role']=='zero')
            model_receipt=read(row['evidence']);zero_state=states[zero['id']]
            assert states[job['id']]['status']=='done' and zero_state['status']=='done'
            assert model_receipt['status']=='ok'
            assert model_receipt['cases'][row['case']]['primary_metric']['value']==row['best']
            assert zero_state['result']['value']==row['zero']
            config=read(Path(job['training_root'])/'config.json')
            item.update(recipe=config['recipe'],seed=config['seed'],training_config=config,
                model_sha256=read(Path(row['best_package'])/'config.json')['model_sha256'],
                submission_sha256=model_receipt['submission_sha256'],
                evaluator_sha256=model_receipt['evaluator_sha256'],
                replay_result_sha256=sha(row['evidence']),
                uncontrolled_result_sha256=sha(zero_state['result']['evidence']),
                native_profile=model_receipt['profiles_sha256'],
                evidence_job=job['id'],uncontrolled_job=zero['id'])
        rows.append(item)
    artifact=dict(version='hydrogym-specialist-evidence/v1',as_of_utc=report['updated_utc'],
        cases=39,completed=sum(r['accepted'] for r in rows),rows=rows,paper_references=refs,
        training_seed=42,evaluation_seed=123,data_release_sha256=plan['data_release_sha256'],
        metric_version=plan['metric_version'],metrics_sha256=plan['metrics_sha256'],
        execution_image_digest=plan['image'],execution_source_sha256=plan['source_hashes'],
        selection='Best fixed physical metric among positive-step development checkpoints, on visible evaluation conditions; not blind test selection.',
        scope='Specialists, not a jointly trained model. Budgets are organizer baselines, not full paper convergence. One replay initialization, not confidence intervals.',
        legacy=[{k:v for k,v in r.items() if k not in ('checkpoint','evidence')} for r in report['legacy_paper_configuration']])
    output.mkdir(parents=True,exist_ok=True)
    (output/'results.json').write_text(json.dumps(artifact,indent=2,allow_nan=False)+'\n')
    lines=['# Verified specialist baselines','',f"Snapshot: {artifact['as_of_utc']}; {artifact['completed']}/39 complete.",'',
        'Same frozen Research v2 physics. Uncontrolled means zero physical action, not a random model.',
        'Models are per-case comparisons, not valid shared-model submissions. No paper-convergence or confidence-interval claim.',
        'Results and provenance: [results.json](results.json). Code: environment/references/specialists/RESEARCH_BASELINES.md in Git, or /workspace/references/specialists/README.md in the agent image.','',
        '| Case | Metric | Uncontrolled | Best trained | Algorithm / step | Recipe | Training budget | Paper same-metric reference |',
        '| --- | --- | ---: | ---: | --- | --- | ---: | --- |']
    keys={'Cylinder_2D_Re1000':'cylinder1000_ppo','Cylinder_2D_Re3900':'cylinder3900_ppo',
          'SquareCylinder_2D_Re200':'square200_ppo','Pinball_2D_Re100':'pinball2d_ddpg','Pinball_3D_Re150':'pinball3d_ppo'}
    fmt=lambda v:'pending' if v is None else f'{v:.4f}'
    for r in rows:
        m=r['primary_metric'];arrow='↓' if m['direction']=='lower_is_better' else '↑'
        ref=refs.get(keys.get(r['case']),{});paper=f"≈{ref['value']:.2f}" if ref else '—'
        lines.append(f"| {r['case']} | {m['id']} {arrow} | {fmt(r['zero'])} | {fmt(r['best'])} | {r['algorithm'] or 'pending'} / {r['best_step'] or '—'} | {r.get('recipe','pending')} | {r['training_budget']} | {paper} |")
    lines+=['','Paper references are approximate readings of published test curves, with different configurations/budgets; see JSON figure provenance.',
        'All cases remain in the benchmark, including the pending Cylinder_3D_Re3900. Missing results are not zeros.',
        'Historical paper-profile results are separately recorded in JSON and are never pooled with Research v2.']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'completed':artifact['completed'],'cases':39,'evidence_bytes':(output/'results.json').stat().st_size}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();publish(a.campaign,a.output)
