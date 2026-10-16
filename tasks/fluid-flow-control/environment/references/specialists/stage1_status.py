"""Snapshot manual reproduction evidence, without assigning paper-match scores.

This is an operator report, not a new benchmark manifest or evaluation contract.
Each plan row explicitly chooses a training run and an independent checkpoint
evaluation. A short pilot's score must never be presented as the full run's score.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from summarize_public import records


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(root, spec):
    report = dict(spec)
    training = root / spec['training_run']
    rows = records(training / 'episodes.jsonl')
    steps = sum(row['steps'] for row in rows if row['phase'] == 'training')
    report.update(completed_training_controls=steps,
                  budget_progress_percent=100 * steps / spec['planned_controls'],
                  run_budget_completed=(training / 'completed.json').exists(),
                  planned_budget_completed=steps >= spec['planned_controls'])
    config = training / 'config.json'
    report['training_run_started'] = config.exists()
    if config.exists():
        cfg = json.loads(config.read_text())
        report.update(algorithm=cfg['algorithm'], seed=cfg['seed'], recipe=cfg['recipe'],
                      actual_run_budget=cfg['training_steps'], config_sha256=digest(config))
    evaluation = root / spec['evaluation_dir']
    completed = evaluation / 'completed.json'
    if completed.exists():
        completion = json.loads(completed.read_text())
        checkpoint = root / spec['checkpoint']
        if digest(checkpoint) != completion['checkpoint_sha256']:
            raise ValueError(f'Evaluated checkpoint was changed: {checkpoint}')
        protocol = json.loads((evaluation / 'protocol.json').read_text())
        if protocol['environment'] != spec['case']:
            raise ValueError('Evaluation case mismatch')
        measured = records(evaluation / 'episodes.jsonl')
        if len(measured) != len(completion['returns']) or not measured:
            raise ValueError('Independent evaluation is incomplete')
        for row, result in zip(measured, completion['returns'], strict=True):
            if row['phase'] != 'evaluation' or row['steps'] != protocol['horizon']:
                raise ValueError('Wrong phase or incomplete evaluation horizon')
            if row['return'] != result:
                raise ValueError('Completed return and episode log disagree')
            for key in ('return', 'mean_drag', 'rms_lift', 'last_half_mean_drag'):
                if not math.isfinite(row[key]):
                    raise ValueError('Nonfinite independent evaluation')
        report['independent_evaluations'] = measured
        report['checkpoint_sha256'] = completion['checkpoint_sha256']
        report['protocol'] = protocol
        report['evidence_sha256'] = {name: digest(evaluation / name) for name in (
            'episodes.jsonl', 'evaluation-steps.jsonl', 'completed.json', 'protocol.json')}
    else:
        report['independent_evaluations'] = []
    report.update(paper_match_accepted=False, paper_gap=None,
                  gap_note='Figure late-time ranges and our last-half means are diagnostic only; '
                           'exact paper test duration/window and seed aggregation are unresolved.')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    result = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  status='interim-specialist-reconstruction-not-paper-acceptance',
                  runs_root=str(args.runs_root.resolve()), plan_sha256=digest(args.plan),
                  cases=[snapshot(args.runs_root, spec) for spec in plan])
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    for row in result['cases']:
        print(row['case'], f"{row['budget_progress_percent']:.3f}%",
              [(x['training_step'], round(x['return'], 4), round(x['last_half_mean_drag'], 5))
               for x in row['independent_evaluations']])


if __name__ == '__main__':
    main()
