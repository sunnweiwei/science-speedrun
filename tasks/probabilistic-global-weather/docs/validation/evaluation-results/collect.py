"""Read archived evaluation receipts and export compact, explicitly partial results."""
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-root', required=True, type=Path)
parser.add_argument('--paper-root', required=True, type=Path)
parser.add_argument('--replay-root', required=True, type=Path)
args = parser.parse_args()
out = Path(__file__).resolve().parent
result = {'checked_utc': datetime.now(timezone.utc).isoformat(),
          'status': 'partial_results_not_final_acceptance',
          'original_protocol': {'requested_cases': 32, 'members': 8, 'horizon_days': 10},
          'paper_protocol': {'requested_cases': 8, 'members': 56, 'horizon_days': 15},
          'aggregation': 'Equal weight per completed date; CRPS is mean of area-weighted case CRPS; RMSE is sqrt(mean of area-weighted case MSE).',
          'original_results': [], 'paper_progress': {}, 'final_replay': []}
for seed in (0, 1):
    reports, sources = [], []
    for path in sorted((args.run_root / f'pipeline-{seed}/curriculum/evaluation').glob('case-*/weather-scores.json')):
        raw = path.read_bytes()
        d = json.loads(raw)
        if d.get('status') != 'passed' or not d['initialization'].startswith('2022-'):
            continue
        sources.append({'report': str(path.relative_to(args.run_root)),
                        'sha256': hashlib.sha256(raw).hexdigest(),
                        'initialization': d['initialization'],
                        'protocol_sha256': d['protocol_sha256'],
                        'training_provenance_verified': d.get('training_provenance_verified')})
        reports.append(d)
    rows = []
    for lead in (24, 72, 120, 240):
        panels = []
        for d in reports:
            matches = [r for r in d['per_case'] if r['variable'] == '2m_temperature'
                       and r['lead_hours'] == lead and r['pool_cells'] == 1]
            assert len(matches) == 1
            panels.extend(matches)
        if panels:
            crps = sum(r['crps'] for r in panels) / len(panels)
            rmse = math.sqrt(sum(r['mse'] for r in panels) / len(panels))
            assert math.isfinite(crps) and math.isfinite(rmse)
            rows.append(dict(lead_hours=lead, cases=len(panels), crps_kelvin=crps, rmse_kelvin=rmse))
    result['original_results'].append(dict(seed=seed, variable='2m_temperature',
        completed_dates=len(reports), sources=sources, rows=rows))
    p = args.replay_root / f'seed-{seed}/exact-replay.json'
    if p.exists():
        raw = p.read_bytes()
        d = json.loads(raw)
        result['final_replay'].append(dict(seed=seed, status=d.get('status'),
            report=f'seed-{seed}/exact-replay.json', sha256=hashlib.sha256(raw).hexdigest(),
            scope='One development case, 8 members, 40 six-hour leads; exact decoded cold/warm comparison'))
for model in ('wn2-seed-0', 'wn2-seed-1', 'gencast'):
    dates = {}
    for path in (args.paper_root / 'predictions' / model).glob('*.complete.json'):
        d = json.loads(path.read_text())
        if d.get('status') == 'passed':
            dates[d['initialization']] = dates.get(d['initialization'], 0) + 1
    endpoints = {}
    for endpoint in ('scores', 'pooling-scores', 'spectra', 'cyclone-scores'):
        sources = []
        for path in sorted((args.paper_root / endpoint / model).glob('case-*.json')):
            raw = path.read_bytes()
            d = json.loads(raw)
            if d.get('status') == 'passed':
                item = dict(report=str(path.relative_to(args.paper_root)), sha256=hashlib.sha256(raw).hexdigest())
                if endpoint == 'cyclone-scores':
                    item['position_rows'] = len(d.get('position_rows', []))
                sources.append(item)
        endpoints[endpoint] = sources
    result['paper_progress'][model] = dict(completed_trajectories=sum(dates.values()),
        requested_trajectories=448, members_by_date=dates, endpoints=endpoints)
(out / 'snapshot.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['original_results','paper_progress']},indent=2))
for item in result['original_results']:
    print(item['seed'],item['completed_dates'],item['rows'])
