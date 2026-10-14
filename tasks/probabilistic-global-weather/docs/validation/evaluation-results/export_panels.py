"""Export all native panels on the exact completed-date support in snapshot.json."""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--run-root', required=True, type=Path)
args = ap.parse_args()
out = Path(__file__).resolve().parent
snapshot = json.loads((out / 'snapshot.json').read_text())
units = dict(geopotential='m2 s-2', temperature='K', specific_humidity='kg kg-1',
             u_component_of_wind='m s-1', v_component_of_wind='m s-1',
             vertical_velocity='Pa s-1', mean_sea_level_pressure='Pa',
             total_precipitation_6hr='m')
units.update({'2m_temperature': 'K', '10m_u_component_of_wind': 'm s-1',
              '10m_v_component_of_wind': 'm s-1'})
rows, sources = [], []
for seed_result in snapshot['original_results']:
    seed = seed_result['seed']
    for endpoint in ('weather', 'precipitation'):
        groups = defaultdict(list)
        expected_keys, identity = None, None
        for source in seed_result['sources']:
            path = args.run_root / source['report']
            if endpoint == 'precipitation':
                path = path.with_name('precipitation-scores.json')
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if endpoint == 'weather':
                assert digest == source['sha256'], 'Snapshot source changed'
            report = json.loads(raw)
            assert report['status'] == 'passed'
            assert report['initialization'] == source['initialization']
            current = tuple(report[k] for k in ('protocol_sha256', 'evaluator_sha256',
                'shard_scorer_sha256', 'weatherbenchx_revision', 'estimator'))
            assert identity is None or current == identity, 'Inconsistent score provenance'
            identity = current
            panels = [r for r in report['per_case'] if r['pool_cells'] == 1]
            keys = {(r['variable'], r['level_hpa'], r['lead_hours']) for r in panels}
            assert len(keys) == len(panels), 'Duplicate panels'
            assert expected_keys is None or keys == expected_keys, 'Unequal panel coverage'
            expected_keys = keys
            for r in panels:
                assert r['case'] == source['initialization']
                assert all(math.isfinite(r[k]) and r[k] >= 0 for k in ('crps', 'mse', 'ensemble_variance'))
                groups[r['variable'], r['level_hpa'], r['lead_hours']].append(r)
            sources.append(dict(seed=seed, endpoint=endpoint, date=source['initialization'],
                report=str(path.relative_to(args.run_root)), sha256=digest,
                protocol_sha256=report['protocol_sha256']))
        for (variable, level, lead), panels in sorted(groups.items(), key=lambda x:(x[0][0],x[0][1] or 0,x[0][2])):
            n = len(panels)
            assert n == seed_result['completed_dates']
            mean = lambda key: math.fsum(r[key] for r in panels) / n
            rmse, spread = math.sqrt(mean('mse')), math.sqrt(mean('ensemble_variance'))
            rows.append(dict(seed=seed, endpoint=endpoint, variable=variable, level_hpa=level,
                lead_hours=lead, cases=n, units=units[variable], crps=mean('crps'), rmse=rmse,
                spread_ddof1=spread, spread_skill=spread/rmse if rmse else None,
                minimum_case_valid_area_fraction=min(r['valid_area_fraction'] for r in panels)))
fields = ['seed','endpoint','variable','level_hpa','lead_hours','cases','units','crps','rmse',
          'spread_ddof1','spread_skill','minimum_case_valid_area_fraction']
with (out / 'native-panels.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
(out / 'native-panel-sources.json').write_text(json.dumps(dict(
    snapshot_sha256=hashlib.sha256((out/'snapshot.json').read_bytes()).hexdigest(),
    csv_sha256=hashlib.sha256((out/'native-panels.csv').read_bytes()).hexdigest(),
    scope='All native-grid panels from exactly the partial dates in snapshot.json; no full-acceptance or paired-seed claim',
    rows=len(rows), sources=sources), indent=2) + '\n')
# Check the PR examples using the independently aggregated panel export.
for seed_result in snapshot['original_results']:
    for example in seed_result['rows']:
        matches = [r for r in rows if r['seed']==seed_result['seed']
                   and r['variable']=='2m_temperature' and r['lead_hours']==example['lead_hours']]
        assert len(matches)==1
        assert math.isclose(matches[0]['crps'],example['crps_kelvin'],rel_tol=1e-12)
        assert math.isclose(matches[0]['rmse'],example['rmse_kelvin'],rel_tol=1e-12)
print(f'Exported {len(rows)} native panels from {len(sources)} score reports; PR examples agree.')
