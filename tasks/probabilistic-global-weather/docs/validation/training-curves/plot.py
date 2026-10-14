"""Export verified stage logs with --run-root PATH; otherwise replot checked-in CSV."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
FIELDS = ['seed', 'stage_index', 'stage', 'stage_update', 'global_updates',
          'loss', 'gradient_norm', 'learning_rate', 'seconds']
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-root', type=Path)
args = parser.parse_args()
if args.run_root:
    rows, sources = [], []
    for seed in (0, 1):
        reports = sorted((args.run_root / f'pipeline-{seed}/curriculum').glob('stage-*/report.json'))
        assert len(reports) == 11
        expected = json.loads((OUT.parent / f'wn2-seed-{seed}-final-training.json').read_text())
        for index, path in enumerate(reports):
            raw = path.read_bytes()
            report = json.loads(raw)
            digest = hashlib.sha256(raw).hexdigest()
            assert digest == expected['stages'][index]['report_sha256']
            assert report['status'] == 'passed' and not report['profile']
            assert len(report['updates']) == expected['stages'][index]['updates']
            sources.append(dict(seed=seed, stage=report['stage']['name'],
                                report=str(path.relative_to(args.run_root)), sha256=digest))
            for update in report['updates']:
                row = {key: update[key] for key in FIELDS[3:]}
                row.update(seed=seed, stage_index=index, stage=report['stage']['name'])
                rows.append(row)
    with (OUT / 'updates.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (OUT / 'sources.json').write_text(json.dumps(dict(
        scope='Two final non-profile runs; hashes match final-training evidence.',
        reports=sources), indent=2) + '\n')
with (OUT / 'updates.csv').open() as stream:
    rows = list(csv.DictReader(stream))
for row in rows:
    for key in FIELDS:
        if key != 'stage':
            row[key] = float(row[key])
assert len(rows) == 1124
for seed in (0, 1):
    subset = [r for r in rows if r['seed'] == seed]
    assert [r['global_updates'] for r in subset] == list(range(1, 563))
    assert all(np.isfinite(r[k]) for r in subset for k in FIELDS if k != 'stage')

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.facecolor': 'white'})
colors = ['#2563eb', '#d97706']
fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
groups = [(0, 0, 'ERA5 1° / 12 h · 410 updates'),
          (1, 1, 'ERA5 1° / 6 h · 102 updates'),
          (2, 2, 'ERA5 0.25° / 6 h · 32 updates'),
          (3, 10, 'HRES 0.25° / AR1–AR8 · 18 updates')]
for ax, (lo, hi, title) in zip(axes.flat, groups):
    for seed, color in enumerate(colors):
        selected = [r for r in rows if r['seed'] == seed and lo <= r['stage_index'] <= hi]
        for stage in range(lo, hi + 1):
            part = [r for r in selected if r['stage_index'] == stage]
            x = [r['global_updates'] for r in part]
            y = [r['loss'] for r in part]
            ax.plot(x, y, color=color, alpha=.38 if lo == hi else .85,
                    marker='o' if lo != hi else None, markersize=4,
                    label=f'Seed {seed} raw' if stage == lo else None)
            if len(y) >= 10:
                window = min(15, max(3, len(y)//10))
                ax.plot(x[window-1:], np.convolve(y, np.ones(window)/window, mode='valid'),
                        color=color, lw=2, label=f'Seed {seed}: {window}-update mean')
    ax.set(title=title, xlabel='Global optimizer update', ylabel='Logged training loss')
    ax.grid(alpha=.18)
    ax.legend(fontsize=8)
    if lo != hi:
        for stage in range(3, 11):
            part = [r for r in rows if r['seed'] == 0 and r['stage_index'] == stage]
            x = np.mean([r['global_updates'] for r in part])
            ax.text(x, .98, f'AR{stage-2}', transform=ax.get_xaxis_transform(),
                    ha='center', va='top', fontsize=7, rotation=90)
fig.suptitle('WN2 training loss · two seeds, 562 updates each\nStage-specific curves; rollout changes alter the loss distribution', fontsize=15)
fig.savefig(OUT / 'loss.png', dpi=180)
plt.close(fig)

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
for seed, color in enumerate(colors):
    part = [r for r in rows if r['seed'] == seed]
    for ax, key, label in zip(axes, ['gradient_norm', 'learning_rate', 'seconds'],
                            ['Logged gradient norm', 'Learning rate', 'Update duration (seconds)']):
        ax.plot([r['global_updates'] for r in part], [r[key] for r in part],
                color=color, alpha=.8, lw=1, label=f'Seed {seed}')
        ax.set_ylabel(label)
        ax.set_yscale('log')
        ax.grid(alpha=.18)
for ax in axes:
    for boundary in [410.5, 512.5, 544.5, 552.5, 556.5, 557.5, 558.5, 559.5, 560.5, 561.5]:
        ax.axvline(boundary, color='gray', alpha=.25, lw=.7)
    ax.legend(loc='upper left')
axes[-1].set_xlabel('Global optimizer update (vertical lines: curriculum transitions)')
fig.suptitle('WN2 optimizer diagnostics · all 1,124 logged updates\nLogarithmic y axes; durations include checkpoint I/O and first-update overhead', fontsize=14)
fig.savefig(OUT / 'diagnostics.png', dpi=180)
plt.close(fig)
print('Verified 22 report hashes; 562 contiguous finite updates per seed.' if args.run_root else 'Verified CSV: 562 contiguous finite updates per seed.')
