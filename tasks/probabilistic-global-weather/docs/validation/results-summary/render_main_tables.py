"""Render complete field/level comparison tables from published aggregate panels."""
import collections
import csv
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
D = json.loads((ROOT / 'results.json').read_text())
E = json.loads((ROOT / 'extended-results.json').read_text())
D['comparisons'].update(E['comparisons'])
D['spectral_power_summary'] = E['extended_spectral_power_summary']
C = D['comparisons']
BASE = 'https://github.com/T0-RSI/ai4sci-tasks/blob/main/probabilistic-global-weather/docs/validation/results-summary/'
UNITS = {
 'geopotential': ('m²/s²', 1), 'geopotential_300_minus_500': ('m²/s²', 1),
 'temperature': ('K', 1), '2m_temperature': ('°C error', 1),
 'specific_humidity': ('g/kg', 1000), 'vertical_velocity': ('Pa/s', 1),
 'mean_sea_level_pressure': ('hPa', .01),
 'total_precipitation_12hr': ('mm / 12 h', 1000),
 'total_precipitation_24hr': ('mm / 24 h', 1000),
}
def unit(v): return UNITS.get(v, ('m/s', 1))
def fmt(v):
 if v is None or not math.isfinite(v): return '—'
 return f'{v:.3g}'
def mean(xs):
 assert xs and all(math.isfinite(x) for x in xs)
 return sum(xs)/len(xs)
def key(r): return (r['variable'], r['level_hpa'], r['lead_hours'])
def table(headers, rows):
 return '| ' + ' | '.join(headers) + ' |\n|' + '|'.join(['---']*len(headers)) + '|\n' + ''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)+'\n'

variance=json.loads((ROOT/'ensemble-variance.json').read_text())
spread={name:{key(r):r['mean_ensemble_variance'] for r in rs} for name,rs in variance.items()}
original={key(r):r for r in C['seed0_vs_gencast_core']['panels']}
core=C['extended_vs_gencast_core']['panels']; simple={key(r):r for r in C['extended_vs_persistence_core']['panels']}
assert len(core)==len(simple)==2969
rows=[]; groups=collections.defaultdict(list)
for r in core:
 s=simple[key(r)]; assert r['cases']==s['cases']==8
 for m in ['crps','rmse']: assert math.isclose(r['candidate_'+m],s['candidate_'+m],rel_tol=1e-12)
 u,f=unit(r['variable']); out=dict(variable=r['variable'],level_hpa=r['level_hpa'],lead_hours=r['lead_hours'],cases=8,unit=u)
 for model,source,prefix in [('extended',r,'candidate'),('seed0',original[key(r)],'candidate'),('gencast',r,'baseline'),('simple',s,'baseline')]:
  for metric in ['crps','rmse']: out[model+'_'+metric]=source[prefix+'_'+metric]*f
 out['extended_spread']=math.sqrt(spread['extended'][key(r)])*f
 out['seed0_spread']=math.sqrt(spread['wn2-seed-0'][key(r)])*f
 out['gencast_spread']=math.sqrt(spread['gencast'][key(r)])*f
 rows.append(out);groups[(r['variable'],r['level_hpa'])].append(out)
with (ROOT/'main-evaluation-by-lead.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
main=[]
for (v,l),rs in groups.items():
 vals=[]
 for metric in ['crps','rmse']:
  for model in ['simple','seed0','extended','gencast']:
   a=[r[model+'_'+metric] for r in rs]; vals.append(fmt(mean(a) if metric=='crps' else math.sqrt(mean([x*x for x in a]))))
 vals.extend(fmt(math.sqrt(mean([r[model+'_spread']**2 for r in rs]))) for model in ['seed0','extended','gencast'])
 main.append([v,str(l) if l is not None else 'surface / derived',unit(v)[0],len(rs),*vals])
assert len(main)==99
text='''### Main evaluation table — all fields and pressure levels

Extended 580-update Seed 0 (main recipe), original 562-update Seed 0, GenCast and persistence / dry, on the **same eight 2022 dates, 56 members and 15-day horizon**. This table includes **all 82 native field/level combinations**, plus 17 precipitation/derived combinations. Lower CRPS and RMSE are better. The simple reference holds initial fields fixed and predicts zero rain.

Each row summarizes **all evaluated leads** (12–360 h every 12 h; 24 h accumulated rain starts at 24 h). CRPS averages equally over dates and leads; RMSE is the square root of the equally averaged MSE, not the mean of lead RMSEs. Pressure levels remain separate. Unit conversions apply equally to all models. Spread is the square root of mean unbiased ensemble variance (ddof=1), in the same units; deterministic simple-reference spread is zero. Spread is a dispersion diagnostic, not an error score. These are physical-unit errors, not the normalized task reward. The temperature-by-lead table below complements this complete field table.

'''+table(['Variable','hPa / type','Error unit','Leads','Simple CRPS','Original CRPS','Extended CRPS','GenCast CRPS','Simple RMSE','Original RMSE','Extended RMSE','GenCast RMSE','Original spread','Extended spread','GenCast spread'],main)
text+=f'[Complete table at every evaluated lead: 2,969 rows, all four models](<{BASE}main-evaluation-by-lead.csv>). No leads or field/level combinations are omitted from this CSV.\n\n'
text+='''### Spatial pooling — all evaluated variable families

All eight dates, both average and maximum pooling, every configured mesh scale, level and lead. Counts below show panels with lower model error than persistence / dry; correlated panels are descriptive, not independent trials. Full absolute scores remain in the aggregate results file.

'''
a=C['extended_vs_persistence_pooling']['panels']; b=C['gencast_vs_persistence_pooling']['panels'];pool=[]
for v in dict.fromkeys(r['variable'] for r in a):
 aa=[r for r in a if r['variable']==v];bb=[r for r in b if r['variable']==v];assert len(aa)==len(bb)
 pool.append([v,len(aa),*[sum(r['candidate_'+m]<r['baseline_'+m] for r in rr) for rr in [aa,bb] for m in ['crps','rmse']]])
text+=table(['Variable','Panels','Extended lower CRPS','Extended lower RMSE','GenCast lower CRPS','GenCast lower RMSE'],pool)
text+='### Precipitation SEEPS — all accumulation / threshold settings\n\nEqual-weight mean of the eight-date score across all available leads; dimensionless, lower is better.\n\n'
ss=C['extended_vs_persistence_core']['seeps_panels']; sg=C['extended_vs_gencast_core']['seeps_panels'];out=[]
for accum,threshold in sorted(set((r['accumulation_hours'],r['dry_threshold_mm']) for r in sg)):
 aa=[r for r in sg if (r['accumulation_hours'],r['dry_threshold_mm'])==(accum,threshold)];bb=[r for r in ss if (r['accumulation_hours'],r['dry_threshold_mm'])==(accum,threshold)]
 out.append([accum,threshold,len(aa),fmt(mean([r['baseline_seeps'] for r in bb])),fmt(mean([r['candidate_seeps'] for r in aa])),fmt(mean([r['baseline_seeps'] for r in aa]))])
text+=table(['Accumulation (h)','Dry threshold (mm)','Leads','Dry SEEPS','Extended SEEPS','GenCast SEEPS'],out)
text+='''### Extreme-event REV — all event definitions

Equal-weight mean across all evaluated leads and the configured cost/loss grid, after each paired eight-date aggregation. Larger is better; climatology has REV zero. This grid-average is a descriptive summary, not a newly optimized operating threshold; full curves remain in the aggregate results file.

'''
rr=C['extended_vs_gencast_core']['rev_panels']; ss=C['extended_vs_persistence_core']['rev_panels'];out=[]
for v,q,direction in sorted(set((r['variable'],r['quantile'],r['direction']) for r in rr)):
 aa=[r for r in rr if (r['variable'],r['quantile'],r['direction'])==(v,q,direction)];bb=[r for r in ss if (r['variable'],r['quantile'],r['direction'])==(v,q,direction)]
 def avg(rs,k): return mean([mean(r[k]) for r in rs])
 out.append([v,q,direction,len(aa),fmt(avg(bb,'baseline_potential_rev')),fmt(avg(aa,'candidate_potential_rev')),fmt(avg(aa,'baseline_potential_rev'))])
text+=table(['Variable','Quantile','Tail','Leads','Simple REV','Extended REV','GenCast REV'],out)
text+='''### Spatial spectra — every evaluated field / level

Ratio of summed nonzero-wavenumber member-mean power to truth power, averaging power over available dates/leads before division. Truth = 1. This measures total spatial variability, not prediction accuracy or agreement at each wavelength. Persistence spectra are not defined in this evaluation. Full wavelength-dependent reports are separate from this scalar summary.

'''
spec=collections.defaultdict(list)
for r in D['spectral_power_summary']:spec[(r['variable'],r['level_hpa'])].append(r)
out=[]
for (v,l),rs in spec.items():
 t=sum(r['truth_power_nonzero_sum'] for r in rs)
 out.append([v,l if l is not None else 'surface',len(rs),fmt(sum(r['candidate_member_mean_power_nonzero_sum'] for r in rs)/t),fmt(sum(r['baseline_member_mean_power_nonzero_sum'] for r in rs)/t),'1.000'])
text+=table(['Variable','hPa / type','Leads','Extended / truth','GenCast / truth','Truth'],out)
text+='### Cyclones — all evaluated leads\n\nPosition errors use storm initializations matched in both forecasts and truth; counts vary with lead. Paired cyclone reports are available only through day 9; days 9.5–15 have no paired report and are unavailable, not zero error. No simple track reference is defined. Position error is in km, lower is better. REV is averaged across its configured cost/loss grid, larger is better.\n\n'
cy=C['extended_vs_gencast_cyclones']; rev={r['lead_hours']:r for r in cy['rev']};out=[]
for r in cy['position']:
 v=rev[r['lead_hours']]
 out.append([r['lead_hours']/24,r['paired_storm_initializations'],fmt(r['candidate_mean_error_km']),fmt(r['baseline_mean_error_km']),fmt(mean(v['candidate_potential_rev'])),fmt(mean(v['baseline_potential_rev']))])
text+=table(['Lead (d)','Matched storms','Extended km','GenCast km','Extended REV','GenCast REV'],out)
(ROOT/'main-evaluation.md').write_text(text)
print('Rows:',len(main),'Native CSV:',len(rows),'Markdown chars:',len(text))
