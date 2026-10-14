"""Reproducible PR figures from frozen paired evaluation tables; no inference."""
import collections
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

P=Path(__file__).resolve().parent
D=json.loads((P/'results.json').read_text()); E=json.loads((P/'extended-results.json').read_text()); D['comparisons'].update(E['comparisons']); C=D['comparisons']
R=list(csv.DictReader((P/'main-evaluation-by-lead.csv').open()))
for r in R:
 r['level_hpa']=int(r['level_hpa']) if r['level_hpa'] else None
 for k in r:
  if k not in ['variable','level_hpa','unit']:r[k]=float(r[k])
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
COL={'extended':'#16866d','seed0':'#d65a31','gencast':'#2474b5','simple':'#64696e','truth':'#20242a'}
LABEL={'extended':'Extended (580)','seed0':'Original (562)','gencast':'GenCast','simple':'Persistence / dry','truth':'Truth'}
SHORT={'geopotential':'Z','temperature':'T','specific_humidity':'q','u_component_of_wind':'u','v_component_of_wind':'v','vertical_velocity':'omega','wind_speed':'Wind speed','2m_temperature':'T2m','mean_sea_level_pressure':'MSLP','10m_u_component_of_wind':'u10','10m_v_component_of_wind':'v10','10m_wind_speed':'Wind10','total_precipitation_12hr':'Rain12h','total_precipitation_24hr':'Rain24h','geopotential_300_minus_500':'Z300 - Z500'}
def label(k):return SHORT[k[0]]+(f' {k[1]} hPa' if k[1] else '')
def save(fig,name):
 fig.savefig(P/(name+'.png'),dpi=160,bbox_inches='tight');plt.close(fig)
G=collections.defaultdict(list)
for r in R:G[(r['variable'],r['level_hpa'])].append(r)
assert len(G)==99 and len(R)==2969
keys=list(G); leads=sorted(set(r['lead_hours'] for r in R))
# Four exhaustive scorecards; visual saturation never alters exported values.
fig,axes=plt.subplots(1,4,figsize=(19,24),sharey=True,layout='constrained')
for ax,(ref,metric) in zip(axes,[(r,m) for r in ['gencast','simple'] for m in ['crps','rmse']]):
 a=np.full((len(keys),len(leads)),np.nan)
 for i,k in enumerate(keys):
  for row in G[k]:
   den=row[ref+'_'+metric];assert den>0
   a[i,leads.index(row['lead_hours'])]=100*(1-row['extended_'+metric]/den)
 im=ax.imshow(a,cmap='RdBu',vmin=-200,vmax=200,aspect='auto',interpolation='nearest')
 ax.set_title(f'{metric.upper()} vs {LABEL[ref]}',fontweight='bold');ax.set_xticks([0,5,11,17,23,29],['0.5','3','6','9','12','15']);ax.set_xlabel('Forecast day')
 ax.set_yticks(range(len(keys)),[label(k) for k in keys],fontsize=8)
 for i in range(1,len(keys)):
  if keys[i][0]!=keys[i-1][0]:ax.axhline(i-.5,color='black',lw=.6)
fig.suptitle('Complete scorecard | Extended relative error reduction (%)\n8 paired dates · 56 members · all 99 field/level combinations · all 30 leads\nBlue: lower error; red: higher error. Colors saturate at ±200%; blank = unavailable. No significance marks.',fontsize=15)
fig.colorbar(im,ax=axes,orientation='horizontal',fraction=.018,pad=.012,extend='both',label='100 × (1 − Extended error / reference error)');save(fig,'evaluation-scorecard')
# Actual physical-unit errors at every lead for representative fields.
sel=[('2m_temperature',None),('mean_sea_level_pressure',None),('geopotential',500),('10m_wind_speed',None),('total_precipitation_24hr',None)]
fig,axes=plt.subplots(5,2,figsize=(12,16),layout='constrained')
for aa,k in zip(axes,sel):
 rs=sorted(G[k],key=lambda r:r['lead_hours'])
 for ax,m in zip(aa,['crps','rmse']):
  for model in COL:
   if model=='truth':continue
   ax.plot([r['lead_hours']/24 for r in rs],[r[model+'_'+m] for r in rs],color=COL[model],label=LABEL[model],lw=2)
  ax.set_title(f'{label(k)} · {m.upper()}');ax.set_ylabel(rs[0]['unit']);ax.set_xlabel('Forecast day');ax.grid(alpha=.2);ax.set_xlim(.5,15)
axes[0,0].legend();fig.suptitle('Physical-unit forecast errors | all evaluated leads\nEight paired 2022 dates · lower is better · Original (562) and extended (580) Seed 0',fontsize=15);save(fig,'evaluation-errors')
# Pooling: retain mode and scale, average levelwise percentage skill within variable.
a=C['extended_vs_gencast_pooling']['panels']; vs=list(dict.fromkeys(r['variable'] for r in a)); scales=sorted(set(r['mesh_refinement'] for r in a))
fig,axes=plt.subplots(len(scales),2,figsize=(14,13),layout='constrained')
for row,scale in zip(axes,scales):
 for ax,mode in zip(row,['average','maximum']):
  arr=np.full((len(vs),len(leads)),np.nan)
  for i,v in enumerate(vs):
   for j,t in enumerate(leads):
    rs=[r for r in a if r['variable']==v and r['lead_hours']==t and r['mesh_refinement']==scale and r['pooling']==mode]
    if rs:arr[i,j]=np.mean([100*(1-r['candidate_crps']/r['baseline_crps']) for r in rs])
  im=ax.imshow(arr,cmap='RdBu',vmin=-200,vmax=200,aspect='auto');ax.set_yticks(range(len(vs)),[SHORT[v] for v in vs]);ax.set_xticks([0,5,11,17,23,29],['0.5','3','6','9','12','15']);ax.set_xlabel('Forecast day');ax.set_title(f'{mode.title()} pooling · mesh refinement {scale}')
fig.suptitle('Spatial pooling | Extended CRPS reduction vs GenCast (%)\nEqual-weight mean of levelwise percentage reductions within each variable; all dates and leads\nBlue: better; red: worse. Colors saturate at ±200%. Full level/scale values remain in extended-results.json.',fontsize=13);fig.colorbar(im,ax=axes,orientation='horizontal',fraction=.025,extend='both');save(fig,'evaluation-pooling')
# Spectra: plot shape, not only total-power ratios; all configured fields.
oldspec=json.loads((P/'spectra-for-figures.json').read_text()); oldrows={(r['variable'],r['level_hpa'],r['lead_hours']):r for r in oldspec['rows']}
s=json.loads((P/'extended-spectra-for-figures.json').read_text());sk=list(dict.fromkeys((r['variable'],r['level_hpa']) for r in s['rows']));assert len(sk)==10
fig,axes=plt.subplots(len(sk),2,figsize=(13,24),layout='constrained')
for aa,k in zip(axes,sk):
 for ax,t in zip(aa,[24,360]):
  r=next(r for r in s['rows'] if (r['variable'],r['level_hpa'])==k and r['lead_hours']==t)
  for model,field in [('extended','candidate_member_mean_power'),('gencast','baseline_member_mean_power'),('truth','truth_power')]:
   ax.loglog(s['total_wavenumber'][1:],r[field][1:],color=COL[model],label=LABEL[model],lw=1.2)
  old=oldrows[(k[0],k[1],t)]
  assert old['cases']==r['cases']
  ax.loglog(s['total_wavenumber'][1:],old['candidate_member_mean_power'][1:],color=COL['seed0'],label=LABEL['seed0'],lw=1.1,linestyle='--')
  ax.set_title(f'{label(k)} · day {t//24} · {r["cases"]} dates');ax.set_xlabel('Total spherical wavenumber (larger = finer scale)');ax.set_ylabel('Power (native unit²)');ax.grid(alpha=.15)
axes[0,0].legend();fig.suptitle('Spatial power spectra | every configured field at days 1 and 15\nMember-mean spectra vs truth · nonzero wavenumbers · power is structure, not prediction accuracy',fontsize=14);save(fig,'evaluation-spectra')
# Extremes, categorical precipitation and tracks.
fig,axes=plt.subplots(2,3,figsize=(17,9),layout='constrained')
for ax,(acc,th) in zip(axes[0,:2],[(12,.1),(24,.1)]):
 aa=[r for r in C['extended_vs_gencast_core']['seeps_panels'] if r['accumulation_hours']==acc and r['dry_threshold_mm']==th];bb=[r for r in C['extended_vs_persistence_core']['seeps_panels'] if r['accumulation_hours']==acc and r['dry_threshold_mm']==th]
 for model,rs,field in [('extended',aa,'candidate_seeps'),('gencast',aa,'baseline_seeps'),('simple',bb,'baseline_seeps')]:ax.plot([r['lead_hours']/24 for r in rs],[r[field] for r in rs],label=LABEL[model],color=COL[model])
 ax.set_title(f'{acc} h precipitation SEEPS · dry threshold {th} mm');ax.set_xlabel('Forecast day');ax.set_ylabel('SEEPS (lower better)');ax.legend()
cy=C['extended_vs_gencast_cyclones']['position'];ax=axes[0,2]
for model,prefix in [('extended','candidate'),('gencast','baseline')]:ax.plot([r['lead_hours']/24 for r in cy],[r[prefix+'_mean_error_km'] for r in cy],color=COL[model],label=LABEL[model])
ax.set_title('Cyclone track position error');ax.set_xlabel('Forecast day');ax.set_ylabel('km (lower better)');ax.legend();ax.text(.02,.68,'Matched storms: 8 → 1\nNo paired reports after day 9',transform=ax.transAxes,fontsize=9)
for ax,v in zip(axes[1,:2],['2m_temperature','10m_wind_speed']):
 rr=[r for r in C['extended_vs_gencast_core']['rev_panels'] if r['variable']==v];q=max(r['quantile'] for r in rr)
 for t,style in [(24,'-'),(120,'--'),(168,':')]:
  r=next(r for r in rr if r['quantile']==q and r['lead_hours']==t and r['direction']=='high')
  for model,prefix in [('extended','candidate'),('gencast','baseline')]:ax.semilogx(r['cost_loss_ratios'],r[prefix+'_potential_rev'],style,color=COL[model],label=f'{LABEL[model]} d{t//24}')
 ax.set_title(f'{SHORT[v]} extreme high · quantile {q}');ax.set_xlabel('Cost / loss ratio');ax.set_ylabel('REV (higher better)');ax.legend(fontsize=7,ncol=2)
ax=axes[1,2]
for t,style in [(24,'-'),(120,'--')]:
 r=next(r for r in C['extended_vs_gencast_cyclones']['rev'] if r['lead_hours']==t)
 for model,prefix in [('extended','candidate'),('gencast','baseline')]:ax.semilogx(r['cost_loss_ratios'],r[prefix+'_potential_rev'],style,color=COL[model],label=f'{LABEL[model]} d{t//24}')
ax.set_title('Cyclone probability REV');ax.set_xlabel('Cost / loss ratio');ax.set_ylabel('REV (higher better)');ax.legend(fontsize=8)
for ax in axes.flat:ax.grid(alpha=.15)
fig.suptitle('Precipitation, extremes and tropical cyclones\nPaired 2022 subset; selected REV thresholds/leads shown here, complete settings retained in tables',fontsize=15);save(fig,'evaluation-events')


fig,axes=plt.subplots(1,2,figsize=(12,24),sharey=True,layout='constrained')
for ax,metric in zip(axes,['crps','rmse']):
 a=np.full((len(keys),len(leads)),np.nan)
 for i,k in enumerate(keys):
  for row in G[k]:a[i,leads.index(row['lead_hours'])]=100*(1-row['extended_'+metric]/row['seed0_'+metric])
 im=ax.imshow(a,cmap='RdBu',vmin=-10,vmax=10,aspect='auto',interpolation='nearest')
 ax.set_title(metric.upper());ax.set_xticks([0,5,11,17,23,29],['0.5','3','6','9','12','15']);ax.set_xlabel('Forecast day');ax.set_yticks(range(len(keys)),[label(k) for k in keys],fontsize=8)
 for i in range(1,len(keys)):
  if keys[i][0]!=keys[i-1][0]:ax.axhline(i-.5,color='black',lw=.6)
fig.suptitle('Additional AR training | Extended (580) vs original (562)\nAll 99 field/level combinations and all 30 leads; same eight 2022 dates\nBlue: lower error; red: higher error. Colors saturate at ±10%. No significance marks.',fontsize=14)
fig.colorbar(im,ax=axes,orientation='horizontal',fraction=.018,extend='both',label='Error reduction (%)');save(fig,'evaluation-ar-improvement')

print('Generated six evaluation figures')
