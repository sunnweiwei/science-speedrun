"""Render the complete paired development report without selecting winning metrics."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).resolve().parent
r=json.loads((root/'scores.json').read_text())
assert r['status']=='complete' and len(r['completed_dates'])==4
fields=[('2m_temperature','2m temperature'),('mean_sea_level_pressure','Sea-level pressure'),
        ('geopotential','500 hPa geopotential'),('10m_wind_speed','10m wind speed'),
        ('total_precipitation_6hr','6-hour precipitation')]
index={(s['variable'],s['lead_hours']):s for s in r['summary']}
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(13,5.4),sharey=True,constrained_layout=True)
y=np.arange(len(fields))
for ax,metric in zip(axes,['crps','rmse']):
    for lead,offset,color in [(6,-.18,'#2563eb'),(24,.18,'#d97706')]:
        values=[index[field,lead][metric+'_improvement_percent'] for field,_ in fields]
        bars=ax.barh(y+offset,values,height=.32,color=color,label=f'{lead}-hour lead')
        for yy,value in zip(y+offset,values):
            ax.text(value+(.08 if value>=0 else -.08),yy,f'{value:+.2f}%',va='center',ha='left' if value>=0 else 'right',fontsize=9)
    ax.axvline(0,color='#374151',lw=1)
    ax.set_title(metric.upper(),fontweight='bold')
    ax.set_xlim(-1.7,7.2)
    ax.set_xlabel('Error reduction (%) · positive is better')
    ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
axes[0].set_yticks(y,[label for _,label in fields]);axes[0].invert_yaxis()
axes[1].legend(loc='lower right',frameon=False)
fig.suptitle('HRES fine-tuning: paired development evaluation\nUpdates 544 → 562 · 4 fixed 2021 dates × 8 members · identical HRES normalization',fontsize=14)
fig.savefig(root/'improvement.png',dpi=180)
fig.savefig(root/'improvement.pdf')
with (root/'summary.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(r['summary'][0]),lineterminator='\n')
    writer.writeheader();writer.writerows(r['summary'])
print('Rendered all 10 CRPS/RMSE panels; full-date support verified.')
