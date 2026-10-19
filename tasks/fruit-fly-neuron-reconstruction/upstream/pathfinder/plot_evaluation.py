"""Static paired local evaluation; never draw a cross-dataset SOTA gap."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from compare_report import local_change
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import external,sha256,write_json


def render(before,after,output,diagnostics=None):
    change=local_change(before,after)
    if diagnostics is not None:
        for key,expected in [('before',before['inputs']['segmentation']),('after',after['inputs']['segmentation']),
                             ('skeletons',before['inputs']['skeletons']),('metadata',before['inputs']['metadata'])]:
            if diagnostics['inputs'][key]!=expected:raise ValueError('Diagnostic input mismatch: '+key)
    colors=['#537ca5','#dd9741']
    fig,axes=plt.subplots(1,3,figsize=(10,3.7),constrained_layout=True)
    ax=axes[0];values=[before['nerl'],after['nerl']]
    ax.bar([0,1],values,color=colors,width=.6)
    upper=max(.001,max(values)*1.35) if max(values)<.1 else 1.12
    digits=6 if max(values)<.01 else 3
    for x,v in enumerate(values):ax.text(x,v+upper*.025,f'{v:.{digits}f}',ha='center',fontsize=14)
    ax.set_ylim(0,upper);ax.set_xticks([0,1],['FFN','Final'])
    if upper==1.12:ax.set_yticks([0,.5,1])
    ax.set_title('nERL (0–1) ↑');ax.set_ylabel('Same frozen local paths')
    ax=axes[1]
    for j,key in enumerate(['split_errors','merge_errors']):
        for i,metrics in enumerate([before,after]):
            value=metrics[key];x=j+(i-.5)*.32
            ax.bar(x,value,width=.3,color=colors[i]);ax.text(x,value,f'{value:,}',ha='center',va='bottom',fontsize=10)
    ax.set_xticks([0,1],['Split','Merge']);ax.set_title('Observed errors ↓');ax.margins(y=.3)
    ax=axes[2]
    if diagnostics is None:
        values=[100*before['omitted_node_fraction'],100*after['omitted_node_fraction']]
        ax.bar([0,1],values,color=colors,width=.6)
        for x,v in enumerate(values):ax.text(x,v,f'{v:.1f}%',ha='center',va='bottom')
        ax.set_xticks([0,1],['FFN','Final']);ax.set_ylim(0,110);ax.set_title('Omitted nodes ↓')
    else:
        rows=diagnostics['per_skeleton'];ys=np.arange(len(rows))
        for i,prefix in enumerate(['before','after']):
            values=[100*r[prefix+'_adjacent_connected_nm']/r['path_nm'] if r['path_nm']>0 else float('nan') for r in rows]
            ax.scatter(values,ys+(i-.5)*.15,color=colors[i],s=25)
        ax.set_yticks(ys,[str(r['skeleton_id']) for r in rows]);ax.set_xlim(0,105)
        ax.set_title('Adjacent continuity (%)');ax.set_ylabel('Skeleton');ax.set_xlabel('Can rise with wrong merges')
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    status='Identical prediction' if change['identical_prediction_bytes'] else f"Δ nERL {after['nerl']-before['nerl']:+.6f}"
    fig.suptitle(f'Local validation · {status}',fontsize=14)
    axes[1].legend([plt.Rectangle((0,0),1,1,color=c) for c in colors],['FFN','Final'],loc='upper right',frameon=False,fontsize=9)
    fig.text(.5,-.025,'Sparse GT: unlabelled false merges are invisible · No paper/SOTA comparison',ha='center',fontsize=9)
    output=external(output);output.parent.mkdir(parents=True,exist_ok=True)
    temp=output.with_name(output.name+'.tmp.png');fig.savefig(temp,dpi=160,bbox_inches='tight');plt.close(fig);temp.replace(output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('before','after','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--diagnostics',type=Path)
    a=p.parse_args();read=lambda path:json.loads(path.read_text())
    render(read(a.before),read(a.after),a.output,read(a.diagnostics) if a.diagnostics else None)
    inputs={k:sha256(getattr(a,k)) for k in ('before','after','diagnostics') if getattr(a,k) is not None}
    write_json(external(a.output.with_suffix('.json')),dict(inputs=inputs,image_sha256=sha256(a.output),paper_comparable=False))

if __name__=='__main__':main()
