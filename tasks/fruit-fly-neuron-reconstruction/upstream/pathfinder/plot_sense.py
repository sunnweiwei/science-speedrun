"""Generate a static SENSE progress chart; no dashboard."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import external


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();rows=[]
    lines=(a.run/'progress.jsonl').read_text().splitlines()
    for index,line in enumerate(lines):
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines)-1:raise
    if not rows:raise ValueError('No completed validation yet')
    fig,axes=plt.subplots(1,2,figsize=(8,3),constrained_layout=True)
    x=[r['step'] for r in rows]
    axes[0].plot(x,[r['train_loss'] for r in rows],label='Train')
    axes[0].plot(x,[r['validation_bce'] for r in rows],label='Validation');axes[0].set_ylabel('BCE');axes[0].legend(frameon=False)
    axes[1].plot(x,[r['validation_dice'] for r in rows],color='#008c95');axes[1].set_ylabel('Validation Dice');axes[1].set_ylim(0,1)
    for ax in axes:
        ax.set_xlabel('Step');ax.spines[['top','right']].set_visible(False)
    out=external(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=160);plt.close(fig)

if __name__=='__main__':main()
