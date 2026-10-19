"""Static physical sections of the eligible full-context center domain."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import external,sha256,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('audit','training-manifest','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();report=json.loads(a.audit.read_text());training=json.loads(a.training_manifest.read_text())
    if report['schema']!='pathfinder-geometric-center-coverage/v1':raise ValueError('Geometry audit required')
    if report['inputs'].get(str(a.training_manifest.resolve()))!=sha256(a.training_manifest):
        raise ValueError('Training geometry differs from audited manifest')
    origin=np.array(report['roi_origin_nm'])/1000;extent=np.array(report['roi_extent_nm'])/1000
    centers=np.array(report['eligible_center_cell_bounds_nm'])/1000
    if (np.array(report['eligible_center_grid_shape'])<=0).any():raise ValueError('No eligible center domain to plot')
    blocks=[r for r in training['splits']['train'] if r['name'] in report['training_raw_blocks_containing_all_eligible_centers']]
    fig,axes=plt.subplots(1,2,figsize=(8,5.2))
    for ax,vertical,title in zip(axes,[1,0],['XY','XZ']):
        ax.add_patch(Rectangle((origin[2],origin[vertical]),extent[2],extent[vertical],
            facecolor='#edf1f4',edgecolor='#657584',linewidth=1.5))
        for row in blocks:
            start=np.array(row['origin_nm'])/1000;size=np.array(row['shape'])*training['voxel_size_nm']/1000
            ax.add_patch(Rectangle((start[2],start[vertical]),size[2],size[vertical],
                fill=False,edgecolor='#c47925',linewidth=2,linestyle='--'))
        ax.add_patch(Rectangle((centers[2,0],centers[vertical,0]),np.diff(centers[2])[0],np.diff(centers[vertical])[0],
            facecolor='#168b9b',edgecolor='#106b77'))
        ax.set_xlim(origin[2]-1,origin[2]+extent[2]+1);ax.set_ylim(origin[vertical]-1,origin[vertical]+extent[vertical]+1)
        ax.set_aspect('equal');ax.set_title(title);ax.set_xlabel('X (µm)');ax.set_ylabel(('Y' if vertical==1 else 'Z')+' (µm)')
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f'10 µm radius → {100*report["eligible_center_volume_fraction"]:.3f}% eligible center positions',fontsize=14)
    fig.legend(handles=[Patch(facecolor='#edf1f4',edgecolor='#657584',label='EM ROI'),
        Patch(facecolor='#168b9b',label='Eligible centers'),
        Patch(facecolor='none',edgecolor='#c47925',linestyle='--',label='Training image extent')],
        loc='lower center',bbox_to_anchor=(.5,.065),ncol=3,frameon=False)
    fig.text(.5,.025,'Geometry only · not instance coverage',ha='center',fontsize=9,color='#52616b')
    fig.tight_layout(rect=(0,.22,1,.94));output=external(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=180);plt.close(fig)
    write_json(output.with_suffix('.json'),dict(audit_sha256=sha256(a.audit),
        training_manifest_sha256=sha256(a.training_manifest),figure_sha256=sha256(output),
        scope=report['scope'],model_performance_shown=False))


if __name__=='__main__':main()
