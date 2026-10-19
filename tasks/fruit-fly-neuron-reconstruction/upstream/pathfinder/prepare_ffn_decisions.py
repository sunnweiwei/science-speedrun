"""Generate prediction-only decision points using Google's pinned FFN utility.

The production entry uses whole-volume official watershed and globally exact
centroid reduction. The historical local generator is retained for old receipts
only; it is not equivalent to the official rule. No GT input exists.
"""
import os
import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from dataset_contract import sha256

FFN_COMMIT='b8df2d96d7c8da057fd2702ced19732a08ab6ae4'
CONNECTOMICS_COMMIT='bf2f6caf083bb5d459ec86f5e859c366857be1ea'


def author_function(ffn_source, connectomics_source):
    for path, expected in ((ffn_source,FFN_COMMIT),(connectomics_source,CONNECTOMICS_COMMIT)):
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip()
        dirty=subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=path,text=True).strip()
        if head != expected or dirty:raise ValueError('Author source commit/worktree mismatch')
        sys.path.insert(0,str(path.resolve()))
    from ffn.utils.decision_point import find_decision_points
    from connectomics.common.bounding_box import BoundingBox
    return find_decision_points,BoundingBox


def generate(base, roi, find_points, box_class, *, tile=128, max_distance_nm=64):
    if base.ndim!=3 or base.dtype.kind!='u' or tile<1 or max_distance_nm<=0:
        raise ValueError('Unsigned 3D base, positive tile and max distance required')
    lo=np.asarray(roi[::2],dtype=int);hi=np.asarray(roi[1::2],dtype=int)
    if len(roi)!=6 or (lo<50).any() or (hi>np.array(base.shape)-50).any() or (hi<=lo).any():
        raise ValueError('Decision ROI requires 50-voxel resegmentation context')
    halo=int(np.ceil(max_distance_nm/16))+2; merged={};chunks=0
    for start in itertools.product(*(range(a,b,tile) for a,b in zip(lo,hi))):
        start=np.array(start);stop=np.minimum(start+tile,hi)
        # Include the next voxel so adjacent fragments across tile faces meet.
        scan_stop=np.minimum(stop+1,hi)
        context_lo=np.maximum(start-halo,0);context_hi=np.minimum(scan_stop+halo,base.shape)
        seg=np.array(base[tuple(slice(int(a),int(b)) for a,b in zip(context_lo,context_hi))],dtype=np.uint64)
        box=box_class(start=(start-context_lo)[::-1],size=(scan_stop-start)[::-1])
        found=find_points(seg,[16]*3,max_distance=max_distance_nm,subvol_box=box,optimize_sparse=True)
        for pair,(distance,xyz) in found.items():
            # Author utility returns coordinates relative to the selected subvolume.
            center=tuple(map(int,np.asarray(xyz)[::-1]+start))
            candidate=(float(distance),center);pair=tuple(map(int,pair))
            if pair not in merged or candidate<merged[pair]:merged[pair]=candidate
        chunks+=1
    decisions=[dict(pair=list(pair),distance_nm=d,center_zyx=list(c)) for pair,(d,c) in sorted(merged.items())]
    return dict(decisions=decisions,chunks=chunks,roi_zyx=list(roi),tile=tile,
                max_distance_nm=max_distance_nm,halo_voxels=halo)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('base','output','ffn_source','connectomics_source'):
        parser.add_argument('--'+name.replace('_','-'),required=True,type=Path)
    parser.add_argument('--roi',required=True,nargs=6,type=int)
    parser.add_argument('--memory-gib',required=True,type=int)
    parser.add_argument('--max-pairs',required=True,type=int)
    parser.add_argument('--spill-directory',type=Path)
    parser.add_argument('--spill-partitions',type=int,default=256)
    parser.add_argument('--spill-workers',type=int,default=1)
    parser.add_argument('--spill-disk-gib',type=int)
    parser.add_argument('--max-pairs-per-partition',type=int)
    a=parser.parse_args()
    if not str(a.output.resolve()).startswith(os.environ.get('PATHFINDER_SCRATCH_ROOT', '/scratch').rstrip('/') + '/') or a.output.exists():
        raise ValueError('New Xtemp output required')
    find_points,box_class=author_function(a.ffn_source,a.connectomics_source)
    from connectomics.segmentation.labels import watershed_expand
    from decision_centroid import generate_global
    base=np.load(a.base,mmap_mode='r')
    if a.spill_directory:
        from decision_centroid import memory_preflight,groups
        from decision_spill import partition_votes,reduce_partitions,sorted_results,RESULT
        from decision_table import write_table
        if (not str(a.spill_directory.resolve()).startswith(os.environ.get('PATHFINDER_SCRATCH_ROOT', '/scratch').rstrip('/') + '/')
            or not a.spill_disk_gib or a.spill_disk_gib<1
            or not a.max_pairs_per_partition or a.max_pairs_per_partition<1
            or not 1<=a.spill_partitions<=1024 or not 1<=a.spill_workers<=32):
            raise ValueError('Bounded Xtemp spill resources required')
        estimate=memory_preflight(list(base.shape),a.roi,memory_gib=a.memory_gib,
                                  max_pairs=a.max_pairs_per_partition)
        import resource,math
        if resource.getrlimit(resource.RLIMIT_NOFILE)[0]<a.spill_partitions+64:
            raise ValueError('Insufficient file descriptor limit for declared partitions')
        expansion_reserve=0;expansion_receipt=None
        if a.spill_workers>1:
            from parallel_decisions import memory_preflight as parallel_memory
            estimate=max(estimate,parallel_memory(base.shape,workers=a.spill_workers,
                max_pairs_per_partition=a.max_pairs_per_partition,memory_gib=a.memory_gib))
            expansion_reserve=16*math.prod(base.shape)+(a.spill_workers+1)*2**20
            expansion_dir=a.spill_directory.with_name(a.spill_directory.name+'-expansion')
            if expansion_dir.exists():raise ValueError('New full expansion directory required')
        disk_bytes=a.spill_disk_gib*2**30
        # Reserve the final table too; retain votes and per-partition results.
        final_reserve=a.max_pairs*RESULT.itemsize+256+expansion_reserve
        if final_reserve>=disk_bytes:raise ValueError('Spill disk cannot hold final decision table')
        expanded,distance=watershed_expand(base,[16]*3,max_distance=64)
        if expanded.shape!=base.shape or distance.shape!=base.shape:
            raise ValueError('Official expansion geometry differs from base')
        if a.spill_workers==1:
            manifest=partition_votes(lambda:groups(expanded,distance,a.roi,tile=128),a.spill_directory,
                                      partitions=a.spill_partitions,max_disk_bytes=disk_bytes-final_reserve)
            del expanded,distance
            paths,count=reduce_partitions(a.spill_directory,manifest['counts'],
                max_pairs_per_partition=a.max_pairs_per_partition,max_pairs=a.max_pairs,
                max_disk_bytes=disk_bytes-final_reserve)
        else:
            from parallel_decisions import extract,reduce
            expansion_dir.mkdir(parents=True)
            if expanded.nbytes+distance.nbytes+4096>expansion_reserve:
                raise ValueError('Expansion dtype exceeds explicit disk reservation')
            np.save(expansion_dir/'expanded.npy',expanded,allow_pickle=False)
            np.save(expansion_dir/'distance.npy',distance,allow_pickle=False)
            del expanded,distance
            expansion_receipt=dict(base_sha256=sha256(a.base),shape=list(base.shape),max_distance_nm=64,
                files={name:sha256(expansion_dir/name) for name in ('expanded.npy','distance.npy')},
                ffn_commit=FFN_COMMIT,connectomics_commit=CONNECTOMICS_COMMIT,
                scope='Complete global official watershed arrays, never independent local expansions')
            (expansion_dir/'receipt.json').write_text(json.dumps(expansion_receipt,indent=2)+'\n')
            manifest=extract(expansion_dir/'expanded.npy',expansion_dir/'distance.npy',a.roi,a.spill_directory,
                workers=a.spill_workers,partitions=a.spill_partitions,tile=128,max_disk_bytes=disk_bytes-final_reserve)
            (a.spill_directory/'extraction.json').write_text(json.dumps(manifest,indent=2)+'\n')
            paths,count=reduce(a.spill_directory,manifest,workers=a.spill_workers,
                max_pairs_per_partition=a.max_pairs_per_partition,max_pairs=a.max_pairs,
                max_disk_bytes=disk_bytes-final_reserve)
            for name,digest in expansion_receipt['files'].items():
                if sha256(expansion_dir/name)!=digest:raise ValueError('Full expansion changed during reduction')
        a.output.parent.mkdir(parents=True,exist_ok=True)
        descriptor=write_table(a.output.with_suffix('.npy'),sorted_results(paths),count,a.roi[::2])
        result=dict(decision_table=descriptor,roi_zyx=a.roi,tile=128,max_distance_nm=64,
                    chunks=int(np.prod((np.array(a.roi[1::2])-np.array(a.roi[::2])+127)//128)),
                    max_pairs=a.max_pairs,max_pairs_per_partition=a.max_pairs_per_partition,
                    estimated_memory_bytes=estimate,declared_memory_gib=a.memory_gib,
                    spill_partitions=a.spill_partitions,spill_workers=a.spill_workers,spill_disk_bytes=disk_bytes,
                    global_expansion=expansion_receipt,
                    vote_bytes=manifest['bytes'],passes=2,
                    expansion_scope='Pinned official watershed on the entire input volume; disk-partitioned exact candidate reduction',
                    cross_chunk_tie_rule='All global-minimum votes, original point groups, nearest global XYZ centroid; official offset then ZYX anchor',
                    spill_sha256=sha256(Path(__file__).with_name('decision_spill.py')),
                    table_adapter_sha256=sha256(Path(__file__).with_name('decision_table.py')))
    else:
        if a.spill_disk_gib is not None or a.max_pairs_per_partition is not None or a.spill_workers!=1:
            raise ValueError('Spill resource settings require a spill directory')
        result=generate_global(base,a.roi,watershed_expand,memory_gib=a.memory_gib,max_pairs=a.max_pairs)
    result.update(schema='ffn-decision-points/v1',base_sha256=sha256(a.base),ground_truth_used=False,
        author_sources={'ffn_commit':FFN_COMMIT,'connectomics_commit':CONNECTOMICS_COMMIT,
                        'decision_point_sha256':sha256(a.ffn_source/'ffn/utils/decision_point.py'),
                        'labels_sha256':sha256(a.connectomics_source/'connectomics/segmentation/labels.py')},
        adapter_sha256=sha256(__file__),reducer_sha256=sha256(Path(__file__).with_name('decision_centroid.py')))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(candidates=result['decision_table']['count'] if 'decision_table' in result else len(result['decisions']),chunks=result['chunks'])))


if __name__=='__main__':main()
