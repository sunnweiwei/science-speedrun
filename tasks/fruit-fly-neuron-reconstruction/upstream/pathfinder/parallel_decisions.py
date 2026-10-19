"""Opt-in local CPU chunk extraction and exact cross-worker pair reduction.

Requires complete global official expansion arrays on disk. It never runs a
local watershed approximation and never drops chunks, votes or pair groups.
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import numpy as np
from decision_centroid import groups
from decision_spill import partition_votes,VOTE,_init_parallel,_reduce_partition_job


def memory_preflight(shape,*,workers,max_pairs_per_partition,memory_gib):
    import math
    if type(workers) is not int or not 1<=workers<=32 or max_pairs_per_partition<1:
        raise ValueError('Bounded parallel resources required')
    estimate=32*math.prod(shape)+workers*(512*129**3+64*2**20+1024*max_pairs_per_partition)
    if estimate>memory_gib*2**30:raise ValueError('Parallel decision memory exceeds declared allocation')
    return estimate


def _extract(task):
    expanded_path,distance_path,roi,directory,rank,workers,partitions,tile,budget=task
    expanded=np.load(expanded_path,mmap_mode='r',allow_pickle=False)
    distance=np.load(distance_path,mmap_mode='r',allow_pickle=False)
    try:
        if expanded.ndim!=3 or expanded.shape!=distance.shape or expanded.dtype.kind!='u' or distance.dtype.kind!='f':
            raise ValueError('Aligned full expansion arrays required')
        return partition_votes(lambda:groups(expanded,distance,roi,tile=tile,chunk_rank=rank,chunk_shards=workers),
                               Path(directory)/f'worker-{rank}',partitions=partitions,max_disk_bytes=budget)
    finally:
        expanded._mmap.close();distance._mmap.close()


def extract(expanded_path,distance_path,roi,directory,*,workers,partitions,tile,max_disk_bytes):
    directory=Path(directory)
    if type(workers) is not int or not 1<=workers<=32 or directory.exists() or max_disk_bytes<workers:
        raise ValueError('New output and explicit1–32 CPU workers/disk bounds required')
    directory.mkdir(parents=True)
    tasks=[(str(expanded_path),str(distance_path),roi,str(directory),rank,workers,partitions,tile,max_disk_bytes//workers)
           for rank in range(workers)]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        manifests=list(pool.map(_extract,tasks))
    return dict(workers=manifests,partitions=partitions,bytes=sum(m['bytes'] for m in manifests),
                chunk_ownership='lexicographic chunk ordinal modulo worker count; unchanged owned anchors')


def reduce(directory,manifest,*,workers,max_pairs_per_partition,max_pairs,max_disk_bytes):
    directory=Path(directory)
    if type(workers) is not int or not 1<=workers<=32 or min(max_pairs_per_partition,max_pairs,max_disk_bytes)<1:
        raise ValueError('Explicit positive bounded resources required')
    tasks=[];used=0
    for i in range(manifest['partitions']):
        inputs=[]
        for rank,m in enumerate(manifest['workers']):
            n=m['counts'][i]
            if not n:continue
            path=directory/f'worker-{rank}/votes-{i}.bin'
            if path.stat().st_size!=n*VOTE.itemsize:raise ValueError('Incomplete extraction shard')
            used+=path.stat().st_size;inputs.append((str(path),n))
        if inputs:tasks.append((str(directory),i,sum(n for _,n in inputs),max_pairs_per_partition,inputs))
    if used!=manifest['bytes'] or used>max_disk_bytes:raise ValueError('Extraction bytes/budget mismatch')
    ctx=mp.get_context('spawn');lock=ctx.Lock();used_counter=ctx.Value('Q',used);total=ctx.Value('Q',0)
    with ProcessPoolExecutor(max_workers=workers,mp_context=ctx,initializer=_init_parallel,
                             initargs=(lock,used_counter,total,max_disk_bytes,max_pairs)) as pool:
        results=list(pool.map(_reduce_partition_job,tasks))
    results.sort();return [p for _,p,_ in results],sum(n for _,_,n in results)
