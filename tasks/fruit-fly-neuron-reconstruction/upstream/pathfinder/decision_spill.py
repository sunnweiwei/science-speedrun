"""Bounded disk partitioning for exact global FFN decision reduction.

This changes storage only. Every local minimum vote, including repeated votes
from different offsets, is retained. Production integration is deliberately
separate from qualification of this reducer.
"""
import heapq
from contextlib import ExitStack
from pathlib import Path
import numpy as np
from decision_centroid import reduce_centroid

VOTE = np.dtype([('a','<u8'),('b','<u8'),('d','<f8'),('z','<i8'),
                 ('y','<i8'),('x','<i8'),('offset','u1'),('group_size','<u8')])
RESULT = np.dtype([('a','<u8'),('b','<u8'),('d','<f8'),
                   ('z','<i8'),('y','<i8'),('x','<i8')])


def partition_votes(group_factory, directory, *, partitions, max_disk_bytes):
    directory=Path(directory)
    if type(partitions) is not int or not 1 <= partitions <= 1024 or max_disk_bytes < 1 or directory.exists():
        raise ValueError('New directory and positive explicit bounds required')
    directory.mkdir(parents=True)
    counts=[0]*partitions; used=0
    # At most1024 handles/64MiB buffers; flush/close also on failure.
    with ExitStack() as stack:
        handles={}
        for pair,distance,points,offset in group_factory():
            if not (0 < pair[0] < pair[1] < 2**64 and np.isfinite(distance)
                    and 0 <= offset < 7 and points.ndim == 2 and points.shape[1] == 3
                    and points.dtype.kind in 'iu' and len(points)):
                raise ValueError('Invalid decision vote')
            required=len(points)*VOTE.itemsize
            if used+required > max_disk_bytes:
                raise ValueError('Decision spill disk limit exceeded; no complete result')
            bucket=((int(pair[0])*11400714819323198485) ^ (int(pair[1])*14029467366897019727))%partitions
            rows=np.empty(len(points),VOTE)
            rows['a']=pair[0]; rows['b']=pair[1]; rows['d']=distance
            for axis,name in enumerate(('z','y','x')):rows[name]=points[:,axis]
            rows['offset']=offset; rows['group_size']=len(points)
            if bucket not in handles:
                handles[bucket]=stack.enter_context((directory/f'votes-{bucket}.bin').open('ab',buffering=65536))
            handles[bucket].write(rows.tobytes())
            counts[bucket]+=len(rows);used+=required
    return dict(counts=counts,bytes=used)


def vote_groups(path):
    if not path.exists():return
    rows=np.memmap(path,dtype=VOTE,mode='r')
    try:
        start=0
        while start < len(rows):
            row=rows[start];count=int(row['group_size'])
            if count < 1 or start+count > len(rows):raise ValueError('Incomplete vote group')
            group=rows[start:start+count]
            points=np.column_stack([group[k] for k in ('z','y','x')])
            yield ((int(row['a']),int(row['b'])),float(row['d']),points,int(row['offset']))
            start+=count
    finally:
        rows._mmap.close()


def reduce_partitions(directory, counts, *, max_pairs_per_partition, max_pairs, max_disk_bytes):
    directory=Path(directory);total=0
    used=sum((directory/f'votes-{i}.bin').stat().st_size for i,n in enumerate(counts) if n)
    paths=[]
    for i,n in enumerate(counts):
        if not n:continue
        path=directory/f'votes-{i}.bin'
        if path.stat().st_size != n*VOTE.itemsize:raise ValueError('Incomplete vote partition')
        rows=reduce_centroid(lambda:vote_groups(path),max_pairs=max_pairs_per_partition)
        total+=len(rows)
        if total>max_pairs:raise ValueError('Decision pair limit exceeded; no complete result')
        # Include NPY header allowance, retain original votes as evidence.
        required=len(rows)*RESULT.itemsize+256
        if used+required>max_disk_bytes:raise ValueError('Decision spill disk limit exceeded; no complete result')
        result=np.empty(len(rows),RESULT)
        for k,(pair,distance,point) in enumerate(rows):result[k]=(*pair,distance,*point)
        out=directory/f'result-{i}.npy';np.save(out,result,allow_pickle=False)
        used+=out.stat().st_size;paths.append(out)
    return paths,total


def sorted_results(paths):
    def records(path):
        array=np.load(path,mmap_mode='r',allow_pickle=False)
        try:
            for row in array:
                yield ((int(row['a']),int(row['b'])),float(row['d']),
                       tuple(int(row[k]) for k in ('z','y','x')))
        finally:
            array._mmap.close()
    yield from heapq.merge(*(records(p) for p in paths),key=lambda row:row[0])

# Separate opt-in reducer. The serial production path remains unchanged until
# independently qualified parallel execution is admitted in a new snapshot.
_PARALLEL_STATE=None


def _init_parallel(lock,used,total,max_disk_bytes,max_pairs):
    global _PARALLEL_STATE
    _PARALLEL_STATE=(lock,used,total,max_disk_bytes,max_pairs)


def _reduce_partition_job(task):
    directory,i,n,max_pairs_per_partition,*extras=task;directory=Path(directory)
    inputs=extras[0] if extras else [(str(directory/f'votes-{i}.bin'),n)]
    for name,count in inputs:
        if Path(name).stat().st_size!=count*VOTE.itemsize:raise ValueError('Incomplete vote partition')
    if sum(count for _,count in inputs)!=n:raise ValueError('Partition vote count mismatch')
    def all_votes():
        for name,_ in inputs:yield from vote_groups(Path(name))
    rows=reduce_centroid(all_votes,max_pairs=max_pairs_per_partition)
    required=len(rows)*RESULT.itemsize+256
    lock,used,total,max_disk_bytes,max_pairs=_PARALLEL_STATE
    with lock:
        if total.value+len(rows)>max_pairs:raise ValueError('Decision pair limit exceeded; no complete result')
        if used.value+required>max_disk_bytes:raise ValueError('Decision spill disk limit exceeded; no complete result')
        total.value+=len(rows);used.value+=required
    # Reserve before writing. On failure, keep reservation and partial evidence;
    # no complete collection or table is published by the failed caller.
    out=directory/f'result-{i}.npy';partial=out.with_suffix('.npy.partial')
    if out.exists() or partial.exists():raise ValueError('New partition result required')
    with partial.open('xb') as f:
        result=np.empty(len(rows),RESULT)
        for k,(pair,distance,point) in enumerate(rows):result[k]=(*pair,distance,*point)
        np.save(f,result,allow_pickle=False)
    if partial.stat().st_size>required:raise ValueError('NPY result exceeds reserved bytes')
    partial.replace(out)
    return i,out,len(rows)


def reduce_partitions_parallel(directory,counts,*,workers,max_pairs_per_partition,max_pairs,max_disk_bytes):
    """Independent local CPU partitions; shared locked capacity reservations."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    if type(workers) is not int or not 1<=workers<=32:
        raise ValueError('Explicit1–32 local CPU workers required')
    if min(max_pairs_per_partition,max_pairs,max_disk_bytes)<1:raise ValueError('Positive bounds required')
    directory=Path(directory)
    used=sum((directory/f'votes-{i}.bin').stat().st_size for i,n in enumerate(counts) if n)
    if used>max_disk_bytes:raise ValueError('Existing votes exceed disk limit')
    ctx=mp.get_context('spawn');lock=ctx.Lock();used_counter=ctx.Value('Q',used);total=ctx.Value('Q',0)
    tasks=[(str(directory),i,n,max_pairs_per_partition) for i,n in enumerate(counts) if n]
    with ProcessPoolExecutor(max_workers=workers,mp_context=ctx,initializer=_init_parallel,
                             initargs=(lock,used_counter,total,max_disk_bytes,max_pairs)) as pool:
        results=list(pool.map(_reduce_partition_job,tasks))
    results.sort()
    return [p for _,p,_ in results],sum(n for _,_,n in results)
