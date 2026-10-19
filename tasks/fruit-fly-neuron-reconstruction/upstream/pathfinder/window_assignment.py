"""Deterministic scheduling only: preserve each original FFN window verbatim."""
import math


POLICIES=('modulo','voxel-balanced-v1')


def assign(windows,shards,policy='modulo'):
    if type(shards) is not int or shards<1 or policy not in POLICIES:
        raise ValueError('Positive shard count and known window assignment required')
    indices=[w['index'] for w in windows]
    if any(type(i) is not int or i<0 for i in indices) or len(set(indices))!=len(indices):
        raise ValueError('Unique nonnegative window indices required')
    weights={}
    for w in windows:
        lengths=[b-a for a,b in zip(w['start'],w['stop'])]
        if len(w['start'])!=3 or len(w['stop'])!=3 or any(type(n) is not int or n<=0 for n in lengths):
            raise ValueError('Positive 3D window geometry required')
        weights[w['index']]=math.prod(lengths)
    if policy=='modulo':return {i:i%shards for i in indices}
    loads=[0]*shards;owners={}
    for index in sorted(indices,key=lambda i:(-weights[i],i)):
        rank=min(range(shards),key=lambda r:(loads[r],r))
        owners[index]=rank;loads[rank]+=weights[index]
    return owners
