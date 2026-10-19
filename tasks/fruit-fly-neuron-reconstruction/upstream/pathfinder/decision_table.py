"""Hash-bound, memory-mapped decision rows with legacy list compatibility."""
from collections.abc import Sequence
from pathlib import Path
import operator
import numpy as np
from dataset_contract import sha256
from decision_spill import RESULT

class DecisionRows(Sequence):
    def __init__(self,array,indices=None):
        self.array=array; self.indices=range(len(array)) if indices is None else indices
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):
        if isinstance(index,slice):return DecisionRows(self.array,self.indices[index])
        row=self.array[self.indices[operator.index(index)]]
        return dict(pair=[int(row['a']),int(row['b'])],distance_nm=float(row['d']),
                    center_zyx=[int(row[k]) for k in ('z','y','x')])

    def pair_index(self,pair):
        lo=0;hi=len(self.array);pair=tuple(pair)
        while lo<hi:
            mid=(lo+hi)//2;row=self.array[mid]
            current=(int(row['a']),int(row['b']))
            if current<pair:lo=mid+1
            else:hi=mid
        if lo==len(self.array):raise ValueError('Unexpected candidate result')
        row=self.array[lo]
        if (int(row['a']),int(row['b']))!=pair:raise ValueError('Unexpected candidate result')
        try:return self.indices.index(lo)
        except ValueError:raise ValueError('Candidate belongs to another shard') from None


def write_table(path,rows,count,origin):
    path=Path(path);partial=path.with_suffix(path.suffix+'.partial')
    if path.exists() or partial.exists():raise ValueError('New decision table required')
    if count<0 or len(origin)!=3:raise ValueError('Invalid decision geometry')
    array=np.lib.format.open_memmap(partial,mode='w+',dtype=RESULT,shape=(count,))
    previous=None;written=0
    try:
        for pair,distance,point in rows:
            if written>=count or (previous is not None and pair<=previous):
                raise ValueError('Decision count/order mismatch')
            array[written]=(*pair,distance,*(int(v)+int(o) for v,o in zip(point,origin)))
            previous=pair;written+=1
        if written!=count:raise ValueError('Incomplete decision table')
        array.flush()
    finally:array._mmap.close()
    partial.replace(path)
    return dict(storage='ffn-decision-table/v1',file=path.name,count=count,sha256=sha256(path))


def load_rows(plan,manifest_path):
    if 'decision_table' not in plan:return plan['decisions']
    if 'decisions' in plan:raise ValueError('Ambiguous decision encoding')
    descriptor=plan['decision_table'];name=descriptor['file']
    if descriptor['storage']!='ffn-decision-table/v1' or Path(name).name!=name:
        raise ValueError('Invalid decision table descriptor')
    path=Path(manifest_path).parent/name
    if path.is_symlink() or sha256(path)!=descriptor['sha256']:
        raise ValueError('Decision table identity mismatch')
    array=np.load(path,mmap_mode='r',allow_pickle=False)
    if array.dtype!=RESULT or array.shape!=(descriptor['count'],):
        array._mmap.close();raise ValueError('Decision table shape/dtype mismatch')
    previous=None
    for start in range(0,len(array),100000):
        block=array[start:start+100000];a=block['a'];b=block['b']
        if (np.any(a==0) or np.any(a>=b) or np.any(~np.isfinite(block['d']))
            or np.any((a[1:]<a[:-1]) | ((a[1:]==a[:-1]) & (b[1:]<=b[:-1])))
            or (previous is not None and (int(a[0]),int(b[0]))<=previous)):
            array._mmap.close();raise ValueError('Invalid or unordered decision table')
        previous=(int(a[-1]),int(b[-1]))
    return DecisionRows(array)
