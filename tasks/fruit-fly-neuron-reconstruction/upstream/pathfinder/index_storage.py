"""Read legacy or hashed numeric instance indices without expanding all records."""
import hashlib,json,sys
from pathlib import Path
from collections.abc import Sequence,Mapping
import numpy as np
from compact_index import DTYPE
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import sha256


class TableRecords(Sequence):
    def __init__(self,table,shape,ids=None):
        self.table=table;self.shape=tuple(shape)
        self.ids=np.flatnonzero(table['size']) if ids is None else ids
    def __len__(self):return len(self.ids)
    def __getitem__(self,item):
        if isinstance(item,slice):return TableRecords(self.table,self.shape,self.ids[item])
        return self.by_id(int(self.ids[item]))
    def by_id(self,sid):
        if sid<=0 or sid>=len(self.table) or not self.table['size'][sid]:raise KeyError(sid)
        r=self.table[sid]
        return dict(id=int(sid),size=int(r['size']),bbox_zyx=[[int(a),int(b)] for a,b in zip(r['lo'],r['hi'])],
            center_zyx=list(map(int,np.unravel_index(int(r['center']),self.shape))))


class TableMapping(Mapping):
    def __init__(self,rows,*,sizes=False):self.rows=rows;self.sizes=sizes
    def __len__(self):return len(self.rows)
    def __iter__(self):return map(int,self.rows.ids)
    def sizes_for(self,ids):
        """Bulk size lookup with the same missing-ID checks as __getitem__."""
        ids=np.asarray(ids)
        if not self.sizes or ids.ndim!=1 or ids.dtype.kind not in 'iu':raise ValueError('Numeric size mapping required')
        if np.any(ids<=0) or np.any(ids>=len(self.rows.table)):raise KeyError('Invalid numeric instance ID')
        sizes=self.rows.table['size'][ids]
        if np.any(sizes==0):raise KeyError('Missing numeric instance ID')
        return sizes
    def __getitem__(self,sid):
        if not isinstance(sid,(int,np.integer)) or sid<=0 or sid>=len(self.rows.table) or not self.rows.table['size'][sid]:raise KeyError(sid)
        return int(self.rows.table['size'][sid]) if self.sizes else self.rows.by_id(sid)


def load_index(path):
    path=Path(path);index=json.loads(path.read_text())
    if index.get('storage')!='numeric-table-v1':return index
    if 'records' in index:raise ValueError('Ambiguous numeric and legacy records')
    d=index['table'];name=d['file']
    if Path(name).name!=name or name in ('.','..'):raise ValueError('Numeric table must be a local file')
    p=path.parent/name
    if p.is_symlink() or sha256(p)!=d['sha256'] or p.stat().st_size!=d['bytes']:raise ValueError('Numeric index table changed')
    table=np.load(p,mmap_mode='r',allow_pickle=False)
    try:
        if table.ndim!=1 or table.dtype!=DTYPE or len(table)!=d['labels']:raise ValueError('Numeric index table format mismatch')
        if not len(table) or table['size'][0]:raise ValueError('Background must not be an instance')
        shape=index['shape']
        if len(shape)!=3 or any(type(x) is not int or x<1 for x in shape):raise ValueError('Invalid index shape')
        rows=TableRecords(table,shape)
        if len(rows)!=d['instances']:raise ValueError('Numeric instance count mismatch')
        index['records']=rows;return index
    except BaseException:table._mmap.close();raise


def validate_table(rows,seg):
    if tuple(seg.shape)!=rows.shape:raise ValueError('Numeric index geometry mismatch')
    for start in range(0,len(rows),100000):
        ids=rows.ids[start:start+100000];r=rows.table[ids]
        if np.any(r['center']>=seg.size):raise ValueError('Numeric center outside volume')
        xyz=np.column_stack(np.unravel_index(r['center'].astype(np.intp),seg.shape))
        if (np.any(r['lo']<0) or np.any(r['hi']>seg.shape) or np.any(r['hi']<=r['lo'])
            or np.any(xyz<r['lo']) or np.any(xyz>=r['hi'])
            or np.any(r['size']>np.prod((r['hi']-r['lo']).astype(np.int64),axis=1))):
            raise ValueError('Invalid numeric instance geometry')
        if not np.array_equal(seg[tuple(xyz.T)],ids):raise ValueError('Numeric center label mismatch')


def select_table_centers(rows,shape,shard_index,shards):
    eligible=[];small_skipped=[];skipped_count=0;skipped_digest=hashlib.sha256()
    for start in range(0,len(rows),100000):
        ids=rows.ids[start:start+100000]
        physical=(np.column_stack(np.unravel_index(rows.table['center'][ids].astype(np.intp),shape))+.5)*16
        keep=((physical>=10000)&(physical<=np.asarray(shape)*16-10000)).all(axis=1)
        eligible.append(ids[keep]);excluded=ids[~keep]
        skipped_count+=len(excluded)
        skipped_digest.update(np.asarray(excluded,dtype='<u8').tobytes())
        if skipped_count<=10000:small_skipped.extend(map(int,excluded))
        else:small_skipped.clear()
    eligible=np.concatenate(eligible) if eligible else np.empty(0,np.intp)
    # Large excluded populations need identity evidence, not hundreds of
    # millions of Python integer objects in every GPU process. The full
    # hashed index and this fixed predicate permit exact regeneration.
    skipped=small_skipped if skipped_count<=10000 else dict(
        storage='ordered-excluded-id-digest/v1',count=skipped_count,
        sha256=skipped_digest.hexdigest(),encoding='ascending little-endian uint64',
        shape=list(shape),voxel_size_nm=[16]*3,radius_nm=10000,
        selection='positive indexed IDs whose representative center fails the full geometric context predicate')
    selected=np.arange(len(eligible))%shards==shard_index
    return TableRecords(rows.table,shape,eligible[selected]),skipped,eligible[~selected].tolist(),eligible.tolist()


class NumericNodes:
    """Validated sorted unique IDs; membership uses no Python hash set."""
    def __init__(self,ids):
        ids=np.asarray(ids)
        if ids.ndim!=1 or ids.dtype.kind not in 'iu' or (len(ids) and (ids[0]<=0 or np.any(ids[1:]<=ids[:-1]))):
            raise ValueError('Strictly increasing positive numeric node IDs required')
        self.ids=ids
    def __len__(self):return len(self.ids)
    def __iter__(self):return map(int,self.ids)
    def __contains__(self,node):
        if type(node) is not int or node<=0:return False
        pos=int(np.searchsorted(self.ids,node))
        return pos<len(self.ids) and int(self.ids[pos])==node
