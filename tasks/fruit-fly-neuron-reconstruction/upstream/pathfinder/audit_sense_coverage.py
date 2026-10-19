"""Independent numeric-index census of merged SENSE coverage, not model accuracy."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import sha256,external,write_json


def audit_coverage(table,shape,coverage,chunk=262144):
    shape=np.asarray(shape,dtype=np.int64)
    if shape.shape!=(3,) or np.any(shape<=0) or chunk<1:raise ValueError('Invalid census geometry')
    if coverage['unprocessed_instance_ids']:raise ValueError('Unprocessed centers remain')
    eligible=coverage['eligible_instance_ids'];processed=coverage['processed']
    selected_hash=hashlib.sha256();excluded_hash=hashlib.sha256();offset=0;excluded_count=0;total=0
    small_excluded=[]
    for start in range(0,len(table),chunk):
        rows=table[start:start+chunk];local=np.flatnonzero(rows['size']>0)
        ids=local.astype(np.uint64)+start
        if np.any(ids==0):raise ValueError('Background is not a positive instance')
        center=rows['center'][local]
        if np.any(center>=int(np.prod(shape))):raise ValueError('Center outside volume')
        z=center//int(shape[1]*shape[2]);y=(center//int(shape[2]))%int(shape[1]);x=center%int(shape[2])
        xyz=np.column_stack((z,y,x)).astype(np.int64)
        # Integer arithmetic, independently implemented from production's
        # floating point center selection and different chunk boundaries.
        physical=xyz*16+8
        keep=np.all((physical>=10000)&(physical<=shape*16-10000),axis=1)
        selected=ids[keep];excluded=ids[~keep];selected_xyz=xyz[keep]
        selected_hash.update(selected.astype('<u8').tobytes());excluded_hash.update(excluded.astype('<u8').tobytes())
        count=len(selected)
        if eligible[offset:offset+count]!=selected.tolist():raise ValueError('Eligible identity/order differs from independent census')
        batch=processed[offset:offset+count]
        if len(batch)!=count:raise ValueError('Missing processed records')
        for row,sid,coord in zip(batch,selected,selected_xyz):
            if row['instance_id']!=int(sid) or row['center_zyx']!=coord.tolist() or row.get('geometric_shape_context_complete') is not True:
                raise ValueError('Processed center identity/geometry differs')
        offset+=count;excluded_count+=len(excluded);total+=len(ids)
        if excluded_count<=10000:small_excluded.extend(excluded.tolist())
        else:small_excluded.clear()
    if len(eligible)!=offset or len(processed)!=offset:raise ValueError('Extra eligible/processed records')
    if coverage['eligible_count']!=offset or coverage['processed_count']!=offset:raise ValueError('Coverage counters differ')
    excluded=coverage['skipped_geometric_context_instance_ids']
    if isinstance(excluded,list):
        if excluded_count>10000 or excluded!=small_excluded:raise ValueError('Excluded identity differs')
    elif (excluded.get('storage')!='ordered-excluded-id-digest/v1' or excluded.get('count')!=excluded_count
          or excluded.get('sha256')!=excluded_hash.hexdigest() or excluded.get('shape')!=shape.tolist()
          or excluded.get('radius_nm')!=10000 or excluded.get('voxel_size_nm')!=[16]*3
          or excluded.get('encoding')!='ascending little-endian uint64'):
        raise ValueError('Excluded digest or geometry differs')
    return dict(complete=True,eligible_count=offset,excluded_count=excluded_count,total_instances=total,
                eligible_sha256=selected_hash.hexdigest(),excluded_sha256=excluded_hash.hexdigest(),
                full_pipeline_complete=False,scope='Independent full numeric-index center coverage and identity; not SENSE predictions, SHAPE or score audit')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('index','merged','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=external(a.output)
    if out.exists():raise FileExistsError(out)
    pins={str(path):sha256(path) for path in (a.index,a.merged)}
    index=json.loads(a.index.read_text());desc=index['table'];name=desc['file']
    if Path(name).name!=name:raise ValueError('Local numeric table required')
    path=a.index.parent/name
    if sha256(path)!=desc['sha256'] or path.stat().st_size!=desc['bytes']:raise ValueError('Index table changed')
    result=json.loads(a.merged.read_text())
    if result['provenance']['instance_index_sha256']!=pins[str(a.index)]:raise ValueError('Merged index provenance differs')
    table_path=path
    table=np.load(path,mmap_mode='r',allow_pickle=False)
    try:report=audit_coverage(table,index['shape'],result['coverage'])
    finally:table._mmap.close()
    if report['total_instances']!=desc['instances']:raise ValueError('Index total differs')
    for path,digest in pins.items():
        if sha256(path)!=digest:raise ValueError('Audit input changed')
    if sha256(table_path)!=desc['sha256']:raise ValueError('Index table changed during audit')
    report['input_sha256']=pins;report['table_sha256']=desc['sha256'];write_json(out,report)

if __name__=='__main__':main()
