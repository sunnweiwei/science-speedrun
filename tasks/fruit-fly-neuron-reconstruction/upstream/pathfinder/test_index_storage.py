import hashlib,json,unittest
from pathlib import Path
import numpy as np
from compact_index import build_table,DTYPE
from index_storage import TableRecords,load_index,TableMapping,validate_table,select_table_centers
from common import sha256
from test_fs_support import TemporaryDirectory
from infer_continuous_sense import select_centers

class StorageTests(unittest.TestCase):
    def test_large_exclusion_digest_matches_explicit_legacy_ids(self):
        shape=(1468,)*3;table=np.zeros(10020,dtype=DTYPE)
        table['size'][1:]=1
        table['center'][1:]=np.ravel_multi_index((100,100,100),shape)
        table['center'][[7,101,9999]]=np.ravel_multi_index((700,700,700),shape)
        rows=TableRecords(table,shape);legacy=list(rows)
        for rank in range(3):
            chosen,digest,other,eligible=select_table_centers(rows,shape,rank,3)
            expected=select_centers(legacy,shape,rank,3)
            self.assertEqual((list(chosen),other,eligible),(expected[0],expected[2],expected[3]))
            self.assertEqual(digest['count'],len(expected[1]))
            self.assertEqual(digest['sha256'],hashlib.sha256(np.asarray(expected[1],dtype='<u8').tobytes()).hexdigest())

    def test_nonempty_shard_selection_preserves_ascending_eligible_ordinal(self):
        shape=(1468,)*3;table=np.zeros(18,dtype=DTYPE)
        for sid,c in [(2,624),(4,625),(9,700),(13,842),(17,843)]:
            table['size'][sid]=1;table['lo'][sid]=c;table['hi'][sid]=c+1
            table['center'][sid]=np.ravel_multi_index((c,c,c),shape)
        rows=TableRecords(table,shape);legacy=list(rows)
        for shard in range(3):
            actual=select_table_centers(rows,shape,shard,3)
            self.assertEqual((list(actual[0]),*actual[1:]),select_centers(legacy,shape,shard,3))
        self.assertEqual(select_table_centers(rows,shape,0,3)[3],[4,9,13])

    def test_roundtrip_selection_mapping_and_tamper(self):
        with TemporaryDirectory() as directory:
            p=Path(directory);a=np.arange(120,dtype=np.uint32).reshape(5,6,4)%7
            table=build_table(a,max_label=10,max_instances=10);np.save(p/'table.npy',table)
            index=dict(shape=list(a.shape),storage='numeric-table-v1',table=dict(file='table.npy',sha256=sha256(p/'table.npy'),bytes=(p/'table.npy').stat().st_size,labels=len(table),instances=6))
            (p/'index.json').write_text(json.dumps(index));v=load_index(p/'index.json');rows=v['records'];validate_table(rows,a)
            legacy=list(rows)
            self.assertEqual(dict(TableMapping(rows)),{r['id']:r for r in legacy})
            self.assertEqual(dict(TableMapping(rows,sizes=True)),{r['id']:r['size'] for r in legacy})
            for shard in range(3):
                expected=select_centers(legacy,a.shape,shard,3);actual=select_table_centers(rows,a.shape,shard,3)
                self.assertEqual((list(actual[0]),*actual[1:]),expected)
            rows.table._mmap.close()
            with (p/'table.npy').open('ab') as f:f.write(b'x')
            with self.assertRaisesRegex(ValueError,'changed'):load_index(p/'index.json')

if __name__=='__main__':unittest.main()
