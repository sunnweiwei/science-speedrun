import copy,hashlib,unittest
import numpy as np
from audit_sense_coverage import audit_coverage


class CoverageAuditTests(unittest.TestCase):
    def fixture(self):
        shape=[1300]*3;table=np.zeros(7,dtype=[('size','<u8'),('center','<u8')])
        centers={1:[624,650,650],2:[625]*3,3:[674]*3,4:[675,650,650],6:[650]*3}
        for sid,coord in centers.items():table[sid]=(1,np.ravel_multi_index(coord,shape))
        coverage=dict(eligible_instance_ids=[2,3,6],processed=[dict(instance_id=i,center_zyx=centers[i],geometric_shape_context_complete=True) for i in [2,3,6]],unprocessed_instance_ids=[],eligible_count=3,processed_count=3,skipped_geometric_context_instance_ids=[1,4])
        return table,shape,coverage

    def test_physical_boundaries_holes_and_chunk_independence(self):
        table,shape,c=self.fixture()
        rows=[audit_coverage(table,shape,c,chunk=n) for n in (1,2,7,100)]
        self.assertTrue(all(r==rows[0] for r in rows))
        self.assertEqual(rows[0]['eligible_sha256'],hashlib.sha256(np.array([2,3,6],dtype='<u8').tobytes()).hexdigest())
        self.assertEqual(rows[0]['total_instances'],5)
        c['skipped_geometric_context_instance_ids']=dict(storage='ordered-excluded-id-digest/v1',count=2,sha256=rows[0]['excluded_sha256'],shape=shape,radius_nm=10000,voxel_size_nm=[16]*3,encoding='ascending little-endian uint64')
        self.assertEqual(audit_coverage(table,shape,c),rows[0])

    def test_missing_duplicate_wrong_center_and_excluded_digest_rejected(self):
        table,shape,c=self.fixture()
        mutations=[lambda x:x['eligible_instance_ids'].__setitem__(1,2),lambda x:x['processed'].pop(),lambda x:x['processed'][0].update(center_zyx=[624]*3),lambda x:x.update(unprocessed_instance_ids=[6]),lambda x:x.update(skipped_geometric_context_instance_ids=[1])]
        for mutate in mutations:
            bad=copy.deepcopy(c);mutate(bad)
            with self.assertRaises(ValueError):audit_coverage(table,shape,bad,2)

if __name__=='__main__':unittest.main()
