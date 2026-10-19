import unittest
import numpy as np
from run_ffn_agglomeration import execute


class BatchedAgglomerationTest(unittest.TestCase):
    def test_independent_candidates_complete_with_rejection(self):
        image=np.zeros((101,)*3,np.uint8);base=np.zeros_like(image,dtype=np.uint32)
        base[40:50,40:50,40:50]=1;base[52:62,40:50,40:50]=2
        rows=[dict(pair=[1,2],center_zyx=[50]*3) for _ in range(2)]
        def reject(raw,seed):return np.full_like(seed,-4.)
        result=execute(image,base,rows,reject,[0]*3,workers=2,seconds=30)
        self.assertTrue(result['complete']);self.assertEqual(result['accepted_pairs'],0)
        self.assertGreater(result['model_windows'],0)

    def test_sink_keeps_rows_out_of_summary_without_changing_scores(self):
        image=np.zeros((101,)*3,np.uint8);base=np.zeros_like(image,dtype=np.uint32)
        base[40:50,40:50,40:50]=1;base[52:62,40:50,40:50]=2
        rows=[dict(pair=[1,2],center_zyx=[50]*3) for _ in range(3)]
        def reject(raw,seed):return np.full_like(seed,-4.)
        old=execute(image,base,rows,reject,[0]*3,workers=2,seconds=30)
        collected=[];new=execute(image,base,rows,reject,[0]*3,workers=2,seconds=30,result_sink=collected.append)
        self.assertEqual(new['results'],[]);self.assertEqual(collected,old['results'])
        for key in ('complete','completed_candidates','requested_candidates','accepted_pairs'):
            self.assertEqual(old[key],new[key])

    def test_predictor_error_drains_worker_requests(self):
        image=np.zeros((101,)*3,np.uint8);base=np.zeros_like(image,dtype=np.uint32)
        base[40:50,40:50,40:50]=1;base[52:62,40:50,40:50]=2
        rows=[dict(pair=[1,2],center_zyx=[50]*3) for _ in range(2)]
        def invalid(raw,seed):return np.zeros((1,))
        with self.assertRaises(RuntimeError):
            execute(image,base,rows,invalid,[0]*3,workers=2,seconds=10)

    def test_no_context_is_explicitly_incomplete(self):
        image=np.zeros((101,)*3,np.uint8);base=np.zeros_like(image,dtype=np.uint32)
        result=execute(image,base,[dict(pair=[1,2],center_zyx=[1]*3)],None,[0]*3,workers=1,seconds=10)
        self.assertFalse(result['complete']);self.assertEqual(result['gpu_batches'],0)


if __name__=='__main__':unittest.main()
