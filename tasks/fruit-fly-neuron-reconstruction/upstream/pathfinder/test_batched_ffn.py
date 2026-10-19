import unittest
from unittest.mock import patch
import numpy as np
from batched_ffn import schedule,grid_tiles,grow_tile,tile_id_stride,global_tile_labels,window_finished


class BatchedFFNTests(unittest.TestCase):
    @patch('batched_ffn.policy_peaks',new=lambda image:np.asarray([np.asarray(image.shape)-17]))
    def test_progress_preserves_requests_predictions_and_completion(self):
        image=np.full((64,64,64),200,np.uint8);tiles=list(grid_tiles([16]*3,[48]*3,16));runs=[]
        for enabled in (False,True):
            requests=[];outputs={};progress=[]
            def predict(x,q):requests.append((x.copy(),q.copy()));return np.full_like(q,3.1)
            report=schedule(image,tiles,predict,batch=3,seconds=10,clock=lambda:0,
                on_result=lambda t,s,r:outputs.update({t['index']:(s.copy(),r)}),on_progress=progress.append if enabled else None)
            runs.append((requests,outputs,report,progress))
        self.assertEqual(runs[0][2],runs[1][2])
        for old,new in zip(runs[0][0],runs[1][0]):
            for a,b in zip(old,new):np.testing.assert_array_equal(a,b)
        self.assertEqual(len(runs[0][0]),len(runs[1][0]))
        for key,(a,r) in runs[0][1].items():
            b,s=runs[1][1][key];np.testing.assert_array_equal(a,b);self.assertEqual(r,s)
        progress=runs[1][3]
        self.assertEqual(len(progress),runs[1][2]['model_batches'])
        self.assertEqual(progress[-1]['completed_windows'],len(tiles))
        self.assertEqual(progress[-1]['model_windows'],runs[1][2]['model_windows'])
        self.assertEqual(progress[-1]['active_window_ids'],[])
        self.assertTrue(any(p['active_window_ids'] for p in progress[:-1]))

    def test_successive_objects_do_not_share_logits_or_initialization(self):
        image=np.zeros((112,112,112),np.uint8)
        seeds=np.array([[32,32,32],[48,48,48],[33,33,33]])
        with patch('batched_ffn.policy_peaks',return_value=seeds):
            gen=grow_tile(image,[16]*3,[96]*3)
            _,first=next(gen)
            prediction=np.full_like(first,-3.2)
            prediction[13:20,13:20,13:20]=3.2
            _,second=gen.send(prediction)
            # The second window overlaps the first write. Prior negative logits
            # and initialized flags must both be cleared for the new object.
            np.testing.assert_array_equal(first,second)
            with self.assertRaises(StopIteration) as done:gen.send(prediction)
            segmentation,stats=done.exception.value
            self.assertEqual(stats['objects'],2)
            self.assertEqual(stats['calls'],2)
            self.assertEqual(int((segmentation==1).sum()),343)
            self.assertEqual(int((segmentation==2).sum()),343)
            self.assertFalse(stats['budget_exhausted'])

    def test_seed_order_changes_first_request_without_changing_seed_set(self):
        image=np.arange(64**3,dtype=np.uint8).reshape(64,64,64)
        seeds=np.array([[20,20,20],[40,40,40]])
        with patch('batched_ffn.policy_peaks',return_value=seeds):
            for order,center in [('ascending',20),('descending',40)]:
                generator=grow_tile(image,[16]*3,[48]*3,seed_order=order)
                raw,_=next(generator)
                np.testing.assert_array_equal(raw,image[center-16:center+17,center-16:center+17,center-16:center+17].astype(np.float32)/255)
                generator.close()
            with self.assertRaises(ValueError):next(grow_tile(image,[16]*3,[48]*3,seed_order='unknown'))

    def test_tight_id_bound_preserves_zero_and_global_uniqueness(self):
        self.assertEqual(tile_id_stride(64),26215)
        tile=np.array([0,1,26214],dtype=np.uint32).reshape(1,1,3)
        first=global_tile_labels(tile,0,21952,64)
        last=global_tile_labels(tile,21951,21952,64)
        self.assertEqual(last.dtype,np.uint32)
        self.assertEqual(int(last[0,0,0]),0)
        np.testing.assert_array_equal(last[last>0]-21951*26215,tile[tile>0])
        self.assertFalse(set(first[first>0])&set(last[last>0]))
        huge=global_tile_labels(tile,200000,200001,64)
        self.assertEqual(huge.dtype,np.uint64)
        with self.assertRaises(ValueError):global_tile_labels(np.full((1,1,1),26215,dtype=np.uint32),0,1,64)

    @patch('batched_ffn.policy_peaks',new=lambda image:np.asarray([np.asarray(image.shape)-17]))
    def test_batch_invariance_and_unique_tile_coverage(self):
        image=np.full((64,64,64),200,np.uint8)
        tiles=list(grid_tiles([16]*3,[48]*3,16))
        def predict(x,q):return np.full_like(q,3.1)
        outputs=[]
        for batch in [1,3,8]:
            results={}
            report=schedule(image,tiles,predict,batch=batch,seconds=10,on_result=lambda t,s,r:results.update({t['index']:(s,r)}))
            self.assertEqual(len(results),8);self.assertFalse(report['time_exhausted'])
            self.assertTrue(all((seg>0).all() for seg,_ in results.values()))
            outputs.append((results,report))
        for results,report in outputs[1:]:
            self.assertEqual(report['model_windows'],outputs[0][1]['model_windows'])
            self.assertLess(report['model_batches'],outputs[0][1]['model_batches'])
            for key,(seg,stats) in results.items():
                np.testing.assert_array_equal(seg,outputs[0][0][key][0]);self.assertEqual(stats,outputs[0][0][key][1])
    @patch('batched_ffn.policy_peaks',new=lambda image:np.asarray([np.asarray(image.shape)-17]))
    def test_bounded_calls_and_partial_result(self):
        image=np.full((64,64,64),200,np.uint8)
        results=[]
        report=schedule(image,list(grid_tiles([16]*3,[48]*3,32)),lambda x,q:np.full_like(q,3.1),
            max_calls=1,on_result=lambda t,s,r:results.append((s,r)))
        self.assertEqual(report['model_windows'],1);self.assertTrue(results[0][1]['budget_exhausted'])
        self.assertTrue((results[0][0]>0).any());self.assertTrue((results[0][0]==0).any())
        self.assertTrue(results[0][1]['call_limit_reached']);self.assertTrue(window_finished(results[0][1]))
    @patch('batched_ffn.policy_peaks',new=lambda image:np.asarray([np.asarray(image.shape)-17]))
    def test_explicit_stop_retains_existing_predictions(self):
        image=np.full((64,64,64),200,np.uint8);gen=grow_tile(image,[16]*3,[48]*3)
        x,q=next(gen);gen.send(np.full_like(q,3.1))
        with self.assertRaises(StopIteration) as result:gen.send(None)
        seg,stats=result.exception.value
        self.assertEqual(stats['calls'],1);self.assertTrue(stats['budget_exhausted']);self.assertTrue((seg>0).any())
        # A time-limit stop is not a protocol call cap; such windows stay unfinished.
        self.assertFalse(stats['call_limit_reached']);self.assertFalse(window_finished(stats))
    def test_blank_volume_is_completed_without_model(self):
        image=np.zeros((48,48,48),np.uint8)
        def invalid_predict(*args):raise AssertionError('No seeds, no model requests')
        report=schedule(image,list(grid_tiles([16]*3,[32]*3,16)),invalid_predict)
        self.assertEqual(report['model_windows'],0);self.assertEqual(len(report['tiles']),1)
        self.assertFalse(report['tiles'][0]['budget_exhausted']);self.assertEqual(report['tiles'][0]['omitted_voxel_fraction'],1)

if __name__=='__main__':unittest.main()
