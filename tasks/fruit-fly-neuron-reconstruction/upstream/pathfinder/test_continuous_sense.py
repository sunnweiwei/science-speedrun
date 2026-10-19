import copy
import unittest
import numpy as np
from infer_continuous_sense import infer,validate_inputs,select_centers


class VirtualVolume:
    """Large coordinate extent, materialize only requested crops for CPU tests."""
    def __init__(self,shape,dtype,boxes=None,value=0):
        self.shape=tuple(shape);self.ndim=3;self.dtype=np.dtype(dtype);self.boxes=boxes or [];self.value=value
    def __getitem__(self,key):
        if all(isinstance(k,(int,np.integer)) for k in key):
            for sid,lo,hi in self.boxes:
                if all(a<=k<b for k,a,b in zip(key,lo,hi)):return self.dtype.type(sid)
            return self.dtype.type(self.value)
        lo=np.array([s.start for s in key]);hi=np.array([s.stop for s in key])
        array=np.full(tuple(hi-lo),self.value,self.dtype)
        for sid,a,b in self.boxes:
            low=np.maximum(lo,a);high=np.minimum(hi,b)
            if (high>low).all():array[tuple(slice(int(x-l),int(y-l)) for x,y,l in zip(low,high,lo))]=sid
        return array


class ContinuousSenseTest(unittest.TestCase):
    def fixture(self):
        rows=[];boxes=[]
        for sid,c in [(1,700),(2,712),(3,100)]:
            lo=[c-5]*3;hi=[c+5]*3
            rows.append(dict(id=sid,size=1000,center_zyx=[c]*3,bbox_zyx=list(map(list,zip(lo,hi)))))
            boxes.append((sid,lo,hi))
        raw=VirtualVolume([1500]*3,np.uint8,value=128);seg=VirtualVolume([1468]*3,np.uint32,boxes)
        valid=np.ones(1500,bool);valid[720]=False
        image=dict(axes='zyx',shape=[1500]*3,voxel_size_nm=[16]*3,origin_nm=[3000,4384,4384],coordinate_convention='voxel_outer_corner',image_valid=dict(scope='known filler planes'))
        meta=dict(image);meta.update(shape=[1468]*3,origin_nm=[3256,4640,4640],raw_origin_nm=image['origin_nm'])
        assembled=dict(ground_truth_used=False,segmentation_sha256='a'*64,metadata=meta,all_tiles_complete=False)
        index=dict(schema='predicted-instance-index/v1',axes='zyx',shape=[1468]*3,records=rows,segmentation_sha256='a'*64,ground_truth_used=False)
        return raw,seg,valid,image,assembled,index

    def test_batch_geometry_canonical_edges_and_observation_gaps(self):
        args=self.fixture();batches=[]
        def predict(raw,prompt):
            batches.append(raw.shape)
            self.assertTrue((prompt[:,0,0,0]==.5).all())
            self.assertAlmostEqual(float(prompt[0,64,64,64]),.95,places=6)
            return np.ones(raw.shape,np.float32)
        result,audit=infer(*args,predict,max_calls=8,seconds=10)
        self.assertEqual(batches,[(2,128,128,128)])
        self.assertEqual(len(result['edges']),1)
        self.assertEqual((result['edges'][0]['source'],result['edges'][0]['target']),(1,2))
        self.assertEqual(result['edges'][0]['sense'],1.)
        self.assertEqual(audit['skipped_geometric_context_instance_ids'],[3])
        self.assertEqual(audit['processed'][0]['sense_input_missing_z_fraction'],1/128)
        self.assertGreater(audit['processed'][0]['shape_context_missing_z_fraction'],0)
        self.assertFalse(audit['all_tiles_complete']);self.assertFalse(audit['budget_exhausted'])

    def test_four_shards_disjoint_and_call_budget(self):
        args=self.fixture();chosen=[]
        for shard in range(4):
            rows,skipped,other,eligible=select_centers(args[-1]['records'],args[1].shape,shard,4)
            chosen.extend(r['id'] for r in rows)
        self.assertEqual(sorted(chosen),[1,2])
        _,audit=infer(*args,lambda a,b:np.ones(a.shape,np.float32),max_calls=1,seconds=10)
        self.assertEqual(audit['unprocessed_instance_ids'],[2]);self.assertTrue(audit['budget_exhausted'])

    def test_reject_hash_geometry_and_wrong_actual_center(self):
        args=self.fixture();args[-1]['segmentation_sha256']='b'*64
        with self.assertRaisesRegex(ValueError,'hash'):validate_inputs(*args)
        args=self.fixture();args[-2]['metadata']['origin_nm'][0]+=1
        with self.assertRaisesRegex(ValueError,'align'):validate_inputs(*args)
        args=self.fixture();args[-1]['records'][0]['center_zyx']=[705]*3
        args[-1]['records'][0]['bbox_zyx']=[[695,710]]*3
        with self.assertRaisesRegex(ValueError,'positive'):validate_inputs(*args)

    def test_empty_and_all_geometrically_skipped(self):
        args=self.fixture();args[-1]['records']=[]
        _,audit=infer(*args,lambda a,b:self.fail('no model call'),seconds=10)
        self.assertTrue(audit['empty_segmentation'])
        args=self.fixture();args[-1]['records']=args[-1]['records'][2:]
        _,audit=infer(*args,lambda a,b:self.fail('no model call'),seconds=10)
        self.assertFalse(audit['empty_segmentation']);self.assertEqual(audit['geometrically_eligible_representative_centers'],0)


if __name__=='__main__':unittest.main()
