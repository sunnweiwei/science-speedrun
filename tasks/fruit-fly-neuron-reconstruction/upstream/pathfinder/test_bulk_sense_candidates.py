import unittest
import numpy as np
from compact_index import DTYPE
from index_storage import TableRecords,TableMapping
from sense import candidates

class BulkCandidatesTests(unittest.TestCase):
    def test_bulk_and_scalar_outputs_match_at_probability_and_overlap_boundaries(self):
        rng=np.random.default_rng(23)
        for threshold in (0.,.5,1.):
            for minimum in (1,30,500):
                labels=rng.integers(0,100,size=(17,19,21),dtype=np.uint32)
                table=np.zeros(100,dtype=DTYPE)
                ids,counts=np.unique(labels,return_counts=True)
                table['size'][ids]=counts+rng.integers(0,200,len(ids));table['size'][0]=0
                sizes={int(i):int(table['size'][i]) for i in ids if i}
                mapping=TableMapping(TableRecords(table,labels.shape),sizes=True)
                for probs in (np.zeros(labels.shape),np.ones(labels.shape),rng.choice([0.,.5,np.nextafter(.5,1.),1.],size=labels.shape)):
                    self.assertEqual(candidates(probs,labels,sizes,1,minimum,threshold),candidates(probs,labels,mapping,1,minimum,threshold))
    def test_missing_and_undersized_labels_still_rejected(self):
        labels=np.array([[[1,2,2]]],dtype=np.uint32);table=np.zeros(3,dtype=DTYPE);table['size'][1]=1
        for size in (0,1):
            table['size'][2]=size
            with self.assertRaises(ValueError):candidates(np.zeros(labels.shape),labels,TableMapping(TableRecords(table,labels.shape),sizes=True),1)
    def test_large_size_scalar_fallback(self):
        labels=np.array([[[1,2]]],dtype=np.uint32);table=np.zeros(3,dtype=DTYPE);table['size'][1]=1;table['size'][2]=2**53+1
        self.assertEqual(candidates(np.ones(labels.shape),labels,{1:1,2:2**53+1},1,1,0),candidates(np.ones(labels.shape),labels,TableMapping(TableRecords(table,labels.shape),sizes=True),1,1,0))

if __name__=='__main__':unittest.main()
