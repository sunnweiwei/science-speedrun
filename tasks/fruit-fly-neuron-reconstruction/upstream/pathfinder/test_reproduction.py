import json
from pathlib import Path
import unittest
import numpy as np
import torch
from sense import Sense, prompt_mask, candidates
from search import filter_graph
from compare_report import compare

class ReproductionTests(unittest.TestCase):
    def test_prompt_and_full_segment_overlap(self):
        labels=np.ones((8,8,8),dtype=np.uint64);labels[:2]=2
        prompt=prompt_mask(labels,1,radius=2)
        self.assertAlmostEqual(float(prompt[4,4,4]),.95)
        self.assertAlmostEqual(float(prompt[0,0,0]),.5)
        probabilities=np.ones(labels.shape)
        self.assertEqual(candidates(probabilities,labels,{1:384,2:1000},1),[])
        rows=candidates(probabilities,labels,{1:384,2:128},1,min_size=100)
        self.assertEqual(rows[0]['target'],2);self.assertEqual(rows[0]['overlap'],1)
        with self.assertRaises(ValueError):candidates(probabilities,labels,{1:384,2:10},1)

    def test_triplet_can_reject_valid_pairs(self):
        edges=[dict(source=1,target=2,sense=.9,shape=.9),dict(source=2,target=3,sense=.8,shape=.8)]
        result=filter_graph([1,2,3],edges,[([1,2,3],.1)])
        self.assertEqual(result['components'],[[1,2],[3]])
        self.assertEqual(filter_graph([1,2,3],edges,[])['components'],[[1,2,3]])
        self.assertFalse(result['author_parity'])
        with self.assertRaises(ValueError):filter_graph([1,2,3],edges+edges,[])

    def test_reference_is_not_local_gap(self):
        reference=json.loads(Path(__file__).with_name('paper_reference.json').read_text())
        r=compare(reference,dict(nerl=.99,split_errors=0,merge_errors=0,paper_protocol_verified=True))
        self.assertFalse(r['directly_comparable']);self.assertIsNone(r['numeric_gap'])
        self.assertEqual(r['reference']['nerl'],.942)

if __name__=='__main__':unittest.main()
