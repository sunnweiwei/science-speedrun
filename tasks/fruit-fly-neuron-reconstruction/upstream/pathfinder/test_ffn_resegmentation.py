import unittest
import numpy as np
from ffn_resegmentation import grow, resegment_pair


class ResegmentationExecutionTest(unittest.TestCase):
    def test_call_cap_cannot_claim_complete(self):
        image = np.zeros((101,)*3, np.uint8)
        def fill(raw, seed):
            return np.full_like(seed, 4.)
        mask, report = grow(image, [50]*3, fill, max_calls=1)
        self.assertFalse(report['complete'])
        self.assertEqual(report['calls'], 1)
        self.assertEqual(int(mask.sum()), 33**3)

    def test_exact_edt_preserves_seeds_and_complete_resegmentation(self):
        from unittest.mock import patch
        from scipy.ndimage import distance_transform_edt
        image=np.zeros((101,)*3,np.uint8);base=np.zeros_like(image,dtype=np.uint32)
        base[40:50,40:50,40:50]=1;base[52:62,40:50,40:50]=2
        def reject(raw,seed):return np.full_like(seed,-4.)
        with patch('ffn_resegmentation.fragment_distance',side_effect=distance_transform_edt):
            original=resegment_pair(image,base,[1,2],reject,retries=3)
        actual=resegment_pair(image,base,[1,2],reject,retries=3)
        self.assertEqual(original,actual)

    def test_model_does_not_connect_two_original_fragments(self):
        image = np.zeros((101,)*3, np.uint8)
        base = np.zeros_like(image, dtype=np.uint32)
        base[40:50,40:50,40:50] = 1
        base[52:62,40:50,40:50] = 2
        def reject(raw, seed):
            return np.full_like(seed, -4.)
        report = resegment_pair(image, base, [1,2], reject, retries=2)
        self.assertTrue(report['complete'])
        self.assertFalse(report['accepted'])
        self.assertEqual(report['iou'], 0.)
        self.assertEqual(len(report['attempts']), 2)


if __name__ == '__main__':
    unittest.main()
