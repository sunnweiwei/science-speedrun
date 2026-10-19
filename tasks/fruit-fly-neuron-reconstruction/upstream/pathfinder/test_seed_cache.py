import json,os
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from seed_cache import get_seeds
from ffn_seeds import policy_peaks
from test_fs_support import TemporaryDirectory

@unittest.skipUnless(os.environ.get('PATHFINDER_TEST_XTEMP'),'Xtemp required')
class SeedCacheTests(unittest.TestCase):
    def test_exact_hit_input_invalidation_and_corruption(self):
        with TemporaryDirectory(dir=os.environ['PATHFINDER_TEST_XTEMP']) as folder:
            root=Path(folder);image=np.random.RandomState(7).randint(0,256,(48,)*3).astype(np.uint8)
            expected=policy_peaks(image)
            first,a=get_seeds(image,root,root/'cache',dict(raw_sha256='fixture',start=[16]*3))
            self.assertFalse(a['hit']);np.testing.assert_array_equal(first,expected)
            with patch('seed_cache.policy_peaks',side_effect=AssertionError('Recomputed cache hit')):
                second,b=get_seeds(image,root,root/'cache',dict(raw_sha256='fixture',start=[16]*3))
            self.assertTrue(b['hit']);np.testing.assert_array_equal(second[::-1],expected[::-1])
            image=image.copy();image[0,0,0]^=1
            third,c=get_seeds(image,root,root/'cache',dict(raw_sha256='fixture',start=[16]*3))
            self.assertNotEqual(a['key'],c['key']);np.testing.assert_array_equal(third,policy_peaks(image))
            receipt=Path(c['receipt']);record=json.loads(receipt.read_text());record['sha256']='0'*64;receipt.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'integrity'):get_seeds(image,root,root/'cache',dict(raw_sha256='fixture',start=[16]*3))

    def test_cached_growth_preserves_requests_and_full_labels(self):
        from batched_ffn import grow_tile
        with TemporaryDirectory(dir=os.environ['PATHFINDER_TEST_XTEMP']) as folder:
            root=Path(folder);image=np.random.RandomState(7).randint(0,256,(48,)*3).astype(np.uint8)
            def provider(context,start,stop):
                return get_seeds(context,root,root/'cache',dict(start=start,stop=stop))[0]
            for order in ('ascending','descending'):
                runs=[]
                for source in (None,provider,provider):
                    g=grow_tile(image,[16]*3,[32]*3,seed_order=order,seed_provider=source);requests=[]
                    try:
                        request=next(g)
                        while True:
                            requests.append(tuple(v.copy() for v in request))
                            prediction=np.full((33,)*3,-4.,np.float32)
                            prediction[13:20,13:20,13:20]=3.1
                            request=g.send(prediction)
                    except StopIteration as end:result=end.value
                    self.assertTrue((result[0]>0).any())
                    runs.append((requests,result))
                for requests,result in runs[1:]:
                    self.assertEqual(len(requests),len(runs[0][0]))
                    for actual,expected in zip(requests,runs[0][0]):
                        for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
                    np.testing.assert_array_equal(result[0],runs[0][1][0]);self.assertEqual(result[1],runs[0][1][1])

if __name__=='__main__':unittest.main()
