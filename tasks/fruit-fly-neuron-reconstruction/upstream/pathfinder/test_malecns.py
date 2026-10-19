import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from fetch_malecns import geometry, pinned_metadata, region_plan
from build_skeletons import convert_skeletons
from topology import sample

class MaleCNSTests(unittest.TestCase):
    def test_spatial_plan_resolves_frozen_splits_without_pilot_weights(self):
        plan=json.loads((Path(__file__).parent/'malecns-spatial-comparison.json').read_text())
        for row in plan['regions']:
            resolved=region_plan(plan,row['split'])
            self.assertEqual(resolved['center_xyz_16nm'],row['center_xyz_16nm'])
            self.assertEqual(resolved['region_id'],row['region_id'])
            self.assertEqual(resolved['expected_image_manifest_sha256'],row['image_manifest_sha256'])
            self.assertEqual(resolved['expected_core_labels_sha256'],row['core_labels_sha256'])
            self.assertEqual(resolved['em_url']+'attributes.json',plan['metadata']['em_s1']['url'])
            self.assertEqual(geometry(resolved)[1],1794)
            self.assertNotIn('model_package_manifest_sha256',resolved)
        with self.assertRaises(ValueError):region_plan(plan)
        with self.assertRaises(ValueError):region_plan(dict(plan,regions=plan['regions']*2),'test')
        pilot=json.loads((Path(__file__).parent/'malecns-evaluation.json').read_text())
        self.assertEqual(region_plan(pilot),pilot)
        with self.assertRaises(ValueError):region_plan(pilot,'train')

    def test_scored_core_has_full_context_after_ffn_crop(self):
        plan=json.loads((Path(__file__).parent/'malecns-evaluation.json').read_text())
        start,size,halo=geometry(plan)
        self.assertEqual(size,1538)
        np.testing.assert_array_equal(start,[23574,12988,11591])
        self.assertEqual((halo-16)*16,10000)
        self.assertEqual((size-16-(halo+256))*16,10000)
        for center in ([10,10,10],[1.5,2,3]):
            with self.assertRaises(ValueError): geometry(dict(plan,center_xyz_16nm=center))

    def test_larger_core_retains_context_and_strict_size_contract(self):
        plan=dict(center_xyz_16nm=[2000,3000,4000],core_size_voxels=512)
        start,size,halo=geometry(plan)
        self.assertEqual(size,1794)
        np.testing.assert_array_equal(start+halo,[1744,2744,3744])
        self.assertEqual((size-16-(halo+512))*16,10000)
        for bad in (True,256.0,0,-1,257,1024):
            with self.assertRaises(ValueError):geometry(dict(plan,core_size_voxels=bad))

    def test_frozen_regions_have_disjoint_contexts_and_fit_source(self):
        plan=json.loads((Path(__file__).parent/'malecns-multiregion.json').read_text())
        source=plan['source']; boxes=[]
        for region in [source]+plan['regions']:
            start,size,_=geometry(region)
            self.assertTrue((start+size <= [47044,39158,67288]).all())
            for lo,hi in boxes:
                self.assertTrue(((start+size<=lo)|(hi<=start)).any())
            boxes.append((start,start+size))
        self.assertEqual(len(plan['regions']),3)
        self.assertEqual(plan['nested_scale_voxels'],[256,512])

    def test_reference_voxel_centers_sample_correct_global_prediction(self):
        start=np.array([1000,2000,3000]); core_start=start+641
        gt=np.zeros((3,4,5),dtype=np.uint64); gt[1,2,3:5]=660718757
        skeleton=SimpleNamespace(vertices=np.array([[16,32,48],[16,32,64]]),edges=np.array([[0,1]]))
        truth,covered=convert_skeletons({660718757:skeleton},gt,core_start[::-1]*16,[16]*3)
        np.testing.assert_array_equal(truth['nodes'][0]['position_nm'],core_start[::-1]*16+[24,40,56])
        ids=sample(gt,[n['position_nm'] for n in truth['nodes']],core_start[::-1]*16,[16]*3)
        np.testing.assert_array_equal(ids,[660718757,660718757])
        self.assertEqual(covered,{660718757})

    def test_changed_public_metadata_rejected(self):
        content=b'{"dimensions": [1,2,3]}'
        entry=dict(url='https://example.invalid/attributes.json',sha256=hashlib.sha256(content).hexdigest())
        with patch('urllib.request.urlopen',return_value=io.BytesIO(content)):
            self.assertEqual(pinned_metadata(entry)['dimensions'],[1,2,3])
        with patch('urllib.request.urlopen',return_value=io.BytesIO(b'{}')):
            with self.assertRaises(ValueError): pinned_metadata(entry)

if __name__=='__main__':unittest.main()
