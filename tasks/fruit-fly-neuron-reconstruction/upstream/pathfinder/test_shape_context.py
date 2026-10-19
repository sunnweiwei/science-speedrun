import unittest
from unittest.mock import patch
import numpy as np
from shape_context import indexed_mesh,context_coverage,MeshCache
from infer_shape import mesh_segment
from instance_index import build_index
from shape_data import sample_surface


class ContextTests(unittest.TestCase):
    def test_marginal_centroids_preserve_exact_meshes_without_coordinate_expansion(self):
        rng=np.random.default_rng(8)
        for seg in [np.ones((12,13,14),np.uint32),rng.integers(0,4,(16,17,18),dtype=np.uint32)]:
            for row in build_index(seg,3):
                expected=mesh_segment(seg,row['id'],np.array([.08,.016,.016]))
                with patch('shape_context.np.argwhere',side_effect=AssertionError('Full coordinates must not be expanded')):
                    actual=indexed_mesh(seg,row['id'],np.array([.08,.016,.016]),row)
                for a,b in zip(expected,actual):np.testing.assert_array_equal(a,b)

    def test_bounded_cache_preserves_full_8192_point_inputs(self):
        seg=np.zeros((24,24,24),np.uint32)
        for sid in range(1,5):seg[2:21,2+4*sid:4+4*sid,5:18]=sid
        rows={r['id']:r for r in build_index(seg)};spacing=np.array([.016]*3)
        expected={sid:mesh_segment(seg,sid,spacing) for sid in rows}
        largest=max(sum(x.nbytes for x in mesh) for mesh in expected.values())
        motifs=[(1,2),(2,3),(3,4),(1,2,3),(2,3,4),(1,2,3,4)]
        def points(meshes,seed):
            vs=[];fs=[];offset=0
            for v,f,_ in meshes:vs.append(v);fs.append(f+offset);offset+=len(v)
            return sample_surface(np.concatenate(vs),np.concatenate(fs),np.array([.2]*3),np.random.default_rng(seed))
        for limit in (0,largest,largest*2,largest*4):
            cache=MeshCache(limit)
            for i,nodes in enumerate(motifs):
                actual=[cache.get(sid,lambda key:indexed_mesh(seg,key,spacing,rows[key])) for sid in nodes]
                np.testing.assert_array_equal(points(actual,i),points([expected[sid] for sid in nodes],i))
                self.assertLessEqual(cache.statistics()['retained_bytes'],limit)
            self.assertLessEqual(cache.statistics()['peak_retained_bytes'],limit)
            self.assertEqual(cache.statistics()['largest_mesh_bytes'],largest)
        with self.assertRaises(ValueError):MeshCache(-1)

    def test_indexed_mesh_matches_full_scan(self):
        seg=np.zeros((20,20,20),np.uint32);seg[3:8,7:10,8:12]=100
        row=dict(id=100,size=60,bbox_zyx=[[3,8],[7,10],[8,12]])
        for a,b in zip(mesh_segment(seg,100,np.array([.08,.016,.016])),indexed_mesh(seg,100,np.array([.08,.016,.016]),row)):
            np.testing.assert_allclose(a,b)
        row['size']=61
        with self.assertRaises(ValueError):indexed_mesh(seg,100,np.ones(3),row)
    def test_missing_prediction_and_image_observation_are_separate(self):
        meta=dict(origin_nm=[1000]*3,raw_origin_nm=[0]*3,shape=[30]*3)
        coverage=dict(tiles=[dict(status='unstarted',start=[29]*3,stop=[31]*3)],image_observation=dict(unknown_output_z=[15]))
        r=context_coverage([15]*3,np.ones(3),meta,coverage)
        self.assertTrue(r['prediction_context_complete']);self.assertLess(r['image_observed_sphere_fraction'],1)
        coverage['tiles'][0].update(start=[15]*3,stop=[17]*3)
        self.assertFalse(context_coverage([15]*3,np.ones(3),meta,coverage)['prediction_context_complete'])

if __name__=='__main__':unittest.main()
