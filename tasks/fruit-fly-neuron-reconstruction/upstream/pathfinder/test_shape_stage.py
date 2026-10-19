import unittest
import numpy as np
from shape_data import sample_surface
from infer_shape import connected_motifs,complete_context,canonical_edges


class ShapeStageTests(unittest.TestCase):
    def test_surface_area_weighting_and_radius(self):
        # Two disjoint triangles, area ratio1:4, both inside radius10.
        v=np.array([[0,0,0],[1,0,0],[0,1,0],[3,0,0],[5,0,0],[3,2,0]],float)
        p=sample_surface(v,[[0,1,2],[3,4,5]],[0,0,0],np.random.default_rng(4))
        self.assertEqual(p.shape,(8192,3));self.assertTrue((np.linalg.norm(p,axis=1)<=1).all())
        self.assertAlmostEqual(float((p[:,0]>.2).mean()),.8,delta=.02)
    def test_reject_empty_sphere(self):
        with self.assertRaises(ValueError):sample_surface([[50,0,0],[51,0,0],[50,1,0]],[[0,1,2]],[0,0,0],np.random.default_rng(1))
    def test_connected_motifs_and_cap(self):
        edges=[dict(source=1,target=2),dict(source=2,target=3),dict(source=3,target=4),dict(source=9,target=10)]
        motifs,truncated=connected_motifs(edges)
        self.assertIn((1,2,3,4),motifs);self.assertNotIn((1,3),motifs);self.assertNotIn((1,2,9),motifs)
        self.assertFalse(truncated);self.assertEqual(len(connected_motifs(edges,limit=2)[0]),2)
        self.assertTrue(connected_motifs(edges,limit=2)[1])
    def test_reciprocal_edges(self):
        rows=canonical_edges([dict(source=2,target=1,sense=.3),dict(source=1,target=2,sense=.9)])
        self.assertEqual(rows,[dict(source=1,target=2,sense=.9)])

    def test_full_context(self):
        self.assertFalse(complete_context([112,192,192],[225,384,384],np.array([.016]*3)))
        self.assertTrue(complete_context([199,199,199],[400]*3,np.array([.08]*3)))

if __name__=='__main__':unittest.main()
