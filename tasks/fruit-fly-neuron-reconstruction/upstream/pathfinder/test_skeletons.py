import unittest
from types import SimpleNamespace
import numpy as np

from build_skeletons import convert_skeletons, compatible_osteoid_merge
from topology import sample


class SkeletonTests(unittest.TestCase):
    def test_osteoid_merge_compat_preserves_disconnected_paths(self):
        try:
            from osteoid import Skeleton
        except ImportError:
            self.skipTest('Optional skeletonization dependencies not installed')
        a = Skeleton.from_path(np.array([[0,0,0], [1,0,0]], dtype=np.float32))
        b = Skeleton.from_path(np.array([[0,2,0], [1,2,0]], dtype=np.float32))
        original = Skeleton.__dict__['simple_merge']
        with compatible_osteoid_merge():
            merged = Skeleton.simple_merge([a, b])
        np.testing.assert_array_equal(merged.edges, [[0,1], [2,3]])
        self.assertEqual(a.edges.dtype, np.uint32)
        self.assertIs(Skeleton.__dict__['simple_merge'], original)

    def test_physical_centers_roundtrip_anisotropic_and_nonzero_origin(self):
        labels = np.zeros((4, 5, 6), dtype=np.uint16)
        labels[1:3, 2, 3] = 7
        skel = SimpleNamespace(vertices=np.array([[40, 16, 24], [80, 16, 24]]),
                               edges=np.array([[0, 1]]))
        truth, represented = convert_skeletons({7: skel}, labels, [100, 200, 300], [40, 8, 8])
        self.assertEqual(represented, {7})
        self.assertEqual(truth['nodes'][0]['position_nm'], [160, 220, 328])
        sampled = sample(labels, [n['position_nm'] for n in truth['nodes']], [100, 200, 300], [40, 8, 8])
        np.testing.assert_array_equal(sampled, [7, 7])
        self.assertEqual(truth['edges'], [[0, 1]])

    def test_unknown_voxel_or_shortcut_is_rejected(self):
        labels = np.ones((4, 4, 4), dtype=np.uint16)
        skel = SimpleNamespace(vertices=np.array([[0, 0, 0], [2, 0, 0]]), edges=np.array([[0, 1]]))
        with self.assertRaisesRegex(ValueError, 'Nonadjacent'):
            convert_skeletons({1: skel}, labels, [0]*3, [1]*3)
        labels[0, 0, 0] = 0
        with self.assertRaisesRegex(ValueError, 'valid GT'):
            convert_skeletons({1: skel}, labels, [0]*3, [1]*3)


if __name__ == '__main__':
    unittest.main()
