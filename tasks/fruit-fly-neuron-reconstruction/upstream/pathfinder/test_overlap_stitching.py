import unittest
import numpy as np
from overlap_stitching import mutual_maximum_pairs,overlap_midplane,component_interface_plane


class OverlapTests(unittest.TestCase):
    def test_only_reciprocal_maximum_is_accepted(self):
        a=np.array([[1]*7+[2]*6],np.uint32)
        b=np.array([[8]*4+[9]*3+[9]*6],np.uint32)
        self.assertEqual(mutual_maximum_pairs(a,b)['pairs'],[
            dict(a=1,b=8,overlap_voxels=4),dict(a=2,b=9,overlap_voxels=6)])

    def test_background_majority_and_ties_do_not_connect_fragments(self):
        a=np.array([[1,1,1,1,2,2,2,2]],np.uint32)
        b=np.array([[0,0,0,8,8,8,9,9]],np.uint32)
        self.assertEqual(mutual_maximum_pairs(a,b)['pairs'],[])

    def test_high_uint64_ids_and_separate_namespaces(self):
        high=2**63+11
        a=np.array([[high,high,0]],np.uint64);b=np.array([[high,high,0]],np.uint64)
        self.assertEqual(mutual_maximum_pairs(a,b)['pairs'],[dict(a=high,b=high,overlap_voxels=2)])

    def test_planes_are_sampled_at_same_global_coordinate_on_each_axis(self):
        for axis in range(3):
            global_data=np.arange(10**3,dtype=np.uint32).reshape(10,10,10)
            sa=[slice(0,6)]*3;sb=sa.copy();sb[axis]=slice(2,8)
            a=global_data[tuple(sa)];b=global_data[tuple(sb)]
            oa=[0,0,0];ob=[0,0,0];ob[axis]=2
            pa,pb,meta=overlap_midplane(a,b,oa,ob,axis)
            np.testing.assert_array_equal(pa,pb);self.assertEqual(meta['coordinate_voxels'],3)
            pb2,pa2,meta2=overlap_midplane(b,a,ob,oa,axis)
            np.testing.assert_array_equal(pa,pa2);np.testing.assert_array_equal(pb,pb2)
            self.assertEqual(meta,meta2)

    def test_touching_but_not_overlapping_is_rejected(self):
        a=np.ones((4,4,4),np.uint32)
        with self.assertRaisesRegex(ValueError,'do not overlap'):overlap_midplane(a,a,[0,0,0],[4,0,0],0)
        with self.assertRaisesRegex(ValueError,'Axis-neighbor'):overlap_midplane(a,a,[0,0,0],[1,1,0],0)

    def test_interface_mapping_preserves_distinct_cropped_components(self):
        # One original ID can split after cropping; never map by original ID alone.
        pred=np.full((6,4,4),7,np.uint32)
        components=np.ones((4,4,4),np.uint32);components[:,:,1]=0;components[:,:,2:]=2
        pred[1:5,:,1]=0
        pred[0,0,0]=8
        plane=component_interface_plane(pred,components,[0,0,0],[1,0,0],0,0,[[0,1],[0,4],[0,4]])
        self.assertEqual(plane[0,0],0)
        self.assertEqual(plane[1,0],1);self.assertEqual(plane[1,3],2)
        inside=component_interface_plane(pred,components,[0,0,0],[1,0,0],0,1,[[0,1],[0,4],[0,4]])
        np.testing.assert_array_equal(inside,components[0])
        with self.assertRaisesRegex(ValueError,'touch core'):
            component_interface_plane(np.ones((8,4,4),np.uint32),components,[0,0,0],[2,0,0],0,0,[[0,1],[0,4],[0,4]])

    def test_invalid_arrays_rejected(self):
        for a,b in [(np.zeros((2,2)),np.zeros((2,2))),
                    (np.zeros((0,2),np.uint32),np.zeros((0,2),np.uint32)),
                    (np.zeros((2,2),np.uint32),np.zeros((2,3),np.uint32))]:
            with self.assertRaises(ValueError):mutual_maximum_pairs(a,b)


if __name__=='__main__':unittest.main()
