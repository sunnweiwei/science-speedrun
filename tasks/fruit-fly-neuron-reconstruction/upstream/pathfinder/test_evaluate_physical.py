import copy
import unittest
import numpy as np
from evaluate_physical import assignments,grid,reference_bounds,reference_graph

class PhysicalScoringTests(unittest.TestCase):
    def setUp(self):
        self.reference=dict(axes='zyx',coordinate_convention='voxel_outer_corner',origin_nm=[-16,32,48],voxel_size_nm=[16]*3,shape=[4]*3)
        self.native=dict(axes='xyz',coordinate_convention='voxel_outer_corner',origin_nm=[40,30,-20],voxel_size_nm=[9,9,20],shape=[8,8,4])
        self.ids=(2**60+3,2**60+7)
        self.points=np.array([[z,y,x] for z in (-8,8,24,40) for y in (40,56,72,88) for x in (56,72,88,104)])

    def test_axis_order_and_native_spacing_preserve_assignments(self):
        ours=np.full((4,4,4),self.ids[0],dtype=np.uint64);ours[:,:,2:]=self.ids[1]
        native=np.full((8,8,4),self.ids[0],dtype=np.uint64);native[4:]=self.ids[1]
        a=assignments(ours,self.reference,self.reference,self.points)
        np.testing.assert_array_equal(a,assignments(native,self.native,self.reference,self.points))
        self.assertEqual(set(map(int,a)),set(self.ids))
        meta=dict(self.native,axes='zyx',origin_nm=self.native['origin_nm'][::-1],voxel_size_nm=self.native['voxel_size_nm'][::-1],shape=self.native['shape'][::-1])
        np.testing.assert_array_equal(a,assignments(native.transpose(2,1,0),meta,self.reference,self.points))

    def test_floor_boundary_and_omissions(self):
        a=np.ones((4,4,4),dtype=np.uint32);a[:,:,1]=0
        np.testing.assert_array_equal(assignments(a,self.reference,self.reference,[[0,40,63.999],[0,40,64]]),[1,0])

    def test_no_partial_coverage_or_dropped_nodes(self):
        a=np.ones((8,8,4),dtype=np.uint32)
        with self.assertRaisesRegex(ValueError,'entire frozen'):
            assignments(a,dict(self.native,origin_nm=[49,30,-20]),self.reference,self.points)
        for point in ([48,40,56],[0,40,float('nan')]):
            with self.assertRaises(ValueError):assignments(a,self.native,self.reference,[point])

    def test_explicit_core_without_inference_halo(self):
        ref=dict(self.reference,origin_nm=[-32,16,32],shape=[6]*3,scored_core_origin_nm=self.reference['origin_nm'],scored_core_shape=[4]*3)
        np.testing.assert_array_equal(reference_bounds(ref),reference_bounds(self.reference))
        assignments(np.ones((4,4,4),dtype=np.uint32),self.reference,ref,self.points)
        with self.assertRaises(ValueError):reference_bounds(dict(ref,scored_core_shape=[99]*3))

    def test_reject_ambiguous_grids(self):
        for change in ({'axes':'yxz'},{'voxel_size_nm':[9,0,20]},{'shape':[8.,8.,4.]},{'coordinate_convention':'center'}):
            with self.assertRaises(ValueError):grid(dict(self.native,**change))
        with self.assertRaises(ValueError):grid(self.native,(8,4,8))
        with self.assertRaises(ValueError):assignments(np.ones((4,4,4)),self.reference,self.reference,self.points)

    def test_topology_cannot_be_silently_filtered(self):
        truth=dict(axes='zyx',coordinate_units='nm',nodes=[dict(id=0,skeleton_id=1,position_nm=[0,0,0]),dict(id=1,skeleton_id=1,position_nm=[0,0,16])],edges=[[0,1]])
        self.assertEqual(reference_graph(truth).number_of_edges(),1)
        bad=copy.deepcopy(truth);bad['nodes'][1]['skeleton_id']=2
        with self.assertRaises(ValueError):reference_graph(bad)
        bad=copy.deepcopy(truth);bad['edges'].append([0,1])
        with self.assertRaises(ValueError):reference_graph(bad)

if __name__=='__main__':unittest.main()
