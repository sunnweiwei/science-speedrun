import unittest
from nested_reference import subset


class NestedReferenceTests(unittest.TestCase):
    def setUp(self):
        self.meta = dict(axes='zyx', coordinate_convention='voxel_outer_corner',
            origin_nm=[0,0,0], voxel_size_nm=[16,16,16], shape=[12]*3,
            scored_core_origin_nm=[32]*3, scored_core_shape=[8]*3)
        self.truth = dict(axes='zyx', coordinate_units='nm', nodes=[
            dict(id=i, skeleton_id=2**60+1, position_nm=[v]*3)
            for i,v in enumerate([40,64,80,127,128,152])],
            edges=[[i,i+1] for i in range(5)])

    def test_half_open_nested_box_preserves_nodes_edges_and_ids(self):
        truth, meta = subset(self.truth,self.meta,4)
        self.assertEqual([n['id'] for n in truth['nodes']],[1,2,3])
        self.assertEqual(truth['edges'],[[1,2],[2,3]])
        self.assertEqual(meta['scored_core_origin_nm'],[64]*3)
        self.assertEqual(meta['shape'],[12]*3)
        self.assertEqual(truth['nodes'][0],self.truth['nodes'][1])
        self.assertEqual(len(self.truth['nodes']),6)

    def test_full_size_is_identity(self):
        truth,meta=subset(self.truth,self.meta,8)
        self.assertEqual(truth,self.truth)
        self.assertEqual(meta,self.meta)

    def test_invalid_size_alignment_empty_and_outside_fail_closed(self):
        for size in (True,4.0,0,-2,3,10,2):
            with self.assertRaises(ValueError):subset(self.truth,self.meta,size)
        self.truth['nodes'][0]['position_nm']=[0]*3
        with self.assertRaises(ValueError):subset(self.truth,self.meta,4)


if __name__=='__main__':unittest.main()
