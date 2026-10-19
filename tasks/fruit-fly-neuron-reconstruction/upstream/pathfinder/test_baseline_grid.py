import unittest
import numpy as np
from baseline_grid import target_grid, image_chunk


class BaselineGridTests(unittest.TestCase):
    def setUp(self):
        self.meta=dict(axes='zyx',coordinate_convention='voxel_outer_corner',
            origin_nm=[1600,3200,4800],voxel_size_nm=[16]*3,shape=[8]*3)

    def test_physical_center_ramp_and_chunk_seams(self):
        target=target_grid(self.meta)
        self.assertEqual(target['shape'],[14,14,6])
        self.assertEqual(target['origin_nm'],[4800,3200,1600])
        raw=np.broadcast_to((np.arange(8)*20).astype(np.uint8)[None,None,:],(8,8,8))
        full=image_chunk(raw,self.meta,target,[0,0,0],target['shape'])
        expected=np.rint(np.clip((np.arange(14)+.5)*9/16-.5,0,7)*20).astype('u1')
        np.testing.assert_array_equal(full[:,0,0],expected)
        a=image_chunk(raw,self.meta,target,[0,0,0],[7,14,6])
        b=image_chunk(raw,self.meta,target,[7,0,0],[14,14,6])
        np.testing.assert_array_equal(np.concatenate([a,b]),full)

    def test_identity_grid_preserves_pixels(self):
        target=dict(self.meta,axes='xyz',origin_nm=[4800,3200,1600])
        raw=np.arange(512,dtype=np.uint16).reshape(8,8,8).astype('u1')
        np.testing.assert_array_equal(image_chunk(raw,self.meta,target,[0]*3,[8]*3),raw.transpose(2,1,0))

    def test_bad_grid_and_chunks_rejected(self):
        with self.assertRaises(ValueError):target_grid(dict(self.meta,voxel_size_nm=[8]*3))
        raw=np.zeros((8,8,8),dtype='u1');target=target_grid(self.meta)
        for start,stop in [([-1,0,0],[1]*3),([0]*3,[99]*3),([0]*3,[0]*3)]:
            with self.assertRaises(ValueError):image_chunk(raw,self.meta,target,start,stop)


if __name__=='__main__':unittest.main()
