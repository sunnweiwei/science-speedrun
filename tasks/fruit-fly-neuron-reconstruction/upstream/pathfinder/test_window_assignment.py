import copy
import math
import unittest
from window_assignment import assign
from stitch_ffn import windows


class AssignmentTests(unittest.TestCase):
    def test_frozen_full_geometry_coverage_and_load(self):
        rows=windows([16,1778]*3,436,32,[1794]*3);original=copy.deepcopy(rows)
        owners=assign(rows,20,'voxel-balanced-v1')
        self.assertEqual(len(rows),125)
        self.assertEqual(set(owners),set(range(125)))
        self.assertEqual(set(owners.values()),set(range(20)))
        self.assertEqual(owners,assign(list(reversed(rows)),20,'voxel-balanced-v1'))
        self.assertEqual(rows,original)
        loads=[0]*20
        for w in rows:loads[owners[w['index']]]+=math.prod(b-a for a,b in zip(w['start'],w['stop']))
        self.assertEqual(sum(loads),8048096064)
        self.assertLess(max(loads)/(sum(loads)/20),1.14)

    def test_old_modulo_unchanged_and_bad_input_rejected(self):
        rows=windows([24,88]*3,32,8)
        self.assertEqual(assign(rows,4),{w['index']:w['index']%4 for w in rows})
        for shards,policy in [(0,'modulo'),(4,'unknown')]:
            with self.assertRaises(ValueError):assign(rows,shards,policy)
        with self.assertRaises(ValueError):assign(rows+[rows[0]],4)


if __name__=='__main__':unittest.main()
