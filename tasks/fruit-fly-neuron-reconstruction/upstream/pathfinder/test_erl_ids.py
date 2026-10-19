import unittest
import networkx as nx
from evaluate_erl import compact_ids


class ErlIdTests(unittest.TestCase):
    def test_uint64_ids_stay_distinct_and_zero_remains_omitted(self):
        g=nx.Graph([(0,1),(2,3)])
        for i in g:g.nodes[i]['skeleton_id']=2**63+i//2
        lookup={0:0,1:2**64-2,2:2**64-1,3:2**64-1}
        mapped,labels=compact_ids(g,lookup)
        self.assertEqual(labels,{0:0,1:1,2:2,3:2})
        self.assertEqual([mapped.nodes[i]['skeleton_id'] for i in range(4)],[1,1,2,2])
        self.assertEqual(g.nodes[0]['skeleton_id'],2**63)
        with self.assertRaises(ValueError):compact_ids(g,{0:True,1:1,2:2,3:2})
        with self.assertRaises(ValueError):compact_ids(g,{0:1,1:1})

if __name__=='__main__':unittest.main()
