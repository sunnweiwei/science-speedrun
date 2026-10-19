"""Bounded enumeration must match independently enumerated connected subsets."""
import itertools
import random
import unittest
from infer_shape import connected_motifs


def exhaustive(edges,max_nodes):
    nodes=sorted({n for e in edges for n in (e['source'],e['target'])})
    pairs={frozenset((e['source'],e['target'])) for e in edges};result=[]
    for size in range(2,max_nodes+1):
        for group in itertools.combinations(nodes,size):
            seen={group[0]}
            while True:
                more={v for u in seen for v in group if frozenset((u,v)) in pairs}
                if more<=seen:break
                seen|=more
            if len(seen)==size:result.append(group)
    return result


class MotifEnumerationTests(unittest.TestCase):
    def test_exact_prefix_and_truncation_against_all_connected_subsets(self):
        rng=random.Random(71)
        for _ in range(15):
            edges=[dict(source=a,target=b) for a,b in itertools.combinations(range(1,9),2) if rng.random()<.45]
            for size in (2,3,4):
                expected=exhaustive(edges,size)
                for limit in (1,4,15,40,200):
                    actual,truncated=connected_motifs(edges,size,limit)
                    self.assertEqual(actual,expected[:limit]);self.assertEqual(truncated,len(expected)>limit)
    def test_dense_pair_cap_does_not_need_higher_order_frontier(self):
        edges=[dict(source=a,target=b) for a,b in itertools.combinations(range(1,101),2)]
        actual,truncated=connected_motifs(edges,4,128)
        self.assertEqual(actual,[(e['source'],e['target']) for e in edges[:128]])
        self.assertTrue(truncated)

if __name__=='__main__':unittest.main()
