import itertools
import random
import unittest

from lazy_search import filter_large_component
from reconcile import connected, reconcile
from search import filter_graph


def edge(u, v, sense=.9):
    return dict(source=u, target=v, sense=sense, shape=.9)


class LazySearchTests(unittest.TestCase):
    def test_matches_exhaustive_constraints(self):
        rng = random.Random(829)
        for _ in range(100):
            nodes = list(range(1, 9))
            edges = [edge(u, v, rng.random()) for u, v in itertools.combinations(nodes, 2)
                     if rng.random() < .4]
            pairs = [(e['source'], e['target']) for e in edges]
            scores = {}
            forbidden = []
            threshold = rng.choice([0., .2, .7])
            for size in range(2, 5):
                for motif in itertools.combinations(nodes, size):
                    if rng.random() < .9:
                        scores[motif] = rng.choice([.1, .8, 1.])
                    score = scores.get(motif)
                    if ((score is not None and score < threshold) or
                        (connected(motif, pairs) and score is None)):
                        forbidden.append((motif, 0.))
            reference = filter_graph(nodes, edges, forbidden, 1., priority='sense')
            actual = filter_large_component(nodes, edges, scores, threshold)
            self.assertEqual(actual['components'], reference['components'])
            self.assertEqual(actual['retained'], reference['retained'])
            self.assertEqual(actual['check_budget_rejections'], 0)

    def chain(self, count=100):
        nodes = list(range(1, count+1))
        edges = [edge(n, n+1) for n in nodes[:-1]]
        scores = {tuple(range(n, n+size)): .9
                  for size in range(2, 5) for n in range(1, count-size+2)}
        return nodes, edges, scores

    def test_large_chain_merges_when_evidence_complete(self):
        nodes, edges, scores = self.chain()
        result = reconcile(nodes, edges, list(scores.items()))
        self.assertEqual(result['components'], [nodes])
        self.assertEqual(result['methods'][0]['method'], 'lazy_conservative_greedy')
        self.assertEqual(result['missing_motif_scores'], 0)

    def test_missing_quadruple_and_disconnected_negative(self):
        for mutation in ['missing', 'negative']:
            nodes, edges, scores = self.chain()
            forbidden = (1, 2, 3, 4) if mutation == 'missing' else (1, 4)
            if mutation == 'missing':
                del scores[forbidden]
            else:
                scores[forbidden] = 0.
            result = reconcile(nodes, edges, list(scores.items()))
            self.assertFalse(any(set(forbidden) <= set(g) for g in result['components']))
            self.assertGreater(len(result['retained']), 0)

    def test_budget_exhaustion_abstains(self):
        nodes, edges, scores = self.chain()
        result = filter_large_component(nodes, edges, scores, .2, max_checks=1)
        self.assertEqual(result['retained'], [])
        self.assertEqual(result['check_budget_rejections'], len(edges))


if __name__ == '__main__':
    unittest.main()
