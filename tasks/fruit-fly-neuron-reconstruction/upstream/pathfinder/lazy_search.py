"""Conservative greedy reconciliation with lazy, bounded motif checks.

Inputs are validated by reconcile(). Every newly assembled connected 2--4 node
motif needs a passing score. Known negative motifs also constrain indirect joins.
This is an independent heuristic, not the paper's combinatorial optimizer.
"""


def filter_large_component(nodes, edges, scores, threshold, max_checks=100000):
    if type(max_checks) is not int or max_checks < 1:
        raise ValueError('Positive integer check budget required')
    groups = {n: {n} for n in nodes}
    owner = {n: n for n in nodes}
    adjacency = {n: set() for n in nodes}
    for edge in edges:
        u, v = edge['source'], edge['target']
        adjacency[u].add(v); adjacency[v].add(u)
    negative = {n: [] for n in nodes}
    for motif, score in scores.items():
        if score < threshold and all(n in owner for n in motif):
            for n in motif:
                negative[n].append(motif)
    retained = []
    counts = dict(missing_motif_scores=0, low_score_rejections=0,
                  check_budget_rejections=0, checks=0)

    class CheckBudget(Exception):
        pass

    for edge in sorted(edges, key=lambda e: (-e['sense'], min(e['source'], e['target']),
                                           max(e['source'], e['target']))):
        a, b = owner[edge['source']], owner[edge['target']]
        if a == b:
            retained.append(edge)
            continue
        if len(groups[a]) > len(groups[b]):
            a, b = b, a
        small, large = groups[a], groups[b]
        checks = 0

        def tick():
            nonlocal checks
            checks += 1
            if checks > max_checks:
                raise CheckBudget

        def allowed():
            # Every newly violated negative constraint touches the smaller group.
            for n in small:
                tick()
                for motif in negative[n]:
                    tick()
                    if all(v in small or v in large for v in motif):
                        counts['low_score_rejections'] += 1
                        return False
            # Any newly assembled connected motif contains a cross-group edge.
            # Grow all such pairs to size four, restricted to the proposed union.
            frontier = set()
            for n in small:
                for v in adjacency[n]:
                    tick()
                    if v in large:
                        frontier.add(tuple(sorted((n, v))))
            for size in range(2, 5):
                following = set()
                for motif in sorted(frontier):
                    tick()
                    score = scores.get(motif)
                    if score is None:
                        counts['missing_motif_scores'] += 1
                        return False
                    if score < threshold:
                        counts['low_score_rejections'] += 1
                        return False
                    if size == 4:
                        continue
                    for n in motif:
                        for v in adjacency[n]:
                            tick()
                            if v not in motif and (v in small or v in large):
                                following.add(tuple(sorted((*motif, v))))
                frontier = following
            return True

        try:
            accept = allowed()
        except CheckBudget:
            accept = False
            counts['check_budget_rejections'] += 1
        counts['checks'] += checks
        if accept:
            retained.append(edge)
            for n in small:
                owner[n] = b
            large.update(small)
            del groups[a]
    return dict(retained=retained, components=sorted(sorted(g) for g in groups.values()),
                max_checks_per_join=max_checks, **counts)
