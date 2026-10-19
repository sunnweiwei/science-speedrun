"""Independent local forbidden-motif graph filter. No global reconciliation yet."""
import math


def filter_graph(nodes, edges, motif_scores, threshold=.2, priority='shape'):
    """Edges: dict(source,target,sense,shape); motifs: (node tuple, score).

    Prevent an invalid motif's nodes from joining the same component, including
    through indirect paths. Stable ID tie breaks are an implementation choice.
    """
    nodes = list(nodes)
    if len(nodes) != len(set(nodes)) or any(type(n) is not int or n <= 0 for n in nodes):
        raise ValueError('Unique positive integer node IDs required')
    if priority not in ('shape', 'sense') or not 0 <= threshold <= 1:
        raise ValueError('Invalid search configuration')
    groups = {n: {n} for n in nodes}; owner = {n: n for n in nodes}
    invalid = []
    for motif, score in motif_scores:
        motif = frozenset(motif)
        if not 2 <= len(motif) <= 4 or not motif <= set(nodes) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid motif or probability')
        if score < threshold: invalid.append(motif)
    seen = set(); eligible = []; rejected = []
    for e in edges:
        u, v = e['source'], e['target']; key = tuple(sorted((u, v)))
        if u not in owner or v not in owner or u == v or key in seen:
            raise ValueError('Unknown, self or duplicate edge')
        if any(not math.isfinite(e[k]) or not 0 <= e[k] <= 1 for k in ('sense', 'shape')):
            raise ValueError('Invalid edge probability')
        seen.add(key)
        if e['shape'] < .5: rejected.append(dict(edge=list(key), reason='pair_shape'))
        else: eligible.append(e)
    retained = []
    for e in sorted(eligible, key=lambda e: (-e[priority], min(e['source'],e['target']), max(e['source'],e['target']))):
        u, v = e['source'], e['target']; a, b = owner[u], owner[v]
        merged = groups[a] | groups[b]
        if any(m <= merged for m in invalid):
            rejected.append(dict(edge=sorted((u,v)), reason='invalid_motif')); continue
        retained.append(e)
        if a != b:
            for n in groups[b]: owner[n] = a
            groups[a] = merged; del groups[b]
    return dict(components=sorted([sorted(g) for g in groups.values()]), retained=retained,
                rejected=rejected, global_reconciliation=False, author_parity=False)
