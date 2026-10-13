"""Validation-only condition diagnostics and molecule-cluster uncertainty."""

from collections import defaultdict
import math

import numpy as np


def released_domain(row):
    energy = row.get('collision_energy')
    return (row.get('adduct') == '[M+H]+'
            and row.get('instrument_type') in {'Orbitrap', 'QTOF'}
            and isinstance(energy, (int, float)) and math.isfinite(energy))


def cluster_interval(outcomes, groups, *, resamples=2000, seed=0, confidence=0.95):
    """Resample molecules, retaining the official spectrum-weighted statistic."""
    if len(outcomes) != len(groups) or not groups or resamples < 1 or not 0 < confidence < 1:
        raise ValueError('invalid cluster-bootstrap inputs')
    grouped = defaultdict(lambda: [0.0, 0])
    for outcome, group in zip(outcomes, groups):
        grouped[group][0] += outcome
        grouped[group][1] += 1
    values = np.asarray(list(grouped.values()), dtype=float)
    rng = np.random.default_rng(seed)
    draws = []
    for offset in range(0, resamples, 128):
        indices = rng.integers(len(values), size=(min(128, resamples - offset), len(values)))
        sample = values[indices].sum(axis=1)
        draws.extend((sample[:, 0] / sample[:, 1]).tolist())
    alpha = (1 - confidence) / 2
    low, high = np.quantile(draws, [alpha, 1 - alpha])
    return {'estimate': float(np.mean(outcomes)), 'lower': float(low), 'upper': float(high),
            'confidence': confidence, 'resamples': resamples, 'seed': seed,
            'molecule_groups': len(values), 'queries': len(groups),
            'grouping': 'supplied target_inchikey connectivity block',
            'statistic': 'spectrum-weighted mean; molecules sampled with replacement'}


def describe(rows, pools, rankings, correct, *, resamples=2000, seed=0, baseline=None):
    strata = {name: defaultdict(list) for name in
              ['adduct', 'instrument_type', 'collision_energy', 'candidate_pool_size', 'routing_domain']}
    hits, groups, differences = [], [], []
    for query_id, row in rows.items():
        hit = int(query_id in rankings and rankings[query_id][0] == correct[query_id])
        hits.append(hit)
        groups.append(row['target_inchikey'].split('-')[0])
        if baseline is not None:
            differences.append(hit - int(query_id in baseline and baseline[query_id][0] == correct[query_id]))
        energy = row.get('collision_energy')
        energy_bin = ('missing' if energy is None else '<20' if energy < 20 else
                      '20-39.999' if energy < 40 else '40-59.999' if energy < 60 else '>=60')
        size = len(pools[row['candidate_pool_id']])
        labels = {'adduct': row['adduct'], 'instrument_type': row['instrument_type'],
                  'collision_energy': energy_bin,
                  'candidate_pool_size': '1-32' if size <= 32 else '33-128' if size <= 128 else '129-256',
                  'routing_domain': 'glacier_released' if released_domain(row) else 'fallback'}
        for name, label in labels.items():
            strata[name][label].append(hit)
    result = {'strata': {name: {label: {'queries': len(values), 'hit_rate_at_1': sum(values) / len(values)}
                                      for label, values in sorted(bins.items())}
                         for name, bins in strata.items()},
              'uncertainty': cluster_interval(hits, groups, resamples=resamples, seed=seed)}
    if baseline is not None:
        result['paired_baseline_delta'] = cluster_interval(differences, groups, resamples=resamples, seed=seed)
    return result
