#!/usr/bin/env python3
"""Paired Hit@1 bootstrap comparison for two complete ranking files."""

import argparse
import json
from pathlib import Path

import numpy as np


def jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error


def load_truth(path: Path) -> dict[str, str]:
    truth = {}
    for row in jsonl(path):
        query_id = row.get("query_id")
        candidate_id = row.get("correct_candidate_id")
        if (
            not isinstance(query_id, str)
            or not isinstance(candidate_id, str)
            or query_id in truth
        ):
            raise ValueError("invalid or duplicate label row")
        truth[query_id] = candidate_id
    if not truth:
        raise ValueError("label file is empty")
    return truth


def load_top1(path: Path, expected: set[str]) -> dict[str, str]:
    predictions = {}
    for row in jsonl(path):
        if set(row) != {"query_id", "ranking"}:
            raise ValueError(f"invalid ranking fields in {path}")
        query_id = row["query_id"]
        ranking = row["ranking"]
        if (
            not isinstance(query_id, str)
            or query_id not in expected
            or query_id in predictions
            or not isinstance(ranking, list)
            or not ranking
            or any(not isinstance(candidate_id, str) for candidate_id in ranking)
            or len(set(ranking)) != len(ranking)
        ):
            raise ValueError(f"invalid or duplicate ranking in {path}")
        predictions[query_id] = ranking[0]
    if set(predictions) != expected:
        raise ValueError(f"ranking file does not cover the label query set: {path}")
    return predictions


def compare(
    truth: dict[str, str],
    baseline: dict[str, str],
    challenger: dict[str, str],
    *,
    confidence: float,
    resamples: int,
    seed: int,
    groups: dict[str, str] | None = None,
) -> dict:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    outcomes = {"challenger_only": 0, "baseline_only": 0, "tie": 0}
    baseline_hits = challenger_hits = 0
    for query_id, correct in truth.items():
        baseline_hit = baseline[query_id] == correct
        challenger_hit = challenger[query_id] == correct
        baseline_hits += baseline_hit
        challenger_hits += challenger_hit
        if challenger_hit and not baseline_hit:
            outcomes["challenger_only"] += 1
        elif baseline_hit and not challenger_hit:
            outcomes["baseline_only"] += 1
        else:
            outcomes["tie"] += 1

    count = len(truth)
    probabilities = np.array([
        outcomes["baseline_only"],
        outcomes["tie"],
        outcomes["challenger_only"],
    ], dtype=np.float64) / count
    rng = np.random.default_rng(seed)
    deltas = np.empty(resamples, dtype=np.float64)
    offset = 0
    while offset < resamples:
        batch = min(10_000, resamples - offset)
        samples = rng.multinomial(count, probabilities, size=batch)
        deltas[offset:offset + batch] = (samples[:, 2] - samples[:, 0]) / count
        offset += batch
    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(deltas, [alpha, 1 - alpha])
    delta = (challenger_hits - baseline_hits) / count
    report = {
        "schema": "massspecgym-retrieval-paired-bootstrap/v1",
        "queries": count,
        "metric": "hit_rate_at_1",
        "baseline_hit_rate": baseline_hits / count,
        "challenger_hit_rate": challenger_hits / count,
        "delta": delta,
        "paired_outcomes": outcomes,
        "confidence": confidence,
        "resamples": resamples,
        "seed": seed,
        "interval": [float(lower), float(upper)],
        "beats_reference": bool(lower > 0),
        "method": "paired nonparametric percentile bootstrap; multinomial-compressed {-1,0,1} outcomes",
    }
    if groups is not None:
        from diagnostics import cluster_interval
        if set(groups) != set(truth):
            raise ValueError('molecule groups must cover the exact query set')
        result = cluster_interval(
            [int(challenger[q] == truth[q]) - int(baseline[q] == truth[q]) for q in truth],
            [groups[q] for q in truth], resamples=resamples, seed=seed, confidence=confidence,
        )
        report.update(schema='massspecgym-retrieval-paired-bootstrap/v2',
                      interval=[result['lower'], result['upper']],
                      beats_reference=result['lower'] > 0,
                      molecule_groups=result['molecule_groups'],
                      method='paired molecule-cluster percentile bootstrap; spectrum-weighted Hit@1 delta')
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--challenger", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--confidence", type=float, default=0.999)
    parser.add_argument("--resamples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--unit', choices=['molecule', 'query'], default='molecule',
                        help='Molecules are the default uncertainty unit; query reproduces historical reports')
    args = parser.parse_args()
    truth = load_truth(args.labels)
    baseline = load_top1(args.baseline, set(truth))
    challenger = load_top1(args.challenger, set(truth))
    groups = None
    if args.unit == 'molecule':
        from rdkit import Chem, rdBase
        groups = {}
        for row in jsonl(args.labels):
            molecule = Chem.MolFromSmiles(row['true_smiles'])
            if molecule is None:
                raise ValueError('cannot assign a molecule group to an invalid true SMILES')
            groups[row['query_id']] = Chem.MolToInchiKey(molecule).split('-')[0]
    report = compare(
        truth,
        baseline,
        challenger,
        confidence=args.confidence,
        resamples=args.resamples,
        seed=args.seed,
        groups=groups,
    )
    if groups is not None:
        report['grouping'] = 'RDKit InChIKey connectivity block from private true_smiles'
        report['rdkit_version'] = rdBase.rdkitVersion
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
