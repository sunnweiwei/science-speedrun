#!/usr/bin/env python3
"""Score visible validation rankings with fixed-denominator hit rates."""

import argparse
import hashlib
import json
from pathlib import Path

from diagnostics import describe
from benchmark_score import normalize


REFERENCE_BASELINE = "GLACIER no-CF with strict-clean MIST fallback"
# Filled from the frozen full-denominator operator replay documented in
# ../reference/reference_baseline.json.
REFERENCE_HIT_RATE_AT_1 = 0.4939052175894281


def molecule_id(smiles):
    return hashlib.sha256(smiles.encode()).hexdigest()


def jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_rankings(path, expected, pools):
    rankings = {}
    for row in jsonl(path):
        if not isinstance(row, dict) or set(row) != {"query_id", "ranking"}:
            raise ValueError("malformed ranking row")
        query_id, ranking = row["query_id"], row["ranking"]
        if not isinstance(query_id, str) or query_id not in expected or query_id in rankings:
            raise ValueError("unknown or duplicate query ranking")
        candidate_ids = [candidate["candidate_id"] for candidate in pools[expected[query_id]["pool"]]]
        if (not isinstance(ranking, list) or any(not isinstance(value, str) for value in ranking)
                or len(ranking) != len(candidate_ids) or set(ranking) != set(candidate_ids)):
            raise ValueError("ranking is not an exact candidate permutation")
        rankings[query_id] = ranking
    return rankings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--rankings", required=True, type=Path)
    parser.add_argument("--query-input", type=Path)
    parser.add_argument("--mces", action="store_true")
    parser.add_argument("--baseline-rankings", type=Path, help="A ranking on the same validation query set")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    selected_queries = None
    if args.query_input is not None:
        selected_queries = {
            row["query_id"] for row in jsonl(args.query_input / "queries.jsonl")
        }
        if not selected_queries:
            raise ValueError("query input is empty")
    pools = {
        row["candidate_pool_id"]: row["candidates"]
        for row in jsonl(args.input_root / "validation_candidate_pools.jsonl")
    }
    expected, selected_rows = {}, {}
    for row in jsonl(args.input_root / "validation.jsonl"):
        if selected_queries is not None and row["query_id"] not in selected_queries:
            continue
        expected[row["query_id"]] = {
            "pool": row["candidate_pool_id"],
            "correct": molecule_id(row["target_smiles"]),
            "smiles": row["target_smiles"],
        }
        selected_rows[row["query_id"]] = row
    if not expected or (selected_queries is not None and set(expected) != selected_queries):
        raise ValueError("query input must be a nonempty subset of validation")
    rankings, global_error = {}, None
    try:
        rankings = load_rankings(args.rankings, expected, pools)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        rankings, global_error = {}, str(error)

    metrics = {}
    for k in (1, 5, 20):
        metrics[f"hit_rate_at_{k}"] = sum(
            query_id in rankings and item["correct"] in rankings[query_id][:k]
            for query_id, item in expected.items()
        ) / len(expected)
    if args.mces:
        from myopic_mces.myopic_mces import MCES
        import pulp
        if not pulp.PULP_CBC_CMD(msg=False).available():
            raise RuntimeError("Pulp's bundled CBC solver is not available")
        total = 0.0
        for query_id, item in expected.items():
            if query_id not in rankings:
                total += 100.0
                continue
            by_id = {
                candidate["candidate_id"]: candidate["smiles"]
                for candidate in pools[item["pool"]]
            }
            predicted = by_id[rankings[query_id][0]]
            total += MCES(
                s1=item["smiles"], s2=predicted, ind=0, threshold=15,
                always_stronger_bound=True, solver="PULP_CBC_CMD",
                solver_options={"msg": 0},
            )[1]
        metrics["mces_at_1"] = total / len(expected)
    baseline = None
    if args.baseline_rankings:
        baseline = load_rankings(args.baseline_rankings, expected, pools)
        if set(baseline) != set(expected):
            raise ValueError("paired baseline must cover the same complete validation query set")
    report = {
        "schema": "massspecgym-retrieval-development-report/v1",
        "queries": len(expected),
        "valid_rankings": len(rankings),
        "global_error": global_error,
        "external_reference_context": {
            "name": REFERENCE_BASELINE,
            "hit_rate_at_1": REFERENCE_HIT_RATE_AT_1,
            "fold": "test",
            "comparable_to_this_validation_report": False,
        },
        "metrics": metrics,
        "benchmark_score": {**normalize(metrics["hit_rate_at_1"]),
                            "evaluation_fold": "validation",
                            "comparable_to_test_reference": False},
        **describe(selected_rows, pools, rankings, {q: item['correct'] for q, item in expected.items()},
                   resamples=args.bootstrap_resamples, seed=args.seed, baseline=baseline),
    }
    text = json.dumps(report, sort_keys=True, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end='')


if __name__ == "__main__":
    main()
