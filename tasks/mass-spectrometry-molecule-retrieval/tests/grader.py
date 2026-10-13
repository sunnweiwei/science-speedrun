#!/usr/bin/env python3
"""Trusted label-side grading for MassSpecGym candidate rankings."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import time
import sys

# In the verifier image these shared files are copied beside this module.
# In a checkout import the same source used by development validation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "environment/evaluation"))
from benchmark_score import normalize

from common import jsonl, sha256, validate_input, write_json


REFERENCE_BASELINE = "GLACIER no-CF with strict-clean MIST fallback"
# Filled from the frozen full-denominator operator replay documented in
# environment/references/reference_baseline.json.
REFERENCE_HIT_RATE_AT_1 = 0.4939052175894281
REFERENCE_INPUT_FILES = {
    "queries.jsonl": "ece08fafaa86d8b6b57d973c0c4c8457f45d97a6aa42e43f303e0a71fe8cd4fc",
    "candidate_pools.jsonl": "fc7905baf25028cb21fb5fe57dbd4f77bfbf455f362e9c8e3528283d198fbbce",
}


def reference_comparison(metrics, input_manifest):
    comparable = input_manifest.get("files") == REFERENCE_INPUT_FILES
    delta = metrics["hit_rate_at_1"] - REFERENCE_HIT_RATE_AT_1 if comparable else None
    return {
        "name": REFERENCE_BASELINE, "hit_rate_at_1": REFERENCE_HIT_RATE_AT_1,
        "comparable": comparable, "reference_delta_hit_at_1": delta,
        "beats_reference": delta > 0 if comparable else None,
    }


def _mces(pair):
    truth, prediction = pair
    if truth == prediction:
        return 0.0
    import pulp
    from myopic_mces.myopic_mces import MCES
    if not pulp.PULP_CBC_CMD(msg=False).available():
        raise RuntimeError("Pulp's bundled CBC solver is not available")
    return float(MCES(
        s1=truth,
        s2=prediction,
        ind=0,
        threshold=15,
        always_stronger_bound=True,
        solver="PULP_CBC_CMD",
        solver_options={"msg": 0},
    )[1])


def load_labels(path: Path, expected_queries: set[str]):
    labels = {}
    for row in jsonl(path):
        if set(row) != {"query_id", "correct_candidate_id", "true_smiles"}:
            raise ValueError("invalid private label fields")
        query_id = row["query_id"]
        if query_id not in expected_queries or query_id in labels:
            raise ValueError("unknown or duplicate private label")
        if not isinstance(row["correct_candidate_id"], str) or not isinstance(row["true_smiles"], str):
            raise ValueError("invalid private label types")
        labels[query_id] = row
    if set(labels) != expected_queries:
        raise ValueError("private labels do not cover all input queries")
    return labels


def load_rankings(path: Path, queries: dict, pools: dict):
    rankings, global_error = {}, None
    try:
        for row in jsonl(path):
            if set(row) != {"query_id", "ranking"}:
                raise ValueError("invalid ranking fields")
            query_id = row["query_id"]
            if not isinstance(query_id, str) or query_id not in queries or query_id in rankings:
                raise ValueError("unknown or duplicate query ranking")
            expected = set(pools[queries[query_id]["candidate_pool_id"]])
            ranking = row["ranking"]
            if (
                not isinstance(ranking, list)
                or any(not isinstance(value, str) for value in ranking)
                or len(ranking) != len(expected)
                or set(ranking) != expected
            ):
                raise ValueError("ranking is not an exact candidate permutation")
            rankings[query_id] = ranking
    except (OSError, ValueError, json.JSONDecodeError) as error:
        rankings, global_error = {}, str(error)
    return rankings, global_error


def score(rankings, queries, pools, labels, compute_mces=True, workers=None):
    count = len(queries)
    metrics = {
        f"hit_rate_at_{k}": sum(
            query_id in rankings
            and labels[query_id]["correct_candidate_id"] in rankings[query_id][:k]
            for query_id in queries
        ) / count
        for k in (1, 5, 20)
    }
    mces_values = []
    if compute_mces:
        pair_indexes = {}
        for query_id, query in queries.items():
            if query_id not in rankings:
                mces_values.append(100.0)
                continue
            predicted_id = rankings[query_id][0]
            predicted_smiles = pools[query["candidate_pool_id"]][predicted_id]
            pair = (labels[query_id]["true_smiles"], predicted_smiles)
            pair_indexes.setdefault(pair, []).append(len(mces_values))
            mces_values.append(None)
        worker_count = workers or min(16, os.cpu_count() or 1)
        if pair_indexes:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                values = executor.map(_mces, pair_indexes, chunksize=16)
                for indexes, value in zip(pair_indexes.values(), values):
                    for index in indexes:
                        mces_values[index] = value
        metrics["mces_at_1"] = sum(mces_values) / count
    return metrics


def run(args, compute_mces=True):
    ready = args.sealed.parent / "READY"
    deadline = time.monotonic() + args.wait_seconds
    while not ready.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("ranking snapshot was not published")
        time.sleep(1)
    snapshot_path = args.sealed / "snapshot.json"
    if ready.read_text().strip() != sha256(snapshot_path):
        raise ValueError("snapshot ready hash mismatch")
    snapshot = json.loads(snapshot_path.read_text())
    if snapshot.get("schema") != "massspecgym-retrieval-ranking-snapshot/v1":
        raise ValueError("invalid snapshot schema")
    if snapshot.get("input_manifest_sha256") != sha256(args.input / "manifest.json"):
        raise ValueError("snapshot/input mismatch")
    for name, expected_hash in snapshot.get("files", {}).items():
        path = args.sealed / name
        if not path.is_file() or path.is_symlink() or sha256(path) != expected_hash:
            raise ValueError(f"sealed output hash mismatch: {name}")

    input_manifest, queries, pools = validate_input(args.input)
    private_manifest = json.loads((args.labels / "manifest.json").read_text())
    if private_manifest.get("schema") != "massspecgym-retrieval-private-labels/v1":
        raise ValueError("invalid private manifest schema")
    if private_manifest.get("files", {}).get("labels.jsonl") != sha256(args.labels / "labels.jsonl"):
        raise ValueError("private label hash mismatch")
    labels = load_labels(args.labels / "labels.jsonl", set(queries))
    ranking_path = args.sealed / "rankings.jsonl"
    rankings, global_error = load_rankings(ranking_path, queries, pools)
    if not ranking_path.is_file():
        global_error = "rankings.jsonl is missing"
    metrics = score(
        rankings, queries, pools, labels,
        compute_mces=compute_mces,
        workers=args.mces_workers,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "massspecgym-retrieval-report/v1",
        "evaluation_mode": "isolated_model_replay",
        "status": ("invalid_output" if global_error else
                   "complete" if len(rankings) == len(queries) else "missing_predictions"),
        "model_quality_measured": global_error is None,
        "queries": len(queries),
        "valid_rankings": len(rankings),
        "invalid_or_missing_rankings": len(queries) - len(rankings),
        "global_error": global_error,
        "metric_directions": {
            "hit_rate_at_1": "maximize",
            "hit_rate_at_5": "maximize",
            "hit_rate_at_20": "maximize",
            "mces_at_1": "minimize",
        },
        "reference_comparison": reference_comparison(metrics, input_manifest),
        "metrics": metrics,
        "benchmark_score": normalize(metrics["hit_rate_at_1"]),
    }
    write_json(args.output / "report.json", report)
    write_json(args.output / "reward.json", {"reward": metrics["hit_rate_at_1"]})
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sealed", required=True, type=lambda x: Path(x).resolve())
    parser.add_argument("--input", required=True, type=lambda x: Path(x).resolve())
    parser.add_argument("--labels", required=True, type=lambda x: Path(x).resolve())
    parser.add_argument("--output", required=True, type=lambda x: Path(x).resolve())
    parser.add_argument("--wait-seconds", type=int, default=21600)
    parser.add_argument("--mces-workers", type=int, default=16)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "reward.json").unlink(missing_ok=True)
        write_json(args.output / "error.json", {
            "status": "infrastructure_error",
            "error": f"{type(error).__name__}: {error}",
        })
        raise


if __name__ == "__main__":
    main()
