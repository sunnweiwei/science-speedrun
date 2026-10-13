#!/usr/bin/env python3
"""Compare the task's exact-candidate metric with GLACIER's 2D metric.

This is a diagnostic only. The benchmark score remains the exact candidate-ID
metric implemented by the verifier.
"""

import argparse
import hashlib
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error


def candidate_id(smiles: str) -> str:
    return hashlib.sha256(smiles.encode()).hexdigest()


@lru_cache(maxsize=262144)
def inchikey(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    return "" if molecule is None else Chem.MolToInchiKey(molecule)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument(
        "--queries",
        type=Path,
        help=(
            "Public queries.jsonl used to recover candidate_pool_id when the "
            "label file is the private verifier schema."
        ),
    )
    parser.add_argument("--candidate-pools", required=True, type=Path)
    parser.add_argument("--rankings", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    pools = {
        row["candidate_pool_id"]: row["candidates"]
        for row in read_jsonl(args.candidate_pools)
    }
    labels = {row["query_id"]: row for row in read_jsonl(args.labels)}
    query_pools = (
        {
            row["query_id"]: row["candidate_pool_id"]
            for row in read_jsonl(args.queries)
        }
        if args.queries is not None
        else {}
    )
    rankings = {row["query_id"]: row["ranking"] for row in read_jsonl(args.rankings)}
    selected = sorted(set(labels) & set(rankings))
    if not selected:
        raise ValueError("labels and rankings have no queries in common")

    hit_counts = {k: Counter() for k in (1, 5, 20)}
    pool_duplicate_counts = Counter()
    pool_identity_cache = {}
    for query_id in selected:
        label = labels[query_id]
        pool_id = label.get("candidate_pool_id", query_pools.get(query_id))
        if pool_id is None:
            raise ValueError(
                "labels omit candidate_pool_id; pass the matching public --queries file"
            )
        candidates = pools[pool_id]
        pool_identity = pool_identity_cache.get(pool_id)
        if pool_identity is None:
            by_id = {
                candidate["candidate_id"]: candidate["smiles"]
                for candidate in candidates
            }
            key_by_id = {
                candidate_id: inchikey(smiles)
                for candidate_id, smiles in by_id.items()
            }
            full_keys = list(key_by_id.values())
            pool_identity = (
                by_id,
                key_by_id,
                len(set(full_keys)) < len(full_keys),
                len({key.split("-")[0] for key in full_keys}) < len(full_keys),
            )
            pool_identity_cache[pool_id] = pool_identity
        by_id, key_by_id, has_exact_duplicates, has_2d_duplicates = pool_identity
        ranking = rankings[query_id]
        if len(ranking) != len(by_id) or set(ranking) != set(by_id):
            raise ValueError(f"invalid candidate permutation for {query_id}")
        target_smiles = label.get("target_smiles", label.get("true_smiles"))
        if target_smiles is None:
            raise ValueError(f"label for {query_id} has no target SMILES")
        target_exact = label.get("correct_candidate_id", candidate_id(target_smiles))
        if target_exact != candidate_id(target_smiles):
            raise ValueError(f"label identity mismatch for {query_id}")
        target_ikey = inchikey(target_smiles)
        target_2d = target_ikey.split("-")[0]
        ranked_ikeys = [key_by_id[item] for item in ranking]
        pool_duplicate_counts["queries_with_exact_ikey_duplicates"] += (
            has_exact_duplicates
        )
        pool_duplicate_counts["queries_with_2d_ikey_duplicates"] += (
            has_2d_duplicates
        )
        for k in hit_counts:
            hit_counts[k]["exact_candidate_id"] += target_exact in ranking[:k]
            hit_counts[k]["exact_inchikey"] += target_ikey in ranked_ikeys[:k]
            hit_counts[k]["two_dimensional_inchikey"] += target_2d in {
                key.split("-")[0] for key in ranked_ikeys[:k]
            }

    count = len(selected)
    report = {
        "schema": "glacier-protocol-audit/v1",
        "queries": count,
        "metrics": {
            metric: {
                f"hit_rate_at_{k}": hit_counts[k][metric] / count
                for k in hit_counts
            }
            for metric in (
                "exact_candidate_id",
                "exact_inchikey",
                "two_dimensional_inchikey",
            )
        },
        "candidate_pool_diagnostics": {
            key: {"count": value, "fraction": value / count}
            for key, value in pool_duplicate_counts.items()
        },
        "warning": "The task verifier uses exact_candidate_id; other metrics are diagnostics only.",
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
