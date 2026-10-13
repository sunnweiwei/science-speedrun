#!/usr/bin/env python3
"""Merge complete-domain GLACIER rankings with a complementary fallback.

GLACIER's released MassSpecGym checkpoint was trained for spectra with a known
collision energy, an Orbitrap or QTOF instrument, and the ``[M+H]+`` adduct.
This adapter uses
GLACIER only on that released domain and requires fallback rankings everywhere
else. The fallback may additionally cover GLACIER queries; those rows are not
used.
"""

import argparse
import json
from pathlib import Path

from glacier_contract import SELECTION_RULE, is_released_domain


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            yield value


def load_rankings(path: Path, expected_candidates: dict, *, complete: bool) -> dict:
    rankings = {}
    for row in read_jsonl(path):
        if set(row) != {"query_id", "ranking"}:
            raise ValueError(f"invalid ranking fields in {path}")
        query_id = row["query_id"]
        ranking = row["ranking"]
        candidates = expected_candidates.get(query_id)
        if candidates is None or query_id in rankings:
            raise ValueError(f"unknown or duplicate query {query_id} in {path}")
        if (
            not isinstance(ranking, list)
            or len(ranking) != len(candidates)
            or set(ranking) != candidates
        ):
            raise ValueError(f"ranking for {query_id} is not an exact candidate permutation")
        rankings[query_id] = ranking
    if complete and set(rankings) != set(expected_candidates):
        missing = sorted(set(expected_candidates) - set(rankings))
        raise ValueError(f"fallback does not cover every query; first missing: {missing[:5]}")
    return rankings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--glacier", required=True, type=Path)
    parser.add_argument("--fallback", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--glacier-query-id-file", type=Path,
                        help="Native starter support subset; defaults to the full released domain.")
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    queries = {row["query_id"]: row for row in read_jsonl(input_root / "queries.jsonl")}
    pools = {
        row["candidate_pool_id"]: {
            candidate["candidate_id"] for candidate in row["candidates"]
        }
        for row in read_jsonl(input_root / "candidate_pools.jsonl")
    }
    expected_candidates = {
        query_id: pools[query["candidate_pool_id"]]
        for query_id, query in queries.items()
    }
    fallback = load_rankings(args.fallback.resolve(), expected_candidates, complete=False)
    glacier = load_rankings(args.glacier.resolve(), expected_candidates, complete=False)

    eligible_queries = {
        query_id for query_id, query in queries.items() if is_released_domain(query)
    }
    if args.glacier_query_id_file is not None:
        selected = set(args.glacier_query_id_file.read_text().splitlines())
        if not selected <= eligible_queries:
            raise ValueError('native support subset includes unknown or out-of-domain queries')
        eligible_queries = selected
    for query_id in glacier:
        if query_id not in eligible_queries:
            raise ValueError(f"GLACIER ranking is outside its published domain: {query_id}")
    missing_glacier = sorted(eligible_queries - set(glacier))
    if missing_glacier:
        raise ValueError(f"GLACIER does not cover its released domain: {missing_glacier[:5]}")
    fallback_required = set(queries) - eligible_queries
    missing_fallback = sorted(fallback_required - set(fallback))
    if missing_fallback:
        raise ValueError(f"fallback does not cover its required domain: {missing_fallback[:5]}")

    output_root = args.output_root.resolve()
    if output_root.exists():
        if not output_root.is_dir() or any(output_root.iterdir()):
            raise FileExistsError(f"output root is not an empty directory: {output_root}")
    else:
        output_root.mkdir(parents=True)
    with (output_root / "rankings.jsonl").open("w", encoding="utf-8") as stream:
        for query_id in queries:
            row = {
                "query_id": query_id,
                "ranking": (
                    glacier[query_id]
                    if query_id in glacier
                    else fallback[query_id]
                ),
            }
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    report = {
        "schema": "glacier-mist-hybrid/v1",
        "queries": len(queries),
        "glacier_queries": len(glacier),
        "fallback_queries": len(queries) - len(glacier),
        "selection_rule": SELECTION_RULE,
    }
    if args.glacier_query_id_file is not None:
        report['selection_rule'] += '; intersect with public native support subset'
    (output_root / "runtime.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
