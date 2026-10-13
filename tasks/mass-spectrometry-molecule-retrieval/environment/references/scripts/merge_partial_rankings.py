#!/usr/bin/env python3
"""Validate and merge disjoint partial task-contract ranking files."""

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--rankings", required=True, type=Path, nargs="+")
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    queries = list(read_jsonl(input_root / "queries.jsonl"))
    pools = {
        row["candidate_pool_id"]: {
            candidate["candidate_id"] for candidate in row["candidates"]
        }
        for row in read_jsonl(input_root / "candidate_pools.jsonl")
    }
    expected = {
        query["query_id"]: pools[query["candidate_pool_id"]]
        for query in queries
    }
    merged = {}
    for path in args.rankings:
        for row in read_jsonl(path.resolve()):
            if set(row) != {"query_id", "ranking"}:
                raise ValueError(f"invalid ranking fields in {path}")
            query_id = row["query_id"]
            candidates = expected.get(query_id)
            if candidates is None or query_id in merged:
                raise ValueError(f"unknown or duplicate query {query_id}")
            ranking = row["ranking"]
            if (
                not isinstance(ranking, list)
                or len(ranking) != len(candidates)
                or set(ranking) != candidates
            ):
                raise ValueError(f"invalid candidate permutation for {query_id}")
            merged[query_id] = ranking

    output_root = args.output_root.resolve()
    if output_root.exists():
        if not output_root.is_dir() or any(output_root.iterdir()):
            raise FileExistsError(f"output root is not an empty directory: {output_root}")
    else:
        output_root.mkdir(parents=True)
    with (output_root / "rankings.jsonl").open("w", encoding="utf-8") as stream:
        for query in queries:
            query_id = query["query_id"]
            if query_id in merged:
                stream.write(
                    json.dumps(
                        {"query_id": query_id, "ranking": merged[query_id]},
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
    report = {
        "schema": "massspecgym-partial-ranking-merge/v1",
        "input_files": len(args.rankings),
        "ranked_queries": len(merged),
        "contract_queries": len(expected),
    }
    (output_root / "runtime.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
