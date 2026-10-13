#!/usr/bin/env python3
"""Write a spectrum-free ranking that preserves each supplied candidate order."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSON at {path}:{line_number}")
            yield value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--candidate-pools", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    pools = {}
    for row in jsonl(args.candidate_pools):
        pool_id = str(row["candidate_pool_id"])
        if pool_id in pools:
            raise ValueError(f"duplicate candidate pool: {pool_id}")
        ranking = [str(candidate["candidate_id"]) for candidate in row["candidates"]]
        if not ranking or len(ranking) != len(set(ranking)):
            raise ValueError(f"invalid candidate pool: {pool_id}")
        pools[pool_id] = ranking

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    seen = set()
    with temporary.open("w", encoding="utf-8") as stream:
        for row in jsonl(args.queries):
            query_id = str(row["query_id"])
            pool_id = str(row["candidate_pool_id"])
            if query_id in seen or pool_id not in pools:
                raise ValueError(f"duplicate query or unknown pool: {query_id}")
            seen.add(query_id)
            stream.write(json.dumps(
                {"query_id": query_id, "ranking": pools[pool_id]},
                sort_keys=True,
                separators=(",", ":"),
            ) + "\n")
    if not seen:
        raise ValueError("query input is empty")
    os.replace(temporary, args.output)
    print(json.dumps({"queries": len(seen), "candidate_pools": len(pools)}))


if __name__ == "__main__":
    main()
