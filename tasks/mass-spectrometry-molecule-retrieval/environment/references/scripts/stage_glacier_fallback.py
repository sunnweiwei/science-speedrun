#!/usr/bin/env python3
"""Stage only queries outside the released GLACIER domain for a fallback."""

import argparse
import hashlib
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    selected_pools = set()
    selected_queries = 0
    query_path = output_root / "queries.jsonl"
    with query_path.open("w", encoding="utf-8") as stream:
        for query in read_jsonl(input_root / "queries.jsonl"):
            if is_released_domain(query):
                continue
            pool_id = query.get("candidate_pool_id")
            if not isinstance(pool_id, str):
                raise ValueError("query has no candidate_pool_id")
            selected_pools.add(pool_id)
            stream.write(json.dumps(query, sort_keys=True, separators=(",", ":")) + "\n")
            selected_queries += 1

    found_pools = set()
    pool_path = output_root / "candidate_pools.jsonl"
    with pool_path.open("w", encoding="utf-8") as stream:
        for pool in read_jsonl(input_root / "candidate_pools.jsonl"):
            pool_id = pool.get("candidate_pool_id")
            if pool_id not in selected_pools:
                continue
            if pool_id in found_pools:
                raise ValueError(f"duplicate candidate pool: {pool_id}")
            found_pools.add(pool_id)
            stream.write(json.dumps(pool, sort_keys=True, separators=(",", ":")) + "\n")
    if found_pools != selected_pools:
        missing = sorted(selected_pools - found_pools)
        raise ValueError(f"missing candidate pools: {missing[:5]}")

    manifest = {
        "schema": "massspecgym-retrieval-input/v1",
        "source": "released GLACIER domain fallback partition",
        "counts": {
            "queries": selected_queries,
            "candidate_pools": len(found_pools),
        },
        "files": {
            "queries.jsonl": sha256(query_path),
            "candidate_pools.jsonl": sha256(pool_path),
        },
        "selection_rule": SELECTION_RULE,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
