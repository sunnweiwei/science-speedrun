#!/usr/bin/env python3
"""Stage labeled development validation data as contract-compatible input."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-size", type=int, help="Uniform deterministic spectrum sample, instead of a file prefix")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--query-id-file",
        type=Path,
        help="Optional newline-delimited allowlist of validation query IDs.",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    if sum(value is not None for value in [args.limit, args.sample_size, args.query_id_file]) > 1:
        raise ValueError('choose one of limit, sample-size or query-id-file')
    selected_query_ids = None
    if args.query_id_file is not None:
        selected_query_ids = {
            line.strip()
            for line in args.query_id_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if not selected_query_ids:
            raise ValueError("query ID allowlist is empty")
    if args.sample_size is not None:
        identifiers = [json.loads(line)['query_id'] for line in (args.input_root / 'validation.jsonl').open() if line.strip()]
        if not 1 <= args.sample_size <= len(identifiers):
            raise ValueError('sample-size must fit the validation fold')
        selected_query_ids = set(random.Random(args.seed).sample(sorted(identifiers), args.sample_size))
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    queries = output / "queries.jsonl"
    selected_pools = set()
    query_count = 0
    with (args.input_root / "validation.jsonl").open() as source, queries.open("w") as target:
        for line in source:
            if args.limit is not None and query_count >= args.limit:
                break
            row = json.loads(line)
            if (
                selected_query_ids is not None
                and row["query_id"] not in selected_query_ids
            ):
                continue
            for key in ("target_smiles", "target_inchikey", "target_formula"):
                row.pop(key)
            target.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            selected_pools.add(row["candidate_pool_id"])
            query_count += 1
    if selected_query_ids is not None and query_count != len(selected_query_ids):
        found = {
            json.loads(line)["query_id"]
            for line in queries.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        missing = sorted(selected_query_ids - found)
        raise ValueError(f"unknown validation query IDs: {missing[:5]}")
    pools = output / "candidate_pools.jsonl"
    if args.limit is None and selected_query_ids is None:
        shutil.copyfile(args.input_root / "validation_candidate_pools.jsonl", pools)
        pool_count = sum(1 for line in pools.open() if line.strip())
    else:
        pool_count = 0
        with (args.input_root / "validation_candidate_pools.jsonl").open() as source, pools.open("w") as target:
            for line in source:
                row = json.loads(line)
                if row["candidate_pool_id"] in selected_pools:
                    target.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                    pool_count += 1
    manifest = {
        "schema": "massspecgym-retrieval-input/v1",
        "source": "development validation",
        "counts": {"queries": query_count, "candidate_pools": pool_count},
        "files": {
            "queries.jsonl": sha256(queries),
            "candidate_pools.jsonl": sha256(pools),
        },
    }
    if args.sample_size is not None:
        manifest['selection'] = {'method': 'uniform spectrum sample', 'size': args.sample_size, 'seed': args.seed}
    (output / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    )


if __name__ == "__main__":
    main()
