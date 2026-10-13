#!/usr/bin/env python3
"""Split a GLACIER candidate table into query-intact round-robin shards."""

import argparse
import csv
import json
from contextlib import ExitStack
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--num-shards", required=True, type=int)
    args = parser.parse_args()
    if args.num_shards < 1:
        raise ValueError("num-shards must be positive")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    query_counts = [0] * args.num_shards
    row_counts = [0] * args.num_shards
    seen_queries = set()
    current_query = None
    shard_index = -1

    with args.input.resolve().open(encoding="utf-8", newline="") as source, ExitStack() as stack:
        reader = csv.DictReader(source, delimiter="\t")
        if reader.fieldnames is None or "spec" not in reader.fieldnames:
            raise ValueError("candidate table is missing the spec column")
        writers = []
        for index in range(args.num_shards):
            path = output_root / f"candidates_{index:02d}.tsv"
            stream = stack.enter_context(path.open("w", encoding="utf-8", newline=""))
            writer = csv.DictWriter(stream, fieldnames=reader.fieldnames, delimiter="\t")
            writer.writeheader()
            writers.append(writer)

        for row in reader:
            query_id = row["spec"]
            if query_id != current_query:
                if query_id in seen_queries:
                    raise ValueError(f"query rows are not contiguous: {query_id}")
                seen_queries.add(query_id)
                current_query = query_id
                shard_index = (shard_index + 1) % args.num_shards
                query_counts[shard_index] += 1
            writers[shard_index].writerow(row)
            row_counts[shard_index] += 1

    report = {
        "schema": "glacier-candidate-shards/v1",
        "num_shards": args.num_shards,
        "queries": sum(query_counts),
        "candidate_rows": sum(row_counts),
        "query_counts": query_counts,
        "candidate_row_counts": row_counts,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
