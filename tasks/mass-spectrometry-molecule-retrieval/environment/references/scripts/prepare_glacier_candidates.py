#!/usr/bin/env python3
"""Convert the benchmark JSONL contract to the table consumed by GLACIER.

This adapter is deliberately answer-free: it reads only public query metadata
and public candidate SMILES. It is useful for faithful released-checkpoint
replays with the pinned upstream ``predict_smis_joint.py`` entry point.
"""

import argparse
import csv
import json
import math
from pathlib import Path

from glacier_contract import is_released_domain


FIELDS = (
    "dataset",
    "spec",
    "ionization",
    "smiles",
    "instrument",
    "collision_energies",
    "precursor",
    "candidate_id",
)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            yield value


def load_pools(path: Path) -> dict:
    pools = {}
    for row in read_jsonl(path):
        pool_id = row.get("candidate_pool_id")
        candidates = row.get("candidates")
        if not isinstance(pool_id, str) or not isinstance(candidates, list):
            raise ValueError(f"invalid candidate-pool row in {path}")
        if pool_id in pools:
            raise ValueError(f"duplicate candidate pool: {pool_id}")
        pools[pool_id] = candidates
    return pools


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument(
        "--eligible-only",
        action="store_true",
        help=(
            "Keep the released GLACIER domain: [M+H]+ with known collision "
            "energy and an Orbitrap or QTOF instrument."
        ),
    )
    parser.add_argument(
        "--one-per-pool",
        action="store_true",
        help="Select at most one query per molecular candidate pool for diverse probes.",
    )
    parser.add_argument(
        "--query-id-file",
        type=Path,
        help="Optional newline-delimited allowlist of query IDs.",
    )
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    allow = None
    if args.query_id_file is not None:
        allow = {
            line.strip()
            for line in args.query_id_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    pools = load_pools(input_root / "candidate_pools.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    query_count = 0
    candidate_count = 0
    seen_pools = set()
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        for query in read_jsonl(input_root / "queries.jsonl"):
            query_id = query.get("query_id")
            if allow is not None and query_id not in allow:
                continue
            collision_energy = query.get("collision_energy")
            eligible = is_released_domain(query)
            if args.eligible_only and not eligible:
                continue
            pool_id = query.get("candidate_pool_id")
            if args.one_per_pool and pool_id in seen_pools:
                continue
            if not isinstance(collision_energy, (int, float)) or not math.isfinite(collision_energy):
                raise ValueError(
                    f"query {query_id} has no finite collision energy; choose an explicit "
                    "imputation strategy before GLACIER prediction"
                )
            candidates = pools.get(pool_id)
            if candidates is None:
                raise ValueError(f"missing candidate pool for {query_id}")
            instrument = query.get("instrument_type") or "Unknown"
            for candidate in candidates:
                writer.writerow(
                    {
                        "dataset": "MassSpecGym",
                        "spec": query_id,
                        "ionization": query["adduct"],
                        "smiles": candidate["smiles"],
                        "instrument": instrument,
                        "collision_energies": repr([float(collision_energy)]),
                        "precursor": query["precursor_mz"],
                        "candidate_id": candidate["candidate_id"],
                    }
                )
                candidate_count += 1
            query_count += 1
            seen_pools.add(pool_id)
            if args.max_queries is not None and query_count >= args.max_queries:
                break

    print(
        json.dumps(
            {
                "schema": "glacier-candidate-adapter/v1",
                "queries": query_count,
                "candidate_rows": candidate_count,
                "eligible_only": args.eligible_only,
                "one_per_pool": args.one_per_pool,
                "answer_fields_read": [],
                "output": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
