"""Extract a real validation spectrum and its archived GLACIER candidate ranking."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

QUERY_ID = "MassSpecGymID0079471"


def find_row(path: Path, key: str, value: str) -> tuple[dict[str, Any], str]:
    with path.open("rb") as handle:
        for line in handle:
            row = json.loads(line)
            if row[key] == value:
                return row, hashlib.sha256(line).hexdigest()
    raise ValueError(f"Missing {key}={value} in {path.name}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def extract(development: Path, rankings: Path) -> dict[str, Any]:
    query, query_hash = find_row(development / "validation.jsonl", "query_id", QUERY_ID)
    pool, pool_hash = find_row(development / "validation_candidate_pools.jsonl", "candidate_pool_id", query["candidate_pool_id"])
    prediction, prediction_hash = find_row(rankings, "query_id", QUERY_ID)
    manifest = json.loads((development / "manifest.json").read_text())
    candidates = {item["candidate_id"]: item for item in pool["candidates"]}
    target = hashlib.sha256(query["target_smiles"].encode()).hexdigest()
    assert candidates[target]["smiles"] == query["target_smiles"]
    assert len(set(prediction["ranking"])) == len(candidates)
    assert set(prediction["ranking"]) == set(candidates)
    selected_ids = [target, *prediction["ranking"][:2]]
    assert len(set(selected_ids)) == 3
    return {
        "schema": "massspec-science-example/v1",
        "split": "validation",
        "selection": "First available archived GLACIER validation query among early source records with precursor m/z below 260, at least eight peaks, [M+H]+ adduct, and at most 22 heavy atoms; selected before checking the target rank",
        "source": manifest["source"],
        "source_url": "https://huggingface.co/datasets/roman-bushuiev/MassSpecGym",
        "source_license": "MIT",
        "source_attribution": "MassSpecGym developers, 2024",
        "source_file_sha256_from_manifest": {name: manifest["files"][name] for name in ["validation.jsonl", "validation_candidate_pools.jsonl"]},
        "source_record_sha256": {"validation_query_line": query_hash, "validation_candidate_pool_line": pool_hash, "validation_ranking_line": prediction_hash},
        "ranking_source": {"run": "glacier_full_validation/scored_entropy_all/rankings.jsonl", "sha256": file_sha256(rankings), "reference": "Official GLACIER MassSpecGym no-CF checkpoint; archived validation replay", "query_id": QUERY_ID},
        "query": query,
        "candidate_count": len(candidates),
        "shown_candidates": [dict(candidates[identifier], display_name=letter, rank=prediction["ranking"].index(identifier) + 1) for letter, identifier in zip("ABC", selected_ids, strict=True)],
        "correct_candidate_id": target,
        "correct_candidate_rank": prediction["ranking"].index(target) + 1,
        "notes": ["Candidate structures are three examples from the actual mass-compatible pool.", "All plotted peaks and intensities are from the materialized validation record; intensities are normalized to the largest peak.", "The actual target is rank 22. This figure shows a retrieval error and does not claim benchmark-wide performance.", "No test sample, test label, or trained weight is included."],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("development", type=Path)
    parser.add_argument("rankings", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("example.json"))
    args = parser.parse_args()
    args.output.write_text(json.dumps(extract(args.development, args.rankings), indent=2) + "\n")


if __name__ == "__main__":
    main()
