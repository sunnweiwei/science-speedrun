#!/usr/bin/env python3
"""Run the starter model and rank every candidate in each query pool."""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from common import (
    compact_json, load_candidate_pools, morgan_fingerprint, read_jsonl,
    sha256, spectrum_tensor,
)
from model import DeepSetsFingerprint


def contained(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("manifest path escapes package")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    manifest = json.loads((package / "model_manifest.json").read_text())
    checkpoint_path = contained(package, manifest["checkpoint"]["path"])
    if sha256(checkpoint_path) != manifest["checkpoint"]["sha256"]:
        raise ValueError("checkpoint checksum mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["model_config"]
    declared_config = json.loads(
        contained(package, manifest["model_config"]).read_text()
    )
    if declared_config != config:
        raise ValueError("checkpoint and declared model config disagree")
    model = DeepSetsFingerprint(
        hidden_size=config["hidden_size"],
        fingerprint_size=config["fingerprint_size"],
        layers_per_mlp=config["layers_per_mlp"],
        dropout=config["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()

    input_root = args.input_root.resolve()
    queries = read_jsonl(input_root / "queries.jsonl")
    pools = load_candidate_pools(input_root / "candidate_pools.jsonl")
    grouped = defaultdict(list)
    query_ids = set()
    for row in queries:
        query_id = row.get("query_id")
        pool_id = row.get("candidate_pool_id")
        if not isinstance(query_id, str) or query_id in query_ids:
            raise ValueError("missing or duplicate query ID")
        if pool_id not in pools:
            raise ValueError(f"unknown candidate pool for {query_id}")
        query_ids.add(query_id)
        grouped[pool_id].append(row)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if any(output_root.iterdir()):
        raise ValueError("output directory must start empty")
    temporary = output_root / "rankings.jsonl.tmp"
    with temporary.open("w", encoding="utf-8") as stream, torch.no_grad():
        for pool_id in sorted(grouped):
            candidates = pools[pool_id]
            candidate_fingerprints = torch.stack([
                morgan_fingerprint(
                    candidate["smiles"],
                    size=config["fingerprint_size"],
                    radius=config["fingerprint_radius"],
                    allow_unparseable=True,
                )
                for candidate in candidates
            ]).to(device)
            candidate_fingerprints = F.normalize(candidate_fingerprints, dim=-1)
            pool_queries = grouped[pool_id]
            for offset in range(0, len(pool_queries), 256):
                batch = pool_queries[offset:offset + 256]
                spectra = torch.stack([
                    spectrum_tensor(
                        row,
                        max_peaks=config["max_peaks"],
                        mz_min=config["mz_min"],
                        mz_max=config["mz_max"],
                    )
                    for row in batch
                ]).to(device)
                predictions = F.normalize(model(spectra), dim=-1)
                scores = predictions @ candidate_fingerprints.T
                for row, row_scores in zip(batch, scores.cpu().tolist()):
                    order = sorted(
                        range(len(candidates)),
                        key=lambda index: (-row_scores[index], candidates[index]["candidate_id"]),
                    )
                    stream.write(compact_json({
                        "query_id": row["query_id"],
                        "ranking": [candidates[index]["candidate_id"] for index in order],
                    }) + "\n")
    os.replace(temporary, output_root / "rankings.jsonl")


if __name__ == "__main__":
    main()
