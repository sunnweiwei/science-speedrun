#!/usr/bin/env python3
"""Project the frozen unified split into DPLM sequence/folding data views."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


FOLDING_TASKS = frozenset({"monomer_protein", "specialist_disorder"})
CANONICAL_DPLM_AA = frozenset("LAGVSERTIDPKQNFYMHWCXBUZO")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protein_sequences(record: dict) -> list[dict]:
    return [
        sequence
        for sequence in record.get("sequences", [])
        if sequence.get("molecule_type") == "protein"
    ]


def main() -> None:
    args = arguments()
    from datasets import Dataset

    args.output_root.mkdir(parents=True, exist_ok=True)
    sequence_root = args.output_root / "sequence"
    staging_root = args.output_root / "folding_staging"
    sequence_root.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=True)
    seen_training_sequences: set[str] = set()
    counters: dict[str, Counter] = {}
    source_digests: dict[str, str] = {}

    for split in ("train", "valid"):
        source_path = args.split_dir / f"{split}.jsonl"
        source_digests[split] = sha256(source_path)
        counts = counters[split] = Counter()
        sequence_rows: list[dict] = []
        seen_in_split: set[str] = set()
        folding_path = staging_root / f"{split}.jsonl"
        with source_path.open() as source, folding_path.open("w") as folding:
            for line in source:
                record = json.loads(line)
                proteins = protein_sequences(record)
                counts["global_examples"] += 1
                counts["protein_chains"] += len(proteins)
                for protein in proteins:
                    sequence = protein["sequence"]
                    digest = protein.get("sha256") or hashlib.sha256(
                        sequence.encode()
                    ).hexdigest()
                    if len(sequence) > args.max_length:
                        counts["sequence_too_long"] += 1
                        continue
                    if not set(sequence) <= CANONICAL_DPLM_AA:
                        counts["sequence_unsupported_alphabet"] += 1
                        continue
                    if digest in seen_in_split:
                        counts["sequence_duplicate_within_split"] += 1
                        continue
                    if split == "valid" and digest in seen_training_sequences:
                        counts["validation_sequence_seen_in_train"] += 1
                        continue
                    seen_in_split.add(digest)
                    sequence_rows.append(
                        {
                            "seq": sequence,
                            "length": len(sequence),
                            "sequence_sha256": digest,
                        }
                    )
                if record.get("task") not in FOLDING_TASKS:
                    counts["folding_unsupported_task"] += 1
                    continue
                if len(proteins) != 1 or len(record.get("sequences", [])) != 1:
                    counts["folding_not_single_protein"] += 1
                    continue
                protein = proteins[0]
                sequence = protein["sequence"]
                if len(sequence) > args.max_length:
                    counts["folding_too_long"] += 1
                    continue
                if not set(sequence) <= CANONICAL_DPLM_AA:
                    counts["folding_unsupported_alphabet"] += 1
                    continue
                projected = {
                    "example_id": record["example_id"],
                    "pdb_name": record["example_id"],
                    "aa_seq": sequence,
                    "length": len(sequence),
                    "chain_id": protein["chain_id"],
                    "structure_path": record["structure_path"],
                    "source": record["source"],
                    "source_id": record["source_id"],
                    "unified_split": split,
                    "unified_split_group": record["unified_split_group"],
                }
                folding.write(json.dumps(projected, sort_keys=True) + "\n")
                counts["folding_supported"] += 1
        if split == "train":
            seen_training_sequences = seen_in_split
        Dataset.from_list(sequence_rows).save_to_disk(str(sequence_root / split))
        counts["sequence_supported_unique"] = len(sequence_rows)
        counts["sequence_output_sha256"] = sha256(folding_path)

    sequence_audit = {
        "schema_version": "foldingtrain-dplm-sequence-adapter-v1",
        "source_split_schema": "foldingtrain-unified-split-v1",
        "source_split_digests": source_digests,
        "max_length": args.max_length,
        "counts": {split: dict(value) for split, value in counters.items()},
        "data_policy": "same frozen FoldingTrain records; no UniRef/PDB-SwissProt substitution",
    }
    (sequence_root / "adapter_audit.json").write_text(
        json.dumps(sequence_audit, indent=2, sort_keys=True) + "\n"
    )
    (staging_root / "adapter_audit.json").write_text(
        json.dumps(sequence_audit, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(sequence_audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
