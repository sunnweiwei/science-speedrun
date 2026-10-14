#!/usr/bin/env python3
"""Run one DPLM-2 ModelBundle shard on FoldBench protein monomers.

This is a throughput-oriented evaluator adapter.  It preserves the exact model
loading, generation, and structure-decoder path used by ``contract_inference``
but loads the model once for a shard instead of once per target.  Target
sequences come only from the coordinate-free FoldBench input manifest.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any


TRACK = "monomer_protein"
MAX_SEQUENCE_LENGTH = 1022
SAFE_HEADER = re.compile(r"[^A-Za-z0-9_.-]")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--struct-tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def load_generation_module():
    path = Path(os.environ.get("DPLM_ROOT", "/opt/dplm")) / "generate_dplm2.py"
    spec = importlib.util.spec_from_file_location("pinned_dplm2_generation", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pinned DPLM-2 generation module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_target_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(rows[0]) != {"pdb_id", "chain_id"}:
        raise ValueError(f"unexpected {TRACK} target schema: {path}")
    return rows


def target_sequence(record: dict[str, Any], row: dict[str, str]) -> str:
    """Resolve the exact polymer entity referenced by a FoldBench target row."""

    matches = []
    for target_row in record.get("target_rows", []):
        if target_row.get("track") != TRACK:
            continue
        if all(str(target_row.get(key)) == value for key, value in row.items()):
            matches.append(target_row)
    if len(matches) != 1:
        raise ValueError(
            f"{row['pdb_id']}: expected one coordinate-free target-row match, "
            f"found {len(matches)}"
        )

    entity_ids = {
        str(reference["entity_id"])
        for reference in matches[0].get("referenced_chains", [])
        if reference.get("field") == "chain_id"
    }
    polymers = {
        str(polymer["entity_id"]): polymer
        for polymer in record.get("polymers", [])
        if polymer.get("molecule_type") == "protein"
    }
    selected = [polymers[entity_id] for entity_id in entity_ids if entity_id in polymers]
    if len(selected) != 1:
        raise ValueError(
            f"{row['pdb_id']}: target must resolve to exactly one protein entity"
        )
    sequence = str(selected[0]["sequence"]).upper()
    if not sequence:
        raise ValueError(f"{row['pdb_id']}: empty sequence")
    if len(sequence) > MAX_SEQUENCE_LENGTH:
        raise ValueError(
            f"{row['pdb_id']}: length {len(sequence)} exceeds "
            f"DPLM-2 limit {MAX_SEQUENCE_LENGTH}"
        )
    if not sequence.isalpha() or not sequence.isascii():
        raise ValueError(f"{row['pdb_id']}: unsupported sequence alphabet")
    return sequence


def prepare_targets(
    inputs: Path, targets: Path, num_shards: int, shard_index: int
) -> tuple[list[dict[str, Any]], int, list[dict[str, str]]]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("invalid shard configuration")
    records: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(inputs):
        key = str(record["pdb_id"]).lower()
        if key in records:
            raise ValueError(f"duplicate coordinate-free FoldBench input: {key}")
        records[key] = record

    target_rows = load_target_rows(targets)
    resolved = []
    rejected = []
    for ordinal, row in enumerate(target_rows):
        key = row["pdb_id"].lower()
        if key not in records:
            raise KeyError(f"FoldBench input missing target {row['pdb_id']}")
        try:
            sequence = target_sequence(records[key], row)
        except ValueError as error:
            rejected.append(
                {
                    "pdb_id": row["pdb_id"],
                    "chain_id": row["chain_id"],
                    "reason": str(error),
                }
            )
            continue
        resolved.append(
            {
                "ordinal": ordinal,
                "pdb_id": row["pdb_id"],
                "chain_id": row["chain_id"],
                "sequence": sequence,
            }
        )

    # Longest-first round-robin gives shards similar token workloads while the
    # assignment remains deterministic and independent of GPU availability.
    balanced = sorted(resolved, key=lambda item: (-len(item["sequence"]), item["ordinal"]))
    selected = [item for index, item in enumerate(balanced) if index % num_shards == shard_index]
    return selected, len(target_rows), rejected


def write_fasta(path: Path, targets: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_header = {}
    with path.open("w", encoding="utf-8") as handle:
        for item in targets:
            safe_id = SAFE_HEADER.sub("_", item["pdb_id"])
            header = f"fb{item['ordinal']:04d}_{safe_id}"
            if header in by_header:
                raise ValueError(f"duplicate native header: {header}")
            by_header[header] = item
            handle.write(f">{header}\n{item['sequence']}\n")
    return by_header


def main() -> None:
    args = arguments()
    if args.batch_size < 1 or args.max_iter < 1:
        raise ValueError("batch size and iteration count must be positive")
    for path in (args.model_dir, args.struct_tokenizer):
        if not path.is_dir():
            raise FileNotFoundError(path)

    selected, target_count, rejected = prepare_targets(
        args.inputs, args.targets, args.num_shards, args.shard_index
    )
    if not selected:
        raise ValueError("this shard contains no targets")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fasta = args.output_dir / f"inputs.shard-{args.shard_index}.fasta"
    by_header = write_fasta(fasta, selected)

    import torch
    from biotite.structure.io import load_structure, save_structure
    from byprot.models.dplm2.dplm2 import MultimodalDiffusionProteinLanguageModel
    from official_baselines.dplm2.tokenize_unified_view import load_tokenizer

    torch.manual_seed(args.seed + args.shard_index)
    torch.cuda.manual_seed_all(args.seed + args.shard_index)
    model = MultimodalDiffusionProteinLanguageModel.from_pretrained(
        str(args.model_dir),
        cfg_override={
            "tokenizer": {"vocab_file": str(args.model_dir)},
            "struct_tokenizer": {"exp_path": str(args.struct_tokenizer)},
        },
    ).eval().cuda()
    model._struct_tokenizer = load_tokenizer(
        args.struct_tokenizer, next(model.parameters()).device
    )
    generation_args = SimpleNamespace(
        task="folding",
        batch_size=args.batch_size,
        max_iter=args.max_iter,
        temperature=1.0,
        unmasking_strategy="stochastic1.0",
        sampling_strategy="annealing@2.0:0.1",
    )
    upstream = load_generation_module()
    batches, name_lists = upstream.initialize_conditional_generation(
        str(fasta),
        model.tokenizer,
        next(model.parameters()).device,
        generation_args,
        model=model,
    )
    native_dir = args.output_dir / f"native.shard-{args.shard_index}"
    prediction_dir = args.output_dir / "predictions"
    prediction_dir.mkdir(exist_ok=True)
    rows = []
    for batch_index, (batch, names) in enumerate(
        zip(batches, name_lists, strict=True)
    ):
        torch.manual_seed(args.seed + args.shard_index * 100_000 + batch_index)
        torch.cuda.manual_seed_all(args.seed + args.shard_index * 100_000 + batch_index)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model.generate(
                input_tokens=batch["input_tokens"],
                max_iter=args.max_iter,
                temperature=1.0,
                unmasking_strategy="stochastic1.0",
                sampling_strategy="annealing@2.0:0.1",
                partial_masks=batch["partial_mask"],
            )
        upstream.save_results(
            tokenizer=model.tokenizer,
            struct_tokenizer=model.struct_tokenizer,
            save_dir=str(native_dir),
            task="folding",
            outputs=outputs,
            headers=names,
            save_pdb=True,
            continue_write=True,
        )
        for header in names:
            item = by_header[header]
            source = native_dir / "pdb" / f"{header}.pdb"
            if not source.is_file():
                raise FileNotFoundError(source)
            prediction = (prediction_dir / f"{item['pdb_id']}.cif").resolve()
            save_structure(prediction, load_structure(source))
            rows.append(
                {
                    "pdb_id": item["pdb_id"],
                    "seed": args.seed,
                    "sample": 0,
                    "ranking_score": 0.0,
                    "prediction_path": str(prediction),
                }
            )

    rows.sort(key=lambda row: row["pdb_id"])
    reference = args.output_dir / f"prediction_reference.shard-{args.shard_index}.csv"
    with reference.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "schema_version": "foldbench-inference-shard-audit/v1",
        "model_id": "dplm2_50m",
        "track": TRACK,
        "coordinate_free_inputs": str(args.inputs),
        "target_count": target_count,
        "supported_target_count": target_count - len(rejected),
        "rejected_targets": rejected,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "selected_targets": len(selected),
        "predictions_written": len(rows),
        "max_iter": args.max_iter,
        "seed": args.seed,
    }
    (args.output_dir / f"inference_audit.shard-{args.shard_index}.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
