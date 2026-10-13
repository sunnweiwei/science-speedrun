#!/usr/bin/env python3
"""Calibrate MIST fingerprint binarization using visible validation labels only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator


def jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"non-object JSON at {path}:{line_number}")
                yield row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--mist-src", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    script_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_root))
    from predict_mist import load_model, predict_fingerprints

    rows = list(jsonl(args.validation))
    required = {
        "query_id", "mzs", "intensities", "precursor_mz", "adduct",
        "instrument_type", "target_formula", "target_smiles",
    }
    if any(not required.issubset(row) for row in rows):
        raise ValueError("validation rows do not contain the required labels and inputs")

    model_args = SimpleNamespace(
        mist_src=args.mist_src,
        checkpoint=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
    )
    formulas = {str(row["query_id"]): str(row["target_formula"]) for row in rows}
    model = load_model(model_args)
    predictions = predict_fingerprints(rows, formulas, model, model_args)

    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
    reference_on_bits = 0
    for row in rows:
        molecule = Chem.MolFromSmiles(str(row["target_smiles"]))
        if molecule is None:
            raise ValueError(f"invalid validation SMILES for {row['query_id']}")
        reference_on_bits += int(generator.GetFingerprint(molecule).GetNumOnBits())

    values = np.concatenate([predictions[str(row["query_id"])] for row in rows])
    bit_count = int(values.size)
    if not 0 < reference_on_bits < bit_count:
        raise ValueError("invalid validation reference on-bit count")
    split_index = bit_count - reference_on_bits
    threshold = float(np.partition(values, split_index)[split_index])
    predicted_on_bits = int(np.count_nonzero(values > threshold))
    report = {
        "schema": "massspecgym-retrieval-mist-threshold/v1",
        "validation_queries": len(rows),
        "fingerprint_bits": 4096,
        "reference_on_bits": reference_on_bits,
        "reference_on_bit_density": reference_on_bits / bit_count,
        "threshold": threshold,
        "comparison": "strictly_greater_than",
        "predicted_on_bits": predicted_on_bits,
        "predicted_on_bit_density": predicted_on_bits / bit_count,
        "formula_source": "visible_validation_target_formula",
        "formula_aggregation": "single_formula",
        "checkpoint": str(args.checkpoint),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
