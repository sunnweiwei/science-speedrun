#!/usr/bin/env python3
"""Replay data-safe MIST-CF -> MIST retrieval under the task contract.

The molecular formula for each spectrum comes only from ``predict_mistcf.py``.
MIST predicts a Morgan4096 fingerprint.  The primary protocol binarizes it at
a validation-calibrated threshold and ranks with Tanimoto similarity, matching
the MIST retrieval protocol. Continuous-Tanimoto and cosine are diagnostics.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--candidate-pools", type=Path, required=True)
    parser.add_argument(
        "--formula-predictions", type=Path, nargs="+", required=True,
        help="One or more JSONL shards written by predict_mistcf.py.",
    )
    parser.add_argument("--mist-src", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--formula-top-k",
        type=int,
        default=1,
        choices=range(1, 6),
        metavar="{1,2,3,4,5}",
        help="Number of ranked MIST-CF formulae to encode for each spectrum.",
    )
    parser.add_argument(
        "--formula-aggregation",
        choices=["max", "mean"],
        default="max",
        help="Element-wise aggregation for multiple formula-conditioned fingerprints.",
    )
    parser.add_argument(
        "--similarity",
        choices=["tanimoto", "continuous_tanimoto", "cosine"],
        default="tanimoto",
    )
    parser.add_argument(
        "--fingerprint-output",
        type=Path,
        help="Optional operator-side cache of query IDs and continuous predictions.",
    )
    parser.add_argument(
        "--fingerprint-input",
        type=Path,
        help=(
            "Optional NPZ cache previously written by --fingerprint-output. "
            "When supplied, skip MIST model loading and spectrum inference."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.20346222817897797,
        help=(
            "Validation-calibrated fingerprint binarization threshold used by "
            "the released MassSpecGym MIST model. Applied only for Tanimoto."
        ),
    )
    return parser.parse_args()


def jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"non-object JSON at {path}:{line_number}")
                yield value


def load_formulas(paths: list[Path], top_k: int = 1) -> dict[str, list[str]]:
    predictions: dict[str, list[str]] = {}
    for path in paths:
        for row in jsonl(path):
            query_id = str(row["identifier"])
            formulas = row.get("formulas", [])
            if query_id in predictions:
                raise ValueError(f"duplicate formula prediction for {query_id}")
            if not formulas:
                raise ValueError(f"empty formula prediction for {query_id}")
            predictions[query_id] = [str(value) for value in formulas[:top_k]]
    return predictions


def load_model(args: argparse.Namespace):
    sys.path.insert(0, str(args.mist_src.resolve()))
    from mist.models.spectra_encoder import SpectraEncoderGrowing

    model = SpectraEncoderGrowing(
        form_embedder="pos-cos",
        output_size=4096,
        hidden_size=640,
        spectra_dropout=0.1,
        top_layers=1,
        refine_layers=4,
        magma_modulo=2048,
        peak_attn_layers=2,
        num_heads=8,
        set_pooling="cls",
        pairwise_featurization=True,
        embed_instrument=False,
        no_diffs=True,
    )
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    return model


def make_feature(row: dict, formula: str, utils) -> dict:
    """Build exactly the inference representation used by PeakFormula."""
    spectrum = np.column_stack(
        [np.asarray(row["mzs"], dtype=float), np.asarray(row["intensities"], dtype=float)]
    )
    spectrum = utils.process_spec_file(
        {"parentmass": float(row["precursor_mz"])}, [("ms2", spectrum)]
    )
    assignment = utils.get_output_dict(
        spec_name=str(row["query_id"]),
        spec=spectrum,
        form=formula,
        mass_diff_type="ppm",
        mass_diff_thresh=20,
        ion_type=str(row["adduct"]),
    )
    table = assignment["output_tbl"]
    if table is None:
        fragments: list[str] = []
        intensities: list[float] = []
        ions: list[str] = []
    else:
        order = np.argsort(np.asarray(table["ms2_inten"], dtype=float))[::-1][:10]
        fragments = np.asarray(table["formula"], dtype=object)[order].tolist()
        intensities = np.asarray(table["ms2_inten"], dtype=float)[order].tolist()
        ions = np.asarray(table["ions"], dtype=object)[order].tolist()

    root = utils.formula_to_dense(formula)
    forms = [utils.formula_to_dense(value) for value in fragments]
    peak_types = [0] * len(forms)
    ion_indices = [utils.get_ion_idx(value) for value in ions]
    # The released MassSpecGym model used PeakFormula(cls_type="ms1").
    forms.append(root)
    intensities.append(1.0)
    peak_types.append(3)
    ion_indices.append(utils.get_ion_idx(str(row["adduct"])))
    # The pinned FRIGID training labels retain MassSpecGym's raw strings
    # ("Orbitrap" and "QTOF"). They intentionally fall through the upstream
    # vocabulary to its shared unknown-instrument embedding. Remapping them to
    # the superficially similar "Orbitrap (LCMS)"/"Q-ToF (LCMS)" categories
    # changes the learned model input and materially lowers retrieval accuracy.
    instrument = str(row.get("instrument_type", ""))
    return {
        "name": str(row["query_id"]),
        "peak_type": np.asarray(peak_types),
        "form_vec": np.asarray(forms),
        "ion_vec": np.asarray(ion_indices),
        "frag_intens": np.asarray(intensities),
        "instrument": utils.get_instr_idx(instrument),
    }


def collate(features: list[dict], featurizers, device: torch.device) -> dict:
    batch = featurizers.PeakFormula.collate_fn(features)
    for key in ("types", "form_vec", "ion_vec", "intens", "num_peaks", "instruments"):
        batch[key] = batch[key].to(device)
    return batch


def aggregate_conditioned_fingerprints(
    conditioned: dict[str, list[tuple[int, np.ndarray]]], aggregation: str
) -> dict[str, np.ndarray]:
    if aggregation not in {"max", "mean"}:
        raise ValueError(f"unsupported formula aggregation: {aggregation}")
    output = {}
    for query_id, values in conditioned.items():
        if not values:
            raise ValueError(f"no conditioned fingerprints for {query_id}")
        matrix = np.stack([value for _, value in sorted(values)])
        output[query_id] = (
            matrix.max(axis=0) if aggregation == "max" else matrix.mean(axis=0)
        ).astype(np.float32, copy=False)
    return output


def predict_fingerprints(
    queries: list[dict], formulas: dict[str, str | list[str]], model, args: argparse.Namespace
) -> dict[str, np.ndarray]:
    from mist import utils
    from mist.data import featurizers
    import mist.utils.spectra_utils as spectra_utils

    cached_subsets = functools.lru_cache(maxsize=100_000)(spectra_utils.get_all_subsets)
    spectra_utils.get_all_subsets = cached_subsets
    utils.get_all_subsets = cached_subsets

    device = torch.device(args.device)
    model = model.to(device).eval()
    conditioned: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    started = time.time()
    formula_rows = [
        (row, formula, rank)
        for row in queries
        for rank, formula in enumerate(
            formulas[row["query_id"]]
            if isinstance(formulas[row["query_id"]], list)
            else [formulas[row["query_id"]]]
        )
    ]
    ordered_rows = sorted(
        formula_rows, key=lambda item: (item[1], item[0]["query_id"], item[2])
    )
    previous_formula = None
    with torch.no_grad():
        for offset in range(0, len(ordered_rows), args.batch_size):
            rows = ordered_rows[offset : offset + args.batch_size]
            features = []
            for row, formula, _ in rows:
                if formula != previous_formula:
                    cached_subsets.cache_clear()
                    previous_formula = formula
                features.append(make_feature(row, formula, utils))
            values, _ = model(collate(features, featurizers, device))
            for (row, _, rank), value in zip(rows, values.cpu().numpy()):
                conditioned[str(row["query_id"])].append(
                    (rank, value.astype(np.float32, copy=False))
                )
            if offset == 0 or offset + len(rows) == len(ordered_rows) or (offset // args.batch_size) % 10 == 0:
                print(
                    json.dumps(
                        {
                            "fingerprints": offset + len(rows),
                            "total": len(ordered_rows),
                            "elapsed_seconds": time.time() - started,
                        }
                    ),
                    flush=True,
                )
    aggregation = getattr(args, "formula_aggregation", "max")
    return aggregate_conditioned_fingerprints(conditioned, aggregation)


def fingerprint_matrix(candidates: list[dict], generator) -> torch.Tensor:
    rows = []
    for candidate in candidates:
        molecule = Chem.MolFromSmiles(candidate["smiles"])
        if molecule is None:
            raise ValueError(f"invalid candidate SMILES for {candidate['candidate_id']}")
        rows.append(generator.GetFingerprintAsNumPy(molecule))
    return torch.from_numpy(np.stack(rows).astype(np.float32, copy=False))


def load_fingerprint_cache(
    path: Path, queries: list[dict], args: argparse.Namespace
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as cache:
        fields = set(cache.files)
        legacy = fields == {"query_ids", "fingerprints"}
        expected_fields = {
            "query_ids", "fingerprints", "formula_top_k", "formula_aggregation",
        }
        if not legacy and fields != expected_fields:
            raise ValueError(f"unexpected fingerprint cache fields in {path}")
        if legacy and (args.formula_top_k != 1 or args.formula_aggregation != "max"):
            raise ValueError("legacy fingerprint cache is valid only for top-1/max")
        if not legacy and (
            int(cache["formula_top_k"]) != args.formula_top_k
            or str(cache["formula_aggregation"]) != args.formula_aggregation
        ):
            raise ValueError("fingerprint cache formula aggregation mismatch")
        query_ids = cache["query_ids"]
        values = cache["fingerprints"]
    if values.ndim != 2 or values.shape != (len(query_ids), 4096):
        raise ValueError(f"unexpected fingerprint cache shape {values.shape}")
    fingerprints = {
        str(query_id): value.astype(np.float32, copy=False)
        for query_id, value in zip(query_ids, values)
    }
    expected = {str(row["query_id"]) for row in queries}
    if set(fingerprints) != expected:
        missing = sorted(expected - set(fingerprints))
        extra = sorted(set(fingerprints) - expected)
        raise ValueError(
            f"fingerprint cache/query mismatch: missing={len(missing)}, extra={len(extra)}"
        )
    return fingerprints


def main() -> None:
    args = parse_args()
    started = time.time()
    RDLogger.DisableLog("rdApp.*")
    formulas = load_formulas(args.formula_predictions, args.formula_top_k)
    queries = list(jsonl(args.queries))
    if args.limit is not None:
        queries = queries[: args.limit]
    missing = sorted({row["query_id"] for row in queries} - set(formulas))
    if missing:
        raise ValueError(f"missing {len(missing)} formula predictions; first={missing[0]}")

    pools = {row["candidate_pool_id"]: row["candidates"] for row in jsonl(args.candidate_pools)}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in queries:
        grouped[str(row["candidate_pool_id"])].append(row)

    if args.fingerprint_input is not None:
        fingerprints = load_fingerprint_cache(args.fingerprint_input, queries, args)
    else:
        model = load_model(args)
        fingerprints = predict_fingerprints(queries, formulas, model, args)
        if args.fingerprint_output is not None:
            args.fingerprint_output.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                args.fingerprint_output,
                query_ids=np.asarray([row["query_id"] for row in queries]),
                fingerprints=np.stack([fingerprints[row["query_id"]] for row in queries]),
                formula_top_k=np.asarray(args.formula_top_k),
                formula_aggregation=np.asarray(args.formula_aggregation),
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    device = torch.device(args.device)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    ranking_started = time.time()
    rows_written = 0
    with temporary.open("w", encoding="utf-8") as stream, torch.no_grad():
        for pool_number, pool_id in enumerate(sorted(grouped), 1):
            candidates = pools[pool_id]
            candidate_fp = fingerprint_matrix(candidates, generator).to(device)
            candidate_sum = candidate_fp.sum(dim=1)[None, :]
            pool_queries = grouped[pool_id]
            query_fp = torch.from_numpy(
                np.stack([fingerprints[row["query_id"]] for row in pool_queries])
            ).to(device)
            if args.similarity == "tanimoto":
                # The audit protocol uses ordinary binary Tanimoto after a
                # per-model threshold is calibrated on validation on-bit
                # density. The released checkpoint's value is pinned below.
                query_fp = (query_fp > args.threshold).to(candidate_fp.dtype)
            intersections = query_fp @ candidate_fp.T
            if args.similarity in {"tanimoto", "continuous_tanimoto"}:
                scores = intersections / (
                    query_fp.sum(dim=1)[:, None] + candidate_sum - intersections
                ).clamp(min=1e-8)
            else:
                scores = intersections / (
                    query_fp.norm(dim=1)[:, None] * candidate_fp.norm(dim=1)[None, :]
                ).clamp(min=1e-8)
            for query, values in zip(pool_queries, scores.cpu().tolist()):
                order = sorted(
                    range(len(candidates)),
                    key=lambda index: (-values[index], candidates[index]["candidate_id"]),
                )
                stream.write(
                    json.dumps(
                        {
                            "query_id": query["query_id"],
                            "ranking": [candidates[index]["candidate_id"] for index in order],
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                rows_written += 1
            if pool_number % 100 == 0 or pool_number == len(grouped):
                print(
                    json.dumps(
                        {
                            "candidate_pools": pool_number,
                            "total_pools": len(grouped),
                            "rankings": rows_written,
                            "ranking_seconds": time.time() - ranking_started,
                        }
                    ),
                    flush=True,
                )
    os.replace(temporary, args.output)
    metrics = {
        "queries": len(queries),
        "candidate_pools": len(grouped),
        "similarity": args.similarity,
        "formula_top_k": args.formula_top_k,
        "formula_aggregation": args.formula_aggregation,
        "fingerprint_threshold": args.threshold if args.similarity == "tanimoto" else None,
        "checkpoint": str(args.checkpoint),
        "elapsed_seconds": time.time() - started,
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
