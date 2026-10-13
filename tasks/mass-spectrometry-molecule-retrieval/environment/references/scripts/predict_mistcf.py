#!/usr/bin/env python3
"""Replay the public data-safe MassSpecGym MIST-CF formula annotator.

The normal, contract-conforming input is the unlabelled ``queries.jsonl`` file.
For operator-side audits, an upstream labelled TSV can instead be supplied to
select a fold and compute formula-accuracy diagnostics.  Candidate generation
and model inference never use the ``formula`` column.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch


INSTRUMENT_REMAP = {
    "Orbitrap": "Orbitrap (LCMS)",
    "QTOF": "Q-ToF (LCMS)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--queries", type=Path)
    inputs.add_argument("--tsv", type=Path)
    parser.add_argument("--fold", choices=["train", "val", "test"])
    parser.add_argument("--mist-cf-src", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fast-filter-checkpoint", type=Path, required=True)
    parser.add_argument("--sirius", type=Path, required=True)
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ppm", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--sirius-cores", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-candidates-only", action="store_true")
    return parser.parse_args()


def configure_imports(args: argparse.Namespace):
    os.environ["SIRIUS_PATH"] = str(args.sirius.resolve())
    sys.path.insert(0, str(args.mist_cf_src.resolve()))

    import mist_cf.common as common
    import mist_cf.common.chem_utils as chem_utils
    from mist_cf.decomp.sirius_decomp import run_sirius
    from mist_cf.fast_form_score.fast_form_model import FastFFN
    from mist_cf.mist_cf_score.mist_cf_data import JsonExtractor, PredDataset
    from mist_cf.mist_cf_score.mist_cf_model import MistNet

    # Subformula enumeration depends only on the candidate formula and is much
    # more expensive than matching a spectrum against the resulting masses.
    # The upstream function returns immutable-by-convention arrays, so caching
    # it preserves exact model inputs while avoiding repeated work for replicate
    # spectra and shared mass candidates.
    cached_subsets = functools.lru_cache(maxsize=100_000)(chem_utils.get_all_subsets)
    # assign_subforms resolves this name in chem_utils' module globals, not via
    # mist_cf.common's re-export, so patch both references.
    chem_utils.get_all_subsets = cached_subsets
    common.get_all_subsets = cached_subsets
    common._reproduction_subset_cache = cached_subsets
    return common, run_sirius, FastFFN, JsonExtractor, PredDataset, MistNet


def cache_key(adduct: str, precursor_mz: float) -> str:
    return f"{adduct}\t{float(precursor_mz):.4f}"


def load_rows(args: argparse.Namespace) -> pd.DataFrame:
    if args.tsv is not None:
        if args.fold is None:
            raise ValueError("--fold is required with --tsv")
        usecols = [
            "identifier",
            "mzs",
            "intensities",
            "formula",
            "precursor_mz",
            "adduct",
            "instrument_type",
            "fold",
        ]
        frame = pd.read_csv(args.tsv, sep="\t", usecols=usecols)
        frame = frame.loc[frame["fold"].eq(args.fold)].reset_index(drop=True)
    else:
        records = []
        with args.queries.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(
                        f"non-object JSON at {args.queries}:{line_number}"
                    )
                records.append(
                    {
                        "identifier": str(row["query_id"]),
                        "mzs": row["mzs"],
                        "intensities": row["intensities"],
                        "precursor_mz": row["precursor_mz"],
                        "adduct": row["adduct"],
                        "instrument_type": row.get("instrument_type", ""),
                    }
                )
        frame = pd.DataFrame.from_records(records)
    if args.limit is not None:
        frame = frame.iloc[: args.limit].copy()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    # Keep replicate spectra with the same precursor/adduct on one worker.  Their
    # candidate formulas and subformula enumerations are identical, so this is
    # substantially faster than round-robin row sharding while remaining fully
    # deterministic across machines and Python versions.
    if args.num_shards > 1:
        group_keys = [
            cache_key(str(adduct), float(precursor_mz))
            for adduct, precursor_mz in zip(frame["adduct"], frame["precursor_mz"])
        ]
        owners = np.fromiter(
            (
                int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
                % args.num_shards
                for key in group_keys
            ),
            dtype=np.int64,
            count=len(group_keys),
        )
        frame = frame.loc[owners == args.shard_index].reset_index(drop=True)
    # Stable grouping maximizes exact subformula-cache reuse for replicate
    # spectra.  Output order is deliberately not part of the JSONL contract.
    frame["_candidate_key"] = [
        cache_key(str(adduct), float(precursor_mz))
        for adduct, precursor_mz in zip(frame["adduct"], frame["precursor_mz"])
    ]
    frame = frame.sort_values(["_candidate_key", "identifier"], kind="stable").reset_index(drop=True)
    return frame


def fast_filter_candidate_map(
    candidate_map: dict[str, list[str]],
    model,
    common,
    max_candidates: int,
    device: torch.device,
) -> dict[str, list[str]]:
    """Apply the public formula-only FastFFN exactly once per unique formula."""
    oversized = {
        key: values for key, values in candidate_map.items() if len(values) > max_candidates
    }
    if not oversized:
        return candidate_map

    unique_formulas = sorted({formula for values in oversized.values() for formula in values})
    model = model.to(device).eval()
    score_map: dict[str, float] = {}
    with torch.no_grad():
        for start in range(0, len(unique_formulas), 8192):
            forms = unique_formulas[start : start + 8192]
            dense = np.stack([common.formula_to_dense(formula) for formula in forms])
            scores = model(torch.from_numpy(dense).float().to(device)).reshape(-1)
            score_map.update(zip(forms, scores.cpu().tolist()))

    for key, formulas in oversized.items():
        # Stable formula tie-break makes the cache invariant to SIRIUS output
        # ordering without changing any non-tied neural decision.
        candidate_map[key] = sorted(
            formulas, key=lambda formula: (-score_map[formula], formula)
        )[:max_candidates]
    return candidate_map


def prepare_candidate_cache(
    args: argparse.Namespace,
    rows: pd.DataFrame,
    run_sirius,
    fast_model,
    common,
    device: torch.device,
) -> dict[str, list[str]]:
    if args.candidate_cache.exists():
        return json.loads(args.candidate_cache.read_text())

    candidate_map: dict[str, list[str]] = {}
    for adduct, adduct_rows in rows.groupby("adduct", sort=True):
        masses = sorted({round(float(value), 4) for value in adduct_rows["precursor_mz"]})
        decompositions = run_sirius(
            masses,
            adduct=adduct,
            filter_="RDBE",
            ppm=args.ppm,
            cores=args.sirius_cores,
            loglevel="NONE",
        )
        for mass in masses:
            candidate_map[cache_key(adduct, mass)] = list(
                decompositions.get(round(mass, 4), [])
            )

    candidate_map = fast_filter_candidate_map(
        candidate_map, fast_model, common, args.max_candidates, device
    )
    args.candidate_cache.parent.mkdir(parents=True, exist_ok=True)
    args.candidate_cache.write_text(json.dumps(candidate_map, sort_keys=True))
    return candidate_map


def preprocess_spectrum(row: pd.Series, common) -> np.ndarray:
    raw_mzs = row["mzs"]
    raw_intensities = row["intensities"]
    mzs = (
        np.asarray(raw_mzs, dtype=float)
        if isinstance(raw_mzs, (list, tuple, np.ndarray))
        else np.fromstring(str(raw_mzs), sep=",")
    )
    intensities = (
        np.asarray(raw_intensities, dtype=float)
        if isinstance(raw_intensities, (list, tuple, np.ndarray))
        else np.fromstring(str(raw_intensities), sep=",")
    )
    spectrum = common.merge_spec_tuples(
        [np.column_stack([mzs, intensities])], float(row["precursor_mz"])
    )
    return common.max_thresh_spec(spectrum, max_peaks=20, inten_thresh=0.003)


def make_feature_bucket(
    row: pd.Series,
    formulas: list[str],
    common,
    extractor,
) -> dict[str, list]:
    instrument_value = row["instrument_type"]
    if pd.isna(instrument_value) or not str(instrument_value).strip():
        instrument = "Unknown (LCMS)"
    else:
        instrument = INSTRUMENT_REMAP.get(str(instrument_value), str(instrument_value))
    spectrum = preprocess_spectrum(row, common)
    tolerance = common.get_instr_tol(instrument)
    adduct = str(row["adduct"])
    parentmass = float(row["precursor_mz"])

    bucket: dict[str, list] = defaultdict(list)
    for formula in formulas:
        candidate_table = common.assign_subforms(
            formula, spectrum, adduct, mass_diff_thresh=tolerance
        )
        features = extractor.extract_json(
            json_obj={
                "cand_ion": adduct,
                "cand_tbl": candidate_table["output_tbl"],
            },
            form=formula,
            ion=adduct,
            parentmass=parentmass,
            instrument=instrument,
            # The released model was trained with --no-cls-mass-diff.
            ablate_cls_error=True,
        )
        bucket["name"].append(str(row["identifier"]))
        bucket["formula"].append(formula)
        bucket["ion"].append(adduct)
        bucket["instrument"].append(instrument)
        bucket["parentmass"].append(parentmass)
        bucket["peak_type"].append(features["peak_types"])
        bucket["form_vecs"].append(features["form_vecs"])
        bucket["ion_vecs"].append(features["ion_vecs"])
        bucket["instrument_vecs"].append(features["instrument_vecs"])
        bucket["frag_intens"].append(features["frag_intens"])
        bucket["rel_mass_diffs"].append(features["rel_mass_diffs"])
    return bucket


def score_bucket(bucket, model, collate_fn, device: torch.device) -> np.ndarray:
    batch = collate_fn([bucket])
    tensor_keys = [
        "types",
        "form_vec",
        "ion_vec",
        "instrument_vec",
        "intens",
        "rel_mass_diffs",
        "num_peaks",
    ]
    for key in tensor_keys:
        batch[key] = batch[key].to(device)
    with torch.no_grad():
        scores = model(
            batch["num_peaks"],
            batch["types"],
            batch["form_vec"],
            batch["ion_vec"],
            batch["instrument_vec"],
            batch["intens"],
            batch["rel_mass_diffs"],
        )
    return scores.reshape(-1).detach().cpu().numpy()


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    found = set()
    with path.open() as handle:
        for line in handle:
            if line.strip():
                found.add(str(json.loads(line)["identifier"]))
    return found


def main() -> None:
    args = parse_args()
    started = time.time()
    common, run_sirius, FastFFN, JsonExtractor, PredDataset, MistNet = configure_imports(args)
    device = torch.device(args.device)
    rows = load_rows(args)
    fast_model = FastFFN.load_from_checkpoint(
        args.fast_filter_checkpoint, map_location="cpu"
    ).eval()
    candidates = prepare_candidate_cache(
        args, rows, run_sirius, fast_model, common, device
    )
    if args.prepare_candidates_only:
        print(
            json.dumps(
                {
                    "candidate_cache": str(args.candidate_cache),
                    "unique_precursor_adduct_pairs": len(candidates),
                    "elapsed_seconds": time.time() - started,
                },
                indent=2,
            )
        )
        return
    model = MistNet.load_from_checkpoint(args.checkpoint, map_location="cpu")
    model = model.to(device).eval()
    extractor = JsonExtractor(max_subpeak=model.max_subpeak)
    collate_fn = PredDataset.collate_fn

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = existing_ids(args.output) if args.resume else set()
    mode = "a" if args.resume else "w"
    total = labelled = top1 = top5 = covered = empty = 0
    previous_candidate_key = None
    with args.output.open(mode) as output_handle:
        for index, row in rows.iterrows():
            identifier = str(row["identifier"])
            if identifier in done:
                continue
            candidate_key = str(row["_candidate_key"])
            if candidate_key != previous_candidate_key:
                # Some high-mass formulas have millions of subformulas.  Keep
                # one precursor group's cache, then release it before moving on.
                common._reproduction_subset_cache.cache_clear()
                previous_candidate_key = candidate_key
            formulas = candidates.get(candidate_key, [])
            canonical_candidates = [common.standardize_form(formula) for formula in formulas]
            has_truth = "formula" in row.index and not pd.isna(row["formula"])
            truth = (
                common.standardize_form(str(row["formula"])) if has_truth else None
            )
            if has_truth:
                labelled += 1
                covered += int(truth in canonical_candidates)
            total += 1
            if not formulas:
                empty += 1
                prediction = {"identifier": identifier, "formulas": [], "scores": []}
            else:
                bucket = make_feature_bucket(row, formulas, common, extractor)
                scores = score_bucket(bucket, model, collate_fn, device)
                order = sorted(
                    range(len(formulas)), key=lambda i: (-float(scores[i]), formulas[i])
                )
                ranked = [canonical_candidates[i] for i in order[: args.top_k]]
                if has_truth:
                    top1 += int(bool(ranked) and ranked[0] == truth)
                    top5 += int(truth in ranked[:5])
                prediction = {
                    "identifier": identifier,
                    "formulas": ranked,
                    "scores": [float(scores[i]) for i in order[: args.top_k]],
                    "candidate_count": len(formulas),
                }
            output_handle.write(json.dumps(prediction, separators=(",", ":")) + "\n")
            if (index + 1) % 100 == 0:
                output_handle.flush()
                print(
                    json.dumps(
                        {
                            "processed": total,
                            "coverage": covered / labelled if labelled else None,
                            "top1": top1 / labelled if labelled else None,
                            "top5": top5 / labelled if labelled else None,
                            "elapsed_seconds": time.time() - started,
                        }
                    ),
                    flush=True,
                )

    metrics = {
        "fold": args.fold,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "processed": total,
        "labelled_diagnostic_rows": labelled,
        "candidate_coverage": covered / labelled if labelled else None,
        "top1_formula_accuracy": top1 / labelled if labelled else None,
        "top5_formula_accuracy": top5 / labelled if labelled else None,
        "empty_candidate_sets": empty,
        "elapsed_seconds": time.time() - started,
        "ppm": args.ppm,
        "max_candidates": args.max_candidates,
        "checkpoint": str(args.checkpoint),
        "fast_filter_checkpoint": str(args.fast_filter_checkpoint),
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
