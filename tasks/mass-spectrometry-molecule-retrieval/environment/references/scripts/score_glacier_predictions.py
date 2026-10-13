#!/usr/bin/env python3
"""Score upstream GLACIER PredSpecDB output under the benchmark contract."""

import argparse
from collections import Counter
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np
from rdkit import Chem

import ms_pred.common as common
from ms_pred.common import chem_utils
from ms_pred.retrieval.retrieval_benchmark import _cosine_sparse, _entropy_sparse


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
        pool_id = row["candidate_pool_id"]
        if pool_id in pools:
            raise ValueError(f"duplicate candidate pool {pool_id}")
        pools[pool_id] = row["candidates"]
    return pools


@lru_cache(maxsize=262144)
def prediction_key_and_support(smiles: str) -> tuple[str, str]:
    """Mirror GLACIER's public inference-domain structure checks."""
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return "", "rdkit_invalid"
        if any(atom.GetAtomicNum() == 0 for atom in molecule.GetAtoms()):
            return "", "wildcard_atom"
        if any(
            atom.GetAtomicNum() not in chem_utils.VALID_ATOM_NUM
            for atom in molecule.GetAtoms()
        ):
            return Chem.MolToInchiKey(molecule), "unsupported_atom"
        molecule = Chem.RemoveHs(molecule)
        if molecule.GetNumAtoms() > 100:
            return Chem.MolToInchiKey(molecule), "more_than_100_heavy_atoms"
        return Chem.MolToInchiKey(molecule), "supported"
    except Exception:
        return "", "structure_exception"


def sparse_observed(
    query: dict,
    num_bins: int,
    upper_limit: int,
    max_peaks: int,
    intensity_threshold: float,
):
    mzs = np.asarray(query["mzs"], dtype=np.float32)
    intensities = np.asarray(query["intensities"], dtype=np.float32)
    keep = (mzs <= float(query["precursor_mz"]) + 1) & (intensities > 0)
    mzs = mzs[keep]
    intensities = intensities[keep]
    if intensities.size == 0:
        raise ValueError(f"query {query['query_id']} has no usable intensity")
    maximum = float(intensities.max())
    if maximum <= 0:
        raise ValueError(f"query {query['query_id']} has no positive intensity")
    # This reproduces process_spec_file() followed by max_inten_spec() in the
    # official no_subform MassSpecGym preprocessing: max-normalize, square
    # root, retain the strongest 50 peaks, then apply the 0.001 threshold.
    intensities = np.sqrt(intensities / maximum)
    strongest = np.argsort(intensities)[::-1][:max_peaks]
    mzs = mzs[strongest]
    intensities = intensities[strongest]
    above_threshold = intensities > intensity_threshold
    peaks = np.column_stack([mzs[above_threshold], intensities[above_threshold]])
    dense = common.bin_spectra([peaks], num_bins=num_bins, upper_limit=upper_limit)[0]
    indices = np.flatnonzero(dense > 0).astype(np.uint32)
    return indices, dense[indices].astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument(
        "--pred-file",
        type=Path,
        required=True,
        nargs="+",
        help="One or more GLACIER PredSpecDB shards.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--distance", choices=("entropy", "cos"), default="entropy")
    parser.add_argument(
        "--tie-break",
        choices=("candidate_id", "input_order"),
        default="candidate_id",
        help=(
            "Use content-derived IDs for the benchmark. input_order exists only "
            "to audit answer-keyed upstream candidate tables."
        ),
    )
    parser.add_argument("--num-bins", type=int, default=150000)
    parser.add_argument("--upper-limit", type=int, default=1500)
    parser.add_argument("--max-observed-peaks", type=int, default=50)
    parser.add_argument("--observed-intensity-threshold", type=float, default=0.001)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument(
        "--query-id-file",
        type=Path,
        help="Optional newline-delimited allowlist for query-disjoint scoring shards.",
    )
    parser.add_argument(
        "--exclude-ranking",
        type=Path,
        action="append",
        default=[],
        help=(
            "Skip query IDs already present in a partial ranking file. This is "
            "an answer-free recovery option; merge the disjoint outputs with "
            "merge_partial_rankings.py."
        ),
    )
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    output_root.mkdir(parents=True)
    pools = load_pools(input_root / "candidate_pools.jsonl")
    pool_prediction_keys = {}
    predictions = [
        common.PredSpecDB(h5_path=path.resolve(), mode="r")
        for path in args.pred_file
    ]
    selected_queries = None
    if args.query_id_file is not None:
        selected_queries = {
            line.strip()
            for line in args.query_id_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if not selected_queries:
            raise ValueError("query ID allowlist is empty")
    excluded_queries = {
        row["query_id"]
        for ranking_path in args.exclude_ranking
        for row in read_jsonl(ranking_path.resolve())
    }

    ranked_queries = 0
    candidate_rows = 0
    missing_predictions = 0
    missing_supported_predictions = 0
    unsupported_missing = Counter()
    with (output_root / "rankings.jsonl").open("w", encoding="utf-8") as stream:
        for query in read_jsonl(input_root / "queries.jsonl"):
            if (
                selected_queries is not None
                and query["query_id"] not in selected_queries
            ):
                continue
            if args.max_queries is not None and ranked_queries >= args.max_queries:
                break
            if query["query_id"] in excluded_queries:
                continue
            collision_energy = query.get("collision_energy")
            if not isinstance(collision_energy, (int, float)) or not math.isfinite(collision_energy):
                continue
            pred_name = f"pred_{query['query_id']}"
            shard_specs = []
            for prediction_db in predictions:
                spec_dict, present = prediction_db.read_from_name(pred_name)
                if present:
                    shard_specs.append(spec_dict)
            if not shard_specs:
                continue
            by_ikey = {}
            for spec_dict in shard_specs:
                for remark, ce_dict in spec_dict.items():
                    normalized = str(remark).removeprefix("ikey ").strip()
                    normalized_specs = by_ikey.setdefault(normalized, {})
                    normalized_specs.update({
                        common.get_collision_energy(ce): spectrum.binned_spec_sparse
                        if spectrum.has_matching_binned_spec(
                            mass_upper_limit=args.upper_limit,
                            num_bins=args.num_bins,
                            pool_fn="add",
                        )
                        else (
                            spectrum.ensure_binned_spectrum(
                                mass_upper_limit=args.upper_limit,
                                num_bins=args.num_bins,
                                pool_fn="add",
                            )
                            or spectrum.binned_spec_sparse
                        )
                        for ce, spectrum in ce_dict.items()
                    })

            pool_id = query["candidate_pool_id"]
            candidates = pools[pool_id]
            candidate_keys = pool_prediction_keys.get(pool_id)
            if candidate_keys is None:
                candidate_keys = [
                    prediction_key_and_support(candidate["smiles"])
                    for candidate in candidates
                ]
                pool_prediction_keys[pool_id] = candidate_keys
            observed = sparse_observed(
                query,
                args.num_bins,
                args.upper_limit,
                args.max_observed_peaks,
                args.observed_intensity_threshold,
            )
            ce_key = common.get_collision_energy(collision_energy)
            candidate_predictions = []
            for candidate_ikey, support in candidate_keys:
                pred = by_ikey.get(candidate_ikey)
                if pred is None:
                    missing_predictions += 1
                    if support == "supported":
                        missing_supported_predictions += 1
                    else:
                        unsupported_missing[support] += 1
                    pred = {}
                candidate_predictions.append(pred)
            distance_fn = _entropy_sparse if args.distance == "entropy" else _cosine_sparse
            distances = distance_fn(candidate_predictions, observed, ce_key, sqrt_inten=True)
            # GLACIER emits identical predictions for candidates that collapse to
            # the same InChIKey. Break exact score ties by the content-derived
            # candidate ID, never by the public candidate-file order.
            candidate_ids = np.asarray(
                [candidate["candidate_id"] for candidate in candidates]
            )
            if args.tie_break == "candidate_id":
                order = np.lexsort((candidate_ids, distances))
            else:
                order = np.argsort(distances, kind="stable")
            ranking = [candidates[int(index)]["candidate_id"] for index in order]
            stream.write(
                json.dumps(
                    {"query_id": query["query_id"], "ranking": ranking},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            ranked_queries += 1
            candidate_rows += len(candidates)

    for prediction_db in predictions:
        prediction_db.close()
    if selected_queries is not None and ranked_queries != len(selected_queries):
        raise ValueError(
            "query ID allowlist was not fully ranked: "
            f"{ranked_queries}/{len(selected_queries)}"
        )
    report = {
        "schema": "glacier-contract-score/v2",
        "distance": args.distance,
        "tie_break": args.tie_break,
        "ranked_queries": ranked_queries,
        "candidate_rows": candidate_rows,
        "missing_candidate_predictions": missing_predictions,
        "missing_supported_candidate_predictions": missing_supported_predictions,
        "model_unsupported_candidate_rows": dict(sorted(unsupported_missing.items())),
        "num_bins": args.num_bins,
        "upper_limit": args.upper_limit,
        "prediction_shards": len(predictions),
        "max_observed_peaks": args.max_observed_peaks,
        "observed_intensity_threshold": args.observed_intensity_threshold,
        "excluded_queries": len(excluded_queries),
        "selected_queries": (
            None if selected_queries is None else len(selected_queries)
        ),
    }
    (output_root / "runtime.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
