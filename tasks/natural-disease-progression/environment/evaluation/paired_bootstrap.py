"""Exact patient-level paired bootstrap for primary AUROC differences."""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from risk_scoring import HORIZONS, read_predictions, weighted_macro_time_auroc


def bootstrap_difference(labels: np.ndarray, valid: np.ndarray,
                         reference: np.ndarray, candidate: np.ndarray,
                         weights: np.ndarray, device_name: str) -> np.ndarray:
    """Exact batched AUROC recomputation; prediction ties remain grouped."""
    device = torch.device(device_name)
    weight = torch.as_tensor(weights, dtype=torch.float32, device=device)
    replicates, diseases = len(weights), labels.shape[1]
    reference_sum = torch.zeros((replicates, diseases), device=device)
    candidate_sum = torch.zeros_like(reference_sum)
    cell_count = torch.zeros_like(reference_sum)

    def batch_auc(y_numpy, p_numpy, patient_index):
        order = np.argsort(p_numpy, kind="stable")
        sorted_prediction = p_numpy[order]
        group = np.r_[0, np.cumsum(sorted_prediction[1:] != sorted_prediction[:-1])]
        group_count = int(group[-1]) + 1
        w = weight[:, torch.as_tensor(patient_index[order], device=device)]
        y = torch.as_tensor(y_numpy[order], dtype=torch.float32, device=device)
        group_index = torch.as_tensor(group, dtype=torch.int64, device=device).expand(replicates, -1)
        positive_group = torch.zeros((replicates, group_count), device=device)
        negative_group = torch.zeros_like(positive_group)
        positive_group.scatter_add_(1, group_index, w * y)
        negative_group.scatter_add_(1, group_index, w * (1 - y))
        positive = positive_group.sum(1)
        negative = negative_group.sum(1)
        lower_negative = torch.cumsum(negative_group, 1) - negative_group
        auc = (positive_group * (lower_negative + .5 * negative_group)).sum(1)
        auc = auc / (positive * negative).clamp(min=1)
        return auc, (positive > 0) & (negative > 0)

    for j in range(diseases):
        for k in range(labels.shape[2]):
            patient_index = np.flatnonzero(valid[:, j, k])
            y = labels[patient_index, j, k]
            if not len(y) or y.min() == y.max():
                continue
            ref_auc, scorable = batch_auc(y, reference[patient_index, j, k], patient_index)
            cand_auc, candidate_scorable = batch_auc(
                y, candidate[patient_index, j, k], patient_index)
            if not torch.equal(scorable, candidate_scorable):
                raise AssertionError("Bootstrap cell availability must depend only on labels")
            keep = scorable.float()
            reference_sum[:, j] += ref_auc * keep
            candidate_sum[:, j] += cand_auc * keep
            cell_count[:, j] += keep
    disease_available = cell_count > 0
    reference_disease = reference_sum / cell_count.clamp(min=1)
    candidate_disease = candidate_sum / cell_count.clamp(min=1)
    difference = ((candidate_disease - reference_disease) * disease_available).sum(1)
    difference /= disease_available.sum(1).clamp(min=1)
    return difference.cpu().numpy()


def load_truth(input_dir: Path, trusted: Path):
    manifest = json.loads((input_dir / "manifest.json").read_text())
    diseases = manifest["scored_diseases"]
    samples = pq.read_table(input_dir / "samples.parquet").to_pylist()
    ids = [row["sample_id"] for row in samples]
    sample_index = {sid: i for i, sid in enumerate(ids)}
    disease_index = {code: j for j, code in enumerate(diseases)}
    shape = (len(ids), len(diseases), len(HORIZONS))
    y, valid = np.zeros(shape, np.uint8), np.zeros(shape, bool)
    seen = set()
    columns = ["sample_id", "disease_code"] + [f"{field}_{horizon}"
        for horizon in HORIZONS for field in ("disease_label", "disease_valid")]
    for batch in pq.ParquetFile(trusted / "label_windows.parquet").iter_batches(
            batch_size=65536, columns=columns):
        data = batch.to_pydict()
        for row in range(batch.num_rows):
            key = (data["sample_id"][row], data["disease_code"][row])
            if key in seen or key[0] not in sample_index or key[1] not in disease_index:
                raise ValueError("Unknown or duplicate keyed label row")
            seen.add(key)
            i, j = sample_index[key[0]], disease_index[key[1]]
            for k, horizon in enumerate(HORIZONS):
                y[i, j, k] = data[f"disease_label_{horizon}"][row]
                valid[i, j, k] = data[f"disease_valid_{horizon}"][row]
    if len(seen) != len(ids) * len(diseases):
        raise ValueError("Trusted labels do not cover the public key grid")
    boundaries = {row["sample_id"]: row for row in pq.read_table(
        trusted / "sample_boundaries.parquet").to_pylist()}
    primary = []
    for sample in samples:
        cutoff = sample["cutoff"].date() if hasattr(sample["cutoff"], "date") else sample["cutoff"]
        row = boundaries[sample["sample_id"]]
        primary.append(cutoff + timedelta(days=365) <= row["diagnosis_observation_end"]
            and cutoff + timedelta(days=365) <= row["death_observation_end"])
    primary = np.asarray(primary, dtype=bool)
    return ids, diseases, y[primary], valid[primary], primary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--trusted", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--device", default="cpu",
                        help="cpu or cuda; batched computation is exact on either device")
    args = parser.parse_args()
    ids, diseases, y, valid, primary = load_truth(args.input, args.trusted)
    reference, _ = read_predictions(args.reference, ids, len(diseases))
    reference = reference[primary]
    unit_weight = np.ones(len(y), dtype=np.float64)
    reference_point = weighted_macro_time_auroc(y, reference, valid, unit_weight)
    rng = np.random.default_rng(args.seed)
    weights = np.stack([np.bincount(rng.integers(0, len(y), len(y)), minlength=len(y))
                        for _ in range(args.replicates)]).astype(np.float32)
    result = {
        "schema": "ndp-paired-patient-bootstrap-auroc/v2",
        "metric": "disease-first macro time-averaged AUROC",
        "replicates": args.replicates,
        "seed": args.seed,
        "reference": str(args.reference),
        "reference_point": reference_point,
        "primary_patients": len(y),
        "method": ("same complete-year patients resampled with replacement for both models; "
                   "AUROC is exactly recomputed in every replicate; a resampled single-class "
                   "disease-horizon cell is unavailable for both models, followed by time-within-"
                   "disease then disease averaging"),
        "comparisons": {},
    }
    for path in args.candidate:
        candidate, _ = read_predictions(path, ids, len(diseases))
        candidate = candidate[primary]
        candidate_point = weighted_macro_time_auroc(y, candidate, valid, unit_weight)
        point = candidate_point - reference_point
        draws = bootstrap_difference(y, valid, reference, candidate, weights, args.device)
        result["comparisons"][str(path)] = {
            "candidate_point": candidate_point,
            "candidate_minus_reference": point,
            "ci95": np.quantile(draws, [.025, .975]).tolist(),
            "probability_candidate_better": float(np.mean(draws > 0)),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
