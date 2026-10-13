"""Variant-effect metrics and perturbation outputs."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "environment" / "evaluation"))
from variant_scoring import (PERTURBATION_SCHEMA, by_chrom_weighted, perturbation_scores,  # noqa: E402
                             tissue_weighted, validate_perturbations, AUROC, AUPRC)


def test_tissue_weighted_mean_weights_by_rows_and_skips_single_class_tissues():
    frame = pd.DataFrame({"tissue": ["a"] * 4 + ["b"] * 2 + ["c"] * 3,
                          "target": [0, 1, 0, 1, 0, 1, 1, 1, 1],
                          "prediction": [0.1, 0.9, 0.2, 0.8, 0.9, 0.1, 0.5, 0.6, 0.7]})
    # a: AUROC 1 over 4 rows, b: AUROC 0 over 2 rows, c: single class -> skipped
    assert tissue_weighted(frame, AUROC) == pytest.approx((1 * 4 + 0 * 2) / 6)


def test_traitgym_metric_is_per_chromosome_auprc_weighted_by_variants():
    frame = pd.DataFrame({"chrom": ["chr1"] * 4 + ["chr2"] * 2,
                          "label": [1, 0, 0, 0, 1, 0], "prediction": [0.9, 0.1, 0.2, 0.3, 0.1, 0.9]})
    expected = (AUPRC([1, 0, 0, 0], [0.9, 0.1, 0.2, 0.3]) * 4 + AUPRC([1, 0], [0.1, 0.9]) * 2) / 6
    assert by_chrom_weighted(frame, AUPRC) == pytest.approx(expected)


def _outputs(tmp_path, families, arrays):
    root = zarr.open_group(str(tmp_path / "perturbations.zarr"), mode="w")
    for index, (a, b) in enumerate(arrays):
        for family in families:
            root.create_dataset(f"perturbations/p{index:06d}/a/{family}", data=a.astype(np.float16))
            root.create_dataset(f"perturbations/p{index:06d}/b/{family}", data=b.astype(np.float16))
    root.attrs.update({"schema": PERTURBATION_SCHEMA, "perturbation_ids": [f"v{i}" for i in range(len(arrays))]})
    return root


def test_center_score_tracks_the_allelic_change_and_validation_checks_shapes(tmp_path):
    tracks = pd.DataFrame({"family": ["DNASE"] * 305, "ontology_curie": ["EFO:0002784"] + ["X"] * 304,
                           "name": ["n"] * 305})
    flat = np.zeros((4, 305)); up = flat.copy(); up[:, 0] = np.arcsinh(100.0)
    outputs = _outputs(tmp_path, ["dnase"], [(flat, flat), (flat, up)])
    labels = pd.DataFrame({"perturbation_id": ["v0", "v1"]})
    scores = perturbation_scores("caqtl", labels, outputs, {"v0": 0, "v1": 1}, tracks)
    assert scores.iloc[0] == 0 and scores.iloc[1] > 6  # log2(4*100+1) - log2(1)
    queries = pd.DataFrame({"perturbation_id": ["v0", "v1"], "families": [["dnase"], ["dnase"]],
                            "dense_bin_start": [0, 0], "dense_bin_end": [4, 4], "splice_start": [0, 0],
                            "splice_end": [0, 0]})
    validate_perturbations(tmp_path / "perturbations.zarr", queries)
    with pytest.raises(ValueError, match="expected"):
        validate_perturbations(tmp_path / "perturbations.zarr", queries.assign(dense_bin_end=[5, 5]))


def test_splice_score_is_the_largest_probability_change(tmp_path):
    a = np.full((10, 4), -5.0); b = a.copy(); b[3, 1] = 5.0
    outputs = _outputs(tmp_path, ["splice_sites"], [(a, b)])
    score = perturbation_scores("clinvar_noncoding", pd.DataFrame({"perturbation_id": ["v0"]}),
                                outputs, {"v0": 0}, pd.DataFrame()).iloc[0]
    assert score == pytest.approx(1 / (1 + np.exp(-5)) - 1 / (1 + np.exp(5)))
