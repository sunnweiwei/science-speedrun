#!/usr/bin/env python3
"""Trusted grader: checks the sealed snapshot, then scores both tracks and variant effects
with environment/evaluation/scoring.py and variant_scoring.py (exact SSI V4 / V3 two-axis score)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from common import sha256, tree_sha256, write_json

sys.path[:0] = [
    "/evaluation",
    str(Path(__file__).resolve().parents[1] / "environment" / "evaluation"),
]  # image, then repository
from scoring import REPORT_SCHEMA, score, ssi_v4_genome_score  # noqa: E402
from variant_scoring import score_perturbations  # noqa: E402

SNAPSHOT_SCHEMA = "genome-sequence-to-function-prediction-snapshot/v4"


def _check_labels_mounted(labels: Path) -> None:
    for required in (
        labels / "public/test.parquet",
        labels / "public/junction_tracks.parquet",
        labels / "labels/test.parquet",
        labels / "variant_evals/public/perturbations.parquet",
        labels / "variant_evals/labels/tracks.parquet",
    ):
        if not required.is_file():
            raise FileNotFoundError(f"missing verifier label asset: {required}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sealed", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--wait-seconds", type=int, default=43200)
    a = p.parse_args()

    _check_labels_mounted(a.labels)
    ready = a.sealed.parent / "READY"
    deadline = time.monotonic() + a.wait_seconds
    while not ready.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("prediction snapshot was not published")
        time.sleep(1)

    snapshot = json.loads((a.sealed / "snapshot.json").read_text())
    if ready.read_text().strip() != sha256(a.sealed / "snapshot.json") or snapshot.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError("invalid sealed snapshot")
    if snapshot.get("input_manifest_sha256") != sha256(a.labels / "public/test.parquet"):
        raise ValueError("sealed predictions and public manifest differ")
    if tree_sha256(a.sealed / "predictions.zarr") != snapshot["prediction_tree_sha256"]:
        raise ValueError("sealed prediction hash mismatch")
    if snapshot.get("perturbation_manifest_sha256") != sha256(a.labels / "variant_evals/public/perturbations.parquet"):
        raise ValueError("sealed perturbations and public queries differ")
    if tree_sha256(a.sealed / "perturbations.zarr") != snapshot["perturbation_tree_sha256"]:
        raise ValueError("sealed perturbation hash mismatch")

    try:
        report = score(a.sealed / "predictions.zarr", a.labels, "test")
        report["variant_effects"] = score_perturbations(
            a.sealed / "perturbations.zarr", a.labels / "variant_evals"
        )
        v4 = ssi_v4_genome_score(report["families"], report["variant_effects"])
        report["ssi_v4"] = v4
        reward_payload = {
            "score": v4["score"],
            "reward": v4["score"],
            "signed_score": v4["signed_score"],
            "tracks_axis": v4["tracks_axis"],
            "variants_axis": v4["variants_axis"],
            "overall_mean_family_pearson": report["overall_mean_family_pearson"],
        }
    except (ValueError, KeyError) as error:
        report = {
            "schema": REPORT_SCHEMA,
            "split": "test",
            "status": "invalid_submission",
            "examples": 0,
            "families": {},
            "variant_effects": {},
            "score": 0.0,
            "overall_mean_family_pearson": 0.0,
            "invalid_reason": f"{type(error).__name__}: {error}",
        }
        reward_payload = {
            "score": 0.0,
            "reward": 0.0,
            "invalid_submission": 1.0,
            "overall_mean_family_pearson": 0.0,
        }

    a.output.mkdir(parents=True, exist_ok=True)
    write_json(a.output / "report.json", report)
    write_json(a.output / "reward.json", reward_payload)


if __name__ == "__main__":
    main()
