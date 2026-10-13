#!/usr/bin/env python3
"""Trusted grader: checks the sealed snapshot, then scores it with environment/evaluation/scoring.py."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
from common import sha256,tree_sha256,write_json
sys.path[:0]=["/evaluation",str(Path(__file__).resolve().parents[1]/"environment"/"evaluation")]  # image, then repository
from scoring import REPORT_SCHEMA,score  # noqa: E402
from variant_scoring import score_perturbations  # noqa: E402

SNAPSHOT_SCHEMA="genome-sequence-to-function-prediction-snapshot/v4"

def main():
    p=argparse.ArgumentParser(); p.add_argument("--sealed",type=Path,required=True); p.add_argument("--labels",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--wait-seconds",type=int,default=43200); a=p.parse_args()
    ready=a.sealed.parent/"READY"; deadline=time.monotonic()+a.wait_seconds
    while not ready.is_file():
        if time.monotonic()>=deadline: raise TimeoutError("prediction snapshot was not published")
        time.sleep(1)
    snapshot=json.loads((a.sealed/"snapshot.json").read_text())
    if ready.read_text().strip()!=sha256(a.sealed/"snapshot.json") or snapshot.get("schema")!=SNAPSHOT_SCHEMA: raise ValueError("invalid sealed snapshot")
    if snapshot.get("input_manifest_sha256")!=sha256(a.labels/"public/test.parquet"): raise ValueError("sealed predictions and public manifest differ")
    if tree_sha256(a.sealed/"predictions.zarr")!=snapshot["prediction_tree_sha256"]: raise ValueError("sealed prediction hash mismatch")
    if snapshot.get("perturbation_manifest_sha256")!=sha256(a.labels/"variant_evals/public/perturbations.parquet"): raise ValueError("sealed perturbations and public queries differ")
    if tree_sha256(a.sealed/"perturbations.zarr")!=snapshot["perturbation_tree_sha256"]: raise ValueError("sealed perturbation hash mismatch")
    try: report=score(a.sealed/"predictions.zarr",a.labels,"test")
    except Exception as error: report={"schema":REPORT_SCHEMA,"split":"test","examples":0,"families":{},"overall_mean_family_pearson":0.0,"invalid_reason":str(error)}
    # Variant-effect and perturbation evaluations: raw metrics with anchors on the same rows.
    # They are reported, not folded into the reward, until the shared score map is defined.
    try: report["variant_effects"]=score_perturbations(a.sealed/"perturbations.zarr",a.labels/"variant_evals")
    except Exception as error: report["variant_effects"]={"invalid_reason":str(error)}
    reward=report["overall_mean_family_pearson"]
    a.output.mkdir(parents=True,exist_ok=True); write_json(a.output/"report.json",report)
    write_json(a.output/"reward.json",{"reward":reward,"overall_mean_family_pearson":reward})
if __name__=="__main__": main()
