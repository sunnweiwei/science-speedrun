#!/usr/bin/env python3
"""Contract smoke check: run a package's infer.sh on one input directory in a scratch location
(relocated copy, no workspace access) and validate the output exactly as the verifier does.

    python check_submission.py --package /workspace/submission --input /workspace/data/test/ctx_far/public
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cellscore import read_requests  # noqa: E402
from query_independence import DEFAULT_ATOL, DEFAULT_RTOL, compare_prediction_files, select_probe_subset  # noqa: E402


def validate(pred_path: Path, basal_path: Path, requests_path: Path) -> dict:
    expected = read_requests(requests_path)
    basal = ad.read_h5ad(basal_path, backed="r")
    pred = ad.read_h5ad(pred_path)
    problems = []
    if "request_id" not in pred.obs:
        problems.append("obs.request_id missing")
    if pred.var_names.has_duplicates:
        problems.append("duplicate genes")
    if list(map(str, pred.var_names)) != list(map(str, basal.var_names)):
        problems.append("gene order differs from basal.h5ad")
    if "request_id" in pred.obs:
        counts = pred.obs["request_id"].astype(str).value_counts().to_dict()
        want = {r["request_id"]: r["n_cells"] for r in expected}
        if counts != want:
            missing = sorted(set(want) - set(counts))[:5]
            extra = sorted(set(counts) - set(want))[:5]
            wrong = [k for k in want if k in counts and counts[k] != want[k]][:5]
            problems.append(f"request/cell-count mismatch: missing={missing} extra={extra} wrong_count={wrong}")
    values = pred.X.data if sparse.issparse(pred.X) else np.asarray(pred.X)
    if not np.isfinite(values).all() or (values < 0).any():
        problems.append("negative or non-finite values")
    if values.size and values.max() < 15 and not np.allclose(values, np.round(values)):
        problems.append("values look log-normalised; the contract requires count-scale output")
    return {"ok": not problems, "problems": problems, "n_cells": int(pred.n_obs), "n_genes": int(pred.n_vars)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True, type=Path)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="vcz-check-"))
    pkg = tmp / "package"
    shutil.copytree(a.package, pkg, symlinks=False)
    out = tmp / "out"
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp), "PYTHONUNBUFFERED": "1", "PIP_NO_INDEX": "1",
           "HF_HUB_OFFLINE": "1", "SE600M_DIR": os.environ.get("SE600M_DIR", "/opt/se600m"),
           "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")}
    t0 = time.time()
    proc = subprocess.run(["bash", str(pkg / "infer.sh"), str(a.input.resolve()), str(out)], cwd=pkg, env=env,
                          capture_output=True, text=True, timeout=a.timeout)
    result = {"exit_code": proc.returncode, "seconds": round(time.time() - t0, 1), "scratch": str(tmp)}
    if proc.returncode != 0:
        result.update(ok=False, problems=["infer.sh failed"], stderr_tail=proc.stderr[-4000:])
    else:
        val = validate(out / "predictions.h5ad", a.input / "basal.h5ad", a.input / "requests.jsonl")
        result.update(val)
        if val["ok"]:
            sub_in = tmp / "subset_in"
            sub_out = tmp / "subset_out"
            sub_in.mkdir()
            full_reqs = read_requests(a.input / "requests.jsonl")
            sub_reqs = select_probe_subset(full_reqs, context_name=a.input.name)
            shutil.copyfile(a.input / "basal.h5ad", sub_in / "basal.h5ad")
            (sub_in / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sub_reqs))
            if (a.input / "context.json").is_file():
                meta = json.loads((a.input / "context.json").read_text())
                meta["n_requests"] = len(sub_reqs)
                (sub_in / "context.json").write_text(json.dumps(meta, indent=2) + "\n")
            proc2 = subprocess.run(["bash", str(pkg / "infer.sh"), str(sub_in), str(sub_out)], cwd=pkg, env=env,
                                   capture_output=True, text=True, timeout=a.timeout)
            if proc2.returncode != 0:
                result.update(ok=False, problems=["subset infer.sh failed"], stderr_tail=proc2.stderr[-4000:])
            else:
                qi = compare_prediction_files(
                    out / "predictions.h5ad", sub_out / "predictions.h5ad",
                    a.input / "basal.h5ad", a.input / "requests.jsonl", sub_in / "requests.jsonl",
                    atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL,
                )
                result["query_independence"] = {
                    "passed": qi["passed"],
                    "changed_requests": qi["changed_requests"],
                    "max_difference": qi["max_difference"],
                    "atol": qi["atol"],
                    "rtol": qi["rtol"],
                }
                if not qi["passed"]:
                    result["ok"] = False
                    result["problems"].append(
                        f"query independence failed on {qi['changed_requests']}/{qi['checked_requests']} subset requests"
                    )
    print(json.dumps(result, indent=2))
    if not a.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
