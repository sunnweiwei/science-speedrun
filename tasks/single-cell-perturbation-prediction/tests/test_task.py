"""Unit tests for the task plumbing (run inside the dev or verifier image: needs cell-eval).

    python -m pytest tests/test_task.py -q        (from the task directory)
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / "tests"))
import cellscore  # noqa: E402


def test_scorer_copies_identical():
    a = (TASK / "environment" / "evaluation" / "cellscore.py").read_bytes()
    b = (TASK / "tests" / "cellscore.py").read_bytes()
    assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest()


def test_download_manifest_wellformed():
    m = json.loads((TASK / "environment" / "data" / "download-source.json").read_text())
    for key, entry in m["files"].items():
        assert "path" in entry and ("url" in entry or "hf_repo" in entry), key


def _synthetic(seed=0, n_genes=300, n_basal=400, n_targets=12, n_cells=60):
    rng = np.random.default_rng(seed)
    base_rate = rng.gamma(2.0, 1.0, size=n_genes)
    basal = rng.poisson(base_rate, size=(n_basal, n_genes)).astype(np.float32)
    genes = [f"G{i}" for i in range(n_genes)]
    targets = [f"T{i}" for i in range(n_targets)]
    requests = [{"request_id": f"ctx:{i:04d}", "context": "ctx", "perturbation": {"type": "crispri", "target": t},
                 "n_cells": n_cells, "seed": i} for i, t in enumerate(targets)]
    resp_blocks, resp_targets = [], []
    effects = {}
    for t in targets:
        eff = np.ones(n_genes); hit = rng.choice(n_genes, 30, replace=False)
        eff[hit] = rng.choice([0.2, 3.0], size=30)
        effects[t] = eff
        resp_blocks.append(rng.poisson(base_rate * eff, size=(80, n_genes)).astype(np.float32))
        resp_targets += [t] * 80
    basal_a = ad.AnnData(X=sparse.csr_matrix(basal), obs=pd.DataFrame({"target_gene": "non-targeting"}, index=[f"c{i}" for i in range(n_basal)]),
                         var=pd.DataFrame(index=genes))
    resp_a = ad.AnnData(X=sparse.csr_matrix(np.vstack(resp_blocks)), obs=pd.DataFrame({"target_gene": resp_targets}, index=[f"r{i}" for i in range(len(resp_targets))]),
                        var=pd.DataFrame(index=genes))
    # an "oracle" prediction: fresh draws from the true perturbed rates
    pred_blocks, ids = [], []
    for r in requests:
        pred_blocks.append(rng.poisson(base_rate * effects[r["perturbation"]["target"]], size=(n_cells, n_genes)).astype(np.float32))
        ids += [r["request_id"]] * n_cells
    pred_a = ad.AnnData(X=sparse.csr_matrix(np.vstack(pred_blocks)), obs=pd.DataFrame({"request_id": ids}, index=[f"p{i}" for i in range(len(ids))]),
                        var=pd.DataFrame(index=genes))
    return basal_a, resp_a, pred_a, requests


def test_scorer_oracle_beats_noeffect(tmp_path):
    basal, resp, pred, requests = _synthetic()
    (tmp_path / "requests.jsonl").write_text("\n".join(json.dumps(r) for r in requests) + "\n")
    basal.write_h5ad(tmp_path / "basal.h5ad"); resp.write_h5ad(tmp_path / "responses.h5ad"); pred.write_h5ad(tmp_path / "pred.h5ad")
    noeff = cellscore.noeffect_prediction(basal, requests); noeff.write_h5ad(tmp_path / "noeffect.h5ad")
    good = cellscore.score_context(tmp_path / "pred.h5ad", tmp_path / "basal.h5ad", tmp_path / "responses.h5ad",
                                   tmp_path / "requests.jsonl", tmp_path / "out_good", threads=4)
    base = cellscore.score_context(tmp_path / "noeffect.h5ad", tmp_path / "basal.h5ad", tmp_path / "responses.h5ad",
                                   tmp_path / "requests.jsonl", tmp_path / "out_base", threads=4)
    assert good["all"]["score"] > 0.3
    assert base["all"]["score"] < 0.05


def test_scorer_rejects_bad_predictions(tmp_path):
    basal, resp, pred, requests = _synthetic()
    bad = pred[:-5].copy()  # one request short of cells
    with pytest.raises(ValueError):
        cellscore.assemble_prediction(bad, basal, requests)
    shuffled = pred.copy(); shuffled.obs["request_id"] = shuffled.obs["request_id"].map(lambda s: s.replace("ctx:", "zzz:"))
    with pytest.raises(ValueError):
        cellscore.assemble_prediction(shuffled, basal, requests)
    wrong_genes = pred[:, ::-1].copy()
    with pytest.raises(ValueError):
        cellscore.assemble_prediction(wrong_genes, basal, requests)


def test_noeffect_package_runs(tmp_path):
    basal, resp, pred, requests = _synthetic()
    inp = tmp_path / "in"; inp.mkdir()
    basal.write_h5ad(inp / "basal.h5ad")
    (inp / "requests.jsonl").write_text("\n".join(json.dumps(r) for r in requests) + "\n")
    (inp / "context.json").write_text(json.dumps({"context_id": "ctx", "chemistry": "10x_3p_v3", "n_genes": basal.n_vars}))
    out = tmp_path / "out"
    subprocess.run(["bash", str(TASK / "environment" / "baseline" / "noeffect" / "infer.sh"), str(inp), str(out)], check=True)
    p = ad.read_h5ad(out / "predictions.h5ad")
    assert p.n_obs == sum(r["n_cells"] for r in requests)
    cellscore.assemble_prediction(p, basal, requests)


def test_publish_swaps_whole_package(tmp_path):
    src = tmp_path / "pkg"
    (src / "st_data").mkdir(parents=True)
    (src / "infer.sh").write_text("#!/usr/bin/env bash\n")
    (src / "st_data" / "train.h5ad").write_text("training-only")
    out = tmp_path / "submission"
    for version in ("1", "2"):
        (src / "weights.txt").write_text(version)
        subprocess.run([sys.executable, str(TASK / "environment" / "baseline" / "publish.py"), str(src),
                        "--output", str(out)], check=True, capture_output=True)
        assert (out / "weights.txt").read_text() == version
    assert not (out / "st_data" / "train.h5ad").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["pkg", "submission"]  # no staging leftovers


def test_development_mounts_follow_task_standard():
    compose = (TASK / "environment" / "docker-compose.yaml").read_text()
    for entry in ("train", "test", "priors", "se600m", "manifest.json", "splits.json"):
        assert f'/{entry}", target: /workspace/data/{entry}}}' in compose
    assert "TASK_PRIVATE_DIR" not in compose and "network_mode" not in compose
    assert "target: /workspace/research" in compose
    dockerfile = (TASK / "environment" / "Dockerfile").read_text()
    assert "ln -s /workspace/data/se600m /opt/se600m" in dockerfile
    assert "COPY instruction.md /workspace/instruction.md" in dockerfile
    verifier = (TASK / "tests" / "docker-compose.yaml").read_text()
    main = verifier.split("  main:")[1].split("  grader:")[0]
    assert "TASK_PRIVATE_DIR" not in main and "TASK_PRIVATE_DIR" in verifier
