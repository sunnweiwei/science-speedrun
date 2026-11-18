"""v2 grader path: normalisation, reference bundle, calibrated reward and grader wiring.

    python -m pytest tests/test_wmse_v2_grader.py -q     (from the task directory; the last test needs cell-eval)
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
from calibrated_reward_v2 import score_context, score_context_v3  # noqa: E402
from population_metrics_v2 import (load_or_build_bundle, log1p_cp10k, prediction_means,  # noqa: E402
                                   score_against_bundle, target_means)


def test_v2_copies_identical():
    for name in ("population_metrics_v2.py", "calibrated_reward_v2.py"):
        assert (TASK / "tests" / name).read_bytes() == (TASK / "environment" / "evaluation" / name).read_bytes(), name


def test_normalisation_scale_invariant_including_totals_below_one():
    rng = np.random.default_rng(0)
    x = rng.gamma(.5, 1., size=(40, 30)) * (rng.random((40, 30)) < .4)
    x[:10] *= 1e-3                      # rows whose total is far below 1
    x[10, :] = 0.                       # an all-zero row
    x[11, :] = 0.; x[11, 3] = 1e-6      # a single tiny count
    totals = x.sum(1)
    assert (totals[:10] < 1).all() and totals[10] == 0
    base = log1p_cp10k(x).toarray()
    for scale in (1e-4, .37, 1., 12.5, 1e5):
        assert np.allclose(log1p_cp10k(x * scale).toarray(), base, rtol=1e-12, atol=1e-12)
    per_row = rng.uniform(.01, 100., size=(40, 1))   # cell-specific depth
    assert np.allclose(log1p_cp10k(x * per_row).toarray(), base, rtol=1e-12, atol=1e-12)
    positive = totals > 0
    assert np.allclose(np.expm1(base[positive]).sum(1), 1e4)   # CP10k for every positive row
    assert not base[10].any()                                    # zero row stays zero, no error
    assert np.isclose(np.expm1(base[11, 3]), 1e4)


@pytest.mark.parametrize("bad", [-1e-9, np.nan, np.inf])
def test_normalisation_rejects_negative_or_nonfinite(bad):
    x = np.ones((3, 4)); x[1, 2] = bad
    with pytest.raises(ValueError):
        log1p_cp10k(x)
    with pytest.raises(ValueError):
        log1p_cp10k(sparse.csr_matrix(x))


def test_target_means_chunking_and_zero_rows():
    rng = np.random.default_rng(1)
    x = rng.poisson(2., size=(50, 8)).astype(float); x[[3, 17]] = 0
    labels = np.array(["a", "b", "c", "skip", "a"] * 10)
    expected = np.stack([log1p_cp10k(x[labels == t]).toarray().mean(0) for t in "abc"])
    for chunk in (1, 7, 64):
        assert np.allclose(target_means(sparse.csr_matrix(x), labels, list("abc"), chunk=chunk), expected, rtol=1e-12)
    with pytest.raises(ValueError):
        target_means(x, labels, ["a", "missing"])


# ---------------------------------------------------------------- synthetic context helpers

def _write_context(root, *, noise, seed=0, n_genes=60, n_targets=8, n_ref=24, n_cells=24, name="ctx_t"):
    """Private + public tree of one context. Each target's readout gene is in the panel."""
    rng = np.random.default_rng(seed)
    genes = [f"G{i}" for i in range(n_genes)]
    targets = [f"G{i}" for i in range(n_targets)]           # targets are panel genes
    rate = rng.gamma(2., 1., size=n_genes) + .5
    profiles = {}
    for k, t in enumerate(targets):
        eff = np.ones(n_genes); eff[k] = .1
        hit = rng.choice(np.arange(n_targets, n_genes), 8, replace=False); eff[hit] = rng.choice([.3, 3.], 8)
        profiles[t] = rate * eff * 20
    if noise:
        blocks = [rng.poisson(profiles[t], size=(n_ref, n_genes)) for t in targets]
    else:                                                    # identical cells within a target
        blocks = [np.tile(np.round(profiles[t]), (n_ref, 1)) for t in targets]
    resp_x = np.vstack(blocks).astype(np.float32)
    labels = np.repeat(targets, n_ref)
    order = rng.permutation(len(labels))                    # responses not grouped by target
    resp = ad.AnnData(sparse.csr_matrix(resp_x[order]),
                      obs=pd.DataFrame({"target_gene": labels[order], "target_ensembl": [f"ENSG{int(t[1:]):011d}.3" for t in labels[order]],
                                        "is_control": False}, index=[f"r{i}" for i in range(len(labels))]),
                      var=pd.DataFrame(index=genes))
    basal = ad.AnnData(sparse.csr_matrix(rng.poisson(rate * 20, size=(30, n_genes)).astype(np.float32)),
                       obs=pd.DataFrame({"target_gene": "non-targeting"}, index=[f"c{i}" for i in range(30)]),
                       var=pd.DataFrame({"ensembl_id": [f"ENSG{i:011d}.1" for i in range(n_genes)]}, index=genes))
    requests = [{"request_id": f"{name}:{i:04d}", "context": name, "perturbation": {"type": "crispri", "target": t},
                 "n_cells": n_cells, "seed": i} for i, t in enumerate(targets)]
    priv = root / "private" / name; pub = root / "test" / name / "public"
    priv.mkdir(parents=True); pub.mkdir(parents=True)
    resp.write_h5ad(priv / "responses.h5ad"); basal.write_h5ad(priv / "basal.h5ad"); basal.write_h5ad(pub / "basal.h5ad")
    (pub / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in requests))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    manifest = {"context_id": name, "tier": "near", "chemistry": "10x_3p_v3",
                "responses_sha256": sha(priv / "responses.h5ad"), "requests_sha256": sha(pub / "requests.jsonl"),
                "public_basal_sha256": sha(pub / "basal.h5ad"), "basal_sha256": sha(priv / "basal.h5ad"),
                "requests": [{"request_id": r["request_id"], "target": r["perturbation"]["target"],
                              "subset": "leaderboard" if i % 4 == 0 else "final", "stratum": "s", "withheld": False,
                              "seen_in_training": True} for i, r in enumerate(requests)]}
    (priv / "manifest.json").write_text(json.dumps(manifest))
    return {"priv": priv, "pub": pub, "requests": requests, "resp": resp, "basal": basal, "profiles": profiles}


def _prediction(ctx, make_block, path, order_seed=None):
    blocks, ids = [], []
    reqs = list(ctx["requests"])
    if order_seed is not None:
        reqs = [reqs[i] for i in np.random.default_rng(order_seed).permutation(len(reqs))]
    for r in reqs:
        b = make_block(r); blocks.append(b); ids += [r["request_id"]] * len(b)
    x = np.vstack(blocks).astype(np.float32)
    if order_seed is not None:
        rows = np.random.default_rng(order_seed + 1).permutation(len(ids)); x = x[rows]; ids = list(np.asarray(ids)[rows])
    ad.AnnData(sparse.csr_matrix(x), obs=pd.DataFrame({"request_id": ids}, index=[f"p{i}" for i in range(len(ids))]),
               var=pd.DataFrame(index=ctx["basal"].var_names)).write_h5ad(path)
    return path


def _reward(ctx, path, bundle=None):
    bundle = bundle or load_or_build_bundle(ctx["priv"], ctx["pub"] / "requests.jsonl")
    means = prediction_means(path, ctx["pub"] / "basal.h5ad", ctx["pub"] / "requests.jsonl", bundle["roster"])
    loss = score_against_bundle(means, bundle)["loss"]
    return score_context(loss=loss, negative_loss=bundle["negative_loss"], repeat_loss=bundle["repeat_loss"],
                         gap_lower95=bundle["gap_lower95"])


def _real_cells(ctx, target):
    return ctx["resp"][ctx["resp"].obs["target_gene"] == target].X.toarray()


def test_bundle_is_cached_and_tamper_checked(tmp_path):
    ctx = _write_context(tmp_path, noise=True)
    first = load_or_build_bundle(ctx["priv"], ctx["pub"] / "requests.jsonl")
    assert first["cache"].endswith("wmse_v2") and (ctx["priv"] / "wmse_v2" / "bundle.npz").is_file()
    again = load_or_build_bundle(ctx["priv"], ctx["pub"] / "requests.jsonl")
    assert np.array_equal(first["weights"], again["weights"]) and first["negative_loss"] == again["negative_loss"]
    final = [r for r in json.loads((ctx["priv"] / "manifest.json").read_text())["requests"] if r["subset"] == "final"]
    assert [x["request_id"] for x in first["roster"]] == sorted(r["request_id"] for r in final)
    assert all(x["readout_gene_index"] == int(x["target"][1:]) for x in first["roster"])  # stable-ID resolution
    w = first["weights"]
    assert w.shape[0] == 10 and np.allclose(w.sum(2), 1)
    assert all(w[:, k, x["readout_gene_index"]].max() == 0 for k, x in enumerate(first["roster"]))
    assert first["negative_loss"] > first["repeat_loss"] and 0 < first["gap_lower95"] <= first["gap"]
    meta = json.loads((ctx["priv"] / "wmse_v2" / "bundle.json").read_text())
    meta["negative_loss"] *= 2
    (ctx["priv"] / "wmse_v2" / "bundle.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):           # edited constants no longer match the stored hash/roster check
        load_or_build_bundle(ctx["priv"], ctx["pub"] / "requests.jsonl")


def test_noise_free_endpoints_are_exact(tmp_path):
    """Identical cells within a target: repeat error P = 0, so the true profile scores exactly 1
    and the equal-target generic mean scores exactly the negative anchor (reward 0)."""
    ctx = _write_context(tmp_path, noise=False)
    bundle = load_or_build_bundle(ctx["priv"], ctx["pub"] / "requests.jsonl")
    assert bundle["repeat_loss"] < 1e-20
    perfect = _reward(ctx, _prediction(ctx, lambda r: _real_cells(ctx, r["perturbation"]["target"]), tmp_path / "p.h5ad"), bundle)
    assert perfect["reward"] == 1. and perfect["raw_skill"] == pytest.approx(1., abs=1e-9)
    roster_targets = [x["target"] for x in bundle["roster"]]
    per = 24 // len(roster_targets); assert per * len(roster_targets) == 24
    pooled = np.vstack([_real_cells(ctx, t)[:per] for t in roster_targets])   # equal cells per roster target
    generic = _reward(ctx, _prediction(ctx, lambda r: pooled, tmp_path / "g.h5ad"), bundle)
    assert generic["raw_skill"] == pytest.approx(0., abs=1e-9) and generic["reward"] == pytest.approx(0., abs=1e-9)


def test_noisy_controls_and_reordering(tmp_path):
    ctx = _write_context(tmp_path, noise=True)
    bundle = load_or_build_bundle(ctx["priv"], ctx["pub"] / "requests.jsonl")
    oracle = lambda r: _real_cells(ctx, r["perturbation"]["target"])  # noqa: E731
    perfect = _reward(ctx, _prediction(ctx, oracle, tmp_path / "p.h5ad"), bundle)
    assert perfect["raw_skill"] > 1 and perfect["reward"] == 1.        # full-data mean beats the half repeat
    basal = ctx["basal"].X.toarray()
    noeffect = _reward(ctx, _prediction(ctx, lambda r: basal[:24], tmp_path / "n.h5ad"), bundle)
    assert noeffect["reward"] == 0. and noeffect["raw_skill"] < 0
    rng = np.random.default_rng(5)
    noisy = lambda r: rng.poisson(ctx["profiles"][r["perturbation"]["target"]], size=(24, 60)).astype(float)  # noqa: E731
    path = _prediction(ctx, noisy, tmp_path / "a.h5ad")
    ref = _reward(ctx, path, bundle)
    assert 0 < ref["reward"]
    # same cells, shuffled rows and request blocks in a different order
    x = ad.read_h5ad(path)
    for seed in (1, 2):
        rows = np.random.default_rng(seed).permutation(x.n_obs)
        x[rows].copy().write_h5ad(tmp_path / f"shuffled{seed}.h5ad")
        moved = _reward(ctx, tmp_path / f"shuffled{seed}.h5ad", bundle)
        assert moved["loss"] == pytest.approx(ref["loss"], rel=1e-12) and moved["reward"] == pytest.approx(ref["reward"], rel=1e-12)
    # counts rescaled per cell (sequencing depth) leave the reward unchanged
    x.X = sparse.csr_matrix(x.X.toarray().astype(np.float64) * np.random.default_rng(3).uniform(.05, 20., size=(x.n_obs, 1)))
    x.write_h5ad(tmp_path / "scaled.h5ad")
    assert _reward(ctx, tmp_path / "scaled.h5ad", bundle)["reward"] == pytest.approx(ref["reward"], rel=1e-9)


def test_request_file_order_does_not_change_reward(tmp_path):
    a = _write_context(tmp_path / "a", noise=True)
    b = _write_context(tmp_path / "b", noise=True)
    lines = (b["pub"] / "requests.jsonl").read_text().splitlines(keepends=True)
    (b["pub"] / "requests.jsonl").write_text("".join(lines[::-1]))
    rng = np.random.default_rng(9)
    blocks = {r["request_id"]: rng.poisson(a["profiles"][r["perturbation"]["target"]], size=(24, 60)).astype(float) for r in a["requests"]}
    pa = _prediction(a, lambda r: blocks[r["request_id"]], tmp_path / "pa.h5ad")
    pb = _prediction(b, lambda r: blocks[r["request_id"]], tmp_path / "pb.h5ad", order_seed=11)
    ra, rb = _reward(a, pa), _reward(b, pb)
    assert ra["reward"] == pytest.approx(rb["reward"], rel=1e-12) and ra["negative_loss"] == rb["negative_loss"]


def test_prediction_contract_violations(tmp_path):
    ctx = _write_context(tmp_path, noise=True)
    bundle = load_or_build_bundle(ctx["priv"], ctx["pub"] / "requests.jsonl")
    good = lambda r: np.ones((24, 60))  # noqa: E731
    path = _prediction(ctx, good, tmp_path / "ok.h5ad")
    args = (ctx["pub"] / "basal.h5ad", ctx["pub"] / "requests.jsonl", bundle["roster"])
    prediction_means(path, *args)
    x = ad.read_h5ad(path)
    cases = {"missing": x[x.obs["request_id"] != ctx["requests"][1]["request_id"]].copy(),
             "short": x[2:].copy(), "genes": x[:, ::-1].copy()}
    extra = x[:24].copy(); extra.obs["request_id"] = "ctx_t:9999"; extra.obs_names = [f"e{i}" for i in range(24)]
    cases["extra"] = ad.concat([x, extra])
    neg = x.copy(); neg.X = sparse.csr_matrix(neg.X.toarray() - 2); cases["negative"] = neg
    for name, bad in cases.items():
        bad.write_h5ad(tmp_path / f"{name}.h5ad")
        with pytest.raises(ValueError):
            prediction_means(tmp_path / f"{name}.h5ad", *args)


def test_grader_emits_v2_reward_and_keeps_v1(tmp_path):
    """End-to-end grader on two synthetic contexts; one context's inference failed."""
    ok = _write_context(tmp_path, noise=True, name="ctx_a", seed=1)
    bad = _write_context(tmp_path, noise=True, name="ctx_b", seed=2)
    sealed = tmp_path / "sealed" / "run"; (sealed / "ctx_a").mkdir(parents=True)
    rng = np.random.default_rng(4)
    _prediction(ok, lambda r: rng.poisson(ok["profiles"][r["perturbation"]["target"]], size=(24, 60)).astype(float),
                sealed / "ctx_a" / "predictions.h5ad")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    snap = {"schema": "virtual-cell-zeroshot-snapshot/v1", "contexts": {
        "ctx_a": {"predictions_sha256": sha(sealed / "ctx_a" / "predictions.h5ad"), "requests_sha256": sha(ok["pub"] / "requests.jsonl"),
                  "basal_sha256": sha(ok["pub"] / "basal.h5ad"), "seconds": 1.0},
        "ctx_b": {"failed": True, "error": "CalledProcessError: exit 1"}}}
    (sealed / "snapshot.json").write_text(json.dumps(snap))
    (sealed.parent / "READY").write_text(sha(sealed / "snapshot.json") + "\n")
    out = tmp_path / "results"
    subprocess.run([sys.executable, str(TASK / "tests" / "grader.py"), "--sealed", str(sealed), "--benchmark", str(tmp_path / "test"),
                    "--private", str(tmp_path / "private"), "--output", str(out), "--threads", "2"], check=True)
    reward = json.loads((out / "reward.json").read_text()); report = json.loads((out / "report.json").read_text())
    expected = _reward(ok, sealed / "ctx_a" / "predictions.h5ad")
    bundle_a = load_or_build_bundle(ok["priv"], ok["pub"] / "requests.jsonl")
    means_a = prediction_means(sealed / "ctx_a" / "predictions.h5ad", ok["pub"] / "basal.h5ad", ok["pub"] / "requests.jsonl", bundle_a["roster"])
    meas_a = score_against_bundle(means_a, bundle_a)
    expected_v3_a = score_context_v3(
        loss=meas_a["loss"], negative_loss=bundle_a["negative_loss"], repeat_loss=bundle_a["repeat_loss"],
        gap_lower95=bundle_a["gap_lower95"], nir=meas_a["nir"], repeat_nir=meas_a["repeat_nir"], targets=len(bundle_a["roster"])
    )
    assert reward["v2_ctx_a"] == pytest.approx(expected["reward"], rel=1e-12) and reward["v2_ctx_b"] == 0.
    assert reward["v2_reward"] == pytest.approx(expected["reward"] / 2, rel=1e-12)
    assert reward["v3_ctx_a"] == pytest.approx(expected_v3_a["context_score"], rel=1e-12) and reward["v3_ctx_b"] == 0.
    assert reward["score"] == pytest.approx(min(1.0, expected_v3_a["context_score"] / 2), rel=1e-12)
    assert reward["reward"] == pytest.approx(reward["score"], rel=1e-12)
    assert report["score"] == reward["score"] and report["reward"] == reward["reward"]
    assert report["contexts"]["ctx_b"]["v2"]["status"] == "invalid_submission" and report["contexts"]["ctx_b"]["v3"]["status"] == "invalid_submission"
    assert reward["v1_ctx_b_final"] == 0. and "v1_reward" in reward and "v1_ctx_a_final" in reward and "leaderboard_reward" not in reward
    assert reward["v1_reward"] == pytest.approx(report["contexts"]["ctx_a"]["v1_final_score"] / 2)
    assert report["contexts"]["ctx_a"]["v2"]["n_targets"] == 6 and len(report["contexts"]["ctx_a"]["v2"]["per_target_loss"]) == 6
    assert 0.5 < report["contexts"]["ctx_a"]["v2"]["nir"] <= 1.0
    assert 0.5 < report["contexts"]["ctx_a"]["v2"]["repeat_nir"] <= 1.0
    assert len(report["contexts"]["ctx_a"]["v2"]["per_target_nir"]) == 6
    assert reward["v2_ctx_a_nir"] == pytest.approx(report["contexts"]["ctx_a"]["v2"]["nir"], rel=1e-12)
    assert "v2_ctx_b_nir" not in reward and report["contexts"]["ctx_b"]["v2"]["nir"] is None
    # Verify target-blind constant prediction yields exact analytical NIR = 0.5 and panel target genes are excluded
    bundle_a = load_or_build_bundle(ok["priv"], ok["pub"] / "requests.jsonl")
    const_means = np.tile(bundle_a["truth"][0].mean(0), (len(bundle_a["roster"]), 1))
    assert score_against_bundle(const_means, bundle_a)["nir"] == pytest.approx(0.5, abs=1e-12)


def test_public_development_score_uses_full_dev_roster_without_private_manifest(tmp_path):
    ctx = _write_context(tmp_path, noise=True)
    req_path = ctx['pub'] / 'requests.jsonl'
    bundle = load_or_build_bundle(ctx['priv'], req_path, tmp_path/'dev-cache', development=True)
    expected_ids = {json.loads(row)['request_id'] for row in req_path.read_text().splitlines()}
    assert {r['request_id'] for r in bundle['roster']} == expected_ids
    assert not (TASK/'environment/evaluation/wmse_roster_exclusions.json').exists()
