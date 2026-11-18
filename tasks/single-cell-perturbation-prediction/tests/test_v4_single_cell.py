#!/usr/bin/env python3
"""CPU-runnable V4 unit & integration tests for #16 single-cell-perturbation-prediction.

Covers:
1. Exact V3 pre-clip signed-mean context formula (`amplitude_skill = +0.4`,
   `identification_skill = -0.2` -> `context_score = +0.10`, NOT `0.20`) and
   `< 1e-12` parity with `harness/ssi_index_v3.py:cell()`.
2. End-to-end two-pass `main_verifier.py` -> `grader.py` query-independence gate:
   (a) Legitimate stochastic model seeded deterministically per `request_id` passes
       with zero false positives (`score > 0`, `< 1e-12` parity with `ssi_index_v3.cell()`).
   (b) Cross-query James-Stein / batch-mean shrinkage cheater (including attempted
       `$HOME`/`$TMPDIR` cross-pass caching) fails `compare_request_predictions` with
       zero false negatives (`score == 0.0`, `reward == 0.0`).
3. Byte-identical scorer module copies, visibility boundary (`tests/` never in dev
   `/workspace`), and 3-part Literature Justification in `tests/README.md`.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "harness" / "ssi_index_v3.py").is_file())

_PY_BIN_DIR = str(Path(sys.executable).parent)
if _PY_BIN_DIR not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _PY_BIN_DIR + os.pathsep + os.environ.get("PATH", "")
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _install_pytest_fallback_if_missing() -> None:
    try:
        import pytest  # noqa: F401
    except ModuleNotFoundError:
        import contextlib
        import re

        class _Approx:
            def __init__(self, target, rel=1e-6, abs=1e-12):
                self.target = target
                self.rel = rel
                self.abs = abs

            def __eq__(self, other):
                if isinstance(self.target, dict) and isinstance(other, dict):
                    return set(self.target) == set(other) and all(
                        _Approx(self.target[k], self.rel, self.abs) == other[k] for k in self.target
                    )
                a = np.asarray(other, dtype=np.float64)
                b = np.asarray(self.target, dtype=np.float64)
                return bool(np.allclose(a, b, rtol=self.rel, atol=self.abs))

        @contextlib.contextmanager
        def _raises(exc_type, match=None):
            try:
                yield
            except exc_type as exc:
                if match is not None and not re.search(match, str(exc)):
                    raise AssertionError(f"{exc!r} did not match {match!r}")
                return
            raise AssertionError(f"Did not raise {exc_type}")

        class _Mark:
            @staticmethod
            def parametrize(argnames, argvalues):
                def _dec(fn):
                    fn._parametrize = (argnames, list(argvalues))
                    return fn
                return _dec

        pm = types.ModuleType("pytest")
        pm.approx = lambda val, rel=1e-6, abs=1e-12: _Approx(val, rel=rel, abs=abs)
        pm.raises = _raises
        pm.mark = _Mark()
        sys.modules["pytest"] = pm


_install_pytest_fallback_if_missing()

sys.path.insert(0, str(TASK_DIR / "tests"))
sys.path.insert(0, str(REPO_ROOT / "harness"))

import grader as vcz_grader  # noqa: E402
import main_verifier as vcz_main  # noqa: E402
import ssi_index_v3  # noqa: E402
from calibrated_reward_v2 import aggregate_contexts_v3, score_context_v3  # noqa: E402
from test_wmse_v2_grader import _write_context  # noqa: E402


class V4SingleCellTests(unittest.TestCase):
    def test_preclip_vs_postclip_order_of_operations_and_v3_parity(self) -> None:
        # Construct exact anchor inputs producing amplitude_skill = +0.4, identification_skill = -0.2:
        # negative_loss = 2.0, repeat_loss = 1.0 -> gap = 1.0; loss = 1.6 -> amplitude_skill = +0.4
        # repeat_nir = 0.75 -> denom = 0.25; nir = 0.45 -> identification_skill = (0.45 - 0.5) / 0.25 = -0.2
        ctx_near = score_context_v3(
            loss=1.6,
            negative_loss=2.0,
            repeat_loss=1.0,
            gap_lower95=0.2,
            nir=0.45,
            repeat_nir=0.75,
            targets=50,
        )
        self.assertAlmostEqual(ctx_near["amplitude_skill"], 0.4, places=12)
        self.assertAlmostEqual(ctx_near["identification_skill"], -0.2, places=12)
        # Pre-clip average = 0.5 * (0.4 + (-0.2)) = +0.10 (NOT 0.5 * (0.4 + 0.0) = +0.20!)
        self.assertAlmostEqual(ctx_near["signed_context"], 0.10, places=12)
        self.assertAlmostEqual(ctx_near["context_score"], 0.10, places=12)

        # Second context where negative identification outweighs positive amplitude -> floors to 0.0:
        # amplitude_skill = +0.2, identification_skill = -0.6 -> signed_context = -0.20 -> context_score = 0.0
        ctx_mid = score_context_v3(
            loss=1.8,
            negative_loss=2.0,
            repeat_loss=1.0,
            gap_lower95=0.2,
            nir=0.35,
            repeat_nir=0.75,
            targets=50,
        )
        self.assertAlmostEqual(ctx_mid["signed_context"], -0.20, places=12)
        self.assertAlmostEqual(ctx_mid["context_score"], 0.0, places=12)

        # Third context with both skills positive:
        # amplitude_skill = +0.5, identification_skill = +0.8 -> signed_context = +0.65 -> context_score = +0.65
        ctx_far = score_context_v3(
            loss=1.5,
            negative_loss=2.0,
            repeat_loss=1.0,
            gap_lower95=0.2,
            nir=0.70,
            repeat_nir=0.75,
            targets=50,
        )
        self.assertAlmostEqual(ctx_far["signed_context"], 0.65, places=12)
        self.assertAlmostEqual(ctx_far["context_score"], 0.65, places=12)

        agg = aggregate_contexts_v3(
            {
                "near": {
                    "loss": 1.6, "negative_loss": 2.0, "repeat_loss": 1.0,
                    "gap_lower95": 0.2, "nir": 0.45, "repeat_nir": 0.75, "targets": 50,
                },
                "mid": {
                    "loss": 1.8, "negative_loss": 2.0, "repeat_loss": 1.0,
                    "gap_lower95": 0.2, "nir": 0.35, "repeat_nir": 0.75, "targets": 50,
                },
                "far": {
                    "loss": 1.5, "negative_loss": 2.0, "repeat_loss": 1.0,
                    "gap_lower95": 0.2, "nir": 0.70, "repeat_nir": 0.75, "targets": 50,
                },
            },
            expected_contexts=("near", "mid", "far"),
        )
        v3_ref = ssi_index_v3.cell(agg["contexts"])
        self.assertLess(abs(agg["score"] - v3_ref["score"]), 1e-12)
        self.assertLess(abs(agg["reward"] - v3_ref["score"]), 1e-12)
        self.assertAlmostEqual(agg["score"], (0.10 + 0.0 + 0.65) / 3.0, places=12)

    def test_two_pass_end_to_end_legitimate_stochastic_vs_cross_query_shrinkage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctx_specs = [("ctx_near", "near", 11), ("ctx_mid", "mid", 22), ("ctx_far", "far", 33)]
            true_profiles: dict[str, dict[str, list[float]]] = {}
            for cname, tier, seed in ctx_specs:
                info = _write_context(root, noise=True, seed=seed, n_genes=60, n_targets=8, n_ref=24, n_cells=24, name=cname)
                # Set distinct tier in private manifest ("near", "mid", "far")
                mpath = info["priv"] / "manifest.json"
                manifest = json.loads(mpath.read_text())
                manifest["tier"] = tier
                mpath.write_text(json.dumps(manifest))
                (info["pub"] / "context.json").write_text(json.dumps({
                    "context_id": cname, "chemistry": "10x_3p_v3", "n_genes": 60,
                    "n_basal_cells": 30, "n_requests": len(info["requests"]), "perturbation_type": "crispri",
                }))
                true_profiles[cname] = {t: prof.tolist() for t, prof in info["profiles"].items()}

            profiles_json = root / "oracle_profiles.json"
            profiles_json.write_text(json.dumps(true_profiles))

            fake_pw = types.SimpleNamespace(pw_uid=os.getuid(), pw_gid=os.getgid(), pw_dir=str(root / "unused_home"))

            def _make_submission(sub_dir: Path, mode: str) -> None:
                sub_dir.mkdir(parents=True, exist_ok=True)
                (sub_dir / "README.md").write_text(f"Test submission ({mode})")
                (sub_dir / "model.py").write_text(
                    f"""import hashlib, json, os, sys
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

inp, out = Path(sys.argv[1]), Path(sys.argv[2])
basal = ad.read_h5ad(inp / "basal.h5ad")
requests = [json.loads(line) for line in (inp / "requests.jsonl").read_text().splitlines() if line.strip()]
profiles = json.loads(Path({str(profiles_json)!r}).read_text())
mode = {mode!r}

# Attempted cross-pass cache cheat via $HOME (blocked by ephemeral HOME per pass!)
cache_file = Path(os.environ["HOME"]) / "pass1_cache.h5ad"

if mode == "legitimate_stochastic":
    blocks, ids = [], []
    for r in requests:
        rid = r["request_id"]
        ctx_name = r["context"]
        target = r["perturbation"]["target"]
        # Seeded strictly by request_id — invariant to batch composition and request order!
        seed_int = int.from_bytes(hashlib.sha256(rid.encode()).digest()[:8], "big")
        rng = np.random.default_rng(seed_int)
        rate = np.asarray(profiles[ctx_name][target], dtype=np.float64)
        sample = rng.poisson(rate, size=(int(r["n_cells"]), basal.n_vars)).astype(np.float32)
        blocks.append(sparse.csr_matrix(sample))
        ids.extend([rid] * int(r["n_cells"]))
else:
    # Cross-query James-Stein / batch-mean shrinkage cheater:
    # shrinks every target's rate 35% toward the mean across all requests in requests.jsonl!
    raw_rates = [np.asarray(profiles[r["context"]][r["perturbation"]["target"]], dtype=np.float64) for r in requests]
    batch_mean = np.mean(raw_rates, axis=0)
    blocks, ids = [], []
    for r, rate in zip(requests, raw_rates):
        rid = r["request_id"]
        shrunk = 0.65 * rate + 0.35 * batch_mean
        sample = np.tile(np.clip(shrunk, 0.0, None).astype(np.float32), (int(r["n_cells"]), 1))
        blocks.append(sparse.csr_matrix(sample))
        ids.extend([rid] * int(r["n_cells"]))

out.mkdir(parents=True, exist_ok=True)
pred = ad.AnnData(
    X=sparse.vstack(blocks).tocsr(),
    obs=pd.DataFrame({{"request_id": ids}}, index=[f"p{{i}}" for i in range(len(ids))]),
    var=pd.DataFrame(index=basal.var_names.astype(str)),
)
if mode != "legitimate_stochastic" and cache_file.is_file():
    # If HOME leaked across passes, a cheater could try replaying cached Pass-1 rows
    cached = ad.read_h5ad(cache_file)
    wanted = {{r["request_id"] for r in requests}}
    pred = cached[cached.obs["request_id"].isin(wanted)].copy()
else:
    pred.write_h5ad(cache_file)
pred.write_h5ad(out / "predictions.h5ad")
"""
                )
                (sub_dir / "infer.sh").write_text(
                    f'#!/usr/bin/env bash\nset -euo pipefail\n"{sys.executable}" model.py "$1" "$2"\n'
                )

            def _run_two_stage(sub_dir: Path, run_dir: Path) -> tuple[dict, dict]:
                sealed_run = run_dir / "sealed" / "run"
                main_out = run_dir / "main_logs"
                grader_out = run_dir / "grader_results"
                with (
                    mock.patch.object(vcz_main.pwd, "getpwnam", return_value=fake_pw),
                    mock.patch.object(vcz_main.os, "chown", lambda *a, **kw: None),
                    mock.patch.object(vcz_main.os, "setuid", lambda *a, **kw: None),
                    mock.patch.object(vcz_main.os, "setgid", lambda *a, **kw: None),
                    mock.patch.object(vcz_main, "kill_uid", lambda *a, **kw: None),
                    mock.patch.object(
                        sys,
                        "argv",
                        [
                            "main_verifier.py",
                            "--submission", str(sub_dir),
                            "--benchmark", str(root / "test"),
                            "--sealed", str(sealed_run),
                            "--output", str(main_out),
                            "--timeout-per-context", "60",
                            "--no-wait",
                        ],
                    ),
                ):
                    vcz_main.main()
                with mock.patch.object(
                    sys,
                    "argv",
                    [
                        "grader.py",
                        "--sealed", str(sealed_run),
                        "--benchmark", str(root / "test"),
                        "--private", str(root / "private"),
                        "--output", str(grader_out),
                        "--expected-contexts", "ctx_near", "ctx_mid", "ctx_far",
                        "--require-query-independence",
                        "--threads", "2",
                    ],
                ):
                    vcz_grader.main()
                rew = json.loads((grader_out / "reward.json").read_text())
                rep = json.loads((grader_out / "report.json").read_text())
                return rew, rep

            # (a) Legitimate stochastic model seeded per request_id -> PASSES with 0 false positives!
            legit_pkg = root / "pkg_legit"
            _make_submission(legit_pkg, "legitimate_stochastic")
            rew_ok, rep_ok = _run_two_stage(legit_pkg, root / "run_legit")
            self.assertEqual(rep_ok["status"], "ok")
            self.assertTrue(rep_ok["query_independence"]["passed"])
            self.assertEqual(rep_ok["query_independence"]["total_changed_requests"], 0)
            self.assertGreater(rew_ok["score"], 0.5)
            self.assertAlmostEqual(rew_ok["score"], rew_ok["reward"], places=12)
            v3_exact = ssi_index_v3.cell(rep_ok["v3_contexts"])
            self.assertLess(abs(rew_ok["score"] - v3_exact["score"]), 1e-12)
            self.assertLess(abs(rep_ok["score"] - v3_exact["score"]), 1e-12)

            # (b) Cross-query James-Stein / batch-mean shrinkage cheater -> REJECTED with score = 0.0!
            cheat_pkg = root / "pkg_cheat"
            _make_submission(cheat_pkg, "batch_shrinkage_cheater")
            rew_bad, rep_bad = _run_two_stage(cheat_pkg, root / "run_cheat")
            self.assertEqual(rep_bad["status"], "rejected_query_dependence")
            self.assertFalse(rep_bad["query_independence"]["passed"])
            self.assertEqual(
                set(rep_bad["query_independence"]["failed_contexts"]),
                {"ctx_near", "ctx_mid", "ctx_far"},
            )
            self.assertGreater(rep_bad["query_independence"]["total_changed_requests"], 0)
            self.assertEqual(rew_bad["score"], 0.0)
            self.assertEqual(rew_bad["reward"], 0.0)
            self.assertEqual(rep_bad["score"], 0.0)
            self.assertEqual(rep_bad["reward"], 0.0)

    def test_copies_identical_and_visibility_boundary(self) -> None:
        for name in (
            "calibrated_reward_v2.py",
            "population_metrics_v2.py",
            "query_independence.py",
            "cellscore.py",
            "prediction_safety.py",
        ):
            a = (TASK_DIR / "tests" / name).read_bytes()
            b = (TASK_DIR / "environment" / "evaluation" / name).read_bytes()
            self.assertEqual(
                hashlib.sha256(a).hexdigest(),
                hashlib.sha256(b).hexdigest(),
                f"{name} differs between tests/ and environment/evaluation/",
            )

        dev_dockerfile = (TASK_DIR / "environment" / "Dockerfile").read_text()
        dev_compose = (TASK_DIR / "environment" / "docker-compose.yaml").read_text()
        self.assertNotIn("COPY tests", dev_dockerfile)
        self.assertNotIn("/tests", dev_dockerfile)
        self.assertNotIn("tests/", dev_compose)
        self.assertNotIn("TASK_PRIVATE_DIR", dev_compose)

        readme = (TASK_DIR / "tests" / "README.md").read_text()
        for heading in (
            "## Scientific & Literature Justification of Verifier Design",
            "### 1. Primary Metric & Upstream Paper Provenance",
            "### 2. Why Additional Ad-Hoc Constraints Are Intentionally Omitted",
            "### 3. Diagnostic Signals Retained in report.json",
        ):
            self.assertIn(heading, readme)

    def test_existing_single_cell_suites_pass_on_cpu(self) -> None:
        import inspect
        import test_calibrated_reward_v2
        import test_prediction_safety
        import test_query_independence
        import test_task
        import test_wmse_v2_grader

        for mod in (
            test_calibrated_reward_v2,
            test_query_independence,
            test_prediction_safety,
            test_task,
            test_wmse_v2_grader,
        ):
            for name, fn in inspect.getmembers(mod, inspect.isfunction):
                if not name.startswith("test_"):
                    continue
                sig = inspect.signature(fn)
                params = list(sig.parameters)
                if hasattr(fn, "_parametrize"):
                    arg_name, values = fn._parametrize
                    for val in values:
                        with tempfile.TemporaryDirectory() as td:
                            kwargs = {}
                            if "tmp_path" in params:
                                kwargs["tmp_path"] = Path(td)
                            kwargs[arg_name] = val
                            fn(**kwargs)
                else:
                    with tempfile.TemporaryDirectory() as td:
                        kwargs = {"tmp_path": Path(td)} if "tmp_path" in params else {}
                        fn(**kwargs)


if __name__ == "__main__":
    unittest.main()
