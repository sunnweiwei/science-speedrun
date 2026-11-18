"""V4 security, seed-isolation, full-tree leak audit, and ssi_index_v3 parity tests (CPU-only)."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import stat
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
TASK_ROOT = TESTS_DIR.parent
REPO_ROOT = TASK_ROOT.parent

for _p in (TASK_ROOT / "environment", TASK_ROOT / "environment" / "evaluation", TESTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tokagym import CASES  # noqa: E402
import harbor_verify as hv  # noqa: E402
import generate_sealed_perturbations as gen_sealed  # noqa: E402


def _load_ssi_index_v3():
    spec_path = next(p for p in Path(__file__).resolve().parents if (p / "harness" / "ssi_index_v3.py").is_file()) / "harness" / "ssi_index_v3.py"
    spec = importlib.util.spec_from_file_location("ssi_index_v3", spec_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


LEGACY_SEED_RE = re.compile(r"(?<![0-9])(100[1-9]|101[0-5])(?![0-9])")


class TokamakV4SealedSecurityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ssi_v3 = _load_ssi_index_v3()
        cls.sealed_path = TESTS_DIR / "sealed" / "perturbations.json"
        if not cls.sealed_path.is_file():
            raise unittest.SkipTest("sealed files are distributed separately (tests/sealed/README.md)")
        cls.legacy_path = TESTS_DIR / "fixtures" / "legacy_perturbations_1001_1015.json"
        cls.sealed = json.loads(cls.sealed_path.read_text())
        cls.legacy = json.loads(cls.legacy_path.read_text())

    def test_sealed_perturbations_expand_to_105_unique_episodes_and_zero_legacy_overlap(self):
        seeds = self.sealed["seeds"]
        legacy_seeds = set(self.legacy["seeds"])
        self.assertEqual(legacy_seeds, set(range(1001, 1016)))
        self.assertEqual(self.sealed["name"], "sealed-v0.4")
        self.assertEqual(len(CASES), 7)
        self.assertEqual(len(seeds), 15)
        self.assertEqual(len(set(seeds)), 15)

        # Every seed must be a positive 31-bit integer, non-consecutive, and disjoint from legacy/dev/public ranges.
        for s in seeds:
            self.assertIsInstance(s, int)
            self.assertGreaterEqual(s, 1_000_000)
            self.assertLess(s, 1 << 31)
        sorted_seeds = sorted(seeds)
        for a, b in zip(sorted_seeds, sorted_seeds[1:]):
            self.assertGreater(b - a, 1000, f"Seeds {a} and {b} are too close / consecutive")

        self.assertEqual(set(seeds) & legacy_seeds, set())
        self.assertEqual(set(seeds) & set(range(0, 10_001)), set())
        self.assertEqual(set(seeds) & set(range(100_000, 101_000)), set())

        scheduled_pairs = [(case_name, seed) for case_name in sorted(CASES) for seed in seeds]
        self.assertEqual(len(scheduled_pairs), 105)
        self.assertEqual(len(set(scheduled_pairs)), 105)

        # Physics distribution matches frozen FAIR-MAST specification.
        self.assertEqual(self.sealed["randomization"], self.legacy["randomization"])

    def test_generator_reproducibility_and_key_sensitivity(self):
        key_a = b"unit-test-secret-key-alpha-0123456789"
        key_b = b"unit-test-secret-key-bravo-9876543210"
        seeds_a1 = gen_sealed.derive_sealed_seeds(key_a)
        seeds_a2 = gen_sealed.derive_sealed_seeds(key_a)
        seeds_b = gen_sealed.derive_sealed_seeds(key_b)
        self.assertEqual(seeds_a1, seeds_a2)
        self.assertEqual(len(seeds_a1), 15)
        self.assertEqual(set(seeds_a1) & set(seeds_b), set())
        with self.assertRaises(ValueError):
            gen_sealed.derive_sealed_seeds(b"short")

    def test_no_file_outside_tests_contains_legacy_or_new_sealed_seeds(self):
        new_seed_tokens = {str(s) for s in self.sealed["seeds"]}
        leaked_legacy = []
        leaked_new = []

        for path in sorted(TASK_ROOT.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(TASK_ROOT)
            if rel.parts[0] == "tests" or "__pycache__" in rel.parts:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                if LEGACY_SEED_RE.search(line):
                    leaked_legacy.append(f"{rel}:{line_no}: {line.strip()}")
                for tok in new_seed_tokens:
                    if tok in line:
                        leaked_new.append(f"{rel}:{line_no}: {tok}")

        self.assertEqual(leaked_legacy, [], f"Legacy 1001..1015 seed references found outside tests/: {leaked_legacy}")
        self.assertEqual(leaked_new, [], f"New sealed-v0.4 seeds leaked outside tests/: {leaked_new}")

    def test_dockerfile_and_compose_never_expose_tests_to_development_image(self):
        dockerfile = (TASK_ROOT / "environment" / "Dockerfile").read_text()
        # Extract the development stage (everything from 'FROM base AS development' to EOF).
        self.assertIn("FROM base AS development", dockerfile)
        dev_stage = dockerfile.split("FROM base AS development", 1)[1]
        self.assertNotIn("COPY tests", dev_stage)
        self.assertNotIn("sealed", dev_stage)

        dev_compose = (TASK_ROOT / "environment" / "docker-compose.yaml").read_text()
        self.assertIn("target: development", dev_compose)
        self.assertNotIn("tests", dev_compose)
        self.assertNotIn("sealed", dev_compose)

    def test_spy_controller_never_learns_secret_seed_via_reset_context_or_fds(self):
        # 1. Verify reset_context across all 105 scheduled episodes.
        dummy_env = types.SimpleNamespace(spec=types.SimpleNamespace(name="MAST-U-like"), horizon=700)
        secret_seeds = self.sealed["seeds"]
        observed_public_seeds = []
        for ep_idx, (case_name, _secret_seed) in enumerate(
            (c, s) for c in sorted(CASES) for s in secret_seeds
        ):
            ctx = hv.reset_context(case_name, dummy_env, ep_idx)
            observed_public_seeds.append(ctx["seed"])

        self.assertEqual(observed_public_seeds, [100_000 + i for i in range(105)])
        self.assertEqual(set(observed_public_seeds) & set(secret_seeds), set())

        # 2. End-to-end subprocess spy policy replay through run_case_stream + measure_latency.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            spy_log = tmp_path / "spy_observations.jsonl"
            pkg = tmp_path / "spy_pkg"
            pkg.mkdir()
            spy_code = f"""import json, os, sys
LOG_PATH = {str(spy_log)!r}
def record(kind, payload):
    fd2 = None
    try:
        fd2 = os.readlink("/proc/self/fd/2")
    except OSError:
        fd2 = ""
    leaked_env = {{k: v for k, v in os.environ.items() if "SEALED" in k or "SSI_" in k}}
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({{"kind": kind, "payload": payload, "fd2": fd2, "env": leaked_env}}) + "\\n")

for line in sys.stdin:
    req = json.loads(line)
    op = req["op"]
    out = {{"request_id": req["request_id"]}}
    if op == "reset":
        record("reset", req["context"])
        out["ready"] = True
    elif op == "act":
        out["action"] = [0.0] * 12
    elif op == "close":
        out["closed"] = True
    print(json.dumps(out), flush=True)
    if op == "close":
        break
"""
            (pkg / "policy.py").write_text(spy_code)
            infer_sh = pkg / "infer.sh"
            infer_sh.write_text('#!/usr/bin/env bash\ncd "$(dirname "$0")"\nexec "' + sys.executable + '" policy.py\n')
            os.chmod(infer_sh, os.stat(infer_sh).st_mode | stat.S_IEXEC)

            simulator_seeds_seen = []

            class FakeTokamakEnv:
                def __init__(self, case_name, randomization=None, solver="nonlinear", strict_actions=True):
                    self.case_name = case_name
                    self.spec = types.SimpleNamespace(name="MAST-U-like")
                    self.n_obs = 585
                    self.n_active = 12
                    self.vmax = np.full(12, 750.0)
                    self.horizon = 3
                    self.budget_scale = 1.0
                    self.on_progress = lambda: None
                    self.numerics_level = 0
                    self.failure_kind = None
                    self.terminated_reason = None
                    self.records = []
                    self._step = 0

                def reset(self, seed=None):
                    simulator_seeds_seen.append((self.case_name, seed))
                    self._step = 0
                    self.records = []
                    return np.zeros(self.n_obs, dtype=float), {"plasma_resistivity": 3.8e-7}

                def step(self, action):
                    self._step += 1
                    rec = {"step_time_sec": 0.001, "shape_rms": 0.004, "ip_error_rel": 0.002}
                    self.records.append(rec)
                    done = self._step >= self.horizon
                    return np.zeros(self.n_obs, dtype=float), 1.0, False, done, rec

                def episode_summary(self):
                    return {
                        "outcome": "success",
                        "feasible": True,
                        "complete": True,
                        "steps": self.horizon,
                        "tolerance_time_fraction": 1.0,
                        "shape_rms_steady": 0.004,
                        "xpoint_error_steady": 0.003,
                        "rin_error_steady": 0.003,
                        "rout_error_steady": 0.003,
                        "zx_error_steady": 0.003,
                        "ip_error_rel_steady": 0.002,
                    }

            class ListQueue(list):
                put = list.append

            def _run_suite_with_seeds(test_seeds, out_subdir):
                out_dir = tmp_path / out_subdir
                q = ListQueue()
                cases_subset = sorted(CASES)[:2]
                for c_idx, c_name in enumerate(cases_subset):
                    ep_map = {s: c_idx * len(test_seeds) + s_idx for s_idx, s in enumerate(test_seeds)}
                    args = (c_name, str(pkg), str(out_dir), "visible", 30.0, 0, test_seeds[0], ep_map)
                    hv.run_case_stream(args, test_seeds, q)
                hv.measure_latency(cases_subset, str(pkg), out_dir, 30.0)

            secret_batch_1 = secret_seeds[:3]
            secret_batch_2 = secret_seeds[3:6]
            with mock.patch.object(hv, "TokamakMagneticEnv", FakeTokamakEnv), \
                 mock.patch.dict(os.environ, {"SSI_TOKAMAK_SEALED_KEY": "top-secret-key-do-not-leak"}):
                _run_suite_with_seeds(secret_batch_1, "replay_1")
                spy_entries_1 = [json.loads(line) for line in spy_log.read_text().splitlines()]
                spy_log.unlink()
                _run_suite_with_seeds(secret_batch_2, "replay_2")
                spy_entries_2 = [json.loads(line) for line in spy_log.read_text().splitlines()]

            # Physics simulator received exact secret seeds for both batches.
            expected_sim_calls = (
                [(c, s) for c in sorted(CASES)[:2] for s in secret_batch_1]
                + [(c, s) for c in sorted(CASES)[:2] for s in secret_batch_2]
            )
            self.assertEqual(simulator_seeds_seen, expected_sim_calls)

            # Policy received identical public contexts across two completely disjoint secret seed batches!
            contexts_1 = [e["payload"] for e in spy_entries_1]
            contexts_2 = [e["payload"] for e in spy_entries_2]
            self.assertEqual(contexts_1, contexts_2)

            all_secret_strs = {str(s) for s in secret_batch_1 + secret_batch_2}
            for entry in spy_entries_1 + spy_entries_2:
                self.assertNotIn(str(entry["payload"]["seed"]), all_secret_strs)
                self.assertEqual(entry["env"], {})
                for s_str in all_secret_strs:
                    self.assertNotIn(s_str, entry["fd2"])

    def test_void_and_failure_denominator_and_ssi_index_v3_parity(self):
        case_names = sorted(CASES)

        def make_ep(outcome):
            feasible = outcome in ("success", "kept")
            return {
                "outcome": outcome,
                "feasible": feasible,
                "tolerance_time_fraction": 1.0 if outcome == "success" else (0.3 if feasible else 0.0),
                "shape_rms_steady": 0.005 if feasible else None,
                "xpoint_error_steady": 0.006 if feasible else None,
                "rin_error_steady": 0.004 if feasible else None,
                "rout_error_steady": 0.004 if feasible else None,
                "zx_error_steady": 0.004 if feasible else None,
                "ip_error_rel_steady": 0.01 if feasible else None,
            }

        # 1. Normal mixed 105-episode fixture WITH void episodes included in the 105 denominator:
        # 42 success, 7 void, 21 kept, 21 plasma_lost, 14 solver_failure = 105 scheduled episodes.
        normal_cases = []
        for idx, name in enumerate(case_names):
            eps = (
                [make_ep("success")] * 6
                + [make_ep("void")] * 1
                + [make_ep("kept")] * 3
                + [make_ep("plasma_lost")] * 3
                + [make_ep("solver_failure")] * 2
            )
            self.assertEqual(len(eps), 15)
            normal_cases.append({"case": name, "episodes": eps, "n_scheduled": 15, "status": "ok"})

        rew_normal, _ = hv.aggregate(normal_cases, policy_compute_ms=0.12)
        self.assertEqual(rew_normal["scheduled_episodes"], 105)
        self.assertEqual(rew_normal["successful_episodes"], 42)
        self.assertEqual(rew_normal["void_episodes"], 7)
        self.assertAlmostEqual(rew_normal["score"], 42 / 105, places=15)
        self.assertEqual(rew_normal["score"], rew_normal["reward"])

        v3_normal = self.ssi_v3.tokamak(rew_normal, expected_episodes=105)
        self.assertLess(abs(rew_normal["score"] - v3_normal["score"]), 1e-12)
        self.assertEqual(v3_normal["diagnostics"]["Void episodes"], 7)

        # Also check fallback per-case branch of ssi_index_v3.tokamak:
        rew_fallback = {k: v for k, v in rew_normal.items() if k not in ("successful_episodes", "scheduled_episodes")}
        v3_fallback = self.ssi_v3.tokamak(rew_fallback, expected_episodes=105)
        self.assertLess(abs(rew_normal["score"] - v3_fallback["score"]), 1e-12)

        # 2. Partial / policy_error fixture (case 3 crashed at step 0 with 0 completed episodes, still 105 scheduled).
        partial_cases = [dict(c) for c in normal_cases]
        partial_cases[3] = {"case": case_names[3], "episodes": [], "n_scheduled": 15, "status": "policy_error"}
        rew_partial, _ = hv.aggregate(partial_cases, policy_compute_ms=0.12)
        self.assertEqual(rew_partial["scheduled_episodes"], 105)
        self.assertEqual(rew_partial["successful_episodes"], 0)
        self.assertEqual(rew_partial["score"], 0.0)
        self.assertEqual(rew_partial["reward"], 0.0)
        self.assertLess(abs(rew_partial["score"] - self.ssi_v3.tokamak(rew_partial)["score"]), 1e-12)

        # 3. Latency-gate rejection fixture (all 105 episodes succeeded physically, but policy took 0.75 ms > 0.50 ms).
        rew_slow, _ = hv.aggregate(normal_cases, policy_compute_ms=0.75)
        self.assertEqual(rew_slow["score"], 0.0)
        self.assertEqual(rew_slow["reward"], 0.0)
        self.assertEqual(rew_slow["successful_episodes"], 0)
        self.assertLess(abs(rew_slow["score"] - self.ssi_v3.tokamak(rew_slow)["score"]), 1e-12)

        # 4. Boundary fixtures: 0/105 and 105/105.
        zero_cases = [{"case": n, "episodes": [make_ep("plasma_lost")] * 15, "n_scheduled": 15, "status": "ok"}
                      for n in case_names]
        full_cases = [{"case": n, "episodes": [make_ep("success")] * 15, "n_scheduled": 15, "status": "ok"}
                      for n in case_names]
        rew_zero, _ = hv.aggregate(zero_cases, policy_compute_ms=0.05)
        rew_full, _ = hv.aggregate(full_cases, policy_compute_ms=0.05)
        self.assertEqual(rew_zero["score"], 0.0)
        self.assertEqual(rew_full["score"], 1.0)
        self.assertLess(abs(rew_zero["score"] - self.ssi_v3.tokamak(rew_zero)["score"]), 1e-12)
        self.assertLess(abs(rew_full["score"] - self.ssi_v3.tokamak(rew_full)["score"]), 1e-12)

    def test_tests_readme_contains_required_literature_justification_sections(self):
        readme = (TESTS_DIR / "README.md").read_text()
        for heading in (
            "## Scientific & Literature Justification of Verifier Design",
            "### 1. Primary Metric & Upstream Paper Provenance",
            "### 2. Why Additional Ad-Hoc Constraints Are Intentionally Omitted",
            "### 3. Diagnostic Signals Retained in report.json",
        ):
            self.assertIn(heading, readme)
        self.assertIn("10.1038/s41586-021-04301-9", readme)
        self.assertIn("FreeGSNKE", readme)


if __name__ == "__main__":
    unittest.main()
