"""SSI V4 regression tests for fluid-flow-control (#05):
1. Dual-rank single-GPU sharing in `maia_gpu_rank.py` + OpenMPI sandbox env vars;
2. Blind test seed roster (`blind_flow_seeds.json`), multi-seed per-case aggregation & failure (-1.0);
3. Secret simulator seed masking in `policy.reset(...)` (public `episode_index` only);
4. Exact `< 1e-12` parity with `harness/ssi_index_v3.flow()` across normal, diverged, and negative fixtures;
5. `check_dev_mpi_all_cases.py` smoke (26 cases = 4 HydroGym + 22 MAIA) & full-dual-rank (3 cases) modes;
6. Visibility boundary (`tests/` never copied into `development` image).
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

TASK = Path(__file__).resolve().parents[1]
REPO = TASK.parent
sys.path.insert(0, str(TASK / "environment" / "evaluation"))
sys.path.insert(0, str(TASK / "tests"))

from aggregate import anchors, task_score  # noqa: E402
from case_spec import BLIND_SEEDS_ENV, VERIFIER_CASES, select_cases  # noqa: E402
import check_dev_mpi_all_cases as mpi_check  # noqa: E402
from harbor_verify import verify  # noqa: E402
import metrics as physical_metrics  # noqa: E402
from protocol import replay_episode  # noqa: E402


def load_ssi_index_v3():
    v3_path = next(p for p in Path(__file__).resolve().parents if (p / "harness" / "ssi_index_v3.py").is_file()) / "harness" / "ssi_index_v3.py"
    spec = importlib.util.spec_from_file_location("ssi_index_v3", v3_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class V4FlowUpgradesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v3 = load_ssi_index_v3()

    def test_seed_masking_hides_secret_seed_and_passes_episode_index(self):
        env_seeds = []
        policy_contexts = []

        class ToySpace:
            shape = (2,)
            low = [-1.0, -1.0]
            high = [1.0, 1.0]

        class ToyEnv:
            observation_space = ToySpace()
            action_space = ToySpace()

            def reset(self, seed=None):
                env_seeds.append(seed)
                self.t = 0
                return [0.1, -0.1], {}

            def step(self, action):
                self.t += 1
                return [0.0, 0.0], -0.5, self.t >= 2, False, {}

        class SpyPolicy:
            def reset(self, context, obs_shape, act_shape, low, high):
                policy_contexts.append(dict(context))

            def act(self, obs):
                return [0.0, 0.0]

        case = {"environment": "Cylinder_2D_Re200", "family": "cylinder", "horizon": 2}
        secret_seeds = [849201, 193847, 772615]
        episodes = [
            replay_episode(ToyEnv(), SpyPolicy(), case, s, episode_index=idx)
            for idx, s in enumerate(secret_seeds)
        ]
        self.assertEqual(env_seeds, secret_seeds)
        self.assertEqual([c["seed"] for c in policy_contexts], [0, 1, 2])
        self.assertEqual([c["episode_index"] for c in policy_contexts], [0, 1, 2])
        for ctx in policy_contexts:
            self.assertNotIn(ctx["seed"], secret_seeds)
        self.assertEqual([e["seed"] for e in episodes], secret_seeds)
        self.assertEqual([e["episode_index"] for e in episodes], [0, 1, 2])

    def test_sealed_blind_seed_roster_and_dev_default_decoupling(self):
        blind_path = TASK / "tests" / "sealed" / "blind_flow_seeds.json"
        if not blind_path.is_file():
            self.skipTest("sealed files are distributed separately (tests/sealed/README.md)")
        payload = json.loads(blind_path.read_text())
        self.assertEqual(payload["schema"], "hydrogym-blind-seeds/v1")
        self.assertEqual(payload["seeds_per_case"], 3)

        # Default dev mode keeps seeds=[123]
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(BLIND_SEEDS_ENV, None)
            _, dev_cases = select_cases(["verifier"])
            self.assertTrue(all(c["seeds"] == [123] for c in dev_cases))

        # Sealed blind mode loads 3 private perturbation seeds != 123 per case
        with patch.dict(os.environ, {BLIND_SEEDS_ENV: str(blind_path)}):
            _, blind_cases = select_cases(["verifier"])
            for c in blind_cases:
                self.assertEqual(len(c["seeds"]), 3)
                self.assertEqual(len(set(c["seeds"])), 3)
                self.assertNotIn(123, c["seeds"])

    def test_multi_seed_aggregation_and_invalid_regime_propagation(self):
        defn = physical_metrics.definition("NACA0012_2D_Re100_AOA20")
        ok_eps = [
            {"primary_metric": dict(defn, status="ok", value=1.2)},
            {"primary_metric": dict(defn, status="ok", value=1.5)},
            {"primary_metric": dict(defn, status="ok", value=1.8)},
        ]
        agg = physical_metrics.aggregate(defn, ok_eps)
        self.assertEqual(agg["status"], "ok")
        self.assertAlmostEqual(agg["value"], 1.5, places=12)
        self.assertEqual(agg["episodes"], 3)

        bad_eps = [
            ok_eps[0],
            {"primary_metric": dict(defn, status="invalid_operating_regime", value=None)},
            ok_eps[2],
        ]
        agg_bad = physical_metrics.aggregate(defn, bad_eps)
        self.assertEqual(agg_bad["status"], "invalid_operating_regime")
        self.assertIsNone(agg_bad["value"])

    def test_v3_10_family_score_parity_under_1e12(self):
        anc = anchors()
        _, ver_cases = select_cases(["verifier"], blind_seeds_path="")
        indexed = {c["id"]: c for c in ver_cases}

        # 1. Normal fixture with heterogeneous gains across 6-case vs 1-case families
        # Give +80% gain on Channel_3D_Retau180 (1-case family) and -10% on NACA0012Gust (6-case family)
        def make_eval_result(mode):
            out = {}
            for cid in VERIFIER_CASES:
                c = indexed[cid]
                z = anc[cid]["zero"]
                direction = c["primary_metric"]["direction"]
                if mode == "normal":
                    fam = cid.split("_")[0]
                    target_gain = {
                        "Channel": 0.80,
                        "Cube": 0.45,
                        "Cylinder": 0.20,
                        "Kolmogorov": 0.65,
                        "NACA0012": 0.30,
                        "NACA0012Gust": -0.10,
                        "Nozzle2D": 0.25,
                        "Pinball": 0.15,
                        "Sphere": 0.50,
                        "SquareCylinder": 0.35,
                    }[fam]
                    val = z * (1.0 - target_gain) if direction == "lower_is_better" else z * (1.0 + target_gain)
                    out[cid] = {"status": "ok", "primary_metric": dict(c["primary_metric"], status="ok", value=val)}
                elif mode == "diverged_partial":
                    if cid in ("Cube_3D_Re3700", "Pinball_2D_Re100", "NACA0012_2D_Re1000_AOA40"):
                        out[cid] = {"status": "policy_error", "error": "diverged"}
                    else:
                        val = z * 0.7 if direction == "lower_is_better" else z * 1.3
                        out[cid] = {"status": "ok", "primary_metric": dict(c["primary_metric"], status="ok", value=val)}
                elif mode == "negative_signed":
                    val = z * 1.4 if direction == "lower_is_better" else z * 0.6
                    out[cid] = {"status": "ok", "primary_metric": dict(c["primary_metric"], status="ok", value=val)}
            return {"cases": out}

        for mode in ("normal", "diverged_partial", "negative_signed"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                with patch.dict(os.environ, {}, clear=False):
                    os.environ.pop(BLIND_SEEDS_ENV, None)
                    rew = verify("/unused", "/unused-sub", tmp, evaluator=lambda *a, m=mode: make_eval_result(m))
                    grading = json.loads((Path(tmp) / "grading.json").read_text())
                    failed = set(grading["failed_cases"])
                    v3_res = self.v3.flow(rew, failed)

                    self.assertLess(abs(rew["score"] - v3_res["score"]), 1e-12)
                    self.assertLess(abs(rew["reward"] - v3_res["score"]), 1e-12)
                    self.assertLess(abs(grading["signed_score"] - v3_res["signed_score"]), 1e-12)
                    for fam, fam_val in v3_res["components"].items():
                        self.assertLess(
                            abs(grading["signed_improvement"]["by_family"][fam] - fam_val), 1e-12
                        )
                    if mode == "normal":
                        # Verify 10-family mean differs from flat 26-case mean on unbalanced family sizes
                        flat = grading["signed_improvement"]["flat_case_mean_improvement"]
                        self.assertGreater(abs(rew["score"] - flat), 0.05)
                    if mode == "negative_signed":
                        self.assertLess(grading["signed_score"], 0.0)
                        self.assertEqual(rew["score"], 0.0)

    def test_mpi_check_script_modes_and_summary(self):
        smoke_cases = mpi_check.cases_for_mode("smoke")
        self.assertEqual(len(smoke_cases), 26)
        self.assertEqual(sum(1 for c in smoke_cases if c["backend"] != "maia"), 4)
        self.assertEqual(sum(1 for c in smoke_cases if c["backend"] == "maia"), 22)

        dual_cases = mpi_check.cases_for_mode("full-dual-rank")
        self.assertEqual([c["id"] for c in dual_cases], list(mpi_check.DUAL_RANK_CASES))
        self.assertTrue(all(c["solver_processes"] == 2 and c["backend"] == "maia" for c in dual_cases))

        with tempfile.TemporaryDirectory() as tmp:
            summary = mpi_check.run_suite(
                "full-dual-rank",
                Path(tmp),
                Path(tmp) / "out",
                verify_data=False,
                runner=lambda case, d, out, m, s: {
                    "status": "ok",
                    "case": case["id"],
                    "wall_sec": 1.0,
                    "gpu_memory": {"max_peak_used_mib": 12345.0},
                    "zero_control_stability": {"finite_stable": True, "rel_diff": 0.0},
                },
            )
            self.assertEqual(summary["status"], "ok")
            self.assertEqual(summary["passed_cases"], 3)
            self.assertTrue((Path(tmp) / "out" / "full_dual_rank_summary.json").is_file())

    def test_openmpi_container_env_and_visibility_boundary(self):
        dockerfile = (TASK / "environment" / "Dockerfile").read_text()
        dev_compose = (TASK / "environment" / "docker-compose.yaml").read_text()
        self.assertIn("OMPI_MCA_btl=self,vader,tcp", dockerfile)
        self.assertIn("OMPI_MCA_btl_vader_single_copy_mechanism=none", dockerfile)
        self.assertIn("OMPI_MCA_btl", dev_compose)

        # Ensure development target never copies tests/ into /workspace
        dev_section = dockerfile.split("FROM runtime AS development", 1)[1]
        self.assertNotIn("tests/", dev_section)
        self.assertNotIn("/tests", dev_compose)


if __name__ == "__main__":
    unittest.main()
