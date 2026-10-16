"""CPU-only interface tests. Fake dynamics below are NOT HydroGym benchmark cases."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch


TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / "environment/evaluation"))
import evaluate
import run_maia
from protocol import InfrastructureError, Policy, PolicyError, replay_episode

spec = importlib.util.spec_from_file_location("hydro_materialize", TASK / "environment/data/materialize.py")
materialize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(materialize)


def fixture_suite():
    suite = json.loads((TASK / "environment/data/suite.example.json").read_text())
    # A test-only protocol fixture, never written to production data.
    suite.update(status="validated", release_id="synthetic-interface-test-only")
    suite["coverage"]["status"] = "complete"
    suite["cases"][0].update(id="fixture-a", environment="fixture-a", horizon=3,
                              observation_size=1, action_size=1)
    suite["cases"][0]["baseline"] = {"algorithm": "PPO", "anchor_description": "Synthetic zero action", "anchor_return": -3.0, "reference_return": 0.0}
    second = copy.deepcopy(suite["cases"][0])
    second.update(id="fixture-b", environment="fixture-b")
    suite["cases"].append(second)
    return suite


class FakeEnv:
    """Toy scalar feedback problem used only to check who computes the reward."""
    class Space:
        shape = (1,)
        low = [-1.0]
        high = [1.0]
    observation_space = action_space = Space()

    def reset(self, seed=None):
        self.state = 1.0
        self.steps = 0
        return [self.state], {}

    def step(self, action):
        self.steps += 1
        self.state += action[0]
        return [self.state], -abs(self.state), False, self.steps >= 3, {"state": self.state}


class TaskTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def package(self, action="[-request['observation'][0]]", altered=""):
        package = self.root / "package"
        package.mkdir(exist_ok=True)
        (package / "README.md").write_text("Synthetic protocol fixture, not a CFD controller.\n")
        (package / "infer.sh").write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "$(dirname "$0")/policy.py"\n')
        (package / "policy.py").write_text(textwrap.dedent(f"""
            import json,sys,time
            for line in sys.stdin:
                request=json.loads(line)
                reply={{'request_id':request['request_id']}}
                if request['op']=='reset': reply['ready']=True
                elif request['op']=='act':
                    {altered or 'pass'}
                    reply['action']={action}
                else:
                    reply['closed']=True
                    print(json.dumps(reply),flush=True)
                    break
                print(json.dumps(reply),flush=True)
        """))
        return package

    def data(self, suite=None):
        suite = suite or fixture_suite()
        root = self.root / "data"
        root.mkdir()
        release = {"release_id": "synthetic-interface-test-only",
                   "cases": fixture_suite()["cases"],
                   "test": {"case_ids": ["fixture-a", "fixture-b"]}}
        # Replace only the verifier-owned fixture identity in these synthetic
        # protocol tests. Real 47-case selection/binding is tested separately.
        def fixture_release(path=None):
            return json.loads(Path(path).read_text()) if path is not None else release
        mocked = patch.object(evaluate, "load_release", side_effect=fixture_release)
        mocked.start()
        self.addCleanup(mocked.stop)
        suite.update(data_release_id=release["release_id"],
                     data_release_sha256=evaluate.data_release_hash(release))
        (root / "release.json").write_text(json.dumps(release))
        (root / "suite.json").write_text(json.dumps(suite))
        directories = []
        for case in suite["cases"]:
            folder = root / "environments" / case["environment"]
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "environment_config.yaml").write_text("test-only: true\n")
            directories.append(case["environment"])
        (root / "materialized.json").write_text(json.dumps({
            "revision": evaluate.SOURCE_REVISION, "verified_directories": directories,
            "data_release_id": release["release_id"],
            "data_release_sha256": evaluate.data_release_hash(release)}))
        return root

    def test_missing_manifest_and_binding_cannot_enable_legacy_subset(self):
        data = self.data()
        (data / "release.json").unlink()
        suite = json.loads((data / "suite.json").read_text())
        suite.pop("data_release_id")
        suite.pop("data_release_sha256")
        (data / "suite.json").write_text(json.dumps(suite))
        with self.assertRaisesRegex(InfrastructureError, "Incomplete evaluation release"):
            evaluate.preflight(data, require_reference=False)

    def test_receipt_must_match_current_data_release(self):
        data = self.data()
        receipt = json.loads((data / "materialized.json").read_text())
        receipt["data_release_id"] = "an-old-data-release"
        (data / "materialized.json").write_text(json.dumps(receipt))
        with self.assertRaisesRegex(InfrastructureError, "not been materialized"):
            evaluate.preflight(data, require_reference=False)

    def policy(self, package, timeout=2):
        policy = Policy(package, self.root / "policy.log", timeout=timeout)
        self.addCleanup(policy.abort)
        policy.reset({"environment": "unseen-id"}, [1], [1], [-1.0], [1.0])
        return policy

    def test_feedback_and_episode_reset(self):
        policy = self.policy(self.package())
        case = fixture_suite()["cases"][0]
        for seed in (83, 12):
            result = replay_episode(FakeEnv(), policy, case, seed)
            self.assertEqual(result["return"], 0.0)
            self.assertEqual(result["steps"], 3)
        policy.close()

    def test_bad_actions(self):
        for value in ("[]", "[True]", "[float('nan')]", "[2.0]", "['0']"):
            with self.subTest(value=value):
                policy = self.policy(self.package(action=value))
                with self.assertRaises(PolicyError):
                    policy.act([1.0])
                policy.abort()

    def test_wrong_id(self):
        policy = self.policy(self.package(altered="reply['request_id']='wrong'"))
        with self.assertRaises(PolicyError):
            policy.act([1.0])

    def test_timeout(self):
        policy = self.policy(self.package(altered="time.sleep(2)"), timeout=0.2)
        with self.assertRaisesRegex(PolicyError, "timeout"):
            policy.act([1.0])

    def test_invalid_observation_is_infrastructure(self):
        policy = self.policy(self.package())
        for observation in ([float("nan")], [], [1.0, 2.0]):
            with self.assertRaises(InfrastructureError):
                policy.act(observation)

    def test_default_example_cannot_be_graded(self):
        example = json.loads((TASK / "environment/data/suite.example.json").read_text())
        with self.assertRaises(InfrastructureError):
            evaluate.validate_suite(example)

    def test_legacy_cache_cannot_shadow_release(self):
        case = fixture_suite()["cases"][0]
        data = self.data()
        fake_home = self.root / "legacy-home"
        with patch.object(run_maia.Path, "home", return_value=fake_home):
            run_maia.check_native_cache(case, data)
            cached = fake_home / ".cache" / "maiagym" / case["environment"]
            cached.mkdir(parents=True)
            with self.assertRaisesRegex(InfrastructureError, "shadow"):
                run_maia.check_native_cache(case, data)

    def test_native_reward_and_horizon_errors(self):
        case = fixture_suite()["cases"][0]
        policy = self.policy(self.package())
        class BrokenReward(FakeEnv):
            def step(self, action):
                return [0.0], float("nan"), False, True, {}
        # A nonfinite reward after the policy acted is a diverged simulation: the case fails.
        with self.assertRaisesRegex(PolicyError, "diverged.*finite scalar"):
            replay_episode(BrokenReward(), policy, case, 1)
        with self.assertRaisesRegex(InfrastructureError, "horizon"):
            replay_episode(FakeEnv(), policy, dict(case, horizon=2), 1)

    def test_reject_bad_suite(self):
        transforms = [
            lambda s: s.update(cases=[]),
            lambda s: s["cases"].append(copy.deepcopy(s["cases"][0])),
            lambda s: s["cases"][0].update(backend="nek"),
            lambda s: s["cases"][0].update(id="../escape"),
            lambda s: s["cases"][0].update(horizon=0),
            lambda s: s["cases"][0].update(seeds=[1, 1]),
            lambda s: s["cases"][0]["baseline"].update(reference_return=None),
            lambda s: s["cases"][0]["baseline"].update(reference_return=-4.0),
            lambda s: s["cases"][0]["env_kwargs"].update(configuration_file="/tmp/changed"),
            lambda s: s["cases"][0]["env_kwargs"].update(obs_normalization_strategy="probewise_mean_std"),
            lambda s: s["cases"][0]["env_kwargs"].update(obs_normalization_strategy="customized", obs_loc=[0.0], obs_scale=[0.0]),
        ]
        for change in transforms:
            suite = fixture_suite()
            change(suite)
            with self.assertRaises(InfrastructureError):
                evaluate.validate_suite(suite)

    def test_incomplete_release_does_not_score_missing_submission(self):
        output = self.root / "out"
        with self.assertRaises(InfrastructureError):
            evaluate.evaluate(self.root / "missing-data", self.root / "missing-model", output)
        self.assertFalse((output / "reward.json").exists())
        self.assertTrue((output / "error.json").exists())

    def test_missing_submission_scores_zero_only_after_preflight(self):
        output = self.root / "out"
        evaluate.evaluate(self.data(), self.root / "missing-model", output)
        self.assertEqual(json.loads((output / "reward.json").read_text())["score"], 0)

    def test_relocation_and_scoring(self):
        package = self.package()
        def runner(case, suite, data, relocated, output):
            self.assertNotEqual(relocated, package)
            policy = Policy(relocated, output.with_suffix(".policy.log"), timeout=2)
            try:
                episodes = [replay_episode(FakeEnv(), policy, case, seed) for seed in case["seeds"]]
                policy.close()
            finally:
                policy.abort()
            return {"status": "ok", "episodes": episodes}
        output = self.root / "out"
        evaluate.evaluate(self.data(), package, output, runner=runner)
        self.assertEqual(json.loads((output / "reward.json").read_text()), {"score": 1.0, "case_coverage": 1.0})

    def test_policy_failures_remain_in_denominator(self):
        suite = fixture_suite()
        results = {"fixture-a": {"status": "policy_error"}, "fixture-b": {
            "status": "ok", "episodes": [{"seed": seed, "return": 0.0} for seed in suite["cases"][1]["seeds"]]}}
        self.assertEqual(evaluate.aggregate(suite, results), {"score": 0.5, "case_coverage": 0.5})
        del results["fixture-a"]
        with self.assertRaises(InfrastructureError):
            evaluate.aggregate(suite, results)

    def test_raw_replay_all_cases_without_reference_scores(self):
        suite = fixture_suite()
        for case in suite["cases"]:
            case.pop("baseline")
        with self.assertRaises(InfrastructureError):
            evaluate.validate_suite(suite)
        evaluate.validate_suite(suite, require_reference=False)
        seen = []
        package = self.package()
        package_digests = []
        def runner(case, release, data, relocated, output):
            seen.append(case["id"])
            self.assertNotEqual(relocated, package)
            package_digests.append(evaluate.package_hash(relocated))
            policy = Policy(relocated, output.with_suffix(".policy.log"), timeout=2)
            try:
                episodes = [replay_episode(FakeEnv(), policy, case, seed) for seed in case["seeds"]]
                policy.close()
            finally:
                policy.abort()
            return {"status": "ok", "episodes": episodes}
        output = self.root / "raw"
        evaluate.evaluate(self.data(suite), package, output, runner=runner, raw_only=True)
        result = json.loads((output / "results.json").read_text())
        self.assertEqual(seen, ["fixture-a", "fixture-b"])
        self.assertEqual(len(set(package_digests)), 1)
        self.assertEqual(result["suite_sha256"], evaluate.suite_hash(suite))
        self.assertEqual(result["evaluation_mode"], "native-returns")
        self.assertEqual(result["case_coverage"], 1.0)
        self.assertTrue(all(c["mean_return"] == 0.0 for c in result["cases"].values()))
        self.assertTrue(all("score" not in c for c in result["cases"].values()))
        self.assertFalse((output / "reward.json").exists())

    def test_raw_mode_does_not_bypass_physics_or_coverage(self):
        for change in (lambda s: s.update(status="draft"),
                       lambda s: s["coverage"].update(status="pending"),
                       lambda s: s["cases"][0].update(backend="jax"),
                       lambda s: s["cases"][0].update(horizon=0),
                       lambda s: s["cases"][0]["env_kwargs"].update(
                           obs_normalization_strategy="probewise_mean_std")):
            suite = fixture_suite()
            change(suite)
            with self.assertRaises(InfrastructureError):
                evaluate.validate_suite(suite, require_reference=False)
        with self.assertRaisesRegex(InfrastructureError, "coverage"):
            evaluate.aggregate(fixture_suite(), {}, normalize=False)

    def test_raw_missing_model_does_not_fabricate_a_return(self):
        output = self.root / "raw-missing"
        evaluate.evaluate(self.data(), self.root / "missing", output, raw_only=True)
        result = json.loads((output / "results.json").read_text())
        self.assertEqual(set(result["cases"]), {"fixture-a", "fixture-b"})
        self.assertTrue(all("return" not in c for c in result["cases"].values()))
        self.assertFalse((output / "reward.json").exists())

    def test_raw_policy_error_is_not_silently_omitted(self):
        suite = fixture_suite()
        results = {"fixture-a": {"status": "policy_error"}, "fixture-b": {
            "status": "ok", "episodes": [{"seed": seed, "return": -2.0}
                                            for seed in suite["cases"][1]["seeds"]]}}
        self.assertEqual(evaluate.aggregate(suite, results, normalize=False), {"case_coverage": 0.5})
        self.assertIsNone(results["fixture-a"]["mean_return"])
        self.assertEqual(results["fixture-b"]["mean_return"], -2.0)

    def test_release_mutation_during_replay_is_not_a_score(self):
        data = self.data()
        def runner(case, suite, data, relocated, output):
            changed = json.loads((data / "suite.json").read_text())
            changed["policy_timeout_sec"] += 1
            (data / "suite.json").write_text(json.dumps(changed))
            return {"status": "policy_error"}
        output = self.root / "changed-release"
        with self.assertRaisesRegex(InfrastructureError, "release changed"):
            evaluate.evaluate(data, self.package(), output, runner=runner)
        self.assertFalse((output / "reward.json").exists())

    def test_negative_gains_remain_visible(self):
        suite = fixture_suite()
        results = {c["id"]: {"status": "ok", "episodes": [{"seed": seed, "return": -6.0} for seed in c["seeds"]]}
                   for c in suite["cases"]}
        self.assertEqual(evaluate.aggregate(suite, results)["score"], 0)
        self.assertEqual(results["fixture-a"]["normalized_gain"], -1)

    def test_mutation_and_solver_error_are_not_scores(self):
        for mode in ("mutation", "solver"):
            with self.subTest(mode=mode):
                def runner(case, suite, data, package, output):
                    if mode == "solver":
                        raise InfrastructureError("solver failed")
                    (package / "README.md").write_text("mutated")
                    return {"status": "policy_error"}
                data = self.root / "data"
                if not data.exists():
                    data = self.data()
                output = self.root / mode
                with self.assertRaises(InfrastructureError):
                    evaluate.evaluate(data, self.package(), output, runner=runner)
                self.assertFalse((output / "reward.json").exists())

    def test_escaping_package_symlink(self):
        package = self.package()
        (package / "escape").symlink_to(self.root)
        with self.assertRaises(PolicyError):
            evaluate.package_hash(package)

    def test_stale_output_rejected(self):
        output = self.root / "out"
        output.mkdir()
        (output / "reward.json").write_text('{"score":123}')
        with self.assertRaises(InfrastructureError):
            evaluate.evaluate(self.data(), self.package(), output)

    def test_materialize_plan_downloads_nothing(self):
        destination = self.root / "do-not-create"
        result = subprocess.run([sys.executable, str(materialize.CATALOG.with_name("materialize.py")),
                                 str(destination), "--case", "Cylinder_2D_Re200"], capture_output=True, text=True, check=True)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["mode"], "plan")
        self.assertEqual(len(plan["download_commands"]), 1)
        self.assertFalse(destination.exists())

    def test_resource_checksums_and_truncation(self):
        path = self.root / "asset"
        path.write_bytes(b"verified")
        item = {"rfilename": "asset", "size": 8,
                "lfs": {"sha256": hashlib.sha256(b"verified").hexdigest()}}
        materialize.verify_files(self.root, [item])
        path.write_bytes(b"modified")
        with self.assertRaises(ValueError):
            materialize.verify_files(self.root, [item])
        path.write_bytes(b"v")
        with self.assertRaises(ValueError):
            materialize.verify_files(self.root, [item])

    def test_catalog_pins_and_inventory_is_not_suite(self):
        catalog = json.loads(materialize.CATALOG.read_text())
        self.assertEqual(catalog["revision"], evaluate.SOURCE_REVISION)
        self.assertEqual(len(catalog["directories"]), 89)
        self.assertEqual(sum(c["has_environment_config"] for c in catalog["directories"]), 66)
        self.assertNotIn("cases", catalog)


if __name__ == "__main__":
    unittest.main()
