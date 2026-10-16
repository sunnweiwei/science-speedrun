"""Research v2 metadata/integration tests, not CFD or trained-policy scores."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

DATA = Path(__file__).resolve().parents[1] / "environment/data"
sys.path.insert(0, str(DATA))
import hour_release
import public_release as public
import research_release as research


class ResearchReleaseTests(unittest.TestCase):
    def setUp(self):
        self.selection = research.read("selection-research-v2.json")
        self.evidence = research.read("cost-evidence-coverage-v1.json")
        self.prior = hour_release.build_current()

    def build(self, selection=None, evidence=None):
        return research.build(selection or self.selection, evidence or self.evidence, self.prior)

    def test_current_default_matches_frozen_build_and_full_membership(self):
        release = public.load_release()
        self.assertEqual(release, research.build_current())
        self.assertEqual(release["release_id"], research.RELEASE_ID)
        self.assertEqual(release["case_count"], 39)
        self.assertEqual(sum(c["dimension"] == 3 for c in release["cases"]), 19)
        self.assertEqual(release["files"], {"count": 558, "bytes": 453756315982})
        self.assertEqual(release["backend_counts"], {"jax": 2, "jax-fluids": 2, "maia": 35})
        self.assertEqual(release["training"]["case_ids"], release["test"]["case_ids"])
        self.assertTrue(release["test"]["one_shared_checkpoint"])
        self.assertTrue(release["test"]["all_selected_cases_required"])
        self.assertFalse(release["test"]["cost_tiers_are_optional"])

    def test_cost_status_is_not_completed_training_or_evaluation(self):
        release = self.build()
        self.assertEqual(release["status"], "data-release-not-cfd-certified")
        self.assertEqual(release["test"]["execution_manifest_status"],
                         "requires_native_protocol_binding_and_full_replay")
        self.assertEqual(sum("training_evidence_reference" in c for c in release["cases"]), 19)
        self.assertEqual(sum(c["required_solver_gpus"] == 2 for c in release["cases"]), 3)
        self.assertFalse(release["reference_training_budget"]["agent_training_limit"])
        self.assertEqual(release["reference_training_budget"]["total_controls"], 156000)

    def test_all_62_complete_resources_have_unique_decisions(self):
        release = self.build()
        selected = {c["id"] for c in release["cases"]}
        excluded = {c["environment"] for c in release["exclusions"]
                    if c.get("resource_status") == "required_payloads_present"}
        self.assertFalse(selected & excluded)
        self.assertEqual(len(selected | excluded), 62)
        cavity = next(c for c in release["exclusions"] if c.get("environment") == "Cavity_2D_Re7500")
        self.assertEqual(cavity["reason"], "nonfinite_native_episode")

    def test_rebuilt_old_hour_manifest_and_other_archives_remain_exact(self):
        self.assertEqual(public.load_release(DATA / "release-hour-v1.json"), self.prior)
        for name, expected in [
            ("hour", "b5c398208b0bd4bcbd8ae5fe67daff3066a0d83cf1f090a711a3eec2ada1c11f"),
            ("coverage", "365e9531671ae4eb24491e880f7e6c1d9768013b5da2bb389df9342525ca1896"),
            ("speed", "23c53782f87b91c9be8babee3cab5e7cd16730c9dc79eb46d1b19c796be82890"),
            ("500gb", "226b7059b4b6b34bb4a8f169374f2a6bc74bc4f780b22ccbc3171c06b34778f3")]:
            self.assertEqual(public.digest(public.load_release(DATA / f"release-{name}-v1.json")), expected)

    def test_unapproved_or_changed_budget_rejected(self):
        for mutate in [lambda p: p.update(status="draft"),
                       lambda p: p["policy"].update(max_source_bytes=501000000000)]:
            selection = copy.deepcopy(self.selection)
            mutate(selection)
            with self.assertRaises(ValueError):
                self.build(selection)

    def test_cannot_drop_change_or_duplicate_a_case(self):
        for mutate in [lambda p: p["cases"].pop(),
                       lambda p: p["cases"].append(p["cases"][0]),
                       lambda p: p["contract"]["evaluation_case_ids"].pop(),
                       lambda p: p["contract"].update(all_selected_cases_required=False)]:
            selection = copy.deepcopy(self.selection)
            mutate(selection)
            with self.assertRaises(ValueError):
                self.build(selection)

    def test_unknown_cost_and_one_step_probe_cannot_qualify(self):
        for field, value in [("status", "timeout"), ("reset_seconds", None),
                             ("timing_samples_seconds", [1.0]), ("observed_field_response", False)]:
            evidence = copy.deepcopy(self.evidence)
            next(r for r in evidence["cases"] if r["case"] == "Cube_3D_Re300")[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.build(evidence=evidence)

    def test_two_gpu_results_cannot_be_called_single_gpu(self):
        selection = copy.deepcopy(self.selection)
        next(c for c in selection["cases"] if c["case"] == "Cube_3D_Re3700")["hardware"] = "1 H100"
        with self.assertRaisesRegex(ValueError, "hardware"):
            self.build(selection)
        evidence = copy.deepcopy(self.evidence)
        record = next(r for r in evidence["cases"] if r["case"] == "Cube_3D_Re3700")
        record["gpu_bindings"][1]["cuda_device_uuid"] = record["gpu_bindings"][0]["cuda_device_uuid"]
        with self.assertRaisesRegex(ValueError, "rank bindings"):
            self.build(evidence=evidence)

    def test_false_training_claim_cost_tampering_or_missing_exclusion_rejected(self):
        for mutate in [lambda p: p["cases"][0].update(full_4000_step_training_validated=True),
                       lambda p: p["cases"][0].update(whole_experiment_planning_hours=0.0),
                       lambda p: p["cases"][0].update(tier="multigpu_exception"),
                       lambda p: p["excluded_complete_resources"].pop()]:
            selection = copy.deepcopy(self.selection)
            mutate(selection)
            with self.assertRaises(ValueError):
                self.build(selection)

    def test_channel_repair_preserved_without_changing_native_source(self):
        release = self.build()
        self.assertEqual(release["source"]["organizer_runtime_adapters"],
                         self.prior["source"]["organizer_runtime_adapters"])
        old = {c["id"]: c for c in self.prior["cases"]}
        for case in release["cases"]:
            if case["id"] in old:
                self.assertEqual(case["files_sha256"], old[case["id"]]["files_sha256"])
                self.assertEqual(case["runtime_profile"], old[case["id"]]["runtime_profile"])

    def test_default_materialization_plan_is_pinned_and_nonmutating(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "data"
            result = subprocess.run([sys.executable, str(DATA / "materialize.py"), str(destination)],
                                    capture_output=True, text=True, check=True)
            plan = json.loads(result.stdout)
            self.assertEqual(plan["data_release_id"], research.RELEASE_ID)
            self.assertEqual(plan["bytes"], 453756315982)
            self.assertEqual(len(plan["directories"]), 39)
            self.assertFalse(destination.exists())
            for command in plan["download_commands"]:
                self.assertEqual(command[0], "hf")
                self.assertEqual(command[command.index("--revision") + 1], plan["download_source"]["revision"])
                self.assertEqual(command[2], plan["download_source"]["repo_id"])

    def test_partial_suite_and_tampered_release_are_rejected(self):
        release = self.build()
        suite = {"data_release_id": research.RELEASE_ID, "data_release_sha256": public.digest(release),
                 "cases": [{"id": c["id"], "environment": c["environment"], "backend": c["backend"]}
                           for c in release["cases"]]}
        public.validate_suite_binding(release, suite)
        suite["cases"].pop()
        with self.assertRaisesRegex(ValueError, "exact frozen"):
            public.validate_suite_binding(release, suite)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "release.json"
            release["test"]["normalizer_updates"] = True
            path.write_text(json.dumps(release))
            with self.assertRaisesRegex(ValueError, "exact frozen"):
                public.load_release(path)


if __name__ == "__main__":
    unittest.main()
