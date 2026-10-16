"""Evidence catalog integrity, not CFD execution or paper-performance proof."""

import hashlib
import json
from pathlib import Path
import unittest


TASK = Path(__file__).resolve().parents[1]


class PaperEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = json.loads((TASK / "environment/references/specialists/paper-targets.json").read_text())
        # These immutable paper targets describe the original 47-case release,
        # not a regenerated set of claimed scores for the new speed selection.
        cls.release = json.loads((TASK / "environment/data/release-500gb-v1.json").read_text())
        cls.cases = {case["id"]: case for case in cls.audit["cases"]}

    def test_exact_frozen_coverage_and_source_identity(self):
        canonical = json.dumps(self.release, sort_keys=True, separators=(",", ":"),
                               allow_nan=False).encode()
        self.assertEqual(self.audit["data_release_sha256"], hashlib.sha256(canonical).hexdigest())
        self.assertEqual(self.audit["data_release_id"], self.release["release_id"])
        self.assertEqual(self.audit["sources"]["data"]["revision"], self.release["source"]["revision"])
        self.assertEqual(self.audit["sources"]["platform"]["revision"], self.release["source"]["code_revision"])
        self.assertEqual(len(self.cases), 47)
        self.assertEqual(len(self.audit["cases"]), 47)
        self.assertEqual(list(self.cases), self.release["test"]["case_ids"])
        self.assertEqual(self.audit["benchmark_primary_metric"], self.release["test"]["primary_metric"])

    def test_every_row_has_resolved_references_and_unchanged_public_values(self):
        for source in self.release["cases"]:
            with self.subTest(case=source["id"]):
                row = self.cases[source["id"]]
                self.assertIn(row["metric_profile"], self.audit["metric_profiles"])
                for ref in row["curve_refs"]:
                    self.assertIn(ref, self.audit["curve_evidence"])
                for ref in row["issue_refs"] + self.audit["global_unresolved"]:
                    self.assertIn(ref, self.audit["issues"])
                config = source["native_configuration"]
                native = config.get("maia", config.get("jax", config.get("jaxfluids", {})))
                self.assertEqual(row["public_config_values"], {
                    "episode_limit": config.get("env", {}).get("max_episode_steps",
                        config.get("runner", {}).get("nb_interactions")),
                    "cfd_substeps": native.get("num_sim_substeps_per_actuation", native.get("steps_per_action")),
                    "action_inputs": native.get("num_action_inputs", native.get("num_actuators")),
                    "omega": native.get("omega"),
                })

    def test_curve_estimates_do_not_become_acceptance_scores(self):
        self.assertEqual(self.audit["status"], "evidence-audit-not-executable-protocol")
        counts = {}
        for row in self.cases.values():
            status = row["learned_result_status"]
            counts[status] = counts.get(status, 0) + 1
            self.assertEqual(row["paper_acceptance"], {
                "metric": None, "target": None, "tolerance": None, "ready": False})
        self.assertEqual(counts, {
            "case_plot_located_setup_unresolved": 5,
            "family_plot_located_case_unresolved": 1,
            "not_located_for_exact_variant": 41,
        })
        for curve in self.audit["curve_evidence"].values():
            self.assertEqual(curve["readout_kind"], "coarse_visual_estimate_not_acceptance_target")
            self.assertIsNone(curve["test_scalar_target"])
            for bounds in curve["approximate_training_endpoint_ranges"].values():
                self.assertEqual(len(bounds), 2)
                self.assertLess(bounds[0], bounds[1])
        self.assertEqual(self.audit["curve_evidence"]["fig3b"]["training_seeds_reported"], 1)
        self.assertEqual(self.audit["curve_evidence"]["ed2a"]["training_seeds_reported"], 3)

    def test_no_transfer_across_case_solver_or_actuation_identity(self):
        self.assertEqual(self.cases["SquareCylinder_2D_Re200"]["curve_refs"], ["ed3c"])
        for source in self.release["cases"]:
            row = self.cases[source["id"]]
            if (source["backend"] == "firedrake" or source["id"].startswith("NACA0012")
                    or source["id"] in {"Cylinder_3D_Re200", "Cylinder_3D_Re3900",
                                       "Cavity_2D_Re7500", "TCFmini_3D_Re180"}):
                self.assertEqual(row["curve_refs"], [], source["id"])
            if source["backend"] == "firedrake":
                self.assertIsNone(row["paper_training_defaults"]["cfd_steps_per_control_action"])
                self.assertIsNone(row["physics_validation"])

    def test_physics_validation_is_separate_and_conflicts_remain_visible(self):
        checks = [c["physics_validation"] for c in self.cases.values() if c["physics_validation"]]
        self.assertEqual(len(checks), 15)
        for check in checks:
            self.assertEqual(check["kind"], "uncontrolled_flow_validation_not_learned_policy")
            self.assertIsNone(check["averaging_window"])
            self.assertIsNone(check["acceptance_tolerance"])
        pinball = self.cases["Pinball_3D_Re150"]
        self.assertEqual(pinball["physics_validation"]["values"]["mean_total_CD"], 2.949)
        self.assertEqual(pinball["public_config_values"]["omega"], 0)
        self.assertEqual(self.audit["metric_profiles"]["pinball"]["paper_parameters"]["omega"], 1)
        self.assertIn("pinball_omega", pinball["issue_refs"])
        self.assertIn("cavity_reward", self.cases["Cavity_2D_Re7500"]["issue_refs"])
        self.assertIn("kolmogorov_objective", self.cases["Kolmogorov_2D"]["issue_refs"])


if __name__ == "__main__":
    unittest.main()
