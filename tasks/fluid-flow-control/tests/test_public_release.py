"""Frozen public train/test membership and materialization, never fake CFD proof."""

import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


DATA = Path(__file__).resolve().parents[1] / "environment/data"
sys.path.insert(0, str(DATA))
import public_release as public
import materialize


class PublicReleaseTests(unittest.TestCase):
    def setUp(self):
        self.release = public.load_release(DATA / "release-500gb-v1.json")
        self.index = json.loads((DATA / "source-files.json").read_text())

    def test_selection_is_reproducible_and_not_cfd_certification(self):
        actual = public.build(json.loads((DATA / "resource-audit.json").read_text()),
            self.index, json.loads((DATA / "native-configs.json").read_text()))
        self.assertEqual(actual, self.release)
        self.assertEqual(actual["case_count"], 47)
        self.assertEqual(actual["status"], "data-release-not-cfd-certified")
        self.assertEqual(actual["training"]["case_ids"], actual["test"]["case_ids"])
        self.assertEqual(actual["files"]["count"], 730)
        self.assertEqual(actual["files"]["bytes"], 488770756805)
        self.assertEqual(actual["data_budget"]["max_bytes"], 500000000000)
        self.assertIsNone(actual["training"]["fixed_transition_count"])
        self.assertEqual(sum(c["dimension"] == 2 for c in actual["cases"]), 30)
        self.assertEqual(sum(c["dimension"] == 3 for c in actual["cases"]), 17)
        self.assertTrue(all(c["execution_status"] != "validated" for c in actual["cases"]))

    def test_selected_backends_and_scientific_pairs_are_kept(self):
        self.assertEqual(self.release["backend_counts"], {
            "maia": 31, "maia-structured": 1, "firedrake": 9,
            "jax": 2, "jax-fluids": 3, "nek5000-v19": 1})
        ids = set(self.release["test"]["case_ids"])
        for re in (10000, 50000):
            for objective in ("NACA0012", "NACA0012Gust"):
                self.assertIn(f"{objective}_3D_Re{re}_AOA12", ids)
        for re in (30, 75, 100, 150):
            self.assertIn(f"Pinball_3D_Re{re}", ids)
        self.assertTrue({"Cylinder_3D_Re200", "Cylinder_3D_Re3900",
            "Sphere_3D_Re300", "Sphere_3D_Re3700", "Cube_3D_Re3700",
            "DRA2303Jet_3D_Re400000_Ma02", "Nozzle3D_coarse",
            "Channel_3D_Retau180", "TCFmini_3D_Re180"}.issubset(ids))

    def test_budget_exclusions_are_not_missing_data(self):
        excluded = [e for e in self.release["exclusions"] if e["reason"] == "not_selected_500gb_scope"]
        self.assertEqual(len(excluded), 15)
        self.assertEqual(sum(e["bytes"] for e in excluded), 1093326110913 - 488770756805)
        self.assertTrue(all(e["resource_status"] == "required_payloads_present" for e in excluded))
        self.assertTrue({"Nozzle3D_fine", "NACA0012_3D_Re200000_AOA0"}.issubset(
            {e["environment"] for e in excluded}))
        self.assertTrue(set(e["environment"] for e in excluded).isdisjoint(self.release["test"]["case_ids"]))

    def test_missing_selected_asset_cannot_silently_shrink_release(self):
        index = copy.deepcopy(self.index)
        index["files"] = [f for f in index["files"] if f["rfilename"] !=
                          "DRA2303Jet_3D_Re400000_Ma02/restart_.hdf5"]
        with self.assertRaisesRegex(ValueError, "Selected cases lack complete resources"):
            public.build(json.loads((DATA / "resource-audit.json").read_text()),
                index, json.loads((DATA / "native-configs.json").read_text()))

    def test_oversize_source_assets_are_rejected(self):
        index = copy.deepcopy(self.index)
        item = next(f for f in index["files"] if f["rfilename"].startswith("Cube_3D_Re3700/"))
        item["size"] += 500000000000
        with self.assertRaisesRegex(ValueError, "exceed.*500 GB"):
            public.build(json.loads((DATA / "resource-audit.json").read_text()),
                index, json.loads((DATA / "native-configs.json").read_text()))

    def test_real_exclusions_are_explicit_not_score_based(self):
        excluded = self.release["exclusions"]
        self.assertEqual(sum(e["reason"] == "missing_payload" for e in excluded), 9)
        self.assertEqual(sum(e["reason"] == "parameter_mismatch" for e in excluded), 1)
        self.assertTrue(any(e.get("environment") == "SquareCylinder_2D_Re3900"
                            and e["missing"] == ["restart_.Netcdf"] for e in excluded))
        self.assertTrue(all("score" not in e["reason"] for e in excluded))

    def test_partition_variants_and_analytic_initialization_are_not_missing(self):
        configs = {c["name"]: c["config"] for c in json.loads((DATA / "native-configs.json").read_text())["configs"]}
        for name in ["Kolmogorov_2D", "Nozzle3D_fine", "NACA0012_3D_Re200000_AOA0"]:
            files = [f for f in self.index["files"] if f["rfilename"].startswith(name + "/")]
            self.assertEqual(public.resource_inputs(name, files, configs[name])[1], [])
        files = [f for f in self.index["files"] if f["rfilename"].startswith("Nozzle3D_fine/")
                 and not f["rfilename"].endswith("restart_proc2.h5")]
        missing = public.resource_inputs("Nozzle3D_fine", files, configs["Nozzle3D_fine"])[1]
        self.assertIn("restart_proc2.h5", missing)

    def test_no_unvalidated_physics_reconstruction(self):
        reconstructed = [c for c in self.release["cases"] if c["configuration_identity_reconstructed"]]
        self.assertEqual(len(reconstructed), 9)
        self.assertTrue(all(c["backend"] == "firedrake" for c in reconstructed))
        self.assertNotIn("Cavity_2D_Re4200", self.release["test"]["case_ids"])
        self.assertNotIn("Cavity_2D_Re4140", self.release["test"]["case_ids"])

    def test_exact_suite_binding_not_a_subset(self):
        suite = {"data_release_id": self.release["release_id"],
            "data_release_sha256": public.digest(self.release),
            "cases": [{"id": c["id"], "environment": c["environment"], "backend": c["backend"]}
                      for c in self.release["cases"]]}
        public.validate_suite_binding(self.release, suite)
        bad = copy.deepcopy(suite)
        bad["cases"].pop()
        with self.assertRaisesRegex(ValueError, "exact frozen"):
            public.validate_suite_binding(self.release, bad)
        bad = copy.deepcopy(suite)
        bad["cases"][0]["environment"] = "another-Re"
        with self.assertRaisesRegex(ValueError, "replaced"):
            public.validate_suite_binding(self.release, bad)
        bad = copy.deepcopy(suite)
        bad["cases"][0]["backend"] = "fake"
        with self.assertRaisesRegex(ValueError, "backend"):
            public.validate_suite_binding(self.release, bad)
        bad = copy.deepcopy(suite)
        bad["data_release_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "not bound"):
            public.validate_suite_binding(self.release, bad)

    def test_no_source_path_or_hash_ambiguity(self):
        for field, value in [("rfilename", "../outside"), ("blobId", "invalid"), ("size", -1)]:
            bad = copy.deepcopy(self.index)
            bad["files"][0] = {"rfilename": "file", "size": 0, "blobId": "a" * 40, field: value}
            with self.assertRaises(ValueError):
                public.validate_index(bad)
        bad = copy.deepcopy(self.index)
        bad["files"].append(bad["files"][0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            public.validate_index(bad)

    def test_train_test_drift_is_rejected(self):
        release = copy.deepcopy(self.release)
        release["test"]["case_ids"] = release["test"]["case_ids"][:-1]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "release.json"
            path.write_text(json.dumps(release))
            with self.assertRaisesRegex(ValueError, "drift"):
                public.load_release(path)

    def test_mutated_membership_budget_and_totals_are_rejected(self):
        for mutation in ("membership", "budget", "totals", "old-version"):
            release = copy.deepcopy(self.release)
            if mutation == "membership":
                release["cases"].pop()
                release["case_count"] -= 1
                release["training"]["case_ids"] = [c["id"] for c in release["cases"]]
                release["test"]["case_ids"] = list(release["training"]["case_ids"])
            elif mutation == "budget":
                release["data_budget"]["max_bytes"] *= 2
            elif mutation == "totals":
                release["files"]["bytes"] += 1
            else:
                release["release_id"] = "hydrogym-public-resources-v1"
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "release.json"
                path.write_text(json.dumps(release))
                with self.assertRaises(ValueError):
                    public.load_release(path)

    def test_default_plan_is_only_the_frozen_resource_set(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "not-created"
            result = subprocess.run([sys.executable, str(DATA / "materialize.py"), str(path),
                "--release", str(DATA / "release-500gb-v1.json")],
                check=True, capture_output=True, text=True)
            plan = json.loads(result.stdout)
            self.assertEqual(plan["directories"], self.release["training"]["case_ids"])
            self.assertEqual(plan["bytes"], 488770756805)
            self.assertTrue(plan["complete_release_selection"])
            self.assertFalse(path.exists())
            diagnostic = subprocess.run([sys.executable, str(DATA / "materialize.py"), str(path),
                "--release", str(DATA / "release-500gb-v1.json"),
                "--case", "Cylinder_2D_Re200"], check=True, capture_output=True, text=True)
            self.assertFalse(json.loads(diagnostic.stdout)["complete_release_selection"])

    def materialization_fixture(self, root):
        source = root / "archive"
        source.mkdir()
        payload = b"synthetic source bytes, not CFD"
        file = source / "environments/fixture/input"
        file.parent.mkdir(parents=True)
        file.write_bytes(payload)
        (source / "environments/excluded").mkdir()
        (source / "environments/excluded/private-research").write_text("not part of the release")
        index = {"repo_id": "dynamicslab/HydroGym-environments", "revision": public.REVISION,
            "files": [{"rfilename": "fixture/input", "size": len(payload),
                       "blobId": hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()}]}
        catalog = {"repo_id": index["repo_id"], "revision": public.REVISION,
            "directories": [{"name": "fixture", "bytes": len(payload)}]}
        config = root / "config"
        config.mkdir()
        (config / "catalog.json").write_text(json.dumps(catalog))
        (config / "source-files.json").write_text(json.dumps(index))
        release = {"release_id": public.RELEASE_ID, "source_index_sha256": public.digest(index),
            "training": {"case_ids": ["fixture"]}}
        return source, config / "catalog.json", release

    def test_local_reuse_is_selected_portable_and_hash_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, catalog, release = self.materialization_fixture(root)
            target = root / "release"
            argv = ["materialize", str(target), "--from-local", str(source)]
            with patch.object(materialize, "CATALOG", catalog), patch.object(public, "load_release", return_value=release), \
                 patch.object(sys, "argv", argv), patch.object(materialize.subprocess, "run", side_effect=AssertionError("No download")), \
                 redirect_stdout(io.StringIO()):
                materialize.main()
            file = target / "environments/fixture/input"
            self.assertFalse(file.is_symlink())
            self.assertTrue(file.samefile(source / "environments/fixture/input"))
            self.assertFalse((target / "environments/excluded").exists())
            receipt = json.loads((target / "materialized.json").read_text())
            self.assertEqual(receipt["data_release_sha256"], public.digest(release))
            self.assertFalse(receipt["benchmark_validated"])
            self.assertFalse((target / "suite.json").exists())

    def test_corrupt_local_input_cannot_publish_a_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, catalog, release = self.materialization_fixture(root)
            file = source / "environments/fixture/input"
            file.write_bytes(b"x" * file.stat().st_size)
            target = root / "release"
            with patch.object(materialize, "CATALOG", catalog), patch.object(public, "load_release", return_value=release), \
                 patch.object(sys, "argv", ["materialize", str(target), "--from-local", str(source)]), \
                 redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "checksum"):
                materialize.main()
            self.assertFalse((target / "release.json").exists())
            self.assertFalse((target / "materialized.json").exists())

    def test_old_release_root_cannot_be_relabelled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, catalog, release = self.materialization_fixture(root)
            prior = json.dumps({"release_id": "hydrogym-public-resources-v1"})
            (source / "release.json").write_text(prior)
            old_receipt = json.dumps({"data_release_id": "hydrogym-public-resources-v1"})
            (source / "materialized.json").write_text(old_receipt)
            with patch.object(materialize, "CATALOG", catalog), patch.object(public, "load_release", return_value=release), \
                 patch.object(sys, "argv", ["materialize", str(source), "--verify-only"]), \
                 redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "another release"):
                materialize.main()
            self.assertEqual((source / "release.json").read_text(), prior)
            self.assertEqual((source / "materialized.json").read_text(), old_receipt)

    def test_verification_preserves_unrelated_verified_archive_entries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, catalog, release = self.materialization_fixture(root)
            (source / "materialized.json").write_text(json.dumps({"revision": public.REVISION,
                "verified_directories": ["prior-entry"]}))
            with patch.object(materialize, "CATALOG", catalog), patch.object(public, "load_release", return_value=release), \
                 patch.object(sys, "argv", ["materialize", str(source), "--verify-only"]), redirect_stdout(io.StringIO()):
                materialize.main()
            receipt = json.loads((source / "materialized.json").read_text())
            self.assertEqual(receipt["verified_directories"], ["fixture", "prior-entry"])


if __name__ == "__main__":
    unittest.main()
