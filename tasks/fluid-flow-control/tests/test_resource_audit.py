"""Resource coverage checks, not native CFD or scientific acceptance tests."""

import copy
import importlib.util
import json
from pathlib import Path
import unittest


DATA = Path(__file__).resolve().parents[1] / "environment/data"
spec = importlib.util.spec_from_file_location("resource_audit", DATA / "resource_audit.py")
resource_audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resource_audit)


class ResourceAuditTests(unittest.TestCase):
    def setUp(self):
        self.coverage = json.loads((DATA / "paper-coverage.json").read_text())
        self.catalog = json.loads((DATA / "catalog.json").read_text())
        self.report = resource_audit.audit(self.coverage, self.catalog)

    def test_complete_table_accounting_does_not_exclude_cases(self):
        self.assertEqual(self.report["paper_rows"], 55)
        self.assertEqual(sum(self.report["resource_counts"].values()), 55)
        self.assertEqual(self.report["scope_exclusions"], [])
        self.assertEqual(self.report["resource_counts"]["missing_payload"], 9)
        self.assertAlmostEqual(self.report["missing_payload_fraction"], 9 / 55)
        self.assertNotIn("cases", self.report)

    def test_checked_in_report_matches_reproducible_audit(self):
        self.assertEqual(json.loads((DATA / "resource-audit.json").read_text()), self.report)

    def test_4140_is_not_silently_4200_or_a_missing_family(self):
        row = next(r for r in self.report["rows"] if r["family"] == "cavity"
                   and r["dimension"] == 2 and r["parameters"]["Re"] == 4200)
        self.assertEqual(row["resource_status"], "parameter_mismatch")
        self.assertEqual(row["candidate_payloads"], [])
        self.assertIn("Cavity_2D_Re4140", row["related_not_equivalent"])

    def test_dra_marker_and_mach_are_not_usable_payloads(self):
        rows = {r["parameters"]["Mach"]: r for r in self.report["rows"]
                if r["family"] == "dra2303"}
        self.assertEqual(rows[0.2]["resource_status"], "partial_variant_payload")
        self.assertEqual(rows[0.7]["resource_status"], "missing_payload")
        self.assertEqual(len(rows[0.7]["marker_only"]), 2)
        self.assertEqual(rows[0.7]["candidate_payloads"], [])

    def test_firedrake_and_marl_are_not_discarded_for_adapter_gaps(self):
        row = next(r for r in self.report["rows"] if r["family"] == "cylinder"
                   and r["dimension"] == 2 and r["parameters"]["Re"] == 100)
        self.assertEqual(len(row["candidate_payloads"]), 2)
        self.assertEqual(row["resource_status"], "candidate_payload_unvalidated")
        marl = [r for r in self.report["rows"] if r["family"] == "cylinder-marl"]
        self.assertEqual(len(marl), 2)
        self.assertTrue(all(r["candidate_payloads"] for r in marl))
        self.assertTrue(all(r["evaluation_status"] == "not_full_suite_validated" for r in marl))

    def test_other_reynolds_number_does_not_replace_channel180(self):
        catalog = copy.deepcopy(self.catalog)
        catalog["directories"] = [d for d in catalog["directories"]
                                  if d["name"] != "TCFmini_3D_Re180"]
        report = resource_audit.audit(self.coverage, catalog)
        row = next(r for r in report["rows"] if r["family"] == "turbulent-channel")
        self.assertEqual(row["resource_status"], "parameter_mismatch")
        self.assertEqual(row["related_not_equivalent"], ["TCFlarge_3D_Re207"])

    def test_reject_inconsistent_revision(self):
        self.catalog["revision"] = "different"
        with self.assertRaisesRegex(ValueError, "revisions"):
            resource_audit.audit(self.coverage, self.catalog)


if __name__ == "__main__":
    unittest.main()
