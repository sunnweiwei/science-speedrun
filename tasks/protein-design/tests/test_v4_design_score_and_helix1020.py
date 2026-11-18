"""CPU unit tests for V4 protein-design score parity, diagnostics, and 1,020-slot helix control."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

_TESTS_DIR = Path(__file__).resolve().parent
_TASK_DIR = _TESTS_DIR.parent
_HARNESS_DIR = next(p for p in Path(__file__).resolve().parents if (p / "harness" / "ssi_index_v3.py").is_file()) / "harness"
for _p in (_TESTS_DIR, _HARNESS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import generate_trivial_helix_1020 as helix1020  # noqa: E402
import grader  # noqa: E402
import main_verifier  # noqa: E402
import ssi_index_v3  # noqa: E402
from common import TRACKS, validate_harness, validate_input  # noqa: E402
from geometry import parse_allatom_pdb  # noqa: E402
from test_task import Fixture, StubScorer  # noqa: E402


class ProteinDesignV4ParityAndControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="pd_v4_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_combined_score_matches_ssi_index_v3_within_1e12(self) -> None:
        cases = [
            # Reference rfd3-small pass rates
            {"unconditional": 0.420, "sm_binder": 0.0275, "ppi_binder": 0.002, "enzyme": 0.0624},
            # Operator harness pass rates
            {"unconditional": 0.410, "sm_binder": 0.030, "ppi_binder": 0.004, "enzyme": 0.054},
            # Partial / asymmetric
            {"unconditional": 1.0, "sm_binder": 0.0, "ppi_binder": 0.0, "enzyme": 0.0},
            {"unconditional": 0.0, "sm_binder": 0.25, "ppi_binder": 0.50, "enzyme": 0.125},
            # Boundaries
            {"unconditional": 0.0, "sm_binder": 0.0, "ppi_binder": 0.0, "enzyme": 0.0},
            {"unconditional": 1.0, "sm_binder": 1.0, "ppi_binder": 1.0, "enzyme": 1.0},
        ]
        for rates in cases:
            with self.subTest(rates=rates):
                got = grader.combined_score(rates)
                expected = ssi_index_v3.design(rates)["score"]
                self.assertLess(abs(got - expected), 1e-12)

    def test_grader_main_writes_score_reward_and_diagnostics(self) -> None:
        fixture = Fixture(self.tmpdir / "fixture")
        untrusted = self.tmpdir / "untrusted"
        (untrusted / "predictions" / "structures").mkdir(parents=True)
        # Copy compliant predictions into untrusted/predictions/ so seal_designs seals them
        for track in TRACKS:
            fixture.submit(track, fixture.compliant_builder(track))
            shutil.copyfile(
                fixture.snapshot / f"{track}.jsonl",
                untrusted / "predictions" / f"{track}.jsonl",
            )
        for pdb in (fixture.snapshot / "structures").glob("*.pdb"):
            shutil.copyfile(pdb, untrusted / "predictions" / "structures" / pdb.name)

        sealed = self.tmpdir / "exchange" / "sealed"
        sealed.parent.mkdir(parents=True)
        main_verifier.seal_designs(untrusted, sealed, fixture.input / "manifest.json")

        out = self.tmpdir / "verifier_out"
        argv = [
            "grader.py",
            "--sealed",
            str(sealed),
            "--contexts",
            str(fixture.input),
            "--output",
            str(out),
            "--reference",
            str(self.tmpdir / "empty_ref"),
        ]
        with patch("sys.argv", argv), patch("novelty.FitNovelty") as mock_nov:
            mock_nov.return_value.max_tm.return_value = {}
            grader.main(scorer=StubScorer(sc_rmsd=1.125))

        reward = json.loads((out / "reward.json").read_text())
        report = json.loads((out / "report.json").read_text())
        rates = {t: report["tracks"][t]["pass_rate"] for t in TRACKS}
        expected = ssi_index_v3.design(rates)["score"]

        self.assertIn("score", reward)
        self.assertIn("reward", reward)
        self.assertLess(abs(reward["score"] - expected), 1e-12)
        self.assertLess(abs(reward["reward"] - expected), 1e-12)
        self.assertLess(abs(report["score"] - expected), 1e-12)
        self.assertLess(abs(report["reward"] - expected), 1e-12)
        for track in TRACKS:
            self.assertAlmostEqual(report["tracks"][track]["backbone_sc_rmsd_mean"], 1.125)
            self.assertAlmostEqual(
                report["tracks"][track]["designable_backbone_sc_rmsd_mean"], 1.125
            )

    def test_invalid_submission_writes_zero_score_and_reward(self) -> None:
        out = self.tmpdir / "invalid_out"
        sealed = self.tmpdir / "invalid_exchange" / "sealed"
        main_verifier.invalid_submission(out, sealed, ValueError("bad package"))
        reward = json.loads((out / "reward.json").read_text())
        report = json.loads((out / "report.json").read_text())
        self.assertEqual(reward["score"], 0.0)
        self.assertEqual(reward["reward"], 0.0)
        self.assertEqual(report["score"], 0.0)
        self.assertEqual(report["reward"], 0.0)

    def test_full_1020_slot_trivial_helix_control_contract_and_stub_score(self) -> None:
        targets_root = _TASK_DIR / "environment" / "data" / "targets"
        view_dir = self.tmpdir / "view_51"
        out_dir = self.tmpdir / "trivial_1020"
        sealed_dir = self.tmpdir / "exchange_1020" / "sealed"
        pkg_dir = self.tmpdir / "package_1020"

        counts = helix1020.materialize_targets_view(targets_root, view_dir)
        self.assertEqual(
            counts,
            {"unconditional": 1, "sm_binder": 4, "ppi_binder": 5, "enzyme": 41},
        )
        manifest, requests = validate_input(view_dir)
        self.assertEqual(sum(manifest["counts"].values()), 51)

        emitted = helix1020.generate_predictions(view_dir, out_dir)
        self.assertEqual(
            emitted,
            {"unconditional": 20, "sm_binder": 80, "ppi_binder": 100, "enzyme": 820},
        )
        self.assertEqual(sum(emitted.values()), 1020)

        # Verify every single one of the 1,020 PDB files parses and passes request validation.
        structures_root = out_dir / "predictions" / "structures"
        parsed_total = 0
        for track in TRACKS:
            rows = [
                json.loads(line)
                for line in (out_dir / "predictions" / f"{track}.jsonl").read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), len(requests[track]))
            for row in rows:
                req = requests[track][row["example_id"]]
                ctx = grader.load_context(view_dir / req["context_path"])
                self.assertEqual(len(row["designs"]), 20)
                for design in row["designs"]:
                    pdb_path = structures_root / design["structure_path"]
                    structure = parse_allatom_pdb(pdb_path, req["design_length"])
                    reason, _ = grader.validate_request(
                        track, structure, design.get("motif_map"), ctx
                    )
                    self.assertIsNone(reason, f"{track}:{row['example_id']} failed: {reason}")
                    parsed_total += 1
        self.assertEqual(parsed_total, 1020)

        # Seal and run end-to-end through grader.main with StubScorer.
        sealed_dir.parent.mkdir(parents=True, exist_ok=True)
        main_verifier.seal_designs(out_dir, sealed_dir, view_dir / "manifest.json")
        grade_out = self.tmpdir / "grade_1020"
        argv = [
            "grader.py",
            "--sealed",
            str(sealed_dir),
            "--contexts",
            str(view_dir),
            "--output",
            str(grade_out),
            "--reference",
            str(self.tmpdir / "no_ref"),
        ]
        with patch("sys.argv", argv), patch("novelty.FitNovelty") as mock_nov:
            mock_nov.return_value.max_tm.return_value = {}
            grader.main(scorer=StubScorer())

        report = json.loads((grade_out / "report.json").read_text())
        reward = json.loads((grade_out / "reward.json").read_text())
        for track, expected_req in (
            ("unconditional", 20),
            ("sm_binder", 80),
            ("ppi_binder", 100),
            ("enzyme", 820),
        ):
            self.assertEqual(report["tracks"][track]["designs_requested"], expected_req)
            self.assertEqual(report["tracks"][track]["designs_valid"], expected_req)
            self.assertEqual(report["tracks"][track]["conditioning_satisfied"], expected_req)

        # Conditioned tracks must all score 0.0 on un-grafted canonical straight helices
        for cond_track in ("sm_binder", "ppi_binder", "enzyme"):
            self.assertEqual(report["tracks"][cond_track]["pass_rate"], 0.0, cond_track)

        rates = {t: report["tracks"][t]["pass_rate"] for t in TRACKS}
        expected_v3 = ssi_index_v3.design(rates)["score"]
        self.assertLess(abs(reward["score"] - expected_v3), 1e-12)
        self.assertLess(abs(reward["reward"] - expected_v3), 1e-12)

        # Verify submission package passes harness validation and executes infer.sh cleanly
        helix1020.write_submission_package(pkg_dir)
        main_verifier.validate_submission(pkg_dir)
        harness_manifest = validate_harness(pkg_dir, self.tmpdir / "empty_pretrained")
        self.assertEqual(harness_manifest["declared_parameters"], 0)

        fixture = Fixture(self.tmpdir / "pkg_smoke_fixture")
        smoke_out = self.tmpdir / "pkg_smoke_out"
        smoke_out.mkdir()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        subprocess.run(
            ["bash", str(pkg_dir / "infer.sh"), str(fixture.input), str(smoke_out)],
            check=True,
            env=env,
        )
        self.assertTrue((smoke_out / "predictions" / "enzyme.jsonl").is_file())

    def test_docs_formula_and_visibility_invariants(self) -> None:
        instruction = (_TASK_DIR / "instruction.md").read_text()
        contract = (_TASK_DIR / "environment" / "contract.md").read_text()
        tests_readme = (_TESTS_DIR / "README.md").read_text()
        dockerfile = (_TASK_DIR / "environment" / "Dockerfile").read_text()

        for text in (instruction, contract, tests_readme):
            lower = text.lower()
            self.assertNotIn("there is no single combined score", lower)
            self.assertNotIn("there is no composite score", lower)
            self.assertIn("unconditional / 51 + (50 / 153) * (sm_binder + ppi_binder + enzyme)", text)

        for heading in (
            "## Scientific & Literature Justification of Verifier Design",
            "### 1. Primary Metric & Upstream Paper Provenance",
            "### 2. Why Additional Ad-Hoc Constraints Are Intentionally Omitted",
            "### 3. Diagnostic Signals Retained in report.json",
        ):
            self.assertIn(heading, tests_readme)

        # Verify /assets remains intact and private verifier files never enter /workspace
        self.assertIn("/assets", dockerfile)
        for forbidden in (
            "tests/README.md",
            "tests/main_verifier.py",
            "tests/generate_trivial_helix_1020.py",
        ):
            self.assertNotIn(forbidden, dockerfile)


if __name__ == "__main__":
    unittest.main(verbosity=2)
