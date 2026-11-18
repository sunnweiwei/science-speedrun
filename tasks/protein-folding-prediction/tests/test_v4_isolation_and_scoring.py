"""V4 tests for two-container label isolation (`--phase infer` / `--phase score`),
dual-entrypoint (`setup.sh` + `infer.sh`) canary & symlink escape defenses,
`tests/docker-compose.yaml` volume invariants, CAPRI anti-collapse geometry, and
`< 1e-12` parity with `harness/ssi_index_v3.folding()`."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import struct
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

import yaml

TEST_ROOT = Path(__file__).resolve().parent
TASK_ROOT = TEST_ROOT.parent
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "harness" / "ssi_index_v3.py").is_file())

sys.path.insert(0, str(TEST_ROOT))
import grade


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ssi_index_v3 = _load_module("ssi_index_v3", REPO_ROOT / "harness/ssi_index_v3.py")


VALID_CIF = """data_pred
loop_
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
CA ALA A 1.0 2.0 3.0
#
"""

SENTINEL_GT_CIF = """data_SECRET_GROUND_TRUTH_SENTINEL_998877
loop_
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
CA ALA A 99.0 88.0 77.0
#
"""


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_synthetic_safetensors(path: Path, digest: str, length: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    value_bytes = length * grade.FEATURE_WIDTH
    scale_bytes = length * (grade.FEATURE_WIDTH // grade.FEATURE_GROUP_SIZE) * 2
    header = {
        "values": {
            "dtype": "I8",
            "shape": [length, grade.FEATURE_WIDTH],
            "data_offsets": [0, value_bytes],
        },
        "scales": {
            "dtype": "BF16",
            "shape": [length, grade.FEATURE_WIDTH // grade.FEATURE_GROUP_SIZE],
            "data_offsets": [value_bytes, value_bytes + scale_bytes],
        },
        "__metadata__": {
            "sequence_sha256": digest,
            "esmc_repo": "biohub/ESMC-6B",
            "esmc_revision": grade.ESMC_REVISION,
            "boundary": grade.FEATURE_BOUNDARY,
            "kernel_backend": grade.KERNEL_BACKEND,
            "schema_version": grade.FEATURE_VERSION,
            "quantization": "symmetric-int8",
            "group_size": str(grade.FEATURE_GROUP_SIZE),
            "scale_dtype": "bfloat16",
        },
    }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(value_bytes + scale_bytes))


def build_nine_track_benchmark(root: Path):
    """Build separated inference benchmark mount (no ground_truths) + private ground_truths dir."""
    infer_mount = root / "main_benchmark"
    suite = infer_mount / "suite"
    (suite / "requests").mkdir(parents=True)
    (suite / "targets").mkdir(parents=True)
    gt_dir = root / "grader_only_ground_truths"
    gt_dir.mkdir(parents=True)

    manifest_requests = []
    feature_rows = []
    seen_seqs = set()

    for idx, track in enumerate(grade.TRACKS):
        req_id = f"target-{idx:02d}"
        seq = "ACDEFGHIKLMNPQRSTVWY"[: 6 + idx]
        req = {
            "schema_version": "foldbench-folding-request/v0",
            "request_id": req_id,
            "target": {"entities": [{"id": "E1", "type": "protein", "sequence": seq, "copies": 1}]},
            "run": {"seeds": [101], "num_samples": 1},
        }
        req_path = suite / "requests" / f"{req_id}.json"
        req_path.write_text(json.dumps(req), encoding="utf-8")
        manifest_requests.append({
            "request_id": req_id,
            "path": f"requests/{req_id}.json",
            "sha256": hashlib.sha256(req_path.read_bytes()).hexdigest(),
        })
        (gt_dir / f"{req_id}.cif").write_text(SENTINEL_GT_CIF, encoding="utf-8")

        if track.startswith("monomer_"):
            write_csv(suite / "targets" / f"{track}.csv", [{"pdb_id": req_id, "chain_id": "A"}])
        elif track == "interface_protein_ligand":
            write_csv(
                suite / "targets" / f"{track}.csv",
                [{"pdb_id": req_id, "native_chain_id_1": "A", "native_chain_id_2": "L"}],
            )
        else:
            write_csv(
                suite / "targets" / f"{track}.csv",
                [{"pdb_id": req_id, "interface_chain_id_1": "A", "interface_chain_id_2": "B"}],
            )

        if seq not in seen_seqs:
            seen_seqs.add(seq)
            digest = hashlib.sha256(seq.encode("utf-8")).hexdigest()
            rel_feat = f"features/esmc6b-final-int8/{digest}.safetensors"
            write_synthetic_safetensors(infer_mount / rel_feat, digest, len(seq))
            feature_rows.append(json.dumps({
                "sequence_sha256": digest,
                "length": len(seq),
                "shape": [len(seq), 2560],
                "dtype": "int8",
                "scale_shape": [len(seq), 40],
                "scale_dtype": "bfloat16",
                "quantization": "symmetric-int8",
                "group_size": 64,
                "feature_version": grade.FEATURE_VERSION,
                "path": rel_feat,
            }))

    (suite / "suite_manifest.json").write_text(json.dumps({
        "contains_ground_truth_coordinates": False,
        "requests": manifest_requests,
    }), encoding="utf-8")
    feature_manifest = infer_mount / "features.esmc6b_final_int8.jsonl"
    feature_manifest.write_text("\n".join(feature_rows) + "\n", encoding="utf-8")
    return infer_mount, suite, feature_manifest, gt_dir


class V4IsolationAndScoringTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_five_modality_score_matches_ssi_index_v3_across_all_fixtures(self):
        fixtures = {
            "esmfold2_fast_ref": {
                "monomer_protein": 0.822, "monomer_rna": 0.569, "monomer_dna": 0.372,
                "interface_protein_protein": 0.552, "interface_antibody_antigen": 0.291,
                "interface_protein_peptide": 0.608, "interface_protein_rna": 0.529,
                "interface_protein_dna": 0.649, "interface_protein_ligand": 0.511,
            },
            "from_scratch_starter": {
                "monomer_protein": 0.0, "monomer_rna": 0.0, "monomer_dna": 0.0,
                "interface_protein_protein": 0.057, "interface_antibody_antigen": 0.017,
                "interface_protein_peptide": 0.118, "interface_protein_rna": 0.086,
                "interface_protein_dna": 0.058, "interface_protein_ligand": 0.0,
            },
            "asymmetric_modality_weights": {
                # Deliberately asymmetric: flat 9-track mean = 3/9 = 0.3333...,
                # whereas 5-modality equal mean = (0 + 0 + 1.0 + 0 + 0)/5 = 0.2!
                "monomer_protein": 0.0, "monomer_rna": 0.0, "monomer_dna": 0.0,
                "interface_protein_protein": 1.0, "interface_antibody_antigen": 1.0,
                "interface_protein_peptide": 1.0, "interface_protein_rna": 0.0,
                "interface_protein_dna": 0.0, "interface_protein_ligand": 0.0,
            },
            "zero": {track: 0.0 for track in grade.TRACKS},
            "perfect": {track: 1.0 for track in grade.TRACKS},
        }
        for name, track_scores in fixtures.items():
            with self.subTest(fixture=name):
                score, modalities = grade.aggregate_modalities(track_scores)
                v3 = ssi_index_v3.folding(track_scores)
                self.assertLess(abs(score - v3["score"]), 1e-12)
                for group_name, group_val in v3["components"].items():
                    self.assertLess(abs(modalities[group_name] - group_val), 1e-12)
        # Verify asymmetric fixture specifically differs from legacy 1/9 flat average:
        asym_score, _ = grade.aggregate_modalities(fixtures["asymmetric_modality_weights"])
        self.assertAlmostEqual(asym_score, 0.2, places=12)
        self.assertNotAlmostEqual(asym_score, 3.0 / 9.0, places=3)

    def test_chain_collapse_with_high_fnat_and_dockq_is_strictly_rejected(self):
        targets = [
            {"pdb_id": "collapse-1", "interface_chain_id_1": "A", "interface_chain_id_2": "B"},
            {"pdb_id": "collapse-2", "interface_chain_id_1": "A", "interface_chain_id_2": "B"},
        ]
        # Chain-collapse gaming: fnat=1.0, DockQ=0.35 >= 0.23, but LRMSD=25.0 > 10 and iRMSD=12.0 > 4.
        # Must NEVER pass via an `fnat >= 0.1` OR clause!
        collapsed_raw = [
            {
                "pdb_id": "collapse-1", "interface_chain_id_1": "A", "interface_chain_id_2": "B",
                "ranking_score": "0.99", "fnat": "1.0", "dockq_score": "0.35",
                "lrmsd": "25.0", "irmsd": "12.0",
            },
            # Missing second target stays in fixed denominator (0/2 = 0.0).
        ]
        for track in (
            "interface_protein_protein",
            "interface_antibody_antigen",
            "interface_protein_peptide",
            "interface_protein_rna",
            "interface_protein_dna",
        ):
            res = grade.track_score(track, targets, collapsed_raw)
            self.assertEqual(res["score"], 0.0, track)
            self.assertEqual(res["targets"], 2)
            self.assertEqual(res["missing_or_invalid"], 1)

        # Exact CAPRI boundary checks:
        one_target = [targets[0]]
        boundary_irmsd = [{
            "pdb_id": "collapse-1", "interface_chain_id_1": "A", "interface_chain_id_2": "B",
            "ranking_score": "1.0", "dockq_score": "0.23", "irmsd": "4.0", "lrmsd": "25.0",
        }]
        boundary_lrmsd = [{
            "pdb_id": "collapse-1", "interface_chain_id_1": "A", "interface_chain_id_2": "B",
            "ranking_score": "1.0", "dockq_score": "0.23", "irmsd": "12.0", "lrmsd": "10.0",
        }]
        just_outside = [{
            "pdb_id": "collapse-1", "interface_chain_id_1": "A", "interface_chain_id_2": "B",
            "ranking_score": "1.0", "fnat": "1.0", "dockq_score": "0.35",
            "irmsd": "4.0001", "lrmsd": "10.0001",
        }]
        self.assertEqual(grade.track_score("interface_protein_protein", one_target, boundary_irmsd)["score"], 1.0)
        self.assertEqual(grade.track_score("interface_protein_protein", one_target, boundary_lrmsd)["score"], 1.0)
        self.assertEqual(grade.track_score("interface_protein_protein", one_target, just_outside)["score"], 0.0)

    def test_two_phase_isolated_pipeline_end_to_end_and_partial_missing_targets(self):
        _, suite, feature_manifest, gt_dir = build_nine_track_benchmark(self.root / "e2e")
        submission = self.root / "compliant_sub"
        submission.mkdir()
        (submission / "infer.sh").write_text(
            '#!/usr/bin/env bash\nset -euo pipefail\nexec python3 "$(dirname "$0")/run_infer.py" "$1" "$2"\n',
            encoding="utf-8",
        )
        (submission / "infer.sh").chmod(0o755)
        # Predict 8 of 9 targets (omit target-08 to exercise partial/missing-target denominator too!)
        (submission / "run_infer.py").write_text(textwrap.dedent(f"""\
            import csv, json, pathlib, sys
            inp, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
            feat_rows = [json.loads(l) for l in (inp / "features.esmc6b_final_int8.jsonl").read_text().splitlines() if l.strip()]
            assert len(feat_rows) == 9
            for row in feat_rows:
                fpath = inp / row["path"]
                assert fpath.is_symlink() and fpath.is_file() and fpath.stat().st_size > 0
            (out / "structures").mkdir(parents=True)
            csv_rows = []
            for line in (inp / "requests.jsonl").read_text().splitlines():
                req = json.loads(line)
                rid = req["request_id"]
                if rid == "target-08":
                    continue  # Intentional missing target -> scores 0 in fixed denominator
                rel = f"structures/{{rid}}.cif"
                (out / rel).write_text({VALID_CIF!r}, encoding="utf-8")
                csv_rows.append({{"pdb_id": rid, "seed": 101, "sample": 0, "ranking_score": 0.9, "prediction_path": rel}})
            with (out / "prediction_reference.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames={grade.FIELDS!r})
                w.writeheader()
                w.writerows(csv_rows)
        """), encoding="utf-8")

        sealed = self.root / "sealed_vol/run"
        infer_out = self.root / "main_logs"
        infer_out.mkdir()
        score_out = self.root / "grader_results"
        score_out.mkdir()

        infer_args = argparse.Namespace(
            phase="infer",
            submission=submission,
            suite=suite,
            features=feature_manifest,
            ground_truth=None,
            sealed=sealed,
            results=None,
            output=infer_out,
            wait_seconds=5,
        )
        grade.run(infer_args)
        self.assertTrue((sealed.parent / "READY").is_file())

        real_run = subprocess.run

        def mock_foldbench_eval(command, **kwargs):
            if command[0] == "/opt/foldbench-env/bin/python":
                eval_raw = score_out / "evaluation/submission/raw"
                for idx, track in enumerate(grade.TRACKS):
                    rid = f"target-{idx:02d}"
                    if rid == "target-08":
                        continue
                    suffix = "dockqv2" if track in grade.NA_TRACKS else "ost"
                    if track.startswith("monomer_"):
                        row = {"pdb_id": rid, "chain_id": "A", "ranking_score": "0.9", "lddt": f"{0.4 + 0.05 * idx:.4f}"}
                    else:
                        row = {
                            "pdb_id": rid,
                            "interface_chain_id_1": "A",
                            "interface_chain_id_2": "B",
                            "ranking_score": "0.9",
                            "dockq_score": "0.55" if idx % 2 == 0 else "0.15",
                            "irmsd": "2.1",
                            "lrmsd": "5.5",
                        }
                    write_csv(eval_raw / f"{track}_{suffix}.csv", [row])
                return subprocess.CompletedProcess(command, 0)
            return real_run(command, **kwargs)

        score_args = argparse.Namespace(
            phase="score",
            submission=None,
            suite=suite,
            features=None,
            ground_truth=gt_dir,
            sealed=sealed,
            results=None,
            output=score_out,
            wait_seconds=2,
        )
        with patch.object(grade.subprocess, "run", side_effect=mock_foldbench_eval):
            report = grade.run(score_args)

        reward = json.loads((score_out / "reward.json").read_text(encoding="utf-8"))
        self.assertTrue(report["label_isolated_sandbox"])
        self.assertEqual(reward["score"], reward["reward"])
        self.assertEqual(reward["interface_protein_ligand"], 0.0)  # missing target-08 scored 0
        v3 = ssi_index_v3.folding(reward)
        self.assertLess(abs(reward["score"] - v3["score"]), 1e-12)
        self.assertLess(abs(report["score"] - v3["score"]), 1e-12)
        for mod_name, mod_score in v3["components"].items():
            self.assertLess(abs(report["modalities"][mod_name] - mod_score), 1e-12)

    def test_dual_entrypoint_red_team_canary_and_symlink_escape_rejected(self):
        infer_mount, suite, feature_manifest, gt_dir = build_nine_track_benchmark(self.root / "canary")
        canary_log = self.root / "exfiltration_report.jsonl"
        grader_exec_marker = self.root / "UNTRUSTED_CODE_EXECUTED_IN_GRADER"

        malicious = self.root / "malicious_sub"
        malicious.mkdir()
        # Both setup.sh AND infer.sh probe for ground_truths sentinels AND check feature readability.
        probe_py = malicious / "probe.py"
        probe_py.write_text(textwrap.dedent(f"""\
            import json, os, pathlib, sys
            stage = sys.argv[1]
            marker = pathlib.Path({str(grader_exec_marker)!r})
            if os.environ.get("IN_TRUSTED_GRADER") == "1":
                marker.write_text(stage)
            search_roots = [pathlib.Path("/benchmark"), pathlib.Path({str(infer_mount)!r})]
            if len(sys.argv) > 2:
                search_roots.append(pathlib.Path(sys.argv[2]))
            leaked = []
            for root in search_roots:
                if not root.exists():
                    continue
                for p in root.rglob("*"):
                    if "ground_truth" in p.name:
                        leaked.append(str(p))
                    elif p.is_file() and p.suffix == ".cif":
                        try:
                            if "SECRET_GROUND_TRUTH_SENTINEL" in p.read_text(errors="ignore"):
                                leaked.append(str(p))
                        except OSError:
                            pass
            feat_ok = False
            if len(sys.argv) > 2:
                inp = pathlib.Path(sys.argv[2])
                fmanifest = inp / "features.esmc6b_final_int8.jsonl"
                if fmanifest.is_file():
                    rows = [json.loads(l) for l in fmanifest.read_text().splitlines() if l.strip()]
                    feat_ok = len(rows) == 9 and all((inp / r["path"]).is_file() for r in rows)
            with pathlib.Path({str(canary_log)!r}).open("a", encoding="utf-8") as out:
                out.write(json.dumps({{"stage": stage, "leaked": leaked, "features_readable": feat_ok}}) + "\\n")
        """), encoding="utf-8")

        (malicious / "setup.sh").write_text(
            '#!/usr/bin/env bash\nset -euo pipefail\npython3 "$(dirname "$0")/probe.py" setup\n',
            encoding="utf-8",
        )
        (malicious / "setup.sh").chmod(0o755)
        # Malicious infer.sh ALSO attempts to drop a symlink pointing to `/benchmark/ground_truths/target-00.cif`!
        (malicious / "infer.sh").write_text(textwrap.dedent(f"""\
            #!/usr/bin/env bash
            set -euo pipefail
            dir="$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd)"
            python3 "$dir/probe.py" infer "$1"
            python3 - "$1" "$2" <<'PY'
            import csv, json, os, pathlib, sys
            inp, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
            os.symlink("/benchmark/ground_truths/target-00.cif", out / "escaped.cif")
            with (out / "prediction_reference.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames={grade.FIELDS!r})
                w.writeheader()
                w.writerow({{"pdb_id": "target-00", "seed": 101, "sample": 0, "ranking_score": 1.0, "prediction_path": "escaped.cif"}})
            PY
        """), encoding="utf-8")
        (malicious / "infer.sh").chmod(0o755)

        sealed = self.root / "canary_sealed/run"
        infer_out = self.root / "canary_main_logs"
        infer_out.mkdir()

        infer_args = argparse.Namespace(
            phase="infer",
            submission=malicious,
            suite=suite,
            features=feature_manifest,
            ground_truth=None,
            sealed=sealed,
            results=None,
            output=infer_out,
            wait_seconds=2,
        )
        with self.assertRaisesRegex(grade.InvalidSubmission, "Symlink forbidden"):
            grade.run(infer_args)

        # Verify BOTH setup.sh and infer.sh ran in --phase infer, saw ZERO ground-truth files,
        # AND saw valid readable .safetensors features in infer.sh!
        probes = [json.loads(l) for l in canary_log.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([p["stage"] for p in probes], ["setup", "infer"])
        self.assertEqual(probes[0]["leaked"], [])
        self.assertEqual(probes[1]["leaked"], [])
        self.assertTrue(probes[1]["features_readable"])

        # Verify --phase score processes the sealed invalid_submission without touching malicious submission
        score_out = self.root / "canary_grader_out"
        score_out.mkdir()
        score_args = argparse.Namespace(
            phase="score",
            submission=None,
            suite=suite,
            features=None,
            ground_truth=gt_dir,
            sealed=sealed,
            results=None,
            output=score_out,
            wait_seconds=2,
        )
        with patch.dict(os.environ, {"IN_TRUSTED_GRADER": "1"}):
            with self.assertRaises(grade.InvalidSubmission):
                grade.run(score_args)
        self.assertFalse(grader_exec_marker.exists())

        # Defense-in-depth: even if an attacker bypassed `main` and planted a direct symlink to
        # `gt_dir / "target-00.cif"` inside a forged `sealed` directory, `--phase score` rejects it
        # BEFORE reading any .cif or invoking FoldBench!
        forged_sealed = self.root / "forged_sealed/run"
        (forged_sealed / "structures").mkdir(parents=True)
        os.symlink(gt_dir / "target-00.cif", forged_sealed / "structures/000000.cif")
        write_csv(forged_sealed / "prediction_reference.csv", [{
            "pdb_id": "target-00", "seed": 101, "sample": 0,
            "ranking_score": 1.0, "prediction_path": "structures/000000.cif",
        }])
        grade.write_json(forged_sealed / "sealed_manifest.json", {
            "schema": grade.SEALED_SCHEMA,
            "status": "ok",
            "suite_manifest_sha256": hashlib.sha256((suite / "suite_manifest.json").read_bytes()).hexdigest(),
            "files": {},
        })
        (forged_sealed.parent / "READY").write_text(
            grade.sha256_file(forged_sealed / "sealed_manifest.json") + "\n", encoding="utf-8"
        )
        forged_out = self.root / "forged_grader_out"
        forged_out.mkdir()
        with (
            patch.object(grade.subprocess, "run") as spy_subproc,
            patch.object(grade, "validate_mmcif") as spy_mmcif,
        ):
            with self.assertRaisesRegex(grade.InvalidSubmission, "Symlink forbidden"):
                grade.run(argparse.Namespace(
                    phase="score",
                    submission=None,
                    suite=suite,
                    features=None,
                    ground_truth=gt_dir,
                    sealed=forged_sealed,
                    results=None,
                    output=forged_out,
                    wait_seconds=1,
                ))
            spy_subproc.assert_not_called()
            spy_mmcif.assert_not_called()

        proc_res = subprocess.run(
            [
                sys.executable, str(TEST_ROOT / "grade.py"),
                "--phase", "score",
                "--suite", str(suite),
                "--ground-truth", str(gt_dir),
                "--sealed", str(forged_sealed),
                "--output", str(forged_out),
                "--wait-seconds", "1",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("invalid submission", proc_res.stdout.lower())
        forged_reward = json.loads((forged_out / "reward.json").read_text(encoding="utf-8"))
        self.assertEqual(forged_reward["score"], 0.0)
        self.assertEqual(forged_reward["reward"], 0.0)

        # Also verify `--submission` is strictly forbidden in `--phase score` and `--ground-truth` in `--phase infer`.
        with self.assertRaisesRegex(ValueError, "forbidden"):
            grade.run(argparse.Namespace(phase="score", submission=malicious))
        with self.assertRaisesRegex(RuntimeError, "ground_truths must NOT be mounted"):
            grade.run(argparse.Namespace(phase="infer", ground_truth=gt_dir))

    def test_docker_compose_enforces_two_container_label_isolation_and_feature_mounts(self):
        compose = yaml.safe_load((TEST_ROOT / "docker-compose.yaml").read_text(encoding="utf-8"))
        services = compose["services"]
        self.assertEqual(set(services), {"main", "grader"})
        self.assertEqual(services["main"].get("network_mode"), "none")
        self.assertEqual(services["grader"].get("network_mode"), "none")

        def normalize_mounts(vol_list):
            parsed = []
            for entry in vol_list:
                if isinstance(entry, dict):
                    parsed.append((
                        str(entry["source"]),
                        str(entry["target"]),
                        bool(entry.get("read_only", False)),
                    ))
                else:
                    parts = str(entry).split(":")
                    src, tgt = parts[0], parts[1]
                    ro = len(parts) >= 3 and "ro" in parts[2].split(",")
                    parsed.append((src, tgt, ro))
            return parsed

        main_mounts = normalize_mounts(services["main"]["volumes"])
        grader_mounts = normalize_mounts(services["grader"]["volumes"])

        # 1. `main` MUST have ZERO `ground_truths` mounts and must NOT mount the parent `/benchmark` root!
        for src, tgt, _ in main_mounts:
            self.assertNotIn("ground_truth", src)
            self.assertNotIn("ground_truth", tgt)
            self.assertNotEqual(PurePosixPath(tgt), PurePosixPath("/benchmark"))

        # 2. `main` MUST mount `/benchmark/suite:ro`, `/benchmark/features.esmc6b_final_int8.jsonl:ro`,
        #    and cover `/benchmark/features/esmc6b-final-int8` read-only!
        self.assertTrue(any(tgt == "/benchmark/suite" and ro for _, tgt, ro in main_mounts))
        self.assertTrue(any(tgt == "/benchmark/features.esmc6b_final_int8.jsonl" and ro for _, tgt, ro in main_mounts))
        esmc_subpath = PurePosixPath("/benchmark/features/esmc6b-final-int8")
        self.assertTrue(
            any(
                ro and PurePosixPath(tgt) != PurePosixPath("/benchmark")
                and esmc_subpath.is_relative_to(PurePosixPath(tgt))
                for _, tgt, ro in main_mounts
            ),
            f"main mounts {main_mounts} do not cover {esmc_subpath} read-only",
        )

        # 3. `grader` MUST mount `sealed` read-only (`:ro`), mount `ground_truths` read-only,
        #    run `--phase score`, and never mount or reference `submission`.
        self.assertTrue(
            any(src == "sealed" and tgt == "/sealed" and ro for src, tgt, ro in grader_mounts),
            f"grader must mount sealed read-only (:ro), got {grader_mounts}",
        )
        self.assertTrue(
            any("ground_truths" in src and tgt == "/benchmark/ground_truths" and ro for src, tgt, ro in grader_mounts)
        )
        for src, tgt, _ in grader_mounts:
            self.assertNotIn("submission", src)
            self.assertNotIn("submission", tgt)
        grader_cmd = " ".join(map(str, services["grader"]["command"]))
        self.assertIn("--phase score", grader_cmd)
        self.assertNotIn("submission", grader_cmd)

        test_sh = (TEST_ROOT / "test.sh").read_text(encoding="utf-8")
        self.assertIn("--phase infer", test_sh)
        self.assertNotIn("ground-truth", test_sh)
        self.assertNotIn("ground_truth", test_sh)

        # 4. Agent visibility boundary & sealed README justification headings:
        dev_dockerfile = (TASK_ROOT / "environment/Dockerfile").read_text(encoding="utf-8")
        dev_compose = (TASK_ROOT / "environment/docker-compose.yaml").read_text(encoding="utf-8")
        self.assertNotIn("COPY tests", dev_dockerfile)
        self.assertNotIn("COPY . /tests", dev_dockerfile)
        self.assertNotIn("/tests", dev_compose)

        sealed_readme = (TEST_ROOT / "README.md").read_text(encoding="utf-8")
        for heading in (
            "## Scientific & Literature Justification of Verifier Design",
            "### 1. Primary Metric & Upstream Paper Provenance",
            "### 2. Why Additional Ad-Hoc Constraints Are Intentionally Omitted",
            "### 3. Diagnostic Signals Retained in report.json",
        ):
            self.assertIn(heading, sealed_readme)


if __name__ == "__main__":
    unittest.main()
