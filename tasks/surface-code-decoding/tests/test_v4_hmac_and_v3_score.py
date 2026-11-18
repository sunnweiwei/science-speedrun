"""V4 security & V3 metric parity unit tests for #17 surface-code-decoding."""
from __future__ import annotations

import hashlib
import hmac
import importlib.util
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

import numpy as np

TASK = Path(__file__).resolve().parents[1]
PROJECT = (
    TASK / "environment" / "baseline" / "alphaqubit"
    if (TASK / "environment" / "baseline" / "alphaqubit").is_dir()
    else TASK
)
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from baselines.alphaqubit_artifact import EXPECTED_GROUPS
from qecbench.data import SCHEMA, load_condition, sha256, write_json
from qecbench.scoring import score as qec_score


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


build_release = _load_module("qec_build_release", TASK / "environment" / "data" / "build_release.py")
materializer = _load_module("qec_materializer_v4", TASK / "environment" / "data" / "materialize.py")
grader = _load_module("qec_grader_v4", TASK / "tests" / "grader.py")
supervisor = _load_module("qec_supervisor_v4", TASK / "tests" / "main_verifier.py")
ssi_v3 = _load_module("ssi_index_v3", next(p for p in Path(__file__).resolve().parents if (p / "harness" / "ssi_index_v3.py").is_file()) / "harness" / "ssi_index_v3.py")


class SurfaceCodeV4SecurityAndScoreTests(unittest.TestCase):
    def test_no_blake2b_and_key_required(self) -> None:
        src_text = (TASK / "environment" / "data" / "build_release.py").read_text()
        self.assertNotIn("blake2b", src_text)

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(build_release.HMAC_KEY_ENV, None)
            with self.assertRaises(ValueError):
                build_release.resolve_secret_key()
            with self.assertRaises(ValueError):
                build_release.resolve_secret_key(secret_key=b"too-short")

    def test_hmac_sample_ids_differ_across_keys_and_block_unkeyed_bruteforce(self) -> None:
        cond = "surface_code_bX_d3_r05_center_3_5"
        indices = np.arange(1, 201, 2, dtype=np.int64)
        key_a = b"v4-test-secret-key-alpha-0123456789"
        key_b = b"v4-test-secret-key-bravo-9876543210"

        ids_a, tokens_a = build_release.sample_ids_with_tokens(cond, indices, key_a)
        ids_b, tokens_b = build_release.sample_ids_with_tokens(cond, indices, key_b)

        self.assertEqual(ids_a.dtype, np.uint64)
        self.assertEqual(len(ids_a), 100)
        self.assertEqual(len(set(ids_a.tolist()) & set(ids_b.tolist())), 0)
        self.assertEqual(len(set(tokens_a) & set(tokens_b)), 0)

        # Verify exact HMAC-SHA256[:32] formula and 64-bit uint64 projection
        for idx, uid, tok in zip(indices, ids_a, tokens_a):
            expected_tok = hmac.new(
                key_a, f"{cond}:{int(idx)}".encode("utf-8"), hashlib.sha256
            ).hexdigest()[:32]
            self.assertEqual(tok, expected_tok)
            self.assertEqual(int(uid), int(expected_tok[:16], 16))

        # Verify zero overlap with legacy unkeyed blake2b or unkeyed sha256
        legacy_blake = {
            int.from_bytes(
                hashlib.blake2b(
                    f"{build_release.RELEASE}|{cond}|{int(i)}".encode(), digest_size=8
                ).digest(),
                "little",
            )
            for i in range(50_000)
        }
        unkeyed_sha = {
            int(hashlib.sha256(f"{cond}:{int(i)}".encode()).hexdigest()[:16], 16)
            for i in range(50_000)
        }
        self.assertEqual(len(set(ids_a.tolist()) & legacy_blake), 0)
        self.assertEqual(len(set(ids_a.tolist()) & unkeyed_sha), 0)

    def test_deterministic_keyed_test_shot_shuffle(self) -> None:
        cond = "surface_code_bZ_d5_r25_center_5_5"
        raw_indices = np.arange(1, 201, 2, dtype=np.int64)  # 100 odd shots
        detectors = np.arange(200, dtype=np.uint8).reshape(100, 2)
        labels = (np.arange(100, dtype=np.uint8) % 2)

        key_a = b"shuffle-hmac-secret-key-alpha-123456"
        key_b = b"shuffle-hmac-secret-key-bravo-654321"

        prep_a1 = build_release.prepare_test_condition_split(cond, detectors, labels, raw_indices, key_a)
        prep_a2 = build_release.prepare_test_condition_split(cond, detectors, labels, raw_indices, key_a)
        prep_b = build_release.prepare_test_condition_split(cond, detectors, labels, raw_indices, key_b)

        # Deterministic under identical secret_key
        np.testing.assert_array_equal(prep_a1["sample_ids"], prep_a2["sample_ids"])
        np.testing.assert_array_equal(prep_a1["detectors"], prep_a2["detectors"])
        self.assertEqual(prep_a1["source_shot_indices"], prep_a2["source_shot_indices"])

        # Destroys sequential temporal shot order and differs across keys
        self.assertNotEqual(prep_a1["source_shot_indices"], raw_indices.tolist())
        self.assertNotEqual(prep_a1["source_shot_indices"], prep_b["source_shot_indices"])
        self.assertEqual(sorted(prep_a1["source_shot_indices"]), raw_indices.tolist())

    def test_end_to_end_release_build_isolates_private_map_and_grades_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = tmp / "raw"
            ref_results = tmp / "ref_results"
            out_dir = tmp / "release_out"
            key_file = tmp / "hmac_secret.key"
            secret_key = b"end-to-end-surface-code-secret-key-2026-v4"
            key_file.write_bytes(secret_key + b"\n")
            ref_results.mkdir(parents=True)
            (ref_results / "README.md").write_text("fixture reference results\n")

            from dataclasses import dataclass

            @dataclass(frozen=True)
            class FakeCondition:
                path: Path
                name: str
                distance: int
                basis: str
                rounds: int
                group_id: str

            fake_conditions: list[FakeCondition] = []
            for group in EXPECTED_GROUPS:
                distance, basis = int(group[1]), group[4]
                center = group.split("center_")[1]
                for rounds in range(1, 26, 2):
                    name = f"surface_code_b{basis}_d{distance}_r{rounds:02}_center_{center}"
                    cdir = source / name
                    cdir.mkdir(parents=True)
                    (cdir / "circuit_ideal.stim").write_text("# ideal fixture\n")
                    (cdir / "pij_from_even_for_odd.dem").write_text("# dem fixture\n")
                    fake_conditions.append(
                        FakeCondition(
                            path=cdir,
                            name=name,
                            distance=distance,
                            basis=basis,
                            rounds=rounds,
                            group_id=group,
                        )
                    )

            split_map = {
                "train": np.array([0, 2, 4, 6], dtype=np.int64),
                "validation": np.array([8, 10, 12, 14], dtype=np.int64),
                "test": np.array([1, 3, 5, 7, 9, 11, 13, 15], dtype=np.int64),
            }

            def fake_load_packed(condition: FakeCondition, split: str):
                idxs = split_map[split]
                n_bytes = (condition.rounds * (condition.distance**2 - 1) + 7) // 8
                dets = np.zeros((len(idxs), n_bytes), dtype=np.uint8)
                dets[:, 0] = (idxs % 251).astype(np.uint8)
                labs = ((idxs // 2) % 2).astype(np.uint8).reshape(-1, 1)
                return dets, labs

            import types

            fake_baselines_data = types.ModuleType("baselines.data")
            fake_baselines_data.discover_conditions = lambda _root: fake_conditions
            fake_baselines_data.hardware_split_indices = lambda split, _n=50_000: split_map[split]
            fake_baselines_data.load_hardware_packed = fake_load_packed

            with mock.patch.dict(sys.modules, {"baselines.data": fake_baselines_data}):
                manifest_res = build_release.build(
                    source, ref_results, out_dir, hmac_key_file=key_file
                )

            self.assertEqual(
                set(manifest_res["files"]),
                {"development.tar.gz", "verifier-inputs.tar.gz", "grader-labels.tar.gz"},
            )

            # Verify archive member boundaries: private_sample_id_map.json ONLY in grader-labels.tar.gz
            for archive_name, must_contain_map in (
                ("development.tar.gz", False),
                ("verifier-inputs.tar.gz", False),
                ("grader-labels.tar.gz", True),
            ):
                with tarfile.open(out_dir / archive_name, "r:gz") as tar:
                    names = {m.name for m in tar.getmembers()}
                has_map = any("private_sample_id_map" in n for n in names)
                self.assertEqual(has_map, must_contain_map, archive_name)

            # Verify role inventories pass strict materializer check
            for role in ("development", "verifier", "grader"):
                materializer.verify_role(out_dir / "materialized" / role)

            # Verify private map accurately maps shuffled sample_ids back to source_shot_indices
            priv_map = json.loads(
                (out_dir / "materialized" / "grader" / "private_sample_id_map.json").read_text()
            )
            self.assertEqual(len(priv_map["conditions"]), 130)
            sample_cond = fake_conditions[0].name
            entry = priv_map["conditions"][sample_cond]
            self.assertEqual(sorted(entry["source_shot_indices"]), split_map["test"].tolist())
            self.assertNotEqual(entry["source_shot_indices"], split_map["test"].tolist())

            # Verify full grading compatibility through grader.grade() + ssi_index_v3 parity
            verifier_root = out_dir / "materialized" / "verifier"
            grader_root = out_dir / "materialized" / "grader"
            sealed = tmp / "sealed" / "run"
            sealed.mkdir(parents=True)
            v_manifest = json.loads((verifier_root / "manifest.json").read_text())
            snap_files = {}
            for row in v_manifest["conditions"]:
                cond_arrays = load_condition(verifier_root, row)
                ids = cond_arrays["sample_ids"]
                with np.load(grader_root / "conditions" / f"{row['name']}.npz", allow_pickle=False) as truth:
                    # Flip 1 error on d3 and 2 errors on d5 to test non-trivial V3 score
                    probs = truth["labels"].astype(np.float32)
                    if row["distance"] == 3:
                        probs[0] = 1.0 - probs[0]
                    else:
                        probs[:2] = 1.0 - probs[:2]
                pred_file = sealed / f"{row['name']}.npz"
                # Reverse IDs in prediction to verify ID-aligned grading on HMAC uint64 IDs
                np.savez(pred_file, sample_ids=ids[::-1], probabilities=probs[::-1])
                snap_files[pred_file.name] = sha256(pred_file)

            run_id = "b" * 32
            write_json(
                sealed / "snapshot.json",
                {
                    "schema": "surface-code-decoding-snapshot/v1",
                    "run_id": run_id,
                    "input_manifest_sha256": sha256(verifier_root / "manifest.json"),
                    "files": snap_files,
                },
            )
            results_dir = tmp / "sealed" / "results"
            grader.grade(
                sealed,
                verifier_root,
                grader_root,
                PROJECT / "score_config.json",
                results_dir,
                {"run_id": run_id, "snapshot_sha256": sha256(sealed / "snapshot.json")},
            )
            report = json.loads((results_dir / run_id / "report.json").read_text())
            reward = json.loads((results_dir / run_id / "reward.json").read_text())
            v3_res = ssi_v3.surface_code(report["failure_by_distance"])

            # d3 fails 1/8 = 0.125; d5 fails 2/8 = 0.25 -> P = 1 - 0.5*(0.125 + 0.25) = 0.8125
            self.assertAlmostEqual(v3_res["score"], 0.8125, places=12)
            self.assertLess(abs(reward["score"] - v3_res["score"]), 1e-12)
            self.assertLess(abs(reward["reward"] - v3_res["score"]), 1e-12)
            self.assertLess(abs(report["performance"] - v3_res["score"]), 1e-12)

            # Also verify invalid prediction produces {"score": 0.0, "reward": 0.0}
            bad_sealed = tmp / "sealed_bad" / "run"
            bad_sealed.mkdir(parents=True)
            bad_run_id = "c" * 32
            write_json(
                bad_sealed / "snapshot.json",
                {
                    "schema": "surface-code-decoding-snapshot/v1",
                    "run_id": bad_run_id,
                    "input_manifest_sha256": sha256(verifier_root / "manifest.json"),
                    "files": {},
                },
            )
            grader.grade(
                bad_sealed,
                verifier_root,
                grader_root,
                PROJECT / "score_config.json",
                results_dir,
                {"run_id": bad_run_id, "snapshot_sha256": sha256(bad_sealed / "snapshot.json")},
            )
            bad_reward = json.loads((results_dir / bad_run_id / "reward.json").read_text())
            self.assertEqual(bad_reward, {"score": 0.0, "reward": 0.0})

    def test_v3_metric_boundary_fixtures_and_visibility_isolation(self) -> None:
        for fail_d3, fail_d5 in ((0.0, 0.0), (1.0, 1.0), (0.18, 0.39), (0.5, 0.5), (0.0412, 0.2879)):
            by_dist = {"d3": fail_d3, "d5": fail_d5}
            v3_score = ssi_v3.surface_code(by_dist)["score"]
            container_score = max(0.0, min(1.0, 0.5 * ((1.0 - fail_d3) + (1.0 - fail_d5))))
            self.assertLess(abs(container_score - v3_score), 1e-12)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            supervisor.invalid(tmp, "invalid_submission", ValueError("missing manifest"))
            self.assertEqual(
                json.loads((tmp / "reward.json").read_text()),
                {"score": 0.0, "reward": 0.0},
            )

        dockerfile_text = (TASK / "environment" / "Dockerfile").read_text()
        compose_text = (TASK / "environment" / "docker-compose.yaml").read_text()
        self.assertNotIn("COPY tests", dockerfile_text)
        self.assertNotIn("/tests", dockerfile_text)
        self.assertNotIn("tests", compose_text)

        readme_text = (TASK / "tests" / "README.md").read_text()
        self.assertIn("## Scientific & Literature Justification of Verifier Design", readme_text)
        self.assertIn("### 1. Primary Metric & Upstream Paper Provenance", readme_text)
        self.assertIn("### 2. Why Additional Ad-Hoc Constraints Are Intentionally Omitted", readme_text)
        self.assertIn("### 3. Diagnostic Signals Retained in report.json", readme_text)
        self.assertIn("10.1038/s41586-024-08148-8", readme_text)
        self.assertIn("6804040", readme_text)


if __name__ == "__main__":
    unittest.main()
