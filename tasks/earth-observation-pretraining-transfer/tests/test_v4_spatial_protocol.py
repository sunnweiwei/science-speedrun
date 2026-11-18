"""V4 spatial blind AWF split, dev materialization filter, and ssi_index_v3 parity tests."""

import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

TASK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "harness" / "ssi_index_v3.py").is_file())
sys.path.insert(0, str(TASK_ROOT / "tests"))
sys.path.insert(0, str(TASK_ROOT / "environment" / "evaluation"))
sys.path.insert(0, str(TASK_ROOT / "environment" / "data"))

import grade
import grade_awf_knn
import prepare_hf
import screening


def load_ssi_v3():
    harness_dir = REPO_ROOT / "harness"
    if str(harness_dir) not in sys.path:
        sys.path.insert(0, str(harness_dir))
    import ssi_index_v3

    return ssi_index_v3


class V4SpatialProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.roster_path = TASK_ROOT / "tests" / "sealed" / "spatial-roster.json"
        cls.audit_path = TASK_ROOT / "tests" / "sealed" / "spatial-roster-audit.json"
        if not cls.roster_path.is_file():
            raise unittest.SkipTest("sealed files are distributed separately (tests/sealed/README.md)")
        cls.dev_splits_path = (
            TASK_ROOT / "environment" / "data" / "awf_spatial_dev_splits.json"
        )
        cls.roster_bytes = cls.roster_path.read_bytes()
        cls.roster = json.loads(cls.roster_bytes)
        cls.audit = json.loads(cls.audit_path.read_text())
        cls.dev_splits = json.loads(cls.dev_splits_path.read_text())

    def _build_source_awf_rows(self):
        """Construct a 1,459-row source fixture matching historical_exposure and class counts."""
        by_split = {"train": [], "valid": [], "test": []}
        for row in self.roster["rows"]:
            by_split[row["split"]].append(row["id"])
        for items in by_split.values():
            items.sort()

        # Assign labels matching audit class_counts_application_order with a deterministic permutation
        # so every local neighbourhood reflects the global majority class (class 2 -> 59/205 in test).
        rng = np.random.default_rng(0)
        labels_for = {}
        for split_name, window_ids in by_split.items():
            counts = list(self.audit["class_counts_application_order"][split_name])
            seq = []
            for cls, cnt in enumerate(counts):
                seq.extend([cls] * cnt)
            assert len(seq) == len(window_ids)
            seq = rng.permutation(np.array(seq, dtype=int)).tolist()
            for wid, cls in zip(window_ids, seq):
                labels_for[wid] = cls

        legacy_split_for = {}
        for split_name, exposure in self.audit["historical_exposure"].items():
            n_train = exposure["train"]
            n_val = exposure["val"]
            window_ids = by_split[split_name]
            assert n_train + n_val == len(window_ids)
            for idx, wid in enumerate(window_ids):
                legacy_split_for[wid] = "train" if idx < n_train else "val"

        rows = []
        for idx, entry in enumerate(self.roster["rows"]):
            wid = entry["id"]
            rows.append(
                {
                    "index": idx,
                    "id": wid,
                    "split": legacy_split_for[wid],
                    "label": labels_for[wid],
                    "latlon": [-3.0 - 0.001 * idx, 36.0 + 0.001 * idx],
                }
            )
        return rows

    def test_sealed_roster_hash_and_zero_split_overlap(self):
        sha = hashlib.sha256(self.roster_bytes).hexdigest()
        self.assertEqual(
            sha, "ed0cf3cebf2d417011b367b7616bc8e3df6950a18462ec52a46356b5ba457f86"
        )
        self.assertEqual(self.audit["pairs_across_splits"], 0)
        self.assertEqual(
            self.audit["split_sizes"], {"train": 1053, "valid": 201, "test": 205}
        )
        self.assertEqual(
            self.audit["historical_exposure"]["test"], {"train": 156, "val": 49}
        )

        rows = self._build_source_awf_rows()
        splits = grade_awf_knn.aligned_splits(rows, self.roster)
        train_ids = set(splits["train"][0])
        valid_ids = set(splits["valid"][0])
        test_ids = set(splits["test"][0])

        self.assertEqual(len(train_ids), 1053)
        self.assertEqual(len(valid_ids), 201)
        self.assertEqual(len(test_ids), 205)
        self.assertEqual(len(train_ids & test_ids), 0)
        self.assertEqual(len(valid_ids & test_ids), 0)
        self.assertEqual(len(train_ids & valid_ids), 0)

        # Verify label-free dev allowlist exposes zero test IDs and zero labels
        self.assertNotIn("test", self.dev_splits)
        self.assertEqual(set(self.dev_splits.keys()), {"protocol", "source", "counts", "train", "val"})
        self.assertEqual(set(self.dev_splits["train"]), train_ids)
        self.assertEqual(set(self.dev_splits["val"]), valid_ids)
        self.assertEqual(len((set(self.dev_splits["train"]) | set(self.dev_splits["val"])) & test_ids), 0)

    def test_dev_materialization_strips_all_205_spatial_test_windows(self):
        awf_rows = self._build_source_awf_rows()
        test_ids = {
            r["id"] for r in self.roster["rows"] if r["split"] == "test"
        }
        legacy_train_test_ids = {
            r["id"] for r in awf_rows if r["id"] in test_ids and r["split"] == "train"
        }
        legacy_val_test_ids = {
            r["id"] for r in awf_rows if r["id"] in test_ids and r["split"] == "val"
        }
        self.assertEqual(len(legacy_train_test_ids), 156)
        self.assertEqual(len(legacy_val_test_ids), 49)

        lfmc_rows = [
            {"index": 0, "id": "lfmc-tr", "split": "train", "label": 80.0, "latlon": [1.0, 2.0]},
            {"index": 1, "id": "lfmc-va", "split": "val", "label": 90.0, "latlon": [1.0, 2.0]},
            {"index": 2, "id": "lfmc-te", "split": "test", "label": 999.0, "latlon": [5.0, 6.0]},
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for task, rows in (("awf", awf_rows), ("lfmc", lfmc_rows)):
                folder = root / "prepared" / "eval" / task
                folder.mkdir(parents=True)
                (folder / "labels.json").write_text(json.dumps(rows))
                with h5py.File(folder / "inputs.h5", "w") as h5:
                    h5.attrs["task"] = task
                    h5.create_dataset(
                        "ids",
                        data=[r["id"] for r in rows],
                        dtype=h5py.string_dtype(),
                    )
                    h5.create_dataset(
                        "sentinel2_l2a",
                        data=np.arange(len(rows) * 2, dtype=np.uint16).reshape(len(rows), 2),
                    )

            original_loads = json.loads

            def guarded_loads(raw, *args, **kwargs):
                parsed = original_loads(raw, *args, **kwargs)
                if isinstance(parsed, dict) and "id" in parsed and "label" in parsed:
                    self.assertNotIn(
                        parsed["id"],
                        test_ids,
                        "Excluded blind AWF test row was deserialized during dev materialization!",
                    )
                    self.assertNotEqual(parsed["id"], "lfmc-te")
                return parsed

            out_view = root / "dev_eval"
            try:
                with patch.object(screening.json, "loads", guarded_loads):
                    manifest = screening.prepare(
                        root / "prepared", out_view, fraction=1.0
                    )
                self.assertEqual(manifest["awf_excluded_spatial_test_rows"], 205)
                self.assertEqual(
                    manifest["tasks"]["awf"]["rows"], {"train": 1053, "val": 201}
                )
                dev_labels = json.loads((out_view / "awf" / "labels.json").read_text())
                self.assertEqual(len(dev_labels), 1254)
                dev_ids = {r["id"] for r in dev_labels}
                self.assertEqual(len(dev_ids & test_ids), 0)
                self.assertEqual(len(dev_ids & legacy_train_test_ids), 0)
                self.assertEqual(len(dev_ids & legacy_val_test_ids), 0)
                self.assertEqual(
                    sum(1 for r in dev_labels if r["split"] == "train"), 1053
                )
                self.assertEqual(
                    sum(1 for r in dev_labels if r["split"] == "val"), 201
                )

                with h5py.File(out_view / "awf" / "inputs.h5", "r") as h5:
                    h5_ids = h5["ids"].asstr()[:].tolist()
                self.assertEqual(h5_ids, [r["id"] for r in dev_labels])
                self.assertEqual(len(set(h5_ids) & test_ids), 0)
            finally:
                if out_view.exists():
                    out_view.chmod(0o755)
                    for child in out_view.iterdir():
                        if child.is_dir():
                            child.chmod(0o755)

    def test_prepare_hf_downstream_strips_all_205_spatial_test_windows(self):
        awf_rows = self._build_source_awf_rows()
        test_ids = {r["id"] for r in self.roster["rows"] if r["split"] == "test"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tar_path = root / "awf.tar"
            with tarfile.open(tar_path, "w") as tar:
                cfg = b"{}"
                info = tarfile.TarInfo("awf/config.json")
                info.size = len(cfg)
                tar.addfile(info, io.BytesIO(cfg))
                for r in awf_rows:
                    wid = r["id"]
                    prefix = f"awf/windows/spatial_split/{wid}"
                    payload = f"img-{wid}".encode()
                    p_info = tarfile.TarInfo(prefix + "/layers/sentinel2/geotiff.tif")
                    p_info.size = len(payload)
                    tar.addfile(p_info, io.BytesIO(payload))
                    meta = json.dumps({"name": wid, "options": {"split": r["split"]}}).encode()
                    m_info = tarfile.TarInfo(prefix + "/metadata.json")
                    m_info.size = len(meta)
                    tar.addfile(m_info, io.BytesIO(meta))

            allowed = prepare_hf.load_awf_dev_splits()
            out_dir = root / "awf-downstream-trainval"
            report = prepare_hf.prepare_windows(
                tar_path,
                out_dir,
                {"train": 1053, "val": 201, "test": 205},
                allowed_splits=allowed,
            )
            self.assertEqual(
                report["window_counts"], {"train": 1053, "val": 201, "test": 205}
            )
            extracted_dirs = {
                p.name for p in (out_dir / "windows" / "spatial_split").iterdir()
            }
            self.assertEqual(len(extracted_dirs), 1254)
            self.assertEqual(len(extracted_dirs & test_ids), 0)

    def test_constant_20nn_baseline_yields_exact_zero_awf_skill(self):
        label_sources = [self._build_source_awf_rows()]
        canonical_labels = Path(
            "/usr1/data/weiweis/openssi-preview/reruns/earth/public-reference/awf/labels.json"
        )
        if canonical_labels.is_file():
            label_sources.append(json.loads(canonical_labels.read_text()))

        for rows in label_sources:
            splits = grade_awf_knn.aligned_splits(rows, self.roster)
            with tempfile.TemporaryDirectory() as tmp:
                feat_path = Path(tmp) / "constant.npz"
                ids = np.array([r["id"] for r in rows])
                for fill_value in (0.0, 1.0):
                    feats = np.full((len(rows), 16), fill_value, dtype=np.float32)
                    np.savez(feat_path, ids=ids, features=feats)
                    scored = grade_awf_knn.score_features(feat_path, splits)
                    self.assertEqual(scored["test"]["correct"], 59)
                    self.assertEqual(scored["test"]["n"], 205)
                    self.assertAlmostEqual(
                        scored["test"]["accuracy"], 59.0 / 205.0, places=15
                    )
                    v4 = grade.compute_earth_v4_score(
                        awf_test_accuracy=scored["test"]["accuracy"],
                        lfmc_test_mae=grade.LFMC_CONSTANT_TEST_MAE,
                        awf_validation_accuracy=scored["valid"]["accuracy"],
                    )
                    self.assertAlmostEqual(v4["awf_skill"], 0.0, places=15)
                    self.assertAlmostEqual(v4["lfmc_skill"], 0.0, places=15)
                    self.assertAlmostEqual(v4["score"], 0.0, places=15)
                    self.assertAlmostEqual(v4["reward"], 0.0, places=15)

    def test_grade_matches_ssi_index_v3_across_all_regimes(self):
        ssi_v3 = load_ssi_v3()
        fixtures = [
            # Normal models (Astra, Claude, OlmoEarth-v1-Large style)
            (135.0 / 205.0, 19.03375),
            (150.0 / 205.0, 19.34380),
            (168.0 / 205.0, 18.76887),
            # Exact trivial baseline (0.0)
            (59.0 / 205.0, 24.435180349813233),
            # Perfect boundary (1.0)
            (1.0, 0.0),
            # Partial / mixed signed cancellation (one positive skill, one negative skill!)
            (0.85, 35.0),
            (0.10, 12.0),
            # Failed / worse-than-chance negative signed score clipped to 0.0
            (0.05, 40.0),
        ]
        for awf_acc, lfmc_mae in fixtures:
            container_out = grade.compute_earth_v4_score(
                awf_test_accuracy=awf_acc,
                lfmc_test_mae=lfmc_mae,
                awf_validation_accuracy=0.70,
            )
            harness_out = ssi_v3.earth(container_out)
            self.assertLess(
                abs(container_out["score"] - harness_out["score"]),
                1e-12,
                f"Score mismatch for {(awf_acc, lfmc_mae)}",
            )
            self.assertLess(
                abs(container_out["reward"] - harness_out["score"]),
                1e-12,
            )
            self.assertLess(
                abs(container_out["signed_score"] - harness_out["signed_score"]),
                1e-12,
            )
            self.assertLess(
                abs(
                    container_out["awf_skill"]
                    - harness_out["components"]["AWF classification skill"]
                ),
                1e-12,
            )
            self.assertLess(
                abs(
                    container_out["lfmc_skill"]
                    - harness_out["components"]["LFMC regression skill"]
                ),
                1e-12,
            )

    def test_visibility_and_contract_sync(self):
        dockerfile = (TASK_ROOT / "environment" / "Dockerfile").read_text()
        compose = (TASK_ROOT / "environment" / "docker-compose.yaml").read_text()
        self.assertNotIn("tests/sealed", dockerfile)
        self.assertNotIn("tests/README.md", dockerfile)
        self.assertNotIn("COPY tests ", dockerfile)
        self.assertNotIn("tests", compose)

        for doc_rel in ("instruction.md", "environment/contract.md"):
            text = (TASK_ROOT / doc_rel).read_text()
            self.assertIn("59/205", text)
            self.assertIn("24.435180349813233", text)
            self.assertIn("awf_skill", text)
            self.assertIn("lfmc_skill", text)

        readme = (TASK_ROOT / "tests" / "README.md").read_text()
        self.assertIn("## Scientific & Literature Justification of Verifier Design", readme)
        self.assertIn("### 1. Primary Metric & Upstream Paper Provenance", readme)
        self.assertIn("### 2. Why Additional Ad-Hoc Constraints Are Intentionally Omitted", readme)
        self.assertIn("### 3. Diagnostic Signals Retained in report.json", readme)


if __name__ == "__main__":
    unittest.main()
