"""Screening never changes canonical membership or admits final-test payloads."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "environment/evaluation"))
import screening


class TestScreening(unittest.TestCase):
    def rows(self):
        return [
            {
                "index": 0,
                "id": "train-a",
                "split": "train",
                "label": 1,
                "latlon": [1, 2],
            },
            {"index": 1, "id": "test", "split": "test", "label": 999, "latlon": [9, 9]},
            {"index": 2, "id": "val-a", "split": "val", "label": 2, "latlon": [1, 2]},
            {
                "index": 3,
                "id": "train-b",
                "split": "train",
                "label": 3,
                "latlon": [3, 4],
            },
            {"index": 4, "id": "val-b", "split": "val", "label": 4, "latlon": [3, 4]},
        ]

    def test_no_test_row_is_deserialized(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.json"
            path.write_text(json.dumps(self.rows()))
            original = json.loads

            def guarded(raw):
                self.assertNotIn('"split": "test"', raw)
                return original(raw)

            with patch.object(screening.json, "loads", guarded):
                rows = screening.trainval_rows(path)
            self.assertEqual([row["index"] for row in rows], [0, 2, 3, 4])

    def test_seeded_site_selection_preserves_entire_sites_and_splits(self):
        rows = [row for row in self.rows() if row["split"] != "test"]
        selected, sites, count = screening.sample_lfmc(rows, 20260930, 0.5)
        reversed_selection, _, _ = screening.sample_lfmc(rows[::-1], 20260930, 0.5)
        self.assertEqual(count, 2)
        self.assertEqual(len(sites), 1)
        self.assertEqual(len(selected), 2)
        self.assertEqual({row["split"] for row in selected}, {"train", "val"})
        self.assertEqual(
            {row["id"] for row in selected}, {row["id"] for row in reversed_selection}
        )

    def test_prepare_copies_only_trainval_without_modifying_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = self.rows()
            snapshots = {}
            for task in ("awf", "lfmc"):
                folder = root / "data" / "eval" / task
                folder.mkdir(parents=True)
                (folder / "labels.json").write_text(json.dumps(rows))
                with h5py.File(folder / "inputs.h5", "w") as source:
                    source.attrs["task"] = task
                    source.create_dataset(
                        "ids",
                        data=[row["id"] for row in rows],
                        dtype=h5py.string_dtype(),
                    )
                    source.create_dataset(
                        "pixels",
                        data=np.arange(10).reshape(5, 2),
                        chunks=(1, 2),
                        compression="lzf",
                    )
                snapshots[task] = [
                    screening.sha256(folder / name)
                    for name in ("inputs.h5", "labels.json")
                ]
            output = root / "view"
            try:
                result = screening.prepare(root / "data", output, fraction=1)
                self.assertEqual(result["test_rows_in_view"], 0)
                self.assertEqual(result["kind"], "fixed_validation_view")
                self.assertEqual(
                    screening.population(output)["kind"], "full_validation"
                )
                for task in ("awf", "lfmc"):
                    self.assertEqual(
                        result["tasks"][task]["rows"], {"train": 2, "val": 2}
                    )
                    source = root / "data" / "eval" / task
                    self.assertEqual(
                        snapshots[task],
                        [
                            screening.sha256(source / name)
                            for name in ("inputs.h5", "labels.json")
                        ],
                    )
                    with h5py.File(output / task / "inputs.h5", "r") as view:
                        self.assertEqual(
                            view["ids"].asstr()[:].tolist(),
                            ["train-a", "val-a", "train-b", "val-b"],
                        )
                        np.testing.assert_array_equal(
                            view["pixels"][:], np.arange(10).reshape(5, 2)[[0, 2, 3, 4]]
                        )
                    self.assertEqual(
                        (output / task / "inputs.h5").stat().st_mode & 0o222, 0
                    )
                with self.assertRaises(FileExistsError):
                    screening.prepare(root / "data", output)
                with self.assertRaises(ValueError):
                    screening.prepare(root / "data", root / "data" / "eval" / "view")
            finally:
                # Let TemporaryDirectory remove the immutable test fixture.
                if output.exists():
                    output.chmod(0o755)
                    for folder in output.iterdir():
                        if folder.is_dir():
                            folder.chmod(0o755)

    def test_cache_rejects_unrecorded_or_changed_input_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            packages = [Path(tmp) / name for name in ("random", "trained")]
            for package, origin in zip(
                packages, ("random_initialization", "own_from_scratch_training")
            ):
                package.mkdir()
                (package / "config.json").write_text('{"embedding_size": 2}')
                (package / "infer.py").write_text("same adapter")
                (package / "encoder.pt").write_bytes(b"weights")
                info = {
                    "origin": origin,
                    "checkpoint": (
                        None if origin == "random_initialization" else "checkpoint"
                    ),
                    "upstream": "pinned",
                    "initialization_seed": 12536,
                    "parameters": 2,
                    "inference_adapter_version": 3,
                    "infer_sha256": screening.sha256(package / "infer.py"),
                    "encoder_sha256": screening.sha256(package / "encoder.pt"),
                }
                (package / "provenance.json").write_text(json.dumps(info))
            context = {
                "gpu_type": "H200",
                "torch": "pinned",
                "cuda": "pinned",
                "batch_size": 16,
                "python_hash_seed": 0,
                "input_sha256": {"lfmc": "fixed-input"},
            }
            self.assertTrue(
                screening.check_cache(*packages, context, context, "lfmc")["compatible"]
            )
            for changed in (
                {**context, "input_sha256": {}},
                {**context, "input_sha256": {"lfmc": "different-input"}},
                {**context, "batch_size": 32},
            ):
                with self.assertRaises(ValueError):
                    screening.check_cache(*packages, context, changed, "lfmc")
            (packages[1] / "config.json").write_text('{"embedding_size": 3}')
            with self.assertRaisesRegex(ValueError, "architecture"):
                screening.check_cache(*packages, context, context, "lfmc")

    def test_raw_baseline_forwarded_to_unchanged_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            view = root / "view"
            view.mkdir()
            (view / "manifest.json").write_text(
                json.dumps(
                    {
                        "kind": "fixed_screening_view",
                        "test_rows_in_view": 0,
                        "population": "fixed train/validation view",
                    }
                )
            )
            evaluation = root / "evaluation"
            evaluation.mkdir()
            (evaluation / "eval").symlink_to(view)
            output = evaluation / "comparison.json"

            def fake_compare(command, **kwargs):
                self.assertEqual(command[-2:], ["--raw-baseline", "raw"])
                self.assertEqual(Path(command[1]).name, "compare_seeds.py")
                self.assertNotIn("--final-test", command)
                output.write_text('{"tasks": {}}')

            argv = [
                "screening.py",
                "compare",
                "--view",
                str(view),
                "--data-root",
                str(evaluation),
                "--pairs",
                "random:trained",
                "--raw-baseline",
                "raw",
                "--output",
                str(output),
            ]
            with patch.object(sys, "argv", argv), patch.object(
                screening.subprocess, "run", fake_compare
            ), contextlib.redirect_stdout(io.StringIO()):
                screening.main()
            report = json.loads(output.read_text())
            self.assertEqual(report["evaluation_population"]["kind"], "screening")
            self.assertFalse(
                report["evaluation_population"]["final_recipe_criterion_established"]
            )


if __name__ == "__main__":
    unittest.main()
