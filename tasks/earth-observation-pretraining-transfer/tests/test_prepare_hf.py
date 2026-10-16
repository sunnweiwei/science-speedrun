"""Offline integrity and split-isolation checks for HF preparation."""

import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "environment/data"))
from prepare_hf import (
    copy_stream,
    prepare_pretraining,
    prepare_windows,
    safe_path,
    verify,
)


def archive(path, rows):
    with tarfile.open(path, "w") as tar:
        for name, data in rows:
            data = data.encode()
            m = tarfile.TarInfo(name)
            m.size = len(data)
            tar.addfile(m, io.BytesIO(data))


class PreparationTests(unittest.TestCase):
    def test_manifest_paths(self):
        for name in ["../escape", "/absolute", "a/../../b", "a\\b"]:
            with self.assertRaises(ValueError):
                safe_path(name)
        self.assertEqual(str(safe_path("archives/train.tar")), "archives/train.tar")

    def test_checksum_and_size(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "data"
            p.write_bytes(b"abc")
            entry = dict(path="data", size=3, sha256=hashlib.sha256(b"abc").hexdigest())
            verify(p, entry)
            p.write_bytes(b"abd")
            with self.assertRaises(ValueError):
                verify(p, entry)

    def test_resume_must_match(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "data"
            copy_stream(io.BytesIO(b"abc"), p)
            copy_stream(io.BytesIO(b"abc"), p)
            for value in [b"ab", b"abcd", b"xyz"]:
                with self.assertRaises(ValueError):
                    copy_stream(io.BytesIO(value), p)
            self.assertEqual(p.read_bytes(), b"abc")

    def test_selected_aliases_and_exclusions(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            archive(
                p / "a.tar",
                [("root/sample_9.h5", "selected"), ("root/sample_3.h5", "excluded")],
            )
            selection = dict(
                source_samples=2,
                kept=1,
                samples=[dict(sample_index=0, source="sample_9.h5")],
                excluded=["sample_3.h5"],
            )
            self.assertEqual(
                prepare_pretraining([p / "a.tar"], selection, p / "out"), 1
            )
            self.assertEqual((p / "out/sample_0.h5").read_text(), "selected")
            self.assertEqual(len(list((p / "out").iterdir())), 1)
            self.assertEqual(
                prepare_pretraining([p / "a.tar"], selection, p / "out"), 1
            )
            with self.assertRaises(ValueError):
                prepare_pretraining([p / "a.tar", p / "a.tar"], selection, p / "out")

    def test_missing_selected_member(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            archive(p / "a.tar", [])
            selection = dict(
                source_samples=1,
                kept=1,
                samples=[dict(sample_index=0, source="sample_9.h5")],
                excluded=[],
            )
            with self.assertRaises(ValueError):
                prepare_pretraining([p / "a.tar"], selection, p / "out")

    def test_native_trainval_payload_before_metadata(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            rows = [("dataset/config.json", "{}")]
            for split in ["train", "val", "test"]:
                prefix = f"dataset/windows/spatial_split/{split}"
                rows.extend(
                    [
                        (prefix + "/layers/label/value", "label-" + split),
                        (
                            prefix + "/metadata.json",
                            json.dumps({"options": {"split": split}}),
                        ),
                    ]
                )
            archive(p / "a.tar", rows)
            expected = {"train": 1, "val": 1, "test": 1}
            report = prepare_windows(p / "a.tar", p / "out", expected)
            self.assertFalse(report["heldout_window_payloads_read"])
            self.assertFalse((p / "out/windows/spatial_split/test").exists())
            self.assertTrue(
                (p / "out/windows/spatial_split/train/layers/label/value").exists()
            )
            self.assertEqual(
                prepare_windows(p / "a.tar", p / "out", expected)["files"], 5
            )
            with self.assertRaises(ValueError):
                prepare_windows(
                    p / "a.tar", p / "other", {"train": 2, "val": 1, "test": 1}
                )

    def test_unsafe_archive_member(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            archive(p / "a.tar", [("../sample_9.h5", "x")])
            selection = dict(
                source_samples=1,
                kept=1,
                samples=[dict(sample_index=0, source="sample_9.h5")],
                excluded=[],
            )
            with self.assertRaises(ValueError):
                prepare_pretraining([p / "a.tar"], selection, p / "out")


if __name__ == "__main__":
    unittest.main()


class DownloadPipelineTests(unittest.TestCase):
    def test_hub_revision_pin_and_component_selection(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import prepare_hf

        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            payload = snapshot / "inputs"
            payload.write_bytes(b"fixture")
            entries = [
                dict(
                    path="eval/awf/inputs.h5",
                    size=7,
                    sha256=hashlib.sha256(b"fixture").hexdigest(),
                    components=["probe"],
                ),
                dict(
                    path="archives/unused.tar",
                    size=99,
                    sha256="0" * 64,
                    components=["pretrain"],
                ),
            ]
            (snapshot / "manifest.json").write_text(
                json.dumps(dict(schema_version=1, files=entries))
            )
            calls = []
            revision = "a" * 40

            def download(repo, name, **kwargs):
                calls.append((repo, name, kwargs["revision"]))
                return str(
                    snapshot / "manifest.json" if name == "manifest.json" else payload
                )

            hub = SimpleNamespace(
                HfApi=lambda: SimpleNamespace(
                    dataset_info=lambda *args, **kwargs: SimpleNamespace(sha=revision)
                ),
                hf_hub_download=download,
            )
            args = [
                "prepare_hf",
                "--repo",
                "example/data",
                "--revision",
                "v1",
                "--output",
                str(root / "output"),
                "--components",
                "probe",
            ]
            with (
                patch.dict(sys.modules, {"huggingface_hub": hub}),
                patch.object(sys, "argv", args),
            ):
                prepare_hf.main()
                prepare_hf.main()
            self.assertEqual(
                (root / "output/eval/awf/inputs.h5").read_bytes(), b"fixture"
            )
            self.assertTrue(all(r == revision for _, _, r in calls))
            self.assertNotIn("archives/unused.tar", [n for _, n, _ in calls])
            report = json.loads((root / "output/hf-prepared.json").read_text())
            self.assertEqual(report["revision"], revision)
