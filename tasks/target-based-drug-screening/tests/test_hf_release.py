from __future__ import annotations

import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from drugclip_data import hf_release


COMMIT = "c8d2febc996e4c96b10f3aecd89fb7375150449f"


class HuggingFaceReleaseTest(unittest.TestCase):
    def test_default_source_pins_verified_dataset_release(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                hf_release.huggingface_source(),
                ("zifeng-ai/target-based-drug-screening", COMMIT),
            )
            self.assertEqual(hf_release.data_source(), "huggingface")
        with patch.dict(
            "os.environ",
            {
                "TARGET_BASED_DRUG_SCREENING_HF_REPO": "",
                "TARGET_BASED_DRUG_SCREENING_HF_REVISION": "",
            },
            clear=True,
        ):
            self.assertEqual(
                hf_release.huggingface_source(),
                ("zifeng-ai/target-based-drug-screening", COMMIT),
            )
        with patch.dict(
            "os.environ",
            {
                "TARGET_BASED_DRUG_SCREENING_HF_REPO": "owner/repo",
                "TARGET_BASED_DRUG_SCREENING_HF_REVISION": "release-v2",
                "TARGET_BASED_DRUG_SCREENING_DATA_SOURCE": "upstream",
            },
        ):
            self.assertEqual(
                hf_release.huggingface_source(), ("owner/repo", "release-v2")
            )
            self.assertEqual(hf_release.data_source(), "upstream")

    def _hub_module(self):
        class Api:
            def repo_info(self, **kwargs):
                return types.SimpleNamespace(sha=COMMIT)

        return types.SimpleNamespace(HfApi=Api)

    @staticmethod
    def _download(stage: Path, repo: str, revision: str) -> str:
        (stage / ".complete.json").write_text('{"status":"PASS"}\n')
        (stage / "new-data.bin").write_bytes(b"new")
        return COMMIT

    def test_valid_staged_release_replaces_previous_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            root.mkdir()
            (root / "old-data.bin").write_bytes(b"old")
            with (
                patch.dict("sys.modules", {"huggingface_hub": self._hub_module()}),
                patch.object(hf_release, "_download_release", side_effect=self._download),
                patch.object(hf_release, "_prepare_layout"),
                patch.object(hf_release, "validate_dataset"),
            ):
                receipt = hf_release.materialize_huggingface(root)
            self.assertFalse((root / "old-data.bin").exists())
            self.assertEqual((root / "new-data.bin").read_bytes(), b"new")
            self.assertEqual(receipt["resolved_commit"], COMMIT)
            self.assertEqual(
                json.loads((root / "HF_MATERIALIZATION.json").read_text()), receipt
            )

    def test_failed_validation_preserves_previous_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            root.mkdir()
            (root / "old-data.bin").write_bytes(b"old")
            with (
                patch.dict("sys.modules", {"huggingface_hub": self._hub_module()}),
                patch.object(hf_release, "_download_release", side_effect=self._download),
                patch.object(hf_release, "_prepare_layout"),
                patch.object(hf_release, "validate_dataset", side_effect=ValueError("bad release")),
            ):
                with self.assertRaisesRegex(ValueError, "bad release"):
                    hf_release.materialize_huggingface(root)
            self.assertEqual((root / "old-data.bin").read_bytes(), b"old")
            self.assertFalse((root / "HF_MATERIALIZATION.json").exists())
            self.assertFalse(any(root.glob(".hf-release-*")))

    def test_release_membership_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "unexpected.bin").write_bytes(b"no")
            with self.assertRaisesRegex(ValueError, "unexpected file"):
                hf_release._release_files(root)


if __name__ == "__main__":
    unittest.main()
