"""New baselines bind frozen data; old archives are not relabelled."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from native import release_identity


class ReleaseIdentityTests(unittest.TestCase):
    def test_release_receipt_and_membership(self):
        release = {"release_id": "test-release", "training": {"case_ids": ["cylinder"]},
                   "cases": [{"id": "cylinder", "files_sha256": "f" * 64}]}
        digest = hashlib.sha256(json.dumps(release, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        receipt = {"data_release_id": "test-release", "data_release_sha256": digest}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(release_identity(root, "legacy"), {})
            (root / "release.json").write_text(json.dumps(release))
            (root / "materialized.json").write_text(json.dumps(receipt))
            self.assertEqual(release_identity(root, "cylinder"), {
                "data_release_id": "test-release", "data_release_sha256": digest,
                "case_files_sha256": "f" * 64})
            with self.assertRaisesRegex(ValueError, "outside"):
                release_identity(root, "other-reynolds")
            changed = copy.deepcopy(receipt)
            changed["data_release_sha256"] = "0" * 64
            (root / "materialized.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "mismatch"):
                release_identity(root, "cylinder")


if __name__ == "__main__":
    unittest.main()
