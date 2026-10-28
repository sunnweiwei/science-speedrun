import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'environment/evaluation'))
from data_identity import split_identity


class DataIdentityTests(unittest.TestCase):
    def test_identity_binds_shards_and_private_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'manifest.json').write_text('{}')
            (root / 'checksums.json').write_text('["first-shard-hash"]')
            truth = root / 'truth.jsonl'
            truth.write_text('{"label":0}')
            original = split_identity(root, truth)
            (root / 'checksums.json').write_text('["different-shard-hash"]')
            self.assertNotEqual(original, split_identity(root, truth))
            changed_shards = split_identity(root, truth)
            truth.write_text('{"label":1}')
            self.assertNotEqual(changed_shards, split_identity(root, truth))
            self.assertNotEqual(split_identity(root), split_identity(root, truth))


if __name__ == '__main__':
    unittest.main()
