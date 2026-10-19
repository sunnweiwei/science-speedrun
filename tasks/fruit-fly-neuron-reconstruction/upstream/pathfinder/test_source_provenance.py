"""Portable provenance must retain content identity without a Git installation."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from test_fs_support import TemporaryDirectory
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import common


class SourceProvenanceTests(unittest.TestCase):
    def test_no_git_records_source_hash_and_detects_changed_code(self):
        with TemporaryDirectory() as tmp:
            source=Path(tmp)/'common.py';source.write_text('value = 1\n')
            with patch.object(common,'__file__',str(source)),patch.dict(os.environ,{'UPSTREAM_SOURCE_COMMIT':''}),patch.object(common.subprocess,'run',side_effect=FileNotFoundError('git')):
                first=common.source_provenance();source.write_text('value = 2\n');second=common.source_provenance()
            self.assertEqual(first['commit'],'unversioned-copy')
            self.assertEqual(len(first['python_tree_sha256']),64)
            self.assertNotEqual(first['python_tree_sha256'],second['python_tree_sha256'])
    def test_explicit_commit_does_not_require_git(self):
        with patch.dict(os.environ,{'UPSTREAM_SOURCE_COMMIT':'frozen-source-commit'}),patch.object(common.subprocess,'run') as run:
            result=common.source_provenance()
        run.assert_not_called();self.assertEqual(result['commit'],'frozen-source-commit')

if __name__=='__main__':unittest.main()
