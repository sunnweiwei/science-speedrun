import copy
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import download_hf as download
import package_hf


class ReleaseTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / 'source'
        self.root.mkdir()
        self.name = 'training-2017-2020/statistics/manifest.json'
        p = self.root / self.name
        p.parent.mkdir(parents=True)
        p.write_text('{"example":true}\n')
        self.out = self.base / 'release'
        with patch.object(package_hf, 'components', return_value=[('development', 'training-2017-2020/statistics')]):
            self.manifest = package_hf.pack(self.root, self.out)
        self.shard = self.manifest['shards'][0]
        self.archive = self.out / self.shard['path']
        self.dest = self.base / 'restored'
        self.dest.mkdir()

    def test_roundtrip(self):
        download.validate_manifest(self.manifest)
        download.extract_shard(self.archive, self.dest, self.shard, self.manifest['files'])
        self.assertEqual((self.root / self.name).read_bytes(), (self.dest / self.name).read_bytes())
        self.assertTrue(download.matches(self.dest / self.name, self.manifest['files'][self.name]))

    def test_archive_corruption(self):
        with self.archive.open('r+b') as f:
            f.write(b'corrupted')
        with self.assertRaisesRegex(ValueError, 'Corrupt archive'):
            download.extract_shard(self.archive, self.dest, self.shard, self.manifest['files'])

    def test_wrong_member_hash(self):
        self.manifest['files'][self.name]['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Corrupt extracted'):
            download.extract_shard(self.archive, self.dest, self.shard, self.manifest['files'])
        self.assertFalse((self.dest / self.name).exists())

    def test_member_missing_or_link(self):
        for kind in ('missing', 'link'):
            with self.subTest(kind=kind):
                with tarfile.open(self.archive, 'w') as tar:
                    if kind == 'link':
                        info = tarfile.TarInfo(self.name)
                        info.type, info.linkname = tarfile.SYMTYPE, '/etc/passwd'
                        tar.addfile(info)
                self.shard.update(bytes=self.archive.stat().st_size, sha256=download.digest(self.archive))
                with self.assertRaises(ValueError):
                    download.extract_shard(self.archive, self.dest, self.shard, self.manifest['files'])

    def test_destination_symlink(self):
        (self.dest / 'training-2017-2020').symlink_to(self.root / 'training-2017-2020', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'destination'):
            download.extract_shard(self.archive, self.dest, self.shard, self.manifest['files'])

    def test_manifest_boundaries(self):
        for change in ('duplicate', 'traversal', 'scope', 'incomplete'):
            m = copy.deepcopy(self.manifest)
            if change == 'duplicate':
                m['shards'][0]['members'].append(self.name)
            elif change == 'traversal':
                m['shards'][0]['members'] = ['../outside']
            elif change == 'scope':
                m['shards'][0]['scope'] = 'evaluation'
            else:
                m['total_files'] += 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                download.validate_manifest(m)

    def test_scoped_download_resume_and_offline_corruption(self):
        evaluation = self.root / 'evaluation-resources/resources.json'
        evaluation.parent.mkdir()
        evaluation.write_text('{"test_truth":true}')
        release = self.base / 'scoped-release'
        with patch.object(package_hf, 'components', return_value=[
                ('development', 'training-2017-2020/statistics'),
                ('evaluation', 'evaluation-resources')]):
            package_hf.pack(self.root, release)
        source = self.base / 'download-source.json'
        source.write_text(json.dumps({'repo_id': 'fixture/weather', 'revision': 'a' * 40,
                                     'manifest_sha256': download.digest(release / 'release.json')}))
        args = SimpleNamespace(source=source, target=self.dest, scope='development',
                               verify_only=False, keep_archives=True)
        def fetch(**kwargs):
            self.assertIs(kwargs['token'], False)
            self.assertEqual(kwargs['revision'], 'a' * 40)
            return str(release / kwargs['filename'])
        with patch.dict('sys.modules', {'huggingface_hub': SimpleNamespace(hf_hub_download=fetch)}):
            download.run(args)
            self.assertFalse((self.dest / 'evaluation-resources').exists())
            download.run(args)
            args.verify_only = True
            download.run(args)
            (self.dest / self.name).write_bytes(b'broken')
            with self.assertRaisesRegex(ValueError, 'Missing/corrupt resources'):
                download.run(args)
            self.assertFalse((self.dest / '.hf-release/development-ready.json').exists())
            args.verify_only = False
            download.run(args)
            self.assertEqual((self.dest / self.name).read_bytes(), (self.root / self.name).read_bytes())


if __name__ == '__main__':
    unittest.main()
