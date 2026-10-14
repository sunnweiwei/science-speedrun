"""Offline tests of the exact archive codec used for the public data release."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch

DATA = Path(__file__).resolve().parents[1]/'environment/data'


def module(name):
    spec = importlib.util.spec_from_file_location(name, DATA/f'{name}.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


packer, downloader = module('pack_release'), module('materialize')


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source, self.upload, self.destination = (self.root/p for p in ('source','upload','data'))
        self.payloads = {
            'manifests/manifest.json': b'{"frozen":true}',
            'objects/structure.npz': b'fixture structure',
            'features/a.safetensors': b'fixture feature A',
            'features/b.safetensors': b'fixture feature B',
            'foldbench/suite/requests/sample.json': b'{"id":"sample"}',
            'models/esmfold2-fast-b28d8ace/model.safetensors': b'fixture reference weight',
        }
        for name, content in self.payloads.items():
            path = self.source/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(content)
        with contextlib.redirect_stdout(io.StringIO()):
            self.manifest = packer.pack(self.source,self.upload,shard_bytes=20)
        self.fetches = []

    def fake_download(self, command, **kwargs):
        self.assertEqual(command[:3], ['hf','download','owner/data'])
        self.assertEqual(command[4:8], ['--type','dataset','--revision','a'*40])
        self.assertTrue(kwargs['check'])
        self.fetches.append(command[3])
        destination = Path(command[9])/command[3]
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(self.upload/command[3],destination)

    def restore(self, **kwargs):
        with patch.object(downloader.subprocess,'run',side_effect=self.fake_download), \
                contextlib.redirect_stdout(io.StringIO()):
            return downloader.materialize(self.destination,self.manifest,'owner/data','a'*40,**kwargs)

    def test_manifest_archive_and_inventory_hashes(self):
        manifest = downloader.load_manifest(self.upload/'release-manifest.json')
        rows = downloader.load_inventory(self.upload/'payload-files.jsonl',manifest)
        self.assertEqual({row['path'] for row in rows},set(self.payloads))
        for row in rows:
            self.assertEqual(row['sha256'],hashlib.sha256(self.payloads[row['path']]).hexdigest())
        for archive in manifest['archives']:
            downloader.verify_file(self.upload/archive['path'],archive)
        self.assertEqual(manifest['payload_bytes'],sum(map(len,self.payloads.values())))

    def test_round_trip_resume_and_offline_verification(self):
        receipt = self.restore()
        self.assertEqual(receipt['payload_files'],len(self.payloads))
        for name, content in self.payloads.items():
            self.assertEqual((self.destination/name).read_bytes(),content)
        self.assertFalse(list((self.destination/'.foldbench-download/archives').glob('*.tar')))
        self.fetches.clear()
        self.restore()
        self.assertEqual(self.fetches,['payload-files.jsonl'])
        self.fetches.clear()
        self.restore(verify_only=True)
        self.assertEqual(self.fetches,[])

    def test_keep_archives(self):
        self.restore(keep_archives=True)
        self.assertEqual(len(list((self.destination/'.foldbench-download/archives').glob('*.tar'))),
                         len(self.manifest['archives']))

    def test_packing_is_byte_reproducible(self):
        with contextlib.redirect_stdout(io.StringIO()):
            again = packer.pack(self.source,self.root/'again',shard_bytes=20)
        self.assertEqual(self.manifest,again)

    def test_interrupted_temporary_write_does_not_pollute_payloads(self):
        temporary = self.destination/'.foldbench-download/tmp/interrupted'
        temporary.parent.mkdir(parents=True)
        temporary.write_bytes(b'partial disposable write')
        self.restore()
        self.restore(verify_only=True)

    def test_resume_incomplete_shard(self):
        target = self.destination/'features/a.safetensors'
        target.parent.mkdir(parents=True)
        target.write_bytes(self.payloads['features/a.safetensors'])
        inode = target.stat().st_ino
        self.restore()
        self.assertEqual(target.stat().st_ino,inode)

    def test_existing_wrong_file_is_not_overwritten(self):
        target = self.destination/'features/a.safetensors'
        target.parent.mkdir(parents=True)
        target.write_bytes(b'existing unrelated bytes')
        with self.assertRaisesRegex(ValueError,'checksum/size'):
            self.restore()
        self.assertEqual(target.read_bytes(),b'existing unrelated bytes')

    def test_archive_corruption_rejected(self):
        archive = self.upload/self.manifest['archives'][0]['path']
        content = bytearray(archive.read_bytes()); content[-1] ^= 1
        archive.write_bytes(content)
        with self.assertRaisesRegex(ValueError,'checksum/size'):
            self.restore()

    def test_inventory_corruption_rejected(self):
        (self.upload/'payload-files.jsonl').write_text('{}\n')
        with self.assertRaisesRegex(ValueError,'checksum/size'):
            self.restore()

    def make_tar(self, entries):
        target = self.root/'test.tar'
        with tarfile.open(target,'w') as archive:
            for name, content, kind in entries:
                info = tarfile.TarInfo(name); info.type = kind
                info.size = len(content) if kind==tarfile.REGTYPE else 0
                info.linkname = '../outside'
                archive.addfile(info,io.BytesIO(content))
        return target

    def test_bad_tar_entries_and_missing_members_rejected(self):
        name, content = 'features/a', b'payload'
        record = {'path':name,'size':len(content),'sha256':hashlib.sha256(content).hexdigest()}
        for entries in (
            [('../outside',content,tarfile.REGTYPE)],
            [(name,content,tarfile.SYMTYPE)],
            [(name,content,tarfile.REGTYPE)]*2,
            [],
            [(name,b'changed',tarfile.REGTYPE)],
        ):
            with self.subTest(entries=entries):
                destination = self.root/f'bad-{len(list(self.root.glob("bad-*")))}'
                with self.assertRaises(ValueError):
                    downloader.unpack(self.make_tar(entries),destination,[record])
        self.assertFalse((self.root/'outside').exists())

    def test_symlink_and_extra_payload_rejected(self):
        self.destination.mkdir()
        (self.destination/'features').symlink_to(self.source/'features',target_is_directory=True)
        with self.assertRaises(ValueError):
            self.restore()
        (self.destination/'features').unlink()
        self.restore()
        (self.destination/'features/extra').write_bytes(b'extra')
        with self.assertRaisesRegex(ValueError,'membership'):
            self.restore(verify_only=True)
        self.assertTrue((self.destination/'features/extra').exists())

    def test_offline_missing_payload_rejected(self):
        self.restore()
        (self.destination/'features/a.safetensors').unlink()
        with self.assertRaisesRegex(ValueError,'Missing files'):
            self.restore(verify_only=True)

    def test_packer_rejects_unapproved_weights(self):
        (self.source/'models/esmc6b').mkdir()
        (self.source/'models/esmc6b/weights').write_bytes(b'not for publication')
        with self.assertRaisesRegex(ValueError,'approved reference'):
            packer.pack(self.source,self.root/'rejected')

    def test_publication_pin_matches_checked_in_manifest(self):
        config = json.loads((DATA/'download-source.json').read_text())
        self.assertEqual(config['repo_id'],'sunweiwei/ai4sci-foldbench-data')
        self.assertRegex(config['revision'],r'^[0-9a-f]{40}$')
        self.assertEqual(config['manifest_sha256'],downloader.sha256(DATA/'release-manifest.json'))
        manifest = downloader.load_manifest(DATA/'release-manifest.json')
        self.assertEqual(manifest['payload_files'],185995)
        self.assertEqual(manifest['payload_bytes'],73068582840)


if __name__=='__main__':
    unittest.main()
