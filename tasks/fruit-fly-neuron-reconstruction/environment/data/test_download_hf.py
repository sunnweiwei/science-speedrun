import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
from download_hf import run


class DownloadTests(unittest.TestCase):
    def test_anonymous_pinned_download_integrity_and_offline(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d); remote=r/'remote';remote.mkdir();(remote/'blobs').mkdir()
            files={}
            for name in ('development/training/value','private/suite/value'):
                value=name.encode();sha=hashlib.sha256(value).hexdigest()
                (remote/'blobs'/sha).write_bytes(value);files[name]={'bytes':len(value),'sha256':sha}
            m=remote/'release.json';m.write_text(json.dumps({'schema':'pathfinder-resources/v1','files':files}))
            config=r/'source.json';config.write_text(json.dumps({'repo_id':'owner/data','revision':'a'*40,'manifest_sha256':hashlib.sha256(m.read_bytes()).hexdigest()}))
            args=argparse.Namespace(source=config,cache=r/'cache',data=r/'dev',private=r/'private',verify_only=False)
            calls=[]
            def fetch(**kw):
                self.assertIs(kw['token'],False);self.assertEqual(kw['revision'],'a'*40)
                calls.append(kw);return remote/kw['filename']
            with patch.dict('sys.modules',{'huggingface_hub':types.SimpleNamespace(hf_hub_download=fetch)}):
                self.assertTrue(run(args)['complete']);self.assertEqual(len(calls),3)
                self.assertFalse((args.data/'suite').exists())
                args.verify_only=True;run(args);self.assertTrue(calls[-1]['local_files_only'])
                (args.data/'training/value').write_text('broken')
                with self.assertRaisesRegex(ValueError,'corrupt'):run(args)
                args.verify_only=False;run(args)
                next((remote/'blobs').iterdir()).write_text('bad')
                import shutil
                shutil.rmtree(args.data);shutil.rmtree(args.private)
                with self.assertRaises(ValueError):run(args)
            args.data=args.cache/'nested'
            with self.assertRaisesRegex(ValueError,'disjoint'):run(args)

    def test_mutable_revision_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'source.json';p.write_text(json.dumps({'revision':'main'}))
            with self.assertRaisesRegex(ValueError,'Immutable'):run(argparse.Namespace(source=p))

    def test_package_preserves_git_paths_and_rejects_corruption(self):
        from package_hf import pack
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);root=r/'source';p=root/'development/runtime/ffn/.git/HEAD'
            p.parent.mkdir(parents=True);p.write_text('pinned revision')
            sha=hashlib.sha256(p.read_bytes()).hexdigest()
            (root/'release.json').write_text(json.dumps({'schema':'pathfinder-resources/v1','files':{p.relative_to(root).as_posix():{'bytes':p.stat().st_size,'sha256':sha}}}))
            pack(root,r/'upload')
            self.assertEqual((r/'upload/blobs'/sha).read_bytes(),p.read_bytes())
            self.assertEqual((r/'upload/release.json').read_bytes(),(root/'release.json').read_bytes())
            p.write_text('corrupt')
            with self.assertRaisesRegex(ValueError,'Corrupt'):pack(root,r/'bad-upload')
