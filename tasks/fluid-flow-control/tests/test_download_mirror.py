"""No-network tests for delivering identical frozen data from our Hub mirror."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'environment/data'))
import materialize
import public_release as public


class DownloadMirrorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.catalog = {'repo_id':'authors/source', 'revision':'a'*40,
                        'directories':[{'name':'fixture','bytes':3}]}
        self.release = {'release_id':'fixture-v1','training':{'case_ids':['fixture']}}
        self.mirror = {'repo_id':'organizer/data','revision':'b'*40,
            'path_prefix':'environments','data_release_id':'fixture-v1',
            'data_release_sha256':public.digest(self.release)}
        self.patch = patch.object(materialize, 'CATALOG', self.root/'catalog.json')
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def write_mirror(self, value=None):
        (self.root/'download-source.json').write_text(json.dumps(value or self.mirror))

    def test_current_release_uses_pinned_data_repo(self):
        self.write_mirror()
        result=materialize.download_source(self.catalog,self.release,['fixture'])
        self.assertEqual(result, {k:self.mirror[k] for k in ('repo_id','revision','path_prefix')})

    def test_missing_mirror_preserves_legacy_source(self):
        self.assertEqual(materialize.download_source(self.catalog,self.release,['fixture']),
            {'repo_id':'authors/source','revision':'a'*40,'path_prefix':''})

    def test_archive_and_extra_inventory_use_original_source(self):
        self.write_mirror()
        older=copy.deepcopy(self.release);older['release_id']='older'
        for release,names in [(older,['fixture']),(self.release,['fixture','excluded'])]:
            self.assertEqual(materialize.download_source(self.catalog,release,names)['repo_id'],'authors/source')

    def test_release_hash_must_match(self):
        self.write_mirror(dict(self.mirror,data_release_sha256='wrong'))
        with self.assertRaisesRegex(ValueError,'another frozen release'):
            materialize.download_source(self.catalog,self.release,['fixture'])

    def test_unpinned_revision_and_unsafe_locations_rejected(self):
        for field,value in [('revision','main'),('revision',None),('repo_id','https://host/path'),
                            ('repo_id',None),('path_prefix','../escape')]:
            with self.subTest(field=field,value=value):
                self.write_mirror(dict(self.mirror,**{field:value}))
                with self.assertRaises(ValueError):
                    materialize.download_source(self.catalog,self.release,['fixture'])

    def test_download_layout_checksums_and_receipt_keep_source_identity(self):
        index={'files':[{'rfilename':'fixture/input','size':3,
            'blobId':hashlib.sha1(b'blob 3\0abc').hexdigest()}]}
        self.release['source_index_sha256']=public.digest(index)
        self.mirror['data_release_sha256']=public.digest(self.release)
        self.write_mirror()
        (self.root/'source-files.json').write_text(json.dumps(index))
        (self.root/'catalog.json').write_text(json.dumps(self.catalog))
        target=self.root/'prepared'
        def download(command, **kwargs):
            self.assertEqual(command[2],'organizer/data')
            self.assertEqual(command[command.index('--local-dir')+1],str(target))
            self.assertEqual(command[command.index('--include')+1],'environments/fixture/*')
            payload=target/'environments/fixture/input'
            payload.parent.mkdir(parents=True);payload.write_bytes(b'abc')
        with patch.object(public,'load_release',return_value=self.release), \
             patch.object(public,'validate_index'), \
             patch.object(sys,'argv',['materialize',str(target),'--download']), \
             patch.object(materialize.subprocess,'run',side_effect=download), \
             contextlib.redirect_stdout(io.StringIO()):
            materialize.main()
        receipt=json.loads((target/'materialized.json').read_text())
        self.assertEqual(receipt['repo_id'],'authors/source')
        self.assertEqual(receipt['revision'],'a'*40)
        self.assertEqual(receipt['download_source']['repo_id'],'organizer/data')
        self.assertEqual(receipt['download_source']['revision'],'b'*40)
        self.assertEqual(receipt['data_release_sha256'],public.digest(self.release))
        self.assertEqual(receipt['verified_files'],1)


if __name__=='__main__':unittest.main()
