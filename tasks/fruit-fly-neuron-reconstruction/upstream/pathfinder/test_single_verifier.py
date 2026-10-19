import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch
from test_fs_support import TemporaryDirectory
from verify_single import stage_weights,InvalidSubmission,namespace_command,run_process
from common import sha256


class SingleVerifierTests(unittest.TestCase):
    def fixture(self,root):
        package=root/'input';(package/'weights').mkdir(parents=True)
        files={}
        for name in ('ffn','sense','shape'):
            p=package/f'weights/{name}.pt';p.write_bytes(name.encode());files[f'weights/{name}.pt']=sha256(p)
        (package/'package.json').write_text(json.dumps(dict(schema='pathfinder-model-package/v1',
            files=files,models={},inference=dict(search_threshold=.8))))
        return package

    def test_does_not_copy_or_execute_submitted_code(self):
        with TemporaryDirectory() as d:
            root=Path(d);p=self.fixture(root);(p/'runtime').mkdir()
            (p/'runtime/evil.py').write_text('raise RuntimeError("executed submitted code")')
            (p/'point-ops').mkdir();(p/'point-ops/pathfinder_fps.so').write_bytes(b'not a shared library')
            with patch('model_package.verify'):
                result=stage_weights(p,root/'out')
            self.assertFalse(result['submitted_code_executed'])
            self.assertEqual(set(x.name for x in (root/'out').iterdir()),{'package.json','weights'})

    def test_bad_hash_symlink_and_ensemble_rejected(self):
        for defect in ('hash','symlink','ensemble','threshold'):
            with self.subTest(defect=defect),TemporaryDirectory() as d:
                root=Path(d);p=self.fixture(root);manifest=json.loads((p/'package.json').read_text())
                if defect=='hash':(p/'weights/ffn.pt').write_bytes(b'corrupt')
                if defect=='symlink':
                    (root/'outside').write_bytes(b'ffn');(p/'weights/ffn.pt').unlink();(p/'weights/ffn.pt').symlink_to(root/'outside')
                if defect=='ensemble':manifest['ffn_ensemble']={}
                if defect=='threshold':manifest['inference']['search_threshold']=.2
                (p/'package.json').write_text(json.dumps(manifest))
                with self.assertRaises(InvalidSubmission):stage_weights(p,root/'out')

    def test_timeout_terminates_worker(self):
        with TemporaryDirectory() as d:
            with self.assertRaises(subprocess.TimeoutExpired):
                run_process([sys.executable,'-c','import time; time.sleep(30)'],.1,Path(d)/'log')

    def test_namespace_excludes_suite_and_submission(self):
        cmd=namespace_command(Path('/workspace/research/run'),Path('/benchmark/runtime'),Path('/opt/pathfinder/limits.json'))
        self.assertNotIn('/benchmark/suite',cmd)
        self.assertNotIn('/workspace/submission',cmd)
        self.assertNotIn('--ro-bind-try',cmd)
        self.assertIn('--unshare-all',cmd)
        self.assertIn('--clearenv',cmd)
        # Stitching runs git in the read-only runtime checkouts; their safe.directory entries live here.
        if Path('/etc/gitconfig').exists():self.assertIn('/etc/gitconfig',cmd)

    @unittest.skipUnless(shutil.which('bwrap'),'bubblewrap unavailable')
    def test_real_namespace_hides_private_sentinel(self):
        with TemporaryDirectory() as d:
            base=Path(d);root=base/'work';root.mkdir()
            runtime=base/'runtime';runtime.mkdir();(runtime/'visible').write_text('runtime')
            limits=base/'limits.json';limits.write_text('{}')
            secret=base/'private-label';secret.write_text('not visible to inference')
            marker=base/'docker-marker';marker.touch()
            cmd=namespace_command(root,runtime,limits)
            for i,value in enumerate(cmd):
                if value=='--ro-bind' and cmd[i+1]=='/opt/pathfinder':cmd[i+1]=str(runtime)
                if value=='--ro-bind' and cmd[i+1]=='/.dockerenv':cmd[i+1]=str(marker)
            end=cmd.index('--chdir')+2
            probe="from pathlib import Path; assert Path('/runtime/visible').read_text()=='runtime'; assert not Path("+repr(str(secret))+").exists(); assert not Path('/benchmark/suite').exists(); assert not Path('/workspace/submission').exists()"
            result=subprocess.run(cmd[:end]+[sys.executable,'-c',probe],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_native_gpu_guard_has_no_environment_bypass(self):
        from execution_resources import device_guard
        with patch('execution_resources.MARKER',Path('/nonexistent-marker')),patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(RuntimeError,'Slurm'):device_guard('cuda')

    def test_atomic_publication_retains_previous_package(self):
        from publish_weights import exchange
        with TemporaryDirectory() as d:
            root=Path(d);old=root/'submission';new=root/'staged'
            old.mkdir();new.mkdir();(old/'id').write_text('old');(new/'id').write_text('new')
            exchange(new,old)
            self.assertEqual((old/'id').read_text(),'new')
            self.assertEqual((new/'id').read_text(),'old')

    def test_invalid_package_zero_and_infrastructure_no_reward(self):
        from verify_single import failure_result
        with TemporaryDirectory() as d:
            logs=Path(d)
            failure_result(logs,InvalidSubmission('bad weights'))
            self.assertEqual(json.loads((logs/'reward.json').read_text()),{'reward':0,'invalid_submission':1})
            failure_result(logs,RuntimeError('namespace failed'))
            self.assertFalse((logs/'reward.json').exists())

    def test_reward_holds_numbers_only(self):
        from verify_single import reward_metrics,METRICS
        method={key:1.0 for key in METRICS}|dict(protocol='local-funlib-erl/v1',inputs={})
        result=reward_metrics(.5,{'ffn':method,'disjoint-pair-shape':method})
        self.assertEqual(result['reward'],.5);self.assertEqual(result['disjoint_pair_shape_nerl'],1.0)
        self.assertTrue(all(type(v) in (int,float) for v in result.values()))

    def test_infrastructure_failure_never_emits_reward(self):
        from verify_single import main
        with TemporaryDirectory() as d:
            root=Path(d);logs=root/'logs'
            args=['verify',*[x for n in ('submission','image','runtime','suite','limits','root') for x in ('--'+n,str(root/n))],
                  '--logs',str(logs),'--seconds','1']
            with patch.object(sys,'argv',args),patch('execution_resources.container_execution',return_value=False):
                with self.assertRaises(RuntimeError):main()
            self.assertFalse((logs/'reward.json').exists())
            self.assertEqual(json.loads((logs/'status.json').read_text())['status'],'infrastructure_failure')



if __name__=='__main__':unittest.main()
