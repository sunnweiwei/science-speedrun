import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from isolate_inference import invocation
from test_fs_support import TemporaryDirectory


class IsolationTests(unittest.TestCase):
    def fixture(self, base):
        base=Path(base);script=base/'model.py';script.write_text('print(1)\n')
        image=base/'runtime.sif';image.write_bytes(b'unit-test-placeholder')
        output=base/'output';output.mkdir()
        return script,image,output

    def test_narrow_mounts_and_environment(self):
        with TemporaryDirectory() as temp:
            script,image,output=self.fixture(temp)
            with patch.dict(os.environ,{'APPTAINER_BIND':'/:/host','APPTAINERENV_SECRET':'hidden'}):
                args,env=invocation(['python',str(script)],image,[script],output)
            self.assertIn(f'{script}:{script}:ro',args)
            self.assertIn(f'{output}:{output}:rw',args)
            self.assertNotIn('--nv',args)
            self.assertNotIn('APPTAINER_BIND',env)
            self.assertNotIn('APPTAINERENV_SECRET',env)
            self.assertEqual(args[-3:],[str(image),'python3',str(script)])

    def test_refuses_mount_overlap_and_unallocated_gpu(self):
        with TemporaryDirectory() as temp:
            script,image,output=self.fixture(temp)
            with self.assertRaises(ValueError):
                invocation(['python',str(script)],image,[Path(temp)],output)
            with patch.dict(os.environ,{'SLURM_JOB_ID':'123','SLURM_JOB_PARTITION':'gpu',
                                         'CUDA_VISIBLE_DEVICES':'0,1,2,3'}):
                with self.assertRaises(ValueError):
                    invocation(['python',str(script)],image,[script],output,gpu='4')
                args,_=invocation(['python',str(script)],image,[script],output,gpu='2')
                self.assertIn('--nv',args);self.assertIn('CUDA_VISIBLE_DEVICES=2',args)

    def test_configured_partition_reaches_clean_container_environment(self):
        with TemporaryDirectory() as temp:
            script,image,output=self.fixture(temp)
            with patch.dict(os.environ, {'SLURM_JOB_ID':'example',
                    'SLURM_JOB_PARTITION':'local-gpu', 'PATHFINDER_GPU_PARTITION':'local-gpu',
                    'CUDA_VISIBLE_DEVICES':'0'}):
                args,_=invocation(['python',str(script)],image,[script],output,gpu='0')
            self.assertIn('SLURM_JOB_PARTITION=local-gpu',args)
            self.assertIn('PATHFINDER_GPU_PARTITION=local-gpu',args)

    @unittest.skipUnless(os.environ.get('PATHFINDER_TEST_IMAGE'), 'Explicit Apptainer image needed for host CPU isolation test')
    def test_actual_container_cannot_read_sibling_gt(self):
        with TemporaryDirectory() as temp:
            script,_,output=self.fixture(temp)
            secret=Path(temp)/'private-ground-truth.txt';secret.write_text('not an inference input')
            data=Path(temp)/'em-input.txt';data.write_text('image-only fixture')
            script.write_text('from pathlib import Path\n'
                +f'assert not Path({str(secret)!r}).exists()\n'
                +f'assert Path({str(data)!r}).read_text()=="image-only fixture"\n'
                +f'Path({str(output/"done.txt")!r}).write_text("isolated")\n')
            args,env=invocation(['python',str(script)],Path(os.environ['PATHFINDER_TEST_IMAGE']),[script,data],output)
            result=subprocess.run(args,env=env,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
            self.assertEqual((output/'done.txt').read_text(),'isolated')


if __name__=='__main__':unittest.main()
