import subprocess
import sys
import argparse
import json
import os
from pathlib import Path
import unittest
from unittest.mock import MagicMock,patch
from launch_overlap import run_workers,run
from common import sha256
from test_stitch_ffn import fixture,xtemp_directory


class LaunchOverlapTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('PATHFINDER_TEST_XTEMP'),'Xtemp test directory required')
    def test_join_cli_imports_without_inherited_pythonpath(self):
        with xtemp_directory() as folder:
            env=dict(os.environ)
            env.pop('PYTHONPATH',None)
            result=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve().with_name('join_overlap_groups.py')),'--help'],
                cwd=folder,env=env,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('--groups',result.stdout)

    @unittest.skipUnless(os.environ.get('PATHFINDER_TEST_XTEMP'),'Xtemp test directory required')
    def test_two_group_controller_join_rejects_partial_duplicate_and_changed_receipts(self):
        from window_assignment import assign
        from join_overlap_groups import collect
        with xtemp_directory() as folder:
            root=Path(folder);manifest=fixture(root);checkpoint=root/'checkpoint.pt';checkpoint.write_bytes(b'synthetic model')
            manifest['checkpoint_sha256']=sha256(checkpoint)
            for row in manifest['windows']:row['checkpoint_sha256']=sha256(checkpoint)
            (root/'budget.json').write_text(json.dumps({'jobs':[{'job_id':str(i),'gpus':4,'reserved_seconds':1000} for i in (1,2)]}))
            owners=assign(manifest['windows'],8,'voxel-balanced-v1');groups=[]
            def fake_workers(commands,masks,out,timeout):
                source=json.loads((out/'start.json').read_text())['source_sha256']
                for command in commands:
                    rank=int(command[command.index('--shard-index')+1]);destination=Path(command[command.index('--output')+1]);destination.mkdir()
                    row=dict(manifest,source_sha256=source,shards=8,shard_index=rank,complete=True,
                        assignment_policy='voxel-balanced-v1',job_id=os.environ['SLURM_JOB_ID'],
                        requested_inference_backend='norm32',inference_backend='norm32',
                        windows=[r for r in manifest['windows'] if owners[r['index']]==rank])
                    (destination/'receipt.json').write_text(json.dumps(row))
                return [dict(rank=i,pid=900000+i,returncode=0,command=c) for i,c in enumerate(commands)]
            for i in range(2):
                out=root/f'group-{i}';groups.append(out)
                args=argparse.Namespace(root=str(root),output=str(out),image_manifest=manifest['image_manifest'],checkpoint=str(checkpoint),
                    roi=manifest['roi'],core_size=4,margin=2,batch=2,seconds=300,max_calls=10,seed_order='ascending',
                    total_shards=8,shard_offset=i*4,assignment_policy='voxel-balanced-v1')
                with patch.dict(os.environ,SLURM_JOB_ID=str(i+1),SLURM_JOB_PARTITION='gpu',CUDA_VISIBLE_DEVICES='0,1,2,3'),patch('launch_overlap.run_workers',side_effect=fake_workers):run(args)
                self.assertTrue((out/'group-complete.json').is_file())
                self.assertFalse((out/'manifest.json').exists());self.assertFalse((out/'complete.json').exists())
            result=collect(root,list(reversed(groups)))
            self.assertEqual(len(result['windows']),8);self.assertEqual(len(result['shard_receipts']),8)
            for incomplete in (groups[:1],[groups[0],groups[0]]):
                with self.assertRaisesRegex(ValueError,'distinct'):collect(root,incomplete)
            changed=groups[0]/'shard-0/receipt.json';changed.write_text(changed.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'Changed shard'):collect(root,groups)

    @unittest.skipUnless(os.environ.get('PATHFINDER_TEST_XTEMP'),'Xtemp test directory required')
    def test_controller_completion_requires_complete_matching_worker_receipts(self):
        # Exercise real controller I/O, source hashing and merge validation;
        # only the GPU subprocesses are replaced by synthetic worker receipts.
        for defect in (None,'missing-window','changed-source','changed-boundary','changed-backend'):
            with self.subTest(defect=defect),xtemp_directory() as folder:
                root=Path(folder);manifest=fixture(root)
                checkpoint=root/'checkpoint.pt';checkpoint.write_bytes(b'synthetic model')
                model_hash=sha256(checkpoint);manifest['checkpoint_sha256']=model_hash
                for record in manifest['windows']:record['checkpoint_sha256']=model_hash
                (root/'budget.json').write_text(json.dumps({'jobs':[{'job_id':'test','gpus':4,'reserved_seconds':1000}]}))
                output=root/'inference'
                args=argparse.Namespace(root=str(root),output=str(output),image_manifest=manifest['image_manifest'],
                    checkpoint=str(checkpoint),roi=manifest['roi'],core_size=4,margin=2,batch=2,seconds=300,max_calls=10,seed_order='ascending',inference_backend='norm32-graph')
                def fake_workers(commands,masks,out,timeout):
                    source=json.loads((out/'start.json').read_text())['source_sha256']
                    for rank,command in enumerate(commands):
                        destination=Path(command[command.index('--output')+1]);destination.mkdir()
                        self.assertEqual(destination,out/f'shard-{rank}')
                        self.assertEqual(command[command.index('--inference-backend')+1],'norm32-graph')
                        shard=dict(manifest,source_sha256=dict(source),shards=4,shard_index=rank,complete=True,
                            inference_backend='norm32-graph',
                            windows=[r for r in manifest['windows'] if r['index']%4==rank])
                        if rank==0 and defect=='missing-window':shard['windows'].pop()
                        if defect=='changed-source':shard['source_sha256']['invented.py']='wrong'
                        if defect=='changed-boundary':shard['boundary_image_shape']=[64]*3
                        if defect=='changed-backend':shard['inference_backend']='eager'
                        (destination/'receipt.json').write_text(json.dumps(shard))
                    return [dict(rank=i,pid=900000+i,returncode=0,command=c) for i,c in enumerate(commands)]
                with patch.dict(os.environ,SLURM_JOB_ID='test',SLURM_JOB_PARTITION='gpu',CUDA_VISIBLE_DEVICES='0,1,2,3'),patch('launch_overlap.run_workers',side_effect=fake_workers):
                    if defect:
                        with self.assertRaises(ValueError):run(args)
                        self.assertFalse((output/'complete.json').exists())
                        self.assertFalse((output/'manifest.json').exists())
                    else:
                        run(args)
                        complete=json.loads((output/'complete.json').read_text())
                        self.assertTrue(complete['complete'])
                        self.assertEqual(complete['manifest_sha256'],sha256(output/'manifest.json'))
                        final=json.loads((output/'manifest.json').read_text())
                        self.assertEqual(len(final['windows']),8)
                        self.assertEqual(len(final['shard_receipts']),4)

    @unittest.skipUnless(os.environ.get('PATHFINDER_TEST_XTEMP'),'Xtemp test directory required')
    def test_gpu_isolation_and_checked_processes(self):
        commands=[[sys.executable,'-c','import os; print(os.environ["CUDA_VISIBLE_DEVICES"])'] for _ in range(4)]
        masks=['GPU-a','GPU-b','GPU-c','GPU-d']
        with xtemp_directory() as folder:
            output=Path(folder);rows=run_workers(commands,masks,output,10)
            self.assertEqual(len({r['pid'] for r in rows}),4)
            for i,row in enumerate(rows):
                self.assertEqual(row['rank'],i);self.assertEqual(row['returncode'],0)
                self.assertEqual((output/f'worker-{i}.log').read_text().strip(),masks[i])
                with self.assertRaises(ProcessLookupError):os.kill(row['pid'],0)

    @unittest.skipUnless(os.environ.get('PATHFINDER_TEST_XTEMP'),'Xtemp test directory required')
    def test_worker_failure_propagates(self):
        with xtemp_directory() as folder:
            with self.assertRaises(subprocess.CalledProcessError):
                run_workers([[sys.executable,'-c','raise SystemExit(7)']]*4,['0','1','2','3'],Path(folder),10)

    @unittest.skipUnless(os.environ.get('PATHFINDER_TEST_XTEMP'),'Xtemp test directory required')
    def test_timeout_kills_and_joins_every_worker(self):
        children=[];original=subprocess.Popen
        def spawn(*args,**kwargs):
            child=original(*args,**kwargs);children.append(child);return child
        with xtemp_directory() as folder,patch('launch_overlap.subprocess.Popen',side_effect=spawn):
            with self.assertRaises(subprocess.TimeoutExpired):
                run_workers([[sys.executable,'-c','import time; time.sleep(30)']]*4,['0','1','2','3'],Path(folder),.05)
        self.assertEqual(len(children),4)
        for child in children:
            self.assertIsNotNone(child.returncode)
            with self.assertRaises(ProcessLookupError):os.kill(child.pid,0)

    def test_reject_missing_devices_before_launch(self):
        # Several workers may share one GPU (MPS); every worker still needs a device.
        for masks in ([],['0','1','2'],['0','1','2',''],['0','0','1',' ']):
            with patch('launch_overlap.subprocess.Popen') as spawn:
                with self.assertRaises(ValueError):run_workers([['worker']]*4,masks,Path('/unused'),300)
                spawn.assert_not_called()

if __name__=='__main__':unittest.main()
