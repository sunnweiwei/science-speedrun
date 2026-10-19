"""Exercise full orchestration and rejection of incomplete workers without GPUs."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from test_fs_support import TemporaryDirectory
import single_pipeline as pipeline


class PipelineTests(unittest.TestCase):
    def exercise(self,partial=False,backend="norm32"):
        with TemporaryDirectory() as d:
            root=Path(d);(root/'image').mkdir();(root/'model/weights').mkdir(parents=True)
            def write(path,data):
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data))
            write(root/'image/manifest.json',dict(shape=[1794]*3,origin_nm=[0]*3))
            write(root/'model/package.json',{})
            (root/'model/weights/sense.pt').write_bytes(b'sense')
            calls=[]
            class Process:
                pid=999999999
                def __init__(self,cmd,**kwargs):
                    script=next(x for x in cmd if str(x).endswith('.py'));name=Path(script).name;calls.append(cmd)
                    def val(flag):return cmd[cmd.index('--'+flag)+1]
                    out=root/'prediction'
                    if name=='stitch_ffn.py':
                        (out/'stitched').mkdir();(out/'stitched/segmentation.npy').write_bytes(b'prediction')
                    if name=='prepare_prediction_context.py':
                        for n in ('receipt.json','instance-index.json','coverage.json'):write(out/'context'/n,{})
                        write(out/'context/metadata.json',dict(voxel_size_nm=[16]*3,origin_nm=[0]*3))
                    if name=='shared_proposals.py':write(Path(val('output')),dict(decisions=[{}]*5))
                    if name=='infer_shape.py':
                        rank=int(val('shard-index'))
                        write(Path(val('output')),dict(budget_exhausted=partial and rank==1,truncated=False,motifs=[{}]*len(range(rank,5,int(val('shards'))))))
                    if name=='merge_inference_shards.py' and val('stage')=='shape':
                        write(Path(val('output')),dict(truncated=False,unprocessed_motifs=[]))
                    if name=='finalize_prediction.py':
                        p=Path(val('output'));p.mkdir();(p/'segmentation.npy').write_bytes(b'final')
                def wait(self):return 0
            table=SimpleNamespace(_mmap=SimpleNamespace(close=lambda:None))
            index=dict(records=SimpleNamespace(table=table),shape=[1762]*3)
            limits=dict(ffn_seconds=3600,sense_seconds=3600,baseline_seconds=3600,shape_seconds=3600,
                        memory_gib=192,max_instances=60000000,numeric_max_label=60000000,window_bytes=600000000)
            with patch('model_package.verify',return_value=dict(search_threshold=.8,package_manifest_sha256='model')),\
                 patch.object(pipeline,'allocation'),patch.object(pipeline.subprocess,'Popen',Process),\
                 patch.object(pipeline,'start_mps',return_value=lambda:None),\
                 patch('prepare_prediction_context.audit',return_value=dict(segmentation_sha256='base')),\
                 patch('index_storage.load_index',return_value=index),\
                 patch('index_storage.select_table_centers',return_value=([1,2,3,4,5],)):
                if partial:
                    with self.assertRaisesRegex(RuntimeError,'Incomplete SHAPE'):pipeline.run(root,root/'runtime',limits,ffn_inference_backend=backend)
                    self.assertFalse((root/'prediction/submission').exists())
                else:
                    pipeline.run(root,root/'runtime',limits,ffn_inference_backend=backend)
                    result=json.loads((root/'prediction/execution.json').read_text())
                    self.assertTrue(result['complete']);self.assertEqual(result['shared_candidates'],5)
                    self.assertEqual(result['ffn_inference_backend'],backend)
                    self.assertFalse(result['consensus']);self.assertEqual(result['checkpoint_count'],1)
            for cmd in calls:
                self.assertNotIn('--suite',cmd)
                if any(str(x).endswith('/infer_shape.py') for x in cmd):
                    self.assertEqual(cmd[cmd.index('--max-nodes')+1],'2')
                if any(str(x).endswith('/launch_overlap.py') for x in cmd):
                    self.assertEqual(cmd[cmd.index('--seed-order')+1],'ascending')
                    self.assertEqual(cmd[cmd.index('--inference-backend')+1],backend)
                    self.assertEqual(cmd[cmd.index('--workers-per-gpu')+1],'4')

    def test_complete_route(self):self.exercise()
    def test_explicit_eager_route(self):self.exercise(backend="eager")
    def test_incomplete_shape_never_finalizes(self):self.exercise(partial=True)

if __name__=='__main__':unittest.main()
