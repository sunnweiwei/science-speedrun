import json,os,unittest
from pathlib import Path
from test_fs_support import TemporaryDirectory
from score_stream import ScoreWriter,load_shard
from merge_ffn_agglomeration import combine

class StreamTests(unittest.TestCase):
    def test_scores_round_trip_and_merge_exactly(self):
        with TemporaryDirectory(dir=os.environ['PATHFINDER_TEST_XTEMP']) as tmp:
            tmp=Path(tmp);path=tmp/'shard.json';writer=ScoreWriter(tmp/'scores.jsonl')
            row=dict(pair=[1,2],complete=True,accepted=True,iou=.95,
                     consistency=[[.95]*2]*2,prediction_voxels=[100,100],deleted_voxels=[0,0])
            shard=dict(shards=1,shard_index=0,checkpoint_sha256='c',raw_sha256='r',base_sha256='b',decisions_sha256='d',ground_truth_used=False)
            writer.append(row);descriptor=writer.finish();path.write_text(json.dumps(dict(shard,result_stream=descriptor)))
            loaded=load_shard(path);self.assertEqual(list(loaded['results']),[row]);self.assertEqual(len(loaded['results']),1)
            plan=dict(base_sha256='b',decisions=[dict(pair=[1,2])])
            self.assertEqual(combine(plan,[loaded],'d'),combine(plan,[dict(shard,results=[row])],'d'))
            descriptor['count']=2;path.write_text(json.dumps(dict(shard,result_stream=descriptor)))
            with self.assertRaisesRegex(ValueError,'Incomplete'):list(load_shard(path)['results'])
            with (tmp/'scores.jsonl').open('a') as f:f.write('{}\n')
            with self.assertRaisesRegex(ValueError,'identity'):load_shard(path)

    def test_unfinished_writer_never_publishes_final_file(self):
        with TemporaryDirectory(dir=os.environ['PATHFINDER_TEST_XTEMP']) as tmp:
            path=Path(tmp)/'scores.jsonl';writer=ScoreWriter(path)
            writer.append(dict(pair=[1,2]));writer.close()
            self.assertFalse(path.exists());self.assertTrue(writer.partial.exists())

if __name__=='__main__':unittest.main()
