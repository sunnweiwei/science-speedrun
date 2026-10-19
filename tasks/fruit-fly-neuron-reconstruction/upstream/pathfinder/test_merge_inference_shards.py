import copy
import unittest
import tempfile
import json
from pathlib import Path
from merge_inference_shards import merge_sense,merge_shape,merge_sense_stream,write_merge_json


def sense_fixture():
    results=[];receipts=[]
    for rank in range(2):
        common=dict(shards=2,shard_index=rank,ground_truth_used=False,paper_comparable=False)
        results.append(dict(common,schema='pathfinder-candidate-edges/v1',metadata={'shape':[10]*3},
            edges=[dict(source=1,target=2,sense=.7+rank*.1,evidence_seed_id=rank+1,center_zyx=[rank]*3)]))
        receipts.append(dict(common,segmentation_sha256='seg',image_manifest_sha256='raw',
            instance_index_sha256='index',assembled_receipt_sha256='assembly',coverage_sha256='coverage',
            checkpoint_sha256='weights',eligible_instance_ids=[1,2,3,4],
            skipped_geometric_context_instance_ids=[5],total_indexed_instances=5,input_shape=[128]*3,
            prompt_radius=16,shape_radius_nm=10000,min_size=500,threshold=.5,all_tiles_complete=False,
            assigned_instance_ids=[rank+1,rank+3],processed=[dict(instance_id=rank+1)],
            unprocessed_instance_ids=[rank+3],calls=1))
    return results,receipts


class MergeTests(unittest.TestCase):
    def test_mutually_consistent_wrong_inputs_rejected_by_external_provenance(self):
        outputs,receipts=sense_fixture()
        expected={'checkpoint_sha256':'weights','threshold':.5,'image_manifest_sha256':'raw'}
        merge_sense_stream(zip(receipts,outputs),2,expected_provenance=expected)
        for row in receipts:row['checkpoint_sha256']='same wrong checkpoint'
        with self.assertRaisesRegex(ValueError,'admitted provenance'):
            merge_sense_stream(zip(receipts,outputs),2,expected_provenance=expected)
        with self.assertRaisesRegex(ValueError,'provenance fields'):
            merge_sense_stream(zip(receipts,outputs),2,expected_provenance={'unrecognized':'value'})

    def test_strict_full_coverage_rejects_expiry_missing_flag_and_wrong_census(self):
        outputs,receipts=sense_fixture()
        for r in receipts:
            r['processed']=[dict(instance_id=i) for i in r['assigned_instance_ids']]
            r.update(unprocessed_instance_ids=[],calls=2,budget_exhausted=False)
        result=merge_sense_stream(zip(receipts,outputs),2,True,4)
        self.assertEqual(result['coverage']['processed_count'],4)
        for value in (True,None):
            bad=copy.deepcopy(receipts);bad[1]['budget_exhausted']=value
            with self.assertRaisesRegex(ValueError,'Incomplete SENSE'):
                merge_sense_stream(zip(bad,outputs),2,True,4)
        with self.assertRaisesRegex(ValueError,'admitted campaign'):
            merge_sense_stream(zip(receipts,outputs),2,True,5)
        partial_outputs,partial=sense_fixture()
        for r in partial:r['budget_exhausted']=False
        with self.assertRaisesRegex(ValueError,'Incomplete SENSE'):
            merge_sense_stream(zip(partial,partial_outputs),2,True,4)

    def test_incremental_json_exact_bytes_and_failure_does_not_publish(self):
        results,receipts=sense_fixture();value=merge_sense(results,receipts)
        value['unicode_test']='果蝇';value['special_test']='line\nquote"'
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'merged.json'
            write_merge_json(path,value)
            expected=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
            self.assertEqual(path.read_bytes(),expected)
            with self.assertRaises(ValueError):write_merge_json(path,dict(invalid=float('nan')))
            self.assertEqual(path.read_bytes(),expected)
            self.assertTrue(path.with_suffix('.json.tmp').exists())

    def test_streaming_matches_expected_coverage_and_tie_order(self):
        results,receipts=sense_fixture()
        results[1]['edges'][0]['sense']=results[0]['edges'][0]['sense']
        merged=merge_sense_stream(((r,o) for r,o in zip(receipts,results)),2)
        self.assertEqual(merged,merge_sense(results[::-1],receipts[::-1]))
        self.assertEqual(merged['edges'][0]['evidence_seed_id'],1)
        self.assertEqual(merged['coverage']['processed_count'],2)
        self.assertEqual(merged['coverage']['unprocessed_instance_ids'],[3,4])

    def test_streaming_rejects_missing_extra_reordered_and_changed_provenance(self):
        results,receipts=sense_fixture()
        pairs=list(zip(receipts,results))
        for rows,n in [(pairs[:1],2),(pairs+pairs[:1],2),(pairs[::-1],2),([],2)]:
            with self.assertRaises(ValueError):merge_sense_stream(iter(rows),n)
        for field,value in [('checkpoint_sha256','wrong'),('eligible_instance_ids',[1,2,4,3])]:
            bad=copy.deepcopy(receipts);bad[1][field]=value
            with self.assertRaises(ValueError):merge_sense_stream(zip(bad,results),2)

    def test_compact_exclusion_identity_and_count_are_preserved(self):
        results,receipts=sense_fixture()
        compact=dict(storage='ordered-excluded-id-digest/v1',count=1,sha256='a'*64,
                     encoding='ascending little-endian uint64',shape=[10]*3,voxel_size_nm=[16]*3,radius_nm=10000)
        for row in receipts:row['skipped_geometric_context_instance_ids']=copy.deepcopy(compact)
        self.assertEqual(merge_sense(results,receipts)['coverage']['skipped_geometric_context_instance_ids'],compact)
        for row in receipts:row['skipped_geometric_context_instance_ids']['count']=2
        with self.assertRaises(ValueError):merge_sense(results,receipts)

    def test_sense_max_evidence_preserves_center_and_partial_coverage(self):
        results,receipts=sense_fixture();r=merge_sense(results[::-1],receipts)
        self.assertEqual(len(r['edges']),1);self.assertEqual(r['edges'][0]['evidence_seed_id'],2)
        self.assertEqual(r['edges'][0]['center_zyx'],[1]*3)
        self.assertEqual(r['coverage']['unprocessed_instance_ids'],[3,4])
        self.assertEqual(r['coverage']['eligible_instance_fraction'],.5)
        self.assertEqual(r['coverage']['total_instance_fraction'],.4)

    def test_sense_rejects_missing_mismatched_or_misattributed_shards(self):
        results,receipts=sense_fixture()
        for mutation in ('missing','weights','coverage','seed'):
            a=copy.deepcopy(results);b=copy.deepcopy(receipts)
            if mutation=='missing':b.pop()
            if mutation=='weights':b[1]['checkpoint_sha256']='wrong'
            if mutation=='coverage':b[1]['unprocessed_instance_ids']=[]
            if mutation=='seed':a[1]['edges'][0]['evidence_seed_id']=4
            with self.assertRaises(ValueError):merge_sense(a,b)

    def test_shape_prefix_coverage_and_missing_motif(self):
        candidates=dict(edges=[dict(source=1,target=2,sense=.8),dict(source=2,target=3,sense=.9)])
        base=dict(shards=2,ground_truth_used=False,paper_comparable=False,
            segments_sha256='seg',edges_sha256='edges',checkpoint_sha256='weights',voxel_size_nm=[16]*3,
            origin_nm=[0]*3,axes='zyx',radius_um=10,points=8192,motif_limit=100,truncated=False,
            allow_partial_context=False,total_enumerated_motifs=3,index_sha256='index',coverage_sha256='coverage',
            max_nodes=4,seed=42)
        rows=[dict(base,shard_index=0,motifs=[dict(nodes=[1,2],shape=.9,context_complete=True,prediction_context_complete=True)]),
              dict(base,shard_index=1,motifs=[dict(nodes=[2,3],shape=None,context_complete=True,prediction_context_complete=False)])]
        r=merge_shape(rows,candidates)
        self.assertEqual(r['unprocessed_motifs'],[[1,2,3]])
        self.assertEqual(len(r['edges']),1);self.assertEqual(r['full_context_scored'],1)
        rows[0]['motifs'][0]['prediction_context_complete']=False
        with self.assertRaises(ValueError):merge_shape(rows,candidates)
        rows[0]['motifs'][0]['nodes']=[1,2,3]
        with self.assertRaises(ValueError):merge_shape(rows,candidates)

if __name__=='__main__':unittest.main()
