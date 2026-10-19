import unittest
from domain_comparison import summarize, resource_costs
import json,tempfile
from pathlib import Path
import numpy as np
from dataset_contract import sha256
from domain_comparison import reconstruction_coverage


class CoverageEvidenceTest(unittest.TestCase):
    def fixture(self,root,complete):
        tile=dict(index=0,start=[16]*3,stop=[18]*3)
        metadata=dict(shape=[2]*3)
        shard=dict(roi=[16,18]*3,tile_size=2,id_stride=9,shards=1,shard_index=0,
            checkpoint_sha256='x',raw_sha256='y',metadata=metadata,ground_truth_used=False,
            tile_results=[dict(tile,budget_exhausted=not complete)])
        source=root/'shard.json';source.write_text(json.dumps(shard))
        np.save(root/'segmentation.npy',np.zeros((2,2,2),np.uint32))
        receipt=dict(inputs=[dict(path=str(source),sha256=sha256(source))],metadata=metadata,
            segmentation_sha256=sha256(root/'segmentation.npy'),all_tiles_complete=complete)
        (root/'receipt.json').write_text(json.dumps(receipt));(root/'metadata.json').write_text(json.dumps(metadata))
        coverage=dict(tiles=[dict(tile,status='complete' if complete else 'partial')],total_tiles=1,
            complete_tiles=int(complete),all_tiles_complete=complete)
        (root/'coverage.json').write_text(json.dumps(coverage))

    def test_partial_cannot_be_promoted_by_summary_flags(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root,False)
            self.assertFalse(reconstruction_coverage(root)['full_context_reconstruction_complete'])
            p=root/'coverage.json';j=json.loads(p.read_text());j['all_tiles_complete']=True;p.write_text(json.dumps(j))
            with self.assertRaisesRegex(ValueError,'summary'):reconstruction_coverage(root)

    def test_complete_coverage_requires_unchanged_prediction(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root,True)
            self.assertTrue(reconstruction_coverage(root)['full_context_reconstruction_complete'])
            with (root/'segmentation.npy').open('ab') as f:f.write(b'changed')
            with self.assertRaisesRegex(ValueError,'prediction'):reconstruction_coverage(root)


def metric(value):
    return dict(nerl=value,erl_nm=value*100,split_errors=2,merge_errors=1,omitted_node_fraction=0.,
        adapter_sha256='a',core_evaluator_adapter_sha256='c',topology_sha256='t',suite_sha256='s',
        inputs=dict(skeletons='k',metadata='m',segmentation=str(value)),evaluator_sha256='e',
        protocol='test',boundary_margin_voxels=[0]*3,maximum_erl_nm=100,total_path_length_nm=1000)


class DomainReportTest(unittest.TestCase):
    def test_negative_gain_survives_failed_separate_native_check(self):
        record=dict(region='full-core',ffn=metric(.1),baseline=metric(.3),full=metric(.2))
        out=summarize({'full-core':'s'},[record],{'investigation_required':True})
        self.assertEqual(out['primary']['direction'],'worse')
        self.assertAlmostEqual(out['primary']['relative_nerl_change'],-1/3)
        self.assertIsNone(out['direct_pathfinder_paper_gap'])

    def test_missing_region_not_silently_completed(self):
        out=summarize({'full-core':'s'},[],{})
        self.assertFalse(out['all_regions_scored']);self.assertIsNone(out['primary'])

    def test_mixed_reference_rejected(self):
        record=dict(region='full-core',ffn=metric(.1),baseline=metric(.3),full=metric(.2))
        record['full']['inputs']['skeletons']='other'
        with self.assertRaises(ValueError):summarize({'full-core':'s'},[record],{})

    def test_joint_allocation_not_double_charged_and_pending_not_free(self):
        ledger=dict(jobs=[dict(job_id='1',gpus=4,reserved_seconds=7200,actual_seconds=3600),
                          dict(job_id='2',gpus=1,reserved_seconds=1800)],bytes_on_disk=123)
        out=resource_costs(ledger,{'1':['ffn','full'],'2':['baseline']})
        self.assertEqual(out['charged_gpu_hours'],4.5)
        self.assertEqual(out['charged_allocation_hours'],1.5)
        self.assertFalse(out['accounting_complete'])
        self.assertIsNone(out['measured_peak_disk_bytes'])

    def test_missing_job_cannot_silently_reduce_cost(self):
        with self.assertRaises(ValueError):resource_costs({'jobs':[]},{'missing':['test']})


if __name__=='__main__':unittest.main()
