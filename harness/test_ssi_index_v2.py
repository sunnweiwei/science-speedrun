import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from ssi_index_v2 import task_score

class V2IndexTests(unittest.TestCase):
    def score(self,name,rewards,**files):
        with TemporaryDirectory() as d:
            p=Path(d);(p/'reward.json').write_text(json.dumps(rewards))
            for n,doc in files.items():(p/n).write_text(json.dumps(doc))
            return task_score(name,p)['value']

    def test_reduced_eeg_is_not_mistaken_for_a_complete_run(self):
        t={'tuev':.5, 'ad65':.6,'openbmi':.7,'longitudinal':.8,'faced':.9}
        a={'reward':.1,'tracks':['tuev','ad65','openbmi','longitudinal']}
        with self.assertRaises(ValueError):
            self.score('brain-computer-interface-decoding',t,**{'report.json':dict(scope='smoke',aggregate=a)})

    def test_a_low_valid_correlation_remains_negative(self):
        self.assertEqual(self.score('genome-sequence-to-function',{'reward':-.1}),-.1)

    def test_fly_uses_archived_display_and_keeps_distinct_subgrid_predictions(self):
        task='fruit-fly-neuron-reconstruction'
        for native in (.00001,.0001):
            self.assertEqual(self.score(task,dict(reward=native,disjoint_pair_shape_nerl=native)),native**.25)
        with self.assertRaises(ValueError):
            self.score(task,dict(reward=0.,disjoint_pair_shape_nerl=.0001,grid_control_nerl=.002))

    def test_massspec_rejects_a_single_call_with_an_impressive_hit_rate(self):
        with self.assertRaises((KeyError,ValueError)):
            self.score('mass-spectrometry-molecule-retrieval',{'reward':.91},**{'report.json':dict(status='complete',model_quality_measured=True,metrics={'hit_rate_at_1':.91})})

    def test_protein_design_requires_new_protocol_and_the_full_five_target_ppi_panel(self):
        tracks={'unconditional':20,'sm_binder':80,'ppi_binder':100,'enzyme':820}
        rewards={name+'_pass_rate':.1 for name in tracks}
        native={t:{'pass_rate':.1,'designs_requested':n} for t,n in tracks.items()}
        report={'tracks':native}
        with self.assertRaises(ValueError):
            self.score('protein-design',rewards,**{'report.json':report})
        report['ppi_inverse_folding']='proteinmpnn-fixed-target/v2'
        self.assertEqual(self.score('protein-design',rewards,**{'report.json':report}),.1)
        native['ppi_binder']['designs_requested']=2
        with self.assertRaises(ValueError):
            self.score('protein-design',rewards,**{'report.json':report})

    def test_crystal_invalid_submission_has_a_bound_v2_zero(self):
        report=dict(schema='crystalgen-report/v4-uma200-hull-aligned',protocol='crystalgen-evaluation/v4-uma200-hull-aligned',
                    primary_geometry='uma200_relaxed',pre_scoring_relaxation=True,error='InvalidSubmission: malformed output',
                    status='invalid_submission',evaluation_mode='sealed_replay_invalid_submission',requests_sha256='a'*64)
        self.assertEqual(self.score('crystal-structure-generation',{'reward':0.,'invalid_submission':1},**{'report.json':report}),0.)
        with self.assertRaises(ValueError):
            self.score('crystal-structure-generation',{'reward':0.,'invalid_submission':1},**{'report.json':dict(report,status='infrastructure_error')})

    def test_crystal_as_submitted_measurement_is_not_a_historical_uma_score(self):
        report=dict(schema='crystalgen-report/v3-submitted-geometry',protocol='crystalgen-evaluation/v3-submitted-geometry',
                    primary_geometry='submitted',pre_scoring_relaxation=False,requests_sha256='b'*64)
        with self.assertRaises(ValueError):
            self.score('crystal-structure-generation',{'reward':0.},**{'report.json':report})

if __name__=='__main__':unittest.main()
