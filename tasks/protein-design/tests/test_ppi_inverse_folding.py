"""Exercise the actual MPNN invocation and FASTA parser at the binder/target boundary."""
from pathlib import Path
from subprocess import CompletedProcess
from unittest import TestCase, mock

from pipeline import DesignabilityScorer, ScorerUnavailable
from test_task import build_structure, helix


class TargetConditioningTests(TestCase):
    def setUp(self):
        with mock.patch('pipeline.verify_assets',return_value={'ligandmpnn':'/runner','ligandmpnn_params':'/weights'}):
            self.scorer=DesignabilityScorer(devices=['0'])
        self.binder=build_structure(helix(7),['ALA']*7)
        target=build_structure(helix(5,origin=(5,8,3)),['GLY']*5)
        self.context=dict(target_coords=target[0],target_mask=target[1],target_restypes=target[2])

    def run_with_records(self,records):
        def fake(command,**kwargs):
            def arg(name):return command[command.index(name)+1]
            self.assertEqual(arg('--chains_to_design'),'A')
            self.assertEqual(arg('--fasta_seq_separation'),'/')
            self.assertEqual(arg('--model_type'),'protein_mpnn')
            self.assertEqual(arg('--number_of_batches'),'4')
            lines=Path(arg('--pdb_path')).read_text().splitlines()
            atoms=[line for line in lines if line.startswith('ATOM')]
            self.assertEqual({line[21] for line in atoms},{'A','B'})
            self.assertEqual(len(set(line[6:11] for line in atoms)),len(atoms))
            target=[line for line in atoms if line[21]=='B' and line[12:16].strip()=='CA']
            self.assertEqual(len(target),5)
            self.assertTrue(all(line[17:20]=='GLY' for line in target))
            for line,xyz in zip(target,self.context['target_coords'][:,1],strict=True):
                for field,value in zip((line[30:38],line[38:46],line[46:54]),xyz,strict=True):
                    self.assertAlmostEqual(float(field),value,delta=.00051)
            output=Path(arg('--out_folder'))/'seqs'
            output.mkdir()
            (output/'design.fa').write_text(''.join(f'>sample{i}\n{s}\n' for i,s in enumerate(records)))
            return CompletedProcess(command,0,'','')
        with mock.patch('pipeline.subprocess.run',side_effect=fake):
            return self.scorer.design_sequences('ppi_binder',self.binder,self.context,None)

    def test_only_designed_binder_is_passed_to_refolding(self):
        seqs=['CCCCCCC','DDDDDDD','EEEEEEE','FFFFFFF']
        self.assertEqual(self.run_with_records([s+'/GGGGG' for s in ['AAAAAAA',*seqs]]),seqs)

    def test_incomplete_or_redesigned_target_output_is_a_scorer_error(self):
        for records in (['AAAAAAA/GGGGG'], ['AAAAAAA/GGGGG']+['CCCCCCC']*4,
                        ['AAAAAAA/GGGGG']+['CCCCCCC/EEEEE']*4):
            with self.subTest(records=records),self.assertRaises(ScorerUnavailable):
                self.run_with_records(records)

    def test_missing_target_cannot_be_scored_as_a_binder_only_design(self):
        with self.assertRaises(ScorerUnavailable):
            self.scorer.design_sequences('ppi_binder',self.binder,{},None)


    def test_rf3_empty_output_is_an_infrastructure_failure(self):
        self.scorer.paths["rf3_ckpt"]="/weights/predictor"
        with mock.patch("pipeline.subprocess.run",return_value=CompletedProcess([],0,"","")):
            with self.assertRaisesRegex(ScorerUnavailable,"RF3 produced 0 of 1"):
                self.scorer.predict_all("design",["AAAAAAA"],self.context,[0])

    def test_missing_interface_confidence_does_not_become_zero(self):
        model={"backbone":[{"atoms":{"CA":v}} for v in self.binder[0][:,1]],
               "iptm":None,"interface_pae":None}
        with mock.patch.object(self.scorer,"design_sequences",return_value=["AAAAAAA"]), \
             mock.patch.object(self.scorer,"predict_all",return_value=[[model]]):
            with self.assertRaisesRegex(ScorerUnavailable,"interface confidence"):
                self.scorer.evaluate("ppi_binder",self.binder,self.context)
