"""Review-only scoring invariants, including a cross-bin duplicate fixture."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from scoring.experimental_v2 import global_unique_low_energy_score, negative_formation_low_energy_score
from scoring.upstream import normalize_result


def record(scores,loaded=2,valid=2):
    return {'run_info':{'n_structures':loaded},'validity_filtering':{'valid_structures':valid},
            'results':{'sun':'BenchmarkResult(final_scores='+repr(scores)+')'}}


class CandidateScoreTests(unittest.TestCase):
    def test_equivalent_materials_in_different_energy_bins_count_once(self):
        # The upstream global-uniqueness pass identifies one equivalence class;
        # the legacy per-bin sums report one stable + one metastable hit.
        r=record({'total_structures_evaluated':2,'sun_count':1,'msun_count':1,
                  'metasun_count':1,'metasun_structure_indices':[0]})
        self.assertEqual(global_unique_low_energy_score(r,2)['reward'],.5)

    def test_invalid_attempts_remain_in_denominator(self):
        r=record({'total_structures_evaluated':2,'metasun_count':2,'metasun_structure_indices':[0,1]})
        self.assertEqual(global_unique_low_energy_score(r,4)['reward'],.5)

    def test_no_silent_fallback_to_legacy_sum(self):
        with self.assertRaisesRegex(ValueError,'globally'):
            global_unique_low_energy_score(record({'sun_count':1,'msun_count':1}),2)

    def test_duplicate_or_out_of_range_indices_fail(self):
        for indices in [[0,0],[0,2],[0,True]]:
            with self.subTest(indices=indices),self.assertRaises(ValueError):
                global_unique_low_energy_score(record({'total_structures_evaluated':2,'metasun_count':2,
                                                       'metasun_structure_indices':indices}),2)

    def test_infrastructure_missing_output_is_not_zero(self):
        with self.assertRaises(ValueError):global_unique_low_energy_score(record({}),2)
        self.assertEqual(global_unique_low_energy_score(record({},loaded=0,valid=0),2)['reward'],0.)

    def test_nonfinite_counts_and_upstream_alias(self):
        for bad in [float('inf'),float('nan'),True,-1]:
            with self.subTest(count=bad),self.assertRaises(ValueError):
                global_unique_low_energy_score(record({}),bad)
        r=record({'total_structures_evaluated':2,'metasun_count':1,'metasun_structure_indices':[0]})
        r['results']['sun_new']=r['results'].pop('sun')
        self.assertEqual(global_unique_low_energy_score(r,2)['reward'],.5)


def stability(energies):
    rows=', '.join("{'value': %r, 'std': 0.01}" % e for e in energies)
    return ("BenchmarkResult(evaluator_results={'formation_energy': {'combined_value': np.float64(0.1), "
            "'metric_results': {'formation_energy': MetricResult(metrics={'n': 1}, individual_values=[%s])}}})" % rows)


class NegativeFormationTests(unittest.TestCase):
    def doc(self, energies, indices):
        r=record({'total_structures_evaluated':len(energies),'metasun_count':len(indices),
                  'metasun_structure_indices':indices},loaded=len(energies),valid=len(energies))
        r['results']['stability']=stability(energies)
        return r

    def test_positive_formation_energy_is_not_counted(self):
        # Structures above the elemental reference pass the 0.1 eV/atom hull window in flat-hull systems.
        result=negative_formation_low_energy_score(self.doc([-0.5,0.2,-0.1,0.05],[0,1,2,3]),8)
        self.assertEqual((result['accepted'],result['unique_novel_low_energy'],result['reward']),(2,4,.25))

    def test_only_accepted_structures_need_finite_energy(self):
        self.assertEqual(negative_formation_low_energy_score(self.doc([-0.5,float('nan')],[0]),2)['reward'],.5)
        with self.assertRaises(ValueError):negative_formation_low_energy_score(self.doc([-0.5,float('nan')],[0,1]),2)

    def test_missing_or_misaligned_energies_fail(self):
        r=self.doc([-0.5,-0.4],[0,1]);del r['results']['stability']
        with self.assertRaisesRegex(ValueError,'stability'):negative_formation_low_energy_score(r,2)
        r=self.doc([-0.5,-0.4],[0,1]);r['results']['stability']=stability([-0.5])
        with self.assertRaisesRegex(ValueError,'cover'):negative_formation_low_energy_score(r,2)

    def test_absent_counts_are_not_zero_valid(self):
        for document in ({},{'run_info':{'n_structures':2}}):
            with self.subTest(document=document):
                with self.assertRaises(ValueError):negative_formation_low_energy_score(document,2)
                with self.assertRaises(ValueError):global_unique_low_energy_score(document,2)
                with self.assertRaises(RuntimeError):normalize_result(document,2)

    def test_zero_valid_with_accepted_counts_fails(self):
        document={'run_info':{'n_structures':2},'validity_filtering':{'valid_structures':0},
                  'results':{'sun':"BenchmarkResult(final_scores={'sun_count': 5})"}}
        with self.assertRaises(RuntimeError):normalize_result(document,2)

    def test_measured_zero_valid_scores_zero(self):
        document={'run_info':{'n_structures':2},'validity_filtering':{'valid_structures':0},'results':{}}
        self.assertEqual(normalize_result(document,2)[0]['sun_metasun'],0.)
        self.assertEqual(negative_formation_low_energy_score(document,2)['reward'],0.)


if __name__=='__main__':unittest.main()
