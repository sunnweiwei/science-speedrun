"""Unit tests of explicit loader correction, not CFD performance evidence."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'environment/data'))
from native_resource_bindings import bind_nozzle_restart


class NativeResourceBindingsTests(unittest.TestCase):
    def test_only_missing_restart_switch_added_without_source_mutation(self):
        original={'domain':{'x':{'cells':500}},'initial_condition':{'levelset':'placeholder'},
                  'boundary_conditions':{'unchanged':True}}
        before=copy.deepcopy(original)
        for name in ['Nozzle2D_fine','Nozzle3D_coarse']:
            result=bind_nozzle_restart(name,original,'/data/'+name+'/restart.h5')
            self.assertEqual(result.pop('restart'),{'is_restart':True,'file_path':'/data/'+name+'/restart.h5'})
            self.assertEqual(result,original)
        self.assertEqual(original,before)

    def test_existing_choice_and_unreviewed_cases_are_rejected(self):
        for case,setup in [('Nozzle2D_coarse',{}),('Nozzle3D_fine',{}),
                           ('Nozzle2D_fine',{'restart':{'is_restart':False}})]:
            with self.subTest(case=case),self.assertRaises(ValueError):
                bind_nozzle_restart(case,setup,'/data/restart.h5')


if __name__=='__main__':
    unittest.main()
