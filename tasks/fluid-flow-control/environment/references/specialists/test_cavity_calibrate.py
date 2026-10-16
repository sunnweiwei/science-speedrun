import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cavity_calibrate import calibrate


class CalibrationTests(unittest.TestCase):
    def test_completed_binding_and_rejections(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            samples = np.tile(np.array([[1., 2.], [-1., -2.]]), (2000, 1))
            np.savez_compressed(run / 'raw-measurements.npz', observations=samples)
            protocol = {'action': 'zero', 'minimum_calibration_cycles': 1000, 'steps': 4000,
                'reynolds': 4200, 'probe_locations': [1.3, 0.], 'observations': ['u', 'v'],
                'substeps_per_action': 400, 'probe_provenance': 'synthetic-test-not-CFD'}
            result = {'estimated_cycle_coverage_met': True, 'max_block_mean_drift_in_std': 0.,
                'samples': 4000, 'mean': [0., 0.], 'std_population': [1., 2.],
                'estimated_tail_cycles': 1000., 'tail_peak_count': 1000}
            (run / 'protocol.json').write_text(json.dumps(protocol))
            (run / 'diagnostic.json').write_text(json.dumps(result))
            (run / 'source.py').write_text('# Synthetic calibration test\n')
            output = run / 'calibration.json'
            document = calibrate(run, output)
            self.assertEqual(document['measurement']['tail_mean_eq36_reward'], -2.)
            with self.assertRaises(FileExistsError):
                calibrate(run, output)
            for change in ({'estimated_cycle_coverage_met': False},
                           {'max_block_mean_drift_in_std': .5}):
                (run / 'diagnostic.json').write_text(json.dumps(dict(result, **change)))
                with self.assertRaises(ValueError):
                    calibrate(run, run / 'rejected.json')
                self.assertFalse((run / 'rejected.json').exists())


if __name__ == '__main__':
    unittest.main()
