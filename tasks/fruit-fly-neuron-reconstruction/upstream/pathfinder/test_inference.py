"""CPU checks for connecting the active trainers to inference."""
import unittest

from infer_sense import checkpoint_config
from sense import Sense


class CheckpointTests(unittest.TestCase):
    def test_active_volume_checkpoint_is_accepted(self):
        architecture = dict(stage='sense', width=8, factors=[1, 2, 4, 8, 8],
                            input_shape=[128]*3, voxel_size_nm=[16]*3)
        config = checkpoint_config(dict(architecture=architecture), [16]*3)
        self.assertEqual(config['prompt_radius'], 16)
        self.assertEqual(config['input_shape'], [128]*3)
        model = Sense(config['width'], tuple(config['factors']))
        self.assertEqual(model.levels, 5)

    def test_legacy_checkpoint_is_preserved(self):
        config = dict(width=2, factors=[1, 2], prompt_radius=7)
        checkpoint = dict(axes='zyx', voxel_size_nm=[40, 8, 8], config=config)
        self.assertEqual(checkpoint_config(checkpoint, [40, 8, 8]), config)

    def test_wrong_stage_or_resolution_is_rejected(self):
        architecture = dict(stage='ffn', width=32, factors=None,
                            input_shape=[33]*3, voxel_size_nm=[16]*3)
        with self.assertRaisesRegex(ValueError, 'SENSE'):
            checkpoint_config(dict(architecture=architecture), [16]*3)
        architecture.update(stage='sense', width=8, factors=[1, 2, 4, 8, 8])
        with self.assertRaisesRegex(ValueError, 'resolution'):
            checkpoint_config(dict(architecture=architecture), [40, 8, 8])


if __name__ == '__main__':
    unittest.main()
