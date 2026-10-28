"""Numerical input semantics and resumable sampler coverage, beyond shape checks."""
import itertools
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'environment/baseline/braindecode'))
from shared import chunk_signal
from sampling import CyclingBatchSampler


class SharedSemanticsTests(unittest.TestCase):
    def test_chunk_channels_time_and_batch_are_preserved(self):
        x = torch.arange(2 * 3 * 1000).reshape(2, 3, 1000).float()
        chunks, weights = chunk_signal(x)
        self.assertEqual(tuple(chunks.shape), (6, 3, 400))
        for batch in range(2):
            for chunk in range(3):
                length = min(400, 1000 - chunk * 400)
                torch.testing.assert_close(chunks[batch * 3 + chunk, :, :length],
                                           x[batch, :, chunk * 400:chunk * 400 + length])
                self.assertTrue(torch.all(chunks[batch * 3 + chunk, :, length:] == 0))
        torch.testing.assert_close(weights, torch.tensor([1., 1., .5]))

    def test_exact_chunks_round_trip(self):
        x = torch.randn(2, 19, 2000)
        chunks, weights = chunk_signal(x)
        restored = chunks.reshape(2, 5, 19, 400).permute(0, 2, 1, 3).reshape_as(x)
        torch.testing.assert_close(restored, x)
        self.assertTrue(torch.all(weights == 1))

    def test_resume_ignores_prefetch_and_preserves_coverage(self):
        sampler = CyclingBatchSampler(23, 4, seed=3407)
        expected = list(itertools.islice(iter(sampler), 20))
        self.assertEqual(sorted(sum(expected[:6], [])), list(range(23)))
        # A worker may prefetch all 20, while the trainer consumed only 7.
        resumed = CyclingBatchSampler(23, 4, seed=3407, consumed_batches=7)
        self.assertEqual(list(itertools.islice(iter(resumed), 13)), expected[7:])


if __name__ == '__main__':
    unittest.main()
