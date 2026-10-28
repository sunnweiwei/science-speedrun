"""Reproducible task-local batches, including resume after worker prefetch."""
import math

import torch


class CyclingBatchSampler:
    def __init__(self, size, batch_size, seed, consumed_batches=0):
        if size < 1 or batch_size < 1 or consumed_batches < 0:
            raise ValueError('invalid sampler arguments')
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.consumed_batches = consumed_batches
        self.batches_per_cycle = math.ceil(size / batch_size)

    def __iter__(self):
        cycle, offset = divmod(self.consumed_batches, self.batches_per_cycle)
        while True:
            generator = torch.Generator().manual_seed(self.seed + cycle)
            order = torch.randperm(self.size, generator=generator).tolist()
            for batch in range(offset, self.batches_per_cycle):
                yield order[batch * self.batch_size:(batch + 1) * self.batch_size]
            cycle += 1
            offset = 0
