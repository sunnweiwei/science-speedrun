import unittest
from unittest.mock import patch
import precompute_seeds as seeds


class SeedWorkerTests(unittest.TestCase):
    def test_large_host_cannot_spawn_one_memory_heavy_worker_per_core(self):
        with patch.object(seeds.os,'cpu_count',return_value=256),patch.object(seeds.os,'sched_getaffinity',return_value=set(range(256))):
            self.assertEqual(seeds.default_workers(),8)

    def test_small_allocation_is_respected(self):
        with patch.object(seeds.os,'sched_getaffinity',return_value={3,4}):
            self.assertEqual(seeds.default_workers(),2)


if __name__=='__main__':unittest.main()
