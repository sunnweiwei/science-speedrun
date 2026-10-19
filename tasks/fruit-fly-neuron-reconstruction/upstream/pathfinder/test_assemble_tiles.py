import copy,unittest
from assemble_tiles import validate_shards
from batched_ffn import grid_tiles


class ShardTests(unittest.TestCase):
    def fixture(self):
        tiles=list(grid_tiles([16]*3,[80]*3,32))
        rows=[]
        for shard in range(2):
            rows.append(dict(roi=[16,80]*3,tile_size=32,id_stride=32**3+1,shards=2,shard_index=shard,
                checkpoint_sha256='x',raw_sha256='y',metadata={'shape':[64]*3},ground_truth_used=False,
                tile_results=[dict(t,budget_exhausted=False) for t in tiles if t['index']%2==shard]))
        return rows
    def test_exact_partition_and_partial_input(self):
        rows=self.fixture();self.assertEqual(len(validate_shards(rows)[2]),8)
        self.assertEqual(len(validate_shards(rows[:1])[2]),4)
    def test_reject_duplicate_incompatible_and_foreign(self):
        rows=self.fixture()
        with self.assertRaises(ValueError):validate_shards(rows+rows[:1])
        bad=copy.deepcopy(rows);bad[1]['raw_sha256']='other'
        with self.assertRaises(ValueError):validate_shards(bad)
        mixed=self.fixture();mixed[1]['seed_order']='descending'
        with self.assertRaises(ValueError):validate_shards(mixed)
        for r in mixed:r['seed_order']='descending'
        self.assertEqual(len(validate_shards(mixed)[2]),8)
        bad=copy.deepcopy(rows);bad[0]['tile_results'].append(rows[1]['tile_results'][0])
        with self.assertRaises(ValueError):validate_shards(bad)

if __name__=='__main__':unittest.main()
