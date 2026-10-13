"""Read task-prepared MAGMa supervision using native GLACIER featurization."""

import json
from pathlib import Path
import sqlite3

from torch.utils.data import Dataset
from ms_pred import common
from ms_pred.glacier.dataset import IntenDataset, TreeProcessor


class PreparedDataset(Dataset):
    def __init__(self, root, split, model_config):
        if split not in {'train', 'validation'}:
            raise ValueError('only train and validation are development inputs')
        root = Path(root).resolve()
        self.rows = [json.loads(line) for line in (root / f'{split}.jsonl').open()]
        self.database = root / 'targets.sqlite'
        self.connection = None
        self.processor = TreeProcessor(
            pe_embed_k=model_config['pe_embed_k'], root_encode='graphormer',
            embed_elem_group=model_config['embed_elem_group'],
            multi_hop_max_dist=model_config['multi_hop_max_dist'],
        )

    def __getstate__(self):
        return dict(self.__dict__, connection=None)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if self.connection is None:
            self.connection = sqlite3.connect(self.database.as_uri() + '?mode=ro', uri=True)
        row = self.rows[index]
        result = self.connection.execute('SELECT payload FROM targets WHERE query_id=?', (row['query_id'],)).fetchone()
        if result is None or result[0] is None:
            raise ValueError('missing prepared supervision')
        item = self.processor.featurize_tree(json.loads(result[0]))
        item.update(name=row['query_id'], adduct=common.ion2onehot_pos[row['adduct']],
                    instrument=common.instrument2onehot_pos[row['instrument_type']],
                    precursor=row['precursor_mz'])
        return item

    collate_fn = staticmethod(IntenDataset.collate_fn)
