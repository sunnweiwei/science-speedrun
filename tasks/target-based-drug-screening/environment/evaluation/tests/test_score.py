from pathlib import Path
import pickle
import sys

import lmdb

EVALUATION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVALUATION))

from score import score_target  # noqa: E402


def test_complete_target_metrics_read_labels_from_lmdb(tmp_path):
    path = tmp_path / "test/dude/example/mols.lmdb"
    path.parent.mkdir(parents=True)
    environment = lmdb.open(str(path), subdir=False, map_size=1 << 24)
    with environment.begin(write=True) as transaction:
        for index in range(1000):
            transaction.put(str(index).encode(), pickle.dumps({"label": int(index % 50 == 0)}))
    environment.close()
    rows = [
        {"benchmark": "dude", "target_id": "example",
         "ligand_index": str(index), "score": str(float(1000 - index))}
        for index in range(1000)
    ]
    result = score_target(rows, tmp_path)
    assert result["ligands"] == 1000
    assert result["actives"] == 20
    assert 0 <= result["roc_auc"] <= 1
    assert 0 <= result["bedroc_80.5"] <= 1
    assert result["ef_0.01"] > 0
