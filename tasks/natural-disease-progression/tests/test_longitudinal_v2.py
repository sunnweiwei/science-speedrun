import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environment" / "baseline" / "temporal-gru"))

from longitudinal import (HORIZONS, LongitudinalRiskModel, _load_labels_aligned,
                          _numeric_value, _select_sequence, _sequence_delta, collate)


def test_v2_protocol_is_versioned_and_patient_weighted():
    protocol = json.loads((ROOT / "environment" / "data" / "protocols" / "longitudinal-v2.json").read_text())
    assert protocol["index_time"] == "second_eligible_live_adult_discharge"
    assert protocol["samples_per_patient"] == 1
    assert protocol["selection_uses_future_information"] is False
    assert protocol["horizons_days"] == list(HORIZONS)


def test_history_spanning_selection_keeps_old_and_recent_events():
    selected = _select_sequence(2000, 512)
    assert len(selected) == 512
    assert selected[0] == 0
    assert selected[-1] == 1999
    assert (selected[1:] > selected[:-1]).all()
    assert sum(selected >= 1744) == 256


def test_missing_numeric_is_mean_imputed_with_explicit_indicator():
    stats = {"lab:L:mmol/L": [10.0, 2.0]}
    assert _numeric_value("lab", "lab:L:mmol/L", "14", stats) == (2.0, 0.0)
    assert _numeric_value("lab", "lab:L:mmol/L", "not-a-number", stats) == (0.0, 1.0)
    assert _numeric_value("lab", "lab:unknown:mmol/L", "0", stats) == (0.0, 1.0)
    assert _numeric_value("diagnosis", "diagnosis:A00", None, stats) == (0.0, 0.0)


def test_sequence_gap_uses_event_time_order():
    # Backwards-from-cutoff time decreases for oldest-to-newest event order.
    assert _sequence_delta(np.asarray([5.0, 4.5, 1.0], np.float32)).tolist() == [0.0, .5, 3.5]


def test_label_loader_aligns_keys_and_rejects_duplicates(tmp_path):
    rows = []
    for sid in ("b", "a"):
        for disease in ("B", "A"):
            row = {"sample_id": sid, "disease_code": disease,
                   "cutoff": date(2020, 1, 1),
                   "diagnosis_observation_end": date(2022, 1, 1),
                   "death_observation_end": date(2022, 1, 1)}
            for horizon in HORIZONS:
                row[f"disease_label_{horizon}"] = sid == "b" and disease == "B"
                row[f"disease_valid_{horizon}"] = True
                row[f"death_label_{horizon}"] = sid == "b"
                row[f"death_valid_{horizon}"] = True
            rows.append(row)
    path = tmp_path / "labels.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    loaded = _load_labels_aligned(path, ["a", "b"], ["A", "B"])
    assert loaded["y"][:, :, -1].tolist() == [[0, 0], [0, 1]]
    assert loaded["death_y"][:, -1].tolist() == [0, 1]
    assert loaded["primary"].tolist() == [1, 1]
    pq.write_table(pa.Table.from_pylist(rows + [rows[0]]), path)
    with pytest.raises(ValueError, match="duplicated"):
        _load_labels_aligned(path, ["a", "b"], ["A", "B"])


def _row():
    return {"id": "x", "age": torch.tensor(70.0), "sex": torch.tensor(1.0),
            "all_code": torch.tensor([2,3,4],dtype=torch.int32),
            "all_mod": torch.tensor([1,2,1],dtype=torch.uint8),
            "all_value": torch.tensor([0.0,.5,0.0]),
            "all_missing": torch.tensor([0.0,1.0,0.0]),
            "seq_code": torch.tensor([2,3,4,0],dtype=torch.int32),
            "seq_mod": torch.tensor([1,2,1,0],dtype=torch.uint8),
            "seq_value": torch.tensor([0.0,.5,0.0,0.0]),
            "seq_missing": torch.tensor([0.0,1.0,0.0,0.0]),
            "seq_time": torch.tensor([3.0,2.0,1.0,0.0]),
            "seq_delta": torch.tensor([0.0,1.0,1.0,0.0]),
            "seq_len": torch.tensor(3,dtype=torch.int16)}


def test_diagnosis_ablation_removes_other_modalities_and_risk_is_monotone():
    batch = collate([_row()], diagnosis_only=True)
    assert batch["all_mod"].tolist() == [1,1]
    assert batch["seq_code"][0,:2].tolist() == [2,4]
    assert batch["seq_value"][0,:2].tolist() == [0.0,0.0]
    assert batch["seq_missing"][0,:2].tolist() == [0.0,0.0]
    assert batch["seq_time"][0,:2].tolist() == [3.0,1.0]
    assert batch["seq_delta"][0,:2].tolist() == [0.0,2.0]
    model = LongitudinalRiskModel(vocab=8,diseases=3,kind="temporal")
    disease, death = model(batch)
    assert disease.shape == (1,3,len(HORIZONS))
    assert death.shape == (1,len(HORIZONS))
    assert torch.all(torch.diff(disease,dim=2) >= 0)
    assert torch.all(torch.diff(death,dim=1) >= 0)
